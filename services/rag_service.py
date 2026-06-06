"""RAG pipeline: PDF -> Chunk -> Embed -> Vector DB -> Retrieve -> LLM completion."""
from typing import List, Dict, Any, Optional
import logging
import os

from services.embedding_service import embed_text, embed_texts, embed_image
from services.media_store import save_media
from utils.pdf_processor import extract_text_from_pdf
from utils.text_chunker import chunk_text_recursive, chunk_tokens, chunk_by_sentences

logger = logging.getLogger(__name__)


def openai_chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    max_tokens: int = 500,
    temperature: float = 0.3,
) -> str:
    """Call OpenAI chat completion."""
    try:
        from openai import OpenAI
    except ImportError:
        return "OpenAI SDK not installed. Install: pip install openai"
    try:
        client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"OpenAI API error: {exc}"


class RAGService:
    """Retrieval-Augmented Generation pipeline."""

    def __init__(self, db_session=None):
        self.db = db_session
        self._vector_service = None
        self._collection_service = None
        self._index_service = None

    def _get_collection_service(self):
        if self._collection_service is None and self.db is not None:
            from services.collection_service import CollectionService
            self._collection_service = CollectionService(self.db)
        return self._collection_service

    def _get_vector_service(self):
        if self._vector_service is None and self.db is not None:
            from services.vector_service import VectorService
            self._vector_service = VectorService(self.db)
        return self._vector_service

    def _search_vectors(self, collection_id: str, query_vector: List[float],
                        k: int = 5, filters: Optional[Dict] = None) -> Dict[str, Any]:
        from services.collection_index_service import CollectionIndexService
        svc = CollectionIndexService(self.db)
        return svc.search_collection_indexed(
            collection_id=collection_id, query_vector=query_vector,
            k=k, method="brute", filters=filters, distance_metric="cosine",
        )

    def ingest_pdf(
        self, collection_id: str, pdf_path: str,
        chunk_strategy: str = "recursive", chunk_size: int = 500, chunk_overlap: int = 50,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Extract, chunk, embed, and store a PDF's contents."""
        try:
            coll_svc = self._get_collection_service()
            if coll_svc:
                coll = coll_svc.get_collection(collection_id)
                if not coll.get("success"):
                    return coll

            full_text = extract_text_from_pdf(pdf_path)
            if not full_text.strip():
                return {"success": False, "message": "No text extracted from PDF"}

            chunks = self._chunk_text(full_text, chunk_strategy, chunk_size, chunk_overlap)

            if not chunks:
                return {"success": False, "message": "No chunks generated"}

            embeddings = embed_texts(chunks)
            vec_svc = self._get_vector_service()
            stored = 0
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                chunk_meta = {
                    **(metadata or {}),
                    "source": os.path.basename(pdf_path),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content_type": "rag_chunk",
                    "text": chunk[:500],
                }
                result = vec_svc.create_vector(vector_data=emb, metadata=chunk_meta, collection_id=collection_id)
                if result.get("success"):
                    stored += 1

            return {
                "success": True,
                "message": f"Ingested {stored}/{len(chunks)} chunks from PDF",
                "total_chunks": len(chunks),
                "stored": stored,
            }
        except Exception as exc:
            logger.exception("PDF ingestion failed")
            return {"success": False, "message": f"PDF ingestion error: {exc}"}

    def _chunk_text(self, text: str, strategy: str, chunk_size: int, overlap: int) -> List[str]:
        if strategy == "tokens":
            return chunk_tokens(text, chunk_size=chunk_size, overlap=overlap)
        elif strategy == "sentences":
            return chunk_by_sentences(text, max_sentences=chunk_size, overlap_sentences=overlap)
        return chunk_text_recursive(text, chunk_size=chunk_size, overlap=overlap)

    def ingest_document(
        self,
        collection_id: str,
        file_path: str,
        chunk_strategy: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        extract_images: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest any supported document, optionally extracting embedded images."""
        from utils.document_processor import extract_text, extract_images_from_pdf

        try:
            text = extract_text(file_path)
            if not text.strip():
                return {"success": False, "message": "No text extracted"}

            chunks = self._chunk_text(text, chunk_strategy, chunk_size, chunk_overlap)
            if not chunks:
                return {"success": False, "message": "No chunks generated"}

            embeddings = embed_texts(chunks)
            vec_svc = self._get_vector_service()
            stored = 0

            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                chunk_meta = {
                    **(metadata or {}),
                    "source": os.path.basename(file_path),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content_type": "rag_chunk",
                    "text": chunk[:500],
                }
                result = vec_svc.create_vector(
                    vector_data=emb, metadata=chunk_meta, collection_id=collection_id
                )
                if result.get("success"):
                    stored += 1

            ext = os.path.splitext(file_path)[1].lower()
            image_count = 0
            if extract_images and ext == ".pdf":
                images = extract_images_from_pdf(file_path)
                for img_bytes in images:
                    if len(img_bytes) < 1024:
                        continue
                    content_uri = save_media(collection_id, f"pdf_img_{stored}_{image_count}.png", img_bytes)
                    try:
                        img_emb = embed_image(img_bytes)
                        img_meta = {
                            **(metadata or {}),
                            "source": os.path.basename(file_path),
                            "content_type": "rag_image",
                            "content_uri": content_uri,
                        }
                        vec_svc.create_vector(
                            vector_data=img_emb, metadata=img_meta, collection_id=collection_id
                        )
                        image_count += 1
                    except Exception:
                        pass

            return {
                "success": True,
                "message": f"Ingested {stored} text chunks + {image_count} images",
                "total_chunks": len(chunks),
                "stored": stored,
                "images_extracted": image_count,
            }
        except Exception as exc:
            logger.exception("Document ingestion failed")
            return {"success": False, "message": str(exc)}

    def query(
        self, collection_id: str, query: str, k: int = 5,
        llm_model: str = "gpt-4o-mini", api_key: Optional[str] = None,
        system_prompt: Optional[str] = None, max_tokens: int = 500, temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """RAG query: embed -> retrieve -> build context -> LLM answer."""
        try:
            query_vector = embed_text(query)
            search_result = self._search_vectors(
                collection_id=collection_id, query_vector=query_vector, k=k,
                filters={"content_type": "rag_chunk"},
            )
            if not search_result.get("success"):
                return {"success": False, "message": "Search failed", "error": search_result.get("message")}

            results = search_result.get("results", [])
            if not results:
                return {"success": True, "answer": "No relevant documents found.", "context": [], "query": query}

            context, context_list = self._build_context(results)
            answer = self._llm_answer(query, context, system_prompt, llm_model, api_key, max_tokens, temperature)
            return {
                "success": True, "answer": answer, "query": query,
                "context": context_list,
                "total_results": len(results),
            }
        except Exception as exc:
            logger.exception("RAG query failed")
            return {"success": False, "message": f"RAG query error: {exc}"}

    # ------------------------------------------------------------------ #
    #  Advanced RAG strategies                                            #
    # ------------------------------------------------------------------ #

    def query_with_rewrite(
        self, collection_id: str, query: str, k: int = 5,
        llm_model: str = "gpt-4o-mini", api_key: Optional[str] = None,
        system_prompt: Optional[str] = None, max_tokens: int = 500, temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Query rewriting RAG: ask the LLM to rephrase/expand the query
        before embedding, so the retrieval step matches more relevant chunks.

        Flow: query -> LLM rewrite -> embed rewritten -> retrieve -> LLM answer
        """
        try:
            rewrite_messages = [
                {"role": "system", "content": (
                    "You are a search query optimizer. Rewrite the user's question "
                    "to be more specific and detailed for semantic search. "
                    "Output ONLY the rewritten query, nothing else."
                )},
                {"role": "user", "content": query},
            ]
            rewritten = openai_chat_completion(
                messages=rewrite_messages, model=llm_model, api_key=api_key,
                max_tokens=150, temperature=0.3,
            )

            query_vector = embed_text(rewritten)
            search_result = self._search_vectors(
                collection_id=collection_id, query_vector=query_vector, k=k,
                filters={"content_type": "rag_chunk"},
            )
            if not search_result.get("success"):
                return {"success": False, "message": "Search failed"}

            results = search_result.get("results", [])
            if not results:
                return {"success": True, "answer": "No relevant documents found.",
                        "context": [], "query": query, "rewritten_query": rewritten}

            context, context_list = self._build_context(results)
            answer = self._llm_answer(query, context, system_prompt, llm_model, api_key, max_tokens, temperature)
            return {
                "success": True, "answer": answer, "query": query,
                "rewritten_query": rewritten,
                "context": context_list, "total_results": len(results),
                "strategy": "rewrite",
            }
        except Exception as exc:
            logger.exception("RAG rewrite query failed")
            return {"success": False, "message": f"RAG rewrite error: {exc}"}

    def query_with_hyde(
        self, collection_id: str, query: str, k: int = 5,
        llm_model: str = "gpt-4o-mini", api_key: Optional[str] = None,
        system_prompt: Optional[str] = None, max_tokens: int = 500, temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Hypothetical Document Embeddings (HyDE) RAG.

        Instead of embedding the raw question, we ask the LLM to generate
        a hypothetical answer paragraph, embed THAT, and use it for retrieval.
        The hypothetical answer lives in the same embedding space as the actual
        documents, so retrieval recall improves 10-30%.

        Flow: query -> LLM hypothetical answer -> embed hypothesis -> retrieve -> LLM answer
        """
        try:
            hyde_messages = [
                {"role": "system", "content": (
                    "Write a short, detailed paragraph that answers the following question. "
                    "Be specific and factual. This will be used for document retrieval, "
                    "so include key terms and concepts. Output ONLY the paragraph."
                )},
                {"role": "user", "content": query},
            ]
            hypothetical = openai_chat_completion(
                messages=hyde_messages, model=llm_model, api_key=api_key,
                max_tokens=200, temperature=0.5,
            )

            # Embed the hypothetical answer instead of the raw query
            hyde_vector = embed_text(hypothetical)
            search_result = self._search_vectors(
                collection_id=collection_id, query_vector=hyde_vector, k=k,
                filters={"content_type": "rag_chunk"},
            )
            if not search_result.get("success"):
                return {"success": False, "message": "Search failed"}

            results = search_result.get("results", [])
            if not results:
                return {"success": True, "answer": "No relevant documents found.",
                        "context": [], "query": query, "hypothetical_answer": hypothetical}

            context, context_list = self._build_context(results)
            # Use original query for final answer (not the hypothesis)
            answer = self._llm_answer(query, context, system_prompt, llm_model, api_key, max_tokens, temperature)
            return {
                "success": True, "answer": answer, "query": query,
                "hypothetical_answer": hypothetical,
                "context": context_list, "total_results": len(results),
                "strategy": "hyde",
            }
        except Exception as exc:
            logger.exception("RAG HyDE query failed")
            return {"success": False, "message": f"RAG HyDE error: {exc}"}

    def query_multihop(
        self, collection_id: str, query: str, k: int = 5, hops: int = 2,
        llm_model: str = "gpt-4o-mini", api_key: Optional[str] = None,
        system_prompt: Optional[str] = None, max_tokens: int = 500, temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Multi-hop iterative RAG for complex questions.

        Performs multiple retrieval rounds:
          1. Retrieve with original query
          2. Ask LLM what information is still missing
          3. Retrieve with the refined follow-up query
          4. Combine all context for final answer

        Args:
            hops: Number of retrieval iterations (default 2)
        """
        try:
            all_results = []
            current_query = query

            for hop in range(hops):
                query_vector = embed_text(current_query)
                search_result = self._search_vectors(
                    collection_id=collection_id, query_vector=query_vector, k=k,
                    filters={"content_type": "rag_chunk"},
                )
                if search_result.get("success"):
                    hop_results = search_result.get("results", [])
                    all_results.extend(hop_results)

                if hop < hops - 1:
                    # Ask LLM what's missing before next hop
                    hop_context, _ = self._build_context(all_results)
                    refine_messages = [
                        {"role": "system", "content": (
                            "Based on the context below, determine what additional "
                            "information is needed to fully answer the question. "
                            "Output a focused search query to find the missing information. "
                            "Output ONLY the search query."
                        )},
                        {"role": "user", "content": (
                            f"Question: {query}\n\nContext so far:\n{hop_context[:2000]}"
                        )},
                    ]
                    current_query = openai_chat_completion(
                        messages=refine_messages, model=llm_model, api_key=api_key,
                        max_tokens=100, temperature=0.3,
                    )

            # Deduplicate results by vector_id
            seen_ids = set()
            unique_results = []
            for r in all_results:
                vid = r.get("vector_id", "")
                if vid not in seen_ids:
                    seen_ids.add(vid)
                    unique_results.append(r)

            if not unique_results:
                return {"success": True, "answer": "No relevant documents found.",
                        "context": [], "query": query}

            context, context_list = self._build_context(unique_results)
            answer = self._llm_answer(query, context, system_prompt, llm_model, api_key, max_tokens, temperature)
            return {
                "success": True, "answer": answer, "query": query,
                "context": context_list, "total_results": len(unique_results),
                "hops_performed": hops, "strategy": "multihop",
            }
        except Exception as exc:
            logger.exception("RAG multihop query failed")
            return {"success": False, "message": f"RAG multihop error: {exc}"}

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_context(self, results: List[Dict]) -> tuple:
        """
        Build context string and structured list from search results.
        Returns (context_string, context_list).
        """
        context_parts = []
        context_list = []
        for r in results:
            meta = r.get("metadata", {})
            chunk_text = meta.get("text", "")
            source = meta.get("source", "unknown")
            if chunk_text:
                context_parts.append(f"[Source: {source}]\n{chunk_text}")
                context_list.append({
                    "text": chunk_text, "source": source,
                    "distance": r.get("distance", 0),
                })
        return "\n\n---\n\n".join(context_parts), context_list

    def _llm_answer(self, query: str, context: str,
                    system_prompt: Optional[str], llm_model: str,
                    api_key: Optional[str], max_tokens: int,
                    temperature: float) -> str:
        """Call LLM with context to produce a grounded answer."""
        default_system = (
            "You are a helpful assistant. Answer the user's question based solely "
            "on the provided context. If the context doesn't contain enough "
            "information, say so. Cite the source document when possible."
        )
        messages = [
            {"role": "system", "content": system_prompt or default_system},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        return openai_chat_completion(
            messages=messages, model=llm_model, api_key=api_key,
            max_tokens=max_tokens, temperature=temperature,
        )

