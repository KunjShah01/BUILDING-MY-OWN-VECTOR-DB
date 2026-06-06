"""Text chunking strategies for RAG."""
from typing import List, Optional


def chunk_text_recursive(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Recursive character text splitting with overlap."""
    if not text:
        return []

    separators = ["\n\n", "\n", ". ", "! ", "? ", ", ", " "]

    def _split(text: str, seps: List[str]) -> List[str]:
        if not seps or len(text) <= chunk_size:
            return [text]
        sep = seps[0]
        parts = text.split(sep)
        chunks = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)
        result = []
        for c in chunks:
            if len(c) > chunk_size:
                result.extend(_split(c, seps[1:]))
            else:
                result.append(c)
        return result

    chunks = _split(text, separators)

    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                prev_end = chunks[i - 1][-overlap:]
                chunk = prev_end + chunk
            overlapped.append(chunk)
        chunks = overlapped

    return chunks


def chunk_by_sentences(text: str, max_sentences: int = 5, overlap_sentences: int = 1) -> List[str]:
    """Split text into chunks by sentence count."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]
    if not sentences:
        return []
    chunks = []
    start = 0
    while start < len(sentences):
        end = start + max_sentences
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        start += max_sentences - overlap_sentences
    return chunks


def chunk_tokens(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """Token-aware chunking using tiktoken with fallback."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_text = enc.decode(tokens[start:end])
            chunks.append(chunk_text)
            start += chunk_size - overlap
        return chunks
    except ImportError:
        return chunk_text_recursive(text, chunk_size=chunk_size * 4, overlap=overlap * 4)


def chunk_contextual(text: str, document_summary: Optional[str] = None, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Contextual chunking: prepends a document-level summary to each chunk.
    This provides global context to localized chunks, which has been shown 
    to significantly improve retrieval performance (e.g., Anthropic's Contextual Retrieval).
    
    Args:
        text: Full document text
        document_summary: High-level summary of the entire document
        chunk_size: Target size of each chunk
        overlap: Character overlap between chunks
        
    Returns:
        List of context-enriched chunks
    """
    base_chunks = chunk_text_recursive(text, chunk_size=chunk_size, overlap=overlap)
    
    if not document_summary:
        return base_chunks
    
    # Prepend summary to each chunk to anchor its meaning
    return [
        f"DOCUMENT SUMMARY: {document_summary}\n---\nCHUNK CONTENT:\n{chunk}" 
        for chunk in base_chunks
    ]


def chunk_semantic(text: str, embed_fn, distance_threshold: float = 0.3) -> List[str]:
    """
    Semantic chunking: split text into sentences, embed them, and break into
    new chunks whenever the cosine distance between adjacent sentences 
    exceeds the given threshold (indicating a topic shift).
    
    Args:
        text: Full document text
        embed_fn: Callable that takes a string and returns a vector (List[float])
        distance_threshold: Cosine distance cutoff to trigger a chunk split
        
    Returns:
        List of semantically coherent chunks
    """
    import re
    from utils.distance import cosine_distance
    
    # Basic sentence split
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if len(s.strip()) > 5]
    
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences
        
    # Embed all sentences
    embeddings = [embed_fn(s) for s in sentences]
    
    chunks = []
    current_chunk_sentences = [sentences[0]]
    
    for i in range(1, len(sentences)):
        # Calculate semantic shift between adjacent sentences
        dist = cosine_distance(embeddings[i-1], embeddings[i])
        
        if dist > distance_threshold:
            # Significant topic shift detected -> flush current chunk
            chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentences[i]]
        else:
            # Coherent -> keep appending
            current_chunk_sentences.append(sentences[i])
            
    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
        
    return chunks
