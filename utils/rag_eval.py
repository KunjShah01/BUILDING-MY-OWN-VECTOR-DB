"""
RAG Evaluation Harness.

Provides a RAGAS-style evaluation suite to measure the quality of the RAG pipeline.
Metrics:
- Context Relevancy: How relevant the retrieved context is to the query (using embeddings).
- Answer Faithfulness: Is the answer grounded in the retrieved context? (LLM as judge)
- Answer Relevancy: Does the answer actually address the user's question? (LLM as judge)
"""

import json
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

from services.embedding_service import embed_text
from services.rag_service import openai_chat_completion
from utils.distance import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class RAGScore:
    context_relevancy: float      # [0.0, 1.0]
    answer_faithfulness: float    # [0.0, 1.0]
    answer_relevancy: float       # [0.0, 1.0]
    overall_score: float          # [0.0, 1.0]
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "context_relevancy": round(self.context_relevancy, 3),
            "answer_faithfulness": round(self.answer_faithfulness, 3),
            "answer_relevancy": round(self.answer_relevancy, 3),
            "overall_score": round(self.overall_score, 3)
        }


class RAGEvaluator:
    def __init__(self, llm_model: str = "gpt-4o-mini", api_key: str = None):
        self.llm_model = llm_model
        self.api_key = api_key

    def _eval_context_relevancy(self, query: str, contexts: List[str]) -> float:
        """
        Measure how semantically relevant the retrieved contexts are to the query.
        Uses cosine similarity between the query embedding and the chunk embeddings.
        Returns the average similarity of the top chunks.
        """
        if not contexts:
            return 0.0
            
        try:
            query_emb = embed_text(query)
            similarities = []
            for ctx in contexts:
                ctx_emb = embed_text(ctx)
                sim = cosine_similarity(query_emb, ctx_emb)
                similarities.append(sim)
                
            # Cap at 0, average the scores
            similarities = [max(0.0, s) for s in similarities]
            return sum(similarities) / len(similarities)
        except Exception as e:
            logger.error(f"Context relevancy eval failed: {e}")
            return 0.0

    def _eval_answer_faithfulness(self, question: str, answer: str, context_str: str) -> float:
        """
        Measure if the answer is grounded in the retrieved context (no hallucinations).
        Asks the LLM to output a binary 1 or 0.
        """
        if not answer or not context_str:
            return 0.0
            
        prompt = (
            "Given the following question, context, and answer, evaluate if the answer "
            "is strictly faithful to and grounded in the context. "
            "Ignore whether the answer actually addresses the question, only check if "
            "the facts in the answer are present in the context. "
            "Output ONLY '1' if faithful, or '0' if it contains hallucinations or ungrounded facts."
        )
        user_content = f"Question: {question}\n\nContext:\n{context_str}\n\nAnswer: {answer}"
        
        try:
            result = openai_chat_completion(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content}
                ],
                model=self.llm_model,
                api_key=self.api_key,
                temperature=0.0,
                max_tokens=10
            )
            return 1.0 if "1" in result else 0.0
        except Exception as e:
            logger.error(f"Answer faithfulness eval failed: {e}")
            return 0.0

    def _eval_answer_relevancy(self, question: str, answer: str) -> float:
        """
        Measure if the answer actually addresses the user's question.
        Asks the LLM to score from 0.0 to 1.0.
        """
        if not answer:
            return 0.0
            
        prompt = (
            "Evaluate how relevant and direct the following answer is to the given question. "
            "Score from 0.0 (completely irrelevant or dodges the question) to 1.0 (perfectly addresses the question). "
            "Output ONLY the float score, e.g., '0.8'."
        )
        user_content = f"Question: {question}\n\nAnswer: {answer}"
        
        try:
            result = openai_chat_completion(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_content}
                ],
                model=self.llm_model,
                api_key=self.api_key,
                temperature=0.0,
                max_tokens=10
            )
            import re
            match = re.search(r"0\.\d+|1\.0", result)
            if match:
                return float(match.group(0))
            return 0.0
        except Exception as e:
            logger.error(f"Answer relevancy eval failed: {e}")
            return 0.0

    def evaluate(self, query: str, answer: str, contexts: List[str]) -> RAGScore:
        """
        Evaluate a single RAG interaction across all three dimensions.
        """
        cr = self._eval_context_relevancy(query, contexts)
        context_str = "\n".join(contexts)
        af = self._eval_answer_faithfulness(query, answer, context_str)
        ar = self._eval_answer_relevancy(query, answer)
        
        # Overall score is the harmonic mean, to penalize any single bad metric
        # Avoid division by zero
        if cr == 0 or af == 0 or ar == 0:
            overall = 0.0
        else:
            overall = 3 / ((1/cr) + (1/af) + (1/ar))
            
        return RAGScore(
            context_relevancy=cr,
            answer_faithfulness=af,
            answer_relevancy=ar,
            overall_score=overall
        )

    def evaluate_dataset(self, qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate an entire dataset.
        Expected input format:
        [
            {
                "query": "...",
                "answer": "...",
                "contexts": ["..."]
            }, ...
        ]
        """
        scores = []
        for i, pair in enumerate(qa_pairs):
            query = pair.get("query", "")
            answer = pair.get("answer", "")
            contexts = pair.get("contexts", [])
            
            logger.info(f"Evaluating example {i+1}/{len(qa_pairs)}...")
            score = self.evaluate(query, answer, contexts)
            scores.append(score)
            
        if not scores:
            return {}
            
        avg_cr = sum(s.context_relevancy for s in scores) / len(scores)
        avg_af = sum(s.answer_faithfulness for s in scores) / len(scores)
        avg_ar = sum(s.answer_relevancy for s in scores) / len(scores)
        avg_overall = sum(s.overall_score for s in scores) / len(scores)
        
        return {
            "total_evaluated": len(scores),
            "average_metrics": {
                "context_relevancy": round(avg_cr, 3),
                "answer_faithfulness": round(avg_af, 3),
                "answer_relevancy": round(avg_ar, 3),
                "overall_score": round(avg_overall, 3)
            },
            "individual_scores": [s.to_dict() for s in scores]
        }
