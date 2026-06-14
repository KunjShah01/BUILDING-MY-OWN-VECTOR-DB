"""Natural Language Query router: English-to-structured-query via LLM."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from config.database import get_db
from models.pydantic_models import NLQueryRequest, NLQueryResponse
from services.rag_service import openai_chat_completion
from services.search_engine_service import SearchEngineService
from services.vector_service import VectorService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Natural Language Query"])


def _tenant_id(request: Request) -> str:
    return getattr(request.state, "tenant_id", None) or "default"


@router.post("/search/nl", response_model=NLQueryResponse)
def nl_query(
    request: Request,
    body: NLQueryRequest,
    db: Session = Depends(get_db),
):
    prompt = (
        "You are a query parser. Convert the following natural language search query "
        "into a structured JSON object with these fields:\n"
        "  - text (string): the core search terms\n"
        "  - categories (list of strings): category filters\n"
        "  - date_from (string or null): ISO date lower bound\n"
        "  - date_to (string or null): ISO date upper bound\n"
        "  - limit (int): max results (default 10)\n"
        "  - method (string): 'hybrid', 'sparse', or 'dense' (default 'hybrid')\n\n"
        f"Query: {body.query}\n\n"
        "Return ONLY valid JSON, no other text."
    )

    structured = {"text": body.query, "categories": [], "date_from": None, "date_to": None, "limit": body.limit, "method": "hybrid"}
    llm_response = openai_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        model="gpt-4o-mini",
        max_tokens=300,
        temperature=0.1,
    )

    try:
        parsed = json.loads(llm_response)
        structured.update(parsed)
    except (json.JSONDecodeError, TypeError):
        logger.warning("LLM response was not valid JSON, falling back: %s", llm_response)

    vs = VectorService()
    ses = SearchEngineService(vs)

    search_text = structured.get("text", body.query)
    limit = structured.get("limit", body.limit)
    method = structured.get("method", "hybrid")

    search_results = []
    try:
        from models.pydantic_models import SearchMethod
        search_method = SearchMethod.HYBRID if method == "hybrid" else SearchMethod.BRUTE
        resp = ses.search(
            query_vector=[0.0] * 384,
            query_text=search_text,
            k=limit,
            method=search_method,
        )
        search_results = resp if isinstance(resp, list) else resp.get("results", [])
    except Exception as exc:
        logger.exception("Search execution failed")

    return NLQueryResponse(
        success=True,
        structured_query=structured,
        results=search_results,
        total=len(search_results),
    )
