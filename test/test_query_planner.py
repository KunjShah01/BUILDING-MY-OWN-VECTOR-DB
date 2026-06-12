"""Tests for the AST hybrid query parser and cost-based optimizer."""

import pytest

from utils.query_planner import (
    parse_query, plan_query, eval_metadata, estimate_selectivity,
    Predicate, SemanticMatch, BoolOp, NotOp,
)


# ---------------- parsing ----------------

def test_parse_simple_predicate():
    ast = parse_query("price < 100")
    assert isinstance(ast, Predicate)
    assert ast.field == "price"
    assert ast.op == "<"
    assert ast.value == 100


def test_parse_string_value():
    ast = parse_query("category = 'tech'")
    assert ast.value == "tech"


def test_parse_semantic_match_with_k():
    ast = parse_query('semantic_match("laptops", 5)')
    assert isinstance(ast, SemanticMatch)
    assert ast.query == "laptops"
    assert ast.k == 5


def test_parse_and_or_precedence():
    ast = parse_query("a = 1 AND b = 2 OR c = 3")
    # OR is lowest precedence -> root is OR
    assert isinstance(ast, BoolOp)
    assert ast.op == "OR"
    assert ast.children[0].op == "AND"


def test_parse_parentheses_override():
    ast = parse_query("a = 1 AND (b = 2 OR c = 3)")
    assert ast.op == "AND"
    assert ast.children[1].op == "OR"


def test_parse_not():
    ast = parse_query("NOT price > 100")
    assert isinstance(ast, NotOp)


def test_parse_hybrid_query():
    ast = parse_query("(category = 'tech' AND price < 100) OR semantic_match(\"laptops\")")
    assert isinstance(ast, BoolOp)
    assert ast.op == "OR"


def test_parse_invalid_raises():
    with pytest.raises(ValueError):
        parse_query("price <")


# ---------------- evaluation ----------------

def test_eval_metadata_predicate():
    ast = parse_query("price < 100")
    assert eval_metadata(ast, {"price": 50}) is True
    assert eval_metadata(ast, {"price": 150}) is False
    assert eval_metadata(ast, {}) is False


def test_eval_in_operator():
    ast = parse_query("category in 'tech'")  # value parsed as string, membership
    assert eval_metadata(ast, {"category": "t"}) is True


def test_eval_and_or():
    ast = parse_query("a = 1 AND b = 2")
    assert eval_metadata(ast, {"a": 1, "b": 2}) is True
    assert eval_metadata(ast, {"a": 1, "b": 3}) is False


def test_eval_semantic_leaf_is_true_for_metadata():
    ast = parse_query('semantic_match("x")')
    assert eval_metadata(ast, {"anything": 1}) is True


def test_eval_not():
    ast = parse_query("NOT price > 100")
    assert eval_metadata(ast, {"price": 50}) is True
    assert eval_metadata(ast, {"price": 150}) is False


# ---------------- optimizer ----------------

def test_plan_filter_only():
    plan = plan_query("category = 'tech' AND price < 100")
    assert plan.strategy == "filter_only"
    assert plan.semantic_query is None


def test_plan_vector_only():
    plan = plan_query('semantic_match("laptops")')
    assert plan.strategy == "vector_only"
    assert plan.semantic_query == "laptops"


def test_plan_filter_first_when_selective():
    plan = plan_query("category = 'tech' AND semantic_match(\"x\")",
                      stats={"category": {"distinct": 100}})
    assert plan.strategy == "filter_first"
    assert plan.estimated_selectivity < 0.1


def test_plan_vector_first_when_broad():
    plan = plan_query("price > 10 AND semantic_match(\"x\")")
    assert plan.strategy == "vector_first"


def test_plan_default_k():
    plan = plan_query('semantic_match("x")', default_k=25)
    assert plan.semantic_k == 25


def test_plan_predicate_fn_callable():
    plan = plan_query("category = 'tech' AND semantic_match(\"x\")")
    assert plan.predicate_fn({"category": "tech"}) is True
    assert plan.predicate_fn({"category": "food"}) is False


def test_selectivity_and_lower_than_or():
    and_sel = estimate_selectivity(parse_query("a = 1 AND b = 2"))
    or_sel = estimate_selectivity(parse_query("a = 1 OR b = 2"))
    assert and_sel < or_sel
