"""
AST-based hybrid query parser + cost-based optimizer (ROADMAP Phase 4).

Parses hybrid query expressions that mix metadata predicates with semantic
search, e.g.:

    (category = 'tech' AND price < 100) OR semantic_match("laptops")

into an Abstract Syntax Tree of typed nodes, then a cost-based optimizer
decides execution order: run the metadata filter first when it is highly
selective (small candidate set), otherwise run vector search first and
post-filter.

Grammar (recursive-descent, precedence: OR < AND < NOT < primary):

    expr    := or_expr
    or_expr := and_expr ( OR and_expr )*
    and_expr:= not_expr ( AND not_expr )*
    not_expr:= NOT not_expr | primary
    primary := '(' expr ')'
             | semantic_match '(' STRING [ ',' NUMBER ] ')'
             | FIELD OP VALUE
    OP      := = | != | < | <= | > | >= | in | contains
"""

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict, Callable, Tuple


# ----------------------------- AST nodes -----------------------------

class Node:
    """Base AST node."""


@dataclass
class Predicate(Node):
    """A metadata comparison: field OP value."""
    field: str
    op: str
    value: Any

    def matches(self, metadata: Optional[Dict[str, Any]]) -> bool:
        if metadata is None:
            return False
        actual = metadata.get(self.field)
        op = self.op
        try:
            if op == "=":
                return actual == self.value
            if op == "!=":
                return actual != self.value
            if actual is None:
                return False
            if op == "<":
                return actual < self.value
            if op == "<=":
                return actual <= self.value
            if op == ">":
                return actual > self.value
            if op == ">=":
                return actual >= self.value
            if op == "in":
                return actual in self.value
            if op == "contains":
                return self.value in actual
        except TypeError:
            return False
        return False


@dataclass
class SemanticMatch(Node):
    """A vector-search leaf: semantic_match("query text", k)."""
    query: str
    k: Optional[int] = None


@dataclass
class BoolOp(Node):
    """AND / OR over children."""
    op: str  # "AND" | "OR"
    children: List[Node] = field(default_factory=list)


@dataclass
class NotOp(Node):
    child: Node


# ----------------------------- tokenizer -----------------------------

_TOKEN_RE = re.compile(r"""
    \s*(?:
        (?P<lparen>\()
      | (?P<rparen>\))
      | (?P<comma>,)
      | (?P<op><=|>=|!=|=|<|>)
      | (?P<string>'[^']*'|"[^"]*")
      | (?P<number>-?\d+\.?\d*)
      | (?P<word>[A-Za-z_][A-Za-z0-9_\.]*)
    )
""", re.VERBOSE)

_KEYWORDS = {"and", "or", "not", "in", "contains", "semantic_match"}


@dataclass
class Token:
    kind: str
    value: str


def tokenize(text: str) -> List[Token]:
    tokens: List[Token] = []
    pos = 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            raise ValueError(f"Unexpected character at {pos}: {text[pos:pos+10]!r}")
        pos = m.end()
        kind = m.lastgroup
        value = m.group(kind)
        if kind == "word" and value.lower() in _KEYWORDS:
            tokens.append(Token(value.lower(), value.lower()))
        else:
            tokens.append(Token(kind, value))
    return tokens


# ----------------------------- parser -----------------------------

class QueryParser:
    """Recursive-descent parser producing an AST."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.i = 0

    def parse(self) -> Node:
        node = self._or()
        if self.i != len(self.tokens):
            raise ValueError(f"Trailing tokens from {self.tokens[self.i]}")
        return node

    def _peek(self) -> Optional[Token]:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def _next(self) -> Token:
        tok = self.tokens[self.i]
        self.i += 1
        return tok

    def _expect(self, kind: str) -> Token:
        tok = self._peek()
        if tok is None or tok.kind != kind:
            raise ValueError(f"Expected {kind}, got {tok}")
        return self._next()

    def _or(self) -> Node:
        children = [self._and()]
        while self._peek() and self._peek().kind == "or":
            self._next()
            children.append(self._and())
        return children[0] if len(children) == 1 else BoolOp("OR", children)

    def _and(self) -> Node:
        children = [self._not()]
        while self._peek() and self._peek().kind == "and":
            self._next()
            children.append(self._not())
        return children[0] if len(children) == 1 else BoolOp("AND", children)

    def _not(self) -> Node:
        if self._peek() and self._peek().kind == "not":
            self._next()
            return NotOp(self._not())
        return self._primary()

    def _primary(self) -> Node:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of query")

        if tok.kind == "lparen":
            self._next()
            node = self._or()
            self._expect("rparen")
            return node

        if tok.kind == "semantic_match":
            self._next()
            self._expect("lparen")
            query = self._literal_value(self._expect("string"))
            k = None
            if self._peek() and self._peek().kind == "comma":
                self._next()
                k = int(self._literal_value(self._expect("number")))
            self._expect("rparen")
            return SemanticMatch(query=query, k=k)

        # field OP value
        field_tok = self._expect("word")
        op_tok = self._peek()
        if op_tok and op_tok.kind in ("op", "in", "contains"):
            op = self._next().value
        else:
            raise ValueError(f"Expected operator after field {field_tok.value!r}")
        if self._peek() is None:
            raise ValueError(f"Expected value after operator {op!r}")
        value_tok = self._next()
        return Predicate(field_tok.value, op, self._literal_value(value_tok))

    @staticmethod
    def _literal_value(tok: Token) -> Any:
        if tok.kind == "string":
            return tok.value[1:-1]
        if tok.kind == "number":
            return float(tok.value) if "." in tok.value else int(tok.value)
        if tok.kind == "word":
            low = tok.value.lower()
            if low == "true":
                return True
            if low == "false":
                return False
            return tok.value
        raise ValueError(f"Invalid literal {tok}")


def parse_query(text: str) -> Node:
    """Parse a hybrid query string into an AST."""
    return QueryParser(tokenize(text)).parse()


# ----------------------------- predicate evaluation -----------------------------

def eval_metadata(node: Node, metadata: Optional[Dict[str, Any]]) -> bool:
    """
    Evaluate the metadata-only portion of an AST against one record.
    SemanticMatch leaves are treated as True here (they are resolved by the
    vector engine, not by metadata).
    """
    if isinstance(node, Predicate):
        return node.matches(metadata)
    if isinstance(node, SemanticMatch):
        return True
    if isinstance(node, NotOp):
        return not eval_metadata(node.child, metadata)
    if isinstance(node, BoolOp):
        if node.op == "AND":
            return all(eval_metadata(c, metadata) for c in node.children)
        return any(eval_metadata(c, metadata) for c in node.children)
    raise TypeError(f"Unknown node {node}")


def to_predicate_fn(node: Node) -> Callable[[Optional[Dict[str, Any]]], bool]:
    """Compile an AST into a metadata predicate callable for index filtering."""
    return lambda metadata: eval_metadata(node, metadata)


# ----------------------------- cost-based optimizer -----------------------------

@dataclass
class QueryPlan:
    """The chosen execution strategy for a hybrid query."""
    strategy: str            # "filter_first" | "vector_first" | "filter_only" | "vector_only"
    semantic_query: Optional[str]
    semantic_k: Optional[int]
    predicate_fn: Optional[Callable[[Optional[Dict[str, Any]]], bool]]
    estimated_selectivity: float
    reason: str
    ast: Node


def collect_semantic(node: Node) -> List[SemanticMatch]:
    """Return all SemanticMatch leaves in the AST."""
    if isinstance(node, SemanticMatch):
        return [node]
    if isinstance(node, NotOp):
        return collect_semantic(node.child)
    if isinstance(node, BoolOp):
        out: List[SemanticMatch] = []
        for c in node.children:
            out.extend(collect_semantic(c))
        return out
    return []


def collect_predicates(node: Node) -> List[Predicate]:
    if isinstance(node, Predicate):
        return [node]
    if isinstance(node, NotOp):
        return collect_predicates(node.child)
    if isinstance(node, BoolOp):
        out: List[Predicate] = []
        for c in node.children:
            out.extend(collect_predicates(c))
        return out
    return []


# Heuristic selectivity per operator: fraction of corpus expected to match.
# Lower = more selective = cheaper to evaluate first.
_OP_SELECTIVITY = {
    "=": 0.05,
    "in": 0.15,
    "contains": 0.2,
    "!=": 0.9,
    "<": 0.33,
    "<=": 0.33,
    ">": 0.33,
    ">=": 0.33,
}


def estimate_selectivity(node: Node,
                         stats: Optional[Dict[str, Dict[str, Any]]] = None) -> float:
    """
    Estimate the fraction of records passing the metadata portion of the AST.
    ``stats`` may provide per-field cardinality: {field: {"distinct": N}} to
    refine equality selectivity (1/distinct).
    """
    if isinstance(node, Predicate):
        base = _OP_SELECTIVITY.get(node.op, 0.5)
        if node.op == "=" and stats and node.field in stats:
            distinct = stats[node.field].get("distinct")
            if distinct:
                base = 1.0 / max(distinct, 1)
        return base
    if isinstance(node, SemanticMatch):
        return 1.0  # semantic leaf doesn't pre-filter metadata
    if isinstance(node, NotOp):
        return max(0.0, 1.0 - estimate_selectivity(node.child, stats))
    if isinstance(node, BoolOp):
        sels = [estimate_selectivity(c, stats) for c in node.children]
        if node.op == "AND":
            prod = 1.0
            for s in sels:
                prod *= s
            return prod
        # OR: inclusion-exclusion upper bound, clamped
        combined = 0.0
        for s in sels:
            combined = combined + s - combined * s
        return min(combined, 1.0)
    return 1.0


def plan_query(text_or_ast,
               stats: Optional[Dict[str, Dict[str, Any]]] = None,
               selectivity_threshold: float = 0.1,
               default_k: int = 10) -> QueryPlan:
    """
    Build a cost-based execution plan for a hybrid query.

    Decision rule:
        - No semantic leaf  -> filter_only
        - No metadata predicate -> vector_only
        - Selective filter (<= threshold) -> filter_first (filter then rank)
        - Otherwise -> vector_first (search then post-filter)
    """
    ast = text_or_ast if isinstance(text_or_ast, Node) else parse_query(text_or_ast)

    semantics = collect_semantic(ast)
    predicates = collect_predicates(ast)
    selectivity = estimate_selectivity(ast, stats)

    sem = semantics[0] if semantics else None
    sem_query = sem.query if sem else None
    sem_k = (sem.k if sem and sem.k else default_k) if sem else None
    pred_fn = to_predicate_fn(ast) if predicates else None

    if not semantics:
        strategy, reason = "filter_only", "no semantic_match leaf; pure metadata query"
    elif not predicates:
        strategy, reason = "vector_only", "no metadata predicates; pure vector search"
    elif selectivity <= selectivity_threshold:
        strategy = "filter_first"
        reason = (f"filter selectivity {selectivity:.3f} <= {selectivity_threshold}; "
                  "filter then rank candidates")
    else:
        strategy = "vector_first"
        reason = (f"filter selectivity {selectivity:.3f} > {selectivity_threshold}; "
                  "vector search then post-filter")

    return QueryPlan(
        strategy=strategy,
        semantic_query=sem_query,
        semantic_k=sem_k,
        predicate_fn=pred_fn,
        estimated_selectivity=selectivity,
        reason=reason,
        ast=ast,
    )
