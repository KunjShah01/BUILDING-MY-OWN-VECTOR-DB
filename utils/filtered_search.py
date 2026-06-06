"""
Filtered search utilities for HNSW metadata-aware graph traversal.

Instead of post-filtering search results (which can return fewer than k matches),
these predicates are applied *during* graph traversal via HNSWIndex.search()'s
metadata_filter parameter.

Usage:
    from utils.filtered_search import Filter

    # Simple equality
    f = Filter.eq("category", "tech")

    # Compound filter
    f = Filter.eq("category", "tech") & Filter.gt("score", 0.5)

    # Use in search
    results = hnsw.search(query, k=10, metadata_filter=f)
"""
from typing import Dict, Any, Optional, List, Callable


class FilterPredicate:
    """
    Composable metadata filter predicate.

    Each predicate wraps a callable that takes a metadata dict (or None)
    and returns True/False.  Predicates can be combined with & (AND) and
    | (OR) operators.
    """

    def __init__(self, fn: Callable[[Optional[Dict[str, Any]]], bool]):
        self._fn = fn

    def __call__(self, metadata: Optional[Dict[str, Any]]) -> bool:
        if metadata is None:
            return False
        return self._fn(metadata)

    def __and__(self, other: 'FilterPredicate') -> 'FilterPredicate':
        """Combine two filters with AND logic."""
        left, right = self._fn, other._fn
        return FilterPredicate(lambda m: m is not None and left(m) and right(m))

    def __or__(self, other: 'FilterPredicate') -> 'FilterPredicate':
        """Combine two filters with OR logic."""
        left, right = self._fn, other._fn
        return FilterPredicate(lambda m: m is not None and (left(m) or right(m)))

    def __invert__(self) -> 'FilterPredicate':
        """Negate a filter."""
        fn = self._fn
        return FilterPredicate(lambda m: m is not None and not fn(m))


class Filter:
    """
    Builder API for creating filter predicates.

    Examples:
        Filter.eq("status", "active")
        Filter.gt("score", 0.8) & Filter.in_("tag", ["ml", "nlp"])
        Filter.exists("embedding_version")
    """

    @staticmethod
    def eq(key: str, value: Any) -> FilterPredicate:
        """Metadata[key] == value"""
        return FilterPredicate(lambda m: m.get(key) == value)

    @staticmethod
    def neq(key: str, value: Any) -> FilterPredicate:
        """Metadata[key] != value"""
        return FilterPredicate(lambda m: m.get(key) != value)

    @staticmethod
    def gt(key: str, value: float) -> FilterPredicate:
        """Metadata[key] > value"""
        def _gt(m: Dict) -> bool:
            v = m.get(key)
            return v is not None and v > value
        return FilterPredicate(_gt)

    @staticmethod
    def gte(key: str, value: float) -> FilterPredicate:
        """Metadata[key] >= value"""
        def _gte(m: Dict) -> bool:
            v = m.get(key)
            return v is not None and v >= value
        return FilterPredicate(_gte)

    @staticmethod
    def lt(key: str, value: float) -> FilterPredicate:
        """Metadata[key] < value"""
        def _lt(m: Dict) -> bool:
            v = m.get(key)
            return v is not None and v < value
        return FilterPredicate(_lt)

    @staticmethod
    def lte(key: str, value: float) -> FilterPredicate:
        """Metadata[key] <= value"""
        def _lte(m: Dict) -> bool:
            v = m.get(key)
            return v is not None and v <= value
        return FilterPredicate(_lte)

    @staticmethod
    def in_(key: str, values: List[Any]) -> FilterPredicate:
        """Metadata[key] is in the given list of values."""
        value_set = set(values)
        return FilterPredicate(lambda m: m.get(key) in value_set)

    @staticmethod
    def contains(key: str, substring: str) -> FilterPredicate:
        """Metadata[key] contains substring (for string values)."""
        def _contains(m: Dict) -> bool:
            v = m.get(key)
            return isinstance(v, str) and substring in v
        return FilterPredicate(_contains)

    @staticmethod
    def exists(key: str) -> FilterPredicate:
        """Metadata has the given key (and it's not None)."""
        return FilterPredicate(lambda m: m.get(key) is not None)

    @staticmethod
    def list_contains(key: str, item: Any) -> FilterPredicate:
        """Metadata[key] is a list containing item."""
        def _list_contains(m: Dict) -> bool:
            v = m.get(key)
            return isinstance(v, (list, tuple, set)) and item in v
        return FilterPredicate(_list_contains)

    @staticmethod
    def from_dict(filter_dict: Dict[str, Any]) -> Optional[FilterPredicate]:
        """
        Build a filter from a simple {key: value} dictionary.
        Each entry becomes an equality check; all are ANDed together.
        Entries with dict values are treated as operator specs:
            {"score": {"$gt": 0.5}, "category": "tech"}

        Supported operators: $gt, $gte, $lt, $lte, $ne, $in, $contains, $exists

        Returns None if filter_dict is empty.
        """
        if not filter_dict:
            return None

        predicates: List[FilterPredicate] = []
        operator_map = {
            "$gt": Filter.gt,
            "$gte": Filter.gte,
            "$lt": Filter.lt,
            "$lte": Filter.lte,
            "$ne": Filter.neq,
            "$in": Filter.in_,
            "$contains": Filter.contains,
        }

        for key, value in filter_dict.items():
            if isinstance(value, dict):
                for op, op_value in value.items():
                    if op == "$exists":
                        if op_value:
                            predicates.append(Filter.exists(key))
                        else:
                            predicates.append(~Filter.exists(key))
                    elif op in operator_map:
                        predicates.append(operator_map[op](key, op_value))
            else:
                predicates.append(Filter.eq(key, value))

        if not predicates:
            return None

        result = predicates[0]
        for p in predicates[1:]:
            result = result & p
        return result
