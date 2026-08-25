def confirmed_prefix_hash(token_ids: list[int]) -> int:
    """Thin utility shim for target-side confirmed-prefix cache identity."""
    return hash(tuple(token_ids))


__all__ = ["confirmed_prefix_hash"]
