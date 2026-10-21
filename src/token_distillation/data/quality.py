"""Admission of actual source facts, distinct from configured input units."""
from __future__ import annotations


def validate_raw_quality(frame, *, verified_sources) -> dict:
    """Fail closed before treating a raw-price matrix as an admitted sample."""
    required = {"source", "has_trade_price"}
    if not required.issubset(frame.columns):
        raise ValueError("Raw input lacks source/trade-price verification")
    attrs = frame.attrs
    if (attrs.get("price_basis") != "unadjusted" or attrs.get("volume_unit") != "shares"
            or attrs.get("amount_unit") != "CNY"):
        raise ValueError("Actual raw input price basis or volume/amount units are unverified")
    if not frame["source"].isin(frozenset(verified_sources)).all():
        raise ValueError("Raw input includes an unverified or adjusted source")
    if not frame["has_trade_price"].eq(1).all():
        raise ValueError("Raw input includes unverified trade prices")
    if ((frame["volume"] <= 0) | (frame["amount"] <= 0)).any():
        raise ValueError("Suspended/zero-turnover sessions are not admitted by the initial complete-window contract")
    return {"eligible": True, "price_basis": attrs["price_basis"],
            "volume_unit": attrs["volume_unit"], "amount_unit": attrs["amount_unit"],
            "source_values": sorted(frame["source"].unique().tolist()),
            "has_trade_price": "all_verified", "formal_qualified": False}
