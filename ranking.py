"""
ranking.py — Scoring and ranking logic for flights and hotels.

Ranking is intentionally kept outside the agent so that:
  1. The same cached (unranked) data can be re-ranked instantly when the
     user changes their budget tier without re-hitting the network.
  2. The scoring weights are transparent and easy to tune independently
     of any LLM behaviour.

Each ranked result gains a `rank_reason` key with a short human-readable
explanation of why it was placed in that position — this is the
"explainability" feature shown in the UI tables.

Also provides export_itinerary_pdf() for one-click PDF downloads.
"""

from __future__ import annotations

import io
import logging
import re
import textwrap
from typing import Any

import pandas as pd

from tools import FlightResult, HotelResult

logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _minmax(series: pd.Series) -> pd.Series:
    """Min-max normalise a pandas Series to [0, 1]. Safe when all values equal."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - mn) / (mx - mn)


# ── Flight ranking ────────────────────────────────────────────────────────────

def rank_flights(
    flights: list[FlightResult],
    budget: str,  # "Low" | "Medium" | "High"
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    Rank flights by budget preference and return the top_n with an added
    `rank_reason` explainability string.

    Scoring weights:
      Low    — 100 % price (cheapest wins)
      Medium — 50 % price, 30 % duration, 20 % stops
      High   — 10 % price, 30 % duration, 60 % stops (comfort first)

    Lower normalised price / duration / stops = better score.
    """
    if not flights:
        return []

    df = pd.DataFrame(flights)

    # Normalise the three numeric dimensions
    df["price_n"]    = _minmax(df["price"])
    df["duration_n"] = _minmax(df["duration_min"])
    df["stops_n"]    = _minmax(df["stops"].astype(float))

    budget_lower = budget.lower()

    if budget_lower == "low":
        df["score"] = df["price_n"]
        reason_fn = lambda r: (
            f"Cheapest option at ${r['price']:.0f} — prioritised for low budget"
        )
    elif budget_lower == "high":
        df["score"] = 0.1 * df["price_n"] + 0.3 * df["duration_n"] + 0.6 * df["stops_n"]
        reason_fn = lambda r: (
            f"{'Non-stop' if r['stops'] == 0 else f\"{r['stops']}-stop\"} flight "
            f"({r['duration_min'] // 60}h {r['duration_min'] % 60}m) at ${r['price']:.0f} "
            f"— comfort-prioritised for high budget"
        )
    else:  # Medium
        df["score"] = 0.5 * df["price_n"] + 0.3 * df["duration_n"] + 0.2 * df["stops_n"]
        reason_fn = lambda r: (
            f"${r['price']:.0f}, {r['duration_min'] // 60}h {r['duration_min'] % 60}m, "
            f"{'non-stop' if r['stops'] == 0 else f\"{r['stops']} stop(s)\"} "
            f"— best price/duration balance"
        )

    df = df.sort_values("score").head(top_n).reset_index(drop=True)

    results: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        record = row.to_dict()
        record["rank_reason"] = reason_fn(record)
        # Drop internal normalised columns
        for col in ("price_n", "duration_n", "stops_n", "score"):
            record.pop(col, None)
        results.append(record)

    return results


def flights_to_dataframe(ranked_flights: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert ranked flight dicts to a display-ready DataFrame."""
    if not ranked_flights:
        return pd.DataFrame()

    rows = []
    for f in ranked_flights:
        rows.append({
            "Airline":          f["airline"],
            "Price (USD)":      f"${f['price']:.0f}",
            "Duration":         f"{f['duration_min'] // 60}h {f['duration_min'] % 60}m",
            "Stops":            "Non-stop" if f["stops"] == 0 else str(f["stops"]),
            "Departure":        f["departure"],
            "Arrival":          f["arrival"],
            "Why Recommended":  f.get("rank_reason", ""),
            "Book":             f["booking_url"],
        })
    return pd.DataFrame(rows)


# ── Hotel ranking ─────────────────────────────────────────────────────────────

def rank_hotels(
    hotels: list[HotelResult],
    budget: str,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    Rank hotels by budget preference and return the top_n.

    Pre-filter: prefer hotels with rating >= 7.0. If fewer than top_n
    qualify, the filter is relaxed to include all hotels.

    Scoring weights:
      Low    — 100 % price (cheapest wins)
      Medium — 50 % rating (higher better), 50 % price (lower better)
      High   — 70 % rating, 30 % distance from centre (closer better)
    """
    if not hotels:
        return []

    df = pd.DataFrame(hotels)

    # Prefer well-rated hotels; relax filter if not enough options
    filtered = df[df["rating"] >= 7.0]
    if len(filtered) < top_n:
        filtered = df

    # Normalise dimensions (for price + distance: lower is better → invert)
    filtered = filtered.copy()
    filtered["price_n"]    = _minmax(filtered["price_per_night"])
    filtered["rating_n"]   = _minmax(filtered["rating"])
    filtered["distance_n"] = _minmax(filtered["distance_km"])

    budget_lower = budget.lower()

    if budget_lower == "low":
        filtered["score"] = filtered["price_n"]
        reason_fn = lambda r: (
            f"Most affordable at ${r['price_per_night']:.0f}/night "
            f"(rating {r['rating']:.1f})"
        )
    elif budget_lower == "high":
        filtered["score"] = (
            -0.7 * filtered["rating_n"] + 0.3 * filtered["distance_n"]
        )
        reason_fn = lambda r: (
            f"Top-rated ({r['rating']:.1f}/10), "
            f"{r['distance_km']:.1f} km from centre at ${r['price_per_night']:.0f}/night"
        )
    else:  # Medium
        filtered["score"] = (
            -0.5 * filtered["rating_n"] + 0.5 * filtered["price_n"]
        )
        reason_fn = lambda r: (
            f"${r['price_per_night']:.0f}/night, rating {r['rating']:.1f}/10 "
            f"— best value balance"
        )

    filtered = filtered.sort_values("score").head(top_n).reset_index(drop=True)

    results: list[dict[str, Any]] = []
    for _, row in filtered.iterrows():
        record = row.to_dict()
        record["rank_reason"] = reason_fn(record)
        for col in ("price_n", "rating_n", "distance_n", "score"):
            record.pop(col, None)
        results.append(record)

    return results


def hotels_to_dataframe(ranked_hotels: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert ranked hotel dicts to a display-ready DataFrame."""
    if not ranked_hotels:
        return pd.DataFrame()

    rows = []
    for h in ranked_hotels:
        rows.append({
            "Hotel":             h["name"],
            "Price/Night (USD)": f"${h['price_per_night']:.0f}",
            "Rating":            f"{h['rating']:.1f} / 10",
            "Distance (km)":     f"{h['distance_km']:.1f} km",
            "Address":           h["address"],
            "Why Recommended":   h.get("rank_reason", ""),
            "Book":              h["booking_url"],
        })
    return pd.DataFrame(rows)


# ── PDF export ────────────────────────────────────────────────────────────────

def export_itinerary_pdf(itinerary_md: str) -> bytes:
    """
    Convert a markdown itinerary string to a PDF and return the raw bytes.

    Uses fpdf2.  Handles:
      # H1 headings  → large bold
      ## H2 headings → medium bold
      **bold** text  → bold inline (simplified: whole line rendered bold)
      Plain lines    → regular body text
      Empty lines    → vertical spacing

    Returns bytes suitable for st.download_button(data=...).
    """
    try:
        from fpdf import FPDF  # fpdf2
    except ImportError:
        logger.error("fpdf2 not installed; cannot generate PDF.")
        return b""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", style="B", size=18)
    pdf.cell(0, 10, "Travel Itinerary", ln=True, align="C")
    pdf.ln(4)

    for raw_line in itinerary_md.splitlines():
        line = raw_line.rstrip()

        if line.startswith("# "):
            pdf.set_font("Helvetica", style="B", size=15)
            pdf.ln(4)
            pdf.multi_cell(0, 8, line[2:])
            pdf.ln(2)

        elif line.startswith("## "):
            pdf.set_font("Helvetica", style="B", size=13)
            pdf.ln(3)
            pdf.multi_cell(0, 7, line[3:])
            pdf.ln(1)

        elif line.startswith("### "):
            pdf.set_font("Helvetica", style="B", size=11)
            pdf.multi_cell(0, 6, line[4:])

        elif not line:
            pdf.ln(4)

        else:
            # Strip markdown bold markers for plain rendering
            clean = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
            clean = re.sub(r"\*(.*?)\*", r"\1", clean)
            # Bullet list items
            if clean.startswith("- "):
                clean = "\u2022 " + clean[2:]
            is_bold = "**" in line
            pdf.set_font("Helvetica", style="B" if is_bold else "", size=10)
            # Wrap long lines
            for chunk in textwrap.wrap(clean, width=100) or [""]:
                pdf.multi_cell(0, 5, chunk)

    return bytes(pdf.output())
