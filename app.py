"""
app.py — Streamlit UI for the AI Travel Assistant.

Layout
------
Sidebar    : Structured travel input form (source, destination, dates, budget)
Main area  : Tabs for Flights and Hotels results (ranked tables)
             Follow-up form (purpose, days, mood) → Itinerary generation
             Itinerary display + PDF download button
Chat area  : Free-form follow-up Q&A with the travel agent

Session state keys
------------------
  search_complete  bool          — True once a successful search has run
  flights_raw      list[dict]    — Unranked flights from tools.py (cached)
  hotels_raw       list[dict]    — Unranked hotels from tools.py (cached)
  flights_ranked   list[dict]    — Ranked flights from ranking.py
  hotels_ranked    list[dict]    — Ranked hotels from ranking.py
  travel_agent     TravelAgent   — Stateful agent for this session
  itinerary        str           — Generated markdown itinerary
  chat_history     list[dict]    — [{role, content}, ...]
  search_params    dict          — Last used search params (for cache key)
"""

from __future__ import annotations

import datetime
import logging
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import TravelAgent
from ranking import (
    export_itinerary_pdf,
    flights_to_dataframe,
    hotels_to_dataframe,
    rank_flights,
    rank_hotels,
)
from tools import fetch_flights, fetch_hotels

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ── Streamlit Cloud secret injection ──────────────────────────────────────────
# Streamlit Cloud stores secrets in st.secrets (not env vars).
# We copy them into os.environ here so that tools.py and agent.py
# (which use os.getenv) work identically on Cloud and locally.
_SECRET_KEYS = (
    "OPENAI_API_KEY",
    "SERPAPI_API_KEY",
)
for _k in _SECRET_KEYS:
    if not os.getenv(_k):
        try:
            os.environ[_k] = st.secrets[_k]
        except (KeyError, FileNotFoundError):
            pass  # Key not configured — demo/mock mode will be used

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Travel Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialisation ──────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "search_complete": False,
        "flights_raw": [],
        "hotels_raw": [],
        "flights_ranked": [],
        "hotels_ranked": [],
        "travel_agent": None,
        "itinerary": "",
        "chat_history": [],
        "search_params": {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ── Cached search wrapper ─────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _cached_search(
    source: str,
    destination: str,
    dep_date: str,
    ret_date: str,
    budget: str,
) -> dict:
    """
    Fetch raw (unranked) flights and hotels and return them as plain dicts.

    Uses st.cache_data so repeated searches with the same parameters
    never re-hit the network.  The TravelAgent is NOT cached here because
    it is stateful; it is created fresh in session_state after the cache
    returns.
    """
    agent = TravelAgent()
    result = agent.search(source, destination, dep_date, ret_date, budget)
    return {
        "flights": result["flights"],
        "hotels": result["hotels"],
        "summary": result["summary"],
        # Store agent messages so the session agent can be pre-seeded
        "agent_messages": agent.messages,
    }


# ── Input validation ──────────────────────────────────────────────────────────

def _validate_inputs(
    source: str,
    destination: str,
    dep_date: datetime.date,
    ret_date: datetime.date,
) -> list[str]:
    """Return a list of validation error strings (empty list = all OK)."""
    errors: list[str] = []
    today = datetime.date.today()

    if not source.strip():
        errors.append("Source city cannot be empty.")
    if not destination.strip():
        errors.append("Destination city cannot be empty.")
    if source.strip().lower() == destination.strip().lower():
        errors.append("Source and destination must be different cities.")
    if dep_date < today:
        errors.append("Departure date cannot be in the past.")
    if ret_date <= dep_date:
        errors.append("Return date must be after the departure date.")

    return errors


# ── Sidebar — Input form ──────────────────────────────────────────────────────

with st.sidebar:
    st.title("✈️ AI Travel Assistant")

    # ── User-supplied API keys (session-only, never persisted) ────────────────
    # Anything typed here is held in st.session_state for the duration of this
    # browser session and copied into os.environ so tools.py / agent.py pick it
    # up. Nothing is written to disk, logged, or sent anywhere except the
    # respective upstream provider when a search runs.
    with st.expander("🔑 API Keys (optional — for live data)", expanded=False):
        st.caption(
            "Paste your own keys to run with live data. "
            "Keys are kept in memory for this session only — they are never "
            "saved to disk or logged. Leave blank to use demo/mock mode."
        )
        st.text_input("OPENAI_API_KEY", type="password", key="ui_OPENAI_API_KEY")
        st.text_input("SERPAPI_API_KEY", type="password", key="ui_SERPAPI_API_KEY")

    # Apply UI-provided keys to os.environ (overrides .env / st.secrets).
    # Invalidate the cached search if any key changed so the next search
    # actually re-runs against the new credentials.
    _keys_changed = False
    for _k in _SECRET_KEYS:
        _v = st.session_state.get(f"ui_{_k}", "").strip()
        if _v and os.environ.get(_k) != _v:
            os.environ[_k] = _v
            _keys_changed = True
    if _keys_changed:
        _cached_search.clear()

    st.markdown("---")
    st.subheader("Trip Details")

    source_city = st.text_input(
        "From (city or airport code)",
        placeholder="e.g. New York or JFK",
    )
    dest_city = st.text_input(
        "To (city or airport code)",
        placeholder="e.g. London or LHR",
    )

    today = datetime.date.today()
    dep_date = st.date_input(
        "Departure date",
        value=today + datetime.timedelta(days=7),
        min_value=today,
    )
    ret_date = st.date_input(
        "Return date",
        value=today + datetime.timedelta(days=14),
        min_value=today + datetime.timedelta(days=1),
    )

    budget = st.selectbox(
        "Budget category",
        options=["Medium", "Low", "High"],
        index=0,
        help=(
            "Low — prioritise cheapest options\n"
            "Medium — balance cost and quality\n"
            "High — comfort and convenience first"
        ),
    )

    search_clicked = st.button("🔍 Search Flights & Hotels", use_container_width=True)

    # Trigger search
    if search_clicked:
        errors = _validate_inputs(source_city, dest_city, dep_date, ret_date)
        if errors:
            for err in errors:
                st.error(err)
        else:
            dep_str = dep_date.strftime("%Y-%m-%d")
            ret_str = ret_date.strftime("%Y-%m-%d")

            with st.spinner("Searching flights and hotels…"):
                raw = _cached_search(
                    source_city.strip(),
                    dest_city.strip(),
                    dep_str,
                    ret_str,
                    budget,
                )

            # Rank results based on current budget
            st.session_state.flights_raw = raw["flights"]
            st.session_state.hotels_raw = raw["hotels"]
            st.session_state.flights_ranked = rank_flights(raw["flights"], budget)
            st.session_state.hotels_ranked = rank_hotels(raw["hotels"], budget)

            # Create a fresh session-level agent pre-seeded with search context
            agent = TravelAgent()
            agent.messages = raw["agent_messages"]
            st.session_state.travel_agent = agent

            # Store params for reference in itinerary generation
            st.session_state.search_params = {
                "source": source_city.strip(),
                "destination": dest_city.strip(),
                "dep_date": dep_str,
                "ret_date": ret_str,
                "budget": budget,
            }

            st.session_state.search_complete = True
            st.session_state.itinerary = ""
            st.session_state.chat_history = []

    st.markdown("---")
    st.caption("Powered by OpenAI · SerpAPI")


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🌍 AI Travel Assistant")

if not st.session_state.search_complete:
    st.info(
        "Fill in your trip details in the sidebar and click **Search Flights & Hotels** to get started."
    )
    st.markdown(
        """
        ### What this assistant can do for you
        - **Real-time flight search** via Google Flights (SerpAPI)
        - **Hotel recommendations** via Google Hotels (SerpAPI)
        - **Smart ranking** tailored to your budget preference
        - **Personalised itinerary generation** — day-by-day plans with meeting slots,
          café recommendations, and commute buffers
        - **AI-powered chat** for follow-up questions and refinements
        - **PDF export** of your final itinerary

        > *Works in demo mode without API keys — set them in `.env` for live results.*
        """
    )

else:
    params = st.session_state.search_params

    # Re-rank if budget changed (user tweaked the sidebar without re-searching)
    current_budget = budget
    if (
        st.session_state.flights_raw
        and st.session_state.search_params.get("budget") != current_budget
    ):
        st.session_state.flights_ranked = rank_flights(
            st.session_state.flights_raw, current_budget
        )
        st.session_state.hotels_ranked = rank_hotels(
            st.session_state.hotels_raw, current_budget
        )

    st.subheader(
        f"Results: {params['source']} → {params['destination']}  "
        f"({params['dep_date']} – {params['ret_date']}, {params['budget']} budget)"
    )

    # ── Results tabs ──────────────────────────────────────────────────────────
    tab_flights, tab_hotels = st.tabs(["🛫 Flights", "🏨 Hotels"])

    with tab_flights:
        st.markdown("#### Top 5 Flights")
        if st.session_state.flights_ranked:
            df = flights_to_dataframe(st.session_state.flights_ranked)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Book": st.column_config.LinkColumn("Book", display_text="Book →"),
                },
            )
        else:
            st.warning("No flight results found.")

    with tab_hotels:
        st.markdown("#### Top 5 Hotels")
        if st.session_state.hotels_ranked:
            df = hotels_to_dataframe(st.session_state.hotels_ranked)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Book": st.column_config.LinkColumn("Book", display_text="Book →"),
                },
            )
        else:
            st.warning("No hotel results found.")

    st.markdown("---")

    # ── Follow-up form → Itinerary ────────────────────────────────────────────
    st.subheader("📋 Generate Your Itinerary")

    with st.form("itinerary_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            purpose = st.radio(
                "Purpose of visit",
                options=["Business", "Leisure", "Mixed"],
                index=0,
            )
        with col2:
            dep = datetime.date.fromisoformat(params["dep_date"])
            ret = datetime.date.fromisoformat(params["ret_date"])
            max_days = max(1, (ret - dep).days)
            num_days = st.slider(
                "Number of days",
                min_value=1,
                max_value=max_days,
                value=min(3, max_days),
            )
        with col3:
            mood = st.radio(
                "Travel mood",
                options=["Balanced", "Relaxed", "Packed"],
                index=0,
                help=(
                    "Relaxed — 2–3 activities/day\n"
                    "Balanced — 3–4 activities/day\n"
                    "Packed — 4–6 activities/day"
                ),
            )

        generate_clicked = st.form_submit_button(
            "✨ Generate Itinerary", use_container_width=True
        )

    if generate_clicked:
        # Pick the top-ranked hotel name for the itinerary prompt
        hotel_name = ""
        if st.session_state.hotels_ranked:
            hotel_name = st.session_state.hotels_ranked[0].get("name", "")

        agent: TravelAgent = st.session_state.travel_agent

        with st.spinner("Generating your personalised itinerary…"):
            itinerary_md = agent.generate_itinerary(
                destination=params["destination"],
                num_days=num_days,
                purpose=purpose,
                mood=mood,
                hotel_name=hotel_name,
                dep_date=params["dep_date"],
                ret_date=params["ret_date"],
            )

        st.session_state.itinerary = itinerary_md

    # ── Itinerary display ─────────────────────────────────────────────────────
    if st.session_state.itinerary:
        st.markdown("---")
        st.subheader("🗓️ Your Itinerary")
        st.markdown(st.session_state.itinerary)

        # PDF export
        pdf_bytes = export_itinerary_pdf(st.session_state.itinerary)
        if pdf_bytes:
            st.download_button(
                label="📄 Export to PDF",
                data=pdf_bytes,
                file_name=f"itinerary_{params['destination'].replace(' ', '_')}.pdf",
                mime="application/pdf",
            )

    st.markdown("---")

    # ── Chat section ──────────────────────────────────────────────────────────
    st.subheader("💬 Ask the Travel Assistant")
    st.caption(
        "Refine your itinerary, ask about visa requirements, local transport, "
        "restaurant recommendations, or anything else travel-related."
    )

    # Render existing chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask anything about your trip…")
    if user_input:
        # Display the user message immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get the agent reply
        agent: TravelAgent = st.session_state.travel_agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = agent.chat(user_input)
            st.markdown(reply)

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
