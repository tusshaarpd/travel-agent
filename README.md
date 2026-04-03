# AI Travel Assistant

A production-grade AI travel planner powered by **GPT-4o-mini**, **SerpAPI (Google Flights)**, and **Booking.com MCP**. Built with Streamlit for a clean, interactive UI and a native OpenAI function-calling agent loop — no LangChain required.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.32%2B-red)
![OpenAI](https://img.shields.io/badge/openai-GPT--4o--mini-green)
![License: MIT](https://img.shields.io/badge/license-MIT-yellow)

---

## Features

- **Flight search** — queries Google Flights via SerpAPI and extracts airline, price, duration, stops, and booking link
- **Hotel search** — fetches hotels from a Booking.com MCP endpoint with price/night, rating, and distance from city centre
- **Budget-aware ranking** — scores and ranks top-5 flights and hotels using weighted algorithms tuned for Low / Medium / High budget tiers
- **Explainability** — every ranked result includes a plain-English "Why Recommended" reason
- **Itinerary generation** — day-by-day markdown plan tailored to trip purpose (Business / Leisure / Mixed) and travel mood (Relaxed / Balanced / Packed)
- **Persistent chat** — follow-up Q&A with the agent to refine results or the itinerary
- **PDF export** — download the generated itinerary as a formatted PDF
- **Demo mode** — all features work with zero API keys; realistic mock data is used as a fallback automatically

---

## Architecture

```
Streamlit UI  (app.py)
  └─→ TravelAgent  (agent.py)
        ├─→ OpenAI function-calling loop  (gpt-4o-mini)
        │     └─→ _dispatch_tool()
        │           ├─→ fetch_flights()  ──→ SerpAPI  [live]
        │           │                   └─→ mock data [fallback]
        │           └─→ fetch_hotels()  ──→ Booking.com MCP  [live]
        │                               └─→ mock data [fallback]
        └─→ rank_flights() / rank_hotels()  (ranking.py)
              ├─→ flights_to_dataframe() / hotels_to_dataframe()
              └─→ export_itinerary_pdf()
```

**Data flow:**

1. User fills the sidebar form (cities, dates, budget) and clicks **Search**
2. `app.py` validates inputs and calls `_cached_search()` (`@st.cache_data`)
3. `TravelAgent.search()` runs the OpenAI function-calling loop; the LLM issues `search_flights` and `search_hotels` tool calls
4. `tools.py` fetches raw results from live APIs (or falls back to mock data)
5. `ranking.py` scores and sorts the raw results; top-5 returned with `rank_reason`
6. Results displayed as interactive `pd.DataFrame` tables in two tabs
7. User fills the itinerary form → `TravelAgent.generate_itinerary()` → markdown rendered in the main area
8. User chats → `TravelAgent.chat()` → conversational refinement
9. User clicks **Export to PDF** → `export_itinerary_pdf()` → browser download

---

## File Structure

```
travel-agent/
├── app.py           # Streamlit UI: sidebar form, result tabs, itinerary, chat, PDF download
├── agent.py         # TravelAgent class: OpenAI function-calling agent loop
├── tools.py         # SerpAPI + Booking.com MCP HTTP clients + mock fallbacks + retry logic
├── ranking.py       # Budget-weighted scoring, DataFrame formatters, fpdf2 PDF export
├── requirements.txt # Python dependencies
└── .env.example     # Template for environment variables
```

---

## Prerequisites

- Python **3.11+**
- `pip`

---

## Installation & Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd travel-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Open .env and fill in your API keys (all optional — see below)

# 4. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Environment Variables

Copy `.env.example` to `.env` and populate as needed. **All variables are optional** — the app falls back to mock data when any key is missing.

| Variable | Required | Purpose | How to obtain |
|---|---|---|---|
| `OPENAI_API_KEY` | No* | Powers GPT-4o-mini for the agent loop, itinerary generation, and chat | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `SERPAPI_API_KEY` | No* | Live Google Flights search via SerpAPI | [serpapi.com](https://serpapi.com) |
| `BOOKING_MCP_ENDPOINT` | No* | HTTP endpoint of your Booking.com MCP deployment | Your own MCP server URL |
| `BOOKING_MCP_API_KEY` | No* | Bearer token for authenticating with the MCP endpoint | Your MCP server credentials |

\* Without these keys the app runs in **demo mode** using rich, realistic mock data.

---

## Usage Guide

### Step 1 — Search
Fill in the sidebar:
- **Source city** and **Destination city** (e.g. New York, Paris)
- **Departure date** and **Return date**
- **Budget** — Low / Medium / High (controls ranking weights)

Click **Search Flights & Hotels**. Results are cached so changing the budget tier re-ranks without hitting the APIs again.

### Step 2 — Review Results
Two tabs appear in the main area:

| Tab | Columns |
|---|---|
| **Flights** | Airline, Price (USD), Duration, Stops, Departure, Arrival, Why Recommended, Book |
| **Hotels** | Hotel, Price/Night (USD), Rating, Distance (km), Address, Why Recommended, Book |

### Step 3 — Generate Itinerary
Below the results, fill the follow-up form:
- **Purpose** — Business / Leisure / Mixed
- **Number of days** — slider
- **Travel mood** — Relaxed / Balanced / Packed

Click **Generate Itinerary**. A day-by-day markdown plan is rendered, including meeting slots (for Business), commute buffers, café/workspace recommendations, and key attractions tuned to your mood.

### Step 4 — Export to PDF
Click **Export to PDF** to download the formatted itinerary as a `.pdf` file.

### Step 5 — Refine via Chat
Use the chat input at the bottom to ask follow-up questions, request changes to the itinerary, or get additional recommendations. The agent retains full conversation context within the session.

---

## Module Reference

### `tools.py` — Data Contracts

All modules share these TypedDicts as the universal data contract:

```python
class FlightResult(TypedDict):
    airline:      str    # e.g. "Delta Airlines"
    price:        float  # USD total fare
    duration_min: int    # total travel time in minutes
    stops:        int    # 0 = non-stop
    departure:    str    # e.g. "08:30"
    arrival:      str    # e.g. "14:15"
    booking_url:  str

class HotelResult(TypedDict):
    name:             str
    price_per_night:  float  # USD
    rating:           float  # 0–10 scale
    distance_km:      float  # from city centre
    address:          str
    booking_url:      str
```

### `tools.py` — Booking.com MCP Protocol

The MCP client sends a JSON POST to `BOOKING_MCP_ENDPOINT`:

**Request:**
```json
{
  "tool": "search_hotels",
  "arguments": {
    "destination":    "Paris",
    "checkin_date":   "2025-06-01",
    "checkout_date":  "2025-06-05",
    "budget_tier":    "Medium",
    "currency":       "USD",
    "language":       "en-us"
  }
}
```

**Expected response:**
```json
{
  "result": {
    "hotels": [
      {
        "name": "Hotel Example",
        "price_per_night": 120.0,
        "rating": 8.4,
        "distance_km": 1.2,
        "address": "12 Rue de Rivoli, Paris",
        "booking_url": "https://booking.com/..."
      }
    ]
  }
}
```

### `ranking.py` — Scoring Weights

All scores use **min-max normalised** values in `[0, 1]` before combining.

**Flights:**

| Budget | Price weight | Duration weight | Stops weight |
|---|---|---|---|
| Low | 100% | 0% | 0% |
| Medium | 50% | 30% | 20% |
| High | 10% | 30% | 60% |

**Hotels:**

Hotels with `rating < 7.0` are filtered to the back unless fewer than `top_n` qualify above the threshold.

| Budget | Price weight | Rating weight | Distance weight |
|---|---|---|---|
| Low | 100% | 0% | 0% |
| Medium | 50% | 50% | 0% |
| High | 0% | 70% | 30% |

### `agent.py` — Temperature Strategy

| Turn type | Temperature | Rationale |
|---|---|---|
| Search / tool calling | `0.3` | Deterministic argument generation for tools |
| Itinerary generation | `0.7` | Creative, varied day plans |
| Follow-up chat | `0.5` | Balanced, conversational responses |

### `agent.py` — OpenAI Tool Definitions

Two functions are registered with the OpenAI API:

| Tool | Parameters |
|---|---|
| `search_flights` | `source`, `destination`, `date` (YYYY-MM-DD), `budget` (Low\|Medium\|High) |
| `search_hotels` | `destination`, `checkin` (YYYY-MM-DD), `checkout` (YYYY-MM-DD), `budget` |

---

## Caching Strategy

| Layer | Mechanism | Scope |
|---|---|---|
| Network search results | `@st.cache_data` on `_cached_search()` | Per Streamlit session; keyed on `(source, dest, dep_date, ret_date, budget)` |
| API fetch functions | `@functools.lru_cache(maxsize=32)` on `fetch_flights()` / `fetch_hotels()` | In-process; avoids duplicate API calls within the same run |

Changing the **budget** tier re-ranks cached data without re-calling any external API.

---

## Retry & Fallback Policy

| Scenario | Behaviour |
|---|---|
| API call fails | Retry up to **3 times** with exponential backoff: 2 s → 4 s → 8 s |
| All retries exhausted | Silently fall back to 8-entry mock dataset |
| `OPENAI_API_KEY` not set | `_mock_llm_response()` calls tools directly and returns a demo-mode message |
| Agent loop takes too long | Hard limit of **10 iterations**; returns partial result with a warning |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.32.0 | Web UI framework |
| `openai` | ≥ 1.0.0 | LLM and function-calling API client |
| `pandas` | ≥ 2.0.0 | DataFrames for ranking and display |
| `httpx` | ≥ 0.27.0 | HTTP client for Booking.com MCP |
| `requests` | ≥ 2.31.0 | HTTP client for SerpAPI |
| `python-dotenv` | ≥ 1.0.0 | Load `.env` file at startup |
| `fpdf2` | ≥ 2.7.0 | PDF generation for itinerary export |
| `google-search-results` | ≥ 2.4.2 | SerpAPI Python wrapper |

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Project Decisions

**No LangChain** — The agent uses a hand-rolled OpenAI function-calling loop (`_run_agent_loop`) with a simple `while` iteration. This keeps the dependency footprint small, makes the control flow transparent, and avoids LangChain version-compatibility churn.

**Ranking outside the LLM** — Raw results flow from `tools.py` → `agent.py` → `app.py` as plain Python lists. `ranking.py` applies scoring after the cached search returns. This means:
- The cache stores budget-independent raw data
- Switching budget re-ranks instantly without a new API call
- Scoring weights can be tuned without touching the agent

**Mock-first design** — Every external call has a deterministic fallback. This makes the app immediately runnable and testable without any API credentials, and prevents hard failures in production when a third-party service is unavailable.

**Stateful agent per session** — One `TravelAgent` instance is stored in `st.session_state`. It accumulates `messages` history so that follow-up chat turns have full context of the original search and generated itinerary.

---

## License

MIT
