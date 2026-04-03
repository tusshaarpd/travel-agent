"""
agent.py — LLM orchestration layer for the AI Travel Assistant.

Architecture
------------
TravelAgent wraps an OpenAI client and maintains a conversation history
list that grows across calls, enabling coherent multi-turn chat.

The agent uses OpenAI's native function-calling (tool_choice="auto") so
the LLM autonomously decides:
  • WHEN to invoke search_flights / search_hotels
  • WHAT arguments to pass (derived from the user's natural-language input)
  • HOW to combine and narrate the results

No LangChain or other orchestration framework is used — the agent loop is
a straightforward while-loop capped at MAX_ITERATIONS.

Public interface
----------------
  agent = TravelAgent()
  result = agent.search(source, destination, dep_date, ret_date, budget)
      → {"flights": [...], "hotels": [...], "summary": str}

  itinerary_md = agent.generate_itinerary(destination, num_days, purpose, mood,
                                           hotel_name, flights, hotels)
      → markdown string

  reply = agent.chat(user_message)
      → plain text assistant reply

Temperature strategy
--------------------
  0.3 — search / tool-calling turns  (want deterministic tool argument generation)
  0.7 — itinerary generation         (want creative, varied day plans)
  0.5 — follow-up chat               (balance coherence and variety)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from tools import fetch_flights, fetch_hotels

load_dotenv()

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 10  # safety cap on the agent loop

# ── OpenAI tool definitions ───────────────────────────────────────────────────
# These are passed verbatim to the OpenAI API. The LLM uses them to decide
# which function to call and what arguments to supply.

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": (
                "Search for available flights between two cities on a given date. "
                "Returns a list of flight options with airline, price, duration, stops, "
                "and booking URL. Always call this before recommending flights."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Departure city or airport code (e.g. 'New York' or 'JFK')",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination city or airport code (e.g. 'London' or 'LHR')",
                    },
                    "date": {
                        "type": "string",
                        "description": "Departure date in YYYY-MM-DD format",
                    },
                    "budget": {
                        "type": "string",
                        "enum": ["Low", "Medium", "High"],
                        "description": "Traveller's budget preference",
                    },
                },
                "required": ["source", "destination", "date", "budget"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": (
                "Search for available hotels in a destination city. "
                "Returns a list of hotels with name, price per night, rating, "
                "distance from city centre, and booking URL. "
                "Always call this before recommending hotels."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "Destination city name (e.g. 'London')",
                    },
                    "checkin": {
                        "type": "string",
                        "description": "Check-in date in YYYY-MM-DD format",
                    },
                    "checkout": {
                        "type": "string",
                        "description": "Check-out date in YYYY-MM-DD format",
                    },
                    "budget": {
                        "type": "string",
                        "enum": ["Low", "Medium", "High"],
                        "description": "Traveller's budget preference",
                    },
                },
                "required": ["destination", "checkin", "checkout", "budget"],
            },
        },
    },
]

# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_SEARCH = """You are an expert AI travel concierge. Your job is to help
business travellers find the best flights and hotels for their trip.

When asked to search for travel options you MUST:
1. Call search_flights to get live flight data.
2. Call search_hotels to get live hotel data.
3. Only after both tool calls return results, write a brief (3-4 sentence)
   summary of what you found, highlighting the standout options.

Never make up flights or hotels. Always use the tool results."""

SYSTEM_PROMPT_ITINERARY = """You are a seasoned travel planner specialising in
business trips. Generate detailed, practical itineraries in clean markdown.

Structure every itinerary as:
  ## Day N — <date or day name>
  - **Morning (09:00–12:00):** ...
  - **Afternoon (13:00–17:00):** ...
  - **Evening (18:00–21:00):** ...

For business trips always include:
  - Morning meeting slots with 30-minute buffer before/after
  - Recommended co-working spaces or business-friendly cafes
  - Commute time estimates between venues

Adjust density to the travel mood:
  Relaxed  → 2–3 activities per day max, long breaks
  Packed   → 4–6 activities per day, tight but realistic
  Balanced → 3–4 activities per day, one leisure slot each afternoon

End with a "## Practical Tips" section with local transport advice."""

SYSTEM_PROMPT_CHAT = """You are an AI travel assistant helping a user refine their
travel plans. You have access to the flights, hotels, and itinerary already generated
in this conversation. Answer questions helpfully and concisely. If the user asks you
to change the itinerary, rewrite the relevant days only."""


# ── TravelAgent ───────────────────────────────────────────────────────────────

class TravelAgent:
    """
    Stateful travel agent that maintains conversation history across calls.

    One instance is created per Streamlit session (stored in st.session_state)
    so the LLM retains context across the search → itinerary → chat flow.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning(
                "OPENAI_API_KEY not set. LLM calls will fail; mock summaries will be used."
            )
        self.client = OpenAI(api_key=api_key or "sk-placeholder")
        self.model = model
        self.messages: list[ChatCompletionMessageParam] = []
        # Cache raw results so itinerary generation can reference them
        self._last_flights: list[dict] = []
        self._last_hotels: list[dict] = []

    # ── Public methods ────────────────────────────────────────────────────────

    def search(
        self,
        source: str,
        destination: str,
        dep_date: str,
        ret_date: str,
        budget: str,
    ) -> dict[str, Any]:
        """
        Trigger the agent to search for flights and hotels.

        The agent loop will automatically call search_flights and search_hotels
        (via OpenAI function calling), then produce a summary.

        Returns:
            {
              "flights":  list[FlightResult],
              "hotels":   list[HotelResult],
              "summary":  str,
            }
        """
        # Reset conversation for a fresh search
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT_SEARCH}]

        user_msg = (
            f"I need to travel from {source} to {destination}. "
            f"Departure: {dep_date}, Return: {ret_date}. Budget tier: {budget}. "
            "Please search for the best flights and hotels available."
        )

        summary = self._run_agent_loop(user_msg, temperature=0.3)

        return {
            "flights": self._last_flights,
            "hotels": self._last_hotels,
            "summary": summary,
        }

    def generate_itinerary(
        self,
        destination: str,
        num_days: int,
        purpose: str,   # "Business" | "Leisure" | "Mixed"
        mood: str,      # "Relaxed" | "Packed" | "Balanced"
        hotel_name: str,
        dep_date: str,
        ret_date: str,
    ) -> str:
        """
        Generate a day-by-day itinerary in markdown.

        Injects a structured prompt into the existing conversation so the LLM
        can reference the flights/hotels it already retrieved.
        """
        # Switch to itinerary system prompt while preserving search history
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0] = {"role": "system", "content": SYSTEM_PROMPT_ITINERARY}
        else:
            self.messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT_ITINERARY})

        # Build a rich context message
        hotel_snippet = f"staying at {hotel_name}" if hotel_name else "at a suitable hotel"
        user_msg = (
            f"Generate a {num_days}-day itinerary for {destination} "
            f"({dep_date} to {ret_date}), {hotel_snippet}. "
            f"Purpose: {purpose}. Travel mood: {mood}. "
            "Include day-wise breakdown, meeting slots (if business), commute buffers, "
            "recommended cafes / co-working spaces, and key places to visit. "
            "Format in clean markdown with Day 1, Day 2, etc."
        )

        return self._run_agent_loop(user_msg, temperature=0.7)

    def chat(self, user_message: str) -> str:
        """
        Handle a follow-up chat message from the user.

        Switches to the conversational system prompt so the LLM can draw on
        the full search + itinerary history already in self.messages.
        """
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0] = {"role": "system", "content": SYSTEM_PROMPT_CHAT}
        else:
            self.messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT_CHAT})

        return self._run_agent_loop(user_message, temperature=0.5)

    # ── Private methods ───────────────────────────────────────────────────────

    def _run_agent_loop(self, user_message: str, temperature: float = 0.5) -> str:
        """
        Core agentic loop.

        Appends user_message then iterates:
          1. Call LLM
          2. If tool_calls → dispatch each tool → append results → loop
          3. If no tool_calls → return the assistant's text response

        Capped at MAX_ITERATIONS to prevent runaway loops.
        """
        self.messages.append({"role": "user", "content": user_message})

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            # Graceful degradation: return a mock summary without calling OpenAI
            mock_reply = self._mock_llm_response(user_message)
            self.messages.append({"role": "assistant", "content": mock_reply})
            return mock_reply

        for iteration in range(MAX_ITERATIONS):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    temperature=temperature,
                )
            except Exception as exc:
                logger.error("OpenAI API call failed: %s", exc)
                fallback = (
                    "I encountered an error reaching the AI service. "
                    "Please check your OPENAI_API_KEY and try again."
                )
                return fallback

            assistant_msg = response.choices[0].message

            # No tool calls → final text answer
            if not assistant_msg.tool_calls:
                content = assistant_msg.content or ""
                self.messages.append({"role": "assistant", "content": content})
                return content

            # Append assistant's tool-call message (must preserve tool_calls field)
            self.messages.append(assistant_msg.model_dump(exclude_unset=True))

            # Dispatch each tool call and feed results back
            for tool_call in assistant_msg.tool_calls:
                tool_result = self._dispatch_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments),
                )
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result),
                })

        logger.warning("Agent loop hit MAX_ITERATIONS (%d) — returning partial result", MAX_ITERATIONS)
        return "Search completed. Please review the results above."

    def _dispatch_tool(self, name: str, args: dict[str, Any]) -> Any:
        """
        Route a function name from the LLM to the appropriate Python function.

        Results are stored in instance variables so search() can return them
        alongside the LLM summary.
        """
        if name == "search_flights":
            flights = fetch_flights(
                source=args.get("source", ""),
                destination=args.get("destination", ""),
                date=args.get("date", ""),
                budget=args.get("budget", "Medium"),
            )
            self._last_flights = list(flights)
            return {"flights": flights, "count": len(flights)}

        if name == "search_hotels":
            hotels = fetch_hotels(
                destination=args.get("destination", ""),
                checkin=args.get("checkin", ""),
                checkout=args.get("checkout", ""),
                budget=args.get("budget", "Medium"),
            )
            self._last_hotels = list(hotels)
            return {"hotels": hotels, "count": len(hotels)}

        logger.error("Unknown tool called: %s", name)
        return {"error": f"Unknown tool: {name}"}

    def _mock_llm_response(self, user_message: str) -> str:
        """
        Return a helpful placeholder when OPENAI_API_KEY is not set.
        The app still shows real (or mock) flight/hotel data from tools.py.
        """
        # If this looks like a search request, trigger the tools directly
        if any(kw in user_message.lower() for kw in ("travel", "flight", "hotel", "search")):
            # Parse the message heuristically to call tools
            words = user_message.split()
            for i, w in enumerate(words):
                if w.lower() == "from" and i + 1 < len(words):
                    source = words[i + 1]
                    break
            else:
                source = "origin"
            for i, w in enumerate(words):
                if w.lower() == "to" and i + 1 < len(words):
                    dest = words[i + 1].rstrip(".")
                    break
            else:
                dest = "destination"

            self._last_flights = list(fetch_flights(source, dest, "2025-01-01", "Medium"))
            self._last_hotels = list(fetch_hotels(dest, "2025-01-01", "2025-01-07", "Medium"))

            return (
                f"**Demo Mode** — OPENAI_API_KEY is not set.\n\n"
                f"Found {len(self._last_flights)} sample flights and "
                f"{len(self._last_hotels)} sample hotels. "
                "Set OPENAI_API_KEY in your .env file for AI-powered recommendations and itinerary generation."
            )

        return (
            "**Demo Mode** — set OPENAI_API_KEY in your .env file to enable "
            "AI-powered responses. The travel search and ranking features still work "
            "with or without an API key."
        )
