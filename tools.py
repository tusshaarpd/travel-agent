"""
tools.py — External API clients for the AI Travel Assistant.

Provides two public fetch functions:
  - fetch_flights()  → SerpAPI / Google Flights
  - fetch_hotels()   → SerpAPI / Google Hotels

Both functions fall back to rich mock data when API keys are absent or
the upstream service is unreachable, so the app remains fully functional
in demo / development mode without any credentials.

Retry logic uses simple exponential back-off (2s → 4s → 8s).
Results are cached in-process with functools.lru_cache so repeated
searches with identical parameters never hit the network twice.
"""

from __future__ import annotations

import functools
import logging
import os
import time
from typing import TypedDict

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Shared data contracts ─────────────────────────────────────────────────────

class FlightResult(TypedDict):
    airline: str
    price: float          # USD
    duration_min: int     # total travel time in minutes
    stops: int            # 0 = non-stop
    departure: str        # e.g. "08:30"
    arrival: str          # e.g. "14:15"
    booking_url: str


class HotelResult(TypedDict):
    name: str
    price_per_night: float   # USD
    rating: float            # 0–10
    distance_km: float       # from city centre
    address: str
    booking_url: str


# ── Retry helper ──────────────────────────────────────────────────────────────

def _with_retry(fn, *args, retries: int = 3, **kwargs):
    """Call fn(*args, **kwargs) up to `retries` times with exponential back-off."""
    delay = 2
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("Attempt %d/%d failed: %s", attempt + 1, retries, exc)
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
    raise last_exc  # type: ignore[misc]


# ── Flight search ─────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=32)
def fetch_flights(
    source: str,
    destination: str,
    date: str,      # "YYYY-MM-DD"
    budget: str,    # "Low" | "Medium" | "High"
) -> list[FlightResult]:
    """
    Search for available flights.

    Attempts SerpAPI Google Flights first; falls back to mock data
    if SERPAPI_API_KEY is missing or the request fails.

    Returns a list of FlightResult dicts (unranked; ranking is in ranking.py).
    """
    api_key = os.getenv("SERPAPI_API_KEY", "")
    if not api_key:
        logger.info("SERPAPI_API_KEY not set — using mock flight data.")
        return _mock_flights(source, destination)

    try:
        return _with_retry(_serpapi_flights, source, destination, date, api_key)
    except Exception as exc:
        logger.error("SerpAPI flight search failed: %s — falling back to mocks", exc)
        return _mock_flights(source, destination)


def _serpapi_flights(
    source: str,
    destination: str,
    date: str,
    api_key: str,
) -> list[FlightResult]:
    """Hit the SerpAPI Google Flights endpoint and parse results."""
    params = {
        "engine": "google_flights",
        "departure_id": source,
        "arrival_id": destination,
        "outbound_date": date,
        "currency": "USD",
        "hl": "en",
        "api_key": api_key,
    }
    response = requests.get(
        "https://serpapi.com/search",
        params=params,
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    flights: list[FlightResult] = []
    # SerpAPI returns best_flights and other_flights arrays
    for section in ("best_flights", "other_flights"):
        for item in data.get(section, []):
            # Each item may contain multiple legs; pick the first airline
            legs = item.get("flights", [{}])
            airline = legs[0].get("airline", "Unknown Airline")
            # Duration is in minutes in the top-level item
            duration = item.get("total_duration", 0)
            departure = legs[0].get("departure_airport", {}).get("time", "N/A")
            arrival = legs[-1].get("arrival_airport", {}).get("time", "N/A")
            stops = max(0, len(legs) - 1)
            price = float(item.get("price", 0))
            link = item.get("booking_token", "")
            if link:
                link = f"https://www.google.com/flights?token={link}"

            flights.append(
                FlightResult(
                    airline=airline,
                    price=price,
                    duration_min=duration,
                    stops=stops,
                    departure=departure,
                    arrival=arrival,
                    booking_url=link,
                )
            )

    return flights if flights else _mock_flights(source, destination)


def _mock_flights(source: str, destination: str) -> list[FlightResult]:
    """
    Return 8 realistic-looking mock flights between source and destination.
    Used when SERPAPI_API_KEY is absent or the API call fails.
    """
    src = source[:3].upper()
    dst = destination[:3].upper()
    base_url = f"https://www.google.com/flights?q={src}+to+{dst}"

    return [
        FlightResult(airline="Delta Airlines",      price=312.0,  duration_min=185, stops=0, departure="06:00", arrival="09:05", booking_url=base_url),
        FlightResult(airline="United Airlines",     price=289.0,  duration_min=195, stops=0, departure="08:30", arrival="11:45", booking_url=base_url),
        FlightResult(airline="American Airlines",   price=275.0,  duration_min=200, stops=0, departure="10:00", arrival="13:20", booking_url=base_url),
        FlightResult(airline="Southwest Airlines",  price=198.0,  duration_min=240, stops=1, departure="05:45", arrival="10:05", booking_url=base_url),
        FlightResult(airline="JetBlue Airways",     price=220.0,  duration_min=215, stops=0, departure="13:30", arrival="17:05", booking_url=base_url),
        FlightResult(airline="Alaska Airlines",     price=265.0,  duration_min=205, stops=0, departure="16:00", arrival="19:25", booking_url=base_url),
        FlightResult(airline="Spirit Airlines",     price=149.0,  duration_min=295, stops=2, departure="04:30", arrival="09:25", booking_url=base_url),
        FlightResult(airline="Frontier Airlines",   price=163.0,  duration_min=270, stops=1, departure="07:15", arrival="11:45", booking_url=base_url),
    ]


# ── Hotel search ──────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=32)
def fetch_hotels(
    destination: str,
    checkin: str,    # "YYYY-MM-DD"
    checkout: str,   # "YYYY-MM-DD"
    budget: str,     # "Low" | "Medium" | "High"
) -> list[HotelResult]:
    """
    Search for available hotels.

    Attempts SerpAPI Google Hotels first; falls back to mock data when
    SERPAPI_API_KEY is missing or the request fails.
    """
    api_key = os.getenv("SERPAPI_API_KEY", "")
    if not api_key:
        logger.info("SERPAPI_API_KEY not set — using mock hotel data.")
        return _mock_hotels(destination)

    try:
        return _with_retry(
            _serpapi_hotels, destination, checkin, checkout, api_key
        )
    except Exception as exc:
        logger.error("SerpAPI hotel search failed: %s — falling back to mocks", exc)
        return _mock_hotels(destination)


def _serpapi_hotels(
    destination: str,
    checkin: str,
    checkout: str,
    api_key: str,
) -> list[HotelResult]:
    """Hit the SerpAPI Google Hotels endpoint and parse results."""
    params = {
        "engine": "google_hotels",
        "q": destination,
        "check_in_date": checkin,
        "check_out_date": checkout,
        "currency": "USD",
        "hl": "en",
        "api_key": api_key,
    }
    response = requests.get(
        "https://serpapi.com/search",
        params=params,
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()

    hotels: list[HotelResult] = []
    for item in data.get("properties", []):
        rate = item.get("rate_per_night") or {}
        price = float(rate.get("extracted_lowest") or 0)
        # Skip listings without a usable nightly price (e.g. some vacation rentals)
        if price <= 0:
            continue

        # Google reviews use a 0–5 scale; HotelResult.rating uses 0–10
        raw_rating = float(item.get("overall_rating") or 0)
        rating = raw_rating * 2 if raw_rating <= 5 else raw_rating

        hotels.append(
            HotelResult(
                name=item.get("name", "Unknown Hotel"),
                price_per_night=price,
                rating=rating,
                distance_km=0.0,
                address=item.get("address", "") or "",
                booking_url=item.get("link", "") or "",
            )
        )

    return hotels if hotels else _mock_hotels(destination)


def _mock_hotels(destination: str) -> list[HotelResult]:
    """
    Return 8 realistic-looking mock hotels for the given destination.
    Used when SERPAPI_API_KEY is absent or the API call fails.
    """
    dest = destination.title()
    base_url = f"https://www.booking.com/search.html?ss={destination}"

    return [
        HotelResult(name=f"The {dest} Grand",          price_per_night=320.0, rating=9.1, distance_km=0.3, address=f"1 Central Ave, {dest}",         booking_url=base_url),
        HotelResult(name=f"{dest} Marriott",            price_per_night=289.0, rating=8.7, distance_km=0.8, address=f"200 Business Blvd, {dest}",     booking_url=base_url),
        HotelResult(name=f"Hilton {dest} Downtown",     price_per_night=265.0, rating=8.5, distance_km=1.1, address=f"55 Commerce St, {dest}",        booking_url=base_url),
        HotelResult(name=f"Hyatt Regency {dest}",       price_per_night=245.0, rating=8.3, distance_km=1.4, address=f"300 Park Lane, {dest}",         booking_url=base_url),
        HotelResult(name=f"{dest} Boutique Inn",        price_per_night=175.0, rating=8.0, distance_km=2.0, address=f"88 Old Quarter, {dest}",        booking_url=base_url),
        HotelResult(name=f"Comfort Suites {dest}",      price_per_night=139.0, rating=7.6, distance_km=3.5, address=f"450 Highway Rd, {dest}",        booking_url=base_url),
        HotelResult(name=f"Holiday Inn {dest}",         price_per_night=118.0, rating=7.2, distance_km=4.2, address=f"600 Airport Connector, {dest}", booking_url=base_url),
        HotelResult(name=f"{dest} Budget Lodge",        price_per_night=79.0,  rating=6.8, distance_km=5.0, address=f"12 Outskirts Rd, {dest}",       booking_url=base_url),
    ]
