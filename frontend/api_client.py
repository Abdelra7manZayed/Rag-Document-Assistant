"""
Thin wrapper around the RAG backend API.
All network calls live here — the Streamlit app never calls requests directly.
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
QUERY_ENDPOINT = f"{BASE_URL}/api/v1/query"
HEALTH_ENDPOINT = f"{BASE_URL}/health"
TIMEOUT = 60  # seconds — LLM inference can be slow


def check_health() -> bool:
    """Return True if the backend is reachable."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def ask_question(question: str) -> dict:
    """
    Send a question to the backend and return the parsed response dict.

    Returns:
        {
            "answer": str,
            "sources": [ {"source": str, "chunk_id": str, "excerpt": str} ],
            "model_used": str,
        }

    Raises:
        requests.HTTPError  — on 4xx/5xx responses
        requests.Timeout    — if the backend takes too long
        requests.ConnectionError — if backend is unreachable
    """
    payload = {"question": question}
    response = requests.post(QUERY_ENDPOINT, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()
