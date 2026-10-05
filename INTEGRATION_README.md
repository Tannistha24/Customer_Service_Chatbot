# Unified Customer Service Bot

## Run the website

From the `src` directory:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The website calls `runtime.py`, which orchestrates the existing Task 1-6
components.

## Pipeline

Customer message -> security -> session -> multilingual NLU -> entities ->
intent -> sentiment/risk -> ticket/priority -> routing -> existing RAG/KB ->
customer response.

The existing task implementations remain in their original folders.

## Important

The uploaded `.env` contained a Gemini API key. It was intentionally removed
from this package. Rotate/revoke that key if it was a real credential, and
place the new key in a local `.env` file.

The RAG answer path requires an activated knowledge-base version. Without an
active KB, the runtime returns a safe "knowledge base unavailable" response
rather than inventing an answer.
