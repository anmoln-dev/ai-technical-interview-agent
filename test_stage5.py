"""
Stage 5 Verification & End-to-End Test Suite
Tests:
- API Health & Metadata
- Model Selection & BYOK Config
- Curriculum & Candidate Profiles Data Layer
- Adaptive Interview State Machine (start, 8+ questions across 4+ days, follow-ups)
- Live Code Evaluation Engine & Syntax Checker
- Structured Evaluation Feedback Generation
- Static Assets & HTML/JS Accessibility markup presence
"""

import os
import pytest
from fastapi.testclient import TestClient
from app import app, SESSIONS, LIVE_CODE_CHALLENGES

client = TestClient(app)

def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["default_model"] == "gemma-4-31b-it"
    assert data["byok_supported"] is True

def test_get_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert data["default_model"] == "gemma-4-31b-it"
    model_ids = [m["id"] for m in data["models"]]
    assert "gemma-4-31b-it" in model_ids
    assert "gemini-2.5-flash" in model_ids
    assert "gemini-2.5-pro" in model_ids

def test_byok_validation():
    # Invalid key test
    bad_resp = client.post("/api/config/byok", json={"api_key": "short", "model_name": "gemma-4-31b-it"})
    assert bad_resp.status_code == 400

    # Valid key test
    good_resp = client.post("/api/config/byok", json={"api_key": "AIzaSy_test_valid_key_12345", "model_name": "gemini-2.5-flash"})
    assert good_resp.status_code == 200
    data = good_resp.json()
    assert data["status"] == "valid"
    assert data["model"] == "gemini-2.5-flash"

def test_data_layer_endpoints():
    # Curriculum
    curr_resp = client.get("/api/curriculum")
    assert curr_resp.status_code == 200
    curr_data = curr_resp.json()
    assert "cohort" in curr_data
    assert len(curr_data["days"]) == 31

    # Candidates list
    cand_resp = client.get("/api/candidates")
    assert cand_resp.status_code == 200
    candidates = cand_resp.json()
    assert len(candidates) >= 1

    # Single Candidate detail
    cand_id = candidates[0]["member"]["id"]
    single_resp = client.get(f"/api/candidates/{cand_id}")
    assert single_resp.status_code == 200
    assert single_resp.json()["member"]["id"] == cand_id

def test_full_e2e_interview_session_flow():
    # 1. Fetch first candidate
    candidates = client.get("/api/candidates").json()
    cand_id = candidates[0]["member"]["id"]

    # 2. Start Interview
    start_resp = client.post("/api/interview/start", json={
        "candidate_id": cand_id,
        "model_name": "gemma-4-31b-it",
        "api_key": "AIzaSy_test_key_sample"
    })
    assert start_resp.status_code == 200
    start_data = start_resp.json()
    session_id = start_data["session_id"]
    assert session_id in SESSIONS
    assert start_data["questions_asked"] == 1
    assert "initial_question" in start_data

    # 3. Simulate multi-turn technical conversation until session completes (8+ Qs across 4+ days)
    turns = 0
    max_turns = 15
    is_complete = False

    responses_pool = [
        "In Day 7, I used dense vector embeddings via SentenceTransformers sentence-transformers/all-MiniLM-L6-v2. We evaluated cosine similarity against Euclidean distance and found cosine similarity performed better because magnitude normalization prevented document length bias. We measured recall@k on healthcare claim queries.",
        "For Day 8 vector search, we configured ChromaDB with HNSW index parameters M=16, efConstruction=200. Metadata filtering on patient_id was applied during vector traversal to keep retrieval latency under 45ms.",
        "In Day 10 hybrid retrieval, our query router analyzed query intent using regex and lightweight intent classification to split structured claims queries to PostgreSQL and semantic policy questions to ChromaDB. Results were deduplicated by canonical document ID.",
        "Regarding prompt engineering in Day 12, we used system instructions with strict grounding directives and zero-shot schema enforcement to minimize LLM hallucination in clinical summaries.",
        "For Day 13 function calling, we defined Pydantic models for tool schemas. If the model hallucinated arguments, our API backend caught the validation error and returned a structured retry context.",
        "In Day 16 FastAPI backend development, we used async endpoints and CORS middleware, maintaining session state safely with thread-safe data structures.",
        "For Day 20 conversation memory, we implemented a sliding token window strategy keeping the last 6 turns and summarizing older history.",
        "In Day 22 multi-agent orchestration, we used a centralized supervisor router agent to dispatch sub-tasks to specialist retriever and evaluator agents.",
        "In Day 23 Model Context Protocol, MCP standardized server tools JSON-RPC protocol enabling plug-and-play tool integration.",
        "On Day 27 security, we implemented prompt injection detection filters using regex guardrails and PII redaction.",
        "For Day 28 deployment, multi-stage Docker builds reduced our image size to under 250MB with Kubernetes liveness and readiness probes configured.",
        "For Capstone Day 31, our architecture connected Next.js client to FastAPI backend streaming SSE chunks from ChromaDB RAG and MCP tools."
    ]

    while not is_complete and turns < max_turns:
        resp_text = responses_pool[turns % len(responses_pool)]
        chat_resp = client.post("/api/interview/chat", json={
            "session_id": session_id,
            "message": resp_text
        })
        assert chat_resp.status_code == 200
        chat_data = chat_resp.json()
        is_complete = chat_data["is_complete"]
        turns += 1

    assert is_complete is True
    assert chat_data["questions_asked"] >= 8
    assert chat_data["days_covered_count"] >= 4

    # 4. Verify Feedback Report Endpoint
    feedback_resp = client.get(f"/api/interview/session/{session_id}/feedback")
    assert feedback_resp.status_code == 200
    report = feedback_resp.json()
    assert "overall_score" in report
    assert "readiness_level" in report
    assert "domain_scores" in report
    assert "strengths" in report
    assert "growth_areas" in report
    assert "recommended_review_days" in report
    assert report["overall_score"] >= 70

def test_live_code_challenge_submission():
    # Start session
    cand_id = client.get("/api/candidates").json()[0]["member"]["id"]
    session_id = client.post("/api/interview/start", json={"candidate_id": cand_id}).json()["session_id"]

    # Valid Python Code submission
    valid_code = (
        "import math\n"
        "from typing import List\n\n"
        "def cosine_similarity(a: List[float], b: List[float]) -> float:\n"
        "    dot_product = sum(x * y for x, y in zip(a, b))\n"
        "    mag_a = math.sqrt(sum(x * x for x in a))\n"
        "    mag_b = math.sqrt(sum(y * y for y in b))\n"
        "    if mag_a == 0 or mag_b == 0:\n"
        "        return 0.0\n"
        "    return dot_product / (mag_a * mag_b)\n"
    )

    sub_resp = client.post("/api/interview/live-test/submit", json={
        "session_id": session_id,
        "test_id": "challenge_day_7",
        "code": valid_code
    })
    assert sub_resp.status_code == 200
    res = sub_resp.json()
    assert res["passed"] is True
    assert res["score"] >= 90

    # Invalid syntax code submission
    bad_code = "def cosine_similarity(a, b):\n return a +"
    bad_sub = client.post("/api/interview/live-test/submit", json={
        "session_id": session_id,
        "test_id": "challenge_day_7",
        "code": bad_code
    })
    assert bad_sub.status_code == 200
    bad_res = bad_sub.json()
    assert bad_res["passed"] is False
    assert bad_res["score"] < 50
    assert "Syntax error" in bad_res["feedback"]

def test_frontend_static_serving_and_accessibility_markers():
    response = client.get("/")
    assert response.status_code == 200
    html = response.text

    # Check WCAG / Accessibility elements in root HTML
    assert '<html lang="en"' in html
    assert 'class="skip-link"' in html
    assert 'id="sr-announcements"' in html
    assert 'role="status"' in html
    assert 'role="banner"' in html
    assert 'role="main"' in html
    assert 'role="dialog"' in html
    assert 'aria-modal="true"' in html
    assert 'data-contrast="normal"' in html
    assert 'data-size="normal"' in html
    assert 'app.js' in html
    assert 'styles.css' in html

if __name__ == "__main__":
    pytest.main(["-v", __file__])
