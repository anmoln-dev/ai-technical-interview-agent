"""
Stage 3 Test Script - Verify All REST API Endpoints via FastAPI TestClient
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_stage_3_api_endpoints():
    print("==========================================")
    print("Running Stage 3 API Endpoints Verification Suite")
    print("==========================================")

    # 1. Health Check
    r = client.get("/api/health")
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    data = r.json()
    assert data["status"] == "ok"
    assert data["default_model"] == "gemma-4-31b-it"
    print("[OK] GET /api/health -> 200 OK (default_model: gemma-4-31b-it)")

    # 2. Get Models List
    r = client.get("/api/models")
    assert r.status_code == 200
    models = r.json().get("models", [])
    assert len(models) == 3
    print(f"[OK] GET /api/models -> 200 OK ({len(models)} models available)")

    # 3. Validate BYOK API Key
    r = client.post("/api/config/byok", json={"api_key": "AIzaSyFakeKeyForTesting123", "model_name": "gemma-4-31b-it"})
    assert r.status_code == 200
    assert r.json()["status"] == "valid"
    print("[OK] POST /api/config/byok -> 200 OK (Validated key)")

    # 4. Get Candidates List & Single Candidate
    r = client.get("/api/candidates")
    assert r.status_code == 200
    cands = r.json()
    assert len(cands) == 20
    print(f"[OK] GET /api/candidates -> 200 OK ({len(cands)} candidates returned)")

    r = client.get("/api/candidates/CAND-001")
    assert r.status_code == 200
    assert r.json()["member"]["name"] == "Sarah Johnson"
    print("[OK] GET /api/candidates/CAND-001 -> 200 OK (Sarah Johnson)")

    r = client.get("/api/candidates/INVALID_ID")
    assert r.status_code == 404
    print("[OK] GET /api/candidates/INVALID_ID -> 404 Not Found (Correct Error Handling)")

    # 5. Get Curriculum
    r = client.get("/api/curriculum")
    assert r.status_code == 200
    curr = r.json()
    assert len(curr["modules"]) == 8
    assert len(curr["days"]) == 31
    print("[OK] GET /api/curriculum -> 200 OK (8 Modules, 31 Days)")

    # 6. Start Interview
    r = client.post("/api/interview/start", json={"candidate_id": "CAND-001", "model_name": "gemma-4-31b-it"})
    assert r.status_code == 200
    start_data = r.json()
    session_id = start_data["session_id"]
    assert "initial_question" in start_data
    print(f"[OK] POST /api/interview/start -> 200 OK (Session: {session_id[:8]}...)")

    # 7. Post Chat Messages
    r = client.post("/api/interview/chat", json={"session_id": session_id, "message": "I used Sentence Transformers for vector embeddings and ChromaDB for local indexing."})
    assert r.status_code == 200
    chat_data = r.json()
    assert "agent_response" in chat_data
    print("[OK] POST /api/interview/chat -> 200 OK (Agent response received)")

    # 8. Live Test Submission
    r = client.post("/api/interview/live-test/submit", json={
        "session_id": session_id,
        "test_id": "challenge_day_7",
        "code": "def cosine_similarity(a, b):\n    return 1.0\n"
    })
    assert r.status_code == 200
    live_data = r.json()
    assert live_data["passed"] is True
    print("[OK] POST /api/interview/live-test/submit -> 200 OK (Code evaluated)")

    # 9. Get Session Status
    r = client.get(f"/api/interview/session/{session_id}")
    assert r.status_code == 200
    assert len(r.json()["messages"]) >= 2
    print("[OK] GET /api/interview/session/{session_id} -> 200 OK (Session history returned)")

    # 10. End Interview & Get Feedback
    r = client.post("/api/interview/end", json={"session_id": session_id})
    assert r.status_code == 200
    feedback_data = r.json()
    assert "overall_score" in feedback_data
    assert "readiness_level" in feedback_data
    print(f"[OK] POST /api/interview/end -> 200 OK (Overall Score: {feedback_data['overall_score']})")

    r = client.get(f"/api/interview/session/{session_id}/feedback")
    assert r.status_code == 200
    print("[OK] GET /api/interview/session/{session_id}/feedback -> 200 OK (Feedback report retrieved)")

    print("==========================================")
    print("STAGE 3 API VERIFICATION SUCCESSFUL!")
    print("==========================================")

if __name__ == "__main__":
    test_stage_3_api_endpoints()
