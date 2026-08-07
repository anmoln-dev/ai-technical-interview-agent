"""
Stage 2 Test Script - Verify Interview State Machine, Live Testing Tool Engine, and Feedback Generator
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import (
    start_interview, StartInterviewRequest,
    interview_chat, ChatMessageRequest,
    submit_live_test, LiveCodeSubmissionRequest,
    end_interview, EndInterviewRequest,
    SESSIONS, LIVE_CODE_CHALLENGES
)

def test_stage_2():
    print("==========================================")
    print("Running Stage 2 Verification Suite")
    print("==========================================")

    # 1. Test Session Initialization for Candidate CAND-001 (Sarah Johnson)
    start_req = StartInterviewRequest(candidate_id="CAND-001")
    start_res = start_interview(start_req)
    session_id = start_res["session_id"]
    
    assert session_id in SESSIONS, "Session ID not stored"
    assert start_res["questions_asked"] == 1, "Initial question count should be 1"
    assert len(start_res["days_covered_list"]) >= 1, "At least 1 day covered initially"
    print(f"[OK] Interview Started: Session {session_id[:8]}... for {start_res['candidate']['name']}")

    # 2. Test Multi-Turn Dynamic Conversation (Simulate 8 Questions)
    sample_answers = [
        "In Day 7, I used Sentence Transformers to create 384-dimensional dense vectors and evaluated cosine similarity against Euclidean distance.",
        "We chose ChromaDB for local prototyping with HNSW indexing and metadata filtering for fast search.",
        "Our query router checks if keywords match SQL schema or semantic vector DB index.",
        "We used few-shot prompt templates to enforce Pydantic output formatting.",
        "Function calling schema validation caught invalid tool arguments before tool execution.",
        "FastAPI async endpoints managed session concurrency with SQLite persistence.",
        "We used CrewAI multi-agent router to delegate domain specific queries.",
        "Containerized backend with Docker multi-stage build and deployed to Kubernetes."
    ]

    for idx, answer in enumerate(sample_answers):
        chat_req = ChatMessageRequest(session_id=session_id, message=answer)
        chat_res = interview_chat(chat_req)
        print(f"[OK] Turn {idx+1}: Q_count={chat_res['questions_asked']}, Days_covered={chat_res['days_covered_count']}, Complete={chat_res['is_complete']}")

    session = SESSIONS[session_id]
    assert session["questions_asked"] >= 8, f"Expected at least 8 questions, got {session['questions_asked']}"
    assert len(set(session["days_covered"])) >= 4, f"Expected at least 4 unique days, got {len(set(session['days_covered']))}"

    # 3. Test Live Coding Challenge Execution & Submission
    challenge_id = "challenge_day_7"
    valid_code = (
        "import math\n"
        "def cosine_similarity(a, b):\n"
        "    dot = sum(x*y for x,y in zip(a,b))\n"
        "    mag_a = math.sqrt(sum(x*x for x in a))\n"
        "    mag_b = math.sqrt(sum(y*y for y in b))\n"
        "    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0\n"
    )
    live_req = LiveCodeSubmissionRequest(session_id=session_id, test_id=challenge_id, code=valid_code)
    live_res = submit_live_test(live_req)
    
    assert live_res["passed"] is True, "Valid code should pass test"
    assert live_res["score"] == 95, "Valid code score should be 95"
    print(f"[OK] Live Code Submission Passed: Challenge '{challenge_id}' Score={live_res['score']}")

    # 4. Test Final Feedback Generator
    end_req = EndInterviewRequest(session_id=session_id)
    feedback_res = end_interview(end_req)
    
    assert feedback_res["overall_score"] > 0, "Overall score should be > 0"
    assert "readiness_level" in feedback_res, "Readiness level missing"
    assert len(feedback_res["domain_scores"]) >= 5, "Domain breakdown scores missing"
    assert len(feedback_res["strengths"]) >= 3, "Strengths list missing"
    assert len(feedback_res["growth_areas"]) >= 3, "Growth areas list missing"
    print(f"[OK] Final Feedback Report Generated: Score={feedback_res['overall_score']}, Readiness='{feedback_res['readiness_level']}'")

    print("==========================================")
    print("STAGE 2 VERIFICATION SUCCESSFUL!")
    print("==========================================")

if __name__ == "__main__":
    test_stage_2()
