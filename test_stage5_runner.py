"""
Stage 5 Verification Runner using Standard Library (unittest)
"""

import sys
import unittest
from fastapi.testclient import TestClient
from app import app, SESSIONS, LIVE_CODE_CHALLENGES

client = TestClient(app)

class TestStage5E2E(unittest.TestCase):

    def test_api_health(self):
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["default_model"], "gemma-4-31b-it")
        self.assertTrue(data["byok_supported"])

    def test_get_models(self):
        response = client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models", data)
        self.assertEqual(data["default_model"], "gemma-4-31b-it")
        model_ids = [m["id"] for m in data["models"]]
        self.assertIn("gemma-4-31b-it", model_ids)
        self.assertIn("gemini-2.5-flash", model_ids)
        self.assertIn("gemini-2.5-pro", model_ids)

    def test_byok_validation(self):
        bad_resp = client.post("/api/config/byok", json={"api_key": "short", "model_name": "gemma-4-31b-it"})
        self.assertEqual(bad_resp.status_code, 400)

        good_resp = client.post("/api/config/byok", json={"api_key": "AIzaSy_test_valid_key_12345", "model_name": "gemini-2.5-flash"})
        self.assertEqual(good_resp.status_code, 200)
        data = good_resp.json()
        self.assertEqual(data["status"], "valid")
        self.assertEqual(data["model"], "gemini-2.5-flash")

    def test_data_layer_endpoints(self):
        curr_resp = client.get("/api/curriculum")
        self.assertEqual(curr_resp.status_code, 200)
        curr_data = curr_resp.json()
        self.assertIn("cohort", curr_data)
        self.assertEqual(len(curr_data["days"]), 31)

        cand_resp = client.get("/api/candidates")
        self.assertEqual(cand_resp.status_code, 200)
        candidates = cand_resp.json()
        self.assertGreaterEqual(len(candidates), 1)

        cand_id = candidates[0]["member"]["id"]
        single_resp = client.get(f"/api/candidates/{cand_id}")
        self.assertEqual(single_resp.status_code, 200)
        self.assertEqual(single_resp.json()["member"]["id"], cand_id)

    def test_full_e2e_interview_session_flow(self):
        candidates = client.get("/api/candidates").json()
        cand_id = candidates[0]["member"]["id"]

        start_resp = client.post("/api/interview/start", json={
            "candidate_id": cand_id,
            "model_name": "gemma-4-31b-it",
            "api_key": "AIzaSy_test_key_sample"
        })
        self.assertEqual(start_resp.status_code, 200)
        start_data = start_resp.json()
        session_id = start_data["session_id"]
        self.assertIn(session_id, SESSIONS)
        self.assertEqual(start_data["questions_asked"], 1)
        self.assertIn("initial_question", start_data)

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
            self.assertEqual(chat_resp.status_code, 200)
            chat_data = chat_resp.json()
            is_complete = chat_data["is_complete"]
            turns += 1

        self.assertTrue(is_complete)
        self.assertGreaterEqual(chat_data["questions_asked"], 8)
        self.assertGreaterEqual(chat_data["days_covered_count"], 4)

        feedback_resp = client.get(f"/api/interview/session/{session_id}/feedback")
        self.assertEqual(feedback_resp.status_code, 200)
        report = feedback_resp.json()
        self.assertIn("overall_score", report)
        self.assertIn("readiness_level", report)
        self.assertIn("domain_scores", report)
        self.assertIn("strengths", report)
        self.assertIn("growth_areas", report)
        self.assertIn("recommended_review_days", report)
        self.assertGreaterEqual(report["overall_score"], 70)

    def test_live_code_challenge_submission(self):
        cand_id = client.get("/api/candidates").json()[0]["member"]["id"]
        session_id = client.post("/api/interview/start", json={"candidate_id": cand_id}).json()["session_id"]

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
        self.assertEqual(sub_resp.status_code, 200)
        res = sub_resp.json()
        self.assertTrue(res["passed"])
        self.assertGreaterEqual(res["score"], 90)

        bad_code = "def cosine_similarity(a, b):\n return a +"
        bad_sub = client.post("/api/interview/live-test/submit", json={
            "session_id": session_id,
            "test_id": "challenge_day_7",
            "code": bad_code
        })
        self.assertEqual(bad_sub.status_code, 200)
        bad_res = bad_sub.json()
        self.assertFalse(bad_res["passed"])
        self.assertLess(bad_res["score"], 50)
        self.assertIn("Syntax error", bad_res["feedback"])

    def test_repeat_and_unsure_intent_handling(self):
        cand_id = client.get("/api/candidates").json()[0]["member"]["id"]
        start_data = client.post("/api/interview/start", json={"candidate_id": cand_id}).json()
        session_id = start_data["session_id"]
        self.assertEqual(start_data["mode"], "demo")
        self.assertIn("Demo Mode", start_data["mode_notice"])

        # Candidate asks to repeat question
        repeat_resp = client.post("/api/interview/chat", json={
            "session_id": session_id,
            "message": "Sorry, could you repeat that?"
        }).json()

        # Should NOT advance topic or say "great engineering reasoning"
        self.assertNotIn("great engineering reasoning", repeat_phrases := repeat_resp["agent_response"].lower())
        self.assertIn("restate", repeat_phrases)
        self.assertEqual(repeat_resp["questions_asked"], 1)

        # Candidate says unsure
        unsure_resp = client.post("/api/interview/chat", json={
            "session_id": session_id,
            "message": "I don't know"
        }).json()

        self.assertNotIn("great engineering reasoning", unsure_resp["agent_response"].lower())
        self.assertIn("no worries", unsure_resp["agent_response"].lower())
        self.assertEqual(unsure_resp["questions_asked"], 2)

    def test_demo_vs_live_mode_notifications(self):
        cand_id = client.get("/api/candidates").json()[0]["member"]["id"]
        
        # Demo mode test
        demo_data = client.post("/api/interview/start", json={"candidate_id": cand_id}).json()
        self.assertEqual(demo_data["mode"], "demo")
        self.assertIn("Demo Mode", demo_data["mode_notice"])

        # Live mode test
        live_data = client.post("/api/interview/start", json={
            "candidate_id": cand_id,
            "api_key": "AIzaSy_sample_test_key_long_enough",
            "model_name": "gemini-2.5-flash"
        }).json()
        self.assertEqual(live_data["mode"], "live")
        self.assertIn("Live Mode", live_data["mode_notice"])

    def test_frontend_static_serving_and_accessibility_markers(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.text

        self.assertIn('<html lang="en"', html)
        self.assertIn('class="skip-link"', html)
        self.assertIn('id="sr-announcements"', html)
        self.assertIn('role="status"', html)
        self.assertIn('role="banner"', html)
        self.assertIn('role="main"', html)
        self.assertIn('role="dialog"', html)
        self.assertIn('aria-modal="true"', html)
        self.assertIn('data-contrast="normal"', html)
        self.assertIn('data-size="normal"', html)
        self.assertIn('mode-status-badge', html)
        self.assertIn('app.js', html)
        self.assertIn('styles.css', html)

if __name__ == "__main__":
    unittest.main()
