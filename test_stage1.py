"""
Stage 1 Test Script - Verify Google AI Studio BYOK Configuration & Data Loading
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import load_json, CURRICULUM_DATA, CANDIDATE_DATA, SUPPORTED_MODELS

def test_stage_1():
    print("==========================================")
    print("Running Stage 1 Verification Suite")
    print("==========================================")

    # 1. Verify Curriculum Data Loading
    modules = CURRICULUM_DATA.get("modules", [])
    days = CURRICULUM_DATA.get("days", [])
    print(f"[OK] Curriculum Loaded: {len(modules)} Modules, {len(days)} Days")
    assert len(modules) == 8, f"Expected 8 modules, got {len(modules)}"
    assert len(days) == 31, f"Expected 31 days, got {len(days)}"

    # 2. Verify Candidates Data Loading
    candidates = CANDIDATE_DATA.get("candidates", [])
    print(f"[OK] Candidates Loaded: {len(candidates)} Cohort Candidates")
    assert len(candidates) == 20, f"Expected 20 candidates, got {len(candidates)}"

    # 3. Verify Supported Models & Gemma 4 31B Default
    print(f"[OK] Supported Models List: {[m['id'] for m in SUPPORTED_MODELS]}")
    default_model = next((m for m in SUPPORTED_MODELS if m.get("default")), None)
    assert default_model is not None, "Default model not configured"
    assert default_model["id"] == "gemma-4-31b-it", f"Expected default model 'gemma-4-31b-it', got {default_model['id']}"
    print(f"[OK] Default Model Verified: {default_model['name']} ({default_model['id']})")

    print("==========================================")
    print("STAGE 1 VERIFICATION SUCCESSFUL!")
    print("==========================================")

if __name__ == "__main__":
    test_stage_1()
