import sys
import os
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.schemas.semantic_profile import SemanticProfile
from backend.agents.dataset_understanding import DatasetUnderstandingAgent
from scripts.agent_runners.utils import (
    print_phase_header,
    print_input_payload,
    print_output_payload,
    save_snapshot,
    load_snapshot,
)

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Diagnostic Runner: Dataset Understanding Agent")
    parser.add_argument("--input", type=str, default=None, help="Path to SemanticProfile JSON file (default: loads step_01_output.json)")
    args = parser.parse_args()

    print_phase_header(2, "Dataset Understanding Agent", "backend.agents.DatasetUnderstandingAgent")

    raw_profile = None
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            raw_profile = json.load(f)
    else:
        raw_profile = load_snapshot("step_01_output.json")

    if not raw_profile:
        print("❌ Error: No SemanticProfile input found. Run step 1 first or pass --input <file.json>.")
        sys.exit(1)

    profile_obj = SemanticProfile(**raw_profile) if isinstance(raw_profile, dict) else raw_profile

    inputs = {"semantic_profile": profile_obj}
    print_input_payload(inputs)

    print("🧠 Executing DatasetUnderstandingAgent.run()...")
    agent = DatasetUnderstandingAgent()
    domain_context = agent.run(inputs)

    print_output_payload(domain_context)
    save_snapshot("step_02_output.json", domain_context)

if __name__ == "__main__":
    main()
