import sys
import os
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.schemas.evaluation import EvaluationReport
from backend.agents.research_director import ResearchDirectorAgent
from scripts.agent_runners.utils import (
    print_phase_header,
    print_input_payload,
    print_output_payload,
    save_snapshot,
    load_snapshot,
)

def main():
    parser = argparse.ArgumentParser(description="Phase 7 Diagnostic Runner: Research Director Agent")
    parser.add_argument("--eval", type=str, default=None, help="Path to EvaluationReport JSON file (default: step_06_output.json)")
    parser.add_argument("--iteration", type=int, default=1, help="Current iteration number")
    parser.add_argument("--max-iterations", type=int, default=5, help="Maximum iterations limit")
    args = parser.parse_args()

    print_phase_header(7, "Research Director Agent", "backend.agents.ResearchDirectorAgent")

    raw_eval = load_snapshot("step_06_output.json") if not args.eval else json.load(open(args.eval))
    if not raw_eval:
        print("❌ Error: No EvaluationReport input found. Run step 6 first.")
        sys.exit(1)

    eval_obj = EvaluationReport(**raw_eval) if isinstance(raw_eval, dict) else raw_eval

    inputs = {
        "evaluation_report": eval_obj,
        "iteration_count": args.iteration,
        "max_iterations": args.max_iterations,
    }
    print_input_payload(inputs)

    print("🧭 Executing ResearchDirectorAgent.run()...")
    agent = ResearchDirectorAgent()
    decision = agent.run(inputs)

    print_output_payload(decision)
    save_snapshot("step_07_output.json", decision)

if __name__ == "__main__":
    main()
