import sys
import os
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.evaluation.evaluator import EvaluationEngine
from scripts.agent_runners.utils import (
    print_phase_header,
    print_input_payload,
    print_output_payload,
    save_snapshot,
    load_snapshot,
)

def main():
    parser = argparse.ArgumentParser(description="Phase 6 Diagnostic Runner: Evaluation Engine")
    parser.add_argument("--results", type=str, default=None, help="Path to ExperimentResults JSON file (default: step_05_output.json)")
    args = parser.parse_args()

    print_phase_header(6, "Evaluation Engine", "backend.evaluation.evaluator.EvaluationEngine")

    raw_results = load_snapshot("step_05_output.json") if not args.results else json.load(open(args.results))
    if not raw_results:
        print("❌ Error: No ExperimentResults input found. Run step 5 first.")
        sys.exit(1)

    print_input_payload(raw_results, label="EXPERIMENT RESULTS LIST")

    from backend.schemas.experiment import ExperimentResult
    results_objs = [ExperimentResult(**r) if isinstance(r, dict) else r for r in raw_results]

    print("📈 Executing EvaluationEngine.evaluate_batch()...")
    evaluator = EvaluationEngine()
    report, decision = evaluator.evaluate_batch(results_objs, job_id="debug_job")

    print_output_payload(report)
    save_snapshot("step_06_output.json", report)

if __name__ == "__main__":
    main()
