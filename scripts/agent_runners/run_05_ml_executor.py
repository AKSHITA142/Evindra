import sys
import os
import argparse
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.ml_execution.executor import MLExecutionEngine
from backend.schemas.experiment import ExperimentSpec
from scripts.agent_runners.utils import (
    print_phase_header,
    print_input_payload,
    print_output_payload,
    save_snapshot,
    load_snapshot,
)

def main():
    parser = argparse.ArgumentParser(description="Phase 5 Diagnostic Runner: ML Execution Engine")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV dataset file")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--plan", type=str, default=None, help="Path to ExperimentPlan JSON file (default: step_04_output.json)")
    parser.add_argument("--exp-index", type=int, default=0, help="Index of candidate experiment in plan to execute (default: 0)")
    args = parser.parse_args()

    print_phase_header(5, "ML Execution Engine", "backend.ml_execution.executor.MLExecutionEngine")

    if not os.path.isfile(args.file):
        print(f"❌ Error: Dataset CSV file '{args.file}' not found.")
        sys.exit(1)

    raw_plan = load_snapshot("step_04_output.json") if not args.plan else json.load(open(args.plan))
    if not raw_plan:
        print("❌ Error: No ExperimentPlan input found. Run step 4 first.")
        sys.exit(1)

    exp_list = raw_plan.get("experiments", [])
    if not exp_list or args.exp_index >= len(exp_list):
        print(f"❌ Error: Experiment index {args.exp_index} out of range (found {len(exp_list)} experiments).")
        sys.exit(1)

    raw_spec = exp_list[args.exp_index]
    spec_obj = ExperimentSpec(**raw_spec) if isinstance(raw_spec, dict) else raw_spec

    df = pd.read_csv(args.file)

    inputs = {
        "dataset_rows": len(df),
        "target_column": args.target,
        "experiment_spec": spec_obj.model_dump(),
        "test_size": 0.2,
        "cv_folds": 5,
    }
    print_input_payload(inputs)

    model_name = getattr(spec_obj, "model_name", getattr(spec_obj, "model", "Model"))
    print(f"⚡ Executing single experiment '{spec_obj.experiment_id}' ({model_name})...")
    engine = MLExecutionEngine()
    result = engine.execute_single_experiment(
        spec=spec_obj,
        df=df,
        target_column=args.target,
    )

    print_output_payload(result)
    save_snapshot("step_05_output.json", [result])

if __name__ == "__main__":
    main()
