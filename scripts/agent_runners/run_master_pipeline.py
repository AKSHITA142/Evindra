import sys
import os
import argparse
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.profiling.engine import ProfilingEngine
from backend.agents.dataset_understanding import DatasetUnderstandingAgent
from backend.agents.constraint_analyzer import ConstraintGoalAnalyzer
from backend.agents.strategy_planner import StrategyPlannerAgent
from backend.ml_execution.executor import MLExecutionEngine
from backend.evaluation.evaluator import EvaluationEngine
from backend.agents.research_director import ResearchDirectorAgent
from backend.agents.report_generator import ReportGeneratorAgent
from backend.reports.html_generator import HTMLReportGenerator
from backend.reports.markdown_generator import MarkdownReportGenerator

from scripts.agent_runners.utils import (
    print_phase_header,
    print_input_payload,
    print_output_payload,
    save_snapshot,
)

def main():
    parser = argparse.ArgumentParser(description="Master DataPilot-AI Pipeline Diagnostic Runner")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV dataset file")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--mission", type=str, default="Predict target with high precision and zero feature leakage", help="Research mission text")
    parser.add_argument("--budget", type=int, default=4, help="Experiment budget")
    parser.add_argument("--step-by-step", action="store_true", help="Pause after each step for manual review")
    args = parser.parse_args()

    def pause_if_step():
        if args.step_by_step:
            input("\n[Press Enter to continue to next pipeline step...]")

    print("\n🚀 STARTING MASTER PIPELINE DIAGNOSTIC RUN 🚀\n")

    # 1. Profiling
    print_phase_header(1, "Profiling Engine", "backend.profiling.engine.ProfilingEngine")
    print_input_payload({"file_path": args.file, "target_column": args.target})
    profile, _ = ProfilingEngine.profile_file(args.file, target_column=args.target)
    print_output_payload(profile)
    save_snapshot("step_01_output.json", profile)
    pause_if_step()

    # 2. Dataset Understanding
    print_phase_header(2, "Dataset Understanding Agent", "backend.agents.DatasetUnderstandingAgent")
    print_input_payload({"semantic_profile": profile})
    understanding_agent = DatasetUnderstandingAgent()
    domain_context = understanding_agent.run({"semantic_profile": profile})
    print_output_payload(domain_context)
    save_snapshot("step_02_output.json", domain_context)
    pause_if_step()

    # 3. Constraint & Goal Analyzer
    print_phase_header(3, "Constraint & Goal Analyzer Agent", "backend.agents.ConstraintGoalAnalyzer")
    print_input_payload({"user_goal": args.mission, "semantic_profile": profile})
    constraint_agent = ConstraintGoalAnalyzer()
    mission_brief = constraint_agent.run({"user_goal": args.mission, "semantic_profile": profile})
    print_output_payload(mission_brief)
    save_snapshot("step_03_output.json", mission_brief)
    pause_if_step()

    # 4. Strategy Planner
    print_phase_header(4, "Strategy Planner Agent", "backend.agents.StrategyPlannerAgent")
    print_input_payload({"semantic_profile": profile, "experiment_budget": args.budget, "mission_brief": mission_brief})
    planner_agent = StrategyPlannerAgent()
    experiment_plan = planner_agent.run({"semantic_profile": profile, "experiment_budget": args.budget, "mission_brief": mission_brief})
    print_output_payload(experiment_plan)
    save_snapshot("step_04_output.json", experiment_plan)
    pause_if_step()

    # 5. ML Execution Engine
    print_phase_header(5, "ML Execution Engine", "backend.ml_execution.executor.MLExecutionEngine")
    df = pd.read_csv(args.file)
    results = []
    ml_engine = MLExecutionEngine()
    for idx, spec in enumerate(experiment_plan.experiments):
        model_name = getattr(spec, "model_name", getattr(spec, "model", "Model"))
        print(f"\n⚡ Running Experiment {idx + 1}/{len(experiment_plan.experiments)} ({spec.experiment_id}: {model_name})...")
        res = ml_engine.execute_single_experiment(spec=spec, df=df, target_column=args.target)
        results.append(res)
    print_output_payload(results, label=f"EXECUTED {len(results)} EXPERIMENTS")
    save_snapshot("step_05_output.json", results)
    pause_if_step()

    # 6. Evaluation Engine
    print_phase_header(6, "Evaluation Engine", "backend.evaluation.evaluator.EvaluationEngine")
    print_input_payload(results, label="EXPERIMENT RESULTS")
    evaluator = EvaluationEngine()
    eval_report, _ = evaluator.evaluate_batch(results, job_id="master_debug_job")
    print_output_payload(eval_report)
    save_snapshot("step_06_output.json", eval_report)
    pause_if_step()

    # 7. Research Director Agent
    print_phase_header(7, "Research Director Agent", "backend.agents.ResearchDirectorAgent")
    print_input_payload({"evaluation_report": eval_report, "iteration_count": 1, "max_iterations": 2})
    director_agent = ResearchDirectorAgent()
    decision = director_agent.run({"evaluation_report": eval_report, "iteration_count": 1, "max_iterations": 2})
    print_output_payload(decision)
    save_snapshot("step_07_output.json", decision)
    pause_if_step()

    # 8. Report Generator Agent
    print_phase_header(8, "Report Generator Agent", "backend.agents.ReportGeneratorAgent & Renderers")
    print_input_payload({"evaluation_report": eval_report, "experiment_results": results, "semantic_profile": profile})
    reporter_agent = ReportGeneratorAgent()
    recommendation = reporter_agent.run({"evaluation_report": eval_report, "experiment_results": results, "semantic_profile": profile})
    print_output_payload(recommendation)

    html_str = HTMLReportGenerator.generate_html(recommendation=recommendation, evaluation_report=eval_report, profile=profile, experiment_results=results)
    out_dir = os.path.join("storage", "debug_runs")
    os.makedirs(out_dir, exist_ok=True)
    html_file = os.path.join(out_dir, "report.html")
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"📄 Final HTML Audit Report generated: {html_file}")
    save_snapshot("step_08_output.json", recommendation)

    print("\n✅ MASTER PIPELINE DIAGNOSTIC RUN COMPLETED 100% SUCCESSFULLY!\n")

if __name__ == "__main__":
    main()
