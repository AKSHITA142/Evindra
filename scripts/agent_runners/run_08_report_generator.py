import sys
import os
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.schemas.evaluation import EvaluationReport
from backend.schemas.semantic_profile import SemanticProfile
from backend.agents.report_generator import ReportGeneratorAgent
from backend.reports.html_generator import HTMLReportGenerator
from backend.reports.markdown_generator import MarkdownReportGenerator
from scripts.agent_runners.utils import (
    print_phase_header,
    print_input_payload,
    print_output_payload,
    save_snapshot,
    load_snapshot,
)

def main():
    parser = argparse.ArgumentParser(description="Phase 8 Diagnostic Runner: Report Generator Agent & HTML Renderer")
    parser.add_argument("--eval", type=str, default=None, help="Path to EvaluationReport JSON")
    parser.add_argument("--results", type=str, default=None, help="Path to ExperimentResults JSON")
    parser.add_argument("--profile", type=str, default=None, help="Path to SemanticProfile JSON")
    args = parser.parse_args()

    print_phase_header(8, "Report Generator Agent", "backend.agents.ReportGeneratorAgent & HTMLReportGenerator")

    raw_eval = load_snapshot("step_06_output.json") if not args.eval else json.load(open(args.eval))
    raw_results = load_snapshot("step_05_output.json") if not args.results else json.load(open(args.results))
    raw_profile = load_snapshot("step_01_output.json") if not args.profile else json.load(open(args.profile))

    if not raw_eval:
        print("❌ Error: EvaluationReport input missing.")
        sys.exit(1)

    eval_obj = EvaluationReport(**raw_eval) if isinstance(raw_eval, dict) else raw_eval
    profile_obj = SemanticProfile(**raw_profile) if isinstance(raw_profile, dict) else raw_profile

    inputs = {
        "evaluation_report": eval_obj,
        "experiment_results": raw_results or [],
        "semantic_profile": profile_obj,
    }
    print_input_payload(inputs)

    print("📝 Executing ReportGeneratorAgent.run()...")
    agent = ReportGeneratorAgent()
    recommendation = agent.run(inputs)

    print_output_payload(recommendation, label="FINAL RECOMMENDATION JSON")

    print("🎨 Rendering HTML & Markdown report templates...")
    html_str = HTMLReportGenerator.generate_html(
        recommendation=recommendation,
        evaluation_report=eval_obj,
        profile=profile_obj,
        experiment_results=raw_results or [],
    )

    md_str = MarkdownReportGenerator.generate_markdown(
        recommendation=recommendation,
        evaluation_report=eval_obj,
        profile=profile_obj,
    )

    out_dir = os.path.join("storage", "debug_runs")
    os.makedirs(out_dir, exist_ok=True)
    html_file = os.path.join(out_dir, "report.html")
    md_file = os.path.join(out_dir, "report.md")

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_str)
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_str)

    print(f"📄 HTML Report rendered to: {html_file}")
    print(f"📄 Markdown Report rendered to: {md_file}")
    save_snapshot("step_08_output.json", recommendation)

if __name__ == "__main__":
    main()
