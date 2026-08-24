import sys
import os
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.schemas.semantic_profile import SemanticProfile
from backend.schemas.mission_brief import MissionBrief
from backend.agents.strategy_planner import StrategyPlannerAgent
from scripts.agent_runners.utils import (
    print_phase_header,
    print_input_payload,
    print_output_payload,
    save_snapshot,
    load_snapshot,
)

def main():
    parser = argparse.ArgumentParser(description="Phase 4 Diagnostic Runner: Strategy Planner Agent")
    parser.add_argument("--budget", type=int, default=5, help="Experiment budget limit")
    parser.add_argument("--profile", type=str, default=None, help="Path to SemanticProfile JSON file")
    parser.add_argument("--mission", type=str, default=None, help="Path to MissionBrief JSON file")
    args = parser.parse_args()

    print_phase_header(4, "Strategy Planner Agent", "backend.agents.StrategyPlannerAgent")

    raw_profile = load_snapshot("step_01_output.json") if not args.profile else json.load(open(args.profile))
    if not raw_profile:
        print("❌ Error: No SemanticProfile input found.")
        sys.exit(1)

    profile_obj = SemanticProfile(**raw_profile) if isinstance(raw_profile, dict) else raw_profile

    raw_mission = load_snapshot("step_03_output.json") if not args.mission else (json.load(open(args.mission)) if os.path.isfile(args.mission) else None)
    mission_obj = None
    if raw_mission and isinstance(raw_mission, dict):
        try:
            mission_obj = MissionBrief(**raw_mission)
        except Exception:
            from backend.schemas.mission_brief import MissionConstraints
            try:
                mission_obj = MissionConstraints(**raw_mission)
            except Exception:
                mission_obj = raw_mission

    inputs = {
        "semantic_profile": profile_obj,
        "experiment_budget": args.budget,
        "mission_brief": mission_obj,
    }
    print_input_payload(inputs)

    print("📊 Executing StrategyPlannerAgent.run()...")
    agent = StrategyPlannerAgent()
    experiment_plan = agent.run(inputs)

    print_output_payload(experiment_plan)
    save_snapshot("step_04_output.json", experiment_plan)

if __name__ == "__main__":
    main()
