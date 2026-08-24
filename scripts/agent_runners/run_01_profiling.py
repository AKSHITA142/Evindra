import sys
import os
import argparse

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.profiling.engine import ProfilingEngine
from scripts.agent_runners.utils import print_phase_header, print_input_payload, print_output_payload, save_snapshot

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Diagnostic Runner: Profiling Engine")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV dataset file")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    args = parser.parse_args()

    print_phase_header(1, "Profiling Engine", "backend.profiling.engine.ProfilingEngine")

    input_payload = {
        "file_path": args.file,
        "target_column": args.target,
    }
    print_input_payload(input_payload)

    if not os.path.isfile(args.file):
        print(f"❌ Error: File not found at '{args.file}'")
        sys.exit(1)

    print("⚙️ Executing ProfilingEngine.profile_file()...")
    profile, hints = ProfilingEngine.profile_file(args.file, target_column=args.target)

    print_output_payload(profile)
    save_snapshot("step_01_output.json", profile)

if __name__ == "__main__":
    main()
