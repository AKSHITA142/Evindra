import os
import json
from typing import Any, Dict, Optional
from pydantic import BaseModel

# ANSI Color codes for clean CLI terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SNAPSHOT_DIR = os.path.join(PROJECT_ROOT, "storage", "debug_runs")

def print_phase_header(phase_num: int, phase_name: str, module_name: str):
    banner = f"--- PHASE {phase_num}: {phase_name.upper()} ({module_name}) ---"
    line = "=" * len(banner)
    print(f"\n{CYAN}{BOLD}{line}")
    print(banner)
    print(f"{line}{RESET}\n")

def print_input_payload(data: Any, label: str = "AGENT INPUT PAYLOAD"):
    print(f"{YELLOW}{BOLD}📥 [{label}]{RESET}")
    if isinstance(data, BaseModel):
        formatted = json.dumps(data.model_dump(), indent=2, default=str)
    elif isinstance(data, dict) or isinstance(data, list):
        formatted = json.dumps(data, indent=2, default=str)
    else:
        formatted = str(data)
    print(f"{formatted}\n")

def print_output_payload(data: Any, label: str = "AGENT STRUCTURED OUTPUT"):
    print(f"{GREEN}{BOLD}📤 [{label}]{RESET}")
    if isinstance(data, BaseModel):
        formatted = json.dumps(data.model_dump(), indent=2, default=str)
    elif isinstance(data, dict) or isinstance(data, list):
        formatted = json.dumps(data, indent=2, default=str)
    else:
        formatted = str(data)
    print(f"{formatted}\n")

def serialize_payload(data: Any) -> Any:
    if isinstance(data, BaseModel):
        return data.model_dump()
    if isinstance(data, list):
        return [serialize_payload(item) for item in data]
    if isinstance(data, dict):
        return {k: serialize_payload(v) for k, v in data.items()}
    return data

def save_snapshot(filename: str, data: Any):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    filepath = os.path.join(SNAPSHOT_DIR, filename)
    payload = serialize_payload(data)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"💾 Snapshot saved to: {filepath}")

def load_snapshot(filename: str) -> Optional[Dict[str, Any]]:
    filepath = os.path.join(SNAPSHOT_DIR, filename)
    if not os.path.isfile(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
