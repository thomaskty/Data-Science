
from pathlib import Path
import json

INCOMING_DIR = Path("data/Portal")
def fetch_new_cases():
    cases = []
    for file in INCOMING_DIR.glob("*.json"):
        with open(file, "r") as f:
            data = json.load(f)
        cases.append(data)
    return cases
