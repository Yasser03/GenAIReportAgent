"""
report_store.py — JSON-based report persistence
"""

import json
from pathlib import Path
from datetime import datetime

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def save_report(report: dict) -> Path:
    """Save report as timestamped JSON file."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"report_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path


def load_latest_report() -> dict | None:
    """Load the most recently generated report."""
    files = sorted(REPORTS_DIR.glob("report_*.json"))
    if not files:
        return None
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)


def load_all_reports() -> list[dict]:
    """Load all reports ordered oldest → newest."""
    reports = []
    for path in sorted(REPORTS_DIR.glob("report_*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                reports.append(json.load(f))
        except Exception:
            pass
    return reports
