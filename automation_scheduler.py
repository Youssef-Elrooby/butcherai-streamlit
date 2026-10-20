"""
ButcherAI — Windows Task Scheduler Setup
==========================================
Creates Windows Scheduled Tasks to run automations automatically.
No server needed — uses Windows Task Scheduler (built into Windows).

Usage:
    python automation_scheduler.py            # Setup all tasks
    python automation_scheduler.py --list     # List scheduled tasks
    python automation_scheduler.py --remove   # Remove all tasks
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

TASK_PREFIX = "ButcherAI_"
PYTHON_EXE = sys.executable
ENGINE_SCRIPT = Path(__file__).parent / "automation_engine.py"


def _run_schtasks(args: list) -> tuple:
    """Run schtasks command and return (success, output)."""
    try:
        result = subprocess.run(
            ["schtasks"] + args,
            capture_output=True,
            text=True,
            shell=True,
        )
        return result.returncode == 0, (result.stdout or result.stderr)
    except Exception as exc:
        return False, str(exc)


def create_task(name: str, schedule: str, task: str, db_path: str) -> bool:
    """
    Create a Windows scheduled task.

    Args:
        name: Task name
        schedule: "DAILY", "WEEKLY", "HOURLY", "MINUTE"
        task: Automation task to run
        db_path: Database path
    """
    task_name = f"{TASK_PREFIX}{name}"
    command = f'"{PYTHON_EXE}" "{ENGINE_SCRIPT}" --task {task} --db "{db_path}"'

    # Map schedule to schtasks format
    if schedule == "DAILY":
        sched_args = ["/SC", "DAILY", "/ST", "06:00"]
    elif schedule == "WEEKLY":
        sched_args = ["/SC", "WEEKLY", "/D", "SUN", "/ST", "20:00"]
    elif schedule == "HOURLY":
        sched_args = ["/SC", "HOURLY"]
    else:
        sched_args = ["/SC", "DAILY", "/ST", "06:00"]

    args = [
        "/Create",
        "/TN", task_name,
        "/TR", command,
        *sched_args,
        "/F",  # Force overwrite
    ]

    success, output = _run_schtasks(args)
    if success:
        print(f"[OK] Created task: {task_name} ({schedule})")
    else:
        print(f"[FAIL] {task_name}: {output}")
    return success


def setup_all(db_path: str) -> bool:
    """Create all automation scheduled tasks."""
    tasks = [
        ("backup", "DAILY", "backup", db_path),
        ("daily_summary", "DAILY", "daily_summary", db_path),
        ("low_stock", "DAILY", "low_stock", db_path),
        ("holiday_prep", "WEEKLY", "holiday_prep", db_path),
        ("wac_monitor", "DAILY", "wac_monitor", db_path),
        ("weekly_report", "WEEKLY", "weekly_report", db_path),
        ("marketing", "DAILY", "marketing", db_path),
        ("test_run", "DAILY", "test_run", db_path),
    ]

    print("Creating Windows Scheduled Tasks...")
    print(f"Python: {PYTHON_EXE}")
    print(f"Engine:  {ENGINE_SCRIPT}")
    print(f"DB:      {db_path}")
    print()

    results = []
    for name, schedule, task, db in tasks:
        results.append(create_task(name, schedule, task, db))

    print()
    passed = sum(results)
    print(f"Created {passed}/{len(results)} tasks")
    return all(results)


def list_tasks() -> bool:
    """List all ButcherAI scheduled tasks."""
    success, output = _run_schtasks(["/Query", "/FO", "LIST"])
    if success:
        lines = [l for l in output.split("\n") if TASK_PREFIX in l]
        if lines:
            print("ButcherAI Scheduled Tasks:")
            for line in lines:
                print(f"  {line.strip()}")
        else:
            print("No ButcherAI tasks found")
    else:
        print(f"Error: {output}")
    return success


def remove_all() -> bool:
    """Remove all ButcherAI scheduled tasks."""
    success, output = _run_schtasks(["/Query", "/FO", "LIST"])
    if not success:
        print(f"Error: {output}")
        return False

    tasks = [l.split()[0] for l in output.split("\n") if TASK_PREFIX in l and l.strip()]
    if not tasks:
        print("No ButcherAI tasks to remove")
        return True

    results = []
    for task in tasks:
        ok, out = _run_schtasks(["/Delete", "/TN", task, "/F"])
        if ok:
            print(f"[OK] Removed: {task}")
        else:
            print(f"[FAIL] {task}: {out}")
        results.append(ok)

    return all(results)


def main():
    parser = argparse.ArgumentParser(description="ButcherAI Task Scheduler")
    parser.add_argument("--list", action="store_true", help="List tasks")
    parser.add_argument("--remove", action="store_true", help="Remove all tasks")
    parser.add_argument("--db", default=os.environ.get("DB_PATH", "ButcherAI.db"))

    args = parser.parse_args()

    if args.list:
        list_tasks()
    elif args.remove:
        remove_all()
    else:
        setup_all(args.db)


if __name__ == "__main__":
    main()
