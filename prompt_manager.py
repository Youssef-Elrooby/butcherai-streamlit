"""
ButcherAI — Prompt Manager (Exercise 4.2)
===========================================
Loads versioned prompts from the prompts/ directory.
Set ACTIVE_PROMPT_VERSION in .env to switch versions.

Usage:
    from prompt_manager import get_system_prompt
    prompt = get_system_prompt("system")  # loads prompts/v{N}_system.txt
"""

from __future__ import annotations

import os
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent / "prompts"
ACTIVE_VERSION = os.environ.get("ACTIVE_PROMPT_VERSION", "2")


def get_system_prompt(prompt_type: str = "system") -> str:
    """
    Load a versioned prompt from the prompts/ directory.

    Args:
        prompt_type: The type of prompt (e.g., "system", "reflection", "synthesis")

    Returns:
        The prompt text from the versioned file
    """
    version = ACTIVE_VERSION
    prompt_file = PROMPTS_DIR / f"v{version}_{prompt_type}.txt"

    if not prompt_file.exists():
        # Fallback to v1 if the requested version doesn't exist
        prompt_file = PROMPTS_DIR / f"v1_{prompt_type}.txt"

    if not prompt_file.exists():
        return f"Prompt file not found: {prompt_file}"

    return prompt_file.read_text(encoding="utf-8").strip()


def list_available_versions() -> list:
    """List all available prompt versions."""
    versions = set()
    for f in PROMPTS_DIR.glob("v*_*.txt"):
        # Extract version number from filename (e.g., v2_system.txt -> 2)
        parts = f.stem.split("_")
        if parts[0].startswith("v") and parts[0][1:].isdigit():
            versions.add(int(parts[0][1:]))
    return sorted(versions)


def get_prompt_history(prompt_type: str = "system") -> dict:
    """Get all versions of a prompt type."""
    history = {}
    for version in list_available_versions():
        prompt_file = PROMPTS_DIR / f"v{version}_{prompt_type}.txt"
        if prompt_file.exists():
            history[f"v{version}"] = prompt_file.read_text(encoding="utf-8").strip()
    return history


if __name__ == "__main__":
    print(f"Active prompt version: {ACTIVE_VERSION}")
    print(f"Available versions: {list_available_versions()}")
    print()
    print("Current system prompt:")
    print("-" * 40)
    print(get_system_prompt("system"))
