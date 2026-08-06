#!/usr/bin/env python3
"""
Standalone Diagnostic Tool for Groq API Integration.

This script tests connectivity, authentication, model validity, and completion
performance for both the reasoning model (GROQ_MODEL_REASONING) and the coding
model (GROQ_MODEL_CODE) configured in your environment.

Usage:
    python backend/test_groq.py
"""

import os
import sys
import time
import json
import logging
from dotenv import load_dotenv

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure backend directory is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.providers import GroqProvider

# Setup logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("GroqDiagnostic")


def mask_api_key(key: str) -> str:
    """Returns a masked representation of the API key for safe display."""
    if not key:
        return "<MISSING>"
    if len(key) <= 8:
        return "gsk_***"
    return f"{key[:6]}...{key[-4:]}"


def run_diagnostic():
    print("=" * 65)
    print("         ANALYST.AI — GROQ API DIAGNOSTIC SUITE          ")
    print("=" * 65)

    # 1. Environment Loading
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"[OK] Loaded environment variables from {env_path}")
    else:
        load_dotenv()
        print("[INFO] Loaded environment variables from system defaults")

    # 2. Configuration Inspection
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    reasoning_model = os.getenv("GROQ_MODEL_REASONING", "deepseek-r1-distill-llama-70b").strip()
    code_model = os.getenv("GROQ_MODEL_CODE", "qwen-2.5-coder-32b").strip()
    timeout = int(os.getenv("LLM_REQUEST_TIMEOUT", "60"))

    print("\n--- Configuration Settings ---")
    print(f"Provider          : groq")
    print(f"API Key           : {mask_api_key(api_key)}")
    print(f"Reasoning Model   : {reasoning_model}")
    print(f"Code Model        : {code_model}")
    print(f"Timeout           : {timeout} seconds")
    print("-" * 35)

    if not api_key:
        print("\n[ERROR] GROQ_API_KEY is not set or empty in .env!")
        print("Please add your key to .env: GROQ_API_KEY=gsk_your_key_here")
        sys.exit(1)

    # 3. Instantiate Groq Provider
    try:
        provider = GroqProvider()
    except Exception as e:
        print(f"\n[ERROR] Failed to instantiate GroqProvider: {e}")
        sys.exit(1)

    test_results = []

    # 4. Test Reasoning Model
    print(f"\n[Test 1/2] Testing Reasoning Model: '{reasoning_model}'...")
    reasoning_prompt = "What is data analysis? Provide a clear 1-sentence answer."
    reasoning_sys_prompt = "You are a concise enterprise data analyst."

    print(f"Request Payload (Sanitized):")
    print(json.dumps({
        "model": reasoning_model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": reasoning_sys_prompt},
            {"role": "user", "content": reasoning_prompt}
        ]
    }, indent=2))

    t0 = time.time()
    try:
        response_text = provider.generate(
            prompt=reasoning_prompt,
            system_prompt=reasoning_sys_prompt,
            model=reasoning_model,
            temperature=0.1
        )
        elapsed = time.time() - t0
        print(f"\n[SUCCESS] Request Succeeded ({elapsed:.2f}s)")
        print(f"Response:\n{response_text}\n")
        test_results.append(("Reasoning Model", reasoning_model, True, f"{elapsed:.2f}s", len(response_text)))
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n[FAILED] Request Failed ({elapsed:.2f}s)")
        print(f"Error Details:\n{e}\n")
        test_results.append(("Reasoning Model", reasoning_model, False, f"{elapsed:.2f}s", 0))

    # 5. Test Code Generation Model
    print(f"\n[Test 2/2] Testing Code Model: '{code_model}'...")
    code_prompt = "Calculate the average value of column 'salary' in dataframe df."
    code_sys_prompt = "Goal: Write ONLY valid, executable python pandas code to analyze dataframe df. Assign to variable result."

    print(f"Request Payload (Sanitized):")
    print(json.dumps({
        "model": code_model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": code_sys_prompt},
            {"role": "user", "content": code_prompt}
        ]
    }, indent=2))

    t0 = time.time()
    try:
        response_text = provider.generate(
            prompt=code_prompt,
            system_prompt=code_sys_prompt,
            model=code_model,
            temperature=0.1
        )
        elapsed = time.time() - t0
        print(f"\n[SUCCESS] Request Succeeded ({elapsed:.2f}s)")
        print(f"Response:\n{response_text}\n")
        test_results.append(("Code Model", code_model, True, f"{elapsed:.2f}s", len(response_text)))
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n[FAILED] Request Failed ({elapsed:.2f}s)")
        print(f"Error Details:\n{e}\n")
        test_results.append(("Code Model", code_model, False, f"{elapsed:.2f}s", 0))

    # 6. Diagnostic Summary Table
    print("\n" + "=" * 65)
    print("                 GROQ API DIAGNOSTIC SUMMARY             ")
    print("=" * 65)
    all_passed = True
    for name, m_name, success, latency, char_count in test_results:
        status_str = "SUCCESS" if success else "FAILED"
        if not success:
            all_passed = False
        print(f"• {name:<18} [{m_name:<30}] : {status_str} ({latency}, {char_count} chars)")
    print("=" * 65)

    if all_passed:
        print("\n[OK] ALL GROQ API TESTS PASSED SUCCESSFULLY! The integration is ready for production.")
        sys.exit(0)
    else:
        print("\n[FAIL] ONE OR MORE TESTS FAILED. Review error output above to resolve issues.")
        sys.exit(1)


if __name__ == "__main__":
    run_diagnostic()
