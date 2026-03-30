import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# override this if your llama-cli is in a separate location or if youre using a systemwide install. Mine didnt work, thats why the manual build version
LLAMA_BUILD_DIR = Path("/home/anamay/Programming/open-source/llama.cpp/build")
TOKEN_LIMIT = 1000
RESULTS_FILE = BASE_DIR / "peak_ram.json"

NOW_LOCAL = datetime.now().astimezone()
TODAY_DATE = NOW_LOCAL.date().isoformat()
UTC_OFFSET = NOW_LOCAL.strftime("%z")
UTC_OFFSET_FMT = (
    f"UTC{UTC_OFFSET[:3]}:{UTC_OFFSET[3:]}" if UTC_OFFSET else "UTC"
)
TIMEZONE_LABEL = f"{NOW_LOCAL.tzname() or 'Local'} ({UTC_OFFSET_FMT})"

SYSTEM_PROMPT_TEMPLATE = """You are an advanced Named Entity Recognition (NER) and semantic query parsing engine. Your sole function is to process natural language search queries and map them into a strictly formatted, validated JSON object. You do not engage in conversation, provide explanations, or output markdown formatting outside of the raw JSON block.

### OBJECTIVE
Extract entities such as people, date ranges, and explicit tags from the user's query. Separate the remaining descriptive elements into a generalized visual semantic string designed to be passed to an image embedding model (e.g., SigLIP or CLIP).

### JSON SCHEMA
Your output must strictly adhere to the following JSON structure:
{
  "people": ["array of strings (extract explicit names only)"],
  "date_from": "YYYY-MM-DD (ISO 8601 format, or null)",
  "date_to": "YYYY-MM-DD (ISO 8601 format, or null)",
  "tags": ["array of strings (explicitly requested categories or tags)"],
  "visual_query": "string (generalized scene description without specific names)",
  "confidence": float (0.0 to 1.0)
}

### EXTRACTION RULES
1. **People**: Extract specific human names. Do not include pronouns or generic identifiers (like "my brother" or "a guy").
2. **Dates**: Assume the current date is __TODAY_DATE__ in timezone __TIMEZONE__. Resolve relative dates (e.g., "last winter", "August 2022", "yesterday") into concrete `date_from` and `date_to` bounds. If a single day is referenced, `date_from` and `date_to` must be identical. If no date is implied, output `null` for both.
3. **Tags**: Only populate this array if the user explicitly asks results with tags x, y . then and only then add these tagsin the array
4. **Visual Query Abstraction**: This is the most critical step. Take the descriptive action or scene from the query and generalize it. Remove the specific names extracted in the "people" array and replace them with generic nouns (e.g., replace "Sarah and Jason" with "people", replace "my dog Max" with "a dog"). This string must be highly optimized for a visual embedding space.
5. **Format Restriction**: Output raw JSON only. Do not wrap the output in ```json ... ``` code blocks. Do not add any preamble.

### EXAMPLES

User: "Find pictures of Sarah and Jason skiing in the Alps last winter which could have the tags Activities, Snowboarding"
{
  "people": [
    "Sarah",
    "Jason"
  ],
  "date_from": "2025-12-01",
  "date_to": "2026-02-28",
  "tags": [
    "Activities",
    "Snowboarding"
  ],
  "visual_query": "people skiing in snowy mountain ranges",
  "confidence": 0.95
}

User: "Show me the photos of David at the beach house from July 4th 2021"
{
  "people": [
    "David"
  ],
  "date_from": "2021-07-04",
  "date_to": "2021-07-04",
  "tags": [],
  "visual_query": "person at a beach house",
  "confidence": 0.98
}

User: "just a dark moody sunset over the city skyline, no specific people"
{
  "people": [],
  "date_from": null,
  "date_to": null,
  "tags": [],
  "visual_query": "dark moody sunset over a city skyline",
  "confidence": 0.85
}"""

SYSTEM_PROMPT = (
        SYSTEM_PROMPT_TEMPLATE
        .replace("__TODAY_DATE__", TODAY_DATE)
        .replace("__TIMEZONE__", TIMEZONE_LABEL)
)

USER_QUERY = """I need to find those photos of Michael and Emma hiking through the redwood forest during our road trip last August, they might be in the album tagged Pacific Northwest or Vacation."""


def extract_json_object(text: str):
    starts = [i for i, ch in enumerate(text) if ch == "{"]
    for start in reversed(starts):
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:idx + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    return None


def find_llama_cli():
    env_cli = os.environ.get("LLAMA_CLI", "")
    if env_cli and os.path.isfile(env_cli) and os.access(env_cli, os.X_OK):
        return env_cli

    search_root = LLAMA_BUILD_DIR
    if search_root.is_file():
        search_root = search_root.parent

    candidates = [
        search_root / "bin" / "llama-cli",
        search_root / "llama-cli",
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)

    for found in search_root.rglob("llama-cli"):
        if found.is_file() and os.access(found, os.X_OK):
            return str(found)

    return None


def read_peak_kb(time_file: Path):
    content = time_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", content)
    if not match:
        return None
    return int(match.group(1))


def run_model(llama_cli: str, model_path: Path):
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {USER_QUERY}\nAssistant:"

    with tempfile.NamedTemporaryFile(prefix="llama_time_", delete=False) as tf:
        time_file = Path(tf.name)

    cmd = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(time_file),
        llama_cli,
        "-m",
        str(model_path),
        "-p",
        full_prompt,
        "-n",
        str(TOKEN_LIMIT),
        "--temp",
        "0.0",
        "--top-k",
        "1",
        "--no-display-prompt",
        "--no-conversation",
        "--single-turn",
        "--simple-io",
        "--log-disable",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        peak_kb = read_peak_kb(time_file)
    finally:
        time_file.unlink(missing_ok=True)

    stdout = (proc.stdout or "").strip()
    parsed = extract_json_object(stdout)

    return {
        "model": model_path.name,
        "exit_code": proc.returncode,
        "peak_ram_kb": peak_kb,
        "peak_ram_mb": round(peak_kb / 1024, 2) if peak_kb is not None else None,
        "parsed_json": parsed,
        "raw_output": stdout,
        "stderr": (proc.stderr or "").strip(),
    }


def main():
    if not Path("/usr/bin/time").exists():
        raise SystemExit("Error: /usr/bin/time is required but was not found.")

    llama_cli = find_llama_cli()
    if not llama_cli:
        raise SystemExit(
            f"Error: could not find executable llama-cli under: {LLAMA_BUILD_DIR}\n"
            "Set LLAMA_CLI=/full/path/to/llama-cli and run again."
        )

    if MODELS_DIR.is_dir():
        models = sorted(MODELS_DIR.glob("*.gguf"))
    else:
        models = sorted(BASE_DIR.glob("*.gguf"))

    if not models:
        raise SystemExit(f"No .gguf files found in {MODELS_DIR}")

    results = [run_model(llama_cli, model) for model in models]
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "llama_cli": llama_cli,
        "models_dir": str(MODELS_DIR),
        "token_limit": TOKEN_LIMIT,
        "system_prompt": SYSTEM_PROMPT,
        "user_query": USER_QUERY,
        "results": results,
    }

    RESULTS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved: {RESULTS_FILE}")
    print(f"Using llama-cli: {llama_cli}")
    for item in results:
        print(f"{item['model']}: peak_ram_kb={item['peak_ram_kb']} exit_code={item['exit_code']}")


if __name__ == "__main__":
    main()