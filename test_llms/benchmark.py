from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import psutil
from openai import AzureOpenAI


MODELS_DIR = Path("models")
RESULTS_FILE = Path("results.json")
LLAMA_CLI_BIN = "/home/anamay/Programming/open-source/llama.cpp/build/bin/llama-cli"

DEFAULT_TIMEOUT_S = 60
MAX_RESPONSE_TOKENS = 256
CTX_SIZE = 1024

AZURE_CONFIG = {
    "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT_URL", ""),
    "api_key": os.getenv("AZURE_OPENAI_API_KEY", ""),
    "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01"),
    "deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1"),
}

GBNF_GRAMMAR = r"""root ::= "{" ws people-field "," ws date-from-field "," ws date-to-field "," ws location-field "," ws tags-field "," ws visual-query-field "," ws confidence-field ws "}"
people-field ::= "\"people\"" ws ":" ws str-array
date-from-field ::= "\"date_from\"" ws ":" ws nullable-string
date-to-field ::= "\"date_to\"" ws ":" ws nullable-string
location-field ::= "\"location\"" ws ":" ws nullable-string
tags-field ::= "\"tags\"" ws ":" ws str-array
visual-query-field ::= "\"visual_query\"" ws ":" ws string
confidence-field ::= "\"confidence\"" ws ":" ws number
number ::= "-"? int frac? exp?
int ::= "0" | [1-9] [0-9]*
frac ::= "." [0-9] [0-9]*
exp ::= [eE] [+-]? [0-9] [0-9]*
nullable-string ::= string | "null"
str-array ::= "[" ws (string ("," ws string)* ws)? "]"
string ::= "\"" char* "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
ws ::= [ \t\n\r]*"""

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
}
"""


def _build_runtime_clock() -> tuple[str, str]:
    now_local = datetime.now().astimezone()
    today_date = now_local.date().isoformat()
    utc_offset = now_local.strftime("%z")
    utc_label = f"UTC{utc_offset[:3]}:{utc_offset[3:]}" if utc_offset else "UTC"
    timezone_label = f"{now_local.tzname() or 'Local'} ({utc_label})"
    return today_date, timezone_label


def build_system_prompt() -> str:
    today_date, timezone_label = _build_runtime_clock()
    # Avoid str.format here because the template intentionally contains many JSON braces.
    return (
        SYSTEM_PROMPT_TEMPLATE.replace("__TODAY_DATE__", today_date)
        .replace("__TIMEZONE__", timezone_label)
    )


TEST_QUERIES = [
    {
        "query": "Find photos of Sarah and Jason skiing in the Alps last winter.",
        "ground_truth": {
            "people": ["Sarah", "Jason"],
            "date_from": "2025-12-01",
            "date_to": "2026-02-28",
            "location": "Alps",
            "tags": [],
            "visual_query": "people skiing in snowy mountain landscape",
        },
    },
    {
        "query": "Show photos of David at the beach house on July 4th 2021.",
        "ground_truth": {
            "people": ["David"],
            "date_from": "2021-07-04",
            "date_to": "2021-07-04",
            "location": "beach house",
            "tags": [],
            "visual_query": "person at a beach house",
        },
    },
    {
        "query": "Show photos from Goa with the tags vacation and sunset.",
        "ground_truth": {
            "people": [],
            "date_from": None,
            "date_to": None,
            "location": "Goa",
            "tags": ["vacation", "sunset"],
            "visual_query": "travel photos in a coastal setting at sunset",
        },
    },
    {
        "query": "Photos of Rahul and Priya hiking in Ladakh from August 2022.",
        "ground_truth": {
            "people": ["Rahul", "Priya"],
            "date_from": "2022-08-01",
            "date_to": "2022-08-31",
            "location": "Ladakh",
            "tags": [],
            "visual_query": "people hiking in mountain terrain",
        },
    },
    {
        "query": "Need wedding reception shots from June 2024 with tags wedding and family.",
        "ground_truth": {
            "people": [],
            "date_from": "2024-06-01",
            "date_to": "2024-06-30",
            "location": None,
            "tags": ["wedding", "family"],
            "visual_query": "outdoor wedding reception celebration",
        },
    },
    {
        "query": "Kids playing in the park last summer.",
        "ground_truth": {
            "people": [],
            "date_from": "2025-06-01",
            "date_to": "2025-08-31",
            "location": "park",
            "tags": [],
            "visual_query": "children playing outdoors in a park",
        },
    },
    {
        "query": "Rohan and Meera engagement photos from December 2025.",
        "ground_truth": {
            "people": ["Rohan", "Meera"],
            "date_from": "2025-12-01",
            "date_to": "2025-12-31",
            "location": None,
            "tags": [],
            "visual_query": "couple engagement ceremony",
        },
    },
    {
        "query": "Black and white street photography from Mumbai.",
        "ground_truth": {
            "people": [],
            "date_from": None,
            "date_to": None,
            "location": "Mumbai",
            "tags": [],
            "visual_query": "black and white street photography in an urban city",
        },
    },
    {
        "query": "Neha singing on stage at the college fest with tags concert and performance.",
        "ground_truth": {
            "people": ["Neha"],
            "date_from": None,
            "date_to": None,
            "location": "college fest",
            "tags": ["concert", "performance"],
            "visual_query": "person singing on stage at a live event",
        },
    },
    {
        "query": "Travel photos from Europe in 2019 with Lisa.",
        "ground_truth": {
            "people": ["Lisa"],
            "date_from": "2019-01-01",
            "date_to": "2019-12-31",
            "location": "Europe",
            "tags": [],
            "visual_query": "travel sightseeing around european landmarks",
        },
    },
    {
        "query": "Indoor plant photos near the window.",
        "ground_truth": {
            "people": [],
            "date_from": None,
            "date_to": None,
            "location": "near window",
            "tags": [],
            "visual_query": "indoor plants with natural window light",
        },
    },
    {
        "query": "Just a dark moody sunset over the city skyline.",
        "ground_truth": {
            "people": [],
            "date_from": None,
            "date_to": None,
            "location": "city skyline",
            "tags": [],
            "visual_query": "dark moody sunset over city skyline",
        },
    },
]


def get_system_ram_used_mb() -> float:
    return psutil.virtual_memory().used / (1024 * 1024)


def find_llama_binary() -> str | None:
    if os.path.isfile(LLAMA_CLI_BIN) and os.access(LLAMA_CLI_BIN, os.X_OK):
        return LLAMA_CLI_BIN
    for binary in ["llama-cli", "llama.cpp", "main"]:
        found = shutil.which(binary)
        if found:
            return found
    return None


def extract_last_json_object(text: str) -> str | None:
    if not text:
        return None

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
                    candidate = text[start : idx + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break
    return None


def parse_model_output(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        extracted = extract_last_json_object(raw)
        if extracted is None:
            return {}
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            return {}


def run_model_query(
    model_path: str,
    query: str,
    llama_bin: str,
    grammar_file: str,
    system_prompt: str,
    timeout: int = DEFAULT_TIMEOUT_S,
    verbose: bool = False,
) -> tuple[str | None, float]:
    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{query}\n<|assistant|>\n"
    cmd = [
        llama_bin,
        "-m",
        model_path,
        "-p",
        prompt,
        "-n",
        str(MAX_RESPONSE_TOKENS),
        "--temp",
        "0.0",
        "--top-k",
        "1",
        "-c",
        str(CTX_SIZE),
        "--no-display-prompt",
        "--no-conversation",
        "--single-turn",
        "--simple-io",
        "-ngl",
        "0",
        "--log-disable",
        "--grammar-file",
        grammar_file,
    ]

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - start
        output = result.stdout.strip()

        if verbose:
            print(f"         [raw stdout] {output!r}")
            if result.stderr.strip():
                print(f"         [stderr] {result.stderr.strip()}")

        if result.returncode != 0 and not output:
            if not verbose and result.stderr.strip():
                print(f"         [llama-cli error] {result.stderr.strip()}")
            return None, elapsed

        extracted = extract_last_json_object(output)
        return extracted if extracted is not None else output, elapsed
    except subprocess.TimeoutExpired:
        return None, timeout
    except Exception:
        return None, time.time() - start


def heuristic_score(ground_truth: dict, model_output: dict) -> dict:
    scores = {}

    gt_people = {p.lower() for p in ground_truth.get("people", [])}
    out_people = {p.lower() for p in model_output.get("people", [])}
    if gt_people:
        scores["people_accuracy"] = len(gt_people & out_people) / len(gt_people)
    else:
        scores["people_accuracy"] = 1.0 if not out_people else 0.5

    gt_date = (ground_truth.get("date_from"), ground_truth.get("date_to"))
    out_date = (model_output.get("date_from"), model_output.get("date_to"))
    scores["date_accuracy"] = 1.0 if gt_date == out_date else 0.0

    gt_loc = str(ground_truth.get("location") or "").lower().strip()
    out_loc = str(model_output.get("location") or "").lower().strip()
    visual_text = str(model_output.get("visual_query") or "").lower()
    if gt_loc:
        scores["location_accuracy"] = 1.0 if gt_loc in out_loc or gt_loc in visual_text else 0.0
    else:
        scores["location_accuracy"] = 1.0 if not out_loc else 0.5

    gt_tags = {t.lower() for t in ground_truth.get("tags", [])}
    out_tags = {t.lower() for t in model_output.get("tags", [])}
    if gt_tags:
        scores["tags_accuracy"] = len(gt_tags & out_tags) / len(gt_tags)
    else:
        scores["tags_accuracy"] = 1.0 if not out_tags else 0.5

    visual_query = str(model_output.get("visual_query") or "").strip()
    scores["visual_query_quality"] = 1.0 if len(visual_query) >= 8 else 0.0
    scores["json_validity"] = 1.0 if model_output else 0.0

    scores["overall"] = sum(scores.values()) / len(scores)
    scores["notes"] = "heuristic scoring"
    return scores


def judge_ner_output(query: str, ground_truth: dict, model_output: dict) -> dict:
    if not AZURE_CONFIG["api_key"] or not AZURE_CONFIG["endpoint"]:
        return heuristic_score(ground_truth, model_output)

    client = AzureOpenAI(
        azure_endpoint=AZURE_CONFIG["endpoint"],
        api_key=AZURE_CONFIG["api_key"],
        api_version=AZURE_CONFIG["api_version"],
    )

    judge_prompt = f"""You are evaluating a Named Entity Recognition output for a photo search system.

Query:
{query}

Ground Truth:
{json.dumps(ground_truth, indent=2)}

Model Output:
{json.dumps(model_output, indent=2)}

Score each item from 0.0 to 1.0:
1) people_accuracy
2) date_accuracy
3) location_accuracy
4) tags_accuracy
5) visual_query_quality
6) json_validity

Return only:
{{
  "people_accuracy": 0.0,
  "date_accuracy": 0.0,
  "location_accuracy": 0.0,
  "tags_accuracy": 0.0,
  "visual_query_quality": 0.0,
  "json_validity": 0.0,
  "overall": 0.0,
  "notes": "brief explanation"
}}"""

    try:
        response = client.chat.completions.create(
            model=AZURE_CONFIG["deployment"],
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)
    except Exception as exc:
        print(f"    [judge error] {exc}; falling back to heuristic")
        return heuristic_score(ground_truth, model_output)


def benchmark_model(
    model_path: Path,
    llama_bin: str,
    grammar_file: str,
    system_prompt: str,
    verbose: bool = False,
) -> dict:
    print(f"\n{'=' * 60}")
    print(f"  Model: {model_path.name}")
    print(f"{'=' * 60}")

    ram_before = get_system_ram_used_mb()
    query_results = []

    for idx, case in enumerate(TEST_QUERIES, start=1):
        query = case["query"]
        ground_truth = case["ground_truth"]
        print(f"  [{idx:02d}/{len(TEST_QUERIES)}] {query}")

        raw_output, elapsed = run_model_query(
            model_path=str(model_path),
            query=query,
            llama_bin=llama_bin,
            grammar_file=grammar_file,
            system_prompt=system_prompt,
            verbose=verbose,
        )
        parsed = parse_model_output(raw_output)
        scores = judge_ner_output(query, ground_truth, parsed)

        print(
            f"         -> score={scores.get('overall', 0.0):.2f} "
            f"time={elapsed:.1f}s json_ok={'yes' if parsed else 'NO'}"
        )
        if scores.get("notes"):
            print(f"         -> {scores['notes']}")

        query_results.append(
            {
                "query": query,
                "ground_truth": ground_truth,
                "raw_output": raw_output,
                "parsed": parsed,
                "scores": scores,
                "elapsed_s": elapsed,
            }
        )

    ram_after = get_system_ram_used_mb()
    ram_delta_mb = max(0.0, ram_after - ram_before)

    score_keys = [
        "people_accuracy",
        "date_accuracy",
        "location_accuracy",
        "tags_accuracy",
        "visual_query_quality",
        "json_validity",
        "overall",
    ]
    avg_scores = {
        key: round(sum(row["scores"].get(key, 0.0) for row in query_results) / len(query_results), 3)
        for key in score_keys
    }
    avg_time = sum(row["elapsed_s"] for row in query_results) / len(query_results)

    result = {
        "model": model_path.name,
        "disk_mb": round(model_path.stat().st_size / (1024 * 1024), 1),
        "ram_delta_mb": round(ram_delta_mb, 1),
        "avg_time_s": round(avg_time, 2),
        "avg_scores": avg_scores,
        "query_results": query_results,
    }

    print("\n  Summary")
    print(f"  Disk: {result['disk_mb']} MB")
    print(f"  RAM delta: {result['ram_delta_mb']} MB")
    print(f"  Avg time/query: {result['avg_time_s']}s")
    print(f"  NER overall: {avg_scores['overall']:.3f}")
    print(
        "  People: "
        f"{avg_scores['people_accuracy']:.3f} | "
        f"Dates: {avg_scores['date_accuracy']:.3f} | "
        f"Visual: {avg_scores['visual_query_quality']:.3f}"
    )
    return result


def print_summary_table(all_results: list[dict]) -> None:
    print(f"\n\n{'=' * 90}")
    print("  FINAL BENCHMARK RESULTS")
    print(f"{'=' * 90}")
    print(
        f"{'Model':<45} {'Disk':>7} {'RAM':>7} {'Time':>6} "
        f"{'NER':>6} {'People':>8} {'Dates':>7} {'Visual':>8}"
    )
    print("-" * 90)

    ordered = sorted(all_results, key=lambda row: -row["avg_scores"]["overall"])
    for row in ordered:
        scores = row["avg_scores"]
        print(
            f"{row['model'][:44]:<45} {row['disk_mb']:>6.0f}M {row['ram_delta_mb']:>6.0f}M "
            f"{row['avg_time_s']:>5.1f}s {scores['overall']:>6.3f} "
            f"{scores['people_accuracy']:>8.3f} {scores['date_accuracy']:>7.3f} "
            f"{scores['visual_query_quality']:>8.3f}"
        )
    print("=" * 90)

    proposal_rows = [
        {
            "model": row["model"].replace(".gguf", ""),
            "disk_mb": row["disk_mb"],
            "peak_ram_mb": row["ram_delta_mb"],
            "ner_score": row["avg_scores"]["overall"],
            "people_score": row["avg_scores"]["people_accuracy"],
            "date_score": row["avg_scores"]["date_accuracy"],
            "visual_score": row["avg_scores"]["visual_query_quality"],
            "avg_time_s": row["avg_time_s"],
        }
        for row in ordered
    ]
    print("\nJSON TABLE (paste into proposal):")
    print(json.dumps(proposal_rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="GSoC digiKam NER benchmark")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print raw model output")
    args = parser.parse_args()

    print("GSoC digiKam NER benchmark")
    print(f"Models dir: {MODELS_DIR.resolve()}")
    print(
        "Azure judge: "
        f"{'ENABLED' if AZURE_CONFIG['api_key'] and AZURE_CONFIG['endpoint'] else 'DISABLED (heuristic fallback)'}"
    )
    print("Mode: llama-cli single-turn, no-conversation")

    llama_bin = find_llama_binary()
    if not llama_bin:
        print("\nERROR: llama-cli not found.")
        print("Install llama.cpp/llama-cpp-python and ensure llama-cli is executable.")
        sys.exit(1)
    print(f"llama binary: {llama_bin}")

    models = sorted(MODELS_DIR.glob("*.gguf"))
    if not models:
        print(f"ERROR: no .gguf files found in {MODELS_DIR}")
        sys.exit(1)
    print(f"Found {len(models)} model(s): {[m.name for m in models]}\n")

    system_prompt = build_system_prompt()
    grammar_fd, grammar_file = tempfile.mkstemp(suffix=".gbnf")

    try:
        with os.fdopen(grammar_fd, "w") as handle:
            handle.write(GBNF_GRAMMAR)

        all_results = []
        for model_path in models:
            try:
                result = benchmark_model(
                    model_path=model_path,
                    llama_bin=llama_bin,
                    grammar_file=grammar_file,
                    system_prompt=system_prompt,
                    verbose=args.verbose,
                )
                all_results.append(result)

                with open(RESULTS_FILE, "w", encoding="utf-8") as handle:
                    json.dump(all_results, handle, indent=2)
                print(f"  [saved to {RESULTS_FILE}]")
            except KeyboardInterrupt:
                print("\nInterrupted. Saving partial results...")
                break
            except Exception as exc:
                print(f"  ERROR benchmarking {model_path.name}: {exc}")
                traceback.print_exc()

        if all_results:
            print_summary_table(all_results)
            with open(RESULTS_FILE, "w", encoding="utf-8") as handle:
                json.dump(all_results, handle, indent=2)
            print(f"\nFull results saved to: {RESULTS_FILE}")
    finally:
        os.unlink(grammar_file)


if __name__ == "__main__":
    main()
