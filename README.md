# Model Benchmark Suite

This workspace contains two independent benchmarking tracks:

- `test_llms/`: GGUF LLM benchmark for NER and semantic query parsing quality + peak RAM.
- `test_embedding_ram/`: ONNX embedding-model smoke test + per-model peak RAM.

Both tracks are used to evaluate models for the digiKam natural-language photo search pipeline.

---

## Repository layout

- `test_llms/benchmark.py`: runs multi-query NER benchmark and writes `test_llms/results.json`.
- `test_llms/run_llama_peak_ram.py`: runs one representative prompt per GGUF model and writes `test_llms/peak_ram.json`.
- `test_embedding_ram/run_onnx_once.py`: runs one ONNX model once and prints JSON metadata for inputs/outputs.
- `test_embedding_ram/run_all_onnx_peak_ram.sh`: runs all ONNX models with `/usr/bin/time -v` and writes `test_embedding_ram/result.json`.

---

## 1) LLM benchmark (`test_llms`)

### What it measures

Each GGUF model is evaluated on 12 test queries across:

- `people_accuracy`
- `date_accuracy`
- `location_accuracy`
- `tags_accuracy`
- `visual_query_quality`
- `json_validity`

Scoring uses Azure OpenAI as judge when credentials are present; otherwise it falls back to heuristic scoring.

### Requirements

- Python 3.10+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) with `llama-cli`
- GNU time at `/usr/bin/time`

Install dependencies:

```bash
cd test_llms
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure llama binary

The benchmark script currently points to a local path in `test_llms/benchmark.py`:

```python
LLAMA_CLI_BIN = "/home/anamay/Programming/open-source/llama.cpp/build/bin/llama-cli"
```

Update this constant if your binary is elsewhere.

For `test_llms/run_llama_peak_ram.py`, either:

- set `LLAMA_CLI=/full/path/to/llama-cli`, or
- update `LLAMA_BUILD_DIR` in the script.

### Run quality benchmark

```bash
cd test_llms
source venv/bin/activate
python benchmark.py
# verbose stdout/stderr per query
python benchmark.py -v
```

Output: `test_llms/results.json` (saved incrementally after each model).

### Run LLM peak RAM benchmark

```bash
cd test_llms
source venv/bin/activate
python run_llama_peak_ram.py
```

Output: `test_llms/peak_ram.json`.

### Optional Azure judge configuration

```bash
export AZURE_OPENAI_ENDPOINT_URL="https://<resource>.openai.azure.com/"
export AZURE_OPENAI_API_KEY="<key>"
export AZURE_OPENAI_DEPLOYMENT="gpt-4.1"
# optional override
export AZURE_OPENAI_API_VERSION="2025-01-01"
```

### Current LLM snapshot

From `test_llms/results.json` and `test_llms/peak_ram.json` currently in this workspace:

| Model | NER overall | Avg time/query | Peak RAM (MB) |
|---|---:|---:|---:|
| `llama-3.2-1b-instruct-q4_k_m` | 0.816 | 10.5s | 5234 |
| `google_gemma-3-1b-it-Q4_K_M` | 0.731 | 11.1s | 1181 |
| `qwen2.5-0.5b-instruct-q4_k_m` | n/a | n/a | 982 |
| `qwen2.5-1.5b-instruct-q4_k_m` | n/a | n/a | 2557 |
| `smollm2-1.7b-instruct-q4_k_m` | n/a | n/a | 3353 |

Note: peak RAM values above are from `run_llama_peak_ram.py` and can differ from `ram_delta_mb` in `benchmark.py`.

---

## 2) Embedding RAM benchmark (`test_embedding_ram`)

### What it does

Runs each `.onnx` model in an isolated process, feeds synthetic text/image tensors based on model input metadata, and records:

- peak RSS (`/usr/bin/time -v`)
- model status (`ok` or `error(code)`)
- parsed output tensor summaries (dtype, shape, min/max/mean, sample values)

### Requirements

- Python 3.10+
- GNU time at `/usr/bin/time`

Install dependencies:

```bash
cd test_embedding_ram
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run all ONNX models

```bash
cd test_embedding_ram
source venv/bin/activate
bash run_all_onnx_peak_ram.sh
```

Optional custom query and output path:

```bash
OUT_JSON=/tmp/onnx_result.json bash run_all_onnx_peak_ram.sh "a photo of a golden retriever in sunlight"
```

Output: `test_embedding_ram/result.json` (default).

### Run a single ONNX model once

```bash
cd test_embedding_ram
source venv/bin/activate
python run_onnx_once.py models/vision_model_int8.onnx --query "a photo of a cat on a chair"
```

### Current ONNX peak RAM snapshot

From `test_embedding_ram/result.json` currently in this workspace:

| Model | Status | Peak RSS (MB) |
|---|---|---:|
| `clip-vit-base-patch16-textual-qint8.onnx` | ok | 159 |
| `clip-vit-base-patch16-visual-qint8.onnx` | ok | 208 |
| `siglip2_text_model_int8.onnx` | ok | 531 |
| `vision_model_int8.onnx` | ok | 199 |

---

## Notes

- `test_embedding_ram/run_onnx_once.py` downloads a sample image from `https://picsum.photos/512` for image-input models.
- If your `llama-cli` build no longer supports some flags used in these scripts, adapt arguments to your local llama.cpp version.

---

## Context

This benchmark workspace was developed for benchmarking models to digiKam natural-language photo search, covering both:

- structured NER/query parsing quality on small GGUF LLMs, and
- memory footprint validation for ONNX text/vision embedding models.
