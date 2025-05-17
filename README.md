# Agentic Video‑Scraping Utilities

This repo contains two standalone Python scripts that together enable
1) automatic generation of diverse, visually grounded video search prompts and  
2) post‑hoc evaluation of downloaded videos with GPT‑4 Vision.

| Script | Purpose |
| ------ | ------- |
| **`prompt_generation.py`** | Generates meta‑prompts and concrete prompts for a given keyword, scores them for *diversity* and *faithfulness*, and writes the accepted prompts to `generated_prompts.json`. |
| **`video_evaluation.py`**  | Reads a Parquet catalog of videos, finds the corresponding `.mp4` files, extracts key frames, and asks GPT‑4 Vision to score each video for *diversity* and *faithfulness*. Aggregates pass‑rate statistics. |

---

## Quick Start

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install core requirements
pip install openai pandas opencv-python pillow nltk tqdm pyarrow matplotlib
# (plus any extras your workflow needs)

# 3. Set your OpenAI key (never hard‑code it!)
export OPENAI_API_KEY="sk-..."   # or use direnv / secrets manager
