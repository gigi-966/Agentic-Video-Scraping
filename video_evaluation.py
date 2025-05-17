"""
batch_video_eval_parquet.py
---------------------------
• Read a Parquet file that lists video metadata.
• For every row, locate the downloaded video file (via
  ‑ local_path  if present,
  ‑ id.mp4      fallback,
  ‑ sanitized title.mp4 fallback).
• Ask GPT‑4 Vision (or GPT‑4o) for:
    { "diversity": 0‑1 float, "faithful": true/false }
• A video *passes* if diversity ≥ 0.5  AND  faithful == true.
• Print per‑video results + overall pass‑rate.

Requires: openai ≥ 1.0.0, pandas, opencv‑python, pillow
"""

from __future__ import annotations
import base64
import json
import os
import time
import re
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import cv2
import pandas as pd
from PIL import Image
from openai import OpenAI

# ▶️ CONFIG ──────────────────────────────────────────────────────────────────
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "sk‑..."  # put key or export env
OPENAI_API_KEY = "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"     
KEYWORD = "Dog Training"
PARQUET_FILE   = "/home/mcallisterdavid/yt-sandbox/llm_class_hs_videos/search_results_20250505_104211.parquet"  # your .parquet
VIDEO_DIR      = Path("/home/mcallisterdavid/yt-sandbox/llm_class_hs_videos")                 # where .mp4 files live
EXT            = ".mp4"
NUM_FRAMES     = 5
MODEL_NAME     = "gpt-4o-mini"        # or "gpt-4o" / "gpt-4-vision-preview"
UNIQUE_THRESH  = 0.7                  # pass threshold
# ────────────────────────────────────────────────────────────────────────────

client = OpenAI(api_key=OPENAI_API_KEY)


# ─── helper functions ───────────────────────────────────────────────────────
def sanitize(text: str) -> str:
    """Create a safe filename stem from arbitrary text."""
    return re.sub(r"[^A-Za-z0-9 _\-]", "", text).strip().replace(" ", "_")


def find_video(row: pd.Series) -> Path:
    """Return Path to video file for a DataFrame row."""
    # 1) explicit local_path column
    lp = row.get("local_path", "")
    if lp and Path(lp).exists():
        return Path(lp)

    # 2) by YouTube id (id.mp4)
    vid_by_id = VIDEO_DIR / f"{row.get('id','')}{EXT}"
    if vid_by_id.exists():
        return vid_by_id

    # 3) by sanitized title
    vid_by_title = VIDEO_DIR / f"{sanitize(row['title'])}{EXT}"
    if vid_by_title.exists():
        return vid_by_title

    raise FileNotFoundError(f"No video file for row: {row['title']}")


def extract_frames(path: Path, n: int) -> List[Image.Image]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(total // n, 1)
    frames = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    return frames


def pil_to_data_uri(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def gpt4v_diversity(
    frames: List[Image.Image], prompt: str
) -> Tuple[float, bool]:
    blocks = [
        {
            "type": "text",
            "text": (
                "You will see key frames from a video.\n"
                'Return STRICT JSON: {"diversity":0‑1,"faithful":true|false}\n'
                "Definitions:\n"
                f"• diversity: 0 = generic/repetitive, 1 = very novel/rare. Score between 0-1 based on how unique the content is compared to typical {KEYWORD} videos on the internet.\n"
                "• faithful: true if video clearly depicts the prompt below else false. If the video is not about {KEYWORD} then it is not faithful.\n\n"
                f"Prompt:\n\"{prompt}\""
            ),
        }
    ] + [
        {"type": "image_url", "image_url": {"url": pil_to_data_uri(f)}}
        for f in frames
    ]

    try:
        top_p=0.98
        frequency_penalty=0.001
        presence_penalty=0.001
        temperature=0.1
        max_tokens=150
        rsp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": blocks}],
            max_tokens=max_tokens,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            top_p=top_p
        ).choices[0].message.content
        time.sleep(10)
    except Exception as e:
        print(f"Error: {e}")
        return None, False

    try:
        # Clean the response by removing markdown code block syntax if present
        cleaned_rsp = rsp.strip()
        if cleaned_rsp.startswith("```"):
            # Remove the first line (```json) and last line (```)
            cleaned_rsp = "\n".join(cleaned_rsp.split("\n")[1:-1])
        data = json.loads(cleaned_rsp)
        return float(data["diversity"]), bool(data["faithful"])
    except Exception:
        raise RuntimeError(f"Bad JSON from GPT:\n{rsp}")


# ─── main ───────────────────────────────────────────────────────────────────
def main():
    df = pd.read_parquet(PARQUET_FILE)
    print(f"Loaded {len(df)} rows from {PARQUET_FILE}")

    max_videos_to_eval = 100
    results = []
    passed = 0
    uniq_scores = []
    faithful_passed = 0
    for idx, row in df.iterrows():        
        try:
            video_path = find_video(row)
        except FileNotFoundError as e:
            print(f"[{idx+1}] {e}")
            continue

        print(f"[{idx+1}] {video_path.name}")
        frames = extract_frames(video_path, NUM_FRAMES)
        # Save frames to output folder
        output_dir = Path("output") / video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            frame_path = output_dir / f"frame_{i:03d}.jpg"
            # Save PIL Image to file
            frame.save(str(frame_path))

        uniq, faithful = gpt4v_diversity(frames, row["title"])
        if uniq is None:
            print(f"OpenAI time limit hit")
            continue
        ok = (uniq >= UNIQUE_THRESH) and faithful
        passed += ok
        results.append((video_path.name, uniq, faithful, ok))
        uniq_scores.append(uniq)
        faithful_passed += faithful
        print(f"  diversity={uniq:.2f}, faithful={faithful}, pass={ok}")

        if len(results) >= max_videos_to_eval:
            break

    if results:
        rate = passed / len(results)
        print("\n===== SUMMARY =====")
        for name, u, f, ok in results:
            print(f"{name:<40}  uniq={u:.2f}  faith={f}  pass={ok}")
        print(f"\nOverall pass‑rate: {passed}/{len(results)} = {rate:.2%}")
        # print overall uniqueness score 
        print(f"Overall uniqueness score: {sum(uniq_scores) / len(uniq_scores):.2f}")
        print(f"Overall faitthfulness pass rate: {faithful_passed}/{len(results)} = {faithful_passed/len(results):.2%}")
    else:
        print("No videos evaluated.")


if __name__ == "__main__":
    main()
