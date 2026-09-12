"""Generate synthetic Zoom-chat text from behavior captions with Claude — LABEL-BLIND.

Plan (docs/plan.md Section 17 / professor brief): for each clip's OpenFace behavior
caption, an LLM writes the chat message the student would plausibly have typed in
that 10 s window (or nothing — most windows are silent). The engagement label is
NEVER given to the model, so the generated text cannot encode the answer; the
existing audit (audit_text_leakage.py) is still run on the output as a hard gate.

The 11,206 clips share 6,614 unique captions, so we make one API call per unique
caption asking for several message variants. Per-clip assignment is deterministic
(hash of the clip path) and targets ~40% overall text coverage, prioritised toward
SILENT / LOW-AUDIO clips: a student who is speaking isn't typing, so chat text is
assigned mostly to clips whose wav energy is low (quietest tertile p=0.80, middle
p=0.35, loudest p=0.05 -> mean ~0.40). Audio RMS is an input feature, so this
conditioning introduces no label leakage.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python preprocessing/engagenet/generate_chat.py            # generate + build v3
    python preprocessing/engagenet/generate_chat.py --limit 20 # smoke test

Resumable: responses are checkpointed to chat_generation_cache.json; rerunning
skips completed captions.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import anthropic

HERE = Path(__file__).resolve().parent
SRC_ANNOTATIONS = HERE / "annotations_engagement_v2_a10.txt"
OUT_ANNOTATIONS = HERE / "annotations_engagement_v3_a10.txt"
CACHE = HERE / "chat_generation_cache.json"

MODEL = "claude-opus-5"
CONCURRENCY = 8

RMS_CACHE = HERE / "audio_rms_cache.json"
TEXT_PROBS = (0.80, 0.35, 0.05)  # quietest / middle / loudest tertile -> mean ~0.40

SCHEMA = {
    "type": "object",
    "properties": {
        "silence_prob": {"type": "number"},
        "messages": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["silence_prob", "messages"],
    "additionalProperties": False,
}

SYSTEM = """You generate realistic Zoom chat messages for a dataset of online-class recordings.

You are given a short behavioral description of a student during one 10-second webcam clip
from an online lecture (an intro ML / statistics style class). Infer what, if anything, that
student might plausibly type into the class Zoom chat around that moment.

Rules:
- Most 10-second windows have NO chat message. Estimate silence_prob: the probability this
  student types nothing in this window. Typical values are 0.85-0.98; students visibly mid-typing
  or reacting strongly might be lower.
- Provide exactly 3 candidate messages for the rare case they do type. Make them varied in
  length and register: short reactions ("lol", "yes", "^^this"), answers to an instructor
  check-for-understanding prompt, questions ("wait why do we divide by n-1?"), or off-topic
  asides if the behavior suggests disengagement.
- Write like real students: lowercase, typos occasionally, abbreviations, no perfect punctuation.
- Base the message ONLY on the behavioral description given. Never mention the description
  itself, facial expressions, or cameras — students don't narrate their own body language.
- Output JSON only."""


def unique_captions(path: Path) -> list[str]:
    caps = []
    seen = set()
    for line in path.read_text().splitlines():
        parts = line.split(";")
        if len(parts) >= 5:
            c = parts[4]
            if c and c not in seen:
                seen.add(c)
                caps.append(c)
    return caps


async def generate(captions: list[str], cache: dict) -> dict:
    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0
    lock = asyncio.Lock()

    async def one(cap: str):
        nonlocal done
        async with sem:
            for attempt in range(4):
                try:
                    r = await client.messages.create(
                        model=MODEL,
                        max_tokens=1024,
                        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                        messages=[{"role": "user", "content": f"Behavioral description: {cap}"}],
                    )
                    if r.stop_reason == "refusal":
                        raise RuntimeError("refusal")
                    text = next(b.text for b in r.content if b.type == "text")
                    obj = json.loads(text)
                    obj["silence_prob"] = min(max(float(obj["silence_prob"]), 0.0), 1.0)
                    obj["messages"] = [m.replace(";", ",").replace("\n", " ").strip() for m in obj["messages"]][:3]
                    async with lock:
                        cache[cap] = obj
                        done += 1
                        if done % 50 == 0:
                            CACHE.write_text(json.dumps(cache))
                            print(f"  {done}/{len(captions)} captions done")
                    return
                except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError):
                    await asyncio.sleep(2 ** (attempt + 1))
                except Exception as e:
                    print(f"  skipping caption ({type(e).__name__}): {cap[:60]}")
                    return

    await asyncio.gather(*(one(c) for c in captions))
    CACHE.write_text(json.dumps(cache))
    return cache


def det_uniform(key: str) -> float:
    """Deterministic pseudo-uniform in [0,1) from a string key."""
    h = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def audio_rms(wav_path: str) -> float:
    """Mean absolute amplitude of a 16-bit PCM wav; inf when unreadable (treated as loud)."""
    import wave

    import numpy as np

    try:
        with wave.open(wav_path, "rb") as w:
            frames = w.readframes(w.getnframes())
        x = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(x**2))) if x.size else float("inf")
    except Exception:
        return float("inf")


def load_rms(lines: list[list[str]]) -> dict:
    rms = json.loads(RMS_CACHE.read_text()) if RMS_CACHE.exists() else {}
    missing = [p for p in (parts[1] for parts in lines) if p not in rms]
    if missing:
        print(f"computing audio RMS for {len(missing)} wavs...")
        for i, p in enumerate(missing):
            rms[p] = audio_rms(p)
            if (i + 1) % 2000 == 0:
                RMS_CACHE.write_text(json.dumps(rms))
                print(f"  {i + 1}/{len(missing)}")
        RMS_CACHE.write_text(json.dumps(rms))
    return rms


def build_v3(cache: dict) -> tuple[int, int]:
    lines = [l.split(";") for l in SRC_ANNOTATIONS.read_text().splitlines() if l.strip()]
    rms = load_rms(lines)

    # Tertile thresholds over finite RMS values (unreadable wavs count as loudest).
    finite = sorted(v for v in rms.values() if v != float("inf"))
    t1 = finite[len(finite) // 3]
    t2 = finite[2 * len(finite) // 3]

    lines_out, with_text = [], 0
    for parts in lines:
        cap = parts[4] if len(parts) >= 5 else ""
        entry = cache.get(cap)
        chat = ""
        if entry and entry["messages"]:
            r = rms.get(parts[1], float("inf"))
            p_text = TEXT_PROBS[0] if r <= t1 else TEXT_PROBS[1] if r <= t2 else TEXT_PROBS[2]
            if det_uniform(parts[0] + "|text") < p_text:
                idx = int(det_uniform(parts[0] + "|variant") * len(entry["messages"]))
                chat = entry["messages"][idx]
                with_text += 1
        lines_out.append(";".join(parts[:4] + [chat]))
    OUT_ANNOTATIONS.write_text("\n".join(lines_out) + "\n")
    return len(lines_out), with_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only generate for first N captions (smoke test)")
    ap.add_argument("--build-only", action="store_true", help="skip generation, rebuild v3 from cache")
    args = ap.parse_args()

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    caps = unique_captions(SRC_ANNOTATIONS)
    todo = [c for c in caps if c not in cache]
    if args.limit is not None:
        todo = todo[: args.limit]
    print(f"{len(caps)} unique captions; {len(cache)} cached; generating {len(todo)}")

    if todo and not args.build_only:
        asyncio.run(generate(todo, cache))

    total, with_text = build_v3(cache)
    print(f"wrote {OUT_ANNOTATIONS.name}: {total} clips, {with_text} with chat text "
          f"({100 * with_text / total:.1f}%), {total - with_text} silent")
    print("Next: python preprocessing/engagenet/audit_text_leakage.py "
          f"--annotation_path {OUT_ANNOTATIONS}")


if __name__ == "__main__":
    main()
