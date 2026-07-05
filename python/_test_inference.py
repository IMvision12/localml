"""Smoke test - exercise each supported kind with a small model.

Run:
    python _test_inference.py [suite_name...]
    python _test_inference.py quick    # cached-only
    python _test_inference.py all

Uses the same adapters the sidecar uses, so passing tests here ≈ passing in the app.
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

import os
os.environ.setdefault("HF_HOME", str(Path(os.environ.get("APPDATA", "")) / "InferML" / "hf-cache"))

from routing import inspect_model, pick_adapter  # noqa: E402
from io_utils import resolve_device  # noqa: E402

def _make_test_image():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (384, 256), (40, 60, 90))
    d = ImageDraw.Draw(img)
    d.ellipse((120, 30, 180, 90), fill=(240, 200, 180))
    d.rectangle((110, 95, 200, 220), fill=(220, 220, 220))
    d.rectangle((210, 130, 320, 180), outline=(200, 30, 30), width=4)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def _make_test_audio(seconds=1.5):
    import numpy as np
    import soundfile as sf
    sr = 16000
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False, dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    buf = io.BytesIO(); sf.write(buf, audio, sr, format="WAV")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

TEST_IMG = _make_test_image()
TEST_AUDIO = _make_test_audio()

CASES = [
    ("quick", "text-gen: Qwen3-0.6B",
     "Qwen/Qwen3-0.6B",
     {"input": {"kind": "text", "text": "The capital of France is"},
      "params": {"max_new_tokens": 12, "do_sample": False}},
     "text",
     lambda o: len(o["text"]) > 0),

    ("quick", "object-detection: detr-resnet-50",
     "facebook/detr-resnet-50",
     {"input": {"kind": "image", "dataUrl": TEST_IMG},
      "params": {"threshold": 0.3}},
     "boxes",
     lambda o: isinstance(o["boxes"], list)),

    ("quick", "segmentation: segformer (cityscapes)",
     "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
     {"input": {"kind": "image", "dataUrl": TEST_IMG}, "params": {}},
     "masks",
     lambda o: o["overlay"].startswith("data:image/png;base64,") and isinstance(o["legend"], list)),

    ("quick", "segmentation: detr-panoptic",
     "facebook/detr-resnet-50-panoptic",
     {"input": {"kind": "image", "dataUrl": TEST_IMG}, "params": {}},
     "masks",
     lambda o: o["overlay"].startswith("data:image/png;base64,") and isinstance(o["legend"], list)),

    ("all", "image-classification: vit-tiny",
     "WinKawaks/vit-tiny-patch16-224",
     {"input": {"kind": "image", "dataUrl": TEST_IMG}, "params": {"top_k": 3}},
     "labels",
     lambda o: isinstance(o["labels"], list) and len(o["labels"]) > 0 and "label" in o["labels"][0]),

    ("all", "asr: whisper-tiny",
     "openai/whisper-tiny",
     {"input": {"kind": "audio", "dataUrl": TEST_AUDIO}, "params": {}},
     "text",
     lambda o: "text" in o),

    ("all", "image-to-text: blip-base captioning",
     "Salesforce/blip-image-captioning-base",
     {"input": {"kind": "image", "dataUrl": TEST_IMG}, "params": {}},
     "text",
     lambda o: len(o["text"].strip()) > 0),

    ("all", "zero-shot-image-cls: clip-vit-base",
     "openai/clip-vit-base-patch32",
     {"input": {"kind": "image", "dataUrl": TEST_IMG, "text": "a photo of a person, a photo of a landscape, a photo of food"},
      "params": {}},
     "labels",
     lambda o: isinstance(o["labels"], list)),
]

def run_case(case):
    suite, name, model_id, payload, expected_kind, validator = case
    t0 = time.time()
    try:
        info = inspect_model(model_id)
        adapter = pick_adapter(info)
        dev = resolve_device()
        adapter.load(info, dev)
        out = adapter.run(payload["input"], payload.get("params") or {})
    except Exception as e:
        return {
            "name": name, "model": model_id, "status": "ERROR",
            "elapsed": f"{time.time()-t0:.1f}s",
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=4),
        }
    kind_ok = out.get("kind") == expected_kind
    valid = False
    try:
        valid = bool(kind_ok and validator(out))
    except Exception as e:
        return {
            "name": name, "model": model_id, "status": "INVALID",
            "elapsed": f"{time.time()-t0:.1f}s",
            "got_kind": out.get("kind"), "expected": expected_kind,
            "error": f"validator raised: {e}",
        }
    adapter_name = type(adapter).__name__
    return {
        "name": name, "model": model_id,
        "status": "PASS" if valid else "FAIL",
        "elapsed": f"{time.time()-t0:.1f}s",
        "kind": out.get("kind"), "expected_kind": expected_kind,
        "adapter": adapter_name,
        "preview": _preview(out),
    }

def _preview(o):
    k = o.get("kind")
    if k == "text":   return {"text": o["text"][:80]}
    if k == "boxes":  return {"count": len(o["boxes"]), "top": o["boxes"][:2]}
    if k == "labels": return {"count": len(o["labels"]), "top": o["labels"][:3]}
    if k == "masks":  return {"legend_count": len(o.get("legend", [])),
                              "labels": [l["label"] for l in (o.get("legend") or [])[:5]],
                              "overlay_bytes": len(o.get("overlay", ""))}
    return {"kind": k}

def main():
    suites = set(sys.argv[1:]) or {"quick"}
    cases = [c for c in CASES if c[0] in suites or "all" in suites]
    print(f"\n=== InferML inference smoke test - {len(cases)} cases on device={resolve_device()} ===\n")
    results = []
    for case in cases:
        print(f">> {case[1]}  ({case[2]})")
        r = run_case(case)
        results.append(r)
        print(json.dumps(r, indent=2, default=str))
        print()
    pas = sum(1 for r in results if r["status"] == "PASS")
    fail = sum(1 for r in results if r["status"] != "PASS")
    print("=" * 60)
    print(f"Summary: {pas} pass / {fail} fail / {len(results)} total")
    for r in results:
        mark = "[PASS]" if r["status"] == "PASS" else "[FAIL]"
        print(f"  {mark} {r['name']:45s} {r['status']:8s} {r['elapsed']}")

if __name__ == "__main__":
    main()
