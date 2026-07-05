"""Full architecture-matrix smoke test.

Runs every model_type from supported_architectures.json against the same
backend the sidecar uses (routing.pick_adapter -> adapter.load -> adapter.run).
Nothing here calls transformers directly; the test path is the production path.

Per row:
  1. Probe HF for total repo bytes via huggingface_hub.HfApi.
  2. Skip if >10 GB (toolarge), 401/403 (gated), or unreachable (no-checkpoint).
  3. Otherwise download (HF_HOME = InferML's cache), load via adapter.load(),
     run adapter.run() with a task-appropriate payload, validate output kind.
  4. Delete the cached snapshot, free CUDA + Python memory, move on.

Output: append one JSON line per case to MODEL_TEST_RESULTS_PATH.

Run:
    python _test_matrix.py                  # full sweep
    python _test_matrix.py small            # only rows whose probed size <1 GB
    python _test_matrix.py task=object-detection  # only one task category
    python _test_matrix.py model=Qwen/Qwen3-0.6B   # one specific model
"""
from __future__ import annotations

import base64
import gc
import io
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

os.environ.setdefault(
    "HF_HOME",
    str(Path(os.environ.get("APPDATA", "")) / "InferML" / "hf-cache"),
)

import _win_compat  # noqa: F401, E402

from routing import inspect_model, pick_adapter           # noqa: E402
from io_utils import resolve_device                        # noqa: E402

MAX_BYTES = 10 * 1024 * 1024 * 1024

RESULTS_PATH = HERE.parent / "test_results.jsonl"
OUTPUTS_DIR = HERE.parent / "test_outputs"
HF_CACHE_HUB = Path(os.environ["HF_HOME"]) / "hub"

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
SUMM_TEXT = (
    "The Hubble Space Telescope is a space telescope that was launched into low Earth "
    "orbit in 1990 and remains in operation. It was not the first space telescope, but "
    "it is one of the largest and most versatile, well known as a vital research tool "
    "and as a public relations boon for astronomy. The Hubble telescope is named after "
    "astronomer Edwin Hubble and is one of NASA's Great Observatories."
)

TASK_PAYLOAD = {
    "object-detection":              {"input": {"kind": "image", "dataUrl": TEST_IMG}, "params": {"threshold": 0.3}},
    "zero-shot-object-detection":    {"input": {"kind": "image", "dataUrl": TEST_IMG, "text": "person, dog, bicycle"}, "params": {"threshold": 0.1}},
    "image-segmentation":            {"input": {"kind": "image", "dataUrl": TEST_IMG}, "params": {}},
    "mask-generation":               {"input": {"kind": "image", "dataUrl": TEST_IMG}, "params": {"max_masks": 8}},
    "image-classification":          {"input": {"kind": "image", "dataUrl": TEST_IMG}, "params": {"top_k": 3}},
    "zero-shot-image-classification":{"input": {"kind": "image", "dataUrl": TEST_IMG, "text": "a photo of a person, a photo of food, a photo of a building"}, "params": {}},
    "image-to-text":                 {"input": {"kind": "image", "dataUrl": TEST_IMG}, "params": {"max_new_tokens": 32}},
    "image-text-to-text":            {"input": {"kind": "image", "dataUrl": TEST_IMG, "text": "Describe this image."}, "params": {"max_new_tokens": 32, "do_sample": False}},
    "depth-estimation":              {"input": {"kind": "image", "dataUrl": TEST_IMG}, "params": {}},
    "document-question-answering":   {"input": {"kind": "image", "dataUrl": TEST_IMG, "text": "What is in this image?"}, "params": {}},
    "automatic-speech-recognition":  {"input": {"kind": "audio", "dataUrl": TEST_AUDIO}, "params": {}},
    "text-to-speech":                {"input": {"kind": "text", "text": "Hello, this is a test."}, "params": {}},
    "text-generation":               {"input": {"kind": "text", "text": "The capital of France is"}, "params": {"max_new_tokens": 12, "do_sample": False}},
    "translation":                   {"input": {"kind": "text", "text": "Hello, how are you?"}, "params": {"max_new_tokens": 30, "tgt_lang": "fra_Latn"}},
    "summarization":                 {"input": {"kind": "text", "text": SUMM_TEXT}, "params": {"max_new_tokens": 60, "do_sample": False}},
    "text-to-image":                 {"input": {"kind": "text", "text": "a red apple on a wooden table"}, "params": {"num_inference_steps": 4}},
    "image-to-image":                {"input": {"kind": "image", "dataUrl": TEST_IMG, "text": "sunset version"}, "params": {"num_inference_steps": 4, "strength": 0.7}},
}

EXPECTED_KIND = {
    "object-detection":               "boxes",
    "zero-shot-object-detection":     "boxes",
    "image-segmentation":             "masks",
    "mask-generation":                "masks",
    "image-classification":           "labels",
    "zero-shot-image-classification": "labels",
    "image-to-text":                  "text",
    "image-text-to-text":             "text",
    "depth-estimation":               "image",
    "document-question-answering":    "text",
    "automatic-speech-recognition":   "text",
    "text-to-speech":                 "audio",
    "text-generation":                "text",
    "translation":                    "text",
    "summarization":                  "text",
    "text-to-image":                  "image",
    "image-to-image":                 "image",
}

CASES = [
    ("facebook/detr-resnet-50",                                "object-detection"),
    ("microsoft/conditional-detr-resnet-50",                   "object-detection"),
    ("SenseTime/deformable-detr",                              "object-detection"),
    ("microsoft/table-transformer-detection",                  "object-detection"),
    ("hustvl/yolos-tiny",                                      "object-detection"),
    ("PekingU/rtdetr_r18vd",                                   "object-detection"),
    ("PekingU/rtdetr_v2_r18vd",                                "object-detection"),
    ("ustc-community/dfine-nano-coco",                         "object-detection"),

    ("google/owlvit-base-patch32",                             "zero-shot-object-detection"),
    ("google/owlv2-base-patch16-ensemble",                     "zero-shot-object-detection"),
    ("IDEA-Research/grounding-dino-tiny",                      "zero-shot-object-detection"),

    ("nvidia/segformer-b0-finetuned-ade-512-512",              "image-segmentation"),
    ("facebook/maskformer-swin-tiny-coco",                     "image-segmentation"),
    ("facebook/mask2former-swin-tiny-coco-instance",           "image-segmentation"),
    ("shi-labs/oneformer_ade20k_swin_tiny",                    "image-segmentation"),
    ("openmmlab/upernet-convnext-tiny",                        "image-segmentation"),
    ("microsoft/beit-base-finetuned-ade-640-640",              "image-segmentation"),
    ("Intel/dpt-large-ade",                                    "image-segmentation"),
    ("facebook/detr-resnet-50-panoptic",                       "image-segmentation"),
    ("facebook/data2vec-vision-base-ft1k",                     "image-segmentation"),
    ("google/deeplabv3_mobilenet_v2_1.0_513",                  "image-segmentation"),
    ("apple/deeplabv3-mobilevit-xx-small",                     "image-segmentation"),
    ("tue-mps/coco_instance_eomt_large_640",                   "image-segmentation"),

    ("facebook/sam-vit-base",                                  "mask-generation"),
    ("facebook/sam2.1-hiera-tiny",                             "mask-generation"),

    ("google/vit-base-patch16-224",                            "image-classification"),
    ("facebook/deit-tiny-patch16-224",                         "image-classification"),
    ("microsoft/beit-base-patch16-224",                        "image-classification"),
    ("microsoft/resnet-50",                                    "image-classification"),
    ("facebook/convnext-tiny-224",                             "image-classification"),
    ("facebook/convnextv2-tiny-1k-224",                        "image-classification"),
    ("microsoft/swin-tiny-patch4-window7-224",                 "image-classification"),
    ("microsoft/swinv2-tiny-patch4-window8-256",               "image-classification"),
    ("google/efficientnet-b0",                                 "image-classification"),
    ("google/mobilenet_v1_1.0_224",                            "image-classification"),
    ("google/mobilenet_v2_1.0_224",                            "image-classification"),
    ("apple/mobilevit-xx-small",                               "image-classification"),
    ("apple/mobilevitv2-1.0-imagenet1k-256",                   "image-classification"),
    ("sail/poolformer_s12",                                    "image-classification"),
    ("Zetatech/pvt-tiny-224",                                  "image-classification"),
    ("OpenGVLab/pvt_v2_b0",                                    "image-classification"),
    ("facebook/regnet-y-040",                                  "image-classification"),
    ("microsoft/focalnet-tiny",                                "image-classification"),
    ("facebook/levit-128S",                                    "image-classification"),
    ("google/bit-50",                                          "image-classification"),
    ("microsoft/cvt-13",                                       "image-classification"),
    ("MBZUAI/swiftformer-xs",                                  "image-classification"),
    ("timm/resnet18.a1_in1k",                                  "image-classification"),

    ("openai/clip-vit-base-patch32",                           "zero-shot-image-classification"),
    ("google/siglip-base-patch16-224",                         "zero-shot-image-classification"),
    ("google/siglip2-base-patch16-224",                        "zero-shot-image-classification"),
    ("OFA-Sys/chinese-clip-vit-base-patch16",                  "zero-shot-image-classification"),
    ("BAAI/AltCLIP-m9",                                        "zero-shot-image-classification"),
    ("Salesforce/blip-itm-base-coco",                          "zero-shot-image-classification"),
    ("facebook/metaclip-b32-400m",                             "zero-shot-image-classification"),

    ("Salesforce/blip-image-captioning-base",                  "image-to-text"),
    ("microsoft/git-base",                                     "image-to-text"),
    ("nlpconnect/vit-gpt2-image-captioning",                   "image-to-text"),
    ("google/pix2struct-base",                                 "image-to-text"),
    ("microsoft/trocr-small-printed",                          "image-to-text"),
    ("naver-clova-ix/donut-base",                              "image-to-text"),

    ("Intel/dpt-hybrid-midas",                                 "depth-estimation"),
    ("vinvino02/glpn-kitti",                                   "depth-estimation"),
    ("Intel/zoedepth-nyu-kitti",                               "depth-estimation"),
    ("LiheYoung/depth-anything-small-hf",                      "depth-estimation"),
    ("apple/DepthPro-hf",                                      "depth-estimation"),
    ("depth-anything/prompt-depth-anything-vits-hf",           "depth-estimation"),

    ("microsoft/layoutlmv3-base",                              "document-question-answering"),
    ("naver-clova-ix/donut-base-finetuned-docvqa",             "document-question-answering"),

    ("llava-hf/llava-onevision-qwen2-0.5b-ov-hf",              "image-text-to-text"),
    ("Qwen/Qwen2-VL-2B-Instruct",                              "image-text-to-text"),
    ("Qwen/Qwen2.5-VL-3B-Instruct",                            "image-text-to-text"),
    ("Qwen/Qwen3-VL-2B-Instruct",                              "image-text-to-text"),
    ("HuggingFaceTB/SmolVLM-256M-Instruct",                    "image-text-to-text"),
    ("HuggingFaceTB/SmolVLM2-256M-Video-Instruct",             "image-text-to-text"),
    ("microsoft/kosmos-2-patch14-224",                         "image-text-to-text"),
    ("microsoft/kosmos-2.5",                                   "image-text-to-text"),
    ("microsoft/Florence-2-base",                              "image-text-to-text"),
    ("vikhyatk/moondream2",                                    "image-text-to-text"),
    ("openbmb/MiniCPM-V-2",                                    "image-text-to-text"),

    ("Qwen/Qwen2-0.5B",                                        "text-generation"),
    ("Qwen/Qwen3-0.6B",                                        "text-generation"),
    ("microsoft/phi-2",                                        "text-generation"),
    ("microsoft/Phi-3-mini-4k-instruct",                       "text-generation"),
    ("microsoft/Phi-4-mini-instruct",                          "text-generation"),
    ("deepseek-ai/deepseek-coder-1.3b-instruct",               "text-generation"),
    ("stabilityai/stablelm-2-1_6b",                            "text-generation"),
    ("bigcode/starcoder2-3b",                                  "text-generation"),
    ("Salesforce/codegen-350M-mono",                           "text-generation"),
    ("state-spaces/mamba-130m-hf",                             "text-generation"),
    ("state-spaces/mamba2-130m",                               "text-generation"),
    ("RWKV/rwkv-4-169m-pile",                                  "text-generation"),
    ("allenai/OLMo-1B-hf",                                     "text-generation"),
    ("EleutherAI/pythia-70m",                                  "text-generation"),
    ("EleutherAI/gpt-neo-125m",                                "text-generation"),
    ("gpt2",                                                   "text-generation"),
    ("facebook/opt-125m",                                      "text-generation"),
    ("bigscience/bloom-560m",                                  "text-generation"),
    ("ibm-granite/granite-3.0-2b-instruct",                    "text-generation"),
    ("ibm-granite/granite-3.0-1b-a400m-instruct",              "text-generation"),
    ("ai21labs/Jamba-tiny-dev",                                "text-generation"),
    ("Zyphra/Zamba2-1.2B",                                     "text-generation"),
    ("LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",                   "text-generation"),

    ("Helsinki-NLP/opus-mt-en-fr",                             "translation"),
    ("facebook/m2m100_418M",                                   "translation"),
    ("facebook/wmt19-en-de",                                   "translation"),

    ("microsoft/prophetnet-large-uncased-cnndm",               "summarization"),

    ("openai/whisper-tiny",                                    "automatic-speech-recognition"),
    ("facebook/wav2vec2-base-960h",                            "automatic-speech-recognition"),
    ("facebook/wav2vec2-conformer-rope-large-960h-ft",         "automatic-speech-recognition"),
    ("facebook/w2v-bert-2.0",                                  "automatic-speech-recognition"),
    ("facebook/hubert-base-ls960",                             "automatic-speech-recognition"),
    ("asapp/sew-tiny-100k-ft-ls100h",                          "automatic-speech-recognition"),
    ("microsoft/unispeech-large-1500h-cv",                     "automatic-speech-recognition"),
    ("microsoft/unispeech-sat-base-100h-libri-ft",             "automatic-speech-recognition"),
    ("microsoft/wavlm-base-plus",                              "automatic-speech-recognition"),
    ("facebook/data2vec-audio-base-960h",                      "automatic-speech-recognition"),
    ("facebook/s2t-small-librispeech-asr",                     "automatic-speech-recognition"),
    ("facebook/hf-seamless-m4t-medium",                        "automatic-speech-recognition"),
    ("facebook/seamless-m4t-v2-large",                         "automatic-speech-recognition"),
    ("UsefulSensors/moonshine-tiny",                           "automatic-speech-recognition"),
    ("nvidia/parakeet-ctc-0.6b",                               "automatic-speech-recognition"),

    ("microsoft/speecht5_tts",                                 "text-to-speech"),
    ("kakao-enterprise/vits-ljs",                              "text-to-speech"),
    ("suno/bark-small",                                        "text-to-speech"),
    ("facebook/musicgen-small",                                "text-to-speech"),

    ("stabilityai/sdxl-turbo",                                 "text-to-image"),
    ("kandinsky-community/kandinsky-2-2-decoder",              "text-to-image"),
]

def probe_repo_size(model_id):
    """Sum the bytes of every file in the repo. Returns (bytes, gated_bool)
    or (-1, True) if the repo is gated/auth-required."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(model_id, files_metadata=True)
    except Exception as e:
        msg = str(e).lower()
        if "401" in msg or "403" in msg or "gated" in msg or "restricted" in msg:
            return -1, True
        return -1, False
    total = 0
    for s in (info.siblings or []):
        sz = getattr(s, "size", None) or getattr(s, "lfs", None) and getattr(s.lfs, "size", None)
        if isinstance(sz, int):
            total += sz
    return total, False

def cleanup_cache(model_id):
    """Delete the cached snapshot directory to free disk."""
    folder = HF_CACHE_HUB / f"models--{model_id.replace('/', '--')}"
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)

def free_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def preview(out):
    k = out.get("kind")
    if k == "text":   return {"text": (out.get("text") or "")[:80]}
    if k == "boxes":  return {"count": len(out.get("boxes") or []), "labels": [b.get("label") for b in (out.get("boxes") or [])[:3]]}
    if k == "labels": return {"count": len(out.get("labels") or []), "top": [l.get("label") for l in (out.get("labels") or [])[:3]]}
    if k == "masks":  return {"legend_count": len(out.get("legend") or []), "labels": [l.get("label") for l in (out.get("legend") or [])[:5]]}
    if k == "image":  return {"has_dataUrl": bool(out.get("dataUrl"))}
    if k == "audio":  return {"has_dataUrl": bool(out.get("dataUrl"))}
    return {"kind": k}

def write_result(rec):
    rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec

def _decode_data_url(s):
    """Decode a `data:<mime>;base64,<data>` string to raw bytes."""
    if not isinstance(s, str) or not s.startswith("data:"):
        return None
    try:
        _, b64 = s.split(",", 1)
        return base64.b64decode(b64)
    except Exception:
        return None

def _safe_filename(model_id):
    return model_id.replace("/", "__")

def save_outputs(model_id, task, out, payload):
    """Persist any visual artifacts the adapter returned to OUTPUTS_DIR.
    For boxes / masks: write the server-rendered annotated PNG.
    For mask overlays: also write the standalone overlay PNG.
    For depth / text-to-image / image-to-image: write the dataUrl as PNG.
    For TTS: write the audio dataUrl as WAV.
    """
    task_dir = OUTPUTS_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)
    base = task_dir / _safe_filename(model_id)
    saved = []

    annotated = out.get("annotated")
    if annotated:
        b = _decode_data_url(annotated)
        if b:
            p = base.with_name(base.name + "_annotated.png")
            p.write_bytes(b); saved.append(p.name)
    overlay = out.get("overlay")
    if overlay:
        b = _decode_data_url(overlay)
        if b:
            p = base.with_name(base.name + "_overlay.png")
            p.write_bytes(b); saved.append(p.name)
    if out.get("kind") == "image" and out.get("dataUrl"):
        b = _decode_data_url(out["dataUrl"])
        if b:
            p = base.with_name(base.name + ".png")
            p.write_bytes(b); saved.append(p.name)
    if out.get("kind") == "audio" and out.get("dataUrl"):
        b = _decode_data_url(out["dataUrl"])
        if b:
            p = base.with_name(base.name + ".wav")
            p.write_bytes(b); saved.append(p.name)
    meta = {"model_id": model_id, "task": task, "kind": out.get("kind")}
    if out.get("kind") == "boxes":
        meta["boxes"] = out.get("boxes")
    elif out.get("kind") == "masks":
        meta["legend"] = out.get("legend")
    elif out.get("kind") == "labels":
        meta["labels"] = out.get("labels")
    elif out.get("kind") == "text":
        meta["text"] = out.get("text")
    meta_path = base.with_name(base.name + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    saved.append(meta_path.name)
    return saved

def save_test_image_once():
    """Write the synthetic test image to OUTPUTS_DIR once so users can see
    what every visual model was being shown."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    p = OUTPUTS_DIR / "_input_image.png"
    if not p.exists():
        b = _decode_data_url(TEST_IMG)
        if b:
            p.write_bytes(b)
    a = OUTPUTS_DIR / "_input_audio.wav"
    if not a.exists():
        b = _decode_data_url(TEST_AUDIO)
        if b:
            a.write_bytes(b)

def run_one(model_id, task):
    t0 = time.time()
    base = {"model_id": model_id, "task": task}

    size_bytes, gated = probe_repo_size(model_id)
    if gated:
        return write_result({**base, "status": "SKIP_GATED", "reason": "gated repo, no token", "elapsed": f"{time.time()-t0:.1f}s"})
    if size_bytes < 0:
        return write_result({**base, "status": "SKIP_UNREACHABLE", "reason": "could not probe HF", "elapsed": f"{time.time()-t0:.1f}s"})
    if size_bytes > MAX_BYTES:
        return write_result({**base, "status": "SKIP_TOOLARGE", "size_gb": round(size_bytes / 1e9, 2), "elapsed": f"{time.time()-t0:.1f}s"})

    payload = TASK_PAYLOAD.get(task)
    if payload is None:
        return write_result({**base, "status": "SKIP_NO_PAYLOAD", "reason": f"no test payload for task {task!r}", "elapsed": f"{time.time()-t0:.1f}s"})
    expected = EXPECTED_KIND.get(task)

    adapter = None
    try:
        info = inspect_model(model_id)
        adapter = pick_adapter(info)
        adapter.load(info, resolve_device())
        out = adapter.run(payload["input"], payload.get("params") or {})
    except Exception as e:
        rec = write_result({
            **base, "status": "ERROR",
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=4),
            "size_gb": round(size_bytes / 1e9, 3),
            "elapsed": f"{time.time()-t0:.1f}s",
        })
        if adapter is not None:
            del adapter
        free_memory()
        cleanup_cache(model_id)
        return rec

    kind_ok = out.get("kind") == expected
    saved_files = []
    if kind_ok:
        try:
            saved_files = save_outputs(model_id, task, out, payload)
        except Exception as e:
            print(f"  [warn] failed to save outputs for {model_id}: {e}")
    rec = write_result({
        **base,
        "status": "PASS" if kind_ok else "WRONG_KIND",
        "got_kind": out.get("kind"),
        "expected_kind": expected,
        "adapter": type(adapter).__name__,
        "preview": preview(out),
        "saved": saved_files,
        "size_gb": round(size_bytes / 1e9, 3),
        "elapsed": f"{time.time()-t0:.1f}s",
    })

    del adapter
    free_memory()
    cleanup_cache(model_id)
    return rec

def main():
    args = sys.argv[1:]
    cases = list(CASES)

    if "small" in args:
        global MAX_BYTES
        MAX_BYTES = 1 * 1024 * 1024 * 1024
    for a in args:
        if a.startswith("task="):
            t = a.split("=", 1)[1]
            cases = [c for c in cases if c[1] == t]
        elif a.startswith("model="):
            m = a.split("=", 1)[1]
            cases = [c for c in cases if c[0] == m]

    save_test_image_once()
    print(f"=== InferML matrix smoke test - {len(cases)} cases on device={resolve_device()} ===")
    print(f"  HF_HOME    = {os.environ['HF_HOME']}")
    print(f"  Results    -> {RESULTS_PATH}")
    print(f"  Outputs    -> {OUTPUTS_DIR}")
    print(f"  Size cap   = {MAX_BYTES / 1e9:.1f} GB")
    print()

    counters = {"PASS": 0, "WRONG_KIND": 0, "ERROR": 0, "SKIP_GATED": 0, "SKIP_TOOLARGE": 0, "SKIP_UNREACHABLE": 0, "SKIP_NO_PAYLOAD": 0}
    for i, (mid, task) in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {mid:60s} ({task})")
        rec = run_one(mid, task)
        counters[rec["status"]] = counters.get(rec["status"], 0) + 1
        marker = {
            "PASS": "[PASS]", "WRONG_KIND": "[WRONG]", "ERROR": "[FAIL]",
            "SKIP_GATED": "[GATED]", "SKIP_TOOLARGE": "[TOOLARGE]",
            "SKIP_UNREACHABLE": "[UNREACH]", "SKIP_NO_PAYLOAD": "[NOPLOAD]",
        }.get(rec["status"], "[??]")
        extra = rec.get("preview") or rec.get("error") or rec.get("size_gb") or ""
        print(f"  {marker:10s} {rec.get('elapsed','')}  {extra}")
        print()

    print("=" * 60)
    print(f"Summary: {sum(counters.values())} cases")
    for k, v in counters.items():
        print(f"  {k:18s} {v}")

if __name__ == "__main__":
    main()
