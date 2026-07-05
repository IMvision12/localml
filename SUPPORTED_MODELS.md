# Supported Models

Every supported family lives in its own folder under `python/models/`. Per-family inference code is local to the folder. Cross-cutting fallbacks (`StandardPipelineAdapter`, `DiffusersAdapter`) live in `python/adapters/` and only run when no family folder claims the model_type.

## How routing works

1. The Model Hub filter in the renderer (`src/main/services/huggingface.js`) only surfaces repos whose `model_type` tag is in `python/supported_architectures.json`. Repos in unsupported runtimes (Ultralytics YOLO, PaddleOCR, GGUF-only, ExLlama, ONNX-only, Keras, etc.) are filtered out before the user sees them.
2. When the user hits Run, the sidecar reads the repo's `model_type` and looks it up in the `python/models/` registry. O(1) dispatch, one adapter per family.
3. If no family folder claims the model_type, falls back to `DiffusersAdapter` (for `library_name: diffusers`) and then `StandardPipelineAdapter` (for any registered `pipeline_tag`). If those also miss, routing raises a clear "no adapter matched" error.

Only the family name is listed below. Every public variant of a family that ships compatible weights loads through. When a particular variant of a family does NOT load (gated, removed, needs an external runtime we don't ship), it's called out in a note.

---

## Object Detection

- DETR
- Conditional DETR
- Deformable DETR
- YOLOS *(transformers-native YOLOS only)*
- RT-DETR / RT-DETRv2
- D-FINE
- Table Transformer

> Not supported: Ultralytics YOLO (v5-v12), RF-DETR, detectron2-based models, mmdetection. They use different runtimes that InferML doesn't ship.

## Zero-Shot Object Detection

Pass comma-separated text prompts as the candidate label list.

- OWL-ViT / OWLv2
- Grounding-DINO
- MM-Grounding-DINO *(multimodal pretraining recipe over Grounding-DINO)*
- OmDet-Turbo *(real-time open-vocab detector. `omlab/omdet-turbo-swin-tiny-hf`)*

## Image Segmentation

Semantic / panoptic / instance. Cityscapes palette is applied when labels match; otherwise a distinct color per class.

- SegFormer
- MaskFormer
- Mask2Former
- OneFormer *(with semantic / instance / panoptic mode picker)*
- EoMT
- EoMT-DINOv3 *(separate model_type; DINOv3-backbone variant. `tue-mps/eomt_dinov3_*`)*
- UperNet
- BEiT
- DPT
- DETR *(panoptic-fine-tuned checkpoints only)*
- DeepLabv3 *(via MobileNet v2 / MobileViT backbones)*
- Data2Vec-Vision

## Mask Generation (SAM family)

Automatic grid-sampling + interactive point/box prompting in the SAM workspace.

- SAM
- SAM 2 / 2.1
- SAM 3
- SAM-HQ *(higher-quality boundary masks. `syscv-community/sam-hq-vit-base`)*
- EdgeTAM *(lightweight SAM variant for fast tracking. only transformers-canonical EdgeTAM repos surface; vendor / ONNX forks under `library_name: edgetam` stay out)*

> MedSAM and similar fine-tuned SAM checkpoints work. they share the SAM model_type.

## Image Classification

- ViT / DeiT / BEiT
- ResNet
- ConvNeXt / ConvNeXtV2
- Swin / SwinV2
- EfficientNet
- MobileNet v1 / v2
- MobileViT / MobileViTv2
- PoolFormer
- PVT / PVTv2
- RegNet
- FocalNet
- LeViT
- BiT
- CvT
- SegFormer *(with `*-imagenet1k` head)*
- SwiftFormer
- Any **timm** repo *(library passthrough. `library_name: timm` always loads)*

## Zero-Shot Image Classification

Comma-separated candidate labels in the text input; the model scores each against the image.

- CLIP
- SigLIP / SigLIP2
- Chinese-CLIP
- AltCLIP
- BLIP *(image-text matching)*
- MetaCLIP

## Image-to-Text (captioning, OCR)

- BLIP / BLIP-2 / InstructBLIP
- GIT
- Vision-Encoder-Decoder *(e.g. ViT-GPT2 captioning, also covers Nougat (`facebook/nougat-*`) for academic PDF OCR)*
- Pix2Struct
- TrOCR
- Donut
- MGP-STR *(`alibaba-damo/mgp-str-base`. scene text recognition)*
- GOT-OCR 2.0 *(`stepfun-ai/GOT-OCR2_0`. general-purpose OCR. handles formulas, tables, polygons. trust_remote_code preset via model_overrides.json)*

> BLIP-2 / InstructBLIP are routed but the smallest public checkpoints are >10 GB so they're impractical for typical hardware.

## Image-Text-to-Text (VLM / VQA)

Each VLM family has its own folder under `python/models/`. Custom-code families have their own `adapter.py`; pipeline-based families delegate to the standard image-text-to-text task handler.

| Family | Adapter | Notes |
|---|---|---|
| Florence-2 | `models/florence2/` (custom) | 11-task picker UI |
| Qwen-VL / 2-VL / 2.5-VL / 3-VL | `models/qwen_vl/` (custom) | |
| LLaVA / Next / OneVision / ViP-LLaVA | `models/llava/` (custom) | |
| Moondream | `models/moondream/` (custom) | |
| FastVLM | `models/fastvlm/` (custom) | Apple. `library_name: ml-fastvlm` |
| Janus / Janus-Pro | `models/janus/` (custom) | Use `deepseek-community/Janus-Pro-*` fork. 2-mode picker (Understand / Generate) |
| DeepSeek-VL v1 | `models/deepseek_vl/` (custom) | Use `deepseek-community/deepseek-vl-*` fork |
| PaliGemma / PaliGemma 2 | `models/paligemma/` | gated |
| Idefics / Idefics2 / Idefics3 | `models/idefics/` | |
| SmolVLM / SmolVLM2 | `models/smolvlm/` | |
| Kosmos-2 / Kosmos-2.5 | `models/kosmos/` | |
| InternVL | `models/internvl/` | smallest is `OpenGVLab/InternVL2_5-1B` |
| Fuyu | `models/fuyu/` | Adept's persimmon-based VLM |
| Gemma3 (multimodal) | `models/gemma3_vlm/` | gated |
| MLlama | `models/mllama/` | Llama-3.2-Vision. gated |
| MiniCPM-V | `models/minicpm_v/` | |
| Pixtral | `models/llava/` *(Pixtral declares `model_type: llava`)* | |
| Aria | `models/aria/` | rhymes-ai 25B MoE |
| Cohere2-Vision | `models/cohere2_vision/` | |
| GLM4V / GLM4V-MoE | `models/glm4v/` | THUDM |
| Emu3 | `models/emu3/` | BAAI unified gen + understanding. needs trust_remote_code |
| Chameleon | `models/chameleon/` | Meta's early-fusion VLM |
| Ovis | `models/ovis/` | AIDC-AI 1.x / 2.x. needs trust_remote_code |
| LFM2-VL | `models/lfm2_vl/` | Liquid AI 450M |
| Hunyuan-VL | `models/hunyuan_vl/` | Tencent multimodal |
| Kimi-VL | `models/kimi_vl/` | Moonshot AI |

> Not loadable today (filter excludes them):
> - **Original DeepSeek Janus repos** (`deepseek-ai/Janus-Pro-1B`, `deepseek-ai/Janus-Pro-7B`). `model_type: multi_modality`. transformers only registers `model_type: janus`. Use the **community fork** (`deepseek-community/Janus-Pro-*`) which reuploads identical weights with the canonical config.
> - **Original DeepSeek-VL v1 repos** (`deepseek-ai/deepseek-vl-1.3b-*`, `7b-*`). Same `multi_modality` problem. Use the **community fork** (`deepseek-community/deepseek-vl-*`).
> - **DeepSeek-VL v2** (`deepseek-ai/deepseek-vl2-*`). `model_type: deepseek_vl_v2` is not yet in transformers' auto-config registry. Repos ship no `modeling_*.py` so `trust_remote_code` does not help. Blocked until transformers adds support.
> - **MLCD-Embodied-7B**. Underlying `model_type` is just `qwen2`, so it surfaces under text-generation, not as a separate VLM family.
> - **transformers.js / litert-lm FastVLM forks**. Different runtimes. only `apple/FastVLM-*` originals load.

> Many of these are gated (PaliGemma, Gemma3, Llama 3.2-Vision, Pixtral). Set your HF token in Settings → HF Token before downloading. Some of the largest variants exceed common VRAM/RAM budgets. pick the smallest published variant first (LFM2-VL 450M, InternVL2.5-1B, Janus-Pro 1B, Moondream).

## Depth Estimation

- DPT
- GLPN
- ZoeDepth
- Depth-Anything *(v1 + v2)* / Prompt-Depth-Anything
- Depth Pro

## Document Question Answering

- LayoutLMv3
- Donut
- Vision-Encoder-Decoder *(Donut-flavoured DocVQA)*

> Not supported: **LayoutLM v1** needs the `tesseract` OCR binary (not bundled). **LayoutLMv2** needs `detectron2` (notoriously hard to install on Windows). Both intentionally removed; LayoutLMv3 is self-contained and covers the same use cases.

## Automatic Speech Recognition

- Whisper *(transcribe / translate toggle in the ASR workspace)*
- Wav2Vec2 / Wav2Vec2-Conformer / Wav2Vec2-BERT
- HuBERT
- WavLM
- SEW / SEW-D
- UniSpeech / UniSpeech-SAT
- Data2Vec-Audio
- Seamless-M4T / Seamless-M4T v2
- Moonshine
- Parakeet
- Kyutai STT *(`kyutai/stt-1b-en_fr` is the smallest)*
- Granite Speech *(`ibm-granite/granite-speech-3.3-2b` is the smallest. handles ASR + speech translation)*
- Voxtral *(`mistralai/Voxtral-Mini-3B-2507`. gated. The Hub filter accepts the `mistral-common` library label; the larger `Voxtral-Small-24B-2507` is tagged `vllm` and stays out)*
- Pop2Piano *(`sweetcocoa/pop2piano`. audio → piano MIDI. routed through ASR; output is symbolic tokens until a dedicated MIDI workspace is built)*

> Long audio (> 30 s) automatically routes through the chunked pipeline variant.

## Text-to-Speech

- SpeechT5 *(curated CMU-Arctic voice picker)*
- VITS *(Facebook MMS-TTS)*
- Bark
- FastSpeech 2 Conformer
- MusicGen / MusicGen Melody
- Dia *(`nari-labs/Dia-1.6B`. needs a transformers-tagged repo; the original `nari-labs` repo declares no library_name and may not surface)*
- CSM *(`sesame/csm-1b`. Sesame's Conversational Speech Model)*

## Text Generation (LLMs)

Causal LMs run through `pipeline("text-generation")`. Reasoning models (R1-distill, QwQ) auto-route through a `<think>`-stripping variant.

| Family | Folder |
|---|---|
| Llama (3.x, 4) *(gated)* | `models/llama/` |
| Mistral / Mistral3 / Ministral3 / Mixtral | `models/mistral/` |
| Qwen2 / Qwen2-MoE / Qwen3 / Qwen3-MoE | `models/qwen/` |
| Gemma / Gemma2 / Gemma3 (text) / Gemma 3n / Recurrent Gemma *(gated)* | `models/gemma/` |
| Phi / Phi3 / Phi4 / Phi-MoE / Phi-4-Multimodal | `models/phi/` |
| DeepSeek / V2 / V3 *(incl. R1-distill)* | `models/deepseek/` |
| Falcon / Falcon-Mamba / Falcon-H1 | `models/falcon/` |
| OLMo / OLMo2 / OLMo3 / OLMoE | `models/olmo/` |
| SmolLM3 | `models/smollm/` |
| GPT-OSS | `models/gpt_oss/` |
| GPT-2 / GPT-Neo / GPT-NeoX / GPT-J / GPT-BigCode | `models/gpt2/` |
| Mamba / Mamba2 | `models/mamba/` |
| RWKV | `models/rwkv/` |
| StableLM | `models/stablelm/` |
| StarCoder2 | `models/starcoder2/` |
| CodeGen | `models/codegen/` |
| Cohere / Cohere2 (Command R / R+) | `models/cohere/` |
| OPT | `models/opt/` |
| BLOOM | `models/bloom/` |
| BitNet (1.58-bit) | `models/bitnet/` |
| Granite / Granite-MoE | `models/granite/` |
| Jamba | `models/jamba/` |
| Persimmon | `models/persimmon/` |
| GLM / GLM4 / GLM-MoE-DSA | `models/glm/` |
| Nemotron / Nemotron-H | `models/nemotron/` |
| Zamba / Zamba2 | `models/zamba/` |
| EXAONE | `models/exaone/` |
| DBRX | `models/dbrx/` |
| Bamba | `models/bamba/` |
| MiniMax / MiniMax-M2 / MiMo | `models/minimax/` |
| MPT | `models/mpt/` |
| XGLM (multilingual) | `models/xglm/` |
| XLNet | `models/xlnet/` |

> Many large variants (70B+, MoE) exceed typical VRAM. Pick the smallest non-gated variant (e.g. Qwen3-0.6B, SmolLM3-3B, OLMoE-1B-7B, Phi-3-mini) to test first. AWQ / GPTQ quantized versions show up but need `pip install autoawq` / `auto-gptq` (not in `requirements.txt`). GPT-OSS, Mistral3, Ministral3, Gemma 3n are gated. set your HF token in Settings.

## Translation

- Marian *(Helsinki-NLP/opus-mt-*, ~1000 language pairs)*
- M2M-100 / NLLB-MoE / FSMT

> Not supported: **T5**, **mT5**, **mBART**, **BART**-translation fine-tunes. The T5 / BART / Pegasus families have been intentionally removed for now.

## Summarization

- ProphetNet

> Not supported: **BART**, **Pegasus**, **Pegasus-X**, **LongT5**, **T5**-summarizers. The T5 / BART / Pegasus families have been intentionally removed.

## Text-to-Image (diffusers)

Each diffusion family has its own folder under `python/models/`. Routing matches on `library_name: diffusers` + a repo pattern; per-family folders set sensible defaults (steps, guidance) without rewriting load + run.

| Family | Folder | Default steps | Default guidance |
|---|---|---|---|
| Stable Diffusion 1.5 / 2.x / 3 / 3.5 | `models/stable_diffusion/` | 30 | 7.5 |
| SDXL | `models/sdxl/` | 30 | 7.0 |
| SDXL-Turbo / SD-Turbo | `models/sdxl_turbo/` | 1 | 0.0 |
| FLUX | `models/flux/` | 4 | 0.0 *(targets schnell. dev users override)* |
| Kandinsky 2.2 / 3 | `models/kandinsky/` | 50 | 4.0 |
| Kolors | `models/kolors/` | 50 | 5.0 |
| PixArt-α / PixArt-Σ | `models/pixart/` | 20 | 4.5 |
| Sana | `models/sana/` | 20 | 4.5 |
| Playground v2 / v2.5 | `models/playground/` | 30 | 3.0 |

User-provided params still win over the per-family defaults. Any unmatched diffusers checkpoint falls through to a generic `DiffusersAdapter` with steps=20 / guidance=7.5.

> Janus also generates images. it's listed under VLMs above (image-text-to-text) because it's a unified gen + understanding model with a mode picker rather than a dedicated diffusion pipeline. Switch the chat composer's mode toggle to "Generate" and type a prompt.

## Image-to-Image / Inpainting (diffusers)

| Family | Folder | Notes |
|---|---|---|
| SDXL Refiner | `models/sdxl_refiner/` | Two-stage refinement after SDXL base |
| SD Inpainting (1.5, 2.0, SDXL) | `models/sd_inpainting/` | Needs image + mask. mask UI not yet wired |
| InstructPix2Pix | `models/instructpix2pix/` | "Make it a winter scene"-style edits |

> The diffusers library passthrough means **any** `library_name: diffusers` checkpoint also loads, even if the family isn't named above. it just gets the generic `DiffusersAdapter` defaults instead of family-specific ones.

---

## What's filtered out of the Hub

These runtimes / formats won't show up in the Hub because the sidecar can't load them. The filter rejects them by tag (`library_name` or `tags` or `config.architectures`):

| Category | Tags / libraries hit | Why |
|---|---|---|
| YOLO family | `ultralytics`, `yolov5`-`yolov12`, `yolo11`, `yolo12`, `yoloe`, `yolox` | Ultralytics runtime, not transformers |
| Other detection stacks | `rfdetr`, `rf-detr`, `detectron2`, `mmdetection`, `mmcv`, `mmpose`, `mmsegmentation` | Custom runtimes |
| Paddle stack | `paddleocr`, `paddledetection`, `paddlenlp`, `paddlepaddle` | PaddlePaddle runtime |
| Other ML frameworks | `keras`, `spacy`, `flair`, `fastai`, `stable-baselines3` | Different ecosystems |
| llama.cpp formats | `gguf`, `ggml`, `llama.cpp` | Wrong runtime; repos contain only `.gguf` weights |
| ExLlama formats | `exl2`, `exllama`, `exllamav2` | Wrong runtime; exllamav2-only weights |
| Inference-only exports | `onnx-only`, `tensorrt`, `coreml`, `tflite` | No PyTorch weights to load |
| Browser / edge | `transformers.js`, `litert-lm`, `litert` | Different runtimes (e.g. FastVLM ONNX/litert forks) |
| Removed architectures | `Deta*` class name | Class was removed from transformers |

**Quant formats kept:**
- **AWQ** (transformers loads with `autoawq`)
- **GPTQ** (transformers loads with `auto-gptq`)
- **bitsandbytes** / 8-bit / 4-bit (built-in transformers)

## Caveats

- **VRAM-heavy models** (70B+ LLMs, Mixtral-8x7B, FLUX-dev, SD-3.5-large) will OOM on 8 GB. Pick smaller variants or AWQ/GPTQ quantized versions.
- **Gated models** (most Llama, some Gemma, PaliGemma, Pixtral) need an HF token granted access on the model page. Set it in Settings → HF Token.
- **AWQ / GPTQ quantized repos** show in the Hub but require `pip install autoawq` / `pip install auto-gptq`, which aren't in `requirements.txt` by default.
- **Video models** (CogVideoX, HunyuanVideo, LTX-Video) are out of scope for v1.
- **Embeddings** (BERT, BGE, E5, Jina, Nomic, Snowflake, GTE, mxbai, MiniLM) are intentionally excluded. raw vectors aren't a useful end-user output without a similarity comparator or RAG playground around them.

## Adding a new family

The canonical path is a folder under `python/models/`. See `python/models/README.md` for the full convention; in summary:

1. **For a family that uses the standard pipeline.** Three lines:
   ```python
   # python/models/myfamily/__init__.py
   from models._pipeline_helper import make_pipeline_adapter
   MODEL_TYPES = ["myfamily"]
   TASK = "object-detection"
   ADAPTER = make_pipeline_adapter("object-detection", name="MyFamilyAdapter")
   ```
   Then add `myfamily` to the relevant task array in `python/supported_architectures.json` so the Hub filter surfaces matching repos.

2. **For a family with custom inference logic.** Add `python/models/<family>/adapter.py` with a class that subclasses `adapters.base.Adapter` and implements `load()` + `run()`. Reference `models/janus/adapter.py` or `models/florence2/adapter.py` for the pattern.

3. **For a single one-off repo that needs a knob flipped.** Add an entry to `python/model_overrides.json`:
   ```json
   { "overrides": { "owner/repo": { "trust_remote_code": true, "params": { "max_new_tokens": 512 } } } }
   ```

4. **For a fully custom adapter outside the tree.** Drop a `.py` file in `python/plugins/` with an `Adapter` subclass. Auto-discovered at sidecar startup.
