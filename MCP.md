# Using InferML from Claude (MCP)

InferML ships an [MCP](https://modelcontextprotocol.io) server, `inferml-mcp`, that
hands your local models to Claude and any other MCP-speaking client. Claude can
then look at an image and detect what's in it, segment it, transcribe a recording,
speak, generate a picture, or run a small on-device LLM — all without anything
leaving your machine.

`inferml-mcp` is a **client of a running InferML server**, not a second copy of the
engine. That means it shares the models the browser UI already has loaded: the
first call to a model pays the load cost, every later call — from the UI or from
Claude — is instant. It also means **InferML has to be running**; the MCP server
will not start it for you.

## Install

```bash
pipx install "inferml[mcp]"
```

Already have InferML? Re-run the same command to add the `mcp` extra and register
the `inferml-mcp` command. From a source checkout, `pip install -e ".[mcp]"`.

The extra pulls in just `mcp` and `httpx`. The inference stack lives in the
server process, not here.

## Connect

Start the model server and leave it running:

```bash
inferml --no-browser
```

**Claude Code**

```bash
claude mcp add inferml -- inferml-mcp
```

**Claude Desktop** — add this to `claude_desktop_config.json` (Settings →
Developer → Edit Config):

```json
{
  "mcpServers": {
    "inferml": {
      "command": "inferml-mcp"
    }
  }
}
```

If the client can't find the command, use the absolute path that
`pipx list` (or `which inferml-mcp`) prints. Restart the client after editing.

**Anything else** — `inferml-mcp` speaks MCP over stdio, so any client that can
spawn a subprocess works. Pass `--url` if your server isn't on the default port.

## Check it's working

Ask Claude:

> Use inferml_status to check my local model server.

You should get back the server version, whether the inference stack is ready,
the active accelerator, and which models are resident. If it reports that the
server is unreachable, InferML isn't running.

## The tools

| Tool | What it does |
| --- | --- |
| `inferml_status` | Server reachable? Stack ready? Which accelerator, which models loaded. |
| `list_models` | Loaded, downloaded, and servable models. |
| `search_models` | Search Hugging Face, filtered to architectures InferML can run. |
| `download_model` | Fetch weights into the local cache. Idempotent. |
| `detect_objects` | Object detection, fixed-vocabulary or open-vocabulary. |
| `segment_image` | Semantic / instance / panoptic segmentation, and SAM. |
| `generate_image` | Text-to-image with a local diffusion model. |
| `transcribe_audio` | Speech to text. |
| `text_to_speech` | Text to a WAV file. |
| `generate_text` | Prompt a local LLM. |
| `embed_text` | Embed strings; returns similarity, not raw vectors. |

A few behaviours worth knowing before you use them.

**Images and audio go in as file paths.** Give Claude a path — `detect_objects`
with `image_path: "C:/photos/street.jpg"` — not base64. That's what an LLM client
actually has on hand.

**Detection boxes are normalized** to `[0,1]` as `{x, y, w, h}` with the origin at
top-left, so they don't depend on the image's resolution. Pass `labels` to switch
from the 80 fixed COCO classes to open-vocabulary detection, which finds whatever
you describe:

> Find every traffic light and fire hydrant in street.jpg.

**Segmentation and image generation return the picture inline**, so Claude can see
the result and talk about it, and they also write a PNG to disk. Speech only
writes a file — a model can't listen to audio, so inlining it would just burn
context.

**`embed_text` never returns raw vectors.** Several hundred floats per string is
unreadable and floods the context. For two or more inputs it returns the pairwise
cosine similarity matrix, which is what the vectors were for anyway.

**`generate_image` requires an explicit `model`.** Diffusion weights are
gigabytes, so there's no default — the tool won't silently download 5 GB. It does
pre-fetch the model you name (see the Windows note below); pass
`ensure_downloaded: false` to skip that.

## Which model runs

Every tool except `generate_image` has a small default, so you can start without
naming one. If you've already downloaded a model for that task in the InferML UI,
that one is preferred over the default.

| Tool | Default when you don't pass `model` |
| --- | --- |
| `detect_objects` | `facebook/detr-resnet-50` |
| `detect_objects` with `labels` | `IDEA-Research/grounding-dino-tiny` |
| `segment_image` | `nvidia/segformer-b0-finetuned-ade-512-512` |
| `transcribe_audio` | `openai/whisper-tiny` |
| `text_to_speech` | `microsoft/speecht5_tts` |
| `embed_text` | `sentence-transformers/all-MiniLM-L6-v2` |
| `generate_text` | whichever LLM is currently loaded |
| `generate_image` | none — you must pass one |

Anything else is a Hugging Face repo id. To find one:

> Search for a small object detection model, then download it.

Claude will call `search_models` and `download_model`. Results are pre-filtered to
architectures InferML supports, so anything it finds will actually load — GGUF,
Ultralytics YOLO, and other foreign runtimes are excluded.

The first call to a model downloads and loads it, which can take from seconds to
minutes. Subsequent calls reuse it.

## Where files land

Generated images and speech are written to `~/inferml-outputs`. Override per call
with `output_path`, or globally:

```bash
inferml-mcp --output-dir /path/to/outputs
```

## Configuration

| Flag | Environment variable | Default |
| --- | --- | --- |
| `--url` | `INFERML_URL` | `http://127.0.0.1:11500` |
| `--output-dir` | `INFERML_MCP_OUTPUT_DIR` | `~/inferml-outputs` |

To point Claude at an InferML running on another port or machine:

```json
{
  "mcpServers": {
    "inferml": {
      "command": "inferml-mcp",
      "args": ["--url", "http://192.168.1.50:11500"]
    }
  }
}
```

Note that InferML's HTTP API has **no authentication**. It binds `127.0.0.1` by
default, which is what keeps it private; only expose it on a LAN
(`inferml --host 0.0.0.0`) on a network you trust.

## Troubleshooting

**"Can't reach the InferML server"** — `inferml --no-browser` isn't running, or
it's on a different port. Start it, or pass `--url`.

**A tool fails and you don't know why** — ask Claude to run `inferml_status`. It
reports `ready: false` and lists `missing_packages` when the inference stack isn't
installed; run `inferml`, open the UI, and complete the first-run install.

**"This model is gated or private"** — the repo needs a Hugging Face token. Open
the InferML UI → Settings → HF Token and paste one from
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**`generate_image` fails on an uncached model** — on Windows, diffusers builds its
download file-patterns with `os.path.join`, which produces `vae\config.json`.
Hugging Face repo paths use forward slashes, so those patterns match nothing and
the component configs never download; the pipeline then fails to load. This is a
diffusers bug, not an InferML one. `generate_image` works around it by fetching
the weights through InferML's own downloader first, which is why
`ensure_downloaded` defaults to true. If you turn it off, run `download_model`
yourself first.

**`generate_text` says no LLM is loaded** — pass an explicit `model`, or open a
text-generation model in the InferML UI first.

**The first call is very slow** — it's downloading weights. `download_model` ahead
of time makes the timing visible instead of hiding it inside another tool call.
