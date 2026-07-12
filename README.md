<p align="center">
  <img src="assets/logo.png" alt="InferML logo" width="140" />
</p>

# InferML

Any Hugging Face model. Local. Multi-modal. A **desktop app** - download it and
run models on your own machine.

Run 143+ model families fully on-device: LLMs, VLMs, diffusion, ASR, TTS,
segmentation, detection. Nothing listens on the network, nothing leaves your
machine.

## Install

One line in your terminal:

```powershell
# Windows (PowerShell)
irm https://inferml.vercel.app/install.ps1 | iex
```

```bash
# macOS / Linux
curl -fsSL https://inferml.vercel.app/install.sh | sh
```

The script grabs the latest build for your OS from GitHub Releases and installs
it. Prefer to click? Download it from the
[**Releases**](https://github.com/IMvision12/InferML/releases/latest) page:

| Platform | File |
| --- | --- |
| Windows | `InferML-Setup-<version>.exe` |
| macOS (Apple Silicon) | `InferML-<version>-arm64.dmg` |
| macOS (Intel) | `InferML-<version>-x64.dmg` |
| Linux | `InferML-<version>.AppImage` or `.deb` |

> The builds are not code-signed yet. Downloaded in a browser, Windows
> SmartScreen will say "unknown publisher" (More info → Run anyway) and macOS
> Gatekeeper will need right-click → Open the first time. **The install script
> avoids this on macOS** - a `curl`-fetched file isn't quarantined, so Gatekeeper
> doesn't challenge it.

### Requirements

**Python 3.10 or newer**, already installed and on your PATH. InferML uses it to
build a private environment for the model runtime - it does not touch your
system packages, and it never installs Python for you.

- **Windows** - [python.org](https://www.python.org/downloads/); tick *"Add
  python.exe to PATH"* in the installer.
- **macOS** - `brew install python@3.12`
- **Linux** - `sudo apt install python3 python3-venv`

If Python is missing or too old, the app says so on launch and links you to the
download - nothing else breaks.

## First launch

1. InferML creates its own Python environment (a few seconds, once).
2. It asks whether to install the inference stack for **CPU** or **GPU** and
   fetches the matching PyTorch build. This is a one-time download of ~0.5-2.5 GB,
   with a progress bar.
3. Download a model from the Hub tab and run it.

Everything - the environment, models, chats, settings - lives in the app's data
folder, and nothing is sent anywhere.

## Lives in your tray

InferML sits in the system tray (menu bar on macOS). **Closing the window doesn't
quit it** - the engine and every model you've loaded stay resident, so reopening
is instant instead of paying the load again. Getting the window back is one click.

The tray menu has: **Open InferML**, **Launch at login**, and **Quit InferML**.
Quit is the only thing that actually stops the engine and frees the memory your
models are holding.

## The app itself never uses a port

The window is a local file and the Python engine is a child process Electron
talks to over stdin/stdout. Nothing is served, so **the InferML interface can
never be opened in a browser** - there is no URL that returns it.

That stays true even with the API below switched on. The API serves `/v1` and
nothing else; a browser pointed at it gets a 404.

## OpenAI-compatible API

Off by default. Turn it on in **Settings → API & MCP**, and InferML serves an
OpenAI-compatible API on `http://localhost:11500/v1` (any api key). It routes to
whichever LLM is loaded, or lazy-loads the one you name.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:11500/v1", api_key="not-needed")
client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

Supports streaming (`stream=True`), `GET /v1/models`, embeddings, audio, images,
and tool/function calling for the Qwen/Hermes, Llama and Mistral families.

It runs inside the engine process, so it shares the models the app already has
warm rather than loading a second copy. Because the app is tray-resident, the API
stays up with the window closed.

> **What it is:** a loopback-only listener with no CORS headers and no
> credential routes. **What it isn't:** authenticated. Any program running as you
> can use it while it's on - the same trade Ollama makes. That's why it's a
> switch rather than something always listening.

## MCP server

Gives Claude direct access to your local models. It's an HTTP client of the API
above, so **turn the API on first**.

Copy the exact command from **Settings → API & MCP**, or:

```powershell
# Windows
claude mcp add inferml -- "$env:APPDATA\InferML\venv\Scripts\python.exe" "$env:APPDATA\InferML\inferml-mcp.py"
```

```bash
# macOS
claude mcp add inferml -- \
  "$HOME/Library/Application Support/InferML/venv/bin/python" \
  "$HOME/Library/Application Support/InferML/inferml-mcp.py"
```

The app rewrites that launcher on every boot, so it survives updates.

Tools: `detect_objects`, `segment_image`, `generate_image`, `transcribe_audio`,
`text_to_speech`, `generate_text`, `embed_text`, plus `search_models`,
`download_model`, `list_models` and `inferml_status`. Keep InferML running while
you use them.

## Updating

The app checks GitHub Releases and updates itself from Settings. Updates replace
the app only - your models and the installed PyTorch stack are left alone, so a
UI fix never costs you a 2 GB re-download.

## Uninstall

Remove the app the normal way for your OS (Add/Remove Programs on Windows, drag
to Trash on macOS, `apt remove inferml` for the .deb, or delete the AppImage).

That leaves your settings and downloaded weights on disk. To wipe those too:

```bash
# macOS
rm -rf ~/Library/Application\ Support/InferML   # env, settings, chats, HF token
rm -rf ~/.cache/huggingface                     # downloaded models (GBs)
```

```bash
# Linux
rm -rf ~/.config/InferML
rm -rf ~/.cache/huggingface
```

```powershell
# Windows (PowerShell)
Remove-Item -Recurse -Force "$env:APPDATA\InferML"
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\huggingface"
```

The Hugging Face cache is **shared with every other HF tool** on the machine, so
clearing it makes `transformers`/`diffusers` re-download elsewhere - skip that
line if you'd rather keep the weights. If you set `HF_HOME` or `HF_HUB_CACHE`,
delete those locations instead.

## Development

The app is an Electron shell (`src/main/`) around a Python engine (`python/`).
The shell finds a Python, builds a venv in its data folder, and spawns
`python/runner.py` as a child process - then talks to it in newline-delimited
JSON over stdin/stdout. Nothing binds a port.

The React UI in `src/renderer/` is loaded from disk with `loadFile()`, runs with
no node integration, and reaches the engine only through `window.inferml`, which
`src/main/preload.js` defines over IPC. If a capability isn't on that object, the
UI doesn't have it.

```bash
npm install
npm start            # build the renderer + launch the app
```

Useful pieces:

| Path | What it is |
| --- | --- |
| `src/main/main.js` | boot sequence + window/tray lifecycle |
| `src/main/python-runner.js` | spawns the engine, JSON over stdin/stdout |
| `src/main/ipc.js` | every operation the UI is allowed to invoke |
| `src/main/preload.js` | defines `window.inferml` - the renderer's only door out |
| `src/main/python-env.js` | Python discovery + the app-managed venv |
| `src/main/tray.js` | tray icon, close-to-tray, launch at login |
| `python/runner.py` | the stdio protocol + dispatch table |
| `python/engine.py` | adapter cache, run/download/unload |
| `python/routing.py` | picks an adapter for a model |
| `python/models/<family>/` | one folder per model family (144 of them) |

Build installers locally:

```bash
npm run dist:win     # or dist:mac / dist:linux  → dist-app/
```

> On Windows, `dist:win` needs **Developer Mode** on (Settings → System → For
> developers). Without it, extracting electron-builder's signing toolchain fails
> on `Cannot create symbolic link` - Windows won't let a normal user create
> symlinks. CI is unaffected.

To poke at the engine without the UI, talk to it the way Electron does - it reads
one JSON object per line on stdin and answers on stdout:

```bash
cd python
echo '{"id":"1","type":"hf.search","q":"detr"}' | python -u runner.py
```

`runner.py`'s `OPS` table is the complete list of what it can do.
