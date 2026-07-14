/**
 * Python interpreter discovery + the app-managed venv.
 *
 * InferML ships the Electron shell and the `python/` source tree, but NOT a
 * Python runtime - the user supplies one (3.10+). Everything that depends on
 * that decision is confined to this file: to switch to a bundled interpreter
 * later, only `findSystemPython()` has to change (return the bundled path) and
 * the rest of the app is unaffected.
 *
 * The venv lives under userData, is owned entirely by the app, and is where the
 * onboarding screen's torch install lands - runner.py runs *inside* it, so the
 * setup op (which pips into sys.executable) targets it with no changes.
 */
'use strict';

const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const MIN_PYTHON = [3, 10];

// What the engine needs to answer its first request. Deliberately small: this is
// installed synchronously on first launch, while the user watches a progress
// bar, so anything in here is time they spend staring at a spinner.
//
// The heavy stack (torch, transformers, diffusers) is NOT here - it's gigabytes,
// it depends on a CPU/GPU choice we haven't asked for yet, and it's installed
// later into this same venv by the onboarding screen.
//
// The app itself does not speak HTTP - it drives the engine over stdio. fastapi
// and uvicorn are here for the *optional* local API (python/api/), and mcp/httpx
// for the MCP server that talks to it. They're a few MB and they have to be
// present before the user can flip the setting on, so they ship in the base venv
// rather than being installed on demand.
const SERVER_DEPS = [
  'huggingface_hub',            // model search + downloads
  'platformdirs>=4',            // data dir resolution
  'psutil>=5.9',                // hardware sampling
  'fastapi>=0.110',             // the optional /v1 API
  'uvicorn[standard]>=0.29',
  'python-multipart>=0.0.9',    // /v1/audio/transcriptions takes a file upload
  'mcp>=1.2',                   // the MCP server
  'httpx>=0.27',                // ...which is an HTTP client of the API above
];

// Ask a candidate interpreter what it actually is. A candidate that isn't real
// Python (most importantly the Windows Store "app execution alias" stub, a
// 0-byte reparse point that pops open the Store instead of running) either
// fails, times out, or prints nothing - all of which we treat as "not Python".
const PROBE = 'import sys,json;print(json.dumps({"exe":sys.executable,"ver":list(sys.version_info[:3])}))';

function probe(cmd, args) {
  let r;
  try {
    r = spawnSync(cmd, [...args, '-c', PROBE], {
      encoding: 'utf8',
      timeout: 5000,
      windowsHide: true,
    });
  } catch {
    return null;
  }
  if (!r || r.error || r.status !== 0 || !r.stdout) return null;
  let info;
  try {
    info = JSON.parse(r.stdout.trim().split('\n').pop());
  } catch {
    return null;
  }
  if (!info || !info.exe || !Array.isArray(info.ver)) return null;
  return { exe: info.exe, version: info.ver };
}

function meetsMinimum(version) {
  const [maj, min] = version;
  if (maj !== MIN_PYTHON[0]) return maj > MIN_PYTHON[0];
  return min >= MIN_PYTHON[1];
}

function candidates() {
  if (process.platform === 'win32') {
    // The `py` launcher first: it resolves the newest registered interpreter
    // and, unlike bare `python`, is never the Store stub.
    return [['py', ['-3']], ['py', []], ['python', []], ['python3', []]];
  }
  return [
    ['python3', []],
    ['python', []],
    ['/opt/homebrew/bin/python3', []], // Apple Silicon Homebrew
    ['/usr/local/bin/python3', []],    // Intel Homebrew
    ['/usr/bin/python3', []],
  ];
}

/**
 * The first interpreter on this machine that is real Python >= 3.10.
 * Returns { cmd, args, exe, version } or null.
 */
function findSystemPython() {
  const tooOld = [];
  for (const [cmd, args] of candidates()) {
    const info = probe(cmd, args);
    if (!info) continue;
    if (!meetsMinimum(info.version)) {
      tooOld.push(`${info.exe} (${info.version.join('.')})`);
      continue;
    }
    return { cmd, args, exe: info.exe, version: info.version, tooOld };
  }
  return { cmd: null, args: null, exe: null, version: null, tooOld };
}

function venvDir(userData) {
  return path.join(userData, 'venv');
}

/** Path to the venv's interpreter (platform-dependent layout). */
function venvPython(userData) {
  const dir = venvDir(userData);
  return process.platform === 'win32'
    ? path.join(dir, 'Scripts', 'python.exe')
    : path.join(dir, 'bin', 'python');
}

/**
 * Path to the uv *binary* the `uv` wheel drops next to the interpreter.
 *
 * Call this, never `python -m uv`. That module is a shim that re-spawns uv.exe
 * through subprocess, and on Windows the venv's python.exe is itself a launcher
 * stub that re-execs the real interpreter - so `python -m uv` becomes four
 * nested processes, and uv's output never survives the trip back down the
 * inherited pipes. The install runs; you just never see a single line of it.
 */
function venvUv(userData) {
  const dir = venvDir(userData);
  return process.platform === 'win32'
    ? path.join(dir, 'Scripts', 'uv.exe')
    : path.join(dir, 'bin', 'uv');
}

/**
 * InferML's own wheel cache, and the env every uv call must run with.
 *
 * Left alone, uv caches into a machine-global directory (%LOCALAPPDATA%\uv\cache).
 * That is the right default for a developer tool and the wrong one for us. Our
 * wheels are enormous - a CUDA torch is 2.4GB - and the cache grows without bound:
 * on the machine this was found, it had reached 10.7GB while the venv itself was
 * 140MB. Two things follow, and both are bugs the user actually hit:
 *
 *   - "Delete Python runtime" cannot honestly free disk space it does not own. It
 *     deleted the venv and left ten gigabytes of wheels behind.
 *   - Reinstalling then restored torch from cache in seconds without downloading
 *     anything, which looks like the delete never happened.
 *
 * And we cannot simply wipe the global cache instead: it is shared, and it may
 * hold wheels the user's own projects depend on. Owning a cache under userData is
 * what makes the delete real - we can measure it, we remove it with the runtime,
 * and a reinstall genuinely re-downloads.
 */
function uvCacheDir(userData) {
  return path.join(userData, 'uv-cache');
}

function uvEnv(userData) {
  return { ...process.env, UV_CACHE_DIR: uvCacheDir(userData) };
}

/**
 * True once the venv exists AND runner.py's dependencies are importable from it.
 *
 * This must stay in step with SERVER_DEPS. It probes for what the engine needs
 * *today* - not what it needed when the venv was built - which is what makes the
 * venv self-repairing: an install carried over from the FastAPI era fails this
 * check, and boot rebuilds it instead of starting an engine that would die on
 * its first import.
 */
function isVenvReady(userData) {
  const py = venvPython(userData);
  if (!fs.existsSync(py)) return false;
  const r = spawnSync(py, ['-c', 'import huggingface_hub, platformdirs, psutil, fastapi, uvicorn, mcp, httpx, uv'], {
    encoding: 'utf8',
    timeout: 20000,
    windowsHide: true,
  });
  return !!r && !r.error && r.status === 0;
}

/**
 * Split a stream into log lines, tolerating both terminators and chunk seams.
 *
 * Two things make this less trivial than `.split('\n')`:
 *
 *  - Installers redraw progress with carriage returns, so a \r ends a line just
 *    as much as a \n does. Splitting on newlines alone means a multi-GB download
 *    emits nothing at all until it finishes, which is exactly what made torch
 *    look frozen.
 *  - 'data' chunks land on arbitrary byte boundaries, so the tail of a chunk is
 *    usually half a line. It has to be held back and glued to the next chunk, or
 *    the log comes out shredded ("+ fastapi" / "==0.139.0" on separate lines).
 */
function lineSplitter(onLog) {
  let rest = '';
  return (buf) => {
    const parts = (rest + String(buf)).split(/\r\n|\r|\n/);
    rest = parts.pop();   // may be a partial line, or a progress bar mid-redraw
    for (const line of parts) {
      if (line.trim() && onLog) onLog(line.trim());
    }
  };
}

function run(cmd, args, onLog, env) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { windowsHide: true, env: env || process.env });

    // Keep the tail of what it said. An installer that fails always explains
    // itself - "No module named pip", a resolver conflict, a proxy refusing the
    // connection - and reporting only the exit code throws that away, leaving a
    // user (and me) staring at `python.exe -m exited 1` with nothing to act on.
    const tail = [];
    const record = (line) => {
      tail.push(line);
      if (tail.length > 12) tail.shift();
      if (onLog) onLog(line);
    };

    child.stdout.on('data', lineSplitter(record));
    child.stderr.on('data', lineSplitter(record));
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) return resolve();
      const what = `${path.basename(cmd)} ${args.slice(0, 3).join(' ')}`.trim();
      const why = tail.length ? `\n\n${tail.join('\n')}` : '';
      reject(new Error(`\`${what}\` exited ${code}.${why}`));
    });
  });
}

/**
 * Create the venv (if needed) and install the server layer into it.
 * `onProgress({ step, log })` drives the bootstrap screen.
 */
async function ensureVenv(userData, systemPython, onProgress) {
  const emit = (step) => onProgress && onProgress({ step });
  const log = (line) => onProgress && onProgress({ log: line });
  const py = venvPython(userData);

  if (!fs.existsSync(py)) {
    emit('Creating the Python environment');
    fs.mkdirSync(userData, { recursive: true });
    await run(systemPython.cmd, [...systemPython.args, '-m', 'venv', venvDir(userData)], log);
  }

  // pip installs uv, and that is the last thing pip is used for. Everything after
  // this - here and in runner.py's torch install - goes through uv: it resolves
  // and downloads several times faster, and (the reason it is here at all) it
  // reports progress on a pipe. pip draws its download bar with carriage returns
  // and only flushes once the file has landed, so a multi-GB install showed
  // nothing whatsoever for minutes and looked like a hang.
  emit('Preparing the installer');
  await run(py, ['-m', 'pip', 'install', '--upgrade', 'pip', 'uv'], log);

  emit('Installing the InferML engine');
  await run(venvUv(userData), ['pip', 'install', '--python', py, ...SERVER_DEPS], log, uvEnv(userData));

  return py;
}

/**
 * Strip the inference stack, leaving the base environment exactly as it was.
 *
 * This is what "Delete Python runtime" runs. It does NOT delete the venv, and the
 * distinction matters: the engine *is* a process inside that venv, and chats,
 * model search, hardware and settings all go through it. Deleting the venv wholesale
 * takes the whole app down and forces a rebuild - creating the environment,
 * bootstrapping pip, reinstalling the engine's own dependencies - which is a
 * minute of work to get back to where we already were, and one more thing that can
 * fail and leave the app with no Python at all.
 *
 * So instead of removing everything and putting half of it back, remove only the
 * half we mean: torch, transformers, diffusers and every package that came in with
 * them. `uv pip sync` does this exactly - it makes the environment match the given
 * list and uninstalls whatever is not in it, orphaned transitive dependencies
 * included, which a plain `uv pip uninstall torch` would leave behind.
 *
 * Two details that are easy to get wrong:
 *   - sync wants a fully-resolved list, not a top-level one. Handed SERVER_DEPS
 *     directly it uninstalls starlette out from under fastapi. Hence the compile.
 *   - `uv` itself must be in the list. It is not a SERVER_DEP, so sync would
 *     happily uninstall it - and on Windows that means deleting the running
 *     uv.exe that is doing the uninstalling.
 */
async function resetToBaseEnv(userData, onProgress) {
  const log = (line) => onProgress && onProgress({ log: line });
  const py = venvPython(userData);
  const uv = venvUv(userData);
  const env = uvEnv(userData);

  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'inferml-reset-'));
  const reqs = path.join(tmp, 'base.txt');
  const lock = path.join(tmp, 'base.lock');

  try {
    fs.writeFileSync(reqs, [...SERVER_DEPS, 'uv'].join('\n') + '\n');
    await run(uv, ['pip', 'compile', '--python', py, '-q', '-o', lock, reqs], log, env);
    await run(uv, ['pip', 'sync', '--python', py, lock], log, env);
  } finally {
    try { fs.rmSync(tmp, { recursive: true, force: true }); } catch { /* a temp dir; not worth failing over */ }
  }
}

module.exports = {
  MIN_PYTHON,
  findSystemPython,
  venvDir,
  venvPython,
  uvCacheDir,
  uvEnv,
  isVenvReady,
  ensureVenv,
  resetToBaseEnv,
};
