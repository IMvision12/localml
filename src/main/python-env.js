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
  const r = spawnSync(py, ['-c', 'import huggingface_hub, platformdirs, psutil, fastapi, uvicorn, mcp, httpx'], {
    encoding: 'utf8',
    timeout: 20000,
    windowsHide: true,
  });
  return !!r && !r.error && r.status === 0;
}

function run(cmd, args, onLog) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { windowsHide: true });
    const emit = (buf) => {
      for (const line of String(buf).split(/\r?\n/)) {
        if (line.trim() && onLog) onLog(line.trim());
      }
    };
    child.stdout.on('data', emit);
    child.stderr.on('data', emit);
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${path.basename(cmd)} ${args[0] || ''} exited ${code}`));
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

  emit('Installing the InferML server');
  await run(py, ['-m', 'pip', 'install', '--upgrade', 'pip'], log);
  await run(py, ['-m', 'pip', 'install', ...SERVER_DEPS], log);

  return py;
}

module.exports = {
  MIN_PYTHON,
  findSystemPython,
  venvDir,
  venvPython,
  isVenvReady,
  ensureVenv,
};
