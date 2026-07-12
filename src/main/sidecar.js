/**
 * The FastAPI server, run as a child process of the Electron shell.
 *
 * This is the same `server.cli` the web build ships - it is not modified or
 * reimplemented here. The shell allocates a loopback port, starts the server on
 * it with the browser suppressed, waits for /api/health, and points the window
 * at it. The renderer's `window.inferml` bridge already speaks HTTP+SSE to this
 * origin, so nothing in the UI or the Python engine has to know it's running
 * under Electron.
 *
 * Teardown is deliberately aggressive: a server that outlives the window keeps
 * a torch process (and GPU memory, and a locked python.exe) alive with no UI to
 * stop it.
 */
'use strict';

const { spawn } = require('child_process');
const crypto = require('crypto');
const http = require('http');
const net = require('net');
const path = require('path');

const HOST = '127.0.0.1';

// The OpenAI-compatible API is a documented integration point - people hardcode
// `http://localhost:11500/v1` into LangChain/OpenAI SDK clients. So we always
// try the canonical port first and only fall back to an ephemeral one if it's
// genuinely taken (a second instance, or something else squatting on it).
const DEFAULT_PORT = 11500;

function isFree(port) {
  return new Promise((resolve) => {
    const srv = net.createServer();
    srv.unref();
    srv.once('error', () => resolve(false));
    srv.listen(port, HOST, () => srv.close(() => resolve(true)));
  });
}

/** An ephemeral port the OS says is free right now. */
function freePort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, HOST, () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

/** The canonical port when available, otherwise any free one. */
async function pickPort() {
  if (await isFree(DEFAULT_PORT)) return DEFAULT_PORT;
  return freePort();
}

function healthOnce(port) {
  return new Promise((resolve) => {
    const req = http.get(
      { host: HOST, port, path: '/api/health', timeout: 1500 },
      (res) => {
        res.resume();
        resolve(res.statusCode === 200);
      },
    );
    req.on('error', () => resolve(false));
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
  });
}

async function waitForHealth(port, { timeoutMs = 90000, isDead } = {}) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (isDead && isDead()) throw new Error('The server process exited during startup.');
    if (await healthOnce(port)) return true;
    await new Promise((r) => setTimeout(r, 300));
  }
  throw new Error(`The server did not become healthy within ${Math.round(timeoutMs / 1000)}s.`);
}

class Sidecar {
  constructor({ pythonPath, pythonDir, dataDir }) {
    this.pythonPath = pythonPath;   // the managed venv's interpreter
    this.pythonDir = pythonDir;     // the shipped `python/` source tree
    this.dataDir = dataDir;         // Electron's userData - see below
    this.child = null;
    this.port = null;
    this.exited = false;
    this.stderr = [];

    // A fresh secret per launch, shared only with the server we're about to
    // spawn. The shell attaches it to every request the window makes; the server
    // serves the UI to nobody else. That's what stops someone opening the app in
    // a browser at localhost - they can reach the port, but they can't know this.
    // It never touches disk and dies with the process.
    this.uiToken = crypto.randomBytes(32).toString('hex');
  }

  get url() {
    return this.port ? `http://${HOST}:${this.port}` : null;
  }

  /** Start the server and resolve once it answers /api/health. */
  async start(onLog) {
    this.port = await pickPort();
    this.exited = false;

    // The server hardcodes the loopback bind and never opens a browser - it has
    // no other mode - so the port is the only thing to pass.
    this.child = spawn(
      this.pythonPath,
      ['-m', 'server.cli', '--port', String(this.port)],
      {
        cwd: this.pythonDir,
        windowsHide: true,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1',
          // cwd is enough for `-m server.cli`, but be explicit: a stray
          // PYTHONPATH in the user's shell must not shadow the shipped tree.
          PYTHONPATH: this.pythonDir,
          // Left to itself, server/appdata.py resolves its own platformdirs
          // location (%LOCALAPPDATA%\InferML\InferML, ~/.local/share/InferML),
          // which is NOT where Electron keeps userData. That would scatter the
          // venv and the chats/settings/token across two folders and make
          // "delete your data" a two-step answer. Pin the server to Electron's
          // userData so there is exactly one InferML directory per platform.
          INFERML_DATA_DIR: this.dataDir,
          // Locks the UI to this shell. See the comment on `uiToken` above.
          INFERML_UI_TOKEN: this.uiToken,
        },
      },
    );

    const tap = (buf) => {
      for (const line of String(buf).split(/\r?\n/)) {
        if (!line.trim()) continue;
        this.stderr.push(line);
        if (this.stderr.length > 200) this.stderr.shift();
        if (onLog) onLog(line);
      }
    };
    this.child.stdout.on('data', tap);
    this.child.stderr.on('data', tap);
    this.child.on('exit', () => { this.exited = true; });

    try {
      await waitForHealth(this.port, { isDead: () => this.exited });
    } catch (e) {
      // Surface what Python actually said - "did not become healthy" alone is
      // useless when the real cause is an ImportError three frames down.
      const tail = this.stderr.slice(-12).join('\n');
      this.stop();
      throw new Error(tail ? `${e.message}\n\n${tail}` : e.message);
    }
    return this.url;
  }

  stop() {
    const child = this.child;
    this.child = null;
    if (!child || child.exitCode !== null) return;
    if (process.platform === 'win32') {
      // SIGTERM doesn't reliably reach a Python child on Windows, and a
      // survivor holds python.exe open. Kill the whole tree.
      try {
        spawn('taskkill', ['/pid', String(child.pid), '/T', '/F'], { windowsHide: true });
      } catch {
        try { child.kill(); } catch { /* already gone */ }
      }
    } else {
      try { child.kill('SIGTERM'); } catch { /* already gone */ }
      // Escalate if it ignores SIGTERM (torch can block in native code).
      setTimeout(() => {
        try { if (child.exitCode === null) child.kill('SIGKILL'); } catch { /* gone */ }
      }, 3000).unref();
    }
  }
}

module.exports = { Sidecar, freePort, pickPort, waitForHealth, DEFAULT_PORT };
