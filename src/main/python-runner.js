/**
 * The Python engine, as a child process.
 *
 * Replaces the FastAPI/uvicorn sidecar. Instead of binding a port and speaking
 * HTTP to itself over loopback, the engine is now a plain child process that we
 * talk to in newline-delimited JSON over its stdin/stdout (see python/runner.py
 * for the protocol). Nothing listens on the network, so nothing on the machine
 * can reach the engine except this process.
 *
 * `call()` returns a promise per request id; long operations also stream
 * `progress` frames, which is how model downloads and the runtime install
 * report back without polling. Unsolicited `event` frames (hardware ticks,
 * store changes) are broadcast to whoever subscribed.
 *
 * The engine is restarted lazily rather than eagerly: if it dies (OOM during a
 * 7B load is the realistic case), every in-flight promise rejects with a real
 * message and the *next* call spawns a fresh one. Eager respawning would fight
 * a crash loop.
 */
'use strict';

const { spawn } = require('child_process');
const path = require('path');
const readline = require('readline');

class PythonRunner {
  constructor({ pythonPath, pythonDir, dataDir, version }) {
    this.pythonPath = pythonPath;   // the managed venv's interpreter
    this.pythonDir = pythonDir;     // the shipped `python/` tree
    this.dataDir = dataDir;         // Electron's userData
    this.version = version || '0.0.0';

    this.proc = null;
    this.seq = 0;
    this.pending = new Map();       // id -> { resolve, reject, onProgress }
    this.listeners = new Map();     // event name -> Set<cb>
    this.stderr = [];               // ring buffer, surfaced in diagnostics
    this.starting = null;
  }

  get running() {
    return !!this.proc && this.proc.exitCode === null;
  }

  /** Spawn the engine. Resolves when it reports `ready`. */
  start() {
    if (this.running) return Promise.resolve();
    if (this.starting) return this.starting;

    this.starting = new Promise((resolve, reject) => {
      const script = path.join(this.pythonDir, 'runner.py');

      // -u: unbuffered. Without it Python holds stdout in a 4KB buffer and every
      // response arrives late, or not at all - the protocol just appears to hang.
      this.proc = spawn(this.pythonPath, ['-u', script], {
        cwd: this.pythonDir,
        windowsHide: true,
        stdio: ['pipe', 'pipe', 'pipe'],
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1',
          PYTHONIOENCODING: 'utf-8',
          // cwd is enough, but be explicit: a stray PYTHONPATH in the user's
          // shell must not shadow the shipped tree.
          PYTHONPATH: this.pythonDir,
          // Pin the engine's data directory to Electron's userData. Left alone,
          // appdata.py would resolve its own platformdirs location, which is a
          // *different* folder - and chats, settings and the HF token would
          // quietly split across two of them.
          INFERML_DATA_DIR: this.dataDir,
          // package.json is the single source of truth for the version, and the
          // Python side has no way to read it. It needs one because /api/health
          // reports it and the MCP server's `inferml_status` tool surfaces it.
          INFERML_VERSION: this.version,
        },
      });

      const ready = setTimeout(
        () => reject(new Error('The Python engine did not start within 60s.')),
        60000,
      );

      readline.createInterface({ input: this.proc.stdout }).on('line', (line) => {
        let frame;
        try {
          frame = JSON.parse(line);
        } catch {
          return;   // not ours; runner.py keeps stray prints off stdout, but be safe
        }
        if (frame.event === 'ready') {
          clearTimeout(ready);
          resolve();
          return;
        }
        this._onFrame(frame);
      });

      readline.createInterface({ input: this.proc.stderr }).on('line', (line) => {
        if (!line.trim()) return;
        this.stderr.push(line);
        if (this.stderr.length > 300) this.stderr.shift();
      });

      this.proc.on('error', (e) => {
        clearTimeout(ready);
        reject(e);
      });

      this.proc.on('exit', (code) => {
        clearTimeout(ready);
        this._failAllPending(
          `The Python engine exited (code ${code}).\n\n${this.stderr.slice(-10).join('\n')}`,
        );
        this.proc = null;
        this.starting = null;
      });
    }).finally(() => { this.starting = null; });

    return this.starting;
  }

  _onFrame(frame) {
    if (frame.event) {
      const subs = this.listeners.get(frame.event);
      if (subs) for (const cb of subs) { try { cb(frame.data); } catch { /* a bad subscriber is not our problem */ } }
      return;
    }

    const p = this.pending.get(frame.id);
    if (!p) return;

    if (frame.progress) {
      if (p.onProgress) { try { p.onProgress(frame.progress); } catch { /* ditto */ } }
      return;   // not terminal - the request is still running
    }

    this.pending.delete(frame.id);
    if (frame.ok) p.resolve(frame.result);
    else p.reject(new Error(frame.error || 'unknown engine error'));
  }

  _failAllPending(message) {
    for (const [, p] of this.pending) p.reject(new Error(message));
    this.pending.clear();
  }

  /**
   * Invoke an op. `onProgress` receives streaming frames for long operations
   * (download, setup) and is ignored for everything else.
   */
  async call(type, payload = {}, onProgress = null) {
    await this.start();
    const id = String(++this.seq);
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      try {
        this.proc.stdin.write(JSON.stringify({ id, type, ...payload }) + '\n');
      } catch (e) {
        this.pending.delete(id);
        reject(new Error(`Could not reach the Python engine: ${e.message}`));
      }
    });
  }

  /** Subscribe to a broadcast (`hw:update`, `chats:updated`, `hf:installsChanged`). */
  on(event, cb) {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event).add(cb);
    return () => this.listeners.get(event).delete(cb);
  }

  stop() {
    const proc = this.proc;
    this.proc = null;
    if (!proc || proc.exitCode !== null) return;

    // Closing stdin is the graceful path: runner.py's read loop sees EOF and
    // exits on its own. But a Python blocked inside a native torch call won't
    // notice, so we escalate.
    try { proc.stdin.end(); } catch { /* already gone */ }

    if (process.platform === 'win32') {
      // SIGTERM is not a real thing on Windows and does not reliably reach a
      // Python child. A survivor holds GPU memory and a locked python.exe with
      // no UI left to stop it, so kill the tree.
      try {
        spawn('taskkill', ['/pid', String(proc.pid), '/T', '/F'], { windowsHide: true });
      } catch {
        try { proc.kill(); } catch { /* already gone */ }
      }
    } else {
      try { proc.kill('SIGTERM'); } catch { /* already gone */ }
      setTimeout(() => {
        try { if (proc.exitCode === null) proc.kill('SIGKILL'); } catch { /* gone */ }
      }, 3000).unref();
    }
  }
}

module.exports = { PythonRunner };
