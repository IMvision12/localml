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

const { uvCacheDir } = require('./python-env');

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
    this.suspended = null;   // when set, the reason the engine may NOT start, ever
    this.gate = null;        // when set, a promise that resolves when calls may resume
    this.openGate = null;
    this.replay = [];        // requests that outlived the engine, awaiting the next one
  }

  get running() {
    return !!this.proc && this.proc.exitCode === null;
  }

  /**
   * Take the engine offline for a moment, parking callers instead of failing them.
   *
   * The engine is restarted on demand by `call()`, which is normally what you want
   * and is exactly wrong while its packages are being uninstalled: a user who clicks
   * over to Models mid-removal would trigger a search, respawn Python out of an
   * environment that is halfway gone, and re-lock the files being deleted.
   *
   * Rejecting those calls stops the damage but is a poor answer to give a user who
   * did nothing wrong - they get an error for clicking a button during a routine
   * nine-second operation. So hold them at the gate instead. They wait, the engine
   * comes back, and they run. The UI never learns anything happened.
   */
  hold() {
    if (this.gate) return;
    this.gate = new Promise((resolve) => { this.openGate = resolve; });
  }

  /** Let parked callers through (they re-enter start(), which may now reject). */
  release() {
    const open = this.openGate;
    this.gate = null;
    this.openGate = null;
    if (open) open();
  }

  /**
   * Refuse to start at all, with the reason to report. Terminal, unlike hold():
   * for when there is no working interpreter left and waiting would be a lie.
   * Without this, every later call fails with a raw `spawn ... ENOENT` instead.
   *
   * Always follow with release(), or anything already parked waits forever.
   */
  suspend(reason) {
    this.suspended = reason || 'The Python runtime is unavailable.';
    // Nothing is coming to run these. Waiting forever is the one outcome worse
    // than an error, so give them the error - it at least says what happened.
    for (const p of this.replay) p.reject(new Error(this.suspended));
    this.replay = [];
  }

  resume() { this.suspended = null; }

  /** Wait for in-flight requests to finish, so a deliberate stop doesn't kill them. */
  async drain(timeoutMs = 5000) {
    const deadline = Date.now() + timeoutMs;
    while (this.pending.size && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 50));
    }
  }

  /** Spawn the engine. Resolves when it reports `ready`. */
  start() {
    if (this.suspended) return Promise.reject(new Error(this.suspended));
    // Park behind the gate, then try again - by which time the engine is back up
    // and this resolves normally, or it is suspended and this rejects with why.
    if (this.gate) return this.gate.then(() => this.start());
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
          // The engine installs torch with uv (runner.py's setup op). Point it at
          // the app's own wheel cache, not uv's machine-global one - otherwise the
          // multi-GB wheels land somewhere we can neither measure nor delete, and
          // "Delete Python runtime" silently leaves them on disk.
          UV_CACHE_DIR: uvCacheDir(this.dataDir),
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
          this._flushReplay();   // anything the previous engine died holding
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

      const mine = this.proc;
      this.proc.on('exit', (code) => {
        clearTimeout(ready);
        if (this.gate) {
          // We killed it on purpose and another engine is coming. The caller is
          // still holding a promise; hand their request to the next one.
          this._parkPending();
        } else {
          this._failAllPending(
            `The Python engine exited (code ${code}).\n\n${this.stderr.slice(-10).join('\n')}`,
          );
        }
        // Only if this is still *the* engine. A stop that times out lets the next
        // one spawn before the old one's exit event lands, and clearing the fields
        // unconditionally would orphan the live process: `running` goes false, a
        // third engine gets spawned, and its frames arrive on a pipe nobody owns.
        if (this.proc === mine) {
          this.proc = null;
          this.starting = null;
        }
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
    const text = this.suspended || message;
    for (const [, p] of this.pending) p.reject(new Error(text));
    this.pending.clear();
  }

  /**
   * Set in-flight requests aside to be re-sent to the next engine.
   *
   * These are requests the UI made in good faith that happened to be in the pipe
   * when we deliberately killed the process - a hardware poll, a size check, a
   * chat list that landed as the user clicked "delete runtime". Failing them
   * produces an error the user cannot act on, for something they did not do wrong.
   * Draining first catches most of them, but not one whose write lands in the
   * microtask *after* the drain check, which is why the failure kept coming back.
   *
   * Re-sending is safe because these are reads and whole-object writes: running one
   * twice is the same as running it once. The exception is the long, visible work -
   * an inference or a download - where silently starting over would be a surprise
   * and, once the runtime is gone, pointless. Those get told the truth.
   */
  _parkPending() {
    const NO_REPLAY = new Set(['tasks.run', 'tasks.download', 'tasks.setup']);
    for (const [, p] of this.pending) {
      if (NO_REPLAY.has(p.type)) {
        p.reject(new Error('Cancelled: the Python runtime was removed.'));
      } else {
        this.replay.push(p);
      }
    }
    this.pending.clear();
  }

  _flushReplay() {
    const queued = this.replay;
    this.replay = [];
    for (const p of queued) this._send(p);
  }

  /**
   * Put a request on the wire, allocating its id.
   *
   * The payload is nested, never spread across the frame. Spread, an argument
   * called `id` - and six ops take one, including every "do this to that model" -
   * lands after the frame's own id and overwrites it. The engine then replies
   * quoting the *argument* as the request id, `_onFrame` finds no pending entry
   * for it, and the reply is dropped on the floor. The promise never settles and
   * never rejects; the UI just waits, forever.
   */
  _send(p) {
    const id = String(++this.seq);
    this.pending.set(id, p);
    try {
      this.proc.stdin.write(JSON.stringify({ id, type: p.type, payload: p.payload || {} }) + '\n');
    } catch (e) {
      this.pending.delete(id);
      p.reject(new Error(`Could not reach the Python engine: ${e.message}`));
    }
  }

  /**
   * Invoke an op. `onProgress` receives streaming frames for long operations
   * (download, setup) and is ignored for everything else.
   */
  async call(type, payload = {}, onProgress = null) {
    await this.start();
    // `type` and `payload` are kept on the pending entry, not just written to the
    // wire: if the engine is deliberately stopped before answering, that is what
    // lets the request be re-sent to its successor instead of failed.
    return new Promise((resolve, reject) => {
      this._send({ resolve, reject, onProgress, type, payload: payload || {} });
    });
  }

  /** Subscribe to a broadcast (`hw:update`, `chats:updated`, `hf:installsChanged`). */
  on(event, cb) {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event).add(cb);
    return () => this.listeners.get(event).delete(cb);
  }

  /**
   * Stop the engine and resolve once the OS has really reaped it.
   *
   * `stop()` returns the moment it has *asked* the process to die - which is all
   * quitting needs. Uninstalling the engine's packages needs more: until the
   * interpreter is gone, Windows holds open handles on python.exe and on every
   * torch DLL it mapped, and those files will not delete. So this waits for exit.
   *
   * It also drains first. Killing the process while requests are in flight rejects
   * every one of them, and the user sees errors for a hardware poll that happened
   * to land as they clicked. They finish in milliseconds; wait for them. (Call
   * hold() before this, or new requests keep arriving and the drain never ends.)
   */
  async stopAndWait(timeoutMs = 15000) {
    await this.drain();
    const proc = this.proc;
    this.stop();
    if (!proc || proc.exitCode !== null) return;
    await new Promise((resolve) => {
      const done = () => { clearTimeout(timer); resolve(); };
      const timer = setTimeout(done, timeoutMs);
      proc.once('exit', done);
    });
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
