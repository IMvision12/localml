/**
 * The renderer's whole view of the outside world.
 *
 * The UI calls `window.inferml.*` (see preload.js); every one of those lands
 * here, and this is the only place that can reach the Python engine, the disk,
 * or the network. The renderer itself is loaded from file:// with no node
 * integration, so it cannot fetch, spawn, or read anything on its own.
 *
 * Three shapes of traffic:
 *   request/response   `inferml:call`      - the common case
 *   streaming          `inferml:download`  - progress frames while it runs
 *                      `inferml:setup`
 *   broadcast          `inferml:event`     - the engine talking unprompted
 */
'use strict';

const { app, ipcMain, dialog, shell } = require('electron');
const fs = require('fs');
const path = require('path');

const {
  venvDir, uvCacheDir, ensureVenv, resetToBaseEnv, isVenvReady, findSystemPython, MIN_PYTHON,
} = require('./python-env');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/**
 * Delete a directory Windows may not have finished letting go of.
 *
 * Killing a process is not the same as the OS closing its handles. For a moment
 * after the engine dies, python.exe and the torch DLLs it mapped are still
 * locked, and rmSync fails with EBUSY/EPERM. rmSync's own maxRetries covers the
 * usual case; the outer loop covers a slow reap on a machine under load.
 */
async function rmDirWithRetry(dir, attempts = 12) {
  for (let i = 0; i < attempts; i++) {
    try {
      fs.rmSync(dir, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
      if (!fs.existsSync(dir)) return;
    } catch (e) {
      if (i === attempts - 1) {
        throw new Error(
          `Could not delete the runtime: ${e.message}. Something still has a file open in it - close any terminal running from this environment and try again.`,
        );
      }
    }
    await sleep(400);
  }
  throw new Error('The runtime folder is still in use and could not be deleted.');
}

// The operations the UI is allowed to invoke, mirroring runner.py's OPS table.
// The renderer is trusted code, but it also renders markdown from HuggingFace
// model cards - so if XSS ever got past DOMPurify, this is the blast radius.
// Keeping it an explicit list means a hole in the UI can't reach an op the UI
// was never meant to have.
const ALLOWED = new Set([
  'tasks.run', 'tasks.stop', 'tasks.status', 'tasks.cancelDownload',
  'hf.search', 'hf.installed', 'hf.markInstalled', 'hf.uninstall', 'hf.modelInfo',
  'hf.getToken', 'hf.setToken', 'hf.clearToken', 'hf.verifyToken',
  'chats.list', 'chats.get', 'chats.save', 'chats.patch', 'chats.delete',
  'settings.get', 'settings.save',
  'hw.get',
  'storage.size', 'storage.clear',
  'api.status', 'api.start', 'api.stop',
]);

const BROADCASTS = ['hw:update', 'chats:updated', 'hf:installsChanged'];

// The renderer needs a status synchronously during its first render (before any
// promise can resolve), so main keeps the last one it saw. Seeded pessimistically:
// "not ready" renders the setup prompt, which is the safe thing to show if we
// genuinely don't know yet.
let lastStatus = { ready: false, runtimeInstalled: false, sidecarRunning: false };

function registerIpc({ runner, getWin }) {
  const send = (channel, payload) => {
    const win = getWin();
    if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
  };

  for (const name of BROADCASTS) {
    runner.on(name, (data) => send('inferml:event', { name, data }));
  }

  ipcMain.handle('inferml:call', async (_e, type, payload) => {
    if (!ALLOWED.has(type)) throw new Error(`operation not permitted: ${type}`);
    const result = await runner.call(type, payload || {});
    if (type === 'tasks.status' && result) lastStatus = result;
    return result;
  });

  // Synchronous by design. It only reads a cached object in this process - no
  // engine round-trip - so it costs microseconds, and it replaces the blocking
  // XHR the old web bridge used for exactly this.
  ipcMain.on('inferml:statusSync', (e) => { e.returnValue = lastStatus; });

  // --- streaming operations --------------------------------------------------
  //
  // These resolve like any other call, but emit progress along the way. The
  // renderer subscribes once (onDownloadProgress / onSetupProgress) rather than
  // correlating per call, which is what the existing UI already expects.

  ipcMain.handle('inferml:download', async (_e, modelId) => {
    try {
      return await runner.call('tasks.download', { modelId },
        (p) => send('inferml:progress', { kind: 'download', data: p }));
    } catch (e) {
      return { ok: false, error: String((e && e.message) || e) };
    }
  });

  ipcMain.handle('inferml:setup', async (_e, opts) => {
    try {
      const res = await runner.call('tasks.setup', opts || {},
        (p) => send('inferml:progress', { kind: 'setup', data: p }));
      // The inference stack just changed underneath us; make sure the next
      // statusSync tells the truth instead of the pre-install answer.
      try { lastStatus = await runner.call('tasks.status', {}); } catch { /* non-fatal */ }
      return res;
    } catch (e) {
      return { ok: false, error: String((e && e.message) || e) };
    }
  });

  // --- things only the shell can do -----------------------------------------

  ipcMain.handle('inferml:app.version', async () => app.getVersion());

  ipcMain.handle('inferml:dataDir', async () => app.getPath('userData'));

  ipcMain.handle('inferml:openExternal', async (_e, url) => {
    // Only ever a real web link. A file:// or, on Windows, a .lnk/.exe path
    // handed to the shell would be an arbitrary-execution hole, and model cards
    // are full of attacker-supplied links.
    if (typeof url !== 'string' || !/^https?:\/\//i.test(url)) return { ok: false };
    await shell.openExternal(url);
    return { ok: true };
  });

  ipcMain.handle('inferml:showDataDir', async () => {
    const dir = app.getPath('userData');
    fs.mkdirSync(dir, { recursive: true });
    await shell.openPath(dir);
    return { ok: true };
  });

  // Native file pickers. The old web bridge used a hidden <input type="file">
  // and FileReader, which only worked because the UI was a real web page. Under
  // file:// the renderer has no such privilege, and shouldn't.
  const pickFile = async (kind, filters) => {
    const win = getWin();
    const res = await dialog.showOpenDialog(win, { properties: ['openFile'], filters });
    if (res.canceled || !res.filePaths.length) return null;
    const file = res.filePaths[0];
    const mime = {
      '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
      '.webp': 'image/webp', '.gif': 'image/gif', '.bmp': 'image/bmp',
      '.wav': 'audio/wav', '.mp3': 'audio/mpeg', '.flac': 'audio/flac',
      '.ogg': 'audio/ogg', '.m4a': 'audio/mp4',
    }[path.extname(file).toLowerCase()] || 'application/octet-stream';
    const b64 = fs.readFileSync(file).toString('base64');
    return { kind, name: path.basename(file), dataUrl: `data:${mime};base64,${b64}` };
  };

  ipcMain.handle('inferml:pickImage', () => pickFile('image', [
    { name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp'] },
  ]));

  ipcMain.handle('inferml:pickAudio', () => pickFile('audio', [
    { name: 'Audio', extensions: ['wav', 'mp3', 'flac', 'ogg', 'm4a'] },
  ]));

  /**
   * Delete the Python runtime: torch, transformers, diffusers, everything that
   * came in with them, and the wheels cached to install them.
   *
   * This is a *shell* job, not an engine one. The engine cannot do it - it is a
   * process running inside the very environment being emptied, and on Windows the
   * torch DLLs it has mapped are locked for as long as it lives. It can no more
   * uninstall them than saw off the branch it is sitting on. That is why this was
   * once a hardcoded "go and delete the folder yourself" error in the preload.
   *
   * What it deliberately does NOT delete is the venv. The engine lives there too,
   * and so do the handful of MB it needs to answer any request at all; removing it
   * takes the whole app down and then has to put half of it straight back - a
   * minute of creating an environment, bootstrapping pip and reinstalling the
   * engine's own dependencies to arrive exactly where we started. `resetToBaseEnv`
   * strips only the inference stack, in place, in seconds.
   */
  let clearing = false;

  ipcMain.handle('inferml:clearPyRuntime', async () => {
    // Never twice at once. A second pass would tear packages out from under the
    // first one mid-uninstall and leave the environment in pieces. The UI guards
    // this too, but the guard belongs where the damage happens.
    if (clearing) return { ok: false, error: 'The runtime is already being removed.' };
    clearing = true;

    const userData = app.getPath('userData');
    const dir = venvDir(userData);

    // A silent long operation reads as a hang, so this narrates itself - but on its
    // own channel, NOT the setup one. A setup frame is how the app knows an install
    // is running; borrowing that channel made the app announce it was downloading a
    // runtime at the exact moment it was deleting one.
    const step = (text) => send('inferml:progress', { kind: 'runtime', data: { kind: 'step', text } });
    const log = (text) => send('inferml:progress', { kind: 'runtime', data: { kind: 'log', text } });

    // Nothing may lazily respawn the engine while its own packages are moving.
    // Callers that arrive meanwhile are parked, not failed - they wait out the few
    // seconds and then run normally.
    runner.hold();
    try {
      // The engine has to be down for this: its interpreter is the one being
      // modified, and on Windows a torch DLL it has mapped cannot be deleted.
      step('Stopping the inference engine');
      await runner.stopAndWait();

      step('Removing the inference runtime');
      await resetToBaseEnv(userData, (e) => { if (e.log) log(e.log); });

      // And the wheel cache, or nothing is really freed: what remains in the venv
      // is a few hundred MB, but the cached wheels behind it are gigabytes (a CUDA
      // torch alone is 2.4GB). Leaving it also lets the next install restore torch
      // from disk in seconds, which makes the delete look like it never happened.
      step('Deleting cached downloads');
      await rmDirWithRetry(uvCacheDir(userData));

      // Open the gate *before* restarting: start() parks behind it like any other
      // caller, so restarting through a shut gate would wait on itself forever.
      // Everything parked during the removal wakes here and rides the same spawn.
      step('Restarting the inference engine');
      runner.release();
      await runner.start();

      // torch is gone now, and this is what flips the UI back to "install runtime".
      try { lastStatus = await runner.call('tasks.status', {}); } catch { /* the next status call will catch it */ }

      step('Runtime removed');
      return { ok: true };
    } catch (e) {
      const error = String((e && e.message) || e);

      if (isVenvReady(userData)) {
        // The environment still runs the engine, so put the app back on its feet:
        // chats, model search and settings all need one, torch or no torch.
        runner.release();
        try { await runner.start(); } catch { /* falls through to the error below */ }
      } else {
        // Stripping it left it unusable. Now - and only now - is a full rebuild
        // worth the minute it costs, because there is nothing left to preserve.
        try {
          step('Repairing the Python environment');
          await rmDirWithRetry(dir);
          const py = findSystemPython();
          if (!py.exe) throw new Error(`no Python ${MIN_PYTHON.join('.')}+ found`);
          await ensureVenv(userData, py, (ev) => {
            if (ev.step) step(ev.step);
            if (ev.log) log(ev.log);
          });
          runner.release();
          await runner.start();
        } catch (repair) {
          // Out of options. Parked callers must not wait for an engine that is
          // never coming - suspend so they reject with the reason instead, and let
          // them through. The next launch repairs the venv on its own: boot()
          // rebuilds whenever isVenvReady is false.
          runner.suspend(
            `The Python environment is damaged and could not be repaired: ${(repair && repair.message) || repair}\n\nRestart InferML to try again.`,
          );
        }
      }

      return { ok: false, error };
    } finally {
      // Always, on every path. A gate left shut parks every future call forever.
      runner.release();
      clearing = false;
    }
  });

  return {
    // Fill the status cache before the UI is shown. Without this the first
    // render reads the pessimistic seed and flashes "Setup Python runtime" at
    // people who already have torch installed, then corrects itself a beat
    // later - which reads as a bug.
    async primeStatus() {
      try { lastStatus = await runner.call('tasks.status', {}); } catch { /* boot will surface it */ }
    },
  };
}

module.exports = { registerIpc };
