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
