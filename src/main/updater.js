/**
 * Auto-update against GitHub Releases.
 *
 * Replaces the old pipx self-update path (`pipx upgrade inferml` + relaunch),
 * which is meaningless now that the app ships as an installer rather than a
 * PyPI package. electron-updater reads the same release assets CI publishes and
 * swaps the app in place.
 *
 * The IPC surface here mirrors the shape the renderer's Settings screen already
 * consumes (`{ ok, hasUpdate, currentVersion, latestVersion, ... }`), so the UI
 * did not have to change.
 *
 * Note this updates the *shell*, not the venv. The Python layer is shipped as
 * source inside the app bundle, so a new installer brings a new `python/` tree
 * with it; the venv (torch et al.) is left alone, which is the whole point -
 * users don't re-download 2GB of CUDA wheels to get a UI fix.
 */
'use strict';

const { app, ipcMain } = require('electron');

let wired = false;

/**
 * @param getWin - resolves the current window *at send time*. Not a `win` value:
 *   these handlers are registered once at startup, but the window is destroyed and
 *   recreated whenever the user closes to the tray and reopens, so a captured
 *   reference would go stale and download progress would vanish into a dead window.
 */
function initUpdater(getWin) {
  if (wired) return;
  wired = true;

  const send = (channel, payload) => {
    const win = getWin();
    if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
  };

  // electron-updater is a no-op (and throws on checkForUpdates) in an unpacked
  // dev run. Register handlers that answer honestly instead of blowing up.
  if (!app.isPackaged) {
    ipcMain.handle('updates:check', async () => ({
      ok: true,
      hasUpdate: false,
      currentVersion: app.getVersion(),
      latestVersion: app.getVersion(),
      canAutoUpdate: false,
      note: 'Updates are disabled in a development run.',
    }));
    ipcMain.handle('updates:download', async () => ({ ok: false, error: 'dev build' }));
    ipcMain.handle('updates:install', async () => ({ ok: false, error: 'dev build' }));
    return;
  }

  const { autoUpdater } = require('electron-updater');

  autoUpdater.autoDownload = false;          // the user presses Download.
  autoUpdater.autoInstallOnAppQuit = true;
  autoUpdater.logger = null;

  autoUpdater.on('download-progress', (p) => {
    send('updates:progress', {
      percent: Math.round(p.percent || 0),
      transferred: p.transferred,
      total: p.total,
      bytesPerSecond: p.bytesPerSecond,
    });
  });
  autoUpdater.on('update-downloaded', (info) => {
    send('updates:downloaded', { version: info && info.version });
  });
  autoUpdater.on('error', (err) => {
    send('updates:error', { error: String((err && err.message) || err) });
  });

  ipcMain.handle('updates:check', async () => {
    try {
      const res = await autoUpdater.checkForUpdates();
      const latest = (res && res.updateInfo && res.updateInfo.version) || app.getVersion();
      const current = app.getVersion();
      return {
        ok: true,
        hasUpdate: latest !== current,
        currentVersion: current,
        latestVersion: latest,
        canAutoUpdate: true,
        releaseUrl: `https://github.com/IMvision12/InferML/releases/tag/v${latest}`,
        downloadPageUrl: 'https://github.com/IMvision12/InferML/releases/latest',
      };
    } catch (e) {
      return {
        ok: false,
        error: String((e && e.message) || e),
        currentVersion: app.getVersion(),
      };
    }
  });

  ipcMain.handle('updates:download', async () => {
    try {
      await autoUpdater.downloadUpdate();
      return { ok: true };
    } catch (e) {
      return { ok: false, error: String((e && e.message) || e) };
    }
  });

  // Quits immediately - the installer takes over. `sidecar.stop()` still runs,
  // because Electron fires before-quit/will-quit on this path.
  ipcMain.handle('updates:install', async () => {
    setImmediate(() => autoUpdater.quitAndInstall(false, true));
    return { ok: true };
  });
}

module.exports = { initUpdater };
