/**
 * Electron main process - the InferML desktop shell.
 *
 * Boot sequence (the bootstrap window narrates each step):
 *   1. find a system Python >= 3.10          -> python-env.findSystemPython
 *   2. create/repair the managed venv        -> python-env.ensureVenv
 *   3. start the FastAPI server on loopback  -> sidecar.start
 *   4. point the window at it                -> win.loadURL(sidecar.url)
 *
 * Steps 1-2 only do real work on first launch; afterwards the venv is warm and
 * boot is just step 3-4. Once the window loads the sidecar URL, the app *is*
 * the existing web UI - `web-bridge.js` talks HTTP+SSE to the same origin, so
 * the renderer and the Python engine are untouched by this shell.
 *
 * Lifecycle: the window is a *view onto* a background service, not the service
 * itself. Closing it hides it and leaves the server, the loaded models, and the
 * OpenAI-compatible API running behind the tray icon; only "Quit" from the tray
 * (or Cmd-Q) actually stops the sidecar. Anything less would mean an agent
 * talking to http://localhost:11500/v1 dies the moment someone tidies away a
 * window.
 */
'use strict';

const { app, BrowserWindow, Menu, shell, ipcMain, dialog } = require('electron');
const fs = require('fs');
const path = require('path');

const { findSystemPython, ensureVenv, isVenvReady, venvPython, MIN_PYTHON } = require('./python-env');
const { Sidecar } = require('./sidecar');
const { initUpdater } = require('./updater');
const { writeMcpLauncher, launcherPath } = require('./mcp-setup');
const { migrateLegacyData } = require('./migrate');
const { createTray, destroyTray } = require('./tray');

const isDev = !app.isPackaged;

/** The shipped `python/` source tree (extraResources in the packaged app). */
function pythonDir() {
  return isDev
    ? path.join(__dirname, '..', '..', 'python')
    : path.join(process.resourcesPath, 'python');
}

/** Auto-started at login: warm up in the tray, don't pop a window. */
function startedHidden() {
  return process.argv.includes('--hidden')
    || app.getLoginItemSettings().wasOpenedAsHidden;
}

/**
 * Kill Electron's stock "File / Edit / View / Window / Help" menu.
 *
 * The app has its own in-window navigation; the default menu is pure Chromium
 * boilerplate (Reload, Toggle DevTools, Zoom...) that just eats vertical space.
 *
 * macOS is the exception: there the menu bar lives in the system bar, not the
 * window, and removing it outright would take Cmd-Q, Cmd-C and Cmd-V with it -
 * standard-role menu items are what actually bind those shortcuts. So macOS
 * keeps a minimal menu, and Windows/Linux get none at all.
 */
function installMenu() {
  if (process.platform === 'darwin') {
    Menu.setApplicationMenu(Menu.buildFromTemplate([
      { role: 'appMenu' },     // About / Hide / Quit
      { role: 'editMenu' },    // Undo / Cut / Copy / Paste / Select All
      { role: 'windowMenu' },  // Minimize / Zoom / Close
    ]));
  } else {
    Menu.setApplicationMenu(null);
  }
}

let win = null;
let tray = null;
let sidecar = null;
let booted = false;
let isQuitting = false;

function createWindow() {
  win = new BrowserWindow({
    width: 1280,
    height: 860,
    minWidth: 940,
    minHeight: 640,
    show: false,
    backgroundColor: '#0b0d12',
    title: 'InferML',
    autoHideMenuBar: true,   // belt-and-braces: no menu strip, and no Alt to reveal one
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  win.once('ready-to-show', () => {
    if (!startedHidden()) win.show();
  });

  // Closing hides. The sidecar - and everything loaded into it - survives, and
  // the tray is how you get back. Quit is the only thing that really stops it.
  win.on('close', (e) => {
    if (isQuitting) return;
    e.preventDefault();
    win.hide();
    // Keep the Dock icon out of the way on macOS, matching the menu-bar-app feel.
    if (process.platform === 'darwin' && app.dock) app.dock.hide();
  });

  win.on('closed', () => { win = null; });

  // Keep external links in the user's browser, not in an app window with no
  // chrome to escape from.
  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  win.loadFile(path.join(__dirname, 'bootstrap.html'));
  return win;
}

function send(channel, payload) {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
}

/** Show the window, recreating it if it was closed to the tray and destroyed. */
function showWindow() {
  if (process.platform === 'darwin' && app.dock) app.dock.show();
  if (!win || win.isDestroyed()) {
    createWindow();
    // A window recreated after boot goes straight back to the live server.
    if (booted && sidecar && sidecar.url) win.loadURL(sidecar.url);
    else boot();
    return;
  }
  if (win.isMinimized()) win.restore();
  win.show();
  win.focus();
}

/** The only path that actually stops the server. */
function quitApp() {
  isQuitting = true;
  if (sidecar) sidecar.stop();
  destroyTray();
  app.quit();
}

/** Run the boot sequence, narrating it to the bootstrap page. */
async function boot() {
  // Carry chats/settings over from a pre-desktop (pipx) install, if any.
  const migrated = migrateLegacyData(app.getPath('userData'));
  if (migrated.length) {
    send('boot:log', { line: `Migrated from your previous install: ${migrated.join(', ')}` });
  }

  const py = findSystemPython();

  if (!py.exe) {
    send('boot:python-missing', {
      min: MIN_PYTHON.join('.'),
      tooOld: py.tooOld || [],
      platform: process.platform,
    });
    return;
  }

  send('boot:status', {
    step: `Found Python ${py.version.join('.')}`,
    detail: py.exe,
  });

  try {
    if (!isVenvReady(app.getPath('userData'))) {
      send('boot:status', { step: 'Setting up InferML (first launch only)' });
      await ensureVenv(app.getPath('userData'), py, (evt) => {
        if (evt.step) send('boot:status', { step: evt.step });
        if (evt.log) send('boot:log', { line: evt.log });
      });
    }
  } catch (e) {
    send('boot:error', {
      title: 'Could not set up the Python environment',
      message: String((e && e.message) || e),
    });
    return;
  }

  send('boot:status', { step: 'Starting the model server' });
  sidecar = new Sidecar({
    pythonPath: venvPython(app.getPath('userData')),
    pythonDir: pythonDir(),
    dataDir: app.getPath('userData'),
  });

  try {
    const url = await sidecar.start((line) => send('boot:log', { line }));
    booted = true;
    // Point the MCP launcher at this install before the UI can offer it.
    writeMcpLauncher(app.getPath('userData'), pythonDir());
    if (tray && tray.refresh) tray.refresh();   // "starting..." -> the live URL

    // Stamp every request this window makes to the server with the shell's
    // secret. The server hands the UI to holders of that secret and to nobody
    // else, so pointing a browser at localhost gets a 403 instead of the app.
    // Applies to page loads, fetch/XHR and EventSource alike.
    win.webContents.session.webRequest.onBeforeSendHeaders(
      { urls: [`${url}/*`] },
      (details, cb) => cb({
        requestHeaders: { ...details.requestHeaders, 'X-InferML-Shell': sidecar.uiToken },
      }),
    );

    if (win && !win.isDestroyed()) await win.loadURL(url);
    initUpdater(win);
  } catch (e) {
    send('boot:error', {
      title: 'The model server failed to start',
      message: String((e && e.message) || e),
    });
  }
}

// One instance only: a second copy would start a second server, load a second
// copy of every model, and fight over the same venv.
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  // Re-running the app (double-clicking the icon while it sits in the tray)
  // surfaces the existing instance instead of starting a rival server.
  app.on('second-instance', () => showWindow());

  app.whenReady().then(() => {
    installMenu();

    tray = createTray({
      onOpen: showWindow,
      onQuit: quitApp,
      getUrl: () => (sidecar && sidecar.url) || null,
    });

    createWindow();
    boot();

    // macOS: clicking the Dock icon reopens the window we hid on close.
    app.on('activate', () => showWindow());
  });
}

// Deliberately empty. The default behaviour quits the app when the last window
// closes - which is exactly what must NOT happen here: the window is a view onto
// a running service, and the tray is how you get it back.
app.on('window-all-closed', () => { /* keep running in the tray */ });

// Cmd-Q / taskbar "close window" / OS shutdown all funnel through here, so this
// is the backstop that guarantees the sidecar dies with the app. A survivor
// would hold GPU memory and a locked interpreter with no UI left to stop it.
app.on('before-quit', () => { isQuitting = true; if (sidecar) sidecar.stop(); });
app.on('will-quit', () => { if (sidecar) sidecar.stop(); });
process.on('exit', () => { if (sidecar) sidecar.stop(); });

// --- bootstrap page IPC ------------------------------------------------------

ipcMain.handle('boot:retry', async () => { await boot(); });

ipcMain.handle('boot:open-python-download', async () => {
  await shell.openExternal('https://www.python.org/downloads/');
});

ipcMain.handle('app:paths', async () => ({
  userData: app.getPath('userData'),
  venvPython: venvPython(app.getPath('userData')),
  pythonDir: pythonDir(),
  version: app.getVersion(),
}));

// The exact, copy-pasteable command that registers this install as an MCP
// server. Both paths are stable across app updates.
ipcMain.handle('app:mcp-command', async () => {
  const userData = app.getPath('userData');
  const py = venvPython(userData);
  const script = launcherPath(userData);
  return {
    command: `claude mcp add inferml -- "${py}" "${script}"`,
    python: py,
    script,
  };
});

ipcMain.handle('app:show-logs', async () => {
  const dir = app.getPath('userData');
  fs.mkdirSync(dir, { recursive: true });
  await shell.openPath(dir);
});

ipcMain.handle('app:copy-diagnostics', async () => {
  const lines = (sidecar && sidecar.stderr) || [];
  await dialog.showMessageBox(win, {
    type: 'info',
    title: 'Server log',
    message: lines.slice(-20).join('\n') || 'No server output captured.',
  });
});
