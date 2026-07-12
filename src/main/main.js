/**
 * Electron main process - the InferML desktop app.
 *
 * Boot sequence (the bootstrap window narrates each step):
 *   1. find a system Python >= 3.10          -> python-env.findSystemPython
 *   2. create/repair the managed venv        -> python-env.ensureVenv
 *   3. start the Python engine               -> PythonRunner.start   (stdin/stdout)
 *   4. show the UI                           -> win.loadFile(renderer)
 *
 * Steps 1-2 only do real work on first launch. There is no server and no port:
 * the engine is a child process we talk to in JSON over its stdio (see
 * python/runner.py), and the UI is a local file. Nothing InferML runs is
 * reachable from the network, or from a browser.
 *
 * Lifecycle: the window is a *view onto* a running engine, not the engine
 * itself. Closing it hides it and leaves the engine - and every model it has
 * loaded - resident behind the tray icon, so reopening is instant. Only "Quit"
 * from the tray (or Cmd-Q) actually stops it.
 */
'use strict';

const { app, BrowserWindow, Menu, shell, ipcMain } = require('electron');
const path = require('path');

const { findSystemPython, ensureVenv, isVenvReady, venvPython, MIN_PYTHON } = require('./python-env');
const { PythonRunner } = require('./python-runner');
const { registerIpc } = require('./ipc');
const { initUpdater } = require('./updater');
const { writeMcpLauncher, launcherPath } = require('./mcp-setup');
const { migrateLegacyData } = require('./migrate');
const { createTray, destroyTray } = require('./tray');

const isDev = !app.isPackaged;

/** The shipped `python/` tree (extraResources in the packaged app). */
function pythonDir() {
  return isDev
    ? path.join(__dirname, '..', '..', 'python')
    : path.join(process.resourcesPath, 'python');
}

/**
 * The UI, as a file on disk. Always the compiled build - dev included.
 *
 * The source index.html compiles JSX in the browser with Babel Standalone, which
 * needs to *fetch* each .jsx over XHR. That worked when a server was serving the
 * page. It cannot work now: the document's origin is file://, which is opaque,
 * so those fetches are cross-origin and Chromium blocks them - React never
 * mounts and the window comes up black, with no error, because the failure is a
 * blocked request rather than a thrown exception.
 *
 * The alternative would be `webSecurity: false`, which is not a trade worth
 * making to save a build step - it would also switch off the protections that
 * matter when the UI renders markdown from HuggingFace model cards.
 *
 * So both paths load src/renderer/dist. `npm start` builds it first, and
 * `npm run build:renderer:watch` rebuilds on save if you want a fast loop. The
 * bonus is that dev and production now run byte-identical code, and the CSP
 * loses 'unsafe-eval' everywhere instead of only in packaged builds.
 */
function rendererHtml() {
  return path.join(__dirname, '..', 'renderer', 'dist', 'index.html');
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
let runner = null;
let ipc = null;
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

  // Closing hides. The engine - and everything loaded into it - survives, and
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
    // A window recreated after boot goes straight back to the UI.
    if (booted) win.loadFile(rendererHtml());
    else boot();
    return;
  }
  if (win.isMinimized()) win.restore();
  win.show();
  win.focus();
}

/** The only path that actually stops the engine. */
function quitApp() {
  isQuitting = true;
  if (runner) runner.stop();
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

  send('boot:status', { step: `Found Python ${py.version.join('.')}`, detail: py.exe });

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

  send('boot:status', { step: 'Starting the inference engine' });

  try {
    await runner.start();
    await ipc.primeStatus();
    // Point the MCP launcher at this install. Cheap, and doing it every boot is
    // what keeps `claude mcp add inferml -- ...` working across app updates.
    writeMcpLauncher(app.getPath('userData'), pythonDir());
    booted = true;
    if (tray && tray.refresh) tray.refresh();
    if (win && !win.isDestroyed()) await win.loadFile(rendererHtml());
  } catch (e) {
    send('boot:error', {
      title: 'The inference engine failed to start',
      message: String((e && e.message) || e),
    });
  }
}

// One instance only: a second copy would start a second engine and load a second
// copy of every model into the same GPU.
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  // Re-running the app (double-clicking the icon while it sits in the tray)
  // surfaces the existing instance instead of starting a rival engine.
  app.on('second-instance', () => showWindow());

  app.whenReady().then(() => {
    installMenu();

    runner = new PythonRunner({
      pythonPath: venvPython(app.getPath('userData')),
      pythonDir: pythonDir(),
      dataDir: app.getPath('userData'),
      version: app.getVersion(),
    });

    // Registered before any window exists, let alone loads. The renderer asks
    // for state the moment it mounts, and an ipcMain handler that isn't there
    // yet throws "No handler registered for ..." instead of answering.
    ipc = registerIpc({ runner, getWin: () => win });
    initUpdater(() => win);

    tray = createTray({
      onOpen: showWindow,
      onQuit: quitApp,
      isRunning: () => !!runner && runner.running,
    });

    createWindow();
    boot();

    // macOS: clicking the Dock icon reopens the window we hid on close.
    app.on('activate', () => showWindow());
  });
}

// Deliberately empty. The default behaviour quits the app when the last window
// closes - which is exactly what must NOT happen here: the window is a view onto
// a running engine, and the tray is how you get it back.
app.on('window-all-closed', () => { /* keep running in the tray */ });

// Cmd-Q / taskbar "close window" / OS shutdown all funnel through here, so this
// is the backstop that guarantees the engine dies with the app. A survivor would
// hold GPU memory and a locked interpreter with no UI left to stop it.
app.on('before-quit', () => { isQuitting = true; if (runner) runner.stop(); });
app.on('will-quit', () => { if (runner) runner.stop(); });
process.on('exit', () => { if (runner) runner.stop(); });

// --- bootstrap page IPC ------------------------------------------------------
//
// The only two channels bootstrap.html uses. Everything the *app* needs lives in
// ipc.js; these exist because the bootstrap page runs before the app does.

ipcMain.handle('boot:retry', async () => { await boot(); });

ipcMain.handle('boot:open-python-download', async () => {
  await shell.openExternal('https://www.python.org/downloads/');
});

// The exact command that registers this install as an MCP server. Both paths are
// stable across app updates, which is the whole point of the generated launcher.
ipcMain.handle('inferml:mcpCommand', async () => {
  const userData = app.getPath('userData');
  return {
    command: `claude mcp add inferml -- "${venvPython(userData)}" "${launcherPath(userData)}"`,
  };
});
