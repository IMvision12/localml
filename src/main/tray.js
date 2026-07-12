/**
 * System tray / menu bar presence.
 *
 * Closing the window hides it; the engine and every model it has loaded stay
 * resident behind the tray icon, so reopening is instant instead of paying the
 * multi-second load again. Quit is the only thing that actually stops it, and it
 * is an explicit act because it drops multi-GB models out of memory.
 *
 * The menu is rebuilt on demand rather than kept in sync with a live handle, so
 * the status line can't drift from reality.
 */
'use strict';

const { app, Menu, Tray, nativeImage } = require('electron');
const path = require('path');

let tray = null;

function iconPath() {
  return app.isPackaged
    ? path.join(process.resourcesPath, 'logo.png')
    : path.join(__dirname, '..', '..', 'assets', 'logo.png');
}

function trayImage() {
  // The source art is 1024x1024; a tray needs ~16-22px. Electron will happily
  // render the full-size image and produce a comically large icon, so resize.
  const img = nativeImage.createFromPath(iconPath());
  if (img.isEmpty()) return img;
  return img.resize({ width: 18, height: 18, quality: 'best' });
}

/**
 * @param {object} o
 * @param {() => void}    o.onOpen      show/focus the main window
 * @param {() => void}    o.onQuit      really quit (stops the engine)
 * @param {() => boolean} o.isRunning   whether the Python engine is up
 */
function createTray({ onOpen, onQuit, isRunning }) {
  if (tray) return tray;

  tray = new Tray(trayImage());
  tray.setToolTip('InferML');

  const rebuild = () => {
    const running = isRunning();
    const openAtLogin = app.getLoginItemSettings().openAtLogin;

    tray.setContextMenu(Menu.buildFromTemplate([
      {
        label: running ? 'InferML - engine running' : 'InferML - starting...',
        enabled: false,
      },
      { type: 'separator' },
      { label: 'Open InferML', click: onOpen },
      { type: 'separator' },
      {
        label: 'Launch at login',
        type: 'checkbox',
        checked: openAtLogin,
        click: (item) => {
          // `--hidden` is read back in main.js: an auto-started InferML warms up
          // in the tray without stealing focus with a window nobody asked for.
          app.setLoginItemSettings({
            openAtLogin: item.checked,
            openAsHidden: true,
            args: ['--hidden'],
          });
          rebuild();
        },
      },
      { type: 'separator' },
      { label: 'Quit InferML', click: onQuit },
    ]));
  };

  rebuild();
  tray.on('double-click', onOpen);
  // Left-click opens on Windows/Linux, where a click isn't expected to just show
  // the menu the way it is on macOS.
  if (process.platform !== 'darwin') tray.on('click', onOpen);

  tray.refresh = rebuild;
  return tray;
}

function destroyTray() {
  if (tray) {
    tray.destroy();
    tray = null;
  }
}

module.exports = { createTray, destroyTray };
