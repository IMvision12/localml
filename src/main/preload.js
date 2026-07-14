/**
 * Preload bridge - the renderer's only door out.
 *
 * `window.inferml` is defined here, and this is the whole of it. The renderer
 * runs with contextIsolation on and no node integration, loaded from file://,
 * so it has no fetch to a server, no require, no child_process: if a capability
 * isn't on this object, the UI does not have it.
 *
 * This file used to be almost empty. When InferML was a web app the server
 * served `web-bridge.js`, which built `window.inferml` out of fetch() calls to
 * localhost, and the preload only added the few things HTTP couldn't do. The
 * server is gone, that shim is gone, and this is what the UI was always meant to
 * talk to - so the shape below is deliberately identical to what web-bridge.js
 * exposed. None of the React components had to change.
 */
'use strict';

const { contextBridge, ipcRenderer } = require('electron');

const on = (channel) => (cb) => {
  const handler = (_evt, payload) => cb(payload);
  ipcRenderer.on(channel, handler);
  return () => ipcRenderer.removeListener(channel, handler);
};

/** Invoke an engine op. */
const call = (type, payload) => ipcRenderer.invoke('inferml:call', type, payload || {});

// --- broadcasts --------------------------------------------------------------
// The engine pushes these unprompted (hardware ticks, store changes). One IPC
// listener fans out to however many components subscribed.

const subs = {
  'hw:update': new Set(),
  'chats:updated': new Set(),
  'hf:installsChanged': new Set(),
};
ipcRenderer.on('inferml:event', (_e, { name, data }) => {
  const set = subs[name];
  if (!set) return;
  for (const cb of set) { try { cb(data); } catch { /* one bad subscriber shouldn't break the rest */ } }
});
const onEvent = (name) => (cb) => {
  subs[name].add(cb);
  return () => subs[name].delete(cb);
};

// Progress from the long operations. Same fan-out, keyed by which one.
//
// `runtime` is deliberately not `setup`, though both narrate uv installing things.
// The app treats a setup frame as proof that an install is running - it will spin
// up a progress bar and tell the user their runtime is being downloaded - and
// removing the runtime is the one operation where that is exactly the wrong
// conclusion to draw.
const progress = { download: new Set(), setup: new Set(), runtime: new Set() };
ipcRenderer.on('inferml:progress', (_e, { kind, data }) => {
  for (const cb of progress[kind] || []) { try { cb(data); } catch { /* ditto */ } }
});
const onProgress = (kind) => (cb) => {
  progress[kind].add(cb);
  return () => progress[kind].delete(cb);
};

contextBridge.exposeInMainWorld('inferml', {
  tasks: {
    run: (payload) => call('tasks.run', payload),
    stop: () => call('tasks.stop'),
    status: () => call('tasks.status'),

    // Synchronous: the first render needs to know whether the runtime is
    // installed before any promise can resolve. Reads a cache in the main
    // process, not the engine.
    statusSync: () => ipcRenderer.sendSync('inferml:statusSync'),

    setup: (opts) => ipcRenderer.invoke('inferml:setup', opts || {}),
    download: (modelId) => ipcRenderer.invoke('inferml:download', modelId),
    cancelDownload: (modelId) => call('tasks.cancelDownload', { modelId }),

    onDownloadProgress: onProgress('download'),
    onSetupProgress: onProgress('setup'),
  },

  hf: {
    // The UI wants the array; the engine returns {items:[...]} or {error}.
    search: (q, task) => call('hf.search', { q, task })
      .then((r) => (r && Array.isArray(r.items) ? r.items : (r || { error: 'search failed' }))),
    installed: () => call('hf.installed'),
    markInstalled: (id, meta) => call('hf.markInstalled', { id, meta }),
    uninstall: (id) => call('hf.uninstall', { id }),
    modelInfo: (id) => call('hf.modelInfo', { id }),
    getToken: () => call('hf.getToken').then((r) => (r && r.token) || null),
    setToken: (token) => call('hf.setToken', { token }),
    clearToken: () => call('hf.clearToken'),
    verifyToken: (token) => call('hf.verifyToken', { token }),
    onInstallsChanged: onEvent('hf:installsChanged'),
  },

  chats: {
    list: () => call('chats.list'),
    get: (id) => call('chats.get', { id }),
    save: (chat) => call('chats.save', { chat }),
    patch: (id, patch) => call('chats.patch', { id, patch }),
    delete: (id) => call('chats.delete', { id }),
    onUpdate: onEvent('chats:updated'),
  },

  settings: {
    get: () => call('settings.get'),
    save: (patch) => call('settings.save', { patch }),
  },

  hw: {
    get: () => call('hw.get'),
    subscribe: onEvent('hw:update'),
  },

  dialog: {
    openImage: () => ipcRenderer.invoke('inferml:pickImage'),
    openAudio: () => ipcRenderer.invoke('inferml:pickAudio'),
  },

  // The optional local HTTP API (OpenAI-compatible /v1). Off unless the user
  // turns it on; the engine persists the choice, so this survives a restart.
  api: {
    status: () => call('api.status'),
    start: (port) => call('api.start', { port }),
    stop: () => call('api.stop'),
    // Ready-to-paste `claude mcp add inferml -- ...` for this exact install.
    mcpCommand: () => ipcRenderer.invoke('inferml:mcpCommand'),
  },

  app: {
    version: () => ipcRenderer.invoke('inferml:app.version'),
    openExternal: (url) => ipcRenderer.invoke('inferml:openExternal', url),
  },

  // "Logs" is a button in Settings that reveals the app's data folder. There is
  // no log *list* - the engine's output goes to Electron's stderr, not to a ring
  // buffer the UI reads back.
  logs: {
    view: () => ipcRenderer.invoke('inferml:showDataDir'),
    path: () => ipcRenderer.invoke('inferml:dataDir'),
  },

  storage: {
    size: (key) => call('storage.size', { key }),
    clearHfCache: () => call('storage.clear', { key: 'hfCache' }),

    // Not an engine op, unlike its neighbours: the engine runs *inside* the venv
    // whose packages this strips, so only the main process can do it (see ipc.js).
    clearPyRuntime: () => ipcRenderer.invoke('inferml:clearPyRuntime'),
    onClearProgress: onProgress('runtime'),
  },

  updates: {
    check: (opts) => ipcRenderer.invoke('updates:check', opts || {}),
    download: () => ipcRenderer.invoke('updates:download'),
    install: () => ipcRenderer.invoke('updates:install'),
    onProgress: on('updates:progress'),
    onDownloaded: on('updates:downloaded'),
    onError: on('updates:error'),
  },
});

// The bootstrap page (shown while the venv is built) is a separate document with
// its own script, and it predates window.inferml. Left as its own namespace.
contextBridge.exposeInMainWorld('infermlDesktop', {
  isDesktop: true,
  boot: {
    onStatus: on('boot:status'),
    onLog: on('boot:log'),
    onPythonMissing: on('boot:python-missing'),
    onError: on('boot:error'),
    retry: () => ipcRenderer.invoke('boot:retry'),
    openPythonDownload: () => ipcRenderer.invoke('boot:open-python-download'),
  },
});
