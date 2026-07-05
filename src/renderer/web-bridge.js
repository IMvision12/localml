

(function () {
  'use strict';

  const BASE = ''; 

  async function jget(path) {
    const r = await fetch(BASE + path, { headers: { Accept: 'application/json' } });
    return r.json();
  }
  async function jsend(path, method, body) {
    const r = await fetch(BASE + path, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    return r.json();
  }
  const jpost = (path, body) => jsend(path, 'POST', body);
  const jpatch = (path, body) => jsend(path, 'PATCH', body);
  const jdelete = (path) => jsend(path, 'DELETE');



  const _subs = { 'hw:update': new Set(), 'chats:updated': new Set(), 'hf:installsChanged': new Set() };
  function _wireEvents() {
    let es;
    try { es = new EventSource(BASE + '/api/events'); } catch { return; }
    for (const name of Object.keys(_subs)) {
      es.addEventListener(name, (e) => {
        let data; try { data = JSON.parse(e.data); } catch { data = undefined; }
        for (const cb of _subs[name]) { try { cb(data); } catch {} }
      });
    }
    es.onerror = () => {  };
  }
  _wireEvents();
  function _on(name, cb) {
    _subs[name].add(cb);
    return () => _subs[name].delete(cb);
  }

  const _dlProgress = new Set();
  const _setupProgress = new Set();




  let _statusCache = { ready: false, runtimeInstalled: false, sidecarRunning: true };
  function _refreshStatusSync() {
    try {
      const xhr = new XMLHttpRequest();
      xhr.open('GET', BASE + '/api/status', false); 
      xhr.send();
      if (xhr.status === 200) _statusCache = JSON.parse(xhr.responseText);
    } catch {}
  }
  _refreshStatusSync();


  async function _consumeSSE(resp, onEvent) {
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const dataLine = frame.split('\n').find((l) => l.startsWith('data:'));
        if (!dataLine) continue;
        try { onEvent(JSON.parse(dataLine.slice(5).trim())); } catch {}
      }
    }
  }

  function _pickFile(accept, kind) {
    return new Promise((resolve) => {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = accept;
      input.style.display = 'none';
      input.onchange = () => {
        const file = input.files && input.files[0];
        if (!file) { resolve(null); cleanup(); return; }
        const reader = new FileReader();
        reader.onload = () => { resolve({ kind, dataUrl: reader.result, name: file.name }); cleanup(); };
        reader.onerror = () => { resolve(null); cleanup(); };
        reader.readAsDataURL(file);
      };

      const cleanup = () => { try { input.remove(); } catch {} window.removeEventListener('focus', onFocus, true); };
      const onFocus = () => setTimeout(() => { if (!input.files || !input.files.length) { resolve(null); cleanup(); } }, 400);
      window.addEventListener('focus', onFocus, true);
      document.body.appendChild(input);
      input.click();
    });
  }

  window.inferml = {

    tasks: {
      run: (payload) => jpost('/api/run', payload || {}),
      stop: () => jpost('/api/stop', {}),
      status: () => jget('/api/status').then((s) => { if (s && typeof s === 'object') _statusCache = s; return s; }),

      statusSync: () => _statusCache,
      setup: async (opts) => {
        try {
          const resp = await fetch(BASE + '/api/setup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
            body: JSON.stringify(opts || {}),
          });
          if (!resp.ok || !resp.body) return { ok: false, error: `setup failed (${resp.status})` };
          let terminal = { ok: false, error: 'setup ended without a result' };
          await _consumeSSE(resp, (evt) => {
            if (evt.kind === 'result') {
              terminal = evt.ok ? { ok: true } : { ok: false, error: evt.error };
            } else {
              for (const cb of _setupProgress) { try { cb(evt); } catch {} }
            }
          });
          _refreshStatusSync(); 
          return terminal;
        } catch (e) {
          return { ok: false, error: String((e && e.message) || e) };
        }
      },
      cancelDownload: (modelId) => jpost('/api/download/cancel', { modelId }),
      download: async (modelId) => {
        try {
          const resp = await fetch(BASE + '/api/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
            body: JSON.stringify({ modelId }),
          });
          if (!resp.ok || !resp.body) return { ok: false, error: `download failed (${resp.status})` };
          let terminal = { ok: false, error: 'download ended without result' };
          await _consumeSSE(resp, (evt) => {
            if (evt.type === 'progress') {
              const msg = { modelId, pct: evt.pct, done: evt.done, total: evt.total, final: evt.final };
              for (const cb of _dlProgress) { try { cb(msg); } catch {} }
            } else if (evt.type === 'result') {
              terminal = evt.ok ? { ok: true, info: evt.info } : { ok: false, error: evt.error, cancelled: evt.cancelled };
            }
          });
          return terminal;
        } catch (e) {
          return { ok: false, error: String((e && e.message) || e) };
        }
      },
      onDownloadProgress: (cb) => { _dlProgress.add(cb); return () => _dlProgress.delete(cb); },
      onSetupProgress: (cb) => { _setupProgress.add(cb); return () => _setupProgress.delete(cb); },
    },

    hf: {
      search: (q, task) => {
        const p = new URLSearchParams();
        if (q) p.set('q', q);
        if (task) p.set('task', task);


        return jget('/api/hf/search?' + p.toString()).then((r) =>
          (r && Array.isArray(r.items)) ? r.items : (r || { error: 'search failed' })
        );
      },
      installed: () => jget('/api/hf/installed'),
      markInstalled: (id, meta) => jpost('/api/hf/markInstalled', { id, meta }),
      uninstall: (id) => jpost('/api/hf/uninstall', { id }),
      modelInfo: (id) => jget('/api/hf/modelInfo?id=' + encodeURIComponent(id)),
      getToken: () => jget('/api/hf/token').then((r) => (r && r.token) || null),
      hasToken: () => jget('/api/hf/hasToken').then((r) => !!(r && r.hasToken)),
      setToken: (token) => jpost('/api/hf/token', { token }),
      clearToken: () => jdelete('/api/hf/token'),
      verifyToken: (token) => jpost('/api/hf/verifyToken', { token }),
      onInstallsChanged: (cb) => _on('hf:installsChanged', cb),
    },

    chats: {
      list: () => jget('/api/chats'),
      get: (id) => jget('/api/chats/' + encodeURIComponent(id)),
      save: (chat) => jpost('/api/chats', chat),
      patch: (id, patch) => jpatch('/api/chats/' + encodeURIComponent(id), patch),
      delete: (id) => jdelete('/api/chats/' + encodeURIComponent(id)),
      onUpdate: (cb) => _on('chats:updated', cb),
    },

    settings: {
      get: () => jget('/api/settings'),
      save: (patch) => jpost('/api/settings', patch),
    },

    hw: {
      get: () => jget('/api/hw'),
      subscribe: (cb) => _on('hw:update', cb),
    },

    dialog: {
      openImage: () => _pickFile('image/*', 'image'),
      openAudio: () => _pickFile('audio/*', 'audio'),
    },

    app: {
      version: () => jget('/api/app').then((r) => (r && r.version) || '0.0.0'),
      paths: () => jget('/api/app'),
      openExternal: (url) => { try { window.open(url, '_blank', 'noopener'); } catch {} return Promise.resolve(); },
    },

    logs: {
      list: () => jget('/api/logs'),
      view: () => { try { window.open('/api/logs', '_blank'); } catch {} return Promise.resolve({ ok: true }); },

      path: () => jget('/api/app').then((r) => (r && r.dataDir) || ''),
      clear: () => Promise.resolve({ ok: true }),
      onUpdate: () => () => {},
    },

    storage: {
      size: (key) => jget('/api/storage/size?key=' + encodeURIComponent(key)),
      clearHfCache: () => jpost('/api/storage/clear', { key: 'hfCache' }),

      clearPyRuntime: () => Promise.resolve({ ok: false, error: 'The Python runtime is installed via pip - manage it with pipx, not from here.' }),
    },

    updates: {
      check: () => Promise.resolve({ ok: true, hasUpdate: false }),
      download: () => Promise.resolve({ ok: false }),
      install: () => Promise.resolve({ ok: false }),
      onProgress: () => () => {},
      onDownloaded: () => () => {},
      onError: () => () => {},
    },

    window: {
      minimize: () => Promise.resolve(),
      maximize: () => Promise.resolve(),
      close: () => Promise.resolve(),
      isMaximized: () => Promise.resolve(false),
      onState: () => () => {},
    },
  };
})();
