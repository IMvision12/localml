const { useState, useEffect, useMemo, useCallback } = React;
const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "theme": "dark"
}/*EDITMODE-END*/;

const CHAT_TASKS = new Set(['text-generation', 'image-text-to-text', 'conversational']);

function App() {
  const [bootReady, setBootReady] = useState(false);
  const [welcome, setWelcome] = useState(true);


  const [onboard, setOnboard] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settingsSection, setSettingsSection] = useState('general');
  const openSettings = (section) => { setSettingsSection(section || 'general'); setSettingsOpen(true); };
  const [view, setView] = useState('hub'); 
  const [activeSession, setActiveSession] = useState(null);
  const [theme, setTheme] = useState(TWEAK_DEFAULTS.theme);
  const [tweaksOpen, setTweaksOpen] = useState(false);
  const [recentsOpen, setRecentsOpen] = useState(true);
  const [hubInstalledMode, setHubInstalledMode] = useState(false);
  const [sessions, setSessions] = useState([]);
  const [hw, setHw] = useState(null);
  const [installedModels, setInstalledModels] = useState({});
  const [version, setVersion] = useState('0.1.0');





  const [pyStatus, setPyStatus] = useState(() => {
    const seed = window.inferml?.tasks?.statusSync?.();
    if (seed && typeof seed === 'object') return seed;
    return { ready: false, runtimeInstalled: false };
  });



  const [hubResetSignal, setHubResetSignal] = useState(0);
  const [pySetup, setPySetup] = useState(null); 
  const [updatingTo, setUpdatingTo] = useState(null); 



  const [updateInfo, setUpdateInfo] = useState(null); 

  useEffect(() => {
    (async () => {
      try {
        const [settings, v] = await Promise.all([
          window.inferml?.settings.get(),
          window.inferml?.app.version(),
        ]);
        if (settings?.theme) setTheme(settings.theme);
        if (v) setVersion(v);
      } catch {}
      setBootReady(true);
    })();
  }, []);

  useEffect(() => {



    let stop;
    let cancelled = false;
    (async () => {
      try {
        const initial = await window.inferml?.hw.get();
        if (!cancelled && initial && !initial.error) setHw(initial);
      } catch {}
      if (cancelled) return;
      stop = window.inferml?.hw.subscribe(data => { if (!cancelled && !data?.error) setHw(data); });
      if (cancelled && stop) { try { stop(); } catch {} stop = null; }
    })();
    return () => {
      cancelled = true;
      if (stop) { try { stop(); } catch {} }
    };
  }, []);

  const reloadSessions = useCallback(async () => {
    try { setSessions(await window.inferml.chats.list() || []); } catch { setSessions([]); }
  }, []);
  useEffect(() => {
    reloadSessions();
    const unsub = window.inferml?.chats.onUpdate(reloadSessions);
    return () => { if (unsub) unsub(); };
  }, [reloadSessions]);

  const reloadInstalled = useCallback(async () => {
    try { setInstalledModels((await window.inferml?.hf.installed()) || {}); } catch {}
  }, []);
  useEffect(() => { reloadInstalled(); }, [reloadInstalled, view]);


  useEffect(() => {
    const off = window.inferml?.hf?.onInstallsChanged?.(() => reloadInstalled());
    return () => { try { off && off(); } catch {} };
  }, [reloadInstalled]);





  useEffect(() => {
    let alive = true;
    const probe = async () => {
      try {
        const r = await window.inferml?.updates?.check?.();
        if (!alive) return;
        if (r?.ok && r.hasUpdate) setUpdateInfo(r);
        else setUpdateInfo(null);
      } catch {}
    };
    probe();
    const id = setInterval(probe, 2 * 24 * 60 * 60 * 1000);
    return () => { alive = false; clearInterval(id); };
  }, []);

  const refreshPyStatus = useCallback(async () => {
    try { setPyStatus((await window.inferml?.tasks.status()) || { ready: false }); } catch {}
  }, []);
  useEffect(() => { refreshPyStatus(); }, [refreshPyStatus]);

  // `pySetup` remembers the last install for as long as the app is open, and
  // `done: true` is what the onboarding screen reads to say "Python runtime ready".
  // Deleting the runtime has to clear it, or onboarding keeps reporting a runtime
  // that is no longer on disk - refreshing pyStatus alone is not enough, because
  // the stale flag wins.
  const resetPySetup = useCallback(() => setPySetup(null), []);





  useEffect(() => {
    const MAX_LOG = 500;
    let pendingLog = [];
    let pendingStep = null;
    let rafScheduled = false;

    const flush = () => {
      rafScheduled = false;
      if (!pendingLog.length && pendingStep == null) return;
      const linesToAdd = pendingLog;
      const stepToAdd = pendingStep;
      pendingLog = [];
      pendingStep = null;
      setPySetup(prev => {
        // Never `running: true` here. A progress frame is evidence that something
        // is talking, not that *this app* started an install - and runPySetup sets
        // running itself, synchronously, before the first frame can arrive. Taking
        // a frame as proof of an install is what let the runtime *removal* render
        // as a runtime download, complete with progress bar, that then never ended.
        const base = prev || { running: false, log: [], step: '' };
        const merged = base.log.concat(linesToAdd);
        const log = merged.length > MAX_LOG ? merged.slice(-MAX_LOG) : merged;
        return { ...base, step: stepToAdd ?? base.step, log };
      });
    };

    const schedule = () => {
      if (rafScheduled) return;
      rafScheduled = true;
      requestAnimationFrame(flush);
    };

    const unsub = window.inferml?.tasks.onSetupProgress((evt) => {
      if (evt.kind === 'step') {
        pendingStep = evt.text;
        pendingLog.push(`» ${evt.text}`);
      } else if (evt.kind === 'log') {
        pendingLog.push(evt.text.replace(/\s+$/, ''));
      } else {
        return;
      }
      schedule();
    });
    return () => {
      if (unsub) unsub();

      if (rafScheduled) flush();
    };
  }, []);

  const runPySetup = async (opts) => {

    setPySetup({ running: true, log: [], step: 'Starting…', error: null });
    const res = await window.inferml?.tasks.setup(opts);
    if (res?.ok) {
      setPySetup(prev => ({ ...(prev || {}), running: false, done: true, step: 'Ready' }));



      const accel = opts?.accelerator;
      setPyStatus(prev => ({
        ...(prev || {}),
        ready: true,
        runtimeInstalled: true,
        activeAccelerator: accel || prev?.activeAccelerator,
        installedAccelerator: accel || prev?.installedAccelerator,
        accelerators: {
          ...(prev?.accelerators || {}),
          ...(accel ? { [accel]: { installed: true, installedAt: new Date().toISOString() } } : {}),
        },
      }));
      refreshPyStatus();
    } else {
      setPySetup(prev => ({ ...(prev || { log: [] }), running: false, error: res?.error || 'setup failed' }));
    }
  };





  useEffect(() => {
    const onStart  = (e) => setUpdatingTo(e.detail?.version || 'latest');
    const onFailed = () => setUpdatingTo(null);
    window.addEventListener('inferml:update-installing', onStart);
    window.addEventListener('inferml:update-install-failed', onFailed);
    return () => {
      window.removeEventListener('inferml:update-installing', onStart);
      window.removeEventListener('inferml:update-install-failed', onFailed);
    };
  }, []);

  useEffect(() => {
    const themeClasses = ['light', 'theme-nord', 'theme-dracula', 'theme-tokyo', 'theme-catppuccin', 'theme-gruvbox', 'theme-onedark'];
    document.body.classList.remove(...themeClasses);
    if (theme === 'light')           document.body.classList.add('light');
    else if (theme === 'nord')       document.body.classList.add('theme-nord');
    else if (theme === 'dracula')    document.body.classList.add('theme-dracula');
    else if (theme === 'tokyo')      document.body.classList.add('theme-tokyo');
    else if (theme === 'catppuccin') document.body.classList.add('theme-catppuccin');
    else if (theme === 'gruvbox')    document.body.classList.add('theme-gruvbox');
    else if (theme === 'onedark')    document.body.classList.add('theme-onedark');

    window.inferml?.settings.save({ theme }).catch(() => {});
  }, [theme]);

  useEffect(() => {
    const handler = (e) => {
      if (!e.data) return;
      if (e.data.type === '__activate_edit_mode') setTweaksOpen(true);
      if (e.data.type === '__deactivate_edit_mode') setTweaksOpen(false);
    };
    window.addEventListener('message', handler);
    window.parent.postMessage({ type: '__edit_mode_available' }, '*');
    return () => window.removeEventListener('message', handler);
  }, []);

  useEffect(() => {
    const onKey = (e) => {
      const mod = e.metaKey || e.ctrlKey;
      if (mod && (e.key === 'k' || e.key === 'K')) {
        e.preventDefault();
        const input = document.querySelector('.side-search input');
        if (input) input.focus();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const finishWelcome = () => {




    const alreadyInstalled = pyStatus?.ready || pyStatus?.runtimeInstalled;
    if (!alreadyInstalled) setOnboard(true);
    setWelcome(false);
  };
  const finishOnboard = () => setOnboard(false);
  const setTheme_ = (t) => { setTheme(t); window.parent.postMessage({ type: '__edit_mode_set_keys', edits: { theme: t } }, '*'); };

  const openHub = () => {
    setView('hub');
    setActiveSession(null);
    setHubInstalledMode(false);
    setHubResetSignal(n => n + 1);
  };
  const openHubInstalled = () => { setView('hub'); setActiveSession(null); setHubInstalledMode(true); };
  const openSession = (id) => { setView('session'); setActiveSession(id); };
  const startSessionWithModel = async (modelId) => {

    const fresh = await window.inferml?.hf.installed().catch(() => null);
    const meta = (fresh && fresh[modelId]) || installedModels[modelId] || {};
    const task = meta?.task || '';
    const kind = CHAT_TASKS.has(task) ? 'chat' : 'task';
    const id = 'c-' + Math.random().toString(36).slice(2) + Date.now().toString(36);
    const session = {
      id,
      title: modelId.split('/').pop(),
      kind,
      modelId,
      task,
      ...(kind === 'chat' ? { messages: [] } : { runs: [] }),
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    try { await window.inferml.chats.save(session); } catch {}
    if (fresh) setInstalledModels(fresh);
    await reloadSessions();
    setView('session');
    setActiveSession(id);
  };

  const installedCount = Object.keys(installedModels).length;
  const gpuPill = hw?.gpu?.model ? `${truncate(hw.gpu.model, 18)} · ${gpuUsage(hw)}` : 'gpu · detecting…';

  const activeSessionObj = useMemo(() => {
    if (!activeSession) return null;
    return sessions.find(s => s.id === activeSession) || null;
  }, [activeSession, sessions]);

  if (!bootReady) return null;

  const renderMain = () => {
    if (view === 'hub') {
      return <ModelHub hw={hw} onOpenModel={startSessionWithModel} onOpenSettings={openSettings} defaultInstalled={hubInstalledMode} resetSignal={hubResetSignal}/>;
    }
    if (view === 'session' && activeSessionObj) {
      return renderWorkspace(activeSessionObj, installedModels, () => {});
    }
    return (
      <Landing
        onOpenHub={openHub}
        installedCount={installedCount}
      />
    );
  };

  return (
    <div className="win">
      <div className="win-frame">
        <div className="app-body">
          <aside className="sidebar">
            <div className="side-section" style={{paddingTop:12}}>
              <button
                className={`new-chat-btn ${view === 'hub' && !hubInstalledMode ? 'active' : ''}`}
                onClick={openHub}
              >
                <Icon name="home" size={14}/> Home
              </button>
              <button
                className={`new-chat-btn ${view === 'hub' && hubInstalledMode ? 'active' : ''}`}
                onClick={openHubInstalled}
                style={{marginTop: 4}}
              >
                <Icon name="plus" size={14}/> Installed only
              </button>
            </div>
            <div className="side-section side-flex">
              <button
                type="button"
                className="side-label side-label-toggle"
                onClick={() => setRecentsOpen(v => !v)}
                aria-expanded={recentsOpen}
              >
                <Icon name="chevron" size={11} className={`side-label-caret ${recentsOpen ? 'open' : ''}`}/>
                <span>Recents</span>
              </button>
              {recentsOpen && (
              <div className="side-chats">
                {sessions.length === 0 && (
                  <div className="side-empty">
                    {installedCount === 0
                      ? 'No models installed. Open Model Hub to download.'
                      : 'No sessions yet. Open Model Hub and pick a model.'}
                  </div>
                )}
                {sessions.map(s => (
                  <ChatItem
                    key={s.id}
                    session={s}
                    isActive={activeSession === s.id}
                    onOpen={openSession}
                    onDeleted={(id) => {


                      if (activeSession === id) { setActiveSession(null); setView('hub'); }
                    }}
                  />
                ))}
              </div>
              )}
            </div>

            <div className="side-footer">
              {!pyStatus?.ready && !pyStatus?.runtimeInstalled && !pySetup?.running && !onboard && (
                <button
                  className="side-setup-btn"
                  onClick={() => setOnboard(true)}
                  title="Python runtime is not installed yet. Click to set it up."
                >
                  <Icon name="alert" size={12}/>
                  <span>Setup Python runtime</span>
                </button>
              )}
              {pySetup?.running && (
                <button
                  className="side-setup-btn"
                  onClick={() => setOnboard(true)}
                  title={pySetup.step || 'Installing Python runtime…'}
                >
                  <span className="dot warn"/>
                  <span>Installing runtime…</span>
                </button>
              )}
              {updateInfo?.hasUpdate && !updatingTo && (
                <button
                  className="side-update-btn"
                  onClick={() => openSettings('general')}
                  title={`v${updateInfo.latestVersion} is available. Click to update from Settings.`}
                >
                  <Icon name="arrow_right" size={12}/>
                  <span>Update · v{String(updateInfo.latestVersion || '').replace(/^v/i, '')}</span>
                </button>
              )}
              <div className="side-stats mono" title={hw?.gpu?.model || ''}>
                {sbGpu(hw) && <span>GPU {sbGpu(hw)}</span>}
                <span>RAM {sbRam(hw)}</span>
                <span>CPU {sbCpu(hw)}</span>
              </div>
              <button className="new-chat-btn" onClick={() => openSettings()} title="Settings">
                <Icon name="settings" size={14}/> Settings
              </button>
            </div>

          </aside>

          <main className="main">
            {renderMain()}
          </main>
        </div>

        <Onboarding
          open={onboard && !welcome}
          onDone={finishOnboard}
          pyStatus={pyStatus}
          pySetup={pySetup}
          refreshPyStatus={refreshPyStatus}
          runSetup={runPySetup}
        />
        {welcome && <Welcome onStart={finishWelcome} version={version}/>}

        <Settings
          open={settingsOpen}
          onClose={() => setSettingsOpen(false)}
          initialSection={settingsSection}
          theme={theme}
          setTheme={setTheme_}
          version={version}
          hw={hw}
          pyStatus={pyStatus}
          pySetup={pySetup}
          runSetup={runPySetup}
          refreshPyStatus={refreshPyStatus}
          resetPySetup={resetPySetup}
        />
        {updatingTo && <UpdatingOverlay version={updatingTo}/>}
      </div>

      {tweaksOpen && (
        <div className="tweaks">
          <span className="lbl">Theme</span>
          <div className="tweaks-toggle">
            <button className={theme === 'dark' ? 'active' : ''} onClick={() => setTheme_('dark')}><Icon name="moon" size={11}/>Dark</button>
            <button className={theme === 'light' ? 'active' : ''} onClick={() => setTheme_('light')}><Icon name="sun" size={11}/>Light</button>
          </div>
          <div style={{width:1,height:16,background:'var(--line)'}}/>
          <button className="hp-chip" onClick={() => { setWelcome(true); }}>Replay welcome</button>
        </div>
      )}
    </div>
  );
}

function renderWorkspace(session, installedModels, onSaved) {
  const modelId = session.modelId || session.model;
  const installedMeta = (modelId && installedModels[modelId]) || null;

  const task = session.task || installedMeta?.task || '';
  const meta = installedMeta || { task };





  const isFlorence = /florence-?2/i.test(modelId || '');
  if (CHAT_TASKS.has(task) && !isFlorence) {
    return (
      <ChatWorkspace
        key={session.id}
        sessionId={session.id}
        modelId={modelId}
        modelMeta={meta}
        onSaved={onSaved}
      />
    );
  }
  return (
    <TaskWorkspace
      key={session.id}
      sessionId={session.id}
      modelId={modelId}
      modelMeta={meta}
      onSaved={onSaved}
    />
  );
}

function Landing({ onOpenHub, installedCount }) {
  return (
    <div className="chat-landing">
      <div className="cl-inner">
        <div className="cl-eyebrow">{timeGreeting()}</div>
        <div className="cl-title">Run any HuggingFace model locally</div>
        <div className="cl-sub">
          Download a model from the Model Hub and open a workspace for its task: chat with VLMs and LLMs, detect and segment images, transcribe audio, generate images.
        </div>
        <div className="cl-actions">
          <button className="cl-cta" onClick={onOpenHub}>
            <Icon name="cube" size={14}/> Open Model Hub
          </button>
        </div>
        <div className="cl-hint">
          {installedCount > 0
            ? `${installedCount} model${installedCount === 1 ? '' : 's'} installed. Pick one from the sidebar`
            : 'No models installed yet'}
        </div>
      </div>
    </div>
  );
}

function prettyTask(t) {
  if (!t) return 'Other';
  const s = String(t).replace(/-/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}
function taskIcon(task) {
  if (!task) return 'cube';
  if (task.includes('segmentation') || task.includes('detection') || task.includes('classification') || task === 'image-to-text') return 'eye';
  if (task.includes('speech') || task.includes('audio')) return 'waveform';
  if (task.includes('text-to-image')) return 'sparkle';
  if (task.includes('image-text') || task.includes('text-generation') || task === 'conversational') return 'chat';
  if (task.includes('feature')) return 'zap';
  return 'cube';
}
function taskColor(task) {
  if (!task) return 'var(--fg-2)';
  if (task.includes('segmentation') || task.includes('detection') || task.includes('classification')) return 'oklch(70% 0.14 155)';
  if (task.includes('speech') || task.includes('audio')) return 'oklch(70% 0.13 65)';
  if (task.includes('text-to-image')) return 'oklch(70% 0.15 320)';
  if (task.includes('image-text') || task.includes('text-generation') || task === 'conversational') return 'oklch(70% 0.12 230)';
  return 'var(--fg-2)';
}

function truncate(s, n) { return s && s.length > n ? s.slice(0, n - 1) + '…' : (s || ''); }
function gpuUsage(hw) {
  const tot = hw?.gpu?.memTotal || hw?.gpu?.vram || 0;
  const used = hw?.gpu?.memUsed || 0;
  if (!tot) return '-';
  return `${bytesToGb(used)}/${bytesToGb(tot)} GB`;
}
function bytesToGb(b) { return (b / (1024 ** 3)).toFixed(1); }
function vramLabel(hw) {
  const tot = hw?.gpu?.memTotal || hw?.gpu?.vram || 0;
  if (!tot) return 'detecting…';
  return `${bytesToGb(hw?.gpu?.memUsed || 0)} / ${bytesToGb(tot)} GB`;
}
function vramPct(hw) {
  const tot = hw?.gpu?.memTotal || hw?.gpu?.vram || 0;
  if (!tot) return 0;
  return Math.min(100, Math.round(((hw?.gpu?.memUsed || 0) / tot) * 100));
}
function ramLabel(hw) {
  if (!hw?.mem?.total) return 'detecting…';
  return `${bytesToGb(hw.mem.used)} / ${bytesToGb(hw.mem.total)} GB`;
}
function sbRam(hw) {
  if (!hw?.mem?.total) return '- GB';
  return `${(hw.mem.used / (1024 ** 3)).toFixed(2)} / ${(hw.mem.total / (1024 ** 3)).toFixed(2)} GB`;
}
function sbCpu(hw) {
  const v = hw?.cpu?.load;
  if (typeof v !== 'number') return '- %';
  return `${v.toFixed(2)} %`;
}

function sbGpu(hw) {
  if (!hw) return null;
  if (hw?.os?.platform === 'darwin' || hw?.gpu?.unified) return null;
  const used  = hw?.gpu?.memUsed;
  const total = hw?.gpu?.memTotal || hw?.gpu?.vram;
  if (typeof used !== 'number' || used <= 0 || !total) return null;
  return `${(used / (1024 ** 3)).toFixed(2)} / ${(total / (1024 ** 3)).toFixed(2)} GB`;
}
function relTime(ts) {
  if (!ts) return '';
  const s = Math.round((Date.now() - ts) / 1000);
  if (s < 60) return 'now';
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h`;
  const d = Math.round(h / 24);
  if (d < 7) return `${d}d`;
  return new Date(ts).toLocaleDateString();
}
function timeGreeting() {
  const h = new Date().getHours();
  if (h < 12) return 'Good morning';
  if (h < 18) return 'Good afternoon';
  return 'Good evening';
}

function ChatItem({ session, isActive, onOpen, onDeleted }) {
  const { useState, useEffect, useRef } = React;
  const [menuOpen, setMenuOpen] = useState(false);
  const [menuPos, setMenuPos] = useState({ top: 0, left: 0 });
  const [renaming, setRenaming] = useState(false);
  const [draft, setDraft] = useState(session.title || '');
  const [confirmOpen, setConfirmOpen] = useState(false);
  const rowRef = useRef(null);
  const menuRef = useRef(null);



  useEffect(() => {
    if (!menuOpen) return;
    const onDown = (e) => {
      const inRow  = rowRef.current?.contains(e.target);
      const inMenu = menuRef.current?.contains(e.target);
      if (!inRow && !inMenu) setMenuOpen(false);
    };
    const onKey    = (e) => { if (e.key === 'Escape') setMenuOpen(false); };
    const onScroll = () => setMenuOpen(false);
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);

    document.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', onScroll);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', onScroll);
    };
  }, [menuOpen]);

  const openMenu = (e) => {
    e.stopPropagation();
    if (menuOpen) { setMenuOpen(false); return; }
    const rect = e.currentTarget.getBoundingClientRect();
    const MENU_W = 150;

    const spaceBelow = window.innerHeight - rect.bottom;
    const top = spaceBelow < 140 ? rect.top - 124 : rect.bottom + 4;
    const left = Math.max(8, Math.min(window.innerWidth - MENU_W - 8, rect.right - MENU_W));
    setMenuPos({ top, left });
    setMenuOpen(true);
  };

  const togglePin = async () => {
    setMenuOpen(false);

    try { await window.inferml?.chats.patch(session.id, { pinned: !session.pinned }); } catch {}
  };

  const startRename = () => {
    setMenuOpen(false);
    setDraft(session.title || '');
    setRenaming(true);
  };

  const submitRename = async () => {
    const t = draft.trim();
    setRenaming(false);
    if (!t || t === (session.title || '')) return;
    try { await window.inferml?.chats.patch(session.id, { title: t, updatedAt: Date.now() }); } catch {}
  };

  const askDelete = () => {
    setMenuOpen(false);
    setConfirmOpen(true);
  };

  const confirmDelete = async () => {
    setConfirmOpen(false);
    try {
      await window.inferml?.chats.delete(session.id);
      onDeleted && onDeleted(session.id);
    } catch {}
  };

  const onRowClick = () => {
    if (renaming || menuOpen) return;
    onOpen(session.id);
  };

  return (
    <div
      ref={rowRef}
      className={`chat-item ${isActive ? 'active' : ''} ${session.pinned ? 'pinned' : ''}`}
      onClick={onRowClick}
    >
      <div className="chat-item-icon" aria-hidden="true">
        <Icon name="history" size={20}/>
      </div>
      <div className="chat-item-body">
        {renaming ? (
          <input
            className="chat-rename-input"
            value={draft}
            autoFocus
            onChange={(e) => setDraft(e.target.value)}
            onClick={(e) => e.stopPropagation()}
            onBlur={submitRename}
            onKeyDown={(e) => {
              if (e.key === 'Enter') submitRename();
              if (e.key === 'Escape') { setRenaming(false); setDraft(session.title || ''); }
            }}
          />
        ) : (
          <div className="t1">
            {session.pinned && <Icon name="pin" size={10} stroke={1.8} style={{marginRight:5,color:'var(--accent)',flexShrink:0}}/>}
            <span style={{overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{session.title || 'Untitled'}</span>
          </div>
        )}
        <div className="t2">
          <span style={{overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap',minWidth:0}}>{prettyTask(session.task) || 'session'}</span>
        </div>
      </div>

      <button
        className={`chat-kebab ${menuOpen ? 'open' : ''}`}
        onClick={openMenu}
        aria-label="More actions"
      >
        <Icon name="dots" size={16}/>
      </button>

      {menuOpen && (
        <div
          ref={menuRef}
          className="chat-menu"
          style={{ top: menuPos.top, left: menuPos.left }}
          onClick={(e) => e.stopPropagation()}
        >
          <button className="chat-menu-item" onClick={togglePin}>
            <Icon name="pin" size={12}/> {session.pinned ? 'Unpin' : 'Pin'}
          </button>
          <button className="chat-menu-item" onClick={startRename}>
            <Icon name="pencil" size={12}/> Rename
          </button>
          <button className="chat-menu-item danger" onClick={askDelete}>
            <Icon name="trash" size={12}/> Delete
          </button>
        </div>
      )}

      <ConfirmDialog
        open={confirmOpen}
        title="Delete session?"
        message={<>This will permanently delete <b>{session.title || 'Untitled'}</b> and its run history. This cannot be undone.</>}
        confirmLabel="Delete"
        danger
        onConfirm={confirmDelete}
        onCancel={() => setConfirmOpen(false)}
      />
    </div>
  );
}

function UpdatingOverlay({ version }) {
  return (
    <div className="updating-overlay">
      <div className="updating-card">
        <Logo size={64}/>
        <div className="updating-title">Installing update</div>
        <div className="updating-version mono">v{version}</div>
        <div className="updating-progress"><div className="updating-progress-bar"/></div>
        <div className="updating-hint">The app will restart automatically.</div>
      </div>
    </div>
  );
}

function ConfirmDialog({ open, title, message, confirmLabel = 'OK', cancelLabel = 'Cancel', danger = false, onConfirm, onCancel }) {
  const { useEffect, useRef } = React;
  const confirmRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e) => {
      if (e.key === 'Escape') { e.stopPropagation(); onCancel && onCancel(); }
      if (e.key === 'Enter')  { e.stopPropagation(); onConfirm && onConfirm(); }
    };
    window.addEventListener('keydown', onKey, true);

    setTimeout(() => confirmRef.current?.focus(), 0);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [open, onCancel, onConfirm]);

  if (!open) return null;
  return (
    <div className="confirm-modal" onClick={onCancel}>
      <div className="confirm-card" onClick={(e) => e.stopPropagation()}>
        {title && <div className="confirm-title">{title}</div>}
        <div className="confirm-message">{message}</div>
        <div className="confirm-actions">
          <button className="mc-btn ghost" onClick={onCancel}>{cancelLabel}</button>
          <button
            ref={confirmRef}
            className={`mc-btn ${danger ? 'danger' : 'primary'}`}
            onClick={onConfirm}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
