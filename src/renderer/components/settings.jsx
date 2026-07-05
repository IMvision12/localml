const { useState: useStateS, useEffect: useEffectS } = React;

function Settings({
  open,
  onClose,
  initialSection,
  theme,
  setTheme,
  version,
  hw,
  pyStatus,
  pySetup,
  runSetup,
  refreshPyStatus,
}) {
  const [section, setSection] = useStateS(initialSection || 'general');


  useEffectS(() => {
    if (open && initialSection) setSection(initialSection);
  }, [open, initialSection]);

  useEffectS(() => {
    if (!open) return;
    const onKey = (e) => { if (e.key === 'Escape') onClose && onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  const sections = [
    { id: 'general',    label: 'General',       icon: 'settings' },
    { id: 'appearance', label: 'Appearance',    icon: 'image' },
    { id: 'hardware',   label: 'Hardware',      icon: 'cpu' },
    { id: 'hf',         label: 'HF Token',      icon: 'cube' },
  ];
  const active = sections.find(s => s.id === section) || sections[0];

  return (
    <div className="settings-modal" onClick={onClose}>
      <div className="settings-card" onClick={(e) => e.stopPropagation()}>
        <aside className="settings-nav">
          <div className="settings-nav-title">Settings</div>
          {sections.map(s => (
            <button
              key={s.id}
              className={`settings-nav-item ${section === s.id ? 'active' : ''}`}
              onClick={() => setSection(s.id)}
            >
              <Icon name={s.icon} size={15}/>
              <span>{s.label}</span>
            </button>
          ))}
        </aside>

        <div className="settings-content">
          <div className="settings-header">
            <h2>{active.label}</h2>
            <button className="settings-close" onClick={onClose} title="Close (Esc)">
              <Icon name="close" size={14}/>
            </button>
          </div>

          <div className="settings-body">
            {section === 'general'    && <GeneralSection    version={version} hw={hw} pyStatus={pyStatus} refreshPyStatus={refreshPyStatus}/>}
            {section === 'appearance' && <AppearanceSection theme={theme} setTheme={setTheme}/>}
            {section === 'hardware'   && <HardwareSection   hw={hw} pyStatus={pyStatus} pySetup={pySetup} runSetup={runSetup} refreshPyStatus={refreshPyStatus}/>}
            {section === 'hf'         && <HFSection/>}
          </div>
        </div>
      </div>
    </div>
  );
}

function Row({ title, sub, value, control, onClick }) {
  return (
    <div className={`s-row ${onClick ? 'clickable' : ''}`} onClick={onClick}>
      <div className="s-row-l">
        <div className="s-row-t">{title}</div>
        {sub && <div className="s-row-s">{sub}</div>}
      </div>
      {value && <div className="s-row-v mono">{value}</div>}
      {control}
    </div>
  );
}

function GeneralSection({ version, hw, pyStatus, refreshPyStatus }) {
  const openExternal = (url) => window.inferml?.app?.openExternal?.(url);
  const [logsPath, setLogsPath] = useStateS('');
  const [cacheStat, setCacheStat] = useStateS({ bytes: null, files: null, paths: [] });
  const [runtimeStat, setRuntimeStat] = useStateS({ bytes: null, files: null });



  const [sizeLoading, setSizeLoading] = useStateS({ hf: true, py: true });
  const [confirmKey, setConfirmKey] = useStateS(null); 
  const [busy, setBusy] = useStateS(null); 
  const [error, setError] = useStateS(null);



  const refreshSizes = async () => {
    const MIN_HOLD_MS = 350;
    const start = performance.now();
    setSizeLoading({ hf: true, py: true });
    const settle = async (key, promise) => {
      try {
        const res = await promise;
        if (key === 'hf' && res?.ok) setCacheStat({ bytes: res.bytes, files: res.files, paths: res.paths || [] });
        if (key === 'py' && res?.ok) setRuntimeStat({ bytes: res.bytes, files: res.files });
      } finally {
        const elapsed = performance.now() - start;
        if (elapsed < MIN_HOLD_MS) {
          await new Promise(r => setTimeout(r, MIN_HOLD_MS - elapsed));
        }
        setSizeLoading(s => ({ ...s, [key]: false }));
      }
    };
    try {
      await Promise.all([
        settle('hf', window.inferml?.storage?.size?.('hfCache')),
        settle('py', window.inferml?.storage?.size?.('pyRuntime')),
      ]);
    } catch {
      setSizeLoading({ hf: false, py: false });
    }
  };

  useEffectS(() => {
    let alive = true;
    (async () => {
      try {
        const p = await window.inferml?.logs?.path?.();
        if (alive && p) setLogsPath(p);
      } catch {}
      if (alive) refreshSizes();
    })();
    return () => { alive = false; };
  }, []);



  useEffectS(() => {
    const off = window.inferml?.hf?.onInstallsChanged?.(() => refreshSizes());
    return () => { try { off && off(); } catch {} };
  }, []);

  const viewLogs = async () => {
    try { await window.inferml?.logs?.view?.(); } catch {}
  };

  const doClear = async (key) => {
    setError(null);
    setBusy(key);
    try {
      const res = key === 'hf'
        ? await window.inferml?.storage?.clearHfCache?.()
        : await window.inferml?.storage?.clearPyRuntime?.();
      if (!res?.ok) setError(res?.error || 'Clear failed');
    } catch (e) {
      setError(String(e?.message || e));
    } finally {
      setBusy(null);
      setConfirmKey(null);
      await refreshSizes();
      try { await refreshPyStatus?.(); } catch {}
    }
  };

  const fmtBytes = (n) => {
    if (n == null) return '-';
    if (n < 1024) return `${n} B`;
    const u = ['KB', 'MB', 'GB', 'TB'];
    let v = n / 1024, i = 0;
    while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
    return `${v.toFixed(v >= 100 ? 0 : 1)} ${u[i]}`;
  };

  return (
    <div className="s-section">
      <h3 className="s-h">App Info</h3>
      <div className="s-card">
        <Row
          title="Version"
          value={`InferML v${version || '-'}`}
          control={<UpdateCheckButton currentVersion={version}/>}
        />
        <Row title="Platform" value={`${hw?.os?.distro || hw?.os?.platform || '-'} · ${hw?.os?.arch || ''}`.trim()}/>
      </div>

      <h3 className="s-h">Help & Feedback</h3>
      <div className="s-card">
        <Row
          title="Report bug or send feedback"
          control={
            <button
              className="mc-btn ghost"
              onClick={(e) => { e.stopPropagation(); openExternal('https://github.com/IMvision12/InferML/issues'); }}
            >
              <Icon name="arrow_right" size={11}/> Open in browser
            </button>
          }
        />
      </div>

      <h3 className="s-h">Storage</h3>
      <div className="s-card">
        <Row
          title="Models cache"
          sub={(() => {
            const paths = cacheStat.paths || [];
            const pathStr = paths.length === 0
              ? (pyStatus?.hfCachePath || '-')
              : (paths.length === 1 ? paths[0] : paths.join('  +  '));
            if (sizeLoading.hf) return `${pathStr} · Calculating…`;
            const sizeStr = `${fmtBytes(cacheStat.bytes)}${cacheStat.files != null ? ` · ${cacheStat.files} file${cacheStat.files === 1 ? '' : 's'}` : ''}`;
            return `${pathStr} · ${sizeStr}`;
          })()}
          control={
            <button
              className="mc-btn ghost danger"
              onClick={(e) => { e.stopPropagation(); setConfirmKey('hf'); }}
              title="Delete every downloaded model in every detected cache. Re-running an installed model will re-download it."
              disabled={busy === 'hf' || sizeLoading.hf || cacheStat.bytes === 0}
            >
              <Icon name="trash" size={11}/> {busy === 'hf' ? 'Clearing…' : sizeLoading.hf ? 'Calculating…' : 'Clean'}
            </button>
          }
        />
        <Row
          title="Python runtime"
          sub={(() => {
            const pathStr = pyStatus?.runtimePath || '-';
            if (sizeLoading.py) return `${pathStr} · Calculating…`;
            const sizeStr = `${fmtBytes(runtimeStat.bytes)}${runtimeStat.files != null ? ` · ${runtimeStat.files} file${runtimeStat.files === 1 ? '' : 's'}` : ''}`;
            return `${pathStr} · ${sizeStr}`;
          })()}
          control={
            <button
              className="mc-btn ghost danger"
              onClick={(e) => { e.stopPropagation(); setConfirmKey('py'); }}
              title="Delete the bundled Python venv. Re-running a model will re-install (~1 GB download)."
              disabled={busy === 'py' || sizeLoading.py || runtimeStat.bytes === 0}
            >
              <Icon name="trash" size={11}/> {busy === 'py' ? 'Clearing…' : sizeLoading.py ? 'Calculating…' : 'Clean'}
            </button>
          }
        />
        <Row
          title="Logs"
          sub={logsPath || '-'}
          control={
            <button
              className="mc-btn ghost"
              onClick={(e) => { e.stopPropagation(); viewLogs(); }}
              title="Open log file in text editor"
              disabled={!logsPath}
            >
              <Icon name="file" size={11}/> View
            </button>
          }
        />
        {error && <div className="s-row-error mono">{error}</div>}
      </div>

      <ConfirmDialog
        open={confirmKey === 'hf'}
        title="Clear models cache?"
        message="Deletes every weight file in hf-cache. Models you've installed will need to be re-downloaded the next time you run them. This does not affect your sessions or settings."
        confirmLabel="Clear cache"
        cancelLabel="Cancel"
        danger
        onConfirm={() => doClear('hf')}
        onCancel={() => setConfirmKey(null)}
      />
      <ConfirmDialog
        open={confirmKey === 'py'}
        title="Clear Python runtime?"
        message="Deletes the bundled Python venv (interpreter, uv, torch, transformers, all deps). Next time you run a model you'll have to re-install the runtime (~1 GB download, several minutes). Use this if torch / torchaudio / transformers are misbehaving."
        confirmLabel="Clear runtime"
        cancelLabel="Cancel"
        danger
        onConfirm={() => doClear('py')}
        onCancel={() => setConfirmKey(null)}
      />
    </div>
  );
}

function AppearanceSection({ theme, setTheme }) {
  const themes = [
    { id: 'dark',       label: 'Dark' },
    { id: 'light',      label: 'Light' },
    { id: 'nord',       label: 'Nord' },
    { id: 'dracula',    label: 'Dracula' },
    { id: 'tokyo',      label: 'Tokyo Night' },
    { id: 'catppuccin', label: 'Catppuccin' },
    { id: 'gruvbox',    label: 'Gruvbox' },
    { id: 'onedark',    label: 'One Dark' },
  ];
  return (
    <div className="s-section">
      <h3 className="s-h">Theme</h3>
      <div className="s-card s-card-pad">
        <div className="s-theme-grid">
          {themes.map(t => (
            <button
              key={t.id}
              className={`s-theme-card ${theme === t.id ? 'active' : ''}`}
              onClick={() => setTheme(t.id)}
            >
              <div className={`s-theme-preview ${t.id}`}>
                <div className="stp-bar"/>
                <div className="stp-row stp-r1"/>
                <div className="stp-row stp-r2"/>
                <div className="stp-row stp-r3"/>
              </div>
              <div className="s-theme-name">{t.label}</div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function HardwareSection({ hw, pyStatus, pySetup, runSetup, refreshPyStatus }) {
  const isMac = pyStatus?.platform === 'darwin';
  const running = !!pySetup?.running;
  const ready = !!pyStatus?.ready;
  const fmtGb = (b) => b ? (b / (1024 ** 3)).toFixed(1) + ' GB' : '-';

  return (
    <div className="s-section">
      <h3 className="s-h">System</h3>
      <div className="s-card">
        <Row title="CPU" sub={`${hw?.cpu?.brand || '-'}${hw?.cpu?.cores ? ` · ${hw.cpu.cores}c / ${hw.cpu.threads}t` : ''}`}/>
        <Row title="GPU" sub={`${hw?.gpu?.model || '-'}${hw?.gpu?.memTotal ? ` · ${fmtGb(hw.gpu.memTotal)} VRAM${hw?.gpu?.unified ? ' (unified)' : ''}` : ''}`}/>
        <Row title="RAM" sub={`${fmtGb(hw?.mem?.total)}${hw?.mem?.free ? ` · ${fmtGb(hw.mem.free)} free` : ''}`}/>
        <Row title="Disk" sub={`${fmtGb(hw?.disk?.free)} free${hw?.disk?.mount ? ` · ${hw.disk.mount}` : ''}`}/>
      </div>

      <h3 className="s-h">Python Runtime</h3>
      <div className="s-card">
        <Row
          title="Status"
          sub={running ? (pySetup?.step || 'Installing…') : ready ? 'Ready to run inference' : 'Not installed. Pick CPU or GPU to begin.'}
          control={
            running
              ? <span className="s-pill warn"><span className="dot"/> installing…</span>
              : ready
                ? <span className="s-pill ok"><span className="dot"/> ready</span>
                : (() => {



                    const suggested = pyStatus?.suggestedAccelerator || 'cpu';
                    const hasNvidia = !!pyStatus?.hasNvidia && !isMac;
                    return (
                      <div style={{display:'flex', gap:8, flexWrap:'wrap'}}>
                        {hasNvidia && (
                          <button
                            className="mc-btn primary"
                            onClick={() => runSetup && runSetup({ accelerator: 'gpu' })}
                            title="Install the CUDA-enabled PyTorch wheel (~1 GB). Recommended on machines with an NVIDIA GPU."
                          >
                            <Icon name="arrow_right" size={12}/> Install GPU runtime
                          </button>
                        )}
                        <button
                          className={hasNvidia ? 'mc-btn ghost' : 'mc-btn primary'}
                          onClick={() => runSetup && runSetup({ accelerator: 'cpu' })}
                          title={hasNvidia
                            ? 'Install the CPU-only PyTorch wheel. Lighter (~600 MB), runs on any hardware.'
                            : `Install the ${isMac ? 'Apple Silicon (MPS)' : 'CPU'} PyTorch wheel.`}
                        >
                          <Icon name="arrow_right" size={12}/> Install {isMac ? 'runtime' : 'CPU runtime'}
                        </button>
                      </div>
                    );
                  })()
          }
        />
        {pySetup?.error && !running && (
          <div className="s-row-error mono" style={{marginTop:6}}>{pySetup.error}</div>
        )}
        {pyStatus?.installedAccelerator && (() => {
          const active = pyStatus.installedAccelerator;
          const next = active === 'gpu' ? 'cpu' : 'gpu';
          const otherInstalled = !!pyStatus?.accelerators?.[next]?.installed;
          const switchLabel = otherInstalled
            ? `Switch to ${next.toUpperCase()}`
            : `Install ${next.toUpperCase()} runtime`;
          return (
            <Row
              title="Accelerator"
              sub={active === 'gpu'
                ? (isMac ? 'Apple Metal (MPS)' : 'GPU (CUDA 12.4)')
                : 'CPU only'}
              control={!isMac && ready && !running
                ? <button
                    className="mc-btn primary"
                    onClick={() => runSetup && runSetup({ accelerator: next })}
                    title={otherInstalled
                      ? `${next.toUpperCase()} runtime is already installed in py-runtime/${next}/. Switching just flips the active pointer.`
                      : `${next.toUpperCase()} runtime not installed yet. Will download torch + deps for the new accelerator (~1 GB).`}
                  >
                    <Icon name="arrow_right" size={12}/> {switchLabel}
                  </button>
                : null}
            />
          );
        })()}
        {running && (
          <div className="s-row" style={{paddingTop: 6}}>
            <div className="s-row-l" style={{flex:1}}>
              <div className="py-progress"><div className="py-progress-bar"/></div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function HFSection() {
  return (
    <div className="s-section">
      <h3 className="s-h">Access Token</h3>
      <div className="s-card s-card-pad">
        <HFTokenCard/>
      </div>
    </div>
  );
}

function UpdateCheckButton({ currentVersion }) {
  const [state, setState] = useStateS('idle');
  const [result, setResult] = useStateS(null);
  const [progress, setProgress] = useStateS(0);   
  const [confirmOpen, setConfirmOpen] = useStateS(false);
  const openExternal = (url) => window.inferml?.app?.openExternal?.(url);





  useEffectS(() => {
    let mounted = true;
    const u = window.inferml?.updates;
    if (!u) return;
    const offProgress = u.onProgress?.((evt) => {
      if (!mounted) return;
      setProgress(Math.round(evt?.percent || 0));
      setState((s) => (s === 'downloading' || s === 'available' ? 'downloading' : s));
    });
    const offDownloaded = u.onDownloaded?.((evt) => {
      if (!mounted) return;
      setProgress(100);
      setResult((r) => ({ ...(r || {}), latestVersion: evt?.version || r?.latestVersion }));
      setState('downloaded');
    });
    const offError = u.onError?.((evt) => {
      if (!mounted) return;




      const errStr = evt?.error || 'Unknown error';
      setResult((r) => {
        const hadUpdate = !!(r && r.hasUpdate);
        return {
          ...(r || {}),
          error: errStr,
          ...(hadUpdate ? { canAutoUpdate: false } : {}),
        };
      });
      setState((s) => {
        if (s === 'downloading' || s === 'downloaded') return 'available';
        return 'idle';
      });
    });

    (async () => {
      if (!mounted) return;
      setState('checking');
      try {
        const r = await u.check?.();
        if (!mounted) return;
        if (!r || !r.ok) { setState('idle'); return; }
        setResult(r);
        setState(r.hasUpdate ? 'available' : 'uptodate');
      } catch { if (mounted) setState('idle'); }
    })();

    return () => {
      mounted = false;
      offProgress && offProgress();
      offDownloaded && offDownloaded();
      offError && offError();
    };
  }, []);

  const check = async () => {
    setState('checking');
    setProgress(0);
    try {

      const r = await window.inferml?.updates?.check?.({ force: true });
      if (!r || !r.ok) {

        setResult(r || { error: 'unknown' });
        setState('idle');
        return;
      }
      setResult(r);
      setState(r.hasUpdate ? 'available' : 'uptodate');
    } catch (e) {
      setResult({ error: String(e?.message || e) });
      setState('idle');
    }
  };



  const requestDownload = () => setConfirmOpen(true);

  const confirmAndDownload = async () => {
    setConfirmOpen(false);
    setProgress(0);
    setState('downloading');
    try {
      const r = await window.inferml?.updates?.download?.();
      if (!r) { setState('error'); setResult({ error: 'No response' }); return; }
      if (r.alreadyDownloaded) { setProgress(100); setState('downloaded'); return; }
      if (!r.ok) { setResult({ error: r.error || 'Download failed' }); setState('error'); return; }

    } catch (e) {
      setResult({ error: String(e?.message || e) });
      setState('error');
    }
  };

  const installAndRestart = async () => {
    setState('installing');



    window.dispatchEvent(new CustomEvent('inferml:update-installing', {
      detail: { version: result?.latestVersion || '' },
    }));



    const timeoutId = setTimeout(() => {
      window.dispatchEvent(new CustomEvent('inferml:update-install-failed'));
      setResult((r) => ({ ...(r || {}), error: 'Install timed out. Try downloading the installer manually from the website.' }));
      setState('idle');
    }, 30000);

    try { await window.inferml?.updates?.install?.(); }
    catch (e) {
      clearTimeout(timeoutId);
      setResult({ error: String(e?.message || e) });
      setState('error');
      window.dispatchEvent(new CustomEvent('inferml:update-install-failed'));
    }

  };

  if (state === 'checking') {
    return (
      <button className="mc-btn ghost" disabled>
        <span className="upd-spin"/> Checking…
      </button>
    );
  }
  if (state === 'available' && result?.hasUpdate) {
    const canAuto = !!result.canAutoUpdate;
    return (
      <>
        <div className="upd-result">
          <span className="upd-tag mono">{result.latestVersion}</span>
          {canAuto ? (
            <button className="mc-btn primary" onClick={requestDownload}>
              <Icon name="arrow_right" size={11}/> Download
            </button>
          ) : (
            <button
              className="mc-btn primary"
              onClick={() => openExternal(result.downloadPageUrl || result.releaseUrl)}
            >
              <Icon name="arrow_right" size={11}/> Open download page
            </button>
          )}
        </div>
        <ConfirmDialog
          open={confirmOpen}
          title={`Download InferML ${result.latestVersion}?`}
          message="The update will download in the background. You'll be prompted to install and restart when it's ready."
          confirmLabel="Download"
          cancelLabel="Cancel"
          onConfirm={confirmAndDownload}
          onCancel={() => setConfirmOpen(false)}
        />
      </>
    );
  }
  if (state === 'downloading') {
    return (
      <div className="upd-result upd-result-col">
        <div className="upd-progress">
          <div className="upd-progress-bar" style={{ width: `${progress}%` }}/>
        </div>
        <span className="upd-tag mono">{progress}%</span>
      </div>
    );
  }
  if (state === 'downloaded') {
    return (
      <div className="upd-result">
        <span className="upd-tag mono">{result?.latestVersion || 'ready'}</span>
        <button className="mc-btn primary" onClick={installAndRestart}>
          <Icon name="arrow_right" size={11}/> Install &amp; restart
        </button>
      </div>
    );
  }
  if (state === 'installing') {
    return (
      <button className="mc-btn ghost" disabled>
        <span className="upd-spin"/> Restarting…
      </button>
    );
  }
  if (state === 'uptodate') {
    return (
      <button className="mc-btn ghost" onClick={check}>
        <Icon name="check_c" size={11}/> Up to date
      </button>
    );
  }
  return (
    <button className="mc-btn ghost" onClick={check} title={result?.error || ''}>
      <Icon name="arrow_right" size={11}/> Check for updates
    </button>
  );
}
