const { useState: useStateOB, useEffect: useEffectOB, useRef: useRefOB } = React;

function Onboarding({ open, onDone, pyStatus = {}, pySetup, refreshPyStatus, runSetup }) {
  const [hw, setHw] = useStateOB(null);
  const [showLog, setShowLog] = useStateOB(false);
  const [copied, setCopied] = useStateOB(false);

  const [accelerator, setAccelerator] = useStateOB(null);
  const logRef = useRefOB(null);
  const triggeredRef = useRefOB(false);
  const copyTimerRef = useRefOB(null);

  useEffectOB(() => {
    if (!open) { triggeredRef.current = false; return; }
    let mounted = true;
    (async () => {
      try {
        const h = await window.inferml?.hw.get();
        if (mounted && h && !h.error) setHw(h);
      } catch {}
    })();
    return () => { mounted = false; };
  }, [open]);


  const suggestedAccel = pyStatus?.suggestedAccelerator || (
    hw?.gpu?.vendor?.toLowerCase().includes('nvidia') ? 'gpu' : 'cpu'
  );

  const startSetup = (accel) => {
    if (pySetup?.running || pySetup?.done) return;
    setAccelerator(accel);
    triggeredRef.current = true;
    runSetup && runSetup({ accelerator: accel });
  };



  useEffectOB(() => {
    if (!open) return;
    if (pySetup?.running || pySetup?.done || triggeredRef.current) return;
    if (pyStatus?.platform === 'darwin') {
      triggeredRef.current = true;
      runSetup && runSetup({ accelerator: 'gpu' });
    }
  }, [open, pyStatus?.platform, pySetup?.running, pySetup?.done]);

  useEffectOB(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [pySetup?.log?.length]);

  const copyLog = () => {
    const text = (pySetup?.log || []).join('\n');
    if (!text) return;
    try { navigator.clipboard?.writeText(text); } catch {}
    setCopied(true);
    if (copyTimerRef.current) clearTimeout(copyTimerRef.current);
    copyTimerRef.current = setTimeout(() => setCopied(false), 1400);
  };
  useEffectOB(() => () => { if (copyTimerRef.current) clearTimeout(copyTimerRef.current); }, []);





  useEffectOB(() => {
    if (!open) return;
    if (!pyStatus?.ready) return;
    if (pySetup?.running || pySetup?.done || triggeredRef.current) return;
    onDone && onDone();
  }, [open, pyStatus?.ready, pySetup?.running, pySetup?.done]);

  const running = !!pySetup?.running;
  const done = !!pyStatus?.ready || !!pySetup?.done;
  const errored = !!pySetup?.error;
  const step = running ? (pySetup?.step || 'Preparing environment…')
             : errored ? 'Setup failed'
             : done ? 'Ready'
             : 'Setting up local runtime…';

  const canContinue = done && !running;

  return (
    <div className="onboard" style={{display: open ? 'flex' : 'none'}}>
      <div className="ob-main">
        <h1 className="ob-h1">Hardware</h1>
        <p className="ob-h2">
          This is what local HuggingFace models will run on. We warn before downloading anything that won't fit.
        </p>

        <div className="hw-detect">
          <h4>System</h4>
          <div className="hw-grid">
            <div className="hw-item"><div className="lbl">GPU</div><div className={`val ${hw?.gpu?.model ? 'ok' : ''}`}>{hw?.gpu?.model || 'Detecting…'}{hw?.gpu?.memTotal ? <span className="u"> · {gbFmt(hw.gpu.memTotal)} GB VRAM</span> : (hw?.gpu?.vram ? <span className="u"> · {gbFmt(hw.gpu.vram)} GB VRAM</span> : null)}</div></div>
            <div className="hw-item"><div className="lbl">BACKEND</div><div className="val">{inferBackend(hw)}</div></div>
            <div className="hw-item"><div className="lbl">CPU</div><div className="val">{hw?.cpu?.brand || 'Detecting…'}{hw?.cpu?.cores && <span className="u"> · {hw.cpu.cores}c / {hw.cpu.threads}t</span>}</div></div>
            <div className="hw-item"><div className="lbl">RAM</div><div className="val">{hw?.mem?.total ? `${gbFmt(hw.mem.total)} GB` : '-'}{hw?.mem?.free && <span className="u"> · {gbFmt(hw.mem.free)} GB free</span>}</div></div>
            <div className="hw-item"><div className="lbl">DISK</div><div className="val">{hw?.disk?.free ? `${gbFmt(hw.disk.free)} GB free` : '-'}{hw?.disk?.mount && <span className="u"> · {hw.disk.mount}</span>}</div></div>
            <div className="hw-item"><div className="lbl">OS</div><div className="val">{(hw?.os?.distro || hw?.os?.platform) || '-'}{hw?.os?.build && <span className="u"> · build {hw.os.build}</span>}</div></div>
          </div>
        </div>

        {!running && !done && !triggeredRef.current && pyStatus?.platform !== 'darwin' && (
          <div className="py-card idle">
            <div className="py-card-head">
              <Icon name="cube" size={18}/>
              <div className="py-card-titles">
                <div className="py-card-t">Choose runtime</div>
                <div className="py-card-s">
                  We'll set up a Python ML runtime in the background. Python, PyTorch and transformers. Pick once; runs locally after that.
                </div>
              </div>
            </div>
            <div className="accel-pick">
              <button
                className={`accel-card ${suggestedAccel === 'cpu' ? 'suggested' : ''}`}
                onClick={() => startSetup('cpu')}
              >
                <div className="accel-h">
                  <Icon name="cpu" size={16}/>
                  <span className="accel-name">CPU</span>
                  {suggestedAccel === 'cpu' && <span className="accel-badge">recommended</span>}
                </div>
                <div className="accel-sub">Works everywhere. Slower for diffusion / VLMs.</div>
                <div className="accel-meta mono">~2 GB download · torch-cpu</div>
              </button>
              <button
                className={`accel-card ${suggestedAccel === 'gpu' ? 'suggested' : ''}`}
                onClick={() => startSetup('gpu')}
              >
                <div className="accel-h">
                  <Icon name="gpu" size={16}/>
                  <span className="accel-name">GPU</span>
                  {suggestedAccel === 'gpu' && <span className="accel-badge">recommended</span>}
                </div>
                <div className="accel-sub">
                  {pyStatus?.platform === 'darwin'
                    ? 'Apple Metal (MPS). Apple Silicon only.'
                    : 'NVIDIA CUDA 12.4. Needs a recent NVIDIA driver.'}
                </div>
                <div className="accel-meta mono">
                  {pyStatus?.platform === 'darwin' ? '~2 GB · MPS-enabled torch' : '~5 GB · torch + CUDA'}
                </div>
              </button>
            </div>
            <div className="py-card-s" style={{marginTop: 10, opacity: 0.7}}>
              {pyStatus?.hasNvidia
                ? 'NVIDIA GPU detected. GPU runtime will use CUDA.'
                : 'No NVIDIA GPU detected. CPU is the safe default.'}
            </div>
          </div>
        )}

        <div className={`py-card ${running ? 'running' : done ? 'ok' : errored ? 'bad' : 'idle'}`} style={{display: (!running && !done && !triggeredRef.current) ? 'none' : ''}}>
          <div className="py-card-head">
            {running && <div className="py-spinner"><span/><span/><span/></div>}
            {done && !running && <Icon name="check_c" size={18}/>}
            {errored && <Icon name="alert" size={18}/>}
            <div className="py-card-titles">
              <div className="py-card-t">{running ? 'Setting up the Python runtime' : done ? 'Python runtime ready' : errored ? 'Setup failed' : 'Setting up the Python runtime'}</div>
              <div className="py-card-s mono">{step}</div>
            </div>
            {(running || errored || (pySetup?.log?.length || 0) > 0) && (
              <div className="py-log-tools" role="group" aria-label="Log actions">
                <button
                  className={`py-log-btn ${copied ? 'is-ok' : ''}`}
                  onClick={copyLog}
                  disabled={!(pySetup?.log?.length)}
                  title="Copy full setup log to clipboard"
                >
                  <Icon name={copied ? 'check_c' : 'paperclip'} size={11}/>
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
                <span className="py-log-tools-sep" aria-hidden="true"/>
                <button
                  className={`py-log-btn ${showLog ? 'is-active' : ''}`}
                  onClick={() => setShowLog(v => !v)}
                  title={showLog ? 'Hide log output' : 'View log output'}
                >
                  <Icon name={showLog ? 'eye_off' : 'eye'} size={11}/>
                  <span>{showLog ? 'Hide' : 'View'}</span>
                </button>
              </div>
            )}
          </div>

          {running && !showLog && (
            <div className="py-progress"><div className="py-progress-bar"/></div>
          )}

          {showLog && (
            <div ref={logRef} className="py-log">
              {(pySetup?.log || []).length === 0
                ? <div className="py-log-empty">Waiting for output…</div>
                : (pySetup.log || []).map((l, i) => <div key={i}>{l}</div>)}
            </div>
          )}

          {errored && (
            <div className="py-card-err">
              {pySetup.error}
              <div className="py-card-hint">
                Common causes: no internet, missing system build tools (Windows → install "Desktop development with C++"; Linux → <code>apt install python3-dev build-essential</code>), or no Python 3.9+ on PATH.
              </div>
            </div>
          )}

          {errored && (
            <div className="py-card-actions">
              <button className="mc-btn primary" onClick={() => runSetup && runSetup({ accelerator: accelerator || suggestedAccel })}>
                <Icon name="arrow_right" size={12}/> Retry
              </button>
            </div>
          )}
        </div>

        {}
        {!running && done && pyStatus?.platform !== 'darwin' && pyStatus?.installedAccelerator && (
          <div className="py-card idle">
            <div className="py-card-head">
              <Icon name="cube" size={18}/>
              <div className="py-card-titles">
                <div className="py-card-t">
                  Accelerator: <span className="mono">{pyStatus.installedAccelerator === 'gpu' ? 'GPU (CUDA)' : 'CPU'}</span>
                </div>
                <div className="py-card-s">
                  Switch to {pyStatus.installedAccelerator === 'gpu' ? 'CPU' : 'GPU'} reinstalls the torch wheels from the {pyStatus.installedAccelerator === 'gpu' ? 'CPU' : 'CUDA 12.4'} index (~{pyStatus.installedAccelerator === 'gpu' ? '2 GB' : '5 GB'} download).
                </div>
              </div>
              <button
                className="mc-btn primary"
                onClick={() => {
                  const next = pyStatus.installedAccelerator === 'gpu' ? 'cpu' : 'gpu';
                  runSetup && runSetup({ accelerator: next });
                }}
              >
                <Icon name="arrow_right" size={12}/> Switch to {pyStatus.installedAccelerator === 'gpu' ? 'CPU' : 'GPU'}
              </button>
            </div>
          </div>
        )}

        {}

        <div className="ob-buttons">
          <button className="ob-btn primary" disabled={!canContinue} onClick={onDone}>
            {canContinue ? 'Continue' : running ? 'Please wait…' : 'Setup in progress'} <Icon name="arrow_right" size={13}/>
          </button>
          {!running && !done && (
            <button className="ob-btn ghost" onClick={onDone}>Skip for now</button>
          )}
        </div>
      </div>
    </div>
  );
}

function HFTokenCard() {
  const [masked, setMasked] = useStateOB(null);
  const [input, setInput]   = useStateOB('');
  const [show, setShow]     = useStateOB(false);
  const [busy, setBusy]     = useStateOB(false);
  const [status, setStatus] = useStateOB(null); 

  const refresh = async () => {
    try { setMasked(await window.inferml?.hf.getToken()); } catch {}
  };
  useEffectOB(() => { refresh(); }, []);

  const save = async () => {
    const token = input.trim();
    if (!token) return;
    setBusy(true);
    setStatus(null);
    try {
      const v = await window.inferml?.hf.verifyToken(token);
      if (!v?.ok) { setStatus({ ok: false, error: v?.error || 'verification failed' }); setBusy(false); return; }
      await window.inferml?.hf.setToken(token);
      setStatus({ ok: true, user: v.user });
      setInput('');
      await refresh();
    } catch (e) {
      setStatus({ ok: false, error: String(e?.message || e) });
    }
    setBusy(false);
  };

  const clear = async () => {
    setBusy(true);
    try {
      await window.inferml?.hf.clearToken();
      setMasked(null);
      setStatus(null);
    } catch {}
    setBusy(false);
  };

  return (
    <div className="py-card idle">
      <div className="py-card-head">
        <Icon name={masked ? 'check_c' : 'alert'} size={18}/>
        <div className="py-card-titles">
          <div className="py-card-t">HuggingFace access token</div>
          <div className="py-card-s">
            {masked
              ? <>Set · <span className="mono">{masked}</span>{status?.user ? <> · connected as <b>{status.user.name}</b></> : null}</>
              : 'Optional. Required to download gated models (Llama, Gemma, some Qwen/DeepSeek).'}
          </div>
        </div>
        <a className="hub-link" onClick={() => window.inferml?.app.openExternal('https://huggingface.co/settings/tokens')}>
          Get a token
        </a>
      </div>

      <div className="hf-token-row">
        <input
          className="hf-token-input mono"
          type={show ? 'text' : 'password'}
          value={input}
          placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') save(); }}
        />
        <button className="tb-btn" onClick={() => setShow(v => !v)} title={show ? 'Hide' : 'Show'}>
          <Icon name={show ? 'eye_off' : 'eye'} size={12}/>
        </button>
        <button className="mc-btn primary" disabled={busy || !input.trim()} onClick={save}>
          {busy ? 'Verifying…' : (masked ? 'Replace' : 'Save')}
        </button>
        {masked && (
          <button className="mc-btn ghost" disabled={busy} onClick={clear}>
            <Icon name="x" size={11}/> Clear
          </button>
        )}
      </div>

      {status && !status.ok && (
        <div className="py-card-err" style={{marginTop: 8}}>{status.error}</div>
      )}
      {status && status.ok && (
        <div className="py-card-s" style={{marginTop: 8, color: 'var(--ok)'}}>
          Saved · verified as <b>{status.user?.name}</b>
          {status.user?.orgs?.length ? <> · orgs: {status.user.orgs.join(', ')}</> : null}
        </div>
      )}
    </div>
  );
}

function gbFmt(b) { return (b / (1024 ** 3)).toFixed(1); }
function inferBackend(hw) {
  const vendor = (hw?.gpu?.vendor || '').toLowerCase();
  const model = (hw?.gpu?.model || '').toLowerCase();
  if (vendor.includes('nvidia') || model.includes('rtx') || model.includes('geforce') || model.includes('quadro')) return 'CUDA · fp16';
  if (vendor.includes('amd') || model.includes('radeon')) return 'ROCm · fp16';
  if (vendor.includes('apple') || model.includes('apple')) return 'Metal · fp16';
  if (model.includes('intel')) return 'OpenVINO · fp32';
  return 'CPU · fp32';
}

window.Onboarding = Onboarding;
