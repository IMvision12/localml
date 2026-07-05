function Home({ onNewChat, onOpenModels, hw, chats, provider, version, onOpenChat }) {
  const [featured, setFeatured] = React.useState([]);
  const [featLoading, setFeatLoading] = React.useState(true);

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const picks = [
          'Qwen/Qwen2-VL-7B-Instruct',
          'black-forest-labs/FLUX.1-schnell',
          'facebook/sam2.1-hiera-large',
          'distil-whisper/distil-large-v3',
        ];
        const r = await window.inferml.hf.search(picks[0], null);
        if (cancelled) return;
        if (Array.isArray(r)) {
          const byId = new Map((r || []).map(m => [m.id, m]));
          const results = [];
          for (const id of picks) {
            if (byId.has(id)) { results.push(byId.get(id)); continue; }
            const sub = await window.inferml.hf.search(id.split('/').pop(), null);
            if (Array.isArray(sub)) {
              const match = sub.find(m => m.id === id) || sub[0];
              if (match) results.push(match);
            }
          }
          if (!cancelled && results.length) setFeatured(results.slice(0, 4));
        }
      } catch {}
      if (!cancelled) setFeatLoading(false);
    })();
    return () => { cancelled = true; };
  }, []);

  const quickStart = [
    { id: 'vlm', ic: 'chat', nm: 'Chat with images', dsc: 'Ask questions about screenshots, photos, documents', model: modelFor(provider), color: 'oklch(70% 0.12 230)', sample: "What's in this screenshot?" },
    { id: 'vision', ic: 'target', nm: 'Detect & segment', dsc: 'Describe objects, boxes, masks. Routed to the orchestrator', model: modelFor(provider), color: 'oklch(70% 0.14 155)', sample: 'Segment the forklifts in this photo' },
    { id: 'diff', ic: 'sparkle', nm: 'Generate images', dsc: 'Refine text-to-image prompts with the orchestrator', model: modelFor(provider), color: 'oklch(70% 0.15 320)', sample: 'Orange electric forklift, golden hour' },
    { id: 'audio', ic: 'waveform', nm: 'Transcribe audio', dsc: 'Attach audio, discuss transcript and summaries', model: modelFor(provider), color: 'oklch(70% 0.13 65)', sample: 'Drop a meeting recording' },
  ];

  const greeting = timeGreeting();
  const recents = (chats || []).slice(0, 5);

  return (
    <div className="home">
      <div className="home-bg" />
      <div className="home-inner">
        <div className="home-hello">
          <div className="home-greeting"><span className="dim">{greeting}</span></div>
          <div className="home-ask">What should we run today?</div>
        </div>

        <HomePrompt onSend={(text, workspace) => { onNewChat(workspace); setTimeout(() => prefillComposer(text), 120); }} provider={provider}/>

        <div className="home-section">
          <div className="sec-head">
            <span className="sec-title">Start a workspace</span>
            <span className="sec-rule"/>
            <span className="sec-hint">4 modalities</span>
          </div>
          <div className="home-grid">
            {quickStart.map(q => (
              <button key={q.id} className="qs-card" onClick={() => onNewChat(q.id)}>
                <div className="qs-icon" style={{color: q.color}}><Icon name={q.ic} size={20} stroke={1.4}/></div>
                <div className="qs-body">
                  <div className="qs-nm">{q.nm}</div>
                  <div className="qs-dsc">{q.dsc}</div>
                  <div className="qs-sample">"{q.sample}"</div>
                </div>
                <div className="qs-foot">
                  <span className="qs-model">{q.model}</span>
                  <span className="qs-arrow"><Icon name="arrow_right" size={12}/></span>
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className="home-split">
          <div className="home-section">
            <div className="sec-head">
              <span className="sec-title">System</span>
              <span className="sec-rule"/>
              <span className={`dot ${provider ? 'ok' : 'warn'}`}/>
              <span className="sec-hint">{provider ? `orchestrator · ${provider}` : 'not connected'}</span>
            </div>
            <div className="sys-panel">
              <SysRow ic="gpu" lbl="GPU" val={hw?.gpu?.model || 'Detecting…'} sub={hw?.gpu?.driver || (hw?.gpu?.vendor || '')} pct={gpuPct(hw)} num={gpuNum(hw)}/>
              <SysRow ic="ram" lbl="RAM" val={hw?.mem?.total ? `${gb(hw.mem.total)} GB` : '-'} sub={hw?.os?.distro || ''} pct={hw?.mem?.pct || 0} num={hw?.mem?.total ? `${gb(hw.mem.used)} / ${gb(hw.mem.total)} GB` : '-'}/>
              <SysRow ic="cpu" lbl="CPU" val={hw?.cpu?.brand || 'Detecting…'} sub={hw?.cpu?.cores ? `${hw.cpu.cores}c / ${hw.cpu.threads}t` : ''} pct={hw?.cpu?.load || 0} num={hw?.cpu?.load != null ? `${hw.cpu.load}% load` : '-'} dim/>
              <SysRow ic="folder" lbl="Disk" val={hw?.disk?.mount || '-'} sub={hw?.disk?.total ? `${gb(hw.disk.total)} GB volume` : ''} pct={hw?.disk?.total ? Math.round((hw.disk.used / hw.disk.total) * 100) : 0} num={hw?.disk?.total ? `${gb(hw.disk.used)} / ${gb(hw.disk.total)} GB` : '-'} dim/>
            </div>
          </div>

          <div className="home-section">
            <div className="sec-head">
              <span className="sec-title">Recent</span>
              <span className="sec-rule"/>
              <button className="sec-link">See all →</button>
            </div>
            <div className="recent-list">
              {recents.length === 0 && (
                <div style={{padding:'18px 14px',color:'var(--fg-3)',fontSize:12}}>No chats yet. Click a workspace above to start one.</div>
              )}
              {recents.map(c => (
                <Recent
                  key={c.id}
                  tag={c.tag || 'CHAT'}
                  color={colorForTag(c.tag)}
                  title={c.title || 'Untitled'}
                  meta={c.sub || ''}
                  time={relTime(c.updatedAt)}
                  running={c.running}
                  onClick={() => onOpenChat && onOpenChat(c.id)}
                />
              ))}
            </div>
          </div>
        </div>

        <div className="home-section">
          <div className="sec-head">
            <span className="sec-title">Featured on HuggingFace</span>
            <span className="sec-rule"/>
            <button className="sec-link" onClick={onOpenModels}>Browse library →</button>
          </div>
          <div className="feat-grid">
            {featLoading && Array.from({length:4}).map((_, i) => (
              <div key={i} className="feat-card" style={{opacity: 0.45}}>
                <div className="feat-head"><span className="feat-tag t-vlm">…</span></div>
                <div className="feat-nm">loading…</div>
                <div className="feat-meta"><span>-</span></div>
              </div>
            ))}
            {!featLoading && featured.map(f => (
              <div key={f.id} className="feat-card" onClick={onOpenModels}>
                <div className="feat-head">
                  <span className={`feat-tag t-${taskShort(f.task)}`}>{taskShort(f.task).toUpperCase()}</span>
                  {f.installed && <span className="feat-new">INSTALLED</span>}
                </div>
                <div className="feat-nm">{f.nm}</div>
                <div className="feat-meta">
                  <span>{f.size || '-'}</span>
                  <span className={`dot ${f.hw}`}/>
                  <span>{f.hw === 'ok' ? 'fits on GPU' : 'may swap'}</span>
                </div>
                <button className="feat-btn" onClick={(e) => { e.stopPropagation(); onOpenModels(); }}>
                  <Icon name={f.installed ? 'check' : 'download'} size={11}/> {f.installed ? 'Installed' : 'Install'}
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="home-footer">
          <span>InferML {version || '0.1.0'}</span><span className="sep">·</span>
          <span>electron</span><span className="sep">·</span>
          <span>{hw?.os?.platform || 'unknown'} {hw?.os?.release || ''}</span><span className="sep">·</span>
          <span>{provider ? `orchestrator · ${provider}` : 'no orchestrator'}</span>
          <span style={{flex:1}}/>
          <a className="home-link" onClick={() => window.inferml?.app.openExternal('https://huggingface.co/')}>HF Hub</a><span className="sep">·</span>
          <a className="home-link" onClick={() => window.inferml?.app.openExternal('https://docs.anthropic.com/')}>Docs</a><span className="sep">·</span>
          <a className="home-link" onClick={() => window.inferml?.app.openExternal('https://github.com/')}>Report issue</a>
        </div>
      </div>
    </div>
  );
}

function HomePrompt({ onSend, provider }) {
  const [v, setV] = React.useState('');
  const submit = () => {
    const t = v.trim();
    if (!t) return;
    const ws = autoRoute(t);
    onSend(t, ws);
    setV('');
  };
  return (
    <div className="home-prompt">
      <div className="hp-inner">
        <Icon name="sparkle" size={14} style={{color:'var(--accent)'}}/>
        <input
          placeholder="Ask anything. InferML will route it to the right workspace…"
          value={v}
          onChange={e => setV(e.target.value)}
          onKeyDown={e => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') submit(); else if (e.key === 'Enter') submit(); }}
        />
        <span className="hp-hint">⌘↵</span>
      </div>
      <div className="hp-foot">
        <button className="hp-chip" onClick={() => onSend('', 'vlm')}><Icon name="paperclip" size={11}/> Attach image</button>
        <button className="hp-chip" onClick={() => onSend('', 'audio')}><Icon name="mic" size={11}/> Attach audio</button>
        <button className="hp-chip"><Icon name="zap" size={11}/> Auto-route <span style={{color:'var(--accent)'}}>ON</span></button>
        <span className="hp-hint">router <span style={{color:'var(--fg-1)'}}>{provider || '-'}</span> · {window.defaultModelFor ? window.defaultModelFor(provider) : ''}</span>
      </div>
    </div>
  );
}

function prefillComposer(text) {
  if (!text) return;
  const ta = document.querySelector('.cc-input');
  if (ta) {
    const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
    setter.call(ta, text);
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    ta.focus();
  }
}

function autoRoute(t) {
  const s = t.toLowerCase();
  if (/(image|photo|screenshot|jpeg|jpg|png)/.test(s)) return 'vlm';
  if (/(segment|detect|bounding box|mask)/.test(s)) return 'vision';
  if (/(generate|render|prompt|draw|paint)/.test(s)) return 'diff';
  if (/(transcribe|audio|recording|podcast|whisper)/.test(s)) return 'audio';
  return 'chat';
}

function modelFor(provider) {
  if (provider === 'anthropic') return 'claude-sonnet-4-6';
  if (provider === 'openai') return 'gpt-4o-mini';
  if (provider === 'google') return 'gemini-2.5-flash';
  return '-';
}
function taskShort(task) {
  const t = (task || '').toLowerCase();
  if (t.includes('image-text')) return 'vlm';
  if (t.includes('text-to-image')) return 'diff';
  if (t.includes('segmentation')) return 'seg';
  if (t.includes('detection')) return 'seg';
  if (t.includes('speech')) return 'asr';
  if (t.includes('audio')) return 'asr';
  return 'vlm';
}
function colorForTag(tag) {
  return ({
    VLM: 'oklch(70% 0.12 230)',
    VISION: 'oklch(70% 0.14 155)',
    DIFF: 'oklch(70% 0.15 320)',
    AUDIO: 'oklch(70% 0.13 65)',
    CHAT: 'oklch(70% 0.10 250)',
  })[tag] || 'oklch(70% 0.10 250)';
}
function timeGreeting() {
  const h = new Date().getHours();
  if (h < 12) return 'Good morning';
  if (h < 18) return 'Good afternoon';
  return 'Good evening';
}
function userName() {
  return 'Kiran';
}
function gb(b) { return (b / (1024 ** 3)).toFixed(1); }
function gpuPct(hw) {
  const tot = hw?.gpu?.memTotal || hw?.gpu?.vram || 0;
  if (!tot) return 0;
  return Math.min(100, Math.round(((hw?.gpu?.memUsed || 0) / tot) * 100));
}
function gpuNum(hw) {
  const tot = hw?.gpu?.memTotal || hw?.gpu?.vram || 0;
  if (!tot) return '-';
  return `${gb(hw?.gpu?.memUsed || 0)} / ${gb(tot)} GB`;
}
function relTime(ts) {
  if (!ts) return '';
  const s = Math.round((Date.now() - ts) / 1000);
  if (s < 60) return 'now';
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.round(h / 24);
  if (d < 7) return `${d}d ago`;
  return new Date(ts).toLocaleDateString();
}

function SysRow({ ic, lbl, val, sub, pct, num, dim }) {
  return (
    <div className="sys-row">
      <div className="sys-ic"><Icon name={ic} size={13}/></div>
      <div className="sys-label">{lbl}</div>
      <div className="sys-value">{val} {sub && <span className="sub">· {sub}</span>}</div>
      <div className="sys-bar"><div className="sys-bar-fill" style={{width: pct + '%', background: dim ? 'var(--fg-2)' : 'var(--accent)'}}/></div>
      <div className="sys-num">{num}</div>
    </div>
  );
}

function Recent({ tag, color, title, meta, time, running, onClick }) {
  return (
    <div className="recent-row" onClick={onClick}>
      <div className="recent-tag" style={{color, borderColor: 'color-mix(in oklab, ' + color + ' 40%, transparent)'}}>{tag}</div>
      <div className="recent-body">
        <div className="recent-title">{title}{running && <span className="recent-live"><span className="d"/>running</span>}</div>
        <div className="recent-meta">{meta}</div>
      </div>
      <div className="recent-time">{time}</div>
      <Icon name="chevron" size={13} style={{color:'var(--fg-3)', transform:'rotate(-90deg)'}}/>
    </div>
  );
}
window.Home = Home;
