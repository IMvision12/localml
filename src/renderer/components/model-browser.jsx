const { useState: useStateMB, useEffect: useEffectMB, useMemo: useMemoMB, useRef: useRefMB } = React;

const HUB_TASKS = [
  { id: 'all',          nm: 'All',          task: null,                           ic: 'cube'     },
  { id: 'vlm',          nm: 'VLM',          task: 'image-text-to-text',           ic: 'chat'     },
  { id: 'text',         nm: 'Text',         task: 'text-generation',              ic: 'chat'     },
  { id: 'segmentation', nm: 'Segmentation', task: 'image-segmentation',           ic: 'eye'      },
  { id: 'sam',          nm: 'SAM',          task: 'mask-generation',              ic: 'sparkle'  },
  { id: 'detection',    nm: 'Detection',    task: 'object-detection',             ic: 'target'   },
  { id: 'classify',     nm: 'Classify',     task: 'image-classification',         ic: 'eye'      },
  { id: 'diffusion',    nm: 'Diffusion',    task: 'text-to-image',                ic: 'sparkle'  },

  { id: 'depth',        nm: 'Depth',        task: 'depth-estimation',             ic: 'layers'   },
  { id: 'docs',         nm: 'Docs / OCR',   task: 'document-question-answering',  ic: 'file'     },
  { id: 'asr',          nm: 'ASR',          task: 'automatic-speech-recognition', ic: 'waveform' },
  { id: 'tts',          nm: 'TTS',          task: 'text-to-speech',               ic: 'waveform' },
];

const MODEL_FAMILIES = [
  { nm: 'Llama',     q: 'llama' },
  { nm: 'Qwen',      q: 'qwen' },
  { nm: 'Mistral',   q: 'mistral' },
  { nm: 'Gemma',     q: 'gemma' },
  { nm: 'Phi',       q: 'phi' },
  { nm: 'DeepSeek',  q: 'deepseek' },
  { nm: 'Florence',  q: 'florence' },
  { nm: 'LLaVA',     q: 'llava' },

  { nm: 'Moondream', q: 'moondream' },
  { nm: 'PaliGemma', q: 'paligemma' },
  { nm: 'SmolVLM',   q: 'smolvlm' },
  { nm: 'Whisper',   q: 'whisper' },
  { nm: 'Parakeet',  q: 'parakeet' },
  { nm: 'DETR',      q: 'detr' },
  { nm: 'YOLOS',     q: 'yolos' },
  { nm: 'RT-DETR',   q: 'rt-detr' },
  { nm: 'SAM',       q: 'sam-vit' },
  { nm: 'BLIP',      q: 'blip' },
  { nm: 'TrOCR',     q: 'trocr' },
  { nm: 'ViT',       q: 'vit' },
  { nm: 'CLIP',      q: 'clip' },
  { nm: 'SigLIP',    q: 'siglip' },
  { nm: 'SD',        q: 'stable-diffusion' },
  { nm: 'FLUX',      q: 'flux' },
  { nm: 'SpeechT5',  q: 'speecht5' },
  { nm: 'Bark',      q: 'bark' },
];

const SUGGEST_TASKS = [
  'image-text-to-text',
  'text-generation',
  'automatic-speech-recognition',
  'object-detection',
  'mask-generation',
  'image-classification',
];

const SUGGEST_MAX_BYTES = 5 * 1024 * 1024 * 1024; 

function _parseSize(s) {
  if (!s || typeof s !== 'string') return NaN;
  const m = s.match(/^([\d.]+)\s*([KMGT]?B)$/i);
  if (!m) return NaN;
  const n = parseFloat(m[1]);
  const unit = m[2].toUpperCase();
  const mul = { B: 1, KB: 1024, MB: 1024 ** 2, GB: 1024 ** 3, TB: 1024 ** 4 }[unit] || 1;
  return n * mul;
}

function _parseParams(s) {
  if (!s || typeof s !== 'string') return NaN;
  const m = s.match(/^([\d.]+)\s*([KMB])$/i);
  if (!m) return NaN;
  const n = parseFloat(m[1]);
  return n * ({ K: 1e3, M: 1e6, B: 1e9 }[m[2].toUpperCase()] || 1);
}

function _filterBytes(m) {
  const real = _parseSize(m.size);
  if (Number.isFinite(real)) return real;
  const params = _parseParams(m.params);
  if (Number.isFinite(params)) return params * 2;
  return NaN;
}

function _formatLandingDate(d = new Date()) {
  return d.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' }).toUpperCase();
}

const SUPPORTED_TASKS = new Set([

  'image-text-to-text',
  'text-generation',
  'translation',
  'summarization',
  'text2text-generation',
  'image-segmentation',
  'mask-generation',
  'object-detection',
  'zero-shot-object-detection',
  'image-classification',
  'zero-shot-image-classification',
  'image-to-text',
  'depth-estimation',
  'document-question-answering',
  'automatic-speech-recognition',
  'text-to-speech',

  'text-to-image',
  'image-to-image',


]);

function ModelHub({ hw, onOpenModel, onOpenSettings, defaultInstalled = false, resetSignal = 0 }) {
  const [tab, setTab] = useStateMB('all');
  const [query, setQuery] = useStateMB('');

  const [debouncedQuery, setDebouncedQuery] = useStateMB('');
  const [showInstalled, setShowInstalled] = useStateMB(defaultInstalled);

  useEffectMB(() => { setShowInstalled(defaultInstalled); }, [defaultInstalled]);



  useEffectMB(() => {
    if (resetSignal === 0) return;
    setTab('all');
    setQuery('');
    setDebouncedQuery('');
  }, [resetSignal]);



  const [suggestedPool, setSuggestedPool] = useStateMB({}); 
  const [suggestedSizes, setSuggestedSizes] = useStateMB({}); 
  const [suggestedLoading, setSuggestedLoading] = useStateMB(true);
  const [results, setResults] = useStateMB([]);
  const [loading, setLoading] = useStateMB(false);
  const [err, setErr] = useStateMB(null);
  const [installed, setInstalled] = useStateMB({});
  const [downloads, setDownloads] = useStateMB({});
  const [sizeMap, setSizeMap] = useStateMB({}); 
  const dlTimers = useRefMB({});

  const refreshInstalled = async () => {
    try { setInstalled((await window.inferml.hf.installed()) || {}); } catch {}
  };

  useEffectMB(() => { refreshInstalled(); }, []);



  useEffectMB(() => {
    const off = window.inferml?.hf?.onInstallsChanged?.(() => refreshInstalled());
    return () => { try { off && off(); } catch {} };
  }, []);



  useEffectMB(() => {
    let cancelled = false;
    (async () => {
      setSuggestedLoading(true);
      const lists = await Promise.allSettled(
        SUGGEST_TASKS.map(t => window.inferml.hf.search('', t))
      );
      if (cancelled) return;
      const pool = {};
      for (let i = 0; i < lists.length; i++) {
        const r = lists[i];
        if (r.status === 'fulfilled' && Array.isArray(r.value)) {
          pool[SUGGEST_TASKS[i]] = r.value;
        }
      }
      if (cancelled) return;
      setSuggestedPool(pool);
      setSuggestedLoading(false);
    })();
    return () => { cancelled = true; };
  }, []);


  const suggestedList = useMemoMB(() => {
    const picks = [];
    const seen = new Set();
    for (const task of SUGGEST_TASKS) {
      const list = suggestedPool[task] || [];
      const pick = list.find(m => {
        if (!m.id || seen.has(m.id)) return false;
        if (installed[m.id]) return false;


        const bytes = _filterBytes(m);
        if (Number.isFinite(bytes) && bytes > SUGGEST_MAX_BYTES) return false;
        return true;
      });
      if (pick) {

        const finalSize = suggestedSizes[pick.id] || pick.size;
        picks.push({ ...pick, size: finalSize });
        seen.add(pick.id);
      }
      if (picks.length >= 5) break;
    }
    return picks;
  }, [suggestedPool, suggestedSizes, installed]);



  useEffectMB(() => {
    let cancelled = false;
    (async () => {
      const needSize = suggestedList.filter(p => !p.size || p.size === '-' || p.size === '-');
      if (needSize.length === 0) return;
      const CONCURRENCY = 3;
      let cursor = 0;
      const worker = async () => {
        while (!cancelled && cursor < needSize.length) {
          const m = needSize[cursor++];
          if (suggestedSizes[m.id]) continue; 
          try {
            const info = await window.inferml?.hf.modelInfo(m.id);
            if (cancelled || !info?.size) continue;
            setSuggestedSizes(prev => ({ ...prev, [m.id]: info.size }));
          } catch {  }
        }
      };
      await Promise.all(Array.from({ length: CONCURRENCY }, worker));
    })();
    return () => { cancelled = true; };
  }, [suggestedList]);


  useEffectMB(() => {
    const t = setTimeout(() => setDebouncedQuery(query), 300);
    return () => clearTimeout(t);
  }, [query]);

  useEffectMB(() => {
    const off = window.inferml?.tasks.onDownloadProgress((evt) => {
      if (!evt || !evt.modelId) return;
      const { modelId, pct, done, total, final } = evt;
      setDownloads(d => {

        if (!d[modelId]) return d;
        return {
          ...d,
          [modelId]: {
            ...d[modelId],
            pct: typeof pct === 'number' ? pct : d[modelId].pct,
            done: typeof done === 'number' ? done : d[modelId].done,
            total: typeof total === 'number' ? total : d[modelId].total,
            final: !!final || !!d[modelId].final,
          },
        };
      });
    });
    return () => { if (off) off(); };
  }, []);



  const hasActiveFilter = (
    showInstalled
    || debouncedQuery.trim().length > 0
    || (tab && tab !== 'all')
  );

  useEffectMB(() => {
    if (!hasActiveFilter) {
      setResults([]);
      setLoading(false);
      setErr(null);
      return;
    }
    let cancelled = false;
    async function run() {
      setLoading(true);
      setErr(null);
      try {
        if (showInstalled) {
          const inst = (await window.inferml.hf.installed()) || {};
          const ids = Object.keys(inst);
          const collected = [];
          for (const id of ids) {
            const sub = await window.inferml.hf.search(id.split('/').pop(), null);
            if (Array.isArray(sub)) {
              const match = sub.find(m => m.id === id);
              if (match) collected.push({ ...match, installed: true });
              else collected.push({ id, nm: id.split('/').pop(), path: id, task: inst[id]?.task, size: inst[id]?.size, installed: true });
            }
          }
          if (!cancelled) setResults(collected);
        } else {
          const sel = HUB_TASKS.find(t => t.id === tab);
          const r = await window.inferml.hf.search(debouncedQuery.trim(), sel?.task || null);
          if (Array.isArray(r)) {
            if (!cancelled) setResults(r);
          } else {


            if (!cancelled) setErr(r?.error || 'search failed');
          }
        }
      } catch (e) {
        if (!cancelled) setErr(String(e.message || e));
      }
      if (!cancelled) setLoading(false);
    }
    run();
    return () => { cancelled = true; };
  }, [tab, debouncedQuery, showInstalled]);

  const allModels = useMemoMB(() =>
    results.map(m => {
      const entry = sizeMap[m.id];
      const fetched = entry && entry.fetched;
      const originalSize = m.size && m.size !== '-' ? m.size : null;
      const size = (entry?.size) || originalSize;
      return {
        ...m,
        installed: !!installed[m.id],
        size,
        sizeFetched: !!fetched || !!originalSize,
        bytes: entry?.bytes || 0,
        hw: entry?.hw || m.hw,
      };
    }),
    [results, installed, sizeMap]);



  const models = useMemoMB(
    () => allModels.filter(m => m.installed || SUPPORTED_TASKS.has(m.task)),
    [allModels]
  );
  const hiddenCount = allModels.length - models.length;


  useEffectMB(() => {
    let cancelled = false;
    (async () => {
      const ids = results
        .filter(m => SUPPORTED_TASKS.has(m.task))
        .filter(m => !m.size || m.size === '-')
        .map(m => m.id)
        .filter(id => !(id in sizeMap));
      if (!ids.length) return;

      const CONCURRENCY = 4;
      let cursor = 0;
      const worker = async () => {
        while (!cancelled && cursor < ids.length) {
          const id = ids[cursor++];
          try {
            const info = await window.inferml?.hf.modelInfo(id);
            if (cancelled) return;
            setSizeMap(prev => ({
              ...prev,
              [id]: { fetched: true, size: info?.size || null, bytes: info?.bytes || 0, hw: info?.hw },
            }));
          } catch {
            if (!cancelled) setSizeMap(prev => ({ ...prev, [id]: { fetched: true, size: null, bytes: 0 } }));
          }
        }
      };
      await Promise.all(Array.from({ length: CONCURRENCY }, worker));
    })();
    return () => { cancelled = true; };
  }, [results]);

  const startDownload = async (m) => {
    if (downloads[m.id]) return;
    setDownloads(d => ({ ...d, [m.id]: { status: 'downloading', size: m.size } }));
    try {
      const res = await window.inferml.tasks.download(m.id);
      if (!res?.ok) {
        const msg = res?.error || 'download failed';

        setDownloads(d => d[m.id]
          ? { ...d, [m.id]: { status: 'error', error: msg } }
          : d);
        return;
      }
      await window.inferml.hf.markInstalled(m.id, { task: m.task, size: m.size, nm: m.nm, localPath: res.info?.path });
      await refreshInstalled();
      setDownloads(d => { const rest = { ...d }; delete rest[m.id]; return rest; });
    } catch (e) {
      setDownloads(d => d[m.id]
        ? { ...d, [m.id]: { status: 'error', error: String(e?.message || e) } }
        : d);
    }
  };
  const cancelDownload = async (id) => {

    setDownloads(d => { const rest = { ...d }; delete rest[id]; return rest; });



    try { await window.inferml?.tasks.cancelDownload(id); } catch {}
  };
  const uninstall = async (id) => {
    await window.inferml.hf.uninstall(id);
    refreshInstalled();
  };

  const installedCount = Object.keys(installed).length;
  const activeDownloads = Object.keys(downloads).length;



  const installedList = Object.entries(installed)
    .map(([id, m]) => ({
      id,
      nm: m.nm || id.split('/').pop(),
      owner: id.split('/')[0],
      task: m.task || '',
      size: m.size || '',
      installedAt: m.installedAt || 0,
    }))
    .sort((a, b) => b.installedAt - a.installedAt)
    .slice(0, 5);

  return (
    <div className={`hub ${!hasActiveFilter ? 'hub-idle' : ''}`}>
      {!hasActiveFilter ? (
        <div className="hub-landing">
          <div className="hub-landing-eyebrow">{_formatLandingDate()}</div>
          <h1 className="hub-landing-title">What would you like to run today?</h1>
          <p className="hub-landing-sub">Pick from your library, browse the Hub, or paste a model id.</p>

          <div className="hub-search-row hub-landing-search">
            <div className="hub-search">
              <Icon name="search" size={14} style={{color:'var(--fg-3)'}}/>
              <input
                placeholder="Search a task, family, or paste an HF id…"
                value={query}
                onChange={e => setQuery(e.target.value)}
                autoFocus
              />
              {query && <button className="hub-search-clear" onClick={() => setQuery('')}><Icon name="x" size={11}/></button>}
            </div>
          </div>

          <div className="hub-tag-list hub-landing-tags">
            {HUB_TASKS.map(t => (
              <button
                key={t.id}
                className={`hub-tag ${tab === t.id ? 'active' : ''}`}
                onClick={() => setTab(tab === t.id ? 'all' : t.id)}
              >
                <Icon name={t.ic} size={11}/> {t.nm}
              </button>
            ))}
          </div>

          <div className="hub-landing-cols">
            <div className="hub-landing-col">
              <div className="hub-landing-col-label">
                Installed{installedList.length > 0 && <span className="hub-landing-col-n"> · {installedList.length}</span>}
              </div>
              {installedList.length === 0 ? (
                <div className="hub-landing-empty">Nothing yet. Pick something from the Suggested list →</div>
              ) : (
                <ul className="hub-landing-list">
                  {installedList.map(m => (
                    <li key={m.id} className="hub-landing-row" onClick={() => onOpenModel && onOpenModel(m.id)}>
                      <div className="hub-landing-row-l">
                        <div className="hub-landing-row-nm">{m.nm}</div>
                        <div className="hub-landing-row-meta">{m.owner}{m.task && <> · <span className="mono">{m.task}</span></>}</div>
                      </div>
                      <div className="hub-landing-row-r mono">{m.size}</div>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="hub-landing-col">
              <div className="hub-landing-col-label">Suggested for you</div>
              {suggestedLoading && suggestedList.length === 0 ? (
                <div className="hub-landing-empty">Loading suggestions from HuggingFace…</div>
              ) : suggestedList.length === 0 ? (
                <div className="hub-landing-empty">No suggestions right now. Browse the Hub above.</div>
              ) : (
                <ul className="hub-landing-list">
                  {suggestedList.map(m => {
                    const owner = (m.id || '').split('/')[0];
                    const nm = m.nm || (m.id || '').split('/').pop();
                    const dl = downloads[m.id];
                    const error = dl?.status === 'error';
                    const gated = error && _looksGated(dl?.error);
                    const hasPct = dl && typeof dl.pct === 'number' && dl.total > 0;
                    const pct = hasPct ? Math.max(0, Math.min(100, dl.pct)) : null;
                    const onRowClick = () => {
                      if (dl && !error) return;          
                      if (error) { cancelDownload(m.id); startDownload(m); return; }
                      startDownload(m);
                    };
                    return (
                      <li
                        key={m.id}
                        className={`hub-landing-row ${dl ? 'is-downloading' : ''}`}
                        onClick={onRowClick}
                      >
                        <div className="hub-landing-row-l">
                          <div className="hub-landing-row-nm">{nm}</div>
                          <div className="hub-landing-row-meta">{owner}{m.task && <> · <span className="mono">{m.task}</span></>}</div>
                          {dl && !error && (
                            <div className={`dl-bar ${hasPct ? '' : 'indeterminate'} hub-landing-row-bar`}>
                              <div className="dl-bar-fill" style={hasPct ? { width: `${pct}%` } : undefined}/>
                            </div>
                          )}
                          {error && !gated && (
                            <div className="hub-landing-row-err">{dl?.error?.slice(0, 80) || 'failed'}</div>
                          )}
                          {gated && (
                            <div className="hub-landing-row-err">Gated repo. Set your HF token in Settings, then retry.</div>
                          )}
                        </div>
                        <div className="hub-landing-row-r mono">
                          {dl && !error ? (
                            <>
                              <span>{hasPct ? `${pct.toFixed(0)}%` : 'starting…'}</span>
                              <button
                                className="hub-landing-row-x"
                                onClick={(e) => { e.stopPropagation(); cancelDownload(m.id); }}
                                title="Cancel download"
                              >
                                <Icon name="x" size={11}/>
                              </button>
                            </>
                          ) : error ? (
                            <button
                              className="hub-landing-row-x"
                              onClick={(e) => { e.stopPropagation(); cancelDownload(m.id); }}
                              title="Dismiss"
                            >
                              <Icon name="x" size={11}/>
                            </button>
                          ) : (
                            <><Icon name="download" size={11}/> {m.size || '-'}</>
                          )}
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="hub-head">
          <div className="hub-title-row">
            <div className="hub-title">Model Hub</div>
            <div className="hub-sub">Browse HuggingFace · download and run locally</div>
            <div style={{flex:1}}/>
            <div className="hub-stats">
              <span><span className="n">{installedCount}</span> installed</span>
              {activeDownloads > 0 && <span><span className="n">{activeDownloads}</span> downloading</span>}
            </div>
          </div>
          <div className="hub-search-row">
            <div className="hub-search">
              <Icon name="search" size={13} style={{color:'var(--fg-3)'}}/>
              <input
                placeholder={showInstalled ? 'Filter installed…' : 'Search HuggingFace Hub…'}
                value={query}
                onChange={e => setQuery(e.target.value)}
              />
              {query && <button className="hub-search-clear" onClick={() => setQuery('')}><Icon name="x" size={11}/></button>}
            </div>
          </div>
          {}
          {!showInstalled && !query.trim() && (
            <div className="hub-tag-list" style={{justifyContent: 'flex-start'}}>
              {HUB_TASKS.map(t => (
                <button
                  key={t.id}
                  className={`hub-tag ${tab === t.id ? 'active' : ''}`}
                  onClick={() => setTab(tab === t.id ? 'all' : t.id)}
                >
                  <Icon name={t.ic} size={11}/> {t.nm}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="hub-body">
        {}
        {(() => {
          const inlineIds = hasActiveFilter ? new Set() : new Set((suggestedList || []).map(s => s.id));
          const stripEntries = Object.entries(downloads).filter(([id]) => !inlineIds.has(id));
          if (stripEntries.length === 0) return null;
          return (
            <div className="hub-section">
              <div className="sec-title-row"><span>Downloading</span><span className="line"/></div>
              {stripEntries.map(([id, st]) => {
                const m = models.find(x => x.id === id) || { nm: id.split('/').pop(), path: id };
                const error = st.status === 'error';
                const gated = error && _looksGated(st.error);
                const hasPct = typeof st.pct === 'number' && st.total > 0;
                const pct = hasPct ? Math.max(0, Math.min(100, st.pct)) : null;
                return (
                  <div key={id} className="dl-card">
                    <div className="dl-head">
                      <span className="nm">{m.nm}</span>
                      <span className="path">{m.path || id}</span>
                      <span className="rate">
                        {error ? 'failed' : (hasPct ? `${fmtBytes(st.done)} / ${fmtBytes(st.total)}` : 'starting…')}
                      </span>
                      <button className="tb-btn" onClick={() => cancelDownload(id)} style={{marginLeft: 8}}><Icon name="x" size={11}/> Dismiss</button>
                    </div>
                    {!error && (
                      <div className={`dl-bar ${hasPct ? '' : 'indeterminate'}`}>
                        <div className="dl-bar-fill" style={hasPct ? { width: `${pct}%` } : undefined}/>
                      </div>
                    )}
                    {!error && hasPct && (
                      <div className="dl-foot"><span className="pct">{pct.toFixed(1)}%</span><span>huggingface_hub</span></div>
                    )}
                    {error && !gated && <div className="dl-foot" style={{color:'var(--bad)'}}>{st.error}</div>}
                    {gated && (
                      <GatedTokenPrompt
                        modelId={id}
                        onOpenSettings={onOpenSettings}
                        onRetry={() => { cancelDownload(id); startDownload(m); }}
                      />
                    )}
                  </div>
                );
              })}
            </div>
          );
        })()}

        {hasActiveFilter && loading && models.length === 0 && <div className="hub-empty">Searching HuggingFace…</div>}
        {hasActiveFilter && err && models.length === 0 && <div className="hub-empty bad">Error: {err}</div>}
        {hasActiveFilter && !loading && !err && models.length === 0 && (
          <div className="hub-empty">
            {showInstalled
              ? 'No models installed yet. Browse a category and click Download.'
              : (hiddenCount > 0
                  ? `No supported models in these results (${hiddenCount} filtered out: task not supported yet).`
                  : (() => {
                      const q = debouncedQuery.trim();
                      const sel = HUB_TASKS.find(t => t.id === tab);
                      const tabNm = sel && sel.id !== 'all' ? sel.nm : null;
                      if (q && tabNm) return `No ${tabNm} models match "${q}". Clear the search or switch to the All tab.`;
                      if (q)          return `No models match "${q}".`;
                      if (tabNm)      return `No ${tabNm} models found.`;
                      return 'No results.';
                    })())}
          </div>
        )}
        {err && models.length > 0 && (
          <div className="hub-empty bad" style={{padding: '8px 12px', margin: '0 0 10px'}}>
            Couldn't refresh: {err}. Showing previous results.
          </div>
        )}
        {!loading && models.length > 0 && (
          <div className="hub-grid">
            {models.map(m => (
              <ModelCard
                key={m.id}
                m={m}
                onInstall={() => startDownload(m)}
                onUninstall={() => uninstall(m.id)}
                onOpen={() => onOpenModel && onOpenModel(m.id)}
                downloading={!!downloads[m.id]}
                dlPct={downloads[m.id]?.pct}
                dlDone={downloads[m.id]?.done}
                dlTotal={downloads[m.id]?.total}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function randRate() { return (Math.random() * 12 + 6).toFixed(1); }

function fmtBytes(b) {
  if (!b || b <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0, n = b;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

function ModelCard({ m, onInstall, onUninstall, onOpen, downloading, dlPct, dlDone, dlTotal }) {
  const sizeDisplay = m.size && m.size !== '-' ? m.size : null;
  const sizeLabel = sizeDisplay
    ? sizeDisplay
    : (m.sizeFetched ? <span className="mc-size-dim">size unknown</span> : <span className="mc-size-dim">fetching…</span>);
  const handleMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    e.currentTarget.style.setProperty('--mx', `${e.clientX - r.left}px`);
    e.currentTarget.style.setProperty('--my', `${e.clientY - r.top}px`);
  };
  return (
    <div className={`model-card ${m.installed ? 'installed' : ''}`} onMouseMove={handleMove}>
      <div className="mc-body">
        <div className="mc-name">
          <span className="nm">{m.nm}</span>
          {m.installed && <span className="mc-chip ok"><Icon name="check" size={10}/> Installed</span>}
        </div>
        <div className="mc-path">{m.path || m.id}</div>
        {m.desc && <div className="mc-desc">{m.desc}</div>}
        <div className="mc-meta">
          {m.task && <span className="mc-tag task">{m.task}</span>}
          {m.params && <span className="mc-tag"><span className="k">params</span>{m.params}</span>}
          {m.dl && <span className="mc-tag"><span className="k">dl</span>{m.dl}/mo</span>}
          {m.license && <span className="mc-tag"><span className="k">lic</span>{m.license}</span>}
          {m.hw === 'ok' && <span className="mc-tag ok">● fits</span>}
          {m.hw === 'warn' && <span className="mc-tag warn">⚠ large</span>}
        </div>
      </div>
      <div className="mc-footer">
        <div className="mc-size">
          <Icon name="download" size={12}/>
          <span className="mc-size-val">{sizeLabel}</span>
        </div>
        <div className="mc-actions">
          {m.installed ? (
            <>
              <button className="mc-btn primary" onClick={onOpen}><Icon name="arrow_right" size={12}/> Open</button>
              <button className="mc-btn ghost" onClick={onUninstall}><Icon name="x" size={11}/> Remove</button>
            </>
          ) : downloading ? (
            (() => {
              const hasPct = typeof dlPct === 'number' && dlTotal > 0;
              const pct = hasPct ? Math.max(0, Math.min(100, dlPct)) : null;
              return (
                <div
                  className={`mc-btn downloading ${hasPct ? '' : 'indeterminate'}`}
                  style={hasPct ? { '--pct': `${pct}%` } : undefined}
                  role="progressbar"
                  aria-valuenow={hasPct ? pct : undefined}
                  aria-valuemin={0}
                  aria-valuemax={100}
                >
                  <span className="mc-dl-fill"/>
                  <span className="mc-dl-text">
                    <Icon name="download" size={12}/>
                    {hasPct
                      ? <>{pct.toFixed(0)}% · {fmtBytes(dlDone)} / {fmtBytes(dlTotal)}</>
                      : <>Downloading…</>}
                  </span>
                </div>
              );
            })()
          ) : (
            <>
              <button className="mc-btn primary" onClick={onInstall}>
                <Icon name="download" size={12}/> Download{sizeDisplay ? ` · ${sizeDisplay}` : ''}
              </button>
              <button className="mc-btn ghost" onClick={() => window.inferml?.app.openExternal(`https://huggingface.co/${m.id || m.path}`)}>View</button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// Heuristic: does an HF download error look like a gated / unauthorized repo?
function _looksGated(msg) {
  const s = String(msg || '').toLowerCase();
  return s.includes('401') || s.includes('403')
      || s.includes('gated') || s.includes('unauthorized')
      || s.includes('access to this model')
      || s.includes('login') || s.includes('token')
      || s.includes('huggingface-cli login');
}

function GatedTokenPrompt({ modelId, onOpenSettings, onRetry }) {
  return (
    <div className="gated-prompt">
      <div className="gated-prompt-msg">
        <Icon name="alert" size={12}/>
        <span>This model is gated. Set your HuggingFace token in Settings, then retry.</span>
      </div>
      <div className="gated-prompt-actions">
        <button className="mc-btn primary" onClick={() => onOpenSettings && onOpenSettings('hf')}>
          <Icon name="settings" size={11}/> Open Settings
        </button>
        <button className="tb-btn" onClick={() => onRetry && onRetry()} title="Re-attempt the download once your token is set">
          <Icon name="arrow_right" size={11}/> Retry
        </button>
        <span style={{flex: 1}}/>
        <a
          className="hub-link"
          onClick={() => window.inferml?.app.openExternal(`https://huggingface.co/${modelId}`)}
        >
          Request access on Hugging Face
        </a>
      </div>
    </div>
  );
}

window.ModelHub = ModelHub;
