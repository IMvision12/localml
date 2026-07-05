const { useState: useStateCH, useEffect: useEffectCH, useRef: useRefCH } = React;

function titleFromFirstMessage(text, atts) {
  const cleaned = (text || '').trim().replace(/\s+/g, ' ');
  if (cleaned) {
    if (cleaned.length <= 60) return cleaned;
    const cut = cleaned.slice(0, 60);
    const lastSpace = cut.lastIndexOf(' ');
    return (lastSpace > 30 ? cut.slice(0, lastSpace) : cut).trimEnd() + '…';
  }
  const first = atts && atts[0];
  if (first?.name) {
    return first.name.replace(/\.[^.]+$/, '').replace(/[_\-]+/g, ' ');
  }
  return 'New chat';
}

const JANUS_MODES = [
  { value: 'understand', label: 'Understand', desc: 'Image in. text out' },
  { value: 'generate',   label: 'Generate',   desc: 'Text in. image out' },
];

function JanusModeBar({ value, onChange }) {
  return (
    <div className="whisper-bar">
      <div className="whisper-bar-head">
        <Icon name="sparkle" size={11}/>
        <span className="whisper-bar-k">Janus mode</span>
      </div>
      <div className="whisper-bar-toggle" role="radiogroup" aria-label="Janus mode">
        {JANUS_MODES.map(m => (
          <button
            key={m.value}
            type="button"
            role="radio"
            aria-checked={value === m.value}
            className={`whisper-pill ${value === m.value ? 'active' : ''}`}
            onClick={() => onChange(m.value)}
            title={m.desc}
          >
            <span className="whisper-pill-l">{m.label}</span>
            <span className="whisper-pill-d">{m.desc}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function ChatWorkspace({ sessionId, modelId, modelMeta, onSaved }) {
  const [chat, setChat] = useStateCH(null);
  const [input, setInput] = useStateCH('');
  const [atts, setAtts] = useStateCH([]);
  const [error, setError] = useStateCH(null);
  const [sending, setSending] = useStateCH(false);


  const [stopping, setStopping] = useStateCH(false);
  const stoppedByUserRef = useRefCH(false);
  const stop = async () => {
    if (!sending || stopping) return;
    setStopping(true);
    stoppedByUserRef.current = true;
    try { await window.inferml?.tasks?.stop?.(); } catch {}
  };
  const [janusMode, setJanusMode] = useStateCH('understand');
  const scrollRef = useRefCH(null);
  const inputRef = useRefCH(null);

  const isVLM = (modelMeta?.task === 'image-text-to-text');
  const isJanus = /janus/i.test(modelId || '');

  useEffectCH(() => {
    let cancelled = false;
    (async () => {
      if (!sessionId) return;
      const c = await window.inferml.chats.get(sessionId);
      if (cancelled || !c) return;

      const healed = (c.messages || []).map(m =>
        (m.streaming && !m.text)
          ? { ...m, streaming: false, text: '[interrupted]', error: true }
          : m
      );
      const changed = healed.some((m, i) => m !== (c.messages || [])[i]);
      const loaded = { ...c, messages: healed };
      if (changed) window.inferml.chats.save(loaded).catch(() => {});
      setChat(loaded);
    })();
    return () => { cancelled = true; };
  }, [sessionId]);

  useEffectCH(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [chat?.messages?.length]);

  useEffectCH(() => {
    if (inputRef.current) inputRef.current.focus();
  }, [sessionId]);

  if (!chat) return <div className="chat-view"><div className="chat-empty">Loading…</div></div>;

  const send = async () => {
    const text = input.trim();
    const isGenerate = isJanus && janusMode === 'generate';
    if ((!text && !atts.length) || sending) return;


    if (isGenerate && !text) {
      setError('Generate mode needs a text prompt describing the image to create.');
      return;
    }
    if (!isGenerate) {
      const hasPriorImg = chat.messages.some(m => (m.attachments || []).some(a => a.kind === 'image'));
      if (isVLM && !atts.some(a => a.kind === 'image') && !hasPriorImg) {
        setError('This is a vision-language model. Attach an image before sending the first message.');
        return;
      }
    }
    setError(null);
    setSending(true);

    const userMsg = {
      id: 'u-' + Math.random().toString(36).slice(2),
      role: 'user',
      text,
      attachments: atts,
      ts: Date.now(),
    };
    const asstId = 'a-' + Math.random().toString(36).slice(2);
    const asstMsg = {
      id: asstId,
      role: 'assistant',
      text: '',
      ts: Date.now(),
      streaming: true,
      model: modelId,
    };
    const nextMsgs = [...chat.messages, userMsg, asstMsg];
    const baseTitle = chat.title && chat.title !== 'New chat'
      ? chat.title
      : titleFromFirstMessage(text, atts);
    const nextChat = {
      ...chat,
      title: baseTitle,
      modelId,
      task: modelMeta?.task,
      sub: `${modelId.split('/').pop()} · ${nextMsgs.length} msgs`,
      messages: nextMsgs,
    };
    setChat(nextChat);
    setInput('');
    setAtts([]);
    try { await window.inferml.chats.save(nextChat); } catch {}
    onSaved && onSaved(nextChat);




    let imageAtt = isGenerate ? null : (atts || []).find(a => a.kind === 'image');
    if (!imageAtt && isVLM && !isGenerate) {
      for (let i = chat.messages.length - 1; i >= 0; i--) {
        const m = chat.messages[i];
        const prior = (m.attachments || []).find(a => a.kind === 'image');
        if (prior) { imageAtt = prior; break; }
      }
    }
    const task = modelMeta?.task || (isVLM ? 'image-text-to-text' : 'text-generation');
    const payload = {
      task,
      modelId,
      input: {
        text,
        ...(imageAtt ? { dataUrl: imageAtt.dataUrl } : {}),
      },
      ...(isJanus ? { params: { janus_mode: janusMode } } : {}),
    };

    const res = await window.inferml.tasks.run(payload).catch(e => ({ ok: false, error: String(e?.message || e) }));

    const patchAssistant = (patch) => {
      setChat(prev => {
        if (!prev) return prev;
        const msgs = prev.messages.map(m => m.id === asstId ? { ...m, ...patch, streaming: false } : m);
        const done = { ...prev, messages: msgs };
        window.inferml.chats.save(done).catch(() => {});
        onSaved && onSaved(done);
        return done;
      });
    };

    if (res?.ok) {
      const out = res.output || {};
      if (out.kind === 'image' && out.dataUrl) {
        patchAssistant({ text: '', image: out.dataUrl });
      } else if (out.kind === 'text') {
        patchAssistant({ text: out.text || '(empty reply)' });
      } else {
        patchAssistant({ text: JSON.stringify(out) });
      }
    } else if (stoppedByUserRef.current) {
      patchAssistant({ text: 'Stopped by user.', cancelled: true });
    } else {
      const errMsg = res?.error || 'inference failed';
      patchAssistant({ text: errMsg, error: true });
      setError(errMsg);
    }
    stoppedByUserRef.current = false;
    setSending(false);
    setStopping(false);
  };

  const attachImage = async () => {
    const att = await window.inferml.dialog.openImage();
    if (att) setAtts(a => [...a, att]);
  };
  const removeAtt = (i) => setAtts(a => a.filter((_, idx) => idx !== i));

  const onKeyDown = (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); send(); }
    else if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  };

  const visibleMessages = chat.messages.filter(m => m.role !== 'system');



  const hasPriorImage = chat.messages.some(m => (m.attachments || []).some(a => a.kind === 'image'));
  const isGenerateMode = isJanus && janusMode === 'generate';
  const vlmNeedsImage = !isGenerateMode && isVLM && atts.length === 0 && !hasPriorImage;
  const generateNeedsText = isGenerateMode && !input.trim();
  const sendDisabled = sending || vlmNeedsImage || generateNeedsText || (!input.trim() && atts.length === 0);

  return (
    <div className="chat-view">
      <div className="chat-head">
        <div className="chat-head-titles">
          <div className="chat-title">{chat.title || 'New chat'}</div>
          <div className="chat-sub">{modelId} · {visibleMessages.length} msg{visibleMessages.length === 1 ? '' : 's'}</div>
        </div>
        <div style={{flex:1}}/>
      </div>

      <div ref={scrollRef} className="chat-body">
        {visibleMessages.length === 0 && (
          <div className="chat-empty">
            <div className="chat-empty-ic"><Icon name={isJanus ? 'sparkle' : (isVLM ? 'eye' : 'chat')} size={28} stroke={1.4}/></div>
            <div className="chat-empty-t">{isJanus ? 'Janus' : (isVLM ? 'Vision-language chat' : 'Chat')}</div>
            <div className="chat-empty-s">
              {isJanus
                ? (isGenerateMode
                    ? <>Type a prompt. Janus will synthesize an image. Running <code>{modelId}</code> locally.</>
                    : <>Attach an image, then ask about it. Running <code>{modelId}</code> locally.</>)
                : (isVLM
                    ? <>Attach an image, then ask about it. Running <code>{modelId}</code> locally.</>
                    : <>Send a message. Running <code>{modelId}</code> locally.</>)}
            </div>
          </div>
        )}
        {visibleMessages.map(m => <Message key={m.id} m={m}/>)}
        {error && <div className="chat-err"><Icon name="alert" size={12}/> {error}</div>}
      </div>

      <div className="chat-composer">
        {isJanus && <JanusModeBar value={janusMode} onChange={setJanusMode}/>}
        {atts.length > 0 && (
          <div className="cc-atts">
            {atts.map((a, i) => (
              <div key={i} className="cc-att">
                {a.kind === 'image' && <img src={a.dataUrl} alt={a.name}/>}
                <span className="cc-att-nm">{a.name}</span>
                <button className="cc-att-x" onClick={() => removeAtt(i)}><Icon name="x" size={11}/></button>
              </div>
            ))}
          </div>
        )}
        <textarea
          ref={inputRef}
          className="cc-input"
          placeholder={
            isGenerateMode
              ? 'Describe the image you want Janus to generate…'
              : vlmNeedsImage
                ? 'Attach an image first, then ask about it…'
                : isVLM
                  ? 'Ask about the attached image…'
                  : 'Send a message…'
          }
          rows={2}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={onKeyDown}
        />
        <div className="cc-foot">
          {isVLM && !isGenerateMode && <button className="hp-chip" onClick={attachImage}><Icon name="paperclip" size={11}/> Image</button>}
          <span className="hp-hint">local · <span style={{color:'var(--fg-1)'}}>{modelId}</span></span>
          <div style={{flex:1}}/>
          <button
            className={`cc-send ${sending ? 'is-stop' : ''}`}
            onClick={sending ? stop : send}
            disabled={sending ? stopping : sendDisabled}
            title={sending ? 'Stop the running inference' : undefined}
          >
            {sending
              ? (stopping ? 'Stopping…' : <><Icon name="x" size={11}/> Stop</>)
              : <>Send <span className="cc-kbd">⌘↵</span></>}
          </button>
        </div>
      </div>
    </div>
  );
}

function Message({ m }) {
  const images = (m.attachments || []).filter(a => a.kind === 'image');
  const saveGenerated = async () => {
    if (!m.image) return;
    try {
      const a = document.createElement('a');
      a.href = m.image;
      a.download = `janus-${m.id}.png`;
      a.click();
    } catch {}
  };
  return (
    <div className={`msg ${m.role}`}>
      <div className="msg-bubble">
        {images.length > 0 && (
          <div className="msg-imgs">
            {images.map((a, i) => <img key={i} src={a.dataUrl} alt={a.name}/>)}
          </div>
        )}
        {m.image && (
          <div className="msg-imgs msg-imgs-gen">
            <img src={m.image} alt="generated"/>
            <button type="button" className="msg-img-save" onClick={saveGenerated} title="Save image">
              <Icon name="download" size={12}/> Save
            </button>
          </div>
        )}
        {m.text && (
          m.role === 'assistant' && !m.error
            ? <MarkdownText text={m.text} className="msg-text"/>
            : <div className={`msg-text ${m.error ? 'err' : ''}`}>{m.text}</div>
        )}
        {m.streaming && !m.text && !m.image && (
          <div className="msg-loading">
            <span className="msg-dot"/><span className="msg-dot"/><span className="msg-dot"/>
            <span>running locally…</span>
          </div>
        )}
        <div className="msg-meta">{m.role === 'user' ? 'you' : (m.model || 'assistant')} · {formatTime(m.ts)}</div>
      </div>
    </div>
  );
}

const _markedConfigured = (() => {
  if (typeof window === 'undefined' || !window.marked) return false;
  try {
    window.marked.setOptions({
      gfm: true,    
      breaks: true, 
    });
  } catch { return false; }




  if (window.DOMPurify && typeof window.DOMPurify.addHook === 'function') {
    try {
      window.DOMPurify.addHook('afterSanitizeAttributes', (node) => {
        if (!node || node.nodeType !== 1) return;
        if (node.tagName === 'A') {
          node.setAttribute('target', '_blank');
          node.setAttribute('rel', 'noopener noreferrer');
        }
      });
    } catch {}
  }
  return true;
})();

function MarkdownText({ text, className }) {

  const html = React.useMemo(() => {
    if (!_markedConfigured || !window.DOMPurify) return null;
    try {
      const parsed = window.marked.parse(text || '', { async: false });
      return window.DOMPurify.sanitize(parsed, { ADD_ATTR: ['target', 'rel'] });
    } catch { return null; }
  }, [text]);


  if (html === null) {
    return <div className={className} style={{whiteSpace:'pre-wrap'}}>{text}</div>;
  }
  return <div className={`${className} markdown`} dangerouslySetInnerHTML={{ __html: html }}/>;
}

function formatTime(ts) {
  if (!ts) return '';
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

window.ChatWorkspace = ChatWorkspace;
