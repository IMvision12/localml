// LocalML landing page - interactions.
//
// LocalML installs from PyPI (`pipx install localml`) and runs as a local web
// server, so there are no OS-specific installers to fetch. This script wires up
// the copy-to-clipboard command blocks, highlights the visitor's own platform
// card, and runs the ambient scroll / hero effects.

// ── copy-to-clipboard for command + code blocks ──────────────────────────
(function () {
  function flash(btn) {
    btn.classList.add('is-copied');
    const label = btn.querySelector('.cmd-copy-label');
    const prev = label ? label.textContent : '';
    if (label) label.textContent = 'Copied';
    setTimeout(() => {
      btn.classList.remove('is-copied');
      if (label) label.textContent = prev || 'Copy';
    }, 1600);
  }

  function fallbackCopy(text, btn) {
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      ta.remove();
      flash(btn);
    } catch { /* clipboard unavailable - nothing we can do */ }
  }

  function copy(text, btn) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(() => flash(btn)).catch(() => fallbackCopy(text, btn));
    } else {
      fallbackCopy(text, btn);
    }
  }

  // Resolve the text to copy: an explicit [data-copy] wins (e.g. commands with
  // markup), otherwise fall back to the block's rendered command / code text.
  function textFor(btn) {
    const holder = btn.closest('[data-copy]');
    if (holder && holder.getAttribute('data-copy')) return holder.getAttribute('data-copy');
    const scope = btn.closest('.cmd, .api-code');
    const src = scope && (scope.querySelector('.cmd-text') || scope.querySelector('.api-pre'));
    return src ? src.textContent : '';
  }

  document.querySelectorAll('.cmd-copy').forEach((btn) => {
    btn.addEventListener('click', () => {
      const text = textFor(btn);
      if (text) copy(text, btn);
    });
  });
})();

// ── OS detection: highlight the visitor's platform card, and set the hero
//    install one-liner. The HTML ships the curl/sh line by default; Windows
//    visitors get the PowerShell `irm … | iex` line instead.
(function () {
  const ua = (navigator.userAgent || '').toLowerCase();
  const plat = (navigator.platform || '').toLowerCase();
  let os = 'windows';
  if (ua.includes('mac') || plat.includes('mac')) os = 'mac';
  else if (ua.includes('linux') || plat.includes('linux')) os = 'linux';

  const card = document.getElementById('plat-' + os);
  if (card) card.classList.add('is-you');

  if (os === 'windows') {
    const ps = 'irm https://www.localml.tech/install.ps1 | iex';
    const cmd = document.getElementById('hero-cmd');
    const text = document.getElementById('hero-cmd-text');
    const prompt = cmd && cmd.querySelector('.cmd-prompt');
    if (cmd) cmd.setAttribute('data-copy', ps);
    if (text) text.textContent = ps;
    if (prompt) prompt.textContent = '>';
  }
})();

// Toggle nav border on scroll for subtle separation.
(function () {
  const nav = document.querySelector('.nav');
  if (!nav) return;
  const onScroll = () => {
    if (window.scrollY > 8) nav.classList.add('scrolled');
    else nav.classList.remove('scrolled');
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();

// Fade-in-on-scroll with staggered delay per card in a grid.
(function () {
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          e.target.style.opacity = '1';
          e.target.style.transform = 'translateY(0)';
          io.unobserve(e.target);
        }
      }
    },
    { threshold: 0.1 }
  );
  // Parent-level reveals (no stagger)
  document.querySelectorAll('.section-head, .screenshot-frame, .api-code').forEach((el) => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(24px)';
    el.style.transition = 'opacity 0.7s cubic-bezier(0.2, 0.8, 0.2, 1), transform 0.7s cubic-bezier(0.2, 0.8, 0.2, 1)';
    io.observe(el);
  });
  // Grid reveals with stagger
  ['.feat-grid', '.model-families', '.plat-grid', '.oneliners', '.api-points'].forEach((gridSel) => {
    const cards = document.querySelectorAll(`${gridSel} > *`);
    cards.forEach((el, i) => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(28px)';
      el.style.transition = `opacity 0.55s cubic-bezier(0.2, 0.8, 0.2, 1) ${i * 60}ms, transform 0.55s cubic-bezier(0.2, 0.8, 0.2, 1) ${i * 60}ms, border-color 0.3s, box-shadow 0.3s`;
      io.observe(el);
    });
  });
})();

// Populate the constellation-particle field in the hero background.
(function () {
  const host = document.getElementById('hero-stars');
  if (!host) return;
  const COUNT = 24;
  for (let i = 0; i < COUNT; i++) {
    const s = document.createElement('span');
    s.style.top = (Math.random() * 100) + '%';
    s.style.left = (Math.random() * 100) + '%';
    s.style.animationDelay = (Math.random() * 6) + 's';
    s.style.animationDuration = (4 + Math.random() * 5) + 's';
    s.style.width = s.style.height = (1.5 + Math.random() * 2.5) + 'px';
    host.appendChild(s);
  }
})();

// Mouse-follow halo on feature cards - reads cursor position, pipes to CSS var.
(function () {
  const cards = document.querySelectorAll('.feat');
  cards.forEach((card) => {
    card.addEventListener('mousemove', (e) => {
      const r = card.getBoundingClientRect();
      card.style.setProperty('--mx', ((e.clientX - r.left) / r.width * 100) + '%');
      card.style.setProperty('--my', ((e.clientY - r.top) / r.height * 100) + '%');
    });
  });
})();

// Subtle parallax on hero aurora - shifts with mouse (2-3px max).
(function () {
  const aurora = document.querySelector('.hero-aurora');
  const stars = document.getElementById('hero-stars');
  if (!aurora) return;
  document.addEventListener('mousemove', (e) => {
    const x = (e.clientX / window.innerWidth - 0.5) * 14;
    const y = (e.clientY / window.innerHeight - 0.5) * 14;
    aurora.style.transform = `translate3d(${x}px, ${y}px, 0)`;
    if (stars) stars.style.transform = `translate3d(${x * 0.5}px, ${y * 0.5}px, 0)`;
  });
})();
