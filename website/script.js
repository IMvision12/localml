// InferML landing page - interactions.
//
// InferML ships as a desktop app from GitHub Releases. This script resolves the
// download buttons to the right asset for the visitor's OS, wires up the
// copy-to-clipboard blocks (still used by the API code sample), highlights the
// visitor's platform card, and runs the ambient scroll / hero effects.

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

// ── OS detection + direct download links.
//
// Clicking Download starts the download. It never lands the visitor on GitHub.
//
// The links point at GitHub's stable "latest asset" path, which 302s straight to
// the newest build:
//
//   /releases/latest/download/InferML-Setup.exe
//
// There is deliberately NO GitHub API call here. The API is rate-limited (60
// req/hr per IP) and returns 404 until the first release exists - either of which
// used to knock every button back to the releases page. Because the artifact
// names carry no version (see electron-builder.yml), these URLs can be hard-coded
// and the platform cards work even with JavaScript disabled.
(function () {
  const REPO = 'IMvision12/InferML';
  const DL = (file) => `https://github.com/${REPO}/releases/latest/download/${file}`;

  const ASSET = {
    win: 'InferML-Setup.exe',
    macArm: 'InferML-arm64.dmg',
    macIntel: 'InferML-x64.dmg',
    linux: 'InferML.AppImage',
  };

  const ua = (navigator.userAgent || '').toLowerCase();
  const plat = (navigator.platform || '').toLowerCase();
  let os = 'windows';
  if (ua.includes('mac') || plat.includes('mac')) os = 'mac';
  else if (ua.includes('linux') || plat.includes('linux')) os = 'linux';

  // Apple Silicon vs Intel. Handing an Intel Mac an arm64 build gives them an app
  // that cannot run, so this needs to be right.
  //
  // The GPU string is the best signal available: every Apple Silicon Mac reports
  // an Apple GPU, and no Intel Mac ever does (they ship Intel/AMD/Nvidia). It also
  // beats userAgentData.architecture, which reports x86 for a browser running
  // under Rosetta on an M-series machine - exactly the case we must not get wrong.
  function isAppleSilicon() {
    try {
      const gl = document.createElement('canvas').getContext('webgl');
      if (!gl) return null;
      const dbg = gl.getExtension('WEBGL_debug_renderer_info');
      const renderer = dbg
        ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) || '')
        : String(gl.getParameter(gl.RENDERER) || '');
      if (!renderer) return null;
      if (/apple/i.test(renderer)) return true;
      if (/intel|amd|radeon|nvidia|geforce/i.test(renderer)) return false;
      return null;
    } catch {
      return null;
    }
  }

  // Which macOS build to offer.
  //
  // The GPU check is only meaningful for a visitor who is ON a Mac. A Windows or
  // Linux visitor clicking the macOS card has an Intel/AMD/Nvidia GPU that says
  // nothing about the Mac they're downloading for - reading it would hand every
  // one of them the Intel build. So only consult it on macOS; everyone else gets
  // Apple Silicon, which is the large majority of Macs in use, and the Intel build
  // is one visible link away.
  const macIsIntel = () => os === 'mac' && isAppleSilicon() === false;
  const macAsset = () => (macIsIntel() ? ASSET.macIntel : ASSET.macArm);

  const forOs = {
    windows: () => ASSET.win,
    mac: macAsset,
    linux: () => ASSET.linux,
  };

  // Platform cards: Windows and Linux are already correct in the markup; macOS is
  // the only one that needs resolving, since its arch isn't known until now.
  const KEY = { windows: 'win', mac: 'mac', linux: 'linux' };
  const macCard = document.querySelector('.dl-card[data-os="mac"]');
  if (macCard) macCard.href = DL(macAsset());

  const you = document.querySelector(`.dl-card[data-os="${KEY[os]}"]`);
  if (you) you.classList.add('is-you');

  // Hero button: point it at this visitor's build and say which one it is.
  const hero = document.getElementById('hero-dl');
  const label = document.getElementById('hero-dl-label');
  if (hero) hero.href = DL(forOs[os]());
  if (label) {
    label.textContent =
      os === 'windows' ? 'Download for Windows'
      : os === 'linux' ? 'Download for Linux'
      : macIsIntel()   ? 'Download for macOS (Intel)'
      : 'Download for macOS';
  }

  // ── Compatibility bridge for releases built before the filenames dropped their
  // version (v2.0.0 shipped `InferML-Setup-2.0.0.exe`, which the stable URLs above
  // cannot name). Ask the API what the newest release actually contains and match
  // by pattern, so the buttons work regardless of which naming scheme is live.
  //
  // This is an upgrade, never a downgrade: if the API is rate-limited, offline, or
  // returns nothing, the hard-coded links above stand and the buttons still work.
  const MATCH = {
    win: /\.exe$/i,
    linux: /\.AppImage$/i,
    macArm: /arm64\.dmg$/i,
    macIntel: /(x64|x86_64|intel)\.dmg$/i,
  };
  const findAsset = (assets, re) => {
    const hit = assets.find((a) => re.test(a.name || '') && !/\.blockmap$/i.test(a.name || ''));
    return hit && hit.browser_download_url;
  };

  fetch(`https://api.github.com/repos/${REPO}/releases/latest`)
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error('no release'))))
    .then((rel) => {
      const assets = rel.assets || [];
      const macKey = macIsIntel() ? 'macIntel' : 'macArm';
      const real = {
        win: findAsset(assets, MATCH.win),
        linux: findAsset(assets, MATCH.linux),
        mac: findAsset(assets, MATCH[macKey]),
      };

      document.querySelectorAll('.dl-card').forEach((el) => {
        const url = real[el.getAttribute('data-os')];
        if (url) el.href = url;
      });
      const heroUrl = real[KEY[os]];
      if (hero && heroUrl) hero.href = heroUrl;

      // Same for the secondary links, which name specific builds.
      const intel = findAsset(assets, MATCH.macIntel);
      const deb = findAsset(assets, /\.deb$/i);
      document.querySelectorAll('.dl-alt a').forEach((a) => {
        if (/x64\.dmg/i.test(a.href) && intel) a.href = intel;
        else if (/\.deb/i.test(a.href) && deb) a.href = deb;
      });
    })
    .catch(() => { /* keep the hard-coded stable URLs */ });

  // The hero command ships as the curl/sh line; Windows visitors get PowerShell.
  if (os === 'windows') {
    const ps = 'irm https://inferml.vercel.app/install.ps1 | iex';
    const cmd = document.getElementById('hero-cmd');
    const text = document.getElementById('hero-cmd-text');
    const prompt = document.getElementById('hero-cmd-prompt');
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
  ['.feat-grid', '.model-families', '.downloads', '.oneliners', '.api-points', '.mcp-tools'].forEach((gridSel) => {
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
