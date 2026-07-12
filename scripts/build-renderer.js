#!/usr/bin/env node
//
// Pre-compile the renderer's JSX files to plain JS so users don't pay the
// Babel-Standalone tax on every cold start.
//
// Without this step, Windows cold launch shows ~3 s of black screen while
// the browser parses Babel (~3.5 MB), then compiles 8 separate JSX files
// in sequence. With this step, the renderer loads pre-transformed JS in a
// single React-mount round, dropping cold start to under ~500 ms.
//
// Output:
//   src/renderer/dist/index.html       (Babel removed, .jsx → .js paths)
//   src/renderer/dist/styles.css       (copied)
//   src/renderer/dist/components/*.js  (esbuild-transformed JSX)
//
// `main.js` chooses between the source HTML (dev, with Babel hot reload)
// and the dist HTML (packaged builds) based on `app.isPackaged`.
//
// Usage:
//   node scripts/build-renderer.js          # one-shot
//   node scripts/build-renderer.js --watch  # rebuild on change

const fs = require('fs');
const path = require('path');
const esbuild = require('esbuild');

const ROOT             = path.resolve(__dirname, '..');
const RENDERER_DIR     = path.join(ROOT, 'src', 'renderer');
const COMPONENTS_DIR   = path.join(RENDERER_DIR, 'components');
const OUT_DIR          = path.join(RENDERER_DIR, 'dist');
const OUT_COMPONENTS   = path.join(OUT_DIR, 'components');
const SRC_HTML         = path.join(RENDERER_DIR, 'index.html');
const OUT_HTML         = path.join(OUT_DIR, 'index.html');

const watchMode = process.argv.includes('--watch');

function rel(p) { return path.relative(ROOT, p).replace(/\\/g, '/'); }

// Read the source HTML (the dev-mode one with Babel Standalone) and rewrite
// it for production:
//   - drop the Babel Standalone <script> tag
//   - rewrite each <script type="text/babel" src=".../X.jsx"> to
//                  <script src=".../X.js">
//   - drop 'unsafe-eval' from the CSP - without Babel-in-browser the
//     renderer no longer needs eval, which is a real security tightening
function generateHtml() {
  let html = fs.readFileSync(SRC_HTML, 'utf8');

  // Remove the Babel Standalone <script> (with optional whitespace either side)
  html = html.replace(
    /[ \t]*<script[^>]*\/@babel\/standalone\/babel\.min\.js[^>]*><\/script>\n?/g,
    '',
  );

  // Rewrite text/babel script tags → plain script tags pointing at .js
  html = html.replace(
    /<script\s+type="text\/babel"\s+src="([^"]+)\.jsx"\s*><\/script>/g,
    '<script src="$1.js"></script>',
  );

  // Vendor the browser runtime deps (react, react-dom, marked, dompurify) so the
  // built frontend is self-contained and doesn't reach into node_modules, which
  // isn't shipped. Each `../../node_modules/<pkg>/<path>/<file>` script src is
  // rewritten to `vendor/<file>`, and the files are copied by copyVendorAssets().
  //
  // The path MUST stay relative. This used to emit `/vendor/<file>`, which was
  // fine when a server was serving the page - but the app loads the UI straight
  // off disk now (file://), and there a leading slash resolves to the root of
  // the filesystem. React would 404, and the window would come up blank.
  html = html.replace(
    /(["'])\.\.\/\.\.\/node_modules\/[^"']*\/([^"'\/]+)(["'])/g,
    '$1vendor/$2$3',
  );

  // Tighten CSP: drop 'unsafe-eval'. Anything else stays.
  html = html.replace(/\s*'unsafe-eval'/g, '');

  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.writeFileSync(OUT_HTML, html);
}

// Files that the rendered index.html references via relative path and that
// aren't transformed (CSS, fonts, anything dropped into src/renderer/).
function copyStaticAssets() {
  fs.mkdirSync(OUT_DIR, { recursive: true });
  for (const name of ['styles.css']) {
    const src = path.join(RENDERER_DIR, name);
    if (fs.existsSync(src)) {
      fs.copyFileSync(src, path.join(OUT_DIR, name));
    }
  }
}

// Copy the browser runtime UMD bundles into dist/vendor/ so the built
// frontend loads them from same-origin `/vendor/*` (see generateHtml's
// rewrite) rather than reaching into node_modules. This is what lets the
// Python web server serve a fully self-contained UI with no Node present.
const VENDOR_FILES = [
  ['react',     'umd/react.production.min.js'],
  ['react-dom', 'umd/react-dom.production.min.js'],
  ['marked',    'lib/marked.umd.js'],
  ['dompurify', 'dist/purify.min.js'],
];
function copyVendorAssets() {
  const outVendor = path.join(OUT_DIR, 'vendor');
  fs.mkdirSync(outVendor, { recursive: true });
  const nm = path.join(ROOT, 'node_modules');
  for (const [pkg, relPath] of VENDOR_FILES) {
    const src = path.join(nm, pkg, relPath);
    if (fs.existsSync(src)) {
      fs.copyFileSync(src, path.join(outVendor, path.basename(relPath)));
    } else {
      console.warn(`[renderer] vendor asset missing: ${pkg}/${relPath} (run npm install)`);
    }
  }
}

function listJsxEntries() {
  if (!fs.existsSync(COMPONENTS_DIR)) return [];
  return fs.readdirSync(COMPONENTS_DIR)
    .filter((f) => f.endsWith('.jsx'))
    .map((f) => path.join(COMPONENTS_DIR, f));
}

const baseBuildOpts = {
  outdir: OUT_COMPONENTS,
  loader: { '.jsx': 'jsx' },
  jsx: 'transform',
  jsxFactory: 'React.createElement',
  jsxFragment: 'React.Fragment',
  // Electron 33 ships Chromium 130 - chrome120 is a safe baseline that
  // doesn't downlevel anything we'd actually use.
  target: ['chrome120'],
  // bundle:false keeps the existing classic-script architecture: each .jsx
  // becomes a self-contained .js, components attach themselves to `window`,
  // load order is preserved by the <script> tags in index.html.
  bundle: false,
  // Skip minify so the `function Foo() { … } window.Foo = Foo` global
  // pattern survives intact. Components use bare `Icon`, `Logo`, etc. as
  // free variables that resolve to window at runtime; minify could rename
  // those when esbuild can't see they're cross-file globals.
  minify: false,
  sourcemap: false,
  logLevel: 'silent',
};

async function buildOnce() {
  const entries = listJsxEntries();
  fs.mkdirSync(OUT_COMPONENTS, { recursive: true });

  await esbuild.build({ ...baseBuildOpts, entryPoints: entries });
  generateHtml();
  copyStaticAssets();
  copyVendorAssets();

  console.log(`[renderer] compiled ${entries.length} component(s) → ${rel(OUT_COMPONENTS)}`);
}

async function buildWatch() {
  const entries = listJsxEntries();
  fs.mkdirSync(OUT_COMPONENTS, { recursive: true });

  const ctx = await esbuild.context({ ...baseBuildOpts, entryPoints: entries });
  await ctx.watch();

  // Initial pass for the HTML + static assets, then re-run when the source
  // HTML or styles change. fs.watch on Windows fires `rename` for content
  // edits in some editors, so we rebuild on any event.
  generateHtml();
  copyStaticAssets();
  copyVendorAssets();
  try {
    fs.watch(SRC_HTML, () => {
      try { generateHtml(); console.log('[renderer] re-emitted index.html'); } catch (e) { console.warn('[renderer] html rebuild failed:', e.message); }
    });
  } catch {}
  try {
    fs.watch(path.join(RENDERER_DIR, 'styles.css'), () => {
      try { copyStaticAssets(); console.log('[renderer] copied styles.css'); } catch {}
    });
  } catch {}

  console.log(`[renderer] watching ${entries.length} component(s) for changes…`);

  const teardown = async () => {
    try { await ctx.dispose(); } catch {}
    process.exit(0);
  };
  process.on('SIGINT', teardown);
  process.on('SIGTERM', teardown);
}

(watchMode ? buildWatch() : buildOnce()).catch((err) => {
  console.error('[renderer] build failed:', err && err.message || err);
  process.exit(1);
});
