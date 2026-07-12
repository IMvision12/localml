#!/usr/bin/env node
// Remove build + dev output, leaving node_modules and source intact.
const fs = require('fs/promises');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const TARGETS = [
  'dist', 'out', 'release', '.cache', 'dist-app',
  'src/renderer/dist',
];

(async () => {
  for (const t of TARGETS) {
    const p = path.join(ROOT, t);
    try {
      await fs.rm(p, { recursive: true, force: true });
      console.log(`removed ${t}`);
    } catch (e) {
      console.log(`skip ${t}: ${e.message}`);
    }
  }
})();
