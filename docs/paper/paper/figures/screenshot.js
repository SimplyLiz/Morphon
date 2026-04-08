// Playwright screenshot script for Morphon WASM visualizer
// Usage: node screenshot.js (requires playwright in node_modules)
//
// Captures the visualizer at multiple states for the paper.
// Produces both full-UI and clean (panels hidden) versions.

const { chromium } = require('playwright');
const path = require('path');

const URL = 'http://localhost:8080/';
const OUT_DIR = path.join(__dirname);

const HIDE_UI_CSS = `
  #left-panel, #right-panel, #timeline-bar, #bottom-panel,
  #log-panel, #log-tabs, #log-content, #log-status,
  .logo, header, nav, footer,
  .overlay, .panel { display: none !important; }
  body { background: #000 !important; }
  /* Hide any floating indicators */
  [class*="page-indicator"], [class*="fps"] { display: none !important; }
`;

(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({
    viewport: { width: 1920, height: 1200 },
    deviceScaleFactor: 2,
  });
  const page = await ctx.newPage();

  console.log('Loading visualizer...');
  await page.goto(URL, { waitUntil: 'networkidle' });
  console.log('Waiting for WASM init...');
  await page.waitForTimeout(3000);

  const shoot = (name) => page.screenshot({
    path: path.join(OUT_DIR, name + '.jpg'),
    type: 'jpeg',
    quality: 80,
    fullPage: false,
  });

  // ─── Snapshot A: full UI, initial state ───
  console.log('Snapshot A: full UI, initial');
  await shoot('visualizer_full');

  // Let it run a bit
  await page.waitForTimeout(15000);

  // ─── Snapshot B: full UI, after 15s ───
  console.log('Snapshot B: full UI, after 15s');
  await shoot('visualizer_running');

  // Hide UI for clean topology shots
  await page.addStyleTag({ content: HIDE_UI_CSS });
  await page.waitForTimeout(500);

  // ─── Snapshot C: clean topology, current state ───
  console.log('Snapshot C: clean topology');
  await shoot('topology_running');

  // Wait more
  await page.waitForTimeout(15000);
  console.log('Snapshot D: clean topology, +15s');
  await shoot('topology_developed');

  await browser.close();
  console.log('Done. Wrote PNGs to', OUT_DIR);
})();
