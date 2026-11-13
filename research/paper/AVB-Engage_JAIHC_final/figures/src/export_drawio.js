// Export a .drawio file to a cropped PDF with draw.io's own export page (app.diagrams.net/export3.html) in headless
// Chrome, as jgraph/draw-image-export2 does; the diagram is rendered locally in the browser, nothing is uploaded.
// Setup (2026-11-13, any scratch folder): npm install puppeteer@24; Chrome also needs libasound.so.2 (conda-forge
// alsa-lib), copied into ./libs next to this script. Usage: node export_drawio.js in.drawio out.pdf
// One draw.io px prints as 0.719 pt: 331 px = 84 mm, 686 px = 174 mm, 12.5 px text = 9 pt (docs/plan.md Sec. 21.13).
const puppeteer = require('puppeteer');
const fs = require('fs');
(async () => {
  const [inp, out] = process.argv.slice(2);
  const xml = fs.readFileSync(inp, 'utf8');
  const browser = await puppeteer.launch({headless: 'shell', env: Object.assign({}, process.env, {LD_LIBRARY_PATH: __dirname + '/libs'}), args: ['--no-sandbox', '--disable-gpu', '--font-render-hinting=none']});
  try {
    const page = await browser.newPage();
    await page.goto('https://app.diagrams.net/export3.html', {waitUntil: 'networkidle0', timeout: 120000});
    await page.evaluate((xml) => render({xml: xml, format: 'pdf', w: 0, h: 0, border: 0, bg: '#ffffff', scale: 1, from: 0, to: 0, crop: '1'}), xml);
    await page.waitForSelector('#LoadingComplete', {timeout: 120000});
    const b = JSON.parse(await page.$eval('#LoadingComplete', d => d.getAttribute('bounds')));
    const fix = 0.959;  // draw-image-export2: Chrome prints PDFs larger than the requested pixel size
    const w = Math.ceil(Math.ceil(b.width + b.x) * fix);
    const h = Math.ceil(Math.ceil(b.height + b.y) * fix + 0.1);
    const pdf = await page.pdf({printBackground: true, width: w + 'px', height: (h + 1) + 'px', pageRanges: '1',
                                margin: {top: '0px', bottom: '0px', left: '0px', right: '0px'}});
    fs.writeFileSync(out, pdf);
    console.log(JSON.stringify({bounds: b, page_px: [w, h + 1]}));
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exit(1); });
