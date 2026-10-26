const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = path.join(__dirname, '..');
test('portable site files exist', () => {
  for (const name of ['index.html','styles.css','viewer.js','viewer-core.js',
    '.nojekyll','provenance.json','assets/brain-maps.mp4','assets/poster.png']) {
    assert.equal(fs.existsSync(path.join(root,name)), true, `missing ${name}`);
  }
});
if (fs.existsSync(path.join(root,'index.html'))) {
  test('the page has accessible controls and an ordinary-video fallback', () => {
    const html = fs.readFileSync(path.join(root,'index.html'),'utf8');
    for (const id of ['brain-video','play-pause','timeline','previous-frame',
      'next-frame','restart','fullscreen','speed','cohort-status','error-message']) {
      assert.ok(html.includes(`id="${id}"`), `missing ${id}`);
    }
    assert.match(html, /<video[^>]*controls/s);
    assert.match(html, /<noscript>/);
    assert.match(html, /aria-label="Playback position"/);
    assert.doesNotMatch(html, /autoplay/);
    assert.equal((html.match(/data-cohort=/g) || []).length, 3);
  });
  test('all runtime scripts and styles are local and every linked asset exists', () => {
    const html = fs.readFileSync(path.join(root,'index.html'),'utf8');
    for (const [, value] of html.matchAll(/(?:src|href|poster)="([^"]+)"/g)) {
      if (value.startsWith('#')) continue;
      assert.doesNotMatch(value, /^(?:https?:|\/\/|file:)/);
      assert.equal(fs.existsSync(path.join(root,value)),true,value);
    }
    for (const name of ['viewer.js','viewer-core.js','styles.css']) {
      assert.doesNotMatch(fs.readFileSync(path.join(root,name),'utf8'),
        /https?:\/\/|\bfetch\s*\(|\bXMLHttpRequest\b|\bWebSocket\b|@import/);
    }
  });
  test('bundled scientific media exactly match their recorded source hashes', () => {
    const provenance = JSON.parse(fs.readFileSync(path.join(root,'provenance.json'),'utf8'));
    for (const item of provenance.assets) {
      const digest = crypto.createHash('sha256').update(fs.readFileSync(path.join(root,item.path))).digest('hex');
      assert.equal(digest,item.sha256,item.path);
    }
    assert.equal(provenance.new_statistical_analysis,false);
    assert.equal(provenance.patient_level_inputs_included,false);
  });
  test('full-screen mode enables native playback controls', () => {
    const js = fs.readFileSync(path.join(root,'viewer.js'),'utf8');
    assert.match(js,/fullscreenchange/);
    assert.match(js,/video\.controls = Boolean\(document\.fullscreenElement\)/);
  });
}
