const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const serverPath = path.join(__dirname,'..','serve.cjs');
test('the optional preview server exists', () => assert.ok(fs.existsSync(serverPath),'range-capable preview server is missing'));
if (fs.existsSync(serverPath)) {
  test('preview serves byte ranges, HEAD requests and rejects non-public paths', async () => {
    const server = require(serverPath).createServer();
    await new Promise(resolve => server.listen(0,'127.0.0.1',resolve));
    const base = `http://127.0.0.1:${server.address().port}`;
    try {
      const partial = await fetch(base+'/assets/brain-maps.mp4',{headers:{Range:'bytes=0-1023'}});
      assert.equal(partial.status,206);
      assert.equal(partial.headers.get('content-range'),'bytes 0-1023/10146838');
      assert.equal((await partial.arrayBuffer()).byteLength,1024);
      const head = await fetch(base+'/assets/brain-maps.mp4',{method:'HEAD'});
      assert.equal(head.headers.get('accept-ranges'),'bytes');
      assert.equal(head.headers.get('content-length'),'10146838');
      const invalid = await fetch(base+'/assets/brain-maps.mp4',{headers:{Range:'bytes=999999999-'}});
      assert.equal(invalid.status,416);
      assert.equal((await fetch(base+'/README.md')).status,404);
      assert.equal((await fetch(base+'/index.html')).status,200);
    } finally { await new Promise(resolve => server.close(resolve)); }
  });
}
