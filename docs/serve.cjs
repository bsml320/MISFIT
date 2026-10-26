// Optional loopback-only preview server. GitHub Pages needs no server script.
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const publicFiles = new Map([
  ['/', ['index.html','text/html; charset=utf-8']],
  ['/index.html', ['index.html','text/html; charset=utf-8']],
  ['/styles.css', ['styles.css','text/css; charset=utf-8']],
  ['/viewer.js', ['viewer.js','application/javascript; charset=utf-8']],
  ['/viewer-core.js', ['viewer-core.js','application/javascript; charset=utf-8']],
  ['/provenance.json', ['provenance.json','application/json; charset=utf-8']],
  ['/assets/brain-maps.mp4', ['assets/brain-maps.mp4','video/mp4']],
  ['/assets/poster.png', ['assets/poster.png','image/png']]
]);
function createServer() {
  return http.createServer((request,response) => {
    if (!['GET','HEAD'].includes(request.method)) { response.writeHead(405); response.end(); return; }
    let route;
    try { route = new URL(request.url,'http://127.0.0.1').pathname; }
    catch { response.writeHead(400); response.end(); return; }
    const item = publicFiles.get(route);
    if (!item) { response.writeHead(404); response.end('Not found'); return; }
    const file = path.join(__dirname,item[0]);
    const size = fs.statSync(file).size;
    let start = 0;
    let end = size - 1;
    let status = 200;
    const headers = {'Content-Type':item[1], 'Accept-Ranges':'bytes', 'Cache-Control':'no-store'};
    if (request.headers.range) {
      const match = /^bytes=(\d*)-(\d*)$/.exec(request.headers.range);
      if (match && (match[1] || match[2])) {
        start = match[1] ? Number(match[1]) : Math.max(0,size-Number(match[2]));
        end = match[1] && match[2] ? Math.min(size-1,Number(match[2])) : size-1;
      } else start = size;
      if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start >= size || end < start) {
        response.writeHead(416,{'Content-Range':`bytes */${size}`}); response.end(); return;
      }
      status = 206;
      headers['Content-Range'] = `bytes ${start}-${end}/${size}`;
    }
    headers['Content-Length'] = end-start+1;
    response.writeHead(status,headers);
    if (request.method === 'HEAD') { response.end(); return; }
    const stream = fs.createReadStream(file,{start,end});
    stream.on('error',() => response.destroy());
    stream.pipe(response);
  });
}
module.exports = {createServer};
if (require.main === module) {
  const port = Number(process.argv[2] || 8766);
  const server = createServer();
  server.on('error',error => { console.error(error.message); process.exitCode=1; });
  server.listen(port,'127.0.0.1',() => console.log(`Local viewer: http://127.0.0.1:${port}/`));
}
