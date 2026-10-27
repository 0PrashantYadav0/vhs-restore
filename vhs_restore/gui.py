# Point-and-click app: a small local web page (standard library only). Only this computer can reach it, and every
# request needs the random key in the URL, so other websites can't drive it. Jobs run the normal command line.
import json, os, re, secrets, signal, subprocess, sys, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

MEDIA = {'.mp4', '.mov', '.mkv', '.avi', '.mpg', '.mpeg', '.m4v', '.ts', '.dv', '.vob', '.wmv', '.webm',
         '.wav', '.mp3', '.m4a', '.flac', '.aac', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'}
FLAGS = {'mono', 'declick', 'speech', 'stabilize', 'dropouts', 'colorize', 'scratches', 'no-color'}
VALUES = {'aspect': r'\d+:\d+', 'crop': r'auto|\d+:\d+:\d+:\d+', 'height': r'\d{3,4}', 'long-edge': r'\d{3,5}',
          'every': r'\d+(\.\d+)?', 'chroma-shift': r'-?\d{1,2}', 'faces': r'0?\.\d+|1', 'ai': r'0?\.\d+|1', 'preview': r'[\d:.]+',
          'denoise': r'off|light|medium|strong', 'deinterlace': r'auto|on|off'}
jobs = []


def argv(req):
    """Build the command from a whitelist - nothing from the page reaches a shell."""
    cmd = req['cmd']
    if cmd not in ('video', 'audio', 'image', 'stills'): raise ValueError('unknown task')
    a = [sys.executable, '-u', '-m', 'vhs_restore', cmd, req['input']]
    a += ['-o', req['output']] if cmd == 'image' else [req['output']]
    for k, v in req.get('options', {}).items():
        if k in FLAGS and v is True: a.append('--' + k)
        elif k in VALUES and v not in (None, '', False):
            if not re.fullmatch(VALUES[k], str(v)): raise ValueError(f'bad value for {k}: {v}')
            a += ['--' + k, str(v)]
    return a


def progress(log):
    for line in reversed(log[-50:]):
        m = re.search(r'(\d+)/(\d+) chunks done, about (\S+) left', line)
        if m: return int(m[1]) / int(m[2]), m[3] + ' left'
        m = re.match(r'(\d+)/(\d+) ', line)
        if m: return int(m[1]) / int(m[2]), ''
    return None, ''


def start(req):
    p = subprocess.Popen(argv(req), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         start_new_session=True)
    job = {'id': len(jobs), 'proc': p, 'log': [], 'output': req['output'], 'cmd': req['cmd']}
    jobs.append(job)
    threading.Thread(target=lambda: [job['log'].append(l.rstrip()) for l in p.stdout], daemon=True).start()
    return job['id']


class Handler(BaseHTTPRequestHandler):
    token = ''

    def log_message(self, *_): pass

    def reply(self, code, body, kind='application/json', extra=()):
        data = body if isinstance(body, bytes) else (json.dumps(body) if kind == 'application/json' else body).encode()
        self.send_response(code)
        self.send_header('Content-Type', kind); self.send_header('Content-Length', str(len(data)))
        for k, v in extra: self.send_header(k, v)
        self.end_headers(); self.wfile.write(data)

    def allowed(self, q):
        host_ok = self.headers.get('Host', '').split(':')[0] in ('127.0.0.1', 'localhost')  # blocks DNS rebinding
        return host_ok and secrets.compare_digest(q.get('t', [''])[0], self.token)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if not self.allowed(q): return self.reply(403, 'open the link printed in the terminal', 'text/plain')
        if u.path == '/': return self.reply(200, PAGE, 'text/html; charset=utf-8')
        if u.path == '/api/ls':
            p = Path(q.get('path', [str(Path.home())])[0]).expanduser()
            if p.is_file(): p = p.parent
            try:
                items = sorted((x for x in p.iterdir() if not x.name.startswith('.')), key=lambda x: x.name.lower())
            except OSError as e:
                return self.reply(400, {'error': str(e)})
            return self.reply(200, {'path': str(p), 'parent': str(p.parent),
                                    'dirs': [x.name for x in items if x.is_dir()],
                                    'files': [x.name for x in items if x.is_file() and x.suffix.lower() in MEDIA]})
        if u.path == '/api/job':
            j = jobs[int(q['id'][0])]
            frac, eta = progress(j['log'])
            return self.reply(200, {'running': j['proc'].poll() is None, 'code': j['proc'].returncode,
                                    'log': j['log'][-300:], 'progress': frac, 'eta': eta, 'output': j['output']})
        if u.path == '/api/file':
            return self.send_file(Path(q['path'][0]))
        self.reply(404, {'error': 'not found'})

    def send_file(self, p):  # with byte ranges, so the browser can play and seek videos
        if not p.is_file() or p.suffix.lower() not in MEDIA: return self.reply(404, {'error': 'not found'})
        size = p.stat().st_size
        kind = {'.mp4': 'video/mp4', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
                '.wav': 'audio/wav', '.m4a': 'audio/mp4', '.flac': 'audio/flac'}.get(p.suffix.lower(), 'application/octet-stream')
        m = re.match(r'bytes=(\d*)-(\d*)', self.headers.get('Range', ''))
        a, b = (int(m[1] or 0), int(m[2] or size - 1)) if m else (0, size - 1)
        b = min(b, size - 1, a + (8 << 20) - 1)
        self.send_response(206 if m else 200)
        self.send_header('Content-Type', kind); self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(b - a + 1))
        if m: self.send_header('Content-Range', f'bytes {a}-{b}/{size}')
        self.end_headers()
        with open(p, 'rb') as f:
            f.seek(a); self.wfile.write(f.read(b - a + 1))

    def do_POST(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if not self.allowed(q): return self.reply(403, {'error': 'forbidden'})
        req = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
        if u.path == '/api/run':
            try:
                return self.reply(200, {'id': start(req)})
            except (ValueError, KeyError) as e:
                return self.reply(400, {'error': str(e)})
        if u.path == '/api/stop':
            j = jobs[int(req['id'])]
            if j['proc'].poll() is None: os.killpg(j['proc'].pid, signal.SIGTERM)  # ffmpeg children too
            return self.reply(200, {'ok': True})
        self.reply(404, {'error': 'not found'})


def serve(port=0, open_browser=True):
    Handler.token = secrets.token_urlsafe(16)
    srv = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    url = f'http://127.0.0.1:{srv.server_address[1]}/?t={Handler.token}'
    print(f'vhs-restore is open at {url}\n(keep this window open; Ctrl+C to quit)', flush=True)
    if open_browser: webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>vhs-restore</title><style>
:root{color-scheme:light dark;--line:#8884;--accent:#2f6fde}
body{font:15px/1.45 system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem}
h1{font-size:1.4rem;margin:.2rem 0 1rem} .tabs button{padding:.5rem .9rem;border:1px solid var(--line);background:none;border-radius:8px;margin-right:.3rem;cursor:pointer;font:inherit}
.tabs button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
fieldset{border:1px solid var(--line);border-radius:10px;margin:1rem 0;padding:.8rem 1rem} legend{font-weight:600;padding:0 .3rem}
label{display:block;margin:.35rem 0} input[type=text]{width:100%;box-sizing:border-box;padding:.45rem;font:inherit}
.row{display:flex;gap:.5rem} .row input{flex:1} button.main{padding:.6rem 1.2rem;font:inherit;border-radius:8px;border:0;background:var(--accent);color:#fff;cursor:pointer}
button.sec{padding:.45rem .8rem;font:inherit;border-radius:8px;border:1px solid var(--line);background:none;cursor:pointer}
small{opacity:.7} #browser{border:1px solid var(--line);border-radius:10px;padding:.5rem;max-height:320px;overflow:auto;display:none}
#browser div{padding:.2rem .4rem;cursor:pointer;border-radius:5px} #browser div:hover{background:#8882}
progress{width:100%;height:1.1rem} pre{background:#8881;padding:.6rem;border-radius:8px;max-height:240px;overflow:auto;font-size:12px;white-space:pre-wrap}
video,img{max-width:100%;border-radius:8px;margin-top:.5rem} [hidden]{display:none!important}
</style></head><body>
<h1>vhs-restore</h1>
<div class="tabs" role="tablist">
 <button data-cmd="video" class="on">Video</button><button data-cmd="audio">Sound only</button><button data-cmd="image">Photos</button><button data-cmd="stills">Best moments → photos</button>
</div>
<fieldset><legend>1. What to restore</legend>
 <div class="row"><input type="text" id="input" placeholder="Choose a file (or a folder of photos)"><button class="sec" onclick="browse()">Browse…</button></div>
 <div id="browser"></div>
 <label>Save as <input type="text" id="output"></label>
 <small>Your original is never changed.</small>
</fieldset>
<fieldset><legend>2. Options</legend>
 <div data-for="video stills image"><label><input type="checkbox" id="aspect"> People look too wide (4:3 tape captured as 16:9)</label>
  <label><input type="checkbox" id="crop"> Remove black borders automatically</label></div>
 <div data-for="video"><label>Result size <select id="height"><option value="2160">4K (2160 lines)</option><option value="1080">HD (1080 lines, ~4× faster)</option></select></label>
  <label><input type="checkbox" id="stabilize"> Steady a shaky picture</label>
  <label><input type="checkbox" id="dropouts"> Remove white specks and streaks (tape dropouts)</label></div>
 <div data-for="video audio"><label><input type="checkbox" id="mono" checked> Mono sound (most tapes are mono; twice as fast)</label>
  <label><input type="checkbox" id="declick"> Repair clicks and distortion (slower)</label>
  <label><input type="checkbox" id="speech"> Mostly talking: stronger voice clean-up (can thin out music)</label></div>
 <div data-for="stills"><label>One photo from every <input type="number" id="every" value="30" min="5" style="width:5em"> seconds</label></div>
 <div data-for="image stills"><label><input type="checkbox" id="ai"> AI sharpening (Real-ESRGAN): more detail, but invented; a few minutes per photo</label>
  <label><input type="checkbox" id="faces"> AI face restoration (GFPGAN): clearer faces, may change how people look</label></div>
 <div data-for="image"><label><input type="checkbox" id="colorize"> Add colour to black-and-white photos (AI)</label>
  <label><input type="checkbox" id="scratches"> Repair long scratches</label></div>
 <small data-for="image stills">AI models download once (up to 1.3 GB) the first time they are used.</small>
</fieldset>
<div class="row" style="align-items:center">
 <button class="main" onclick="run(false)">Start</button>
 <span data-for="video"><button class="sec" onclick="run(true)">Preview 20 s first</button> at <input type="text" id="at" value="10:00" style="width:5em"></span>
 <button class="sec" id="stop" onclick="stop()" hidden>Stop</button>
</div>
<section id="status" hidden><fieldset><legend>Progress</legend><progress id="bar"></progress> <span id="eta"></span>
 <div id="result"></div><details><summary>Details</summary><pre id="log"></pre></details></fieldset></section>
<script>
const T=new URLSearchParams(location.search).get('t'), $=id=>document.getElementById(id);
let cmd='video', job=null, timer=null;
const ext={video:'_restored.mp4',audio:'_clean.wav',image:'_restored',stills:'_best_photos'};
function api(path,body){const u=path+(path.includes('?')?'&':'?')+'t='+T;return fetch(u,body?{method:'POST',body:JSON.stringify(body)}:{}).then(r=>r.json())}
function suggest(){const i=$('input').value;if(!i)return;const b=i.replace(/\.[^/.]+$/,'');$('output').value=b+ext[cmd]}
function show(){document.querySelectorAll('[data-for]').forEach(e=>e.hidden=!e.dataset.for.split(' ').includes(cmd));suggest()}
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('on'));b.classList.add('on');cmd=b.dataset.cmd;show()});
$('input').onchange=suggest;
function browse(path){api('/api/ls?path='+encodeURIComponent(path||$('input').value||'~')).then(d=>{if(d.error)return alert(d.error);const B=$('browser');B.style.display='block';B.innerHTML='';
 const add=(t,f)=>{const e=document.createElement('div');e.textContent=t;e.onclick=f;B.appendChild(e)};
 add('⬆ '+d.path,()=>browse(d.parent));if(cmd=='image')add('✔ Use this whole folder',()=>pick(d.path));
 d.dirs.forEach(n=>add('📁 '+n,()=>browse(d.path+'/'+n)));d.files.forEach(n=>add('🎞 '+n,()=>pick(d.path+'/'+n)))})}
function pick(p){$('input').value=p;$('browser').style.display='none';suggest()}
function options(prev){const o={};['mono','declick','speech','stabilize','dropouts','colorize','scratches'].forEach(k=>{if(!$(k).closest('[data-for]').hidden&&$(k).checked)o[k]=true});
 if($('aspect').checked&&cmd!='audio')o.aspect='4:3';if($('crop').checked&&cmd!='audio')o.crop='auto';if($('faces').checked&&(cmd=='image'||cmd=='stills'))o.faces='0.6';if($('ai').checked&&(cmd=='image'||cmd=='stills'))o.ai='0.5';
 if(cmd=='video')o.height=$('height').value;if(cmd=='stills')o.every=$('every').value;if(prev)o.preview=$('at').value;return o}
function run(prev){let out=$('output').value;if(!$('input').value||!out)return alert('Choose what to restore first');
 if(prev)out=out.replace(/(\.[^/.]+)?$/,'_preview.mp4');
 api('/api/run',{cmd,input:$('input').value,output:out,options:options(prev)}).then(d=>{if(d.error)return alert(d.error);job=d.id;$('status').hidden=false;$('stop').hidden=false;$('result').innerHTML='';poll()})}
function stop(){api('/api/stop',{id:job})}
function poll(){api('/api/job?id='+job).then(d=>{$('log').textContent=d.log.join('\n');$('log').scrollTop=1e9;
 if(d.progress!=null){$('bar').value=d.progress}else $('bar').removeAttribute('value');$('eta').textContent=d.eta;
 if(d.running){timer=setTimeout(poll,1500);return}$('stop').hidden=true;$('bar').value=d.code==0?1:0;
 if(d.code!=0){$('eta').textContent='Stopped or failed - see details';return}$('eta').textContent='Done: '+d.output;
 const f='/api/file?t='+T+'&path='+encodeURIComponent(d.output);
 if(/\.mp4$/.test(d.output))$('result').innerHTML='<video controls src="'+f+'"></video>';
 else if(/\.(wav|m4a|flac)$/.test(d.output))$('result').innerHTML='<audio controls src="'+f+'"></audio>'})}
show();
</script></body></html>'''
