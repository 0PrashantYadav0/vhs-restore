# Shared pieces: ffmpeg helpers, source analysis, the picture "look" (filter chain) and the audio clean-up chain.
import collections, hashlib, json, os, re, subprocess, urllib.request
from fractions import Fraction
from pathlib import Path

from .debuzz import debuzz

DENOISE = {'off': None, 'light': '3:4:6:8', 'medium': '5:8:10:16', 'strong': '7:10:14:20'}
HISS = "afftdn=nr=20:nf=-30:tn=1"
REPAIR = "adeclick,adeclip"
VOICE = "equalizer=f=300:t=q:w=1:g=-2,equalizer=f=2800:t=q:w=1:g=3,acompressor=threshold=-22dB:ratio=2.5:attack=15:release=250:makeup=2"
CACHE = Path(os.environ.get('VHS_RESTORE_CACHE', Path.home() / '.cache' / 'vhs-restore'))


def ffmpeg(*args):
    subprocess.run(['ffmpeg', '-hide_banner', '-v', 'error', '-y', *map(str, args)], check=True)


def probe(path):
    """-> (duration seconds, video stream dict or None, audio stream dict or None)"""
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                          'format=duration:stream=codec_type,width,height,sample_aspect_ratio,channels,avg_frame_rate',
                          '-of', 'json', str(path)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    pick = lambda kind: next((s for s in j['streams'] if s['codec_type'] == kind), None)
    return float(j['format'].get('duration') or 0), pick('video'), pick('audio')


def guard(src, dst):
    """Never let an output path point at the original."""
    if Path(src).resolve() == Path(dst).resolve():
        raise SystemExit(f"refusing to overwrite the original: {src}")


def even(v):
    return max(2, int(round(v / 2)) * 2)


def ratio(s):
    return float(Fraction(str(s).replace(':', '/')))


def seconds(s):
    """'1:02:03', '62:03' or '3723' -> 3723.0"""
    return sum(float(p) * 60 ** i for i, p in enumerate(reversed(str(s).split(':'))))


def fetch(name, url, sha256, log=print):
    """Download a model file once into the cache and check it."""
    dst = CACHE / name
    if dst.exists(): return dst
    CACHE.mkdir(parents=True, exist_ok=True)
    log(f"downloading {name} (one time) from {url}")
    part = dst.with_suffix('.part')
    h = hashlib.sha256()
    with urllib.request.urlopen(url) as r, open(part, 'wb') as f:
        while chunk := r.read(1 << 20):
            f.write(chunk); h.update(chunk)
    if h.hexdigest() != sha256:
        part.unlink(); raise SystemExit(f"{name}: download corrupted (checksum mismatch), try again")
    part.rename(dst)
    return dst


def aspect_of(v, crop=None, aspect=None):
    """Picture shape the result should have: --aspect if given, else what the file says (incl. non-square pixels)."""
    if aspect: return ratio(aspect)
    w, h = (int(n) for n in crop.split(':')[:2]) if crop else (v['width'], v['height'])
    sar = v.get('sample_aspect_ratio', '1:1')
    sar = Fraction(sar.replace(':', '/')) if sar not in ('N/A', '0:1', '') else Fraction(1)
    return float(w * sar / h)


def _samples(dur, n=5):
    return [dur * (i + 1) / (n + 1) for i in range(n)] if dur > 30 else [0]


def detect_interlace(src, dur):
    """'tff' / 'bff' if the picture is interlaced (two half-pictures per frame), else None."""
    count = collections.Counter()
    for t in _samples(dur):
        err = subprocess.run(['ffmpeg', '-hide_banner', '-ss', str(t), '-i', str(src), '-an', '-vf', 'idet',
                              '-frames:v', '200', '-f', 'null', '-'], capture_output=True, text=True).stderr
        for line in err.splitlines():
            if 'Multi frame detection' in line:
                for k, n in re.findall(r'(TFF|BFF|Progressive):\s*(\d+)', line): count[k] += int(n)
    order = max(('TFF', 'BFF'), key=count.get)
    other = count['BFF' if order == 'TFF' else 'TFF']
    # real interlacing is consistently one field order; mixed votes come from fine moving detail in progressive video
    return order.lower() if count[order] > 3 * (other + count['Progressive']) else None


def detect_crop(src, dur):
    """Black borders -> 'W:H:X:Y' covering the picture in every sampled scene (dark scenes can't over-crop)."""
    boxes = []
    for t in _samples(dur, 8):
        err = subprocess.run(['ffmpeg', '-hide_banner', '-ss', str(t), '-i', str(src), '-an', '-vf',
                              'cropdetect=limit=24:round=2:reset=0', '-frames:v', '50', '-f', 'null', '-'],
                             capture_output=True, text=True).stderr
        found = re.findall(r'crop=(\d+):(\d+):(\d+):(\d+)', err)
        if found: boxes.append(tuple(map(int, found[-1])))
    if not boxes: return None
    x0, y0 = min(b[2] for b in boxes), min(b[3] for b in boxes)
    x1, y1 = max(b[2] + b[0] for b in boxes), max(b[3] + b[1] for b in boxes)
    return f'{x1 - x0}:{y1 - y0}:{x0}:{y0}'


# tape dropouts: white horizontal streaks, 1-2 lines tall, that flash for one frame. Where a pixel is much brighter
# than the median of its neighbouring frames AND that bright area is a thin horizontal run, take the median instead.
# (Brightness only; moving objects are rarely thin, horizontal and one frame long.)
DROPOUTS = ('split[d0][d1];[d1]tmedian=radius=1,split[m0][m1];[d0]split[o0][o1];'
            "[o0][m0]blend=all_mode=subtract,lutyuv=y='if(gt(val,60),255,0)':u=0:v=0,"
            + 'erosion=coordinates=24,' * 15 + 'dilation=coordinates=24,' * 15 +  # keep runs >= 31 px (not sparkles)
            'split[k0][k1];'
            '[k1]erosion=coordinates=66,dilation=coordinates=66[tall];'                # ...minus anything >= 3 px tall
            '[k0][tall]blend=all_mode=subtract,dilation[thin];[o1][m1][thin]maskedmerge')


def look(v, height, *, crop=None, aspect=None, work_height=720, denoise='medium', color=True, sharpen=True,
         deinterlace=None, chroma_shift=0, dropouts=False, stabilize=False):
    """ffmpeg -vf chain: deinterlace, crop, repair, fix aspect, clean at the source's real resolution, colour, sharpen,
    upscale to `height`. work_height: VHS only holds ~720 lines of real detail, so cleaning happens there."""
    a = aspect_of(v, crop, aspect)
    src_h = int(crop.split(':')[1]) if crop else v['height']
    wh = min(src_h, work_height) if work_height else src_h
    f = []
    if deinterlace:  # each half-picture becomes a full frame: 25i -> 50p, smooth real motion
        f.append(f'bwdif=mode=send_field:parity={deinterlace}:deint=all')
    if chroma_shift:  # VHS colour often sits a few pixels right of the picture
        f.append(f'chromashift=cbh={-chroma_shift}:crh={-chroma_shift}')
    if crop: f.append(f'crop={crop}')
    if dropouts: f.append(DROPOUTS)
    f.append(f'scale={even(wh * a)}:{even(wh)}:flags=area')
    if DENOISE[denoise]:
        f.append(f'hqdn3d={DENOISE[denoise]}')
    if stabilize: f.append('deshake')
    if color:  # half-strength auto white balance, gentle S-curve, a bit more contrast and colour (tape fades)
        f += ['split[n][g];[g]grayworld[w];[w][n]blend=all_mode=normal:all_opacity=0.5',
              "curves=all='0/0 0.2/0.22 0.5/0.53 0.85/0.86 1/0.96'", 'eq=contrast=1.12:saturation=1.3']
    if sharpen:
        f.append('unsharp=13:13:0.6:5:5:0')  # sharpen at the real resolution, before upscaling, so noise isn't magnified
        # upscale in two steps with contrast-adaptive sharpening in between: crisper than one big jump, no halos
        steps = [(height // 2, 0.9), (height, 0.5)] if height >= 2 * wh else [(height, 0.7)]
        for h, s in steps:
            f += [f'scale={even(h * a)}:{even(h)}:flags=lanczos', f'cas={s}']
    elif height != wh:
        f.append(f'scale={even(height * a)}:{even(height)}:flags=lanczos')
    return ','.join(f + ['setsar=1'])


def video_codec(width, height):
    """Hardware HEVC on Macs (fast), otherwise x265. Bitrate scales with picture size (10 Mbit/s at 2880x2160)."""
    encoders = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True).stdout
    if 'hevc_videotoolbox' in encoders:
        mbps = max(2, round(10 * width * height / (2880 * 2160)))
        return ['-c:v', 'hevc_videotoolbox', '-b:v', f'{mbps}M', '-maxrate', f'{mbps * 1.6:.0f}M',
                '-bufsize', f'{mbps * 3.2:.0f}M', '-tag:v', 'hvc1']
    return ['-c:v', 'libx265', '-crf', '20', '-preset', 'medium', '-pix_fmt', 'yuv420p10le', '-tag:v', 'hvc1']


RNNOISE = ('rnnoise-sh.rnnn', 'https://raw.githubusercontent.com/GregorR/rnnoise-models/master/somnolent-hogwash-2018-09-01/sh.rnnn',
           '70bb6685eb0c2a1d18e2918dca3fbfbd39317010b1802eb1b6ea73a92f3fdec0')


def clean_audio(src, dst, work, *, channels, mono=False, lowpass=12000, hiss=True, repair=False, speech=False,
                loudness=-14, start=None, length=None, log=print):
    """Debuzz -> [click/clip repair] -> hiss reduction [or speech denoiser] -> voice EQ + compression -> loudness.
    Intermediate files live in `work` so an interrupted run picks up where it stopped."""
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    ch = 1 if mono else channels
    clean = work / 'clean.wav'
    if not clean.exists():
        raw, db = work / 'in.f32', work / 'db.f32'
        log("audio: decoding")
        window = (['-ss', start] if start is not None else []) + (['-t', length] if length else [])
        ffmpeg(*window, '-i', src, '-vn', '-af', 'highpass=f=40', '-ac', ch, '-ar', 48000, '-f', 'f32le', raw)
        debuzz(raw, db, ch, lowpass, log=log)
        raw.unlink()
        chain = [REPAIR] if repair else []
        if speech:  # neural speech denoiser (RNNoise): best for talking, can thin out music
            model = fetch(*RNNOISE, log=log).as_posix().replace(':', r'\:')
            chain.append(f"arnndn=m='{model}':mix=0.8")
        elif hiss:
            chain.append(HISS)
        log("audio: " + ", ".join(f.split('=')[0] for f in chain) + ", EQ, compression")
        part = work / 'clean.part.wav'
        ffmpeg('-f', 'f32le', '-ar', 48000, '-ac', ch, '-i', db, '-af', ','.join(chain + [VOICE]), '-c:a', 'pcm_s24le', part)
        part.rename(clean); db.unlink()
    log("audio: loudness")
    target = f'loudnorm=I={loudness}:TP=-1.5:LRA=11'
    t = subprocess.run(['ffmpeg', '-hide_banner', '-i', str(clean), '-af', target + ':print_format=json', '-f', 'null', '-'],
                       capture_output=True, text=True, check=True).stderr
    m = json.JSONDecoder().raw_decode(t[t.rindex('{'):])[0]
    measured = (f"measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
                f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    ffmpeg('-i', clean, '-af', f'{target}:{measured}', '-ar', 48000, dst)


def cpu_jobs():
    return max(1, min(4, (os.cpu_count() or 2) // 2))
