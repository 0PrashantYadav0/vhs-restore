# Shared pieces: ffmpeg helpers, the picture "look" (filter chain) and the audio clean-up chain.
import json, os, subprocess
from fractions import Fraction
from pathlib import Path

from .debuzz import debuzz

DENOISE = {'off': None, 'light': '3:4:6:8', 'medium': '5:8:10:16', 'strong': '7:10:14:20'}
HISS = "afftdn=nr=20:nf=-30:tn=1"
VOICE = "equalizer=f=300:t=q:w=1:g=-2,equalizer=f=2800:t=q:w=1:g=3,acompressor=threshold=-22dB:ratio=2.5:attack=15:release=250:makeup=2"


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


def aspect_of(v, crop=None, aspect=None):
    """Picture shape the result should have: --aspect if given, else what the file says (incl. non-square pixels)."""
    if aspect: return ratio(aspect)
    w, h = (int(n) for n in crop.split(':')[:2]) if crop else (v['width'], v['height'])
    sar = v.get('sample_aspect_ratio', '1:1')
    sar = Fraction(sar.replace(':', '/')) if sar not in ('N/A', '0:1', '') else Fraction(1)
    return float(w * sar / h)


def look(v, height, *, crop=None, aspect=None, work_height=720, denoise='medium', color=True):
    """ffmpeg -vf chain: crop, fix aspect, clean at the source's real resolution, colour, sharpen, upscale to `height`.
    work_height: VHS only holds ~720 lines of real detail, so cleaning happens there (fast, and averages noise away)."""
    a = aspect_of(v, crop, aspect)
    src_h = int(crop.split(':')[1]) if crop else v['height']
    wh = min(src_h, work_height) if work_height else src_h
    f = [f'crop={crop}'] if crop else []
    f.append(f'scale={even(wh * a)}:{even(wh)}:flags=area')
    if DENOISE[denoise]:
        f.append(f'hqdn3d={DENOISE[denoise]}')
    if color:  # half-strength auto white balance, gentle S-curve, a bit more contrast and colour (tape fades)
        f += ['split[n][g];[g]grayworld[w];[w][n]blend=all_mode=normal:all_opacity=0.5',
              "curves=all='0/0 0.2/0.22 0.5/0.53 0.85/0.86 1/0.96'", 'eq=contrast=1.12:saturation=1.3']
    f.append('unsharp=13:13:0.6:5:5:0')  # sharpen at the real resolution, before upscaling, so noise isn't magnified
    # upscale in two steps with contrast-adaptive sharpening in between: crisper than one big jump, no halos
    steps = [(height // 2, 0.9), (height, 0.5)] if height >= 2 * wh else [(height, 0.7)]
    for h, s in steps:
        f += [f'scale={even(h * a)}:{even(h)}:flags=lanczos', f'cas={s}']
    return ','.join(f + ['setsar=1'])


def video_codec(width, height):
    """Hardware HEVC on Macs (fast), otherwise x265. Bitrate scales with picture size (10 Mbit/s at 2880x2160)."""
    encoders = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True).stdout
    if 'hevc_videotoolbox' in encoders:
        mbps = max(2, round(10 * width * height / (2880 * 2160)))
        return ['-c:v', 'hevc_videotoolbox', '-b:v', f'{mbps}M', '-maxrate', f'{mbps * 1.6:.0f}M',
                '-bufsize', f'{mbps * 3.2:.0f}M', '-tag:v', 'hvc1']
    return ['-c:v', 'libx265', '-crf', '20', '-preset', 'medium', '-pix_fmt', 'yuv420p10le', '-tag:v', 'hvc1']


def clean_audio(src, dst, work, *, channels, mono=False, lowpass=12000, hiss=True, loudness=-14, log=print):
    """Debuzz -> hiss reduction -> voice clarity EQ + gentle compression -> two-pass loudness normalisation.
    Intermediate files live in `work` so an interrupted run picks up where it stopped."""
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    ch = 1 if mono else channels
    clean = work / 'clean.wav'
    if not clean.exists():
        raw, db = work / 'in.f32', work / 'db.f32'
        log("audio: decoding")
        ffmpeg('-i', src, '-vn', '-af', 'highpass=f=40', '-ac', ch, '-ar', 48000, '-f', 'f32le', raw)
        debuzz(raw, db, ch, lowpass, log=log)
        raw.unlink()
        log("audio: hiss reduction, EQ, compression")
        part = work / 'clean.part.wav'
        ffmpeg('-f', 'f32le', '-ar', 48000, '-ac', ch, '-i', db, '-af', (HISS + ',' if hiss else '') + VOICE, '-c:a', 'pcm_s24le', part)
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
