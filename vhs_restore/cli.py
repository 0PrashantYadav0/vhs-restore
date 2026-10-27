"""vhs-restore: restore old videos (VHS, Hi8, Video8, early DV), their audio, and old photos - up to 4K.
The original file is only ever read; every result is written to a new file."""
import argparse, math, shutil, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .core import (aspect_of, clean_audio, cpu_jobs, detect_crop, detect_interlace, even, ffmpeg, guard, look, probe,
                   ratio, seconds, video_codec)


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def resolve(a, src, dur, v):
    """Turn 'auto' settings into concrete ones by looking at the source."""
    if a.crop == 'auto':
        a.crop = detect_crop(src, dur)
        log(f"borders: {'crop ' + a.crop if a.crop else 'none found'}")
    a.deint = None
    if getattr(a, 'deinterlace', 'off') != 'off':
        a.deint = detect_interlace(src, dur) or ('tff' if a.deinterlace == 'on' else None)
        log(f"interlaced: {'yes (' + a.deint + '), output gets double the frame rate' if a.deint else 'no'}")


def look_args(a, v, height, sharpen=True, color=True):
    return look(v, height, crop=a.crop, aspect=a.aspect, work_height=a.work_height, denoise=a.denoise,
                color=color and not a.no_color, sharpen=sharpen, deinterlace=getattr(a, 'deint', None),
                chroma_shift=a.chroma_shift, dropouts=getattr(a, 'dropouts', False), stabilize=getattr(a, 'stabilize', False))


def audio_args(a, au):
    return dict(channels=au['channels'], mono=a.mono, lowpass=a.lowpass, hiss=not a.no_hiss, repair=a.declick,
                speech=a.speech, log=log)


def cmd_video(a):
    guard(a.input, a.output)
    dur, v, au = probe(a.input)
    if not v: raise SystemExit("no video stream found")
    if not dur: raise SystemExit("can't read the video's length - remux it first: ffmpeg -i in -c copy fixed.mp4")
    resolve(a, a.input, dur, v)
    vf = look_args(a, v, a.height)
    width = even(a.height * aspect_of(v, a.crop, a.aspect))
    if a.preview is not None: return preview(a, v, au, vf, width)
    codec = video_codec(width, a.height)
    work = Path(str(a.output) + '.work'); work.mkdir(exist_ok=True)
    n = max(1, math.ceil(dur / a.chunk))
    log(f"{n} chunk(s) of {a.chunk}s, {a.jobs} at a time; work files in {work} (re-run the same command to resume)")
    t0, done, lock = time.time(), [0], threading.Lock()
    todo = [i for i in range(n) if not (work / f'c{i:05d}.mp4').exists()]

    def chunk(i):  # finished chunks are kept, so an interrupted run resumes
        part = work / f'c{i:05d}.part.mp4'
        ffmpeg('-ss', i * a.chunk, '-t', a.chunk, '-i', a.input, '-an', '-vf', vf, *codec, part)
        part.rename(work / f'c{i:05d}.mp4')
        with lock:
            done[0] += 1
            left = (time.time() - t0) / done[0] * (len(todo) - done[0])
            log(f"video: {n - len(todo) + done[0]}/{n} chunks done, about {left / 3600:.0f}h{left % 3600 / 60:02.0f}m left")

    audio = None
    with ThreadPoolExecutor(a.jobs) as pool, ThreadPoolExecutor(1) as apool:
        if au and not a.no_audio and not (work / 'audio.wav').exists():
            audio = apool.submit(clean_audio, a.input, work / 'audio.part.wav', work / 'audio', **audio_args(a, au))
        list(pool.map(chunk, todo))
        if audio: audio.result(); (work / 'audio.part.wav').rename(work / 'audio.wav')

    (work / 'list.txt').write_text(''.join(f"file 'c{i:05d}.mp4'\n" for i in range(n)))
    log("joining")
    has_audio = (work / 'audio.wav').exists()
    ffmpeg('-f', 'concat', '-safe', 0, '-i', work / 'list.txt', *(['-i', work / 'audio.wav'] if has_audio else []),
           '-map', '0:v', *(['-map', '1:a', '-c:a', 'aac', '-b:a', '256k', '-shortest'] if has_audio else []),
           '-c:v', 'copy', '-movflags', '+faststart', a.output)
    shutil.rmtree(work)
    log(f"done -> {a.output}")


def preview(a, v, au, vf, width):
    """A short before | after clip, to judge settings before a long render."""
    start, length = seconds(a.preview), a.preview_length
    ow = even(a.height * aspect_of(v))
    graph = f"[0:v]split[pv_src][pv_raw];[pv_src]{vf}[pv_new];[pv_raw]scale={ow}:{a.height},setsar=1[pv_old];[pv_old][pv_new]hstack"
    with tempfile.TemporaryDirectory() as tmp:
        snd = Path(tmp) / 'audio.wav'
        if au and not a.no_audio:
            clean_audio(a.input, snd, Path(tmp) / 'work', start=start, length=length, **audio_args(a, au))
        ffmpeg('-ss', start, '-t', length, '-i', a.input, *(['-i', snd] if snd.exists() else []),
               '-filter_complex', graph, *(['-map', '1:a', '-c:a', 'aac', '-b:a', '256k', '-shortest'] if snd.exists() else []),
               *video_codec(ow + width, a.height), '-movflags', '+faststart', a.output)
    log(f"preview (left: original, right: restored) -> {a.output}")


def cmd_audio(a):
    guard(a.input, a.output)
    _, _, au = probe(a.input)
    if not au: raise SystemExit("no audio stream found")
    work = Path(str(a.output) + '.work')
    clean_audio(a.input, a.output, work, **audio_args(a, au))
    shutil.rmtree(work)
    log(f"done -> {a.output}")


def ai_finish(a, png, dst, width, height):
    """Optional AI steps on a picture already cleaned by the normal chain, then save at the final size."""
    from . import ai
    img = ai.cv2.imread(str(png))
    if a.ai:  # blended with a plain upscale: full-strength AI looks painted on soft tape/print sources
        up = ai.upscale(img, log=log)
        img = ai.cv2.addWeighted(up, a.ai, ai.resize(img, up.shape[1], up.shape[0]), 1 - a.ai, 0)
    if a.faces: img = ai.faces(img, a.faces, log=log)
    ai.cv2.imwrite(str(dst), ai.resize(img, width, height), [ai.cv2.IMWRITE_JPEG_QUALITY, 95])


def cmd_image(a):
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    files = [f for p in map(Path, a.inputs) for f in (sorted(p.iterdir()) if p.is_dir() else [p])
             if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp')]
    if a.ai or a.faces or a.colorize or a.scratches:
        from . import ai; ai.need()
    crop = a.crop
    for i, f in enumerate(files, 1):
        dst = out / (f.stem + '.jpg')
        guard(f, dst)
        _, v, _ = probe(f)
        a.crop = crop; resolve(a, f, 0, v)  # 'auto' is worked out per photo
        asp = aspect_of(v, a.crop, a.aspect)
        height = a.long_edge if asp < 1 else even(a.long_edge / asp)  # long side = --long-edge
        with tempfile.TemporaryDirectory() as tmp:
            src, colorized = f, False
            if a.scratches or a.colorize:  # these repair the original picture, before any cleaning
                img = ai.cv2.imread(str(f))
                if a.scratches: img = ai.scratches(img, log=log)
                if a.colorize:
                    if ai.is_gray(img): img, colorized = ai.colorize(img, log=log), True
                    else: log(f"{f.name}: already in colour, not colourising")
                src = Path(tmp) / 'pre.png'; ai.cv2.imwrite(str(src), img)
            if a.ai or a.faces:
                png = Path(tmp) / 'clean.png'
                h0 = int(a.crop.split(':')[1]) if a.crop else v['height']  # AI upscales from the real size
                ffmpeg('-i', src, '-vf', look_args(a, v, h0 if a.ai else height, sharpen=not a.ai, color=not colorized),
                       '-frames:v', 1, png)
                ai_finish(a, png, dst, even(height * asp), height)
            else:
                ffmpeg('-i', src, '-vf', look_args(a, v, height, color=not colorized), '-frames:v', 1, '-q:v', 2, dst)
        log(f"{i}/{len(files)} {dst}")


def metrics(frames):
    """Per frame: sharpness, colourfulness, exposure error, VHS tracking-tear strength."""
    a = frames.astype(np.float32)
    g = a.mean(3)
    lap = g[:, 1:-1, 1:-1] * 4 - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:]
    sharp = np.log(lap.var((1, 2)) + 1)
    rg, yb = a[..., 0] - a[..., 1], (a[..., 0] + a[..., 1]) / 2 - a[..., 2]
    color = np.hypot(rg.std((1, 2)), yb.std((1, 2))) + 0.3 * np.hypot(rg.mean((1, 2)), yb.mean((1, 2)))
    m = g.mean((1, 2))
    exposure = abs(m - 115) / 115 + ((g < 8).mean((1, 2)) + (g > 248).mean((1, 2))) * 2
    rd = np.abs(np.diff(g, axis=1)).mean(2)
    tear = np.sort(rd, 1)[:, -3:].mean(1) / np.maximum(np.median(rd, 1), 1)  # clean <= 4, glitched > 10
    return np.stack([sharp, color, exposure, tear], 1)


def score(m):
    """'Nice photo' score from metrics(): sharp, colourful, well exposed; torn frames rejected (-inf)."""
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    return np.where(m[:, 3] > 6, -np.inf, z(m[:, 0]) + 0.6 * z(m[:, 1]) - 1.5 * z(m[:, 2]))


def cmd_stills(a):
    dur, v, _ = probe(a.input)
    if not v: raise SystemExit("no video stream found")
    if a.ai or a.faces:
        from . import ai; ai.need()
    resolve(a, a.input, dur, v)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    W, H = 320, 240
    log(f"scoring {a.rate} frame(s)/s of {dur / 60:.0f} min of video")
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', a.input, '-an', '-vf', f'fps={a.rate},scale={W}:{H}',
                          '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'], stdout=subprocess.PIPE)
    m = []
    while len(buf := p.stdout.read(W * H * 3 * 100)) >= W * H * 3:
        m.append(metrics(np.frombuffer(buf[:len(buf) // (W * H * 3) * W * H * 3], np.uint8).reshape(-1, H, W, 3)))
    if p.wait(): raise SystemExit("ffmpeg failed while reading the video")
    s = score(np.concatenate(m))
    best = {}
    for i, sc in enumerate(s):
        w = int(i / a.rate // a.every)
        if sc > -np.inf and (w not in best or sc > s[best[w]]): best[w] = i
    fps = ratio(v['avg_frame_rate']) if v.get('avg_frame_rate', '0/0') != '0/0' else 25.0
    fps *= 2 if a.deint else 1
    wh = min(int(a.crop.split(':')[1]) if a.crop else v['height'], a.work_height or 10 ** 9)
    asp = aspect_of(v, a.crop, a.aspect)
    vf = look_args(a, v, wh if a.ai else a.height, sharpen=not a.ai)
    for k, i in enumerate(sorted(best.values()), 1):
        t = i / a.rate
        start = max(t - 0.4, 0)  # a little lead-in so the temporal denoiser has history
        h, r = divmod(int(t), 3600)
        dst = out / f"{h}h{r // 60:02d}m{r % 60:02d}s.jpg"
        grab = lambda target, *q: ffmpeg('-ss', f'{start:.3f}', '-i', a.input, '-an', '-vf',
                                         f'{vf},select=eq(n\\,{round((t - start) * fps)})', '-frames:v', 1, *q, target)
        if a.ai or a.faces:
            with tempfile.TemporaryDirectory() as tmp:
                grab(Path(tmp) / 'f.png')
                ai_finish(a, Path(tmp) / 'f.png', dst, even(a.height * asp), a.height)
        else:
            grab(dst, '-q:v', 2)
        log(f"{k}/{len(best)} {dst}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog='vhs-restore', description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)

    pic = argparse.ArgumentParser(add_help=False)
    pic.add_argument('--crop', help="W:H:X:Y crop first, e.g. black capture borders (1904:1068:6:6), or 'auto' to find them")
    pic.add_argument('--aspect', help="real picture shape, e.g. 4:3 when a 4:3 tape was captured stretched to 16:9")
    pic.add_argument('--denoise', choices=['off', 'light', 'medium', 'strong'], default='medium')
    pic.add_argument('--no-color', action='store_true', help="skip white balance / colour boost (B&W or already-good colour)")
    pic.add_argument('--chroma-shift', type=int, default=0, metavar='PX',
                     help="move the colour this many pixels left (VHS colour often sits right of the picture; try 2-6)")
    tape = argparse.ArgumentParser(add_help=False)
    tape.add_argument('--deinterlace', choices=['auto', 'on', 'off'], default='auto',
                      help="turn interlaced video (two half-pictures per frame) into smooth double-frame-rate video (default auto)")
    tape.add_argument('--dropouts', action='store_true', help="remove white specks/streaks that flash for one frame (tape dropouts)")
    snd = argparse.ArgumentParser(add_help=False)
    snd.add_argument('--mono', action='store_true', help="mix to mono first (most tapes are mono; halves the work)")
    snd.add_argument('--lowpass', type=int, default=12000, help="cut everything above this many Hz, 0 = off (default 12000: tape hiss + 15.6 kHz whine)")
    snd.add_argument('--no-hiss', action='store_true', help="skip hiss reduction (debuzz only)")
    snd.add_argument('--declick', action='store_true', help="repair clicks, crackle and clipped (distorted) peaks; slow")
    snd.add_argument('--speech', action='store_true', help="neural speech denoiser instead of hiss reduction: best for talking, can thin out music")
    faces = argparse.ArgumentParser(add_help=False)
    faces.add_argument('--ai', type=float, nargs='?', const=0.5, default=0, metavar='STRENGTH',
                       help="AI upscaling (Real-ESRGAN) blended at STRENGTH 0-1 (default 0.5; higher looks painted); a few minutes per photo")
    faces.add_argument('--faces', type=float, nargs='?', const=0.6, default=0, metavar='STRENGTH',
                       help="AI face restoration (GFPGAN), blended at STRENGTH 0-1 (default 0.6; 1 = fully AI face)")

    p = sub.add_parser('video', parents=[pic, tape, snd], help="restore a whole video (picture + sound) to 4K")
    p.add_argument('input'); p.add_argument('output')
    p.add_argument('--height', type=int, default=2160, help="output height (default 2160 = 4K)")
    p.add_argument('--work-height', type=int, default=720, help="resolution the cleaning happens at (default 720; 0 = source)")
    p.add_argument('--stabilize', action='store_true', help="steady a shaky picture or tape wobble")
    p.add_argument('--preview', metavar='TIME', help="only render a short before|after clip starting at TIME (e.g. 10:00)")
    p.add_argument('--preview-length', type=float, default=20, metavar='SECONDS')
    p.add_argument('--jobs', type=int, default=cpu_jobs(), help="chunks rendered in parallel")
    p.add_argument('--chunk', type=int, default=300, help="chunk length in seconds (resume granularity)")
    p.add_argument('--no-audio', action='store_true')
    p.set_defaults(fn=cmd_video)

    p = sub.add_parser('audio', parents=[snd], help="remove buzz/hum/hiss from a video's or audio file's sound")
    p.add_argument('input'); p.add_argument('output', help="e.g. clean.wav, clean.flac or clean.m4a")
    p.set_defaults(fn=cmd_audio)

    p = sub.add_parser('image', parents=[pic, faces], help="restore old photos / scans and upscale them")
    p.add_argument('inputs', nargs='+', help="image files or folders")
    p.add_argument('-o', '--output', required=True, help="output folder")
    p.add_argument('--long-edge', type=int, default=3840, help="long side of the result in pixels (default 3840 = 4K)")
    p.add_argument('--scratches', action='store_true', help="find and paint over long scratches (not AI; only use on scratched photos)")
    p.add_argument('--colorize', action='store_true', help="AI colours for black-and-white photos (DDColor; colour photos are skipped)")
    p.set_defaults(fn=cmd_image, work_height=0)

    p = sub.add_parser('stills', parents=[pic, tape, faces], help="pick the best-looking moments of a video and save them as restored 4K photos")
    p.add_argument('input'); p.add_argument('output', help="output folder")
    p.add_argument('--every', type=float, default=30, help="keep the best frame from every this many seconds (default 30)")
    p.add_argument('--rate', type=float, default=1, help="frames per second to consider (default 1)")
    p.add_argument('--height', type=int, default=2160)
    p.add_argument('--work-height', type=int, default=720)
    p.set_defaults(fn=cmd_stills)

    p = sub.add_parser('gui', help="open the point-and-click app in your browser")
    p.add_argument('--port', type=int, default=0)
    p.add_argument('--no-browser', action='store_true')
    p.set_defaults(fn=lambda a: __import__('vhs_restore.gui', fromlist=['serve']).serve(a.port, not a.no_browser))

    a = ap.parse_args(argv)
    if not shutil.which('ffmpeg'): raise SystemExit("ffmpeg not found - install it first (see README)")
    a.fn(a)


if __name__ == '__main__':
    main()
