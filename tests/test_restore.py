# Run with:  python -m pytest tests   (or just: python tests/test_restore.py)
import os, re, subprocess, sys, tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vhs_restore.cli import main
from vhs_restore.core import DROPOUTS, detect_crop, detect_interlace, probe
from vhs_restore.debuzz import debuzz
from vhs_restore.gui import argv

SR = 48000


def level(x, hz):
    """dB power at exactly `hz` (0.1 Hz resolution)."""
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    b = int(round(hz * len(x) / SR))
    return 10 * np.log10(p[b - 2:b + 3].max())


def test_debuzz_removes_hum_keeps_music():
    rng = np.random.default_rng(0)
    t = np.arange(40 * SR) / SR
    # "music": a new random note every 0.25 s, so nothing in it is steady
    notes = rng.uniform(200, 2000, len(t) // (SR // 4) + 1)[(t * 4).astype(int)]
    music = 0.3 * np.sin(2 * np.pi * np.cumsum(notes) / SR)
    noise = 0.02 * rng.standard_normal(len(t))  # tape hiss
    buzz = sum(0.03 / k * np.sin(2 * np.pi * 50 * k * t) for k in range(1, 8)) + 0.01 * np.sin(2 * np.pi * 15625 * t)
    ref = (music + noise).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        src, dst = Path(d) / 'in.f32', Path(d) / 'out.f32'
        (ref + buzz).astype(np.float32).tofile(src)
        debuzz(src, dst, lowpass=0, log=lambda m: None)
        out = np.fromfile(dst, np.float32)
    mid = slice(5 * SR, 35 * SR)
    for hz in (50, 100, 150, 250, 15625):
        drop = level((ref + buzz)[mid], hz) - level(out[mid], hz)
        assert drop > 15, f"{hz} Hz only {drop:.1f} dB quieter"
    band = lambda y: (np.abs(np.fft.rfft(y[mid])) ** 2)[200 * 30:2000 * 30].sum()  # where the music is
    change = 10 * np.log10(band(out) / band(ref))
    assert abs(change) < 0.5, f"music changed by {change:.2f} dB"


def test_cli_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        src = d / 'tape.mp4'
        # a 4:3 picture stretched to 16:9 with 50 Hz hum on the sound, like a typical capture
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'gradients=s=320x240:r=25:d=24:speed=0.05',
                        '-f', 'lavfi', '-i', 'sine=f=50:d=24', '-vf', 'scale=426:240,setsar=1', '-shortest',
                        '-c:v', 'libx264', '-c:a', 'aac', str(src)], check=True)
        main(['video', str(src), str(d / 'out.mp4'), '--aspect', '4:3', '--height', '480', '--chunk', '10', '--jobs', '2'])
        dur, v, a = probe(d / 'out.mp4')
        assert (v['width'], v['height']) == (640, 480) and a and abs(dur - 24) < 0.2, (v, dur)
        assert not (d / 'out.mp4.work').exists()

        main(['audio', str(src), str(d / 'clean.wav')])
        assert probe(d / 'clean.wav')[2]

        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=s=300x200', '-frames:v', '1', str(d / 'old.png')], check=True)
        main(['image', str(d / 'old.png'), '-o', str(d / 'photos'), '--long-edge', '600'])
        _, v, _ = probe(d / 'photos' / 'old.jpg')
        assert (v['width'], v['height']) == (600, 400), v

        main(['stills', str(src), str(d / 'stills'), '--every', '10', '--height', '480', '--aspect', '4:3'])
        assert len(list((d / 'stills').glob('*.jpg'))) == 3

        try:
            main(['audio', str(src), str(src)])
            assert False, "overwrote the original"
        except SystemExit:
            pass


def lavfi(dst, video, *extra):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', video, *extra, str(dst)], check=True)


def test_tape_repairs():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # interlaced PAL: 50 half-pictures a second woven into 25 frames -> detected, and restored at 50 fps
        lavfi(d / 'il.mp4', 'testsrc2=s=720x576:r=50:d=6', '-vf', 'tinterlace=interleave_top,setfield=tff',
              '-c:v', 'libx264', '-flags', '+ildct+ilme')
        assert detect_interlace(d / 'il.mp4', 6) == 'tff'
        lavfi(d / 'prog.mp4', 'testsrc2=s=720x576:r=25:d=6', '-c:v', 'libx264')
        assert detect_interlace(d / 'prog.mp4', 6) is None
        main(['video', str(d / 'il.mp4'), str(d / 'il_out.mp4'), '--height', '480', '--no-audio'])
        assert probe(d / 'il_out.mp4')[1]['avg_frame_rate'] == '50/1'

        # black capture borders are found
        lavfi(d / 'border.mp4', 'testsrc2=s=640x480:r=25:d=2', '-vf', 'pad=720:576:40:48:black', '-c:v', 'libx264')
        assert detect_crop(d / 'border.mp4', 2) == '640:480:40:48', detect_crop(d / 'border.mp4', 2)

        # every option together still renders, and --preview makes a before|after clip
        lavfi(d / 'tape.mp4', 'gradients=s=426x240:r=25:d=12:speed=0.05', '-f', 'lavfi', '-i', 'sine=f=50:d=12',
              '-shortest', '-c:v', 'libx264', '-c:a', 'aac')
        main(['video', str(d / 'tape.mp4'), str(d / 'all.mp4'), '--height', '480', '--crop', 'auto', '--dropouts',
              '--stabilize', '--chroma-shift', '2', '--declick', '--speech', '--mono', '--chunk', '6'])
        dur, v, a = probe(d / 'all.mp4')
        assert v['height'] == 480 and a and abs(dur - 12) < 0.2
        main(['video', str(d / 'tape.mp4'), str(d / 'pv.mp4'), '--height', '480', '--preview', '0:02', '--preview-length', '5'])
        dur, v, a = probe(d / 'pv.mp4')
        assert v['width'] == 852 * 2 and a and abs(dur - 5) < 0.2, (v, dur)  # before | after, each 16:9 at 480


def test_dropouts_removed_motion_kept():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        lavfi(d / 'm.mkv', 'testsrc2=s=720x576:r=25:d=2', '-c:v', 'ffv1')
        raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(d / 'm.mkv'), '-pix_fmt', 'gray', '-f', 'rawvideo', '-'],
                             capture_output=True, check=True).stdout
        clean = np.frombuffer(raw, np.uint8).reshape(-1, 576, 720)
        dirty = clean.copy()
        for f in dirty:  # white horizontal streaks, 1-2 lines tall, one frame each
            for _ in range(4):
                y, x = rng.integers(0, 574), rng.integers(0, 520)
                f[y:y + rng.integers(1, 3), x:x + rng.integers(30, 200)] = 245
        for name, arr in (('clean', clean), ('dirty', dirty)):
            subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', '720x576', '-r', '25',
                            '-i', '-', '-pix_fmt', 'yuv420p', '-c:v', 'ffv1', str(d / f'{name}.mkv')], input=arr.tobytes(), check=True)

        def psnr(a, graph=None):
            g = f'[0:v]{graph}[a];[a][1:v]psnr' if graph else '[0:v][1:v]psnr'
            e = subprocess.run(['ffmpeg', '-hide_banner', '-i', str(d / a), '-i', str(d / 'clean.mkv'), '-filter_complex', g,
                                '-f', 'null', '-'], capture_output=True, text=True).stderr
            return float(re.search(r'y:(\S+)', e)[1])
        assert psnr('dirty.mkv', DROPOUTS) > psnr('dirty.mkv') + 8   # streaks mostly gone
        assert psnr('clean.mkv', DROPOUTS) > 50                     # moving picture barely touched


def test_gui_only_runs_whitelisted_commands():
    a = argv({'cmd': 'video', 'input': 'in.mp4', 'output': 'out.mp4', 'options': {'aspect': '4:3', 'mono': True, 'ai': False}})
    assert a[3:] == ['vhs_restore', 'video', 'in.mp4', 'out.mp4', '--aspect', '4:3', '--mono']
    for bad in ({'cmd': 'rm'}, {'cmd': 'video', 'options': {'aspect': '4:3; rm -rf ~'}}):
        try:
            argv({'input': 'a', 'output': 'b', **bad}); assert False, bad
        except ValueError:
            pass


def test_scratches():
    try:
        from vhs_restore import ai; ai.need()
    except SystemExit:
        return print("skipped: pip install '.[ai]' to test scratch repair")
    import cv2
    rng = np.random.default_rng(2)
    img = cv2.GaussianBlur(rng.integers(60, 200, (600, 800, 3), dtype=np.uint8), (0, 0), 6)  # soft "photo"
    s = img.copy()
    cv2.line(s, (50, 40), (700, 560), (240, 240, 240), 2); cv2.line(s, (600, 30), (150, 500), (20, 20, 20), 2)
    err = lambda a: np.abs(a.astype(int) - img.astype(int)).mean()
    assert err(ai.scratches(s, log=lambda m: None)) < err(s) / 3
    assert err(ai.scratches(img, log=lambda m: None)) < 0.05      # a clean photo is left alone


def test_ai_models():  # big downloads, so only on request:  VHS_RESTORE_AI_TEST=1 python -m pytest tests
    if not os.environ.get('VHS_RESTORE_AI_TEST'): return
    import cv2
    from vhs_restore import ai
    img = cv2.GaussianBlur(np.random.default_rng(3).integers(0, 255, (64, 80, 3), dtype=np.uint8), (0, 0), 2)
    assert ai.upscale(img, log=lambda m: None).shape == (256, 320, 3)
    g = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    assert ai.is_gray(g) and not ai.is_gray(ai.colorize(g, log=lambda m: None))
    assert ai.faces(img, log=lambda m: None).shape == img.shape  # no face: unchanged size, no crash


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'): fn(); print(name, 'ok')
    print("all good")
