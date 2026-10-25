# Run with:  python -m pytest tests   (or just: python tests/test_restore.py)
import subprocess, sys, tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vhs_restore.cli import main
from vhs_restore.core import probe
from vhs_restore.debuzz import debuzz

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


if __name__ == '__main__':
    test_debuzz_removes_hum_keeps_music()
    test_cli_end_to_end()
    print("all good")
