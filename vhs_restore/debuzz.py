# Adaptive buzz remover: finds steady tones (mains hum + harmonics, head-switching buzz, drifting motor
# rumble, line whine...) and pushes them down to the surrounding noise floor, leaving music and voices alone.
# Works on raw float32 @ 48 kHz (interleaved if more than one channel).
import numpy as np

SR = N = 48000          # 1 s window -> 1 Hz bins
HOP = N // 4            # 75% overlap
BLOCK = 40              # frames per noise estimate (10 s)
LO = 20                 # lowest bin (Hz) checked for tones


def debuzz(src, dst, channels=1, lowpass=12000, thresh=2.5, log=print):
    """src/dst: paths to raw float32 48 kHz audio. lowpass: Hz (0 = off). thresh: tone/neighbour power ratio."""
    x = np.memmap(src, np.float32, 'r').reshape(-1, channels)
    y = np.memmap(dst, np.float32, 'w+', shape=x.shape)
    for c in range(channels):
        _channel(x[:, c], y, c, lowpass, thresh, log)
    y.flush()


def _channel(x, y, c, lowpass, thresh, log):
    hi = min(lowpass, 20000) if lowpass else 20000
    win = np.hanning(N + 1)[:N].astype(np.float32)
    nfr = (len(x) + N - HOP) // HOP + 1     # frames start at -(N-HOP) so every sample gets full overlap
    nbl = (nfr + BLOCK - 1) // BLOCK

    def span(t):
        s = t * HOP - (N - HOP)
        return max(s, 0), min(s + N, len(x)), max(-s, 0)

    def frame(t):
        a, b, o = span(t)
        buf = np.zeros(N, np.float32)
        buf[o:o + b - a] = x[a:b]
        return np.fft.rfft(buf * win)

    # pass 1: per 10 s block, the 20th-percentile power of every bin = what is always there (buzz + floor)
    p20 = np.empty((nbl, hi - LO), np.float32)
    for b in range(nbl):
        p20[b] = np.percentile([np.abs(frame(t)[LO:hi]) ** 2 for t in range(b * BLOCK, min(nfr, (b + 1) * BLOCK))], 20, axis=0)
        if b % 50 == 0: log(f"debuzz ch{c}: measuring {b}/{nbl}")

    # buzz bins = always-there level pokes above the neighbours (+-15 Hz); widen by 1 Hz each side (lines wobble)
    tone, floor = np.zeros(p20.shape, bool), np.zeros_like(p20)
    for b in range(nbl):
        pn = np.median(p20[max(0, b - 1):b + 2], axis=0)  # 30 s, so one sustained note doesn't count as buzz
        floor[b] = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(pn, 15, mode='edge'), 31), axis=1)
        m = pn > thresh * floor[b]
        tone[b] = m | np.roll(m, 1) | np.roll(m, -1)
    log(f"debuzz ch{c}: {tone.mean() * 100:.1f}% of bins treated as buzz")

    # pass 2: push buzz bins down to the surrounding floor (an adaptive notch); everything else untouched
    f = np.fft.rfftfreq(N, 1 / SR)
    lp = np.sqrt(np.clip((lowpass + 500 - f) / 1000, 0, 1)) if lowpass else 1  # smooth 1 kHz-wide cut
    for t in range(nfr):
        X = frame(t)
        blk = min(t // BLOCK, nbl - 1)
        P = np.abs(X[LO:hi]) ** 2 + 1e-20
        X[LO:hi] *= np.where(tone[blk], np.sqrt(np.minimum(P, floor[blk]) / P), 1)
        X *= lp
        a, b, o = span(t)
        y[a:b, c] += (np.fft.irfft(X, N)[o:o + b - a] * win[o:o + b - a] / 1.5).astype(np.float32)  # hann^2 @75% sums to 1.5
        if t % 2000 == 0: log(f"debuzz ch{c}: cleaning {t}/{nfr}")
