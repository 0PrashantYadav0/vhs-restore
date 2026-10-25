# vhs-restore

Restore old home videos and photos on your own computer:

- **Video**: denoise, fix colour and the picture shape, sharpen, and upscale to 4K (2880×2160 for 4:3 tapes, 3840×2160 for 16:9).
- **Sound**: remove buzz and hum (mains hum and its harmonics, head-switching buzz, motor rumble, the 15.6 kHz line whine), then reduce hiss, make voices clearer and even out the volume.
- **Photos**: clean up and recolour old photos or scans, and upscale them to 4K.
- **Stills**: find the best-looking moments in a long video and save them as restored 4K photos.

Your original file is only ever read. Every result goes to a new file, and the tool refuses to write over its input.

It was built to rescue a 2-hour VHS wedding tape and uses no AI models. Everything is classic signal processing on top of [ffmpeg](https://ffmpeg.org), so results are predictable and faces don't get invented details.

## Install

You need **Python 3.9+** and **ffmpeg** (version 5 or newer).

```sh
# macOS
brew install ffmpeg
# Ubuntu / Debian
sudo apt install ffmpeg
# Windows
winget install Gyan.FFmpeg
```

Then install the tool:

```sh
pip install git+https://github.com/0PrashantYadav0/vhs-restore
# or, from a downloaded copy:
pip install .
```

## Use

```sh
# whole tape -> 4K video with cleaned sound
vhs-restore video tape.mp4 tape_4K.mp4

# a 4:3 tape captured stretched to 16:9, with black borders: crop the borders, fix the shape
vhs-restore video tape.mp4 tape_4K.mp4 --crop 1904:1068:6:6 --aspect 4:3 --mono

# only fix the sound (from a video or an audio file)
vhs-restore audio tape.mp4 tape_sound.wav

# old photos or scans (files or whole folders) -> 4K
vhs-restore image scans/ -o restored/

# best moment from every 30 seconds of the tape, as restored 4K photos
vhs-restore stills tape.mp4 best_photos/ --aspect 4:3
```

Long videos are rendered in 5-minute chunks, several at a time. If the computer sleeps, crashes or restarts, **run the same command again** and it continues from the last finished chunk.

On a Mac, rendering uses the hardware HEVC encoder. A 2-hour VHS tape to 4K took about 9 hours on an 8 GB M1 Mac. Other systems use `libx265`, which is slower, so start with a short clip to check the look.

### Options

| Option | What it does |
|---|---|
| `--crop W:H:X:Y` | Crop before anything else, e.g. to remove black or noisy capture borders. |
| `--aspect 4:3` | The real shape of the picture. Use it when a 4:3 tape was captured as 16:9 and everyone looks wide. |
| `--denoise off/light/medium/strong` | How hard to clean noise (default `medium`). |
| `--no-color` | Skip the white-balance and colour boost, for black-and-white footage or colour that is already right. |
| `--height 2160` | Output height (video and stills). `1080` gives HD and renders about 4× faster. |
| `--long-edge 3840` | Long side of restored photos. |
| `--mono` | Mix the sound to mono first. Most tapes are mono anyway, and this halves the audio work. |
| `--lowpass 12000` | Cut everything above this frequency (tape hiss and the 15.6 kHz whine). `0` turns it off. |
| `--no-hiss` | Only remove buzz; keep the hiss reduction off. |
| `--every 30` | (stills) Keep the best frame from every this many seconds. |

Try a setting on a short piece first. This cuts out one minute without re-encoding it:

```sh
ffmpeg -ss 00:10:00 -t 60 -i tape.mp4 -c copy test.mp4
vhs-restore video test.mp4 test_4K.mp4 --aspect 4:3
```

## How it works

**Picture** (`vhs_restore/core.py`, `look()`):

1. Crop, then scale to the tape's *real* resolution (about 720 lines). Cleaning at that size is fast, and the downscale averages noise away.
2. Clean noise across space and time (`hqdn3d`).
3. Apply automatic white balance at half strength, so faded tape colour comes back without looking artificial. Then a gentle S-curve, plus a little more contrast and saturation.
4. Sharpen at the real resolution, so the noise isn't magnified along with the detail.
5. Upscale in two Lanczos steps, with contrast-adaptive sharpening after each one. This is crisper than one big jump and avoids halos.

**Sound** (`vhs_restore/debuzz.py`):

A tape's buzz is a set of steady tones whose frequencies drift and whose level changes along the tape, so fixed notch filters miss most of it. The debuzzer works in two passes:

1. It measures the spectrum in 1 Hz steps. For every 10-second block it finds what is *always* there, the 20th-percentile level of each frequency.
2. Any frequency that always sits clearly above its neighbours (±15 Hz) is buzz, and it is pushed down to the neighbours' level in that block.

Music and speech change all the time, so they pass through untouched. On the wedding tape, hum peaks up to 38 dB above the background ended up 0–5 dB above it, buried in the noise, while the music band (500 Hz–6 kHz) changed by 1.5 dB or less. After that come ffmpeg's FFT hiss reduction, a small EQ for voice clarity, gentle compression and two-pass loudness normalisation to −14 LUFS.

**Stills**: one frame per second is scored on sharpness, colour and exposure. Frames with a VHS tracking tear (a jagged horizontal band) are rejected. The best frame from each window is then rendered at full quality, with a short lead-in so the temporal denoiser has history to work with.

## Limits

- It cleans and sharpens; it does not invent detail. A 1980s tape upscaled to 4K looks like a very clean tape, not like footage shot on a modern camera. If you want that, AI upscalers (Real-ESRGAN, Topaz) can run on the output, at the risk of waxy, invented faces.
- Dropouts (white specks), tracking tears and photo scratches or dust are not repaired yet.
- Interlaced captures should be deinterlaced first, e.g. `ffmpeg -i in.mp4 -vf bwdif -c:a copy out.mp4`. Most capture software already does this.

## Develop

```sh
pip install -e . pytest
python -m pytest tests     # buzz-removal check + a run of every command on a tiny generated clip (~10 s)
```

Pull requests are welcome, especially dropout and scratch removal, automatic border detection, and before/after examples you are happy to share.

## License

MIT, see [LICENSE](LICENSE).
