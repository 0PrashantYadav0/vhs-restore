# vhs-restore

Restore old home videos and photos on your own computer:

- **Video**: deinterlace, denoise, repair tape damage, fix colour and the picture shape, sharpen, and upscale to 4K (2880×2160 for 4:3 tapes, 3840×2160 for 16:9).
- **Sound**: remove buzz and hum (mains hum and its harmonics, head-switching buzz, motor rumble, the 15.6 kHz line whine), then reduce hiss, repair clicks, make voices clearer and even out the volume.
- **Photos**: clean up and recolour old photos or scans, repair scratches, and upscale to 4K. Optional AI can sharpen detail, restore faces and colour black-and-white photos.
- **Stills**: find the best-looking moments in a long video and save them as restored 4K photos.

Your original file is only ever read. Every result goes to a new file, and the tool refuses to write over its input.

It was built to rescue a 2-hour VHS wedding tape. By default it uses classic signal processing on top of [ffmpeg](https://ffmpeg.org), so results are predictable and nobody's face gets invented details. The AI options are separate and off unless you ask for them.

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
pip install git+https://github.com/0PrashantYadav0/vhs-restore            # (on PyPI soon: pip install vhs-restore)
pip install "vhs-restore[ai] @ git+https://github.com/0PrashantYadav0/vhs-restore"   # with the AI photo options
```

## The app

```sh
vhs-restore gui
```

This opens a simple page in your browser. Pick a file, tick the options in plain words, and use **Preview 20 s first** to check the result before the long render. Progress and time left are shown as it runs. The page only works on your own computer, behind a private link printed in the terminal.

## Command line

```sh
# whole tape -> 4K video with cleaned sound
vhs-restore video tape.mp4 tape_4K.mp4

# typical VHS capture: 4:3 picture stretched to 16:9, mono sound, white dropout streaks
vhs-restore video tape.mp4 tape_4K.mp4 --aspect 4:3 --crop auto --mono --dropouts

# check the settings on 20 seconds first: makes a before|after clip
vhs-restore video tape.mp4 check.mp4 --aspect 4:3 --preview 10:00

# only fix the sound (from a video or an audio file)
vhs-restore audio tape.mp4 tape_sound.wav --declick

# old photos or scans (files or whole folders) -> 4K, scratches repaired
vhs-restore image scans/ -o restored/ --scratches

# same, plus AI: sharper detail, restored faces, colour for black-and-white photos
vhs-restore image scans/ -o restored/ --ai --faces --colorize

# best moment from every 30 seconds of the tape, as restored 4K photos
vhs-restore stills tape.mp4 best_photos/ --aspect 4:3
```

Long videos are rendered in 5-minute chunks, several at a time. If the computer sleeps, crashes or restarts, **run the same command again** and it continues from the last finished chunk.

On a Mac, rendering uses the hardware HEVC encoder. A 2-hour VHS tape to 4K took about 9 hours on an 8 GB M1 Mac. Other systems use `libx265`, which is slower. Use `--preview` to check the look first.

### Options

**Picture** (video, stills, photos)

| Option | What it does |
|---|---|
| `--aspect 4:3` | The real shape of the picture. Use it when a 4:3 tape was captured as 16:9 and everyone looks wide. |
| `--crop auto` or `--crop W:H:X:Y` | Remove black capture borders: found automatically, or given exactly. |
| `--deinterlace auto/on/off` | (video, stills) Turns interlaced video, which stores two half-pictures per frame, into smooth video at twice the frame rate. Detected automatically. |
| `--dropouts` | (video, stills) Remove white streaks that flash for one frame (tape dropouts). |
| `--stabilize` | (video) Steady a shaky picture or tape wobble. |
| `--chroma-shift 3` | Move colour that sits a few pixels to the right of the picture back into place. |
| `--denoise off/light/medium/strong` | How hard to clean noise (default `medium`). |
| `--no-color` | Skip the white-balance and colour boost, for black-and-white footage or colour that is already right. |
| `--height 2160` | Output height (video and stills). `1080` gives HD and renders about 4× faster. |
| `--long-edge 3840` | (photos) Long side of the result. |
| `--preview 10:00` | (video) Render only a short before|after clip starting there (`--preview-length 20`). |

**Sound** (video, audio)

| Option | What it does |
|---|---|
| `--mono` | Mix the sound to mono first. Most tapes are mono anyway, and this halves the work. |
| `--declick` | Repair clicks, crackle and clipped (distorted) peaks. Slower. |
| `--speech` | Use a neural speech denoiser (RNNoise) instead of the hiss filter. Best for talking; can thin out music. |
| `--lowpass 12000` | Cut everything above this frequency (tape hiss and the 15.6 kHz whine). `0` turns it off. |
| `--no-hiss` | Only remove buzz; keep the hiss reduction off. |

**Photos** (image, and `--ai`/`--faces` for stills)

| Option | What it does |
|---|---|
| `--scratches` | Find long scratches and paint them in from their surroundings. Not AI. Only use on scratched photos. |
| `--ai [0.5]` | AI upscaling (Real-ESRGAN), blended with a plain upscale at this strength. Higher values look painted. Takes a few minutes per photo. |
| `--faces [0.6]` | AI face restoration (GFPGAN), blended in at this strength. Faces too small to work from are left alone. |
| `--colorize` | AI colours for black-and-white photos (DDColor). Colour photos are skipped. |
| `--every 30` | (stills) Keep the best frame from every this many seconds. |

The AI options need the `[ai]` extra (see Install). Their models download once into `~/.cache/vhs-restore`: upscaler 70 MB, faces 340 MB, colour 980 MB. Each download is checked against a fixed checksum. AI **invents** detail: faces can come out looking slightly different from the real person. Keep your originals, and compare.

## How it works

**Picture** (`vhs_restore/core.py`, `look()`):

1. **Deinterlace**, if interlaced: each half-picture becomes a full frame (`bwdif`), so 25i becomes 50p with real motion.
2. **Repairs**:
   - Colour shift (`chromashift`) and crop.
   - Dropouts: a pixel much brighter than the median of its neighbouring frames, in a thin horizontal run 31 px or longer, is replaced with that median.
3. **Scale to the tape's real resolution**, about 720 lines. Cleaning at that size is fast, and the downscale averages noise away.
4. **Clean noise** across space and time (`hqdn3d`), then steady the picture if asked (`deshake`).
5. **Colour**: automatic white balance at half strength, so faded tape colour comes back without looking artificial. Then a gentle S-curve, plus a little more contrast and saturation.
6. **Sharpen** at the real resolution, then upscale in two Lanczos steps with contrast-adaptive sharpening after each one. This is crisper than one big jump and avoids halos.

**Sound** (`vhs_restore/debuzz.py`):

A tape's buzz is a set of steady tones whose frequencies drift and whose level changes along the tape, so fixed notch filters miss most of it. The debuzzer works in two passes:

1. It measures the spectrum in 1 Hz steps. For every 10-second block it finds what is *always* there, the 20th-percentile level of each frequency.
2. Any frequency that always sits clearly above its neighbours (±15 Hz) is buzz, and it is pushed down to the neighbours' level in that block.

Music and speech change all the time, so they pass through untouched. On the wedding tape, hum peaks up to 38 dB above the background ended up 0–5 dB above it, buried in the noise. The music band (500 Hz–6 kHz) changed by 1.5 dB or less.

After that come click and clipping repair if asked (`adeclick`, `adeclip`), and FFT hiss reduction or RNNoise. Then a small EQ for voice clarity, gentle compression and two-pass loudness normalisation to −14 LUFS.

**Stills**: one frame per second is scored on sharpness, colour and exposure. Frames with a VHS tracking tear (a jagged horizontal band) are rejected. The best frame from each window is then rendered at full quality, with a short lead-in so the temporal denoiser has history to work with.

**AI** (`vhs_restore/ai.py`): ONNX models run with onnxruntime on the CPU, so no GPU is needed:

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) for upscaling
- [GFPGAN](https://github.com/TencentARC/GFPGAN) for faces, found by OpenCV's [YuNet](https://github.com/opencv/opencv_zoo) detector
- [DDColor](https://github.com/piddnad/DDColor) for colour

Scratch repair is not AI. It finds long, thin, straight marks with morphology and line detection, then fills them in with OpenCV's inpainting.

## Limits

- Without `--ai` it cleans and sharpens but does not invent detail. A 1980s tape upscaled to 4K looks like a very clean tape, not like footage shot on a modern camera. AI is offered for photos and stills only. On video it would take days, and invented detail flickers from frame to frame.
- Tracking tears and dust specks on photos are not repaired. Automatic colour-shift detection isn't there yet; set `--chroma-shift` by eye.
- `--crop auto` finds black borders, not noisy grey ones. Use `--crop W:H:X:Y` for those.

## Develop

```sh
pip install -e ".[ai]" pytest
python -m pytest tests                          # everything except the big AI models (~30 s)
VHS_RESTORE_AI_TEST=1 python -m pytest tests    # also runs the AI models (downloads ~1.4 GB once)
```

Releases are published to PyPI automatically when a GitHub release is created (`.github/workflows/publish.yml`).

Pull requests are welcome, especially tracking-tear repair, automatic colour-shift detection, and before/after examples you are happy to share.

## License

MIT, see [LICENSE](LICENSE). The AI models are downloaded (ONNX versions from the [facefusion-assets](https://github.com/facefusion/facefusion-assets) and [opencv_zoo](https://github.com/opencv/opencv_zoo) GitHub repositories) and keep their own licences: Real-ESRGAN BSD-3-Clause, GFPGAN Apache-2.0, DDColor Apache-2.0, YuNet MIT.
