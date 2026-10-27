# Optional AI steps for photos and stills (pip install "vhs-restore[ai]"). Models are downloaded once into the cache.
# Unlike the rest of vhs-restore these INVENT detail: great for faces and old prints, but check the result.
import numpy as np

from .core import fetch

try:
    import cv2, onnxruntime as ort
except ImportError:
    cv2 = ort = None

ASSETS = 'https://github.com/facefusion/facefusion-assets/releases/download/'
MODELS = {  # name: (url, sha256)   upstream licences: Real-ESRGAN BSD-3, GFPGAN Apache-2.0, YuNet MIT, DDColor Apache-2.0
    'real_esrgan_x4.onnx': (ASSETS + 'models-3.0.0/real_esrgan_x4.onnx', 'afa7fa42eed0315a3d7b2a54e79e4e44a672d3c0889c2828c4f84c9ab91edb0b'),
    'gfpgan_1.4.onnx': (ASSETS + 'models-3.0.0/gfpgan_1.4.onnx', 'accc4757b26bdb89b32b4d3500d4f79c9dff97c1dd7c7104bf9dcb95e3311385'),
    'face_detection_yunet_2023mar.onnx': ('https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/'
                                          'models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
                                          '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'),
    'ddcolor.onnx': (ASSETS + 'models-3.0.0/ddcolor.onnx', 'dd6156f056d0690891ef35f9ee87176d4d976ab19bbd36e57d18e841e064d797'),
}
_sessions = {}


def need():
    if cv2 is None:
        raise SystemExit('AI options need extra packages:  pip install "vhs-restore[ai]"')


def session(name, log=print):
    if name not in _sessions:
        path = fetch(name, *MODELS[name], log=log)
        _sessions[name] = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    return _sessions[name]


def upscale(img, log=print, tile=192, pad=16):
    """Real-ESRGAN x4 on a BGR uint8 image, in overlapping tiles so memory stays small."""
    s = session('real_esrgan_x4.onnx', log)
    h, w = img.shape[:2]
    out = np.zeros((h * 4, w * 4, 3), np.float32)
    x = img[:, :, ::-1].astype(np.float32) / 255
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            ya, yb, xa, xb = max(y0 - pad, 0), min(y0 + tile + pad, h), max(x0 - pad, 0), min(x0 + tile + pad, w)
            t = s.run(None, {s.get_inputs()[0].name: x[ya:yb, xa:xb].transpose(2, 0, 1)[None]})[0][0].transpose(1, 2, 0)
            oy, ox = (y0 - ya) * 4, (x0 - xa) * 4
            th, tw = min(tile, h - y0) * 4, min(tile, w - x0) * 4
            out[y0 * 4:y0 * 4 + th, x0 * 4:x0 * 4 + tw] = t[oy:oy + th, ox:ox + tw]
    return (out[:, :, ::-1].clip(0, 1) * 255).round().astype(np.uint8)


# where GFPGAN expects the eyes, nose and mouth corners in its 512x512 face crop
FACE = np.array([[192.98, 239.95], [318.90, 240.19], [256.63, 314.02], [201.26, 371.41], [313.09, 371.15]], np.float32)


def faces(img, strength=0.6, log=print):
    """GFPGAN on every detected face, blended back at `strength` (1 = fully AI face, 0.5 keeps more of the real one)."""
    h, w = img.shape[:2]
    det = cv2.FaceDetectorYN.create(str(fetch('face_detection_yunet_2023mar.onnx', *MODELS['face_detection_yunet_2023mar.onnx'], log=log)), '', (w, h), 0.7)
    _, found = det.detect(img)
    if found is None: return img
    s = session('gfpgan_1.4.onnx', log)
    out, n = img.astype(np.float32), 0
    for f in found:
        pts = f[4:14].reshape(5, 2)
        pts = np.array([*sorted(pts[:2], key=lambda p: p[0]), pts[2], *sorted(pts[3:], key=lambda p: p[0])], np.float32)
        if np.linalg.norm(pts[0] - pts[1]) < 40: continue  # eyes < 40 px apart: too little to go on, GFPGAN would invent a face
        M = cv2.estimateAffinePartial2D(pts, FACE, method=cv2.LMEDS)[0]
        crop = cv2.warpAffine(img, M, (512, 512), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        x = (crop[:, :, ::-1].astype(np.float32) / 255 - 0.5) / 0.5
        y = s.run(None, {s.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0].transpose(1, 2, 0)
        y = ((y.clip(-1, 1) + 1) / 2 * 255)[:, :, ::-1]
        mask = np.zeros((512, 512), np.float32)
        cv2.rectangle(mask, (40, 40), (472, 472), 1, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 20) * strength  # soft edges so the face melts into the photo
        iM = cv2.invertAffineTransform(M)
        back = cv2.warpAffine(y, iM, (w, h), flags=cv2.INTER_CUBIC)
        m = cv2.warpAffine(mask, iM, (w, h))[:, :, None]
        out = out * (1 - m) + back * m; n += 1
    log(f"faces: restored {n} of {len(found)} found" + (" (the rest are too small)" if n < len(found) else ""))
    return out.clip(0, 255).round().astype(np.uint8)


def is_gray(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.abs(lab[:, :, 1:] - 128).mean()) < 4


def colorize(img, log=print):
    """DDColor: guess colours for a black-and-white photo; brightness/detail stay the original's."""
    s = session('ddcolor.onnx', log)
    size = 512  # DDColor's training size
    g = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB).astype(np.float32) / 255
    L = cv2.cvtColor(g, cv2.COLOR_RGB2LAB)[:, :, :1]
    x = cv2.cvtColor(np.concatenate([L, np.zeros_like(L), np.zeros_like(L)], 2), cv2.COLOR_LAB2RGB)
    x = cv2.resize(x, (size, size), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)[None]
    ab = s.run(None, {s.get_inputs()[0].name: x})[0][0].transpose(1, 2, 0)
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(np.concatenate([L, ab], 2), cv2.COLOR_LAB2RGB)
    return (rgb[:, :, ::-1].clip(0, 1) * 255).round().astype(np.uint8)


def scratches(img, log=print):
    """Scratches (not AI): thin bright or dark lines that run straight for a long way. Find them, then paint them in
    from their surroundings. Short marks are left alone, so real detail like jewellery sparkle survives."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = max(3, round(min(g.shape) / 300)) | 1                         # scratch width scales with scan size
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3 * k, 3 * k))
    d = cv2.max(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se), cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se))
    ridge = (d > 50).astype(np.uint8)
    length = int(min(g.shape) * 0.15)  # tuned on busy wedding frames: shorter/fainter lines are usually real detail
    lines = cv2.HoughLinesP(ridge, 1, np.pi / 360, threshold=int(length * 0.6), minLineLength=length, maxLineGap=2 * k)
    mask = np.zeros_like(g)
    for x0, y0, x1, y1 in ([] if lines is None else lines.reshape(-1, 4)):
        cv2.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, k)
    mask &= cv2.dilate(ridge * 255, np.ones((3, 3), np.uint8))  # only the line pixels themselves
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    log(f"scratches: {0 if lines is None else len(lines)} found, {mask.mean() / 2.55:.2f}% of the picture repaired"
        + (" - that's a lot: if this photo isn't scratched, leave --scratches off" if mask.mean() > 2.55 * 2 else ""))
    return cv2.inpaint(img, mask, k, cv2.INPAINT_TELEA)


def resize(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if w < img.shape[1] else cv2.INTER_LANCZOS4)
