"""OCR for scanned packet pages.

The scans are 144-DPI renders degraded with skew, per-line jitter, faint
gridlines, speckle and stray tick marks. Whole-page OCR fails badly because the
ink occupies a small fraction of a noisy page. Instead we isolate glyph-sized
connected components, group them into text lines, and OCR one line at a time.
"""
from __future__ import annotations

import cv2
import fitz
import numpy as np
import pytesseract

_INK_THRESHOLD = 150
_TARGET_GLYPH_H = 44.0  # tesseract likes roughly 30-50px cap height
_PSM_LINE = "--oem 1 --psm 7"
_ORIENT_ACCEPT = 3  # form-vocabulary hits that make a rotation obviously right


def native_image(doc: fitz.Document, page: fitz.Page):
    """Return the largest embedded raster at its native resolution.

    Re-rendering the page would resample an already-lossy 144-DPI JPEG and
    lose detail, so we decode the embedded image directly.
    """
    best = None
    for info in page.get_images(full=True):
        try:
            raw = doc.extract_image(info[0])
        except Exception:
            continue
        arr = cv2.imdecode(np.frombuffer(raw["image"], np.uint8), cv2.IMREAD_GRAYSCALE)
        if arr is not None and (best is None or arr.size > best.size):
            best = arr
    if best is None:
        pix = page.get_pixmap(dpi=150, colorspace=fitz.csGRAY)
        best = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return best


def glyph_mask(img):
    """Keep only character-sized ink: drops rules, borders, photos and stamps."""
    raw = (img < _INK_THRESHOLD).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    height, width = img.shape
    mask = np.zeros_like(raw)
    heights = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < 6 or h < 5 or h > 0.035 * height or w > 0.25 * width:
            continue
        mask[labels == i] = 255
        heights.append(h)
    med_h = float(np.median(heights)) if heights else 12.0
    return mask, med_h


def find_lines(mask, med_h):
    """Merge glyphs into words, drop isolated marks, group words into lines."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(med_h * 0.8), 5), 1))
    words = cv2.dilate(mask, kernel)
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(words, 8)
    boxes = []
    for i in range(1, count):
        x, y, w, h, _area = stats[i]
        # A real word is several glyphs wide; a tick mark or speck is not.
        if w < 2.5 * med_h or h > 3 * med_h:
            continue
        boxes.append((x, y, w, h))
    if not boxes:
        return []

    boxes.sort(key=lambda b: b[1] + b[3] / 2)
    lines, current = [], [boxes[0]]
    centre = boxes[0][1] + boxes[0][3] / 2
    for box in boxes[1:]:
        c = box[1] + box[3] / 2
        if abs(c - centre) <= 0.7 * med_h:
            current.append(box)
            centre = (centre * (len(current) - 1) + c) / len(current)
        else:
            lines.append(current)
            current, centre = [box], c
    lines.append(current)

    out = []
    for line in lines:
        x0 = min(b[0] for b in line)
        x1 = max(b[0] + b[2] for b in line)
        y0 = min(b[1] for b in line)
        y1 = max(b[1] + b[3] for b in line)
        out.append((x0, y0, x1, y1))
    return out


def _deskew_line(gray, mask):
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 20:
        return gray
    angle = cv2.minAreaRect(points)[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.3 or abs(angle) > 15:
        return gray
    h, w = gray.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def orient(img):
    """Rotate a scan upright.

    Purely geometric orientation scores are unreliable here because the scanned
    gridlines and border ticks segment into convincing-looking "text lines" at
    any angle. Deciding on how much real form vocabulary each rotation yields is
    slower but actually correct, so the upright reading short-circuits and only
    doubtful pages pay for the alternatives.
    """
    score = _wordiness(img)
    if score >= _ORIENT_ACCEPT:
        return img
    best, best_score = img, score
    for turns in (1, 3, 2):
        candidate = np.ascontiguousarray(np.rot90(img, turns))
        candidate_score = _wordiness(candidate)
        if candidate_score > best_score:
            best, best_score = candidate, candidate_score
        if best_score >= _ORIENT_ACCEPT:
            break
    return best


_ANCHORS = ("case", "applicant", "species", "world", "visa", "sponsor", "arrival",
            "purpose", "fee", "status", "registry", "flags", "form", "mib", "note")


def _wordiness(img) -> int:
    """Count recognisable form vocabulary in a cheap whole-page pass."""
    small = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    try:
        text = pytesseract.image_to_string(small, config="--oem 1 --psm 11").lower()
    except Exception:
        return 0
    return sum(text.count(anchor) for anchor in _ANCHORS)


def ocr_page(img) -> list[str]:
    img = orient(img)
    mask, med_h = glyph_mask(img)
    lines = find_lines(mask, med_h)
    if not lines:
        return []

    # Keep grayscale antialiasing inside a dilated glyph footprint: binarising
    # here costs several characters per line, but the footprint still removes
    # the gridlines and border noise.
    halo = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    clean = np.where(halo > 0, img, 255).astype(np.uint8)
    scale = float(np.clip(_TARGET_GLYPH_H / max(med_h, 6.0), 1.5, 6.0))

    height, width = img.shape
    texts = []
    for (x0, y0, x1, y1) in lines:
        top, bottom = max(0, y0 - 6), min(height, y1 + 6)
        left, right = max(0, x0 - 6), min(width, x1 + 6)
        patch = _deskew_line(clean[top:bottom, left:right].copy(), mask[top:bottom, left:right])
        patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        patch = cv2.GaussianBlur(patch, (3, 3), 0)
        lo = float(np.percentile(patch, 1))
        patch = np.clip((patch.astype(np.float32) - lo) * 255.0 / max(255.0 - lo, 1.0), 0, 255)
        patch = cv2.copyMakeBorder(patch.astype(np.uint8), 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        text = pytesseract.image_to_string(patch, config=_PSM_LINE).strip()
        if text:
            texts.append(text)
    return texts
