#!/usr/bin/env python3
"""
line_cropper.py

Lined-notebook page photo → one cropped image per text line, ready for OCR.

Designed for student notebooks with printed ruled lines:
  1. Find the page (paper) against the background; keep all writing.
  2. Detect the dominant orientation from long consistent lines (ruled lines / margin).
  3. Deskew so ruled lines become horizontal.
  4. Detect the printed horizontal ruled lines (span most of the width, consistent).
  5. Place seams on those ruled lines (they already separate the text rows).
  6. Crop each band between consecutive ruled lines; drop empty bands.

Usage
  pip install opencv-python-headless numpy
  python line_cropper.py photo.jpg out_folder
"""
import argparse
import os
import sys

import cv2
import numpy as np

MAX_SIDE = 2200


# ---------------------------------------------------------------- page crop
def order_points(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        np.float32,
    )


def crop_page(img):
    """
    Find the paper region and warp to a rectangle.
    Less aggressive than before: prefers keeping content over tight crop.
    Falls back to the full image if the page boundary is unclear.
    """
    h, w = img.shape[:2]
    scale = 900.0 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    if border.mean() > 127:
        mask = cv2.bitwise_not(mask)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img
    c = max(cnts, key=cv2.contourArea)
    area_ratio = cv2.contourArea(c) / mask.size
    if area_ratio < 0.15 or area_ratio > 0.98:
        return img

    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.02, 0.03, 0.05, 0.08, 0.12):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx
            break
    if quad is None:
        rect = cv2.minAreaRect(hull)
        quad = cv2.boxPoints(rect)

    pts = order_points(quad) / scale
    cx, cy = pts.mean(axis=0)
    expand = 1.04
    pts = cx + (pts - cx) * expand
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    tl, tr, br, bl = pts
    out_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    out_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if out_w < 80 or out_h < 80:
        return img
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return warped


# ------------------------------------------------------------ orientation
def _rotate_expand(gray, angle):
    """Rotate without clipping the page or changing its scale."""
    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    out_w = int(np.ceil(h * sin + w * cos))
    out_h = int(np.ceil(h * cos + w * sin))
    M[0, 2] += out_w / 2.0 - center[0]
    M[1, 2] += out_h / 2.0 - center[1]
    return cv2.warpAffine(
        gray, M, (out_w, out_h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )


def _horizontal_structure_score(gray, angle):
    """Measure how much long horizontal structure remains after a rotation."""
    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_AREA,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )
    edges = cv2.Canny(cv2.GaussianBlur(rotated, (3, 3), 0), 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w // 10), 1))
    long_horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    return float(np.count_nonzero(long_horizontal)) / max(float(w * h), 1.0)


def detect_ruled_angle(gray):
    """Find the dominant page-line orientation, including vertical lines."""
    h, w = gray.shape
    scale = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape
    edges = cv2.Canny(cv2.GaussianBlur(small, (3, 3), 0), 40, 120)
    min_len = max(25, int(min(sh, sw) * 0.16))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 360, threshold=max(20, int(min(sh, sw) * 0.08)),
        minLineLength=min_len, maxLineGap=max(10, int(min(sh, sw) * 0.025)),
    )
    if lines is None:
        return 0.0

    # Lines are undirected: -90° and +90° represent the same orientation.
    bins = np.zeros(180, np.float64)
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < min_len:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        angle = ((angle + 90.0) % 180.0) - 90.0
        angles.append((angle, length))
        bins[int(round(angle + 90.0)) % 180] += length ** 1.35
    if len(angles) < 2:
        return 0.0

    # Smooth the circular histogram and refine the winning cluster by length.
    smooth = np.convolve(np.r_[bins[-3:], bins, bins[:3]], np.ones(7) / 7, mode="valid")
    peak = int(np.argmax(smooth)) - 90
    cluster = [(a, length) for a, length in angles if abs(a - peak) <= 7.0]
    if not cluster:
        return float(peak)
    return float(np.average([a for a, _ in cluster], weights=[length ** 1.35 for _, length in cluster]))


def deskew(gray):
    """Rotate so printed ruled lines become horizontal, at any input angle."""
    angle = detect_ruled_angle(gray)
    if abs(angle) < 0.3:
        return gray

    # Hough orientation has no direction, so test both signs. This also makes
    # the result robust to OpenCV's image-coordinate angle convention.
    small_scale = min(1.0, 700.0 / max(gray.shape))
    probe = cv2.resize(gray, None, fx=small_scale, fy=small_scale, interpolation=cv2.INTER_AREA)
    chosen = angle if _horizontal_structure_score(probe, angle) >= _horizontal_structure_score(probe, -angle) else -angle
    return _rotate_expand(gray, chosen)


# ------------------------------------------------------------ ruled-line detection
def binarize_dark(gray):
    h, w = gray.shape
    block = (max(h, w) // 25) | 1
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    inv = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 11
    )
    return inv


def detect_ruled_ys(gray):
    """
    Detect y-positions of printed horizontal ruled lines.
    Suppress short/random ink; keep structures that span a large fraction of width.
    """
    h, w = gray.shape
    inv = binarize_dark(gray)

    min_span = max(w // 5, 40)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (min_span, 1))
    ruled = cv2.morphologyEx(inv, cv2.MORPH_OPEN, k)
    ruled = cv2.dilate(ruled, np.ones((3, 1), np.uint8), iterations=1)

    coverage = (ruled > 0).sum(axis=1).astype(np.float64) / max(w, 1)

    thr = max(0.25, float(np.percentile(coverage[coverage > 0.05], 40)) if np.any(coverage > 0.05) else 0.3)
    thr = min(thr, 0.55)

    s = _smooth1d(coverage, 1.2)
    peaks = []
    for i in range(2, h - 2):
        if s[i] >= s[i - 1] and s[i] >= s[i + 1] and s[i] > thr:
            peaks.append(i)

    if len(peaks) < 2:
        return None, coverage

    peaks = sorted(peaks)
    merged = [peaks[0]]
    min_gap = max(8, int(h * 0.012))
    for p in peaks[1:]:
        if p - merged[-1] >= min_gap:
            merged.append(p)
        else:
            if s[p] > s[merged[-1]]:
                merged[-1] = p

    if len(merged) >= 3:
        diffs = np.diff(merged)
        pitch = float(np.median(diffs))
    else:
        pitch = float(merged[1] - merged[0]) if len(merged) >= 2 else 40.0

    if len(merged) >= 4 and pitch > 10:
        cleaned = [merged[0]]
        for p in merged[1:]:
            expected = cleaned[-1] + pitch
            if abs(p - expected) < pitch * 0.55:
                cleaned.append(p)
            elif p > expected + pitch * 0.8:
                if s[p] > thr * 1.1:
                    cleaned.append(p)
        merged = cleaned

    return merged, coverage


def _smooth1d(a, sigma):
    r = int(max(1, round(sigma * 3)))
    x = np.arange(-r, r + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    return np.convolve(np.pad(a, r, mode="edge"), k, mode="valid")


# ------------------------------------------------------------ fallback freeform
def estimate_pitch(profile):
    p = _smooth1d(profile.astype(np.float64), 1.5)
    p = p - p.mean()
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    lo, hi = 10, max(16, len(p) // 3)
    candidates = []
    for lag in range(lo, min(hi, len(ac) - 1)):
        if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > 0.18 * ac[0]:
            candidates.append((lag, float(ac[lag])))
    if not candidates:
        return lo + int(np.argmax(ac[lo:hi]))
    best = max(candidates, key=lambda t: t[1] * (1.0 + 0.15 * (hi - t[0]) / max(hi - lo, 1)))
    return best[0]


def find_peaks_freeform(s, pitch):
    thr = 0.12 * s.max()
    cands = [i for i in range(1, len(s) - 1) if s[i] >= s[i - 1] and s[i] > s[i + 1] and s[i] > thr]
    cands.sort(key=lambda i: -s[i])
    accepted = []
    min_sep = max(8, int(0.42 * pitch))
    for i in cands:
        if all(abs(i - j) >= min_sep for j in accepted):
            accepted.append(i)
    return sorted(accepted)


def find_seam(cost, y0, half, vpen=0.35, prior_w=0.25):
    h, w = cost.shape
    lo, hi = max(0, y0 - half), min(h, y0 + half + 1)
    n = hi - lo
    ys = np.arange(n)
    band = cost[lo:hi].astype(np.float32).copy()
    band += (prior_w * ((ys - (y0 - lo)) / max(half, 1)) ** 2)[:, None].astype(np.float32)
    D = np.empty_like(band)
    P = np.zeros((n, w), np.int8)
    D[:, 0] = band[:, 0]
    inf = np.float32(np.inf)
    for x in range(1, w):
        prev = D[:, x - 1]
        from_above = np.concatenate(([inf], prev[:-1])) + vpen
        from_below = np.concatenate((prev[1:], [inf])) + vpen
        stack = np.stack((from_above, prev, from_below))
        idx = np.argmin(stack, axis=0)
        D[:, x] = stack[idx, ys] + band[:, x]
        P[:, x] = idx - 1
    path = np.empty(w, np.int32)
    y = int(np.argmin(D[:, -1]))
    path[-1] = y
    for x in range(w - 1, 0, -1):
        y = y + int(P[y, x])
        path[x - 1] = y
    return path + lo


def segment_freeform(gray, ink):
    h, w = gray.shape
    profile = (ink > 0).sum(axis=1).astype(np.float64)
    pitch = estimate_pitch(profile)
    s = _smooth1d(profile, max(2.0, pitch / 6.0))
    peaks = find_peaks_freeform(s, pitch)
    if not peaks:
        return [], pitch

    last_ink = int(np.max(np.where(profile > 3)[0])) if np.any(profile > 3) else h - 1
    peaks = [p for p in peaks if p <= last_ink + int(pitch * 0.4)]
    if not peaks:
        return [], pitch

    dist = cv2.distanceTransform((ink == 0).astype(np.uint8), cv2.DIST_L2, 5)
    cost = 1.0 / (1.0 + dist) + 12.0 * (ink > 0)
    half = max(3, int(pitch * 0.32))

    seams = [np.full(w, max(0, peaks[0] - int(pitch * 0.55)), np.int32)]
    for a, b in zip(peaks[:-1], peaks[1:]):
        y0 = a + int(np.argmin(s[a:b + 1]))
        seams.append(find_seam(cost, y0, half))
    bottom_y = min(h - 1, peaks[-1] + int(pitch * 0.50), last_ink + int(pitch * 0.30))
    seams.append(np.full(w, bottom_y, np.int32))
    for i in range(1, len(seams)):
        seams[i] = np.maximum(seams[i], seams[i - 1] + 1)
    return [np.clip(sm, 0, h - 1) for sm in seams], pitch


# ------------------------------------------------------------ main segmentation
def segment_lines(gray):
    """Prefer ruled-line guided seams. Fall back to freeform only if needed."""
    h, w = gray.shape
    ink = binarize_dark(gray)

    ruled_ys, coverage = detect_ruled_ys(gray)

    if ruled_ys is not None and len(ruled_ys) >= 2:
        seams = [np.full(w, y, np.int32) for y in ruled_ys]
        if ruled_ys[0] > 15:
            seams.insert(0, np.full(w, max(0, ruled_ys[0] - int((ruled_ys[1] - ruled_ys[0]) * 0.9)), np.int32))
        if h - 1 - ruled_ys[-1] > 15:
            pitch_est = float(np.median(np.diff(ruled_ys))) if len(ruled_ys) > 1 else 40
            seams.append(np.full(w, min(h - 1, ruled_ys[-1] + int(pitch_est * 0.9)), np.int32))
        pitch = float(np.median(np.diff(ruled_ys))) if len(ruled_ys) > 1 else 40.0
        return ink, seams, pitch

    seams, pitch = segment_freeform(gray, ink)
    return ink, seams, pitch


def crop_line(gray, ink, top, bot, pad_x=10):
    h, w = gray.shape
    y0, y1 = int(top.min()), int(bot.max()) + 1
    if y1 - y0 < 6:
        return None
    rows = np.arange(y0, y1)[:, None]
    mask = (rows >= top[None, :]) & (rows <= bot[None, :])
    ink_mask = (ink[y0:y1] > 0) & mask
    cols = np.where(ink_mask)[1]
    if cols.size < 30 or (cols.max() - cols.min()) < 25:
        return None
    x0, x1 = max(0, cols.min() - pad_x), min(w, cols.max() + pad_x + 1)
    out = np.where(mask, gray[y0:y1], 255).astype(np.uint8)
    return out[:, x0:x1]


# --------------------------------------------------------------------- main
def process(path, outdir):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"Could not read {path}")
    os.makedirs(os.path.join(outdir, "lines"), exist_ok=True)

    page = crop_page(img)
    h, w = page.shape[:2]
    if max(h, w) > MAX_SIDE:
        f = MAX_SIDE / max(h, w)
        page = cv2.resize(page, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    gray = deskew(cv2.cvtColor(page, cv2.COLOR_BGR2GRAY))
    cv2.imwrite(os.path.join(outdir, "page.png"), gray)

    ink, seams, pitch = segment_lines(gray)
    debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    saved = 0
    for i in range(len(seams) - 1):
        crop = crop_line(gray, ink, seams[i], seams[i + 1])
        if crop is None:
            continue
        saved += 1
        cv2.imwrite(os.path.join(outdir, "lines", f"line_{saved:03d}.png"), crop)
    for sm in seams:
        pts = np.stack((np.arange(len(sm)), sm), axis=1).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(debug, [pts], False, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(outdir, "debug_seams.png"), debug)
    print(f"pitch ~{pitch:.0f}px, {saved} lines saved to {outdir}/lines")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("outdir")
    args = ap.parse_args()
    process(args.image, args.outdir)
