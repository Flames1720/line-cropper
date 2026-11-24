#!/usr/bin/env python3
"""
line_cropper.py

Page photo in, one cropped image per text line out, ready for OCR.

Steps
  1. Find the page against the background and flatten it (perspective warp).
  2. Deskew so lines are roughly horizontal.
  3. Binarize (ink = white on black) and optionally strip printed ruled lines.
  4. Estimate line pitch from the row profile and locate the gaps between lines.
  5. For every gap, run a dynamic programming seam from left to right through
     the cheapest white space, so the cut bends around handwriting.
  6. Cut each line out between its top seam and bottom seam.

Usage
  pip install opencv-python-headless numpy
  python line_cropper.py photo.jpg out_folder
"""
import argparse
import os
import sys

import cv2
import numpy as np

MAX_SIDE = 2400  # working resolution of the page after cropping


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
    """Find the page quadrilateral and warp it flat. Returns img unchanged if unsure."""
    h, w = img.shape[:2]
    scale = 1000.0 / max(h, w)
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    if border.mean() > 127:  # background came out white, so flip
        mask = cv2.bitwise_not(mask)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.2 * mask.size:
        return img

    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull))

    pts = order_points(quad) / scale
    tl, tr, br, bl = pts
    out_w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    out_h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if out_w < 50 or out_h < 50:
        return img
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h), flags=cv2.INTER_CUBIC)


# ------------------------------------------------------------ preprocessing
def binarize(gray):
    h, w = gray.shape
    block = (max(h, w) // 30) | 1
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    ink = cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 12
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 12
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def remove_rules(ink):
    """Delete long horizontal streaks (printed ruled lines) from the ink map."""
    h, w = ink.shape
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 12, 25), 1))
    rules = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k)
    rules = cv2.dilate(rules, np.ones((3, 1), np.uint8))
    return cv2.subtract(ink, rules)


def deskew(gray):
    """Rotate so the row profile is as sharp as possible. Returns rotated gray."""
    ink = remove_rules(binarize(gray))
    s = 800.0 / max(ink.shape)
    small = cv2.resize(ink, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    hh, ww = small.shape
    best_score, best_angle = -1.0, 0.0
    for a in np.arange(-6, 6.01, 0.25):
        M = cv2.getRotationMatrix2D((ww / 2, hh / 2), a, 1.0)
        r = cv2.warpAffine(small, M, (ww, hh))
        score = float(np.var(r.sum(axis=1).astype(np.float64)))
        if score > best_score:
            best_score, best_angle = score, a
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), best_angle, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255
    )


# ------------------------------------------------------------ line finding
def smooth1d(a, sigma):
    r = int(max(1, round(sigma * 3)))
    x = np.arange(-r, r + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    k /= k.sum()
    return np.convolve(np.pad(a, r, mode="edge"), k, mode="valid")


def estimate_pitch(profile):
    p = smooth1d(profile.astype(np.float64), 2.0)
    p = p - p.mean()
    ac = np.correlate(p, p, "full")[len(p) - 1:]
    lo, hi = 12, max(14, len(p) // 3)
    for lag in range(lo, min(hi, len(ac) - 1)):
        if ac[lag] > ac[lag - 1] and ac[lag] >= ac[lag + 1] and ac[lag] > 0.25 * ac[0]:
            return lag
    return lo + int(np.argmax(ac[lo:hi]))


def find_peaks(s, pitch):
    thr = 0.2 * s.max()
    cands = [i for i in range(1, len(s) - 1) if s[i] >= s[i - 1] and s[i] > s[i + 1] and s[i] > thr]
    cands.sort(key=lambda i: -s[i])
    accepted = []
    for i in cands:
        if all(abs(i - j) >= 0.6 * pitch for j in accepted):
            accepted.append(i)
    return sorted(accepted)


def filter_content_peaks(peaks, profile, pitch, min_rel=0.12):
    """Keep only peaks that sit over real ink. Drop empty ruled-line peaks."""
    if not peaks:
        return peaks
    half = max(3, int(pitch * 0.35))
    scores = []
    for y in peaks:
        lo, hi = max(0, y - half), min(len(profile), y + half + 1)
        scores.append(float(profile[lo:hi].sum()))
    scores = np.asarray(scores)
    ref = max(scores.max(), 1.0)
    keep = scores >= (min_rel * ref)
    abs_thr = 8.0 * half
    keep = keep & (scores >= abs_thr)
    filtered = [p for p, k in zip(peaks, keep) if k]
    return filtered


def find_seam(cost, y0, half, vpen=0.08, prior_w=0.1):
    """Cheapest left-to-right path near row y0, moving at most one row per column."""
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


def segment_lines(gray):
    h, w = gray.shape
    ink = binarize(gray)
    seg_ink = remove_rules(ink)

    profile = (seg_ink > 0).sum(axis=1).astype(np.float64)
    pitch = estimate_pitch(profile)
    s = smooth1d(profile, max(2.0, pitch / 6.0))
    peaks = find_peaks(s, pitch)
    peaks = filter_content_peaks(peaks, profile, pitch)
    if not peaks:
        return ink, [], pitch

    last_ink = int(np.max(np.where(profile > 3)[0])) if np.any(profile > 3) else h - 1
    peaks = [p for p in peaks if p <= last_ink + int(pitch * 0.4)]
    if not peaks:
        return ink, [], pitch

    dist = cv2.distanceTransform((seg_ink == 0).astype(np.uint8), cv2.DIST_L2, 5)
    cost = 1.0 / (1.0 + dist) + 2.0 * (seg_ink > 0)
    half = max(4, int(pitch * 0.5))

    seams = [np.full(w, max(0, peaks[0] - int(pitch * 0.6)), np.int32)]
    for a, b in zip(peaks[:-1], peaks[1:]):
        y0 = a + int(np.argmin(s[a:b + 1]))
        seams.append(find_seam(cost, y0, half))
    bottom_y = min(h - 1, peaks[-1] + int(pitch * 0.55), last_ink + int(pitch * 0.35))
    seams.append(np.full(w, bottom_y, np.int32))

    for i in range(1, len(seams)):
        seams[i] = np.maximum(seams[i], seams[i - 1] + 1)
    seams = [np.clip(sm, 0, h - 1) for sm in seams]
    return ink, seams, pitch


def crop_line(gray, ink, top, bot, pad_x=12):
    h, w = gray.shape
    y0, y1 = int(top.min()), int(bot.max()) + 1
    rows = np.arange(y0, y1)[:, None]
    mask = (rows >= top[None, :]) & (rows <= bot[None, :])
    ink_mask = (ink[y0:y1] > 0) & mask
    cols = np.where(ink_mask)[1]
    if cols.size < 40 or (cols.max() - cols.min()) < 30:
        return None
    x0, x1 = max(0, cols.min() - pad_x), min(w, cols.max() + pad_x + 1)
    out = np.where(mask, gray[y0:y1], 255).astype(np.uint8)
    return out[:, x0:x1]


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
    print(f"pitch ~{pitch}px, {saved} lines saved to {outdir}/lines")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("outdir")
    args = ap.parse_args()
    process(args.image, args.outdir)
