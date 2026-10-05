#!/usr/bin/env python3
"""
PDF Image / Figure Extractor
==========================
Heuristically identifies PDF pages that contain figures, diagrams or
illustration grids, then crops the non-text regions and writes them as PNG
files.  Also produces low-resolution thumbnail matrix sheets for quick visual
review.

Typical use case: bulk-processing scanned books, technical manuals or
scientific papers to harvest the figures.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------
DEFAULT_DEST_DIR = "./pdfs"
DEFAULT_IMAGE_OUT_DIR = "./image_out_dir"
DEFAULT_GRID_OUT_DIR = "./grid_out_dir"
DEFAULT_THRESHOLD = 70
DEFAULT_EVAL_DPI = 150
EDGE_MARGIN = 0.035  # fraction of page ignored at each edge when looking for figures
WORK_W = 1100  # working width used by the geometry scorer


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _contours(*args, **kwargs):
    """Compatibility wrapper – OpenCV 3 vs 4 return different tuple lengths."""
    result = cv2.findContours(*args, **kwargs)
    return result[-2]


def _textiness(im: np.ndarray) -> float:
    """
    Estimate how text-like an image is.
    Higher = more text (character-sized blobs + horizontal line structure).
    """
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n < 8:
        return 0.0
    Hs = st[1:, cv2.CC_STAT_HEIGHT]
    Ws = st[1:, cv2.CC_STAT_WIDTH]
    h, w = im.shape[:2]
    k = (Hs > 3) & (Hs < h * 0.08) & (Ws > 2) & (Ws < w * 0.25)
    if k.sum() < 12:
        return 0.0
    xh = int(np.argmax(np.bincount(Hs[k].astype(int))))
    if xh < 4:
        return 0.0
    lo, hi = 0.55 * xh, 1.8 * xh
    same_height = sum(
        1 for i in range(1, n)
        if lo <= st[i, cv2.CC_STAT_HEIGHT] <= hi
        and st[i, cv2.CC_STAT_WIDTH] < w * 0.3
    )
    ratio = same_height / max(1, n - 1)
    line_like = sum(
        1 for i in range(1, n)
        if st[i, cv2.CC_STAT_HEIGHT] < h * 0.06
        and st[i, cv2.CC_STAT_WIDTH] > st[i, cv2.CC_STAT_HEIGHT] * 3
    )
    if line_like > 8:
        ratio = min(1.0, ratio + 0.25)
    return float(ratio)


def _prep(pixmap: fitz.Pixmap) -> np.ndarray:
    """Convert a PyMuPDF pixmap to a grayscale working image of width WORK_W."""
    img = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        (pixmap.height, pixmap.width, pixmap.n)
    )
    if pixmap.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    elif pixmap.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img = img[:, :, 0]
    h, w = img.shape
    return cv2.resize(
        img, (WORK_W, int(h * WORK_W / w)), interpolation=cv2.INTER_AREA
    )


def _strip_artefacts(bw: np.ndarray) -> np.ndarray:
    """Remove large border-touching components that are usually page edges."""
    h, w = bw.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    out = bw.copy()
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        if (
            (x <= 2 or y <= 2 or x + ww >= w - 2 or y + hh >= h - 2)
            and (hh > h * 0.5 or ww > w * 0.5)
            and a > ww * hh * 0.35
        ):
            out[lab == i] = 0
    return out


# ---------------------------------------------------------------------------
# Core algorithms
# ---------------------------------------------------------------------------
def score_page_geometry(pixmap: fitz.Pixmap) -> tuple[int, int, int, int]:
    """
    Score a page for figure-like geometry.
    Returns total, lattice, box, satellite.
    """
    im = _prep(pixmap)
    h, w = im.shape
    bw = cv2.adaptiveThreshold(
        im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    bw = _strip_artefacts(bw)

    # --- lattice ---
    latt = 0
    seg = cv2.HoughLinesP(
        bw, 1, np.pi / 360, threshold=70,
        minLineLength=int(w * 0.16), maxLineGap=12
    )
    if seg is not None and len(seg):
        S = np.asarray(seg, dtype=float).reshape(-1, 4)
        ang = np.degrees(np.arctan2(S[:, 3] - S[:, 1], S[:, 2] - S[:, 0])) % 180
        H = S[(ang < 8) | (ang > 172)]
        V = S[np.abs(ang - 90) < 8]
        xs, ys = [], []
        for a in H[:80]:
            for b in V[:80]:
                ya = (a[1] + a[3]) / 2
                xb = (b[0] + b[2]) / 2
                if (
                    min(a[0], a[2]) - 6 <= xb <= max(a[0], a[2]) + 6
                    and min(b[1], b[3]) - 6 <= ya <= max(b[1], b[3]) + 6
                ):
                    xs.append(xb)
                    ys.append(ya)
        if len(xs) >= 6:
            ux = len(np.unique(np.round(np.array(xs) / 8)))
            uy = len(np.unique(np.round(np.array(ys) / 8)))
            if ux >= 3 and uy >= 3:
                fill = len(xs) / float(ux * uy)
                latt = int(min(ux, 20) * min(uy, 20) * min(fill, 1.0))

    # --- regular boxes ---
    box = 0
    cs = _contours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    R = []
    for c in cs:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww < w * 0.035 or hh < w * 0.035 or ww > w * 0.45 or hh > w * 0.60:
            continue
        if cv2.contourArea(c) < ww * hh * 0.55:
            continue
        R.append((x + ww / 2, y + hh / 2, ww, hh))
    if len(R) >= 6:
        A = np.array(R)
        mw, mh = np.median(A[:, 2]), np.median(A[:, 3])
        keep = A[
            (np.abs(A[:, 2] - mw) < mw * 0.22)
            & (np.abs(A[:, 3] - mh) < mh * 0.22)
        ]
        if len(keep) >= 6:
            gx = len(np.unique(np.round(keep[:, 0] / (mw * 0.6))))
            gy = len(np.unique(np.round(keep[:, 1] / (mh * 0.6))))
            if gx >= 2 and gy >= 2:
                box = min(len(keep), 64)

    # --- satellite blobs ---
    sat = 0
    n, lab, st, cen = cv2.connectedComponentsWithStats(bw, 8)
    big = [
        (st[i, 4], cen[i])
        for i in range(1, n)
        if (w * 0.035) ** 2 < st[i, 4] < (w * 0.45) ** 2
        and 0.15 < st[i, 2] / max(1, st[i, 3]) < 6.0
    ]
    if 6 <= len(big) <= 40:
        C = np.array([c for _, c in big])
        ctr = np.array([w / 2, h / 2])
        d = np.hypot(*(C - ctr).T) / (w / 2)
        outer = int((d > 0.45).sum())
        inner = sum(
            a for a, c in big if np.hypot(*(c - ctr)) < w * 0.18
        )
        if outer >= 6 and inner > 0:
            sat = min(outer, 20)

    # your tuned weights
    total = latt * 2.0 + box * 9.0 + sat * 7.0
    return int(total), latt, box, sat


FULLPAGE_FRAC = 0.60  # an embedded image placed over >= this share of the page is a scan layer, not a figure


def _covers_page(page: fitz.Page, xref: int) -> bool:
    """MRC scans (JPX background + JBIG2 mask) place page-sized layers on every page; skip those."""
    area = page.rect.width * page.rect.height
    try:
        rects = page.get_image_rects(xref)
    except Exception:
        return False
    return any((r.width * r.height) / area >= FULLPAGE_FRAC for r in rects)


def extract_embedded_images(page: fitz.Page, doc: fitz.Document, keep_fullpage: bool = False) -> list[tuple[np.ndarray, float]]:
    """Real embedded raster images (XObjects), not page renders and not page-sized scan layers."""
    out: list[tuple[np.ndarray, float]] = []
    seen: set[int] = set()
    for img in page.get_images(full=True):
        xref = img[0]
        if xref in seen:
            continue
        seen.add(xref)
        if not keep_fullpage and _covers_page(page, xref):
            continue
        try:
            base = doc.extract_image(xref)
        except Exception:
            continue
        data = base.get("image")
        if not data:
            continue
        arr = np.frombuffer(data, dtype=np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if im is None:
            continue
        h, w = im.shape[:2]
        if w < 80 or h < 80 or w * h < 15_000:
            continue
        try:
            r = page.get_image_rects(xref)[0]
            idpi = w / (r.width / 72.0) if r.width > 0 else 72.0
        except Exception:
            idpi = 72.0
        out.append((im, idpi))
    return out


def native_dpi(page: fitz.Page) -> float:
    """Pixel density of the largest embedded image on the page (the scan itself); 0 if none."""
    best = 0.0
    for img in page.get_images(full=True):
        try:
            rects = page.get_image_rects(img[0])
        except Exception:
            continue
        for r in rects:
            if r.width > 0:
                best = max(best, img[2] / (r.width / 72.0))
    return best


def _big_ink_share(bw: np.ndarray, H: int) -> float:
    """Share of ink sitting in large connected structures (card borders, wheels, rules, drawn lines).
    Letters are small components; a paragraph scores ~0, a plate scores high - text INSIDE a plate
    does not matter because the big structures dominate the ink."""
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n < 2:
        return 0.0
    big = np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) > 0.03 * H
    area = st[1:, cv2.CC_STAT_AREA].astype(float)
    return float(area[big].sum() / max(1.0, area.sum()))


def _is_text_line(band: np.ndarray, ww: int, H: int) -> bool:
    """A band (one or several lines) is type if its glyph-sized blobs share one height class and none is huge."""
    n, _, st, _ = cv2.connectedComponentsWithStats(band, 8)
    if n < 7:
        return False
    big_enough = st[1:, cv2.CC_STAT_AREA] >= 6
    hs = st[1:, cv2.CC_STAT_HEIGHT][big_enough]
    ws = st[1:, cv2.CC_STAT_WIDTH][big_enough]
    if hs.size < 6:
        return False
    med = float(np.median(hs))
    if med > 0.04 * H:
        return False
    same = float(((hs >= 0.5 * med) & (hs <= 1.8 * med)).mean())
    return same >= 0.7 and float(np.percentile(ws, 90)) <= 3.5 * med


def nxt_blank_missing(r0: int, r1: int, hh: int, gap_to_fig: int) -> bool:
    """True when the band is the whole region (nothing left to separate it from)."""
    return r0 <= 0 and r1 >= hh - 1


def _trim_stray_lines(bw, x, y, ww, hh, H):
    """Drop body-text lines that got fused onto the top or bottom of a figure region.
    Captions are put back afterwards by _attach_captions."""
    max_band_h = int(0.12 * H)
    gap = max(8, int(0.006 * H))
    cx = x + ww / 2.0
    for end in ("top", "bottom"):
        while hh > 0.06 * H:
            reg = bw[y:y + hh, x:x + ww]
            rows = reg.any(axis=1)
            # fuse rows separated by less than `gap` blank rows, so slivers of a clipped
            # neighbouring line and the line itself form one band
            rows = np.convolve(rows.astype(np.uint8), np.ones(gap, dtype=np.uint8), mode="same") > 0
            if end == "top":
                r0 = int(np.argmax(rows))
                r1 = r0
                while r1 + 1 < hh and rows[r1 + 1]:
                    r1 += 1
                nxt = r1 + 1
                while nxt < hh and not rows[nxt]:
                    nxt += 1
                gap_to_fig = nxt - r1
            else:
                r1 = hh - 1 - int(np.argmax(rows[::-1]))
                r0 = r1
                while r0 - 1 >= 0 and rows[r0 - 1]:
                    r0 -= 1
                prv = r0 - 1
                while prv >= 0 and not rows[prv]:
                    prv -= 1
                gap_to_fig = r0 - prv
            band = reg[r0:r1 + 1]
            if (r1 - r0) > max_band_h or nxt_blank_missing(r0, r1, hh, gap_to_fig):
                break
            if not _is_text_line(band, ww, H):
                break
            # every text-like band goes (captions are re-attached next, only if centred and adjacent)
            if end == "top":
                y, hh = y + r1 + 1, hh - (r1 + 1)
            else:
                hh = r0
    return y, hh


def _attach_captions(bw, x, y, ww, hh, max_lines, H, W):
    """Pull in up to max_lines short, centred text lines directly above/below a figure
    (titles, captions, labels).  Left-aligned or full-width lines are body text and stay out."""
    x0, x1 = x, x + ww
    pad = int(0.10 * ww)
    cx = (x0 + x1) / 2.0
    max_gap, max_line_h = int(0.012 * H) * 3, int(0.03 * H)

    def grab(direction):
        nonlocal y, hh
        got = 0
        while got < max_lines:
            if direction > 0:
                a, b = y + hh, min(H, y + hh + max_gap + max_line_h)
            else:
                a, b = max(0, y - max_gap - max_line_h), y
            strip = bw[a:b, max(0, x0 - pad): min(W, x1 + pad)]
            rows = np.where(strip.sum(axis=1) > 0)[0]
            if rows.size == 0:
                return
            if direction > 0:
                r0 = rows[0]
                if r0 > max_gap:
                    return
                r1 = r0
                while r1 + 1 < strip.shape[0] and strip[r1 + 1].any():
                    r1 += 1
                ys0, ys1 = a + r0, a + r1
            else:
                r1 = rows[-1]
                if (strip.shape[0] - 1 - r1) > max_gap:
                    return
                r0 = r1
                while r0 - 1 >= 0 and strip[r0 - 1].any():
                    r0 -= 1
                ys0, ys1 = a + r0, a + r1
            if ys1 - ys0 > max_line_h:
                return
            cols = np.where(bw[ys0:ys1 + 1, max(0, x0 - pad): min(W, x1 + pad)].any(axis=0))[0]
            if cols.size == 0:
                return
            lx0, lx1 = cols[0] + max(0, x0 - pad), cols[-1] + max(0, x0 - pad)
            if (lx1 - lx0) > 0.9 * ww or abs((lx0 + lx1) / 2.0 - cx) > 0.08 * ww:
                return
            if direction > 0:
                hh = ys1 - y + 1
            else:
                hh += y - ys0
                y = ys0
            got += 1

    grab(-1)
    grab(+1)
    return y, hh


def crop_figures(page: fitz.Page, crop_dpi: float, max_caption_lines: int = 2):
    """
    Find figure regions at 200 dpi, then re-render each region at crop_dpi.
    Text inside a figure is kept.  Only paragraph-like blocks are rejected.
    Returns a list of (BGR crop, (x0, y0, x1, y1) in page points).
    """
    det_dpi = 200
    pix = page.get_pixmap(dpi=det_dpi)
    raw = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        (pix.height, pix.width, pix.n)
    )
    if pix.n >= 3:
        color = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    else:
        gray = raw[:, :, 0]

    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )
    n, lab, st, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n < 20:
        return []

    H, W = gray.shape
    Hs = st[1:, cv2.CC_STAT_HEIGHT]
    Ws = st[1:, cv2.CC_STAT_WIDTH]
    keep = (Hs > 3) & (Hs < H * 0.06) & (Ws < W * 0.08)
    if keep.sum() < 20:
        return []

    xh = int(np.argmax(np.bincount(Hs[keep].astype(int))))
    text = np.zeros_like(bw)
    if xh >= 4:
        lo, hi = 0.55 * xh, 1.7 * xh
        base: dict[int, list[int]] = {}
        for i in range(1, n):
            if (
                lo <= st[i, cv2.CC_STAT_HEIGHT] <= hi
                and st[i, cv2.CC_STAT_WIDTH] <= W * 0.06
            ):
                b = st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT]
                base.setdefault(int(b // max(2, xh // 2)), []).append(i)
        for ids in base.values():
            if len(ids) >= 6:
                for i in ids:
                    text[lab == i] = 255

    fig = bw.copy()
    fig[text > 0] = 0
    # Scan edges and the library footer fuse with the plate under the closing step and turn it into
    # one page-sized blob (which is then discarded).  Figures never touch the page edge in print.
    my, mx = int(EDGE_MARGIN * H), int(EDGE_MARGIN * W)
    fig[:my, :] = 0
    fig[H - my:, :] = 0
    fig[:, :mx] = 0
    fig[:, W - mx:] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    cl = cv2.morphologyEx(fig, cv2.MORPH_CLOSE, k, iterations=2)

    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(cl, 8)
    out = []
    scale = 72.0 / det_dpi
    for i in range(1, n2):
        x, y, ww, hh, a = st2[i]
        if ww < W * 0.12 or hh < H * 0.10:
            continue
        if ww > W * 0.95 and hh > H * 0.95:
            continue
        aspect = ww / max(1, hh)
        if aspect > 3.5 and hh < H * 0.22:
            continue
        if aspect < 0.2 and ww < W * 0.22:
            continue
        if _big_ink_share(bw[y:y + hh, x:x + ww], H) < 0.15:
            continue  # paragraph block, not a figure

        y, hh = _trim_stray_lines(bw, x, y, ww, hh, H)
        if max_caption_lines > 0:
            y, hh = _attach_captions(bw, x, y, ww, hh, max_caption_lines, H, W)
        # pad for breathing room, but never into a neighbouring text line
        PAD = 16

        def room(strip: np.ndarray) -> int:
            """strip: rows ordered outward from the figure edge. Returns pad that stays ink-free."""
            if strip.shape[0] == 0:
                return 0
            ink = np.where(strip.any(axis=1))[0]
            return strip.shape[0] if ink.size == 0 else max(0, int(ink[0]) - 1)

        top = room(bw[max(0, y - PAD):y, x:x + ww][::-1])
        bot = room(bw[y + hh:min(H, y + hh + PAD), x:x + ww])
        lef = room(bw[y:y + hh, max(0, x - PAD):x][:, ::-1].T)
        rig = room(bw[y:y + hh, x + ww:min(W, x + ww + PAD)].T)
        x0, y0 = max(0, x - lef), max(0, y - top)
        x1, y1 = min(W, x + ww + rig), min(H, y + hh + bot)
        rect = fitz.Rect(x0 * scale, y0 * scale, x1 * scale, y1 * scale)
        cp = page.get_pixmap(dpi=crop_dpi, clip=rect)
        arr = np.frombuffer(cp.samples, dtype=np.uint8).reshape(cp.height, cp.width, cp.n)
        crop = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR) if cp.n >= 3 else cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
        out.append((crop, tuple(rect)))
    return out


def save_png(path: str, bgr: np.ndarray, dpi: float) -> None:
    """Write PNG with the true pixel density in the file (cv2.imwrite leaves it unset -> viewers say 72)."""
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path, dpi=(round(dpi), round(dpi)))


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract figures from PDFs by geometric scoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", "-i", default=DEFAULT_DEST_DIR,
                   help="Directory containing source PDF files")
    p.add_argument("--image-dir", "-o", default=DEFAULT_IMAGE_OUT_DIR,
                   help="Directory where cropped figure PNGs are written")
    p.add_argument("--grid-dir", "-g", default=DEFAULT_GRID_OUT_DIR,
                   help="Directory where thumbnail matrix sheets are written")
    p.add_argument("--threshold", "-t", type=int, default=DEFAULT_THRESHOLD,
                   help="Minimum geometry score for a page to be processed")
    p.add_argument("--keep-fullpage", action="store_true",
                   help="Also save page-sized embedded layers (MRC scan background/mask); off by default")
    p.add_argument("--crop-dpi", type=int, default=0,
                   help="DPI for figure crops (default: max(200, native scan dpi))")
    p.add_argument("--captions", type=int, default=2,
                   help="Max centred title/caption lines to attach above and below a figure (0 = off)")
    p.add_argument("--eval-dpi", type=int, default=DEFAULT_EVAL_DPI,
                   help="DPI used for the cheap geometry scoring pass")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    for d in (args.input_dir, args.image_dir, args.grid_dir):
        os.makedirs(d, exist_ok=True)

    pdfs = sorted(glob.glob(os.path.join(args.input_dir, "*.pdf")))
    if not pdfs:
        print(f"No PDF files found in {args.input_dir}", file=sys.stderr)
        return 1

    for pdf_path in pdfs:
        pdf_name = Path(pdf_path).stem
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"  could not open {pdf_name}: {e}")
            continue

        print(f"Auditing {pdf_name} ({len(doc)} pp)...")
        thumbs: list[Image.Image] = []
        hits = 0

        for pn in range(len(doc)):
            page = doc[pn]
            fig_idx = 0

            # 1) Embedded images first
            embedded = extract_embedded_images(page, doc, args.keep_fullpage)
            for crop, idpi in embedded:
                out_path = os.path.join(
                    args.image_dir, f"{pdf_name}_p{pn+1}_img{fig_idx}.png"
                )
                save_png(out_path, crop, idpi)
                fig_idx += 1
            if embedded:
                print(f"  p{pn+1}: {len(embedded)} embedded image(s)")

            # 2) Geometry crops on high-score pages
            score, latt, box, sat = score_page_geometry(
                page.get_pixmap(dpi=args.eval_dpi)
            )
            if score >= args.threshold:
                hits += 1
                print(
                    f"  [HIT] p{pn+1}: score={score} "
                    f"(latt={latt} box={box} sat={sat})"
                )
                cdpi = int(round(args.crop_dpi or max(200.0, native_dpi(page))))
                for crop, _bbox in crop_figures(page, cdpi, args.captions):
                    if crop.size == 0:
                        continue
                    out_path = os.path.join(
                        args.image_dir,
                        f"{pdf_name}_p{pn+1}_fig{fig_idx}.png",
                    )
                    save_png(out_path, crop, cdpi)
                    fig_idx += 1

            if fig_idx > 0 or score >= args.threshold:
                pt = page.get_pixmap(dpi=50)
                t = Image.frombytes("RGB", [pt.width, pt.height], pt.samples)
                t.thumbnail((200, 260))
                thumbs.append(t)

        print(
            f"  {hits} of {len(doc)} pages passed "
            f"({100 * hits / max(1, len(doc)):.1f}%)"
        )

        for s in range(math.ceil(len(thumbs) / 64)):
            sheet = Image.new("RGB", (1600, 2080), (15, 0, 15))
            for i, th in enumerate(thumbs[s * 64 : (s + 1) * 64]):
                sheet.paste(th, ((i % 8) * 200, (i // 8) * 260))
                th.close()
            sheet_path = os.path.join(
                args.grid_dir, f"{pdf_name}_matrix_sheet_{s+1}.jpg"
            )
            sheet.save(sheet_path, quality=85)

        doc.close()

    print("\nSpatial shortlisting complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
