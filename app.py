#!/usr/bin/env python3
"""
Line Cropper web UI — multiple images + PDF support.
"""

import base64
import io
import os
import zipfile
from typing import List, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request

from line_cropper import (
    MAX_SIDE,
    crop_line,
    crop_page,
    deskew,
    segment_lines,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024  # 40 MB total

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Line Cropper</title>
  <style>
    :root {
      --bg: #0f1115;
      --card: #1a1d24;
      --border: #2a2f3a;
      --text: #e8eaed;
      --muted: #9aa0a6;
      --accent: #7c9cff;
      --accent2: #4ade80;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 24px 16px 48px;
      line-height: 1.5;
    }
    .container { max-width: 980px; margin: 0 auto; }
    h1 { font-size: 1.55rem; font-weight: 650; margin-bottom: 6px; }
    .sub { color: var(--muted); margin-bottom: 24px; font-size: 0.95rem; }
    .drop {
      border: 2px dashed var(--border);
      border-radius: 14px;
      background: var(--card);
      padding: 36px 20px;
      text-align: center;
      cursor: pointer;
      transition: border-color .15s, background .15s;
      margin-bottom: 16px;
    }
    .drop.dragover { border-color: var(--accent); background: #1e2430; }
    .drop p { color: var(--muted); margin-top: 8px; font-size: 0.9rem; }
    input[type=file] { display: none; }
    .btn {
      display: inline-block;
      background: var(--accent);
      color: #0f1115;
      font-weight: 600;
      padding: 10px 20px;
      border-radius: 8px;
      border: none;
      cursor: pointer;
      font-size: 0.95rem;
      margin-top: 10px;
      text-decoration: none;
    }
    .status { margin: 14px 0; color: var(--muted); min-height: 1.4em; font-size: 0.95rem; }
    .status.ok { color: var(--accent2); }
    .status.err { color: #f87171; }
    .results { display: none; margin-top: 20px; }
    .results.show { display: block; }
    .page-block {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 20px;
    }
    .page-block h2 { font-size: 1rem; font-weight: 600; margin-bottom: 12px; }
    .page-block h3 { font-size: 0.85rem; color: var(--muted); font-weight: 500; margin: 14px 0 8px; }
    .page-block img.debug { max-width: 100%; height: auto; border-radius: 8px; background: #111; display: block; }
    .lines { display: flex; flex-direction: column; gap: 10px; }
    .line-item {
      background: #111; border-radius: 8px; padding: 8px 10px;
      display: flex; align-items: center; gap: 10px;
    }
    .line-item span { font-size: 0.78rem; color: var(--muted); min-width: 54px; flex-shrink: 0; }
    .line-item img { max-height: 64px; width: auto; border-radius: 4px; }
    .actions { margin-top: 16px; display: flex; gap: 10px; flex-wrap: wrap; }
    .hint { font-size: 0.82rem; color: var(--muted); margin-top: 8px; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Line Cropper</h1>
    <p class="sub">Drop one or more page photos or a PDF. Empty ruled lines are ignored; only real handwriting/print is kept.</p>

    <div class="drop" id="drop">
      <div style="font-size: 1.8rem;">📄</div>
      <p>Drag & drop images or a PDF here<br><span style="opacity:.7">or click to choose (multiple allowed)</span></p>
      <button class="btn" type="button" id="pick">Choose files</button>
      <input type="file" id="file" accept="image/*,application/pdf" multiple>
    </div>

    <div class="status" id="status"></div>
    <div class="results" id="results">
      <div id="pages"></div>
      <div class="actions">
        <a class="btn" id="zip" href="#" download="line_crops.zip">Download all (ZIP)</a>
      </div>
    </div>
  </div>

  <script>
    const drop = document.getElementById('drop');
    const fileInput = document.getElementById('file');
    const pick = document.getElementById('pick');
    const status = document.getElementById('status');
    const results = document.getElementById('results');
    const pagesEl = document.getElementById('pages');
    const zipLink = document.getElementById('zip');

    pick.onclick = (e) => { e.stopPropagation(); fileInput.click(); };
    drop.onclick = (e) => { if (e.target === drop || e.target.tagName === 'P' || e.target.tagName === 'DIV') fileInput.click(); };

    ['dragenter','dragover'].forEach(ev => {
      drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('dragover'); });
    });
    ['dragleave','drop'].forEach(ev => {
      drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('dragover'); });
    });
    drop.addEventListener('drop', e => {
      if (e.dataTransfer.files.length) processFiles(e.dataTransfer.files);
    });
    fileInput.addEventListener('change', () => {
      if (fileInput.files.length) processFiles(fileInput.files);
    });

    async function processFiles(fileList) {
      status.textContent = 'Processing… this can take a few seconds per page';
      status.className = 'status';
      results.classList.remove('show');
      pagesEl.innerHTML = '';

      const fd = new FormData();
      for (const f of fileList) fd.append('files', f);

      try {
        const res = await fetch('/process', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed');

        let totalLines = 0;
        data.pages.forEach((page) => {
          totalLines += page.lines.length;
          const block = document.createElement('div');
          block.className = 'page-block';
          let linesHtml = page.lines.map((b64, i) =>
            `<div class="line-item"><span>Line ${String(i+1).padStart(2,'0')}</span>
             <img src="data:image/png;base64,${b64}" alt="line ${i+1}"></div>`
          ).join('');
          block.innerHTML = `
            <h2>${page.label} · ${page.lines.length} lines (pitch ~${page.pitch}px)</h2>
            <h3>Straightened page + red seams</h3>
            <img class="debug" src="data:image/png;base64,${page.debug}" alt="seams">
            <h3>Cropped lines</h3>
            <div class="lines">${linesHtml || '<p class="hint">No content lines found</p>'}</div>
          `;
          pagesEl.appendChild(block);
        });

        zipLink.href = 'data:application/zip;base64,' + data.zip;
        results.classList.add('show');
        status.textContent = `Done — ${data.pages.length} page(s), ${totalLines} lines total`;
        status.className = 'status ok';
      } catch (err) {
        status.textContent = err.message || 'Something went wrong';
        status.className = 'status err';
      }
    }
  </script>
</body>
</html>
"""


def encode_png(img: np.ndarray) -> str:
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def process_bgr(img: np.ndarray) -> dict:
    page = crop_page(img)
    h, w = page.shape[:2]
    if max(h, w) > MAX_SIDE:
        f = MAX_SIDE / max(h, w)
        page = cv2.resize(page, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)

    gray = deskew(cv2.cvtColor(page, cv2.COLOR_BGR2GRAY))
    ink, seams, pitch = segment_lines(gray)

    debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for sm in seams:
        pts = np.stack((np.arange(len(sm)), sm), axis=1).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(debug, [pts], False, (0, 0, 255), 2)

    lines_b64 = []
    line_pngs: List[Tuple[str, bytes]] = []
    for i in range(len(seams) - 1):
        crop = crop_line(gray, ink, seams[i], seams[i + 1])
        if crop is None:
            continue
        lines_b64.append(encode_png(crop))
        ok, buf = cv2.imencode(".png", crop)
        if ok:
            line_pngs.append((f"line_{len(line_pngs)+1:03d}.png", buf.tobytes()))

    ok, dbg_buf = cv2.imencode(".png", debug)
    ok2, page_buf = cv2.imencode(".png", gray)
    return {
        "pitch": int(pitch) if pitch else 0,
        "debug": encode_png(debug),
        "lines": lines_b64,
        "debug_bytes": dbg_buf.tobytes() if ok else b"",
        "page_bytes": page_buf.tobytes() if ok2 else b"",
        "line_pngs": line_pngs,
    }


def pdf_to_images(pdf_bytes: bytes, max_pages: int = 30) -> List[np.ndarray]:
    try:
        import fitz
    except ImportError:
        raise RuntimeError("PDF support requires pymupdf")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    doc.close()
    return images


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/process", methods=["POST"])
def process():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    pages_out = []
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        page_idx = 0
        for f in files:
            name = (f.filename or "file").lower()
            data = f.read()
            if not data:
                continue

            is_pdf = name.endswith(".pdf") or (f.mimetype or "").startswith("application/pdf")
            try:
                if is_pdf:
                    imgs = pdf_to_images(data)
                    labels = [f"{f.filename} — page {i+1}" for i in range(len(imgs))]
                else:
                    nparr = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None:
                        return jsonify({"error": f"Could not decode image: {f.filename}"}), 400
                    imgs = [img]
                    labels = [f.filename or f"image_{page_idx+1}"]
            except Exception as e:
                return jsonify({"error": f"Failed to open {f.filename}: {e}"}), 400

            for img, label in zip(imgs, labels):
                page_idx += 1
                try:
                    result = process_bgr(img)
                except Exception as e:
                    return jsonify({"error": f"Processing failed on {label}: {e}"}), 500

                folder = f"page_{page_idx:02d}"
                zf.writestr(f"{folder}/debug_seams.png", result["debug_bytes"])
                zf.writestr(f"{folder}/page.png", result["page_bytes"])
                for lname, ldata in result["line_pngs"]:
                    zf.writestr(f"{folder}/{lname}", ldata)

                pages_out.append({
                    "label": label,
                    "pitch": result["pitch"],
                    "debug": result["debug"],
                    "lines": result["lines"],
                })

    if not pages_out:
        return jsonify({"error": "No processable pages found"}), 400

    zip_b64 = base64.b64encode(zip_buf.getvalue()).decode("ascii")
    return jsonify({"pages": pages_out, "zip": zip_b64})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), debug=True)
