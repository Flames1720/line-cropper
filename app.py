#!/usr/bin/env python3
"""
Simple Flask web UI around line_cropper.py
Drop a page photo → get straightened page, debug seams, and individual line crops.
"""

import io
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_file

# Import the core logic from the original script
from line_cropper import (
    crop_page,
    deskew,
    segment_lines,
    crop_line,
    MAX_SIDE,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024  # 12 MB

HTML = """
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
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 24px 16px 48px;
      line-height: 1.5;
    }
    .container { max-width: 960px; margin: 0 auto; }
    h1 { font-size: 1.6rem; font-weight: 650; margin-bottom: 6px; }
    .sub { color: var(--muted); margin-bottom: 28px; font-size: 0.95rem; }
    .drop {
      border: 2px dashed var(--border);
      border-radius: 14px;
      background: var(--card);
      padding: 40px 24px;
      text-align: center;
      cursor: pointer;
      transition: border-color .15s, background .15s;
      margin-bottom: 20px;
    }
    .drop.dragover { border-color: var(--accent); background: #1e2430; }
    .drop p { color: var(--muted); margin-top: 8px; font-size: 0.9rem; }
    input[type=file] { display: none; }
    .btn {
      display: inline-block;
      background: var(--accent);
      color: #0f1115;
      font-weight: 600;
      padding: 10px 22px;
      border-radius: 8px;
      border: none;
      cursor: pointer;
      font-size: 0.95rem;
      margin-top: 12px;
    }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .status { margin: 16px 0; color: var(--muted); min-height: 1.4em; }
    .status.ok { color: var(--accent2); }
    .status.err { color: #f87171; }
    .results { display: none; margin-top: 28px; }
    .results.show { display: block; }
    .grid { display: grid; gap: 20px; }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      overflow: hidden;
    }
    .card h3 { font-size: 0.95rem; margin-bottom: 12px; color: var(--muted); font-weight: 500; }
    .card img {
      max-width: 100%;
      height: auto;
      border-radius: 8px;
      display: block;
      background: #111;
    }
    .lines {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .line-item {
      background: #111;
      border-radius: 8px;
      padding: 10px;
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .line-item span {
      font-size: 0.8rem;
      color: var(--muted);
      min-width: 56px;
    }
    .line-item img {
      max-height: 70px;
      width: auto;
      border-radius: 4px;
    }
    .actions { margin-top: 18px; display: flex; gap: 12px; flex-wrap: wrap; }
    a.btn { text-decoration: none; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Line Cropper</h1>
    <p class="sub">Drop a photo of a page. It will straighten the page, find the gaps between lines, and cut each line out cleanly for OCR.</p>

    <div class="drop" id="drop">
      <div style="font-size: 2rem;">📄</div>
      <p>Drag & drop a page photo here, or click to choose</p>
      <button class="btn" type="button" id="pick">Choose image</button>
      <input type="file" id="file" accept="image/*">
    </div>

    <div class="status" id="status"></div>

    <div class="results" id="results">
      <div class="grid">
        <div class="card">
          <h3>Straightened page + red seams</h3>
          <img id="debug" alt="debug seams">
        </div>
        <div class="card">
          <h3>Cropped lines <span id="count"></span></h3>
          <div class="lines" id="lines"></div>
        </div>
      </div>
      <div class="actions">
        <a class="btn" id="zip" href="#" download="lines.zip">Download all lines (ZIP)</a>
      </div>
    </div>
  </div>

  <script>
    const drop = document.getElementById('drop');
    const fileInput = document.getElementById('file');
    const pick = document.getElementById('pick');
    const status = document.getElementById('status');
    const results = document.getElementById('results');
    const debugImg = document.getElementById('debug');
    const linesEl = document.getElementById('lines');
    const countEl = document.getElementById('count');
    const zipLink = document.getElementById('zip');

    pick.onclick = () => fileInput.click();
    drop.onclick = (e) => { if (e.target === drop) fileInput.click(); };

    ['dragenter','dragover'].forEach(ev => {
      drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('dragover'); });
    });
    ['dragleave','drop'].forEach(ev => {
      drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('dragover'); });
    });
    drop.addEventListener('drop', e => {
      const f = e.dataTransfer.files[0];
      if (f) process(f);
    });
    fileInput.addEventListener('change', () => {
      if (fileInput.files[0]) process(fileInput.files[0]);
    });

    async function process(file) {
      status.textContent = 'Processing… this can take a few seconds';
      status.className = 'status';
      results.classList.remove('show');
      linesEl.innerHTML = '';

      const fd = new FormData();
      fd.append('image', file);

      try {
        const res = await fetch('/process', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Failed');

        debugImg.src = 'data:image/png;base64,' + data.debug;
        countEl.textContent = `(${data.lines.length})`;
        data.lines.forEach((b64, i) => {
          const div = document.createElement('div');
          div.className = 'line-item';
          div.innerHTML = `<span>Line ${String(i+1).padStart(2,'0')}</span><img src="data:image/png;base64,${b64}" alt="line ${i+1}">`;
          linesEl.appendChild(div);
        });
        zipLink.href = 'data:application/zip;base64,' + data.zip;
        results.classList.add('show');
        status.textContent = `Done — ${data.lines.length} lines extracted (pitch ~${data.pitch}px)`;
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
    """Return base64-encoded PNG of a BGR or grayscale image."""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    import base64
    return base64.b64encode(buf.tobytes()).decode("ascii")


def process_image(file_bytes: bytes):
    """Run the full pipeline and return dict with base64 results."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    page = crop_page(img)
    h, w = page.shape[:2]
    if max(h, w) > MAX_SIDE:
        f = MAX_SIDE / max(h, w)
        page = cv2.resize(page, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)

    gray = deskew(cv2.cvtColor(page, cv2.COLOR_BGR2GRAY))
    ink, seams, pitch = segment_lines(gray)

    # Debug image with red seams
    debug = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for sm in seams:
        pts = np.stack((np.arange(len(sm)), sm), axis=1).reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(debug, [pts], False, (0, 0, 255), 2)

    lines_b64 = []
    line_pngs = []
    for i in range(len(seams) - 1):
        crop = crop_line(gray, ink, seams[i], seams[i + 1])
        if crop is None:
            continue
        b64 = encode_png(crop)
        lines_b64.append(b64)
        ok, buf = cv2.imencode(".png", crop)
        if ok:
            line_pngs.append((f"line_{len(line_pngs)+1:03d}.png", buf.tobytes()))

    # ZIP of all lines
    import base64
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in line_pngs:
            zf.writestr(name, data)
        # also include the debug seams
        ok, dbg_buf = cv2.imencode(".png", debug)
        if ok:
            zf.writestr("debug_seams.png", dbg_buf.tobytes())
        ok, page_buf = cv2.imencode(".png", gray)
        if ok:
            zf.writestr("page.png", page_buf.tobytes())
    zip_b64 = base64.b64encode(zip_buf.getvalue()).decode("ascii")

    return {
        "pitch": int(pitch),
        "debug": encode_png(debug),
        "lines": lines_b64,
        "zip": zip_b64,
    }


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    try:
        data = process_image(f.read())
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# For local testing
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), debug=True)
