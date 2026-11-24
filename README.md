# Line Cropper

Page photo → individual text-line images ready for OCR.

## What it does

1. Finds the page against the background and warps it flat.
2. Deskews so lines are roughly horizontal.
3. Estimates line height and finds gaps.
4. Runs a dynamic-programming seam through the emptiest white space between lines (bends around handwriting).
5. Cuts each line out cleanly.

## CLI

```bash
pip install opencv-python-headless numpy
python line_cropper.py photo.jpg out_folder
```

Outputs:
- `page.png` – cropped & straightened page
- `lines/line_001.png` … – individual line crops
- `debug_seams.png` – red seam paths overlaid (check this first)

## Web UI (local)

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:7860 , drop a page photo, see the seams + cropped lines, download a ZIP.

## Deploy notes

- **Vercel**: OpenCV is large; serverless functions may hit size limits. Prefer Hugging Face Spaces (Gradio/Streamlit) or a small VPS for reliable OpenCV usage.
- The original script already works well as a CLI or inside any Python environment that can install `opencv-python-headless`.
