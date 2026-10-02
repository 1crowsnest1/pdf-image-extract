 /░░░░    /░░░     /░░░░                  /░░░    /░░░░░    /█▀▀█    /█▀▀▀    /█▀▀▀/       
│_▒ /▒   │-▒_/▒   │_▒__/                 │/_▒/   │ ▒ ▒ ▒   │ ▓▓▓▓   │ ▓/▓▓   │ ▓▓▓         
│ ▓▓▓/   │ ▓│ ▓   │ ▓▓▓                   │ ▓    │ ▓ ▓ ▓   │_▒_/▒   │ ▒//▒   │_▒_/         
│ █_/    │ ███/   │_█_/                   /███   │ █ █ █   │ ░│ ░   │ ░░░░   │ ░░░░        
│//      │/__/    │//                    │/__/   │//////   │//│//   │/___/   │/___/        
 /█▀▀▀/    /░ /░    /░░░    /░░░░    /█▀▀█    /░░░░    /░░░    /░░░    /░░░░
│ ▓▓▓     │//▒▒/   │//▒/   │_▒ /▒   │ ▓▓▓▓   │ ▒__/   │//▒/   │_▒/▒   │_▒ /▒
│_▒_/      │ ▓▓     │ ▓    │ ▓▓▓/   │_▒_/▒   │ ▓       │ ▓    │ ▓ ▓   │ ▓▓▓/
│ ░░░░     /█ /█    │ █    │ █_/█   │ ░│ ░   │ ████    │ █    │ ███   │ █_/█
│/___/    │//│//    │//    │// //   │//│//   │/___/    │//    │/__/   │// //


# PDF Image Extractor

Heuristically detect and extract figures, diagrams and illustration grids from PDF files.

The tool first scores every page with a fast geometry-based detector (looking for lattices, regular boxes and satellite blobs). Pages that pass a configurable threshold are rendered at higher resolution; text-like regions are masked out and the remaining large connected components are cropped and saved as PNG files. Low-resolution thumbnail matrix sheets are also produced for quick visual review.

## Features

- Pure-Python, no external system dependencies beyond the listed packages
- Geometry scoring (Hough lattices + box grids + satellite blobs) to skip text-only pages
- Text-region suppression before cropping so pure-text blocks are rarely saved
- Configurable input / output directories and score threshold via CLI
- Thumbnail contact sheets (8×8) written as JPEG for rapid browsing

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

1. Put the PDFs you want to process into a directory (default: `./dest_dir`).
2. Run:

```bash
python pdf_image_extract.py
```

Extracted figures appear in `./image_out_dir` as

```
<pdfname>_p<page>_fig<n>.png
```

Thumbnail matrix sheets appear in `./grid_out_dir`.

### Useful options

```bash
python pdf_image_extract.py \
  --input-dir  /path/to/pdfs \
  --image-dir  /path/to/figures \
  --grid-dir   /path/to/thumbnails \
  --threshold  80 \
  --eval-dpi   120
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-i / --input-dir` | `./dest_dir` | Folder containing source PDFs |
| `-o / --image-dir` | `./image_out_dir` | Where cropped PNG figures are written |
| `-g / --grid-dir` | `./grid_out_dir` | Where thumbnail matrix sheets are written |
| `-t / --threshold` | `70` | Minimum geometry score required to process a page |
| `--eval-dpi` | `150` | DPI used for the cheap scoring pass |

## How the scoring works

Three independent detectors contribute to a page score:

| Detector | Weight | Looks for |
|----------|--------|-----------|
| Lattice  | ×1     | Intersecting horizontal / vertical Hough lines forming a grid |
| Box      | ×9     | Regularly sized and spaced rectangular contours |
| Satellite| ×8     | Multiple mid-sized blobs arranged around a centre |

A page whose total score exceeds `--threshold` is considered figure-rich and is sent to the cropping stage.

## Limitations & tips

- The heuristics are tuned for scanned technical / scientific material. Highly artistic or photographic pages may need a lower threshold or manual review.
- Very large PDFs (hundreds of pages) are processed sequentially; memory usage stays modest because only one page is rendered at a time.
- The text-masking step uses a simple height-histogram heuristic; dense multi-column text can occasionally leak into crops. Raising the `_textiness` cutoff (currently 0.68) makes the filter stricter.

## License

MIT – see [LICENSE](LICENSE).

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
