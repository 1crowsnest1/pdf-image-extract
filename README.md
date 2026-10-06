# 🇵​​🇩​​🇫​ ​🇮​​🇲​​🇦​​🇬​​🇪​ ​🇪​​🇽​​🇹​​🇷​​🇦​​🇨​​🇹​​🇴​​🇷​

Heuristically detect and extract figures, diagrams and illustration grids from PDF files.

Pages are scored with a fast geometry detector (lattices, regular boxes, satellite blobs). High-scoring pages are cropped after text masking. **Embedded raster images** are extracted when present (page-sized MRC scan layers are skipped by default).

## 𝔽𝕖𝕒𝕥𝕦𝕣𝕖𝕤 (𝕧𝟘.𝟙.𝟚)

- Embedded image extraction → `*_imgN.png` (skips full-page scan layers)
- Geometry scoring + **paragraph-only** rejection (text *inside* figures is kept)
- Edge margin so scan borders cannot swallow plates
- Trim body-text lines fused onto figure crops
- Attach centred titles/captions (up to N lines)
- Smart padding that does not bleed into neighbouring text
- PNG files tagged with true DPI (native scan density or `--crop-dpi`)
- 8×8 thumbnail matrix sheets

## 𝕀𝕟𝕤𝕥𝕒𝕝𝕝𝕒𝕥𝕚𝕠𝕟

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 𝕌𝕤𝕒𝕘𝕖

```bash
python pdf_image_extract.py -i ./pdfs

python pdf_image_extract.py -i ./pdfs -o ./figures -g ./sheets -t 80
python pdf_image_extract.py -i ./pdfs --crop-dpi 300 --captions 2
python pdf_image_extract.py -i ./pdfs --keep-fullpage   # also save page-sized layers

```
### ℂ𝕃𝕀 𝕠𝕡𝕥𝕚𝕠𝕟𝕤

| Flag | Default | Meaning |
|------|---------|---------|
| `-i / --input-dir` | `./pdfs` | Source PDFs |
| `-o / --image-dir` | `./image_out_dir` | Cropped figures |
| `-g / --grid-dir` | `./grid_out_dir` | Contact sheets |
| `-t / --threshold` | `70` | Min geometry score |
| `--eval-dpi` | `150` | Scoring pass DPI |
| `--crop-dpi` | `0` (auto) | Crop render DPI; `0` = max(200, native scan dpi) |
| `--captions` | `2` | Max centred title/caption lines to attach (0 = off) |
| `--keep-fullpage` | off | Also save page-sized embedded layers (MRC background/mask) |

## 𝕊𝕔𝕠𝕣𝕚𝕟𝕘

| Detector | Weight | Looks for |
|----------|--------|-----------|
| Lattice | ×2 | H/V grid lines |
| Box | ×9 | Regular rectangular contours |
| Satellite | ×7 | Mid-sized blobs around centre |

## 𝕆𝕦𝕥𝕡𝕦𝕥 𝕟𝕒𝕞𝕚𝕟𝕘

- `{name}_p{page}_img{n}.png` — embedded images  
- `{name}_p{page}_fig{n}.png` — geometry crops  
- `{name}_matrix_sheet_{n}.jpg` — contact sheets  

## ℕ𝕠𝕥𝕖𝕤

- A crop is rejected only when it looks like a **paragraph** (low share of large ink structures). Text inside a plate stays.
- The outer ~3.5% of the page is ignored when locating figures so scan edges do not fuse with the plate.
- Occasional partial line fragments at crop edges may still appear; they show up on the thumbnail sheets for review.

## 𝕃𝕚𝕔𝕖𝕟𝕤𝕖

MIT – see [LICENSE](LICENSE).
