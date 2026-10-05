  _______ ______   _______     ___                                 _______         __                    __              
 |   _   |   _  \ |   _   |   |   |.--------.---.-.-----.-----.   |   _   |.--.--.|  |_.----.---.-.----.|  |_.-----.----.
 |.  1   |.  |   \|.  1___|   |.  ||        |  _  |  _  |  -__|   |.  1___||_   _||   _|   _|  _  |  __||   _|  _  |   _|
 |.  ____|.  |    \.  __)     |.  ||__|__|__|___._|___  |_____|   |.  __)_ |__.__||____|__| |___._|____||____|_____|__|  
 |:  |   |:  1    /:  |       |:  |               |_____|         |:  1   |                                              
 |::.|   |::.. . /|::.|       |::.|                               |::.. . |                                              
 `---'   `------' `---'       `---'                               `-------'                                              
                                                                                                                         

# PDF Image Extractor

Heuristically detect and extract figures, diagrams and illustration grids from PDF files.

Pages are scored with a fast geometry detector (lattices, regular boxes, satellite blobs). High-scoring pages are cropped after text masking. **Embedded raster images** are always extracted when present.

## Features

- Embedded image extraction (`page.get_images`) → `*_imgN.png`
- Geometry scoring + strict text rejection → `*_figN.png`
- Paragraph-band filter (rejects wide, shallow text blocks)
- 8×8 thumbnail matrix sheets for quick review
- Configurable threshold and evaluation DPI

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# defaults: ./pdfs → ./image_out_dir + ./grid_out_dir
python pdf_image_extract.py

python pdf_image_extract.py -i ./pdfs -o ./figures -g ./sheets -t 80
```

| Flag | Default | Meaning |
|------|---------|---------|
| `-i / --input-dir` | `./pdfs` | Source PDFs |
| `-o / --image-dir` | `./image_out_dir` | Cropped figures |
| `-g / --grid-dir` | `./grid_out_dir` | Contact sheets |
| `-t / --threshold` | `70` | Min geometry score |
| `--eval-dpi` | `150` | Scoring pass DPI |

## Scoring

| Detector | Weight | Looks for |
|----------|--------|-----------|
| Lattice | ×2 | H/V grid lines |
| Box | ×9 | Regular rectangular contours |
| Satellite | ×7 | Mid-sized blobs around centre |

## Output naming

- `{name}_p{page}_img{n}.png` — embedded images  
- `{name}_p{page}_fig{n}.png` — geometry crops  
- `{name}_matrix_sheet_{n}.jpg` — contact sheets  

## License

MIT – see [LICENSE](LICENSE).
