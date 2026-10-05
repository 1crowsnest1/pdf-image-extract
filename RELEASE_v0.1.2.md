# v0.1.2 – Smarter crops, captions, true DPI

## What's new

- **Paragraph-only rejection** — text inside figures is kept; only paragraph-like blocks are dropped
- **Edge margin (3.5%)** — scan edges no longer swallow plates
- **Caption attachment** — centred titles/captions above/below figures (`--captions`, default 2)
- **Stray line trim** — body-text lines fused onto crops are removed
- **Smart padding** — pad does not reach into neighbouring text lines
- **Skip full-page scan layers** — MRC background/mask XObjects ignored unless `--keep-fullpage`
- **True DPI in PNGs** — density written into the file; auto from native scan or `--crop-dpi`

## Usage

```bash
python pdf_image_extract.py -i ./pdfs
python pdf_image_extract.py -i ./pdfs --crop-dpi 300 --captions 2
```

## Notes

Some partial line fragments at crop edges can still appear; check matrix sheets for QA.
