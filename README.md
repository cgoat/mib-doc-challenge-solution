# MIB Doc Challenge — offline document pipeline

Reads a directory of MIB intake packet PDFs and writes `predictions.jsonl`:
extracted applicant record plus an `APPROVED` / `DENIED` / `NEEDS_REVIEW`
adjudication and a calibrated confidence.

No network, no API keys, no LLM. PP-OCRv4 via ONNX Runtime + classical CV + a rules engine, plus
one small logistic model (27 weights, ~2 KB) fitted on the public training split
for confidence calibration.

## Run it

```bash
docker build -t mib-submission .
mkdir -p /tmp/mib-output
docker run --rm --network none \
  --mount type=bind,src="$PWD/../mib-doc-challenge/data/train",dst=/input,readonly \
  --mount type=bind,src=/tmp/mib-output,dst=/output \
  mib-submission /input /output/predictions.jsonl
```

Locally, without Docker (needs Tesseract on `PATH`, or set `TESSERACT_CMD`):

```bash
python -m mib.main <input_pdf_dir> <output_predictions_path>
python -m pytest tests/ -q
```

## Score on the public training split

`122.34 / 150` — extraction `42.55/50`, classification `63.66/80`, calibration
`16.12/20`, no missing cases, 6 catastrophic false approvals out of 431 denials.
Measured with the challenge's own `scripts/evaluate.py`.

## How it works

| Module | Responsibility |
| --- | --- |
| `mib/pages.py` | Splits each page's text layer into trusted visible spans and untrusted hidden ones; classifies the page kind |
| `mib/ocr.py` | Reads scanned pages with PP-OCRv4; the older Tesseract line-at-a-time path is kept behind `MIB_OCR_ENGINE` |
| `mib/parse.py` | Turns page lines into typed field candidates, in either column layout |
| `mib/lexicon.py` | Snaps noisy OCR to the closed vocabularies, with a distance threshold so unseen values pass through |
| `mib/document.py` | Merges page evidence into one packet record, weighted by the field manual's precedence |
| `mib/rules.py` | Adjudication policy and defaulting |
| `mib/confidence.py` | Calibrated probability that the adjudication is correct |
| `mib/main.py` | Parallel driver, output contract, time-budget fallback |

Design notes, failure modes and what I would do next are in the submission memo.

### Trust model

Hidden text — white fill, sub-6pt type, or drawn outside the page crop — is
separated from the visible text at read time and never reaches the field
parser. Large decorative overlays (`SAMPLE DENIAL`) are classified as
watermarks, not findings. Injected content is recorded as a signal for
calibration but is never used as evidence.

## Regenerating the fitted artefacts

Both are derived from the public training split only:

```bash
python tools/extract_cache.py <pdf_dir> /tmp/train_cache.jsonl     # cache extractions
python tools/fit_calibration.py <train_labels.csv> /tmp/train_cache.jsonl  # -> mib/calibration.json
python tools/score_cache.py <train_labels.csv> /tmp/train_cache.jsonl      # score without re-OCR
```

`mib/lexicon.json` holds the closed vocabularies (species codes, home worlds,
visa classes, purposes, risk flags, and the name-part lists) read off the same
labels. Nothing is keyed to a case id or a filename.

## License

MIT — see [LICENSE](LICENSE). The challenge repository and its public dataset
are MIT-licensed too; the PP-OCRv4 models shipped inside
`rapidocr-onnxruntime` are Apache 2.0.
