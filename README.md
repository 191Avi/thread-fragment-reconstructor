# Thread — Fragment Reconstructor

Point this at a messy folder — notes, screenshots, half-written drafts, PDFs —
and get back a clear picture of what you were actually trying to do, grouped
into coherent "threads," each with a plain-language brief and a next-action
list.

## What it does

Say something like:

- "Reconstruct this folder for me"
- "What was I working on in here?"
- "I have a pile of screenshots and notes about something — help me figure out what it was"
- "Pull these scattered files into a project brief"

Claude will scan the folder, read across formats (OCR on screenshots, text
extraction from PDFs, direct reads of notes), group the fragments by what they
seem to be *about* rather than their file type or date, and hand back a brief
per group: what it looks like you were doing, which files belong to it, and
what to do next.

## How it works

The skill bundles a script (`scripts/thread.py`) that does the heavy lifting —
cross-format extraction plus topic clustering via TF-IDF and agglomerative
clustering — and writes a markdown report. Claude then reads that report
alongside the original fragments and synthesizes the final brief in plain
language, sharpening the script's first-pass guesses with its own judgment.

## Requirements

The script needs a few Python packages, which the skill installs on first run:

```
scikit-learn, pytesseract, pdfplumber, pillow, numpy
```

Image OCR additionally needs the `tesseract-ocr` system binary. If it's not
present, OCR on screenshots returns empty text but everything else (PDFs, text
notes) still works fine.

## Privacy

This skill only reads files in the folder you point it at — it never modifies,
moves, or deletes anything.
