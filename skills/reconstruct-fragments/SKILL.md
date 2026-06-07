---
name: reconstruct-fragments
description: Reconstructs the intent behind a messy folder of notes, screenshots, and PDFs by clustering fragments by topic and producing a plain-language brief with next-action lists per thread. Trigger when the user says things like "reconstruct this folder," "what was I working on here," "make sense of these files," "pull these scattered notes into a project brief," "I have a pile of screenshots and notes about something, help me figure out what it was," or points at a folder full of mixed file types and asks what it adds up to.
---

# Reconstruct fragments into project threads

This skill turns a folder of disconnected fragments — text notes, markdown files,
screenshots, and PDFs — into a small set of "threads": coherent project briefs,
each with the fragments that belong to it and a short list of suggested next
actions. It is for the moment someone opens a folder and thinks "wait, what was
I even doing here?"

## When to use this

Use this when the user wants to:
- Make sense of a messy folder, downloads directory, or project workspace
- Recover context on something they set aside and came back to later
- Turn scattered screenshots/notes/PDFs into a single brief they can act on
- Figure out what a pile of half-finished material was actually for

## How to run it

A working script is bundled at `scripts/thread.py`. It does cross-format text
extraction (OCR for images via pytesseract, text extraction for PDFs via
pdfplumber, direct read for .txt/.md), TF-IDF + clustering to group fragments by
topic, and writes a markdown report.

1. Confirm the target folder with the user (ask for the path if it isn't obvious
   from context — never guess at a folder containing personal files).
2. Make sure the Python dependencies are available, installing if needed:
   ```bash
   pip install scikit-learn pytesseract pdfplumber pillow numpy --break-system-packages
   ```
   (`pytesseract` also needs the `tesseract-ocr` system binary for image OCR —
   if it's missing, OCR on screenshots will silently return empty text, which
   is fine; PDFs and text notes still work.)
3. Run the script against the folder:
   ```bash
   python3 scripts/thread.py "<folder_path>" --out thread_report.md
   ```
4. Read the generated `thread_report.md` and present it to the user. Don't just
   dump the raw file — walk them through what the threads seem to be, in plain
   language, the way a sharp assistant would after going through the pile
   themselves.

## Going beyond the script's heuristics

The bundled script uses keyword heuristics to guess thread titles and next
actions — that's a fast first pass, not the ceiling. After reading the
generated report and the actual fragment contents, use your own judgment to:

- Sharpen vague thread titles into something specific and accurate
- Rewrite the next-action lists so they reflect what's actually in the
  fragments, not just keyword matches (e.g. naming the actual people, dates,
  and decisions mentioned)
- Flag anything that looks like it's missing or blocked
- Note if two "threads" the script split apart actually look related, or if
  one "thread" looks like it's secretly two different things

Treat the script's output as raw material for your own synthesis, not the
final answer to hand back.

## Notes

- This only reads files — it never modifies, moves, or deletes anything in the
  user's folder.
- If the folder is large, mention how many fragments were found and give the
  user the option to narrow the scope (e.g. a subfolder or date range) before
  running a full scan.
- If OCR or PDF extraction comes back empty for a file, say so plainly rather
  than guessing at its content.
