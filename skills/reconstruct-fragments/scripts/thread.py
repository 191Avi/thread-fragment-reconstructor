#!/usr/bin/env python3
"""
Thread -- Fragment Reconstructor (prototype)

Scans a folder of mixed, messy files (notes, screenshots, PDFs, text fragments),
extracts text from each regardless of format, clusters them by apparent intent,
and produces a plain-language brief + next-action list per cluster.

Usage:
    python3 thread.py <folder_to_scan> [--out report.md]

Supported inputs:
    .txt, .md           -> read directly
    .png, .jpg, .jpeg   -> OCR via pytesseract
    .pdf                -> text extraction via pdfplumber
"""

import argparse
import os
import sys
import re
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


SUPPORTED_TEXT = {".txt", ".md"}
SUPPORTED_IMAGE = {".png", ".jpg", ".jpeg"}
SUPPORTED_PDF = {".pdf"}


def extract_text_from_image(path):
    from PIL import Image
    import pytesseract
    try:
        img = Image.open(path)
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


def extract_text_from_pdf(path):
    import pdfplumber
    chunks = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                chunks.append(page.extract_text() or "")
    except Exception:
        pass
    return "\n".join(chunks)


def extract_text_from_txt(path):
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def load_fragments(folder):
    fragments = []
    for root, _dirs, files in os.walk(folder):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            full = os.path.join(root, fn)
            text = ""
            kind = None
            if ext in SUPPORTED_TEXT:
                text, kind = extract_text_from_txt(full), "note"
            elif ext in SUPPORTED_IMAGE:
                text, kind = extract_text_from_image(full), "screenshot"
            elif ext in SUPPORTED_PDF:
                text, kind = extract_text_from_pdf(full), "document"
            else:
                continue
            text = text.strip()
            if len(text) < 5:
                continue
            fragments.append({"path": full, "name": fn, "kind": kind, "text": text})
    return fragments


def cluster_fragments(fragments, distance_threshold=0.9):
    if len(fragments) <= 1:
        return [list(range(len(fragments)))]

    texts = [f["text"] for f in fragments]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=4000)
    matrix = vectorizer.fit_transform(texts)

    sim = cosine_similarity(matrix)
    dist = 1 - sim
    dist[dist < 0] = 0

    n = len(fragments)
    if n == 2:
        return [[0, 1]] if sim[0, 1] > 0.08 else [[0], [1]]

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(dist)

    groups = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[label].append(idx)
    return list(groups.values())


def top_keywords(texts, n=6):
    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=2000, ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(texts)
        scores = np.asarray(matrix.sum(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()
        ranked = sorted(zip(terms, scores), key=lambda x: -x[1])
        keywords = []
        for term, _score in ranked:
            if re.fullmatch(r"[\d\W_]+", term):
                continue
            if any(term in k or k in term for k in keywords):
                continue
            keywords.append(term)
            if len(keywords) >= n:
                break
        return keywords
    except Exception:
        return []


def first_sentence(text, max_len=160):
    text = re.sub(r"\s+", " ", text).strip()
    m = re.search(r".{20,}?[.!?](\s|$)", text)
    snippet = m.group(0).strip() if m else text[:max_len]
    truncated = len(snippet) > max_len or (not m and len(text) > max_len)
    return snippet[:max_len].rstrip() + ("..." if truncated else "")


def guess_next_actions(keywords, fragment_count, full_text=""):
    actions = []
    haystack = (" ".join(keywords) + " " + full_text).lower()

    def has(*words):
        for w in words:
            w = w.lower()
            # Word-boundary match for alphabetic tokens so 'rent' doesn't fire
            # on 'coherent'/'different'. Symbol tokens like '$' fall back to a
            # plain substring check, since \b doesn't apply to non-word chars.
            if all(c.isalnum() or c in " -" for c in w):
                if re.search(r"\b" + re.escape(w) + r"\b", haystack):
                    return True
            elif w in haystack:
                return True
        return False

    if has("proposal", "pitch", "deck", "presentation", "campaign", "client"):
        actions.append("Assemble the scattered drafts into one working proposal/deck and flag what's still missing (e.g. brand assets, confirmed budget).")
        actions.append("Follow up on anything you're waiting on from someone else before this can move forward.")
    if has("apartment", "lease", "rent", "landlord", "move-in", "move in"):
        actions.append("Put the listings/lease terms side by side and note the open questions (parking, pet policy, deposit) before deciding.")
        actions.append("Set a decision deadline so this doesn't stall -- these usually have a shelf life.")
    if has("meeting", "call", "agenda", "transcript", "memo"):
        actions.append("Turn these notes into a short follow-up with clear owners and dates.")
    if has("budget", "invoice", "cost", "price", "expense", "$"):
        actions.append("Reconcile the numbers mentioned across these fragments into one figure you trust.")
    if has("plan", "roadmap", "schedule", "timeline", "deadline"):
        actions.append("Merge the timeline fragments into a single plan so nothing slips through the cracks.")

    if not actions:
        actions.append("Review these fragments together and decide whether this thread is still active or can be archived.")
        actions.append("If it's active, write one line describing what 'done' looks like.")

    seen = set()
    deduped = []
    for a in actions:
        if a not in seen:
            seen.add(a)
            deduped.append(a)

    if fragment_count >= 4 and len(deduped) < 3:
        deduped.append("This thread has pulled together several fragments -- worth giving it its own folder or doc so it stays this coherent.")

    return deduped[:3]


def build_report(fragments, clusters):
    lines = ["# Thread -- Reconstructed Project Briefs\n"]
    lines.append("_Scanned %d fragments and grouped them into %d apparent threads._\n" % (len(fragments), len(clusters)))

    for i, idxs in enumerate(sorted(clusters, key=len, reverse=True), start=1):
        members = [fragments[j] for j in idxs]
        texts = [m["text"] for m in members]
        keywords = top_keywords(texts)
        title_guess = ", ".join(keywords[:3]).title() if keywords else "Untitled thread"

        lines.append("## Thread %d: %s" % (i, title_guess))
        lines.append("**Likely about:** %s" % (", ".join(keywords) if keywords else "(not enough text to tell)"))
        lines.append("**Fragments involved (%d):**" % len(members))
        for m in members:
            lines.append("- `%s` (%s) -- \"%s\"" % (m["name"], m["kind"], first_sentence(m["text"])))

        lines.append("\n**Suggested next actions:**")
        for action in guess_next_actions(keywords, len(members), full_text=" ".join(texts)):
            lines.append("- [ ] %s" % action)
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Reconstruct intent from a folder of messy fragments.")
    parser.add_argument("folder", help="Folder to scan")
    parser.add_argument("--out", default="thread_report.md", help="Output markdown report path")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print("Not a folder: %s" % args.folder, file=sys.stderr)
        sys.exit(1)

    print("Scanning %s ..." % args.folder)
    fragments = load_fragments(args.folder)
    print("Found %d readable fragments." % len(fragments))

    if not fragments:
        print("Nothing to reconstruct -- no readable fragments found.")
        sys.exit(0)

    clusters = cluster_fragments(fragments)
    report = build_report(fragments, clusters)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)

    print("Wrote report to %s\n" % args.out)
    print(report)


if __name__ == "__main__":
    main()
