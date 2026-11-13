# JAIHC submission package: AVB-Engage

Prepared 2026-11-13 from `research/paper/AVB-Engage_JAIHC_final/` against the journal's submission guidelines
(link.springer.com/journal/12652/submission-guidelines) and the Springer Nature LaTeX template manual
(`research/paper/user-manual.pdf`). Details of every check: `docs/plan.md` §21.

## What was checked

- 12 pages including references, two-column `\documentclass[pdflatex,iicol,sn-basic]{sn-jnl}`, author-year references
- abstract 246 words (150–250), 6 keywords (4–6), at most three heading levels
- "Statements and Declarations" before the references; LLM use documented in the Methods (Section 4.3)
- references alphabetical, DOI links where a DOI exists, BibTeX with `sn-basic.bst`
- table footnotes as lower-case letters below the table; captions without final punctuation or colour words
- figures: Fig1 84 mm and Fig2 174 mm wide (Fig3 71 mm), Helvetica-style lettering at 8–9 pt, contrast ≥ 4.5:1,
  EPS (vector, fonts embedded) plus PDF, file names `Fig1`, `Fig2`, `Fig3`
- Online Resource 1 (`ESM_1.pdf`) carries the title, journal, authors and corresponding-author block
- no overfull boxes, no undefined references; numbers checked against the result files and the code

## Files (in `AVB-Engage_JAIHC_submission.zip`)

| Upload as | File(s) |
|---|---|
| Manuscript, LaTeX source | `1_manuscript_source/paper.tex`, `references.bib`, `sn-jnl.cls`, `sn-basic.bst` |
| Figures | `1_manuscript_source/Fig1.eps`, `Fig2.eps`, `Fig3.eps` (the `.pdf` versions are used when the system compiles with pdfLaTeX) |
| Manuscript PDF, for checking the system's build | `paper.pdf` |
| Supplementary material, "Online Resource 1" | `ESM_1.pdf` (caption: decoding rules, threshold search, metrics, cross-attention configuration, training procedure and configuration, full ablation, per-level, cost and dataset tables, and paired significance tests) |
| Cover letter | `cover_letter.pdf` (draft: fill in `cover_letter.tex` and rebuild first) |

## Still to do by the authors, before uploading

1. Title block in `paper.tex` and `supplementary.tex`: author names, affiliations (department, institution, city,
   state, country), corresponding author's e-mail, ORCIDs.
2. Statements and Declarations in `paper.tex`: Funding, Ethics (EngageNet licence terms; whether the institution
   needs an exemption statement), Code availability, Author contributions (CRediT), the AI statement (marked DRAFT).
   Delete the four `% TODO (authors)` comments once these are filled in, since the source goes to the journal.
3. `cover_letter.tex`: names, affiliation, e-mail, ORCID; optional suggested reviewers with institutional e-mails.
4. Sanchita's agreement that Figs. 1–2 were generated from code written with an AI assistant (the AI statement says so).
5. Rebuild `paper.pdf` and `ESM_1.pdf` and confirm 12 pages (page 12 has about 25 free lines for the real author block).

## Also required in the submission form itself

- Author contributions and competing interests: the journal publishes only what is entered in the form.
- Funding information (it goes to CrossMark).
- Graphics programs: Figs. 1–2 draw.io (diagrams.net), Fig. 3 TikZ (LaTeX).
- Article type: original research article.

## Overleaf

The Overleaf project "Springer Nature LaTeX Template" holds the same manuscript files in its `SpringerNature/` folder
(main document `SpringerNature/paper.tex`, pdfLaTeX, TeX Live 2026), and this checklist with `cover_letter.tex` in
its `submission/` folder. Built there on 2026-11-13: paper 12 pages, supplement 11 pages, 0 errors. The three hyperref
warnings ("Difference ... between bookmark levels") come from the template's `\bmhead`, a level-5 heading that follows
a level-1 section, as the template manual prescribes for the back matter; they only flatten the PDF bookmark tree,
not the printed pages.
Overleaf compiles only the main document: to get `ESM_1.pdf` or `cover_letter.pdf`, set that file as the main document
(right-click it in the file tree, or Settings, Main document), recompile, download the PDF, and set `paper.tex` back.

## Rebuilding

From `AVB-Engage_JAIHC_final/`: pdfLaTeX + BibTeX on `paper.tex` and `supplementary.tex` (Overleaf does this). Figures:
`figures/src/make_*_drawio.py` → `figures/src/export_drawio.js` → `FigN.pdf`; `figures/Fig3_source.tex` (XeTeX) →
`Fig3.pdf`; then `gs -q -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile=FigN.eps FigN.pdf`.
