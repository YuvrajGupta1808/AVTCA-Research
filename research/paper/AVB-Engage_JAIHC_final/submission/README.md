# JAIHC submission package: AVB-Engage

Prepared 2026-11-13 from `research/paper/AVB-Engage_JAIHC_final/` against the journal's submission guidelines
(link.springer.com/journal/12652/submission-guidelines) and the Springer Nature LaTeX template manual
(`research/paper/user-manual.pdf`). Details of every check: `docs/plan.md` §21. Revised 2026-11-14 after the full
review of the paper, the supplement and the code (`docs/plan.md` §22): fixes P1–P7 and the smaller items of §22.4 are
in `paper.tex`, `supplementary.tex` and `references.bib`, and the PDFs and this zip were rebuilt from them.

## What was checked

- 12 pages including references, two-column `sn-jnl` class with the options `pdflatex,iicol,sn-basic`, author-year references
- abstract 246 words (150–250), 6 keywords (4–6), at most three heading levels
- "Statements and Declarations" before the references; LLM use documented in the Methods (Section 4.3)
- references alphabetical, DOI links where a DOI exists, BibTeX with `sn-basic.bst`; 23 references, all cited
  (Chumachenko et al. 2022, the network this code descends from, added on 2026-11-14; every field checked on Crossref)
- 2026-11-14 review fixes: the test-selected training recipe is disclosed (Section 4.3, Section S5), the decoding gain is
  stated as recipe-dependent (contribution 2, Section 5.3), the modified EfficientFace backbone and the origin of the
  intermediate fusion and modality dropout are stated (Sections 2, 3.1.2, 3.1.5), the weight range and the equal-thirds
  value are corrected (Section 5.4.3, Table S10), the Fig. 2 stars are explained correctly, and participant-level
  intervals are given (Section 5.4, Section S13); local tectonic build: paper 12 pp with more free room on p. 12 than
  before (right column ends at about 556 of 686 pt, previously 614), Fig. 2 caption still 7 lines, supplement 12 pp,
  no new warnings
- table footnotes as lower-case letters below the table; captions without final punctuation or colour words
- figures: Fig1 and Fig3 84 mm, Fig2 174 mm wide, lettering at 8–9 pt in Helvetica clones (FreeSans in
  Figs. 1–2, TeX Gyre Heros in Fig. 3; fonts embedded, none of Type 3), contrast ≥ 4.5:1, readable in grey, EPS plus
  PDF, file names `Fig1`, `Fig2`, `Fig3`; all vector except the spectrogram in Figs. 1–2 (lossless, 700 dpi or more)
- Fig. 1 (redesigned 2026-11-14) shows the real data of one test clip, and Fig. 2 its inputs and outputs as thumbnails, but with a
  drawn face and a clip without speech,
  so no participant's face or voice is published (Springer's consent rule for identifiable people)
- Online Resource 1 (`ESM_1.pdf`) carries the title, journal, authors and corresponding-author block
- no overfull boxes, no undefined references; numbers checked against the result files and the code

## Files (in `AVB-Engage_JAIHC_submission.zip`)

| Upload as | File(s) |
|---|---|
| Manuscript, LaTeX source | `1_manuscript_source/paper.tex`, `references.bib`, `sn-jnl.cls`, `sn-basic.bst` |
| Figures | `1_manuscript_source/Fig1.eps`, `Fig2.eps`, `Fig3.eps` (the `.pdf` versions are used when the system compiles with pdfLaTeX) |
| Manuscript PDF, for checking the system's build | `paper.pdf` |
| Supplementary material, "Online Resource 1" | `ESM_1.pdf` (caption, as in the paper's Supplementary information: decoding rules and threshold search, training objective, metrics, alignment, cross-attention blocks, training procedure and configuration, and further ablation, cost, label-granularity, per-level, behavior-branch, dataset, modality, fusion-weight and significance results) |
| Cover letter | `cover_letter.pdf` (the optional reviewer-suggestion sentence was removed on 2026-11-14) |

## Still to do by the authors, before uploading

1. Title block in `paper.tex` and `supplementary.tex`: filled 2026-11-14. Yuvraj Gupta (corresponding author,
   ygupta@sfsu.edu) and Sanchita Ghose (sanchitaghose@sfsu.edu, as on her SFSU faculty page), both at the Department
   of Computer Science, San Francisco State University, 1600 Holloway Avenue, San Francisco, CA 94132, USA (chosen by
   Yuvraj; her faculty page lists Computer Engineering in the School of Engineering, so Professor Sanchita should
   confirm her line). ORCIDs go into the submission form.
2. Statements and Declarations in `paper.tex`: written 2026-11-14 (no funding; no competing interests; no new data,
   EngageNet obtained from its authors on request; code at the GitHub repository, whose `main` branch holds the
   paper's code since 2026-11-14 (`development` merged into `main`); CRediT contributions of Yuvraj Gupta and Sanchita Ghose; a short
   AI statement naming "generative AI tools"). Still to confirm with Professor Sanchita: the contribution split.
3. `cover_letter.tex`: signature filled 2026-11-14 (Yuvraj Gupta, corresponding author); the optional bracketed
   sentence suggesting reviewers was deleted on 2026-11-14 (reviewers, with institutional e-mails, can still be
   suggested in the submission form).
4. Professor Sanchita's approval of the generative-AI statement (it covers the code, including the code that draws Figs. 1–2).
5. Rebuilt 2026-11-14 with the real title block: paper 12 pages, supplement 11 pages, 0 overfull boxes. The address
   and the e-mail lines each wrap to two lines (10 mm taller than the placeholder). On Overleaf (pdfLaTeX, as the
   journal compiles it) Section 5 still starts at the foot of page 6 and page 12 has about 11 free lines; the local
   tectonic build breaks lines slightly differently (Section 5 on page 7, about 6 free lines), so judge the page
   budget on Overleaf, and before submitting put Overleaf's PDF in the zip as `paper.pdf`. After the 2026-11-14
   review fixes the local tectonic build is still 12 pages, with Section 5 now on page 6 and about 11 free lines on
   page 12, and the supplement is 12 pages. **The revised `paper.tex`, `supplementary.tex`, `references.bib` and
   `cover_letter.tex` are not yet on Overleaf:** upload them, recompile there, and check the page count before
   submitting.
6. Preprint: the manuscript (LaTeX source and figures) is public in the GitHub repository since 2026-11-14. Springer
   Nature's preprint policy, which covers all its journals, allows this (a preprint is not prior publication and may
   be posted on any legal platform, before or during review) but asks authors to disclose it at submission: the cover
   letter says so, and if the submission form asks about preprints, answer yes and give the repository URL (there is
   no DOI). After acceptance, never post the publisher's typeset PDF, check the licence before pushing the accepted
   version, and add the article's DOI to the repository once it is published.

## Also required in the submission form itself

- Author contributions and competing interests: the journal publishes only what is entered in the form.
- Funding information (it goes to CrossMark).
- Graphics programs: Figs. 1–2 draw.io (diagrams.net) with data panels from Python (NumPy, librosa, Pillow);
  Fig. 3 TikZ (LaTeX), written by Python from the result file.
- Article type: original research article.

## Overleaf

The Overleaf project "Springer Nature LaTeX Template" holds the same manuscript files in its `SpringerNature/` folder
(main document `SpringerNature/paper.tex`, pdfLaTeX, TeX Live 2026), and this checklist with `cover_letter.tex` in
its `submission/` folder. Built there on 2026-11-13: paper 12 pages, supplement 11 pages, 0 errors; the redesigned Fig. 1 (with Fig. 2 and
`paper.tex`) was uploaded on 2026-11-14 and builds at 12 pages, 0 errors; so does the filled title block (`paper.tex`,
`supplementary.tex`, `cover_letter.tex` and this checklist, uploaded the same day). The review fixes of 2026-11-14
(`docs/plan.md` §22) were made after these uploads and are not on Overleaf yet. The three hyperref
warnings ("Difference ... between bookmark levels") come from the template's `\bmhead`, a level-5 heading that follows
a level-1 section, as the template manual prescribes for the back matter; they only flatten the PDF bookmark tree,
not the printed pages.
Overleaf compiles the file open in the editor when that file declares its own LaTeX class (`supplementary.tex`,
`cover_letter.tex`), and the main document `paper.tex` otherwise (checked 2026-11-14): open `supplementary.tex` or
`cover_letter.tex` and recompile to get `ESM_1.pdf` or `cover_letter.pdf`, then reopen `paper.tex`. For the same
reason this checklist never quotes the class command: when it did, Recompile with the checklist open failed with
"README.tex" not found.

## Rebuilding

From `AVB-Engage_JAIHC_final/`: pdfLaTeX + BibTeX on `paper.tex` and `supplementary.tex` (Overleaf does this). Figures:
`figures/src/make_*_drawio.py` (avtca environment; `clip_panels.py` reads the example clip and the result files) →
`figures/src/export_drawio.js`
with Helvetica mapped to FreeSans by a private fontconfig file (`docs/plan.md` §21.15) → `FigN.pdf`;
`figures/src/make_confusion_tikz.py` → `figures/fig_confusion.tex`, then `figures/Fig3_source.tex` (XeTeX) → `Fig3.pdf`; then `gs -q -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write
-sOutputFile=FigN.eps FigN.pdf` (its default keeps the Fig. 1 spectrogram lossless; checked by a round trip).
