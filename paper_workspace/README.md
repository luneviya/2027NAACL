# Paper workspace

This directory keeps the current ACL-format working draft and the recovered
source of the earlier RefCode manuscript.

## Current draft

Use `acl_template_bilingual/main.tex` as the copy-ready Overleaf entry point.
It retains the required ACL submission structure and packages while removing
the instructional text, example tables, sample acknowledgments, and example
appendix from the official template. The revised experiment section is written
directly in `main.tex`; no paper content is split across additional TeX files.

Each English paragraph is followed by a Chinese translation written as a LaTeX
comment beginning with `% 中文：`. The complete file can therefore be pasted
into Overleaf and compiled without loading a Chinese typesetting package.

The draft depends on the adjacent `acl.sty`, `acl_natbib.bst`, and
`custom.bib` files.

## Archived source

`refcode_original/` is the recovered pre-ACL source supplied through Overleaf.
It is retained for comparison and migration only; its older reported values
must not be treated as the final NAACL results.

Downloaded source archives and generated ZIP bundles are intentionally excluded
from Git because their unpacked sources are already present here.
