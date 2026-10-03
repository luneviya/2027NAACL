# Paper workspace

This directory keeps the current ACL-format working draft and the recovered
source of the earlier RefCode manuscript.

## Current draft

Use `acl_template_bilingual/main.tex` as the paper entry point. The revised
experiment section is contained directly in this file rather than split across
multiple `experiments_*.tex` files. English paragraphs are followed by blue
Chinese translations for drafting; set `\showtranslationfalse` before a
submission build.

The draft depends on the adjacent `acl.sty`, `acl_natbib.bst`, and `custom.bib`
files.

## Archived source

`refcode_original/` is the recovered pre-ACL source supplied through Overleaf.
It is retained for comparison and migration only; its older reported values
must not be treated as the final NAACL results.

Downloaded source archives and generated ZIP bundles are intentionally excluded
from Git because their unpacked sources are already present here.
