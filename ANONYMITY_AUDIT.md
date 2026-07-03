# Anonymity Audit

Inspection date: 2026-07-04

Scope: tracked source files, reviewer-facing documentation, scripts, configs, local git metadata, ignored generated files, and local workspace artifacts.

## Resolved Summary

The anonymization cleanup has been applied.

Resolved items:

- Replaced the local git remote with an anonymous placeholder remote.
- Replaced local git email metadata with an anonymous placeholder email.
- Removed the local dataset symlink that pointed to a private server path.
- Removed generated `saved_models/` outputs from the working tree.
- Removed Python cache directories and bytecode files from the working tree.
- Removed personal/nonessential GitHub URLs from code help text.
- Expanded `.gitignore` for generated outputs, caches, temporary files, local environments, datasets, logs, checkpoints, and wandb files.
- Kept README commands concise and aligned with the current three scripts.

Unresolved or intentional remaining items:

- `refcode/utils/parser/my-languages.so` remains as an ignored local build artifact so the current local environment can still import parser-dependent code without rebuilding. It is not tracked by git and should not be included in anonymous archives.
- `refcode/utils/parser/build.sh` contains public third-party GitHub URLs for tree-sitter grammars. These do not identify the authors but should be manually reviewed if the venue prohibits any GitHub links.
- The repository uses `DeepSoftwareAnalytics/CoCoSoDa` as a public pretrained model identifier. This is a reproducibility dependency, not an author identity, but it may reveal project lineage.

## Current Risk Assessment

Safe to submit as anonymous artifact: **yes, if submitted from tracked files only**

Do not submit the raw working directory or `.git/` folder. Use a clean anonymous repository or `git archive` from tracked files.

## Remaining Issues

### 1. Ignored parser shared library remains locally

- File path: `refcode/utils/parser/my-languages.so`
- Line number: n/a
- Risk level: **low**
- Why it may violate double-anonymous review: it is a generated binary artifact and may contain build metadata. It is not identity-revealing in normal source review, but binaries are harder to inspect.
- Recommended fix: do not include this file in the submitted artifact. If reviewers need it, ask them to rebuild it with `refcode/utils/parser/build.sh`.

### 2. Public tree-sitter GitHub URLs

- File path:
  - `refcode/utils/parser/build.sh:1`
  - `refcode/utils/parser/build.sh:2`
  - `refcode/utils/parser/build.sh:3`
  - `refcode/utils/parser/build.sh:4`
  - `refcode/utils/parser/build.sh:5`
  - `refcode/utils/parser/build.sh:6`
  - `refcode/utils/parser/build.sh:7`
- Risk level: **low**
- Why it may violate double-anonymous review: the URLs point to public third-party grammar repositories, not the authors. They are retained for reproducibility.
- Recommended fix: no required anonymity fix. Optional: replace with package-manager instructions if the venue forbids GitHub URLs.

### 3. Public pretrained model identifier

- File path:
  - `README.md`
  - `REPRODUCIBILITY.md`
  - `configs/*.yaml`
  - `scripts/*.sh`
  - `refcode/**/run.py`
  - `refcode/refcode_refinement/rerank.py`
- Risk level: **low**
- Why it may violate double-anonymous review: `DeepSoftwareAnalytics/CoCoSoDa` is a public pretrained model dependency. It does not identify this submission's authors, but it may reveal experimental lineage.
- Recommended fix: keep for reproducibility, or replace README examples with `BASE_MODEL=/path/to/model` if the venue requests stronger anonymization.

## Final Search Results

The following checks were run after cleanup:

```bash
find . -maxdepth 3 -type l -printf '%p -> %l\n'
```

Result: no symlinks found.

```bash
find . -maxdepth 4 \( -name '__pycache__' -o -name 'saved_models' -o -name 'dataset' -o -name 'wandb' \) -printf '%p\n'
```

Result: no matching generated directories found.

```bash
find . -type f \( -name '*.pyc' -o -name '*.bin' -o -name '*.pt' -o -name '*.pkl' -o -name '*.log' \) -printf '%p\n'
```

Result: no matching generated files found.

Identity, path, and credential search:

Result: no author names, real usernames, private local path prefixes, private IPs, credentials, or email addresses were found in tracked source files after excluding `.git/` and the ignored parser binary.

GitHub URL search:

```bash
rg -n "github\.com"
```

Remaining hits are public third-party dependency URLs in `refcode/utils/parser/build.sh`.

## Final Checklist

Safe to submit as anonymous artifact: **yes, with tracked files only**

### Must-Fix Issues Before Submission

- Do not upload `.git/`.
- Do not upload ignored generated files or local build artifacts.
- Ensure the public artifact is hosted under an anonymous repository URL.

### Optional Cleanup Suggestions

- Rebuild parser artifacts on reviewer machines instead of shipping `my-languages.so`.
- If the venue is extremely strict about project lineage, replace the visible default model owner in docs with a placeholder and document `BASE_MODEL`.

### Files That Should Remain Excluded

- `.git/`
- `dataset/`
- `saved_models/`
- `__pycache__/`
- `refcode/**/__pycache__/`
- `refcode/utils/parser/my-languages.so`
- `*.bin`
- `*.pt`
- `*.pkl`
- `*.npy`
- `*.npz`
- `*.log`
- `*.pyc`
- `*.so`
- raw dataset `*.jsonl`
- wandb outputs
- local environment files

### Anonymous GitHub Replacement Rules

Configure replacement rules for:

- real GitHub username -> anonymous placeholder
- real repository URL -> anonymous repository URL
- private workspace path -> `/path/to/workspace`
- private dataset path -> `/path/to/data`
- private output path -> `/path/to/output`

Do not replace public third-party dependency names unless the venue explicitly requires it.
