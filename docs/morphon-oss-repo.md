# Morphon / Morphon-OSS Two-Repo Workflow

> **⚠️ This file is private-only.** It documents internal workflow and is
> excluded from the public Morphon-OSS sync. Do not link to it from any
> user-facing doc that ships to OSS.

## TL;DR

There are two GitHub repositories for this project:

| Repo | URL | Visibility | Purpose |
|------|-----|------------|---------|
| **Morphon** | https://github.com/SimplyLiz/Morphon | **Private** | Daily development, research notes, plans, internal specs |
| **Morphon-OSS** | https://github.com/SimplyLiz/Morphon-OSS | **Public** | Paper code, citable preprint, user-facing docs only |

**On your laptop there is only one working folder**: `/Users/lisa/Work/Projects/Morphon/`. That folder is your private Morphon clone. Morphon-OSS is a filtered snapshot we generate and push whenever we want to release a new public version.

---

## Why two repos?

The open-core pattern. We want:

1. **Freedom to develop in private.** Research notes, half-finished specs, internal findings, funding applications, personal ideas, speculative directions — all of this stays in the private repo.
2. **A clean, citable public repo** containing only the published paper, the code that reproduces its numbers, and user-facing documentation. Reviewers, researchers, and anyone citing the DOI land here.
3. **One working folder.** We don't want to context-switch between two clones on the laptop. All day-to-day work happens in the private repo; the public repo is updated only at release boundaries.

## Folder layout (after the dev/ restructure)

### Private Morphon

```
Morphon/
├── dev/                              ← EXCLUDED from Morphon-OSS
│   ├── Lisa/                         personal research notes
│   ├── internal/                     implementation notes
│   ├── plans/                        roadmap, research plans
│   ├── research/                     research notes
│   ├── specs/                        feature specs (temporal, limbic, etc.)
│   ├── concepts/                     concept explorations
│   ├── features/                     feature docs (dev-facing)
│   ├── official/                     FFG funding application etc.
│   ├── Alt/                          old iterations
│   ├── obsoloete/                    deprecated
│   ├── paper-sources/                (was docs/paper/sources/) findings
│   ├── paper-ZENODO_UPLOAD.md         publishing workflow
│   ├── morphon-oss-repo.md           THIS FILE (but see note below)
│   ├── CLAUDE.md                     Claude Code project instructions
│   ├── MORPHON-V*.md/pdf             concept docs (V2-V6)
│   ├── MORPHON-ANCS-*                ANCS integration concept
│   ├── MORPHON-product-concept.*
│   ├── endoquilibrium-spec.*
│   ├── knowledge-hypergraph-spec.md
│   ├── pulse-kernel-lite-spec.*
│   ├── findings.md
│   ├── pr.docx, Claude.pdf
│
├── docs/                             ← SHIPPED with Morphon-OSS
│   ├── BENCHMARKS.md                 user benchmark guide
│   ├── WHAT-IT-CAN-DO.md             feature overview
│   ├── morphogenic-intelligence-concept.md + .pdf
│   ├── user/
│   │   └── settings.mdx.md           config reference
│   ├── paper/paper/                  published LaTeX + PDF
│   └── benchmark_results/            reproducibility JSONs
│
├── src/, examples/, benches/, tests/, web/, demo/, scripts/
├── Cargo.toml, LICENSE, README.md, CITATION.cff, CHANGELOG.md, CONTRIBUTING.md
└── .git/                             (tracks private Morphon only)
```

> **Note about this file:** `docs/morphon-oss-repo.md` is at the top level
> of `docs/` for convenience, but it is **explicitly excluded** from the
> Morphon-OSS sync (see exclusion list below). During the upcoming `dev/`
> restructure this file may move to `dev/morphon-oss-repo.md` — if so,
> update the sync script accordingly.

### Public Morphon-OSS

Same tree minus `dev/` and minus the explicit file exclusions listed below.

---

## Sync workflow

**One working folder, two push targets.** We never add Morphon-OSS as a git
remote in the main working clone — that would invite accidental pushes of
private content. Instead, we create a fresh temporary clone, strip the
private files, and force-push that to Morphon-OSS.

### The script

A reusable script lives at `scripts/sync-oss.sh` (to be created). Its logic:

```bash
#!/usr/bin/env bash
set -euo pipefail

PRIVATE_REPO=/Users/lisa/Work/Projects/Morphon
OSS_URL=https://github.com/SimplyLiz/Morphon-OSS.git

# 1. Create temp clone from the private repo's current main
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
git clone --branch main --single-branch "$PRIVATE_REPO" "$TMP/oss-sync"
cd "$TMP/oss-sync"

# 2. Remove private-only paths
git rm -rf dev/ 2>/dev/null || true
git rm -f docs/morphon-oss-repo.md 2>/dev/null || true

# 3. Create a single consolidating commit (optional — you can keep full
#    history if you prefer. This squashes into one "release" commit per sync.)
git commit -m "chore: sync from private Morphon — $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 4. Point to OSS remote and force-push
git remote remove origin
git remote add origin "$OSS_URL"
git push --force origin main

# 5. Push the v3.0.0 tag (and any other release tags you want public)
git push --force origin v3.0.0 || true
```

### Running the sync

From the private working folder:

```bash
./scripts/sync-oss.sh
```

That's it. The script:
1. Clones your current private `main` into a temp folder (takes a few seconds)
2. Removes `dev/` and any specifically-excluded files
3. Commits the cleaned state as a single "release" commit
4. Force-pushes the temp clone to Morphon-OSS `main`
5. Pushes release tags
6. Deletes the temp folder

Your main working folder is untouched. Private Morphon's git history is
pristine. Public Morphon-OSS gets a clean snapshot.

### When to run the sync

**Not after every private commit.** Morphon-OSS is a release target, not a
mirror. Run the sync when:

1. Cutting a new release (new tag, new paper version, new benchmark numbers)
2. Fixing a publicly-visible bug that we want people to be able to reproduce
3. Updating the paper PDF in response to reviewer feedback
4. Adding user-facing documentation

For normal day-to-day development (code changes, experiments, notes,
WIP commits), just push to private. Morphon-OSS doesn't need to see those.

---

## Exclusion list

These paths exist in private Morphon but are stripped before pushing to
Morphon-OSS:

| Path | Reason |
|------|--------|
| `dev/` (entire folder) | Private development: plans, research, internal notes, specs drafts, funding applications, personal notes, paper source materials, Claude.md instructions, concept docs |
| `docs/morphon-oss-repo.md` | This file — documents the internal workflow, not useful to public audience |

If you add new private directories or files, update both:
1. The exclusion loop in `scripts/sync-oss.sh`
2. The table above in this file

---

## Manual sync (without the script)

If the script isn't available or you need to do a one-off with different
exclusions:

```bash
# From anywhere, not necessarily inside the private clone
TMP=$(mktemp -d)
git clone --branch main --single-branch \
    https://github.com/SimplyLiz/Morphon.git "$TMP/oss-sync"
cd "$TMP/oss-sync"

# Remove private paths
git rm -rf dev/
git rm -f docs/morphon-oss-repo.md

# Single release commit
git commit -m "chore: sync from private Morphon — $(date -u)"

# Point to OSS and force-push
git remote remove origin
git remote add origin https://github.com/SimplyLiz/Morphon-OSS.git
git push --force origin main
git push --force origin v3.0.0

# Clean up
cd - && rm -rf "$TMP"
```

---

## Git history semantics

**Private Morphon** keeps full history. Every commit, every branch, every
tag. This is the source of truth for how the project evolved.

**Morphon-OSS** keeps a flattened, periodic history. Each sync is a single
"release" commit with a timestamp. We do **not** try to preserve private
commit-by-commit history on the public side for three reasons:

1. Commit messages may reference private paths (`dev/plans/...`)
2. Intermediate commits may contain private files that we later removed
3. Readers of the public repo care about releases, not daily development

If you ever need to let an outside contributor send a PR against
Morphon-OSS, the workflow is:

1. They fork Morphon-OSS and submit a PR there
2. We review it on Morphon-OSS (not merged yet)
3. We manually replay the change onto private Morphon's `main`
4. Next sync picks it up
5. The PR on Morphon-OSS is closed with a comment linking to the private
   commit SHA that includes the change

This is the same pattern Google, Microsoft, and most big open-core projects
use. It keeps the private-public boundary sharp without sacrificing
contributor goodwill.

---

## Troubleshooting

### "I accidentally committed a private file to Morphon-OSS"

1. Remove the file locally on Morphon-OSS
2. Force-push (assuming you're the only maintainer)
3. If the private file contained secrets, consider rotating them — GitHub's
   commit history may still contain the leaked content via the REST API
   even after force-push

### "My sync script failed"

Check the error and fix. The temp folder is auto-cleaned on exit thanks to
`trap rm -rf "$TMP" EXIT`, so you can safely rerun.

### "The OSS repo has extra commits I didn't make"

Only happens if someone else pushed to it directly. Don't. All public
commits should go through the sync script.

### "I want to see what's currently in Morphon-OSS without cloning"

```bash
gh api repos/SimplyLiz/Morphon-OSS/contents | \
    python3 -c "import json,sys; print('\n'.join(sorted(e['name'] for e in json.load(sys.stdin))))"
```

Or just open https://github.com/SimplyLiz/Morphon-OSS in a browser.

---

## DOI and Zenodo

- **Concept DOI** (always latest version): `10.5281/zenodo.19467243`
- Upload workflow is documented in `dev/paper-ZENODO_UPLOAD.md` (private)
- When you release a new paper version, upload to Zenodo as "New version"
  of the existing record — the concept DOI stays the same, a new version
  DOI is minted

---

## Authors

- Lisa Welsch · lisa@tastehub.io
- Martyna Kwiecień · martyna@tastehub.io

If someone else takes over maintenance, update this file with the new
sync-script owner and any changes to the exclusion list.
