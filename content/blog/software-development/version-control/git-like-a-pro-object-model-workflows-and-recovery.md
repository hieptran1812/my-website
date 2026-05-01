---
title: "Git Like a Pro: A Deep Dive Into the Object Model, Daily Workflows, and Recovering From Disaster"
date: "2026-04-30"
description: "A senior engineer's playbook for Git — the object model, the three trees, branching strategies, rebase as a power tool, signed commits, performance at scale, and a long catalog of incidents Git tried to ruin our day."
tags:
  [
    "git",
    "version-control",
    "developer-tools",
    "rebase",
    "workflow",
    "devops",
    "monorepo",
    "troubleshooting",
    "shell",
    "code-review",
  ]
category: "software-development"
subcategory: "Version Control"
author: "Hiep Tran"
featured: true
readTime: 50
aiGenerated: true
---

Most engineers plateau at Git the same way: they learn ten porcelain commands by muscle memory, treat the rest as magic, and pay for that opacity every time something goes sideways at 2 AM. The plateau is not about commands. It is about a mental model. Once you understand that Git is a tiny, content-addressed object database with a few pointers on top, every command becomes obvious and every disaster becomes recoverable. This article is the playbook I wish a senior engineer had walked me through in my first year — the object model in detail, the three trees that every command moves data between, branching strategies that scale past a hundred engineers, rebase as a precision tool, the recovery techniques that have saved me at least a dozen times in production, and a long catalog of incidents where Git tried to ruin our day and what the fix actually was.

![Git's object model: refs, commits, trees, blobs](/imgs/blogs/git-like-a-pro-1.png)

The diagram above is the mental model. Refs (HEAD, branches, tags) are 40-character files that point at commits. Commits point at exactly one tree (the snapshot of the repo at that moment) plus zero or more parent commits. Trees point at blobs (the actual file bytes) and at sub-trees (subdirectories). Every object is keyed by the SHA-1 of its content, so identical bytes are stored exactly once. The rest of this article unpacks why that one design choice — content addressing — makes Git fast, durable, and forgiving in ways that are not obvious until you have lost work and gotten it back.

## 1. Why Most Engineers Plateau at Git

The plateau looks the same on every team I have worked with. People learn `add`, `commit`, `pull`, `push`, `checkout`, `merge`, and a vague form of `rebase`. They google specific recipes when something breaks. They develop superstitions: "always pull before you push," "never rebase," "force push is dangerous," "stash before switching branches." Some of these are right, some are right in only narrow contexts, and a few are wrong but persist because they are usually harmless.

The cost of the plateau is not measured in commands typed. It is measured in three places. First, **incident response**: when a teammate force-pushes the release branch at 11:47 PM the night before a release, you either know that the reflog still holds the old SHA for 90 days, or you don't. Second, **review fatigue**: a team that does not understand merge versus rebase ships pull requests with twenty-commit histories full of "fix typo," "address review," "wip," which makes blame, bisect, and revert all dramatically harder six months later. Third, **review velocity**: engineers who don't trust `git add -p` end up either committing unrelated changes together or context-switching to clean up before each commit. The compound interest on those three taxes over a multi-year project is staggering.

The way past the plateau is not to memorize more commands. It is to learn the **plumbing** — the small set of low-level commands and on-disk objects that all the porcelain (`add`, `commit`, `merge`, `rebase`) is built on top of. Once you have the plumbing in your head, the porcelain is just a thin layer of UX over operations you can already reason about. Three days of plumbing study will buy you a decade of better judgment.

## 2. The Object Model in 1500 Words

Git stores four kinds of objects, and only four. Every object is named by the SHA-1 of `<type> <length>\0<content>`. They live in `.git/objects/`, indexed by the first two hex chars of the SHA. Every porcelain command reduces, eventually, to creating, reading, or referencing these objects.

**Blobs** are file contents. Just bytes. No filename, no permissions, no path — those live in trees. Two files with identical content have the same blob SHA, and Git stores them once. This is why duplicating a 100 MB binary across three directories adds zero bytes to your `.git/`.

**Trees** are directories. A tree is a sorted list of `(mode, type, sha, name)` entries — `100644 blob 5a1b... README.md`, `040000 tree e811... src/`. A tree references blobs (files) and other trees (subdirectories). Trees are also content-addressed, so two unrelated commits that happen to have identical directory contents share the same tree object.

**Commits** are snapshots with metadata. Each commit points at exactly one tree (the root tree), zero or more parent commits, and stores an author, committer, timestamp, and message. The chain of parents is what we call history. A merge commit has two parents; an octopus merge has more. The root commit has none.

**Tags** (the annotated kind) are signed pointers at a commit, with their own message and author. Lightweight tags are just refs that don't go through this object — they are a single 40-char file under `.git/refs/tags/`.

Inspect the actual bytes:

```bash
$ echo "hello git" > greet.txt
$ git init -q && git add greet.txt && git commit -q -m "init"

$ git rev-parse HEAD
9a4b2cf8e... # the commit SHA
$ git cat-file -t 9a4b2cf8       # type
commit
$ git cat-file -p 9a4b2cf8       # pretty-printed contents
tree 1f7a90c5c4...
author Hiep <hiep@x> 1714512000 +0700
committer Hiep <hiep@x> 1714512000 +0700

init

$ git cat-file -p 1f7a90c5       # the root tree
100644 blob ce0136250786...    greet.txt

$ git cat-file -p ce013625        # the blob
hello git
```

That is the entire model. Every other concept — branches, tags, merges, rebases, reverts, cherry-picks — is a transformation defined on top of these four object kinds and a few pointers (refs). When the porcelain confuses you, drop into the plumbing for thirty seconds and re-orient.

A subtle point that pays off forever: **commits don't store diffs**. They store the full tree. The "diff" you see in `git show` is a derived view — Git reconstructs it on demand by walking the tree of the commit and comparing against the tree of its parent. This has consequences. It is why `git revert` of a commit deep in history is cheap (no diff replay, just compute the inverse against the current tree). It is why `git cherry-pick` of a commit onto an unrelated branch is well-defined (the change is the diff against the chosen base, not against the original parent). And it is why renaming a 100 MB file is not free in storage — Git stores the new path with the same blob SHA, but the tree object grows by one entry and there is no diff record.

The on-disk layout has two formats. **Loose objects** (`.git/objects/9a/4b2cf8...`) are zlib-compressed individual files; new commits start here. **Packfiles** (`.git/objects/pack/pack-*.pack` and `.idx`) compact many objects together with **delta compression**: similar blobs are stored as a base plus a delta against another similar blob. `git gc` migrates loose objects into packs, often shrinking a multi-GB repo by 5–10×. Pack delta chains are why `git clone` of a large repo over a slow network can stall on a single CPU after the bytes arrive: the client decompresses and resolves deltas before checkout.

Content addressing has one property that anchors most of Git's design: **integrity is free**. If a single bit of a blob flips on disk, the SHA no longer matches and `git fsck` will scream. SHA-1 is no longer cryptographically secure against adversaries (SHAttered 2017), but as an integrity check against bit rot it is fine, and Git is migrating to SHA-256 for new repos.

## 3. Refs, HEAD, and the Reflog

A **ref** is a 40-character file. Branches live under `.git/refs/heads/`, tags under `.git/refs/tags/`, and remote-tracking branches under `.git/refs/remotes/<remote>/`. Many tools use **packed-refs** (a single `.git/packed-refs` file) for performance once you have thousands of refs.

`HEAD` is the special ref that says "where am I now." Usually it is symbolic — `ref: refs/heads/main` — meaning "follow the `main` branch." When you `git commit` with a symbolic HEAD, Git updates both HEAD's target ref (`main`) and HEAD itself. When HEAD points directly at a commit SHA instead of through a branch ref, you are in **detached HEAD state**: commits you make have no branch to follow them, and a subsequent `git checkout` of any other ref will leave them "unreachable" — still in the object database, but with no ref pointing at them.

A few special pseudo-refs that show up in error messages and commit recovery:

- `ORIG_HEAD` — the previous HEAD before a destructive operation (rebase, reset, merge). `git reset --hard ORIG_HEAD` reverses an unintended rebase or merge.
- `FETCH_HEAD` — what the last `git fetch` brought down. `git diff FETCH_HEAD` is what most people actually want when they ask "what's new on the remote."
- `MERGE_HEAD` — set during a merge in progress, points at the other parent. Cleared after `git commit` finishes the merge.

The single most powerful Git feature for disaster recovery is the **reflog**. Every time HEAD or a branch ref moves, Git appends an entry to `.git/logs/HEAD` (and per-branch logs). The reflog records "you were at SHA X, now you are at SHA Y, here's why" with timestamps. By default, reachable reflog entries are kept for 90 days; unreachable ones for 30. That window is the only thing standing between your team and a permanent loss.

```bash
$ git reflog
9a4b2cf HEAD@{0}: commit: add cache
3b7e2a1 HEAD@{1}: rebase (start): checkout main
1c44d2f HEAD@{2}: commit: lost work I thought I deleted
...

# Recover the "lost" commit:
$ git checkout -b recovery 1c44d2f
```

Repeat that recovery once, in front of a panicked teammate, and you will be remembered as the team's Git oracle for years.

## 4. The Three Trees: Working Dir, Index, HEAD

Almost every confusing Git error reduces to a simple question: which of the three trees did I just move?

![The three trees: working dir, index, HEAD](/imgs/blogs/git-like-a-pro-2.png)

The **working directory** is real files on disk — what your editor sees. The **index** (a single binary file at `.git/index`, also called the "staging area") is Git's proposed next commit: a map from path to blob SHA plus mode. The **HEAD** tree is the snapshot of the commit HEAD currently points at. Every command that mutates state moves data between exactly two of these.

| Command                       | What moves                                                                                                       |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `git add <path>`              | working dir → index                                                                                              |
| `git add -p`                  | working dir → index, hunk by hunk                                                                                |
| `git commit`                  | index → new commit, advance HEAD                                                                                 |
| `git restore <path>`          | HEAD's blob → working dir (discard local edit)                                                                   |
| `git restore --staged <path>` | HEAD's blob → index (un-stage)                                                                                   |
| `git reset --soft <ref>`      | HEAD ref → \<ref\>; index, working dir untouched                                                                 |
| `git reset --mixed <ref>`     | HEAD ref → \<ref\>; index reset to \<ref\>'s tree; working dir untouched (default)                               |
| `git reset --hard <ref>`      | HEAD ref → \<ref\>; index reset; working dir overwritten                                                         |
| `git checkout <ref> -- <path>`| copy path from \<ref\> into both index and working dir (legacy combined op)                                      |
| `git switch <branch>`         | move HEAD to \<branch\>; working dir + index follow if clean; bails if dirty                                     |
| `git stash push`              | working dir + index → temporary commit on the stash ref; reset working dir/index to HEAD                         |

The single most useful exercise to graduate from "Git plateau" is to write that table by hand from memory. Each command becomes obvious once you can recite which trees it touches. The misery of `reset --hard wrong-sha` becomes tractable when you remember it only moved HEAD and overwrote index/working dir — your old commits, including the work you thought you lost, are still in the object database and the reflog still points at them.

A worked example: you have edited `parser.py`, staged half of it, and want to undo *only* the staged half without losing the unstaged edits.

```bash
# Working dir = edits A + B; index = edits A; HEAD = original.
$ git restore --staged parser.py    # index <- HEAD;  working dir untouched
# Working dir still has edits A + B; index now equals HEAD.
$ git add -p parser.py              # re-stage just the parts you actually want
```

Note what we did *not* do: we did not run `git reset --hard`, which would have nuked the working-dir edits along with the staged ones. The three-trees model tells you which command targets exactly the tree you mean.

## 5. Branching Strategies That Scale

![Branching strategies: trunk-based, GitFlow, release-train](/imgs/blogs/git-like-a-pro-3.png)

Most "should we use GitFlow?" arguments are rehashes of a smaller question: how often do you ship, how many parallel versions do you support, and how strong is your CI? The strategy follows from those three answers, not the other way around.

**Trunk-based development** keeps a single long-lived branch (`main`). Feature branches are short-lived (under two days), merged behind feature flags, and `main` is always green and deployable. A release is a tag on `main`. This is what almost every CI/CD-heavy shop ends up at — Google, Facebook (at one point), and most modern startups under a few hundred engineers. The discipline that makes it work is feature flags: incomplete features land on `main` dark and ship dark until they are flipped on.

**GitFlow** maintains long-lived `develop` and `main` branches, plus `feature/*`, `release/*`, and `hotfix/*` branches with prescribed merge rules. It was designed for shipped products where multiple released versions are supported in parallel — a 1.4 mobile app on the App Store while 1.5 is in QA and 1.6 is being built. GitFlow's ceremony pays off when "deploy" means submitting to a store and waiting for review, and it costs you ceremony in any context where deploy is push-button.

**Release-train** branches a release on a fixed cadence (every 2–6 weeks) regardless of feature readiness. Whatever is on `main` at branch time goes; everything else waits for the next train. Only critical fixes get cherry-picked into the release branch after cut. Chrome and Firefox famously use this. It is the only strategy that scales past about 200 engineers without engineering managers spending their week on merge-conflict triage.

| Dimension          | Trunk-based       | GitFlow                       | Release-train                       |
| ------------------ | ----------------- | ----------------------------- | ----------------------------------- |
| Team size          | <20 to medium     | small to medium               | 200+                                |
| Deploy cadence     | hourly to daily   | weekly to monthly             | every 2–6 weeks                     |
| Main shape         | linear            | branchy (develop + release)   | linear `main` + release branches    |
| Feature flags      | required          | optional                      | required for parallel versions      |
| Merge type         | squash            | merge commit                  | rebase + cherry-pick to release     |
| Rollback           | revert one commit | hotfix branch + double-merge  | cut a patch release                 |
| CI strictness      | every PR          | merge queue                   | merge queue + release-branch CI     |
| Failure mode       | flaky CI gates    | release ceremony stalls       | engineers drift from `main`         |

The mistake teams make is picking a strategy on cargo-cult grounds and not adjusting when the real constraints change. A 5-engineer startup that adopted GitFlow because the founder used it at a previous job will spend half its weekly engineering meeting on merge ceremony. A 300-engineer org doing trunk-based with no merge queue will see `main` red for hours every day. **Match the strategy to the constraint that hurts most**: if it's release coordination, GitFlow. If it's stability under high churn, release-train. If it's flow, trunk-based.

## 6. Merge vs Rebase vs Squash

![Merge vs rebase: the same code, two histories](/imgs/blogs/git-like-a-pro-4.png)

The single most contentious style choice in Git. The answer is contextual, but the contexts are knowable.

**Merge** preserves the original branch graph. The result of merging `feature` into `main` is a new merge commit on `main` whose two parents are the previous `main` tip and the `feature` tip. Reviewers can see exactly which work happened on which branch and when. Tooling that generates release notes from merge commit subjects (a common pattern in Java and JavaScript ecosystems) needs merge commits to function. The cost: `git log` on `main` is a graph, not a list, and `git bisect` will sometimes land on a non-functional intermediate commit.

**Rebase** rewrites the feature branch on top of the current `main`. The original commits get new SHAs because they have new parents. The branch is then merged with a fast-forward, leaving `main` linear. Reviewers see one commit per logical change. `bisect` halves cleanly. The cost: history is a fiction — those commits were never actually authored on top of those parents — and rebasing a branch that has been pushed and shared with reviewers requires `--force-with-lease` and a heads-up on Slack.

**Squash-merge** collapses the entire feature branch into one commit on `main`, regardless of how many commits the feature had. GitHub's "Squash and merge" button does this. Pros: every commit on `main` is a PR-sized atom; bisect is dead simple; revert is one line. Cons: in-feature history is lost forever (unless the PR description preserves it), and authors who pre-decompose a feature into 5 carefully-staged commits feel cheated.

The decision rules I follow:

- **Trunk-based + small/medium PRs?** Squash-merge is almost always right.
- **Trunk-based + a large refactor that comes in 6 carefully-decomposed commits?** Rebase + fast-forward, preserving the 6 commits.
- **GitFlow / release-train?** Merge commits, because the branch graph is part of the audit trail and release-note generation depends on it.
- **A long-lived shared branch I am rebasing for any reason?** Always `--force-with-lease`, never `--force`. The lease check ("the remote tip should still be the SHA I last fetched") prevents you from clobbering a teammate's push.

The single most damaging Git anti-pattern I have seen in production is force-pushing without lease on shared branches. Every team needs branch protection on `main` and `release/*` that disables force push outright. GitHub branch protection, GitLab protected branches, Bitbucket merge policies — pick one, turn it on, and never disable it.

## 7. Interactive Rebase as a Power Tool

![Interactive rebase: editing history with intent](/imgs/blogs/git-like-a-pro-5.png)

`git rebase -i HEAD~N` opens an editor with one line per commit. Each commit can be transformed by replacing `pick` with another verb:

```
pick    a1   Add A handler
pick    a2   Tests for A handler
pick    a3   Fix typo in A
pick    a4   Address review: rename helper
pick    a5   wip
```

The verbs:

- **`pick`** — keep the commit as is.
- **`reword`** — keep changes, change the message.
- **`edit`** — pause after applying so you can amend or split.
- **`squash`** — fold into the previous commit, combine messages.
- **`fixup`** — fold into the previous, *discard* this message.
- **`drop`** — throw the commit away.
- **`exec <cmd>`** — run a shell command between commits (great for "run tests after each commit").
- **`reorder`** — just rearrange lines.

The combination that genuinely changes how you work: **`fixup` commits + `--autosquash`**. When you find a typo in a previous commit, instead of opening an editor, do:

```bash
$ git commit --fixup=a1            # creates a commit titled "fixup! Add A handler"
$ ... more work ...
$ git commit --fixup=a3
$ git rebase -i --autosquash HEAD~10
```

Git automatically reorders the fixup commits next to their targets and pre-marks them as `fixup`. You hit save and the history is clean. Wire `git config --global rebase.autosquash true` so `--autosquash` is implied.

The other transformative pattern: **splitting a commit with `edit`**. Rebase, mark the commit `edit`, then:

```bash
$ git reset HEAD^                 # undo the commit but keep the changes
$ git add -p file_a.py            # stage just the part for commit 1
$ git commit -m "Refactor file_a"
$ git add file_b.py               # remaining changes for commit 2
$ git commit -m "Wire up file_b"
$ git rebase --continue
```

This is how you turn one giant "fix everything" commit into three reviewable atoms after the fact. Code reviewers will love you.

## 8. Cherry-pick, Range-diff, and Patch Mode

`git cherry-pick <sha>` re-applies the diff of one commit onto the current branch, creating a new commit with a new SHA. It is how you backport a fix from `main` to `release/24.1`:

```bash
$ git switch release/24.1
$ git cherry-pick -x 9a4b2cf      # -x adds "(cherry picked from commit ...)" to the message
```

The `-x` flag is essential for any cross-branch port — six months later, `git log --grep "cherry picked from"` will show every backport. Without `-x`, you have lost the audit trail.

`git range-diff A..B C..D` compares two ranges of commits, pairing them up and showing how each changed. The killer use case: re-reviewing your own PR after a rebase. You used to have commits at `feature@v1`, you rebased onto a newer `main` and have `feature@v2`, and you want to convince yourself that the rebase didn't accidentally drop or alter any change.

```bash
$ git range-diff feature@{1}..feature feature~5..feature
```

`git add -p` is patch mode. It walks every hunk of every modified file and asks `[y,n,q,a,d,j,J,g,/,e,?]`. You stage exactly the changes you want into the commit and leave the rest in the working dir. Combined with `git commit -p` (same prompt but creates a commit) and `git checkout -p` (discard hunks selectively), patch mode is how mature engineers keep commits atomic without thinking about it.

`git stash --keep-index` is the often-forgotten cousin: stash the working-dir changes that are *not* staged, leaving the staged changes alone. Useful for "I want to run the test suite against just the version of the code I'm about to commit."

## 9. Remotes, Fetch, Pull, Push

![Remote refs: fetch, pull, push, and the lease check](/imgs/blogs/git-like-a-pro-6.png)

A **remote** is a named URL. By convention `origin` is the URL you cloned from. You can have many: `origin`, `upstream`, `fork`, etc. The remote URLs and configured refspecs live in `.git/config`.

A **refspec** is a mapping of "what to copy" rules. The default is `+refs/heads/*:refs/remotes/origin/*` — fetch every branch from the remote, store it under `refs/remotes/origin/`. The `+` means "force update," which is fine for remote-tracking refs (they are local copies and force update only ever overwrites your local mirror, never the remote).

`git fetch` copies remote refs to your local `refs/remotes/origin/*` and downloads any new objects. It does not touch your working dir, your branches, or HEAD. **`fetch` is always safe.** Run it freely.

`git pull` is `fetch` plus a merge or rebase of the upstream into the current branch. The default is merge. If you set `pull.rebase = true`, pull becomes fetch + rebase. Either default is fine, but pick one and standardize across the team — mixed styles produce mysterious commit graphs.

`git push <remote> <branch>` updates the remote ref. The remote rejects pushes that aren't fast-forward (your branch is missing commits the remote has) unless you `--force` or `--force-with-lease`.

**Always use `--force-with-lease`, never `--force`**. The difference saves your team. `--force-with-lease` includes "I expect the remote tip to still be the SHA I last fetched" — if a teammate pushed in between, the lease check fails and your force-push aborts. Plain `--force` overwrites whatever is there, so if your colleague just pushed a critical fix, you have just deleted it from the remote.

Push protections worth turning on at the server level:

- **Block force-push on `main` and `release/*`.**
- **Require linear history on `main`** (no merge commits) if you are trunk-based.
- **Require signed commits** if your industry has any compliance requirements (more in section 11).
- **Require a passing CI run** before merge.
- **Require code review** with at least one reviewer who is not the PR author.

These are all server-side toggles in GitHub/GitLab/Bitbucket. Turn them on once and they save you forever.

## 10. Submodules, Subtrees, Sparse-Checkout, Partial Clone

When the repo gets too big or needs to compose other repos, Git offers four mechanisms with very different trade-offs.

**Submodules** record a pointer to a specific commit in another repo. The pointer is committed in the parent repo as a `.gitmodules` entry plus a special tree entry. `git submodule update --init --recursive` clones the child at the recorded SHA. Submodules are correct but operationally painful: every developer needs to remember the init step, every CI job needs `--recursive`, and merging changes to the submodule pointer is a frequent source of "why is the submodule pointing at a commit that no longer exists?" — see Case Study 5.

**Subtrees** copy the source repo's content into a subdirectory of the parent and merge upstream changes via `git subtree pull`. There is no separate clone step — the files just live in the parent repo's history. Subtrees are easier on consumers but harder on contributors (changes flow back via `git subtree push`).

**Sparse-checkout** lets you check out only part of a repo's working tree. The full history is still fetched, but only matching paths populate the working dir. Useful when one repo holds 50 services and you only work on three.

```bash
$ git sparse-checkout init --cone
$ git sparse-checkout set services/api services/worker
```

**Partial clone** (`--filter=blob:none` or `--filter=tree:0`) skips downloading blob (or tree) contents during clone, fetching them on-demand when something asks for them. Combined with sparse-checkout, this is the foundation of how giants like Microsoft work in their Windows monorepo — a 300 GB repo where a fresh checkout pulls about 5 GB.

```bash
$ git clone --filter=blob:none --sparse https://github.com/big/repo
$ cd repo && git sparse-checkout set <paths>
```

| Tool             | When to use                                                                   | Cost                                                          |
| ---------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Submodule        | Vendoring a third-party lib at a fixed SHA, audit-tracked                     | Operational toil; every developer must remember `--recursive` |
| Subtree          | Vendoring code you'll modify in place                                         | Harder to push back upstream                                  |
| Sparse-checkout  | Big monorepo, you need part of the working dir                                | History is full size                                          |
| Partial clone    | Big repo, you need full history but rarely all blobs                          | First access of each blob makes a network round trip          |
| Monorepo tooling | Truly massive; you need build graph, codeowners, virtual file system          | Bazel/Buck/Nx is a separate skill set                         |

## 11. Hooks, Signed Commits, and Supply Chain Hygiene

Git hooks are shell scripts in `.git/hooks/`. The named ones (`pre-commit`, `commit-msg`, `pre-push`, etc.) run automatically at lifecycle points. Default they are off; enable by making them executable.

Per-repo hooks don't sync with `git clone` (because `.git/` isn't versioned). The standard solution is the [`pre-commit`](https://pre-commit.com) framework, configured via a versioned `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-merge-conflict
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.5.6
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks
```

`pre-commit install` wires it up. Now every commit runs the hooks; failures abort the commit. The `gitleaks` hook in particular has paid for itself dozens of times by catching `.env` or AWS keys before they hit the repo.

**Signed commits.** GPG and SSH signing both work. SSH is dramatically easier — you reuse the same SSH key you push with:

```bash
$ git config --global gpg.format ssh
$ git config --global user.signingkey ~/.ssh/id_ed25519.pub
$ git config --global commit.gpgsign true
$ git config --global tag.gpgsign true
```

Push with signed commits, configure the server to require them on protected branches, and now the audit trail is cryptographic. Sigstore's [`gitsign`](https://github.com/sigstore/gitsign) takes this further by signing with short-lived OIDC certs tied to your corporate SSO, eliminating long-lived signing keys entirely — strongly recommended in regulated environments.

Supply-chain hygiene also means watching what your hooks themselves can do. `pre-commit` runs arbitrary code from third-party repos at the rev you pin. Pin to a SHA, not a tag, when paranoid. Audit any new hook added to a project's config — a malicious `pre-push` hook could exfiltrate your SSH keys and you'd never know.

## 12. Performance at Scale

Stock Git slows down on big repos in predictable ways. The fixes are config flags; turn them on early.

**`core.fsmonitor`** uses the OS's filesystem watcher (Watchman on Mac/Linux, the built-in fsmonitor on newer Git) to skip the expensive `stat()` of every file on `git status`. On a 100k-file repo, status drops from 8 seconds to under 200 ms.

```bash
$ git config core.fsmonitor true
$ git config core.untrackedcache true
```

**Commit-graph** is a precomputed index of the parent relationships of every commit, used to short-circuit walks in `log`, `merge-base`, and `bisect`. On a million-commit repo, the difference is order-of-magnitude.

```bash
$ git config core.commitGraph true
$ git config gc.writeCommitGraph true
$ git commit-graph write --reachable --changed-paths
```

**Multi-pack-index (midx)** lets Git keep many packfiles efficient without forcing repacks.

```bash
$ git config core.multiPackIndex true
$ git multi-pack-index write --bitmap
```

**Partial clone + sparse-checkout** (already covered) are the foundation of "I need the same repo Microsoft has but on my laptop."

**Scalar** is Microsoft's all-in-one wrapper that turns these on by default and adds prefetching. `scalar clone <url>` is a one-line replacement for `git clone` on big repos. Scalar is shipped inside Git itself in modern versions (2.38+).

| Setting                            | Effect on a 5M-commit repo                       |
| ---------------------------------- | ------------------------------------------------ |
| `core.fsmonitor=true`              | `status` 8s → 200ms                              |
| `core.commitGraph=true`            | `log` 12s → 1.2s                                 |
| Partial clone (`--filter=blob:none`)| Initial clone 45min → 4min, 80GB → 6GB           |
| Sparse-checkout (cone)             | Working dir 100k files → 8k                      |
| `core.untrackedcache=true`         | `status` 200ms → 50ms when working tree is large |

## 12.5. Conflict Resolution Beyond Stare-and-Squint

The default merge driver dumps `<<<<<<<` markers into your file and walks away. That is fine for two-line conflicts and miserable for the real ones. There are four techniques that turn conflict resolution from an art into a procedure.

**`git mergetool`** drops you into a three-way diff (yours, theirs, base) inside your configured tool. `git config merge.tool meld` (or `vimdiff`, `kdiff3`, `vscode`, …) and `git mergetool <path>` opens it. The three-way view almost always reveals the answer that the two-way `<<<<<<<` markers obscure: when both sides changed the same line, looking at the *base* tells you which side is fixing a bug and which is doing something orthogonal.

**`git rerere`** ("reuse recorded resolution") records every conflict resolution and applies it automatically when the same conflict appears again. This is transformative on long-lived feature branches that get rebased weekly: you resolve a particular conflict once, and every future rebase replays the resolution without prompting.

```bash
$ git config --global rerere.enabled true
$ git config --global rerere.autoupdate true
```

**`git checkout --conflict=diff3`** rewrites conflict markers to include the merge base in the middle:

```
<<<<<<< ours
return cache.get(key, default=None)
||||||| base
return cache.get(key)
=======
return cache.get(key) or default
>>>>>>> theirs
```

The base section makes it obvious that *both* sides modified the original `cache.get(key)`. Set `merge.conflictStyle = diff3` (or the newer `zdiff3`, which is more compact) globally and never read a 2-way marker again.

**Strategy options for whole-tree decisions.** When you know one side is correct for a whole subtree (a dependency upgrade where you want the new branch's `package-lock.json` regardless), `git checkout --theirs path/` or `--ours path/` resolves a path to one side. For an entire merge in one direction, `git merge -X theirs branch` *prefers* the other side on conflict (note: not the same as `-s ours`, which silently throws away the other side's changes). Used surgically, these are precision tools. Used by reflex, they hide bugs (see Case 16.7).

A pattern I use on tough rebases: open three terminal panes — `git diff <base>..HEAD`, `git diff <base>..<theirs>`, `git diff HEAD..<theirs>` — and resolve the file with all three views in front of me. The "what did each side actually change" comparison is far less error-prone than reading conflict markers in isolation.

## 12.7. Workflow Ergonomics: Aliases, Configs, and the Git You've Never Met

Git ships with about 150 sub-commands. Most engineers know 15. The other 135 contain genuine quality-of-life improvements that compound across a career.

**Aliases I would not work without:**

```bash
git config --global alias.st  "status -sb"
git config --global alias.co  "checkout"
git config --global alias.sw  "switch"
git config --global alias.cm  "commit -m"
git config --global alias.ca  "commit --amend --no-edit"
git config --global alias.lg  "log --graph --oneline --all --decorate"
git config --global alias.last "log -1 HEAD --stat"
git config --global alias.unstage "restore --staged"
git config --global alias.uncommit "reset --soft HEAD^"
git config --global alias.fixup "commit --fixup"
git config --global alias.absorb "absorb --and-rebase"
git config --global alias.wip "!git add -A && git commit -m 'wip' --no-verify"
git config --global alias.unwip "!git log -1 --format=%s | grep -q '^wip' && git reset HEAD~"
```

`git lg` alone is worth the price of admission — it gives you the entire branch graph in one screen, decorated with branch and tag names. `git wip` and `git unwip` form a "save my work, I'll come back to this" pair more ergonomic than stash for short interruptions.

**Configs that change the defaults to something sane:**

```bash
git config --global pull.rebase true               # never merge on pull
git config --global pull.ff only                   # refuse non-ff pull
git config --global merge.conflictStyle zdiff3     # 3-way conflict markers
git config --global rebase.autoSquash true         # --autosquash by default
git config --global rebase.autoStash true          # auto-stash on rebase
git config --global rebase.updateRefs true         # rebase chains of stacked branches
git config --global push.default current           # push to same-named remote branch
git config --global push.autoSetupRemote true      # first push creates upstream automatically
git config --global fetch.prune true               # delete remote-tracking refs for deleted remote branches
git config --global diff.algorithm histogram       # better diffs for refactors
git config --global diff.colorMoved zebra          # highlight moved blocks
git config --global commit.verbose true            # show diff in commit message editor
git config --global core.fsmonitor true            # see Section 12
git config --global core.untrackedcache true
git config --global rerere.enabled true            # remember conflict resolutions
git config --global help.autoCorrect 20            # auto-fix typos like 'git stauts'
```

Three of these deserve highlighting. `rebase.updateRefs = true` is the unsung hero of stacked PRs — when you rebase a chain of dependent branches, Git updates each intermediate branch's pointer automatically. Before this flag (Git 2.38+), people used `git-stack` or hand-managed pointers and lost work weekly.

`push.autoSetupRemote = true` removes the "fatal: the current branch has no upstream branch" annoyance on first push. New branches just push and create their tracking ref. The old habit of `git push -u origin HEAD` becomes unnecessary.

`commit.verbose = true` puts the diff of the commit you're about to make at the bottom of the commit-message editor (commented out so it's not part of the message). You can read what you're committing while writing the message — it catches "oops, I included something I didn't mean to" several times a year.

**Tools to install around Git** that are not Git itself but might as well be:

- **`git-absorb`**: looks at your current diff, figures out which previous commit each hunk should belong to, creates fixup commits automatically, and offers to autosquash. The single biggest workflow improvement of the last five years.
- **`delta`** (or `diff-so-fancy`): a pager that turns Git's plain diff output into a syntax-highlighted side-by-side or inline view. `git config core.pager delta`.
- **`lazygit`** / **`tig`** / **`gitui`**: terminal UIs that make staging, branching, and rebase interactive. Even if you live on the command line, the interactive rebase view of lazygit is faster than editing the rebase todo file by hand.
- **`gh`** / **`glab`**: GitHub/GitLab CLIs. `gh pr create --fill && gh pr view --web` is a one-liner from "I just pushed" to "I'm reviewing my own PR in the browser."
- **`git-branchless`**: smartlog views, undo, and a different rebase model for stacked branches. Originally from the Mercurial world; the experience is genuinely better than raw Git for some workflows.

The cumulative effect of these aliases, configs, and tools is that day-to-day Git stops feeling like a tax and starts feeling like a power tool. Engineers who have not done this lap don't know what they're missing — and the gap shows up most visibly during incident response, when seconds matter and friction kills.

## 13. Internals: How `git gc` Actually Works

`gc` ("garbage collection") does several things in sequence: pack loose objects into a packfile, write reachability bitmaps, prune unreachable objects older than the prune window (default 2 weeks), and update the commit-graph and midx.

The crucial concept is **reachability**. An object is reachable if it can be walked to from any ref (branch, tag, HEAD), the index, the reflog, or any stash entry. `git gc` only deletes objects that are unreachable *and* older than the prune window.

The footgun: `git gc --prune=now`. This forces immediate prune. If you ran `git reset --hard HEAD^^^` an hour ago thinking you knew what you were doing, the "lost" commits are still unreachable but reachable via reflog. `gc --prune=now` does *not* respect the reflog by default — but `gc.reflogExpireUnreachable` (default 30 days) does protect them via the reflog window. The real disaster is `git reflog expire --expire-unreachable=now --all && git gc --prune=now --aggressive`, which is the documented "I really mean it, delete everything unreachable" sequence. People run this when their `.git/` got bloated. They sometimes regret it within 24 hours.

A safer alternative that keeps the reflog window honest: `git gc --auto` is run automatically by Git when the loose-object count crosses a threshold. Trust it. Reach for explicit `gc` only when you have a specific reason.

## 14. Bisect Like a Scientist

![git bisect: binary search for the first bad commit](/imgs/blogs/git-like-a-pro-7.png)

`git bisect` is binary search through history for the commit that introduced a bug. Mark a known-bad commit and a known-good commit, and Git checks out the midpoint. You test, mark good or bad, repeat. `log2(N)` checkouts to find the culprit in N commits.

The non-obvious upgrade is `git bisect run`. Hand it a script that exits 0 (good), 1 (bad), or 125 (skip — can't decide), and Git automates the search:

```bash
#!/usr/bin/env bash
# bisect-test.sh
set -e
pip install -e . > /dev/null 2>&1 || exit 125
pytest tests/test_regression.py::test_the_thing -x -q
```

```bash
$ git bisect start
$ git bisect bad HEAD
$ git bisect good v2.4.0
$ git bisect run ./bisect-test.sh
...
b1c2d3 is the first bad commit
$ git bisect reset
```

The skip exit (`125`) is what makes this practical for real projects: when a particular commit fails to build for an unrelated reason (a dependency was bumped two commits later), skip past it.

Bisecting flaky tests is harder. The trick is to run the test 10 times in your bisect script and decide based on the failure rate:

```bash
fails=0
for i in {1..10}; do pytest tests/flaky.py -x || fails=$((fails+1)); done
[ $fails -ge 3 ]   # bad if 3 or more failed
```

Calibrate the threshold to your flake rate. Below your noise floor and you'll fingerpoint the wrong commit; above and you'll skip the real culprit.

## 15. Worktrees for Parallel Work

`git worktree add ../repo-hotfix release/24.1` creates a *second* working directory for the same repo, on a different branch, sharing the same `.git/`. No second clone, no duplicated objects. You can edit and commit in both simultaneously.

The killer use case: you are mid-refactor on `feature/big-thing` with a dirty working dir, and a P0 bug needs a hotfix on `release/24.1`. Without worktrees you would `git stash`, `git switch release/24.1`, fix, push, switch back, `git stash pop`, and pray nothing weird happened. With worktrees:

```bash
$ git worktree add ../myrepo-hotfix release/24.1
$ cd ../myrepo-hotfix
# fix, commit, push
$ cd ../myrepo            # original worktree, your dirty state intact
$ git worktree remove ../myrepo-hotfix
```

CI agents that build many branches in parallel use worktrees instead of multiple clones — same speed, dramatically less disk.

## 15.5. Stashing, Worktrees, and the Right Tool for "I Need to Switch Contexts"

People reach for `git stash` for at least four different problems, and only one of them is what stash is good at. Knowing which tool fits which problem saves entire afternoons.

**Problem A — "I'm in the middle of work and need to pull the latest main."** Don't stash. `git pull --rebase --autostash` does the right thing automatically: it stashes, fetches, rebases your branch onto the new base, and pops the stash back. Set `rebase.autoStash = true` globally and this becomes a reflex.

**Problem B — "I need to switch to a different branch for 2 minutes to grep something."** Don't stash. Open a worktree: `git worktree add ../repo-temp <branch>`. Your dirty working dir stays put. When done, `git worktree remove ../repo-temp`. No stash juggling, no risk of pop conflicts.

**Problem C — "I started feature A but realized I should ship hotfix B first."** Don't stash. Commit a WIP commit on a new branch with `git switch -c feature-a-wip; git add -A; git commit -m "wip"; git switch -c hotfix-b origin/main`. After hotfix B is done, `git switch feature-a-wip; git reset HEAD^` and you have your dirty state back. Real branches are reflog-tracked; stashes are not (their reflog is short and per-stash-slot).

**Problem D — "I want to test what the working dir would look like without these uncommitted changes."** *Now* you stash. `git stash push -m "test without my changes" --keep-index` if you want to keep the staged changes, plain `git stash` if not. Run your tests, then `git stash pop`. This is what stash was designed for.

The misuse of stash that bites teams: people accumulate dozens of unlabeled stashes over weeks ("I'll come back to that"), then one day they run `git stash drop` to clean up and realize they don't remember which slot held which work. By the time they fish around for SHAs in `git fsck --lost-found`, half the work has been gc'd. Treat stashes like browser tabs: if it's been there a day, it's not coming back. Convert to a real branch or delete it.

A worktree pattern that has saved me dozens of times: keep a `repo/` clone with `main` checked out, and a `repo-pr/` worktree where you check out PR branches for review.

```bash
$ cd repo
$ git fetch origin pull/1234/head:pr-1234
$ git worktree add ../repo-pr pr-1234
$ cd ../repo-pr
# inspect, run tests, edit if you want to suggest changes
$ cd ../repo
$ git worktree remove ../repo-pr
$ git branch -D pr-1234
```

Reviewing a PR locally takes 20 seconds and never disturbs your in-progress branch. The team that adopts this pattern stops blocking on "I'd review that, but I'm in the middle of something" excuses.

## 16. Case Studies: Ten Times Git Tried to Ruin Our Day

Each of these is a real (or composite) incident. Each one taught the team a permanent lesson. The pattern is the same: someone reached for an unfamiliar command at the wrong moment, or trusted a superstition over the model.

### 16.1 The 2 AM force-push that ate the release branch

**Setup**: Friday night, the release branch `release/24.1` is locked but a senior engineer with admin override force-pushed to "fix" what he thought was a bad merge. He had rebased a local branch onto an *old* `release/24.1` SHA — a stale tip from before three other teammates had merged hotfixes — and pushed `--force` (not `--force-with-lease`). Six hours of merged hotfixes from three teammates vanished from the remote. The first sign of trouble was a CI alert that the integration test was passing again, suspiciously, even though one of the hotfixes had been written specifically to make it pass.

**What we did**: Pulled up GitHub's branch protection log (Settings → Branches → Branch protection rules → Push activity) to confirm the SHA *before* the force-push. SSH'd into the engineer's laptop over Tailscale. His local reflog still had the old SHAs — the force-push only updated the remote, not his reflog, because reflog is per-machine. We ran `git push origin <old-sha>:release/24.1` to reset the remote to the pre-force state. The three teammates whose work had been "deleted" still had the commits in their local clones (their last fetch was before the force-push). Total recovery time: 12 minutes from alert to fix. Total ego damage to the senior engineer: substantial.

**Lesson**: Branch protection on release branches must disallow force-push *even for admins*. There is no legitimate "I really mean it" reason to force-push `release/*` — if you genuinely need to rewrite that branch, revert-then-recommit is the right tool. The reflog on the offender's machine is the most likely source of recovery, but it is per-machine and per-clone, so it pays to know where to look. The remote's "ref update" log (GitHub Audit Log, GitLab Push Log, Bitbucket Activity API) is where you confirm the pre-push SHA. Always use `--force-with-lease`, ideally aliased so muscle memory cannot regress. Never push at 2 AM without a second pair of eyes — the failure mode of a tired engineer is to skip the lease check.

### 16.2 The detached-HEAD commits that "disappeared" after a checkout

**Setup**: Engineer ran `git checkout 9a4b2cf` to "look at an old version" of a file because she needed to copy a fragment from a deleted module. Once there, she discovered a small bug, fixed it, committed three times across two hours, then ran `git switch main` to get back to her main work. Git printed a multi-line warning about "leaving N commits behind, you can save them by creating a new branch" which she did not read because the prior fifty `switch` invocations had been silent. By the time she remembered the bug fix two days later, the working dir on `main` had no trace of it and `git log --all` did not show the commits because no ref pointed at them.

**Recovery**: `git reflog` is the source of truth for HEAD's movements regardless of whether refs exist. The output showed three commit SHAs from the work session, sandwiched between the original `checkout 9a4b2cf` and the `switch main`. We ran `git checkout -b rescue-branch HEAD@{3}` (counting from the most recent reflog entry) and the commits were back in a real branch. Took 90 seconds once we knew where to look.

**Lesson**: Read the warnings. Detached HEAD is the only state in Git where commits are not anchored to any ref, which means they live for exactly as long as the reflog window keeps them — 30 days for unreachable entries by default. When you need to look at an old commit, `git switch --detach <sha>` is fine for read-only browsing, but the moment you commit, switch to a real branch with `git switch -c <branch>`. Better: configure `advice.detachedHead = true` (it's on by default) and slow down enough to actually read the message.

### 16.3 The lost stash after `git stash drop`

**Setup**: Engineer had nine stashes accumulated over a month — most stale, two important. He ran `git stash drop` repeatedly to clean up, indexing from `stash@{0}`, and accidentally dropped `stash@{0}` which had today's two-hour debugging session (because each `drop` shifts the indices, the "latest" stash slid into position 0 between drops, but he didn't notice). He realized only when he tried `git stash list` and saw an empty list.

**Recovery**: A stash is just a special commit on the `refs/stash` ref, with the working-dir state as the first parent and a fake "index" tree as the second. When you drop it, the ref no longer points there but the commit object stays in the database for the prune window. There are two ways to find it. The first is the stash reflog — `git reflog stash` shows every push and drop, with SHAs. The second, when even the reflog has been cleared (e.g., after a `gc`), is `git fsck --lost-found`, which walks the object database and reports any commit not reachable from any ref:

```bash
$ git fsck --lost-found
dangling commit b1c2d3a...
$ git show b1c2d3       # confirm it's the right session
$ git stash apply b1c2d3   # apply just like a normal stash
```

If the stash had untracked files (because you did `git stash push -u`), they were stored as a third parent of the stash commit; `git show <sha>^3` gets you to that commit.

**Lesson**: `git stash drop` is recoverable but only briefly — once `gc.reflogExpireUnreachable` (30 days default) plus a `gc` run pass, the commit goes for good. Better practice: never use `git stash` for sessions longer than 30 minutes — make a real branch with a meaningful name. When you do stash, `git stash push -m "debugging the parser bug"` so future-you knows which one to apply. Use `git stash show -p stash@{N}` to peek before dropping. And install the `git restash` alias that re-creates a stash from the reflog so the recovery path is one command, not five.

### 16.4 The rebased PR that broke every reviewer's checkout

**Setup**: Author rebased `feature/x` onto a fresh `main` and force-pushed (with `--force`, not `--force-with-lease`). Three reviewers had `feature/x` checked out locally. Their next `git pull` rejected because the histories had diverged. One reviewer ran `git pull --force` and lost his review notes that he had committed locally.

**Recovery**: The reviewer's reflog had the old branch tip. `git reset --hard <old-sha>` restored his work, then he rebased his local commits onto the new feature branch tip himself.

**Lesson**: When you rebase a shared branch, post a heads-up in the PR thread. Reviewers should `git fetch && git reset --hard origin/feature/x` (not `pull --force`), and only after committing or stashing any local work. The team should standardize on `pull.rebase = true` and `pull.ff = only` so plain `git pull` fails instead of silently merging or fast-forwarding into a weird state.

### 16.4.5 The "git is broken" merge that was actually a clock skew

**Setup**: Engineer reported that `git merge` had produced a commit dated *3 days in the future*, and now `git log` was showing commits in a confusing order. CI was flagging the commit as "future-dated" and refusing to release it. He insisted his clock was correct.

**Recovery**: `git log --pretty=fuller <sha>` shows both author date and committer date. The committer's machine had recently come back from a 4-day suspended state; macOS hadn't re-synced NTP yet, and the clock was 3 days fast. The merge commit was created with that bogus committer date. The fix was to amend the commit with `git commit --amend --no-edit --date=now` after fixing NTP, then force-push (with lease).

**Lesson**: Git stores two timestamps per commit (author and committer) and trusts your local clock. Build pipelines and "future commit" detectors do exist (`pre-receive` hooks that reject commits with timestamps more than a few minutes in the future), and they save you from this exact issue. On laptops, run `sntp -sS time.apple.com` (macOS) or `systemctl restart systemd-timesyncd` (Linux) periodically — or better, install Chrony and let it run. Always-correct clocks are a Git invariant the whole ecosystem implicitly assumes.

### 16.5 The submodule pointing at a deleted SHA

**Setup**: A submodule's parent repo recorded a SHA. The submodule's owning team rebased their default branch and the SHA became unreachable from any branch. CI started failing on every clone with `fatal: reference is not a tree: <sha>`.

**Recovery**: The submodule team's reflog still had the old SHA. They created a `refs/keep-prefer-rebase-was-bad` ref pointing at it, which made the SHA reachable again. Long-term fix: the parent repo updated its submodule pointer to a SHA on the new default branch.

**Lesson**: Submodule pointers are records of "this exact SHA at this point in time." If the upstream rebases, the parent breaks. Submodules need a contract: upstream never rewrites history of the default branch, or the parent uses tags instead of raw SHAs. Better still: don't use submodules unless you have to. For most use cases, a vendored copy or a published artifact is operationally safer.

### 16.6 The `git reset --hard` that wiped 4 hours of staged work

**Setup**: Engineer meant to `git reset --hard origin/main` to wipe one bad commit. He misread his terminal — there was a partial paste from earlier in his shell history, and the command that actually executed was `git reset --hard origin/main^^^^^^^`. Seven carets back. Four hours of staged work, three local commits, and a careful interactive-rebase reorder all gone in one keystroke. He reported it to the team as "I think I just lost half a day."

**Recovery**: `git reflog` showed the SHA right before the reset. The first reflog line was `<old-sha> HEAD@{0}: reset: moving to origin/main^^^^^^^`. We ran `git reset --hard HEAD@{1}` and everything came back, working dir included. Total recovery time, including looking up reflog syntax: 90 seconds.

**Lesson**: `reset --hard` is among the most reflog-recoverable destructive commands because all it does is move HEAD and overwrite working-dir files — the prior commits are still in the object database, still pointed at by the reflog, and the prior working-dir state can be reconstructed by walking the trees. The recipe is always: check reflog first, panic later. Adding `git config --global advice.resetUpstreamMismatch true` makes Git print a clearer warning when you reset across different upstreams. Some teams alias `git rh` to `git reset --hard` and require typing the full long form — friction as a feature for irreversible operations. The deeper lesson: any command with `--hard` deserves a moment of pause. If the keystroke memory is automatic, the destructive operation gets done before the brain catches up.

### 16.7 The merge that silently dropped a feature

**Setup**: An octopus merge of three branches included a strategy hint of `-X ours` for one branch (a teammate had set `merge.ours = true` in their gitconfig two years ago after a frustrating conflict, then forgot it was there). The merge produced a single commit, no conflicts reported, and CI passed. Two days later, QA reported that the third feature — a new pricing rule integration — was completely absent. `git log --all --grep="pricing"` showed the feature's commits were on the merged branch; they just hadn't made it into main's tree.

**Recovery**: `git diff merge-commit~1 merge-commit -- pricing/` showed the merge had silently chosen main's empty version for every conflicting file. The team reverted the merge with `git revert -m 1 <merge-commit>` (the `-m 1` flag selects which parent's view to revert *toward*, here parent 1 is the pre-merge main), then re-merged the three branches in sequence — pairwise, with default strategy — so every real conflict was surfaced and reviewed.

**Lesson**: Merge strategies (`-X ours`, `-X theirs`, `-s ours`, `-s subtree`) silently drop content under conflict and there is no warning. Never set them in gitconfig globally; only pass them on the command line for a specific operation when you know exactly what you're discarding. Octopus merges (more than two parents) compound the risk because each conflicting file is resolved against multiple competing trees and the strategy interaction is harder to reason about. When more than two branches are merging into `main`, prefer sequential pairwise merges so each step's conflicts are reviewable. Audit every developer's gitconfig periodically — a stale `merge.ours = true` from two jobs ago is a landmine.

### 16.8 The `gc --prune=now` that nuked an unreachable branch tip

**Setup**: Disk full alert on the build server. Junior SRE ran `git reflog expire --expire-unreachable=now --all && git gc --prune=now --aggressive`. The previous day, an engineer had force-pushed an experimental branch and then deleted it locally. The branch tip became unreachable. The aggressive gc deleted it permanently.

**Recovery**: None possible from the local repo. The remote (GitHub) still had the SHA in its push log; the team coordinated with GitHub support to restore the branch tip from server-side backup. Took two business days.

**Lesson**: `--expire-unreachable=now` and `--prune=now` are the only operations in Git that genuinely cannot be undone. Server-side reflog (GitHub's GraphQL `ref update` history, GitLab's push events) is sometimes the only backup. Treat aggressive gc the way you'd treat a database `DROP TABLE` — review with another engineer, never run alone, never under time pressure.

### 16.9 The CRLF / autocrlf checksum nightmare

**Setup**: A Windows engineer's gitconfig had `core.autocrlf=true` (the Git for Windows installer enables this by default). He committed a binary protobuf descriptor file that Git misidentified as text because it was small and contained no NUL bytes. Git silently rewrote LF → CRLF in the blob during checkout (and the reverse on commit), the file's bytes diverged from what other developers had, and CI on the Linux build agents started failing because the protobuf descriptor's checksum no longer matched the embedded version. The bug took two days to find because it only manifested after a fresh clone — incremental fetches kept the existing blob.

**Recovery**: Set `core.autocrlf = input` repo-wide (convert on commit only, never rewrite on checkout). Add `.gitattributes` with `*.pb binary` and `*.bin binary` to mark known binaries. Run `git add --renormalize .` to re-canonicalize working-dir contents under the new rules and commit the canonical form.

**Lesson**: Always commit a `.gitattributes` file as part of repo bootstrapping. Set `* text=auto` for the default rule (let Git detect line-ending behavior per file), then add explicit overrides: `*.png binary`, `*.bin binary`, `*.pb binary`, `*.lockb binary`. The `.gitattributes` file overrides per-developer `core.autocrlf` settings — that is exactly what you want, because per-developer settings drift across teams and OSes. The `binary` macro is shorthand for "no text conversion, no diff" and is the safest default for anything that isn't source code.

### 16.10 The signed-commit chain broken by a rebase

**Setup**: Branch protection on `main` required signed commits. An author signed five commits on a feature branch using SSH signing, then rebased onto a new base before merging because main had moved. The rebased commits kept the original author info but had no signature — by default `git rebase` re-creates commits via `cherry-pick` semantics, and `cherry-pick` does not call out to the signing path even when `commit.gpgsign = true`. Push to `main` was rejected with `error: GH006: Protected branch update failed for refs/heads/main; signed commits required`.

**Recovery**: `git rebase -S <base>` re-signs every commit during the rebase. The `-S` flag is the per-invocation form; the corresponding config is `rebase.gpgSign = true` (note the camelCase G — `rebase.gpgsign` is silently ignored, a footgun in itself). Better long-term: set both `commit.gpgsign = true` and `rebase.gpgSign = true` globally.

**Lesson**: Signing config has more knobs than people realize. `commit.gpgsign`, `tag.gpgsign`, `rebase.gpgSign`, and (on newer Git) `pull.gpgSign` are independent. Set all of them globally if you want signing to "just work" through every workflow. The other gotcha is that some Git GUIs and platform tooling create commits via internal APIs that bypass the gitconfig signing path — verify with `git log --show-signature` after every push if your team is just rolling out signed commits, until you've confirmed every tool path signs correctly.

### 16.11 The 14 GB repo that wouldn't clone on CI

**Setup**: Years of accumulated build artifacts checked into the repo (someone had used Git as artifact storage during a sprint, then nobody cleaned it up). Fresh CI agents ran out of disk during clone, and even when they had disk, clones took 18 minutes — burning a third of the build budget on a no-op. Worse, every new engineer's onboarding day started with a 30-minute coffee break waiting on git clone.

**Recovery**: Short-term, `git clone --depth=1 --filter=blob:none` brought the clone to 600 MB and 90 seconds; CI agents could now do `git fetch --unshallow --filter=blob:none` to back-fill history when needed. Long-term, `git filter-repo --strip-blobs-bigger-than 5M --invert-paths` removed every blob bigger than 5 MB from every commit on every branch in history. The repo dropped to 800 MB. Migration required org-wide coordination — the rewritten history has new SHAs, so every developer had to re-clone, every CI cache had to be invalidated, every PR had to be either rebased onto the new history or re-opened, and any branches with the old SHAs were effectively orphaned.

**Lesson**: Large binaries belong in Git LFS or an artifact store, never in raw Git. `git lfs migrate import --include="*.zip,*.tar.gz"` transparently moves matched blobs from history into LFS while preserving the apparent commit shape — this is much less invasive than `filter-repo` and rarely requires re-clones. Audit repos periodically with this one-liner that lists every commit-bound blob bigger than 5 MB:

```bash
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
  | awk '$1=="blob" && $2>5000000 { print $2, $3 }' \
  | sort -nr | head -20
```

Hook this into your CI on a weekly schedule. The first time someone commits a 200 MB JAR or a 1 GB PNG, you find out the next week instead of two years later when the repo is broken.

### 16.12 The leaked `.env` in commit history

**Setup**: An engineer committed a `.env` file containing production AWS access keys, RDS credentials, and a Stripe live secret. The commit was pushed to a public open-source mirror of an internal repo (the `git remote -v` had been mis-configured during a recent migration). The leak was discovered six hours later when GitHub's secret-scanning bot opened a security advisory automatically. By the time the engineer noticed, GitHub's advisory said the keys had already been used to spawn 47 EC2 instances mining cryptocurrency in three regions.

**Recovery**: First, the credential rotation — every key in the `.env` was revoked and reissued in under 15 minutes. The mining instances were terminated; the IAM user that had been compromised was force-deleted. Total bill from the incident: about $4,800, refunded by AWS after we filed an abuse report with the access logs. Then the history rewrite: `git filter-repo --invert-paths --path .env` removed the file from every commit on every branch. We force-pushed (with branch protection temporarily disabled by an admin), then notified everyone with a clone to re-clone. We also coordinated with GitHub Support to purge cached views of the old commits — the public commit URL still resolves for about an hour after a force-push, even after the ref is gone. Crucially: **never trust that history rewrite alone has cleansed a leaked secret.** Anyone who cloned the repo, anyone who pulled the ref before the rewrite, every cache, every CI agent's stale workspace — all of them still have the secret. Rotate first, rewrite second.

**Lesson**: Layer the defenses. `pre-commit` with `gitleaks` catches most secrets before commit; the few-seconds delay on commit is worth it. GitHub secret scanning (free for public repos, paid Advanced Security for private) catches some after. A rotation runbook for every credential type — AWS, GCP, Stripe, internal API keys, database passwords — is the only thing that matters once a leak is confirmed. Document the runbook with one named owner per credential type. `git filter-repo` is the modern replacement for `filter-branch` (which is deprecated, slow, and corrupts certain refs); install it via `pip install git-filter-repo`. The deeper meta-lesson: the cost of a leak is not the rewrite — it's the dollars and trust you spend in the hour after disclosure. Practice the rotation drill once a quarter so the muscle memory is there when you need it.

## 17. When to Reach for Git, When Not To

Git is the right answer for source code, configuration, infrastructure-as-code, documentation, and any text-shaped artifact you want a versioned, auditable history of. Below the line are cases where Git is wrong or where it needs help.

**Large binaries.** A 500 MB model checkpoint or a 2 GB game asset does not belong in a raw Git repo. Use Git LFS for moderate sizes (up to a few GB), or DVC / Git Annex for ML model and dataset versioning where you need pointer files in Git but actual storage in S3/GCS. Treating Git as artifact storage will corner you within 18 months.

**Massive monorepos.** Once a single repo crosses about 10 million LOC and a thousand engineers, raw Git starts to bend. You need a build graph (Bazel, Buck), a virtual file system (VFS for Git, EdenFS), and a code-owners enforcement layer. Microsoft's Windows repo, Google's piper-on-perforce-then-piper-on-bazel, and Meta's Sapling are the public reference points.

**When to give up and start fresh.** Sometimes a repo has been so badly mangled by years of merges, force-pushes, and history rewrites that the cleanest fix is `git checkout --orphan fresh; git add -A; git commit -m "Squashed initial commit"; git push -u origin fresh; git branch -m main main-archive; git branch -m fresh main`. The audit trail is preserved on the archived branch; everyone starts from a clean slate. This is rare — once or twice in a decade — but it is sometimes the right answer when no other technique can untangle the mess.

The deeper meta-lesson of every section above is the one I will end on: **Git rewards mental models, not memorization.** When something goes wrong, do not reach for Stack Overflow first. Open `git status`, `git log --graph --oneline --all`, and `git reflog`, and look at the trees, the refs, and the recent moves. Nine times out of ten, the answer is visible if you know how to look. The tenth time, the reflog still has the SHA you need. That is the entire game.

A short closing checklist for engineers who want to move past the plateau this month: configure the dozen aliases and configs from Section 12.7, install `git-absorb` and `delta`, turn on `core.fsmonitor` and the commit-graph if your repo is bigger than 50k commits, write the three-trees table from Section 4 by hand once, do an interactive rebase using `--autosquash` on your next PR, and the next time someone on the team panics about "lost" commits, run `git reflog` for them and watch their face. Those six steps are what separate the plateau from real fluency, and none of them takes more than an afternoon.

For more on tooling and review hygiene, you might enjoy [Database connection pooling: a senior engineer's guide](/blog/software-development/database/database-connection-pooling) for the same "small choices, big blast radius" framing applied to a different system, [Random UUIDs are killing your database performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance) for another case where the default tool quietly hurts you at scale, and [Design patterns: a practical guide](/blog/software-development/system-design/design-patterns-guide) for adjacent reasoning about codebase structure.
