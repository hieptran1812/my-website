---
title: "Using Git Like a Senior: A Practical Workflow and Troubleshooting Playbook"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "The day-to-day Git workflow a senior actually runs — trunk-based branching, atomic commits, rebase-to-stay-current, a what-do-I-do-now undo decision tree, and twelve real incidents with the exact fix for each."
tags:
  [
    "git",
    "version-control",
    "workflow",
    "trunk-based-development",
    "rebase",
    "merge-conflicts",
    "troubleshooting",
    "developer-tools",
    "code-review",
    "team-workflow",
  ]
category: "software-development"
subcategory: "Version Control"
author: "Hiep Tran"
featured: true
readTime: 51
---

There is a gap between knowing Git commands and knowing Git, and it shows up at the worst possible moment: a release branch breaks at 5pm, someone force-pushed over a teammate's work, a pull request has thirty "wip" commits, and the person at the keyboard is googling "git undo last commit" while six people wait. The commands were never the hard part. What separates a senior is a small set of habits and a mental model that turns every one of those moments from a panic into a thirty-second fix.

This is not the article about Git's internals — the content-addressed object store, the three trees, how `git gc` walks the reachability graph. That material matters, and I wrote it up separately in [Git Like a Pro](/blog/software-development/version-control/git-like-a-pro-object-model-workflows-and-recovery); reach for it when you want to understand *why* recovery works at the object level. This article is the other half: the **workflow** a senior runs every day, the **decisions** they make without thinking, and the **troubleshooting playbook** for the dozen situations that actually come up on a real team. It assumes you already know `add`, `commit`, `push`, `pull`, and `branch`. It aims to give you the judgment that sits on top of them.

![The life of a change: the lifecycle loop every change travels](/imgs/blogs/using-git-like-senior-workflow-troubleshooting-playbook-1.webp)

The diagram above is the mental model. Every change a senior ships — a one-line config fix, a three-hundred-line feature — rides the same loop: sync `main`, branch off, commit in small atomic steps, push, open a small pull request, rebase to stay current, merge, delete the branch, and start again. The entire job of "using Git well" is running that loop tightly and knowing what to do when one step goes sideways. The rest of this article walks the loop, then turns to the things that break it.

## 1. Why the workflow, not the commands, is what makes a senior

Ask a mid-level engineer and a staff engineer to ship the same change and you will not see a difference in *which* commands they type. You will see a difference in **when** and **why**. The junior reaches for whatever recipe their last Stack Overflow search returned. The senior has internalized a model, so the right command is obvious and the blast radius of a mistake is small by construction.

The cost of not having that model is not measured in commands typed. It is measured in three taxes that compound over the life of a project.

| Situation | What most engineers do | What a senior does |
| --- | --- | --- |
| Starting work | branch off whatever is checked out, sometimes stale `main` | `git switch main && git pull --rebase`, *then* branch |
| Committing | one big commit at the end, message "updates" | small atomic commits, message explains *why* |
| Staying current | merge `main` into the branch repeatedly | rebase the branch onto `origin/main` |
| A pushed mistake on a shared branch | `git reset` + force-push, hope nobody noticed | `git revert` — never rewrite shared history |
| "I lost my work" | reclone, redo the work | `git reflog`, recover the SHA in thirty seconds |
| Opening a PR | 25 commits including "fix typo", "address review" | a handful of clean commits, rebased and squashed |

Each row is the same operation done with and without a model. The senior version is not more clever; it is more *boring*, and boring is the point. A boring history bisects cleanly six months later. A boring branch never eats a teammate's commits. A boring undo never loses four hours of work.

> Seniority in Git is not knowing more commands. It is making fewer of your mistakes irreversible.

The way past the plateau is to stop collecting recipes and start running one disciplined loop, every time, until it is muscle memory. The next four sections are that loop. The five after that are what to do when it breaks.

## 2. The life of a change

Look again at the figure that opened the article. Read it as a checklist you run for every change, no matter how small:

1. **Sync `main`.** `git switch main && git pull --rebase`. You want your branch to start from the team's latest, not from whatever was checked out three days ago.
2. **Branch off.** `git switch -c feat/retry-fetch`. Short-lived, named for the work.
3. **Commit often, small and atomic.** Each commit is one coherent step that builds and passes on its own.
4. **Push the branch.** `git push -u origin feat/retry-fetch`. The `-u` sets the upstream so future `push`/`pull` need no arguments.
5. **Open a small PR.** One concern, reviewable in one sitting.
6. **Rebase to stay current.** While the PR is open, `git fetch && git rebase origin/main` so you integrate continuously, not in one giant merge at the end.
7. **Review and merge.** Squash or rebase-merge so `main` gets a clean entry.
8. **Delete the branch**, locally and on the remote, and start the loop again.

Here is the loop as a real terminal session — the commands a senior types on autopilot, with the flags that matter:

```bash
# 1. start from the team's latest main
git switch main
git pull --rebase                 # replay any local main commits on top, no merge bubble

# 2. branch off for one unit of work
git switch -c feat/retry-fetch    # `switch -c` is the modern `checkout -b`

# ... edit, then ...

# 3. stage selectively and commit in atomic steps
git add -p                        # review each hunk; stage only what belongs in THIS commit
git commit                        # opens $EDITOR for a real message (see section 5)

# 4. publish the branch and set upstream
git push -u origin feat/retry-fetch

# 5. open the PR from the CLI
gh pr create --fill --web         # GitHub CLI; --fill seeds title/body from commits

# 6. a day later, main has moved — stay current
git fetch origin
git rebase origin/main            # replay your commits on top of the new main
git push --force-with-lease       # update the PR branch safely (see section 8)

# 7. after approval, merge keeps main linear
gh pr merge --squash --delete-branch   # or --rebase; never --merge for small PRs

# 8. tidy your local copy
git switch main && git pull --rebase
```

Notice three senior tells already. First, `git switch` instead of the overloaded `git checkout` — `switch` only changes branches, `restore` only changes files, and splitting the old `checkout` into two verbs removes a whole class of "I checked out a file when I meant a branch" mistakes. Second, `git add -p` to build each commit deliberately rather than `git add .` and hope. Third, `--force-with-lease` instead of `--force`, which we will come back to because it is the single most important safety habit on this list.

The loop has a property worth naming: **every step is cheap to undo until step 7.** A branch you haven't pushed is private; you can rewrite it freely. A PR you haven't merged can be force-pushed safely with `--force-with-lease`. Only at the merge does your work become part of the shared history that other people build on — and that is exactly the boundary where the undo rules change. Keep that boundary in mind; the entire troubleshooting half of this article is organized around it.

### 2.1 Second-order: small loops beat big loops

The most common way this loop goes wrong is making it too big. A branch that lives for two weeks accumulates conflicts, drifts from `main`, and turns its eventual merge into an event. A branch that lives for a day integrates almost for free. The senior instinct is to **shrink the loop**: smaller PRs, more frequent merges, branches measured in hours or days, not weeks. We make that concrete in section 4.

## 3. Where your work actually lives

Before the workflow can feel obvious, one model has to click: a change in Git lives in exactly four places, and every command you know is just a way to move it from one place to the next.

![Where your work lives: working directory, staging, local repo, remote, and the commands between them](/imgs/blogs/using-git-like-senior-workflow-troubleshooting-playbook-2.webp)

Read the figure top to bottom. Your edit starts in the **working directory** — plain files on disk. `git add` copies a snapshot of the parts you choose into the **staging area** (also called the index), which is the draft of your next commit. `git commit` freezes that draft into the **local repository** — the `.git` folder that holds all your commits and branches, on your machine only. `git push` sends commits up to the **remote** (`origin` on GitHub or GitLab), the shared source of truth your teammates clone and review.

The reverse arrows matter just as much, because they are how you *undo*. `git restore` pulls a file back from staging or from the last commit into your working directory. `git reset` moves commits back out of history into staging or the working directory. `git fetch` brings the remote's commits down into your local repo. Once you see every command as a move between these four boxes, the undo commands stop being magic — `git reset --soft HEAD~1`, for instance, is just "move the last commit back into staging," nothing more.

Here is the same model expressed as commands, with the box each one touches called out:

```bash
# working dir -> staging
git add file.py            # stage the whole file
git add -p                 # stage selected hunks only

# staging -> working dir (unstage, keep edits)
git restore --staged file.py

# working dir -> discard (DESTRUCTIVE: throws away uncommitted edits)
git restore file.py

# staging -> local repo
git commit -m "..."        # freeze the staged snapshot into a commit

# local repo -> staging (uncommit, keep changes staged)
git reset --soft HEAD~1

# local repo -> remote
git push

# remote -> local repo (download, do not touch your files)
git fetch origin

# remote -> local repo + working dir (download AND integrate)
git pull --rebase
```

The single most clarifying distinction here is **`fetch` versus `pull`**. `git fetch` only updates your local copy of the remote's branches (`origin/main` moves; your `main` does not). `git pull` is `fetch` followed by an integration step that *does* move your branch and touch your files. A senior fetches constantly — it is read-only and safe — and pulls deliberately, with `--rebase`, only when ready to integrate. Conflating the two is how people end up with surprise merge commits, which we diagnose in case study 8.

| Command | Reads from | Writes to | Destructive? |
| --- | --- | --- | --- |
| `git add` | working dir | staging | no |
| `git restore --staged` | staging | working dir (unstage) | no |
| `git restore <file>` | last commit | working dir | **yes** (loses edits) |
| `git commit` | staging | local repo | no |
| `git reset --soft` | local repo | staging | no |
| `git reset --hard` | local repo | working dir + staging | **yes** (loses edits) |
| `git fetch` | remote | local repo (`origin/*`) | no |
| `git pull` | remote | local repo + working dir | no (but can conflict) |
| `git push` | local repo | remote | no (but can be rejected) |

Memorize the "destructive" column. Two commands on this list throw away uncommitted work with no confirmation: `git restore <file>` and `git reset --hard`. Both show up later as case studies, because both are how people lose hours of work. Everything else is recoverable.

## 4. Trunk-based development in practice

With the lifecycle and the four boxes in hand, the next decision is your branching model — and it shapes everything about the daily loop. There are three common models, but for most teams the answer is **trunk-based development**: one long-lived branch (`main`), short-lived feature branches off it, and frequent merges back.

![Trunk-based development: short-lived branches fork off a releasable main and merge back](/imgs/blogs/using-git-like-senior-workflow-troubleshooting-playbook-3.webp)

The figure shows the shape. `main` is a straight line of commits that is *releasable at every point*. Feature branches (`feat-a`, `feat-b`) fork off, carry one or two days of work, and merge straight back. There is no `develop` branch, no `release/*` branch that lives for weeks, no long-running integration branch where conflicts pile up. The two diamonds in the figure are the whole pattern, repeated: fork, work, merge, fork, work, merge.

Why this and not the alternatives? Here is the comparison every senior carries:

| Model | Branches | Best for | Main cost |
| --- | --- | --- | --- |
| **Git Flow** | `main`, `develop`, `feature/*`, `release/*`, `hotfix/*` | shrink-wrapped software with explicit, infrequent releases | heavy ceremony; long-lived branches accumulate conflicts |
| **GitHub Flow** | `main` + `feature/*`, deploy from `main` | web apps with continuous deploy | needs solid CI and feature flags |
| **Trunk-based** | `main` + very short `feature/*` (or direct commits behind flags) | high-velocity teams, continuous integration | requires discipline: small PRs, fast review, feature flags |

Git Flow earned its popularity in an era of quarterly releases, and it is still defensible when you genuinely ship versioned artifacts to customers who upgrade on their own schedule. But on a team that deploys daily, its long-lived `develop` and `release` branches are pure overhead: they are where merge conflicts go to breed. Trunk-based development inverts the bet — keep branches so short that they never diverge far enough to conflict badly.

The discipline that makes trunk-based work is the **feature flag**. If a feature is not done but `main` must stay releasable, you merge the incomplete code *dark* — behind a flag that is off in production:

```python
# ship to main continuously, even when the feature isn't done
if feature_flags.enabled("new_checkout", user):
    return new_checkout(cart)
return legacy_checkout(cart)
```

This is the trick that lets you merge a half-finished feature into a releasable `main`: the new path exists in the codebase, is exercised by tests, but is dark until the flag flips. It decouples *deploy* (shipping the code) from *release* (turning it on), and it is what makes "merge within a day or two" possible without shipping broken features.

A few naming and hygiene conventions that pay off on a real team:

```bash
# branch names encode owner + type + ticket, so `git branch` is self-documenting
git switch -c htran/feat/PROJ-482-retry-fetch
git switch -c htran/fix/PROJ-501-null-pointer

# keep your local branch list clean: prune remote-tracking refs that are gone
git fetch --prune
# list local branches whose upstream was deleted (merged + cleaned up on the remote)
git branch -v | grep '\[gone\]'
```

### 4.1 Second-order: protect `main`, don't trust discipline

Discipline does not scale; configuration does. The senior move is to make the *wrong* thing impossible rather than merely discouraged. On the remote, turn on branch protection for `main`: require a pull request, require CI to pass, require at least one review, and disallow force-pushes. Now "I accidentally committed straight to `main`" (case study 1) and "someone force-pushed the shared branch" (case study 5) cannot happen on the server, regardless of who is at the keyboard. The figure's promise — `main` releasable at every commit — is enforced by the platform, not by everyone remembering to be careful.

### 4.2 Keep PRs small, and stack them when you can't

The other discipline that makes trunk-based development work is the *size* of each pull request. A reviewer's attention is the scarce resource, and it degrades sharply with diff size: research on code review consistently finds defect-detection drops once a change passes a few hundred lines, and a thousand-line PR effectively gets rubber-stamped. The senior target is a PR a competent reviewer can fully understand in one sitting — roughly under 400 lines of real change.

When a feature is genuinely too big for one PR, the answer is not a giant branch; it is a **stack** of small PRs, each building on the last:

```bash
git switch -c htran/feat/PROJ-482-part-1-schema main
# ... commit, push, open PR #1 ...
git switch -c htran/feat/PROJ-482-part-2-api    # branch off part-1, not main
# ... commit, push, open PR #2 (base = part-1 branch) ...
```

Each PR is small and reviewable; they merge bottom-up. The friction in stacks is rebasing the whole stack when the bottom changes — which is exactly what `rebase.updateRefs = true` (section 11) automates, moving every branch pointer in the stack in one rebase. Tools like Graphite or `git-branchless` add ergonomics, but the core mechanic is just short branches stacked on short branches, keeping every individual review small.

## 5. Commit like a senior

Commits are the unit of history, and history is documentation that you write once and read for years. The difference between a junior's history and a senior's history is the difference between a pile of receipts and a well-kept ledger.

![Anatomy of a commit a senior would approve: one atomic change plus a structured message](/imgs/blogs/using-git-like-senior-workflow-troubleshooting-playbook-4.webp)

The figure breaks a good commit into its two halves. On the left: **one atomic change** — a single concern. Add the helper, use it, test it, nothing else. The commit builds and passes on its own, which means it can be reverted on its own, cherry-picked on its own, and bisected through cleanly. On the right: a **structured message**. A subject line of fifty characters or fewer in the imperative mood ("fix: retry fetchUser on 503", not "fixed the fetch thing"), a blank line, then a body wrapped at seventy-two columns that explains *why* the change exists — the reasoning a future reader cannot reconstruct from the diff — and a footer with machine-readable trailers (`Fixes #482`, `Co-authored-by:`).

The body is the part juniors skip and seniors treat as the whole point. The diff already tells you *what* changed. The message must tell you *why*, because in six months the why is the only thing nobody remembers:

```
fix: retry fetchUser on transient 503

The upstream identity service returns 503 for ~200ms during its
rolling deploys (roughly every 20 minutes). Without a retry, those
windows surface as user-facing login errors and page ~3 on-call alerts
per deploy. Retry up to 3 times with exponential backoff (100/200/400ms),
which covers the observed blip with margin.

Fixes #482
```

That message will still make sense to someone debugging this code in 2028. "fixed bug" will not.

### 5.1 The tool that makes atomic commits easy: `git add -p`

Atomic commits sound like extra work, and they are — unless you use patch mode. `git add -p` walks you through your changes one hunk at a time and asks whether each belongs in the current commit:

```bash
git add -p
# Stage this hunk [y,n,q,a,d,s,e,?]?
#   y - stage this hunk
#   n - skip it (it'll go in a different commit)
#   s - split it into smaller hunks
#   e - edit the hunk by hand for surgical staging
```

This is how you commit a bug fix and an unrelated refactor *separately* even though you made both edits in the same file in the same sitting. You stage the fix's hunks (`y`), skip the refactor's (`n`), commit, then stage and commit the rest. The result is two clean commits instead of one muddy one — and a reviewer who can understand each in isolation.

### 5.2 Rewrite before anyone sees it

Here is the rule that resolves the eternal "commit early vs. commit clean" tension:

> Commit early and often while you work. Rewrite the history clean before you push it for review. Never rewrite it after it's shared.

While your branch is private, your commits are scratch paper. Use `git commit --amend` to fold a fix into the previous commit, and interactive rebase to reorder, squash, and reword:

```bash
# fold a forgotten change into the last commit (private branch only)
git add forgotten_file.py
git commit --amend --no-edit

# clean up the whole branch before opening the PR
git rebase -i origin/main
# in the editor, mark commits: pick / reword / squash / fixup / drop / reorder
```

The cleanest workflow for review rounds uses **fixup commits** plus autosquash. When a reviewer asks for a change to commit `abc123`, you don't add a "address review" commit. You make a fixup targeting the original:

```bash
git commit --fixup abc123          # creates "fixup! <subject of abc123>"
# ... more review rounds, more fixups ...
git rebase -i --autosquash origin/main   # fixups auto-position under their targets
git push --force-with-lease
```

The reviewer re-reviews a clean set of commits, not a growing pile of "fix review comment" noise. Turn this on permanently with `git config --global rebase.autosquash true`, covered in section 11.

The whole discipline rests on the boundary from section 2: rewriting is free and safe up to the moment you share, and forbidden after. That boundary is also exactly the line the undo decision tree (section 7) is built around.

## 6. Staying in sync: merge vs. rebase

While your PR is open, `main` keeps moving. You have two ways to incorporate those new commits into your branch — merge `main` in, or rebase your branch onto it — and choosing well is what keeps a project's history navigable.

![Merge-everything vs rebase-then-merge: tangled bubbles versus a linear, bisectable history](/imgs/blogs/using-git-like-senior-workflow-troubleshooting-playbook-5.webp)

The figure contrasts the two outcomes. On the left, the engineer merges `main` into their branch every time it moves. The branch fills with merge commits; `git log --graph` becomes a plate of spaghetti; `git bisect` jumps sideways through merges; `git revert` on a range is fraught. On the right, the engineer rebases the branch onto `origin/main`: the branch's commits are replayed on top of the latest `main`, producing a single linear sequence. Bisect walks it cleanly. Revert is trivial. The history reads like a story told in order.

The mechanical difference: **merge preserves the actual branching shape** of what happened; **rebase rewrites your commits** as if you had started from the current `main` all along. Here is the same situation handled both ways:

```bash
# Option A — merge main into your branch (preserves bubbles)
git fetch origin
git merge origin/main
# creates a merge commit on your branch; history now branches and rejoins

# Option B — rebase your branch onto main (linear) — the senior default
git fetch origin
git rebase origin/main
# replays your commits on top of the latest main; no merge commit
```

The senior heuristic, stated plainly:

| Operation | Use merge | Use rebase |
| --- | --- | --- |
| Updating *your own unpushed/private* branch with `main` | — | **yes**, keep it linear |
| Integrating a *shared* feature branch others have pulled | **yes**, don't rewrite their base | — |
| Final merge of a small PR into `main` | squash or rebase-merge | — |
| Combining two long-lived release lines | **yes**, the bubble is real history | — |

The one rule that prevents the most pain: **never rebase commits other people have based work on.** Rebasing rewrites commit SHAs; if a teammate has your old commits in their branch and you rebase, their next pull turns into a mess of duplicated commits and conflicts. Rebase is for *your* branch before it is shared. Once it is shared, merge.

There is a deeper reason seniors prefer linear history, and it is not aesthetics. A linear history makes three high-value tools work well: `git bisect` (binary-search the commit that introduced a bug), `git revert` (cleanly undo one change), and `git log --first-parent` (read `main` as one decision per line). Merge bubbles degrade all three. The discipline of section 5 (clean commits) and this section (linear integration) exists to keep those tools sharp — see the same argument applied to keeping a *codebase* clean in the [design patterns guide](/blog/software-development/system-design/design-patterns-guide).

### 6.1 The config that removes the most common mistake

By default, `git pull` does a *merge*, which is why so many histories are littered with "Merge branch 'main' of github.com:..." commits that nobody intended. Make pull rebase by default:

```bash
git config --global pull.rebase true
git config --global rebase.autoStash true   # stash dirty changes, rebase, unstash
```

Now `git pull` replays your local commits on top of the incoming ones with no merge bubble, and `autoStash` means you can pull even with uncommitted changes. This one setting eliminates the surprise merge commit (case study 8) for good.

### 6.2 A worked rebase, start to finish

Words about rebase are less convincing than watching the graph change. Suppose you branched `feat/retry` off `main`, made two commits, and meanwhile a teammate merged a logging fix to `main`. Your history has diverged:

```bash
git log --oneline --graph --all
# * 7a1c2d3 (HEAD -> feat/retry) use retryFetch in fetchUser
# * 4b5e6f7 add retryFetch helper
# | * 9d8c7b6 (origin/main) fix: structured logging for auth
# |/
# * 2a3b4c5 chore: bump deps
```

The `|/` is the fork: your two commits and the teammate's one commit both descend from `2a3b4c5`. Rebasing replays *your* commits on top of the teammate's:

```bash
git fetch origin
git rebase origin/main
# First, rewinding head to replay your work on top of it...
# Applying: add retryFetch helper
# Applying: use retryFetch in fetchUser

git log --oneline --graph --all
# * f0e1d2c (HEAD -> feat/retry) use retryFetch in fetchUser
# * a9b8c7d add retryFetch helper
# * 9d8c7b6 (origin/main) fix: structured logging for auth
# * 2a3b4c5 chore: bump deps
```

Two things changed. The fork is gone — the history is one straight line. And your two commits have *new SHAs* (`f0e1d2c`, `a9b8c7d` instead of `7a1c2d3`, `4b5e6f7`), because rebase rewrote them onto a new base. That SHA change is exactly why rebasing a *shared* branch is dangerous: anyone holding the old SHAs now has commits that no longer exist on yours. On your own branch, before sharing, it is free and produces a history a reviewer can read top to bottom.

## 7. What do I do now? The undo decision tree

Everything so far is the happy path. Now the other half: something went wrong and you need to undo it. The reason "how do I undo in Git" feels impossibly deep is that people search for the *symptom* ("undo a commit", "discard changes") instead of asking the one question that actually determines the answer.

![The undo decision tree: committed? pushed? shared? determines the safe command](/imgs/blogs/using-git-like-senior-workflow-troubleshooting-playbook-6.webp)

The figure is the entire troubleshooting framework on one page, and it is worth committing to memory because it collapses dozens of "how do I undo X" questions into three yes/no checks:

1. **Is it committed yet?** If the change is still in your working directory or staging area, you have not made history — `git restore` is all you need.
2. **Is it pushed yet?** If you committed but have not pushed, the commits are private — you can freely `git commit --amend`, `git reset`, or rebase to rewrite them.
3. **Is it shared?** If you pushed and others may have pulled it, the history is shared — you must `git revert` (which adds a new inverse commit) and never rewrite.

Walk the tree from the top and the safe command falls out mechanically:

```bash
# --- not committed yet ---
git restore file.py              # discard working-dir edits to one file (DESTRUCTIVE)
git restore --staged file.py     # unstage, keep the edits
git restore .                    # discard ALL working-dir edits (DESTRUCTIVE)
git stash                        # park everything safely instead of discarding

# --- committed, not pushed ---
git commit --amend               # fix the most recent commit (message or content)
git reset --soft HEAD~1          # uncommit, keep changes staged
git reset HEAD~1                 # uncommit, keep changes in working dir (mixed, default)
git rebase -i HEAD~3             # rewrite the last 3 commits (reorder/squash/drop)

# --- pushed, only my branch (nobody else based work on it) ---
git reset --hard origin/main     # ... then:
git push --force-with-lease      # rewrite the remote branch SAFELY

# --- pushed AND shared with the team ---
git revert <sha>                 # add a commit that undoes <sha>; history stays intact
git revert <old>..<new>          # revert a range, oldest applied last
```

The line that separates the bottom two branches is the most important line in Git. Above it, rewriting history is fine — the commits are yours alone. Below it, rewriting history is a hostile act that breaks everyone who has your commits. `git revert` exists precisely for the below-the-line case: it does not erase the bad commit, it *appends a new commit that undoes it*, so everyone's history stays consistent.

> If anyone else might have pulled it, you don't get to rewrite it. You get to revert it.

### 7.1 Second-order: `git revert` of a merge is special

One sharp edge worth knowing before you hit it: reverting a *merge commit* requires telling Git which parent to keep, because a merge has two parents and "undo" is ambiguous. Use `-m 1` to keep the first parent (usually `main`):

```bash
git revert -m 1 <merge-sha>      # undo a merged PR, keeping mainline as parent 1
```

Forget the `-m` and Git refuses with "mainline was not specified"; that error is the decision tree telling you it needs one more bit of information. Reverting a revert (to re-introduce the change later) has its own subtlety, but the rule of thumb holds: on shared history, you only ever add commits, never remove them.

## 8. reset vs. revert vs. restore vs. checkout

The decision tree tells you *which* command to reach for. This section is the reference for what each command actually touches, because the four "undo" commands operate on different layers, and choosing the wrong one is the single most common way people destroy work.

![reset vs revert vs restore vs checkout: which layer each touches and when it is safe](/imgs/blogs/using-git-like-senior-workflow-troubleshooting-playbook-7.webp)

Read the figure as a truth table. The columns are the three places work lives (working directory, staging, commit history) plus the one question that matters on a team (is it safe after pushing?). The rows are the four commands. The color pattern tells the story before you read a single label: `git revert` is the only all-green row — the only undo that is always safe on shared history — while `git reset --hard` is a wall of red, because it rewrites history *and* destroys uncommitted work.

| Command | Working dir | Staging | History | Safe when pushed? |
| --- | --- | --- | --- | --- |
| `git restore <file>` | overwrites file | untouched | untouched | yes (local file only) |
| `git reset --soft HEAD~` | untouched | keeps changes | moves HEAD back | no (rewrites) |
| `git reset --mixed HEAD~` | untouched | unstaged | moves HEAD back | no (rewrites) |
| `git reset --hard HEAD~` | **wipes changes** | wipes | moves HEAD back | no (rewrites) |
| `git revert <sha>` | untouched | untouched | adds inverse commit | **yes (append-only)** |
| `git checkout <branch>` | swaps tree | swaps | untouched | yes (no rewrite) |

Three modes of `reset` are worth internalizing because the difference is exactly *how far back the change is dragged*:

- **`--soft`**: move `HEAD` back, leave everything staged. "Uncommit, but keep my work ready to recommit." This is how you squash the last three commits into one: `git reset --soft HEAD~3 && git commit`.
- **`--mixed`** (the default): move `HEAD` back, unstage the changes, keep them in the working directory. "Uncommit and unstage."
- **`--hard`**: move `HEAD` back and **throw away** the working directory and staging. "Uncommit and obliterate." This is the only destructive one, and it is the source of case study 7.

The senior habit is to *never* type `git reset --hard` without first asking "what uncommitted work am I about to destroy, and have I stashed it?" If there is any doubt, `git stash` first — stashing is free insurance, and a stash you didn't need is trivial to drop.

### 8.1 `--force-with-lease`, the habit that prevents disasters

When you rewrite a pushed-but-private branch (the third branch of the decision tree), you must force-push, because the remote's history no longer matches yours. The naive command is `git push --force`, and it is dangerous: it overwrites whatever is on the remote, including commits a teammate pushed that you never saw. The senior command is:

```bash
git push --force-with-lease
```

`--force-with-lease` refuses the push if the remote has commits you don't have locally — that is, if someone else pushed since your last fetch. It force-pushes only when the remote is where you *think* it is. The difference between `--force` and `--force-with-lease` is the difference between case study 5 happening and not happening. Alias it so you never type the dangerous one:

```bash
git config --global alias.pushf 'push --force-with-lease'
```

## 9. The safety net: reflog, your real undo button

Here is the fact that should change how you feel about Git mistakes: **almost nothing you commit is ever truly lost.** Even after a botched rebase, a `reset --hard`, or a deleted branch, the commits usually sit in the repository for about ninety days, reachable through the reflog.

![The reflog is a time machine for HEAD: every move is logged for about 90 days](/imgs/blogs/using-git-like-senior-workflow-troubleshooting-playbook-8.webp)

The figure shows what the reflog is: a chronological log of every position `HEAD` has occupied. Every commit, checkout, rebase, reset, and merge moves `HEAD`, and Git records each move with a stable address — `HEAD@{0}` is where you are now, `HEAD@{1}` is one move ago, and so on back through time. In the figure, a `reset --hard` at `HEAD@{2}` appears to destroy work, but `HEAD@{3}` still points at the pre-reset state. Recovery is a lookup plus a new branch:

```bash
# something went wrong — see every recent position of HEAD
git reflog
# 0a1b2c3 HEAD@{0}: reset: moving to HEAD~3      <- the mistake
# 9f8e7d6 HEAD@{1}: commit: nearly done
# 3c2b1a0 HEAD@{2}: commit: real work here       <- what I want back
# ...

# recover by pointing a new branch at the good SHA
git switch -c rescue 3c2b1a0
# or, if you just want your current branch back where it was:
git reset --hard HEAD@{2}
```

This single tool dissolves most Git panic. "I rebased and lost three commits" — they are in the reflog. "I did `reset --hard` and my work is gone" — the pre-reset SHA is in the reflog. "I deleted the branch before merging" — its tip is in the reflog (or findable via `git fsck --lost-found`). The recovery is almost always: find the SHA in `git reflog`, then `git switch -c <name> <sha>`.

Two caveats keep this honest. First, the reflog is **local** — it lives in your clone, not on the remote, so it can only rescue work that reached *your* machine. It cannot recover a commit a teammate lost on theirs. Second, it only tracks *committed* work; changes that were never committed (only edited or staged) are not in the reflog, which is why the discipline of committing often (section 5) is also a safety practice. Uncommitted work destroyed by `reset --hard` is recoverable only if it happened to be staged, via dangling blobs — far harder than a reflog lookup.

> Commit often, and the reflog becomes a ninety-day undo. The only work Git can't get back for you is the work you never committed.

### 9.1 Stash: the no-commit parking lot

The reflog only protects committed work, and sometimes you have half-finished edits you don't want to commit but do need to set aside — an urgent bug report comes in while you're mid-feature. `git stash` is the parking lot: it saves your working-directory and staging changes, reverts you to a clean tree, and lets you retrieve them later.

```bash
git stash push -m "wip: half of the retry logic"   # park changes with a label
git switch main && git switch -c hotfix/urgent       # go deal with the fire
# ... fix, commit, push ...
git switch feat/retry
git stash list                                       # stash@{0}: On feat/retry: wip: half of...
git stash pop                                         # reapply AND drop from the stack
# or, to reapply but keep it on the stack:
git stash apply stash@{0}
```

Two senior habits with stash. First, **always label your stashes** with `-m`; an unlabeled `stash@{3}` six entries deep is a mystery you will be tempted to drop blindly. Second, prefer a throwaway branch or a `wip` commit for anything you'll keep more than an hour — the stash stack is a stack, it has no branch structure, and a forgotten stash is invisible in your normal workflow. Stash is for "I'll be right back," not for long-term storage.

There is a sharp edge: `git stash pop` that hits a conflict applies what it can, leaves the conflict markers, **and does not drop the stash** — which is good (your stash is safe) but surprises people who assume "pop" always cleans up. Resolve the conflict, `git add`, then `git stash drop` manually. That surprise is case study 13.

### 9.2 Bisect: let Git find the commit that broke it

The reflog rescues lost work; `git bisect` rescues lost *time*. When a bug appears and you don't know which of the last two hundred commits introduced it, bisect binary-searches the history for you. You mark one known-bad commit and one known-good commit, and Git checks out the midpoint and asks; each answer halves the search. Two hundred commits resolve in about eight steps.

```bash
git bisect start
git bisect bad                      # current commit is broken
git bisect good v2.3.0              # this tag was fine
# Git checks out the midpoint; you test it, then:
git bisect good                     # ... or `git bisect bad`
# repeat ~log2(N) times until Git prints the first bad commit
git bisect reset                    # return to where you started
```

The power move is **automating the test** so bisect runs unattended. Give it a script that exits 0 for good and non-zero for bad, and Git drives the whole search itself:

```bash
git bisect start HEAD v2.3.0        # bad ref, then good ref
git bisect run pytest tests/test_login.py::test_503_retry
# Git checks out, runs, scores, and bisects automatically until it names
# the exact commit that turned the test red.
```

This is "bisect like a scientist": instead of squinting at diffs, you state the hypothesis as a test and let Git find the commit that falsifies it. A linear history (section 6) is what makes this clean — bisecting through a tangle of merge bubbles is where it gets confusing, which is one more reason seniors keep history linear.

### 9.3 Worktrees: a second working copy without stashing

The classic interruption — "review this urgent PR while I'm mid-feature" — usually triggers a stash-switch-stash dance. `git worktree` is the cleaner answer: it checks out a *second* branch into a *separate directory* that shares the same `.git` object store, so you can have your feature branch and the PR branch both checked out at once, in different folders, with no stashing and no losing your build state.

```bash
# from your main clone, create a second working tree for the PR branch
git worktree add ../repo-review pr/1234
cd ../repo-review                 # the PR branch, fully checked out, builds independently
# ... review, run tests, leave your feature dir completely untouched ...
cd ../repo
git worktree remove ../repo-review   # clean up when done
git worktree list                    # see all active worktrees
```

Worktrees shine for anything that needs two branches live simultaneously: comparing behavior between branches, running a long test suite on one branch while coding on another, or keeping a permanent `main` worktree for quick hotfixes next to your feature worktree. Each tree has its own working directory and index but shares history and the reflog, so it's cheaper than a second clone and stays in sync automatically. The one rule: you can't check out the same branch in two worktrees at once — Git refuses, because two trees editing one branch is exactly the confusion worktrees exist to avoid.

## 10. Conflicts without panic

Merge conflicts trigger more flailing than any other Git situation, and the flailing comes from not understanding what a conflict *is*. A conflict is not Git breaking. It is Git refusing to guess.

![Anatomy of a merge conflict: the markers split the hunk into ours and theirs](/imgs/blogs/using-git-like-senior-workflow-troubleshooting-playbook-9.webp)

The figure decodes the markers, which is most of the battle. When two branches changed the same lines, Git can't know which version you want, so it writes both into the file separated by markers and asks you to choose. Everything between `<<<<<<< HEAD` and `=======` is **ours** — the lines from the branch you are currently on. Everything between `=======` and `>>>>>>> feature` is **theirs** — the incoming branch. There is no "right" side; you decide. You keep ours, keep theirs, or blend the two — and then you delete all three marker lines, because the markers are not magic syntax, they are just text Git inserted that you now remove.

The resolution is mechanical once you see the structure:

```bash
git rebase origin/main
# CONFLICT (content): Merge conflict in src/config.js

# 1. see what's conflicted
git status                       # "both modified: src/config.js"

# 2. open the file, find each <<<<<<< ======= >>>>>>> block,
#    edit to the result you want, delete all three marker lines

# 3. mark it resolved and continue
git add src/config.js
git rebase --continue            # or `git merge --continue` if you were merging

# escape hatch: abandon the whole operation and go back to before
git rebase --abort
```

A few senior moves turn conflict resolution from dread into routine:

- **`git checkout --ours` / `--theirs`** for whole-file decisions. If you know one entire side is correct, `git checkout --theirs path/to/file` takes the incoming version wholesale, then `git add` it. (During a *rebase*, "ours" and "theirs" are swapped relative to intuition, because rebase replays your commits onto their base — when in doubt, open the file and read it rather than trusting the flag.)
- **`git rerere`** — "reuse recorded resolution." Turn it on and Git remembers how you resolved a given conflict, then replays that resolution automatically the next time the same conflict appears. This is the cure for case study 9, the conflict you resolve over and over during a long rebase:

```bash
git config --global rerere.enabled true
```

- **A three-way merge tool** for the genuinely hard ones, so you can see the common ancestor alongside both sides:

```bash
git config --global merge.conflictstyle zdiff3   # show the base, not just the two sides
git mergetool                                     # launch your configured visual tool
```

The `zdiff3` conflict style is an underused upgrade: it adds the **common ancestor** between the markers, so you can see what *both* sides started from and reason about intent instead of guessing. Most conflicts are trivial once you can see the base.

### 10.1 Second-order: conflicts are a signal, not just a chore

Frequent, painful conflicts are telling you something about your workflow, not just your luck. They usually mean branches are living too long (section 4) or two people are editing the same code without coordinating. The fix is upstream of Git: smaller branches, more frequent integration, and clearer ownership. A team that integrates daily rarely sees a conflict bigger than a few lines.

## 11. The senior `.gitconfig`

A surprising fraction of the problems in this article are prevented by configuration, not vigilance. Below is a `~/.gitconfig` worth adopting wholesale; each line is a default that removes a class of mistake or a daily friction.

```ini
[init]
    defaultBranch = main

[pull]
    rebase = true            # never create surprise merge commits on pull

[rebase]
    autoStash = true         # pull/rebase even with a dirty working tree
    autosquash = true        # fixup! commits auto-position in interactive rebase
    updateRefs = true        # rebasing a stack updates all the branch pointers

[merge]
    conflictstyle = zdiff3   # show the common ancestor in conflict markers
    ff = only                # refuse accidental merge commits; force a conscious choice

[rerere]
    enabled = true           # remember and replay conflict resolutions

[push]
    default = current        # `git push` pushes the current branch to its match
    autoSetupRemote = true   # first push needs no `-u`; upstream is set automatically

[fetch]
    prune = true             # delete local refs for branches deleted on the remote
    fsckObjects = true       # verify object integrity on fetch (catch corruption early)

[transfer]
    fsckObjects = true

[diff]
    algorithm = histogram    # smarter, more readable diffs than the default
    colorMoved = zebra       # highlight moved (not changed) lines distinctly

[column]
    ui = auto                # multi-column `git branch` / `git status` output

[branch]
    sort = -committerdate    # most recently used branches first in `git branch`

[help]
    autocorrect = prompt     # "git stauts" -> "did you mean status?"

[alias]
    s    = status -sb
    co   = checkout
    sw   = switch
    rs   = restore
    lg   = log --oneline --graph --decorate
    last = log -1 HEAD --stat
    unstage = restore --staged
    pushf = push --force-with-lease     # the only force-push you should ever type
    amend = commit --amend --no-edit
    fixup = commit --fixup
    please = push --force-with-lease     # because being polite to the remote pays off
```

A few of these deserve a sentence each, because they are the high-leverage ones:

- **`push.autoSetupRemote = true`** means you can `git push` a brand-new branch with no `-u` dance; Git creates the upstream for you. Small, but you do it ten times a day.
- **`rebase.updateRefs = true`** is a gift for anyone who stacks PRs: rebasing the bottom of a stack moves all the branch pointers above it in one shot.
- **`fetch.fsckObjects` / `transfer.fsckObjects`** verify object integrity on the way in, so a corrupt object or a malicious pack is caught at fetch time rather than discovered months later.
- **`merge.ff = only`** turns "oops, I made a merge commit when I meant to fast-forward" into a hard error you have to consciously override, which is exactly when you want friction.

### 11.1 `.gitignore` and the secrets discipline

The other half of configuration hygiene is keeping things *out* of history. A leaked secret (case study 11) is the most expensive Git mistake there is, because once a credential is in a pushed commit it is compromised forever, even if you delete it later. Prevention is cheap; cleanup is not.

```bash
# a sane baseline .gitignore for a Python service
cat > .gitignore <<'EOF'
.env
.env.*
*.pem
*.key
__pycache__/
.venv/
*.log
.DS_Store
EOF

# belt-and-suspenders: a pre-commit hook that blocks obvious secrets
pip install pre-commit detect-secrets
cat > .pre-commit-config.yaml <<'EOF'
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.5.0
    hooks:
      - id: detect-secrets
EOF
pre-commit install
```

The hook runs before every commit and refuses to let a high-entropy string that looks like an API key into history in the first place. That is the right place to stop a leak — at the working-directory boundary in the figure from section 3 — long before it reaches the remote and becomes everyone's problem.

## 12. Case studies: the cases you'll actually hit

The following twelve incidents are the ones that come up on real teams, in roughly descending order of frequency. Each follows the same shape: the symptom, the wrong first instinct, the actual fix, and the lesson. They are deliberately concrete — real commands, real flags — because in the moment you don't want principles, you want the three lines that get you out.

### 1. I committed straight to `main`

**Symptom.** You meant to branch first, but you typed `git commit` while on `main`. Now `main` has a commit that should have been on a feature branch.

**Wrong instinct.** Push it and hope the reviewer doesn't mind, or `git reset --hard` and lose the work.

**The fix.** Move the commit(s) to a new branch, then rewind `main`:

```bash
git switch -c feat/the-work       # new branch points at your commits
git switch main
git reset --hard origin/main      # rewind local main to match the remote
```

Because you branched *before* resetting, `feat/the-work` keeps the commits while `main` returns to the remote's state. Nothing is lost. If you had already pushed to `main` and it's protected, you can't — which is the lesson.

**Lesson.** Turn on branch protection (section 4.1) so committing to `main` is impossible on the server, and set a prompt in your shell that shows the current branch so you always know where you are.

### 2. I committed to the wrong branch

**Symptom.** You did good work, committed it, then realized you were on `feat/old-thing` instead of `feat/new-thing` (or on a teammate's branch).

**Wrong instinct.** Manually copy-paste the file changes between branches.

**The fix.** Cherry-pick the commit onto the right branch, then drop it from the wrong one:

```bash
git log --oneline -1              # note the SHA of the commit, e.g. a1b2c3d
git switch feat/new-thing
git cherry-pick a1b2c3d           # copy the commit here
git switch feat/old-thing
git reset --hard HEAD~1           # remove it from here (if not pushed)
```

If you committed several, `git cherry-pick a1b2c3d..f4e5d6c` copies a range. If the wrong branch was already pushed and shared, don't reset it — `git revert` the commit there instead.

**Lesson.** `git cherry-pick` is the precise tool for "this commit belongs over there." Reach for it instead of manual file shuffling.

### 3. Thirty "wip" commits the night before review

**Symptom.** Your branch works, but its history is "wip", "wip2", "fix", "actually fix", "address review", "typo" — twenty-plus commits a reviewer would hate.

**Wrong instinct.** Open the PR anyway and apologize in the description.

**The fix.** Interactive rebase to collapse the noise into a few atomic commits:

```bash
git rebase -i origin/main
# editor opens with one line per commit; change `pick` to:
#   squash (s) - combine into the previous commit, keep both messages
#   fixup  (f) - combine into previous, DISCARD this message
#   reword (r) - keep the commit, rewrite its message
#   drop   (d) - delete the commit entirely
# reorder lines to reorder commits
git push --force-with-lease
```

A typical cleanup squashes twenty "wip" commits into three or four that each tell one part of the story. The reviewer sees a coherent narrative, not your keystrokes.

**Lesson.** Commit messily while you work; rewrite clean before review (section 5.2). Adopt the `--fixup` + `autosquash` flow so the cleanup is automatic next time.

### 4. I need to undo a commit everyone already pulled

**Symptom.** A bad commit is on `main`, it's been pushed, and three teammates have already pulled it.

**Wrong instinct.** `git reset --hard HEAD~1 && git push --force`. This rewrites shared history and breaks everyone who pulled — the original bug is now compounded by a history mismatch.

**The fix.** Revert, which adds a new commit that undoes the bad one without rewriting anything:

```bash
git revert <bad-sha>              # creates "Revert: <original subject>"
git push                          # a normal push; no force needed
```

Everyone pulls the revert like any other commit; nobody's history breaks. If the bad change was a merged PR, revert the merge with `git revert -m 1 <merge-sha>` (section 7.1).

**Lesson.** This is the bottom branch of the decision tree (section 7). Shared history is append-only. You undo by adding, never by removing.

### 5. A force-push ate a teammate's commits

**Symptom.** You force-pushed your rebased branch; a teammate had pushed a commit to the same branch that you never fetched; your force-push overwrote it. Their commit is "gone" from the remote.

**Wrong instinct.** Panic, or assume the commit is unrecoverable.

**The fix.** Their commit is in *their* local reflog. They recover it and re-push:

```bash
# on the teammate's machine:
git reflog                        # find the SHA of their lost commit
git switch -c rescue <sha>        # rescue it onto a branch
git switch the-branch
git cherry-pick <sha>             # reapply it
git push                          # (after re-syncing the branch)
```

If the teammate's clone is gone too, the commit may still be on the remote's reflog or in a CI checkout — but that's the hard path.

**Lesson.** This is the case `--force-with-lease` prevents entirely (section 8.1): it would have refused the push because the remote had a commit you didn't have. Never type `git push --force`. Alias `pushf` to `--force-with-lease` and use it exclusively.

### 6. I deleted a branch that wasn't merged

**Symptom.** `git branch -D feat/important` (capital `-D` forces deletion even if unmerged), and now the branch and its commits appear gone.

**Wrong instinct.** Redo the work from memory.

**The fix.** The branch tip is still in the reflog:

```bash
git reflog                        # look for the last commit on the deleted branch
# ... or, for commits with no ref pointing at them:
git fsck --no-reflogs --lost-found   # lists dangling commits
git switch -c feat/important <sha>   # resurrect the branch at its old tip
```

Git keeps unreachable commits for about ninety days before garbage collection, so a deleted branch is almost always recoverable the same day.

**Lesson.** `git reflog` and `git fsck --lost-found` are the recovery pair for "the ref is gone." Deletion in Git rarely means destruction.

### 7. `git reset --hard` nuked uncommitted work

**Symptom.** You ran `git reset --hard` to undo something and it also obliterated two hours of uncommitted edits in your working directory.

**Wrong instinct.** Assume it's gone (uncommitted work isn't in the reflog) and start over.

**The fix.** If the work was ever *staged*, Git wrote it as a blob, and it may be recoverable from dangling objects:

```bash
git fsck --lost-found             # dangling blobs land in .git/lost-found/other/
# inspect candidates:
git show <blob-sha>               # is this my lost file?
```

This is genuinely harder than a reflog recovery and often fails for work that was only edited, never staged.

**Lesson.** This is the one truly dangerous corner. The habit that prevents it: `git stash` before any `reset --hard`. Stashing is free, and a `git stash` you didn't need is one `git stash drop` away. Never `--hard` over uncommitted work you care about.

### 8. `git pull` created a surprise merge commit

**Symptom.** You ran `git pull` and your history now has a "Merge branch 'main' of github.com:org/repo" commit you didn't intend, and the graph branched and rejoined.

**Wrong instinct.** Leave it; it's "just a merge commit." Over a team and a year, those accrete into unreadable history.

**The fix.** Undo this one and change the default so it never recurs:

```bash
git reset --hard HEAD~1           # if the merge was the last thing and nothing else came in
# then make pull rebase by default:
git config --global pull.rebase true
```

With `pull.rebase true`, `git pull` replays your local commits on top of the incoming ones — no bubble.

**Lesson.** The default behavior of `git pull` is a merge, and it's the wrong default for trunk-based teams. Set `pull.rebase true` globally (section 11) and the surprise merge commit disappears for good.

### 9. Rebase makes me resolve the same conflict on every commit

**Symptom.** You're rebasing a ten-commit branch and the *same* conflict in the same file shows up at commit after commit, and you resolve it identically each time.

**Wrong instinct.** Grind through all ten by hand, cursing.

**The fix.** Turn on `rerere` so Git records your first resolution and replays it automatically:

```bash
git config --global rerere.enabled true
git rebase --abort                # restart so rerere can record from the top
git rebase origin/main            # resolve once; rerere replays it on later commits
```

If the branch is long and the conflicts are really one logical change repeated, consider squashing the commits *first* (`git reset --soft origin/main && git commit`) so there's a single commit to rebase and a single conflict to resolve.

**Lesson.** `rerere.enabled true` belongs in everyone's global config (section 11). Repeated conflicts during rebase are a solved problem; you just have to opt in.

### 10. The PR won't merge: "this branch is behind main"

**Symptom.** Your PR is approved and green, but the merge button is disabled with "this branch is out of date" or "behind the base branch."

**Wrong instinct.** Click "Update branch," which merges `main` into your PR and adds a bubble, or merge `main` in locally for the same effect.

**The fix.** Rebase your branch onto the latest `main` and force-push:

```bash
git fetch origin
git rebase origin/main            # replay your commits on the new main
# resolve any conflicts, then:
git push --force-with-lease
```

The PR updates, the "behind" warning clears, and the history stays linear. If your platform requires branches to be *exactly* up to date before merge (a "require branches up to date" rule), this is the routine you run before every merge.

**Lesson.** "Behind main" is not an error; it's a request to integrate. Rebase, don't merge, to clear it — and keep the history clean in the process.

### 11. I committed a secret (`.env` / API key)

**Symptom.** You pushed a commit that contains an API key, a password, or a `.env` file.

**Wrong instinct.** Delete the file in a new commit and move on. The secret is still in history, reachable by anyone with the repo, forever.

**The fix.** Two steps, in this order. First — **rotate the credential immediately.** Assume it is compromised the moment it was pushed; deleting it from history does not un-leak it. Second, scrub it from history and force-push:

```bash
# 1. ROTATE the leaked credential first — this is the real fix.

# 2. then remove it from all of history (git-filter-repo is the modern tool)
pip install git-filter-repo
git filter-repo --path .env --invert-paths     # drop the file from every commit
# or to scrub a specific string everywhere:
git filter-repo --replace-text <(echo 'AKIA...EXAMPLE==>REDACTED')

git push --force --all            # rewrite the remote (coordinate with the team!)
git push --force --tags
```

Because `filter-repo` rewrites every commit SHA, this is the rare justified force-push — and it requires telling the whole team to re-clone, because their old history no longer matches.

**Lesson.** Rotation is the fix; history rewriting is cleanup. Prevent recurrence with a `detect-secrets` pre-commit hook and a `.gitignore` that excludes `.env` and key files (section 11.1).

### 12. Detached HEAD — "my commits vanished"

**Symptom.** You ran `git checkout <some-sha>` (or checked out a tag) to look at old code, made a few commits, and `git status` says "HEAD detached." When you switch back to `main`, your commits are nowhere.

**Wrong instinct.** Assume the commits are lost.

**The fix.** A detached HEAD means you committed without a branch pointing at those commits — they exist, but nothing references them. Give them a branch:

```bash
# while still detached, before switching away:
git switch -c feat/rescue         # creates a branch at your current commits
# already switched away? find them in the reflog:
git reflog                        # locate the detached commits' SHA
git switch -c feat/rescue <sha>
```

**Lesson.** "Detached HEAD" is not an error state; it's "you're not on a branch." Commits made there are real and reflog-recoverable. The instant you want to keep work in a detached state, `git switch -c` to anchor it to a branch.

### 13. A stash `pop` conflicted and now I think I lost it

**Symptom.** You ran `git stash pop`, it hit a conflict, you panicked and ran `git checkout .` or `git reset --hard` to "clean up," and now both the conflict and your stashed changes seem gone.

**Wrong instinct.** Assume the stash is destroyed.

**The fix.** A conflicted `pop` does **not** drop the stash, so it's still on the stack — or, if you somehow dropped it, it's a dangling commit:

```bash
git stash list                    # is it still there? usually yes:
git stash apply stash@{0}         # reapply, resolve the conflict properly this time

# if the stash entry is gone, recover the dangling commit:
git fsck --no-reflog | awk '/dangling commit/ {print $3}'
git stash apply <dangling-sha>    # stashes are commits; apply by SHA
```

**Lesson.** `git stash pop` keeps the stash when it conflicts (section 9.1) — that's a feature. Resolve, `git add`, then `git stash drop` deliberately. And don't `reset --hard` as a panic response; it's almost never the right reflex.

### 14. Cloning the repo takes ten minutes because someone committed a 2 GB dataset

**Symptom.** A fresh clone is multiple gigabytes and slow, even though the working tree is small. Someone committed a large binary, model checkpoint, or dataset months ago; deleting it later didn't help because every clone still carries every version of it in history.

**Wrong instinct.** Delete the file in a new commit and wonder why clones are still huge.

**The fix.** Two parts. Purge the blob from all of history, and move large assets to Git LFS so it never recurs:

```bash
# find the biggest objects in history
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | sort -k3 -n | tail -20

# purge the offending path from every commit
git filter-repo --path data/big.parquet --invert-paths
git push --force --all            # coordinate: everyone must re-clone

# prevent recurrence: track large file types with LFS
git lfs install
git lfs track "*.parquet" "*.ckpt" "*.bin"
git add .gitattributes
```

**Lesson.** History is forever and so is its size; a giant blob committed once bloats every future clone. Set up Git LFS (or a `.gitignore` + external storage) *before* anyone commits large files, and audit object sizes when clones start feeling slow.

### 15. I need to split one fat commit into two during review

**Symptom.** A reviewer points out that one of your commits does two unrelated things — a bug fix and a refactor — and asks you to separate them.

**Wrong instinct.** Leave a "split as requested" follow-up commit, which doesn't actually split anything.

**The fix.** Interactive rebase, mark the commit `edit`, reset it, and recommit in pieces:

```bash
git rebase -i origin/main
# change `pick` to `edit` on the fat commit, save and close

git reset HEAD~                   # uncommit it; changes return to the working dir
git add -p                        # stage only the bug-fix hunks (section 5.1)
git commit -m "fix: handle null user in fetchUser"
git add -A
git commit -m "refactor: extract userClient construction"
git rebase --continue             # finish replaying the rest of the branch
git push --force-with-lease
```

**Lesson.** `git rebase -i` with `edit` plus `git add -p` is the surgical kit for reshaping history into atomic commits. Splitting a commit is routine once you've done it twice — and it's far better than a history that pretends to be clean.

## 13. When to reach for which tool, and when not to

The hardest part of Git judgment is not knowing the commands; it's knowing which one fits the situation and which moves are traps. Here is the senior decision summary.

**Reach for rebase when:**

- You're updating your own private or unshared branch with the latest `main`.
- You're cleaning up your branch's history before opening a PR (squash, reorder, reword).
- You want a linear history that `bisect` and `revert` can navigate cleanly.
- You're integrating a small, short-lived feature branch.

**Reach for merge when:**

- You're combining two long-lived branches whose divergence is real, meaningful history.
- The branch you'd be rebasing has been shared and others have based work on it.
- You're doing the final integration of a release line where the merge commit documents a real event.

**Reach for revert when:**

- The commit you want to undo has been pushed and others may have pulled it.
- You need to undo something on `main` or any protected/shared branch.
- You want an auditable record that the change was made *and then deliberately undone*.

**Reach for reset when:**

- The commits are private (not pushed, or pushed only to your own unshared branch).
- You want to uncommit while keeping the changes (`--soft` / `--mixed`).
- You're squashing local commits before review.

**Now the anti-patterns — things that feel productive and aren't:**

- **`git push --force` (without `-with-lease`)** on any branch. It's how you erase a teammate's work. There is no situation where `--force` is correct that `--force-with-lease` doesn't also handle more safely.
- **Rebasing a shared branch.** Rewriting SHAs that others have based work on turns their next pull into a conflict storm. The boundary from section 2 is absolute: rewrite before sharing, never after.
- **`git reset --hard` over uncommitted work** without stashing first. The one Git command that destroys work with no undo. Stash first, always.
- **Long-lived feature branches.** Every day a branch lives, it diverges further and its eventual merge gets more painful. Shrink the loop (section 4).
- **`git pull` with the default merge behavior** on a trunk-based team. It litters history with unintended merge bubbles. Set `pull.rebase true`.
- **"Address review" commits.** They turn a clean PR into noise. Use `--fixup` + `autosquash` so review rounds collapse into the commits they fix.
- **Committing generated files, secrets, or huge binaries.** Each is a different flavor of "history you'll wish you could rewrite." Prevent with `.gitignore`, hooks, and Git LFS for large assets.

The thread running through all of it is the same: **make irreversible mistakes impossible by construction.** Branch protection makes committing to `main` impossible. `--force-with-lease` makes eating a teammate's commits impossible. Committing often makes losing work nearly impossible. `pull.rebase true` makes surprise merges impossible. Seniority is not a faster reaction to disasters; it is an arrangement of habits and config where most disasters can't occur in the first place.

## 14. Further reading

- [Git Like a Pro: the object model, daily workflows, and recovering from disaster](/blog/software-development/version-control/git-like-a-pro-object-model-workflows-and-recovery) — the internals companion to this article: the content-addressed object store, the three trees, signed commits, performance at scale, and the object-level mechanics that make every recovery in this playbook work.
- [Design patterns guide](/blog/software-development/system-design/design-patterns-guide) — the same discipline that keeps a commit history clean keeps a codebase clean; the parallels between linear history and well-factored code run deep.
- **Pro Git** (Chacon & Straub), the free official book — chapters 2, 3, and 7 cover branching, rebasing, and the tools (`bisect`, `rerere`, `filter-branch`) referenced here in full depth.
- `git help <command>` and `git help -g` — the built-in manuals are excellent; `git help everyday` in particular is a curated set of the workflows a working engineer actually needs.

If you take one thing from this article, make it the decision tree in section 7: before you touch any undo command, ask *is it committed, is it pushed, is it shared?* That single question, asked every time, is most of what separates a senior's Git from everyone else's.
