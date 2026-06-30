---
title: "Database Disaster Recovery at Scale: RPO, RTO, and the Restore You Never Tested"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "A principal-engineer field guide to database disaster recovery: setting RPO and RTO targets, what replication does not protect against, point-in-time recovery, tested restore drills, and the DR tiers that map recovery objectives to cost."
tags: ["disaster-recovery", "database-scaling", "rpo-rto", "point-in-time-recovery", "backups", "postgres", "pgbackrest", "wal-archiving", "failover", "reliability"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 38
---

The worst outage of my career did not lose a single byte to a hardware fault. The disks were fine. The replicas were healthy. Replication was streaming with sub-second lag to three nodes across two regions. And we still spent a Saturday afternoon staring at an empty `users` table, because at 14:32:07 a maintenance job had run `DELETE FROM users` against the wrong database, and the system we had spent two years hardening for availability did exactly what it was built to do: it copied that delete, faithfully and instantly, to every replica we owned.

That is the thesis of this entire post, and it is uncomfortable: **the machinery that keeps you up — replication, failover, multi-region — is not the machinery that gets you back. A replica is a copy of your mistakes. Only a backup you have actually restored is an undo button, and a backup you have never restored is not a backup at all — it is a hope with a cron schedule.** Disaster recovery at scale is a measured discipline, not a checkbox. It has two numbers that define it, a small set of mechanisms that implement it, and exactly one practice — the tested restore — that separates the teams who recover in an hour from the teams who make the news.

![A recovery timeline: RPO is the data window lost to the left of the incident, RTO is the downtime window burned to the right of it](/imgs/blogs/database-disaster-recovery-at-scale-1.webp)

The diagram above is the mental model for everything that follows. A disaster lands at some instant — here, the bad `DELETE` at 14:32:07. Everything to the *left* of that instant is about **data**: how far back is your last good recovery point, and therefore how much committed work you lose when you roll back to it? That window is your **RPO**, the Recovery Point Objective. Everything to the *right* is about **time**: how long from "we noticed" to "service is verified and serving again"? That window is your **RTO**, the Recovery Time Objective. The rest of this article is a tour of how to shrink each window deliberately, what each mechanism actually protects against, and — the part most teams skip until it is too late — how to prove the whole chain works before the day you need it.

## 1. RPO and RTO: the two clocks that define disaster recovery

**Senior rule of thumb: you do not have "a backup strategy." You have an RPO and an RTO per class of data, and everything else is an implementation detail in service of those two numbers.**

RPO answers *how much data can I afford to lose?* measured in time. An RPO of five minutes means that after the worst day, you are willing to have lost up to the last five minutes of writes. RTO answers *how long can I be down?* An RTO of thirty minutes means thirty minutes from incident declared to service verified. They are independent: you can have a tiny RPO and a huge RTO (you lost nothing, but the restore takes six hours), or a tiny RTO and a large RPO (you failed over in ten seconds, but to a replica that was an hour behind).

The mistake juniors make is to pick one global pair of numbers for the whole company. The mistake seniors avoid is to forget that **different data deserves different targets, and the cost of a recovery objective is roughly exponential as it approaches zero.** A payments ledger and a session cache are not the same problem and must not get the same budget.

| Data class | Example | RPO target | RTO target | Backup approach |
| --- | --- | --- | --- | --- |
| Financial ledger / money movement | double-entry ledger, balances | ~0 (no committed write lost) | seconds to minutes | sync commit + PITR + active-active |
| Orders / user-generated content | orders, posts, comments, issues | seconds to a few minutes | minutes to ~1 hour | async replica + PITR |
| Derived / session state | sessions, feature toggles, search index | minutes to hours, or rebuildable | rebuild from source | snapshot, or regenerate |
| Analytics / warehouse | event logs, rollups, dashboards | hours | hours to a day | nightly snapshot, reload from source |
| Cache | Redis/Memcached hot keys | not applicable | re-warm on cold start | **do not back it up** |

Two rows in that table matter more than the rest. The ledger row, where RPO must be zero, forces synchronous commit and dictates almost every expensive decision you will make. And the cache row, where the right answer is *do not back it up at all* — a cache is derived state; backing it up buys you nothing except the false comfort of restoring stale keys you should have let cold-start regenerate. Knowing which data is which is the first act of disaster-recovery engineering. Spend your zero-RPO budget on the ledger and your zero-dollar budget on the cache.

> A recovery objective you have not priced is a wish. A recovery objective you have not tested is a lie. The job is to turn both into numbers you can defend in a postmortem.

The second-order consequence is organizational: RPO and RTO are *product* decisions wearing an engineering costume. The business owns "how much money does an hour of downtime cost, and how much is a lost order worth?" Engineering owns "here is what RPO=0 costs versus RPO=5min." When those two conversations never meet, you get a ledger backed up nightly (catastrophic) or a click-tracking table replicated synchronously across three regions (wasteful). Map every dataset to a tier on purpose.

## 2. What replication does not protect against

**Senior rule of thumb: replication is for availability, not for undo. It copies every byte you tell it to — including the bytes you did not mean to write.**

This is the single most expensive misunderstanding in operations, and it is worth being precise about *why* it happens. A streaming replica does not "decide" whether a change is good. The primary executes a statement, durably records the resulting change in its write-ahead log, and ships those log records downstream. To the replica, `DELETE FROM users` is not a catastrophe — it is a perfectly valid WAL record describing rows to remove, indistinguishable from any legitimate delete. The replica applies it because applying the log faithfully is the *entire point* of replication. If it second-guessed records, it would not be a replica; it would be a slower, divergent database.

![A bad DELETE executes on the primary, is logged to the WAL, and fans out to every replica including the cross-region one, leaving zero surviving copies](/imgs/blogs/database-disaster-recovery-at-scale-2.webp)

The figure traces the blast radius. The application issues `DELETE FROM users` with no `WHERE` clause. The primary executes it and writes the change to its WAL. Every replica — including the cross-region one you provisioned precisely for resilience — pulls that WAL and applies it. Within milliseconds, every copy of the table is empty. The very property that makes replication wonderful for surviving a *node* failure makes it useless against a *logical* failure, because it has no concept of "this change was a mistake."

Here is the same lesson at a SQL prompt. Run a destructive statement on the primary, then watch it arrive on the replica with no fault, no alarm, no chance to intervene:

```sql
-- On the PRIMARY (psql, port 5432)
SELECT count(*) FROM users;        -- 4_812_390
DELETE FROM users;                 -- oops: forgot the WHERE
DELETE 4812390
SELECT count(*) FROM users;        -- 0

-- On a streaming REPLICA (psql, port 5433), a few hundred ms later
SELECT count(*) FROM users;        -- 0  -- faithfully replicated; nothing to "fail over" to
SELECT pg_last_wal_replay_lsn();   -- caught up; the replica did its job perfectly
```

The category of failures replication will happily propagate is broader than fat fingers:

- **A fat-finger DML** — `DELETE`/`UPDATE`/`TRUNCATE` without the predicate you intended, or against the wrong database.
- **A bad migration** — a deploy that drops a column, rewrites a table with the wrong default, or runs a backfill that corrupts half the rows. Replication copies the migration's effects perfectly. (This is exactly why [zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations) are reversible by construction — expand/contract, never an irreversible drop in one shot.)
- **Application-level logical corruption** — a bug that writes wrong-but-valid data: double-charged invoices, mis-linked foreign keys, a serialization bug that mangles JSON. The database sees valid writes; the replica copies valid garbage.
- **A malicious or compromised credential** — anyone or anything with write access can issue valid-looking destructive statements, and a delete from an attacker looks identical to a delete from a buggy job.

None of these is what failover is for. Failover answers "the primary's host died; promote a replica." It does nothing for "the data itself is wrong on every node." To undo a logical disaster you need a copy of the past that the bad write *did not reach* — a backup, plus the ability to roll forward to the instant *before* the mistake. That is point-in-time recovery, and it is the heart of this post.

> Replication multiplies your good writes and your bad writes with equal enthusiasm. Backups are the only mechanism that remembers a version of reality your mistake never touched.

## 3. The backup zoo: logical, physical, and the WAL archive

**Senior rule of thumb: a backup captures one instant; the WAL archive captures the gaps between instants. You need both, or you can only ever recover to whenever the last snapshot happened to run.**

Backups come in two families, and conflating them is how teams end up with a tool that cannot do the job they assumed it could.

![A taxonomy tree: database backups split into logical, physical, and the continuous WAL/binlog archive, each with its tradeoffs](/imgs/blogs/database-disaster-recovery-at-scale-3.webp)

**Logical backups** (`pg_dump`, `mysqldump`) capture the *logical contents* — the SQL statements and data needed to recreate the database from scratch. They are gloriously portable: a `pg_dump` from Postgres 14 can usually restore into Postgres 16, on a different OS and CPU architecture, because it is just SQL and `COPY` data. That portability is also their curse. Restoring a logical dump means *replaying* every insert and *rebuilding* every index from zero, which on a large database is agonizingly slow — a multi-terabyte cluster can take a day to restore from a dump. Logical backups are the right tool for small databases, for migrating between versions, and for exporting a single table; they are the wrong tool for recovering a large production cluster under an RTO clock.

**Physical backups** (`pg_basebackup`, filesystem/EBS snapshots, Percona XtraBackup) capture the *bytes on disk* — the actual data files. Restoring is fast because there is no replay: you copy the files into place and start the engine. The cost is rigidity. A physical backup is tied to the major version, page format, and often the architecture of the database that produced it; you cannot restore a Postgres 14 basebackup into Postgres 16. For large databases under a real RTO, physical is almost always the answer, and it is the only family that supports incremental backups and point-in-time recovery.

The third thing in the tree is not a backup at all in the snapshot sense — it is the **WAL archive** (Postgres) or **binlog archive** (MySQL). A base backup is a photograph of the database at the instant it completed. The WAL archive is the *continuous recording of every change since*. Stream those WAL segments to durable storage as they are produced, and you can take a base backup once a day yet still recover to *any second* in between by restoring the base and replaying the WAL up to your chosen stopping point. Without the archive, your only recovery points are the discrete instants your snapshots ran — lose-up-to-24-hours territory. With it, your RPO collapses to the lag of the archive, often a few seconds. The WAL is also what makes a database durable in the first place; if the mechanism is unfamiliar, the [write-ahead log deep dive](/blog/software-development/database/write-ahead-log-how-databases-guarantee-durability) is the prerequisite.

| Property | Logical (`pg_dump`) | Physical (basebackup / snapshot) |
| --- | --- | --- |
| Captures | SQL + data to recreate | byte-level data files |
| Portable across major versions / arch | yes | no (same major version, page format) |
| Restore speed | slow (replays SQL, rebuilds indexes) | fast (copy files, start engine) |
| Incremental backups | no (practically full each time) | yes (block/delta) |
| Enables point-in-time recovery | no | yes, paired with WAL archive |
| Best for | small DBs, version migrations, single tables | large DBs, full-cluster DR |

The full-versus-incremental axis lives inside the physical family. A **full** backup copies every data block. An **incremental** copies only the blocks that changed since the last backup (pgBackRest does this at the block level; WAL-G computes deltas). Incrementals make daily backups of a large database cheap enough to actually keep frequently, which in turn keeps your base-backup-to-now WAL replay short — and a shorter replay is a shorter RTO. The classic schedule is a weekly full plus daily (or hourly) incrementals, with WAL archived continuously on top.

## 4. Point-in-time recovery: the single most valuable DR capability

**Senior rule of thumb: if you can only build one recovery capability, build PITR. It is the only thing that lets you say "restore everything except the mistake" instead of "restore to last night and lose a day."**

Point-in-time recovery combines the two pieces from the previous section: a physical base backup as a starting point, and the continuous WAL archive as the roll-forward tape. To recover, you restore the most recent base backup taken *before* your target time, then replay archived WAL **up to a specified stopping point** — the `recovery_target_time` — and stop. Everything committed before that instant is present; everything after it, including the disaster, never gets applied.

<figure class="blog-anim">
<svg viewBox="0 0 820 280" role="img" aria-label="Point-in-time recovery replays WAL segments from a base backup and stops at the recovery target time, just before the bad DELETE" style="width:100%;height:auto;max-width:820px">
<title>WAL replay sweeps from the base backup up to recovery_target_time and stops before the bad transaction</title>
<style>
.pitr-cell{fill:var(--surface,#f3f4f6);stroke:var(--border,#cbd5e1);stroke-width:1.5}
.pitr-base{fill:var(--surface,#f3f4f6);stroke:var(--border,#cbd5e1);stroke-width:2.5}
.pitr-bad{fill:#ffc9c9;stroke:#e03131;stroke-width:2}
.pitr-fill{fill:#2f9e44;opacity:.26;transform-box:fill-box;transform-origin:left center}
.pitr-head{fill:#2f9e44}
.pitr-t{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.pitr-s{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.pitr-bd{font:700 13px ui-sans-serif,system-ui;fill:#e03131;text-anchor:middle}
.pitr-mk{stroke:#e03131;stroke-width:2;stroke-dasharray:5 4}
@keyframes pitr-grow{0%{transform:scaleX(0)}58%{transform:scaleX(1)}85%{transform:scaleX(1)}100%{transform:scaleX(0)}}
@keyframes pitr-move{0%{transform:translateX(0)}58%{transform:translateX(408px)}85%{transform:translateX(408px)}100%{transform:translateX(0)}}
.pitr-fill{animation:pitr-grow 9s ease-in-out infinite}
.pitr-head{animation:pitr-move 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.pitr-fill{animation:none;transform:scaleX(1)}.pitr-head{animation:none;transform:translateX(408px)}}
</style>
<text class="pitr-t" x="410" y="30">Point-in-time recovery: replay WAL up to the target, then stop</text>
<rect class="pitr-base" x="20" y="90" width="100" height="78" rx="8"/>
<rect class="pitr-cell" x="140" y="90" width="72" height="78" rx="6"/>
<rect class="pitr-cell" x="224" y="90" width="72" height="78" rx="6"/>
<rect class="pitr-cell" x="308" y="90" width="72" height="78" rx="6"/>
<rect class="pitr-cell" x="392" y="90" width="72" height="78" rx="6"/>
<rect class="pitr-cell" x="476" y="90" width="72" height="78" rx="6"/>
<rect class="pitr-bad" x="572" y="90" width="96" height="78" rx="6"/>
<rect class="pitr-cell" x="684" y="90" width="72" height="78" rx="6"/>
<rect class="pitr-fill" x="140" y="90" width="408" height="78" rx="6"/>
<rect class="pitr-head" x="136" y="84" width="5" height="90" rx="2"/>
<line class="pitr-mk" x1="560" y1="80" x2="560" y2="182"/>
<text class="pitr-t" x="70" y="134">base backup</text>
<text class="pitr-s" x="176" y="183">14:30</text>
<text class="pitr-s" x="260" y="183">14:31</text>
<text class="pitr-s" x="344" y="183">14:32:00</text>
<text class="pitr-s" x="428" y="183">14:32:05</text>
<text class="pitr-s" x="512" y="183">14:32:06</text>
<text class="pitr-bd" x="620" y="135">DELETE</text>
<text class="pitr-s" x="620" y="183">14:32:07</text>
<text class="pitr-s" x="720" y="183">14:33+</text>
<text class="pitr-t" x="560" y="72">target 14:32:06</text>
<rect class="pitr-fill" x="150" y="236" width="20" height="16" rx="3" style="opacity:.4"/>
<text class="pitr-s" x="262" y="249">replayed (recovered)</text>
<rect class="pitr-bad" x="430" y="236" width="20" height="16" rx="3"/>
<text class="pitr-s" x="560" y="249">bad txn - never replayed</text>
</svg>
<figcaption>The replay head advances from the base backup applying WAL segment by segment and halts at recovery_target_time = 14:32:06 (one second before the 14:32:07 DELETE), so the recovered database holds every committed write except the mistake.</figcaption>
</figure>

The animation is the whole idea in motion: the replay head sweeps forward from the base backup, applying each WAL segment, and stops at the recovery target — one tick before the bad delete — so the result is the entire database *minus* the disaster. Watch it stop short of the red segment. That gap between where it stops and where the mistake lives is the difference between "we lost an afternoon of writes" and "we lost nothing but the delete."

### Setting up continuous archiving (Postgres)

PITR requires two things configured *before* the disaster: WAL archiving turned on, and base backups taken regularly. On the primary's `postgresql.conf`:

```ini
# postgresql.conf — enable continuous WAL archiving
wal_level = replica            # 'replica' or 'logical'; enough for physical PITR
archive_mode = on
# Ship each completed WAL segment to durable storage. In production this is a
# backup tool (pgBackRest / WAL-G), NOT cp — the tool handles retries, parallelism,
# encryption, and S3. The cp form is shown only to make the contract obvious:
archive_command = 'test ! -f /archive/%f && cp %p /archive/%f'
archive_timeout = 60           # force a segment switch at least every 60s, capping RPO
```

`archive_command` runs once per completed 16 MB WAL segment; `%p` is the path to the segment and `%f` is its filename. The contract is brutal and important: if `archive_command` returns non-zero, Postgres keeps the segment and retries — so a broken archive command silently **stalls WAL recycling and fills your disk**, while a command that returns zero *without actually durably storing the file* silently destroys your ability to recover. (Both are real outages people have had. Test that the file truly lands in durable storage.) `archive_timeout = 60` bounds RPO from below: even an idle database flushes a segment every minute, so you never have more than ~60 seconds of un-archived WAL.

Take the base backup with `pg_basebackup` (or, better, your backup tool):

```bash
# A physical base backup, streaming WAL alongside so the backup is self-consistent
pg_basebackup \
  --host=primary.db.internal --username=replicator \
  --pgdata=/var/lib/postgresql/base-2026-06-30 \
  --format=plain --wal-method=stream --progress --checkpoint=fast
```

### Performing the recovery

Now the disaster has happened. You restore the base backup taken before 14:32:07 onto a fresh instance, point it at the WAL archive, and tell it where to stop. On Postgres 12+ the recovery settings live in `postgresql.conf` (or `postgresql.auto.conf`) and recovery is triggered by a `recovery.signal` file:

```ini
# postgresql.conf on the RESTORE instance
restore_command = 'cp /archive/%f %p'         # production: 'wal-g wal-fetch %f %p' etc.
recovery_target_time = '2026-06-30 14:32:06.5+00'
recovery_target_action = 'promote'            # stop recovery and come up read/write
recovery_target_inclusive = off               # stop BEFORE a txn committed at the target
```

```bash
# Trigger recovery and start the instance
touch /var/lib/postgresql/restore/recovery.signal
pg_ctl -D /var/lib/postgresql/restore -o "-p 5433" -w start

# Postgres logs the replay and where it stopped:
#   LOG:  starting point-in-time recovery to 2026-06-30 14:32:06.5+00
#   LOG:  recovery stopping before commit of transaction 90871, time 2026-06-30 14:32:07.13+00
#   LOG:  archive recovery complete
```

The most precise stopping point is not time at all — if you can read the offending transaction's id out of the logs, recover by **transaction id**, which removes the guesswork of sub-second timing:

```ini
recovery_target_xid = '90871'        # the exact bad transaction, from the logs
recovery_target_inclusive = off      # stop just before it commits
```

Postgres also supports `recovery_target_lsn` (stop at an exact WAL position) and `recovery_target_name` (stop at a named restore point you created with `pg_create_restore_point('before-migration')` right before a risky operation). That last one is a gift to your future self: create a named restore point immediately before every scary migration, and rollback becomes a one-line `recovery_target_name` instead of a frantic log-grep.

The MySQL story is the same shape with different nouns: a physical base backup via **Percona XtraBackup** (or a snapshot), plus the **binlog** as the continuous archive, replayed with `mysqlbinlog --stop-datetime` (or `--stop-position` for binlog-coordinate precision) up to the instant before the bad statement.

> The most valuable sentence in incident response is "we can restore to 14:32:06." It is only available to teams who turned on WAL archiving *before* 14:32:07.

## 5. Continuous archiving and the tools that do it

**Senior rule of thumb: do not hand-roll `archive_command` with `cp` and a cron `pg_basebackup`. Use a tool that does parallel, encrypted, incremental, retention-managed, verifiable backups — and copy them somewhere your production credentials cannot delete.**

The naive `cp`-to-a-local-directory archive is a teaching tool, not a production system. Real continuous archiving is layered, and each layer survives a failure the layer above cannot.

![A defense-in-depth stack: live database, WAL archive to object store, base backups, cross-region copy, and an immutable object-lock tier](/imgs/blogs/database-disaster-recovery-at-scale-5.webp)

Read the stack top to bottom as "what survives what." The **live database** survives nothing on its own — it is the thing failing. The **WAL archive to object storage** survives the loss of the database host: the segments are in S3, durable and continuous, seconds behind. **Base backups** survive a corrupted WAL stream: even if recent WAL is unusable, you can restore the last good full and lose only the delta. The **cross-region copy** survives the loss of an entire region — if `us-east-1` is gone, your backups are not. And the **immutable tier** — object-lock WORM, MFA-delete, ideally in a separate cloud account — survives the failure mode none of the others do: *someone with your credentials trying to delete your backups*, whether that is ransomware, a compromised key, or your own automation gone wrong. Code Spaces (we will get there) died precisely because their backups lived where their production credentials could reach them.

The tools that implement this, and what to reach for:

| Tool | Engine | Backup type | Incremental | Continuous WAL/binlog | PITR | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `pg_dump` / `pg_restore` | Postgres | logical | no | no | no | portable; small DBs and migrations only |
| **pgBackRest** | Postgres | physical | yes (block) | yes | yes | parallel, encrypted, S3/Azure/GCS, retention, `verify` |
| **WAL-G** | PG / MySQL | physical | yes (delta) | yes | yes | cloud-native, streams to object storage, fast restore |
| **Percona XtraBackup** | MySQL | physical | yes | with binlog | yes | hot InnoDB backup, no table locks |
| **RDS automated backups** | managed | physical + WAL | yes | yes | yes | one-click PITR to any second in the window |
| **Aurora** | managed | continuous to S3 | n/a | continuous | yes | backup is always-on, no performance hit |

A pgBackRest setup that does full + incremental + WAL archiving to S3 is genuinely a few lines:

```ini
# /etc/pgbackrest/pgbackrest.conf
[global]
repo1-type=s3
repo1-s3-bucket=acme-pg-backups
repo1-s3-region=us-east-1
repo1-path=/main
repo1-cipher-type=aes-256-cbc        # encryption at rest in the repo
repo1-retention-full=4               # keep 4 full backups; prune older automatically
process-max=8                        # parallel compression/transfer

[main]
pg1-path=/var/lib/postgresql/16/main
```

```bash
# One-time: create the repository for this database ("stanza")
pgbackrest --stanza=main stanza-create

# In postgresql.conf, archive WAL through pgBackRest:
#   archive_command = 'pgbackrest --stanza=main archive-push %p'

# Scheduled: weekly full, daily incremental (incremental is the default after a full)
pgbackrest --stanza=main --type=full backup       # Sundays
pgbackrest --stanza=main --type=incr backup       # daily

# Recover to a point in time — this is the payoff of the whole setup:
pgbackrest --stanza=main --type=time \
  --target="2026-06-30 14:32:06" --target-action=promote restore
```

For managed databases the mechanics are hidden but the *concepts are identical*, and you still must reason about them. RDS automated backups give you PITR to any second within the retention window via `restore-db-instance-to-point-in-time --restore-time` — but they create a **new instance**, so your RTO includes provisioning and cutover, and the retention window is whatever you set (default 7 days, max 35). Aurora continuously streams to S3 with effectively no backup-window performance penalty and restores to any point in the retention range. Critically, **managed backups in a single account are not safe from account-level disasters** — enable cross-region automated backup copies and, for anything that matters, replicate the backups into a separate account with restricted delete permissions. A managed database does not exempt you from the immutable-tier conversation; it just changes who runs the daemon. When the requirement is surviving a whole-region loss with low RPO, you have crossed into [multi-region database architecture](/blog/software-development/database-scaling/multi-region-database-architecture), where the backups are one half of the story and a standby in another region is the other.

## 6. The restore drill: an untested backup is Schrödinger's backup

**Senior rule of thumb: a backup is not "done" when the job exits zero. It is done when you have restored it onto a scratch instance, verified the data, run a smoke test, and timed the whole thing. Until then its state is "unknown," and unknown backups have a way of collapsing into "broken" the moment you observe them.**

This is the thesis of the whole article, so let me be blunt about it. The most common cause of a catastrophic data-loss incident is not the absence of backups. It is the presence of backups that *everyone believed worked* and that turned out, at the worst possible moment, to be empty, truncated, version-incompatible, encrypted with a lost key, or missing the WAL needed to make them consistent. A backup you have never restored is in superposition — simultaneously fine and broken — and the act of restoring it is the measurement that collapses the wave function. The teams who survive are the ones who do that measurement on a *schedule*, in a *drill*, not for the first time during the disaster.

![A scheduled restore-drill loop: fetch backup, restore to scratch, verify counts and checksums, smoke test, measure RTO, repeat weekly](/imgs/blogs/database-disaster-recovery-at-scale-7.webp)

The loop in the figure is the practice. Fetch the latest backup and WAL. Restore it onto an isolated scratch instance — isolated so a drill can never touch production. Verify the data with row counts and checksums against known-good values. Run an application smoke test that actually reads and writes. Measure the wall-clock from start to "verified," because *that number is your real RTO* — not the optimistic figure in the runbook, the measured one. Then schedule it to run again, because a backup that restored last week can break this week when someone bumps the Postgres version or rotates the encryption key. The loop never ends; that is the point.

Here is a restore-drill script you can adapt. It restores the latest backup, checks row counts and a content checksum against production, runs a smoke test, and — the line that matters — records the actual RTO and alerts when it drifts past target:

```bash
#!/usr/bin/env bash
set -euo pipefail
# nightly-restore-drill.sh — prove the backup restores, verify it, and measure RTO.
# Runs against an isolated scratch host. NEVER point this at production.

STANZA=main
SCRATCH=/var/lib/postgresql/scratch
SCRATCH_PORT=5433
PROD_DSN="host=primary.db.internal dbname=app user=verifier"
RTO_TARGET_S=1800        # 30 min; the number we are testing against
start=$(date +%s)

# 1. Restore the latest backup + WAL to an isolated scratch instance.
rm -rf "$SCRATCH" && mkdir -p "$SCRATCH"
pgbackrest --stanza="$STANZA" --pg1-path="$SCRATCH" restore
pg_ctl -D "$SCRATCH" -o "-p $SCRATCH_PORT" -w start
SCRATCH_DSN="host=localhost port=$SCRATCH_PORT dbname=app"
echo "restore wall-clock: $(( $(date +%s) - start ))s"

# 2. Verify: row counts per critical table must match prod within tolerance.
for tbl in users orders ledger_entries; do
  prod=$(psql "$PROD_DSN"    -tAc "SELECT count(*) FROM $tbl")
  rest=$(psql "$SCRATCH_DSN" -tAc "SELECT count(*) FROM $tbl")
  if [ "$rest" -lt "$(( prod * 99 / 100 ))" ]; then
    echo "FAIL row-count $tbl: scratch=$rest prod=$prod"; exit 1
  fi
  echo "OK $tbl: scratch=$rest prod=$prod"
done

# 3. Content checksum of a critical table (order-independent, catches silent corruption).
sum=$(psql "$SCRATCH_DSN" -tAc \
  "SELECT md5(string_agg(id::text||':'||amount::text, ',' ORDER BY id)) FROM ledger_entries")
echo "ledger checksum: $sum"

# 4. App smoke test: the service must read AND write against the restored copy.
DATABASE_URL="postgresql://localhost:$SCRATCH_PORT/app" ./smoke_test.sh

# 5. Record the ACTUAL RTO and alert on drift. This is the number that matters.
rto=$(( $(date +%s) - start ))
echo "actual RTO: ${rto}s (target ${RTO_TARGET_S}s)"
if [ "$rto" -gt "$RTO_TARGET_S" ]; then
  curl -fsS -X POST "$ALERT_WEBHOOK" -d "restore drill RTO ${rto}s exceeds target ${RTO_TARGET_S}s"
fi

pg_ctl -D "$SCRATCH" stop
```

A few non-obvious things this catches that a green backup job never would. It catches **version skew** (the restore engine refuses a backup from a newer major version). It catches a **WAL gap** (the restore cannot reach consistency because a segment never archived). It catches **silent content corruption** via the checksum, where the row count is right but the data is wrong. And it catches **RTO drift** — the slow creep where your database doubled in size and your restore quietly went from 20 minutes to 70, blowing an SLA nobody re-measured. Restore time grows with your data; if you do not measure it continuously, you will discover the new number during the incident.

> Run the drill often enough that a failed restore is a Tuesday, not a catastrophe. The goal is to make recovery boring.

The second-order practice is the **game day**: once or twice a year, declare a simulated disaster and have the on-call engineer recover the real production database into a real standby using only the runbook, while someone times it and writes down every place the runbook lied. Game days are where you discover that the runbook says "restore from S3" but nobody documented which bucket, or that the restore needs an IAM role that expired, or that the one person who knew the decryption passphrase left in March. Better to find out on a Wednesday afternoon than at 3 a.m.

## 7. DR strategies by tier: from backup/restore to active-active

**Senior rule of thumb: you cannot buy a low RTO cheaply. The four classic tiers trade money and operational complexity for shorter recovery windows, roughly exponentially. Pick the cheapest tier that meets each dataset's objectives — no higher.**

There is a well-worn ladder of disaster-recovery postures, and every dataset you own sits on exactly one rung. Climbing a rung shrinks RTO and usually RPO, and raises both standing cost and the number of things that can go wrong on a normal Tuesday.

![A matrix of DR tiers versus RPO, RTO, standing cost, and ops complexity, with the diagonal tension between fast recovery and low cost](/imgs/blogs/database-disaster-recovery-at-scale-6.webp)

The figure is the tradeoff made visual: read down the RTO column and it goes green (you recover faster); read down the cost and complexity columns and they go red (you pay more, and there is more to operate and to break). There is no row that is green everywhere. That diagonal tension is the entire decision.

- **Backup / restore** — you keep backups and WAL, and on disaster you provision fresh infrastructure and restore. Cheapest by far; nothing runs in standby. RTO is *hours* because provisioning plus restore plus WAL replay all happen after the incident starts. Right for analytics, internal tools, anything where hours of downtime is an annoyance, not a crisis.
- **Pilot light** — a minimal version of the environment is always running: the database is replicating to a small standby, but the application tier is scaled to near zero. On disaster you scale the app up and promote the database. RTO drops to tens of minutes because the data is already warm; you are only spinning up compute.
- **Warm standby** — a scaled-down but fully functional copy runs continuously in another zone or region, replicating in near-real-time. On disaster you promote and redirect traffic. RTO is minutes, RPO is seconds. This is the sweet spot for most production tiers, and where most teams should land for their important-but-not-money data.
- **Hot / active-active** — full capacity runs in two or more regions, both taking traffic, with synchronous or near-synchronous replication. On disaster you shed the dead region and the survivors absorb the load with seconds of RTO and near-zero RPO. The most expensive and the most complex by a wide margin — multi-primary write conflicts, global consistency, doubled (or more) standing cost. Reserve it for the data that genuinely cannot be down or lose a write: money movement, the parts of the system whose outage is a headline.

| Strategy | RPO | RTO | Standing cost | Complexity | Reach for it when |
| --- | --- | --- | --- | --- | --- |
| Backup / restore | minutes–hours | hours | $ | low | cost-sensitive, downtime-tolerant data |
| Pilot light | minutes | tens of minutes | $$ | medium | important data, hours of RTO unacceptable |
| Warm standby | seconds | minutes | $$$ | high | most production tiers |
| Hot / active-active | ~0 | seconds | $$$$ | highest | money movement, can't-be-down systems |

The decision should be mechanical, not heroic. Given a dataset's RPO and RTO budget, pick the cheapest tier that satisfies both — and refuse to over-provision the cache to active-active because it sits in the same cluster as the ledger. A small function makes the policy explicit and auditable:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Tier:
    name: str
    rpo_s: int       # best achievable recovery point, seconds
    rto_s: int       # best achievable recovery time, seconds
    rel_cost: int    # standing cost rank, 1 (cheapest) .. 4

# Ordered cheapest-to-priciest; numbers are representative, not guarantees.
TIERS = [
    Tier("backup/restore",     rpo_s=3600, rto_s=14400, rel_cost=1),
    Tier("pilot light",        rpo_s=300,  rto_s=1800,  rel_cost=2),
    Tier("warm standby",       rpo_s=10,   rto_s=300,   rel_cost=3),
    Tier("hot/active-active",  rpo_s=0,    rto_s=30,    rel_cost=4),
]

def choose_tier(rpo_budget_s: int, rto_budget_s: int) -> Tier:
    """Cheapest tier meeting BOTH objectives; raise if none qualifies."""
    qualifying = [t for t in TIERS
                  if t.rpo_s <= rpo_budget_s and t.rto_s <= rto_budget_s]
    if not qualifying:
        raise ValueError(
            f"No tier meets RPO<={rpo_budget_s}s and RTO<={rto_budget_s}s; "
            "relax the objectives or budget for active-active.")
    return min(qualifying, key=lambda t: t.rel_cost)

# A payments ledger: lose nothing, recover in under a minute.
print(choose_tier(rpo_budget_s=0, rto_budget_s=60).name)        # hot/active-active
# A reporting warehouse: a few hours is fine.
print(choose_tier(rpo_budget_s=3600, rto_budget_s=14400).name)  # backup/restore
# An orders database: a few seconds of loss, minutes of downtime.
print(choose_tier(rpo_budget_s=30, rto_budget_s=600).name)      # warm standby
```

Wire that logic into a config the business signs off on, and "what is our DR posture for X?" stops being a debate and becomes a lookup. The expensive tiers are not better; they are *more*. Buy exactly as much as each dataset's objectives require.

## 8. Failover and runbooks: promoting a replica without splitting the brain

**Senior rule of thumb: promoting a replica is the easy part. The hard part is guaranteeing the old primary is dead — really dead — before the new one starts taking writes. Two primaries is not a faster recovery; it is a slower, more expensive disaster.**

Failover is the availability half of recovery: the primary's host died (not its data — its host), and you must promote a healthy replica to take over. It is a different operation from PITR, but it shares the same failure-of-imagination risk, and that risk has a name: **split-brain**.

![A safe failover: detect, orchestrator decides, fence the old primary while promoting the replica, then reroute writes to the new primary only](/imgs/blogs/database-disaster-recovery-at-scale-8.webp)

The figure shows the only safe ordering. A health check fails. An orchestrator (Patroni, Orchestrator, RDS's control plane, a Kubernetes operator) confirms the failure with a quorum — not on a single missed ping, because a 200 ms network blip is not a dead primary. Then two things happen *together*: the old primary is **fenced** — forcibly isolated so it cannot accept writes, via STONITH ("shoot the other node in the head"), revoking its network access, or yanking its virtual IP — and a replica is **promoted** to new primary. Only after both does the system **reroute** writes (through a proxy, a DNS change, or a floating VIP) so clients talk exclusively to the new primary.

Skip the fence and you get the classic split-brain: the old primary was not actually dead — it was unreachable, or paused, or partitioned — and it comes back still believing it is the primary, still accepting writes from clients that have not yet been rerouted. Now you have two primaries taking divergent writes, and reconciling them after the fact ranges from painful to impossible (which write wins? what about the rows both sides changed?). Fencing is not optional ceremony; it is the thing that makes promotion safe. The mechanics of fencing, quorum, and why a partition is indistinguishable from a death without it are worth their own study — the [split-brain and fencing deep dive](/blog/software-development/database/split-brain-and-fencing-in-distributed-databases) covers the failure modes in detail.

The runbook around failover is as important as the automation, because the automation will eventually hit a case its authors did not foresee, and a human will be driving. A failover runbook that survives contact with 3 a.m. has, at minimum:

- **A single, unambiguous trigger.** "Promote when the orchestrator has declared the primary down for >30s AND a human has confirmed it is not a monitoring artifact." Ambiguous triggers cause premature failovers, which cause split-brain.
- **The exact fence command**, copy-pasteable, with the specific host/IP, so nobody is composing a STONITH call under pressure.
- **The promotion command and the reroute step**, in order, with how to verify each took effect (`SELECT pg_is_in_recovery()` should return false on the new primary).
- **A rollback / re-replication plan**: once the dust settles, how the old primary is wiped and re-cloned as a replica of the new one — never reattached as-is, because its divergent tail is exactly the split-brain you avoided.
- **Who to call** and how to declare the incident, so the person executing is not also the person managing communications.

And the runbook is only real if it has been *practiced*. A failover game day — kill the primary in staging (or, if you are brave and well-prepared, in production during a low-traffic window) and have on-call run the promotion from the runbook — is the only way to know the automation works, the fencing fires, and the runbook does not lie. The first time you exercise your failover should never be the time you need it.

> A failover you have automated but never triggered is a script with unknown bugs and root on your database. Trigger it on purpose, on your schedule, before it triggers itself on its own.

## Case studies from production

Theory is cheap. Here are seven incidents — public postmortems and lived experience — that each taught one of the lessons above the hard way. The numbers are approximate where the postmortems are; the lessons are exact.

### 1. GitLab 2017: saved by a backup nobody scheduled

On 31 January 2017, a GitLab engineer fighting replication lag and a spam-induced load spike ran `rm -rf` against what he believed was a stuck secondary's data directory. It was the **primary**. Roughly 300 GB evaporated; he caught it with only a few gigabytes left. Then came the part that made it legendary: GitLab had **five** separate backup and replication mechanisms, and on inspection, *none of them worked*. The `pg_dump` backups to S3 were silently empty because the backup host ran `pg_dump` 9.2 against a 9.6 server, which errored out and produced nothing — and the failure-alert emails were being silently rejected by DMARC. Azure disk snapshots were not enabled on the database server. Replication was the very thing that had broken. The team was ultimately rescued by a **chance LVM snapshot** an engineer had taken by hand about six hours earlier to set up a staging replica — a backup nobody had scheduled and nobody was relying on. Restoring it (slowly, over the network) cost roughly six hours of data: thousands of projects, comments, and new users. GitLab live-streamed the recovery and published an unflinching postmortem. The lesson is the thesis of this post in one incident: **you do not have backups; you have backups that restore. They had five backup methods and zero tested restores, and were saved by luck.** Test the restore, or you are gambling.

### 2. Atlassian 2022: working backups, two-week recovery

In April 2022, an Atlassian maintenance script meant to delete a deprecated legacy app was run with the wrong execution mode and the wrong list of identifiers. Instead of removing the app, it issued a **permanent delete** across roughly 880 sites belonging to about 775 customers — Jira, Confluence, the works — hard-deleted between 07:38 and 08:01 UTC. Here is the twist that makes this the perfect counterpoint to GitLab: **the backups were fine.** No customer lost more than a few minutes of data; RPO was excellent. And yet some customers were down for up to **two weeks**, restored progressively over April 8–18. Why? Because the restore tooling had been built to recover **one site at a time**, for the rare case of a single customer needing a rollback. It had never been designed or tested for a *bulk, parallel* restore of hundreds of sites at once, so Atlassian had to build that automation on the fly, mid-incident. The lesson: **RTO is a property of your restore process at the scale of your disaster, not at the scale of your tests.** A restore that works for one row, one table, or one tenant tells you nothing about restoring everything at once. Drill at disaster scale.

### 3. Code Spaces 2014: the backups died with the primary

Code Spaces was a code-hosting and project-management company. In June 2014, an attacker gained access to their AWS control panel and attempted to extort them. When Code Spaces tried to wrest back control, the attacker **deleted their EBS volumes, S3 buckets, AMIs, and snapshots** — and because the backups lived in the **same AWS account, reachable by the same credentials** as production, the backups went down with everything else. There was no copy outside the blast radius. Code Spaces could not recover and **shut down within days**. The lesson is brutal and specific: **a backup inside the same trust domain as production is not a backup against an adversary or a credential compromise.** Backups belong in a separate account, with separate credentials, with object-lock/MFA-delete so that *no single compromised key can erase them*. This is the immutable tier from the archiving stack, and it is the difference between a bad week and a dead company.

### 4. Maersk and NotPetya 2017: the domain controller in Ghana

On 27 June 2017, the NotPetya wiper tore through Maersk's network, destroying tens of thousands of laptops and thousands of servers in hours. Critically, it wiped **nearly every Active Directory domain controller** — the backbone identity system without which almost nothing else can be rebuilt. The domain controllers replicated to each other, so the wipe propagated to all of them. All but one. A controller in **Ghana** had been offline during the attack because of a local power blackout, which left it holding an intact copy of the directory. Maersk recovered the AD from that single chance survivor, physically moving the data to their recovery base, and rebuilt their global infrastructure over roughly ten days at a cost estimated in the hundreds of millions of dollars. The lesson rhymes with GitLab's: **an offline, disconnected copy is the one that survives a replicating disaster.** Replication that synchronizes your good state also synchronizes your destruction; the copy that is *not* connected — air-gapped, offline, immutable — is the one a wiper or a bad write cannot reach. Do not let luck (a power cut in Ghana) be your air gap. Build it on purpose.

### 5. GitHub 2018: choosing RPO over RTO

In October 2018, a 43-second loss of connectivity between GitHub's East Coast data centers triggered their MySQL orchestration to fail over, promoting a primary in a different site. When connectivity returned, the original site had also briefly accepted writes — the makings of a split-brain, with divergent data on two sides. GitHub made a deliberate, costly choice: rather than risk losing or corrupting committed data by guessing how to merge the divergence, they **ran in a degraded, read-mostly mode for roughly 24 hours** while they carefully reconciled and replayed to guarantee data integrity. They explicitly chose a worse RTO to protect RPO. The lesson is that **RPO and RTO genuinely trade against each other in a real incident, and which one wins is a values decision you should make before the incident, not during it.** For source-of-truth data, GitHub's call — protect the data, eat the downtime — is usually right. But you only get to make that call gracefully if you have decided in advance which datasets are "never lose a write" and which are "never be down," because you cannot maximize both at 2 a.m.

### 6. The 14:32 DELETE: PITR earns its keep

This is the incident from the opening, and I have lived versions of it more than once. A scheduled data-cleanup job, pointed at the wrong environment by a stale connection string, ran a `DELETE` that was valid SQL and catastrophic intent against a production table. Replication did its job perfectly and propagated the empty table to every replica within milliseconds — there was nothing to fail over *to*, because every node agreed the rows were gone. What saved us was not failover and not the replicas. It was a base backup from that morning plus a continuously archived WAL stream. We spun up a scratch instance, restored the base, and replayed WAL with `recovery_target_xid` set to the transaction id of the bad delete (read straight out of the database logs) and `recovery_target_inclusive = off`, so recovery stopped one transaction short of the disaster. We extracted the table, reconciled it back into production, and lost essentially nothing but the time to do it. The lesson: **PITR is the capability that turns "we lost the table" into "we lost twenty minutes." It is worth more than any number of replicas, and it costs almost nothing to have turned on in advance — and nothing to have turned on after the fact, which is to say, you cannot.** Enable WAL archiving today.

### 7. The replica that was mistaken for a backup

A smaller, quieter incident, and the most common one in the wild. A team treated their read replica as their disaster-recovery plan: "we have a replica in another region, we are covered." Then a deploy shipped a migration with a logic bug that rewrote a column's values incorrectly across a large table — valid writes, wrong data. Replication faithfully copied the corruption to the replica. When they went looking for a clean copy, both the primary and the "DR" replica held the same wrong data, and there were no real backups because the replica had been the plan. They reconstructed the column from an application event log over several painful days — and only because that log happened to exist. The lesson is the one this entire post is built around, stated plainly: **a replica is not a backup. It protects against a node dying, never against a write being wrong.** If your DR plan is "we have replicas," you have an availability plan and no recovery plan. Add backups and PITR, then test the restore.

## When to reach for each capability, and when not to

Disaster recovery is a portfolio, not a single product. Match the mechanism to the failure it actually defends against.

**Reach for backups + PITR when:**

- The data is a source of truth that cannot be regenerated from somewhere else — a ledger, orders, user content, anything where "lose it and rebuild from upstream" is not an option.
- You need protection against *logical* failures — fat fingers, bad migrations, app bugs, compromised credentials — which is to say, always, because replication gives you none.
- You want the ability to recover to a specific instant, not just to whenever the last snapshot ran. (This is most of the value; turn on continuous WAL/binlog archiving.)
- Compliance or contracts require a demonstrable, tested recovery process with an audit trail.

**Reach for a warm/hot standby and automated failover when:**

- The RTO budget is minutes or seconds and the data is too large to restore from cold backups inside that window.
- You must survive the loss of a host, a zone, or a whole region without a human in the loop for the first response.
- The business has priced an hour of downtime high enough to justify the standing cost of running capacity you hope never to use.

**Skip or scale down DR when:**

- The data is a **cache** or other derived/regenerable state. Do not back up a cache; plan its cold-start re-warm instead. Backing up derived data is cost with negative value — you restore staleness.
- The data is genuinely low-stakes (an internal dashboard's scratch table) and hours of RTO with a nightly snapshot is honestly fine. Not everything earns a standby; over-provisioning DR for trivia steals budget from the ledger.
- You are tempted to buy active-active for a dataset whose objectives a warm standby already meets. The expensive tier is not a prize; buy the cheapest tier that clears the bar.

The one practice that is never optional, regardless of tier, is the **tested restore**. Backup/restore, pilot light, warm standby, active-active — every one of them rests on the assumption that the copy of your data is good and that you can bring it back inside your RTO. That assumption is false until you have measured it, and it decays every time your schema, your version, your data size, or your tooling changes. So restore to a scratch instance on a schedule, check the row counts and the checksums, run the smoke test, time it, and alert when the number drifts. Run a game day. Make recovery boring. The teams that make the news are not the ones without backups — they are the ones who found out, in the worst hour of their year, that the backups they trusted were Schrödinger's all along.

## Further reading

- [Write-ahead log: how databases guarantee durability](/blog/software-development/database/write-ahead-log-how-databases-guarantee-durability) — the mechanism PITR replays.
- [Split-brain and fencing in distributed databases](/blog/software-development/database/split-brain-and-fencing-in-distributed-databases) — why failover without fencing is a disaster.
- [Multi-region database architecture](/blog/software-development/database-scaling/multi-region-database-architecture) — standbys, cross-region replication, and the active-active end of the DR ladder.
- [Zero-downtime schema migrations](/blog/software-development/database/zero-downtime-schema-migrations) — making the "bad migration" failure reversible by construction.
