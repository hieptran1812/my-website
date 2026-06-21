---
title: "Backups That Actually Restore: You Don't Have Backups, You Have Restores"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "A backup you have never restored is a hope, not a safety net — learn RPO and RTO precisely, the backup types and their tradeoffs, why replication is not a backup, the 3-2-1 rule, and the restore drills that prove your data can actually come back inside your recovery objective."
tags:
  [
    "site-reliability-engineering",
    "sre",
    "backups",
    "disaster-recovery",
    "rpo-rto",
    "point-in-time-recovery",
    "restore-drills",
    "data-durability",
    "ransomware",
    "business-continuity",
  ]
category: "software-development"
subcategory: "Site Reliability Engineering"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/backups-that-actually-restore-1.png"
---

At 06:12 on a Tuesday, the primary database for a payments service stopped answering. The disk controller had failed, the volume was gone, and the replica had inherited a corruption that made it refuse to start. No problem, the on-call engineer thought — we back up every night. She opened the bucket where fourteen months of nightly backups lived. There they were: 426 files, one per night, every job in the scheduler green, every run logged "completed successfully." She pulled the most recent one to restore it. It was zero bytes. So was the one before it. So was every file going back ten months, to the week a credential rotation had quietly broken the upload step while leaving the job's exit code at `0`. The backup had been "succeeding" for ten months without writing a single usable byte. The team did not have ten months of backups. They had ten months of empty files and a green dashboard.

This is the most important sentence in this entire field manual, so I am going to put it on its own line: **you don't have backups, you have restores.** A backup is a file you wrote. A restore is data that came back. Those are not the same thing, and the gap between them is where companies die. Almost every team schedules backups. Almost no team tests restores. The backup job going green tells you a process exited zero; it tells you nothing about whether the bytes are correct, complete, decryptable, current with your schema, or restorable inside the window your business can survive. The only backup that counts — the only one — is one you have *proven* you can restore, within your recovery time objective, with acceptable data loss. Everything else is a hope with a cron schedule.

![A before and after comparison contrasting a green backup job that uploaded zero bytes against a tested restore drill that verifies row counts and times the recovery](/imgs/blogs/backups-that-actually-restore-1.png)

This post lives in the recovery corner of the SRE field manual. The series spine is **define reliability, measure it, spend the error budget, reduce toil, respond to incidents, learn, engineer the fix** — and backups are the load-bearing floor under all of it. If your incident response can mitigate every failure *except* irreversible data loss, you do not have a reliability practice; you have luck. By the end of this post you will be able to: state your RPO and RTO as real numbers and defend them; pick the right backup architecture for those numbers instead of cargo-culting nightly dumps; explain to a skeptical executive why your read replica is not a backup; apply the 3-2-1 rule and know which copy actually saves you; and — the part nobody does — stand up an automated restore drill that restores to a throwaway target, verifies the data, times itself against your RTO, and pages you when it is wrong. We are going to turn "I think we have backups" into "I restored production to a scratch box 41 minutes ago and the row counts matched." Start with the inversion that everything else depends on.

## 1. The inversion: backup is the verb everyone does, restore is the verb that matters

Here is the asymmetry that explains nearly every data-loss postmortem. Backing up is *active, scheduled, automated, and visible*. Someone wrote a cron job, it runs every night, it emits a metric, it lights up a dashboard tile green. Restoring is *passive, on-demand, manual, and invisible* — it only happens when something has already gone wrong, which is the worst possible moment to discover that the procedure does not work. The result is that organizations pour effort into the half of the loop they can see and neglect the half that actually saves them.

The principle — the *why* — is that **a backup job's success signal measures the wrong thing.** A typical backup pipeline has a dozen steps: read the data, possibly quiesce or snapshot, serialize it, compress it, encrypt it, upload it to remote storage, write a manifest, exit. The job's exit code reflects whether the *last command in the script returned zero*. It does not reflect whether the upload contained your data, whether the encryption key still exists, whether the compressed stream is intact, whether the file is complete, or whether anything on the far end can read it back. In our opening story the `pg_dump | gzip | aws s3 cp` pipeline lost its credentials; `aws s3 cp` failed to authenticate but, because of how the pipe and the shell's error handling were wired, the *last* stage still exited zero. Green dashboard, empty bucket. The job measured "did the script finish," and the script finished. It never measured "can I get my data back," because nothing in the pipeline ever tried.

A restore measures the right thing because a restore *is* the thing. When you restore a backup to a real target and the database starts, the row counts match, and a smoke query returns the row you expected, you have end-to-end proof that the entire chain — read, serialize, encrypt, upload, store, download, decrypt, deserialize, load — works. There is no proxy, no signal that *correlates* with recoverability. The restore is recoverability. This is exactly the same lesson as redundancy: an untested spare is decoration, and I cover that failover side in the sibling post on [redundancy and failover that actually works](/blog/software-development/site-reliability-engineering/redundancy-and-failover-that-actually-works). Backups are the data-durability twin of the same hard truth — the thing you never exercised will fail the first time you need it.

So the entire discipline reduces to flipping the org's attention from the verb it does to the verb that matters. The rest of this post is mechanics — RPO, RTO, backup types, 3-2-1, PITR — but every one of those mechanics is in service of one outcome: a *tested, timed, runbooked restore* that you trust. Hold that frame and the rest follows.

### What "the backup is good" must actually mean

Let me make the inversion concrete with a checklist of what the word "good" has to cover, because "the job ran" covers exactly one line of it:

| Property | "Job ran green" proves it? | What proves it |
| --- | --- | --- |
| The file exists and is non-trivially sized | No (zero-byte files exit 0) | Alert on backup *size* and growth, not just exit code |
| The file is complete and not truncated | No | A checksum recorded at write time, re-verified at read time |
| The data is internally consistent | No | A restore that starts the engine without corruption errors |
| The encryption key still exists and decrypts it | No | A restore that actually decrypts, on a *different* host |
| The backup matches the current schema | No | A restore that an app of the current version can boot against |
| It can be restored inside your RTO | No | A *timed* restore drill against the clock |
| The right tables and databases are included | No | Row-count and object-presence verification post-restore |

Every "No" in that table is a place a team has lost data while staring at a green dashboard. The fix for all of them is the same single practice — restore drills — which is why this post builds toward it. But you cannot drill a restore until you know what you are drilling *toward*, and that means two numbers.

## 2. RPO and RTO: the only two numbers that define recovery

You cannot design a backup system without committing to two recovery objectives, and most teams have never written them down. They are the two axes of every data-loss conversation, and confusing them is the most common mistake I see.

**RPO — Recovery Point Objective — is how much data you can afford to lose.** It is measured in *time*, and it answers the question "when we recover, how far back in time is the most recent data we will have?" If your last good backup was at 02:00 and the failure hit at 06:00, every write between 02:00 and 06:00 is gone, and your *actual* data loss for this incident is four hours. Your RPO is the *maximum* such gap you have decided is acceptable. The crucial relationship: **RPO is bounded by your backup frequency.** A 24-hour backup interval means that in the worst case — failure right before the next backup — you lose almost 24 hours of data. So a nightly backup gives you, at best, a 24-hour RPO. If your business says "we cannot lose more than one hour of payment records," a nightly backup is already a guaranteed violation, and no amount of restore testing fixes that, because the data simply was never captured.

**RTO — Recovery Time Objective — is how long recovery is allowed to take.** It is also measured in time, but it is wall-clock *forward* from the moment of failure to the moment service is restored. And here is the part teams forget: RTO is not "how long the restore command runs." It is the *sum* of detect plus decide plus restore plus verify plus cut back. You have to notice the failure, decide to invoke recovery (often the longest and most political step), run the restore, verify the data is correct, and switch traffic back to the recovered system. A restore that runs in 90 minutes inside a 2-hour RTO is failing if detection took 20 minutes and the cut-back dance takes another 30.

![A timeline showing the last good backup, accumulating live writes, the failure point defining RPO, and then the detect decide restore and cut-back steps that sum to RTO](/imgs/blogs/backups-that-actually-restore-2.png)

The picture to hold is two clocks running from the moment of failure in opposite directions. RPO runs *backward* — how far back is your last safe point. RTO runs *forward* — how far ahead until you are healthy again. They are independent. A system can have a tiny RPO (you lose almost nothing) but a terrible RTO (it takes a day to bring it back), or vice versa. Both have to satisfy the business, and they often pull the architecture in different directions.

### How RPO drives frequency and RTO drives architecture

This is the design lever, and it is worth saying slowly because it determines what you build.

**RPO drives backup *frequency* and method.** If you can lose 24 hours, a nightly full dump is fine. If you can only lose one hour, you need to capture data at least hourly — and once you push below roughly 15 minutes, periodic snapshots become impractical and you move to *continuous* capture: write-ahead-log shipping, change-data-capture, or synchronous replication into a separate failure domain. The tighter the RPO, the more continuous the capture has to be. There is no nightly job that gives you a 5-minute RPO; the architecture has to change.

**RTO drives backup *architecture* and where the recovered system lives.** Here the killer constraint is restore *throughput*. Suppose you have a 10 TB dataset and a 4-hour RTO. Your storage restores at, say, 1 TB/hour. A cold restore from a logical dump takes 10 hours just to load the bytes — you have already blown the 4-hour RTO by 6 hours before you even verify anything. No backup *frequency* fixes this; the data is captured, it just cannot come back fast enough. To hit a 4-hour RTO on 10 TB you need a fundamentally different architecture: a storage-level snapshot you can mount near-instantly, an incremental restore that only replays a small delta, or a *hot standby* that is already running and only needs to be promoted. The RTO math forces the shape.

#### Worked example: the 99.9% SLO meets a 24-hour RPO

Tie this back to the series currency. Suppose your service has a 99.9% availability SLO. Over a 30-day month, 99.9% allows $43.2$ minutes of downtime — that is your error budget. Now suppose a data-loss incident forces a recovery that takes 6 hours because you only had nightly logical dumps and the restore ran cold. Six hours is $360$ minutes. You have not just blown the month's $43.2$-minute budget; you have blown roughly *eight months* of budget in a single incident. The arithmetic is brutal and clarifying: a slow recovery is not a one-off inconvenience, it is a structural reliability liability that your SLO already priced. If your error budget is small, your RTO has to be small, and that forces your backup architecture. The error budget and the recovery objective are the same conversation in different units — which is exactly the point the [error budget post](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) makes about turning "reliable enough?" into arithmetic.

### Writing the numbers down

The single highest-leverage thing most teams can do this quarter is hold a 30-minute meeting with the business owner and write two numbers per critical dataset. Not "as low as possible" — that is not a number, it is an aspiration that costs infinity. Real numbers:

| Dataset | RPO (max data loss) | RTO (max recovery time) | Implied architecture |
| --- | --- | --- | --- |
| Payment ledger | 1 minute | 30 minutes | Synchronous replication + WAL archiving + hot standby |
| User profiles | 1 hour | 2 hours | WAL archiving (5-min RPO) + warm restore target |
| Analytics events | 24 hours | 8 hours | Nightly snapshot, cold restore acceptable |
| Generated thumbnails | "any" (regenerable) | best-effort | No backup; regenerate from source |

Notice the bottom row. Some data should not be backed up at all because it is *derived* — you can regenerate it from a source of truth cheaper than you can store and restore it. Knowing which data is precious and which is regenerable is half the battle, because it lets you spend your limited recovery effort where it matters. Now that we have the targets, let us look at the tools that hit them.

## 3. Backup types and their tradeoffs

There is no single "backup." There is a small family of techniques, each landing at a different point on the RPO-RTO-cost surface. Picking the wrong one for your objectives is how teams end up with backups that exist but cannot meet the numbers they need.

![A matrix comparing nightly full incremental storage snapshot and WAL archiving point-in-time recovery across RPO recovery time and cost or load](/imgs/blogs/backups-that-actually-restore-3.png)

**Full backup.** A complete copy of the dataset. Simplest to reason about and to restore (one file, one restore), but slow to take and expensive to store if you keep many of them. A nightly full on a large database can hammer the source with I/O and take hours, which itself becomes a reliability concern.

**Incremental backup.** Captures only what changed since the *last backup of any kind*. Tiny and fast to take. The catch is restore: you must replay the last full plus *every* incremental since, in order. A broken link anywhere in the chain breaks the restore. Long chains mean slow, fragile restores — bad for RTO.

**Differential backup.** Captures everything changed since the last *full*. Restore needs only the full plus the *one* latest differential — simpler and more robust than an incremental chain — at the cost of differentials growing larger the further you get from the last full.

**Snapshot (storage-level).** The storage layer (LVM, ZFS, a cloud block-store, a SAN) freezes a point-in-time view, usually copy-on-write, so it is near-instant to take and near-instant to restore by mounting. Fantastic RTO. The tradeoffs: a snapshot often lives on the *same* storage system as the live data (so it does not satisfy 3-2-1 by itself), it can capture an application in an inconsistent state unless you quiesce or use a crash-consistent engine, and it is tied to that storage platform (not portable to a different vendor or version).

**Logical dump.** A portable, human-readable-ish export (`pg_dump`, `mysqldump`, `mongodump`). The output is engine-version-portable and lets you restore a single table, but it is *slow* to produce and especially slow to restore on large data, because the target rebuilds indexes and re-validates constraints. Great for portability and small data; a trap for large-data RTO.

**Continuous archiving / WAL shipping / PITR.** The database already writes every change to a write-ahead log for durability; if you *archive* those log segments off-box continuously, you can restore a base backup and then *replay the log* up to any point in time. This is the technique that gives you a tiny RPO (you lose only the writes since the last shipped segment, often seconds to a few minutes) and the ability to recover to a precise instant. The cost is operational complexity and continuous I/O for archiving. I lean on the [write-ahead log post](/blog/software-development/database/write-ahead-log-how-databases-guarantee-durability) for the mechanism of *why* the WAL exists; here we are using it as the spine of recovery.

| Type | Take speed | Restore speed (RTO) | Typical RPO | Storage cost | Portability |
| --- | --- | --- | --- | --- | --- |
| Full | Slow | Moderate (single load) | = interval | High (many copies) | High |
| Incremental | Fast | Slow (long chain replay) | = interval | Low | Medium |
| Differential | Medium | Moderate (full + 1) | = interval | Medium | Medium |
| Snapshot | Near-instant | Near-instant (mount) | = snapshot age | Medium (COW) | Low (platform-tied) |
| Logical dump | Slow | Slow (rebuild indexes) | = interval | Medium | High (version-portable) |
| WAL / PITR | Continuous | Base + replay | Seconds–minutes | Medium-high | Medium |

A realistic production stack does *not* pick one. It composes them: a periodic full or snapshot as the *base*, continuous WAL archiving for *tiny RPO*, and a warm standby for *tiny RTO*. The full gives you a fast starting point; the WAL closes the gap to seconds; the warm target removes the restore-throughput bottleneck. Each technique is covering a different objective.

### A WAL-archiving config sketch

Here is the shape of continuous archiving in PostgreSQL. This is the part that turns "lose up to 24 hours" into "lose up to a few minutes," and it is the foundation for point-in-time recovery later.

```yaml
# postgresql.conf — turn on continuous WAL archiving
wal_level = replica                  # emit enough WAL to recover and replicate
archive_mode = on
archive_timeout = 60                 # force a segment at least every 60s -> bounds RPO
# Ship each completed WAL segment to durable, OFFSITE storage.
# %p = path to the segment, %f = its filename. Exit non-zero on failure
# so Postgres retries and does NOT recycle an unshipped segment.
archive_command = 'wal-g wal-push %p'
```

```bash
# Take the base backup that WAL replays on top of (run on a schedule).
# wal-g/pgBackRest stream a consistent base to object storage.
wal-g backup-push /var/lib/postgresql/data

# What you should verify, NOT assume:
#   - archive_command exits 0 only on a successful upload
#   - the newest archived segment is < archive_timeout old (else RPO is drifting)
#   - the base backup completed and is non-zero
wal-g backup-list           # confirm a recent base exists
wal-g wal-verify timeline   # confirm the WAL chain is unbroken
```

The single most important line there is `archive_timeout = 60`. It bounds your RPO from above: even on an idle database, a segment is forced at least every 60 seconds, so your worst-case data loss from WAL archiving is roughly the time since the last forced segment plus shipping latency — call it a couple of minutes, not 24 hours. The second most important thing is that `archive_command` must *fail loudly* (exit non-zero) when the upload fails, because Postgres will then refuse to recycle the segment and your archive stays consistent. A silent-success `archive_command` is the WAL equivalent of our opening zero-byte disaster.

### Consistency: the backup that captures a half-written transaction

There is a subtle trap inside the backup-types discussion that deserves its own treatment, because it is the source of backups that pass every check and still restore to garbage: **consistency.** A backup is only useful if it captures a *coherent* state — one the database could legitimately have been in at a single instant. The failure mode is a backup that captures part of a transaction: the debit was written but not the matching credit, an index points at a row that the row-copy missed, a foreign key references a parent that was not in the snapshot. The data is *present* and the file is *non-zero* and the engine even *starts* — and the data is silently wrong in a way that may not surface for weeks.

There are three ways a backup achieves consistency, and knowing which one you rely on is part of knowing your backups are real. **Transactionally consistent dumps** take a snapshot of the database's MVCC state at the start and read the entire dump as of that single transaction — `pg_dump` does this by default within a repeatable-read transaction, so the dump reflects one coherent instant no matter how long it runs. **Crash-consistent snapshots** capture the storage as if the machine had lost power at that instant; the engine then runs its normal crash-recovery (WAL replay) on startup to reach a consistent state — this is what a raw block-level snapshot of a live database gives you, and it works *only* because the engine is designed to recover from a crash. **Application-consistent / quiesced snapshots** briefly pause or flush the application before snapshotting so there is no in-flight work at all — the safest but most disruptive, used when the storage layer cannot guarantee crash consistency across multiple volumes.

The dangerous middle ground is a *file-level copy of a live database that is neither quiesced nor crash-consistent* — for example, `cp -r` or `rsync` over a running database's data directory, or a snapshot that spans multiple volumes without coordinating them so each volume freezes at a slightly different instant. That backup captures a torn state that *no single instant ever produced*, and crash recovery cannot fix it because the WAL and the data files disagree about reality. It will often restore and start and then corrupt subtly. The rule: never back up a live database by copying files unless your snapshot is genuinely atomic across every volume the engine touches, and always prefer a transactionally-consistent dump or an engine-native snapshot mechanism. This is precisely the kind of failure a restore drill with a *checksum* of a stable data slice catches and a "did the engine start?" check misses.

## 4. Replication is not a backup (and this confusion kills companies)

If you remember one thing from this post besides "you have restores, not backups," remember this: **a replica is not a backup.** I have watched smart teams skip backups entirely because "we have three replicas across two regions, the data is super safe." It is not. Replication and backup solve different problems, and conflating them removes your only defense against the failure mode that actually destroys data.

![A before and after figure showing a DROP TABLE replicating to a replica in milliseconds versus a point-in-time backup restoring the table to the instant before the mistake](/imgs/blogs/backups-that-actually-restore-4.png)

The principle is about *what each one protects against.* Replication protects against *infrastructure* failure — a disk dies, a host crashes, a region goes dark — by keeping a continuously-updated, *faithful* copy elsewhere. The operative word is *faithful*. A replica's entire job is to reproduce, as fast as possible, every change made to the primary. So when you run `DROP TABLE customers;` on the primary, or a bad migration that nulls a column, or your application corrupts a million rows because of a logic bug, the replica does its job *perfectly*: it reproduces the destruction in milliseconds. Now you have the disaster in three places across two regions, beautifully synchronized. Replication is an amplifier of correctness *and* of catastrophe; it has no opinion about which is which.

A backup protects against *logical* failure — a human mistake, a software bug, a malicious deletion, a slow corruption — by preserving a copy that is *frozen at a point in the past* and *isolated from the live system's mutations*. That isolation in *time* is the whole point. Because the backup from 02:00 does not know about and cannot be reached by the `DROP TABLE` you ran at 14:00, it still has the table. The replica, which is connected to the live mutation stream, does not.

The taxonomy that makes this click:

| Threat | Replication helps? | Backup helps? |
| --- | --- | --- |
| Disk / host / rack failure | Yes (failover to replica) | Yes, but slower |
| Whole-region outage | Yes (cross-region replica) | Yes, if offsite copy |
| `DROP TABLE` / bad migration | No (replicated instantly) | Yes (point in past) |
| Application corrupts data | No (replicated) | Yes (point before corruption) |
| Ransomware encrypts volumes | No (encrypts replicas too if reachable) | Only if offline/immutable copy |
| Operator deletes the data | No (deletion replicates) | Only if isolated/immutable |

Read the right-hand column. Backups are the *only* defense against the entire bottom half of that table — the human and software threats — and those are the threats that take companies down, because infrastructure failures are now largely handled by managed redundancy while a one-line `DELETE` without a `WHERE` clause is a Tuesday. The architecture-time treatment of how replicas are wired (leader/follower, sync vs async, multi-leader) lives in the database series' [replication post](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless); the operations point I am making here is narrower and sharper: do not let a replica count as your backup, ever. You need both. The replica is for *availability*; the backup is for *recoverability*; they are different objectives and you must fund both.

### The stress test: a replica with delay is still not a backup

"But our replica is *delayed* by an hour, so we have an hour to catch a `DROP TABLE` and stop it." A delayed replica is a useful trick — it buys you a window to notice a catastrophic statement before it propagates — but it is not a backup either, for three reasons. First, the window is fixed and small; if you notice the `DROP` 90 minutes later, the delayed replica has already replayed it. Second, you only get *one* point in time (now-minus-one-hour), not the ability to recover to the second before the mistake. Third, the delayed replica is still online, still reachable, still part of the live system — ransomware or a credential compromise that can reach it can destroy it. A backup that saves you has to be a *point in time* (any time, ideally) and *isolated* from the live system. That isolation is the next idea.

## 5. The 3-2-1 rule and why the offline copy is the one that saves you

The 3-2-1 rule is the oldest piece of backup wisdom and still the most violated. It says: keep **3** copies of your data, on **2** different media or storage systems, with **1** copy offsite — and the modern, ransomware-era amendment: with at least one copy *offline or immutable.* It is a rule about *correlated failure*. Every clause exists to break a different way that "we have backups" turns into "we lost everything anyway."

![A layered stack from the live primary through an online backup a second medium an offsite region and an offline immutable WORM copy](/imgs/blogs/backups-that-actually-restore-5.png)

**Three copies** breaks single-point-of-failure on the backup itself. If your only backup is corrupt or zero-byte (hello again, opening story), you have nothing. With independent copies, one being bad does not sink you.

**Two media / storage systems** breaks *correlated media failure*. If your live data and your backup are both on the same SAN, the same cloud account, the same filesystem, then a failure of that system — a SAN firmware bug, an account suspension, a filesystem corruption — takes out both at once. They are not independent. A snapshot on the same volume as the source is the classic violation: it is convenient and fast, but a volume-level failure eats the snapshot too.

**One offsite** breaks *correlated site failure*: a datacenter fire, a flood, a region outage, a cloud account-level problem. If every copy is in `us-east-1`, then `us-east-1` having a bad day is your disaster across all copies simultaneously.

**One offline or immutable** — the modern amendment — breaks the *online-deletion* failure mode, and this is the one most teams are missing today. Here is the scenario that should keep you up: an attacker (or a compromised credential, or an operator with too much access, or a buggy cleanup script) gains write access to your environment. They encrypt or delete the live data. Then they encrypt or delete the *online backups too* — because those backups are reachable from the same environment, with the same or nearby credentials. Ransomware operators specifically hunt for and destroy backups before triggering the encryption, precisely because they know an untouched backup defeats them. If every copy you have is *online and mutable*, a single sufficiently-privileged actor can erase all of them. The only copy that survives is one that *cannot be modified or deleted* even by an admin: an immutable object-store bucket with object-lock / WORM (write-once-read-many) retention, or a genuinely air-gapped copy on offline media. That copy is your floor.

| 3-2-1 clause | Failure it breaks | Common violation |
| --- | --- | --- |
| 3 copies | The one backup is bad | "We have a backup" (singular) |
| 2 media | Correlated media/account failure | Snapshot on the same volume as source |
| 1 offsite | Site / region / account loss | All copies in one region |
| 1 offline/immutable | Online deletion (ransomware, operator, script) | Every copy is mutable |

#### Worked example: the immutable copy earns its keep

A team runs nightly snapshots into the same cloud account as production and replicates them to a second region — they feel safe. A compromised CI credential (broad permissions, as CI credentials tragically often have) is used to delete the production databases *and* iterate every snapshot in both regions and delete them too, because they are all reachable with that one credential. Everything online is gone in four minutes. What survives? Only the one copy they had pushed to an object-lock bucket with a 30-day compliance-mode retention that *no credential, including root, can shorten or bypass.* They lose the day's edits since that immutable copy and spend a tense afternoon restoring — but the company survives. Without that one immutable copy, the company does not. The immutable copy is cheap insurance against the most expensive failure mode there is. Configure it like this:

```bash
# S3 Object Lock in COMPLIANCE mode: not even the account root
# can delete or shorten retention until the period expires.
aws s3api put-object-lock-configuration \
  --bucket backups-immutable-prod \
  --object-lock-configuration '{
    "ObjectLockEnabled": "Enabled",
    "Rule": { "DefaultRetention": { "Mode": "COMPLIANCE", "Days": 30 } }
  }'

# The backup writer credential gets PutObject ONLY -- no DeleteObject,
# no PutObjectLockConfiguration. Least privilege closes the attack path.
```

The discipline here is least privilege plus immutability: the credential that *writes* backups can only write, never delete or relock, and the storage itself refuses deletion for the retention window. Now even a fully compromised environment cannot erase your floor.

### Where the encryption key lives is part of the backup

Encrypting backups is correct and increasingly mandatory — but encryption introduces a dependency that becomes a single point of failure in exactly the disaster you are preparing for. The trap is brutal in its simplicity: you encrypt the backups with a key, and the key lives *on the system you just lost.* The ciphertext is perfectly intact in your offsite immutable bucket, and no one alive can read it, because the only copy of the decryption key burned down with the primary. I have seen this turn a *successful* restore (the file came back) into a *failed* recovery (the file is unreadable). The key is part of the backup, and if you have not thought about where it lives, you have not finished designing the backup.

The rule is that the key must live in a *different failure domain* from both the live system and the backup storage, and you must *exercise decryption from a clean host* in your drills. Concretely: keep the master key in a dedicated key-management service or hardware security module that survives the loss of the application infrastructure; keep an *escrowed* copy of the recovery key offline (a sealed envelope in a safe is not a joke — it is a legitimate last-resort control for the master key); and grant the *restore* path access to the key separately from the *backup-write* path, so a compromise of one does not hand over the other. The verification angle: a drill that restores on a *different host than production* and successfully decrypts there proves not just that the bytes are good but that the key is reachable from a recovery context. A drill that decrypts using a key cached on the production host proves nothing about a disaster in which that host is gone.

### Retention is a tradeoff, not a maximum

The last piece of 3-2-1 hygiene is retention — how *long* you keep each copy — and the instinct to "keep everything forever to be safe" is wrong on two counts. First, long retention *widens your blast radius*: every backup you keep is another copy an attacker can exfiltrate, another GDPR/CCPA deletion obligation you are violating if it contains personal data past its lawful window, and another bill. Second, retention should match *recovery need*, which is usually short for operational recovery (you want yesterday and last week, not 2019) and longer only for *compliance* archives, which have completely different access patterns and can live in cold, cheap, deeply-immutable storage. The practical pattern is *tiered retention*: hourly PITR for 7 days (tight RPO for recent operational recovery), daily snapshots for 30 days (the operational window), and monthly archives for the compliance horizon in cold storage. Each tier serves a distinct purpose, and collapsing them — keeping a year of hourly backups — pays storage and risk for recoverability no one will ever use. Decide retention from the same place you decided RPO and RTO: the business need, written down, not fear.

## 6. The untested backup that wasn't: five ways a "successful" backup betrays you

We have circled the cautionary core; now let us name the specific ways backups that *look* fine fail at the worst moment. I have personally seen every one of these, and each is a postmortem with a green dashboard in the timeline.

**1. The job was silently failing.** The opening story: an exit-zero pipeline that uploaded nothing for ten months because a credential broke. The signal measured "script finished," not "data landed." The fix is to alert on the *artifact*, not the *job* — backup file size, age, and growth, plus the restore drill that follows.

**2. The backup is missing a critical table.** A `pg_dump` with an `--exclude-table` someone added "temporarily" two years ago. The backup runs fine, is non-zero, restores cleanly — and is missing the `payments` table because of a forgotten flag. The fix is object-presence verification: after a drill restore, assert that *every expected table/collection/bucket is present* and within an expected row-count band.

**3. The backup restores, but to a useless state.** The data comes back but the application cannot use it. Maybe the backup predates a schema migration and the current app version refuses to boot against it. Maybe foreign keys are inconsistent because the dump was not transactionally consistent. The fix is to boot the *current app version* against the restored data in the drill and run a smoke transaction, not just check that the database engine starts.

**4. The decryption key was in the system you lost.** The backups were encrypted — good — but the key (or the KMS access, or the passphrase in a config file) lived *only* on the primary that just died. The backups are perfectly intact ciphertext that no one can read. The fix is to store recovery keys in a *separate* failure domain (a dedicated secrets manager, an offline escrow) and to *exercise decryption on a different host* in the drill.

**5. The restore takes three days.** The backups are complete, correct, decryptable, current — and a cold logical restore of a 5 TB database rebuilds indexes for 70 hours. The data is recoverable in principle and useless in practice, because the business cannot be down for three days. This one only shows up if you *time* the restore against your RTO, which almost nobody does until they live it.

Notice that exit code, file existence, and even "the engine starts" each catch only one or two of these. The single practice that catches *all five* is a real restore drill that restores to a fresh host, decrypts there, verifies object presence and row counts and checksums, boots the current app version, runs a smoke query, and *times the whole thing.* That is the next section, and it is the heart of the post.

## 7. Restore drills: the only thing that proves any of this

Everything so far has been the *why* and the *what*. Here is the *how*, and it is the practice that separates teams that recover from teams that write a very long postmortem. **Regularly, automatically restore from your backups to a throwaway environment, verify the data, time it against your RTO, and page someone when it is wrong.** Make recovery a tested, timed, runbooked procedure that runs on a schedule — ideally as often as you take backups, at minimum weekly for anything critical — instead of a heroic improvisation you attempt for the first time during a Sev1.

![A graph of an automated restore drill that pulls the latest backup spins a scratch target restores and replays then branches on verification to either record the time or page the on-call](/imgs/blogs/backups-that-actually-restore-6.png)

The principle is that a restore drill *closes the loop* between the verb you do (backup) and the verb that matters (restore), and it does so *before* the disaster, on your schedule, in daylight, with no customer impact, by restoring to a *scratch* target you throw away afterward. It converts an unknown ("can we restore?") into a measured, monitored SLI ("our last 30 restore drills all passed, the p95 restore time is 47 minutes against a 2-hour RTO"). That is the same move as every other reliability practice in this series: turn a hope into a number you watch. An old quip in this field — sometimes called Schrödinger's backup — is that the state of any backup is unknown until you attempt a restore. A drill collapses the wavefunction on *your* schedule instead of the disaster's.

Here is a real, copy-and-adapt restore-drill script. It restores to a disposable target, verifies, times itself, and exits non-zero (which your scheduler turns into a page) when anything is wrong.

```bash
#!/usr/bin/env bash
# restore-drill.sh -- prove the latest backup restores within RTO.
# Run nightly via cron/k8s CronJob. Exits non-zero => alert fires.
set -euo pipefail

RTO_SECONDS=7200                 # 2h RTO for this dataset
SCRATCH="restore-drill-$(date +%s)"
START=$(date +%s)

echo "[$(date -u)] drill start: restoring latest base + WAL to ${SCRATCH}"

# 1. Spin a throwaway target (ephemeral container / instance / namespace).
create_scratch_postgres "${SCRATCH}"

# 2. Restore the most recent base backup, then replay WAL to the latest point.
#    This exercises decryption on a DIFFERENT host than production.
wal-g backup-fetch "/scratch/${SCRATCH}/data" LATEST
configure_recovery "/scratch/${SCRATCH}/data"   # restore_command = wal-g wal-fetch
start_scratch_postgres "${SCRATCH}"             # replays WAL, then opens

# 3. VERIFY: presence of critical objects, row counts in band, checksum.
EXPECTED_TABLES="users payments orders ledger_entries"
for t in ${EXPECTED_TABLES}; do
  psql -h "${SCRATCH}" -tAc "SELECT to_regclass('public.${t}') IS NOT NULL" \
    | grep -q t || { echo "FAIL: table ${t} missing"; exit 2; }
done

ROWS=$(psql -h "${SCRATCH}" -tAc "SELECT count(*) FROM payments")
# Compare against an expected band recorded from production (allow drift).
[ "${ROWS}" -ge "${PAYMENTS_MIN:-1}" ] || { echo "FAIL: payments rows=${ROWS}"; exit 3; }

# Content checksum of a stable, ordered slice -- detects silent corruption.
CKSUM=$(psql -h "${SCRATCH}" -tAc \
  "SELECT md5(string_agg(id::text||amount::text, ',' ORDER BY id))
     FROM payments WHERE created_at < date_trunc('day', now())")

# 4. BOOT the CURRENT app version against the restored data + smoke query.
APP_IMAGE="registry/payments-api:$(prod_running_tag)"
run_smoke_test "${APP_IMAGE}" "${SCRATCH}" || { echo "FAIL: app smoke"; exit 4; }

# 5. Time it against RTO and report.
ELAPSED=$(( $(date +%s) - START ))
echo "[$(date -u)] drill OK: ${ELAPSED}s, payments_rows=${ROWS}, cksum=${CKSUM}"
[ "${ELAPSED}" -le "${RTO_SECONDS}" ] \
  || { echo "FAIL: restore ${ELAPSED}s exceeds RTO ${RTO_SECONDS}s"; exit 5; }

destroy_scratch "${SCRATCH}"     # throwaway target, gone after the drill
```

Walk the exit codes, because each one maps to a betrayal from the previous section. Exit 2 catches the *missing critical table*. Exit 3 catches a backup that is structurally fine but *empty or short*. The checksum catches *silent corruption*. Exit 4 catches the *useless-state* failure where the data restores but the current app cannot use it, and because it boots a fresh app image on a different host it also exercises *decryption away from production*. Exit 5 catches the *three-day restore* by failing when the elapsed time exceeds RTO. The opening zero-byte disaster fails at the `backup-fetch` step or exit 3 — loudly, on a normal Tuesday, ten months earlier, when it is a 20-minute fix instead of a company-ending one.

### Turning the drill into an SLI you watch

A drill that runs and fails silently is just a fancier version of the problem. Wire it into the same Prometheus stack the rest of this series uses. Emit metrics from the drill and alert on them:

```yaml
# Prometheus alerting rules for the restore drill (pushed via Pushgateway
# or scraped from the CronJob's metrics).
groups:
  - name: backup-recovery
    rules:
      # The drill has not PASSED recently -> our recoverability is unknown.
      - alert: RestoreDrillStale
        expr: time() - backup_restore_drill_last_success_timestamp_seconds > 129600
        for: 0m
        labels: { severity: page }
        annotations:
          summary: "No successful restore drill in 36h for {{ $labels.dataset }}"
          runbook: "https://runbooks.example.com/restore-drill-stale"

      # Restore is creeping toward the RTO ceiling -> architecture is drifting.
      - alert: RestoreTimeApproachingRTO
        expr: backup_restore_drill_seconds / backup_restore_rto_seconds > 0.8
        for: 0m
        labels: { severity: ticket }
        annotations:
          summary: "Restore time {{ $value | humanize }} of RTO for {{ $labels.dataset }}"

      # The backup ARTIFACT itself looks wrong (size/age) -- catches zero-byte.
      - alert: BackupArtifactSuspect
        expr: |
          backup_last_size_bytes < (0.5 * backup_size_bytes_7d_avg)
          or (time() - backup_last_completed_timestamp_seconds) > 90000
        for: 0m
        labels: { severity: page }
        annotations:
          summary: "Backup for {{ $labels.dataset }} is too small or too old"
```

Three alerts, three different failure modes. `RestoreDrillStale` fires when recoverability becomes *unknown* (the drill stopped passing) — that is the meta-alert that would have caught the ten-month silence. `RestoreTimeApproachingRTO` fires when your restore is creeping toward the RTO ceiling as data grows, giving you months of warning before a real incident blows the objective. `BackupArtifactSuspect` watches the *artifact* — size relative to the 7-day average and age — which is the cheap check that catches a zero-byte upload *the next morning* even before the drill runs. This is the bridge from this post to the alerting discipline in the [alerting-that-doesn't-cry-wolf](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf) sibling: every one of these pages on a real, user-affecting condition (we cannot recover), not on a noisy proxy.

### How deep should the verification go?

"Verify the restore" is easy to say and easy to do badly, so let us be precise about the levels of verification, because each level catches a different class of failure and costs a different amount. The levels form a ladder you climb as the data's importance rises.

| Verification level | What it proves | What it misses | Cost |
| --- | --- | --- | --- |
| File exists, size in band | The artifact landed (catches zero-byte) | Anything about the *content* | Trivial |
| Engine starts / mounts | The backup is structurally loadable | Missing tables, wrong schema, corruption | Minutes |
| Object presence (all tables/buckets) | Nothing critical was excluded | Row-level emptiness or corruption | Minutes |
| Row counts in expected band | The tables actually hold data | Wrong *values*, subtle corruption | Minutes |
| Content checksum of a stable slice | The bytes match what production held | Live data not in the slice | Minutes |
| App boots + smoke transaction | The data is *usable* by the current app | Full functional correctness | Tens of minutes |

The mistake almost everyone makes is stopping at level two — "the database started, we're good." Level two catches almost nothing this post warns about; the missing-table, the empty-table, the wrong-schema, and the silent-corruption failures all pass a "did it start?" check. The pragmatic target for a source-of-truth dataset is to climb to "app boots + smoke transaction," because that is the level that proves *usability*, which is the only thing that matters when you are actually recovering. The checksum step deserves a word: you cannot checksum the *entire* live database in a drill (it changes constantly and the drill restores a slightly older point), so you checksum a *stable, historical slice* — rows older than today, ordered deterministically — and compare it against the same checksum computed on production for that frozen slice. If the historical slice's checksum drifts between drills, your backups are silently corrupting data that *should never change*, which is the most insidious failure of all and the only thing that catches it.

One more discipline: **test the test.** A restore drill is itself code, and code rots. Periodically inject a known-bad backup (a deliberately truncated file, a backup missing a table) and confirm the drill *fails* on it. A drill that has only ever seen good backups may have a bug that makes it pass on bad ones — the verification equivalent of a smoke detector you have never pressed the test button on. The GitLab postmortem's deepest lesson was not "backups can fail" — everyone knows that — it was "the *detection* of backup failure can fail," and the only defense is to occasionally feed your verification a failure on purpose and watch it catch it.

## 8. Point-in-time recovery, end to end

Point-in-time recovery (PITR) is the technique that gives you both a tiny RPO and the surgical ability to recover to a *specific instant* — which is exactly what you need for the logical-failure case where replication is useless. Let us walk it end to end, because it is the single most valuable recovery capability and the one most teams have configured-but-never-tested.

![A timeline of point-in-time recovery restoring a base snapshot fetching archived WAL setting a recovery target replaying the log and promoting the recovered database](/imgs/blogs/backups-that-actually-restore-7.png)

The mechanism builds directly on the write-ahead log. The database writes every change to the WAL before applying it (that is how it guarantees durability — the [WAL post](/blog/software-development/database/write-ahead-log-how-databases-guarantee-durability) covers exactly why). If you have a *base backup* from some past moment and you have *archived every WAL segment* since, then recovery is: restore the base, then replay the archived WAL forward — and you can *stop replaying at any point you choose.* That "stop at a chosen point" is the magic. It means you can recover to 13:59:58, the instant *before* someone ran `DROP TABLE customers;` at 14:00:00, recovering everything up to the second before the mistake and nothing after it.

#### Worked example: recovering from a 14:00 DROP TABLE with PITR

A bad migration ran `DROP TABLE customers;` at exactly 14:00:00. You notice at 14:11 when the app starts throwing errors. Replication will not help — the `DROP` is already on every replica. With PITR:

```bash
# 1. Restore the most recent base backup (say the 02:00 one) to a target.
wal-g backup-fetch /var/lib/postgresql/data LATEST_BEFORE_14:00

# 2. Tell Postgres to replay WAL up to ONE SECOND before the mistake.
cat > /var/lib/postgresql/data/recovery.signal <<'EOF'
EOF
cat >> postgresql.auto.conf <<'EOF'
restore_command = 'wal-g wal-fetch %f %p'
recovery_target_time = '2026-06-20 13:59:58'   # the instant before the DROP
recovery_target_action = 'promote'             # open read-write once we hit it
recovery_target_inclusive = 'off'              # do NOT replay the target txn
EOF

# 3. Start Postgres. It restores base, replays WAL to 13:59:58, then promotes.
pg_ctl start
# 4. VERIFY before you trust it.
psql -tAc "SELECT to_regclass('public.customers') IS NOT NULL"   # expect: t
psql -tAc "SELECT count(*) FROM customers"                       # expect: full count
```

The arithmetic of the win: your RPO for *this* recovery is essentially the 2 seconds between 13:59:58 and the `DROP` at 14:00:00 — you lose only the writes in that sliver, not the 12 hours back to the 02:00 base, because the WAL carried everything in between. And critically, you *did not* lose the writes between 13:59:58 and 14:00:00 that were unrelated to the `DROP` — well, you lose a 2-second sliver, which is the deliberate price of recovering to *before* the mistake. That is the precision PITR buys. The contrast with a nightly-dump-only world is stark: there you would restore the 02:00 dump and lose 12 hours of customer activity. PITR turns a 12-hour loss into a 2-second loss.

The stress test you must run: **does your PITR actually work, or is it configured-and-untested?** This is the most common gap I find in audits — `archive_mode = on` is set, segments are piling up in a bucket, and *no one has ever fetched them and replayed.* The drill is the same restore drill from section 7, parameterized with a `recovery_target_time`. Run it. Replay to an arbitrary past instant on a scratch box and verify. If you have never replayed your WAL, you have WAL files, not PITR.

## 9. Sizing a real system: 2 TB, 1-hour RPO, 2-hour RTO

Let us put the whole framework to work on a concrete sizing problem, with the math fully shown, because this is the calculation every team should be able to do and almost none have written down.

![A before and after figure showing a nightly full backup missing the RPO and RTO targets versus WAL archiving plus a warm target meeting a five minute RPO and under two hour RTO](/imgs/blogs/backups-that-actually-restore-8.png)

#### Worked example: making 2 TB hit 1h RPO and 2h RTO

You run a 2 TB PostgreSQL database for a service whose business owner has signed off on: **RPO = 1 hour** (lose at most an hour of writes) and **RTO = 2 hours** (back in service within two hours of failure). Your storage restores at roughly 0.2 TB/hour for a cold logical restore (index rebuilds dominate). What architecture meets the numbers? Work it from the constraints.

**Check the RPO against your current plan.** You currently take a nightly full dump. Worst case, failure hits at 23:59, one minute before the next dump — your last good point is the 02:00 dump from *yesterday*, so you lose almost 22 hours. Your *actual* RPO with nightly backups is ~24 hours. Target is 1 hour. **Fail by 23×.** No frequency of full dumps that is operationally sane (you cannot dump a 2 TB database hourly without crushing it) closes this. You need *continuous* capture. Add WAL archiving with `archive_timeout = 60`: now your worst-case RPO is the time since the last shipped segment, bounded at ~1–2 minutes. **1-minute RPO < 1-hour target. Pass, with 30× margin.**

**Check the RTO against your current plan.** A cold restore of 2 TB at 0.2 TB/hour is $2 / 0.2 = 10$ hours just to load the bytes, before index rebuilds and verification. Add detect (~10 min) and decide (~15 min) and cut-back (~20 min) and you are at ~11 hours. Target is 2 hours. **Fail by 5.5×.** The bottleneck is restore *throughput*, so you must remove the restore from the critical path. Two options:

- **Storage snapshot as base.** Restore by *mounting* a recent block-level snapshot (near-instant) instead of replaying a logical dump, then replay WAL forward from the snapshot's timestamp to current. Mount is minutes, not hours; WAL replay for one day of changes on 2 TB is typically tens of minutes. Estimate: detect 10 + decide 15 + mount 10 + WAL replay 40 + verify 15 + cut-back 20 = **110 minutes < 120-minute RTO. Pass.**
- **Warm standby.** Keep a continuously-fed standby running (it is *not* your backup — it is a fast restore target seeded from backups + WAL). On failure you verify it is uncorrupted at a chosen point and *promote* it. Promotion is seconds. But beware: a plain hot standby has replicated any logical corruption, so for the *infrastructure-failure* case it gives you a near-zero RTO, and for the *logical-failure* case you fall back to PITR on a scratch target.

The composed answer: **snapshot base + WAL archiving + a pre-staged warm restore target.** The snapshot and warm target attack RTO (remove the throughput bottleneck); the WAL attacks RPO (continuous capture). Here is the sizing as a small calculator you can adapt:

```python
# rpo_rto_sizing.py -- sanity-check an architecture against objectives.
def check(dataset_tb, restore_tbph, rpo_target_min, rto_target_min,
          capture, base):
    # RPO from the capture method.
    rpo = {"nightly_full": 24 * 60, "hourly": 60,
           "wal_1min": 2}[capture]            # worst-case minutes lost
    # RTO from the restore architecture.
    load_min = {"cold_logical": dataset_tb / restore_tbph * 60,
                "snapshot_mount": 10,          # near-instant mount
                "warm_promote": 1}[base]
    overhead = 10 + 15 + 40 + 15 + 20          # detect+decide+wal+verify+cutback
    rto = load_min + overhead
    return {
        "rpo_min": round(rpo, 1), "rpo_ok": rpo <= rpo_target_min,
        "rto_min": round(rto, 1), "rto_ok": rto <= rto_target_min,
    }

# Nightly full only -> both fail.
print(check(2, 0.2, 60, 120, "nightly_full", "cold_logical"))
# {'rpo_min': 1440, 'rpo_ok': False, 'rto_min': 700.0, 'rto_ok': False}

# WAL + snapshot base + warm target -> both pass.
print(check(2, 0.2, 60, 120, "wal_1min", "snapshot_mount"))
# {'rpo_min': 2, 'rpo_ok': True, 'rto_min': 110.0, 'rto_ok': True}
```

The point of writing it as code is that the sizing becomes *reproducible and auditable*. When your data grows to 4 TB, you rerun it and discover the cold-logical RTO doubled to 1400 minutes — the model tells you *before* an incident does that your architecture has aged out of its objectives. That is the same forward-looking spirit as the [capacity planning](/blog/software-development/site-reliability-engineering/capacity-planning-and-forecasting) sibling: model the constraint, watch it drift, act before it breaks.

## 10. War story: real backup disasters and what they teach

These are real, public incidents (and one composite drawn from common patterns, which I flag). Each one cost real money and each one teaches a specific lesson that maps to a section above.

**GitLab, 2017.** During an incident response, an engineer running cleanup commands at 23:00, exhausted, ran a deletion against the *primary* database directory instead of the failing replica. About 300 GB of production data vanished in seconds. Then the truly painful part of the postmortem unfolded: of *five* configured backup and replication mechanisms, none worked when needed. The logical backups had been silently failing (a version mismatch made `pg_dump` produce empty output — the zero-byte pattern), the S3 uploads were not happening, the snapshots were not taken for that system, and the replication had already propagated the deletion. They recovered from a *staging* database snapshot that happened to be six hours old, losing six hours of data and issues. GitLab's transparency was exemplary, and the lesson is brutal and exactly this post's thesis: *they had five backup methods and zero tested restores.* Five green-ish dashboards, nothing that came back. The fix was, predictably, restore testing and alerting on backup success-by-content.

**The credential-rotation zero-byte backup (composite, but I have seen this exact pattern multiple times).** A team's nightly `pg_dump | gzip | aws s3 cp -` pipeline kept exiting zero after an IAM credential rotation broke the S3 upload, because the way the pipeline's error handling was wired, only the last stage's exit code mattered and it had been masked. For months the bucket received zero-byte objects. The job was green; the data was absent. Discovery came during a real primary loss. The fix is the trio this post keeps returning to: alert on backup *size and age* not just exit code, verify backup *content* (object presence, row counts), and run restore *drills* so the absence surfaces the next morning instead of during the disaster.

**Ransomware that ate the online backups (industry pattern).** A well-documented modern pattern: attackers gain access, spend days quietly mapping the environment, locate and *delete or encrypt the online backups first*, and only then trigger the encryption of production. Organizations that survive are the ones with an *immutable* or *air-gapped* copy the attacker could not reach or modify; organizations that do not survive had every backup online and mutable. This is the entire justification for the "1 offline/immutable" amendment to 3-2-1, and it is why I will not sign off on a backup design in 2026 that lacks an object-locked or air-gapped copy.

The common thread across all three: the data was *theoretically* protected (backups existed, methods were configured) and *practically* lost (untested, unverified, reachable). Theory protected; practice failed. A restore drill and an immutable copy convert theory into practice. The architecture-level postmortem craft — how to write these up so the lesson sticks — is in the [blameless postmortem](/blog/software-development/site-reliability-engineering/the-blameless-postmortem) sibling, and the broader incident-anatomy framing is in [the anatomy of an incident](/blog/software-development/site-reliability-engineering/the-anatomy-of-an-incident).

## 11. How to reach for this (and when not to)

Every practice has a cost, and a field manual that only tells you to do everything is useless. Here is the decisive guidance, including the cases where doing *less* is correct.

**Always do, for any data that is a source of truth:** write down RPO and RTO as real numbers with the business owner; take backups frequently enough to meet RPO; keep at least one offsite *and* one immutable/offline copy (3-2-1); alert on backup *size and age*, not just exit code; and — non-negotiable — run an automated restore drill on a schedule that times itself against RTO and pages when it fails. If you do nothing else from this post, do the restore drill. It is the single highest-leverage reliability investment most teams are missing, and it is a weekend to stand up.

**Reach for PITR / WAL archiving** when your RPO is tighter than your sane backup frequency (sub-hour), or when you need to recover to a *specific instant* to undo a logical mistake. It is the right tool for "undo the bad migration to the second before it ran."

**Reach for snapshots + warm standby** when your RTO is tighter than your restore throughput allows — i.e., the dataset is large enough that a cold restore cannot finish in time. The bigger the data, the more you need to remove the restore from the critical path.

**When NOT to:** Do *not* engineer a 5-minute RPO and 15-minute RTO for *derived, regenerable* data — thumbnails, caches, search indexes you can rebuild from a source of truth. Back up the source; regenerate the derivative. Do *not* take expensive synchronous-replication + continuous-archiving + warm-standby infrastructure for an internal batch job that can tolerate losing a day and being down for an afternoon — that is gold-plating that spends real money and operational toil to buy reliability the business does not need. Do *not* keep 14 months of nightly fulls "to be safe" if your retention requirement is 30 days; you are paying storage *and* widening the blast radius of a compromise. And do *not* let backups become un-drillable: if your restore procedure is so complex that you cannot automate a drill, that complexity *is* the bug — simplify until the drill runs unattended. The right amount of backup is exactly enough to meet your written RPO and RTO and not one nine more, because over-engineering recovery is itself toil that steals time from the next outage. Match the [error budget](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) — if users cannot tell the difference, neither should your backup budget.

This is also where backups hand off to the broader continuity story. Backups answer "can I get the *data* back?" The wider questions — can I get the whole *service* back, in another region, with its dependencies, and keep the business running — belong to disaster recovery and business continuity. Those siblings (planned slugs `disaster-recovery-and-business-continuity` and `running-stateful-systems-reliably`) extend this post from "the data survives" to "the business survives"; backups are the foundation they build on, which is why we put the floor in first. And the architecture-time decisions about how to *design* systems so that recovery is even possible — graceful degradation, idempotent writes, isolation of blast radius — live in the system-design series' treatment of [reliability, SLOs, error budgets, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation).

## 12. Tying it back to the reliability loop

Recall the series spine: define reliability, measure it, spend the error budget, reduce toil, respond to incidents, learn, engineer the fix. Backups sit at a specific, load-bearing place in that loop, and seeing where sharpens how you operate them.

You **define** the recovery contract as two numbers — RPO and RTO — exactly the way you define an SLO; they are the durability SLOs. You **measure** recoverability not by watching the backup job (the wrong signal) but by running restore drills and emitting their pass/fail and timing as an SLI you alert on — the drill *is* the measurement. The cost of a slow recovery is denominated in **error budget**: a 6-hour recovery can blow months of a 99.9% budget in one incident, which is how you justify the spend on better architecture to a numbers-driven business. You **reduce toil** by automating the drill so recoverability is verified unattended rather than by an annual heroic manual test (the manual test is pure toil and, worse, gives a false sense of safety because it runs too rarely). When the disaster comes you **respond** by executing the *runbooked, drilled* restore — calm, because you have done it 30 times on a scratch box — instead of improvising. You **learn** from the rare real recovery and the routine drills, feeding restore-time creep back into capacity planning. And you **engineer the fix** by closing whatever gap a drill or incident exposed: a zero-byte upload, a missing table, a restore that aged past RTO. The whole loop runs *on backups* — it is the floor under incident response, because the one failure mode incident response cannot mitigate is irreversible data loss. Put the floor in, prove it holds, and the rest of the reliability practice has something to stand on. Start every backup conversation with the intro map, [reliability is a feature](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset), and treat this post as the answer to its scariest question: when the data is gone, can you actually get it back?

## Key takeaways

- **You don't have backups, you have restores.** A backup you have never restored is a hope. The only backup that counts is one you have *proven* you can restore, within RTO, with acceptable data loss.
- **The backup job going green proves the script finished, not that the data is recoverable.** Alert on backup *size, age, and content*, never just the exit code — that is exactly how the ten-month zero-byte disaster hid in plain sight.
- **RPO is the data you can lose (set by backup frequency); RTO is the time recovery may take (set by backup architecture).** They are independent, both must satisfy the business, and they pull the design in different directions.
- **Replication is not a backup.** A replica copies your `DROP TABLE` in milliseconds. You need a copy frozen at a *point in time* and *isolated* from the live system to survive logical and human failures.
- **3-2-1: three copies, two media, one offsite — and in 2026, one offline or immutable.** Ransomware and over-privileged credentials delete the online backups too; the WORM/air-gapped copy is the floor that survives.
- **Run an automated restore drill on a schedule.** Restore to a throwaway target, verify object presence, row counts, and checksum, boot the current app, time it against RTO, and page when it fails. It is the single highest-leverage backup practice and most teams are missing it.
- **PITR turns a 12-hour loss into a 2-second loss** by replaying archived WAL to the instant before the mistake — but only if you have actually replayed it; configured-and-untested WAL is not PITR.
- **Match recovery to the objective, not to fear.** Don't engineer five-nines recovery for regenerable data or an internal batch job; do write the numbers down and meet them exactly.

## Further reading

- [Reliability Is a Feature: The SRE Mindset](/blog/software-development/site-reliability-engineering/reliability-is-a-feature-the-sre-mindset) — the series intro and map; backups are the floor under everything it describes.
- [Redundancy and Failover That Actually Works](/blog/software-development/site-reliability-engineering/redundancy-and-failover-that-actually-works) — the availability twin of this post; the untested spare and the untested backup are the same lesson.
- [The Error Budget: The Currency of Reliability](/blog/software-development/site-reliability-engineering/the-error-budget-the-currency-of-reliability) — why a slow recovery is denominated in error budget, and how RTO and the budget are the same conversation.
- [Alerting That Doesn't Cry Wolf](/blog/software-development/site-reliability-engineering/alerting-that-doesnt-cry-wolf) — how to wire the backup-size and restore-drill alerts so they page on real recoverability loss, not noise.
- [Write-Ahead Log: How Databases Guarantee Durability](/blog/software-development/database/write-ahead-log-how-databases-guarantee-durability) — the mechanism that makes WAL archiving and point-in-time recovery possible.
- [Distributed Replication: Leader, Multi-Leader, Leaderless](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless) — the architecture-time view of replicas, and why even three of them are not a backup.
- [Reliability, SLOs, Error Budgets, and Graceful Degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) — designing systems so recovery is possible in the first place.
- The Google SRE Book and SRE Workbook (the data-integrity and disaster-recovery chapters), the PostgreSQL documentation on continuous archiving and point-in-time recovery, and your cloud provider's object-lock / WORM and cross-region replication guides — the canonical sources for the artifacts sketched here.
