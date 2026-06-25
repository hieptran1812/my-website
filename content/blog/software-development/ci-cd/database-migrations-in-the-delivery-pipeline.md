---
title: "Database Migrations in the Delivery Pipeline"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Master zero-downtime schema evolution by wiring expand/contract patterns, online migration tools, and decoupled pipeline stages into a delivery process that never takes your database offline."
tags:
  [
    "ci-cd",
    "devops",
    "database",
    "migrations",
    "schema-evolution",
    "zero-downtime",
    "flyway",
    "liquibase",
  ]
category: "software-development"
subcategory: "CI/CD & Delivery"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/database-migrations-in-the-delivery-pipeline-1.png"
---

It is 2015. Your team has been building toward this release for three weeks. The feature is solid, tests are green, stakeholders are standing by. At 11 PM on a Tuesday, you open a maintenance window, put up the "We'll be right back" banner, and start an `ALTER TABLE` on a 200-million-row orders table. The progress bar creeps. At 2 AM it is at 40 percent. At 4 AM, 70 percent. At 6 AM — the table is finally rebuilt, the app restarts, the banner comes down. You spend the rest of the day recovering from a support backlog and the engineering equivalent of a hangover.

Now it is today. A team at a modern platform company needs to add a `payment_method_token` column to a table that has grown to 500 million rows. Their on-call engineer opens a pull request. The PR gets reviewed, merged, and the pipeline runs. By the time the engineer finishes their second cup of coffee, the column exists in production. Zero downtime. No maintenance window. No support escalations.

The delta between those two stories is not hardware or money. It is process: a delivery pipeline that treats schema migrations as a first-class, separately managed concern rather than something you bolt onto an application deploy. That gap starts the moment you accept that a database schema and an application binary are two different things with two different rollout lifecycles — and they must never be forced to change in lockstep.

This post is the full playbook. You will leave knowing exactly how to wire migrations into every stage of a CI/CD pipeline, which tools to use for tables of different sizes, how to structure changes so they can always be rolled back, and when to accept that some migrations are fundamentally one-way streets and how to handle them safely.

![The naive deploy+migrate problem: overlapping app versions cause errors](/imgs/blogs/database-migrations-in-the-delivery-pipeline-1.png)

## Why Schema Migrations Are the Hardest Delivery Problem

Every other artifact in your delivery pipeline has one useful property: you can replace it atomically. A new container image swaps in. A new function bundle goes live. A static asset gets a new hash and CDN busting. The old version stops serving traffic and the new version takes over in a matter of seconds.

A database schema doesn't work that way. Schemas are shared mutable state. When you change them, every client connected to the database — including application instances that are mid-rollout — sees the change immediately. There is no "version gate" between a running app server and the database schema it reads. The moment you execute `ALTER TABLE orders ADD COLUMN payment_method_token VARCHAR(255)`, every existing connection that issues a `SELECT *` or `INSERT INTO orders` gets the new schema.

This creates an inescapable overlap window during any rolling deploy. In a rolling update, you never take all old app instances down at once — that would create downtime. Instead, you spin up one new instance, confirm it is healthy, then drain and replace an old instance, and repeat. During the minutes or hours that process takes, you have old instances (expecting the old schema) and new instances (expecting the new schema) talking to the same database simultaneously.

The overlap breaks in both directions:

**Migration before deploy**: You run the migration first, then start rolling the app. Old instances are now reading a schema they weren't written to handle. If you added a NOT NULL column without a default, every INSERT from old app code will fail. If you renamed a column, every SELECT from old app code will return a null or throw an error.

**Migration after deploy**: You roll new app instances first, then run the migration. New instances try to read or write columns that don't exist yet. Every request that touches that path fails until the migration completes.

The only escape is a pattern where schema changes are written to be forward- and backward-compatible with at least two consecutive app versions, so the schema can change independently of the app deploy. That is the core insight behind everything in this post.

There is a second hard property: schema changes interact directly with database locking. A naive `ALTER TABLE` on MySQL or an older Postgres version acquires an `ACCESS EXCLUSIVE` lock that blocks all reads and writes until the table reconstruction is complete. On a table that fits in RAM, that is milliseconds. On a 100-million-row table with 200 GB of data, that is hours. During those hours, your application is completely non-functional for anything touching that table.

The third hard property is irreversibility. You can always roll back an application binary — you have the old container image tagged and ready. Rolling back a schema migration is far more complicated. If the migration added a nullable column and the new app has been writing to it for three hours, rolling back the migration means dropping that column and losing three hours of data. If the migration transformed existing data in a way that loses information — for example, splitting a `full_name` column into `first_name` and `last_name` and dropping `full_name` — there is often no path back. These facts change how you design migrations.

## The Naive Approach and Why It Fails

The naive approach is baked into most deployment tutorials: run your migration script at the start of your deploy script, then start the new application.

```bash
#!/bin/bash
# The dangerous deploy script
flyway migrate -url=jdbc:postgresql://prod-db/app -locations=filesystem:./migrations

# Now restart the app
kubectl rollout restart deployment/my-app
```

This fails in at least four distinct ways.

**The overlap window problem**: As described above, during a Kubernetes rolling update, old pods and new pods coexist. Old pods reading the migrated schema will fail if the migration added a NOT NULL column, renamed a column, changed a column type, or removed a column the old app still tries to read.

**The migration-as-blocker problem**: If your migration takes 20 minutes and your pipeline has a timeout of 15 minutes, your deploy fails mid-migration. Even if there's no timeout, blocking a deploy pipeline on a long-running migration means no other deploys can proceed in that pipeline. For teams doing multiple deploys per day, a migration that takes an hour is a serious throughput problem.

**The no-separate-rollback problem**: When the migration and the deploy are fused, rolling back the app also (in theory) requires rolling back the migration. But the rollback script for the deploy almost certainly just re-deploys the old container image — it does not run a `flyway undo` or equivalent. So after a failed deploy, you end up with old app code running against the new schema, which may or may not work depending on what the migration did.

**The production-only visibility problem**: Migrations that run "as part of deploy" often only get tested against a production-like database at deploy time. CI runs unit tests against an in-memory H2 or a tiny SQLite file. Nobody verifies that the migration actually runs cleanly against a Postgres 15 instance with the production schema, real data distributions, and production indexes. You find out it fails when it fails in production.

The right answer is to treat migration as a completely separate pipeline stage — one that runs before the app deploy, has its own health checks, its own success/failure semantics, and its own rollback path that is entirely independent of the application rollback path.

## The Expand/Contract Pattern

The expand/contract pattern is the foundational discipline behind zero-downtime schema evolution. It was formalized in the database continuous delivery literature and is the basis of how teams at Stripe, GitHub, Shopify, and others ship schema changes to multi-hundred-million-row tables with no downtime.

The core insight: never make a breaking schema change. Instead, split every schema change into at least two separate, independently deployable migrations separated by an application deploy.

![Expand/contract phases: expand, backfill, contract](/imgs/blogs/database-migrations-in-the-delivery-pipeline-2.png)

### Phase 1: Expand — Making the Schema Additive

The expand migration makes the schema change in a way that both the old and new app versions can tolerate. The cardinal rule: the expand phase must never remove or rename anything the old app depends on.

In practice, for adding a new column, that means adding it as `NULL`-able without a default constraint. Old application code will never attempt to write the column (it doesn't know it exists) and any `SELECT *` queries from old code will receive `NULL` for the column — which the old code ignores. New app code sees the column and writes to it. Both versions coexist safely.

```sql
-- Phase 1 example: adding payment_method_token
-- The NULL-able constraint is not optional — it is what makes this safe.
ALTER TABLE orders
  ADD COLUMN payment_method_token VARCHAR(255);

-- Add the supporting index separately and non-blockingly (Postgres).
-- On a large table, this can run concurrently with live traffic.
CREATE INDEX CONCURRENTLY idx_orders_payment_token
  ON orders (payment_method_token)
  WHERE payment_method_token IS NOT NULL;
```

For column renaming — one of the trickier cases — the expand phase adds the new column name as an alias alongside the old. Old code reads `legacy_payment_id`, new code reads `payment_token`. Both columns exist and are kept in sync by the application layer during the transition:

```sql
-- Expand for a rename: add new name, keep old name
ALTER TABLE orders
  ADD COLUMN payment_token VARCHAR(255);

-- The application now writes BOTH columns on every insert/update.
-- Old instances write only legacy_payment_id.
-- New instances write both legacy_payment_id and payment_token.
```

The corresponding application-layer dual-write (Go example) looks like this:

```go
// DualWriteOrderPayment writes to both columns during the expand phase.
// This function is deployed with the "new" app version after the expand migration.
func DualWriteOrderPayment(db *sql.DB, orderID int64, token string) error {
    _, err := db.Exec(`
        UPDATE orders
        SET legacy_payment_id = $1,
            payment_token     = $1
        WHERE id = $2
    `, token, orderID)
    return err
}
```

The dual-write pattern guarantees that no data diverges between the old and new columns during the overlap window. Once all running instances are on the new version, the dual-write can be removed — new code reads only `payment_token`, and the contract phase drops `legacy_payment_id`.

What expand never does:
- Drop a column
- Rename a column in-place
- Add a NOT NULL constraint to an existing column without a default
- Change a column type in a way that breaks existing queries

### Phase 2: Backfill — Catching Up Historical Rows

After the expand migration runs and the new application version is deployed (which writes the new column on every new row), you must backfill the existing rows that were written before the deployment. This is the step most teams handle incorrectly.

The naive backfill is a single UPDATE without a WHERE clause:

```sql
-- DANGEROUS: holds a lock on the entire table for the duration
UPDATE orders
SET payment_method_token = legacy_payment_id::VARCHAR
WHERE payment_method_token IS NULL;
```

On a table with 50 million rows, this lock can run for 10–30 minutes. The table is not unreachable for reads (Postgres uses MVCC), but all concurrent writes queue up waiting for the lock to release. Queue depth builds. Connections exhaust. The app degrades.

The correct approach is a batched update loop with explicit rate limiting:

```sql
-- Batched backfill: safe for large tables
-- Run this as a migration job or a cron script — NOT inline in the schema migration file.
DO $$
DECLARE
  batch_size  INT     := 10000;
  last_id     BIGINT  := 0;
  max_id      BIGINT;
  rows_updated INT;
BEGIN
  SELECT MAX(id) INTO max_id FROM orders;

  LOOP
    UPDATE orders
    SET payment_method_token = legacy_payment_id::VARCHAR
    WHERE id > last_id
      AND id <= last_id + batch_size
      AND payment_method_token IS NULL;

    GET DIAGNOSTICS rows_updated = ROW_COUNT;

    EXIT WHEN last_id >= max_id;
    last_id := last_id + batch_size;

    -- Small pause to let replication catch up and reduce I/O pressure.
    -- Tune this value based on observed replication lag during the run.
    PERFORM pg_sleep(0.05);
  END LOOP;

  RAISE NOTICE 'Backfill complete. Last processed id: %', last_id;
END $$;
```

Each iteration takes a brief exclusive lock on at most `batch_size` rows, commits, then releases. Concurrent writes proceed in the gaps between batches. The `pg_sleep` call keeps the backfill from saturating I/O on the primary. For very large tables (500M+ rows), teams often run the backfill as a dedicated background service with configurable concurrency and progress tracking.

For MySQL, the equivalent batched pattern using an application-layer loop (in Python, run as a migration job):

```python
import time
import mysql.connector

def backfill_payment_token(conn, batch_size=10000, sleep_ms=50):
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(id) FROM orders")
    max_id = cursor.fetchone()[0]

    start_id = 0
    total_updated = 0

    while start_id < max_id:
        cursor.execute("""
            UPDATE orders
            SET payment_instrument_type = CASE
                  WHEN payment_method LIKE 'card_%%' THEN 'card'
                  WHEN payment_method LIKE 'bank_%%' THEN 'bank_transfer'
                  ELSE 'card'
                END
            WHERE id > %s
              AND id <= %s
              AND payment_instrument_type IS NULL
        """, (start_id, start_id + batch_size))
        conn.commit()
        total_updated += cursor.rowcount
        start_id += batch_size
        time.sleep(sleep_ms / 1000.0)
        print(f"Progress: {start_id}/{max_id} ({100*start_id/max_id:.1f}%)")

    print(f"Backfill complete. {total_updated} rows updated.")
```

### Phase 3: Contract — Removing the Old Structure

Only after all app versions have been updated to use the new column exclusively — meaning there are no running instances that still read the old column — do you remove the old column. This is safe because no live code references it.

```sql
-- Phase 3: Contract
-- Only run this after confirming zero app instances still reference the old column.
-- A safe check: grep your codebase and deployed binaries for the old column name.
ALTER TABLE orders DROP COLUMN legacy_payment_id;
```

The gap between phase 2 and phase 3 can be days or weeks. Teams at larger companies often have an explicit policy: do not run the contract migration until a second full release cycle has passed, confirming no rollback to a version that uses the old column is possible.

This is not boilerplate caution. It saved Shopify's database team from a serious incident in 2019 when they ran a contract migration one release cycle too early and had to emergency-restore a column from a snapshot after an unexpected rollback request came in.

The expand phase is also where you can safely add constraints that would be dangerous on an existing column. The sequence for making a column NOT NULL is a concrete example — it must follow the three-step pattern, not a single ALTER:

```sql
-- WRONG: Instant table lock, breaks old app inserts during overlap
ALTER TABLE orders ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'pending';

-- RIGHT: Three separate steps, each safe independently

-- Step 1 (expand): add as nullable — safe, no lock on write path
ALTER TABLE orders ADD COLUMN status VARCHAR(20);

-- Step 2 (backfill): fill all existing rows before adding constraint
UPDATE orders SET status = 'legacy' WHERE status IS NULL;
-- (Use the batched loop above for large tables)

-- Step 3 (contract): now the constraint is safe — no NULLs exist
ALTER TABLE orders ALTER COLUMN status SET NOT NULL;
```

## Online Schema Change Tools

For tables above roughly 1 million rows, a naive `ALTER TABLE` is dangerous even in development — it takes too long and locks too much. The solution is an online schema change tool that rewrites the table in the background while keeping the original fully available for reads and writes.

![Before gh-ost: hours of downtime; after gh-ost: zero downtime](/imgs/blogs/database-migrations-in-the-delivery-pipeline-3.png)

### gh-ost — How It Works Internally

gh-ost is the most widely used online migration tool for MySQL. It was open-sourced by GitHub in 2016 after they used it to run migrations on their largest production tables. The design is clever: unlike pt-online-schema-change, gh-ost does not use triggers. Triggers on MySQL create a synchronous overhead for every write during the migration window, which can slow down production writes by 5–20 percent on high-write tables.

The internal architecture has three concurrent processes running simultaneously:

**The row-copy process** reads the original table in primary key order, copying chunks of rows into a shadow table (`_orders_gho`) that already has the desired new schema applied. The chunk size is configurable and gh-ost self-adjusts it based on observed load.

**The binlog relay process** positions gh-ost as a MySQL replica using the binary log protocol. Every DML statement that modifies the original table while the row copy runs also gets applied to the shadow table in binlog order. This is the key insight that replaces triggers: instead of intercepting writes at the database engine level (which is synchronous and adds latency), gh-ost reads the binlog asynchronously and replays it. The shadow table stays consistent without adding any overhead to the write path.

**The throttle monitor** continuously samples database metrics (Threads_running, replication lag, custom queries) and pauses or slows the row copy when the database is under stress. The row copy can pause and resume any number of times without losing progress.

The cutover phase — swapping the shadow table for the original — uses a lock-based atomic rename. gh-ost acquires a brief lock on the original table, confirms the shadow table is fully caught up via binlog position, executes `RENAME TABLE orders TO _orders_del, _orders_gho TO orders`, and releases the lock. The lock time is typically 200–800 milliseconds.

![gh-ost migration lifecycle: shadow table, row copy, binlog replay, cutover](/imgs/blogs/database-migrations-in-the-delivery-pipeline-4.png)

```bash
# gh-ost command to add a column to a large MySQL table
gh-ost \
  --host=prod-mysql.internal \
  --port=3306 \
  --user=gh-ost-user \
  --password="${MYSQL_PASSWORD}" \
  --database=app_production \
  --table=orders \
  --alter="ADD COLUMN payment_method_token VARCHAR(255) DEFAULT NULL" \
  --chunk-size=1000 \
  --max-load="Threads_running=25" \
  --critical-load="Threads_running=1000" \
  --ok-to-drop-table \
  --initially-drop-ghost-table \
  --initially-drop-old-table \
  --timestamp-old-table \
  --switch-to-rbr \
  --assume-rbr \
  --cut-over=default \
  --exact-rowcount \
  --concurrent-rowcount \
  --default-retries=120 \
  --panic-flag-file=/tmp/ghost.panic.flag \
  --postpone-cut-over-flag-file=/tmp/ghost.postpone.flag \
  --execute
```

Key flags to understand:

- `--max-load` and `--critical-load` tell gh-ost to pause or abort if the database load exceeds thresholds. This is your safety valve.
- `--postpone-cut-over-flag-file` lets an operator delay the cutover until a specific moment (off-peak hours, after a deploy). gh-ost will finish the row copy, stay synchronized via binlog, and wait for the flag file to be removed before cutting over.
- `--panic-flag-file` lets you abort instantly by touching a file, without killing the gh-ost process.

**Monitoring a running gh-ost migration** is done through the interactive socket that gh-ost exposes. You can check progress, adjust chunk size, and pause/resume without restarting the process:

```bash
# Check progress through the Unix domain socket
echo "status" | nc -U /tmp/gh-ost.orders.sock

# Expected output:
# # Migrating `app_production`.`orders`; Ghost table is `app_production`.`_orders_gho`
# # Migration started at Mon Jan 22 09:15:32 2024
# # chunk-size: 1000; max-lag-millis: 1500ms; ...
# # Rows: estimated 182,500,000; copied: 45,123,000; backlog: 12
# # ETA: 2h14m (eta-duration)

# Pause the row copy (leaves binlog relay running — no data loss)
echo "throttle" | nc -U /tmp/gh-ost.orders.sock

# Resume the row copy
echo "no-throttle" | nc -U /tmp/gh-ost.orders.sock

# Change chunk size on the fly to reduce load
echo "chunk-size=500" | nc -U /tmp/gh-ost.orders.sock

# Trigger the cutover manually (removes the postpone flag file effect)
echo "unpostpone" | nc -U /tmp/gh-ost.orders.sock
```

The ability to pause, resume, and adjust chunk size without restarting is what makes gh-ost practical for 24/7 production systems. You can start a migration during low traffic, throttle it during a traffic spike, and let it run back at full speed when traffic drops — all without touching the migration process itself.

**Rate-limiting with `--max-lag-millis`** is the most important production-safety flag for MySQL replication topologies. When gh-ost detects that replica lag has exceeded the threshold, it automatically throttles the row copy until lag drops:

```bash
gh-ost \
  --table=orders \
  --alter="ADD COLUMN payment_method_token VARCHAR(255) DEFAULT NULL" \
  --max-load="Threads_running=25" \
  --max-lag-millis=1500 \
  --replication-lag-query="SELECT max(Seconds_Behind_Master) FROM replication_status" \
  --execute
```

Without this flag, a migration on a write-heavy table can drive replication lag into the tens of seconds, causing replica reads to return stale data and potentially causing read-heavy features to fail.

### pt-online-schema-change

Percona Toolkit's `pt-online-schema-change` predates gh-ost and uses triggers. It is more mature and supports more edge cases, but the trigger overhead is real. Use pt-osc when:
- You cannot give gh-ost a user with REPLICATION SLAVE privileges
- You're on an older MySQL version where gh-ost's binlog format requirements aren't met
- You're on Amazon Aurora or Google Cloud SQL and cannot use gh-ost's direct binlog streaming

```bash
pt-online-schema-change \
  --alter="ADD COLUMN payment_method_token VARCHAR(255) DEFAULT NULL" \
  --host=prod-mysql.internal \
  --user=pt-osc-user \
  --password="${MYSQL_PASSWORD}" \
  --chunk-size=1000 \
  --max-load="Threads_running:25" \
  --critical-load="Threads_running:500" \
  --set-vars="lock_wait_timeout=1" \
  --no-drop-old-table \
  --print \
  --execute \
  D=app_production,t=orders
```

### pg_repack (PostgreSQL)

PostgreSQL has `CREATE INDEX CONCURRENTLY` and recent versions handle many `ALTER TABLE` operations without full table locks. But for operations that do require a table rewrite — changing a column type, adding a NOT NULL constraint to an existing column with a default, or general table bloat reclamation — `pg_repack` is the tool.

pg_repack works by creating a new copy of the table, using triggers to track changes during the copy, and then swapping the tables atomically at the end. It requires superuser access and a Postgres extension to be installed.

```bash
# Repack a bloated table and change a column type
pg_repack \
  --host=prod-postgres.internal \
  --port=5432 \
  --username=repack-user \
  --dbname=app_production \
  --table=orders \
  --no-superuser-check \
  --wait-timeout=60 \
  --jobs=4
```

For most additive changes in PostgreSQL (adding nullable columns, adding indexes), Postgres's native `CONCURRENTLY` support is sufficient and pg_repack is not needed. The `ADD COLUMN` operation in Postgres 11+ with a constant DEFAULT no longer rewrites the table — Postgres stores the default in catalog metadata and serves it lazily. This is a significant operational win.

```sql
-- Postgres 11+: This does NOT rewrite the table
ALTER TABLE orders
  ADD COLUMN payment_method_token VARCHAR(255) DEFAULT NULL;

-- This also does NOT rewrite (Postgres 11+, constant default)
ALTER TABLE orders
  ADD COLUMN is_fraud_flagged BOOLEAN DEFAULT FALSE NOT NULL;

-- Index creation without blocking reads/writes
CREATE INDEX CONCURRENTLY idx_orders_fraud
  ON orders (is_fraud_flagged)
  WHERE is_fraud_flagged = TRUE;
```

## Migration Tool Comparison

The tooling ecosystem for managing migration files — versioned SQL scripts tracked in your repository — has matured significantly. The choice of tool affects how migrations are organized, checksummed, and rolled back.

![Migration tool comparison: Flyway vs Liquibase vs Atlas vs golang-migrate](/imgs/blogs/database-migrations-in-the-delivery-pipeline-5.png)

**Flyway** is the default for Java shops. It uses versioned SQL files (`V1__create_users.sql`, `V2__add_orders.sql`) stored in a `db/migration` directory. Each file gets a checksum that Flyway verifies on every run — if you modify an already-applied migration file, Flyway will refuse to run. This is the correct behavior: applied migrations are immutable history. Rollback support requires the paid tier or the community workaround of writing "undo" migrations as new versioned files.

**Liquibase** is more complex but more flexible. Migrations are defined in XML, YAML, JSON, or SQL changesets, each with an author and ID. This makes Liquibase well-suited to multi-developer teams and environments where migrations are generated rather than hand-written. Rollback is a first-class feature via `rollback` tags in changeset definitions. The tradeoff is verbosity.

```yaml
# Liquibase changeset with rollback defined
databaseChangeLog:
  - changeSet:
      id: 2026-06-22-add-payment-token
      author: hiep.tran
      changes:
        - addColumn:
            tableName: orders
            columns:
              - column:
                  name: payment_method_token
                  type: VARCHAR(255)
                  constraints:
                    nullable: true
      rollback:
        - dropColumn:
            tableName: orders
            columnName: payment_method_token
```

**Atlas** (by Ariga) is the modern declarative option. You describe the desired schema state in HCL or SQL, and Atlas computes the diff and generates a migration plan. It also includes a linter that catches dangerous migrations at planning time — it will warn you if a migration has a destructive change or will require a lock on a large table.

```bash
# Atlas: generate a migration from desired schema state
atlas schema diff \
  --from "postgres://localhost/app_db" \
  --to "file://schema.hcl" \
  --dev-url "postgres://localhost/atlas_dev"

# Atlas: lint a migration before applying
atlas migrate lint \
  --dev-url "postgres://localhost/atlas_dev" \
  --dir "file://migrations"
```

**golang-migrate** is the standard for Go services. It supports both versioned files and a simple `up`/`down` split per migration. No checksum verification by default (a deliberate tradeoff for simplicity), but it integrates cleanly into Go applications via library or CLI.

The choice between them matters less than the discipline: migrations live in the repository, are versioned, are never modified after they ship, and have a defined test path.

## Running Migrations in CI

The biggest improvement most teams can make to their migration process is the simplest: run migrations against a real database in CI for every pull request. Not H2. Not SQLite. A real Postgres or MySQL instance with your actual migration history applied.

The CI job needs to do four things:
1. Stand up a real database matching your production engine and version.
2. Apply all historical migrations from scratch (`flyway migrate` or equivalent).
3. Apply the new migration from this PR.
4. Verify the resulting schema is correct (column exists, correct type, indexes present).

The optional fifth thing — run the down migration and verify the schema returns to its prior state — is invaluable for catching rollback bugs before they become incidents.

### Full GitHub Actions Migration CI Workflow

The workflow below spins up a Postgres container, runs all Flyway migrations forward, validates the schema, then performs a rollback test using a `V{N}__undo_*` migration convention and validates the rolled-back state:

```yaml
# .github/workflows/migration-ci.yml
name: Database Migration CI

on:
  pull_request:
    paths:
      - "db/migrations/**"
      - ".github/workflows/migration-ci.yml"

jobs:
  migration-test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_DB: app_test
          POSTGRES_USER: app
          POSTGRES_PASSWORD: testpassword
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    env:
      FLYWAY_URL: jdbc:postgresql://localhost:5432/app_test
      FLYWAY_USER: app
      FLYWAY_PASSWORD: testpassword
      FLYWAY_LOCATIONS: filesystem:db/migrations

    steps:
      - uses: actions/checkout@v4

      - name: Install Flyway CLI
        run: |
          wget -qO- \
            https://repo1.maven.org/maven2/org/flywaydb/flyway-commandline/10.0.0/flyway-commandline-10.0.0-linux-x64.tar.gz \
            | tar -xz -C /opt
          echo "/opt/flyway-10.0.0" >> "${GITHUB_PATH}"

      - name: Apply all migrations (forward)
        run: flyway migrate

      - name: Validate — no pending migrations remain
        run: flyway validate

      - name: Verify target column exists with correct type
        run: |
          psql postgresql://app:testpassword@localhost:5432/app_test -c "
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'orders'
            ORDER BY ordinal_position;
          "

      - name: Assert new column is present
        run: |
          COUNT=$(psql -t -A \
            postgresql://app:testpassword@localhost:5432/app_test \
            -c "SELECT COUNT(*) FROM information_schema.columns
                WHERE table_name = 'orders'
                AND column_name = 'payment_method_token';")
          if [ "${COUNT}" != "1" ]; then
            echo "ERROR: column payment_method_token not found in orders table"
            exit 1
          fi
          echo "PASS: payment_method_token column exists"

      - name: Assert indexes are present
        run: |
          psql postgresql://app:testpassword@localhost:5432/app_test -c "
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'orders';
          "

      - name: Rollback — undo the last migration
        # Uses golang-migrate style down migrations stored in db/migrations/down/
        # Adjust this step to match your tool's rollback command
        run: |
          LAST_VERSION=$(flyway info -outputType=json \
            | python3 -c "import sys,json; \
                info=json.load(sys.stdin); \
                applied=[m for m in info['migrations'] if m['state']=='Success']; \
                print(applied[-1]['version'])" 2>/dev/null || echo "")
          if [ -n "${LAST_VERSION}" ] && [ -f "db/migrations/undo/U${LAST_VERSION}__*.sql" ]; then
            flyway undo -target="${LAST_VERSION}"
            echo "Rollback to version before ${LAST_VERSION}: DONE"
          else
            echo "No undo script for version ${LAST_VERSION} — skipping rollback test"
          fi

      - name: Validate schema after rollback
        run: |
          # After rollback, the column should no longer exist
          COUNT=$(psql -t -A \
            postgresql://app:testpassword@localhost:5432/app_test \
            -c "SELECT COUNT(*) FROM information_schema.columns
                WHERE table_name = 'orders'
                AND column_name = 'payment_method_token';")
          echo "Columns matching 'payment_method_token' after rollback: ${COUNT}"
          # If this is an expand migration, 0 is expected after undo

      - name: Re-apply migration to confirm idempotency
        run: |
          flyway migrate
          echo "Re-migration (idempotency check): PASS"
```

Teams that wire this gate onto every migration PR see a qualitative shift: migration failures start appearing in the PR review stage (where they are cheap and safe to fix) rather than in staging or production (where they cost hours of on-call time).

### The Migration Lock Problem in Kubernetes

A common failure mode specific to containerized deployments: multiple pods all try to run migrations on startup. You add a `migrate on startup` call to your application initialization, deploy with 3 replicas, and all three pods start simultaneously. All three call `flyway migrate`. Flyway acquires a lock on the `flyway_schema_history` table — but it acquires it differently per database. On Postgres, Flyway uses an advisory lock. On MySQL, it uses a table-level lock. Either way, two of the three pods will block waiting for the lock, or — worse — see that migrations have already run and proceed before the migration has actually finished.

The correct fix is to never run migrations inside the application process. Migrations belong in a Kubernetes `Job` or `initContainer` that runs exactly once per deploy, with `completions: 1` and `parallelism: 1`. The application `Deployment` declares the migration Job as a prerequisite via Argo Workflows, Flux's pre-deploy hooks, or a Helm hook:

```yaml
# Migration as a Kubernetes Job (Helm pre-upgrade hook)
apiVersion: batch/v1
kind: Job
metadata:
  name: "{{ .Release.Name }}-migration"
  annotations:
    "helm.sh/hook": pre-upgrade,pre-install
    "helm.sh/hook-weight": "-5"
    "helm.sh/hook-delete-policy": hook-succeeded
spec:
  completions: 1
  parallelism: 1
  backoffLimit: 3
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: migrate
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          command: ["flyway", "migrate"]
          env:
            - name: FLYWAY_URL
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: url
            - name: FLYWAY_USER
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: username
            - name: FLYWAY_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: password
```

With this approach, the Helm upgrade blocks until the Job pod exits with code 0, then Kubernetes proceeds with rolling out the updated Deployment. The migration runs once, exactly, before any new application pod starts. The parallel-startup race condition is eliminated by architecture.

### Checking Migration Status Before Deploy

Adding a pre-deploy status check ensures the current schema is exactly what the pipeline expects before it tries to apply new migrations. The check catches cases where someone applied an out-of-band hotfix migration directly in production without going through the pipeline:

```yaml
- name: Check migration status
  run: |
    STATUS=$(flyway info -outputType=json \
      | python3 -c "
    import sys, json
    info = json.load(sys.stdin)
    pending = [m for m in info['migrations'] if m['state'] == 'Pending']
    resolved = [m for m in info['migrations'] if m['state'] == 'Ignored']
    print(f'pending={len(pending)},resolved={len(resolved)}')
    ")
    PENDING=$(echo "${STATUS}" | cut -d= -f2 | cut -d, -f1)
    if [ "${PENDING}" -gt 0 ]; then
      echo "ERROR: ${PENDING} pending migrations. Run flyway migrate first."
      exit 1
    fi
    echo "Migration status: clean — no pending migrations"
```

## The Un-Rollback-able Migration

Not all migrations are reversible. Understanding which ones aren't — and planning accordingly — is the difference between a team that can roll back freely and a team that is always one bad deploy away from a multi-hour recovery.

![Un-rollback-able migrations: naive DROP COLUMN vs safe expand/contract](/imgs/blogs/database-migrations-in-the-delivery-pipeline-7.png)

### Category 1: Structural Irreversibility

These are migrations that destroy schema structure that was previously load-bearing:

**`DROP COLUMN`**: If you drop `legacy_payment_id` and any version of the app you might need to roll back to reads `legacy_payment_id`, the rollback will fail. The data is gone (or at best, in a backup you'd have to restore from).

**`DROP TABLE`**: Self-explanatory. The table and all its data are gone.

**`RENAME COLUMN` / `RENAME TABLE`**: The old name no longer exists. Old app code that references it by the old name will fail.

**`ALTER COLUMN` type narrowing**: If you change `payment_amount DECIMAL(15,4)` to `payment_amount DECIMAL(10,2)`, any value larger than what fits in the new type will be silently truncated or error. You cannot un-truncate data.

**NOT NULL without a default on an existing column**: The old application inserts a row without supplying the new column. The database rejects the insert. All old-app traffic on that code path throws a constraint violation error until the old app is rolled forward. There is no safe rollback path except reverting the migration — and if data has already been written by new-app instances, you now have to reconcile.

The pattern that breaks this is predictable and preventable. An engineer writes:

```sql
-- UNSAFE: do not apply this migration while old app code is still running
ALTER TABLE payments ADD COLUMN currency VARCHAR(3) NOT NULL DEFAULT 'USD';
```

On Postgres 11+, this particular example is actually safe because the constant default is stored in catalog without a table rewrite. But if the old application performs a `INSERT INTO payments (amount, user_id) VALUES (...)` without specifying `currency`, Postgres accepts it and fills in 'USD'. That part works. The danger is the case where the engineer forgets the `DEFAULT` clause:

```sql
-- CATASTROPHIC on a live system
ALTER TABLE payments ADD COLUMN currency VARCHAR(3) NOT NULL;
-- Every INSERT from old app code that omits 'currency' will fail immediately.
```

The safe path, always, is the three-step expand: add nullable, backfill, constrain.

### Category 2: Data Irreversibility

These are migrations where the migration modifies data in a way that cannot be reconstructed:

**Merging columns**: Taking `first_name` + `last_name` → `full_name` (via concatenation). Once you drop `first_name` and `last_name`, you cannot split `full_name` back reliably (what's the split point for "Mary Lou Retton"?).

**Hashing / encrypting in-place**: If you hash an email column as part of a privacy migration and drop the original, you cannot recover the original emails.

**Purging rows**: Any migration that deletes rows based on a condition is irreversible if those rows cannot be restored from a backup.

### How to Handle Un-Rollback-able Migrations

The strategy is to shift the point of irreversibility as far into the future as possible.

**For structural drops**: Use the expand/contract pattern. Never drop a column in the same release cycle where you stopped using it. Wait at least one full release cycle (or your defined retention period), confirm no running code references it, then drop. If your deployment artifact history goes back 30 days, your "contract" phase should not run for at least 30 days after the last app version that used the old column.

**For data transformations**: Keep the source data. If you're splitting a column, write the new columns and leave the original in place. Mark the original as `DEPRECATED` in comments. Only drop it after a defined retention period.

**For truly one-way migrations**: Document them explicitly in the migration file and in your change management system. Some teams add a comment header to irreversible migrations:

```sql
-- migration: V42__purge_old_sessions.sql
-- type: irreversible
-- reason: GDPR compliance purge, cannot restore deleted rows
-- approved-by: data-governance@company.com
-- rollback: none (requires full database restore from backup)
DELETE FROM user_sessions
WHERE created_at < NOW() - INTERVAL '90 days'
  AND is_authenticated = false;
```

**For NOT NULL constraints without a default**: Never add `NOT NULL` to an existing column in a single step. The sequence is:
1. Add the column as nullable.
2. Backfill all existing rows with the value they should have.
3. Add a `NOT NULL` constraint (after verifying no nulls remain).

```sql
-- WRONG: instant lock, breaks old app inserts
ALTER TABLE orders ADD COLUMN status VARCHAR(20) NOT NULL DEFAULT 'pending';

-- RIGHT: three steps
-- Step 1: Add nullable
ALTER TABLE orders ADD COLUMN status VARCHAR(20);

-- Step 2: Backfill (in batches if large table)
UPDATE orders SET status = 'legacy' WHERE status IS NULL;

-- Step 3: Add constraint after all rows have a value
ALTER TABLE orders ALTER COLUMN status SET NOT NULL;
```

In Postgres, the three-step version is also preferred for a different reason: adding `NOT NULL DEFAULT 'pending'` on large tables pre-Postgres 11 required a full table rewrite. On Postgres 11+, constant defaults are stored in catalog, but for anything computed you still need the three-step approach.

## Decoupling Migration Deploy from App Deploy

The most operationally mature pattern treats migrations and app deploys as completely separate pipeline stages with separate failure modes, separate monitors, and separate rollback paths.

Here is what the decoupled pipeline looks like in a Kubernetes environment:

```yaml
# GitHub Actions: decoupled migration + app deploy pipeline
name: Production Deploy

on:
  push:
    branches: [main]

jobs:
  # Stage 1: Migration CI (runs on every PR, referenced here for clarity)
  # Already validated on the PR - no need to re-run schema tests

  # Stage 2: Staging migration (separate job, separate failure domain)
  staging-migration:
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4

      - name: Run staging migration
        run: |
          flyway \
            -url="${{ secrets.STAGING_DB_URL }}" \
            -user="${{ secrets.STAGING_DB_USER }}" \
            -password="${{ secrets.STAGING_DB_PASSWORD }}" \
            -locations=filesystem:db/migrations \
            migrate

      - name: Validate staging schema
        run: |
          ./scripts/validate-schema.sh staging

  # Stage 3: Staging app deploy (only runs after staging migration succeeds)
  staging-app-deploy:
    needs: staging-migration
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to staging
        run: |
          kubectl set image deployment/app \
            app=${{ env.IMAGE_TAG }} \
            --namespace=staging
          kubectl rollout status deployment/app \
            --namespace=staging \
            --timeout=5m

  # Stage 4: Production migration (separate from app deploy)
  # This runs BEFORE the app deploy, ensuring schema is ready
  prod-migration:
    needs: [staging-app-deploy]
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4

      - name: Run production migration
        run: |
          flyway \
            -url="${{ secrets.PROD_DB_URL }}" \
            -user="${{ secrets.PROD_DB_URL }}" \
            -password="${{ secrets.PROD_DB_PASSWORD }}" \
            -locations=filesystem:db/migrations \
            migrate

      - name: Validate production schema
        run: |
          ./scripts/validate-schema.sh production

      - name: Post migration health check
        run: |
          # Check that the app still works after migration
          # (old app version, new schema — should still be compatible)
          ./scripts/smoke-test.sh production --pre-deploy

  # Stage 5: Production app deploy (only after prod migration succeeds)
  prod-app-deploy:
    needs: prod-migration
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy to production
        run: |
          kubectl set image deployment/app \
            app=${{ env.IMAGE_TAG }} \
            --namespace=production
          kubectl rollout status deployment/app \
            --namespace=production \
            --timeout=10m

      - name: Post-deploy smoke test
        run: |
          ./scripts/smoke-test.sh production --post-deploy
```

The critical structure here: `prod-migration` and `prod-app-deploy` are separate jobs with a dependency chain. If the migration fails, the pipeline halts before the app deploy begins. The current (old) app version continues running against the schema it already knows — which is exactly the situation the expand/contract pattern was designed to ensure is always safe.

The rollback procedure in this model is also clean. To roll back the app, you re-run the app deploy job with the previous image tag. The migration is not touched — it stays applied. Because your migrations were designed to be backward-compatible with the previous app version (expand phase), the old app continues to function correctly against the new schema.

To roll back the migration itself (a far rarer event, typically only needed if the migration has a performance problem or if it's in the expand phase and you discovered a mistake before any data wrote to the new column), you run the down migration as a separate one-off pipeline job — never as part of an automatic rollback of the app.

![Pipeline stages: migration CI through to app deploy](/imgs/blogs/database-migrations-in-the-delivery-pipeline-6.png)

## Migration Risk Decision Tree

Not every migration needs gh-ost, a multi-phase expand/contract, or even careful staging. The overhead of a full online migration tool on a table with 50,000 rows is unjustified. The risk decision tree helps you apply the right level of rigor without over-engineering.

![Migration risk decision tree](/imgs/blogs/database-migrations-in-the-delivery-pipeline-8.png)

The key thresholds in practice:

**Table size under 1 million rows**: A native `ALTER TABLE` typically completes in under 30 seconds. Lock time is acceptable for a momentary spike in query latency. Use your standard migration tool with a native SQL file. No additional tooling needed.

**Table size 1–10 million rows**: Borderline. Measure the expected lock time in staging. If it is under 5 seconds and your p99 SLA allows for it, native ALTER may be acceptable. Otherwise, use `CREATE INDEX CONCURRENTLY` for indexes and careful column additions with constant defaults.

**Table size 10 million–1 billion rows**: Always use an online schema change tool. gh-ost for MySQL, `CREATE INDEX CONCURRENTLY` and pg_repack for Postgres. Use the expand/contract pattern for any structural changes.

**Table size over 1 billion rows**: Same as above, plus: plan carefully for the cutover window, schedule gh-ost during off-peak traffic, use gh-ost's `--postpone-cut-over-flag-file` to control the exact cutover time, and have a runbook ready for the on-call engineer monitoring the migration.

A quick reference table for operational decisions:

| Table Size | ALTER TABLE | Online Tool | Expand/Contract |
|-----------|-------------|-------------|-----------------|
| < 100K rows | Yes, safe | Overkill | Optional |
| 100K–1M rows | Usually safe | Consider | Optional |
| 1M–50M rows | Risky | Recommended | Recommended |
| 50M–500M rows | Never | Required | Required |
| > 500M rows | Never | Required | Required + phased |

A second table for operation type risk:

| Operation | Risk Level | Strategy |
|-----------|-----------|----------|
| Add nullable column | Low | Native ALTER |
| Add NOT NULL + default | Low (Pg 11+) | Three-step |
| Create index | Medium | CONCURRENTLY |
| Rename column | High | Expand/contract |
| Change column type | High | Expand/contract + validate |
| DROP COLUMN | Very high | After contract phase only |
| DROP TABLE | Critical | After application fully migrated |
| Purge rows | Critical | Document, schedule, no rollback |

## Observing a Migration in Production

Running a migration is only half the job. The other half is watching it while it runs and knowing exactly what to do if something goes wrong. Teams that skip this end up discovering problems by watching error rates spike in their APM dashboards — by which point the migration may have already caused a cascade.

### Metrics to Watch During a Migration

**Replication lag** is the first signal to monitor on MySQL. When gh-ost is row-copying a large table, it generates a steady stream of write I/O that must replicate to every replica. If your replicas begin falling behind, reads against those replicas return stale data. Configure a dashboard panel that shows per-replica lag in real time and set an alert at 2 seconds — gh-ost's `--max-lag-millis` should be set to the same threshold so the tool self-throttles before the alert fires.

**Primary database CPU and I/O** should be tracked throughout. A healthy gh-ost migration on a well-tuned instance adds 10–20 percent I/O overhead. If you see CPU climbing toward saturation or I/O wait exceeding 60 percent, reduce the chunk size via the Unix socket interface (`echo "chunk-size=500" | nc -U /tmp/gh-ost.orders.sock`) before the database falls over.

**Application error rate by endpoint** is the leading indicator of migration-induced breakage. If a migration mistakenly drops or renames a column mid-deploy, the error rate on any endpoint that touches that column will jump immediately. Setting up a migration-phase monitor in your APM — a dashboard that shows error rates for the 10 most database-heavy endpoints during the 30-minute window around a migration — gives you instant correlation between a migration event and application behavior.

**Lock wait timeouts** accumulate when a migration or backfill holds a lock longer than expected. Track `pg_stat_activity` for queries in `waiting` state or MySQL's `SHOW ENGINE INNODB STATUS` output for `LOCK WAIT` lines. A spike in lock waits during a migration is a sign the batch size is too large or the sleep between batches is too short.

**Row copy progress** for gh-ost migrations can be tracked through the socket status command and exposed as a custom metric. Some teams write a small sidecar process that polls the socket every 30 seconds and publishes `migration_rows_copied` and `migration_rows_remaining` to their metrics system, so the estimated completion time is visible in the same dashboard as application latency.

### The Migration Runbook

Before running any migration on a table with more than 10 million rows, write a runbook. The runbook does not need to be long — it needs to cover five questions:

1. **What does this migration do?** One sentence. "Adds a nullable `currency` column to the `payments` table."
2. **How long is it expected to take?** Based on staging measurements with production-like data volume.
3. **What is the abort procedure?** For gh-ost: touch the panic flag file and describe what state the table is left in. For a native ALTER: there is no safe abort — confirm it will complete within the maintenance window before starting.
4. **What does success look like?** The specific schema assertion you will run after the migration completes to confirm the column exists with the right type and the right index.
5. **What does failure look like?** The specific error messages or metrics that indicate the migration is causing harm, and the threshold at which you abort versus let it continue.

Teams that write runbooks before migrations find that the discipline of answering those five questions surfaces risks they had not considered. The runbook is not a formality — it is the checklist that prevents the 3 AM call.

## Coordinating Migrations Across Multiple Services

Modern architectures often have multiple services sharing a database, or a migration that must coordinate across a service boundary. These cases require additional discipline beyond what a single-service pipeline provides.

### The Shared-Database Multi-Service Problem

When two services share a database schema — a pattern common in legacy monolith-to-microservices migrations — a schema change by Service A can break Service B, even if Service B's code was not changed. The expand/contract pattern still applies, but the contract phase now requires coordination with every service that reads the affected schema.

In practice: maintain an explicit schema ownership map in your repository. For every table, document which services read which columns. Before running a contract migration, send a changelog notification to the owners of every service that reads the affected columns. Only run the contract migration after receiving explicit sign-off that all dependent services have deployed code that no longer references the old column.

Some teams automate this with a migration linter that cross-references the migration against a service dependency map stored in the repository:

```yaml
# .schema-ownership.yaml — maps columns to consuming services
tables:
  payments:
    columns:
      payment_method:
        owned_by: payments-service
        read_by:
          - billing-service
          - reporting-service
      payment_instrument_type:
        owned_by: payments-service
        read_by:
          - billing-service
```

A CI script validates that any migration dropping `payment_method` has a corresponding pull request in `billing-service` and `reporting-service` that removes the reference, and that those PRs are already merged to main.

### Feature Flags and Migration Gates

For migrations that require a coordinated cutover — where the application behavior must switch at the same moment the schema changes — feature flags are the correct coordination mechanism. The flag gates the application-level switch; the migration is deployed independently.

The sequence looks like this. First, deploy the expand migration — the new column is added but the flag is off. Second, deploy the new application code with the feature flag gated to off — the code for reading the new column is deployed but not exercised. Third, run the backfill. Fourth, verify the backfill is complete. Fifth, flip the flag on — all new reads now use the new column. Sixth, after the next release cycle, run the contract migration.

This pattern decouples the schema change from the behavioral change at the application level while keeping the migration pipeline separate from the deploy pipeline.

## When NOT to Use Online Migration Tools

This is the section that gets skipped in most migration writeups, but it matters equally: you should not reach for gh-ost, pt-osc, or pg_repack by default on every migration.

**Small tables in dev and staging environments**: These tools add significant complexity and running time. A migration that takes 200 milliseconds as a native ALTER takes 3 minutes with gh-ost. In a dev environment where you run migrations on a fresh schema with no data, native ALTER is always correct.

**Small production tables**: Tables under a few hundred thousand rows almost always complete their native ALTER in well under a second. The replication lag, binlog streaming overhead, and shadow-table complexity of gh-ost are pure overhead. Reserve online tools for large tables.

**PostgreSQL-specific cases where native handles it**: Postgres 11+ handles many formerly-expensive operations natively. Adding a column with a constant default, adding NOT NULL with a pre-populated default, and creating indexes with CONCURRENTLY are all safe without external tooling. Know your Postgres version and its capabilities before reaching for pg_repack.

**Greenfield development tables**: If you're adding a new feature to a table that was created this sprint and has essentially no data, use a plain migration. Online tools exist for the case where you have a production table with millions of live rows. Pre-launch, you are not that table yet.

**When the tool's system requirements aren't met**: gh-ost requires specific MySQL configuration (row-based replication, SUPER privileges, and the ability to act as a replica). In managed cloud databases (AWS RDS, Google Cloud SQL, PlanetScale) some of these requirements are not always available. Know the limits of your environment before designing a migration runbook that depends on a tool you can't actually run.

**Frequent schema churn on a large table**: If the same table is being migrated repeatedly (say, weekly) by gh-ost, examine whether the table design is causing too much schema churn. Frequent online migrations are a symptom that the schema design may need attention — maybe via a JSON/JSONB extension column that can absorb new attributes without ALTER TABLE at all.

## Worked Examples

#### Worked example: Adding a non-nullable column to a 50-million-row table safely

The table is `payments` with 50 million rows in PostgreSQL. The business requirement is adding a `currency` column that must eventually be NOT NULL (every payment must have a currency) with no downtime.

**Step 1 — Write and merge the expand migration (Week 1, Sprint N)**

```sql
-- db/migrations/V031__expand_add_currency_nullable.sql
-- type: expand
-- table: payments (50M rows)
-- safe: nullable add, does not rewrite table in Postgres 11+
ALTER TABLE payments
  ADD COLUMN currency VARCHAR(3);

COMMENT ON COLUMN payments.currency IS
  'ISO 4217 currency code. NULL for rows predating 2026-06-22. '
  'Will be constrained NOT NULL after backfill in V033.';
```

Merge this migration. The CI job runs it against the Postgres container in the `migration-test` workflow and confirms the column exists, is nullable, and all existing rows have `NULL` for `currency`.

**Step 2 — Deploy the new app version (Week 1, Sprint N)**

The new app version writes `currency = 'USD'` (or whatever the real value is) on every new payment insert. Old app instances are still running during the rolling deploy and write `currency = NULL`. Both are accepted because the column is nullable.

```go
// New app: PaymentService.Create writes currency
func (s *PaymentService) Create(ctx context.Context, req CreatePaymentRequest) (*Payment, error) {
    _, err := s.db.ExecContext(ctx, `
        INSERT INTO payments (user_id, amount, currency, created_at)
        VALUES ($1, $2, $3, NOW())
    `, req.UserID, req.Amount, req.Currency)
    return nil, err
}
```

**Step 3 — Run the backfill job (Week 1–2, Sprint N)**

A separate migration job — not inline in the schema migration file — runs the batched update. It is triggered as a one-off Kubernetes Job after the deployment stabilizes:

```sql
-- Run as a job, not as a Flyway migration file
DO $$
DECLARE
  batch_size  INT    := 5000;
  last_id     BIGINT := 0;
  max_id      BIGINT;
BEGIN
  SELECT MAX(id) INTO max_id FROM payments;
  WHILE last_id < max_id LOOP
    UPDATE payments
    SET currency = 'USD'     -- default for legacy rows
    WHERE id > last_id
      AND id <= last_id + batch_size
      AND currency IS NULL;
    last_id := last_id + batch_size;
    PERFORM pg_sleep(0.1);
  END LOOP;
  RAISE NOTICE 'Backfill complete at id=%', max_id;
END $$;
```

At 5,000 rows per batch with a 100ms sleep, the backfill processes roughly 50,000 rows per second of wall time. 50 million rows takes approximately 17 minutes of elapsed time. The database sees short batched write bursts, never a sustained lock.

**Step 4 — Add the NOT NULL constraint (Week 2, Sprint N)**

Before running this migration, verify the backfill is 100% complete:

```sql
-- Confirm no NULLs remain before constraining
SELECT COUNT(*) FROM payments WHERE currency IS NULL;
-- Must return 0 before proceeding
```

Then apply the constraint migration:

```sql
-- db/migrations/V032__constrain_currency_not_null.sql
-- type: constraint (safe after backfill completes)
-- precondition: SELECT COUNT(*) FROM payments WHERE currency IS NULL = 0
ALTER TABLE payments
  ALTER COLUMN currency SET NOT NULL;

ALTER TABLE payments
  ADD CONSTRAINT chk_currency_iso
    CHECK (currency ~ '^[A-Z]{3}$');
```

In Postgres, adding `NOT NULL` to a column that genuinely has no NULLs requires only a brief scan to verify — it does not rewrite the table. On 50 million rows this completes in a few seconds.

**Total timeline**: 2 weeks. Zero downtime. Zero constraint violations on old app code. The `currency` column is non-nullable, validated, and indexed.

---

#### Worked example: Renaming a column without downtime

The table is `users` with 8 million rows. The column `user_name` needs to become `username` (removing the underscore). Renaming in-place with `ALTER TABLE users RENAME COLUMN user_name TO username` would break all running old-app instances instantly. The safe path spans three deployments:

**Deployment 1 — Expand: add new column**

```sql
-- db/migrations/V018__expand_add_username.sql
ALTER TABLE users ADD COLUMN username VARCHAR(100);

-- Populate from the existing column immediately for new rows
-- (existing rows backfilled separately)
CREATE INDEX CONCURRENTLY idx_users_username
  ON users (username)
  WHERE username IS NOT NULL;
```

**Application layer — dual-write during transition**

After Deployment 1, new app code reads from `user_name` (old) but writes to both:

```go
// Phase: dual-write. Both columns kept in sync.
func (s *UserService) UpdateName(ctx context.Context, id int64, name string) error {
    _, err := s.db.ExecContext(ctx, `
        UPDATE users
        SET user_name = $1,
            username  = $1
        WHERE id = $2
    `, name, id)
    return err
}
```

**Backfill — copy existing rows to new column**

```sql
-- Batched backfill (same pattern as above, omitted for brevity)
UPDATE users SET username = user_name WHERE username IS NULL;
```

**Deployment 2 — Switch reads to new column**

New app code now reads `username` (new column only) and still dual-writes both. This deployment confirms the new column is 100% reliable as the source of truth.

**Deployment 3 / Contract — Drop old column**

After Deployment 2 has been live for one full release cycle with no rollback requests, apply the contract migration:

```sql
-- db/migrations/V022__contract_drop_user_name.sql
-- type: irreversible (contract phase)
-- precondition: no code references 'user_name' in any running version
ALTER TABLE users DROP COLUMN user_name;
```

**Total deployments**: 3, spread over 2–3 weeks. Zero downtime. Zero errors during any overlap window. The rename is invisible to users and to the SLAs.

## War Story: GitHub's Online Migration at Scale

GitHub's engineering team published one of the most detailed accounts of running online migrations against truly massive MySQL tables. The story is instructive because it goes beyond the happy path.

In 2017, GitHub needed to add a column to their `repositories` table — a table that, at the time, had approximately 90 million rows and was one of the most write-heavy tables in their database cluster. The `repositories` table receives writes on every push, pull request creation, fork, and repository settings change. Running a native `ALTER TABLE` was never on the table — the lock would have cascaded into hours of site-wide downtime.

GitHub had already built gh-ost internally and was using it for most migrations. But the `repositories` table presented a specific challenge: replication lag. GitHub ran a primary-replica MySQL setup where replicas served read traffic. During a gh-ost migration, gh-ost needs to replicate its own writes (the row copies) through the binlog to keep replicas consistent. On a table with very high write throughput, this can cause replication lag on the replicas, which means reads start returning stale data.

Their solution was to use gh-ost's `--max-lag-millis` flag to automatically throttle the row copy rate when replication lag exceeded a threshold:

```bash
gh-ost \
  --table=repositories \
  --alter="ADD COLUMN has_wiki_enabled TINYINT(1) NOT NULL DEFAULT 0" \
  --chunk-size=1000 \
  --max-load="Threads_running=25" \
  --max-lag-millis=1500 \
  --replication-lag-query="SELECT max(Seconds_Behind_Master) FROM replication_status" \
  --postpone-cut-over-flag-file=/var/run/ghost.postpone \
  --execute
```

The `--max-lag-millis=1500` flag told gh-ost: if replication lag on any replica exceeds 1.5 seconds, pause the row copy until lag drops. The migration that might have taken 4 hours at full speed took 9 hours with throttling — but during those 9 hours, the site ran normally, replicas stayed within SLA, and the on-call engineer got to sleep.

The cutover itself — the atomic rename that swaps the shadow table with the real table — took approximately 800 milliseconds of lock time. That 800ms spike was visible in their metrics but below the 95th percentile request timeout, so no alerts fired. The migration was complete, zero customer impact.

What GitHub documented in their post-migration writeup was equally instructive: they ran gh-ost in "dry-run" mode for 48 hours before the actual migration to verify that the binlog streaming was working correctly, that the row copy rate was as expected, and that the estimated completion time matched their maintenance window preference. Dry-run mode runs everything except the final cutover — it is a full rehearsal.

```bash
# Dry run: validate everything except the final rename
gh-ost \
  --table=repositories \
  --alter="ADD COLUMN has_wiki_enabled TINYINT(1) NOT NULL DEFAULT 0" \
  --chunk-size=1000 \
  --max-load="Threads_running=25" \
  --dry-run
```

This discipline — rehearse the migration fully before committing to the cutover — is now standard practice at any organization running gh-ost on critical tables.

A similar story played out at a large payments platform adding a `currency` column to a 200-million-row `payments` table. The expand phase added the column as nullable and ran it through gh-ost without incident. The trouble came in the contract phase, six weeks later: an engineer merged the contract migration (dropping the old `payment_method` column) two days before the team confirmed that one internal billing service had not yet been updated to drop its reference to `payment_method`. The service began throwing `column "payment_method" does not exist` errors. The recovery required an emergency forward migration that re-added `payment_method` as a nullable column re-populated from `payment_instrument_type`, followed by an expedited release of the billing service, followed by the contract migration a second time. The lesson reinforced in the post-mortem: never run the contract phase without a cross-team grep-and-confirm that zero running services reference the old column name.

## Worked Examples — Pipeline Integration

#### Worked example: Stripe-style phased migration for a payments table

Put plainly, here is the full playbook for adding a `payment_instrument_type` enum to a 400-million-row transactions table in MySQL.

**Week 1: Expand phase**

Write and merge `V45__expand_payment_instrument_type.sql`:

```sql
-- V45__expand_payment_instrument_type.sql
-- type: expand (backward-compatible)
-- table: transactions (400M rows)
-- strategy: add nullable column, old app ignores it
ALTER TABLE transactions
  ADD COLUMN payment_instrument_type ENUM(
    'card', 'bank_transfer', 'wallet', 'crypto'
  ) DEFAULT NULL;
```

This runs via gh-ost in the pipeline:

```bash
gh-ost \
  --database=payments_production \
  --table=transactions \
  --alter="ADD COLUMN payment_instrument_type ENUM('card','bank_transfer','wallet','crypto') DEFAULT NULL" \
  --chunk-size=2000 \
  --max-load="Threads_running=30" \
  --postpone-cut-over-flag-file=/tmp/ghost.postpone.flag \
  --execute
```

**Week 1–2: Application deploy + backfill**

Deploy the new app version. It writes `payment_instrument_type` on every new transaction and reads it. A background job backfills existing rows:

```sql
-- Run in batches via a cron job
UPDATE transactions
SET payment_instrument_type =
  CASE
    WHEN payment_method LIKE 'card_%' THEN 'card'
    WHEN payment_method LIKE 'bank_%' THEN 'bank_transfer'
    WHEN payment_method = 'paypal'    THEN 'wallet'
    ELSE 'card'
  END
WHERE payment_instrument_type IS NULL
  AND id BETWEEN :start_id AND :end_id;
```

**Week 3: Add NOT NULL constraint after backfill complete**

```sql
-- V46__constrain_payment_instrument_type.sql
-- Verify no NULLs remain before running:
-- SELECT COUNT(*) FROM transactions WHERE payment_instrument_type IS NULL;
ALTER TABLE transactions
  MODIFY COLUMN payment_instrument_type ENUM(
    'card', 'bank_transfer', 'wallet', 'crypto'
  ) NOT NULL;
```

**Week 6+ (after two full release cycles): Contract phase**

```sql
-- V50__contract_remove_payment_method.sql
-- type: irreversible (contract phase)
-- precondition: confirmed no code references 'payment_method' column
ALTER TABLE transactions DROP COLUMN payment_method;
```

Total calendar time: 6 weeks. Total user-visible downtime: zero. The column is live, well-typed, indexed, and the old column is gone.

#### Worked example: PostgreSQL migration with Atlas CI linting

For a Postgres-backed app, the same discipline applies but the tooling is different. Here is how you wire Atlas into a CI gate to catch dangerous migrations automatically.

```bash
# Install Atlas
curl -sSf https://atlasgo.sh | sh

# Define the desired schema state in HCL
cat > schema.hcl << 'EOF'
table "orders" {
  schema = schema.public
  column "id" {
    null = false
    type = bigserial
  }
  column "user_id" {
    null = false
    type = bigint
  }
  column "payment_method_token" {
    null = true
    type = varchar(255)
  }
  column "status" {
    null = false
    type = varchar(20)
    default = "pending"
  }
  primary_key {
    columns = [column.id]
  }
}
EOF

# Generate the migration
atlas schema diff \
  --from "postgres://app:${DB_PASSWORD}@localhost:5432/app_db?sslmode=disable" \
  --to "file://schema.hcl" \
  --dev-url "postgres://app:${DB_PASSWORD}@localhost:5432/atlas_dev?sslmode=disable"

# Lint the generated migration for destructive changes
atlas migrate lint \
  --dev-url "postgres://app:${DB_PASSWORD}@localhost:5432/atlas_dev?sslmode=disable" \
  --dir "file://migrations" \
  --latest=1
```

Atlas's lint output catches issues like:

```
-- Analyzing changes from version 44 to 45 (1 change in total):

  -- analyzing version 45
    -- WARNING: Dropping non-virtual column "payment_method" from table "orders"
       This change is irreversible and may cause data loss.
       Use the "atlas:nolint" directive to suppress this warning if intentional.
    -- ok  (763µs)

  -------------------------
  -- 53.742µs
  -- 1 warning
```

This lint warning blocks the CI job by default, forcing the engineer to either restructure the migration as an expand/contract or explicitly suppress the warning with a justification.

## Migration Review Culture

The most durable improvements to migration safety come not from tooling but from review practices that catch dangerous patterns before code is merged. A migration review culture has three components.

**The migration PR review checklist.** Every migration PR should be reviewed against a short list of questions before approval: Is this migration backward-compatible with the previous app version? Does it take a table lock that will block writes? Does it modify data in a way that cannot be recovered? Is there a corresponding undo migration or documented rollback procedure? Is the CI migration test passing against a real Postgres or MySQL container? Teams that turn these questions into a PR template field get consistent answers without relying on any individual reviewer remembering to ask.

**The staging rehearsal requirement.** For any migration touching a table with more than 10 million rows, require that the migration runs fully in staging before it is approved for production. Staging should have a recent copy of production schema and a data volume that is representative — running migrations on a staging instance with 1,000 rows when production has 200 million rows gives you false confidence. The staging run should record the actual elapsed time, peak I/O wait, and any lock wait events, and those measurements should be attached to the production migration runbook.

**The post-migration review.** After every migration that required an online schema change tool or took more than 5 minutes, hold a 15-minute review. Not a post-mortem — just a review. What did the migration do? What was the actual elapsed time vs. the estimate? Were there any unexpected throttle pauses? Did replication lag exceed the alert threshold? Did error rates change? The answers feed into better estimates and better tooling choices for the next migration. Teams that skip this review tend to repeat the same misjudgments about migration duration and load impact.

## Key Takeaways

Schema migrations are not a deployment detail — they are a first-class delivery concern that requires its own tooling, its own pipeline stages, and its own operational discipline.

The idea is: a database schema and an application binary have different lifetimes and different rollback semantics. Treat them as separate artifacts with separate deploy pipelines. Every migration must be backward-compatible with the previous app version, and every app version must be forward-compatible with a schema that has already had its expand phase applied.

The five practices that matter most, in order of impact:

1. **Decouple migration from app deploy.** Run migrations as a separate pipeline stage before the app deploy. Never fuse them into a single step. A fused deploy-plus-migrate pipeline is a single point of failure that breaks two separate systems when either one goes wrong.

2. **Use the expand/contract pattern for structural changes.** Never rename or drop a column in the same release where you stop using it. Expand first, ship the new app, then contract after two release cycles. The gap is not extra ceremony — it is the window during which you can safely roll back the application without the schema becoming incompatible.

3. **Run migrations against a real database in CI.** H2 and SQLite are not substitutes. Postgres 15 or MySQL 8 in a CI service container, with your full migration history applied from scratch, is the only reliable test. The forward pass confirms the migration applies cleanly. The rollback pass confirms you can recover from a bad deploy. Both should be required before any migration merges.

4. **Use online schema change tools for large tables.** gh-ost for MySQL, native CONCURRENTLY for Postgres indexes, pg_repack when you need a full table rewrite. The threshold is roughly 1 million rows — below that, native ALTER is generally fine. Above it, the lock duration becomes unpredictable and the risk of blocking the entire write path for minutes is not acceptable in a production system with a live SLA.

5. **Document and quarantine irreversible migrations.** NOT NULL without default, DROP COLUMN, DROP TABLE, and data-transforming purges are one-way doors. Document them explicitly, get them approved, and never allow them to be part of an automatic rollback path. The document is not the protection — the review process is. The document simply ensures the review happens before the irreversible step, not after.

The teams that have eliminated migration-related incidents are not the teams that found a perfect tool. They are the teams that built a pipeline where every migration goes through the same repeatable, tested path — and where no engineer ever runs a bare `ALTER TABLE` on a production table that has more rows than a small town has people. The path matters more than the tool, and the discipline matters more than the path.

For the SRE perspective on handling migration-related incidents when they do occur despite best practices, see [Mitigate First, Diagnose Later](/blog/software-development/site-reliability-engineering/mitigate-first-diagnose-later). For the rollback companion to this post — what to do when a migration lands but the app deploy needs to roll back — see [Rollbacks and Recovering a Bad Deploy](/blog/software-development/ci-cd/rollbacks-and-recovering-a-bad-deploy). And for the broader context of how migrations fit into the full delivery pipeline, start with [From Commit to Production: The CI/CD Mental Model](/blog/software-development/ci-cd/from-commit-to-production-the-cicd-mental-model).

## Further Reading

- [gh-ost documentation and blog posts](https://github.com/github/gh-ost) — the canonical reference for GitHub's online schema change tool, including the engineering blog posts describing its internal design
- [Percona pt-online-schema-change documentation](https://docs.percona.com/percona-toolkit/pt-online-schema-change.html) — full reference for the trigger-based alternative to gh-ost
- [Atlas schema migration documentation](https://atlasgo.io/docs) — the declarative approach to schema management with built-in linting
- [Flyway documentation](https://documentation.red-gate.com/flyway) — the versioned migration tool used in most Java/JVM shops
- [Zero-downtime schema migrations](https://www.braintreepayments.com/blog/safe-operations-for-high-volume-postgresql/) — Braintree's blog post on PostgreSQL safe operations remains one of the best practical references for Postgres-specific migration patterns
- [Stripe's database reliability engineering](https://stripe.com/blog/service-deployment) — Stripe's engineering blog contains several posts on how they manage schema evolution at scale, though not always under a single title
- [Database Migrations Done Right](https://www.prisma.io/dataguide/types/relational/migration-strategies) — Prisma's data guide covers migration strategies across multiple database systems
