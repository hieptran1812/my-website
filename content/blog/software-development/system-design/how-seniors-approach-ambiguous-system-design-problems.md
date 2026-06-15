---
title: "How Seniors Approach an Ambiguous System Design Problem: A Repeatable Framework"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "The repeatable five-step loop a staff engineer runs on any vague design prompt — clarify, constrain, sketch, stress, iterate — worked end-to-end on a real problem, with every move naming the trade-off it buys."
tags:
  [
    "system-design",
    "architecture",
    "distributed-systems",
    "scalability",
    "trade-offs",
    "design-review",
    "requirements",
    "optimization",
    "senior-engineering",
  ]
category: "software-development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/how-seniors-approach-ambiguous-system-design-problems-1.webp"
---

There is a moment, about ninety seconds into a system design discussion, where you can tell who has actually shipped large systems and who has only read about them. The prompt is something deliberately vague — "design a service that shortens URLs," "design a notification system," "design Twitter's timeline" — and the junior engineer reaches for the whiteboard marker. Within the first two minutes they have drawn a load balancer, three boxes labeled "microservice," a Cassandra cluster, a Kafka topic, and an arrow to something called "the cache." They are designing. They have not yet asked a single question about what the system is supposed to do, who uses it, how much traffic it sees, or what counts as "working." They have answered a question nobody asked.

The senior engineer, on the same prompt, often does not touch the marker for several minutes. They ask: who's using this, internal or public? What's the read-to-write ratio? Is a redirect that's slow by 100 milliseconds a real problem or a non-problem? What happens if a link resolves to the wrong destination — embarrassing, or a security incident? How many links per day, and is that number going to 10× next year? Only after the shape of the *problem* is clear do they draw anything. And when they finally draw, the first sketch is almost insultingly simple — a web server, a key-value store, maybe a cache — because they know the interesting work is not the first design, it's the stress test that follows.

![The five-step loop a senior runs on any ambiguous prompt, from clarify through iterate](/imgs/blogs/how-seniors-approach-ambiguous-system-design-problems-1.webp)

This post is the opening chapter of a long series, and it is the most important one, because it sets the voice for everything after. It is not about any one technology. It is about the *loop* a senior runs on any ambiguous design prompt — the five repeatable moves shown above: **clarify, constrain, sketch, stress, iterate** — and the meta-skills wrapped around that loop: how to communicate a decision, how to say "it depends" without sounding evasive, how to avoid over-engineering, and what actually separates a junior answer from a staff answer on the *same* prompt. By the end you will have a checklist you can run on any design problem, a fully worked example of the loop applied to a concrete prompt, and a sharpened instinct for the one thing that matters most in senior-level design: every move you make should name what it costs.

## 1. Why ambiguity is the whole point

Design prompts are vague *on purpose*. In an interview, the vagueness tests whether you can impose structure on an open problem. On the job, the vagueness is real — the product manager who says "we need notifications" genuinely does not know yet whether that means transactional email, push notifications, an in-app inbox, or all three, and it is partly your job to help them find out. Treating the prompt as if it had one obvious answer is the first and most common mistake. There is no obvious answer. There are several reasonable designs, each correct under a different set of assumptions, and the senior move is to surface the assumptions before committing to a design.

The reason this matters is economic. The cost of a design mistake is not constant across the project — it grows by roughly an order of magnitude at each stage. A wrong assumption caught in the first five minutes of a whiteboard discussion costs you a sentence. The same wrong assumption caught after the schema is written costs a migration. Caught after launch, under load, it costs an incident, a postmortem, and a quarter of engineering time spent on a rebuild you could have avoided. Seniors are not smarter than juniors in some mystical way; they have simply been on the wrong side of that order-of-magnitude curve enough times to develop an allergy to designing before the problem is clear.

So the first principle of the whole series is this: **resist the urge to design.** The pull is strong because designing feels like progress and asking questions feels like stalling. It is the opposite. Asking the right three questions in the first two minutes is the highest-leverage thing you will do in the entire exercise. The diagram you draw at minute ten is only as good as the questions you asked at minute one.

We'll spend the rest of this post making that abstract advice concrete, because "ask good questions" is itself a vague prompt. The five-step loop is the structure that turns it into a procedure you can run cold.

## 2. The loop, in one paragraph

Here is the entire framework, compressed, so you have the shape before we expand each step. **Clarify**: turn the vague ask into a crisp scope, a list of actors, and the *one* metric that defines success — then stop and confirm it. **Constrain**: extract functional requirements (what it does) and non-functional requirements (how well — latency, availability, durability, consistency), attach real numbers to the scale, and write down the SLOs and the budget. **Sketch**: propose the simplest architecture that could possibly work, naming each component and the data flow between them, and explicitly *not* adding anything the requirements don't demand. **Stress**: attack your own design — what breaks at 10× traffic, at a region outage, at a hot key, at a thundering herd of retries — and find the single biggest bottleneck. **Iterate**: change the design to remove that bottleneck, and for every change, say out loud what trade-off the change buys and what it costs. Then loop back to stress and repeat until the design survives the failures you actually care about.

That's it. Five words. The difference between a junior and a senior is not knowledge of more boxes; it is the discipline to run those five steps in order and to refuse to skip the first two even when the pull to start drawing is overwhelming.

The rest of this post is each step in detail, followed by a complete worked run of the loop on a real prompt, then the meta-skills — communication, "it depends," over-engineering, and the explicit junior-versus-senior contrast — and finally case studies, a when-to-use section, takeaways, and further reading. Let's go.

## 3. Step 1 — Clarify: turn the ask into scope, actors, and one metric

The single most useful output of the clarify step is a sentence you and your audience both agree on, of the form: *"We are building X for Y so that Z, and we will know it's working if M."* X is the scope, Y is the actor, Z is the purpose, and M is the one metric that matters. If you cannot fill in that sentence, you are not ready to design, and any diagram you draw is decoration.

Take "design a URL shortener." That phrase hides at least five different products. A public link shortener like Bitly serves billions of redirects from a huge, long-tailed link corpus, cares enormously about redirect latency and abuse, and barely cares about write throughput. An internal shortener for a company's Slack and email links has a tiny corpus, trivial traffic, and cares mostly about not breaking existing links. A shortener that also does click analytics has a write-heavy fan-out on every redirect and a whole analytics pipeline bolted on. A shortener with custom vanity domains has a multi-tenant routing problem. A shortener that must support link expiration and editing has a mutable-data problem the others don't. Same four words; five completely different systems with different bottlenecks. The clarify step is where you find out which one you're actually being asked to build.

The actors matter as much as the scope. *Who* calls this system shapes the entire design. End users hitting a redirect want sub-50-millisecond latency and don't care about consistency. Internal services creating links want a clean API and idempotency. An abuse team wants to be able to disable a link instantly and globally. A billing system wants accurate per-tenant click counts. Each actor implies requirements, and the requirements conflict — instant global disable fights against aggressive caching, for instance. Naming the actors up front means you discover the conflicts now, on the whiteboard, instead of in production when the abuse team can't take down a phishing link because it's cached at the edge for an hour.

And then the one metric. This is the part juniors skip and seniors obsess over. Every system has many things you *could* optimize — latency, throughput, cost, availability, consistency, developer velocity — but in any given design there is usually one that dominates, the one where being wrong means the project failed. For a public redirect service it's redirect-path p99 latency and availability; the thing must be fast and must basically never be down, because a dead redirect breaks every link anyone ever shared. For an internal analytics shortener, the dominant metric might be click-count accuracy, because the numbers feed billing. Naming the one metric does two things: it tells you where to spend your design budget, and it tells you what you're *allowed to sacrifice*. You cannot optimize everything. The senior who knows the one metric knows which corners are safe to cut.

A practical tactic here: ask numeric questions, not yes/no questions. "Is it high traffic?" gets you "yeah, pretty high," which is useless. "Roughly how many redirects per second at peak — hundreds, thousands, or millions?" gets you an order of magnitude you can design against. You are not looking for precision; you are looking for the exponent. Whether it's 3,000 or 7,000 QPS rarely changes the architecture. Whether it's 3,000 or 3,000,000 changes everything.

#### Worked example: clarifying "design a notification system"

Suppose the prompt is "design a notification system" and you get to ask questions. Here is the clarify conversation a senior runs, with the answers that shape the design:

- **What channels?** "Push, email, and SMS." → Three delivery integrations, each with different latency, cost, and failure semantics. SMS costs real money per message (~\$0.0075 each on a US long code), so cost becomes a design axis email never forces.
- **Who triggers a notification?** "Other internal services — order shipped, password reset, marketing blasts." → Two very different traffic shapes: trickle of transactional events plus occasional huge marketing bursts. The marketing burst is the hard case.
- **What's the volume?** "Maybe 50 million notifications a day, but a marketing campaign can fire 10 million in an hour." → ~580/s average, but ~2,800/s sustained during a campaign, with a burst spike that could be 10× that for a few minutes. The peak, not the average, sizes the system.
- **What's the one metric?** After discussion: for transactional notifications it's *delivery latency* (a password reset email arriving 5 minutes late is a support ticket); for marketing it's *throughput and cost* (nobody cares if a promo arrives 3 minutes late, but a duplicate is embarrassing and a per-message cost overrun is a budget problem).
- **What's the cost of failure?** "A dropped password reset is bad; a duplicate marketing email is mildly bad; a dropped marketing email is fine." → This single answer tells you transactional needs at-least-once with strong delivery tracking, while marketing can tolerate best-effort. Two different reliability tiers in one system.

Notice that we have not drawn anything. But we already know the system has two traffic classes with different SLOs, a cost axis driven by SMS, a burst problem driven by campaigns, and a tiered reliability model. The design almost writes itself once the problem is this clear — and crucially, a *different* set of answers would have produced a different design. That is the whole point of clarifying.

## 4. Step 2 — Constrain: requirements, scale numbers, SLOs, and budget

Clarify gives you the shape of the problem. Constrain gives you the *numbers*, and numbers are what turn a hand-wave into an architecture. This step has four outputs: functional requirements, non-functional requirements, scale estimates, and a budget.

**Functional requirements** are what the system does, stated as capabilities, not implementations. "Create a short link from a long URL." "Redirect a short link to its long URL." "Optionally expire a link after a date." "Optionally record a click event." Keep these in the language of the domain, not the language of databases — "redirect a short link" not "look up a row in Postgres." The implementation is the *sketch* step's job; mixing it in here is how juniors accidentally over-commit to a technology before they've justified it.

**Non-functional requirements** are the *how well* — and this is where systems are actually won or lost. The big five are latency, throughput, availability, durability, and consistency. For each one, you want a target. Latency: redirect p99 under 50ms. Throughput: 10,000 redirects/s sustained, 50,000/s peak. Availability: 99.99% (about 52 minutes of downtime a year). Durability: a created link must never be lost (losing a link breaks every place it was shared). Consistency: a freshly created link must be resolvable within, say, a few seconds globally — but we can tolerate that small window, which is a huge freedom we'll exploit later. The act of writing these down forces conversations you'd otherwise have in production. The moment you write "99.99% availability" next to "redirect path," you've committed to multi-AZ redundancy, health checks, and graceful degradation — and you can decide whether that cost is justified *now*, on purpose.

This is exactly the territory the next two posts in this series go deep on. Turning a vague ask into a crisp set of functional and non-functional requirements with real SLOs is its own discipline, covered in [turning vague asks into requirements and SLOs](/blog/software-development/system-design/turning-vague-asks-into-requirements-and-slos). Attaching believable numbers to scale — QPS, storage, bandwidth — is the back-of-the-envelope skill covered in [back-of-the-envelope estimation for system design](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design). In this post we'll do enough of both to run the loop end-to-end, but those two are the deep dives.

![The same prompt produces a junior answer of technologies and a senior answer of constraints and trade-offs](/imgs/blogs/how-seniors-approach-ambiguous-system-design-problems-2.webp)

**Scale estimates** are the back-of-the-envelope math. You do not need precision; you need the right power of ten, because the power of ten is what selects the architecture. A few rules of thumb that earn their keep: a single modern server handles low tens of thousands of simple requests per second; a single Postgres instance handles a few thousand write transactions per second before you sweat; a gigabit link moves ~125 MB/s; a year is ~31.5 million seconds; "one event per user per day" for 100 million users is ~1,150 events/second average and maybe 5–10× that at peak. Memorize a handful of these and you can size a system in your head while talking.

**Budget** is the constraint juniors forget entirely and seniors treat as a first-class requirement. Every architecture has a dollar cost, and "technically correct but costs \$200,000 a month for a side feature" is a failed design. A senior estimates cost the same way they estimate latency: roughly, with order-of-magnitude numbers. Storing 100 billion link records at ~100 bytes each is ~10 TB, which on object storage is on the order of \$200/month and on a replicated SSD-backed database is more like \$2,000–\$5,000/month — a 10–25× difference that should drive a real decision, not an afterthought. The senior names the budget early because it eliminates whole branches of the design tree before they're drawn.

The output of constrain is a short table you can point at for the rest of the discussion. Here's the one for our URL shortener, which we'll carry into the sketch:

| Dimension | Target | Why it matters |
| --- | --- | --- |
| Read QPS | 10k sustained, 50k peak | Sizes the read path and cache |
| Write QPS | ~200 sustained | Tiny; writes are not the problem |
| Read:write ratio | ~100:1 (often 1000:1) | Optimize reads, not writes |
| Redirect p99 | < 50 ms | The dominant metric |
| Availability | 99.99% | A dead redirect breaks every link |
| Durability | No lost links, ever | Losing a link is unrecoverable |
| Consistency | New link visible in seconds | A freedom we'll exploit |
| Storage (5 yr) | ~10 TB | Cheap; not a constraint |

Look at what that table already tells us before we've drawn a single box. Reads dominate writes by 100:1 or more, so we optimize the read path and the write path can be almost naive. The dominant metric is redirect latency, so caching is going to be central. We can tolerate a few seconds of inconsistency on new links, which means we don't need strong global consistency — a massive cost saving. And storage is trivially cheap, so we won't contort the design to save bytes. The constraints have eliminated most of the design space. *That* is why seniors spend time here.

## 5. Step 3 — Sketch: the simplest thing that could possibly work

Now, and only now, you draw. The discipline of the sketch step is restraint: propose the *simplest* architecture that satisfies the requirements you just wrote down, and not one component more. Every box you add is a box that can fail, needs monitoring, costs money, and has to be explained. The senior's first sketch is deliberately boring. If a single web server plus a single database would meet the requirements, that's the sketch — and the senior is happy to draw exactly that, because the next step (stress) is what reveals whether you actually need more.

For our URL shortener, the simplest thing that could possibly work is genuinely small: a stateless web service behind a load balancer, a key-value store mapping short code to long URL, and a cache in front of the store for the hot reads. Two paths through it: the *write path* (someone creates a link — generate a unique short code, store the mapping) and the *read path* (someone hits a short link — look up the code, return a 301/302 redirect). That's it. We map the read and write paths separately because they have wildly different traffic and wildly different requirements, and separating them lets each scale on its own dimension.

![The simplest URL shortener architecture separating the write path from the cache-fronted read path](/imgs/blogs/how-seniors-approach-ambiguous-system-design-problems-5.webp)

Notice a few senior moves embedded in that boring sketch. First, the service is *stateless*, so we can run many copies behind the load balancer and scale horizontally just by adding instances — a property we got for free by keeping all state in the store. Second, the cache sits on the read path only, because the read path is where the dominant metric (redirect latency) lives and where the 100:1 ratio makes caching pay off enormously. Third, the write path talks directly to the store with no cache, because writes are rare and we don't want stale-cache bugs on link creation. Each of these is a small decision, and each one already names a trade-off: stateless service trades a little request overhead (every request re-fetches state) for trivial horizontal scaling; read-path cache trades possible staleness for latency; cacheless writes trade write latency for correctness on creation.

The one genuinely interesting design decision in the sketch is how to generate the short code, because it's the only place the write path has real choices, and each choice buys something different:

```python
# Option A: random short code, check-and-retry on collision.
# Buys: no coordination, no central counter. Costs: a read before
# every write to check for collision, and rare retries.
import secrets, string

ALPHABET = string.ascii_letters + string.digits  # 62 chars

def make_code(length: int = 7) -> str:
    # 62^7 ~= 3.5 trillion codes; collisions are astronomically rare
    return "".join(secrets.choice(ALPHABET) for _ in range(length))

def create_link(long_url: str, store) -> str:
    for _ in range(5):                # bounded retry
        code = make_code()
        if store.put_if_absent(code, long_url):   # atomic insert
            return code
    raise RuntimeError("collision retries exhausted")  # ~never happens
```

```python
# Option B: monotonic counter, base62-encode it.
# Buys: guaranteed no collisions, dense codes. Costs: a single
# global sequence everyone contends on -- the future bottleneck.
def base62(n: int) -> str:
    alphabet = string.ascii_letters + string.digits
    if n == 0:
        return alphabet[0]
    out = []
    while n:
        n, rem = divmod(n, 62)
        out.append(alphabet[rem])
    return "".join(reversed(out))

def create_link_counter(long_url: str, store, sequence) -> str:
    n = sequence.next()        # <-- the central sequence: looks fine now
    code = base62(n)
    store.put(code, long_url)
    return code
```

A junior picks one of these and moves on. A senior writes both down, names what each buys, and — critically — *flags* that Option B introduces a central sequence that every write contends on. That flag is a gift to the stress step. We've planted the seed of the bottleneck where we can see it, on purpose. The random approach (A) scales writes trivially but pays a read-before-write; the counter approach (B) gives dense sequential codes but creates a coordination point. We'll deliberately pick B for the first sketch precisely so the stress step has something to break — which mirrors what happens on real projects, where the "simple" central counter looks fine at launch and falls over at scale.

Resist the temptation, here, to pre-solve problems you don't have yet. You might already be itching to add a CDN, a write-ahead queue, multi-region replication, a separate analytics pipeline. Don't. None of those are justified by anything in the requirements table *yet*. Adding them now is over-engineering, and over-engineering is a senior failure mode just as real as under-engineering — it's just less visible because the system "works." We add complexity only when the stress step proves we need it, and when we do, we'll know exactly what trade-off it buys. The simplest sketch is not naivety; it's a baseline you'll evolve with intent.

## 6. Step 4 — Stress: attack your own design

This is the step that most distinguishes a senior. Having drawn the simplest design, the senior immediately turns on it and tries to break it. The question is always some variation of: *what happens when this gets harder than the happy path?* And there are four canonical "harder" axes worth running on every design: 10× scale, a region or AZ outage, a hot key, and a thundering herd of retries. Run those four and you'll find the bottleneck in almost any system.

![A design review timeline showing the early minutes spent on the problem and the close on the trade-off](/imgs/blogs/how-seniors-approach-ambiguous-system-design-problems-4.webp)

**What breaks at 10×?** Our requirements said 10k reads/s and 200 writes/s. Multiply by ten: 100k reads/s and 2,000 writes/s. On the read path, the cache absorbs most of it — if 95% of reads hit cache, the store only sees 5k reads/s even at 100k total, which is fine. But the write path is where the seed we planted blooms: that central sequence from Option B now has to mint 2,000 codes per second, every one of them serialized through a single coordination point. A single sequence can do that, but it's a single point of contention, and at 50,000 writes/s (another 25×, the kind of number a viral campaign produces) it falls over. The central counter is the bottleneck. We found it because we ran 10× — twice.

**What breaks at a region outage?** Our sketch has one of everything in one region. If that region goes down, every link in the world stops resolving. For a service whose dominant metric is availability, that's a catastrophic failure mode, and the requirements table demanding 99.99% just made it non-negotiable. The single-region design fails the availability requirement the moment we take it seriously. We need replication across regions — which is exactly where the mechanism deep-dives become relevant, and we'll cross-link rather than re-derive how replication works.

**What breaks at a hot key?** Suppose one short link goes viral — a celebrity tweets it — and it's now 40% of all redirect traffic. That single key, if it lives on one cache node and one database shard, melts that node while every other node sits idle. The hot key is a classic distributed-systems failure that uniform sharding does nothing to fix, because the load isn't uniform. We'll need a strategy for hot keys (replicate the hot entry across all cache nodes, or push it to the CDN edge) that the simple sketch doesn't have.

**What breaks under a thundering herd?** Imagine that viral link's cache entry expires at the exact moment it's getting 40,000 requests/second. All 40,000 of those requests miss the cache simultaneously and stampede the database for the same key — a thundering herd, also called a cache stampede. The database, sized for 5k reads/s, gets hit with 40k requests for one key and falls over, which evicts more cache, which causes more misses, which is a cascading failure. We found a failure mode that only appears at the intersection of a hot key and a TTL expiry — exactly the kind of second-order interaction the stress step exists to surface.

![Stress-testing at ten times traffic exposes the central sequence as the single write-path bottleneck](/imgs/blogs/how-seniors-approach-ambiguous-system-design-problems-8.webp)

There's a fifth axis worth running on systems that handle money or coordinate state: **what breaks under concurrency?** Two users create a custom vanity link with the same code at the same instant; two requests increment the same click counter simultaneously; a write and a delete race on the same key. The happy path assumes operations happen one at a time, and they never do. For our shortener the concurrency hazard is the `put_if_absent` on code creation — if it isn't genuinely atomic, two simultaneous creates can both think they won the code and one silently overwrites the other, resolving the link to the wrong URL, which we said in the clarify step was a security issue. The stress question "what breaks under concurrency?" forces you to find the places where the design quietly assumes serialization it doesn't actually have. A senior runs this axis on anything with mutable shared state, because concurrency bugs are the ones that pass every test (which run single-threaded) and fail in production (which doesn't).

And a sixth, for anything that depends on another service: **what breaks when a dependency is slow, not down?** A dependency that's *down* is easy — you get a fast error and fail over. A dependency that's *slow* — responding in 5 seconds instead of 5 milliseconds — is far more dangerous, because your threads block waiting for it, your connection pool drains, requests queue, and the slowness propagates upstream into a cascading failure that takes down services that don't even use the slow dependency directly. This is the failure mode behind a huge fraction of real outages, and the fix (timeouts, circuit breakers, bulkheads) is something the simple sketch never has. The senior asks "what's my slowest dependency and what happens when it gets slower?" precisely because the answer is almost never "nothing" and is usually "everything."

The output of the stress step is not a fixed design — it's a *ranked list of failure modes*, with the single biggest bottleneck at the top. In our case the central sequence is the write-path bottleneck, and the hot-key/thundering-herd interaction is the read-path bottleneck. We don't try to fix everything at once. We rank, we pick the biggest, and we hand it to the iterate step. The ranking itself is a senior skill: a failure mode that's *likely and severe* (the hot key on a viral link) ranks above one that's *rare and survivable* (a brief AZ blip the load balancer already routes around). You weight by probability times blast radius, fix the top of the list, and explicitly defer the bottom of it with a note rather than pretending it doesn't exist. Stress-testing your own design before someone else's traffic does it for you is the single most senior habit in this entire post.

## 7. Step 5 — Iterate: remove the bottleneck, name the trade-off

Now we evolve. The rule of the iterate step is strict and it's the spine of this whole series: **every change you make names the trade-off it buys.** You never just "add a queue" or "add a cache." You add a queue *to absorb write bursts, paying with eventual-consistency on the create path and the operational cost of running a broker.* You add a cache *to cut read latency, paying with possible staleness and a cache-invalidation problem.* If you can't name what a change costs, you don't understand the change well enough to make it. This is the discipline that turns architecture from cargo-culting boxes into engineering.

Let's iterate on our two bottlenecks.

**Bottleneck 1: the central sequence.** The fix is to stop serializing key generation through a single point. The cleanest move is *ranged allocation*: a central allocator hands each write host a block of, say, 10,000 sequence numbers at a time, and the host mints codes from its local block with no further coordination. The host only talks to the allocator once per 10,000 writes instead of once per write — a 10,000× reduction in contention. What does this buy and cost? It buys near-linear write scaling and removes the bottleneck entirely. It costs *gaps in the sequence* (a host that crashes with 3,000 unused numbers in its block leaves a hole) and *non-monotonic global ordering* (host A and host B mint codes in interleaved ranges). For a URL shortener, both costs are completely acceptable — nobody cares that code `aB3k9` was issued before `aB2x1`, and gaps in a 3.5-trillion code space are irrelevant. We pay in properties we don't need to buy a property we do. That is the trade-off, named.

![Stress-testing exposes the central sequence bottleneck, and ranged allocation removes it by handing each host a local key range](/imgs/blogs/how-seniors-approach-ambiguous-system-design-problems-8.webp)

This pattern — replace a single coordination point with pre-allocated local ranges — recurs constantly in distributed design. It's the same idea behind how databases hand out auto-increment ranges to shards, and it connects directly to the broader topic of [partitioning and sharding without downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime), which goes deep on splitting a write path. We name the pattern, cross-link the deep dive, and move on — the architect's job is to *choose* the pattern and know its cost, not to re-derive the mechanism every time.

**Bottleneck 2: the hot key and the thundering herd.** This is a read-path problem and it has a layered fix. First, for the hot key itself, we replicate the hot entry across all cache nodes (or, better, push wildly popular links to the CDN edge, where they're served without ever touching our infrastructure). That buys the ability to absorb a viral link's traffic across many nodes, paying with a little extra memory and the need to detect which keys are hot. Second, for the thundering herd at TTL expiry, we add *request coalescing* (also called single-flight): when 40,000 requests miss the cache for the same key simultaneously, only *one* of them is allowed to go to the database; the other 39,999 wait for that one result and share it. This buys protection against the stampede, paying with a tiny bit of added latency for the waiters and a more complex cache layer. Here's the shape of it:

```go
// Single-flight: collapse a stampede of identical cache misses
// into ONE database read. Buys herd protection; costs a mutex
// and a little latency for the followers.
type Group struct {
    mu    sync.Mutex
    calls map[string]*call
}
type call struct {
    wg  sync.WaitGroup
    val string
    err error
}

func (g *Group) Do(key string, fn func() (string, error)) (string, error) {
    g.mu.Lock()
    if c, ok := g.calls[key]; ok {   // a fetch is already in flight
        g.mu.Unlock()
        c.wg.Wait()                  // wait for the leader, don't pile on
        return c.val, c.err
    }
    c := &call{}
    c.wg.Add(1)
    g.calls[key] = c                 // we are the leader for this key
    g.mu.Unlock()

    c.val, c.err = fn()              // the ONE real DB read
    c.wg.Done()

    g.mu.Lock()
    delete(g.calls, key)
    g.mu.Unlock()
    return c.val, c.err
}
```

We could also avoid the synchronized-expiry problem entirely by adding jitter to TTLs (so cache entries don't all expire at the same instant) and by refreshing hot entries *before* they expire rather than after. Each of these is a small change that names a trade-off: jitter buys herd avoidance at the cost of slightly less predictable cache freshness; proactive refresh buys zero-miss hot keys at the cost of some wasted refreshes on keys that would have been fine. Caching is deceptively deep and full of these traps — the full treatment is in [caching strategies and the pitfalls that bite](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite). For now, the senior move is to *name* the herd risk and apply the standard mitigations, not to discover cache stampedes in production.

**Bottleneck 3: the region outage.** To hit 99.99%, we replicate across regions. The read path is easy to make multi-region because it's read-mostly and we already tolerate seconds of staleness — we put read replicas in each region and serve redirects locally, accepting that a brand-new link might take a second or two to propagate. That tolerance, which we deliberately negotiated for in the constrain step, is exactly what makes multi-region cheap here. The write path is harder: do we have one global write leader (simple, strongly consistent, but writes from far regions are slow and the leader's region is a single point of failure) or multi-leader writes (fast local writes everywhere, but now we have to resolve conflicts)? This is a genuine CAP/PACELC decision, and which way you go depends entirely on the consistency requirement you wrote down. Because we negotiated "new link visible in seconds" rather than "immediately," we can pick the simpler single-leader-with-async-replicas design and accept the small staleness window — exactly the kind of decision covered in [consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) and the trade-off framing in [the CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc). The architect's job here is not to re-derive CAP; it's to know which requirement makes the choice for you.

After this round of iteration, we loop back to stress: does the new design break anywhere? The ranged allocator is now a smaller single point of failure (it only matters for *new* writes, and we can make it highly available with standard leader election); the multi-region read path survives a region outage; the hot-key and herd mitigations protect the read path. We've removed the top bottlenecks and named every trade-off we bought along the way. This is what "iterate" means — not endless tinkering, but a disciplined cycle of break, fix the biggest thing, name the cost, repeat.

## 8. The full worked run, start to finish

Let me put the whole loop together on one prompt without interruption, so you can see the *rhythm* of it — because the rhythm is the skill. The prompt: **"Design a service that shortens URLs."**

**Clarify (2 minutes).** "Public or internal?" Public. "Read-heavy or write-heavy?" Massively read-heavy, people create a link once and it's clicked thousands of times. "What's the one metric?" Redirect latency and availability — a slow or dead redirect breaks every link anyone ever shared. "Do we need analytics, expiration, custom domains?" Basic click counts are nice-to-have, expiration is out of scope for v1, custom domains out of scope. "What's the cost of resolving to the wrong URL?" High — it's a security and trust issue, so correctness on the mapping is non-negotiable even though we'll relax consistency on *freshness*. **Agreed sentence:** "We're building a public URL shortener for end users so that shared links resolve fast and reliably, and we'll know it works if redirect p99 is under 50ms at four-nines availability."

**Constrain (3 minutes).** Functional: create a link, resolve a link, count clicks (best-effort). Non-functional from the table in §4: 10k reads/s sustained (50k peak), ~200 writes/s, 100:1 read:write, redirect p99 < 50ms, 99.99% availability, no lost links, new links visible in seconds, ~10 TB over five years. Budget: this is core infrastructure but not the company's main product, so target low thousands of dollars a month, not tens of thousands. The constraints tell us: optimize reads, caching is central, we can relax freshness consistency, storage is cheap, and the write path can be simple.

**Sketch (5 minutes).** Stateless read/write service behind an L7 load balancer, a key-value store for code→URL, a Redis cache on the read path, and base62-encoded sequential codes from a central counter (chosen *specifically* so we can break it). Two paths: write mints a code and stores the mapping; read looks up the cache, falls back to the store on miss, returns a 302.

![The redirect latency budget shows the cold-miss datastore read dominating the fifty millisecond target](/imgs/blogs/how-seniors-approach-ambiguous-system-design-problems-7.webp)

Before stressing, the senior sanity-checks the latency budget, because the dominant metric is latency and you can't optimize what you haven't accounted for. The budget above shows where the 50ms goes: DNS and TLS setup, edge, load balancer, service, and the data lookup. The cache hit costs about 3ms; the cold database miss costs about 25ms. That single comparison tells you precisely where the optimization budget should go — the cache hit rate is the lever, because the gap between a 3ms hit and a 25ms miss is where the p99 lives. If we can push the hit rate from 90% to 99%, we move the p99 off the database entirely. Optimization is not "make everything faster"; it's "find the line item that dominates the budget and attack only that one." Here it's cache misses, full stop.

**Stress (5 minutes).** Run the four axes from §6. At 10× the central counter is the write bottleneck. At a region outage the single-region design dies and fails our availability requirement. A viral link is a hot key that melts one cache node. A TTL expiry on that hot link is a thundering herd that stampedes the database. Ranked bottleneck list: (1) central counter, (2) hot-key/herd on reads, (3) single-region availability.

**Iterate (8 minutes).** Replace the central counter with ranged allocation (buys linear write scaling, costs sequence gaps and non-monotonic ordering — both fine here). Add hot-key replication plus single-flight request coalescing plus TTL jitter (buys herd protection, costs cache-layer complexity and a little latency for followers). Add multi-region async read replicas with a single write leader (buys region-outage survival, costs a few seconds of new-link staleness — which we pre-negotiated, so it's free). Loop back: the design now survives all four stress axes, every change has a named trade-off, and we never added a single component the requirements didn't demand. Total: about 23 minutes of structured reasoning, and the difference between this and the junior's two-minute box-drawing is night and day — not because the final boxes are radically different, but because every box is *justified* and every trade-off is *named*.

That's the loop. Clarify, constrain, sketch, stress, iterate. The exact same five steps work on a notification system, a rate limiter, a feed, a chat app, a payment ledger. The prompts change; the loop doesn't.

## 9. The trade-off matrix: how a senior decides

Everything in the iterate step came down to trade-offs, and trade-off analysis is the literal heart of senior-level design. The junior failure mode is to treat technologies as *good* or *bad* — "Cassandra is web-scale," "monoliths are bad," "always use a queue." The senior knows there are no good or bad technologies, only technologies that win a specific access pattern and pay for it elsewhere. The skill is mapping a requirement to the lever that serves it and naming the cost of pulling that lever.

![A matrix mapping each requirement type to the design lever it points at and the cost that lever adds](/imgs/blogs/how-seniors-approach-ambiguous-system-design-problems-3.webp)

The matrix above is the senior's cheat sheet for our URL shortener and systems like it: a read-heavy workload points at caching and CDN, paying with stale reads; a write-heavy workload points at sharding and a log, paying with hot-key risk; a strong-consistency requirement points at a quorum or single leader, paying with higher latency; a low-p99 requirement points at edge and async, paying with eventual data; a huge-storage requirement points at object storage, paying with slow scans. Every row is a requirement, every lever is a choice, and every choice has a cost in the right-hand column. There is no free lever.

Let me make the decision logic explicit with a comparison table for the one decision we hand-waved past — how to store the mappings — because seeing the full trade-off space is the senior habit:

| Store choice | Gain | Pay | When it wins |
| --- | --- | --- | --- |
| Single Postgres | Transactions, easy ops, range scans | Vertical scaling ceiling | Low/medium scale, need SQL |
| Postgres + read replicas | Cheap read scaling | Replication lag, write ceiling unchanged | Read-heavy, our exact case |
| DynamoDB / KV store | Massive scale, low-latency point reads | No range scans, eventual reads | Huge scale, pure key lookups |
| Cassandra | Huge write volume, multi-region writes | Weak transactions, operational weight | Write-heavy, multi-region active-active |

For *our* requirements — read-heavy, point lookups, modest writes, need durability, tolerate seconds of staleness — the winner is Postgres with read replicas, or a managed key-value store. A junior who reaches for Cassandra "because it scales" has chosen the row that pays for massive write volume and multi-region writes we don't need, while sacrificing the transactional simplicity and range scans we might want later. They optimized a dimension that wasn't the bottleneck and paid for it in operational weight. The senior reads the requirement table, finds the row that matches the *actual* access pattern, and names what that choice costs. Choosing a datastore by access pattern rather than by hype is itself a deep topic, covered in [choosing a datastore: SQL, NoSQL, NewSQL](/blog/software-development/system-design/choosing-a-datastore-sql-nosql-newsql) — but the *method* is always the same: match the lever to the requirement, name the cost.

## 10. Optimization: find the real bottleneck, ignore the rest

There's a quote attributed to Donald Knuth that every engineer half-remembers: "premature optimization is the root of all evil." The half people forget is the rest of the sentence — *we should not pass up our opportunities in that critical 3%*. The senior reading of this is precise: don't optimize everything, but *do* relentlessly optimize the one thing that dominates. The whole game is identifying the critical 3% — the real bottleneck — and pouring effort there while leaving the other 97% boring.

Our latency budget made this concrete. The redirect path's p99 is dominated by the gap between a 3ms cache hit and a 25ms cache miss. Therefore the *only* optimization that moves the dominant metric is cache hit rate. Shaving 1ms off the load balancer (already 2ms) is rounding error; getting hit rate from 90% to 99% moves the p99 off the database entirely. A junior, told to "make the redirect faster," might rewrite the service in a faster language, add connection pooling, tune garbage collection — all real but all in the 97% that doesn't matter, because the service code is 4ms and the database miss is 25ms. The senior measures first, finds the dominant term, and optimizes only that. **You cannot optimize what you haven't measured, and you shouldn't optimize what doesn't dominate.**

The same discipline applies to cost optimization, which seniors treat as seriously as latency. If our storage is 10 TB and reads are 100:1 to writes, the cost is dominated by the read path's compute and cache memory, not by storage bytes. Optimizing storage (compressing URLs, deduplicating) saves pennies; optimizing the cache hit rate saves both latency *and* the database compute we'd otherwise need to provision for misses. The cost bottleneck and the latency bottleneck happen to be the same lever here — cache hit rate — which is a happy accident, but the *method* of finding it (decompose the budget, attack the dominant term) is universal.

There's a deeper optimization principle hiding in the latency budget, and it's worth stating because it governs *which* optimizations are even possible. Latency optimizations come in two flavors: making a step faster, and making a step *disappear*. Making the database read faster (better indexes, more memory) is real but bounded — you can shave a 25ms read to 15ms, but you can't make it free. Making the read *disappear* (serving from cache so the database is never touched, or serving from the CDN edge so your infrastructure is never touched) is unbounded — a request that never reaches your origin costs you nothing and is bounded only by the speed of light to the edge. Seniors reach for "make it disappear" before "make it faster," because eliminating a step beats optimizing it every time. In our shortener, the highest-leverage optimization isn't a faster database; it's pushing the hottest links to the CDN edge so that the most-requested links never touch our origin at all. The same instinct applies everywhere: the fastest query is the one you don't run, the fastest network hop is the one you eliminate, and the cheapest server is the one you don't provision. Before you optimize a step, ask whether you can delete it.

This also reframes what a "good" number means. A junior optimizes toward an absolute target ("get p99 under 50ms") and stops when they hit it. A senior optimizes toward the *budget's dominant term* and keeps an eye on the second-order effects: dropping p99 by raising cache memory also drops database load, which lets you provision fewer database replicas, which drops cost — one lever moved three metrics. Conversely, some "optimizations" trade one metric for another without telling you: adding a read replica drops read latency but adds replication lag, which can *raise* the staleness window past your consistency SLO. The senior always asks "what *else* does this lever move?" before pulling it, because in a real system every lever is connected to several others, and the optimization that improves your dashboard metric while quietly breaking your consistency guarantee is the one that becomes next quarter's incident.

#### Worked example: where to spend a \$10,000/month optimization budget

Suppose you're handed \$10,000/month and told to make the URL shortener better, and you must justify where it goes. The senior approach is to decompose the current spend and the current metric and find where a dollar buys the most improvement:

- **Current p99 is 45ms**, dominated by a 90% cache hit rate (10% of requests pay the 25ms miss). Math: p99 is roughly the 99th-percentile request, and with 10% misses, the tail is full of misses. Pushing hit rate to 99% (one more nine) means only 1% of requests hit the slow path, dragging the p99 down toward the cache-hit time. **Spend: \$4,000/month on more cache memory and proactive hot-key refresh.** Expected win: p99 from 45ms to ~15ms. This is the dominant lever.
- **Current availability is 99.95%**, short of the 99.99% target, because we're single-region. **Spend: \$3,000/month on a second region with async read replicas.** Expected win: availability to 99.99%+, surviving a region outage. This buys the dominant *reliability* metric.
- **Current write path occasionally stalls** under campaign bursts. **Spend: \$1,000/month on the ranged-allocator HA setup.** Expected win: write path scales to 50k/s without serialization stalls.
- **Remaining \$2,000/month: hold it.** A senior does not spend a budget just because it exists. The three changes above hit every SLO; the rest is reserve for the next bottleneck the next stress test finds. *Not spending* is itself an optimization decision, and naming it is a senior move juniors rarely make.

Notice that not a single dollar went to making the service code faster, even though that's where a junior's instinct points, because the service code was never the bottleneck. Optimization is the art of spending effort where the budget proves it pays, and refusing to spend it where it doesn't. Measure, find the dominant term, attack it, stop.

## 11. The meta-skills: communicating the decision

Running the loop in your head is half the job. The other half is *communicating* the decision so that the people in the room — a panel, your team, a skeptical staff engineer — trust it. A correct design defended badly loses to a worse design defended well, which is unfair but true, so the communication is not optional polish; it's part of the engineering.

The structure that works is the same loop, narrated out loud. You say what you're doing as you do it: "First let me make sure I understand the problem" (clarify, narrated), "so the dominant requirement here is X, which means I'm going to optimize for Y and I'm willing to sacrifice Z" (constrain, narrated, and notice you've already told them what you're sacrificing — that *builds* trust, it doesn't undermine it). "Here's the simplest thing that works" (sketch), "now let me poke holes in it — what happens at 10×?" (stress, and now you're attacking your own design *in front of them*, which is the single most credibility-building move you can make), "so the bottleneck is the central counter, and I'll fix it with ranged allocation, which costs me sequence gaps that don't matter here" (iterate, with the trade-off named). Narrating the loop turns a static diagram into a reasoning process the audience can follow and trust.

![A design review timeline showing the early minutes on the problem and the close on the named trade-off](/imgs/blogs/how-seniors-approach-ambiguous-system-design-problems-4.webp)

The single biggest communication upgrade from junior to senior is *leading with the trade-off, not hiding it.* A junior, asked "why Redis and not Memcached," gets defensive, as if admitting a downside is admitting a mistake. A senior says: "Redis, because I want the data structures and persistence; the cost is it's single-threaded per node so I shard it, and if I didn't need the data structures I'd use Memcached for the simpler ops model." That answer names the choice, the gain, the cost, and the alternative — in one breath. It signals that you considered the alternatives and chose deliberately. *Naming the downside of your own choice is the strongest possible signal that you understand it.* Juniors hide downsides because they think downsides are weaknesses; seniors lead with downsides because they know downsides are evidence of understanding.

There's also a meta-point about *what diagram to draw*, since this is a design *review* and the picture is the medium. A diagram that shows the data flow and the failure boundaries communicates; a diagram that's a soup of twelve boxes and forty arrows does not. Drawing diagrams that actually communicate is its own skill, covered in [system design diagrams that communicate](/blog/software-development/system-design/system-design-diagrams-that-communicate) — the short version is: show the paths and the failure domains, label the components with their *role and a number*, and leave off everything that isn't load-bearing for the decision you're defending.

## 12. How to say "it depends" without sounding evasive

"It depends" has a bad reputation because juniors use it to dodge. But "it depends" is often the *only* correct answer, and the senior version is not a dodge at all — it's the most informative possible response, *if you name what it depends on and commit to an answer on each branch.* The evasive version is "well, it depends." The senior version is "it depends on whether reads dominate writes; if reads dominate, I cache and add read replicas; if writes dominate, I shard the write path and accept the resharding complexity. Which one are we?" That answer is *more* informative than a flat recommendation, because it teaches the decision rule, not just the conclusion.

![A decision tree turning 'it depends' into committed answers by naming the variable and branching on it](/imgs/blogs/how-seniors-approach-ambiguous-system-design-problems-6.webp)

The decision tree above is the shape of a good "it depends." The root is the question. The branches are the *variables the answer hinges on*, stated explicitly. The leaves are *committed answers*, not more hand-waving. "How should I store the data? It depends on the access shape and the consistency need. If reads dominate: cache plus read replicas. If writes dominate: shard the write path. If you need strong consistency: single-leader quorum, paying latency. If not: async replicas, gaining latency, paying freshness." Every leaf is a decision, not a deferral. That's the difference between a senior "it depends" and a junior one: the senior's branches terminate in commitments, and the senior tells you which branch you're on by asking the one question that resolves it.

The practical tactic: when you feel the pull to say "it depends," immediately ask yourself *depends on what, exactly?* — and say that out loud as a question to the room. "It depends on the read-write ratio — do we know that?" turns a dodge into a clarifying question, which loops you right back to step one. Half the time, asking "depends on what?" reveals that you do actually know the answer because the requirement is already pinned down, and you can just commit. The other half, it surfaces the missing requirement, which is exactly what the clarify step is for. "It depends" is not the end of an answer; it's a pointer to the requirement you still need to clarify.

## 13. Avoiding over-engineering: the other senior failure mode

We've talked a lot about juniors under-thinking the design. But there's a symmetric failure mode that catches *experienced* engineers specifically, and it's worth naming because it's less visible: over-engineering. The senior who has been burned by scale before sometimes over-corrects and builds for a scale that will never arrive — the multi-region active-active, event-sourced, CQRS, Kubernetes-everything architecture for a product with four hundred users. This is just as much a failure as under-engineering; it's slower to build, harder to operate, more expensive, and more likely to fail in novel ways, all to handle a load that isn't coming. Over-engineering is under-engineering's evil twin, and a real senior is vigilant against both.

The antidote is the constrain step, used honestly. Over-engineering happens when you design for *imagined* requirements instead of *stated* ones. The discipline is: every component must trace back to a requirement in the table. If you can't point at the requirement that justifies the queue, the second region, the event store — it's speculative, and speculative complexity is debt you pay interest on every day in operations. The honest question is always "what in the requirements demands this?" If the answer is "well, what if we get huge?", the senior response is "then we'll add it when the requirement is real, and here's the seam I'll leave so adding it later is cheap." Designing for change without building the change *now* is the topic of [evolutionary architecture: designing for change](/blog/software-development/system-design/evolutionary-architecture-designing-for-change), and the core idea is exactly this: leave the seams, defer the complexity, add it when the requirement materializes.

There's a calibration here that only comes with experience: knowing *which* future requirements are cheap to defer and which are expensive to retrofit. Sharding is expensive to retrofit, so if you genuinely expect to need it, designing the schema so it *can* be sharded later (sharding key in every table) is cheap insurance worth buying now. Multi-region is expensive to retrofit, so leaving the seam (no region-pinned assumptions in the code) is worth it. But actually *running* multi-region before you have the traffic is over-engineering. The senior buys the cheap insurance — the seams — and defers the expensive operations until the requirement is real. Cheap-to-add-later things you defer; expensive-to-retrofit things you leave a seam for; nothing gets built before its requirement exists.

## 14. Junior versus senior on the same prompt, made explicit

Let me make the contrast brutally explicit, because seeing the two side by side is the fastest way to internalize the difference. Same prompt — "design a URL shortener" — two engineers.

**The junior** grabs the marker at minute one and draws boxes. They pick technologies first: "We'll use Cassandra for the database because it scales, Kafka for the events, Redis for the cache, and we'll run it all on Kubernetes." They have not asked the read-write ratio, the scale, the latency target, or the cost of failure. When you ask "why Cassandra?", they say "because it's web-scale" — a property statement, not a reasoning statement. When you ask "what happens if a region goes down?", they pause, because they hadn't considered it. When you ask "what's the bottleneck?", they don't have an answer, because they never stress-tested. Their design might even work, but they can't tell you *why* it works, *when* it fails, or *what it cost*. It's a memorized template, not a reasoned design.

**The senior** asks questions for two minutes before drawing. They establish the 100:1 read ratio, the 50ms p99 target, the four-nines availability need, and the tolerance for seconds of staleness — and they state out loud what those imply: "reads dominate, so I optimize the read path and the write path can be simple; I can relax consistency, which makes multi-region cheap." They draw the *simplest* design, then immediately attack it: "at 10× the central counter is my write bottleneck; a region outage kills my availability; a viral link is a hot key with a thundering-herd risk." They fix the biggest bottleneck first, naming the trade-off each fix buys. When you ask "why this datastore?", they name the access pattern it serves and the cost it pays. When you ask "what breaks?", they hand you a ranked list. Their final design might use the *same* technologies as the junior's — but every one is justified, every trade-off is named, and they can tell you exactly where it breaks and what they'd do next.

The difference is not the boxes. It's the *reasoning around the boxes*. The junior produces an artifact; the senior produces a defensible decision. And the gap between them is entirely the five-step loop and the trade-off discipline — both of which are learnable, which is the whole point of this series. You don't become a senior by memorizing more architectures. You become one by running the loop until it's automatic and naming the trade-off until it's reflexive.

![A matrix of where each loop step adds the most value and what skipping it costs](/imgs/blogs/how-seniors-approach-ambiguous-system-design-problems-9.webp)

The matrix above is why the order of the loop matters so much. The early steps — clarify and constrain — are where the *expensive* mistakes get caught, because a wrong problem or a wrong scale assumption forces a full redesign, while a missed bottleneck at least gets caught in the stress step. Skipping clarify means you might build the wrong system entirely; skipping constrain means you build for the wrong scale and rebuild at 10×; skipping stress means you find the bottleneck in production at peak instead of on the whiteboard. The cheapest possible insurance against the most expensive possible mistakes is spending the first five minutes on the problem instead of the solution. Juniors skip the cheap insurance and pay the expensive premium. Seniors buy the insurance every time.

## 15. Case studies: the loop in the wild

The five-step loop isn't an interview trick; it's a compressed version of how good engineering organizations actually reason. Here are four cases where the public record shows the pattern — clarify the real problem, constrain to the real numbers, stress the design against reality, and iterate with named trade-offs.

**Discord and the message store.** Discord publicly described migrating their message storage from MongoDB to Cassandra and later to ScyllaDB as their scale grew into the billions and then trillions of messages. The senior lesson isn't "use ScyllaDB"; it's that they *re-ran the loop* as the constraints changed. The original store was right for the original scale. When the access pattern (massive write volume, time-ordered reads, hot channels acting as hot keys) and the numbers changed by orders of magnitude, the right datastore changed with them. They named the trade-off each time — Cassandra bought write scale but cost them GC pauses and operational pain; ScyllaDB bought lower-latency tails at the cost of a migration. The takeaway: the loop is not a one-time exercise. You re-run clarify-constrain when the scale moves an order of magnitude, because the right answer at 1× is often the wrong answer at 100×.

**Figma and the hot partition.** Figma has written about scaling their Postgres infrastructure, including the work to horizontally shard. A recurring theme in stories like theirs is the *hot partition* — the stress-step failure mode where one shard (one popular file, one giant team) takes load wildly out of proportion to the others, exactly the hot-key problem we found in our URL shortener. The lesson is that uniform sharding does not save you from non-uniform load, and the iterate-step fix involves picking a sharding key that *spreads* the hot entities rather than concentrating them. This is precisely the kind of second-order failure the stress step exists to surface before it's an incident, and it's why "what's the hot key?" is one of the four canonical stress questions.

**Cloudflare and the thundering herd.** CDN and edge providers like Cloudflare operate at a scale where cache stampedes are not a theoretical worry but a daily operational reality, and the public engineering writing from that world is full of the mitigations we discussed: request coalescing so a cache miss for a hot object results in *one* origin fetch rather than a stampede, and TTL jitter so a million edge caches don't all expire the same object simultaneously. The lesson for the architect: the thundering herd is a *predictable* failure of the naive cache design, and the senior anticipates it in the stress step rather than discovering it during a traffic spike. The mitigations are standard; the senior skill is knowing to apply them before the herd arrives.

**The cautionary tale: over-engineering for scale that never came.** The less-told case study is the startup that built a multi-region, microservices, event-sourced architecture for a product that peaked at a few thousand users and then spent its limited engineering runway operating that complexity instead of building features. There's no single public postmortem because these failures are quiet — the company just runs out of money slowly, with most of its engineering time eaten by operational overhead it designed for itself. The lesson is the §13 one: over-engineering is a real, expensive failure mode, and the constrain step — designing for stated requirements, not speculative ones — is the defense. The simplest design that meets the requirements is not the lazy choice; it's frequently the *correct* one, and the burden of proof is on every box you add beyond it.

Across all four cases the meta-pattern is identical: the right design is a function of the *real* constraints, the constraints change as scale changes, the dangerous failures are the second-order ones the stress step surfaces, and every change should name its trade-off. That's the loop, running in production at companies you've heard of.

## 16. When to reach for this framework (and when not to)

**Reach for the full five-step loop** whenever the problem is ambiguous and the cost of getting it wrong is real: a new system, a major redesign, a design-review or interview, any decision where you're choosing between architectures with different failure modes. The loop's whole value is in ambiguous, high-stakes situations where the expensive mistake is committing to a design before the problem is clear. The more vague the prompt and the higher the cost of a wrong turn, the more the loop pays off.

**Run a compressed version** for smaller decisions. Not every choice needs a 25-minute whiteboard session. For a routine feature with well-understood requirements, the loop collapses to a quick mental pass: clarify (do I understand the ask?), constrain (any non-obvious scale or SLO?), sketch (simplest approach), stress (any obvious failure mode?), ship. The loop is a tool whose ceremony should match the stakes — full ritual for a new distributed system, a thirty-second mental check for a CRUD endpoint.

**Don't over-apply it** to problems that aren't actually ambiguous or aren't actually about scale. If the requirement is genuinely "add a field to a form," running a formal trade-off analysis is its own kind of over-engineering — you're spending senior ceremony on a junior problem. Part of seniority is calibrating *how much process* a decision deserves. The loop is most valuable precisely where it's hardest: the open-ended, high-stakes, "design X" problems where juniors flail and the cost of a wrong design is measured in quarters of rework. Match the ceremony to the stakes, and you'll neither under-think the hard problems nor over-think the easy ones.

## 17. Key takeaways

- **Resist the urge to design.** The highest-leverage move on any ambiguous prompt is the question you ask in the first two minutes, not the box you draw at minute ten. Clarify before you sketch, every time.
- **Run the loop: clarify, constrain, sketch, stress, iterate.** It's the same five steps on every prompt — URL shortener, notification system, feed, ledger. The prompts change; the loop doesn't. Discipline in the order beats cleverness in the boxes.
- **Pin the one metric.** Every system has one dominant metric where being wrong means failure. Name it, optimize for it, and know what you're allowed to sacrifice to get it. You cannot optimize everything.
- **Sketch the simplest thing that could possibly work.** Then break it. The first design is a baseline to stress, not a final answer. Add complexity only when the stress step proves you need it, and never one box more.
- **Stress-test against the four axes: 10× scale, region outage, hot key, thundering herd.** Find the single biggest bottleneck before someone else's traffic finds it for you. Output a ranked list, fix the top item first.
- **Name the trade-off on every change.** You never just "add a queue." You add it to buy something specific, paying something specific. If you can't name the cost, you don't understand the change. This is the spine of senior design.
- **Optimize only the dominant term.** Decompose the latency or cost budget, find the line item that dominates, and attack only that. Don't optimize the 97% that doesn't matter; don't pass up the critical 3% that does.
- **"It depends" is an answer when you name the variable and commit on each branch.** Turn the dodge into a decision tree whose leaves are commitments, not more hand-waving.
- **Beware over-engineering as much as under-engineering.** Every component must trace to a stated requirement. Leave seams for the expensive-to-retrofit things; defer the rest until the requirement is real.
- **The difference between junior and senior is the reasoning around the boxes, not the boxes.** Same technologies, but every choice justified, every trade-off named, every failure mode known. That gap is learnable — run the loop until it's reflex.

## 18. Further reading

The siblings in this series go deep on the steps this post sketched. Start with the two that expand the constrain step:

- [Turning vague asks into requirements and SLOs](/blog/software-development/system-design/turning-vague-asks-into-requirements-and-slos) — the discipline of extracting functional and non-functional requirements and pinning real SLOs.
- [Back-of-the-envelope estimation for system design](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design) — how to attach believable QPS, storage, and bandwidth numbers to a design in your head.
- [Articulating trade-offs: CAP, PACELC, and beyond](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond) — the vocabulary for naming the costs every design change buys.
- [System design diagrams that communicate](/blog/software-development/system-design/system-design-diagrams-that-communicate) — drawing the picture that defends the decision in a review.
- [Evolutionary architecture: designing for change](/blog/software-development/system-design/evolutionary-architecture-designing-for-change) — leaving seams for the expensive-to-retrofit requirements without building them now.
- [Choosing a datastore: SQL, NoSQL, NewSQL](/blog/software-development/system-design/choosing-a-datastore-sql-nosql-newsql) — matching the store to the access pattern instead of the hype.
- [Caching strategies and the pitfalls that bite](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite) — the thundering herd, stale reads, and the invalidation traps we touched on.
- [Partitioning and sharding without downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime) — splitting the write path, the ranged-allocation pattern, and hot-partition fixes.

For the mechanisms underneath the architect-level choices, the database deep dives are the reference:

- [Consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) — the spectrum we relaxed to make multi-region cheap.
- [The CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) — the formal frame behind the single-leader-versus-multi-leader write decision.
- [Distributed replication: leader, multi-leader, leaderless](/blog/software-development/database/distributed-replication-leader-multi-leader-leaderless) — the replication choices behind the region-outage fix.

External references worth your time: the *Designing Data-Intensive Applications* book (Kleppmann) for the mechanisms, the public engineering blogs from Discord, Figma, Cloudflare, and Stripe for the war stories, and the original Knuth paper on structured programming for the source of the much-abused optimization quote. Read the war stories with the loop in mind, and you'll start to see the same five steps everywhere.
