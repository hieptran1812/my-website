---
title: "The Actor Model: Mailboxes, Isolation, and Supervision"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How giving each entity private state and a mailbox removes the lock entirely, turns crashes into restarts, and scales the same code from one core to a cluster."
tags:
  [
    "concurrency",
    "parallelism",
    "actor-model",
    "erlang",
    "akka",
    "supervision",
    "message-passing",
    "fault-tolerance",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/the-actor-model-mailboxes-isolation-and-supervision-1.png"
---

A bank balance is the most boring number in computing and the most ruthless. It is one integer. It must never be wrong. And the moment two requests touch it at the same instant — a deposit and a withdrawal arriving on two different threads — you are one unlucky interleaving away from a balance that is off by a hundred dollars and a customer who is, justifiably, furious.

The reflex from the first half of this series is a lock. Wrap the account in a mutex, take the lock before you read the balance, release it after you write. It works. It is also the source of half the production incidents in this whole series: forget to take the lock on one code path and you have a [data race](/blog/software-development/concurrency/data-races-vs-race-conditions-a-precise-distinction); take two locks in the wrong order and you have a [deadlock](/blog/software-development/concurrency/deadlock-the-four-conditions-and-how-to-break-them); hold the lock across a slow call and your throughput collapses under contention. The lock is correct and it is a perpetual liability, because the *sharing* is still there. Every thread can reach into that account. The lock is a discipline bolted onto shared mutable state, and discipline is exactly what fails at 3 AM.

The actor model takes a different bet: **remove the sharing instead of guarding it.** Give the account its own private state that no other thread can touch. Put a single-lane queue — a *mailbox* — in front of it. Let everyone who wants to change the balance *send a message* to that mailbox instead of calling a method. And let exactly one thread, at a time, pull one message off the queue and run it to completion before touching the next. There is no lock because there is nothing to lock: only one piece of code ever touches the balance, and only one message is ever in flight. The race condition is not *guarded* against — it is *structurally impossible*.

This post is about that idea and everything it pulls behind it. The actor — private state plus a mailbox plus a one-message-at-a-time behavior — is the unit. From there we get **supervision trees** and the "let it crash" philosophy that made Erlang systems famous for nine-nines uptime; **location transparency**, where an actor reference works the same whether the actor is on this core or across a datacenter, so the same code scales to a cluster; and a sharp set of **trade-offs** — no synchronous return value, no invariant that spans two actors, mailboxes that can overflow — that tell you exactly when to reach for actors and when not to. Figure 1 is the whole unit in one picture; we will keep coming back to it. By the end you will be able to build a bank-account actor in two languages, wire it under a supervisor, and reason about what actually happens when one of a million of them crashes.

![Two senders fan into one mailbox queue which feeds a single behavior that mutates private state and emits outbound messages](/imgs/blogs/the-actor-model-mailboxes-isolation-and-supervision-1.png)

This is the third turn of the series' spine. We named the hazard — shared mutable state under nondeterministic scheduling — in [why concurrency is hard](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it). We spent the mutual-exclusion track guarding the sharing with [mutexes and critical sections](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections). Now we cross into the models that *avoid* the sharing entirely. Actors and [CSP channels](/blog/software-development/concurrency/message-passing-vs-shared-memory-and-the-csp-philosophy) are two faces of the same coin: don't communicate by sharing memory; share memory by communicating. Actors are the message-passing paradigm with an opinion — that the unit of isolation should be a stateful entity, and that failure should be a first-class part of the design.

## What an actor actually is

Let me define the term precisely, because "actor" gets thrown around loosely. An **actor** is the combination of three things, and it is *only* an actor if it has all three:

1. **Private, isolated state.** The actor holds some data — a balance, a session, a counter, a connection — and *nothing outside the actor can read or write it directly.* There is no public field, no shared pointer, no getter that hands out a reference into its guts. The state is sealed.

2. **A mailbox.** Every actor has exactly one inbox: a queue of messages waiting to be processed. Other actors do not call the actor; they *post* a message to its mailbox and move on. The mailbox is the only door in.

3. **A behavior that processes one message at a time.** The actor has a function — call it `receive` — that takes the next message off the mailbox, does whatever that message says (read or update its private state, send messages to other actors, spawn new actors), runs to completion, and only *then* takes the next message. One message, start to finish, before the next one begins.

That third property is the whole game. The phrase to burn into memory is **single-threaded per actor**. At any instant, at most one thread is executing inside a given actor, processing exactly one message. The runtime might be running ten thousand *other* actors on the same eight cores at the same time — actors are cheap and there are usually far more of them than there are cores — but *within* one actor, processing is strictly serial. That is the guarantee that removes the lock.

This is the formal model Carl Hewitt described in 1973, refined by Gul Agha, and made industrial by Joe Armstrong's Erlang. Hewitt's original three axioms of what an actor can do in response to a message are worth stating because they bound the entire model. On receiving a message, an actor may: **(a) send** a finite number of messages to other actors; **(b) create** a finite number of new actors; and **(c) designate** the behavior to use for the *next* message it receives (which is how an actor changes its own state — by deciding what it will be next time). That is it. No shared variables, no method calls, no return values in the call-stack sense. Everything an actor does is one of those three things.

Figure 1 draws the unit: two senders fan into the mailbox, the mailbox feeds the behavior, the behavior reads and writes the private state and emits outbound messages. Notice what is *not* in that picture: there is no arrow from a sender directly into the state. The only path to the state runs through the mailbox and the single-threaded behavior. That funnel is the actor.

### Why "one at a time" is not a performance disaster

The first objection every engineer raises: if each actor processes messages serially, haven't I just thrown away concurrency? No — and this is the key reframing. You have not made *the system* serial; you have made *each actor* serial. The concurrency lives *between* actors, not inside one. A million actors can each be processing their own message simultaneously across your cores. The bank with ten million accounts has ten million actors, each serial, and the system as a whole is massively concurrent. You traded *intra-entity* parallelism (which you almost never want for a single account anyway — you do not want two threads racing on one balance) for *inter-entity* parallelism (which is exactly the scaling axis that matters: more accounts, more requests, more actors).

The serial-per-actor model maps the *unit of concurrency to the unit of consistency.* One account is one consistency boundary and one actor. You never wanted the balance updated by two threads at once; the actor model makes that not a rule you enforce but a shape you cannot violate.

### How the runtime runs a million actors on eight cores

The claim "a million actors, each serial, all concurrent" only works because the runtime decouples *actors* from *OS threads*. This is the mechanism that makes everything else affordable, and it is worth understanding precisely. The runtime is an **M:N scheduler**: it multiplexes M lightweight actors onto N OS threads (typically N equals the number of cores), with its *own* scheduler in user space deciding which actor runs next on each thread. The OS never sees the actors at all — it sees N busy threads, and the language runtime, not the kernel, picks which actor each thread executes.

Concretely, on the BEAM (Erlang/Elixir's VM): the runtime starts one **scheduler thread per CPU core**, and each scheduler owns a *run queue* of ready actors (Erlang calls them processes). A scheduler pulls a ready actor off its queue, runs it until it either blocks waiting for a message (`receive` with an empty mailbox) or *uses up its time budget*, then puts it back and picks the next one. The time budget is measured in **reductions** — roughly, function calls. After about **2,000 reductions** the BEAM *preempts* the actor mid-flight, saves its tiny state, and lets another actor run. This **preemptive** scheduling is what gives the BEAM its famous *soft-real-time* property: no single actor, however busy, can starve the others, because the scheduler forcibly rotates them. A tight CPU loop in one actor cannot freeze the node — it just gets preempted every 2,000 reductions. Contrast that with cooperative schedulers (a JavaScript event loop, or Go before its preemptive scheduler) where one CPU-bound task that never yields *can* stall everything.

Two more pieces complete the picture. **Work stealing**: if one scheduler's run queue empties while another's is backed up, the idle scheduler *steals* actors from the busy one's queue, keeping all cores fed — the same load-balancing trick a [work-stealing thread pool](/blog/software-development/concurrency/processes-threads-and-how-the-os-scheduler-runs-them) uses. And **per-actor heaps**: each actor gets its own small, independently garbage-collected heap. This is a quietly huge property — because no memory is shared between actors, the garbage collector can collect *one actor's* heap without stopping any other actor. There is no global stop-the-world GC pause across the whole system; GC is per-actor and concurrent. A dead actor's heap is simply freed wholesale. Isolation pays off again, this time in the memory subsystem.

#### Worked example: why a slow message does not block the node

Suppose actor X receives a message whose handler runs a 50-millisecond computation, on a node also running 100,000 latency-sensitive actors. On a naive cooperative runtime, X would hold its scheduler thread for the full 50 ms and 1/8 of the node's throughput would stall. On the BEAM, X is preempted after ~2,000 reductions (microseconds of work), put back on the run queue, and the scheduler interleaves it with everyone else — X still takes 50 ms of *wall-clock* time to finish its handler, but it does so in thousands of small preempted slices, and the other actors keep their low latency. The cost X pays for being slow is that *X's own* next message waits behind its slow handler (intra-actor serial order is preserved) — the cost is correctly *localized to X*, not spread across the node. That localization is the scheduler enforcing the same isolation the mailbox enforces for state.

## Why no lock is ever needed inside an actor

Let me make the mechanism rigorous, because "actors remove the need for locks" is the kind of claim that deserves a proof, not a slogan. Recall *why* a shared counter needs a lock at all. The operation `balance = balance + amount` is not atomic. It compiles to three steps: **load** the current balance into a register, **add** the amount, **store** the result back. Two threads running this concurrently can interleave:

#### Worked example: the lost update an actor cannot have

Suppose `balance = 100`, thread T1 deposits 50, thread T2 deposits 30. The correct result is 180. Here is the losing interleaving on a shared, unlocked balance, step by step:

| Step | Thread | Action | Effect |
| --- | --- | --- | --- |
| 1 | T1 | load `balance` into reg1 | reg1 = 100 |
| 2 | T2 | load `balance` into reg2 | reg2 = 100 (reads the same stale 100) |
| 3 | T1 | add 50 in register | reg1 = 150 |
| 4 | T2 | add 30 in register | reg2 = 130 |
| 5 | T1 | store reg1 to `balance` | balance = 150 |
| 6 | T2 | store reg2 to `balance` | balance = 130 (overwrites T1; the +50 is lost) |

The final balance is 130, not the correct 180. That is the lost update we dissected in [shared mutable state and the anatomy of a race condition](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition). The bug exists because two threads' load-modify-store sequences *interleave*. The lock fixes it by making the whole load-modify-store a critical section: T2 cannot start until T1 finishes.

Now watch what the actor does. The account is one actor. T1's deposit and T2's deposit are not method calls on a shared object — they are *messages* sent to the account actor's mailbox. The runtime serializes them in the mailbox. The actor's behavior dequeues the deposit-50 message, runs `balance = balance + 50` *to completion* (load 100, add 50, store 150), and only then dequeues the deposit-30 message and runs `balance = balance + 30` (load 150, add 30, store 180). The two load-modify-store sequences **cannot interleave because there is no second thread inside the actor to interleave with.** The behavior is single-threaded by construction.

There is the proof. The race required two concurrent load-modify-store sequences on the same memory. The actor admits exactly one execution context touching that memory, processing one message at a time. With no second concurrent accessor, there is no interleaving; with no interleaving, there is no lost update; with no lost update, there is no need for a lock. The lock and the actor are two different answers to the same question — *how do I serialize access to this state?* The lock serializes by mutual exclusion over shared memory. The actor serializes by giving the memory a single owner and a single-lane queue. The actor's answer happens to also be free of the lock's failure modes: you cannot forget to take a lock that does not exist, and you cannot deadlock on a lock you never acquire.

A subtle but important corollary: this guarantee holds **only inside one actor**. The instant your invariant spans *two* actors — "the sum of these two account balances must equal a constant" during a transfer — the single-actor guarantee buys you nothing, because the two halves of the transfer are processed by two different serial contexts with no shared lock between them. Hold that thought; it is the central trade-off and we will return to it.

```python
"""Python illustration only -- a sketch of the serial-per-actor loop."""
# In real systems you'd use a runtime (Erlang, Akka, ...). For Python's
# own concurrency story (GIL, asyncio, threads) see the python-performance
# series linked below; do not build production actors this way by hand.
import queue, threading

class AccountActor:
    def __init__(self):
        self._mailbox = queue.Queue()      # the one door in
        self._balance = 0                  # private, sealed state
        threading.Thread(target=self._run, daemon=True).start()

    def send(self, msg):                   # async: enqueue and return
        self._mailbox.put(msg)

    def _run(self):
        while True:
            msg = self._mailbox.get()      # one message at a time
            kind, amount = msg
            if kind == "deposit":
                self._balance += amount    # no lock: single thread here
            elif kind == "withdraw":
                if amount <= self._balance:
                    self._balance -= amount
```

The `_balance += amount` line has no lock and needs none: only `_run` ever touches `_balance`, and `_run` is one thread pulling one message at a time. (Python's [GIL](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) makes this particular toy extra-serial, but the actor guarantee is independent of the GIL — it comes from the *single consumer thread*, not the interpreter lock.)

## Messages are asynchronous and one-way

Here is where actors feel alien if you come from method calls. When you call `account.deposit(50)` on a normal object, the call **blocks**: your thread sits on the call stack until `deposit` returns, and you get a return value. When you *send* a message to an actor, the send **returns immediately**. You have posted a message to the mailbox; you do not wait for it to be processed; you get nothing back synchronously. Sending is *fire-and-forget*. In Erlang the operator is literally `!` ("bang"); in Akka it is `tell` (also spelled `!`); the semantics are "deliver this, then carry on."

This one-way, asynchronous nature is not a limitation to work around — it is the property that makes the rest of the model possible. If sends blocked, an actor that sent a message and waited could deadlock against an actor doing the same back to it, and you would have reinvented the lock-ordering problem. Because sends never block, the *sender* is never coupled to the *receiver's* execution. The decoupling is the point.

But programs need answers. How do you get the balance *back* if `getBalance` does not return anything? You use **request/reply via a correlation mechanism.** The requester includes, in the message, a reference to *where the reply should go* — either its own actor reference (`self()`) or a one-shot reply channel — plus, often, a **correlation id** so it can match the reply to the request when several are outstanding. The receiver, when it finishes, *sends a reply message* to that address. The reply lands in the requester's mailbox like any other message.

```erlang
%% Erlang: request/reply by hand. The requester's pid is the reply address.
%% Server loop -- an account actor:
account(Balance) ->
    receive
        {deposit, Amount} ->
            account(Balance + Amount);              % new state = next behavior
        {withdraw, Amount} when Amount =< Balance ->
            account(Balance - Amount);
        {balance, From, Ref} ->
            From ! {balance_reply, Ref, Balance},   % reply to the asker
            account(Balance)
    end.

%% Caller -- synchronous-feeling request/reply built on async sends:
get_balance(AccountPid) ->
    Ref = make_ref(),                               % unique correlation id
    AccountPid ! {balance, self(), Ref},            % async send, includes reply addr
    receive
        {balance_reply, Ref, Balance} -> Balance    % match THIS reply by Ref
    after 5000 ->
        {error, timeout}                            % don't wait forever
    end.
```

Three things in that snippet are the whole request/reply pattern, and they recur in every actor system. **`self()`** is the reply address — the account learns where to send the answer from the message itself. **`make_ref()`** is the correlation id — if the caller has several requests in flight, `Ref` lets it match *this* reply to *this* request and ignore stragglers. And **`after 5000`** is the timeout — because a reply is just another message that *might never come* (the account might have crashed, the message might be lost across the network), every blocking receive in a healthy system has a timeout. A method call cannot "time out"; a message exchange must be able to.

Higher-level frameworks wrap this so it feels like a call. Akka's `ask` pattern (`?`) creates a temporary one-shot actor to receive the reply and hands you back a `Future`/`CompletionStage`; Orleans makes grain methods return `Task<T>` so the request/reply is hidden behind `await`. But underneath, it is always: send a message, attach a reply address, await a reply message, time out if it never arrives. Knowing the plumbing matters because the failure modes — a lost reply, a timeout, a reply to a dead requester — are the plumbing's failure modes, not magic.

The other consequence of asynchrony is **ordering**, and the guarantee is narrow and precise: messages are delivered **in send-order per sender pair, and only per pair.** If actor A sends actor B messages `m1` then `m2`, B sees `m1` before `m2`. But if A sends `m1` and C sends `m3` to B "at the same time," B may see them in either order — there is no global ordering across different senders. This per-sender FIFO is all most systems guarantee (Erlang and Akka both guarantee exactly this and no more), and designs that assume a total order across senders are broken. We will see why that matters for invariants shortly.

#### Worked example: the ordering bug that survives every unit test

Two services, the *credit service* C and the *debit service* D, both send messages to one account actor B. The intended business rule is "the salary credit lands before the rent debit." C sends `credit 1000` at 12:00:00.000; D sends `debit 800` at 12:00:00.001 — a full millisecond later. Surely the credit arrives first? No guarantee. C and D are *different senders*; per-sender FIFO says nothing about the order *across* them. If C's message takes a slightly slower network path, B may process `debit 800` against a balance of, say, 100, reject it for insufficient funds, and *then* process the `credit 1000`. The rent payment bounced even though the salary "arrived first" in wall-clock time. This passes every single-machine unit test (where the messages happen to arrive in send order) and fails intermittently in production under network jitter. The fix is not to wish for global ordering — it does not exist — but to *encode the dependency in the data*: make the debit carry a precondition ("only if balance ≥ 800 after all credits up to sequence N"), or route both through a single ordering actor, or use a saga that sequences the steps. The lesson: **if order matters across senders, you must make order part of the protocol, because the transport will not give it to you.**

### Actors and channels: two shapes of message-passing

It is worth placing actors next to the other message-passing model, because engineers conflate them and the distinction sharpens both. In the [CSP world](/blog/software-development/concurrency/message-passing-vs-shared-memory-and-the-csp-philosophy) — Go's goroutines and channels, Hoare's original calculus — the *channel* is the named, first-class thing and the processes are anonymous; you send to a *channel*, and any process reading that channel may receive it. In the *actor* world, the *actor* (its address) is the named, first-class thing and the mailbox is anonymous; you send to an *actor*, and that specific actor receives it. CSP couples to the *pipe*; actors couple to the *recipient*.

That difference cascades. A Go channel is typically *synchronous* (an unbuffered channel rendezvous: the sender blocks until a receiver is ready), which gives you backpressure *for free* — the sender cannot outrun the receiver. An actor mailbox is *asynchronous* (the send returns immediately, the message queues), which gives you decoupling and the overflow hazard. Here is the same producer-to-account handoff in Go, where the synchronous channel makes backpressure automatic:

```go
// Go (CSP): the account "actor" is a goroutine owning private state;
// the channel is the mailbox. An UNBUFFERED channel gives backpressure for free.
type cmd struct {
    op     string
    amount int64
    reply  chan int64 // per-request reply channel = the correlation mechanism
}

func account(mailbox <-chan cmd) {
    var balance int64 // private state: only this goroutine touches it -> no lock
    for c := range mailbox { // one message at a time, serially
        switch c.op {
        case "deposit":
            balance += c.amount
        case "withdraw":
            if c.amount <= balance {
                balance -= c.amount
            }
        case "balance":
            c.reply <- balance // request/reply over a one-shot channel
        }
    }
}

func main() {
    mailbox := make(chan cmd) // UNBUFFERED: send blocks until account is ready (backpressure!)
    go account(mailbox)
    mailbox <- cmd{op: "deposit", amount: 100} // blocks until received
    reply := make(chan int64)
    mailbox <- cmd{op: "balance", reply: reply}
    fmt.Println(<-reply) // 100
}
```

This Go code *is* an actor in everything but name: a goroutine owns private `balance` (no lock — only that goroutine touches it), the channel is its mailbox, the `for c := range mailbox` loop processes one message at a time, and the per-request `reply chan` is the correlation mechanism. The single difference that matters is the unbuffered channel: `mailbox <- cmd{...}` *blocks* until the account goroutine is ready to receive, so a fast producer cannot pile up an unbounded queue — backpressure is structural. Buffer the channel (`make(chan cmd, 1000)`) and you get exactly the actor-style async mailbox back, overflow hazard and all. So actors and CSP are not rivals; they are two points on a spectrum of "share by communicating," differing in *what is named* (recipient vs pipe) and *whether the send blocks* (async mailbox vs sync rendezvous). Pick actors when the *entity and its lifecycle* are the thing you reason about (and you want supervision and distribution); pick CSP channels when the *flow of data through a pipeline* is the thing you reason about.

## A message in flight: the timeline

It helps to walk one message through the system end to end, because the asynchrony hides the steps. Figure 3 lays out a single deposit message's journey on a timeline. The sender fires `deposit 100` and immediately moves on — that is `t1`. The message lands at the *tail* of the account's mailbox at `t2`. If the actor is busy processing an earlier message — `t3` — the deposit simply waits its turn in the queue; nothing spins, nothing blocks the sender, the message just sits there. When the earlier work finishes, the runtime dequeues our deposit at `t4`, runs `balance += 100` *alone* at `t5` (no lock, because nothing else is inside the actor), and then moves on to the next message at `t6`.

![Timeline of a deposit message being sent, enqueued at the mailbox tail, waiting for an earlier message, then dequeued and applied to the balance in isolation before the next message starts](/imgs/blogs/the-actor-model-mailboxes-isolation-and-supervision-3.png)

The picture makes the serialization visible. The balance update at `t5` is sandwiched between "earlier message runs" and "next message starts" — it has the actor entirely to itself. That sandwich is the single-threaded-per-actor guarantee in motion. It is also where you can *see* the mailbox doing the work the lock would have done: the queue is the serialization mechanism. A mutex serializes by blocking threads at a critical section; the mailbox serializes by ordering messages in a queue and feeding them one at a time. Same outcome — exactly one update at a time — achieved by queuing instead of blocking.

This is also the right mental model for *latency*. The time from "send" to "applied" is the queue wait (`t2`→`t4`) plus the processing time (`t4`→`t5`). Under light load the queue is empty and latency is just processing time. Under heavy load the queue grows and latency is dominated by the wait — which is exactly [Little's law](/blog/software-development/concurrency/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws) at work: $L = \lambda W$, the average number of messages waiting equals the arrival rate times the average wait. If messages arrive faster than the actor can process them, $W$ grows without bound and the mailbox does too. That is the overflow problem, and it has its own section.

## Supervision trees and "let it crash"

Now the part that made Erlang legendary. In a lock-based system, error handling is *defensive*: you wrap risky operations in try/catch, check every return code, validate every input, and try to keep the object in a consistent state no matter what goes wrong. The problem is that you cannot anticipate every failure, and the half-handled failures are the worst — an exception caught in the wrong place leaves your object in a torn, half-updated state that is *worse* than a clean crash, because now the corruption is silent.

The actor model's answer is the opposite philosophy: **let it crash.** Do not defend against the unexpected inside the actor. Write the actor for the *happy path* — the messages and states you expect. If something unexpected happens (a malformed message, a bug, a failed assertion, a downstream that vanished), let the actor *crash* — die cleanly, taking its corrupted state with it. Then have a *separate* actor, a **supervisor**, whose entire job is to notice the crash and **restart** the failed actor from a known-good initial state.

Why is this better? Because a restart to a known-good state is a *vastly* simpler and more reliable recovery than trying to repair an arbitrary corrupted state in place. You do not have to enumerate everything that could go wrong; you just have to know the *good* state to restart from. As Joe Armstrong put it, the philosophy is to make systems that are *correct* by isolating the *errors* — a crash in one actor does not propagate, because the actor's state was private, so there is nothing shared to corrupt. The supervisor restarts it, the bad message is gone, and the system heals. This is why Erlang systems quote uptimes in *nines* — the AXD301 telecom switch famously hit nine-nines availability (about 31 milliseconds of downtime per year), not by never failing but by failing in tiny isolated pieces and restarting them faster than anyone noticed.

![Before shows try-catch guarding every operation and producing tangled half-updated state, after shows a happy-path actor that crashes on bad input and is restarted clean by a supervisor](/imgs/blogs/the-actor-model-mailboxes-isolation-and-supervision-6.png)

Figure 6 contrasts the two stances. On the left, defensive code guards every operation and *still* leaks: a missed case survives as a silent bug, a caught-too-early exception leaves half-updated state. On the right, the actor runs only the happy path, crashes on bad input, and a supervisor restarts it clean. The left column is *more* code and *less* reliable; the right column is less code and more reliable. That inversion is counterintuitive until you internalize that the crash is *isolated* — the actor's private state is the only thing lost, and it is exactly the thing you wanted to throw away.

Supervisors compose into a **supervision tree.** A supervisor watches a set of child actors (workers, or other supervisors). When a child crashes, the supervisor applies a **restart strategy**:

- **one-for-one** — restart only the crashed child; its siblings are unaffected. Use this when children are independent (each worker handles its own request).
- **one-for-all** — if one child crashes, restart *all* the children. Use this when children share fate (they form a pipeline where a restart of one invalidates the others).
- **rest-for-one** — restart the crashed child and any children started *after* it. Use this for ordered dependencies.

And a supervisor that keeps restarting a child that keeps crashing will, after a threshold (say, 5 restarts in 10 seconds), give up and *escalate* — crash *itself*, so *its* supervisor decides what to do. This is the **error kernel** pattern: errors bubble up the tree until they reach a level that knows how to handle them, and the deeper, riskier work lives in the leaves where a crash is cheap.

![A root supervisor with a pool supervisor that restarts stateless workers and a database supervisor that owns a stateful database actor](/imgs/blogs/the-actor-model-mailboxes-isolation-and-supervision-4.png)

Figure 4 draws a real tree: a **root supervisor** (one-for-one) sits over two child supervisors. A **pool supervisor** owns a set of stateless workers — if worker 2 crashes, one-for-one restarts just it, and workers 1 and 3 never notice. A **database supervisor** (rest-for-one) owns a stateful db actor that holds a connection. The shape encodes the failure policy: independent things sit under one-for-one so a crash is contained; dependent things sit under a strategy that restarts the dependency chain. The *structure* of the tree *is* the fault-tolerance design. You are not writing error handlers; you are drawing an organization chart of who restarts whom.

Here is a supervisor in Elixir, which gives the OTP machinery a clean syntax:

```elixir
defmodule Account do
  # An account actor as an OTP GenServer -- isolated state, serial handling.
  use GenServer

  def start_link(initial), do: GenServer.start_link(__MODULE__, initial)
  def deposit(pid, amt), do: GenServer.cast(pid, {:deposit, amt})     # async, no reply
  def withdraw(pid, amt), do: GenServer.call(pid, {:withdraw, amt})   # request/reply
  def balance(pid), do: GenServer.call(pid, :balance)

  @impl true
  def init(initial), do: {:ok, initial}                  # known-good start state

  @impl true
  def handle_cast({:deposit, amt}, balance) do
    {:noreply, balance + amt}                            # no lock; serial handler
  end

  @impl true
  def handle_call({:withdraw, amt}, _from, balance) when amt <= balance do
    {:reply, :ok, balance - amt}
  end
  def handle_call({:withdraw, _amt}, _from, balance) do
    {:reply, {:error, :insufficient_funds}, balance}     # expected case: reply, don't crash
  end
  def handle_call(:balance, _from, balance) do
    {:reply, balance, balance}
  end
end

# The supervisor: restart the account to its known-good state if it ever crashes.
defmodule Bank.Supervisor do
  use Supervisor

  def start_link(_), do: Supervisor.start_link(__MODULE__, :ok, name: __MODULE__)

  @impl true
  def init(:ok) do
    children = [
      {Account, 0}                                       # start an account at balance 0
    ]
    # one_for_one: if the account crashes, restart just it -- to balance 0.
    Supervisor.init(children, strategy: :one_for_one, max_restarts: 5, max_seconds: 10)
  end
end
```

Two design choices in that code are the whole "let it crash" discipline in practice. First, the `:insufficient_funds` case is *not* a crash — it is an *expected* business outcome, so the actor replies with an error and keeps running. "Let it crash" means crash on the *unexpected* (a bug, a malformed message), not on every error; you still model your domain's normal failure cases as ordinary replies. Second, the supervisor restarts the account to `0`, its known-good initial state. In a *real* bank you would not restart a balance to zero, of course — you would restart by re-reading the durable balance from the database, because actor state is in-memory and a restart *loses it*. That is the honest caveat: **a restart heals the process, not the data.** Anything an actor must not lose on crash has to be persisted (event-sourced, journaled, or read from a database on init) — which is exactly what frameworks like Akka Persistence and Orleans' state providers give you.

## Location transparency and distribution

So far an actor reference has been a thing on this machine. The model's most consequential property is that *it does not have to be.* An actor reference is an *opaque address* — in Erlang a `pid`, in Akka an `ActorRef`, in Orleans a grain reference. You `send` to that address. **You do not know, and your code does not change, whether the actor behind that address is on the same core, a different core, or a different machine across the network.** This is **location transparency**, and it falls out for free from the fact that you only ever *send messages.*

Think about why this works. A method call cannot be transparently remote — a method call is a jump on the call stack with a synchronous return; you cannot "jump" across a network. But a *message send* is already exactly what a network does: serialize a message, deliver it, the receiver processes it. The actor model defined communication as message-passing from day one, so making a send go to a remote machine is just changing *where the runtime routes the message* — the application code is identical. `account ! {deposit, 50}` looks the same whether `account` is local or three datacenters away. The runtime handles serialization, routing, and delivery.

This is the property that lets the actor model *scale from one core to a cluster with the same programming model.* You do not rewrite for distribution; you reconfigure. Akka Cluster shards millions of actors across a fleet of nodes and routes messages by actor identity; Orleans calls its actors *virtual actors* (grains) that the runtime *automatically* places, activates on demand, and migrates between servers — you never `new` a grain, you just get a reference by id and the runtime materializes it somewhere. The address is logical, not physical.

```scala
// Akka Typed (Scala): the same actor, the same protocol, location-transparent.
// Behaviors describe how to handle the next message; state is captured in the behavior.
object Account {
  sealed trait Cmd
  final case class Deposit(amount: Long) extends Cmd
  final case class Withdraw(amount: Long, replyTo: ActorRef[Reply]) extends Cmd
  final case class GetBalance(replyTo: ActorRef[Reply]) extends Cmd

  sealed trait Reply
  final case class Ok(balance: Long) extends Reply
  case object Insufficient extends Reply

  // The behavior IS the state: it closes over `balance` and returns the next behavior.
  def apply(balance: Long): Behavior[Cmd] = Behaviors.receiveMessage {
    case Deposit(amount) =>
      Account(balance + amount)                       // next behavior = new state; no lock
    case Withdraw(amount, replyTo) if amount <= balance =>
      replyTo ! Ok(balance - amount)                  // request/reply via replyTo address
      Account(balance - amount)
    case Withdraw(_, replyTo) =>
      replyTo ! Insufficient
      Behaviors.same                                  // unchanged state
    case GetBalance(replyTo) =>
      replyTo ! Ok(balance)
      Behaviors.same
  }
}
```

The Akka code embodies Hewitt's third axiom directly: the actor changes its state by *returning the next behavior* (`Account(balance + amount)`), not by mutating a field. `balance` is captured in the closure, never exposed, never shared. The `replyTo: ActorRef[Reply]` is the reply address carried in the message — the same request/reply pattern as the Erlang `Ref`, now type-safe. And critically, *none of this code knows where `replyTo` lives.* It might be a local actor or a remote one; `replyTo ! Ok(...)` is identical either way. That is location transparency in a single line.

Orleans takes location transparency to its logical extreme with **virtual actors**, which it calls *grains*. You never create or destroy a grain and you never hold a physical reference to one. You ask the runtime for a grain *by identity* — `GrainFactory.GetGrain<IAccount>(accountId)` — and the runtime *guarantees* that grain logically always exists: it activates an instance on some server in the cluster on first use, keeps it there while it is busy, deactivates it to reclaim memory when it goes idle, and *re-activates* it (possibly on a different server) the next time a message arrives. The developer writes what looks like ordinary async method calls; the runtime handles placement, activation, and the request/reply plumbing underneath.

```csharp
// Orleans (C#): a virtual actor (grain). The interface looks like async methods;
// the runtime turns each call into a message to a single-threaded grain activation.
public interface IAccount : IGrainWithStringKey
{
    Task Deposit(long amount);                    // async: returns a Task, not a value
    Task<bool> Withdraw(long amount);             // request/reply hidden behind await
    Task<long> Balance();
}

public class AccountGrain : Grain, IAccount
{
    private long _balance;                         // private state, one activation owns it

    // One grain processes one call at a time -> no lock on _balance.
    public Task Deposit(long amount)
    {
        _balance += amount;                        // serial per grain; no lock needed
        return Task.CompletedTask;
    }

    public Task<bool> Withdraw(long amount)
    {
        if (amount <= _balance) { _balance -= amount; return Task.FromResult(true); }
        return Task.FromResult(false);             // expected case -> reply false, don't crash
    }

    public Task<long> Balance() => Task.FromResult(_balance);
}

// Caller -- no `new`, no address: ask the runtime for the grain by id.
IAccount account = grainFactory.GetGrain<IAccount>("acct-42");
await account.Deposit(100);                         // looks like a call; is a message send
bool ok = await account.Withdraw(30);              // await hides the request/reply
```

The crucial line is `GetGrain<IAccount>("acct-42")` — there is no server in it. The grain's *identity* is the account id; the runtime decides *where* it lives. A grain is single-threaded per activation exactly like every other actor (so `_balance += amount` needs no lock), and the `await account.Deposit(100)` that looks like a method call is, underneath, a message to the grain's mailbox with the reply delivered back through the `Task`. Orleans hides the actor machinery behind the language's async syntax so thoroughly that many engineers use it without realizing they are writing actors — which is the strongest possible statement that request/reply, mailboxes, and isolation are the *foundation*, and the surface syntax is a choice. The same model wears four different faces — Erlang's `!`, Akka's typed `Behavior`, Actix's `Handler`, Orleans' awaited `Task` — and underneath all four is one mailbox and one serial behavior.

Distribution is not free, and the honesty here matters: a *local* send is an in-memory enqueue that essentially never fails; a *remote* send goes over a network that *can and will* fail — the message can be lost, the node can be unreachable, the actor can be gone. Location transparency makes the *code* uniform; it does **not** make the *failure modes* uniform. A robust distributed actor system must treat every cross-node send as *potentially undelivered* and design with timeouts, retries, and idempotency — the same discipline you would apply to any [message queue with at-least-once delivery](/blog/software-development/message-queue/backpressure-and-flow-control). The model gives you uniform *syntax* across the local/remote boundary; it does not abolish the boundary. Pretending a remote actor is exactly a local one — the "distributed objects" fallacy of the 1990s — is how you build systems that work in the demo and fall over in the datacenter.

## Backpressure and mailbox overflow

A mailbox is a queue, and the brutal arithmetic of queues is this: if messages arrive faster than the actor processes them, the queue grows **without bound.** Recall $L = \lambda W$ from the timeline section — if the arrival rate $\lambda$ exceeds the service rate, the mailbox length $L$ goes to infinity and so does the wait $W$. An unbounded mailbox under sustained overload does not just slow down; it consumes all your memory and the process dies — and not with a clean "queue full" error but with an out-of-memory kill that takes the *whole node* with it, mailbox and supervisor and all. The very isolation that makes actors safe makes a slow actor a *silent* memory leak: the senders are decoupled, so they happily keep firing into a mailbox no one is draining.

This is the actor model's sharpest operational hazard, and it is the direct cost of fire-and-forget sends. Because a send does not block, the sender gets *no signal* that the receiver is falling behind. In a synchronous call, a slow callee naturally slows the caller — the caller is blocked waiting. In an actor system, a slow receiver does *not* slow a fast sender; the messages just pile up. You have to *engineer* the backpressure that synchronous calls give you for free.

The remedies, roughly in order of how much you should reach for them:

- **Bounded mailboxes.** Cap the mailbox size. When full, you choose a policy: *block the sender* (reintroduces coupling, but bounds memory), *drop the message* (lossy but survives), or *fail the send* (signal the sender to back off). Akka offers bounded mailboxes with exactly these overflow strategies.
- **The reactive-streams / pull model.** Instead of pushing messages, the consumer *requests* `n` messages it is ready for, and the producer sends at most that many. This is *demand-driven* flow control — the consumer's capacity propagates back to the producer. Akka Streams and the Reactive Streams standard build this in; it is the principled fix.
- **Load shedding.** Above a watermark, drop or reject the lowest-value work *early*, before it enters the mailbox, so the system degrades gracefully instead of collapsing.
- **Scaling out.** Add more actors (shards) so each one's arrival rate drops below its service rate.

The single most useful thing you can do operationally is **measure mailbox depth as a first-class metric.** Every mature actor runtime exposes it: in Erlang/Elixir, `Process.info(pid, :message_queue_len)` returns the current backlog for any process, and production systems alert when any actor's queue crosses a threshold (a few hundred, say) because a steadily *growing* queue is the leading indicator of an actor that has become a bottleneck — it is visible *minutes* before the OOM. Plot the mailbox length of your hottest actors over time under a load test: a flat line near zero means the actor keeps up; a line that ramps and never recovers means your arrival rate has crossed the service rate and you are on borrowed time. This is the honest measurement that turns "the system felt slow" into "actor X's mailbox hit 40,000 and climbing at 13:42" — a number you can act on. Treat a rising mailbox the way you treat a rising heap: an alarm, not a curiosity.

This is the same problem, with the same solutions, as [backpressure and flow control in message queues](/blog/software-development/message-queue/backpressure-and-flow-control) and [rate limiting in system design](/blog/software-development/system-design/rate-limiting-and-backpressure) — an actor mailbox is an in-process message queue, and it inherits every flow-control concern queues have. The lesson to carry: **an unbounded mailbox is a production incident waiting for a traffic spike.** Default to bounded mailboxes and an explicit overflow policy for anything that faces unpredictable load. The default unbounded mailbox is fine for internal, rate-limited traffic and a trap for anything fed by the open internet.

## The trade-offs: what actors cost you

No model is free, and a senior engineer's job is to know the bill before signing. Actors cost you three concrete things, and each is the flip side of a strength.

![Matrix of actor trade-offs by axis showing each of isolation, supervision, distribution, and invariants as a strength on one row and a matching weakness on the next](/imgs/blogs/the-actor-model-mailboxes-isolation-and-supervision-8.png)

Figure 8 lays out the symmetry: isolation gives you no-locks (strength) but only async, no-return calls (weakness); supervision gives you self-healing crashes (strength) but loses in-flight messages on a crash (weakness); distribution gives you the same code local or remote (strength) but sends that can silently fail (weakness); and the single-actor invariant guarantee (strength) means there is *no* cross-actor invariant (weakness). Walk the three biggest costs in detail.

**1. No synchronous return — everything is async.** You cannot just "call and get a value." Every interaction is a send, and getting an answer means request/reply with a reply address, a correlation id, and a timeout. For code that is naturally a straight-line computation — call A, use its result to call B, return — actors add ceremony. Frameworks paper over it with `ask`/`Future`/`await`, but the asynchrony is real: you are always one timeout away from "the reply never came." If your problem is request/response with tight latency and no isolated state, actors are overhead.

**2. No invariant that spans multiple actors.** This is the big one and the most common way actor designs go wrong. The single-threaded guarantee holds *inside* one actor. The moment a correctness rule spans two — "transfer 100 from account A to account B, and the total must be conserved at every instant" — you are outside the guarantee. A transfer is two messages to two actors processed by two independent serial contexts. Between A's debit and B's credit, the money is *in flight* and the global invariant is *violated.* There is no lock spanning both actors to make the pair atomic, because the whole point was to not share. To get cross-actor atomicity you must reintroduce coordination *at the application level*: a saga (a sequence of compensable steps), a two-phase commit, or a process-manager actor that owns the transfer as its *own* state — which works precisely because it makes the transfer a single actor's responsibility again. The rule: **if your invariant cannot be drawn inside one actor's boundary, the actor model will not enforce it for you, and you will hand-roll the coordination you came here to escape.**

**3. Ordering is per-sender only.** As established, the only ordering guarantee is FIFO *per sender pair.* Two messages from *different* senders to the same actor can arrive in any order. Any design that assumes a total order across senders — "the credit always arrives before the debit because we sent the credit first, from a different service" — is broken. And delivery itself is, in the base model, *at-most-once* locally (best effort) and *unreliable* remotely; you layer at-least-once and idempotency on top if you need them, exactly as with any [message-delivery-semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) problem.

There is also a quieter cost: **debugging is harder.** A stack trace in a locked program shows you the call chain. In an actor system the "call chain" is a *trail of messages* across mailboxes, possibly across machines, with no synchronous stack linking them. You debug by *tracing messages* — correlation ids, distributed tracing, message logs — not by reading a stack. Tooling has caught up (Erlang's `observer`, Akka's event-stream and Lightbend telemetry, distributed tracing), but the model genuinely costs you the comfortable single-stack mental picture.

## Worked example: a complete bank-account actor

Let me pull it together into the canonical example. We will build an account actor that handles `deposit`, `withdraw`, and `balance`, with no locks, plus a transfer that *demonstrates* the cross-actor problem rather than pretending it away.

#### Worked example: deposit, withdraw, balance with zero locks

The protocol is four messages: `deposit(amount)` (async, no reply needed), `withdraw(amount)` (reply `:ok` or `:insufficient`), `balance()` (reply the number). Each is processed serially, so each touches `balance` alone. Here it is in Rust using the Actix framework, which is the idiomatic actor system in the Rust ecosystem:

```rust
use actix::prelude::*;

// --- Messages (each carries its own reply type via Message::Result) ---
#[derive(Message)]
#[rtype(result = "()")]                 // deposit: no reply (fire-and-forget)
struct Deposit { amount: i64 }

#[derive(Message)]
#[rtype(result = "Result<i64, String>")] // withdraw: reply new balance or an error
struct Withdraw { amount: i64 }

#[derive(Message)]
#[rtype(result = "i64")]                 // balance: reply the number
struct Balance;

// --- The actor: private, sealed state ---
struct Account { balance: i64 }

impl Actor for Account {
    type Context = Context<Self>;        // gives it a mailbox + serial execution
}

// One handler per message. Each runs to completion before the next -- no lock.
impl Handler<Deposit> for Account {
    type Result = ();
    fn handle(&mut self, msg: Deposit, _ctx: &mut Context<Self>) {
        self.balance += msg.amount;      // &mut self is exclusive: single-threaded here
    }
}

impl Handler<Withdraw> for Account {
    type Result = Result<i64, String>;
    fn handle(&mut self, msg: Withdraw, _ctx: &mut Context<Self>) -> Self::Result {
        if msg.amount <= self.balance {
            self.balance -= msg.amount;
            Ok(self.balance)
        } else {
            Err("insufficient funds".into())   // expected case -> reply, don't crash
        }
    }
}

impl Handler<Balance> for Account {
    type Result = i64;
    fn handle(&mut self, _msg: Balance, _ctx: &mut Context<Self>) -> i64 {
        self.balance
    }
}
```

Look at the Rust type system *proving* the isolation for you. The handler signature is `fn handle(&mut self, ...)` — `&mut self` is an *exclusive* borrow. Rust's ownership rules guarantee at compile time that no two `&mut self` borrows exist simultaneously, which is *exactly* the single-threaded-per-actor guarantee, checked by the compiler. The state `balance` is a private field, unreachable from outside. `self.balance += msg.amount` has no lock and needs none, and Rust will not even let you write the racy version — `Account` is not `Sync`, so you cannot share it across threads to begin with. The actor model and Rust's "fearless concurrency" are the same idea from two directions: make the unsafe sharing *unrepresentable*.

Compare the *exact same actor* in Elixir's GenServer (shown earlier) and the shape is identical — `handle_cast`/`handle_call` are Actix's `Handler`s, `{:noreply, new_state}` is returning the next behavior, the `:insufficient_funds` reply is the `Err` branch. Two languages, two type systems, one model. That convergence is the sign you are looking at a real, durable design and not a framework gimmick.

#### Worked example: the transfer that breaks the single-actor guarantee

Now the instructive failure. Transfer 100 from account A to account B:

```rust
// NAIVE transfer -- and it is WRONG as a cross-actor invariant.
async fn transfer(a: &Addr<Account>, b: &Addr<Account>, amount: i64) -> Result<(), String> {
    a.send(Withdraw { amount }).await.unwrap()?;   // step 1: debit A
    // <-- INVARIANT VIOLATED HERE: the money exists in NEITHER account.
    //     A crash, a panic, or a network failure right now LOSES the 100.
    b.send(Deposit { amount }).await;              // step 2: credit B
    Ok(())
}
```

Between the two `await`s the money is gone from A and not yet in B. The system-wide invariant "total money is conserved" is *false* for that window, and no actor's single-threaded guarantee covers it because *two* actors are involved. If the process dies after the withdraw and before the deposit, the 100 simply vanishes. This is not a bug in the framework; it is the model telling you the truth: **it does not do cross-actor atomicity.** The fix is to make the transfer *itself* an actor — a `TransferSaga` that owns the transfer state, drives debit-then-credit, and on failure issues a *compensating* credit back to A. Now the transfer's consistency lives inside *one* actor again, where the guarantee applies, and partial failure becomes a state in that actor's state machine rather than lost money. The model did not solve the distributed-transaction problem; it forced you to *see* it and gave you an actor to put the solution in.

## Measured behavior: actors vs locks, and how many fit

Three numbers matter when you decide whether actors pay off, and I will give defensible orders of magnitude — and tell you exactly how to measure them yourself, because the real numbers depend on your runtime, your hardware, and your message size.

**Throughput: actor vs locked object under contention.** For a *single, hot* piece of state hammered by many threads, an actor and a well-tuned lock are in the same ballpark — both serialize, so both are bounded by the single-threaded service rate, and the throughput ceiling is roughly *one update per (processing time)*. The actor adds the cost of enqueue/dequeue and a context handoff; a lock adds the cost of acquire/release and, under contention, the cost of parking and waking blocked threads. Where the actor *wins* is *under heavy contention*: blocked threads on a contended lock burn CPU spinning or pay the OS the cost of a [context switch](/blog/software-development/concurrency/processes-threads-and-how-the-os-scheduler-runs-them) (order of magnitude ~1–5 µs each) to park and wake, and that overhead *grows* with the number of contending threads — the classic curve that bends *down* as you add threads. An actor's senders never block; they enqueue (tens of nanoseconds) and leave, so adding senders does not add per-update overhead. The honest summary: for low contention a lock is simpler and as fast or faster; for *high* contention on a single entity, the actor's no-blocking-on-send model degrades more gracefully. Measure your own: drive N threads at one counter, plot updates/second vs N for `Arc<Mutex>` and for an actor, and find where each curve bends.

| Property | Locked object | Actor |
| --- | --- | --- |
| State access | shared, every thread reaches in | private, one owner only |
| Serialization | mutex acquire/release | mailbox queue, one at a time |
| Sender under contention | blocks; parks/wakes (~1–5 µs ctx switch) | enqueues (~tens of ns), never blocks |
| Throughput ceiling | ~1 update / critical-section time | ~1 update / message-processing time |
| Failure of one update | exception in caller's stack | crash isolated; supervisor restarts |
| Forgotten synchronization | silent data race | impossible (no shared state) |
| Cross-entity invariant | one lock over both (possible) | not provided; needs a saga/coordinator |
| Scales to | cores on one machine | millions of actors across a cluster |

Figure 5 is the same comparison as a decision matrix you can glance at; Figure 2 is the structural before/after — a contended locked object versus an isolated actor.

![Matrix comparing an actor and a locked object across state, synchronization, failure handling, and scaling, with the actor private and serial and the locked object shared and mutex-guarded](/imgs/blogs/the-actor-model-mailboxes-isolation-and-supervision-5.png)

![Before shows a shared object with a mutex and threads contending, after shows one actor owning private state behind a serial mailbox with no lock and no race](/imgs/blogs/the-actor-model-mailboxes-isolation-and-supervision-2.png)

**How many actors fit?** This is where the model earns its reputation. An actor is *not* an OS thread — that is the whole point. A thread costs you a stack (often ~1 MB of reserved address space) and the kernel's scheduler overhead, so a machine tops out around the low thousands of threads before context-switching dominates. An actor is a lightweight, runtime-scheduled object: an Erlang/Elixir process starts at roughly **300–600 bytes** of heap and is scheduled by the BEAM VM, not the OS; an Akka actor is a few hundred bytes of JVM heap. So instead of thousands you get *millions*. The BEAM is routinely run with **millions of live processes** on a single node; WhatsApp famously reported **over 2 million concurrent TCP connections on one server**, each backed by Erlang processes. The arithmetic is simple: at ~500 bytes per actor, a million actors is ~500 MB — a fraction of a modern server's RAM. You measure it directly: spawn actors in a loop, watch RSS, divide. The number you get tells you whether "one actor per user/connection/account" is affordable — and for the BEAM and Akka, it almost always is.

#### Worked example: the contention curve to expect

Drive a single shared counter from 1, 2, 4, 8, 16, 32, 64 threads. For `Arc<Mutex<i64>>`, throughput typically *rises* to a peak around the core count and then *falls* as more threads pile onto the contended lock — beyond the cores, added threads mostly add park/wake context switches, not work. For an actor wrapping the same counter, throughput rises to the actor's single-threaded service ceiling and then *plateaus* — adding senders does not slow the actor, the mailbox just gets longer (watch it grow!). Neither beats a *single* uncontended thread on the counter (no synchronization at all is always fastest for one hot variable); the actor's advantage is everything *else* — isolation, supervision, and the millions-of-entities scaling that the single-thread baseline cannot give you. The point of the measurement is not "actors are faster" (often they are not for one hot counter) but "actors degrade gracefully and scale in entity-count," which is the axis that matters for real workloads.

## Case studies / real-world

**WhatsApp on Erlang.** The textbook case for actor-model scale. WhatsApp ran its messaging backend on Erlang/OTP and, in a widely cited 2012 engineering report, served **over 2 million concurrent connections on a single FreeBSD server**, later pushing higher — with an engineering team famously in the *dozens* serving hundreds of millions of users. The reason actors fit: a chat connection is a textbook *isolated entity* — one user's session has private state (presence, the socket, in-flight messages) and interacts with others only by sending messages. One Erlang process per connection, supervised, with the BEAM scheduling millions of them, is the model's home turf. The acquisition price (\$19 billion) is the most expensive endorsement the actor model has.

**Discord on Elixir.** Discord built its real-time messaging and presence on Elixir/the BEAM, running *millions* of concurrent users with the same one-process-per-entity approach, GenServers per channel/guild, supervised and distributed across nodes. They have written publicly about scaling Elixir to millions of concurrent users and the engineering they did when a *single* very large GenServer (a "hot" guild) became a mailbox-overflow bottleneck — a perfect illustration of this post's backpressure section: the fix involved sharding the hot actor and adding flow control, because the isolation that scales the common case makes a single hot actor a queue that can overflow.

**The AXD301 telecom switch (Ericsson, Erlang/OTP).** The system Joe Armstrong's thesis is built around: a large ATM switch written in Erlang that reportedly achieved **nine-nines (99.9999999%) availability** in field operation — on the order of *milliseconds* of downtime per year. It got there not by being bug-free but by the supervision-tree discipline: failures were isolated to small processes and restarted faster than they could accumulate into an outage. This is the empirical case that "let it crash plus supervision" produces *more* reliable systems than defensive programming, and it is the origin story of the whole philosophy. (Treat the exact nines as the famous reported figure; the lesson — isolate, restart, supervise — is the durable part.)

**Akka in the JVM world.** Akka brought actors to the JVM and is used in high-throughput systems (Lightbend cites large-scale streaming and stateful-service deployments); the Reactive Streams standard for backpressure grew partly out of the actor/streaming community's need to bound mailboxes. Akka Cluster Sharding routinely distributes millions of *entity actors* (one per order, per device, per session) across a node fleet — the same one-actor-per-entity pattern, now location-transparent across machines.

## When to reach for this (and when not to)

A decisive section, because actors are oversold and underspecified in equal measure.

**Reach for actors when:**

- **Your domain is naturally a population of isolated, stateful entities** — accounts, user sessions, chat rooms, devices, game objects, connections — each with private state and message-based interaction. This is the model's sweet spot; the entity *is* the actor.
- **You need fault tolerance and self-healing** more than you need raw single-thread speed. If "a component crashing should not take down the system, and it should recover on its own," supervision trees are the best tool there is.
- **You need to scale the same code from one machine to a cluster.** Location transparency means you design once and distribute by configuration. If horizontal scale of stateful entities is on the roadmap, actors pay off.
- **You have high contention on many independent entities** (millions of accounts, each lightly contended), where per-entity serialization is exactly right and per-entity parallelism is what you want.

**Do not reach for actors when:**

- **Your core correctness rule is an invariant spanning multiple entities** — a cross-account transfer with a conservation law, a global constraint over many rows, a transaction touching several entities atomically. Actors give you *no* cross-actor atomicity; you will hand-roll sagas or two-phase commit, reintroducing the coordination you came to escape. A database transaction or a single locked aggregate is the simpler, more correct tool here. (See [consistency models](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects).)
- **Your work is straight-line synchronous request/response with no isolated state** — a stateless transform, a pure computation, a CRUD handler over a shared database. Actors add async ceremony and request/reply plumbing for no isolation benefit. A function call or a thread pool is simpler.
- **You have a single hot piece of state and need maximum throughput on it.** An actor serializes it the same as a lock but adds enqueue/handoff overhead; if a plain mutex (or a lock-free structure) is not your bottleneck, do not add an actor runtime. *Measure first.*
- **You cannot afford the operational complexity.** Actors move bugs from "deadlock" to "mailbox overflow" and from "stack trace" to "message trace." If your team is not ready to reason about backpressure, supervision strategies, and distributed-failure semantics, a simpler model with locks or a queue may be the responsible choice.

The honest framing: the actor model is *the best answer to a specific question* — "how do I run a huge population of isolated, fault-tolerant, possibly distributed stateful entities?" — and a *poor answer* to several others. Match the model to the shape of your problem.

![Matrix of actor frameworks listing Erlang on the BEAM for telecom uptime, Akka on the JVM with typed actors, Elixir on the BEAM for web scale, and Orleans and Actix for cloud grains and local async](/imgs/blogs/the-actor-model-mailboxes-isolation-and-supervision-7.png)

Figure 7 maps the framework landscape so you can pick a runtime that fits your stack: **Erlang/OTP** (the BEAM VM, telecom-grade uptime, the original), **Elixir** (also the BEAM, modern syntax, the Phoenix web stack, what Discord uses), **Akka** (Scala/Java on the JVM, typed actors, cluster sharding), and **Orleans** (C#, *virtual* actors/grains the runtime places automatically) and **Actix** (Rust, where the type system enforces the isolation). They are the *same model*; choose by language and operational niche, not by feature checklist.

## Key takeaways

- An **actor** is private isolated state + a **mailbox** + a behavior that processes **one message at a time**. All three are required; the third — single-threaded-per-actor — is what removes the lock.
- **No lock is needed because there is nothing shared to lock.** The race required two concurrent load-modify-store sequences on shared memory; an actor admits exactly one execution context touching its state, so no interleaving is possible. This is a structural guarantee, not a discipline you can forget.
- **Sends are asynchronous and one-way.** Getting an answer means request/reply: carry a reply address and a correlation id in the message, and *always* set a timeout, because a reply is just a message that might never come.
- **"Let it crash" + supervision beats defensive programming** for reliability. Write the happy path, let unexpected failures crash the (isolated) actor, and let a supervisor restart it to a known-good state. The supervision *tree structure* is your fault-tolerance design. But a restart heals the *process*, not the *data* — persist anything you cannot lose.
- **Location transparency** falls out of message-passing: an actor reference works the same locally or remotely, so the same code scales to a cluster. It makes the *syntax* uniform across the network boundary — it does **not** make the *failure modes* uniform. Treat every remote send as potentially undelivered.
- **An unbounded mailbox is an outage waiting for a spike.** Fire-and-forget sends give the sender no backpressure signal, so a slow actor silently grows its queue until OOM. Default to bounded mailboxes with an explicit overflow policy.
- **Actors give no cross-actor invariant and only per-sender ordering.** If a correctness rule cannot be drawn inside one actor's boundary, the model will not enforce it — you will build a saga or coordinator. Match the actor boundary to the consistency boundary.
- **Actors are not faster than a lock on one hot counter** — they serialize the same way plus enqueue overhead. Their wins are isolation, self-healing, and scaling to *millions of entities* across a cluster. Choose them for the shape of the problem (a population of isolated stateful entities), not for raw single-entity speed.

## Further reading

- **Carl Hewitt, Peter Bishop, Richard Steiger — "A Universal Modular ACTOR Formalism for Artificial Intelligence" (1973).** The paper that introduced actors; the three axioms (send, create, designate) are the whole model in one page.
- **Gul Agha — *Actors: A Model of Concurrent Computation in Distributed Systems* (1986).** The formalization that turned Hewitt's idea into a rigorous concurrency model.
- **Joe Armstrong — *Making Reliable Distributed Systems in the Presence of Software Errors* (PhD thesis, 2003).** The "let it crash" and supervision-tree philosophy, derived from building real telecom systems in Erlang. The single best read on *why* the model is shaped the way it is.
- **The Erlang/OTP documentation — `gen_server`, `supervisor`, and the OTP design principles.** How the model is actually engineered: behaviors, restart strategies, the error kernel.
- **The Akka documentation — Actors, Typed, Cluster Sharding, and Akka Streams (Reactive Streams).** The JVM realization, including the backpressure/streaming answer to mailbox overflow.
- **Microsoft Research — "Orleans: Distributed Virtual Actors for Programmability and Scalability."** The *virtual actor* model — runtime-managed placement and activation — and the cloud-scale case for it.
- **The series intro — [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it)** — and the capstone — [the concurrency playbook: choosing the right model](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) — for where actors sit among locks, atomics, channels, and structured concurrency.
- **Siblings: [message-passing vs shared memory and the CSP philosophy](/blog/software-development/concurrency/message-passing-vs-shared-memory-and-the-csp-philosophy)** (the other face of "share by communicating") and **[mutual exclusion: mutexes and critical sections](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections)** (the lock-based model actors replace).
