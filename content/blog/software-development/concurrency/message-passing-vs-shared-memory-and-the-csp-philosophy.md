---
title: "Message Passing vs Shared Memory and the CSP Philosophy"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Two ways to coordinate threads — guard a shared cell with a lock, or give one task ownership and talk to it over a channel — and why the second one deletes a whole class of races."
tags:
  [
    "concurrency",
    "parallelism",
    "message-passing",
    "shared-memory",
    "csp",
    "channels",
    "goroutines",
    "actors",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/message-passing-vs-shared-memory-and-the-csp-philosophy-1.png"
---

Here is a counter. Many threads want to increment it. You have two honest ways to make that safe, and they look nothing alike.

The first way is the one almost everyone learns first. The counter is a number sitting in shared memory. Every thread can reach it, so before any thread touches it you make that thread acquire a lock, do its read-modify-write, and release the lock. The data is shared; you bolt a guard onto it. If you ever forget the guard — one code path, one new contributor, one refactor that moves the increment outside the critical section — you have a data race, and a data race is the kind of bug that passes every test on your laptop and corrupts a balance in production on a Friday.

The second way looks almost strange the first time you see it. You give the counter to **one** task and nobody else is allowed to touch it. That task sits in a loop reading a channel. When another thread wants to add one, it does not reach into the counter — it cannot, the counter is not shared — it *sends a message*: "add 1." The owning task receives the message and does the increment, alone, with no lock, because it is the only thing in the entire program that can write that variable. There is no critical section because there is no sharing. There is no lock because there is nothing to guard. The race is not *prevented*; it is *impossible*, the same way a division-by-zero bug is impossible in code that never divides.

![a comparison table of shared memory with locks versus message passing across who owns the data, synchronization, race risk, and cost](/imgs/blogs/message-passing-vs-shared-memory-and-the-csp-philosophy-1.png)

That second way is the subject of this post, and the figure above is the whole argument in one frame: in the shared-memory column the data is owned by everyone and you pay for a lock to keep them from colliding; in the message-passing column the data has a single owner and threads coordinate by sending. This is the opening of a track in this series I think of as **"avoid sharing."** Everything before it — mutexes, the memory model, atomics, lock-free structures — was an answer to *how do I make shared mutable state safe?* This track asks a different question: *what if I don't share mutable state at all?* The answer has a sixty-year pedigree (Tony Hoare's Communicating Sequential Processes, 1978), a famous one-line slogan from Go ("Do not communicate by sharing memory; instead, share memory by communicating"), and a deep idea underneath both — **ownership** — that Rust later turned into a compiler-enforced guarantee. By the end you will be able to take the same problem, build it both ways, see exactly which class of bugs message passing deletes and which it does not, measure the throughput each one buys, and decide — for a real piece of code in front of you — which paradigm fits.

This is post F1 in the [Concurrency and Parallelism series](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it). It opens the message-passing track and sets up the [actor model](/blog/software-development/concurrency/the-actor-model-mailboxes-isolation-and-supervision) and [CSP channels in practice](/blog/software-development/concurrency/csp-channels-goroutines-and-the-select-statement) that follow. The spine of the whole series is one sentence: **shared mutable state plus nondeterministic scheduling equals the hazard.** Message passing attacks the *first* word of that sentence. Remove the sharing and the hazard cannot form.

## The two paradigms, stated plainly

Let me define the two paradigms precisely, because the rest of the post leans on the distinction and the words get used loosely in the wild.

**Shared memory.** Multiple threads of execution run in one address space — the same heap, the same global variables. A pointer or reference in one thread names the same bytes as the same pointer in another. Threads communicate *implicitly*: thread A writes a field, thread B reads that field and sees the new value (eventually, subject to the memory model). Because the data is reachable by everyone, you must impose order on the accesses by hand — a `mutex`, an atomic, a read-write lock, a barrier — to establish the **happens-before** edges that make a write in one thread visible and indivisible to a read in another. The data is the thing you protect; the synchronization sits *around* the data.

**Message passing.** Each thread (or task, or process, or actor) owns its own private data. No two of them name the same mutable bytes. They communicate *explicitly* by sending values to one another over a conduit — a channel, a mailbox, a queue, a socket. The act of sending and receiving is itself the synchronization: the message *is* the happens-before edge. The data is never shared; what moves between threads is a value (a copy) or an ownership token (a transfer). There is no critical section because there is no shared cell to enter.

The distinction is not about hardware — message passing runs perfectly well inside one process, between two goroutines on the same core, with the "channel" being a small bounded buffer in the same heap. The distinction is about the *programming model*: do two units of execution name the same mutable state (shared memory) or do they only ever exchange messages (message passing)? You can build either model on top of either substrate. Erlang gives you message passing on a single machine; MPI gives you message passing across a cluster; Go gives you message passing between goroutines that share an address space but are *disciplined* not to touch each other's data. Conversely, two processes can share memory through `mmap` or POSIX shared-memory segments, dragging all the locking problems across the process boundary with them.

So the cleanest way to hold the two in your head: **shared memory shares the data and synchronizes the access; message passing isolates the data and synchronizes the communication.** Same goal — establish order over concurrent operations so the result is correct — opposite strategy.

A small honesty note before we go deeper. "Shared memory bad, message passing good" is a slogan, not a law. Both are correct tools; both have failure modes. Shared memory can deadlock on lock cycles; message passing can deadlock on channel cycles. Shared memory races on forgotten locks; message passing can still have *logical* race conditions in the order messages arrive. The point of this post is not to anoint a winner. It is to make the trade precise so you choose deliberately instead of by habit.

## The shared-memory paradigm: lock the data

Let me make the shared-memory version concrete first, because it is the baseline everything else is measured against, and because seeing it clearly is what makes the contrast land.

The canonical hazard is the **lost update**, walked instruction by instruction in the [race-condition post](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition). A line like `count++` is not one operation; it is three — load the current value into a register, add one, store the register back. Two threads can interleave so that both load the same old value, both add one, and both store the same new value, and one of the two increments simply vanishes. The fix in the shared-memory model is to make the read-modify-write **atomic** with respect to other threads by wrapping it in a critical section guarded by a lock.

Here is the lock-guarded counter in Go:

```go
package main

import (
	"sync"
)

type Counter struct {
	mu  sync.Mutex
	val int64
}

func (c *Counter) Add(n int64) {
	c.mu.Lock()
	c.val += n // safe: only one goroutine in here at a time
	c.mu.Unlock()
}

func (c *Counter) Value() int64 {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.val
}
```

Every `Add` and every `Value` takes the same lock, so the read-modify-write is indivisible from any other thread's point of view. This is correct. It is also the model where a single forgotten `Lock()` reintroduces the bug, where the lock can be held across an I/O call and stall everyone, and where, under heavy contention, threads spend more time fighting over the lock than doing work.

The same idea in Rust looks different on the surface but is the same paradigm — shared data, guarded by a lock — and Rust's type system makes the sharing *explicit* in a way Go does not. You cannot share a `Mutex` across threads by accident; you have to wrap it in an `Arc` (an atomically reference-counted pointer) and the compiler checks that the type is safe to share:

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0i64));
    let mut handles = Vec::new();

    for _ in 0..8 {
        let c = Arc::clone(&counter); // a shared handle to the same cell
        handles.push(thread::spawn(move || {
            for _ in 0..1_000_000 {
                let mut guard = c.lock().unwrap(); // acquire
                *guard += 1; // critical section
            } // guard drops here -> release
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    println!("{}", *counter.lock().unwrap());
}
```

Notice `Arc<Mutex<i64>>`. The `Mutex` provides the mutual exclusion; the `Arc` provides the *sharing*. In Rust you literally cannot move a plain `&mut i64` into two threads — the borrow checker forbids two mutable references to the same data — so the only way to share mutable state across threads is to opt in, visibly, with a synchronization wrapper. Rust did not invent shared memory; it made the dangerous part of it impossible to do silently.

And in Java, the same paradigm with the `synchronized` keyword or an explicit `ReentrantLock`:

```java
public final class Counter {
    private long val = 0;
    private final Object lock = new Object();

    public void add(long n) {
        synchronized (lock) {
            val += n; // critical section
        }
    }

    public long value() {
        synchronized (lock) {
            return val;
        }
    }
}
```

All three are the *same paradigm*: the data (`val`, the `i64`, `count`) lives in one place; every thread can reach it; you serialize access with a lock to keep them from colliding. The lock is the price of sharing. Hold that picture. We are about to delete it.

## The message-passing paradigm: own the data, send messages

Now the other way. Instead of letting every thread reach the counter and guarding the counter, we hand the counter to **one** task and let nobody else touch it. Other threads that want to change the counter send it a message. The owner processes messages one at a time, sequentially, so its updates are automatically serialized — not because a lock forces them to be, but because there is only one thread doing them.

![a before and after comparison showing a lock-guarded shared counter versus a counter owned by one task that others update by sending messages](/imgs/blogs/message-passing-vs-shared-memory-and-the-csp-philosophy-2.png)

The figure contrasts the two shapes. On the left, the shared cell with a lock bolted on, reachable by any thread. On the right, the same count owned by a single task; other threads send "add" messages over a channel and the owner — and only the owner — performs the write. There is no lock on the right because there is nothing to guard: only one thread can ever write that variable, so a data race on it is not possible.

Here is the owned-counter version in Go. This is the idiomatic Go way to protect state without a mutex — a goroutine that owns the state and a channel that carries requests to it:

```go
package main

import "sync"

// requests sent to the owner goroutine
type addReq struct{ n int64 }
type getReq struct{ reply chan int64 }

type Counter struct {
	adds chan addReq
	gets chan getReq
}

func NewCounter() *Counter {
	c := &Counter{
		adds: make(chan addReq, 256),
		gets: make(chan getReq),
	}
	go c.run() // the single owner of `val`
	return c
}

// run owns `val` exclusively; no other goroutine can see it.
func (c *Counter) run() {
	var val int64 // private to this goroutine
	for {
		select {
		case req := <-c.adds:
			val += req.n // no lock: only this goroutine touches val
		case req := <-c.gets:
			req.reply <- val
		}
	}
}

func (c *Counter) Add(n int64) { c.adds <- addReq{n} }

func (c *Counter) Value() int64 {
	reply := make(chan int64)
	c.gets <- getReq{reply}
	return <-reply
}

func main() {
	c := NewCounter()
	var wg sync.WaitGroup
	wg.Add(8)
	for i := 0; i < 8; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < 1_000_000; j++ {
				c.Add(1)
			}
		}()
	}
	wg.Wait()
	println(c.Value())
}
```

Look at `val`. It is a local variable inside `run`. No other goroutine has a pointer to it, can name it, or can write it. The only way the outside world affects `val` is by sending an `addReq` on the `adds` channel, and the `run` goroutine applies every increment itself, one at a time, in the order it receives them. There is **no mutex** in this program. The serialization that the mutex used to provide is now provided by the structure: a single owner, a single sequential loop. The synchronization moved from "guard the data" to "route every change through one place."

The `Value()` method shows the round-trip pattern: to *read* the owned state you also send a message — a `getReq` carrying a private reply channel — and the owner sends the current value back. The read is serialized against the writes for free, because they all go through the same `select` loop. This is the request-reply idiom of message passing, and notice it has no risk of reading a half-updated value: the owner is not writing `val` at the moment it is replying, because it does exactly one thing per loop iteration.

The same shape in Rust, using the standard library's `mpsc` (multi-producer, single-consumer) channel. The owner thread holds `val`; senders ship `Msg` values to it:

```rust
use std::sync::mpsc::{self, Sender};
use std::thread;

enum Msg {
    Add(i64),
    Get(Sender<i64>), // a reply channel travels inside the message
}

fn spawn_counter() -> Sender<Msg> {
    let (tx, rx) = mpsc::channel::<Msg>();
    thread::spawn(move || {
        let mut val: i64 = 0; // owned solely by this thread
        for msg in rx { // blocks until a message arrives
            match msg {
                Msg::Add(n) => val += n, // no lock: single owner
                Msg::Get(reply) => {
                    let _ = reply.send(val);
                }
            }
        }
    });
    tx
}

fn main() {
    let tx = spawn_counter();
    let mut handles = Vec::new();
    for _ in 0..8 {
        let tx = tx.clone(); // each producer gets its own sender
        handles.push(thread::spawn(move || {
            for _ in 0..1_000_000 {
                tx.send(Msg::Add(1)).unwrap();
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let (rtx, rrx) = mpsc::channel();
    tx.send(Msg::Get(rtx)).unwrap();
    println!("{}", rrx.recv().unwrap());
}
```

The Rust version makes the ownership story unmistakable: `val` is a local `mut i64` inside the spawned closure. There is no `Mutex`, no `Arc`, no shared reference anywhere. The compiler *moves* the channel receiver `rx` into the owner thread, so no other thread can even name it. The eight producers each own a clone of the sender. Mutable state exists, but it is not shared, so there is no `Send`/`Sync` headache, no lock, no possibility of a forgotten guard.

And Erlang, the language that made this paradigm its entire identity. In Erlang there is *no shared memory at all* — processes cannot share data even if they want to — so message passing is the only option, and the "owned counter" is just a process holding state in its recursive loop:

```erlang
-module(counter).
-export([start/0, add/2, value/1]).

start() ->
    spawn(fun() -> loop(0) end).   % a process owning `Val`

loop(Val) ->
    receive
        {add, N}        -> loop(Val + N);          % apply, recurse with new state
        {get, From}     -> From ! {value, Val},     % reply with the value
                           loop(Val);
        stop            -> ok
    end.

add(Pid, N)  -> Pid ! {add, N}.

value(Pid) ->
    Pid ! {get, self()},
    receive {value, V} -> V end.
```

`Val` is an argument to `loop`. Each message produces a new `loop` call with the updated state. There is nothing to lock because Erlang processes are isolated by the runtime — one process literally cannot reach another's heap. This is the message-passing paradigm in its purest form, and it is why a single Erlang node can run millions of these processes without a single mutex anywhere in the user's code.

Three languages, one idea: *the state has a single owner, and you change it by sending the owner a message.* The lock is gone because the sharing is gone.

## CSP and the channel as the unit of synchronization

The theory under all of this is older than any of these languages. In 1978 Tony Hoare published **Communicating Sequential Processes** (CSP), a formal model of concurrency built on a deceptively small idea: a system is a collection of independent sequential processes that have *no shared state* and interact *only* by communication. The communication is the synchronization. There is no separate notion of a lock; the act of two processes communicating is the only way they coordinate.

In Hoare's original CSP, communication is **synchronous** and **unbuffered** — a *rendezvous*. When process A wants to send a value to process B, A blocks until B is ready to receive, and B blocks until A is ready to send. They meet at a single instant, the value passes, and both continue. Neither side proceeds past the communication until the other has arrived. This is profound: the send and the receive are *the same event*. There is no buffer in between, no "the message is sitting in a queue" — the handoff is a synchronization point as crisp as a lock acquire, except it carries data and it coordinates two named parties rather than guarding a region.

The **channel** is the unit that makes this concrete. A channel is a typed conduit: one side sends, the other receives. In a synchronous (unbuffered) channel, send blocks until receive happens — pure CSP rendezvous. Add a buffer of capacity N and the send only blocks when the buffer is full, decoupling the two parties up to N in-flight messages; this is the asynchronous variant. Either way, the channel operation establishes a **happens-before** edge: everything the sender did before the send *happens-before* everything the receiver does after the receive. That is exactly the ordering guarantee a lock gives you, delivered by the act of communication instead of by guarding memory.

![a tree of communication models splitting into synchronous rendezvous and asynchronous delivery with channels and mailboxes underneath](/imgs/blogs/message-passing-vs-shared-memory-and-the-csp-philosophy-5.png)

The tree shows the taxonomy. Synchronous communication — CSP's rendezvous, a Go channel with capacity zero — blocks both sides until they meet. Asynchronous communication — a buffered channel with N slots, an actor's unbounded mailbox — decouples the sender from the receiver so the sender can run ahead. The way this works is the same regardless of variant: the channel carries the value *and* the ordering edge; what changes is how much the two parties are allowed to drift apart in time.

Go is the language that brought CSP to the mainstream, and its channels are CSP channels almost verbatim. An unbuffered Go channel is a rendezvous:

```go
ch := make(chan int) // unbuffered: send blocks until a receiver is ready

go func() {
	ch <- 42 // blocks here until main receives
}()

v := <-ch // rendezvous: this receive unblocks the send above
// everything the goroutine did before `ch <- 42`
// happens-before everything here after `<-ch`
```

The send `ch <- 42` and the receive `<-ch` are a single synchronized event. The goroutine cannot pass the send until `main` reaches the receive. This is not a queue; it is a handshake that happens to carry an integer. A buffered channel relaxes it:

```go
ch := make(chan int, 3) // buffered: 3 slots

ch <- 1 // does not block (slot available)
ch <- 2 // does not block
ch <- 3 // does not block (buffer now full)
// ch <- 4 would block here until someone receives
```

Now the sender can deposit three values and run ahead without a receiver present; only the fourth send blocks. The buffer is the dial between pure synchronous CSP (capacity 0) and fully decoupled asynchronous messaging (large or unbounded capacity). The right buffer size is a real engineering decision — too small and you serialize unnecessarily, too large and you hide backpressure and let a fast producer pile up unbounded memory. We will return to backpressure in the trade-offs.

The key mental shift CSP asks of you is this: stop thinking of synchronization as something you *add around data* and start thinking of it as something that *happens when two processes communicate*. The channel is not a data structure you protect; the channel is the synchronization primitive itself.

## "Share memory by communicating," explained

Go's documentation states the philosophy as a slogan you have probably seen on a sticker: **"Do not communicate by sharing memory; instead, share memory by communicating."** It is genuinely deep once you unpack it, and genuinely easy to misread as a ban on shared memory, which it is not.

Read it as a contrast between two ways of getting data from thread A to thread B.

**Communicate by sharing memory** is the shared-memory way. A and B both have a reference to a shared structure. A writes into it; B reads from it. To coordinate, they share *the memory* and then bolt synchronization on so B knows when A's write is complete and visible. The data sits still and is touched by both; the lock arbitrates the touching. The communication is *implicit* — B finds out A did something by reading a field A changed.

**Share memory by communicating** inverts it. A does not give B a reference to its data; A *sends* the data — or ownership of it — to B over a channel. The memory is "shared" only in the sense that the *value* makes its way from A to B, but at no instant do both A and B hold a live mutable reference to the same bytes. The communication is *explicit* — A literally hands the value to B — and the handoff is the synchronization, so there is no separate lock and no window where the data is in an inconsistent state visible to both.

The slogan's punchline is that the second style makes the synchronization *fall out of the communication for free*. You were going to move data from A to B anyway; if you do it by sending a message, the happens-before edge you needed comes bundled with the send. You do not have to *also* remember to take a lock, because the channel already serialized things. The bug-prone step — "remember to synchronize around the shared data" — disappears, because the synchronizing and the communicating are the same act.

There is a subtlety worth stating, because people quote the slogan as if shared memory were forbidden in Go. It is not. Go ships `sync.Mutex`, `sync.RWMutex`, `sync/atomic`, and the docs are explicit that for some problems — a reference counter, a small map behind a lock, a hot in-memory cache — a plain mutex is simpler and faster than a channel, and you should use it. The slogan is a *default*, a tie-breaker, a way of thinking — when in doubt, prefer passing the data over sharing it — not a prohibition. The mature reading is: reach for communication first because it tends to produce code where the ownership of every piece of mutable state is obvious; reach for a lock when the data really is shared, hot, and small, and a channel would just be ceremony.

The deepest version of the idea is *ownership*. "Share memory by communicating" works cleanly precisely when the send **transfers ownership** of the data rather than aliasing it. If A sends B a value and then keeps mutating its own copy, you are back to a kind of sharing (two copies that can diverge) — which may be fine, or may be a bug, depending on intent. If A sends B the data and *gives it up* — A no longer touches it after the send — then there is exactly one owner at every instant and the safety is airtight. That is the next section.

## Copying versus ownership transfer

When you "send" a value over a channel, one of two things physically happens, and the difference matters enormously for both correctness and cost.

**Copy.** The runtime duplicates the data and hands the receiver a fresh, independent copy. The sender keeps its original; the receiver has its own. Now there is no aliasing — two separate cells, no shared mutable state — so there is no race, but you paid to copy the bytes. For a small message (an integer, a small struct) this is trivial. For a large message (a megabyte buffer, a big slice) copying on every send is real cost.

**Ownership transfer (move).** Nothing is duplicated. What moves is a *pointer* plus the *right to use it*. The sender hands the receiver the address of the data and *gives up its own access*. After the send, the sender must not touch the data; the receiver is now the sole owner. This is O(1) regardless of how big the data is — you move a pointer, not the bytes — and it is still race-free, because at every instant exactly one party may touch the data.

![a timeline of ownership transfer over a channel where the sender builds a value, gives it up at send, and the receiver owns it after receive](/imgs/blogs/message-passing-vs-shared-memory-and-the-csp-philosophy-4.png)

The timeline traces a single ownership transfer. The sender builds a buffer and owns it — it can still write into it. At the send, it gives up ownership; while the value is in flight, no one writes it. At the receive, the receiver takes ownership and from then on the sender cannot touch it. There is never a moment when both sides hold a live writable handle to the same bytes. That "never both at once" is the entire safety argument, and notice it is a property of *ownership*, not of locking.

![a matrix comparing copy and move semantics across cost, safety, and the language idiom for each](/imgs/blogs/message-passing-vs-shared-memory-and-the-csp-philosophy-7.png)

The matrix lines the two up. Copy is O(size) and leaves both sides with a value; move is O(1) and leaves only the receiver with access. Both are safe. The difference is cost and which language idiom enforces it.

Rust is the language that made ownership transfer a *compiler-checked* guarantee, and this is where it shines. In Rust, sending a non-`Copy` value over a channel **moves** it. After `tx.send(buf)`, the variable `buf` is gone — the compiler will reject any later use of it. There is no convention to remember and no way to get it wrong; the type system enforces "the sender gave it up":

```rust
use std::sync::mpsc;
use std::thread;

fn main() {
    let (tx, rx) = mpsc::channel::<Vec<u8>>();

    thread::spawn(move || {
        let buf = vec![0u8; 1_000_000]; // sender builds + owns a 1 MB buffer
        tx.send(buf).unwrap();          // ownership MOVES into the channel
        // println!("{}", buf.len());   // COMPILE ERROR: value used after move
    });

    let received = rx.recv().unwrap();   // receiver now owns the 1 MB buffer
    println!("got {} bytes", received.len());
}
```

The commented line is the whole point. Uncomment it and the program does not compile — `error[E0382]: borrow of moved value: buf`. The compiler has *proven*, statically, that after the send the sender cannot touch the buffer, so the one-megabyte transfer is O(1) (a pointer moves into the channel) *and* provably race-free. This is what people mean by Rust's "fearless concurrency": the move semantics that make message passing safe are not a runtime check or a programmer convention; they are a compile-time theorem. The marker traits `Send` (this type is safe to move to another thread) and `Sync` (this type is safe to share by reference between threads) are how Rust tracks which transfers are legal — a `Vec<u8>` is `Send`, so it can move across a channel; an `Rc<T>` (non-atomic reference count) is *not* `Send`, so the compiler stops you from sending it to another thread where its count could be raced.

Go takes a different stance: it does *not* enforce ownership transfer; it relies on **convention**. When you send a slice or a pointer over a Go channel, what crosses is the slice header or the pointer — a reference, not a deep copy. The Go runtime does not stop you from continuing to use the slice after sending it. The discipline "don't touch what you sent" is on *you*. Go's race detector (`-race`) will catch you if both sides actually touch the same memory concurrently, but nothing at compile time forbids it:

```go
buf := make([]byte, 1_000_000)
ch <- buf       // the slice HEADER is copied; the backing array is shared
// buf[0] = 9   // legal Go, but a BUG: you just mutated data you "gave away"
```

Sending `buf` copies the 24-byte slice header, not the megabyte of backing storage — so it is cheap, like a move — but the backing array is now reachable from both the sender's `buf` and the value the receiver got. If both touch it, that is a data race the language will not stop at compile time. The Go idiom is therefore "send it and forget it": once you put a slice or pointer on a channel, treat it as gone. It is a *convention* that gives you move-like cost (cheap handoff) without move-like *enforcement* (the compiler does not check you kept your promise).

If you want a true copy in Go — so the sender can keep using its data safely — you copy explicitly before sending, or you send a value type (a struct passed by value is copied into the channel). Sending a struct by value is the cheap, safe default for small messages; sending a pointer or slice is the cheap, *convention-bound* default for large ones.

So the spectrum is: **copy** (safe, costs O(size), both keep a value — Go value sends, an explicit clone) versus **move** (safe, costs O(1), only the receiver keeps access — Rust's enforced moves, Go's by-convention pointer/slice handoff). The race-free property comes from "exactly one owner at any instant." Copy achieves it by making two independent things; move achieves it by transferring the single thing. Rust proves the move at compile time; Go asks you to keep the promise.

#### Worked example: a megabyte buffer, copy versus move

Suppose a producer builds 1 MB image buffers and sends each to a worker for processing, at 10,000 buffers per second.

If every send **copies** the megabyte, you move 10 GB/s of memory traffic just for the handoff — likely your bottleneck, and a waste, because the producer does not need its copy after sending.

If every send **moves** (Rust) or hands off a pointer by convention (Go), you move 10,000 pointers per second — call it 80 KB/s of header traffic — and the producer is forbidden (Rust) or disciplined (Go) from touching the buffer afterward. The work the worker does on the buffer is unchanged; the *handoff* went from 10 GB/s to negligible. Same safety (one owner at a time), four to five orders of magnitude less handoff cost. This is why "transfer ownership, don't copy" is the rule for large messages, and why a language that can *prove* you gave up the buffer (Rust) lets you do this fearlessly while a language that relies on convention (Go) lets you do it but will only catch the mistake at runtime with `-race`.

## Why message passing kills data races by construction

Here is the mechanistic heart of the post — the reason this paradigm is worth a whole track. Message passing does not *reduce* the probability of a data race. It makes a data race *impossible*, by construction, the way an unplugged toaster cannot electrocute you.

Recall the precise definition from earlier in the series: a **data race** is two or more threads accessing the same memory location concurrently, at least one access being a write, with no happens-before edge ordering them. Three ingredients: (1) same location, (2) concurrent, (3) at least one writer, with no synchronization between them. Remove *any one* ingredient and the data race cannot exist.

The lock approach removes ingredient (2)/(3)-without-order: it inserts a happens-before edge (acquire/release) so the accesses are no longer unordered. But the *same location* is still touched by multiple threads — so a single forgotten lock puts ingredient (2) back and the race returns. The lock is a guard you must remember to deploy on every path.

Message passing removes ingredient (1): **there is no same location touched by multiple threads.** The owned data is named by exactly one thread. No other thread holds a reference, a pointer, an alias — nothing. The other threads have a *channel*, not the data. When you grep the entire program, the variable `val` appears only inside the owner. There is no second writer because there is no second namer. With ingredient (1) absent, ingredients (2) and (3) are moot: you cannot have concurrent access to a location that only one thread can reach.

![a before and after figure showing a data race is possible with shared memory and impossible by construction with message passing](/imgs/blogs/message-passing-vs-shared-memory-and-the-csp-philosophy-6.png)

The figure states the contrast as a safety property. On the shared-memory side a race is *possible* — two threads can write one cell, a forgotten lock yields a lost update, the bug appears one time in a million. On the message-passing side a race is *impossible* — there is no shared mutable cell, only one owner writes at a time, so there is nothing to race on. The asymmetry is the whole value proposition: with locks, correctness depends on *every* programmer *every* time remembering the guard; with ownership, correctness is a structural fact you can verify by reading where the data is declared.

This is why the property is "by construction" rather than "by discipline." A property by discipline holds as long as everyone follows the rules; one slip breaks it. A property by construction holds because the alternative is not expressible. In Erlang it is literally not expressible — the runtime gives no way for one process to reach another's heap, so a shared-memory data race between Erlang processes cannot be written. In Rust the borrow checker makes it not expressible without an explicit, visible escape hatch (`unsafe`, or a `Mutex` you opted into). In Go it is expressible if you ignore the convention and keep touching what you sent — Go gives you the gun and trusts you — but the *paradigm*, followed, still removes the race.

Let me be scrupulously honest about the boundary, because this is where people oversell message passing. What it kills is the **data race** — the low-level, memory-model, "torn read / lost update" hazard. It does **not** automatically kill the broader **race condition** — a *logical* timing bug where the *order* of correctly-synchronized operations produces a wrong outcome. If two clients send "withdraw \$100" messages to an account actor that has \$150, the actor processes them one at a time (no data race — its balance is never torn), but if it naively says yes to both before checking, you have a logical race condition that overdrew the account. The actor processing messages sequentially gives you *atomicity per message* for free; it does not give you the *business invariant* for free. You still have to write the check. Message passing deletes the entire category of memory-level races; it leaves the application-level ordering questions on your desk, where they belong.

## The trade-offs: copy cost, no cross-owner invariant, channel deadlocks

If message passing only had upsides everyone would use it for everything, and they don't, because it has three real costs. Naming them precisely is what separates "I read a slogan" from "I can choose."

**Copy cost (or pointer-chasing cost).** Shared memory's whole advantage is that data does not move — a thread reads the shared structure in place, at memory speed, no handoff. Message passing moves data between owners. If you copy, you pay O(size) per message; if you transfer ownership you avoid the copy but you still pay the *channel operation* itself — a send/receive is not free. A channel send involves a lock or a CAS loop on the channel's internal queue, possibly a goroutine/thread park-and-wake if the other side is not ready, and a scheduler hop. Order of magnitude: an uncontended mutex acquire is ~25 ns; a channel send/receive round-trip is more like ~100 ns and up, sometimes much more if it forces a context switch (~1–5 µs). For a hot, tiny, high-frequency update — incrementing a counter ten million times a second — routing every increment through a channel can be *slower* than a plain atomic or a mutex, because the per-message overhead dominates the actual work. We will measure exactly this in a moment.

**No cross-owner invariant.** This is the deep one. Shared memory lets you hold an invariant across multiple pieces of state *atomically*: take one lock, move \$100 from account A to account B, release — and no other thread ever sees a state where the money has left A but not arrived at B. The invariant "total money is conserved" holds at every observable instant. Message passing *isolates* the state, so if A and B are owned by *different* tasks, there is no single moment you can update both. You send "subtract 100" to A and "add 100" to B; between those two messages being processed, the system genuinely is in a state where the \$100 exists in neither account (or in both, depending on order). To restore the cross-owner invariant you need a *protocol* — a two-phase commit, a transaction coordinator, a saga — which is more code and more failure modes than a single lock would have been. **When two pieces of state must change together atomically, they want to live under one owner (or one lock); splitting them across message-passing boundaries makes the invariant your problem to re-establish.**

**Channels can still deadlock.** Message passing does not free you from deadlock; it gives you a *different* deadlock. Two goroutines each waiting to receive from a channel the other will only send to *after* receiving — a classic cyclic wait, exactly the shape locks deadlock in, just with channels as the resource. An unbuffered send with no receiver blocks forever. A `select` with no ready case and no default blocks. A pipeline where the consumer dies and the producer keeps sending into a now-unread channel blocks the producer. The Coffman conditions for deadlock — mutual exclusion, hold-and-wait, no preemption, circular wait — are satisfiable with channels as readily as with locks. You traded "deadlock on a lock cycle" for "deadlock on a channel cycle," not "no deadlock."

There is a fourth, softer cost: **backpressure becomes explicit and you must design it.** A buffered channel with capacity N is a bounded queue; when a producer outruns a consumer, the buffer fills and the producer blocks (good — that *is* backpressure). But choose the buffer too large, or unbounded (Erlang mailboxes are unbounded by default), and a fast producer piles up messages until you run out of memory. Shared memory does not have this failure mode because there is no queue to overflow. With message passing you own the flow-control design — a topic deep enough that the [message-queue series covers it across processes](/blog/software-development/message-queue/message-queues-async-decoupling-and-load-leveling), and the same logic applies in-process between goroutines.

Here is the comparison as a table, because this is the kind of trade you want to be able to recall under pressure:

| Concern | Shared memory + locks | Message passing |
| --- | --- | --- |
| Data race on a cell | Possible (forgotten lock) | Impossible (no shared cell) |
| Where data lives | One place, all touch it | One owner, others send |
| Synchronization | Lock around the data | The send/receive itself |
| Per-op cost (hot path) | Lock ~25 ns, atomic ~5 ns | Channel ~100 ns+, maybe a context switch |
| Large data | Read in place, no copy | Copy O(size), or move O(1) |
| Cross-state invariant | One lock spans both | Needs a protocol (2PC, saga) |
| Failure mode | Lock-cycle deadlock | Channel-cycle deadlock, mailbox overflow |
| Scales to | One address space | One machine, or across the network |

That last row hides a strategic advantage of message passing worth stating: because it never assumes shared memory, the *same* model extends across processes and across machines. An actor that talks only by messages does not care whether the recipient is a goroutine next door or a process on another continent — the send is the send. Shared memory stops dead at the address-space boundary. This is why message-passing systems (Erlang/OTP, Akka, anything built on queues) scale to distribution naturally, and why "share memory by communicating" is also, quietly, "be ready to run on more than one box."

## When to share, when to message

So you have a piece of concurrent code. Which paradigm? Here is the decision the way I actually make it, reduced to the features that swing it.

![a matrix mapping situations like independent workers, a tight shared invariant, large data, and high frequency to message passing or shared memory](/imgs/blogs/message-passing-vs-shared-memory-and-the-csp-philosophy-8.png)

The matrix is the cheat sheet. Read each row, pick the column that fits your situation, and bias toward it.

**Reach for message passing when:**

- **The work decomposes into independent units that pass results along.** A pipeline (parse → transform → write), a pool of workers consuming a job queue, request handlers that don't share session state — these are *naturally* owned-and-communicating. Channels make the dataflow visible and the ownership obvious, and you get parallelism for free with zero locks.
- **You want isolation and fault containment.** If one owner crashes, it does not corrupt anyone else's state — there is no shared state to corrupt. This is the foundation of Erlang's "let it crash" supervision: an isolated process can be killed and restarted without leaving shared data in a torn state. (The [actor model post](/blog/software-development/concurrency/the-actor-model-mailboxes-isolation-and-supervision) builds this out.)
- **You might need to distribute.** If today's two threads could become tomorrow's two services, message passing is the model that survives the move across a network boundary. Shared memory does not.
- **You want the ownership of every mutable thing to be obvious in the code.** Channels force you to name, per datum, "who owns this," which is a documentation and review benefit that pays off long after the code is written.

**Reach for shared memory + a lock (or atomic) when:**

- **Several pieces of state must change together atomically** — a cross-cutting invariant. One lock spanning both is dramatically simpler than a commit protocol across owners. Don't split state that must move as a unit.
- **The data is hot, small, and read or updated at very high frequency.** A reference count, a shared metric counter, a small config map read on every request — an atomic or a tiny mutex beats routing every access through a channel, because the channel's per-op overhead would dominate. Use `sync/atomic`, `AtomicLong`, `std::atomic` — and if even an atomic contends, that is the lock-free / sharding conversation, not the channel one.
- **You truly share a large structure that many readers need in place** — a big in-memory index, a cache — and copying it per access is absurd. A read-write lock (many readers, one writer) or a copy-on-write snapshot fits; a channel does not.

The honest meta-rule: **default to message passing for the *architecture* — how tasks decompose and hand work to each other — and use shared memory with a lock or atomic for the *hot, small, tightly-coupled* state where a channel would be ceremony and overhead.** The two coexist in every real system. A Go service typically has channels structuring the request pipeline *and* a couple of mutex-guarded caches on the hot path. Choosing one paradigm for the *whole* program is the mistake; matching the paradigm to each piece of state is the craft.

And the anti-recommendations, stated bluntly because they are where teams waste weeks: **don't** put a channel in front of a single integer counter you increment in a tight loop — use an atomic. **Don't** model a cross-entity invariant (transfer between two accounts) as messages to two separate owners and then bolt on a distributed transaction — put both accounts under one owner or one lock. **Don't** reach for message passing because "locks are scary" — a single mutex around a small struct is simpler and faster than a goroutine, a channel, and a request-reply protocol. And **don't** assume message passing means no deadlock — draw the channel dependency graph the same way you'd draw the lock graph.

## Worked example: the same problem, both ways

Let me make the choice concrete with one problem solved both ways, then measured. The problem: a bank account that accepts concurrent deposits and withdrawals, where a withdrawal must be rejected if it would overdraw. This problem has a *cross-operation invariant* (balance never negative), which is exactly the kind of thing that pulls toward shared memory — and it is instructive to see message passing handle it too.

#### Worked example: an account, shared-memory version

The shared-memory version is a struct with a mutex. The invariant (no overdraft) is checked and the balance updated inside one critical section, so no other thread can observe or act on an intermediate state:

```go
type Account struct {
	mu      sync.Mutex
	balance int64
}

// Withdraw returns false if it would overdraw. The check-and-update is
// atomic because the whole thing is under one lock.
func (a *Account) Withdraw(amount int64) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.balance < amount {
		return false // reject: would overdraw
	}
	a.balance -= amount
	return true
}

func (a *Account) Deposit(amount int64) {
	a.mu.Lock()
	a.balance += amount
	a.mu.Unlock()
}
```

The invariant lives inside the lock. Two concurrent withdrawals cannot both pass the `balance < amount` check on a balance that only covers one — because the second one waits for the lock, then sees the already-decremented balance, then correctly rejects. Simple, correct, fast. This is shared memory doing what it does best: an invariant that spans a read and a write, held atomically by one lock.

#### Worked example: an account, message-passing version

The message-passing version gives the account to one owner goroutine. Withdrawals are *requests* carrying a reply channel; the owner checks the invariant and replies yes/no. The balance is never shared — it is a local inside the owner — so there is no lock and no possibility of a torn balance:

```go
type withdraw struct {
	amount int64
	reply  chan bool
}
type deposit struct{ amount int64 }

type Account struct {
	withdraws chan withdraw
	deposits  chan deposit
}

func NewAccount(initial int64) *Account {
	a := &Account{
		withdraws: make(chan withdraw),
		deposits:  make(chan deposit, 64),
	}
	go func() {
		balance := initial // owned solely by this goroutine
		for {
			select {
			case w := <-a.withdraws:
				if balance < w.amount {
					w.reply <- false // reject, invariant preserved
				} else {
					balance -= w.amount
					w.reply <- true
				}
			case d := <-a.deposits:
				balance += d.amount
			}
		}
	}()
	return a
}

func (a *Account) Withdraw(amount int64) bool {
	r := make(chan bool)
	a.withdraws <- withdraw{amount, r}
	return <-r
}

func (a *Account) Deposit(amount int64) { a.deposits <- deposit{amount} }
```

Here is the instructive part. The invariant *is* preserved, and not because of a lock — because the owner processes messages **one at a time**. Two concurrent `Withdraw` calls become two `withdraw` messages in the channel; the owner handles the first completely (check, decrement, reply) before it even looks at the second, so the second sees the updated balance and correctly rejects an overdraft. The sequential message loop gives you the same atomicity the lock gave you, for *operations that touch only this one account's state*.

But watch the boundary. Suppose you now need to **transfer** between two accounts atomically — debit A, credit B, never observable in between. In the lock version you take both locks (carefully, in a fixed order to avoid deadlock) and do both updates in one critical section. In the message version, A and B are *different owners*; there is no single message that touches both, so a transfer becomes "send debit to A, wait for ok, send credit to B" — and between those two steps the money is in flight, the cross-account invariant momentarily violated, and if the process crashes after the debit you have lost money. To fix it you need a coordinator or a saga. This is the "no cross-owner invariant" cost made painfully concrete: message passing handled the *single-account* invariant beautifully and made the *two-account* invariant a distributed-transaction problem. The shared-memory version handled both with one more `Lock()`.

That is the real lesson of the worked example, and it is more useful than either version alone: **message passing turns per-owner invariants into free sequential atomicity and turns cross-owner invariants into protocol work.** Match the ownership boundary to your invariant boundary and message passing is a joy; cut across an invariant with an ownership boundary and you have signed up for two-phase commit.

## Measured: throughput, shared versus message-passed

Now the part the slogans skip: what does it *cost*? Let me give you a defensible picture of the throughput trade-off, with the honest caveats, because the numbers are the difference between an informed choice and a cargo-cult one.

The benchmark: eight goroutines hammering a single counter, 1,000,000 increments each (8,000,000 total), four ways — (a) a plain `sync.Mutex`, (b) an `atomic.AddInt64`, (c) a single owner goroutine fed by a buffered channel, (d) the same owner fed by an *unbuffered* channel. The metric is wall-clock time and effective increments per second. **Measure honestly:** warm up the runtime first, run each variant many times, report the median, pin the machine (these are order-of-magnitude figures on a typical modern x86 server-class core, not a guarantee for your hardware — your numbers will differ, and on ARM's weaker memory model the atomic path can shift; re-run on *your* box before quoting any of this).

| Approach | Mechanism per increment | Relative throughput (higher is better) | Why |
| --- | --- | --- | --- |
| Atomic add | One `LOCK XADD`-class instruction | ~10x | No park/wake, no critical section, pure CAS-class op |
| Mutex | Acquire + release (~25 ns uncontended) | ~4x | Fast uncontended; contention degrades it |
| Buffered channel to owner | Send + owner receive + apply | ~1x (baseline) | Channel send/recv overhead per op dominates |
| Unbuffered channel to owner | Rendezvous per op (often a context switch) | ~0.2x | Every send waits for the owner; frequent parking |

Read the shape, not the exact multipliers. For *this* workload — a tiny, hot, high-frequency update — the atomic is roughly an order of magnitude faster than the channel, and the unbuffered channel is several times *slower* than the buffered one because every single increment forces a rendezvous (often a context switch at ~1–5 µs). The channel is doing far more work per increment: a send is a queue operation possibly involving a lock or CAS on the channel's internals plus, when the buffer is empty/full, a goroutine park-and-wake and a scheduler hop. The atomic is a single instruction. **For incrementing a counter, the channel is the wrong tool, and the measurement proves it.**

#### Worked example: where the channel *wins* the throughput argument

Now flip the workload to where message passing belongs. Instead of "increment an integer," make each unit of work substantial — say, 50 µs of CPU (parse a record, run a small model, compress a block). Eight workers pulling jobs from a buffered channel versus eight threads contending on one shared work-list behind a mutex.

Now the math inverts. The per-job *work* is 50 µs; the channel handoff is ~0.1–0.5 µs — under 1% overhead. The mutex-on-the-shared-list version, meanwhile, *serializes* every worker's access to the list: all eight threads fight for one lock on every `pop`, and as you add cores the lock becomes the bottleneck (this is the contention curve that bends *down* — adding threads makes it slower past a point). The channel version has each worker owning its job once received, no shared list, no central lock; throughput scales with cores until you saturate them. Here the channel is not just acceptable, it is *faster and scales better*, because the handoff cost is negligible relative to the work and it avoids the central contention point entirely.

The lesson the two measurements teach together is the one number that actually decides the paradigm: **the ratio of per-message overhead to per-message work.** When the work per message is tiny (a single increment), the channel's fixed overhead dominates and shared memory wins. When the work per message is large (a real job), the overhead is in the noise and message passing's lack of central contention wins. Don't argue the paradigm in the abstract; estimate that ratio for your actual workload and let it decide.

One more honest caveat on *how* to measure this, because microbenchmarks lie constantly. Warm up (the runtime, the allocator, the branch predictor, the caches all need a few iterations to stabilize). Run enough iterations that scheduler jitter averages out, and report the median or a percentile, not the min or the mean of a noisy distribution. Beware the compiler optimizing away work whose result you discard — consume the result. Pin to a known machine and *name it*, because the answer depends on core count, the memory model (x86 TSO vs ARM weak ordering changes the cost of the atomic and the fences a channel needs), the scheduler, and the runtime version. And never quote a single run of a concurrent benchmark as if it were deterministic — it is not. The whole [finding-concurrency-bugs post](/blog/software-development/concurrency/finding-concurrency-bugs-race-detectors-and-stress-testing) is about how nondeterministic these systems are; the same humility applies to measuring them.

## A channel pipeline: the model in the large

To see message passing structure a whole program rather than guard a single variable, look at the pipeline — the pattern where this paradigm is most obviously the right one.

![a directed graph of a channel pipeline where a producer feeds jobs to two workers over a channel and they fan into one consumer](/imgs/blogs/message-passing-vs-shared-memory-and-the-csp-philosophy-3.png)

The graph shows a producer generating jobs into a buffered channel, two workers consuming from it (fan-out — each job goes to whichever worker is free), and the workers sending results into a second channel that one consumer drains (fan-in). There is no shared mutable state anywhere in this picture. Each worker *owns* the job it pulled until it sends the result onward; the producer owns each job until it sends it; the consumer owns each result it receives. Synchronization is entirely the channel sends and receives. You could not draw a lock on this diagram if you tried — there is nothing to lock.

Here it is in Go, which is the language this shape reads most cleanly in:

```go
func pipeline(jobs []Job) []Result {
	jobCh := make(chan Job, 100)
	resCh := make(chan Result, 100)

	// producer: owns each job until it sends it
	go func() {
		for _, j := range jobs {
			jobCh <- j
		}
		close(jobCh) // signal: no more jobs
	}()

	// fan-out: N workers, each owns the job it receives
	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobCh { // ranges until jobCh is closed and drained
				resCh <- process(j) // owns j, produces a result, sends it on
			}
		}()
	}

	// closer: when all workers finish, close the result channel
	go func() {
		wg.Wait()
		close(resCh)
	}()

	// fan-in: consumer drains all results
	var out []Result
	for r := range resCh {
		out = append(out, r)
	}
	return out
}
```

Trace the ownership. A `Job` is created by the producer, owned by it until `jobCh <- j` hands it off, then owned by exactly one worker (whichever receives it) until that worker computes and sends a `Result`. No two workers ever see the same job — the channel receive hands it to exactly one. No lock anywhere. The `close(jobCh)` is the message-passing way to say "no more data," which lets each worker's `for range` terminate cleanly; the `wg.Wait()` then closes `resCh` so the consumer's range terminates too. The only subtle bit is the deadlock-avoidance discipline — close `jobCh` after the last send, close `resCh` only after all workers finish — which is the channel analogue of careful lock ordering. Get the close sequencing wrong and you get the channel deadlock the trade-offs section warned about: a `for range` over a channel that is never closed blocks forever.

This same pipeline in Rust would use `crossbeam` channels (the standard `mpsc` is single-consumer, so a work-stealing pool wants `crossbeam::channel` which is multi-producer multi-consumer) or `Rayon` for the data-parallel case; in Java it would be a `BlockingQueue` feeding an `ExecutorService` or, on modern JVMs, structured concurrency with virtual threads. The shape is universal — produce, fan out to workers, fan in results — and in every language the win is the same: no shared mutable state on the data path, ownership handed cleanly stage to stage, synchronization carried by the queue operations.

## Case studies / real-world

These ideas are not academic. Three real systems show the paradigm shaping production-grade software, and one cautionary tale shows where it does not save you.

**Erlang and the no-shared-state telecom switch.** Erlang was built at Ericsson in the late 1980s to run telephone switches that must not go down. Joe Armstrong and colleagues made a radical bet: *no shared memory at all.* Every Erlang process has its own private heap; processes communicate only by asynchronous message passing; one process literally cannot corrupt another's state because it cannot reach it. This isolation is what makes Erlang's "let it crash" philosophy and OTP supervision trees work — a failing process can be killed and restarted without leaving shared data in a torn state, because there is no shared data. The AXD301 ATM switch built on this reportedly achieved availability famously quoted as around nine nines (a figure often cited from Armstrong's thesis and talks; treat the exact number as an oft-repeated claim rather than an independently audited measurement). The mechanism behind the reliability is exactly this post's thesis: remove shared mutable state and you remove an entire category of failure. (Armstrong's 2003 thesis, *Making reliable distributed systems in the presence of software errors*, is the primary source.)

**Go's design philosophy and the `-race` detector.** Go was designed by Rob Pike, Ken Thompson, and Robert Griesemer with CSP-style channels as a first-class concurrency primitive precisely because the designers had lived through the pain of lock-based shared-memory concurrency in large C++ codebases. The slogan "share memory by communicating" is in the official Go blog and effective-Go documentation as a stated preference. Crucially, Go did *not* ban shared memory — it shipped `sync.Mutex` and `sync/atomic` and a built-in race detector (`go test -race`, ThreadSanitizer-based) that instruments memory accesses to catch data races at runtime. The combination is the pragmatic reading of the philosophy: prefer channels for structure, keep locks for hot shared state, and run the race detector in CI to catch the cases where the convention slipped. The Go standard library itself has had real data races caught by `-race` over the years — evidence that even expert authors need the tool, which is the strongest argument for a paradigm that removes the hazard by construction where you can.

**A real refactor: from a mutex-guarded map to an owner goroutine.** A pattern I have seen repeatedly (and done myself): a service has a shared `map[string]Session` behind a `sync.RWMutex`, accessed from every request handler. Under load, the lock becomes a contention hotspot, and worse, a subtle bug appears where a handler reads a session, makes a decision, then writes it back — a check-then-act race the `RWMutex` does not prevent because the read lock and the write lock are separate critical sections. The refactor: give the session map to a single owner goroutine and have handlers send it request-reply messages (`get session`, `update session`). The check-then-act becomes one message the owner processes atomically, the contention hotspot disappears (the owner is never contended; it just drains a channel), and the code that touches the map shrinks to one function. The cost: every session access is now a channel round-trip (~100s of ns) instead of a lock (~25 ns), which is fine because session access is not the hot inner loop — it happens once per request, where the request itself costs milliseconds. This is the ratio argument from the measurement section, applied: the per-message overhead is negligible relative to the per-request work, so message passing's correctness win comes nearly free.

**The cautionary tale: message passing did not save the logical race.** I have also seen the opposite — a team that moved everything to actors believing it made them "race-free," then shipped a double-spend bug. Two `withdraw` messages to an account actor with insufficient funds for both: the actor processed them one at a time (no data race — the balance was never torn) but an early version checked funds *asynchronously* against a cache that had not yet seen the first withdrawal, and approved both. The data race was gone; the *logical* race condition was not. The fix was to do the check-and-decrement synchronously inside the actor's single-threaded message handler — which is exactly what message passing gives you *if you put the invariant inside one owner's sequential loop*. The lesson: message passing deletes data races by construction, but logical correctness of the *order* of messages is still your job. (This mirrors the precise [data-race-versus-race-condition distinction](/blog/software-development/concurrency/data-races-vs-race-conditions-a-precise-distinction) earlier in the series — they are different bugs, and message passing kills one, not the other.)

## When to reach for this (and when not to)

The decision, distilled to rules you can apply without re-reading the post.

**Reach for message passing when:**

- The work **decomposes into independent units** that produce and consume — pipelines, worker pools, request handlers without shared session state. The dataflow is the program; channels make it explicit and lock-free.
- You want **isolation and fault containment** — a crashing owner cannot corrupt anyone else's state, the foundation of supervision and "let it crash."
- The **per-message work is large** relative to the channel overhead (the ratio is well above ~10–100x), so the handoff cost is in the noise and you gain by avoiding central lock contention.
- You may need to **distribute** later — message passing survives the move across a network boundary; shared memory does not.
- You want **ownership to be obvious** — channels force "who owns this datum" to be answerable by reading the code, which pays off in review and debugging.

**Reach for shared memory + a lock or atomic when:**

- Several pieces of state must **change together atomically** — a cross-cutting invariant. One lock spanning all of it beats a commit protocol across owners. Do not split state that must move as a unit.
- The data is **hot, small, and accessed at very high frequency** — a counter, a reference count, a small config map. An atomic (~5 ns) or a tiny mutex (~25 ns) beats a channel round-trip (~100 ns+). The measurement section showed the channel losing by an order of magnitude here.
- Many readers need a **large structure in place** — a big index or cache — where copying per access is absurd. A read-write lock or copy-on-write snapshot fits; a channel does not.

**Do not:**

- **Do not** put a channel in front of a single hot integer — use an atomic; the channel is pure overhead.
- **Do not** model a cross-entity invariant as messages to two separate owners and then bolt on a distributed transaction — co-locate the state under one owner or one lock.
- **Do not** believe message passing means no deadlock — draw the channel dependency graph the same way you would draw the lock graph, and watch unbuffered sends and `for range` over never-closed channels.
- **Do not** leave a channel unbounded and assume backpressure handles itself — choose buffer sizes deliberately; an unbounded mailbox plus a fast producer is an out-of-memory crash waiting to happen.
- **Do not** choose one paradigm for the *whole* program — real systems use channels for the architecture and locks for the hot, tightly-coupled state. Match the paradigm to each piece of state.

The meta-rule one more time, because it is the takeaway that survives the specifics: **default to communicating for the structure, share for the hot tightly-coupled state, and let the ratio of per-message work to per-message overhead break the tie.** The full decision tree across all the series' models lives in the [capstone playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).

## Key takeaways

- **Two paradigms, opposite strategies.** Shared memory shares the data and synchronizes the access (a lock around a cell); message passing isolates the data and synchronizes the communication (a send over a channel). Same goal — order over concurrent operations — opposite means.
- **Message passing kills data races by construction.** A data race needs the same location touched by multiple threads. Give the data a single owner and no other thread can name it, so the race is not prevented but impossible — like an unplugged toaster.
- **It does not kill logical race conditions.** Per-owner invariants become free sequential atomicity (the owner handles one message at a time), but cross-owner invariants and message-ordering bugs are still your job. Put the invariant inside one owner's loop.
- **CSP is the theory; the channel is the unit.** Synchronous channels are a rendezvous (send blocks until receive); buffered channels decouple the parties up to N in-flight messages. Either way the channel carries the happens-before edge, so communication *is* synchronization.
- **Ownership transfer beats copying for large data.** A move hands off a pointer in O(1) and forbids the sender from touching it; a copy duplicates in O(size). Rust *proves* the move at compile time (`Send`, move semantics); Go relies on the "send it and forget it" *convention*.
- **The costs are real: channel overhead, no cross-owner invariant, channel deadlock.** A channel op (~100 ns+) is slower than an atomic (~5 ns) for tiny hot updates; an invariant across owners needs a protocol; channels deadlock on cycles just like locks.
- **The ratio decides.** Per-message work much larger than per-message overhead → message passing wins (no central contention). Tiny work per message → shared memory wins (overhead dominates). Estimate the ratio for your workload; don't argue it in the abstract.
- **Use both.** Default to communicating for the architecture; use a lock or atomic for the hot, small, tightly-coupled state. Choosing one paradigm for the whole program is the mistake.

## Further reading

- **C. A. R. Hoare, "Communicating Sequential Processes," *Communications of the ACM*, 1978** — the founding paper. Short, formal, and still the clearest statement of "processes that share nothing and communicate by rendezvous." The source of the whole channel model.
- **Joe Armstrong, *Making reliable distributed systems in the presence of software errors* (PhD thesis, 2003)** — the case for no-shared-state message passing as the foundation of fault tolerance, with Erlang/OTP as the proof. The "let it crash" and supervision argument lives here.
- **The Go Blog, "Share Memory By Communicating," and *Effective Go* on concurrency** — the pragmatic mainstream reading: prefer channels, keep locks for hot shared state, run `-race`. The slogan in its original context.
- **Carl Hewitt et al., "A Universal Modular Actor Formalism for Artificial Intelligence" (1973)** — the actor model, message passing's other great tradition, which the [next post](/blog/software-development/concurrency/the-actor-model-mailboxes-isolation-and-supervision) builds on.
- **Maurice Herlihy and Nir Shavit, *The Art of Multiprocessor Programming*** — for the shared-memory side done rigorously: locks, atomics, and why the memory model makes shared state hard, which is exactly what message passing sidesteps.
- **The Rust Book, "Fearless Concurrency" chapter** — how `Send`/`Sync` and move semantics turn ownership transfer into a compile-time guarantee; the cleanest illustration of "the type system enforces one owner at a time."
- **Within this series:** the [intro](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) for the hazard frame; the [race-condition anatomy](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition) for what message passing deletes; the [actor model](/blog/software-development/concurrency/the-actor-model-mailboxes-isolation-and-supervision) and [CSP channels in practice](/blog/software-development/concurrency/csp-channels-goroutines-and-the-select-statement) for the next steps; and the [capstone playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) for choosing among all the models. For cross-process flow control, the [message-queue series on async decoupling](/blog/software-development/message-queue/message-queues-async-decoupling-and-load-leveling).
