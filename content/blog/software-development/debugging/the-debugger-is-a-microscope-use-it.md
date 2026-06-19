---
title: "The Debugger Is a Microscope: Use It (You're Using 5% of It)"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Learn the 95 percent of your debugger you have never touched — conditional breakpoints that fire on iteration 3.8 million, watchpoints that name the line that corrupts a struct, reverse debugging from a segfault back to the bad pointer, and the same moves in gdb, pdb, delve, and DevTools."
tags:
  [
    "debugging",
    "software-engineering",
    "gdb",
    "lldb",
    "pdb",
    "delve",
    "breakpoints",
    "watchpoints",
    "reverse-debugging",
    "rr",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/the-debugger-is-a-microscope-use-it-1.png"
---

Here is a confession that should not be a confession: most professional engineers, including ones with a decade of shipped code behind them, use their debugger the way a tourist uses a telescope at a scenic overlook. They set a breakpoint on a line. They look at one variable. They press the green "continue" button until the program either crashes or finishes. That is roughly five percent of what the tool can do, and it is the five percent that print statements already do, only slower. The other ninety-five percent — the part that turns a "segfault, no idea where" into "the corruption happens on iteration 3,847,221 in a stray `memcpy` two frames up the stack" — sits unused, because nobody ever showed them it was there.

A debugger is not a fancier `print`. It is a microscope for running code. A microscope does not just magnify; it lets you freeze a living thing mid-motion and look at any part of it you want, at any zoom level, changing what you examine on the fly without killing the specimen and starting over. That is exactly what an interactive debugger does to a running process: it freezes the program at the precise instant of interest, with the entire call stack intact, every local variable live and inspectable, and the ability to ask new questions — "what is `node->next` right now? what does `compute_hash(key)` return if I call it here? who, out of all the code in this binary, is about to write to this one byte of memory?" — without recompiling, without re-running, without losing the state you spent ten minutes reaching. The figure below sketches the gap: a `print` samples one value and then the run is gone; the debugger holds the whole frozen world open in front of you.

![Diagram contrasting what a debugger exposes at the moment of failure with what a single print statement can sample, shown as a vertical stack of inspection powers](/imgs/blogs/the-debugger-is-a-microscope-use-it-1.png)

This post teaches the ninety-five percent. By the end of it you will know the full breakpoint taxonomy — not just line breakpoints but conditional breakpoints (the killer feature for a bug that fires once in millions), hit-count and ignore-count breakpoints, data breakpoints and hardware watchpoints (the only sane way to catch "who corrupted this variable"), exception catchpoints, and temporary breakpoints. You will know stepping discipline — when to step *into* the suspect and *over* the boring, why undisciplined stepping wastes half your session, and how to step *backward* in time with reverse debugging and `rr`. You will know how to read all the locals, walk every frame of the stack, evaluate arbitrary expressions, call functions inside the debugged process, and examine raw memory. You will see the same investigation run in gdb, lldb, pdb, delve, and Chrome DevTools, so that learning one debugger genuinely teaches you all of them. You will attach to a running process, open a core dump for a post-mortem, and script the debugger so it auto-prints and continues — a "printf by debugger" that needs no recompile. And, because every powerful tool has a wrong place to use it, you will get an honest account of the bugs where `print` and tracing still beat the debugger flat: timing-sensitive races, async chains, tight production hot loops, and the payments process you must never freeze.

This is post A6 of the field manual, and it slots into the spine the whole series turns on: **observe → reproduce → hypothesize → bisect → fix → prevent.** The debugger is the instrument you reach for in the *observe* and *hypothesize* stages — it is how you watch reality at the instant your belief about the program turns out to be wrong. If you have not read the [scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) intro, read it first; it lays out the loop this post serves. And while we will read stack frames here, the deep dive on *reading* a raw stack trace across languages — symbolization, inlining, demangling — lives in the sibling post `reading-a-stack-trace-across-languages`, and an even deeper Track B treatment of the interactive debugger lives in the planned post `mastering-an-interactive-debugger`. Here, the goal is simpler and broader: to make the microscope stop feeling like a magnifying glass.

## 1. Why a debugger beats print for a whole class of bugs

Let me start by being fair to `print`, because it deserves it. The humble print statement (or `console.log`, or `fmt.Println`, or `log.debug`) is the most-used debugging tool on earth for good reasons. It works everywhere, in every language, with no setup. It survives across process boundaries and async hops. It leaves a durable trail you can grep. It does not require you to attach anything to anything. For a great many bugs, one well-placed log line that prints the value you suspected is the fastest possible path to the answer, and reaching for a debugger instead would be showing off. We will come back to exactly when print wins, in detail, near the end. The point of this section is the *other* class of bug — the one where print quietly costs you an hour and a debugger costs you ninety seconds.

The defining feature of `print` is that it answers exactly one question per run, and the question is baked into the binary before you run it. You decide *in advance* which value to print and where. Then you build, run, read the output, and — almost always — discover you printed the wrong thing, or the right thing but not the context around it, or the value but not the three other values you now realize you need to compare it against. So you edit, rebuild, rerun. Each new question costs a full compile-and-execute cycle. On a C++ project that is sixty seconds of build per question; on a large app it can be minutes. Ten questions is ten rebuilds, and you will ask more than ten questions about any non-trivial bug. The figure below makes the contrast concrete: the print loop is dominated by rebuild time, while the debugger asks all ten questions against a single frozen run.

![Diagram comparing the print debugging loop dominated by repeated rebuilds against the debugger loop that asks many questions against one frozen run](/imgs/blogs/the-debugger-is-a-microscope-use-it-2.png)

A debugger inverts this completely. You run the program *once* under the debugger, let it stop at the point of interest, and then the program sits there, frozen but alive, while you interrogate it interactively. Want a different variable? Type its name. Want to see the caller's locals? Walk up a frame. Want to know what a function would return from here? Call it — the debugger will execute it inside the stopped process and hand you the result. Each new question costs a keystroke and a few milliseconds, not a rebuild. The questions are *interactive*: you ask one, the answer reshapes your hypothesis, and you immediately ask the follow-up that the answer suggested — the tight observe-and-reason loop that is the heart of debugging, running at conversational speed instead of compiler speed.

Before the powers, it helps to see the three main observation tools side by side, because the right reach depends on the *shape* of the bug, not on taste. A print statement, an interactive debugger, and a tracer (sampling profiler, `strace`/`bpftrace`, distributed tracing) each occupy a different point on the trade-off curve between "rich state at one stop" and "low overhead across the whole run."

| Property | `print` / log line | Interactive debugger | Tracer (`strace` / profiler / `rr` record) |
| --- | --- | --- | --- |
| Questions per run | 1 (baked in before build) | Unlimited, interactive | Many, but pre-declared probes |
| Sees full stack + all locals | No (only what you printed) | Yes, every frame, on demand | Partial (stack samples) |
| Requires recompile to ask anew | Yes (rebuild per question) | No | No |
| Overhead on the program | Near zero | High at a stop (freezes process) | Low–moderate, continuous |
| Perturbs timing / races | Mild | Severe (stops the world) | Mild (sampling) to moderate |
| Works in live production | Yes | Dangerous (freezes traffic) | Yes (that is its job) |
| Best for | A value you already suspect | Rich state at a chosen stop | Timing, async, prod, hot loops |

The pattern in the table is the thesis of this whole post: the debugger dominates the "rich state at one stopping point" corner and *loses* the "low overhead across the whole run in production" corner. Sections 11 names exactly where each wins. For now, hold onto the middle column — the debugger's unmatched powers — because they are what most engineers never learn.

There are three powers here that `print` structurally cannot match, and naming them precisely is worth doing.

**You inspect state without modifying the program.** Adding a `print` changes the source, which changes the binary, which can change the bug — especially for optimization-sensitive or timing-sensitive bugs (the classic "heisenbug" that vanishes when you observe it, named after the uncertainty principle). A breakpoint, by contrast, is set on the *already-built* binary; nothing about the code changes. You can debug a release build, a binary you did not compile, a running production process you cannot rebuild, all without touching a line of source.

**You see the whole state at the moment of failure, not a pre-selected slice.** When a `print` fires, you see the values you chose to print. When a *breakpoint* fires, you have the entire frozen world: every local in the current function, every argument, every frame of the call stack with *its* locals, the heap, global state, registers. You did not have to predict in advance which of those forty values would matter. The bug is, by definition, somewhere you did not expect; the debugger's advantage is that it does not require you to have guessed right about *where* to look before you looked.

**You change your questions interactively.** This is the one people underrate most. Debugging is a conversation with the program: you ask, it answers, the answer changes what you ask next. With print, every turn of that conversation is a compile. With a debugger, the conversation runs at the speed of thought. That is why a debugging session that would be ten rebuilds and ten minutes of print becomes ninety seconds of typing once you are fluent.

#### Worked example: the null field nobody could explain with print

A team had a checkout service that threw a `NullPointerException` from a serializer: `row.name` was null, and it should have been impossible for it to be null, because the loader set it on every path. They had been print-debugging it for two days. The trouble was that the field was null only on the order that crashed, and they could not predict which order would crash, so every print run produced thousands of lines of "name = Alice, name = Bob, name = Carol" and then a crash with no print near the failing row, because the failing row was order 5,184 in a batch and nobody scrolls to line 5,184. They added more prints. The log got bigger. Two days.

Under a debugger the investigation took four minutes. Set a breakpoint in the serializer *conditional on the field being null*: `break Serializer.java:88 if name == null`. Run. The program executes 5,183 rows at full speed and stops dead on the exact row where `name` is null, with the whole stack live. Walk up two frames: the loader. Inspect the loader's locals: the row came from a `LEFT JOIN` that returned a null name for users created before a migration. The print approach could never reach this efficiently because print cannot *filter* — it prints every row or no row. The conditional breakpoint fires only on the one that matters. That single capability — "stop only when this is true" — is the difference between two days and four minutes, and it is the subject of the next section.

## 2. Breakpoints, the full taxonomy

If you only ever use one kind of breakpoint, you are leaving the most powerful ones on the table. Breakpoints come in families, and each family answers a different shape of question. The figure below lays out the taxonomy; the families that matter most — conditional, data, and event — are precisely the ones most engineers never touch.

![Diagram showing the breakpoint taxonomy as a tree with location, condition, data, and event families branching from a single breakpoint root](/imgs/blogs/the-debugger-is-a-microscope-use-it-3.png)

**Line breakpoints** are the ones everyone knows: stop when execution reaches this line. `break file.c:42` in gdb, `b file.py:42` in pdb, clicking the gutter in an IDE. Useful, ordinary, and the right tool when you want to stop *every* time control reaches a place. They become a liability the moment the place is reached thousands of times before the interesting one, which is where the next family earns its keep.

**Conditional breakpoints** are the single most valuable feature in this entire post, and the one that converts the most "impossible" bugs into ten-minute ones. A conditional breakpoint stops only when an expression you supply evaluates true. In gdb: `break process_item if item->id == 4096 && item->next == NULL`. The debugger evaluates that condition every time control reaches the line, at near-native speed, and only *stops* when it holds. This is how you catch a bug that fires on iteration 3,847,221 out of four million: you do not step four million times, and you do not print four million lines. You tell the debugger the shape of the bad case and let it run full speed until the universe matches your description. You can condition on anything in scope — a counter reaching a threshold, a pointer being null, two values being unequal, a string matching. The condition is real code in the debugged language's expression grammar, with access to all live state. When people say a debugger "saved them," nine times out of ten they mean a conditional breakpoint.

**Hit-count and ignore-count breakpoints** are conditional breakpoints' simpler cousin: stop on the *N*th time this line is hit, or skip the first *N* hits. In gdb, `ignore 2 3846000` tells breakpoint 2 to skip its next 3,846,000 hits, then stop. This is the move when you already know the bad iteration *number* (perhaps from an assertion message or a prior run) but the state at that iteration is not easily expressed as a condition. It is also faster than a complex condition, because counting a hit is cheaper than evaluating an expression.

**Data breakpoints and hardware watchpoints** are the family that does something no line breakpoint can: they stop when a piece of *memory* changes, regardless of which code changed it. This is the only sane answer to the question every C and C++ engineer eventually asks in despair — "*who* is corrupting this variable?" You do not know who. That is the whole problem. With a line breakpoint you would have to set a breakpoint on every line that might possibly write to it, which is most of the program. A watchpoint instead says: `watch s->len` — and now the debugger stops the instant *any* code anywhere writes to that field, and shows you the exact line that did it. On modern CPUs this is implemented with **hardware debug registers**: the processor itself watches a handful of addresses and traps when one is written, so the program runs at full native speed until the corrupting write happens. (Software watchpoints exist as a fallback when you watch more addresses than the hardware supports, but they are thousands of times slower because the debugger single-steps and checks after every instruction — prefer the hardware kind, and watch a single small field.) We will use a watchpoint to catch a struct corruptor on the one write out of millions in a worked example below.

**Exception and catchpoints** stop on a *kind of event* rather than a location: stop when any exception is thrown, when a `std::bad_alloc` is constructed, when a signal is delivered, when a library is loaded, when a `fork` happens. In gdb, `catch throw` stops at the point a C++ exception is thrown — crucially, *before* the stack unwinds, so you can see where it originated rather than only where it was caught. In Python's pdb you can set `pdb` to break on uncaught exceptions; in DevTools you can toggle "pause on caught/uncaught exceptions." This family answers "stop wherever this *type* of thing happens," which is invaluable when you know the *what* but not the *where*.

**Function and symbol breakpoints** stop on entry to a named function wherever it is defined: `break malloc`, `break MyClass::do_thing`. Handy when you do not know or care about the file and line, or when the function is in a library you cannot easily open. You can break on a symbol you only know by name from a stack trace.

**Temporary breakpoints** (`tbreak` in gdb) fire exactly once and then delete themselves. Perfect for "stop the first time we get into this function, then get out of my way" — no leftover breakpoint to trip over later in the session.

Here is a real gdb session that uses several of these together on a C program that corrupts a length field:

```bash
$ gdb ./server
(gdb) break handle_request          # function breakpoint
Breakpoint 1 at 0x4011a0: file server.c, line 88.
(gdb) condition 1 req->id == 4096    # only stop on request 4096
(gdb) run
Breakpoint 1, handle_request (req=0x55…e80) at server.c:88
88          parse_headers(req);
(gdb) print req->body_len            # inspect a field
$1 = 4
(gdb) watch req->body_len            # hardware watchpoint on the field
Hardware watchpoint 2: req->body_len
(gdb) continue
Hardware watchpoint 2: req->body_len
Old value = 4
New value = 4196353                  # the corruption, caught in the act
copy_body (req=0x55…e80) at body.c:51
51          memcpy(dst, src, n);     # the exact corrupting line
(gdb) backtrace                      # who called this
#0  copy_body  at body.c:51
#1  handle_request at server.c:120
#2  main at server.c:201
(gdb) frame 1                        # hop to the caller
(gdb) print n                        # the bad length
$2 = 4196353
(gdb) finish                         # run to end of this frame, see return
```

Notice the rhythm: break narrowly, inspect, set a watchpoint, continue, catch the write, backtrace, hop frames, inspect. That rhythm — not the individual commands — is the skill. The commands are just vocabulary; we will see the same rhythm in four other languages later.

## 3. The killer feature in depth: conditional breakpoints and the bug at iteration 3.8 million

Conditional breakpoints deserve their own section because they are the technique that most often turns "I can't even find where it goes wrong" into a precise stop. Let me make the *mechanism* concrete, because understanding how the condition is evaluated tells you when it will be fast and when it will be slow.

When you write `break parse if depth > 64`, the debugger does not magically know when `depth > 64`. What it does is set an ordinary breakpoint at the address of `parse`, exactly as for an unconditional one. Every time control reaches that address, the CPU traps into the debugger. The debugger then evaluates `depth > 64` in the current context — it reads `depth` from the stack frame, compares — and if the result is false, it *silently resumes the program* without ever surfacing to you. Only when the condition is true does it actually stop and hand you control. So the cost of a conditional breakpoint is: one trap per hit, plus one expression evaluation per hit. For a line hit four million times, that is four million traps. Each trap is microseconds, so four million traps can be seconds to a couple of minutes of overhead — slow, but finite, and infinitely better than stepping four million times by hand or scrolling four million print lines.

This mechanism explains a crucial optimization. If the line is *extremely* hot — hit billions of times — even microseconds-per-trap is too much. In that case you want to push the condition *into the hardware* or *into the code*. Pushing into hardware: if your condition is "a specific memory location takes a specific value," a hardware watchpoint runs at full speed (no per-hit trap) and is the better tool. Pushing into code: sometimes the cheapest conditional breakpoint is a single line of source — `if (id == 4096) raise(SIGTRAP);` or in higher-level languages a one-line `if cond: breakpoint()` — compiled in, so the *program itself* decides when to stop and there is zero per-hit debugger overhead. That is a legitimate hybrid: you accept one rebuild to make the stop free. Know all three so you can pick by how hot the line is.

A few practical patterns worth memorizing:

- **Stop on a specific key, not a specific iteration.** `break cache_get if strcmp(key, "user:4096") == 0` finds the one lookup you care about across millions, by content rather than by count.
- **Stop on an invariant violation.** `break update_balance if balance < 0` stops the instant a value goes impossible, which is usually *before* the symptom surfaces downstream. You are catching the cause, not the effect.
- **Stop on a relationship between values.** `break merge if left->key > right->key` catches the moment an ordering invariant breaks during a merge, naming the exact inputs.
- **Combine with a temporary breakpoint.** `tbreak main if argc > 3` stops once at startup only under a specific argument shape.

#### Worked example: the conditional breakpoint that fired on iteration 3,847,221

A parser was corrupting one record out of a multi-million-record import, and the corruption only showed up at the very end as a checksum mismatch, with no indication of *which* record. The naive approaches were all dead ends: stepping was out (millions of iterations); printing every record's state produced a multi-gigabyte log nobody could search; and the corruption was a wrong byte in a length prefix, not a crash, so there was no stack trace to start from. The breakthrough was to express the *bad state* as a condition. The team knew the valid length range was 0 to 65,535, so they set `break write_record if rec.len > 65535`. They ran the import. The debugger chewed through 3.8 million records at near-native speed — about ninety seconds of trap overhead — and stopped, once, on record 3,847,221, with `rec.len == 4196353`. The stack showed the length had been written by a routine that read it from a 16-bit field but stored it into a 32-bit field without masking, so a stray high byte from the *next* record leaked in only when that next record's first byte was non-zero, which is why it was rare. Time from "set the breakpoint" to "root cause on screen": under three minutes. The condition did all the searching; the engineer did the thinking. That is the division of labor a debugger is for.

## 4. Watchpoints: catching who corrupted the variable

The hardest bug class in unmanaged languages is *memory corruption by an unknown writer*: a struct field, a global, or a heap object holds the wrong value, and you have no idea which of a thousand lines of code wrote it. You cannot grep for the writer, because corruption usually happens through a *stray* pointer — a `memcpy` that overran, an array index off by one, a dangling pointer reused — and the offending code never mentions your variable by name at all. It writes through an address that, by accident, lands on your variable's bytes. There is nothing to grep for.

This is the situation watchpoints were invented for. The figure below traces the move: set a hardware watchpoint on the field, run at full speed, and let the CPU stop you on the exact instruction that performs the bad write.

![Diagram showing a hardware watchpoint running at full speed until the single corrupting write fires, then a backtrace naming the culprit frame and the fix](/imgs/blogs/the-debugger-is-a-microscope-use-it-4.png)

The *mechanism* is worth understanding because it is genuinely clever and explains both the power and the limits. Modern x86 and ARM CPUs have a small set of **debug registers** (typically four on x86) that can each hold an address and a condition — "trap when this address is written," "...read," "...executed." When you set `watch s->len`, gdb computes the address of `s->len`, loads it into a debug register with a write-trap condition, and lets the program run. The program runs at *full native speed* — no single-stepping, no per-instruction checks — because the *hardware* is doing the watching. The instant any instruction writes to that address, the CPU raises a debug exception, the OS delivers it to the debugger, and the debugger stops you on the very next instruction with the offending one identified. That is why a hardware watchpoint can sit on a variable for millions of operations and cost essentially nothing until the one write you care about.

The limits follow from the mechanism. There are only a handful of debug registers, so you can only watch a few addresses (and each can cover only a few bytes — a register, not an array). If you ask gdb to watch something larger than the hardware can cover, it falls back to a *software watchpoint*, which single-steps the entire program and checks the value after every instruction — correct but thousands of times slower, often unusably so. The discipline that follows: watch the *smallest* thing that pins the bug — one field, one word — not a whole struct or buffer, so the hardware can do the work.

Here is the rhythm against a corrupted struct field in gdb:

```bash
$ gdb ./parser core            # could be live or post-mortem
(gdb) break main
(gdb) run
(gdb) print &session->len      # confirm the address we'll watch
$1 = (uint32_t *) 0x5555557560a8
(gdb) watch *(uint32_t *) 0x5555557560a8
Hardware watchpoint 2: *(uint32_t *) 0x5555557560a8
(gdb) continue
Hardware watchpoint 2: *(uint32_t *) 0x5555557560a8
Old value = 4
New value = 1094795585         # 0x41414141 — 'AAAA', a classic overrun
0x0000555555555280 in copy_into (s=…, src=…, n=20) at buf.c:33
33          memcpy(s->data, src, n);   # n=20 into a 16-byte buffer
(gdb) backtrace
#0  copy_into at buf.c:33
#1  load_session at session.c:88
#2  main at main.c:40
(gdb) frame 0
(gdb) print sizeof s->data     # the buffer is only 16 bytes
$2 = 16
(gdb) print n                  # but we copied 20
$3 = 20
```

The watchpoint named the corruptor — a `memcpy` of 20 bytes into a 16-byte buffer, four bytes overrunning into the adjacent `len` field — in a single `continue`. No grep, no guessing, no stepping. Note one subtlety that trips people up: when the watched variable goes *out of scope* (a stack local whose frame returns), gdb tells you the watchpoint is now invalid and deletes it. That is correct behavior — the address no longer means what it did — but it is why watchpoints on heap or global data are easier to reason about than watchpoints on stack locals. For corruption hunting, watching a heap object's field (which lives until freed) is the sweet spot.

It is worth saying that sanitizers are often a *better* first reach than a manual watchpoint for memory corruption: AddressSanitizer (`-fsanitize=address`) instruments every memory access and will flag the overrunning `memcpy` *at the moment it happens* with a full report, often faster than setting up a watchpoint, and it finds the bug even when the overrun lands on padding rather than a variable you thought to watch. The watchpoint shines when you cannot rebuild with a sanitizer (a release binary, a third-party library, a reproduction you can only get under the real allocator) or when you already have the process stopped and just want to know who touches one specific word. Reach for ASan first if you can rebuild; reach for a watchpoint when you cannot, or when you have a precise address in hand.

## 5. Stepping discipline: into the suspect, over the boring

Stepping is where most debugger time is *wasted*, so getting disciplined about it is one of the highest-leverage skills here. The verbs are simple; the judgment is not.

- **Step over** (`next` in gdb/pdb/delve, F10 in IDEs): execute the next source line, and if it calls a function, run that whole function to completion *without* descending into it. Use this for code you trust — library calls, helpers you are confident in, the boring scaffolding around the suspect.
- **Step into** (`step` in gdb/pdb/delve, F11): execute the next line, and if it calls a function, *descend* into that function and stop at its first line. Use this for the suspect — the function you actually want to watch execute.
- **Step out / finish** (`finish` in gdb, `return` in pdb, `stepout` in delve, Shift-F11): run the *rest* of the current function and stop when it returns to its caller, showing you the return value. Use this when you have stepped into something and realized it is not where the bug is — get out and back up a level.
- **Continue to here / until** (`until` in gdb runs until a line *past* the current one, useful to skip out of a loop; "run to cursor" in IDEs): run until a specific line, then stop — without setting a permanent breakpoint.

The discipline, stated as one rule: **step over the boring, step into the suspect.** Undisciplined stepping means pressing "step into" reflexively and descending into the guts of string formatting, hash computation, and standard-library plumbing you have no reason to suspect — fifty steps to traverse code that was never the problem, while the actual bug is three lines away in *your* function. The cost is not just time; it is *attention*. Every irrelevant frame you step through is state you have to hold in your head and then discard. The expert steps *over* everything they trust until they reach the line where their hypothesis says the wrong value first appears, and only *then* steps *into* the suspect to watch it happen.

A second discipline: do not step when a breakpoint would skip the boredom entirely. If you know the bug is somewhere inside a function called from a loop, do not step through the loop — set a breakpoint inside the function (conditional if needed) and `continue`. Stepping is for the *last few lines* of an investigation, once a breakpoint has dropped you near the scene. Using stepping to *travel* across thousands of lines is the most common time sink in debugging, and the fix is always "set a breakpoint closer and continue."

There is a deeper truth that reframes stepping entirely. Stepping is *forward-only* time travel — you can only go to the next instruction, never the previous one. But the bug's *cause* is almost always *before* the symptom. You notice the bad value at line 200; the value was set wrong at line 80. Stepping forward from line 200 cannot help — you have already passed the cause. The classic move is to set a breakpoint *before* the suspected cause and re-run, stepping forward from there. But that requires re-running, which requires reproducibility, and for a non-deterministic bug the re-run might not reproduce. The way out of this is to step *backward in time*.

### Reverse and replay debugging: stepping backward from the crash

Reverse debugging lets you run the program *backward*: `reverse-step`, `reverse-continue`, `reverse-next` in gdb, which step to the *previous* line, run backward until the *previous* breakpoint, and so on. This is the single most underused power in debugging, and it directly attacks the "cause is before the symptom" problem. You stop at the crash, set a watchpoint on the bad value, and `reverse-continue` — and the debugger runs time backward until the moment the value *became* bad, which is the moment it was written wrong. You have traveled from effect to cause with one command.

There are two ways to get reverse debugging. gdb has a built-in record mode (`record full`) that logs every instruction's effects so it can undo them, but it is slow and memory-hungry and only practical for short windows. The production-grade tool is **`rr`** (record-replay) from Mozilla: you record a run *once* (`rr record ./prog`), and `rr` captures enough of the execution — every non-deterministic input, every syscall result, every signal — that it can *replay the exact same run deterministically*, as many times as you want, forward *and backward*, under a gdb-compatible interface. The recording overhead is modest (often well under 2x), and the replay is fully deterministic, which means a non-reproducible bug, once captured in a recording, becomes infinitely reproducible. That alone is worth the price; reverse stepping is the bonus. (For more on turning a flaky bug deterministic, the sibling post on [reproducing a bug first](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging) covers `rr` as a reproduction tool; here we use its reverse-execution superpower.)

We will do a full `rr` reverse-debugging worked example in the next section. The mental shift to internalize now is this: stepping forward is for the last few lines near a breakpoint; for "how did this value get this way," the right tool is a watchpoint plus *reverse* execution, not more forward stepping and re-running.

## 6. Reverse-debugging a segfault back to the bad pointer

Let me put reverse debugging to work on the bug class it was made for: a use-after-free that crashes far from its cause. The figure below shows the arc — record once, replay to the crash, set a watchpoint on the pointer, run time backward, and land on the earlier line that stored the freed block.

![Diagram showing the rr reverse debugging sequence from recording a run to replaying to the crash and stepping backward to the line that stored the bad pointer](/imgs/blogs/the-debugger-is-a-microscope-use-it-6.png)

First, the *mechanism* of why a use-after-free crashes far from its cause, because it explains why forward debugging is so painful here. When you `free` a block, the allocator does not erase it; it returns it to a free list to hand out again later. Your dangling pointer still points at those bytes, and they still hold the old contents *for a while* — until some unrelated `malloc` elsewhere in the program is handed the same block and writes its own data into it. So the sequence is: you free, you keep using the dangling pointer (and it *works*, because the bytes are still there), then ten thousand lines and three subsystems later something allocates, reuses the block, and now your pointer reads garbage and you crash. The crash site has *no relationship* to the free site. Forward debugging from the crash is hopeless because the cause is tens of thousands of instructions in the past. This is exactly the "cause before symptom" problem at its most extreme, and reverse debugging is the clean answer.

Here is the `rr` session, condensed but faithful:

```bash
# 1. Record the run once. The crash is captured deterministically.
$ rr record ./app --import data.bin
rr: Saving execution to trace directory …
Segmentation fault (core dumped)

# 2. Replay it under a gdb-like interface. Same run, every time.
$ rr replay
(rr) continue
Program received signal SIGSEGV, Segmentation fault.
0x000055…420 in node_value (n=0x602000000050) at node.c:77
77          return n->payload->value;   # n->payload is garbage

# 3. The crash is a deref of a garbage pointer. When did n->payload
#    get set to this value? Watch it and run BACKWARD.
(rr) watch -l n->payload
Hardware watchpoint 1: -location n->payload
(rr) reverse-continue
Hardware watchpoint 1: -location n->payload
Old value = (block *) 0x602000000090
New value = (block *) 0x0           # going back: it was valid, then 0
free_block (b=0x602000000090) at pool.c:54
54          b->in_use = 0;          # the block was freed HERE

# 4. So the block at 0x6020…090 was freed at pool.c:54 while node n
#    still pointed into it. Keep going back to see who freed it.
(rr) backtrace
#0  free_block at pool.c:54
#1  gc_sweep at pool.c:120          # the GC swept it…
#2  run_request at server.c:88      # …mid-request, while still in use
(rr) reverse-continue
# …lands on the line that stored the now-dangling pointer into the node
node_attach (n=…, p=0x602000000090) at node.c:40
40          n->payload = p;         # stored a pointer the GC later freed
```

Read what just happened. We started at a segfault with no context — `n->payload` was garbage and we had no idea why. We set a watchpoint on `n->payload` and ran *backward*. The debugger walked time in reverse and stopped at the instant `n->payload` last changed, which was when its block got freed by a garbage-collection sweep that ran *during* a request while the node still referenced the block. One more reverse step landed on the line that originally attached the pointer. We have the full causal chain — attach, then sweep-and-free while still attached, then later deref the dangling pointer — derived entirely by running backward from the symptom. With forward-only debugging this would have meant guessing where the free might be, setting a breakpoint there, re-running, and hoping the non-deterministic GC timing reproduced. With `rr` the recording *is* deterministic, so the bug reproduces every replay, and reverse execution hands you the cause directly.

#### Worked example: the heisenbug that only `rr` could pin

A service crashed roughly once every few thousand requests with a use-after-free, but only in production, only under concurrent load, and never under a debugger — attaching gdb changed the timing enough that the race between the request handler and the GC sweep never lined up. This is the worst kind of bug: the act of observing it makes it disappear. The fix was to record production-like load under `rr` until a crash was captured — it took about forty minutes of recorded replay traffic to catch one — and then the recording could be replayed *as many times as needed*, deterministically, on a laptop, with reverse execution. The crash that happened "once in a few thousand under load" became "every single replay, on demand," because the recording froze that exact run. Reverse-continuing from the crash on the captured trace took the investigation to the dangling-pointer store in about ten minutes. The measured outcome: a bug that had been open for three weeks with a "could not reproduce" label was root-caused in an afternoon once it was recorded. The lesson — record the rare crash *once*, then you own it forever — is the single biggest reason to keep `rr` in your toolkit.

## 7. Inspecting state and navigating the stack

Stopping at the right place is half the job. The other half is *reading the frozen world* fluently. Here is the vocabulary of inspection, with the gdb names (the others map cleanly, as we will see):

- **`print` / `p` expr** — evaluate any expression in the current context and show the result. Not just variables: `p arr[i]`, `p *node`, `p node->next->key`, `p strlen(name)`, `p i * width + j`. The argument is real code in the language's expression grammar, evaluated against live state.
- **`info locals`** — dump *every* local variable in the current frame at once. This is the "I didn't pre-select what to print" superpower: you see all forty locals, not the three you guessed.
- **`info args`** — the current function's arguments, as they were passed.
- **`backtrace` / `bt` / `where`** — the full call stack, frame by frame, from the current point back to `main`. This is the single most important inspection command: it tells you *how you got here*, which is usually most of the answer.
- **`up` / `down` / `frame N`** — move the "current frame" up toward the caller or down toward the callee, or jump to a numbered frame. Crucially, moving frames changes what `print` and `info locals` see: at frame #3 you read frame #3's locals. This is how you inspect a *caller's* state without re-running.
- **`x/` (examine memory)** — read raw memory at an address with a format: `x/16xb ptr` shows 16 bytes in hex, `x/8i $pc` disassembles 8 instructions at the program counter, `x/s ptr` reads a C string. Indispensable when the high-level view lies (a struct that prints fine but whose raw bytes reveal the corruption).
- **Calling functions** — you can *invoke* a function in the stopped process: `p compute_checksum(buf, len)` actually runs `compute_checksum` inside the debugged program and gives you the result. This is astonishing the first time you see it — you can call a pretty-printer, a validator, a hash function, anything, right at the breakpoint, to test a hypothesis without writing it into the code. (Caveat: the called function runs in the real process and can have side effects; call pure functions freely, mutating ones with care.)

Now, *navigating the stack*, which is where the real diagnostic leverage lives. The crash is at frame #0, but the bug — the place where a value first became wrong — is almost always *higher up*. The figure below shows the move: the segfault is in `strlen` at frame #0 because `name` is null, but `name` became null three frames up in the loader.

![Diagram showing a call stack from a strlen crash at frame zero up through callers to the loader frame where the null value actually originated](/imgs/blogs/the-debugger-is-a-microscope-use-it-7.png)

Read it as a story. Frame #0 is `strlen` dereferencing a null `name` — that is the *symptom*, and a junior engineer "fixes" it by null-checking inside `strlen`'s caller, which only moves the crash. Walk `up`: frame #1 is `format_row`, which was *passed* `name == NULL`. Up again: frame #2 is `render`, where `row.name` was never set. Up again: frame #3 is `load_user`, where the database returned no name and the code silently left the field null. *That* is the bug — the missing null handling at the source, three frames above the crash. The fix belongs at frame #3 (handle the empty result), not frame #0 (paper over the symptom). You found it by walking *up* the stack and reading each caller's locals with `info locals` after each `up`. This is the daily bread of debugger work: the backtrace shows you the path, and you walk *up* it reading state until you find the frame where the value was still correct on the way in but wrong on the way out — that frame contains the bug.

A practical note on inspecting containers and rich objects: a raw `print` of a `std::vector` or a Python dict can be unreadable (internal pointers, capacity, raw bytes). Debuggers support **pretty-printers** — gdb ships Python-based pretty-printers for the C++ standard library so `p myvec` shows `{1, 2, 3}` instead of internal guts, and you can register your own for your types. In pdb, objects print via their `__repr__`. Turn pretty-printing on (it usually is by default for libstdc++) so that inspecting a container is readable; if you ever see raw internals, you are missing a pretty-printer, not looking at corruption.

## 8. The same investigation across the toolchain

The biggest payoff of learning a debugger well is that the *concepts* transfer completely. Break, condition, watch, backtrace, step — these five verbs exist in every serious debugger, with different spellings. Learn the verbs once and you can debug C, Python, Go, and JavaScript with the same playbook. The figure below shows the one-for-one mapping.

![Diagram mapping the five core debugger verbs across gdb, pdb, delve, and DevTools as a matrix showing each tool's spelling of the same command](/imgs/blogs/the-debugger-is-a-microscope-use-it-5.png)

**gdb / lldb (C, C++, Rust).** We have used gdb throughout; lldb (the LLVM debugger, default on macOS) is near-identical in capability with slightly different syntax (`b`, `breakpoint set`, `frame variable`, `thread backtrace`). Both give you conditional breakpoints, hardware watchpoints, full stack navigation, memory examination, expression evaluation, and function calls. For Rust, gdb/lldb work directly on the compiled binary; `rust-gdb`/`rust-lldb` wrappers add Rust-aware pretty-printers.

**pdb / ipdb (Python).** Python's built-in debugger is `pdb`; `ipdb` is the same with IPython's nicer interface. The single most useful entry point is `breakpoint()` — drop that line anywhere in Python 3.7+ and the program drops into pdb when it hits it. Here is a pdb session that mirrors the gdb rhythm exactly:

```python
# In code, or attach with: python -m pdb script.py
import pdb

def process(rows):
    total = 0
    for r in rows:
        total += charge(r)     # suspect: total goes wrong here
    return total

# Run under pdb and drive it:
#   $ python -m pdb app.py
(Pdb) b app.py:8               # break in process
Breakpoint 1 at app.py:8
(Pdb) condition 1 r['id'] == 4096   # only the suspect row
(Pdb) c                        # continue until condition holds
> app.py(8)process()
-> total += charge(r)
(Pdb) p r                      # inspect the row
{'id': 4096, 'amount': None}   # amount is None — there's the bug
(Pdb) w                        # 'where' — the full backtrace
  app.py(20)<module>()
  app.py(8)process()
(Pdb) u                        # up to the caller's frame
(Pdb) p rows[:3]               # caller's locals are now visible
(Pdb) p charge(r)              # call a function right here
*** TypeError: unsupported operand: NoneType
(Pdb) d                        # back down a frame
(Pdb) c                        # continue
```

Same five verbs: `b` (break), `condition` (conditional), `p`/`w` (inspect/backtrace), `u`/`d` (navigate frames), `c` (continue). Python's pdb does not have hardware watchpoints (it is a source-level debugger on a managed runtime), but it has everything else, and `display expr` will auto-print an expression every time you stop. For finding *where* an exception came from, run `python -m pdb -c continue script.py` and pdb drops you into post-mortem mode at the point of an uncaught exception, with the full stack live — the Python equivalent of opening a core dump.

**delve (Go).** Go's debugger is `dlv`. It understands goroutines (Go's lightweight threads), which is essential because a Go stack trace involves many goroutines. The rhythm:

```go
// dlv debug ./cmd/server    (or: dlv attach <pid>)
(dlv) break main.go:88
Breakpoint 1 set at 0x… for main.handleRequest() main.go:88
(dlv) condition 1 req.ID == 4096
(dlv) continue
> main.handleRequest() main.go:88
(dlv) print req                // inspect a struct
(dlv) locals                   // all locals at once
(dlv) stack                    // backtrace of THIS goroutine
(dlv) goroutines               // every goroutine — the Go superpower
(dlv) goroutine 42             // switch to goroutine 42
(dlv) up                       // walk its frames
(dlv) print sharedMap          // read shared state from here
(dlv) next                     // step over
```

Note `goroutines` and `goroutine N`: in a concurrent Go program the bug often lives in a *different* goroutine than the one you stopped in, and delve lets you list all of them and switch context — the same "navigate to where the bug is" move as walking frames, but across concurrency units. For Go race conditions specifically, the race detector (`go test -race`, `go run -race`) instruments memory access and reports the conflicting goroutines and stacks, and is usually the faster first reach than delve for a data race; use delve to inspect state once you know where to look.

**Chrome DevTools / Node `--inspect` (JavaScript).** For browser JS, DevTools' Sources panel is a full debugger: gutter breakpoints, conditional breakpoints (right-click a breakpoint, "Edit breakpoint," enter a condition), the Scope pane (locals and closures), the Call Stack pane (frame navigation), step over/into/out buttons, and "pause on exceptions." For Node, run `node --inspect-brk app.js` and connect DevTools (or use the VS Code debugger) for the identical experience on the server. The async wrinkle — which we will return to in the "when print wins" section — is that an `await` unwinds the synchronous stack, so the call stack at a breakpoint inside an async callback may not show the logical caller; DevTools mitigates this with **async stack traces** (it stitches the logical chain back together), but it is imperfect, which is one reason async bugs sometimes favor logging.

The matrix figure above is worth internalizing as the single most useful artifact in this post: it says that the *skill* is portable. You are not learning gdb; you are learning *debugging*, and gdb, pdb, delve, and DevTools are four dialects of one language. The investigation — break narrowly, inspect widely, walk the stack to the source, step the last few lines — is identical in all of them.

## 9. Remote, post-mortem, and scripted debugging

The breakpoint-and-step model assumes you can launch the program under the debugger on your machine. Three situations break that assumption, and each has a clean answer.

**Remote / attach debugging — the process is already running, possibly elsewhere.** You do not always start the program under the debugger; often it is already running (a server, a stuck process, a container) and you need to look inside *now* without restarting it (restarting would lose the bad state). The move is to *attach*: `gdb -p <pid>` attaches gdb to a live process by PID, freezing it where it is, with full inspection — same `bt`, `info locals`, watchpoints, everything — and `detach` lets it resume. `dlv attach <pid>` does the same for Go; `node --inspect` on an already-running Node process (or sending `SIGUSR1`) opens the inspector; Python can be attached with tools like `pdb-attach` or, read-only, `py-spy dump --pid <pid>` to get a stack without even stopping the process. For a process on another host or in a container, you run a *debug server* on that host (`gdbserver :2345 ./prog` or `gdbserver :2345 --attach <pid>`, or `dlv --listen=:2345 --headless attach <pid>`) and connect your *local* debugger to it over the network (`gdb` then `target remote host:2345`). This is how you debug inside a Kubernetes pod: run `gdbserver`/`dlv` in the pod, port-forward the port, attach from your laptop. The discipline: attaching *freezes the target*, so attaching to a production process stops it serving requests — fine for a stuck or idle process, dangerous for a hot one (more on this in the next section).

**Post-mortem debugging — the process is already dead.** When a process crashes in production at 3am, you cannot attach to it — it is gone. But if it left a **core dump** (a snapshot of its full memory, registers, and stack at the moment of death), you can debug the corpse. `gdb ./prog core` opens the binary together with the core file and drops you *at the crash*, with the full stack and all memory exactly as they were when it died — `bt` shows where it crashed, `info locals` and `print` read the dead process's state, you can walk frames. You cannot *continue* (the process is not running) or set breakpoints (nothing to hit), but for "why did it crash" the post-mortem often *is* the whole answer. Enable cores with `ulimit -c unlimited` (and check `kernel.core_pattern` / use `coredumpctl` on systemd to find them); for a *running* process you can snapshot a core without killing it via `gcore <pid>`. Core dumps are the bridge between "crashed in prod, can't reproduce" and "here is the exact frozen state at death." The sibling post `reading-a-stack-trace-across-languages` goes deeper on symbolizing a core's stack when symbols are stripped.

**Scripted debugging — the debugger as a programmable instrument.** This is the feature that finally makes "printf debugging" and "debugger debugging" the same thing, with the debugger's advantages and none of print's recompile cost. gdb has a full **Python API** and a *breakpoint command* facility: you can attach a list of commands to a breakpoint that run automatically every time it fires, including "print these values and then continue." That is a *scripted printf via the debugger* — it logs exactly what you want, at exactly the lines you want, on the *already-built binary*, with no source edit and no rebuild. Put it in your `.gdbinit` and it loads every session. Here is the pattern:

```bash
# ~/.gdbinit  or  a sourced gdb script
break process_order
commands
  silent                     # don't announce the stop
  printf "order=%d total=%d\n", order->id, order->total
  continue                   # auto-resume — this is "printf by debugger"
end

# A conditional logging breakpoint: log ONLY the suspicious orders,
# without ever stopping interactively.
break charge if amount < 0
commands
  silent
  printf "NEGATIVE charge id=%d amount=%d at ", order->id, amount
  backtrace 1                # one frame of context
  continue
end
```

Run the program under gdb and it streams exactly those log lines, filtered by your condition, with zero rebuilds and zero source changes — and you can change *what* it logs by editing the gdb script and re-running, still no recompile of the program itself. The gdb Python API goes further: you can register a Python function as a breakpoint's `stop()` method, compute arbitrary conditions, walk data structures, and even drive whole automated analyses (gdb's `Breakpoint` and `FrameDecorator` classes). delve has a similar `--init` script facility and a JSON-RPC API; lldb has a full Python scripting API (`script` command and `~/.lldbinit`). The takeaway: a debugger is not only interactive — it is *programmable*, and a scripted breakpoint that prints-and-continues gives you filtered, contextual logging on any binary without a rebuild. When someone says "just use print, it's simpler," the scripted breakpoint is the answer that keeps print's simplicity and adds the debugger's power.

## 10. War story: real bugs the microscope was built for

Concrete history makes the ideas stick. Here are three real bug *classes*, told accurately, where the techniques above are exactly what cracks them. (Where I generalize the investigation rather than quote a specific company's postmortem, I say so; the bug classes and the techniques are real.)

**Heartbleed: a read overrun a watchpoint or sanitizer would have screamed about.** The 2014 Heartbleed vulnerability in OpenSSL was a buffer over-*read*: the TLS heartbeat code trusted a length field from the network and `memcpy`-ed that many bytes out of a buffer to echo back, without checking that the buffer actually held that many bytes. An attacker sent a small heartbeat claiming a 64KB length, and the server copied 64KB starting at the buffer — reading 64KB of *adjacent* heap memory (private keys, session data) and sending it back. The *mechanism* is exactly the overrun family from our watchpoint section, just a read instead of a write. The diagnostic relevance: AddressSanitizer flags an out-of-bounds read at the instant the `memcpy` runs, naming the line; a debugger watchpoint on the buffer's *redzone* (the bytes just past it) would stop on the over-read. The bug shipped because neither a sanitizer nor a bounds check guarded the length, and the lesson the whole industry took was to run fuzzing under ASan continuously — the microscope, automated.

**The use-after-free that only reproduces under load.** A composited but very typical production story: a service crashed intermittently with a segfault in a JSON serializer, but only under concurrent load and never under a debugger (attaching changed the timing). This is the heisenbug pattern from our `rr` section — the bug is a race between a request handler and a background cleanup that frees an object the handler still references, and any perturbation (the debugger's overhead, a log line, a different CPU count) shifts the timing so the windows no longer overlap. The resolution was exactly the `rr` recipe: record production-like load until a crash is captured (a recording is cheap to keep running), then replay the *deterministic* recording with reverse execution to walk from the segfault back to the free. The free turned out to happen in a cleanup goroutine that did not coordinate with in-flight requests. The numbers that made it real: open three weeks as "cannot reproduce," root-caused in an afternoon once recorded. The general lesson — *a heisenbug you can record, you can solve; the recording removes the timing sensitivity that defeats live debugging* — is one of the most important in this entire field manual.

**The Knight Capital deploy: a debugger could not have saved it, and that is the lesson.** In 2012 Knight Capital lost roughly 440 million dollars in 45 minutes because a deploy left old, repurposed code (a feature flag that reactivated a dormant code path) running on one of eight servers. There was no segfault, no exception, no crash to break on — the program ran *exactly as written*, just with logic that should have been dead. This is the honest boundary of the debugger as a tool: a debugger finds where a program does something *unintended by the code*; it cannot find where the *code itself* is wrong but running correctly. For that class — wrong behavior with no fault — you need different instruments: a `git bisect` to find the change that flipped behavior, structured logs and tracing to see the divergence, and deploy discipline so old code cannot be reactivated. We cover bisection and behavioral regressions elsewhere in the series. The point here is calibration: reach for the debugger when there is a *fault* or a *wrong value* to inspect; reach for bisection and tracing when the program is "correct" but the *behavior* regressed.

These three span the space: a memory-safety overrun (watchpoint / sanitizer), a timing-sensitive use-after-free (record-replay / reverse), and a behavioral regression with no fault (not a debugger problem at all). Knowing which is which *before* you start is half the skill — and is exactly what the final section is about.

## 11. How to reach for this (and when not to)

A microscope is the wrong tool for reading a billboard. Every technique in this post has a cost and a wrong place to apply it, and the senior move is knowing when *not* to attach a debugger. The figure below lays out the decision; the honest summary is that the debugger wins decisively for rich state at one stopping point, and *loses* to print and tracing for timing, async, and production cases.

![Diagram showing a decision matrix of when the debugger wins versus when print and tracing win across logic bugs tight loops races async chains and production processes](/imgs/blogs/the-debugger-is-a-microscope-use-it-8.png)

**When the debugger wins (reach for it):**

- *A logic bug with rich state.* You need to see many variables, the stack, and the relationships among them at one point. Stepping and inspecting beats forty print statements. This is the debugger's home turf.
- *A rare case in a huge loop.* The conditional breakpoint that fires on iteration 3.8 million is unbeatable. Print cannot filter; the debugger can.
- *Memory corruption by an unknown writer.* The watchpoint names the corruptor. Nothing else does as directly (except a sanitizer, if you can rebuild).
- *"How did this value get this way."* A watchpoint plus reverse execution (`rr`) walks from effect to cause. This is the killer use of record-replay.
- *A crash with no idea where.* Open the core dump post-mortem; the backtrace is usually most of the answer.

**When print and tracing win (do not reach for the debugger):**

- *Timing-sensitive and race bugs.* A breakpoint *stops the world*, which changes the very timing that causes the bug — the heisenbug vanishes. Lightweight logging (or better, a sanitizer like TSan / `-race`, or recording under `rr` and reverse-debugging the recording) perturbs timing far less. Do not try to catch a live race by setting a breakpoint in the racing code; you will only prove that breakpoints serialize threads.
- *Async / await chains.* At an `await`, the synchronous call stack unwinds, so a breakpoint inside an async continuation often shows you a stack that does not include the logical caller. Async stack traces help but are imperfect. A correlation ID threaded through structured logs across the async hops frequently beats a debugger here, because it preserves the *logical* causality the stack lost.
- *Tight production hot loops.* Even a conditional breakpoint traps on every hit; on a billion-hit line in prod that is unacceptable overhead. Use a scripted print, a sampling profiler, or `bpftrace`/eBPF, which observe with negligible cost.
- *Live production processes that must keep serving.* Attaching gdb *freezes the process*. Do not attach a debugger to the payments process, the order matcher, or any latency-critical request handler under load — you will cause an outage worse than the bug. Read-only tools (`py-spy dump`, `gcore` for a snapshot, a core dump after the fact, distributed tracing) get you the state without freezing the service. The rule: *never freeze a process that is on the critical path for live traffic.*
- *When one log line answers it.* If you already know which value you suspect and one `print` confirms or refutes it, print is faster than launching a debugger. Do not perform surgery to remove a splinter.

To make the choice mechanical, here is the same decision as a lookup table — symptom on the left, the tool that wins, its overhead, and why. Scan to your symptom and reach for that row.

| Symptom | Reach for | Overhead | Why this wins |
| --- | --- | --- | --- |
| Wrong value, rich state | Debugger breakpoint | High at stop, none otherwise | Inspect 40 locals + the stack at one point |
| Bad iteration 1 in millions | Conditional breakpoint | Trap per hit, finite | Filters to the one case; print cannot |
| Who corrupted this word | Hardware watchpoint or ASan | Near zero (hardware) | Stops on the exact writing instruction |
| How did this value get set | `rr` + reverse-continue | Modest record, free replay | Walks effect → cause, deterministically |
| Crashed in prod, no repro | Core dump post-mortem | Zero (process is dead) | Frozen state at death; backtrace is the answer |
| Data race / timing bug | TSan / `-race`, then logs | Moderate (TSan ~5–15x) | Breakpoint perturbs the race away |
| Async / await causality | Correlation-ID logs | Near zero | Stack unwinds at `await`; logs keep causality |
| Hot loop in live prod | `bpftrace` / sampling profiler | Negligible | Observes without trapping every hit |
| Latency-critical prod path | Read-only (`py-spy dump`, `gcore`) | Near zero | Never freeze a process serving traffic |

The table is the operational core of this section: the debugger owns the top four rows (rich state, rare iteration, corruption, causality) and a post-mortem; everything below the midline belongs to tracing, sanitizers, or read-only snapshots precisely because a breakpoint would freeze or perturb the thing under study.

A few cross-cutting cautions worth stating plainly:

- *Don't chase a heisenbug at -O2.* Optimized builds reorder and inline code so the debugger's line mapping is confusing and locals may be "optimized out." Reproduce at `-O0 -g` first if you can; debug the optimized build only when the bug *only* appears optimized (then expect `<optimized out>` and lean on registers and disassembly).
- *Match the tool to the runtime.* Hardware watchpoints exist in gdb/lldb/delve (native code) but not pdb (managed Python) — for Python "who set this attribute," use a property setter or a `__setattr__` trap, not a hardware watchpoint.
- *Prefer the sanitizer when you can rebuild.* For memory and race bugs, AddressSanitizer and ThreadSanitizer find the bug *automatically* at the moment it happens, often faster than a manual watchpoint, and they find bugs you did not think to watch for. Reach for the debugger when you cannot rebuild, or to inspect state once the sanitizer has pointed you at the line.

The meta-rule that ties it together: the debugger is for *inspecting rich state at a chosen stopping point*. When the bug is about *timing*, *causality across async or service boundaries*, or *behavior that is wrong despite correct execution*, the debugger is the wrong microscope — reach for tracing, sanitizers, record-replay, or bisection instead. Choosing right at the start is the difference between a ten-minute session and a wasted afternoon.

## 12. Key takeaways

- **A debugger is a microscope, not a fancier `print`.** It freezes the running process and lets you inspect the whole stack and every local at the moment of failure, change your questions interactively, and ask follow-ups at the speed of thought — no recompile per question. Most engineers use 5% of it; the other 95% is where the hard bugs get solved.
- **Conditional breakpoints are the killer feature.** `break foo if x > 1000 && p == NULL` lets the program run at full speed and stop only on the one bad iteration out of millions. This single capability turns most "I can't even find where it goes wrong" bugs into ten-minute ones.
- **Watchpoints answer "who corrupted this variable."** A hardware watchpoint (`watch s->len`) uses the CPU's debug registers to run at full speed until the exact instruction that writes the field, then names it. Watch the smallest thing that pins the bug so the hardware can do the work; prefer a sanitizer if you can rebuild.
- **Reverse debugging walks from effect to cause.** The bug's cause is almost always *before* the symptom, and forward stepping can never go back. Record once with `rr`, replay deterministically, set a watchpoint on the bad value, and `reverse-continue` to the line that set it wrong — the clean answer to use-after-free and "how did this value get this way."
- **Step over the boring, into the suspect.** Undisciplined stepping wastes half your session traversing code you do not suspect. Set a breakpoint *close* to the scene and `continue`; step only the last few lines. Walk *up* the stack from the crash to the frame where the value was still right on the way in but wrong on the way out — that frame holds the bug.
- **The skill is portable.** Break, condition, watch, backtrace, step are the same five verbs in gdb, lldb, pdb, delve, and DevTools — different spellings, one playbook. Learn one debugger well and you can debug C, Python, Go, and JavaScript with the same investigation.
- **Attach, post-mortem, and script.** Attach to a live or remote process with `gdb -p` / `dlv attach` / `gdbserver`; debug a corpse with `gdb prog core`; and turn the debugger into filtered, no-rebuild logging with breakpoint commands and a `.gdbinit`. A debugger is programmable, not only interactive.
- **Know when print and tracing win.** For timing-sensitive races, async chains, tight production hot loops, and live latency-critical processes, a breakpoint perturbs or freezes the very thing you are studying. Never attach a debugger to a process on the critical path for live traffic. Match the microscope to the specimen.

## Further reading

- [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the series intro and the observe → reproduce → hypothesize → bisect → fix → prevent loop this post serves.
- [Reproduce it first, or you're not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging) — the sibling on turning a ghost into a deterministic repro, including `rr` as a reproduction tool that pairs with the reverse-debugging here.
- `reading-a-stack-trace-across-languages` and `mastering-an-interactive-debugger` — sibling Track B posts (planned) on symbolizing and reading raw stack traces, and a deeper interactive-debugger treatment.
- The **GDB manual** (sourceware.org/gdb/documentation) — the authoritative reference for breakpoints, watchpoints, the Python API, and reverse debugging; and the **LLDB** tutorial for the macOS/LLVM equivalent commands.
- The **`rr` project** (rr-project.org) — record-replay and reverse execution; the "usage" and "reverse execution" docs are the fastest way to get productive.
- The **AddressSanitizer** and **ThreadSanitizer** wikis (the Clang/LLVM docs) — the automated alternative to a manual watchpoint for memory and race bugs; reach for these first when you can rebuild.
- *Debugging* by David J. Agans and *Why Programs Fail* by Andreas Zeller — the two canonical books on the discipline of debugging; Zeller's delta-debugging and scientific-method framing underpins this whole series.
- For production observability instead of attaching a debugger: [observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) and [debugging production at scale](/blog/software-development/system-design/debugging-production-at-scale) — how to get the state you need without freezing a live service.
