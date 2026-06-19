---
title: "Integer Overflow and Floating-Point Traps: When the Numbers Lie and Nothing Crashes"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Catch the arithmetic that silently produces the wrong answer with no crash, using UBSan to trap integer overflow at the operation, full-precision float printing, NaN detection, and the type choices that prevent it."
tags:
  [
    "debugging",
    "software-engineering",
    "integer-overflow",
    "floating-point",
    "ieee-754",
    "undefined-behavior",
    "ubsan",
    "numerical",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/integer-overflow-and-floating-point-traps-1.png"
---

There is a special kind of dread reserved for the bug that does not crash. A segfault is honest. It hands you a signal, a faulting address, a stack you can read. But the bug where a number is simply *wrong* — the counter that should read two billion and reads negative one, the financial total that is three cents short, the search that returns the wrong record at scale — that bug looks you in the eye and says nothing. The program runs to completion. Every test that does not check the exact value passes. And the wrong answer flows downstream into a report, a billing run, a flight control law, a rocket's guidance system, where it does its damage quietly.

I have spent more late nights than I would like chasing these. A reconciliation job that balanced perfectly in staging and was off by pennies in production. A binary search that worked on every unit test and returned garbage when the array got large enough to matter. A data pipeline whose averages came back as `NaN` for one customer and nobody could say why. None of these threw an exception. None left a core dump. They are the quietest, meanest bugs in software, and they all come from the same place: **the machine's numbers are not the numbers you think they are.** Integers have a fixed width and wrap around. Floats are binary approximations of decimal values and carry tiny errors that compound. The arithmetic you write in your head obeys mathematics. The arithmetic the CPU executes obeys two's complement and IEEE-754, and those are different.

![A taxonomy tree splitting silent numeric bugs into an integer family with signed overflow, unsigned wraparound, and narrowing casts, and a floating-point family with equality failure, catastrophic cancellation, and NaN poisoning](/imgs/blogs/integer-overflow-and-floating-point-traps-1.png)

This post is a field manual for the two families of silent numeric bugs in that tree, and how to drag them into the light. We will work the way the whole series works — **observe → reproduce → hypothesize → bisect → fix → prevent** — because a wrong number is exactly the symptom that tempts you to stare and guess. By the end you will be able to: reason about *why* a signed counter at its maximum becomes negative and why a `size - 1` on an empty container becomes a near-infinite loop; understand *why* `0.1 + 0.2` is not `0.3` and why `==` on floats is a landmine; trap integer overflow at the exact operation with UBSan instead of guessing; print floats at full precision to see what is really stored; detect a `NaN` before it poisons a sort; and choose the right type so these bugs cannot happen at all. We will write the wrong version and the right version in C, Python, Go, Rust, and Java, and we will measure the difference in real numbers. If you have only ever debugged with `print`, you can follow every step — and you will leave with diagnostics you can run today.

Throughout, lean on the spine of the series: stop guessing. A wrong number is a falsifiable claim. You can pin down where the value first goes wrong by binary-searching the computation, exactly the discipline laid out in [the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) and sharpened in [hypothesize and falsify, not stare and hope](/blog/software-development/debugging/hypothesize-and-falsify-not-stare-and-hope).

## 1. Two families, one symptom: the wrong answer

Before tools, get the map straight, because the two families fail for completely different reasons and the fix for one is useless for the other. Reaching for an `epsilon` comparison to fix an integer overflow, or widening an integer to fix float drift, is the kind of category error that burns an afternoon.

**Integer bugs** come from *fixed width and modular arithmetic*. A 32-bit signed integer holds exactly $2^{32}$ distinct values, from $-2{,}147{,}483{,}648$ to $2{,}147{,}483{,}647$. There is no value above the maximum; the hardware has no slot for it. When you add one to the maximum, the result is not "the maximum plus one" — there is nowhere to put that — it is whatever the bit pattern happens to mean, which in two's complement is the most negative value. The integer answer is exact (no fractions are lost), but it is the wrong integer, computed modulo $2^{32}$. The bug is that the *range* silently exceeded what the type can hold.

**Floating-point bugs** come from *binary approximation and finite precision*. A `double` (IEEE-754 binary64) holds about 15–17 significant decimal digits, encoded as a sign, an exponent, and a binary fraction. Most decimal numbers — including innocent ones like `0.1` — have no exact binary representation, the way `1/3` has no exact decimal representation. So the stored value is *close to* what you wrote, off by a tiny amount, and those tiny amounts compound, cancel, and occasionally explode into `NaN` or `Inf`. Here the *value itself* is approximate from the start. The bug is that you treated an approximation as exact.

Here is the table I keep in my head when triaging a "the number is wrong" report. Which column the symptom lands in tells me which family I am hunting, and that decides the very first diagnostic I reach for.

| Clue in the symptom | Likely family | First diagnostic |
| --- | --- | --- |
| Value flipped sign or jumped to a huge/negative number | Integer overflow / wraparound | Recompile with UBSan `-fsanitize=signed-integer-overflow,unsigned-integer-overflow` |
| A loop ran billions of times or an allocation was absurdly large | Unsigned wraparound (`size - 1` at `size == 0`) | UBSan integer; print the operand right before the subtraction |
| High-order digits vanished after a cast | Narrowing truncation (64→32) | Print with explicit width; UBSan implicit-conversion |
| Result wrong only by a tiny amount, grows with data size | Float accumulation / rounding | Print `%.17g`; compare against integer-cents or Kahan sum |
| `==` "should" be true but is false | Float equality | Print both operands at full precision; use `isclose` |
| A comparison or sort behaves nonsensically; result is `NaN` | NaN propagation | Test `x != x`; enable FP exception trapping |

Keep that mapping in view; the rest of the post is each row in depth. We will start with integers, because their failure mode is the more surprising of the two — the answer is *exactly* wrong, and on most platforms, by default, nothing tells you.

## 2. The integer mechanism: fixed width and two's complement

To debug integer overflow you have to internalize one fact: **a fixed-width integer is a clock, not a number line.** It counts up, and when it runs off the top it does not stop or grow — it wraps around to the bottom, exactly the way a 12-hour clock goes from 12 back to 1. The whole family of integer bugs is what happens when your value runs off an edge of that clock and you did not notice.

![A circular two's complement diagram showing the value running from zero up through the positive range to INT_MAX, then adding one wraps to INT_MIN and continues up through the negative range back to zero](/imgs/blogs/integer-overflow-and-floating-point-traps-2.png)

Two's complement is the encoding nearly every machine uses for signed integers, and it is why the wrap goes where it goes. In an $n$-bit two's complement integer, the top bit has *negative* weight: for 32 bits, the bit values are $2^{30}, 2^{29}, \ldots, 2^0$ for the lower 31 bits, but the 32nd bit is worth $-2^{31}$. So `0x7FFFFFFF` (all lower bits set, top bit clear) is $2^{31} - 1 = 2{,}147{,}483{,}647$ = `INT_MAX`. Add one and the bits become `0x80000000`: only the top bit is set, which is worth $-2^{31} = -2{,}147{,}483{,}648$ = `INT_MIN`. That is the wrap. It is not a glitch; it is what the encoding *defines* the next bit pattern to mean. The hardware adder does not know or care that you crossed a semantic boundary; it just adds bit patterns modulo $2^{32}$.

For **unsigned** integers, all bits have positive weight, so the clock runs from `0` to $2^{32} - 1 = 4{,}294{,}967{,}295$ and wraps back to `0`. The dangerous direction is *down*: subtract one from `0` and you do not get `-1` (there are no negatives in unsigned), you get $2^{32} - 1$, the maximum. This is the mechanism behind the single most common real-world integer bug I see: code that computes `size - 1` or `count - 1` where `size` is an unsigned type and happens to be `0`. Instead of a small negative number, you get roughly four billion (or eighteen quintillion for 64-bit `size_t`), and the next line either tries to allocate that many bytes or loops that many times.

So the mechanism gives us three named integer traps, each a different way to run off the clock:

- **Signed overflow / wraparound** — a value crosses `INT_MAX` going up (or `INT_MIN` going down) and flips sign.
- **Unsigned wraparound** — a subtraction crosses below `0` and becomes a gigantic positive number.
- **Narrowing truncation** — a value that fits in a wide type is copied into a narrow one, and the high bits are simply discarded.

Two facts make the signed case far nastier than the unsigned case, and they are the reason you cannot just "add a check" and move on. We will take them one at a time.

### 2.1 The first nasty fact: signed overflow is undefined behavior in C and C++

In C and C++, **signed integer overflow is undefined behavior (UB).** Not implementation-defined, not "wraps in two's complement" — undefined. The standard says a program that overflows a signed integer has no defined meaning at all. Unsigned overflow, by contrast, is *defined* to wrap modulo $2^n$.

This sounds like pedantry until you see what optimizers do with it. The compiler is allowed to *assume signed overflow never happens*, and it uses that assumption to delete code — including your overflow check. The classic example:

```c
#include <stdio.h>
#include <limits.h>

int will_overflow(int a, int b) {
    int sum = a + b;
    // A naive "check": did the sum come out smaller than a? Then it must have wrapped.
    if (sum < a) {
        return 1;  // we think we caught the overflow
    }
    return 0;
}

int main(void) {
    printf("%d\n", will_overflow(INT_MAX, 1));
    return 0;
}
```

The intent is clear: if adding a positive `b` to `a` produces something smaller than `a`, the addition must have wrapped. But the compiler reasons differently. It knows `a + b` is a signed addition, and *signed overflow is undefined, so it cannot happen*. If overflow cannot happen, then `sum = a + b` can never be less than `a` when `b >= 0`, so `sum < a` is provably false, so the entire `if` branch is dead code — and it gets deleted. Compile that at `-O2` and the function may return `0` for `INT_MAX + 1`. Your check evaporated because you wrote the check *in terms of the very overflow the compiler assumes away*. This is not theoretical; it has caused real security vulnerabilities where a bounds check was optimized out.

The lesson is sharp: **you cannot detect signed overflow in C by letting it happen and inspecting the result.** You must detect it *before* it happens (check the operands), or use a tool that instruments the operation, or use a language where overflow has defined behavior. We will do all three.

### 2.2 The second nasty fact: the answer is plausible

A wrapped integer is not a wild value like `0xDEADBEEF`. It is a perfectly ordinary-looking number that happens to be wrong. `INT_MAX + 1` is `-2147483648` — a clean, round-looking, totally plausible negative number. A view counter that wraps reads as a believable count. This is why these bugs survive code review and slip past tests: the wrong value does not look wrong. It looks like *a* value. Your eyes will not catch it; only a tool that knows the arithmetic boundary will.

## 3. The integer method: trapping overflow at the operation with UBSan

The most important shift in mindset for integer bugs is this: **do not reason about where the wrong number ended up; trap the moment it was created.** A wrapped value, like a corrupted pointer, travels far from its birthplace before it causes visible harm. The view counter wraps in one service and the wrong number shows up in a dashboard three systems away. Chasing the symptom is hopeless. Instead, instrument the arithmetic so the program halts (or logs) at the exact `+`, `-`, or `*` that overflowed.

The tool for C and C++ is the **Undefined Behavior Sanitizer (UBSan)**, part of Clang and GCC. You compile with a flag and the compiler inserts a check after every arithmetic operation in the covered class; if the operation overflows, it prints the file, line, the operands, and the type, then (optionally) aborts.

```bash
# Compile with integer overflow trapping. -g keeps line numbers; -fno-omit-frame-pointer
# gives clean backtraces. Cover both signed and unsigned, and narrowing conversions.
clang -g -fno-omit-frame-pointer \
  -fsanitize=signed-integer-overflow,unsigned-integer-overflow,implicit-conversion \
  overflow_demo.c -o overflow_demo

# Make the sanitizer halt on first hit and print a stack trace, instead of continuing.
export UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1

./overflow_demo
```

Run our wrapping example under this and instead of a silent `-2147483648` you get a runtime-error report. UBSan prints the source location `overflow_demo.c:7:17`, the message `runtime error: signed integer overflow: 2147483647 + 1 cannot be represented in type 'int'`, and a backtrace pointing at frame `#0` in `compute` at line 7 and frame `#1` in `main` at line 14.

That single line is the whole bug: file, line, the literal operands `2147483647 + 1`, the type `int`. You are no longer staring at a wrong number in a dashboard wondering where it came from; you are standing on the exact line that produced it. This is the difference between "the report is wrong somewhere" and "line 7 overflowed an `int`."

A note on `signed-integer-overflow` vs `unsigned-integer-overflow`: because unsigned overflow is *defined* to wrap, it is technically not undefined behavior, so UBSan only checks it when you explicitly ask. You almost always want to, because a *defined* wrap is still a *bug* if you did not intend it — the `size - 1` disaster is defined behavior and still catastrophic. Add `unsigned-integer-overflow` to the list and UBSan will flag it too. The `implicit-conversion` check catches the narrowing-cast family from the next section.

For a hard stop at the operation that aborts the process the instant any signed overflow occurs (useful in a test binary), there is also `-ftrapv`, which makes signed overflow trap with `SIGABRT`. It is blunter and slower than UBSan and covers only signed overflow, but it is a one-flag way to turn "silent wrong answer" into "loud crash you can debug."

The strategy generalizes beyond C. The whole point is to **make the overflow loud at the operation** rather than letting it travel:

| Language | How overflow behaves by default | How to make it loud |
| --- | --- | --- |
| C / C++ | Signed = UB; unsigned = wraps | `-fsanitize=signed-integer-overflow,unsigned-integer-overflow`; `-ftrapv` |
| Rust | Wraps in `--release`, panics in debug | Use `checked_add`/`overflowing_add`; build with `overflow-checks=true` |
| Go | Silently wraps (defined two's complement) | No built-in trap; check operands or use `math/bits`, `math.MaxInt64` guards |
| Java | Silently wraps (defined two's complement) | `Math.addExact` / `Math.multiplyExact` throw `ArithmeticException` |
| Python | Never overflows (arbitrary precision ints) | N/A for `int`; but `numpy` fixed-width ints DO wrap silently |

That last row is the quiet contrast that makes Python a useful teaching tool. Python's built-in `int` is **arbitrary precision** — it grows as large as memory allows, so `2**100` is just a number and a counter never wraps. This is a deliberate trade: Python pays with slower arithmetic and more memory per integer to buy you freedom from the entire wraparound family. But the moment you reach for `numpy` (or `array`, or `ctypes`), you are back on fixed-width hardware integers that wrap silently, and the bug returns — `np.int32(2147483647) + 1` is `-2147483648` with no warning. For more on how that fixed-width array model works and why it is fast, see [NumPy from first principles](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast).

## 4. The classic integer bug: midpoint overflow in binary search

No integer bug is more famous, or more instructive, than the overflow in binary search. It lived undetected in the Java standard library and in *Programming Pearls* for years before Joshua Bloch wrote it up in 2006. It is the perfect teaching case because the code looks unimpeachable and fails only at scale, which is exactly the property that lets a numeric bug ship.

Here is the buggy midpoint, the line that has bitten thousands of programmers:

```java
// Classic binary search — looks correct, overflows at large indices.
int binarySearch(int[] a, int key) {
    int lo = 0;
    int hi = a.length - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;   // <-- the bug
        if (a[mid] < key)       lo = mid + 1;
        else if (a[mid] > key)  hi = mid - 1;
        else                    return mid;
    }
    return -1;  // not found
}
```

The mechanism: `mid = (lo + hi) / 2` first computes `lo + hi`, then divides. The division is fine, but the *sum* can overflow. When the array is large and the live region sits in the high indices, `lo + hi` can exceed `INT_MAX`. In Java, signed overflow wraps, so `lo + hi` becomes a large negative number, `mid` becomes negative, and `a[mid]` throws `ArrayIndexOutOfBoundsException` — or, in C, silently reads out of bounds. The bug is invisible until the array crosses roughly $2^{30}$ elements, because below that `lo + hi` cannot exceed `INT_MAX`.

![A before-and-after comparison showing the buggy lo plus hi midpoint overflowing to a negative index versus the safe lo plus the gap over two form that stays in range at two billion elements](/imgs/blogs/integer-overflow-and-floating-point-traps-3.png)

The fix is one of the most elegant in all of debugging, because it changes the *shape* of the arithmetic so the overflow cannot occur:

```java
int mid = lo + (hi - lo) / 2;   // <-- the fix
```

Why this is safe: `hi - lo` is the gap, and since `hi >= lo` in a valid search, the gap is non-negative and no larger than `hi`, which fits in `int`. Halving the gap keeps it small. Adding it back to `lo` lands exactly where `(lo + hi) / 2` would have — same midpoint, no intermediate sum that can exceed `INT_MAX`. You never form the dangerous `lo + hi`. The same trick fixes the analogous bug in pointer arithmetic, timestamp averaging, and any "average of two large values" computation. The general rule is: **to average two values without overflow, never add them first; add half the difference to the smaller.**

#### Worked example: catching the midpoint overflow with UBSan

Suppose you maintain a search index over a large sorted array and a user reports that lookups "sometimes return the wrong record on the big shards." It is intermittent, only on the largest indexes, and never reproduces on your laptop's small test data. This is the textbook profile of a scale-dependent integer bug.

**Observe.** The wrong result appears only on shards with more than ~1.5 billion entries. Smaller shards are fine. That size threshold is a screaming clue: something crosses a $2^{30}$-ish boundary.

**Reproduce.** Do not try to reproduce with a 1.5-billion-element array (that is 6 GB of `int`s). Instead, reproduce the *arithmetic* by calling the midpoint computation directly with the boundary values. A C reproducer:

```c
#include <stdio.h>
int mid_buggy(int lo, int hi) { return (lo + hi) / 2; }
int mid_safe(int lo, int hi)  { return lo + (hi - lo) / 2; }

int main(void) {
    int lo = 1500000000, hi = 1700000000;   // both well under INT_MAX, sum is not
    printf("buggy mid = %d\n", mid_buggy(lo, hi));
    printf("safe  mid = %d\n", mid_safe(lo, hi));
    return 0;
}
```

**Hypothesize and trap.** Hypothesis: the sum `lo + hi` overflows `int`. Falsify it with UBSan rather than reading the output:

```bash
clang -g -fsanitize=signed-integer-overflow mid_demo.c -o mid_demo
./mid_demo
# mid_demo.c:2:36: runtime error: signed integer overflow:
# 1500000000 + 1700000000 cannot be represented in type 'int'
# buggy mid = -547483648
# safe  mid = 1600000000
```

There it is, localized to line 2, with the exact operands. The buggy midpoint is `-547483648` (negative — an out-of-bounds index), the safe one is `1600000000` (correct). Note the numbers: $1{,}500{,}000{,}000 + 1{,}700{,}000{,}000 = 3{,}200{,}000{,}000$, which exceeds `INT_MAX` ($2{,}147{,}483{,}647$) by $1{,}052{,}516{,}353$; wrap that around and you land near `-547` million. The arithmetic checks out to the digit.

**Fix and prevent.** Change `(lo + hi) / 2` to `lo + (hi - lo) / 2`. Then add a regression test that calls the midpoint with `lo, hi` near `INT_MAX` and asserts the result is in range — a test that would have failed instantly on the old code and passes on the new. Run the whole suite under UBSan in CI so the *next* overflow anywhere in the codebase fails the build instead of shipping. That CI step is the real prevention; the one-line fix only patches this instance. This boundary-value habit — testing at MAX, MIN, and zero — is the heart of the sibling post on off-by-one and boundary bugs (planned in this series under the slug `off-by-one-and-boundary-bugs`), because overflow is just the boundary bug of the number line.

## 5. Narrowing casts, signed/unsigned comparisons, and time_t

Two more integer traps deserve their own treatment because they fail in ways that are even sneakier than wraparound — they fail *quietly during a comparison or a cast* where no operation looks dangerous at all.

**Narrowing truncation.** When you copy a wide integer into a narrow one — `int64_t` into `int32_t`, `int` into `short`, a Python `numpy.int64` into a `numpy.int32` column — the high-order bits are simply discarded. If the value fit in the narrow type, nothing is lost. If it did not, you keep only the low bits and the high bits vanish without a sound:

```c
#include <stdint.h>
#include <stdio.h>
int main(void) {
    int64_t big = 0x1FFFF0000LL;   // 8589869056, needs 33 bits
    int32_t small = (int32_t)big;  // keeps only low 32 bits
    printf("big   = %lld\n", (long long)big);    // 8589869056
    printf("small = %d\n", small);               // -65536  (high bits gone)
    return 0;
}
```

The value `8589869056` becomes `-65536`. Catch this with UBSan's `implicit-conversion` check (which flags implicit narrowing that changes the value) or by grepping for explicit narrowing casts and asking, for each, "can the source value ever exceed the destination range?" This is the exact mechanism that destroyed the Ariane 5, which we will get to in the war stories — a 64-bit float was converted to a 16-bit signed integer, the value did not fit, and the conversion overflowed.

**Signed/unsigned comparison surprises.** This one violates intuition so hard it deserves to be tattooed somewhere visible. In C, when you compare a signed and an unsigned integer of the same width, the *signed value is converted to unsigned* before the comparison. So:

```c
#include <stdio.h>
int main(void) {
    int s = -1;
    unsigned u = 1;
    if (s > u) printf("yes, -1 > 1u\n");   // THIS PRINTS
    else       printf("no\n");
    return 0;
}
```

`-1 > 1u` is **true**. The `-1` converts to unsigned, becoming `4294967295` (the bit pattern `0xFFFFFFFF` reinterpreted as unsigned), which is indeed greater than `1`. Every loop of the form `for (int i = n - 1; i >= 0 && i < someUnsignedLength; i--)` is a candidate for this, and worse, every loop `for (size_t i = 0; i <= n - 1; i++)` where `n` is `0` runs forever because `n - 1` is `SIZE_MAX`. Compilers warn about signed/unsigned comparison with `-Wsign-compare` (part of `-Wextra`); turn it on and treat it as an error. The discipline is: **pick a signedness for a quantity and stick with it; do not mix signed and unsigned in comparisons.**

**time_t and the year 2038.** A specific, dated instance of overflow worth knowing: traditional Unix time is seconds since 1970-01-01 stored in a signed 32-bit `time_t`. That counter overflows at `2147483647` seconds, which is **03:14:07 UTC on 19 January 2038**. One second later it wraps to `INT_MIN` and the date reads 1901. Any 32-bit system still computing with 32-bit `time_t` in 2038 will see time run backward. The fix is a 64-bit `time_t`, which most modern platforms have adopted, but embedded systems and persisted 32-bit timestamps remain a live hazard. It is the integer-overflow family with a literal deadline.

**Integer division truncates toward zero.** One more integer trap is not about width at all but about the operation: integer division discards the remainder. `7 / 2` is `3`, not `3.5`, in C, Go, Java, and Rust, and in Python's `//` floor division. This is correct and intended, but it bites when you forget you are in integer land — `(a / b) * b` is not `a`, an average computed as `(x1 + x2 + ... + xn) / n` in integers silently floors, and a percentage computed as `count / total * 100` returns `0` whenever `count < total` because `count / total` floored to `0` *before* the multiply. The fix is to either reorder so the multiply happens first (`count * 100 / total`) or to promote one operand to floating point deliberately. The diagnostic is simple once you suspect it: print the intermediate, and if a ratio you expected to be fractional is exactly `0` or an integer, you are dividing too early. Note the subtlety that division also interacts with overflow at exactly one point: `INT_MIN / -1` overflows, because the true result `2147483648` does not fit in `int` — the one division that can overflow, and a real source of `SIGFPE` crashes.

**Mixing integer widths in expressions.** When an expression mixes `int`, `long`, `size_t`, and `int64_t`, the language applies *integer promotion* and *usual arithmetic conversions* before computing, and the result type can surprise you. A common shape: `int a = 100000; int b = 100000; long c = a * b;` — the multiply `a * b` is done in `int` (both operands are `int`), overflows to a wrapped 32-bit value, and only *then* gets widened to `long`. The widening happens too late; the overflow already occurred in the narrow type. The fix is to widen an operand *before* the operation: `long c = (long)a * b;` forces the multiply into `long`. The rule: **the result width is decided by the operand widths, not by where you store it — widen the inputs, not just the output.**

## 6. The floating-point mechanism: why 0.1 + 0.2 is not 0.3

Now the other family. Open a Python REPL and type `0.1 + 0.2`. You get `0.30000000000000004`. This is not a Python bug; you get the same in C, Java, JavaScript, Go, Rust — anywhere that uses IEEE-754 doubles, which is everywhere. To debug floats you have to understand *why*, and the why is genuinely simple once you see it.

![A layered stack showing a binary64 double split into one sign bit, eleven exponent bits, and fifty-two mantissa bits, with zero point one shown as a repeating binary fraction that gets rounded to fit](/imgs/blogs/integer-overflow-and-floating-point-traps-4.png)

A `double` is 64 bits arranged as: 1 sign bit, 11 exponent bits, and 52 mantissa (fraction) bits. The value is, roughly, $\pm\, 1.f \times 2^{e}$, where $f$ is the 52-bit binary fraction and $e$ is the exponent. The mantissa stores binary digits — halves, quarters, eighths — not decimal digits. And here is the crux: **`0.1` in binary is a repeating fraction**, just as `1/3 = 0.3333...` repeats in decimal. In binary, `0.1` is `0.0001100110011001100...` repeating forever. With only 52 bits of mantissa, the machine must *round* this infinite expansion to the nearest representable value. The stored "0.1" is actually `0.1000000000000000055511151231257827021181583404541015625` — close, but not exact. Same for `0.2`. Add the two rounded values and the rounding errors do not cancel; they leave a residue, and the sum prints as `0.30000000000000004`.

The mental model to carry: **a `double` is a grid of representable points on the number line, and the spacing between points grows as the magnitude grows.** Near `1.0` the points are about $2^{-52} \approx 2.2 \times 10^{-16}$ apart. Near `1{,}000{,}000` they are about $2^{-32} \approx 2.3 \times 10^{-10}$ apart. Near `10^{16}$ they are more than `1.0` apart — at that magnitude, a `double` literally cannot represent consecutive integers, so `1e16 + 1 == 1e16` is true. Every decimal you write gets snapped to the nearest grid point, and arithmetic moves you between grid points with rounding at each step. The spacing of that grid, called a **ULP** (unit in the last place), is the natural unit of floating-point error, and it is the key to comparing floats correctly.

This grid structure gives the whole floating-point family its character. Three traps fall straight out of it:

- **Equality is meaningless** — `a == b` asks whether two values landed on the *exact same grid point*, which is almost never what you want when the values came from different computations.
- **Catastrophic cancellation** — subtracting two nearly equal values throws away all the significant digits they shared, leaving only the noisy low bits.
- **Accumulation error** — summing many values lets the per-step rounding errors pile up, and they can pile up surprisingly fast.

And two pathological values guard the edges: **`Inf`**, produced by overflow or divide-by-zero, and **`NaN`** (not a number), produced by `0.0/0.0`, `sqrt(-1)`, `Inf - Inf`, and friends. We will take each in turn.

## 7. Never use == on floats: epsilon, ULP, and isclose

The first rule every numerical programmer learns and then forgets under deadline pressure: **do not compare floats with `==`.** Because each operand is a rounded grid point, two computations that should be mathematically equal usually land on adjacent grid points, and `==` reports them unequal. The naive test:

```python
if 0.1 + 0.2 == 0.3:   # False! the left side is 0.30000000000000004
    print("equal")
```

The instinct is to allow a small tolerance: `abs(a - b) < epsilon`. That is the right idea but a *fixed* epsilon is also wrong, and understanding why is the whole point. Because the grid spacing grows with magnitude, an `epsilon` of `1e-9` is enormous near `0.0` (it spans millions of grid points) and yet *too small* near `1e12` (where adjacent grid points are already farther apart than `1e-9`, so even genuinely-equal values differ by more than your tolerance). A tolerance that makes sense at one magnitude is nonsense at another. The fix is a **relative** tolerance that scales with the magnitude of the operands, plus a small **absolute** tolerance to handle the region near zero where relative tolerance collapses.

This is exactly what Python's `math.isclose` does, and it is the reference design worth copying everywhere:

```python
import math

# rel_tol scales with magnitude; abs_tol rescues the near-zero case.
math.isclose(0.1 + 0.2, 0.3)                 # True (default rel_tol=1e-09)
math.isclose(1e12 + 1.0, 1e12)               # True near a large magnitude
math.isclose(0.0, 1e-12, abs_tol=1e-9)       # True near zero, needs abs_tol
math.isclose(0.0, 1e-12)                     # False! rel_tol alone fails at zero
```

`math.isclose(a, b)` returns true when `abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)`. The relative term handles large magnitudes; the absolute term handles values near zero, where any relative tolerance times a near-zero magnitude is itself near zero. **You must set `abs_tol` explicitly when either value can be zero** — the default `abs_tol=0.0` means `isclose(0.0, 1e-12)` is `False`, which surprises people. Choosing the right tolerances is a judgment call about your domain, not a universal constant, and that is the honest truth: there is no single correct epsilon.

The other rigorous approach is a **ULP comparison**: reinterpret the two floats' bit patterns as integers and ask how many representable grid points apart they are. Two floats one ULP apart are adjacent on the grid; "within 4 ULPs" is a precise, magnitude-aware closeness. In C:

```c
#include <stdint.h>
#include <math.h>
#include <string.h>

int within_ulps(double a, double b, int max_ulps) {
    if (isnan(a) || isnan(b)) return 0;           // NaN is never close
    int64_t ia, ib;
    memcpy(&ia, &a, sizeof ia);                   // type-pun via memcpy (safe)
    memcpy(&ib, &b, sizeof ib);
    if ((ia < 0) != (ib < 0)) return a == b;      // handle signed-zero crossing
    int64_t diff = ia > ib ? ia - ib : ib - ia;
    return diff <= max_ulps;
}
```

This works because IEEE-754 was deliberately designed so that, for same-sign values, the integer ordering of the bit patterns matches the float ordering — adjacent floats have adjacent integer representations. ULP comparison is what test frameworks use internally for "almost equal" assertions. The takeaway: **comparing floats is a real decision (relative tolerance, absolute floor, or ULP count), never a bare `==`.**

There is one method point that ties the whole float family together and is worth stating on its own, because it is the cheapest, most-skipped diagnostic in numerical debugging: **print floats at full precision when you are debugging, never at display precision.** A `double` carries about 17 significant decimal digits, and `%.17g` (C/printf), `repr(x)` or `f"{x!r}"` (Python), `%+.17e` for the sign and exponent, or `Double.toString` with extra digits (Java) all show you the real stored value, residue and all. The reason this matters is structural: every other formatting choice — `%.2f`, `round(x, 2)`, a currency formatter — *hides the bug by rounding it away*. The three-cent drift is invisible at two decimal places and obvious at seventeen. So the instant you suspect a float is wrong, reach for the full-precision print first. It costs one line and it converts "the number looks fine to me" into "the number is `0.30000000000000004`, there is the residue." Half the float bugs I have chased were solved the moment someone printed `repr` instead of trusting the rounded log line.

## 8. Catastrophic cancellation: subtracting away your precision

The most insidious float bug is the one where the type works perfectly and you still lose all your accuracy. It is called **catastrophic cancellation**, and the mechanism is: when you subtract two nearly-equal numbers, the leading digits they share cancel out, and what remains is built entirely from the low-order bits — the *least* accurate part of each operand. You do not lose a little precision; you can lose nearly all of it in a single subtraction.

A concrete demonstration. Suppose `a = 1234567.891234567` and `b = 1234567.891234521`, both stored as doubles. Each holds about 16 significant digits, so each is accurate to roughly the 16th digit. Their difference is `0.000000046` — but the first 9 or so significant digits of `a` and `b` were identical and canceled, so the difference is determined by digits 10 through 16 of the operands, of which only a handful were ever accurate. You started with 16 digits of precision and the subtraction handed you back maybe 6. The error did not appear from nowhere; it was always there in the low bits, hidden under the shared leading digits, and the subtraction *exposed* it by removing everything else.

The textbook example is the quadratic formula. Computing $x = \frac{-b + \sqrt{b^2 - 4ac}}{2a}$ when $b^2 \gg 4ac$ means $\sqrt{b^2 - 4ac} \approx |b|$, so for one root you subtract two nearly equal numbers and lose precision catastrophically. The fix is algebraic — rationalize the formula so the dangerous subtraction becomes an addition, or compute the well-conditioned root first and get the other from $x_1 x_2 = c/a$. A worked numeric case:

```python
import math

def roots_naive(a, b, c):
    d = math.sqrt(b*b - 4*a*c)
    return ((-b + d) / (2*a), (-b - d) / (2*a))

def roots_stable(a, b, c):
    d = math.sqrt(b*b - 4*a*c)
    # add the same-sign terms (no cancellation), then use the product of roots.
    q = -(b + math.copysign(d, b)) / 2
    return (q / a, c / q)

a, b, c = 1.0, 1e8, 1.0
print(roots_naive(a, b, c))   # one root has lost most of its precision
print(roots_stable(a, b, c))  # both roots accurate
```

For `a=1, b=1e8, c=1`, the small root is approximately `-1e-8`. The naive formula computes it as the difference of two numbers near `1e8`, and the answer comes back wrong in most of its digits; the stable formula gets it right. The mechanism to remember: **cancellation does not create error, it reveals error that was always hiding in the low bits — so the cure is to restructure the math to avoid the subtraction, not to add more precision.** No epsilon fixes this; only better-conditioned arithmetic does.

## 9. Accumulation error and Kahan summation

Summing many floats is the everyday version of accumulation error, and it is where money bugs are born. Each addition rounds the running total to the nearest grid point, and over a million additions those tiny roundings accumulate. The error is not random noise that averages out; when you add many small numbers to a large running sum, each small number's low bits fall *below the grid spacing of the large sum* and are silently dropped. Add `1.0` to `1e16` enough times and nothing happens at all, because `1.0` is smaller than the grid spacing near `1e16`.

![A before-and-after comparison showing a million prices summed as doubles drifting three cents off versus the same prices summed as integer cents staying exact to the penny](/imgs/blogs/integer-overflow-and-floating-point-traps-5.png)

The classic illustration: sum the value `0.1` ten million times. Mathematically the answer is `1000000.0`. In naive `double` arithmetic you get something like `999999.9998389754` — off by a fraction, because each `0.1` is slightly wrong and the errors accumulate. For scientific work the fix is **Kahan summation** (compensated summation), which keeps a running correction term that captures the low-order bits lost at each addition and folds them back in:

```python
def naive_sum(xs):
    total = 0.0
    for x in xs:
        total += x
    return total

def kahan_sum(xs):
    total = 0.0
    comp = 0.0           # running compensation for lost low-order bits
    for x in xs:
        y = x - comp     # apply the correction from last time
        t = total + y    # the lossy addition
        comp = (t - total) - y   # recover what y's low bits lost in that add
        total = t
    return total

data = [0.1] * 10_000_000
print(naive_sum(data))   # ~999999.9998389754  (drifted)
print(kahan_sum(data))   # 1000000.0           (exact to display precision)
```

The trick of Kahan's algorithm: `(t - total) - y` recovers the part of `y` that was too small to survive the addition `total + y`, and stashes it in `comp` to be added back next iteration. It roughly squares your effective precision for the cost of three extra operations per element. (Modern alternatives — Neumaier's variant, pairwise summation, `math.fsum` in Python which is exact — go further; `numpy` uses pairwise summation internally for `np.sum`, which is why it drifts less than a naive Python loop.)

But Kahan summation is the right answer for *scientific* sums, not for *money*. For money, the right answer is to **not use floats at all**, which the next worked example makes concrete.

#### Worked example: a financial report off by three cents

A reconciliation job sums roughly 1.2 million transaction amounts into a daily total. Accounting reports the system total is `\$0.03` short of the sum they computed by hand from the ledger. Three cents on millions of dollars — small enough that nobody noticed for months, large enough that it will never reconcile and an auditor will eventually ask.

**Observe.** The discrepancy is tiny, always in the same direction (the float total is slightly *low*), and it grows on days with more transactions. A consistent, magnitude-dependent, sub-cent-scale error in a sum is the unmistakable fingerprint of float accumulation, not of a logic error.

**Reproduce.** Pull the day's amounts and sum them two ways — as `double` and as integer cents:

```python
amounts = load_amounts("2026-06-19.csv")   # e.g. 19.99, 4.50, 1234.07, ...

float_total = 0.0
for a in amounts:
    float_total += a                        # the production code path

cents_total = 0
for a in amounts:
    cents_total += round(a * 100)           # exact: store pennies as ints

print(f"float total : {float_total:.17g}")  # 4821773.9700000007 (say)
print(f"cents total : {cents_total / 100:.2f}")  # 4821774.00
print(f"drift       : {cents_total/100 - float_total:.10f}")  # 0.0299999...
```

**Hypothesize and falsify.** Hypothesis: the `float_total` path accumulates rounding error. Falsify it the right way — by **printing full precision**. The bug is invisible at `:.2f` because the display rounds it away; you must print `:.17g` (or `repr(x)`) to see the real stored value. The instant you print `float_total` at 17 significant digits and see `4821773.9700000007` where the true total is `4821774.00`, the hypothesis is confirmed: the float carries a non-zero residue in its low digits, the cents version does not. The single most useful float-debugging habit is **always print floats with `%.17g` or `repr`, never with rounded formatting, when you are debugging.** Rounded output is what hid the bug in the first place.

**Fix.** Switch the money path off floats. Store and sum **integer cents** (or use a `Decimal`):

```python
from decimal import Decimal, ROUND_HALF_UP

# Option A: integer cents. Parse to pennies, sum exactly, format at the end.
total_cents = sum(round(Decimal(a) * 100) for a in raw_amount_strings)

# Option B: Decimal with explicit quantization and rounding mode.
total = sum((Decimal(s) for s in raw_amount_strings), Decimal("0"))
total = total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
```

Integer cents is exact because integer addition is exact — there is no grid, no rounding, no drift. `Decimal` is exact for decimal fractions because it stores base-10 digits, at the cost of speed; it is the right tool when you need fractional cents, configurable rounding, or to match a regulator's specified rounding mode. Either way the `\$0.03` drift goes to exactly `\$0.00`.

**Prevent.** Add a test that sums a known fixture and asserts an *exact* equality (now possible because integers and `Decimal` are exact), and forbid `float`/`double` in the money layer with a lint rule or a dedicated `Money` type that wraps integer cents. The general rule, learned the hard way by everyone who has shipped a billing system: **money is never a float.** The numbers across this whole investigation are illustrative of the *pattern*, not a specific company's books — but the pattern is exactly what you will see, and the `%.17g` print is exactly how you will see it.

| Money representation | Exact? | Cost / caveat | Use when |
| --- | --- | --- | --- |
| `double` / `float` | No — drifts | Fast; silent rounding error | Never for money |
| Integer cents (`int64`) | Yes | Manual scaling; watch overflow at ~9e16 cents | Most currency math |
| Decimal / BigDecimal | Yes (base-10) | Slower; pick a rounding mode | Fractional cents, regulated rounding |
| Rational / fraction | Yes | Slowest; denominators grow | Rarely; exact ratios |

## 10. NaN and Inf: the values that break comparisons

The last floating-point trap is the strangest, because it involves values that are not numbers at all. IEEE-754 defines two families of special values, and both are *legal* doubles that your code will happily compute with and pass around until they cause visible damage somewhere far away.

**`Inf` (infinity)** is produced by overflow (a result too large to represent, like `1e308 * 10`) or by dividing a nonzero number by zero (`1.0 / 0.0`). It is signed (`+Inf`, `-Inf`) and mostly behaves sensibly: `Inf + 1 == Inf`, `1 / Inf == 0`. It becomes a problem when it flows into a calculation that then produces `NaN` — for example `Inf - Inf` or `Inf / Inf`, both of which are `NaN`.

**`NaN` (not a number)** is produced by genuinely undefined operations: `0.0 / 0.0`, `sqrt(-1.0)`, `log(-1.0)`, `Inf - Inf`, `0.0 * Inf`. And `NaN` has a property that is the single most important fact in floating-point debugging: **`NaN` is not equal to anything, including itself.** `NaN == NaN` is `false`. `NaN < x`, `NaN > x`, `NaN == x` are *all* false for every `x`. This is required by the standard, and it has two enormous consequences — one useful, one catastrophic.

![A flow graph showing a divide of zero by zero producing a NaN that propagates through arithmetic and breaks comparisons so a sort fails and an aggregate becomes NaN, with a side branch detecting it via x not equal to x](/imgs/blogs/integer-overflow-and-floating-point-traps-6.png)

The *useful* consequence: **`x != x` is true if and only if `x` is `NaN`.** That is the canonical, portable NaN test, and it works in every IEEE-754 language. It is what `isnan` does internally. So even without a library you can detect a NaN:

```c
int is_nan(double x) { return x != x; }   // the only value not equal to itself
```

The *catastrophic* consequence: a single `NaN` **poisons everything it touches.** `NaN + anything` is `NaN`. So if one bad division produces a `NaN` and it flows into a running sum, the *entire* sum becomes `NaN` — not just one term, all of it. Worse, because every comparison against `NaN` is false, a `NaN` silently breaks sorting algorithms (which rely on `a < b` being a total order), breaks `min`/`max` (which may return the `NaN` or skip it inconsistently), breaks binary searches, and breaks any threshold check (`if (x > limit)` is false for `NaN`, so the NaN slips past your validation). One invalid operation in one row of a dataset can turn an entire pipeline's output to `NaN` or scramble a sort, and the symptom shows up nowhere near the cause.

The method is the same as for integers: **trap the moment the NaN is born, do not chase where it ended up.** There are two ways. First, detect on read: check `isnan`/`isinf` at the boundaries where data enters your computation, and reject or repair before the NaN can spread. Second — and this is the heavy artillery — **make the FPU raise a signal the instant a NaN or Inf is created**, using the floating-point exception flags:

```c
#define _GNU_SOURCE
#include <fenv.h>
#include <stdio.h>
#include <math.h>

int main(void) {
    // Trap the moment an invalid op (NaN-producing) or divide-by-zero happens.
    feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);

    double a = 0.0, b = 0.0;
    double c = a / b;   // <-- raises SIGFPE here, at the birth of the NaN
    printf("%f\n", c);  // never reached
    return 0;
}
```

`feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW)` (a GNU extension; on macOS/BSD you set the FPU control word directly) makes the CPU raise `SIGFPE` at the exact operation that creates a `NaN` or `Inf`. Now the bug crashes loudly at line 12 instead of producing a silent `NaN` that surfaces as a broken sort three modules later. Attach `gdb`, and you are standing on the division that did it. In Python, `numpy.seterr(invalid='raise', divide='raise')` does the equivalent for array operations, turning a silent `nan` into a `FloatingPointError` you can trace:

```python
import numpy as np
np.seterr(invalid="raise", divide="raise")   # turn silent NaN into an exception
a = np.array([0.0, 1.0, 2.0])
b = np.array([0.0, 1.0, 0.0])
c = a / b    # raises FloatingPointError instead of returning [nan, 1., inf]
```

The prevention is to validate at boundaries (`isnan`/`isinf` on inputs and at module edges), to decide explicitly what a `NaN` *means* in your domain (skip it? treat as zero? reject the record?), and to keep a `NaN` from ever entering a sort or a comparison-based algorithm. **A `NaN` in your data is a question your code must answer before it propagates, not after.**

## 11. The unified method: a debugging matrix and the loop

Step back and the two families share a single debugging shape, which is the reason this post treats them together. For every trap, there is (1) a symptom you can recognize, (2) a tool that traps the error *at the operation* rather than at the distant symptom, and (3) a type-level or algorithmic change that prevents it. That triple — symptom, detector, prevention — is the entire playbook.

![A matrix mapping each numeric trap to its symptom, the runtime detector that traps it, and the prevention, covering signed overflow, unsigned wrap, float equality, NaN or Inf, and money drift](/imgs/blogs/integer-overflow-and-floating-point-traps-7.png)

The matrix is worth memorizing because it converts a vague "the number is wrong" panic into a deterministic procedure. You read the symptom, jump to the detector, run it, and it points at the operation. Then you apply the prevention so it cannot recur. No staring, no guessing.

And the whole thing rides the series' spine — **observe → reproduce → hypothesize → bisect → fix → prevent** — which for numeric bugs has a particularly clean form, because the computation *is* the search space. You binary-search the sequence of operations for the first one whose output is wrong, exactly the way the timeline below narrows the cents-drift case.

![A timeline of the debugging loop for the cents drift case moving from observing a three cent shortfall through reproducing, hypothesizing float rounding, bisecting by printing full precision, fixing with integer cents, and preventing with a money type test](/imgs/blogs/integer-overflow-and-floating-point-traps-8.png)

The "bisect" step deserves emphasis because it is where numeric debugging diverges from the rest of the series. You are not bisecting *commits* (though you might, if a regression introduced the bug — that is when [git bisect](/blog/software-development/version-control/using-git-like-senior-workflow-troubleshooting-playbook) earns its keep). You are bisecting *the data flow*: instrument the computation at its midpoint, check whether the value is already wrong there, and recurse into the half that is wrong. Print `%.17g` at stage 1, stage 2, stage 4, stage 8 — find the first stage where the value diverges from the integer-or-symbolic ground truth, and the operation between the last-good and first-bad stage is your culprit. It is binary search applied to arithmetic.

#### Worked example: the counter that wrapped at 2.1 billion

A metrics service exposes a 32-bit signed counter for total events processed. After a heavy traffic month it starts reporting *negative* event counts, and downstream dashboards show wild negative spikes that trigger false alerts at 3am.

**Observe.** The counter went negative right around the time cumulative events crossed two billion. Two billion is suspiciously close to `INT_MAX` (`2147483647`). That single observation is almost the whole diagnosis — a value going negative near $2.1 \times 10^9$ is a signed 32-bit overflow until proven otherwise.

**Reproduce.** You cannot wait a month for the real counter, so reproduce the *arithmetic* at the boundary. In Go (where the metrics service is written), reproduce the wrap directly:

```go
package main

import (
	"fmt"
	"math"
)

func main() {
	var count int32 = math.MaxInt32   // 2147483647
	fmt.Println("before:", count)     // 2147483647
	count++                           // silently wraps in Go
	fmt.Println("after :", count)     // -2147483648
}
```

**Hypothesize and confirm.** Hypothesis: the counter is an `int32` and wrapped at `INT_MAX`. Go does not have UBSan, so confirm by the wrap behavior above and by inspecting the type declaration. (In a C or Rust service you would compile with `-fsanitize=signed-integer-overflow` or build Rust with overflow checks and catch the exact increment.) The reproduction printing `-2147483648` after incrementing `MaxInt32` is the confirmation. This is precisely the mechanism behind the most famous instance of this bug, which we cover in the war stories.

**Fix.** Widen the type to `int64` (range up to about $9.2 \times 10^{18}$ — at a billion events per second it would take 292 years to overflow), or, in a language that supports it, use checked arithmetic so an overflow is an error rather than a silent wrap:

```rust
// Rust: choose the explicit overflow policy. checked_add returns None on overflow.
let count: i64 = 2_147_483_647;
match count.checked_add(1) {
    Some(n) => println!("ok: {n}"),
    None    => eprintln!("overflow! widen the type or reset"),
}
// wrapping_add is the explicit "I want two's complement wrap" opt-in.
let wrapped = i32::MAX.wrapping_add(1);   // -2147483648, but you ASKED for it
```

```java
// Java: Math.addExact throws ArithmeticException on overflow instead of wrapping.
long count = Integer.MAX_VALUE;
count = Math.addExact(count, 1);   // safe here because count is long
// but: Math.addExact(Integer.MAX_VALUE, 1) on ints THROWS — exactly what you want
```

**Prevent.** Use a 64-bit counter for any monotonically increasing quantity, and in the languages that offer it, make overflow *checked* in the hot path (Rust's `checked_add`, Java's `Math.addExact`, C's UBSan in CI) so a future overflow is a loud failure, not a silent negative number. The contrast worth internalizing: **Rust forces you to choose** — `checked_add` (error on overflow), `wrapping_add` (deliberate wrap), `saturating_add` (clamp at the max), or `overflowing_add` (wrap plus a flag) — so overflow is never accidental. **Python never overflows** its native `int` at all. **C, Go, and Java wrap silently by default** and make you opt into safety. Knowing your language's default is half the battle.

## 12. War stories: when wrong numbers wrote history

These bug classes are not academic. They have destroyed rockets, grounded fleets, and broken the internet's most-watched video. Each is a textbook instance of one trap in this post, and each is worth knowing because the engineers involved were not careless — the bugs were invisible by construction.

**The Ariane 5 (1996): a 64-to-16-bit conversion that destroyed a rocket.** Forty seconds after launch, the European Space Agency's Ariane 5 flight 501 veered off course and self-destructed, taking about \$370 million of rocket and satellites with it. The root cause was a narrowing conversion: the inertial reference system converted a 64-bit floating-point value (horizontal velocity) to a 16-bit signed integer. On the slower Ariane 4 the value always fit. The Ariane 5 flew faster, the horizontal velocity exceeded `32767`, the conversion overflowed, the guidance software raised an unhandled exception, the backup system (running identical code) failed the same way milliseconds later, and the rocket's nozzles swung to a destructive angle. It is the **narrowing-truncation** trap from section 5, at the worst possible scale — and a reminder that "this value has always fit" is an assumption, not a guarantee.

**The Boeing 787 (2015): a 248-day integer overflow.** The U.S. FAA issued an airworthiness directive warning that if a 787 Dreamliner's generator control units were left powered continuously for 248 days, they would all enter failsafe mode simultaneously and shut down — potentially losing all AC electrical power, in flight. The cause was a counter incremented at a fixed rate that overflowed a 32-bit signed integer after about 248 days of continuous operation ($2^{31}$ centiseconds is roughly 248 days). The fix in the field was procedural — power-cycle the units periodically — until a software patch shipped. It is the **signed-overflow** trap with a long fuse: a counter that wraps not at a value you control but after a duration nobody tested for, because who runs the integration test for 248 days?

**Gangnam Style and YouTube's 32-bit view counter (2014).** When Psy's "Gangnam Style" approached `2{,}147{,}483{,}647` views, it approached `INT_MAX` for a signed 32-bit integer — the exact value the view counter used. YouTube announced, with good humor, that they had upgraded the counter to a 64-bit integer (max about $9.2 \times 10^{18}$) so the count would not wrap. This is the friendliest possible version of the **signed-overflow** trap: a counter that simply outgrew its type. The fix — widen to 64 bits — is exactly the fix from the counter worked example above. Nobody died; the internet just got a fun illustration of why `int64` exists.

**Patriot missile clock drift (1991): float accumulation cost lives.** During the Gulf War, a Patriot missile battery failed to intercept an incoming Scud missile that struck a barracks, killing 28 people. The cause was floating-point accumulation error in the system's internal clock. Time was tracked in tenths of a second, multiplied by `0.1` stored as a truncated 24-bit binary fraction — and `0.1` is not exactly representable in binary (exactly the `0.1` problem from section 6). The tiny per-tick error accumulated over about 100 hours of continuous operation into a timing error of about a third of a second, enough to mispredict the fast-moving Scud's position by over half a kilometer and miss the intercept. It is the **accumulation-error** trap from section 9, and it is the most sobering one: the same drift you saw in the cents example, scaled up to a fatal miss because the system ran far longer than its precision budget allowed.

These four span the whole post: narrowing (Ariane), signed overflow (Boeing, Gangnam Style), and float accumulation (Patriot). The throughline is that none of them looked like a bug in the code. The arithmetic was "correct" in the sense that every operation did exactly what the hardware defines — it just did not do what the engineers *meant*. That gap, between what the machine computes and what you intend, is the entire subject of this post.

## 13. Stress-testing your fix: what if it only breaks at scale?

A fix you cannot reproduce the failure for is a fix you cannot trust. Numeric bugs are masters of hiding, so it is worth walking through the conditions under which they surface, because each one changes how you reproduce and verify.

**"It only reproduces at large N."** Integer overflow and accumulation error are both scale-dependent — the midpoint overflow needs $2^{30}$ elements, the cents drift needs millions of rows, the Patriot drift needed 100 hours. You almost never want to reproduce at full scale (too slow, too much memory). Instead, reproduce the *arithmetic at the boundary*: call the overflowing computation directly with `INT_MAX`-adjacent operands, as in the worked examples. The bug is in the operation, not the loop around it, so feed the operation its boundary inputs and you reproduce in microseconds.

**"It only happens in release builds."** This is a red flag for *signed overflow specifically*, because the optimizer's "overflow can't happen" assumption only deletes your check at `-O2`/`-O3`. At `-O0` your hand-rolled check might "work" (the overflow wraps and your `if` sees it) and at `-O2` it vanishes. The fix is to never write the check in terms of the overflow; use UBSan (which works at any optimization level) or a checked-arithmetic primitive. If a numeric bug appears only optimized, suspect UB.

**"It only happens on one machine."** Floating-point results can differ across platforms because of extended-precision intermediate registers (x87's 80-bit registers vs SSE's 64-bit), fused multiply-add (FMA) contraction, different math-library implementations of `sin`/`exp`, and compiler flags like `-ffast-math` that relax IEEE semantics. If `0.1 + 0.2` differs between two hosts, suspect `-ffast-math` or FMA. Pin it down by compiling with strict IEEE settings (`-ffp-contract=off`, never `-ffast-math` for code that cares about exactness) and comparing. The discipline: **`-ffast-math` trades correctness for speed; never enable it in code that must reconcile to the penny or compare floats.**

**"It only fails after hours/days of uptime."** Accumulation error (Patriot) and slow-counter overflow (Boeing's 248 days) both need *time*, not data volume. You cannot run a 248-day test in CI. Reproduce by *fast-forwarding the counter*: seed the accumulator or counter to just below the danger threshold and run a short burst across the boundary. For the 787 you would set the centisecond counter to `INT_MAX - 1000` and watch it cross; for accumulation you would run enough iterations to cross from "error invisible" to "error visible," which is far fewer than the real runtime if you seed the running sum high.

**"I can't attach a debugger / sanitizer in prod."** You usually cannot run UBSan in production (it slows things and changes behavior), and you should not — sanitize in CI and in a reproduction harness. In prod, the move is *defensive instrumentation*: log the operands of the suspect operation when a cheap guard trips (e.g., log when a counter exceeds `INT_MAX/2`, or when an input to a division is near zero, or when an `isnan` check fires at a module boundary). Those logs let you reproduce offline with the real operands, and then you bring the heavy tools to the reproduction, not to prod. This mirrors the broader reproduce-first discipline of the series: get the failing inputs first, then debug them where you can use every tool.

## 14. How to reach for these tools (and when not to)

Every technique here has a cost, and a principal engineer's job is knowing when *not* to pay it. Honest guidance:

**Reach for UBSan / `-fsanitize=...` in CI and reproduction, not in prod.** It catches integer overflow at the exact operation, which is decisive, but it adds overhead (often 1.5–3×) and can change timing. Run your test suite under it so new overflows fail the build; do not ship a sanitized binary to production. The payoff is enormous in CI and negative in prod.

**Do not enable FP exception trapping (`feenableexcept`) blanket-wide in a large codebase.** Plenty of correct code *deliberately* produces `Inf`/`NaN` as sentinel values, and trapping globally will `SIGFPE` on code that is working as intended. Enable it narrowly, around the computation you are debugging, then turn it off. It is a scalpel, not a policy.

**Do not reach for `Decimal`/`BigDecimal` everywhere "to be safe."** It is 10–100× slower than native floats and unnecessary for the vast majority of numeric code (graphics, physics, statistics, ML) where small relative error is fine and floats are the *correct* choice. Use it precisely where exactness in base-10 matters: money, and regulated calculations with specified rounding. Using `Decimal` for a physics simulation is as much an error as using `double` for a ledger.

**Do not chase a float discrepancy with more precision before checking conditioning.** If a result is wildly wrong (not slightly), the problem is usually catastrophic cancellation or a `NaN`, and switching `float` to `double` or `double` to `long double` just delays the failure without fixing it. Restructure the math (the stable quadratic, Kahan summation) or trap the `NaN`. More bits is the lazy non-fix.

**Do not write your own `epsilon` comparison when a library has `isclose`.** People get the relative-vs-absolute tolerance wrong constantly. Use `math.isclose`, `numpy.isclose`, `pytest.approx`, Google Test's `EXPECT_NEAR`/`EXPECT_FLOAT_EQ` (which uses ULPs), or your language's tested equivalent. Hand-rolled float comparison is a bug magnet.

**Do reach for the boundary values first.** The cheapest, highest-value diagnostic for any numeric bug is to run the computation at `MAX`, `MIN`, `0`, and the magnitude where epsilon fails. Most numeric bugs live at exactly those points, and testing them costs nothing. If you do one thing differently after reading this, make it this: add boundary tests.

| Technique | What it catches | Overhead | Reach for it when | Skip it when |
| --- | --- | --- | --- | --- |
| UBSan integer | Signed/unsigned overflow, narrowing | 1.5–3× | CI, reproduction harness | Production binaries |
| `-ftrapv` | Signed overflow only | High | Quick "make it crash" in tests | Anywhere unsigned matters |
| Checked arithmetic | Overflow at the call site | Small | Hot counters, sizes, indices | Code that should wrap (hashing) |
| `feenableexcept` | NaN/Inf at birth | Low, but invasive | Tracing one NaN to its source | Code that uses NaN/Inf as sentinels |
| `%.17g` / `repr` print | Hidden float residue | None | Always, when debugging floats | Production logs (noisy) |
| `Decimal` / integer cents | Money drift | 10–100× (Decimal) | Money, regulated rounding | Physics, ML, graphics |
| Kahan / `fsum` | Accumulation error | ~3× ops | Long scientific sums | Money (use integers instead) |
| `isclose` / ULP | False `==` failures | None | Every float comparison | Comparing to exact zero only |

## 15. Key takeaways

- **A wrong number with no crash is a bug class, not a mystery.** It is either integer wraparound (fixed width, modular arithmetic, two's complement) or float imprecision (binary approximation, finite precision). Diagnose the family first; the fixes do not transfer.
- **Trap the operation, not the symptom.** A wrapped integer or a `NaN` travels far before it does visible harm. Use UBSan, `feenableexcept`, or checked arithmetic to halt at the exact `+`, `-`, `*`, or `/` that produced the bad value.
- **Signed overflow is undefined behavior in C/C++ — the optimizer will delete your check.** Never detect overflow by letting it happen and inspecting the result; check operands before, or use a tool or a language with defined behavior.
- **To average two large values, write `lo + (hi - lo) / 2`, never `(lo + hi) / 2`.** The same shape fixes any "average of two large numbers" overflow.
- **Never compare floats with `==`; never trust a single fixed `epsilon` either.** Use a relative tolerance plus an absolute floor (`math.isclose` with `abs_tol`), or a ULP count. Tolerances are a domain decision, not a constant.
- **Money is never a float.** Use integer cents or `Decimal`. Floats drift, and the drift is invisible until you print `%.17g`, so print full precision when debugging.
- **`NaN != NaN`, so `x != x` detects a NaN — and one NaN poisons every downstream sum, comparison, and sort.** Validate at boundaries; decide what a NaN *means* before it propagates.
- **Catastrophic cancellation reveals error, it does not create it.** Subtracting near-equal values exposes the noisy low bits; the cure is better-conditioned math (stable quadratic, Kahan summation), not more precision.
- **Know your language's default.** Python ints never overflow; Rust forces an explicit overflow policy; C, Go, and Java wrap silently. The default decides whether the bug is even possible.
- **Test the boundaries.** `MAX`, `MIN`, `0`, and the magnitude where epsilon fails are where numeric bugs live. Boundary tests are the cheapest insurance you can buy, and they would have caught nearly every story in this post.

## 16. Further reading

- *"Extra, Extra — Read All About It: Nearly All Binary Searches and Mergesorts are Broken"* — Joshua Bloch's 2006 write-up of the midpoint-overflow bug, the canonical case study for this post.
- *What Every Computer Scientist Should Know About Floating-Point Arithmetic* — David Goldberg's classic paper; the definitive treatment of IEEE-754, ULPs, cancellation, and rounding.
- The **IEEE 754** standard and the [floating-point guide](https://floating-point-gui.de/) — accessible reference for why `0.1 + 0.2 != 0.3` and how to compare floats correctly.
- The **UndefinedBehaviorSanitizer** documentation (Clang/LLVM) — the flags, checks, and `UBSAN_OPTIONS` for trapping integer overflow at the operation.
- *Debugging* by David Agans and *Why Programs Fail* by Andreas Zeller — the methodical, hypothesis-driven debugging mindset this whole series is built on.
- Within this series: [stop guessing — the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) for the observe→reproduce→hypothesize→bisect→fix→prevent loop, and [hypothesize and falsify, not stare and hope](/blog/software-development/debugging/hypothesize-and-falsify-not-stare-and-hope) for turning a wrong number into a falsifiable claim. The companion post on off-by-one and boundary bugs (slug `off-by-one-and-boundary-bugs`) treats overflow as the boundary bug of the number line.
- Cross-system context: [NumPy from first principles](/blog/software-development/python-performance/numpy-from-first-principles-the-ndarray-and-why-its-fast) for where Python's safe arbitrary-precision ints give way to fixed-width wrapping arrays, and [using Git like a senior — the troubleshooting playbook](/blog/software-development/version-control/using-git-like-senior-workflow-troubleshooting-playbook) for bisecting a numeric regression to the commit that introduced it.
