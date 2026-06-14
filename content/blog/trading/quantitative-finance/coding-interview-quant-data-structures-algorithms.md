---
title: "Coding interviews for quant: the data structures and algorithms that actually come up"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A focused, finance-flavored tour of the data structures and algorithms that recur in quant coding interviews — arrays, hashmaps, monotonic stacks, heaps, sliding windows, binary search, sorting, and DP — each built from zero and solved on real dollar examples."
tags:
  [
    "quant-interviews",
    "coding-interviews",
    "data-structures",
    "algorithms",
    "big-o",
    "two-pointers",
    "monotonic-stack",
    "binary-search",
    "dynamic-programming",
    "interview-prep",
    "quantitative-finance",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 45
---

> [!important]
> **TL;DR** — Quant coding interviews are not generic LeetCode grinding. A focused set of data structures and algorithms — usually wrapped in a finance flavor like an order book, a price series, or a P&L curve — plus clean, correct implementation under a clock covers nearly all of it.
>
> - The whole field reduces to about eight tools: **arrays and two pointers**, **hashmaps for counting**, the **monotonic stack**, **heaps / priority queues**, the **sliding window**, **binary search** (including binary-search-on-the-answer), **sorting**, and **dynamic programming**. Graphs show up occasionally.
> - The single highest-value skill is reading the time and space complexity of your own code on sight. An `O(n^2)` solution that an `O(n)` one exists for is the most common reason a desk passes on a candidate.
> - Almost every finance-flavored problem is a thin disguise: **maximum drawdown** is a running maximum, the **best single trade** is a running minimum, **top-k assets by volume** is a size-k heap, **next higher price** is a monotonic stack, **merging two time series** is the merge step of mergesort.
> - The one number worth internalizing: an `O(n)` pass over a million rows is roughly a millisecond; the `O(n^2)` version is roughly fifteen minutes. The complexity class, not the constant factor, is what the clock punishes.
> - Interviewers grade clean, correct, *narrated* code over a clever one-liner. A working `O(n log n)` solution you can explain beats a buggy `O(n)` one you cannot.

Why can a senior quant developer glance at twenty lines of code and tell you, in three seconds, whether it will finish on a million-row file or time out? Why does "find the largest peak-to-trough drop in this price series" — which sounds like it needs you to compare every day against every earlier day — actually take a single left-to-right pass? And why do firms like Jane Street, Two Sigma, Citadel, Hudson River Trading, Jump, and Optiver keep asking variations of the same dozen problems, year after year, instead of inventing new ones?

The answer to all three is the same. There is a small, finite toolbox of data structures and algorithms, and being "good at coding interviews" is mostly the skill of recognizing which tool a problem wants and implementing it cleanly before the timer runs out. The problems *feel* like an endless grab-bag. They are not. Strip away the finance costume — the order books, the trade tapes, the P&L curves — and underneath sits one of about eight structures you can learn cold.

![A small set of data structures unlocks nearly every quant coding interview problem, mapping structures on the left to the problems they solve on the right.](/imgs/blogs/coding-interview-quant-data-structures-algorithms-1.png)

The diagram above is the mental model for this whole post. On the left are the structures. On the right are the problems a quant interview actually poses. Every arrow says "this tool unlocks this problem." If you learn the left column, the right column stops being scary, because a new problem you have never seen is almost always a recombination of tools you already hold.

This is a technique-first walkthrough, finance-flavored throughout. We will build each structure from absolute zero — no computer-science background assumed — define every term on first use, and ground every tool in a worked example with real dollar numbers that you can check by hand. We close with an "In the interview room" section that solves five full problems the way you would on a whiteboard, a list of the misconceptions that cost candidates offers, and a look at how these exact structures show up on a live trading desk. By the end you will not have memorized answers; you will have a checklist for classifying the next problem you have never seen.

## Foundations: what a quant coding interview is actually testing

Before any data structure, you need to know what is being measured, because it changes how you should solve everything below.

A quant coding interview is not a test of whether you have memorized a solution. The interviewer usually assumes you may have seen the problem; many are famous. What they are testing is **whether you can turn a fuzzy requirement into correct, efficient, readable code while talking through your reasoning** — because that is the actual job. A research idea or a trading signal is worthless until someone implements it correctly, and a slow implementation that takes overnight to backtest is nearly as bad as a wrong one. So the rubric has four parts, and they are graded roughly in this order:

1. **Correctness** — does your code produce the right answer on the normal case *and* the edge cases (empty input, one element, all-equal values, negative numbers)?
2. **Time and space complexity** — does it scale? An answer that is correct but `O(n^2)` when `O(n)` is possible will usually fail, because the desk runs these computations on data that does not fit in your patience.
3. **Clean code** — clear names, no dead code, sensible structure, handled edge cases. Reviewers on a desk read each other's code constantly; sloppy code is a real cost.
4. **Communication** — can you explain *why* your approach works, narrate your reasoning while coding, and respond to a "what if the input were a billion rows?" follow-up?

Let me define the handful of terms we will lean on throughout, building from nothing so none is assumed.

- **Data structure** — a way of organizing data in memory so that the operations you care about are fast. A grocery list on paper is a data structure; so is a phone book sorted by last name. The phone book is organized (sorted) precisely so you can *find* a name fast.
- **Algorithm** — a finite, unambiguous recipe for turning an input into an output. "Find the largest number in a list" has a one-line algorithm: scan once, remember the biggest so far.
- **Array** — a numbered row of slots holding values, like a strip of mailboxes labeled `0, 1, 2, …`. In finance, a price series, a list of returns, or a tape of trade sizes is an array. Reading slot `i` is instant; that instant access is the array's superpower.
- **Time complexity** — how the number of operations grows as the input grows, written with **Big-O notation** (defined in the next section). It is the single most important property of an algorithm in an interview.
- **In place** — modifying the input directly without allocating a second large structure, which keeps **space complexity** (extra memory used) low.
- **The online-assessment to onsite funnel** — the hiring pipeline. You start with an automated online assessment (OA), then a live phone or video screen, then an onsite loop of several rounds. Each stage tests a sharper version of the same skill.

![The quant coding-interview funnel narrows from an automated screen to a live onsite where you must talk while you code.](/imgs/blogs/coding-interview-quant-data-structures-algorithms-2.png)

The funnel above shows where each skill is stressed. The **online assessment** is auto-graded: pure correctness and complexity, no human watching, often a 70 to 90 minute window with two to four problems and a pass rate in the rough range of 10 to 20 percent. The **phone screen** is live: now you must talk while you type, so communication starts to count. The **onsite loop** is four to six rounds, mixing coding with probability, mental math, and a behavioral conversation. At every stage, a clean correct solution you can explain beats a clever one-liner you cannot — that is the through-line of the whole process.

### Python or C++?

Most quant firms let you choose your language for the coding rounds, and most candidates pick **Python** for its speed of expression: you write less code, so you spend the clock on the algorithm, not on semicolons. The risk is that Python hides cost — a `list.insert(0, x)` looks like one line but is `O(n)` because it shifts every element. A few firms, especially those hiring for low-latency systems, will ask for **C++** specifically, because on a real trading system the constant factors and memory layout matter enormously. If that is your target, the language itself becomes part of the test; see the companion post on [C++ for low-latency quant interviews](/blog/trading/quantitative-finance/cpp-for-low-latency-quant-interviews). For everything in *this* post we use short Python snippets, because they make the algorithm legible. The algorithms are language-agnostic.

With the vocabulary in place, we start with the one piece of analysis you actually need.

## Complexity and Big-O: the only analysis you need

Big-O notation is a way to describe how an algorithm's running time grows as its input grows, *ignoring constant factors and lower-order terms*. We write `O(n)` ("order n", or "linear") to mean: the work grows in proportion to the input size `n`. Double the input, double the work. We write `O(n^2)` ("quadratic") to mean the work grows with the *square* of the input: double the input, quadruple the work. The reason we throw away constants is that for large inputs, the *class* dominates everything else — an `O(n)` algorithm beats an `O(n^2)` one on big data no matter how clumsy the `O(n)` one's inner loop is.

Here is the menu, from fastest-growing-slowest to most explosive. These are the only classes you need to recognize on sight.

| Class | Name | Plain meaning | Finance example |
|---|---|---|---|
| `O(1)` | constant | same work regardless of input size | look up a price by ticker in a hashmap |
| `O(log n)` | logarithmic | each step halves the remaining work | binary search a sorted list of strikes |
| `O(n)` | linear | one pass over the data | sum a return series, find max drawdown |
| `O(n log n)` | linearithmic | sort, or a linear pass that does a log-work thing each step | sort trades by timestamp; top-k with a heap |
| `O(n^2)` | quadratic | compare every pair | naive "every day vs every earlier day" |
| `O(2^n)` | exponential | try every subset | brute-force "best subset of positions" (avoid!) |

![Same input size produces wildly different operation counts: the complexity class is what the clock punishes, not the constant factor.](/imgs/blogs/coding-interview-quant-data-structures-algorithms-3.png)

The chart above plots operations against input size for each class. Notice the shape: `O(1)` and `O(log n)` hug the bottom — they barely move as the input grows. `O(n)` rises steadily. `O(n^2)` shoots up almost vertically. That vertical wall is the whole game. Let me make it concrete with a number you should carry into every interview.

#### Worked example: why the complexity class, not the constant, decides

Suppose your machine does roughly 100 million simple operations per second (a fair rule of thumb for interpreted Python; compiled C++ is perhaps 100 times faster, but the *ratios* below are identical). You are given a price series with `n = 1,000,000` days.

- An **`O(n)`** algorithm does about 1,000,000 operations. At 100 million per second, that is `1,000,000 / 100,000,000 = 0.01` seconds. Effectively instant.
- An **`O(n log n)`** algorithm does about `n × log2(n) = 1,000,000 × 20 = 20,000,000` operations, or about 0.2 seconds. Fine.
- An **`O(n^2)`** algorithm does `n^2 = 1,000,000,000,000` (a trillion) operations. At 100 million per second, that is `1,000,000,000,000 / 100,000,000 = 10,000` seconds — **just under three hours.**

Same input, same machine. The linear version finishes before you lift your finger off the Enter key; the quadratic version is a coffee break that turns into lunch. This is why an interviewer's first question after you propose a solution is almost always "what's the time complexity?" — and why "can you do better than `O(n^2)`?" is the most common follow-up in the entire field. The one-sentence intuition: **the constant factor is a tip; the complexity class is the bill.**

A note on space. **Space complexity** is the extra memory your algorithm uses, measured the same way. An algorithm that scans a series keeping only a running total is `O(1)` space (one number); one that builds a hashmap of every distinct value is `O(n)` space. Interviewers care about both, and "can you do it in `O(1)` extra space?" is a common second follow-up once you have the time complexity down.

Now to the structures themselves.

## Arrays and two pointers

An array is the most basic structure: a numbered row of values. In finance it is almost always a *time series* — closing prices indexed by day, trade sizes indexed by sequence number, returns indexed by period. The two operations an array makes instant (`O(1)`) are reading the value at a known index and writing a value at a known index. What it does *not* make fast is searching for a value (you may have to scan the whole thing, `O(n)`) or inserting in the middle (everything after the insertion point must shift, `O(n)`).

The first real technique on arrays is the **two-pointer** method. Instead of looping with a single index, you keep two indices — often one at each end — and move them toward each other based on what you see. The payoff is that many problems that look like they need to check every *pair* of elements (which would be `O(n^2)`) collapse to a single `O(n)` pass when the array is sorted, because each pointer only ever moves in one direction.

![On a sorted array, move the pointer that improves the answer; no pair is examined twice, so the whole pass is linear.](/imgs/blogs/coding-interview-quant-data-structures-algorithms-4.png)

The figure walks the canonical case: find two values in a *sorted* array that add to a target. The two pointers start at the ends. You look at their sum. If it equals the target, you are done. If the sum is too small, the only way to grow it is to move the left pointer right to a bigger value. If the sum is too big, move the right pointer left to a smaller value. Crucially, once you move a pointer past an element, you never look at that element again — so each of the `n` elements is visited at most once, and the pass is `O(n)`.

#### Worked example: a pair of trade sizes summing to a target

You have sorted trade sizes `[2, 5, 7, 8, 11, 15]` (in thousands of shares) and want two that sum to exactly 17 (thousand). Start with `left = 0` (value 2) and `right = 5` (value 15).

```python
def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        s = arr[left] + arr[right]
        if s == target:
            return (left, right)  # found the pair
        elif s < target:
            left += 1   # need a bigger sum: grow the left value
        else:
            right -= 1  # need a smaller sum: shrink the right value
    return None         # no pair sums to target
```

On the first iteration, `2 + 15 = 17`, which equals the target, so we return immediately with indices `(0, 5)`. The whole thing took one comparison. Even in the worst case — no pair sums to the target — the two pointers together traverse the array exactly once, so the cost is `O(n)`, against the `O(n^2)` of checking every pair with two nested loops. The one-sentence intuition: **sorting buys you a direction, and a direction lets two pointers replace a double loop.**

Two pointers also power the **merge** of two sorted series (covered under sorting below), **removing duplicates in place**, **reversing a series**, and **partitioning** around a pivot. Whenever you hear "the input is sorted" or "find a pair/triple with property X," reach for two pointers first.

## Hashmaps and counting

A **hashmap** (also called a dictionary, hash table, or `dict` in Python) stores key-value pairs and gives you *average* `O(1)` lookup, insertion, and deletion. The trick is a **hash function**: it takes a key — a ticker symbol, a price, a trade ID — and computes a number that tells the structure which "bucket" to store the value in. Because computing that bucket is constant work, finding "the count for AAPL" does not require scanning anything; you jump straight to the bucket. The catch is that hashmaps have *no order* — you cannot ask a hashmap for "the smallest key" without scanning all of them — and the `O(1)` is an average; pathological inputs can degrade it, though in practice it holds.

The single most common use in interviews is **counting**: how many times does each distinct thing appear? This is a one-pass `O(n)` operation with a hashmap, and it is the building block of dozens of problems — most-frequent element, "are these two series anagrams of each other," "find the first non-repeating value," group-by aggregations.

![Each trade hashes to its ticker bucket in constant time, so one pass over the tape builds the whole frequency table.](/imgs/blogs/coding-interview-quant-data-structures-algorithms-5.png)

The figure shows the pattern: the trade tape streams in on the left, each ticker hashes to a bucket in the middle, and the count table on the right grows as you go. One pass, constant work per trade.

#### Worked example: most-traded ticker on the tape

You are handed the day's trade tape as a list of ticker symbols and must return the most-traded name. The tape is `["AAPL", "MSFT", "AAPL", "AAPL", "MSFT", "NVDA"]`.

```python
from collections import Counter

def most_traded(tape):
    counts = Counter()        # a hashmap of ticker -> count
    for ticker in tape:
        counts[ticker] += 1   # O(1) average per trade
    return counts.most_common(1)[0]   # (ticker, count)

tape = ["AAPL", "MSFT", "AAPL", "AAPL", "MSFT", "NVDA"]
print(most_traded(tape))   # ('AAPL', 3)
```

We make a single pass over the `n` trades, doing `O(1)` work each, so the whole count is `O(n)` time and `O(k)` space where `k` is the number of distinct tickers. The final answer is AAPL with 3 trades, exactly as the figure's count table shows (`AAPL -> 3`, `MSFT -> 2`, `NVDA -> 1`). The one-sentence intuition: **a hashmap turns "how many of each?" from a nested-loop search into a single linear sweep.**

A close cousin is the **seen-set** pattern: keep a hashmap (or its keys-only sibling, a *set*) of values you have already encountered, so you can answer "have I seen this before?" in `O(1)`. That single idea solves "does this series contain a duplicate," "find two numbers that sum to a target in an *unsorted* array" (store each value's complement as you go), and "the first value that repeats." When a problem smells like it needs to remember what it has seen, the answer is almost always a hashmap.

## Stacks and the monotonic stack

A **stack** is a structure with one rule: last in, first out (LIFO). You can **push** a value onto the top and **pop** the top value off, both in `O(1)`, but you can only ever touch the top. The mental picture is a stack of plates — you add and remove from the top, never the middle. Stacks model anything with nesting or "most recent first" structure: matching parentheses, undo history, the call stack of a running program.

The version that earns its keep in quant interviews is the **monotonic stack** — a stack you deliberately keep sorted (monotone), either always-increasing or always-decreasing, by popping elements that would violate the order before you push a new one. It is the standard tool for the family of "next greater / next smaller" problems, which in market-data form become "for each bar, what is the next *higher* price?" or "the next *lower* price?" Those problems look like they need, for each element, a scan forward to find the next bigger one — `O(n^2)` in the worst case. The monotonic stack does the whole thing in `O(n)`, because each element is pushed once and popped at most once.

![The stack holds only bars still waiting for a higher price, and each bar is pushed and popped exactly once, so the pass is linear.](/imgs/blogs/coding-interview-quant-data-structures-algorithms-6.png)

Here is the key idea, shown in the figure. As you scan prices left to right, the stack holds the *indices of bars that have not yet found a higher price to their right*. When a new price arrives, it resolves — and pops — every waiting bar that it exceeds, because for each of those bars, this new price *is* the next higher one. Then you push the new bar to wait its turn. A bar enters the stack once and leaves once, so across the whole array there are at most `n` pushes and `n` pops: `O(n)`.

#### Worked example: next higher price for every bar

Prices are `[5, 3, 8, 4, 9]`. For each bar we want the next price to its right that is strictly higher, or "none" if there isn't one.

```python
def next_higher(prices):
    result = [None] * len(prices)   # default: no higher price ahead
    stack = []                      # holds indices waiting for a higher price
    for i, price in enumerate(prices):
        # this price resolves every waiting bar it exceeds
        while stack and prices[stack[-1]] < price:
            j = stack.pop()
            result[j] = price       # price is the next-higher for bar j
        stack.append(i)             # i now waits for its own next-higher
    return result

print(next_higher([5, 3, 8, 4, 9]))   # -> [8, 8, 9, 9, None]
```

Trace it against the figure. We push 5, then 3 (the stack is `[5, 3]`, decreasing). The 8 arrives and pops both — 3 and 5 each get answer 8 — then 8 is pushed. The 4 arrives, does not exceed 8, so it just waits; the stack is `[8, 4]`. The 9 arrives and pops both 4 and 8 (answers 9 and 9), then waits alone with no higher price ever coming, so its answer is "none." The result is `[8, 8, 9, 9, None]`, matching the answers panel in the figure exactly. The one-sentence intuition: **a monotonic stack remembers only the candidates that could still be the answer, so each element is touched a constant number of times.**

This single pattern, lightly disguised, is "largest rectangle in a histogram," "daily temperatures / days until a warmer day," "stock span" (how many consecutive prior days had a price at or below today's), and the "trapping rain water" problem. If a question asks, for every position, about the *nearest* larger or smaller element in some direction, the monotonic stack is your first guess.

## Heaps and priority queues

A **heap** is a structure that keeps the smallest (a *min-heap*) or largest (a *max-heap*) element instantly available at its "root," while supporting fast insertion and fast removal of that extreme element — both `O(log n)`. It does *not* keep everything fully sorted, which is exactly why it is cheaper than sorting when you only care about the extremes. A heap is the natural implementation of a **priority queue**: a queue where the highest-priority item comes out first, regardless of when it went in. In finance, the order book — the list of resting buy and sell orders, each waiting at its price — is, at its core, a pair of priority queues: the best bid and best offer sit at the roots.

The signature interview use is **top-k**: given a stream or a large array, find the `k` largest (or smallest) items. Sorting the whole thing and taking the top `k` is `O(n log n)`. A heap does it in `O(n log k)`, which is a real win when `k` is small and `n` is huge — and, critically, it works on a *stream* you cannot hold in memory all at once, because the heap only ever stores `k` items.

![The heap's root is the smallest current winner, so a bigger arrival evicts it in logarithmic time, keeping only the top-k in memory.](/imgs/blogs/coding-interview-quant-data-structures-algorithms-7.png)

The figure shows the counterintuitive trick that trips up beginners: to keep the top-`k` *largest* items, you use a **min**-heap of size `k`, not a max-heap. Why? Because the thing you need fast access to is the *weakest current winner* — the smallest of your `k` keepers — so that when a new value arrives you can compare it against that weakest winner in one step. If the newcomer is bigger, it evicts the root (the weakest) and takes its place; if smaller, you discard it. The min-heap keeps the weakest winner at the root, exactly where you need it.

#### Worked example: top-3 assets by volume

A stream of `(ticker, volume)` pairs arrives. You want the three highest-volume names, but the stream is too large to hold entirely, so you keep only a size-3 min-heap.

```python
import heapq

def top_k_by_volume(stream, k=3):
    heap = []  # min-heap of (volume, ticker); root is the smallest volume
    for ticker, volume in stream:
        if len(heap) < k:
            heapq.heappush(heap, (volume, ticker))
        elif volume > heap[0][0]:      # bigger than the weakest winner?
            heapq.heapreplace(heap, (volume, ticker))  # evict root, insert new
    # heap now holds the k largest; sort once for a clean ranking
    return sorted(heap, reverse=True)

stream = [("NVDA", 42), ("AAPL", 31), ("TSLA", 28),
          ("F", 9), ("AMD", 35), ("INTC", 12)]
print(top_k_by_volume(stream))   # -> [(42, 'NVDA'), (35, 'AMD'), (31, 'AAPL')]
```

Walk the stream against the figure. NVDA (42), AAPL (31), and TSLA (28) fill the heap; the root is 28 (TSLA, the weakest). F at 9 is below 28, discarded. AMD at 35 beats the root 28, so TSLA is evicted and AMD enters; now the weakest winner is AAPL at 31. INTC at 12 is below 31, discarded. The final three are NVDA 42, AMD 35, AAPL 31 — exactly the winners in the figure. We did `O(n)` arrivals, each costing at most `O(log k)` to push or replace, for `O(n log k)` total and only `O(k)` memory. The one-sentence intuition: **for the top-k largest, a min-heap lets one comparison against the weakest winner decide every newcomer's fate.**

Heaps also power **merging k sorted lists** (put the head of each list in a heap, repeatedly pop the smallest), the **running median** (a max-heap for the lower half, a min-heap for the upper half — a classic streaming-stats question), and **Dijkstra's shortest path**. If a problem says "k largest/smallest," "stream," or "running/online" anything, think heap.

## The sliding window

The **sliding window** is a technique for computing a metric over every contiguous run of a fixed (or growing) length in an array, *without recomputing the metric from scratch each time*. The window is a sub-range `[left, right]`; as it slides one step right, exactly one element enters and one leaves, so you update the metric incrementally — add the entrant, subtract the leaver — in `O(1)` instead of re-scanning the whole window. Over the full array that is `O(n)` total, versus the `O(n × k)` of naively recomputing each window of width `k`.

This is the workhorse for **rolling statistics** in finance: the rolling sum, rolling average (moving average), rolling max or min, and rolling volatility are all sliding-window computations. Any time you hear "for every window of the last 20 days, compute X," a sliding window is the efficient answer.

![Add the entering element and subtract the leaving one so each slide is constant time, making the whole sweep linear.](/imgs/blogs/coding-interview-quant-data-structures-algorithms-8.png)

The figure makes the saving visible. The window covers days 0 to 2, with sum `10 + 12 + 11 = 33`. To slide to days 1 to 3, you do not re-add three numbers. The `$10` of day 0 leaves and the `$14` of day 3 enters, so the new sum is `33 − 10 + 14 = 37`. The overlap — the 12 and the 11 — is never touched again. That reuse of the overlap is the entire trick.

#### Worked example: a rolling 3-day sum of prices

Prices over five days are `[10, 12, 11, 14, 13]`, and you want the sum of every 3-day window.

```python
def rolling_sum(prices, k=3):
    window = sum(prices[:k])     # the first window, computed once: O(k)
    sums = [window]
    for r in range(k, len(prices)):
        window += prices[r] - prices[r - k]  # add entrant, drop leaver: O(1)
        sums.append(window)
    return sums

print(rolling_sum([10, 12, 11, 14, 13]))   # [33, 37, 38]
```

The first window sums to 33. Sliding right: `33 − 10 + 14 = 37`, then `37 − 12 + 13 = 38`. The result `[33, 37, 38]` matches the figure's first two windows (33 then 37) and continues to 38. We paid `O(k)` once to build the first window and `O(1)` for each of the remaining slides, so the total is `O(n)`. The one-sentence intuition: **a sliding window pays for the overlap once, not once per window.**

The fixed-size window above is the easy case. The *variable-size* window — grow the right edge greedily, and shrink the left edge whenever a constraint is violated — solves "longest run with at most K distinct values," "smallest window whose sum exceeds a target," and "longest profitable streak." The two-pointer machinery underneath is identical; what changes is the rule for when the left pointer advances.

## Binary search (including binary-search-on-the-answer)

**Binary search** finds a target in a *sorted* array in `O(log n)` time by repeatedly halving the search range. You look at the middle element; if it equals the target you are done; if the target is smaller you discard the entire upper half; if larger you discard the lower half. Each step throws away half of what remains, so after `log2(n)` steps the range is down to one element. For a million items that is about 20 steps instead of a million — the difference between a flick and a slog.

The plain version (find a value in a sorted list of strikes, or the insertion point for a new order's price) is worth knowing cold, including the fencepost details: use `lo` and `hi` bounds, compute `mid = lo + (hi - lo) // 2` to avoid overflow in languages where that matters, and be deliberate about whether your loop ends at `lo < hi` or `lo <= hi`. Off-by-one errors here are the most common bug in the entire technique, so interviewers watch this closely.

The version that separates strong candidates is **binary-search-on-the-answer**. Many problems do not hand you a sorted array to search, but they have a hidden monotone structure: there is some threshold answer such that everything below it fails a feasibility test and everything at or above it passes (or vice versa). When feasibility flips exactly once from "no" to "yes," the *space of candidate answers is itself sorted by the yes/no test*, so you can binary-search it — guess an answer, run a feasibility check, and halve the range based on the result.

![Feasibility flips once from no to yes, so the candidate answers are sorted and you can halve the range each step.](/imgs/blogs/coding-interview-quant-data-structures-algorithms-9.png)

The figure shows the structure: a red "infeasible" region, a green "feasible" region, and the single boundary between them that you are hunting. You bracket the boundary with `lo` and `hi`, test the midpoint, and move whichever bound the test tells you to.

#### Worked example: the minimum capital to fill every order in time

You run a smaller version of a real problem: you must allocate enough capital so that a batch of orders all clear within a deadline, and you want the *smallest* capital that works (more capital always helps — that is the monotonicity). Candidate answers run from `\$0` to `\$100` million. There is a function `can_fill(c)` that returns `True` if capital `c` is enough.

```python
def min_capital(can_fill, lo=0, hi=100):
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if can_fill(mid):
            hi = mid        # mid works -> answer is at most mid
        else:
            lo = mid + 1    # mid fails -> answer is strictly above mid
    return lo               # lo == hi: the smallest feasible capital
```

Each call to `can_fill` is the "feasibility test." Because feasibility is monotone — once `\$60` million is enough, so is `\$61` million — the candidate answers from 0 to 100 are effectively sorted "no, no, …, yes, yes," and binary search pins the boundary in about `log2(100) ≈ 7` tests instead of trying all 100 candidates. The one-sentence intuition: **if a yes/no test on the answer flips exactly once, the answers are sorted even when the input isn't — so binary-search them.**

This pattern solves a startling range of problems once you see it: "minimum eating speed to finish in H hours," "split an array into K parts minimizing the largest part's sum," "the smallest capacity ship that delivers all packages in D days," and many "minimize the maximum / maximize the minimum" phrasings. The tell is a problem that asks for the *smallest or largest value* such that *some monotone condition* holds.

## Sorting and its uses

**Sorting** arranges elements in order. You will almost never implement a sort from scratch in an interview — every language ships a good `O(n log n)` sort (`sorted()` in Python, `std::sort` in C++) — but you must know three things: that the standard sort is `O(n log n)`, that this is provably the best possible for comparison-based sorting, and *that sorting is a setup move that makes the next step cheap*. A huge fraction of array problems become easy the moment the data is sorted, because sorting unlocks two pointers, binary search, and greedy scans.

You should be able to name the workhorses. **Mergesort** splits the array in half, sorts each half, and *merges* the two sorted halves — and that merge step is itself a famous interview problem (next worked example). **Quicksort** partitions around a pivot and recurses; it is fast in practice but `O(n^2)` in the worst case. **Heapsort** uses a heap. The detail interviewers probe is **stability** — whether equal elements keep their original relative order — because in finance you often sort trades by price but need ties broken by arrival time, and only a *stable* sort preserves that secondary order for free.

#### Worked example: merge two sorted time series

You have two price feeds, each already sorted by timestamp, and must merge them into one sorted series — the merge step of mergesort, and a direct stand-in for combining two exchanges' tapes. Feed A is `[1, 4, 7, 10]` and feed B is `[2, 3, 8, 9]` (timestamps).

```python
def merge_sorted(a, b):
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:        # <= keeps it stable: ties favor feed a
            result.append(a[i]); i += 1
        else:
            result.append(b[j]); j += 1
    result.extend(a[i:])        # drain whatever remains
    result.extend(b[j:])
    return result

print(merge_sorted([1, 4, 7, 10], [2, 3, 8, 9]))   # -> [1, 2, 3, 4, 7, 8, 9, 10]
```

Two pointers, one on each feed, always take the smaller of the two current heads. Each element is appended exactly once, so the merge is `O(n + m)` — linear in the combined length, not `O((n+m) log(n+m))`, because the inputs were *already* sorted. The result `[1, 2, 3, 4, 7, 8, 9, 10]` interleaves the two feeds in timestamp order. The one-sentence intuition: **merging two sorted series is a two-pointer walk, and it is the reason mergesort is `O(n log n)` rather than `O(n^2)`.** Knowing this also unlocks "merge k sorted lists" (do it pairwise, or with a heap as noted earlier) and the merge-interval family.

## Dynamic programming: the recurring patterns

**Dynamic programming (DP)** is a technique for problems that break into overlapping subproblems, where the answer to a big problem is built from the answers to smaller ones — and you avoid recomputing those smaller answers by storing them in a table. The two ingredients are an **optimal substructure** (the best answer uses best answers to subproblems) and **overlapping subproblems** (the same subproblem recurs, so caching pays off). DP intimidates people, but in interviews it usually reduces to a handful of stock patterns, and the finance-flavored ones are among the friendliest.

The cleanest finance DP is the **best single buy/sell trade**: given a price series, find the maximum profit from buying once and selling once *later*. The naive approach checks every buy day against every later sell day — `O(n^2)`. The DP insight collapses it to one pass: as you scan, carry the **minimum price seen so far**; the best profit achievable *if you sell today* is today's price minus that running minimum, and the overall answer is the best of those daily candidates. You are reusing one computed value — the running minimum — at every step, which is DP in its leanest form.

![Each column reuses the prior column: the minimum-so-far rolls forward, and the best profit is today's price minus that minimum.](/imgs/blogs/coding-interview-quant-data-structures-algorithms-10.png)

The table in the figure fills left to right. The `min so far` row carries the smallest price up to each day; the `best profit` row is the best of "sell today, having bought at the running minimum." Each cell only needs the cell to its left, which is the signature of a one-dimensional DP.

#### Worked example: the best single trade

Prices over five days are `[7, 1, 5, 3, 6]`. We want the maximum profit from one buy followed by one later sell.

```python
def best_single_trade(prices):
    min_so_far = prices[0]
    best = 0                         # 0 = do nothing if prices only fall
    for price in prices:
        min_so_far = min(min_so_far, price)   # cheapest buy up to today
        best = max(best, price - min_so_far)  # best sale if we sell today
    return best

print(best_single_trade([7, 1, 5, 3, 6]))   # 5
```

Trace the table from the figure. Day 0: price 7, min 7, best 0. Day 1: price 1, min drops to 1, best still 0. Day 2: price 5, min 1, best `5 − 1 = 4`. Day 3: price 3, min 1, best stays 4. Day 4: price 6, min 1, best `6 − 1 = 5`. The answer is `\$5` — buy at the `\$1` low on day 1, sell at the `\$6` high on day 4 — exactly matching the figure's answer line. One pass, `O(n)` time, `O(1)` space. The one-sentence intuition: **carrying the running minimum turns an every-pair search into a single forward sweep — that is DP at its leanest.**

The other DP patterns worth having ready: the **1-D "best ending here"** family (maximum-subarray / max profit streak via Kadane's algorithm, which is the running-minimum idea's sibling for sums), **grid path counting** (number of ways through an `m × n` grid, the staple 2-D DP), the **knapsack** family (best value subject to a budget — directly a portfolio-selection flavor), and **edit distance / longest common subsequence** (string DP, less common in quant but it appears). For each, the move is the same: define what a subproblem's answer means, write the recurrence that builds it from smaller ones, decide the fill order, and identify the base cases. If you can state the recurrence in one sentence, the code writes itself.

## Graphs, BFS, and DFS (occasionally)

**Graphs** — nodes connected by edges — show up less often in quant coding rounds than the structures above, but they do appear, so know the basics. A graph models relationships: assets connected by correlation, currencies connected by exchange rates, accounts connected by transfers. The two ways to explore a graph are **breadth-first search (BFS)**, which fans out level by level using a queue (good for shortest paths in an unweighted graph), and **depth-first search (DFS)**, which plunges down one path before backtracking using a stack or recursion (good for connectivity, cycle detection, and topological ordering).

The finance-flavored graph problem you are most likely to meet is **currency arbitrage**: given exchange rates between currencies, is there a cycle of trades that returns more than you started with? That reduces to detecting a negative-weight cycle (after a logarithm transform) with the Bellman-Ford algorithm — a known set-piece. More common are the gentle ones: "how many connected groups of correlated assets are there?" (count connected components with DFS) or "is there a path of trades from currency A to currency B?" (a plain BFS/DFS reachability check). You do not need the exotic graph algorithms for most quant loops; you need clean BFS and DFS and the judgment to recognize when a problem is secretly a graph.

## In the interview room

Time to put it together. The five problems below are the kind a quant interviewer poses — each finance-flavored, each solved in full the way you would on a whiteboard, with the spoken reasoning, the complexity, and the edge cases. Read each one as a script for how to *talk through* a solution, not just how to write it. The discipline that earns marks is: **restate, name the technique, state the complexity target, code it, then test it on a small case and the edges.**

#### Worked example: maximum drawdown of a price series

*Problem.* Given a series of daily prices, compute the **maximum drawdown** — the largest peak-to-trough percentage drop, where the peak must come before the trough. This is the headline risk number for any strategy: it answers "what is the worst loss I would have suffered if I bought at the worst possible moment and held to the bottom?"

*Restate and name the technique.* For each day, the worst drop ending today is `(peak so far − today) / peak so far`. So if I carry the **running maximum** of the prices, I can compute today's drawdown in `O(1)` and take the worst over all days. This is the running-extremum pattern — the same shape as the best-single-trade DP, mirrored. Target: `O(n)` time, `O(1)` space.

![Track the highest price seen so far, and the worst gap below it as you scan left to right is the maximum drawdown.](/imgs/blogs/coding-interview-quant-data-structures-algorithms-11.png)

The figure shows the mechanism: the dashed line is the running peak, and the shaded band is the worst gap between that peak and a later price.

```python
def max_drawdown(prices):
    if not prices:
        return 0.0                  # edge case: empty series
    peak = prices[0]
    mdd = 0.0
    for price in prices:
        peak = max(peak, price)     # highest price seen up to today
        drawdown = (peak - price) / peak
        mdd = max(mdd, drawdown)
    return mdd

series = [78, 88, 100, 80, 62, 60, 72, 84, 90]
print(round(max_drawdown(series) * 100, 1))   # 40.0
```

*Test it.* On the series in the figure, the running peak reaches `\$100`, and the lowest price *after* that peak is `\$60`, giving a drawdown of `(100 − 60) / 100 = 40%`. The code returns `40.0`. Check the edges: an empty series returns 0; a series that only rises returns 0 (no drawdown); a single element returns 0. The peak-before-trough requirement is handled for free, because `peak` only ever reflects prices we have already seen. One pass, `O(n)` time, `O(1)` space. If the interviewer asks for the *dollar* drawdown instead of percentage, drop the division; if they ask for the longest drawdown *duration*, track the index of the running peak too.

#### Worked example: best single buy/sell trade for maximum profit

*Problem.* Given a price series, find the maximum profit from exactly one buy and one later sell. If no profitable trade exists, return 0 (you simply do not trade).

*Restate and name the technique.* This is the running-minimum DP from the section above — the mirror image of max drawdown. Carry the cheapest price seen so far; the best profit if I sell today is today minus that minimum; the answer is the best over all days. Target `O(n)` time, `O(1)` space. I will reuse the `best_single_trade` function verbatim, and the key thing to *say out loud* is why the naive `O(n^2)` double loop is unnecessary: I never need to look back at all earlier buy days, because the running minimum already summarizes them.

```python
def best_single_trade(prices):
    min_so_far = float("inf")
    best = 0
    for price in prices:
        min_so_far = min(min_so_far, price)
        best = max(best, price - min_so_far)
    return best

print(best_single_trade([7, 1, 5, 3, 6]))   # 5  (buy 1, sell 6)
print(best_single_trade([9, 7, 4, 1]))       # 0  (only falls -> don't trade)
```

*Test it.* On `[7, 1, 5, 3, 6]` the answer is `\$5`. On a strictly falling series `[9, 7, 4, 1]`, every `price − min_so_far` is `≤ 0`, so `best` stays 0 — we correctly decline to trade. The edge cases (empty, single element) both give 0. A strong follow-up is "what if you can make *two* trades?" — that extends to a small constant-state DP tracking the best profit after the first buy, first sell, second buy, and second sell; mentioning that you see the extension signals depth even if you do not code it. The companion [mini-backtest coding challenge](/blog/trading/quantitative-finance/mini-backtest-coding-challenge-quant-interviews) takes this idea all the way to a small strategy simulator.

#### Worked example: merge two sorted trade tapes

*Problem.* Two exchanges each give you a tape of trades already sorted by timestamp. Merge them into a single timestamp-sorted tape. Ties (same timestamp) should keep exchange A's trade first.

*Restate and name the technique.* This is the merge step of mergesort: a two-pointer walk that repeatedly takes the earlier of the two current heads, with `<=` to keep it stable so ties favor A. Target `O(n + m)` time. The thing to emphasize is that because both inputs are *already sorted*, we beat the `O((n+m) log(n+m))` of concatenating and re-sorting.

```python
def merge_tapes(a, b):
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][0] <= b[j][0]:       # compare timestamps; <= = stable, A wins ties
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:]); out.extend(b[j:])  # drain the remainder
    return out

a = [(1, "A-buy"), (4, "A-sell"), (7, "A-buy")]
b = [(2, "B-sell"), (4, "B-buy"), (8, "B-sell")]
print([t for t, _ in merge_tapes(a, b)])   # [1, 2, 4, 4, 7, 8]
```

*Test it.* The merged timestamps are `[1, 2, 4, 4, 7, 8]`, and at the tied timestamp 4, A's trade precedes B's — the stability requirement is met by the `<=`. Check the edges: if one tape is empty, the `extend` calls drain the other; if both are empty, the result is empty. One pass over each tape, `O(n + m)` time and `O(n + m)` output space. The natural follow-up — "merge `k` tapes from `k` exchanges" — is where the heap returns: keep the head of each tape in a min-heap and repeatedly pop the earliest, for `O(N log k)` over `N` total trades.

#### Worked example: top-k assets by dollar volume from a stream

*Problem.* A live feed emits `(ticker, dollar_volume)` updates all day. At any moment you must be able to report the `k` highest-volume names. The feed is far too large to store.

*Restate and name the technique.* Top-k over a stream with bounded memory is the size-`k` min-heap. Keep only `k` items; the root is the weakest winner; each new value either beats the root (evict and insert) or is discarded. Target `O(n log k)` time and `O(k)` space — the `O(k)` space is the whole point, since we cannot hold the stream.

```python
import heapq

class TopK:
    def __init__(self, k):
        self.k = k
        self.heap = []   # min-heap of (volume, ticker)

    def update(self, ticker, volume):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (volume, ticker))
        elif volume > self.heap[0][0]:        # beats the weakest winner?
            heapq.heapreplace(self.heap, (volume, ticker))

    def top(self):
        return sorted(self.heap, reverse=True)

tk = TopK(3)
for t, v in [("NVDA", 42), ("AAPL", 31), ("TSLA", 28),
             ("F", 9), ("AMD", 35), ("INTC", 12)]:
    tk.update(t, v)
print(tk.top())   # [(42, 'NVDA'), (35, 'AMD'), (31, 'AAPL')]
```

*Test it.* The final top-3 are NVDA 42, AMD 35, AAPL 31, matching the heap worked example earlier. Edge cases: if fewer than `k` items have arrived, `top()` returns all of them; duplicate tickers would need a small extra hashmap if you want one entry per ticker (a good thing to flag). The interviewer's likely follow-up is "what if a ticker's volume *updates* rather than arrives fresh?" — then you need a hashmap from ticker to its heap entry plus lazy deletion, which is the realistic order-book-maintenance version and worth describing even if you keep the code simple.

#### Worked example: next higher price (monotonic stack)

*Problem.* For each bar in a price series, report the next bar to its right with a strictly higher price, or "none." This is the "stock span" / "next greater element" family and a genuine market-microstructure question (when does price next break out above the current level?).

*Restate and name the technique.* Monotonic stack. Keep a stack of indices whose next-higher price is still unknown; each new price resolves and pops every waiting bar it exceeds. Each bar is pushed once and popped once, so the whole thing is `O(n)` despite looking like it needs a forward scan per bar. I will reuse the `next_higher` function from the monotonic-stack section.

```python
def next_higher(prices):
    result = [None] * len(prices)
    stack = []  # indices of bars still waiting for a higher price
    for i, price in enumerate(prices):
        while stack and prices[stack[-1]] < price:
            result[stack.pop()] = price
        stack.append(i)
    return result

print(next_higher([5, 3, 8, 4, 9]))   # [8, 8, 9, 9, None]
```

*Test it.* The answer `[8, 8, 9, 9, None]` matches the monotonic-stack figure exactly: 5 and 3 both resolve to 8, 8 and 4 both resolve to 9, and 9 has no higher price ahead. Edge cases: a strictly increasing series resolves each bar immediately on the next bar; a strictly decreasing series leaves everything as "none" and the stack holds all `n` indices at the end (still `O(n)` total work, because there were `n` pushes and zero pops until the end). The complexity claim — "each element is pushed and popped at most once, so `O(n)`" — is exactly the kind of amortized-analysis sentence interviewers want to hear you say.

## Common misconceptions

A few beliefs cost candidates offers. Here are the ones worth unlearning.

**"The fastest, cleverest solution wins."** No. A correct, clearly-explained `O(n log n)` solution beats a buggy `O(n)` one every time, and it beats a correct `O(n)` one that you cannot explain. Interviewers are hiring someone whose code their teammates will read and trust. Reach for the simplest correct approach first, get it working and tested, *then* optimize if there is time and the interviewer wants it. Premature cleverness that breaks under questioning is the classic way to fail a round you could have passed.

**"I should jump straight to coding."** The strongest candidates spend the first few minutes *not* coding: restating the problem, pinning down assumptions ("are prices guaranteed positive? can the series be empty? are there duplicate timestamps?"), naming the technique, and stating the target complexity out loud. This is not stalling; it is the part of the job that prevents you from building the wrong thing. Silent fast typing that produces a subtly wrong answer scores worse than a slower, narrated, correct one.

**"Big-O constants never matter."** In analysis we drop constants, and for choosing between complexity *classes* that is right. But on a real desk, a `2n` algorithm versus a `10n` algorithm is a 5x difference in a hot loop, and memory layout (cache-friendliness) can swamp the theoretical class entirely. Interviews mostly test the class, but if you are interviewing for a low-latency role, expect questions where the constant factor and the hardware are the whole point.

**"Sorting is always too slow."** Candidates sometimes contort themselves to avoid an `O(n log n)` sort, reaching for a buggy `O(n)` hashmap trick when sorting would have made the problem trivial and the bug impossible. Sorting is cheap and it unlocks two pointers, binary search, and greedy scans. If sorting turns a hard problem into an easy one and the input is not enormous, sort. State the cost (`O(n log n)`) and move on.

**"Python's built-ins are free."** They are not. `list.insert(0, x)` and `del list[0]` are `O(n)` (everything shifts); `x in some_list` is `O(n)` (a scan), while `x in some_set` is `O(1)`; `list.pop(0)` is `O(n)` but `collections.deque.popleft()` is `O(1)`. Reaching for a list when you needed a set or a deque silently turns your `O(n)` solution into `O(n^2)`. Knowing the cost of your language's standard operations is part of knowing your complexity.

**"Recursion and dynamic programming are different worlds."** DP is usually just recursion with the repeated subproblems cached (memoization), or that same recursion rewritten bottom-up as a table fill. If you can write the recursive relationship — "the answer for `n` in terms of the answer for `n−1`" — you have already done the hard part of the DP; adding the cache is mechanical.

## How it shows up on a real desk

These are not artificial puzzles. The exact structures appear, every day, in the systems quant developers and researchers build. Here is where they live.

**The order book is a pair of priority queues.** A matching engine maintains all resting buy orders and all resting sell orders, and the central operation is "what is the best bid and best offer right now?" — the extremes. That is a heap's home turf. Real matching engines use more specialized structures (price-level arrays, intrusive linked lists) for nanosecond access, but the *conceptual* model an interviewer probes is the priority queue, and "design a simplified order book" is a staple onsite question that tests exactly the heap and hashmap material above.

**Backtests are sliding windows and running statistics over price arrays.** Every moving average, every rolling-volatility estimate, every momentum signal is a sliding-window computation over a return series. Doing them the naive `O(n × k)` way instead of the `O(n)` incremental way is the difference between a research loop that iterates in seconds and one that iterates overnight — and the speed of that loop directly bounds how many ideas a researcher can test. The post on [backtesting done right](/blog/trading/quantitative-finance/backtesting-done-right-quant-research) shows how these windows compose into a full simulation, and why getting the windowing wrong silently leaks future information into the past.

**Risk reports are running extrema.** Maximum drawdown, the running high-water mark, the worst N-day loss — the daily risk pack a desk produces is built from the running-maximum and running-minimum passes you saw above, run across hundreds of strategies. The `O(n)` formulation matters because these run on long histories across a whole book, repeatedly.

**Time-series alignment is the merge step.** Combining data from multiple exchanges, joining a price feed to a news feed, or aligning two instruments' tapes onto a common clock is, at its core, the two-pointer merge of sorted series. When the feeds are out of order, you sort first (`O(n log n)`) and then merge — and you handle the ties and gaps that the clean textbook version glosses over, which is where real engineering judgment shows.

**Frequency and dedup work is hashmaps everywhere.** Counting trades per symbol, detecting duplicate messages on a noisy feed, building a symbol-to-id mapping, computing a group-by aggregation across millions of rows — all hashmaps, all `O(n)`. The "seen-set" pattern in particular shows up constantly in feed handlers that must drop replayed or duplicated messages.

**Threshold-finding is binary-search-on-the-answer.** Sizing a position so that expected risk hits a target, finding the implied volatility that reprices an option to its market price, calibrating a parameter so a model matches a quote — these are monotone-feasibility searches, and binary search on the answer is the workhorse. The implied-vol root-find is the canonical example: price rises monotonically in volatility, so you binary-search (or Newton-step) the volatility that reproduces the observed price.

## When this matters and further reading

If you are preparing for a quant coding loop, the takeaway is to stop grinding random problems and start grinding *techniques*. Pick one structure at a time — arrays and two pointers, then hashmaps, then the monotonic stack, then heaps, then sliding windows, then binary search, then DP — and for each, solve five to ten problems until the pattern is automatic and you can implement it cleanly without thinking about syntax. Then practice the meta-skill that actually gets graded: narrating your reasoning out loud while you code, stating the complexity, and testing on the edge cases. Mock interviews where you must talk through a problem in real time are worth more than another hundred silent solves.

This post is the map; each tool has companions that go deeper. For the probability brain-teasers that fill the *other* half of a quant loop, see [the classic quant probability problem set](/blog/trading/quantitative-finance/classic-quant-probability-problems). For the mental-arithmetic speed rounds that several firms run, see [mental math and arithmetic speed](/blog/trading/quantitative-finance/mental-math-arithmetic-speed-quant-interviews). For the language-specific systems track, see [C++ for low-latency quant interviews](/blog/trading/quantitative-finance/cpp-for-low-latency-quant-interviews). And to see these data structures assembled into a small end-to-end strategy, work through the [mini-backtest coding challenge](/blog/trading/quantitative-finance/mini-backtest-coding-challenge-quant-interviews).

![Pick the structure whose worst operation is the one your problem performs most often, reading speed off the cheat-sheet by color.](/imgs/blogs/coding-interview-quant-data-structures-algorithms-12.png)

The cheat-sheet above is the one table to internalize before you walk in. Every structure trades one operation against another: the array gives you instant access but slow search; the hashmap gives you instant search and insert but no order; the heap gives you the extreme cheaply but nothing else; a balanced tree gives you everything at `O(log n)`; the stack and queue give you `O(1)` at the ends only. There is no universally best structure — there is only the structure whose *fast* operations match the ones your problem performs most, and whose *slow* operations your problem rarely touches. Choosing it correctly, and saying why out loud, is most of what a quant coding interview is testing.

*This article is educational, not financial or career advice. The code is illustrative; production trading systems handle far more edge cases, concurrency, and numerical care than these teaching snippets show.*
