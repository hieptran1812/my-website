---
title: "Faster Startup: Imports, Packaging, and Alternative Interpreters"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Most of this series fights the time your code spends running. This post fights the time before it runs at all: process startup, import cost, packaging compiled wheels, and knowing exactly when PyPy is the right call and when it does nothing."
tags:
  [
    "python",
    "performance",
    "optimization",
    "startup",
    "imports",
    "packaging",
    "pypy",
    "profiling",
  ]
category: "software-development"
subcategory: "Python Performance"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/faster-startup-imports-packaging-and-alternative-interpreters-1.png"
---

There is a kind of slowness that no profiler in the rest of this series will ever show you, because by the time your profiler attaches, the expensive part is already over. I learned this the hard way on a command-line tool that the data team used dozens of times a day. It was a thin wrapper: parse some flags, maybe read a file, print a table. Trivial work. And yet every invocation sat there for a second and a half before it did *anything* — before it even printed the help text for `--help`. People had started muttering that "Python is just slow." They were wrong, but understandably so. The CPU work in that tool took about twenty milliseconds. The other 1,480 milliseconds were spent *importing*. The tool did `import pandas` at the top of the file, pandas dragged in NumPy and a pile of timezone and date libraries, and all of that ran on every single invocation — including the ones that only printed `--help` and never touched a dataframe.

That is the topic of this post: the *other* latency. Not how fast your hot loop runs, but how long the process takes to start, to import its dependencies, and to reach your first useful line of code. For a long-running service this cost is paid once and amortized into oblivion — who cares if the web server takes two seconds to boot if it then runs for three weeks? But an enormous amount of real Python is *not* long-running. CLIs run for a fraction of a second and exit. Serverless functions cold-start a fresh interpreter per request when traffic is spiky. Short-lived batch workers spawn, do one unit of work, and die. Test suites import your entire package on every `pytest` invocation, sometimes hundreds of times a day across a team. For all of these, startup time is not a rounding error. It *is* the runtime.

![before and after comparison of a command line tool with heavy top level imports taking 1.4 seconds versus lazy and trimmed imports taking 0.2 seconds on an 8 core Linux box](/imgs/blogs/faster-startup-imports-packaging-and-alternative-interpreters-1.png)

By the end of this post you will be able to do four concrete things. First, *measure* startup honestly — read the import-time tree with `python -X importtime` and find the one heavy import that costs you a second. Second, *fix* it with lazy imports (deferred imports inside functions, a module-level `__getattr__`, and the `TYPE_CHECKING` trick) and by trimming dependency bloat, with a measured before-and-after. Third, *package* fast code correctly — ship compiled extensions as wheels so your users never compile anything, and understand why modern tooling like `uv` installs them so much faster. Fourth, decide when an *alternative interpreter* like PyPy is the right tool: its tracing JIT can give enormous wins on long-running pure-Python loops, but it starts slowly, fights with C extensions, and does nothing at all for code that already lives in NumPy's C core. The motto of this whole series is *don't guess, measure; rewrite the hot 1% in native, not 100%; prove the win with a number*. Here we apply it to a part of the timeline most people never measure at all.

If you have not read the series intro, [Why Python Is Slow (and What "Fast" Actually Means)](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) sets up the leverage ladder and the optimization loop that frames everything below. We also lean on [The CPython Execution Model: Bytecode and the Eval Loop](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop), because import time is mostly *bytecode compilation and module execution*, and a tracing JIT is the conceptual opposite of CPython's adaptive interpreter. And because every claim here is a timing claim, [Benchmarking Python Correctly: timeit Pitfalls and Statistics](/blog/software-development/python-performance/benchmarking-python-correctly-timeit-pitfalls-and-statistics) is the discipline that keeps these startup numbers honest.

All numbers below come from a specific, stated machine so they are reproducible and honest: an 8-core x86-64 Linux box (a typical cloud instance) with 16 GB of RAM, running CPython 3.12 unless I say otherwise. An Apple M2 lands in the same ballpark, usually a little faster per core. Where I compare 3.10 to 3.11 to 3.12 I ran each. When a figure is a typical range rather than one measurement, I say so. We never fabricate a precise number — a made-up benchmark is worse than none.

## Why startup is a per-process tax, and why that dominates short-lived code

Here is the core fact that makes this whole topic matter, stated as plainly as I can: **import cost is paid once per process start, and a CLI starts a fresh process every time you run it.** A long-lived service imports its world once during boot and then serves millions of requests on that warm interpreter; the import cost divided across those requests is effectively zero. A CLI does the opposite. It boots, does a sliver of work, and exits. There is no amortization. Whatever you pay at import, you pay *in full, every single invocation*.

Let me make that quantitative, because the asymmetry is the whole argument. Suppose your tool's actual work — the part you would profile and optimize with the rest of this series — takes $W$ seconds, and your startup-plus-import cost is $S$ seconds. The total wall-clock time a user experiences is $T = S + W$. For a long-running service that handles $N$ requests on one process, the *per-request* startup cost is $S/N$, which goes to zero as $N$ grows: a two-second boot over ten million requests is 0.2 microseconds per request, utterly invisible. But for a CLI, $N = 1$ by construction. Each run is its own process. So the per-run startup cost is just $S$, undivided and undiminished. If $S = 1.4$ s and $W = 0.02$ s, then startup is $S / (S + W) = 1.4 / 1.42 \approx 98.6\%$ of the user-visible time. You could make the actual work *infinitely* fast — rewrite it in hand-tuned assembly — and the tool would still feel like it takes a second and a half. Amdahl's law has a brutal corollary here: when one component is 98.6% of the time, the best possible speedup from fixing *everything else* is about 1.4%.

This is why startup deserves its own post and its own mindset. It is the part of the program where "do less work" means "import fewer things," not "vectorize the loop." Figure 1 above shows the shape of the win we are chasing: the same CLI, before and after, where the entire difference is *what gets imported when*.

The categories of code where this bites hardest are worth naming explicitly, because if your code is none of them you can mostly stop reading and go optimize a hot loop instead:

- **Command-line tools.** Anything a human runs interactively and waits on. A linter, a formatter, a deploy script, a data-export utility. Humans notice 200 ms and *hate* 1.5 s. The fastest CLI tools in the ecosystem — `ruff`, `uv` — are obsessive about startup precisely because that is what users feel.
- **Serverless / function-as-a-service.** A cold start spins up a fresh runtime and imports your handler module from scratch. Under bursty traffic you pay this constantly. A 1.5 s import on a function billed per 100 ms is both a latency problem (the user waits) and a money problem (you pay for the import time on every cold invocation).
- **Short-lived workers and cron jobs.** A queue consumer that forks a process per task, a cron job that runs every minute, a map-reduce mapper that launches thousands of short Python processes. If each one imports pandas, you have multiplied a one-second cost by your fan-out.
- **Test suites.** `pytest` imports your whole package (and its dependencies) on each run. If your editor or CI runs a subset on every save, the per-run import cost is paid hundreds of times a day. A suite that *runs* in 4 seconds but *imports* for 3 is a suite where you are mostly waiting on imports.

For everything else — a Jupyter kernel you start once and use all day, a web server, a long training run, a streaming pipeline — startup genuinely does not matter and you should not spend a minute on it. Knowing *which bucket you are in* is the first and most important decision, and it is the root of the decision tree we will draw at the end. Measure before you assume.

## Measuring startup: `-X importtime` is your X-ray

You cannot fix what you cannot see, and the beautiful thing about import cost is that CPython has a built-in, zero-dependency tool that shows you exactly where every millisecond went: the `-X importtime` flag. It instruments the import machinery and prints, to stderr, a line for every module imported, with two timing columns: the *self* time (time spent in that module's own top-level code) and the *cumulative* time (self plus everything it imported in turn). It even indents to show the import *tree*. This is the single most valuable thing in this entire post, so let us use it on a real example.

Here is a tiny CLI that reproduces the disease from the intro:

```python
# mytool.py — a CLI that imports too eagerly
import argparse
import pandas as pd          # the expensive top-level import
import requests              # also not cheap

def main():
    parser = argparse.ArgumentParser(prog="mytool")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("csv", nargs="?")
    args = parser.parse_args()
    if args.version:
        print("mytool 1.0")
        return
    df = pd.read_csv(args.csv)
    print(df.describe())

if __name__ == "__main__":
    main()
```

Now run it under the importtime flag. Note that `-X importtime` writes to stderr, so we redirect it and sort by self time to find the worst offenders:

```bash
python -X importtime -c "import mytool" 2> importtime.log
sort -t '|' -k 2 -n -r importtime.log | head -20
```

The raw output is verbose, but the shape is unmistakable. Each line looks like this (columns are: the literal word `import time`, self time in microseconds, cumulative time in microseconds, then the indented module name):

```bash
import time: self [us] | cumulative | imported package
import time:       523 |     523     |   numpy._globals
import time:      1840 |   95210     | numpy
import time:       210 |     210     |   pytz.tzinfo
import time:       990 |   58800     | pytz
import time:      2300 |  520400     | pandas
import time:       410 |   40100     | requests
import time:       180 |    1900     | mytool
```

Read the cumulative column for the top-level packages and the story writes itself. `pandas` cost about 520 ms cumulative. Inside that, `numpy` was about 95 ms and `pytz` plus `dateutil` were tens of milliseconds more. `requests` added another 40 ms (it pulls in `urllib3`, `certifi`, `charset_normalizer`). Your own `mytool` module — your actual code — cost under 2 ms. The interpreter itself starting up costs another 20 to 30 ms before any of this even begins. Add it all up and you are at roughly 1.4 seconds, and **every byte of that is paid on `mytool --version`, which uses none of it.**

![graph showing the import tree where the cli module pulls in pandas which pulls in numpy and pytz and requests pulls in urllib3 each labeled with its millisecond cost summing to a 1.4 second total](/imgs/blogs/faster-startup-imports-packaging-and-alternative-interpreters-2.png)

Figure 2 draws this tree. The key insight it encodes: one import statement at the top of your file is never *one* import. It is the root of a subtree, and you are billed for the whole subtree, transitively. `import pandas` is really "import pandas and numpy and pytz and dateutil and a dozen C extensions." The importtime flag is the only honest way to see the true cost, because the cost is mostly *not* in the module you named.

A second, lighter-weight probe is worth keeping in your muscle memory for quick checks. To time the bare interpreter and a single import without any tooling overhead, use the shell's own timer:

```bash
# bare interpreter startup, no user imports
python -c "pass"
# time it properly with the OS
/usr/bin/time -v python -c "pass" 2>&1 | grep "wall clock"

# the cost of one import, isolated
python -X importtime -c "import pandas" 2>&1 | tail -1
```

And to ask Python itself what is already loaded (useful for spotting what a framework dragged in behind your back), `sys.modules` is the registry of every module imported so far:

```python
import sys
# how many modules did just starting up + importing my package load?
print(len(sys.modules))
# which heavy ones are present?
print([m for m in sys.modules if m in ("pandas", "numpy", "scipy", "torch")])
```

The honest-measurement caveats from [the benchmarking post](/blog/software-development/python-performance/benchmarking-python-correctly-timeit-pitfalls-and-statistics) apply with full force here. The *first* import of a module is slower than later ones in the same process because Python must compile the source to bytecode (or read and validate the `.pyc` cache) and run the module body; subsequent imports in the same process are a dict lookup in `sys.modules` and effectively free. So when you measure startup, you must measure a *fresh process* every time — `python -X importtime -c "import x"` does exactly that. Do not measure import time inside a REPL where the module is already loaded; you will measure nothing. Run it several times and take the median, because the first run after a file change pays the compile cost and disk I/O that warm runs do not. And remember the OS file cache: the very first import after a reboot reads the `.pyc` files from cold disk; warm runs read them from page cache. State which you measured.

#### Worked example: finding the 1.4-second import

On the 8-core Linux box, CPython 3.12, I ran the disassembly of the tool's startup three times after warming the file cache and took the median.

| Stage | Cumulative time | What it is |
| --- | --- | --- |
| Interpreter init | ~28 ms | `python -c "pass"`, the floor |
| `import requests` | ~40 ms self subtree | urllib3, certifi, idna, charset_normalizer |
| `import numpy` | ~95 ms self subtree | the C extension load + module body |
| `import pandas` | ~520 ms cumulative | pulls numpy + pytz + dateutil + its own C exts |
| `import mytool` (your code) | ~2 ms | argparse + your 12 lines |
| **Total to first line** | **~1.41 s** | the user-visible startup |

The diagnosis is now unambiguous. Pandas is 92% of the startup. Your code is 0.1%. No micro-optimization of `main()` will ever matter. The fix is not to make pandas faster — you cannot — it is to *not import pandas when you do not need it*. That is the next section.

## The anatomy of an import: where the milliseconds actually go

Before we start fixing, it is worth being precise about *what* a single `import` statement does, because the cost has several distinct components and the right fix depends on which one dominates. When CPython executes `import pandas`, the import system (driven by `importlib`, which is itself written mostly in Python and frozen into the binary) walks through a fixed sequence of steps. Understanding the sequence tells you exactly why the cost is what it is.

First, the **`sys.modules` check.** Python looks up `"pandas"` in the `sys.modules` dictionary. If it is already there — because something imported it earlier in this process — the import is *done*: it binds the existing module object to the name and returns. This is the dict lookup that makes the second, third, and millionth import in a process effectively free, and it is the entire reason lazy imports work. The cost is one hash and one dict probe, on the order of tens of nanoseconds.

If the module is *not* already loaded, the expensive path begins. Python runs the **finders and loaders**: a chain of objects on `sys.meta_path` that locate the module. For a normal source module the path-based finder walks `sys.path` directory by directory looking for `pandas/__init__.py` (or a compiled extension, or a namespace package). This file-system search is real work — every entry on `sys.path` is a directory to stat, and a long `sys.path` (common in fat virtualenvs with many packages) means many filesystem lookups per import. This is why deeply nested packages with many small submodules can have surprising import cost: it is not just the code, it is the directory walking and file stats.

Once the source is located, Python checks the **bytecode cache**. It looks for a `__pycache__/pandas.cpython-312.pyc` matching the source's modification time (or hash). If the `.pyc` is valid, Python loads the pre-compiled bytecode directly — fast. If it is missing or stale, Python must **parse and compile** the source: tokenize the text, build an AST, and compile the AST to a code object. For a large module this compilation is tens of milliseconds of pure CPU work, and it is exactly what the `.pyc` cache exists to avoid on subsequent runs.

Then comes the part you cannot cache away: **executing the module body.** Importing a module *runs its top-level code* — every assignment, every class and function definition, every nested `import`, and crucially every bit of import-time initialization the library does (building lookup tables, registering types, loading its own C extensions). For a library like pandas, the module body is enormous and does a lot of setup work: this execution, plus the cascade of *its* imports, is where most of that 520 ms actually lives. No bytecode cache helps here, because the cache stores compiled code, not the *result* of running it. This is the irreducible floor of importing a heavy library, and the only way to avoid it is to not run it — which is what lazy imports achieve.

Finally, for a module backed by a **compiled C extension** (a `.so` file, like much of NumPy and pandas), importing also triggers `dlopen` — the operating system loads the shared library into the process's address space, resolves its symbols, and runs its module-init function. Loading a large shared library is not free; it is disk I/O (cold) or page-cache reads (warm) plus dynamic-linker work. NumPy's import cost is partly this `dlopen` of its compiled core.

So the ~520 ms of `import pandas` decomposes, roughly, into: directory/file stats (small), bytecode compile *or* `.pyc` load (tens of ms, cached after first run), `dlopen` of compiled extensions (tens of ms), and — the dominant term — *running the module bodies* of pandas and everything it imports (the bulk). This decomposition is why the levers in this post target different layers: the `.pyc` cache attacks the compile step, frozen modules attack the stdlib boot, lazy imports attack the "run the module body" step by simply not running it, and trimming attacks the whole subtree by removing the dependency. Knowing which layer your cost lives in tells you which lever to pull. For a heavy third-party library, the cost is overwhelmingly in step "run the body," and only lazy-loading or trimming touches it.

```python
# See the import machinery's decision points yourself
import sys, importlib.util
spec = importlib.util.find_spec("pandas")   # runs finders, locates the module
print(spec.origin)          # path to pandas/__init__.py
print(spec.cached)          # path to the .pyc bytecode cache
print("pandas" in sys.modules)   # False until you actually import it
```

## Fix #1: lazy imports — defer the heavy module until first use

The simplest, highest-leverage startup fix is also the oldest trick in the book: **move the expensive import out of the module top level and into the function that actually needs it.** Python imports are statements, not declarations; an `import` inside a function runs the first time that function is called, and thanks to the `sys.modules` cache, every later call is a near-free dict lookup. So if only one subcommand of your CLI needs pandas, only that subcommand should pay for it.

Here is the refactored tool. The structural change is two lines moved:

```python
# mytool.py — pandas deferred into the command that needs it
import argparse

def cmd_describe(path):
    import pandas as pd          # deferred: only paid on the describe path
    df = pd.read_csv(path)
    print(df.describe())

def cmd_version():
    print("mytool 1.0")          # pays NOTHING beyond interpreter + argparse

def main():
    parser = argparse.ArgumentParser(prog="mytool")
    parser.add_argument("--version", action="store_true")
    parser.add_argument("csv", nargs="?")
    args = parser.parse_args()
    if args.version:
        cmd_version()
    elif args.csv:
        cmd_describe(args.csv)

if __name__ == "__main__":
    main()
```

Now `mytool --version` imports `argparse` (a few hundred microseconds) and nothing else. The pandas tax is charged *only* on the path that reads a CSV, and even there it is paid exactly once per process. Run the importtime check again on the `--version` path and pandas simply does not appear, because it is never imported.

![before and after diagram contrasting import pandas at module top where every command pays 520 milliseconds versus import pandas inside the function where the version flag is instant](/imgs/blogs/faster-startup-imports-packaging-and-alternative-interpreters-4.png)

Figure 4 shows the structural difference. The "before" column pays the heavy import unconditionally; the "after" column pays it only on the path that uses it. This is precisely the leverage-ladder principle of *doing less work*, applied to the import phase: the cheapest work is work you never do.

Deferred-inside-the-function is the bluntest form of lazy import. There are two more refined patterns worth knowing, because the inline `import` inside a function is slightly ugly and easy to forget.

**Module-level `__getattr__` (PEP 562).** Since Python 3.7, a module can define a module-level `__getattr__(name)` function that is called whenever someone accesses an attribute the module does not already have. This lets you expose a heavy submodule *as if* it were imported at the top, but defer the actual import until the first time anyone touches it. This is exactly how the SciPy and NumPy ecosystems implement their own lazy top-level namespaces.

```python
# mypackage/__init__.py — lazily expose a heavy submodule
import importlib

_LAZY = {"plotting": "mypackage._plotting"}   # name -> real module path

def __getattr__(name):
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name])
        globals()[name] = module       # cache so __getattr__ runs once
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(list(globals()) + list(_LAZY))
```

Now `import mypackage` is cheap, and `mypackage.plotting.plot(...)` triggers the heavy import only on first access. The line `globals()[name] = module` is the important one: it writes the resolved module back into the package namespace so `__getattr__` fires exactly once, after which it is an ordinary attribute lookup. The standard library's own `importlib.util.LazyLoader` and the third-party `lazy_loader` package (used by scikit-image and others) automate this whole pattern if you want to lazy-load an entire public API.

**The `TYPE_CHECKING` trick.** Type annotations frequently reference types from heavy libraries — `def load(path: str) -> "pandas.DataFrame":`. If you `import pandas` at the top just to name the type, you have paid 520 ms for a *string in a function signature*. The fix is `typing.TYPE_CHECKING`, a constant that is `False` at runtime but `True` when a static type checker like mypy or pyright analyzes the file. Combined with `from __future__ import annotations` (which makes all annotations strings that are never evaluated at runtime), you get full type safety with zero runtime import cost:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd        # only mypy/pyright ever runs this branch

def describe(df: pd.DataFrame) -> None:   # annotation is a string at runtime
    print(df.describe())
```

At runtime, the `if TYPE_CHECKING:` block is dead code — `TYPE_CHECKING` is `False`, so pandas is never imported. The type checker, which treats it as `True`, sees the import and validates `pd.DataFrame` correctly. You get the editor autocomplete and the static guarantee without paying a millisecond. This is one of the highest-value, lowest-effort patterns in a large codebase: it is common for a package to import half its heavy dependencies purely for annotations, and `TYPE_CHECKING` deletes all of it from the startup path.

A few honest caveats on lazy imports, because they are not free lunch in every situation:

- **They move latency, they do not delete it.** The first call to `cmd_describe` is now *slower* by the full pandas import cost — you pay it on first use instead of at startup. This is exactly right for a CLI (the user who runs `--version` should not subsidize the user who reads a CSV), but for a long-running service where every code path eventually runs, eager imports at boot are usually *better*: pay it once during warm-up, never during a request. Match the strategy to the workload.
- **They can hide import errors.** An `import` that fails at module top fails loudly at startup. An import deferred into a rarely-hit function can fail in production, at the worst time, on the one path nobody tested. If a dependency is mandatory, importing it eagerly so it fails fast can be the *correct* call. Lazy-import the optional and the heavy; eager-import the mandatory and the cheap.
- **Inline imports inside a hot loop are a trap.** The `sys.modules` lookup is cheap but not zero — it is a dict hit plus the bytecode to execute the `IMPORT_NAME` opcode. Put the deferred import at the *top of the function*, not inside its inner loop. Once per call is fine; once per iteration is silly.

#### Worked example: lazy-loading pandas in a real CLI

Same machine, CPython 3.12, median of five fresh-process runs of `mytool --version` (the path that needs nothing heavy):

| Version | Imports on `--version` | Startup time | Speedup |
| --- | --- | --- | --- |
| Eager (pandas at top) | argparse, pandas, numpy, pytz, requests | ~1.41 s | 1.0x |
| Lazy (pandas deferred) | argparse only | ~0.20 s | **~7.0x** |

The `--describe` path still pays the pandas cost on first use (so its total is roughly unchanged), but the common, cheap paths — `--version`, `--help`, bad-argument errors — drop from 1.4 s to 0.2 s. For a tool people run dozens of times a day, mostly to check flags or fix typos, that is the difference between "feels broken" and "feels instant." The 0.2 s floor is the interpreter plus argparse plus your code; to go lower you would attack the interpreter init itself, which is the territory of frozen modules (below).

## Fix #2: trim dependency bloat — the import you should not have

Lazy imports defer cost. The next lever *removes* it. A startling amount of startup time comes from heavyweight dependencies pulled in for a sliver of functionality you could get cheaper. The classic offenders:

- **`import pandas` to read one CSV.** If you are reading a CSV and iterating rows, the standard library's `csv` module imports in under a millisecond and does the job. Pandas is 520 ms of startup for the convenience of `.describe()`. For a one-shot CLI that reads a small file, `csv` or the third-party `pyarrow`'s lightweight CSV reader (which loads far faster than pandas) may be the right tradeoff.
- **`import requests` for one HTTP GET.** `requests` pulls urllib3, certifi, idna, and charset_normalizer — about 40 ms. The standard library's `urllib.request` is already imported by other machinery and is essentially free for a simple GET. For a CLI making one call, the stdlib is often plenty.
- **`import numpy` for a sum.** If you are summing a list of a few thousand numbers, Python's built-in `sum()` is instant and NumPy's 95 ms import is pure waste. NumPy earns its import cost when you are doing *vectorized* work over large arrays, as the [vectorization posts](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) in this series cover; for scalar arithmetic it is a liability.
- **A giant framework for a small script.** Importing all of `scipy`, all of `sklearn`, or a full web framework to use one function is the macro version of the same mistake.

The way to find these is the same importtime tree from before: sort by cumulative time, look at the top three packages, and ask of each, "do I actually need this, or did I reach for it out of habit?" Often you can replace a 520 ms import with a 1 ms stdlib equivalent and lose nothing the CLI cares about.

To make the trade concrete, here is the import cost of the heavy library against its stdlib equivalent for the common one-shot tasks, measured on the 8-core box, CPython 3.12, as a fresh-process import:

| Task | Heavy import | Cost | Stdlib equivalent | Cost |
| --- | --- | --- | --- | --- |
| Read a CSV | `import pandas` | ~520 ms | `import csv` | <1 ms |
| One HTTP GET | `import requests` | ~40 ms | `import urllib.request` | ~3 ms |
| Sum some numbers | `import numpy` | ~95 ms | built-in `sum` | 0 ms |
| Parse JSON | a third-party parser | ~10–30 ms | `import json` | ~1 ms |
| Parse a date | `import dateutil` | ~15 ms | `import datetime` | <1 ms |

None of this means the heavy library is bad — pandas is wonderful, and for a tool that does real dataframe work and runs long enough to amortize the import, the 520 ms is well spent. The point is purely about *startup-sensitive code on the cheap path*: a CLI that mostly reads small files, makes single requests, and prints results can often swap the entire right column in for the left and drop a second of startup with no loss of capability the tool actually uses. The stdlib is already partly loaded by the interpreter, so its marginal import cost is near zero, and it ships with Python so it adds no dependency to install. For the 80% case of a small utility, "use the standard library" is itself a performance optimization.

There is a deeper, subtler form of bloat: **transitive imports you did not ask for.** A small utility library you depend on might itself `import pandas` at *its* top level, and now you pay pandas's startup cost transitively, through a dependency you thought was lightweight. The importtime tree exposes this — you will see `pandas` indented under some innocuous-looking package. The fix is either to file an issue asking the upstream to lazy-import (many libraries have done exactly this for startup reasons), to pin to a version before the heavy dependency was added, or to vendor the one function you need. This is also why minimal dependency footprints matter for startup-sensitive code: every dependency is a potential import-time landmine, and you do not control when its maintainers add a heavy transitive import.

![matrix table of startup levers showing lazy import dependency trimming importtime measurement bytecode cache frozen modules and PYTHONOPTIMIZE with their effect and when to use each](/imgs/blogs/faster-startup-imports-packaging-and-alternative-interpreters-3.png)

Figure 3 lays out the full menu of startup levers, their typical effect, and when to reach for each. Read it top to bottom in order of leverage: measure with importtime first, then lazy-import and trim (the big wins), then lean on the bytecode cache and frozen modules (free or near-free), and only bother with `PYTHONOPTIMIZE` once the imports are already lean (it is a small win and rarely the real problem). The ordering matters: trimming a 520 ms import dwarfs anything `-O` will ever buy you, so do the imports first.

## Fix #3: the bytecode cache, frozen modules, and interpreter flags

Even after you have trimmed and deferred your imports, there is a floor: interpreter startup plus the cost of compiling and running whatever modules remain. Several mechanisms attack this floor, mostly for free.

**`__pycache__` and the bytecode cache.** When Python first imports a module, it compiles the `.py` source to bytecode and writes a `.pyc` file into a `__pycache__` directory, tagged with the interpreter version and the source's modification time (or hash). On every later import, if the cache is valid, Python skips the parse-and-compile step entirely and loads the bytecode directly. The compile step is not free — for a large module it can be tens of milliseconds — so a warm `.pyc` cache is a real startup win on the *first run after a deploy*. The practical implications:

- **Ship the `.pyc` files** in your container image or deployment artifact so the very first invocation in production is already warm. A common trick is to run `python -m compileall .` during the image build, which pre-compiles every module so no end-user invocation ever pays the compile cost.
- **Read-only filesystems** (some serverless and container environments) cannot write `__pycache__`, so without pre-compilation you pay the compile cost on *every* cold start. Pre-compiling with `compileall` at build time fixes this.
- The cache is keyed on the interpreter version, so a Python upgrade correctly invalidates it; you do not get stale bytecode across versions.

```bash
# pre-compile everything at image build time so production starts warm
python -m compileall -q /app
# or compile with optimizations baked in (strips asserts, see -O below)
python -O -m compileall -q /app
```

**Frozen modules (the 3.11 startup win).** Here is a genuinely free improvement that arrived with CPython 3.11. The interpreter has to import a handful of standard-library modules just to *boot* — the import machinery itself, `os`, `io`, `abc`, codecs, and others — before it can run anything. Historically these were imported from disk like any other module. In 3.11, the "Faster CPython" work *froze* these startup modules: their bytecode is compiled ahead of time and embedded directly into the Python binary, so importing them at boot is a memory read, not a disk read and compile. The measured effect is a meaningful cut in bare-interpreter startup. On the 8-core box, `python -c "pass"` dropped from roughly 33 ms on 3.10 to roughly 28 ms on 3.11 and 3.12 — about a 10 to 15% reduction in the startup floor, and you get it for free just by upgrading. You can confirm a module is frozen:

```python
import importlib.util
# the loader name tells you it came from frozen bytecode, not disk
spec = importlib.util.find_spec("os")
print(spec.origin)   # 'frozen' for frozen stdlib boot modules on 3.11+
```

![layered stack showing process start from interpreter init through frozen stdlib boot then the import chain then bytecode compile or cache hit then finally your code running](/imgs/blogs/faster-startup-imports-packaging-and-alternative-interpreters-5.png)

Figure 5 shows the full stack of what happens between typing `python mytool.py` and your first line executing. From the bottom: interpreter initialization (20 to 30 ms), the frozen-stdlib boot (faster on 3.11+), the import chain of *your* dependencies (this is the big, controllable layer — 1.2 s in the bad case), bytecode compilation or a `.pyc` cache hit, and finally your code. The point of the figure is that every short-lived process repeats this *entire* stack from scratch. The layers you control most are the import chain (lazy/trim) and the cache (pre-compile); the bottom layers you mostly get for free by upgrading CPython.

**`-O` / `PYTHONOPTIMIZE`.** Running Python with `-O` (or setting `PYTHONOPTIMIZE=1`) strips `assert` statements and code guarded by `__debug__`; `-OO` additionally strips docstrings. This produces slightly smaller `.pyc` files and skips the assert checks at runtime. Be honest about the magnitude: this is a *small* win. It will not rescue a tool whose problem is a 520 ms pandas import. It matters at the margins — a tiny reduction in bytecode size and a few skipped asserts — and it carries a real risk: if any of your code (or a dependency's) relies on `assert` for actual validation rather than debugging checks, `-O` silently disables it, which can be a correctness bug. Use `-O` for production builds where you have audited that asserts are debug-only, but do not expect it to move the startup needle much. The order of operations is: fix imports first, lean on caching and frozen modules (free), and treat `-O` as a final small polish, not a fix.

## Fix #4: packaging fast code — ship wheels, not source

Now flip the perspective. The earlier posts in this series taught you to rewrite the hot 1% in native code — [Cython, a C extension, or Rust via PyO3](/blog/software-development/python-performance/rust-for-python-pyo3-and-maturin). That native code has to *reach your users*, and how you package it determines whether their `pip install` takes three seconds or three minutes (and whether it works at all without a C compiler on their machine). The mechanism that makes this fast and reliable is the **wheel**.

A **wheel** (`.whl`) is a pre-built, ready-to-install Python package: a zip archive with a standardized layout. For a pure-Python package, a wheel is just your `.py` files plus metadata — installing it is a copy, no build step. For a package with a compiled extension (Cython, C, Rust), the wheel contains the *already-compiled* shared library (`.so` on Linux, `.pyd` on Windows, `.dylib`-style on macOS) for a specific platform and Python version. When a user runs `pip install yourpackage`, pip downloads the matching wheel and just unzips it. No compiler. No build. No waiting. This is why installing NumPy or pandas takes seconds even though they are huge piles of compiled C and Fortran: you are downloading a pre-built binary, not compiling it.

The alternative — an **sdist** (source distribution) — ships your source code, and pip must *compile* the extension on the user's machine at install time. That requires a working C/C++/Rust toolchain, the right headers, and minutes of build time, and it fails loudly on any machine missing the compiler. Shipping only an sdist for a compiled package is how you get the dreaded "Microsoft Visual C++ 14.0 is required" or "error: command 'gcc' failed" install errors. Wheels exist to make all of that the *maintainer's* problem, solved once, instead of *every user's* problem, solved never.

The catch is that a compiled wheel is platform-specific. A `.so` built on your Linux laptop will not run on Windows, on macOS, or even on an older Linux with a different glibc. The solution is **manylinux**: a set of standardized, deliberately-old Linux build environments (defined by PEPs 513, 599, 600, and successors) that produce wheels compatible with essentially every modern Linux distribution. You build inside a `manylinux` container — which has an old-enough glibc and a curated set of system libraries — and the resulting wheel, tagged something like `manylinux_2_17_x86_64`, runs on any Linux newer than that baseline. To cover the world you build a *matrix* of wheels: each platform (Linux x86-64, Linux aarch64, macOS x86-64, macOS arm64, Windows) times each supported Python version (and ABI). The tool that automates this matrix is `cibuildwheel`, which spins up the right containers and runs your build across all of them in CI; `maturin` (for Rust/PyO3) and `setuptools`/`scikit-build` (for C/Cython) produce the individual wheels.

```bash
# Build a Rust extension wheel with maturin (from the PyO3 post)
maturin build --release            # one wheel for THIS platform
# Build the full portable matrix in CI with cibuildwheel
cibuildwheel --platform linux      # manylinux wheels, all CPython versions
# Inspect what you built
ls dist/
# yourpkg-1.0-cp312-cp312-manylinux_2_17_x86_64.whl   <- pre-compiled, portable
# yourpkg-1.0-cp312-cp312-macosx_11_0_arm64.whl
```

The payoff chain is worth stating plainly: you do the hard native-compile work once, in CI, across a matrix; you upload the wheels to PyPI; and from then on every `pip install yourpackage` anywhere in the world is a download-and-unzip, no compiler required. That is the packaging half of "ship the fast code."

It helps to read a wheel filename, because the tags encode exactly which machines it runs on. A name like `yourpkg-1.0-cp312-cp312-manylinux_2_17_x86_64.whl` has four meaningful fields after the version: the *Python tag* (`cp312` = CPython 3.12), the *ABI tag* (`cp312` = the CPython 3.12 binary interface, the C-API/ABI the extension was compiled against), and the *platform tag* (`manylinux_2_17_x86_64` = Linux on x86-64 with glibc 2.17 or newer). pip looks at the machine it is running on, computes the set of compatible tags, and downloads the *most specific* wheel that matches. If no compiled wheel matches — say you are on a brand-new Python version the project has not built wheels for yet — pip falls back to the sdist and tries to compile, which is when users hit build failures. This is also why a compiled extension must be rebuilt for each new CPython minor version: the ABI tag changes, and a `cp311` wheel will not be selected on 3.12. (Pure-Python packages and extensions built against the *stable ABI* can use the broader `abi3` tag and skip the per-version rebuild, but most performance-oriented extensions target a specific version's ABI for full API access.)

There is a real performance cost to *getting the tags wrong*, beyond install failures. If your platform tag is too new — you built on a bleeding-edge Linux and tagged it for a recent glibc — older but still-supported systems will not match it and will fall back to compiling from sdist, paying minutes of build time per install instead of seconds of download. The `manylinux` standard exists precisely to push the glibc baseline *old enough* that essentially every production Linux matches, so the download-and-unzip fast path is the one that fires. Building inside the official `manylinux` containers (rather than on your dev machine) is what guarantees this; the container's old glibc is a feature, not a limitation.

**Editable installs** are the development-time complement. When you run `pip install -e .` (an *editable* install, PEP 660), pip does not copy your package into `site-packages`; it installs a link back to your working directory, so edits to your `.py` files take effect immediately without reinstalling. For a pure-Python package this is seamless. For a *compiled* package there is a catch: editing the `.py` is instant, but editing the Rust/C/Cython source requires recompiling the extension — `maturin develop` rebuilds and re-links the extension into your environment in one step, which is the editable-install equivalent for native code. Knowing this distinction saves a lot of "why are my changes not showing up" confusion: pure-Python edits are live; native edits need a rebuild.

**Why `uv` and modern tooling are fast.** Recent years brought a wave of dramatically faster Python tooling, most visibly `uv` (a Rust-based installer and resolver from the makers of `ruff`). When people say `uv pip install` is 10 to 100 times faster than classic pip, the speedup comes from several concrete engineering choices, not magic:

- **A fast dependency resolver written in Rust**, avoiding the slow backtracking that classic pip's pure-Python resolver could fall into on complex dependency graphs.
- **Aggressive parallelism** — it downloads and unpacks wheels concurrently across many connections, rather than mostly serially.
- **A global content-addressed cache with hard-linking** — once a wheel is in the cache, installing it into a new environment hard-links the files instead of copying them, so creating a fresh virtualenv with the same dependencies is nearly instant and uses almost no extra disk.
- **It is itself a fast-starting native binary** — `uv` does not pay Python's own import/startup cost to run, which matters because an installer is exactly the kind of short-lived CLI this whole post is about.

The last point is delicious: `uv` is fast partly *because its authors took the startup problem seriously*. It is the lesson of this post embodied in a tool. The practical takeaway is that for startup-sensitive and install-sensitive work, the tooling around your Python now matters as much as the Python.

## Fix #5: alternative interpreters — when PyPy helps, and when it does nothing

CPython is the reference interpreter — the one almost everyone runs — and everything above applies to it. But it is not the *only* implementation of Python, and the most important alternative for performance is **PyPy**. PyPy can run the same pure-Python code many times faster than CPython, with zero changes to your source. That sounds like a free lunch, and the entire skill of using it well is understanding exactly when it *is* and when it is decidedly *not*.

The reason PyPy is fast is a **tracing JIT (just-in-time) compiler**. CPython is a pure interpreter: it reads your bytecode one opcode at a time and executes each one in C, every iteration, forever (the [adaptive interpreter](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop) in 3.11+ specializes hot opcodes but does not compile to machine code). PyPy does something fundamentally different. While your program runs, PyPy *watches* it. When it notices a loop executing many times — a "hot" loop — it *traces* one iteration: it records the exact sequence of operations that actually happened, including the concrete types it saw, and compiles that linear trace down to optimized **machine code**, with type checks ("guards") inserted to bail back to the interpreter if a later iteration does something different. From then on, the hot loop runs as native machine code, not interpreted bytecode. The same boxing, type-dispatch, and eval-loop overhead that makes CPython's loops slow simply evaporates, because the JIT has specialized the loop to the types it actually sees.

![before and after comparison showing CPython 3.12 with fast 30 millisecond startup running a pure python loop interpreted in 9 seconds versus PyPy with slow startup plus warmup but the traced hot loop finishing in 0.8 seconds an 11 times speedup](/imgs/blogs/faster-startup-imports-packaging-and-alternative-interpreters-6.png)

Figure 6 captures the trade-off that governs everything about PyPy. CPython starts fast and runs the loop slowly. PyPy starts *slower* — both its bare interpreter startup and the JIT *warmup*, the time it spends running interpreted and tracing before it has compiled the hot loop — and only *after* that warmup does it pull dramatically ahead. This is the **amortization argument**, and it is the single most important thing to understand about JITs.

It is worth understanding a little more about *why* the JIT produces such fast code, because it explains both the wins and the limits. The overhead a tracing JIT eliminates is exactly the overhead the [execution-model post](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop) anatomized: in CPython, every `total += x * x * dx` in our loop is a sequence of bytecode opcodes, each dispatched through the eval loop, each operand a boxed `PyObject` on the heap, each arithmetic operation a type check followed by a method-table lookup followed by a fresh heap allocation for the result, and each reference-count bump a memory write. That per-operation tax is paid on *every iteration*, a hundred million times. What PyPy's JIT does, once it has traced the loop and seen that `total`, `x`, and `dx` are always plain floats, is compile a specialized machine-code version where the floats live in CPU registers, the arithmetic is bare floating-point instructions, the boxing and unboxing are gone, the type checks are replaced by one cheap *guard* at the top of the loop, and the redundant allocations are eliminated. The loop body collapses from dozens of interpreted opcodes per iteration to a handful of native instructions. That is where the 11x comes from: not a faster interpreter, but *no interpreter at all* for the hot path.

The "guards" are the subtle part and the reason the warmup matters. The compiled trace is valid only as long as the assumptions it baked in hold — that `x` is always a float, that the loop structure does not change. Each assumption becomes a guard, a cheap runtime check; if a guard fails (say a later iteration suddenly sees an integer), execution *bails out* of the machine code back into the interpreter, and if that bail-out happens often, PyPy may re-trace. This is also why PyPy shines on *type-stable* loops — code where the same types flow through the same path millions of times — and does less well on code that is wildly polymorphic, where guards fail constantly and the JIT cannot settle on a fast trace. Pure numeric loops are the ideal case: stable types, hot inner loop, lots of iterations. That is precisely the workload profile where the amortization formula and the JIT mechanics agree that PyPy wins.

Let me make the amortization rigorous, because it tells you precisely when PyPy is worth it. Let $w$ be PyPy's warmup overhead (startup plus the interpreted iterations before the JIT kicks in and compiles the loop), measured in seconds. Let CPython run your hot loop at rate $r_C$ iterations per second and PyPy's *compiled* loop run at the much faster rate $r_P$. For a workload of $n$ iterations, CPython takes about $T_C = n / r_C$ and PyPy takes about $T_P = w + n / r_P$. PyPy wins — $T_P < T_C$ — when

$$n / r_C > w + n / r_P \;\;\Longleftrightarrow\;\; n \left( \frac{1}{r_C} - \frac{1}{r_P} \right) > w.$$

The left side grows with $n$; the right side ($w$) is a fixed cost. So there is a **break-even iteration count** $n^\star$ below which PyPy is *slower* (you paid the warmup but never ran the loop long enough to recoup it) and above which it is faster, and the further past $n^\star$ you go the more the warmup amortizes toward zero. This is why the rule of thumb is "PyPy wins on long-running, loop-heavy, pure-Python workloads." A CLI that runs for 50 milliseconds never reaches $n^\star$ — it pays PyPy's slow start and exits before the JIT helps, so PyPy makes it *slower*. A simulation that runs a tight numeric loop for ten minutes is miles past $n^\star$ — the warmup is a rounding error and you get the full JIT speedup. The break-even is exactly the amortization boundary the formula predicts.

Here is a benchmark you can run yourself — a deliberately loop-heavy, pure-Python kernel (a naive numeric integration) that is exactly the kind of code PyPy was built for:

```python
# loopbench.py — pure-Python hot loop, no NumPy, no C
import time

def integrate(n):
    total = 0.0
    dx = 1.0 / n
    x = 0.0
    for _ in range(n):
        # f(x) = x*x, accumulated the slow scalar way
        total += x * x * dx
        x += dx
    return total

if __name__ == "__main__":
    n = 100_000_000
    t0 = time.perf_counter()
    result = integrate(n)
    print(f"{result:.6f} in {time.perf_counter() - t0:.3f} s")
```

Run it under both interpreters (PyPy installs alongside CPython; you invoke it as `pypy3`):

```bash
python3  loopbench.py     # CPython 3.12
pypy3    loopbench.py     # PyPy (same source, no changes)
```

On the 8-core Linux box, CPython 3.12 ran the 100-million-iteration loop in roughly 9.0 s. PyPy ran the identical source in roughly 0.8 s — about an **11x speedup**, with *no code change whatsoever*. That is the magic of a tracing JIT on pure-Python loops: it compiled the inner loop to machine code and the boxing/dispatch overhead disappeared. (Your exact numbers will vary with PyPy version and the loop; speedups of 5x to 50x on tight pure-Python loops are typical, and occasionally far more.)

Now the crucial counter-example — **the case where PyPy does nothing.** Take the *same* integration, but written in NumPy, where the loop already lives in compiled C:

```python
# loopbench_numpy.py — the work is already in NumPy's C core
import numpy as np, time

def integrate(n):
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.sum(x * x) / n

if __name__ == "__main__":
    t0 = time.perf_counter()
    print(integrate(100_000_000), time.perf_counter() - t0)
```

Run this under CPython and PyPy and the times are *nearly identical* — PyPy gives essentially no speedup, and historically could be *slower* because PyPy must run NumPy through a C-extension compatibility layer (`cpyext`) that adds overhead at the boundary. Why? Because **a tracing JIT can only speed up code that runs in the Python interpreter.** The NumPy version has almost no Python-level loop to trace; the entire computation happens inside one C function (`np.sum` over a packed buffer), and that C function runs at the same speed no matter which Python interpreter called it. There is no boxed-object interpreter loop for the JIT to compile away, because there is no interpreter loop in the hot path at all. This is the second rigorous claim of the section: **a JIT cannot accelerate work that is already native.** The boxing-and-dispatch overhead it eliminates was never present in vectorized NumPy code, so there is nothing to eliminate.

That single fact resolves most of the "should I use PyPy?" confusion. PyPy is a weapon against *the interpreter overhead of pure-Python loops*. If your hot path is already vectorized NumPy, or already a C/Cython/Rust extension, the interpreter overhead PyPy removes is already absent, and PyPy buys you nothing — while costing you C-extension compatibility headaches and slower startup. The decision is cleaner than people think:

- **Long-running, loop-heavy, mostly pure-Python, few C extensions?** PyPy can be a huge, zero-effort win. Try it.
- **Already vectorized in NumPy, or hot path already in a C/Rust extension?** PyPy gives ~nothing and adds compatibility risk. Stay on CPython.
- **Short-lived CLI or serverless function?** PyPy's slow startup makes it *worse*. Stay on CPython and fix imports.
- **Depends heavily on C extensions PyPy does not support well?** Compatibility cost can outweigh the loop speedup. Test before committing.

#### Worked example: PyPy on three workloads

Same machine, the median of several runs, CPython 3.12 versus PyPy:

| Workload | CPython 3.12 | PyPy | Speedup | Why |
| --- | --- | --- | --- | --- |
| Pure-Python loop (100M iters) | ~9.0 s | ~0.8 s | **~11x** | JIT compiles the hot loop to machine code |
| NumPy version (100M elems) | ~0.35 s | ~0.4 s | ~1.0x (or worse) | work is already in C; nothing to JIT |
| CLI `--version` startup | ~0.20 s | ~0.5 s | **~0.4x (slower)** | PyPy startup + warmup, loop never runs |

The table *is* the decision. PyPy is spectacular on the first row, irrelevant on the second, and actively harmful on the third. The amortization formula predicted all three: the pure-Python loop runs far past $n^\star$, the NumPy version has no interpreter loop to trace, and the CLI exits before warmup completes.

**A brief, honest nod to GraalPy.** PyPy is not the only alternative interpreter. **GraalPy** is a Python implementation built on Oracle's GraalVM (the JVM-based polyglot runtime). It also has a JIT and can be fast on the right workloads, and its headline feature is tight interoperability with Java and other GraalVM languages — if you are embedding Python inside a JVM application, or want to call Java libraries directly, GraalPy is the natural fit. The honest caveat is that it is younger than PyPy, its C-extension compatibility and ecosystem coverage are narrower, and its niche (JVM polyglot embedding) is specific. For most performance work the choice is CPython-with-native-extensions versus PyPy-for-pure-Python-loops; GraalPy is worth knowing exists but is a specialized tool. There is also **Cinder** (Meta's CPython fork with a JIT and other optimizations), used internally at scale but not a general drop-in, and the official CPython JIT effort (the experimental copy-and-patch JIT landing in 3.13+), which aims to bring some JIT benefits to mainline CPython over time without the compatibility cost of switching interpreters.

![matrix comparing CPython PyPy GraalPy manylinux wheels and uv showing what each is best for and its main caveat](/imgs/blogs/faster-startup-imports-packaging-and-alternative-interpreters-7.png)

Figure 7 collects the full runtime-and-packaging menu: CPython as the safe default for short jobs and native libraries, PyPy for long pure-Python loops, GraalPy for JVM polyglot embedding, manylinux wheels for shipping compiled extensions portably, and `uv` for fast installs. Each row carries its caveat, because every one of these is a tradeoff, not a free upgrade. The caveats are where teams get burned: choosing PyPy for a CLI, or shipping an sdist instead of wheels and getting compile failures on user machines.

## Case studies: real startup and runtime numbers

A few real, named results to ground all of this in the ecosystem rather than my benchmarks alone. As always, I name the source and frame anything I did not personally run as approximate.

**The `ruff` and `uv` startup obsession.** The Astral tooling (`ruff` the linter, `uv` the installer) is the loudest proof that startup matters. `ruff` is famously 10 to 100 times faster than the pure-Python linters it replaced, and a large part of the felt difference is that it is a native binary with effectively zero interpreter startup — there is no `import` chain to pay because it is not a Python process at all. `uv` brings the same philosophy to package installation: a Rust binary that starts instantly and parallelizes aggressively. The lesson is not "rewrite everything in Rust"; it is that for tools people run constantly, the startup floor is the product, and the teams that win treat it that way.

**The CPython 3.11 "Faster CPython" startup work.** The 3.11 release (the first big "Faster CPython" milestone) reported roughly 10 to 60% overall speedups on the pyperformance benchmark suite depending on the workload, and specifically improved *startup* by freezing the standard-library boot modules into the binary, as covered above. On the 8-core box I measured bare `python -c "pass"` dropping from ~33 ms (3.10) to ~28 ms (3.11/3.12) — a real, free improvement you get just by upgrading the interpreter. For startup-sensitive code, "upgrade CPython" is a legitimate first move with zero code changes.

**PyPy on pure-Python workloads.** The PyPy project's own benchmark suite reports a geometric-mean speedup of roughly 4x to 5x over CPython across a broad set of programs, with individual loop-heavy pure-Python benchmarks running 10x or more faster. The numbers I measured above (an 11x on a tight integration loop, ~1x on NumPy) are squarely in that documented range. The consistent caveat across all PyPy reporting is the C-extension story: workloads dominated by C-extension calls see little or negative benefit, which is exactly the NumPy result.

**Serverless cold starts.** Across the serverless ecosystem (AWS Lambda, Google Cloud Functions, and the like), import time is a well-documented cold-start contributor for Python functions, and the standard mitigations are precisely the ones in this post: minimize and lazy-import dependencies, ship pre-compiled `.pyc` (or pre-built layers), and avoid pulling in a heavy framework when a small one will do. A function that lazy-imports its heavy SDK and only loads it on the code path that needs it can cut cold-start latency by hundreds of milliseconds — the same 7x-on-the-cheap-path effect from our CLI worked example, but where the cheap path is "respond to a health check" and the expensive one is "run the report."

#### Worked example: a serverless handler's cold start

Consider a function whose handler imports the full cloud SDK plus pandas at module top — a common pattern, because the imports go at the top out of habit. Under bursty traffic, every cold start re-imports the whole world. Approximate cold-start import budget, and the same handler after applying this post's levers:

| Handler version | Import path | Cold-start import time |
| --- | --- | --- |
| Eager (SDK + pandas at top) | full SDK, pandas, numpy on every cold start | ~900 ms |
| Lazy (defer pandas + heavy SDK clients) | only the slice the request needs | ~250 ms |
| Lazy + pre-compiled `.pyc` in the layer | no compile step on read-only FS | ~210 ms |

On a function billed per cold start, shaving ~700 ms off the import budget is both a latency win (the user-facing tail latency drops) and a cost win (you are billed for less runtime per cold invocation). The mechanism is identical to the CLI: the work the request does not need should not be imported, and on a read-only serverless filesystem you must pre-compile the bytecode at build time because the runtime cannot write `__pycache__`. This is the import section of this post applied verbatim to the highest-volume short-lived-process workload there is.

## When to reach for each lever (and when not to)

This post has a lot of levers; the discipline is pulling the *right* one, which always starts with measuring which bucket you are in. Here is the decisive guidance.

**Reach for lazy imports when** your code is a CLI, a serverless function, a short-lived worker, or a test suite, *and* `-X importtime` shows a heavy import dominating startup on a path that does not always need it. This is the highest-leverage startup fix and it is almost always worth it for these workloads. **Do not bother** for a long-running service where every path eventually runs — eager imports at boot are better there, paid once during warm-up rather than during a request, and they fail fast if a mandatory dependency is missing.

**Reach for dependency trimming when** the importtime tree shows a 500 ms package imported for a feature the standard library provides in 1 ms — `pandas` for one CSV, `requests` for one GET, `numpy` for a scalar sum. **Do not** trim a dependency you genuinely use heavily; pandas earns its import cost in a tool that does real dataframe work and runs long enough to amortize it.

**Reach for the bytecode cache and frozen modules always** — they are free. Pre-compile with `compileall` at image build time so production starts warm, and upgrade to CPython 3.11+ for the frozen-stdlib win. **Treat `-O` as a final polish**, not a fix; it is a small win and risks disabling asserts your code relies on.

**Reach for wheels whenever you ship a compiled extension.** Always build and publish a `manylinux` (plus macOS and Windows) wheel matrix so users `pip install` a pre-built binary and never need a compiler. **Never ship only an sdist for a compiled package** unless you enjoy "gcc failed" bug reports.

**Reach for PyPy when** your hot path is a long-running, loop-heavy, mostly pure-Python computation with few C-extension dependencies — it can be a 5x to 50x win with zero code changes, and the amortization formula says you are well past break-even. **Do not reach for PyPy** when your hot path is already vectorized NumPy or already a C/Rust extension (the interpreter overhead it removes is already gone), when your code is a short-lived CLI (its slow startup makes things worse), or when you depend heavily on C extensions PyPy supports poorly (compatibility cost outweighs the gain). And **reach for GraalPy** specifically when you are embedding Python in a JVM polyglot application — otherwise it is a niche tool.

![decision tree starting from slow then branching on whether the cost is startup a long pure python loop or shipping native code and routing to trim and lazy imports PyPy or manylinux wheels](/imgs/blogs/faster-startup-imports-packaging-and-alternative-interpreters-8.png)

Figure 8 is the decision tree in one picture. Start at the top — *is it actually slow, and have you measured?* — then branch on the *kind* of cost. If it is startup (a CLI or serverless function), trim and lazy-load your imports. If it is a long pure-Python hot loop and you have no heavy C dependencies, try PyPy. If you are shipping native code to users, build manylinux wheels. The whole post collapses into that one routing decision, and the routing depends entirely on which kind of slowness you measured — which is why measuring comes first. This same "measure, then route to the right lever" structure is the spine of the series, and it is exactly what the capstone playbook (the decision-framework post that closes this series) generalizes across every lever, not just startup.

## Key takeaways

- **Import cost is paid once per process start.** For long-running services it amortizes to zero; for CLIs, serverless functions, short-lived workers, and test suites it is paid in full every invocation and often *dominates* the user-visible time. Know which bucket you are in before you optimize anything.
- **`python -X importtime` is the X-ray.** It prints the import tree with per-module self and cumulative times. Sort by cumulative, look at the top three packages, and you will usually find one heavy import responsible for most of your startup.
- **Lazy imports defer cost; trimming removes it.** Move a heavy import into the function that needs it (or use a module-level `__getattr__` or the `TYPE_CHECKING` trick), and replace a 500 ms dependency with a 1 ms stdlib equivalent where you can. Together these took our CLI from 1.4 s to 0.2 s on the cheap path — about 7x.
- **The bytecode cache and frozen modules are free wins.** Pre-compile with `compileall` so production starts warm; upgrade to CPython 3.11+ for the frozen-stdlib startup improvement. `-O` is a small final polish, not a fix.
- **Ship wheels, not source.** A `manylinux`/macOS/Windows wheel matrix lets users `pip install` a pre-built binary with no compiler. `uv` is fast because it is a native, fast-starting binary with a parallel resolver and a hard-linking cache — the startup lesson applied to tooling.
- **PyPy is a weapon against pure-Python interpreter overhead.** Its tracing JIT compiles hot loops to machine code, giving 5x to 50x on long-running, loop-heavy, pure-Python code — but only past the warmup break-even point, so it is wrong for short CLIs and gives nothing for already-vectorized NumPy or native code.
- **A JIT cannot speed up code that is already in C.** If your hot path is NumPy or a C/Rust extension, the boxing-and-dispatch overhead PyPy removes was never there. Match the tool to where the time actually is.
- **The order is always: measure, route, fix.** Measure with importtime (or a profiler); route on the *kind* of cost (startup vs hot loop vs shipping native); pull the matching lever; re-measure and prove the win with a number.

## Further reading

- The CPython documentation on the [import system](https://docs.python.org/3/reference/import.html), [`-X importtime`](https://docs.python.org/3/using/cmdline.html#cmdoption-X), and [`compileall`](https://docs.python.org/3/library/compileall.html) — the authoritative sources for the import machinery and the bytecode cache.
- [PEP 562 (module `__getattr__`)](https://peps.python.org/pep-0562/) and [PEP 563 / `from __future__ import annotations`](https://peps.python.org/pep-0563/) for the lazy-import and `TYPE_CHECKING` patterns.
- The [Python Packaging User Guide](https://packaging.python.org/) and the [manylinux](https://github.com/pypa/manylinux) and [`cibuildwheel`](https://cibuildwheel.readthedocs.io/) projects for building and shipping portable compiled wheels.
- The [PyPy project](https://www.pypy.org/) and its [performance benchmarks](https://speed.pypy.org/), plus the [GraalPy](https://www.graalvm.org/python/) documentation for the alternative-interpreter landscape.
- The [Faster CPython](https://github.com/faster-cpython/ideas) notes and the CPython 3.11 What's New for the frozen-modules and startup improvements.
- *High Performance Python* by Micha Gorelick and Ian Ozsvald (O'Reilly) — the book-length treatment of this whole series' subject, including startup and packaging.
- Within this series: the intro [Why Python Is Slow (and What "Fast" Actually Means)](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means), [The CPython Execution Model: Bytecode and the Eval Loop](/blog/software-development/python-performance/the-cpython-execution-model-bytecode-and-the-eval-loop) (for why a JIT is the conceptual opposite of the adaptive interpreter), [Benchmarking Python Correctly: timeit Pitfalls and Statistics](/blog/software-development/python-performance/benchmarking-python-correctly-timeit-pitfalls-and-statistics) (for measuring startup honestly), and [Rust for Python: PyO3 and maturin](/blog/software-development/python-performance/rust-for-python-pyo3-and-maturin) (for building the native wheels you ship). The capstone decision-framework post that closes the series generalizes this "measure, route, fix" loop across every lever.
