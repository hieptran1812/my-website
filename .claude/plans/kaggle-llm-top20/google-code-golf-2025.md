# NeurIPS 2025 - Google Code Golf Championship: top-20 addendum

- Slug: `google-code-golf-2025` (competition id 95282, forum 7155893). Final board pulled from `competitions.LeaderboardService/GetLeaderboard` on 2026-10-28.
- Deep coverage of ranks 1 to 4 (Code Golf International, jailctf merger, ox jam!, FuunAgent) and one-line stubs for 5 to 10, 14, 19, 25 live in `.claude/plans/kaggle-llm-writeup-bank-part3.md`, H2 `## NeurIPS 2025 - Google Code Golf Championship` (lines ~633 to 757). This file adds ranks 5 to 20.
- Board mechanics: 100% public leaderboard, so public rank = private rank for every team (API confirms). Score = sum over 400 tasks of `max(1, 2500 - bytes)`; one byte anywhere = one point. Rank 1 to 20 spans 962,070 to 950,693 (11,377 bytes over 400 tasks, ~28 bytes/task).
- Community index of solution repos: c-number, https://www.kaggle.com/competitions/google-code-golf-2025/discussion/613968 .

## Top-20 table

| Rank | Team | Members (display name (username)) | Score | Public rank | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|
| 1 | Code Golf International | Luke G (lukeg10081); Mukundan (mukundan314); Seek64 (seekerstyle); 4atj (bruhhhhd); sisyphus-cg (sisyphuscg) | 962,070 | 1 | https://www.kaggle.com/competitions/google-code-golf-2025/writeups/cgi (topic 614092) | Expert golfers, all by hand, zlib on 21 tasks (bank) |
| 2 | jailctf merger | lydxn (google64); Quasar-0147 (quasar0147); oh_word (ohword); quasar098 (quasar098); hellopir2 (hellopir2) | 961,805 | 2 | https://www.kaggle.com/competitions/google-code-golf-2025/writeups/solution (topic 618673) | CTF team; char-level fuzzer, GA compression; LLM first drafts for long tasks (bank) |
| 3 | ox jam! | ovs (oskarvs); Kuan Wei Heng (kuanweiheng); MWI (cheeseexports); JoKing (kayjoking); abcd (dcab1234) | 961,784 | 3 | https://www.kaggle.com/competitions/google-code-golf-2025/writeups/3rd-place-submission-ox-jam (topic 619624) | code.golf veterans by hand; pysearch (bank) |
| 4 | FuunAgent | Kohei (confirm); Jokrasa (jonaskreutz); THUNDER THUNDER (thunderthunder) | 957,810 | 4 | https://www.kaggle.com/competitions/google-code-golf-2025/writeups/parallel-sampling-rule-based-prompt-generation (topic 614124) | gpt5-codex, 4 parallel samples, AST-rule prompts (bank) |
| 5 | HIMAGINE THE FUTURE. | keymoon (keymoon); shibh308 (shibh308); Yu_212 (yu212mc); kq5y (kq5yyy) | 957,568 | 5 | https://www.kaggle.com/competitions/google-code-golf-2025/writeups/himagine-the-future-5th-place-writeup (topic 614225) | Student CP/CTF team; manual golf + own deflate optimiser beating Zopfli (+~700) + ARC-GEN seed cracking |
| 6 | Theoretical Syntax Golfers | Yosshi999 (yosshi999); shell_mug (shellmug); sz-223 (sz223n); alcea (althaea); satosjp (satosjp) | 957,527 | 6 | https://www.kaggle.com/competitions/google-code-golf-2025/writeups/6th-place-solution-for-google-code-golf-championsh (topic 615145) | Manual golf (955,251) + zopfli (+2,276) incl. variable-renaming by static analysis + simulated annealing |
| 7 | LogicLynx | siman (simann); Shun_PI (shunrcn); nikuroll (nikuroll); Pech1 (pechi998244353); nesya (nesyaru) | 957,513 | 7 | https://www.kaggle.com/competitions/google-code-golf-2025/writeups/7th-place-write-up-logiclynx (topic 619321) | Semi-automated Codex Web (4 outputs/prompt), AGENTS.md knowledge + per-task memos, plus manual tricks |
| 8 | import itertools | jacekwl (jacekwl); asarvazyan (asarvazyan); Marko Rupnik (markorupnik); Natte (natanalel) | 957,177 | 8 | https://www.kaggle.com/competitions/google-code-golf-2025/writeups/8th-place-write-up-import-itertools (topic 615039) | Mostly manual after LLM-assisted 400/400; pysearch, recursion templates, hash-based edge cases |
| 9 | Santa Golf Club 2500 | Ueddy (ku0807); sash (sash2104); 4eta (bio4eta); Kazuo Watanabe (watanabekazuo); yu_w (yuw000) | 957,151 | 9 | https://www.kaggle.com/competitions/google-code-golf-2025/writeups/9th-place-write-up (topic 614150) | CP team, all manual, 1,000+ hours, dashboard-driven "chasing" |
| 10 | lv1.dev | Shik (shik33137); Darkpi (darkpi) | 956,458 | 10 | https://www.kaggle.com/competitions/google-code-golf-2025/writeups/solution-for-arc-golf-lv1-dev (topic 628087) | Manual golf + iterative LLM refinement (mainly OpenAI Codex Web) |
| 11 | cam jam! | Andy Wu (aw7834); Alex Pylypenko (alexpylypenko); James Leung2123 (jamesleung2123); IntGrah (intgrah); Samuel Trajtenberg (sammyuri) | 956,008 | 11 | no writeup found (repo https://github.com/intGrah/golf per topic 613968, now 404; "worst 20 solutions" topic 613437) | unknown (code.golf-community team, by name; UNVERIFIED) |
| 12 | ﾀｲｽｹﾝｻﾝﾊﾞII | Daylight (daylighttime); kuhaku (kuhaku2000); shogo314 (shogo314); littlegirl112 (littlegirl112); KowerKoint (kokikawamura) | 955,424 | 12 | no writeup found (repo link in 613968, now 404) | unknown |
| 13 | 暗黑AGI | 暗黑AGI (boristown); Chen_Hao_kxlmfc (chchchch123); bemymind_hr (bemymind); Xiang Li (gordonli1014); Tianrui Jia (tianruijia123) | 955,227 | 13 | no writeup found | unknown; boristown published Chinese natural-language hints for all 400 tasks (notebook https://www.kaggle.com/code/boristown/agi-chinese-hints-for-all-400-tasks), used as LLM context by 15th |
| 14 | ALE-Agent | iwiwi (takiba); Inoichan (inoueu1); charmq (charmq); c-number (cnumber); Yuki Imajuku (yukiimajuku) | 954,571 | 14 | no writeup; repo https://github.com/SakanaAI/google-code-golf-2025 | Sakana AI team; tooling: judge, compressor, variable-name optimiser, prompt generators for ChatGPT/Claude |
| 15 | santa2024 | denden12 (denden12); ISAKA Tsuyoshi (isakatsuyoshi); Koshiro (kosirowada); chimaki (chimaki821); Hikari_30 (hikari30) | 954,246 | 15 | https://www.kaggle.com/competitions/google-code-golf-2025/writeups/15th-place-solution (topic 614727) | Human seeds -> GPT-5 Pro/Codex follow-up golf + API loop; 23 rule-based minifier notebooks; pysearch; compression |
| 16 | mystery+bono+Lucien | Lucien de Rubempre (lucienderubempre); bono (kjoonha); moto (yamashitamotokazu); sktkmozt (sktkmozt); Kaz (kaz0603) | 953,632 | 16 | no Kaggle writeup; README https://github.com/m5t0/google-code-golf-2025 | Started from two public solution notebooks; compressor sweep + variable-name optimiser |
| 17 | Golfers Anonymous | Oleg (golffer) | 953,512 | 17 | no writeup found | unknown (solo) |
| 18 | Team JYCDT | Jiwei Liu (jiweiliu); Yiheng Wang (yiheng); Chris Deotte (cdeotte); Darren Amadeus Martin (darrenamadeusmartin); Tony Li (tonylica) | 953,336 | 18 | no writeup found | unknown; Tony Li posted a "TOP 78 Tricks" compilation (topic 614082) and task-priority analysis (topic 608435) |
| 19 | Team | Radek Osmulski (radek1); jazivxt (jazivxt); conor (conormacamhlaoibh); Filtered (huanligong); Yuchen20 (yuchen2066) | 953,039 | 19 | partial: https://www.kaggle.com/competitions/google-code-golf-2025/discussion/614729 (Yuchen20 part) ; blog https://yuchen-mao.vercel.app/blog/google-code-golf | Evolutionary LLM program-synthesis agent (Gemini free tier) to ~0.940, then manual |
| 20 | ShadowPrompt Labs | KaizaburoChubachi (zaburo); shinh (shinh0); y011d4 (y011d4); TDK8823 (daisuketakahashi); darrenks (darrenks) | 950,693 | 20 | no writeup found | unknown |

Writeups found: 13 of 20 (ranks 1 to 10, 15, partial 19, README-only 16). 136 forum topics scanned (8 pages).

## Per-team synthesis (ranks 5 to 20)

### Rank 5: HIMAGINE THE FUTURE. (957,568)
Source: topic 614225; repo https://github.com/key-moon/golf ; deflate visualiser https://key-moon.github.io/deflate-viz/ .
- Student team from competitive programming and CTF, new to both code golf and Kaggle; most work in August/September. Effort split stated as "90% grinding code golf, 9% tooling, and 1% weird hacks". No LLM use is mentioned.
- Golf examples: task 111 uses `iter(list.pop, sentinel)` to walk a list backwards until a value; task 270 uses the regex conditional group `(?(2)...|...)`, which they saw nowhere else.
- **Compression stub**: `#coding:L1` (shortest PEP 263 line; Latin-1 maps bytes 0 to 255 one to one) + `exec(zlib.decompress(bytes("...",'L1'),-9))`; raw deflate (negative wbits) saves the 6-byte header/checksum for 4 bytes of argument; `-9` often works for small streams. lzma header too big below ~1000 bytes; bz2 (BWT) weaker.
- **Own deflate optimiser, stronger than Zopfli, ~+700 points vs native zlib**: hill climbing on a deflate parser to remove characters that need escaping inside the literal (~1 in 100 bytes); metaheuristics on the Huffman tree failed (rugged, discrete landscape); the breakthrough was that with factorisation and the code-length code fixed, optimal Huffman code lengths can be found by **dynamic programming** (example: Zopfli 218 bytes -> 215).
- **Automatic variable renaming** from static dependency analysis: "the single most impactful optimization, netting several hundred bytes". A GA over (names, deflate payload) gave ~100 bytes more.
- Code-level rules under compression: repeat identical fragments, unroll loops, avoid uppercase names (breaks run-length coding of the Huffman header), reuse letters from keywords, tabs over spaces, `'` over `"`.
- **Seed cracking**: test cases come from ARC-GEN; by reading bits leaked through `randint` calls with power-of-two ranges in tasks 043, 127, 142, they found every task used seed `2025 + example_index`. Only task 096 met their five criteria for regenerating outputs from the seed; they wrote a solution that recovers the seed from grid size, colour set and background and replays the generator (seeds for non-ARC-GEN cases found by exhaustive search).

### Rank 6: Theoretical Syntax Golfers (957,527)
Source: topic 615145; repo https://github.com/Yosshi999/ARCGolfVisualizer .
- Score decomposition: hand golf 955,251; zlib +2,276, of which code optimisation for compression ~400. 5 "diamond" (sole-best) and 58 gold tasks on the community spreadsheet (task 288 at 85 B, 245 at 94 B, 225 at 122 B, 382 at 123 B, 349 at 163 B, 9 bytes under the next best).
- Tooling: Flask + Jinja2 visualiser; submissions managed in git.
- Techniques: BFS (slow but short), unrolled "rotate left", non-ASCII byte strings for constant arrays, loop unrolling + exec.
- Compression: zopfli instead of zlib for code over ~240 B; strip head/tail (`[2:-4]`), `-8` wbits.
- Variable-name optimisation two ways: **static analysis** (liveness -> interference graph; candidates from Optuna, earlier a QUBO solver; ~3 h for all tasks, no test cases needed) then **simulated annealing** (black-box zopfli byte count, each candidate validated on 10 test cases; ~12 h, wider search, better). Static result seeds the SA.
- No LLM use mentioned.

### Rank 7: LogicLynx (957,513)
Source: topic 619321; repo https://github.com/shunsuke-toba/kaggle-google-code-golf-2025/ .
- Strategy chosen up front: 5 people cannot hand-golf 400 tasks, so centre on generative AI. **Codex (web)** returns up to 4 results per prompt; since shorter = better, review is just picking the shortest valid one.
- Semi-automation: prompt generation and PR merging automated; prompt entry and correctness review manual.
- Context given to Codex: general golf knowledge from the code.golf Python wiki; competition-derived techniques (they had the AI analyse their own near-optimal code to extract reusable tricks, kept as knowledge in `AGENTS.md`); a **per-task memo** of experiments already tried, with instructions not to repeat them, cleared whenever the code got shorter (old ideas may work again).
- Prompts written in XML ("more understandable than Markdown"); termination condition "shorten by at least 1 character" to keep Codex working long.
- Manual techniques documented: recursion template `p=lambda g,i=N:-i*g or p(...,i-1)`; rotate/transpose instead of 4-direction logic; eval-based expansion of two-stage recursion with `%s` formatting; regex on `str(g)` with fixed-width offsets (`.{19}` = one row down on width 7); **Chinese Remainder Theorem single loop** `k%a, k%b` over a grid when gcd = 1 (task 240: 10 bytes under any published code); zopfli + Latin-1 literal; variable-name search for 2 to 12 bytes per compressed task; writing repetitive code that compresses better (task 086).
- Comment by nesya (team): AI did very well on the sheer number of tasks and tiny optimisations (task 020 almost entirely AI-improved, still shortest standalone); AI struggled with regex tasks, where humans set the approach.

### Rank 8: import itertools (957,177)
Sources: topic 615039; topic 614376 (jacekwl personal summary); repo https://github.com/Natanaelel/GoogleCodeGolf2025 .
- Mostly non-expert team; "our first instinct was to lean heavily on LLMs. It didn't take long to realize it was not going to get us anywhere near the top". jacekwl used LLMs to get 400/400 solved, then did "99% of work manually"; after reading other writeups he concluded LLMs' ceiling is "much much higher, you just have to use them in a smart way".
- Techniques: rotate-and-recurse cellular automaton for flood fill (task 002) with the update expression found by **pysearch**; `[g:=...for _ in g][3]` loop instead of recursion when no counter needed; `g.pop()` with `g*1` shallow copy; `a*0==0` to tell numbers from rows; 4 nested loops flattened into one with bit ops over 4096; **hash-based special-casing** of 1 to 2 failing edge cases (`hash((*c,))%709==8`); `sum(g,r)` instead of `sum(g,[])`; raw byte strings as index tables; bit-count trick for shape sizes; small brute-force scripts over slice templates to pass failing tests (task 346).

### Rank 9: Santa Golf Club 2500 (957,151)
Source: topic 614150; repo https://github.com/4eta/google-code-golf-2025 .
- Five competitive programmers, no prior code golf; every task "manually implemented and optimized"; over 1,000 hours in three months. Learned from ChatGPT, Gemini and Claude and from the code.golf community, but code written by hand.
- Timeline: August solve all 400 regardless of length; September study public-best solutions; October byte-by-byte.
- Custom dashboard ranking tasks by gap to public best, score gap, staleness, and tasks where lower-ranked teams beat them; sorting by last update to encourage "chasing" each other's partial ideas.
- Tools: ARC-GEN generators, pysearch, garrymoss's compressed-variable-name-optimisation notebook, lynn's flateview.

### Rank 10: lv1.dev (956,458)
Source: topic 628087; repo https://github.com/lv1-dev/google-code-golf-2025 . One line only: "a combination of manual golfing, with iterative refinement by LLM (mainly OpenAI Codex Web)". Repo lists per-task solutions, plain and `.zip.py` compressed variants.

### Rank 11 to 13, 17, 18, 20: no writeup
- 11 cam jam! (intgrah, sammyuri and others; repo listed in 613968, 404 now; intgrah posted their 20 longest solutions in topic 613437, all plain hand-written Python BFS/loops). 12 ﾀｲｽｹﾝｻﾝﾊﾞII. 13 暗黑AGI (boristown's public Chinese hints for all 400 tasks). 17 Golfers Anonymous (solo, Oleg). 18 Team JYCDT (Jiwei Liu, Yiheng Wang, Chris Deotte, Darren Amadeus Martin, Tony Li; Deotte opened the "by hand or by LLM?" thread 608310; Tony Li posted a combined "TOP 78 Tricks" list, 614082, and "Top 100 Tasks Worth Optimizing", 608435). 20 ShadowPrompt Labs (includes shinh and darrenks, by name long-time golf-site authors; UNVERIFIED).

### Rank 14: ALE-Agent (Sakana AI team) (954,571)
Source: repo README https://github.com/SakanaAI/google-code-golf-2025 . No writeup. Repo provides a judge (`uv run judge`, 300 s timeout per task), compress/decompress, a variable-name optimiser (1,000 attempts, up to 8 changes each), scoreboard fetch/compare, and "prompt generators to easily paste prompts into ChatGPT or Claude". Degree of automation not stated; team name refers to Sakana's ALE-Agent (AtCoder heuristic agent), but whether that agent was used here is UNVERIFIED.

### Rank 15: santa2024 (954,246)
Source: topic 614727; compression notebook https://www.kaggle.com/code/denden12/golf-compress-all (topic 613993).
- First code-golf competition for all five. Four tracks: hand, LLM, rule-based, compression.
- **By hand**: LLMs could not solve from scratch or change algorithms, so humans wrote the "initial solutions"; 31 tasks solved by regex; grouped near-identical tasks (154/390, 351/400, 246/335) using a PDF of all tasks.
- **By LLM**: tried GPT, Claude, Gemini, Grok. Early: Claude Code with a DSL to get accepted solutions; after GPT-5's release they switched ("astonishing", many 0->1 solves); final mainly **GPT-5 Codex and GPT-5 Pro** doing "follow-up code-golfing" of strong human seeds. Rationale: AI golfs well once the direction is clear but struggles with 0->1 and local optima; switching human->AI or AI->AI adds diversity to escape them.
- Prompt files per task: many examples of code already at first-place length (taught recursion), before/after golf tips, contest-style test cases, boristown's Chinese hints translated via the PLaMo API, refined with OpenAI's GPT-5 prompt optimiser. Even gpt-5-mini then found a better solution about 1 in 50 tries.
- Custom evaluator for agents: plain and zlib/zopfli sizes, gap to first place (GPT-5 Codex often closed ~50-point gaps), and a **hack filter** rejecting outputs like an object whose `__eq__` always returns true (passes locally, fails on LB).
- API loop: for each task, test cases + current shortest code + 30 to 50 random solutions from other tasks; gpt-5 produced a few-byte gain once every ~50 tasks, gpt-5-mini once every ~100; grok-4 also used.
- **Rule-based**: 23 notebooks running every minute (strip comments/whitespace, True/False -> 1/0, AST-wide one-letter renaming, 1-space indentation, builtin aliases, etc.), plus scheduled AI post-processing notebooks (OpenAI, Gemini, Grok APIs); pysearch for small expressions (task 144: `3*(x==y)` -> `3>>x+y`).
- Compression: sometimes write longer raw code that compresses better (e.g. not aliasing `range`), turned into GPT-5 prompt examples; sped-up variable-name search (cheap compressed-size check first, correctness later).
- Members earlier reported a metric vulnerability privately (isakatsuyoshi, topic 599368).

### Rank 16: mystery+bono+Lucien (953,632)
Source: README https://github.com/m5t0/google-code-golf-2025 . Seeded from two public notebooks ("ARC AGI Code Golf - 216/400 solved" by fedimser; "Optimize 391/400 DSL" by lucvan68); tools `compcheck.py` (tries compressors and parameters per solution) and `varopt*.py` (variable names for compression). LLM use not stated.

### Rank 19: Team (Yuchen20 part) (953,039)
Source: topic 614729 + blog. Bank has a stub; additions:
- Agent = **Evolutionary DB** (per task: best solution, public best, all verified longer variants) + **context engineering** + lean loop (propose, extract, validate entry point and stdlib-only, run tests, return traceback and failing I/O, minify with python-minifier, keep shortest). Focus task sampled by softmax over gap to public best (large constant if unsolved); in-task priors sampled with inverse-length bias; cross-task exemplars random at first, later by pattern-tag similarity, enabling "crossover".
- Mostly **gemini-2.5-flash on the free tier**, some OpenRouter tests.
- First prompt with ARC-GEN generator + arc-dsl solution + examples solved about half the tasks; plateau traced to arc-dsl shortcuts that fail ARC-GEN's extra examples; **removing arc-dsl from the prompt** took it to ~360 solved. Rest by hand or vetted public code.
- Golf idioms distilled by ChatGPT agent mode from an archived golf site with visible ranked submissions; the evolutionary DB itself was the best teacher (tasks 246/335 got near-identical solutions).
- Context rot: ~10 I/O pairs + 8 exemplars + ~10 priors made the model drift (fewer used later). Plateau at ~0.940 normalised, then manual; teammates did zlib, AST normalisation, smaller models with golf tricks, Claude Code.

## Technique frequency matrix

Y = stated, N = stated not used / clearly implied, ? = no source. Ranks 11 to 13, 17, 18, 20 have no source and are omitted from counts.

| Technique | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mainly hand-golfed | Y | Y | Y | N | Y | Y | partly | Y | Y | Y | ? | ? | ? | ? | partly | ? | ? | ? | partly | ? |
| LLM writes first drafts (0->1) | N | Y (long tasks) | N | Y | N | N | Y | Y (early) | N (learned from) | ? | ? | ? | ? | ? | Y (GPT-5) | ? | ? | ? | Y | ? |
| LLM golf/refine loop | N | N | N | Y (gpt5-codex) | N | N | Y (Codex Web) | N | N | Y (Codex Web) | ? | ? | ? | Y? (prompt generators) | Y (GPT-5 Pro/Codex, API) | ? | ? | ? | Y (Gemini) | ? |
| Best-of-N sampling + auto verify | N | N | N | Y (N=4) | N | N | Y (4 outputs) | N | N | ? | ? | ? | ? | ? | Y | ? | ? | ? | Y | ? |
| Persistent memory of tried ideas | ? | ? | ? | Y (SQLite logs) | ? | ? | Y (per-task memo) | ? | Y (dashboard) | ? | ? | ? | ? | ? | Y (prompt files) | ? | ? | ? | Y (evolutionary DB) | ? |
| zlib / zopfli compression | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | ? | ? | ? | Y | Y | Y | ? | ? | Y | ? |
| Custom deflate re-encoder beyond Zopfli | Y | ? | ? | ? | Y | N | N | ? | N | ? | ? | ? | ? | ? | N | ? | ? | ? | ? | ? |
| Variable-name optimisation for compression | Y | Y (GA) | ? | ? | Y (static + GA) | Y (static + SA) | Y | Y | Y (garrymoss) | ? | ? | ? | ? | Y | Y | Y | ? | ? | ? | ? |
| pysearch / expression brute force | ? | ? | Y | ? | ? | ? | ? | Y | Y | ? | ? | ? | ? | ? | Y | ? | ? | ? | ? | ? |
| Char-level fuzzer / GA on source | ? | Y | ? | ? | Y (GA on payload) | N | N | N | N | ? | ? | ? | ? | ? | N | ? | ? | ? | ? | ? |
| Rule-based minifier pipeline | N | ? | ? | Y (AST-rule prompts) | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | Y (23 rules) | ? | ? | ? | Y (python-minifier) | ? |
| Regex on str(grid) | Y | ? | ? | Y (prompted) | Y | ? | Y | Y | ? | ? | ? | ? | ? | ? | Y (31 tasks) | ? | ? | ? | N (LLMs never used it) | ? |
| Seed cracking / generator exploitation | ? | ? | ? | ? | Y | N | N | N | N | ? | ? | ? | ? | ? | N | ? | ? | ? | ? | ? |
| Hash/hardcode edge cases | Y | ? | ? | ? | ? | ? | ? | Y | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |

Counted summary (14 teams with a source: 1 to 10, 14, 15, 16, 19):
- 14 of 14 used zlib/zopfli compression on long tasks (rank 10 via the `.zip.py` files in its repo).
- 10 of 14 explicitly optimised variable names for compressibility.
- 7 of 14 used an LLM in a systematic loop or for drafts (2, 4, 7, 8 early, 10, 15, 19); only 3 of those finished in the top 7 (2 drafts only, 4, 7).
- 6 of 14 were essentially hand-golfed with no LLM in the final code path (1, 3, 5, 6, 9, 16 as far as stated).
- 2 of 14 built a deflate encoder beyond Zopfli (1, 5).

## Consensus vs divergence

- Table stakes: hand-verified correctness across all ARC-GEN examples; the recursion/rotation template; compression of anything over ~200 to 240 bytes with a Latin-1 literal and raw-deflate stub; variable renaming for compressibility; the community per-task spreadsheet (topic 596679) as the shared benchmark.
- What separated top 3 from 4 to 20: about 4,000 points (1st 962,070 vs 4th 957,810, ~10 bytes/task). Writeups attribute it to expert golf knowledge (code.golf veterans and a seasoned golfer in 2nd) rather than tooling; 5th to 10th, packed within 1,350 points, are CP/CTF teams new to golf or LLM-assisted teams.
- Contradiction 1 (LLM value): import itertools (8th) says LLMs were "not going to get us anywhere near the top" and "in most cases they were unhelpful" (615039); LogicLynx (7th, 619321) built their whole workflow around Codex and finished one place ahead; santa2024 (15th, 614727) found GPT-5 "astonishing" for 0->1 but weak at escaping local optima; JoKing (3rd) predicted LLMs would miss bitwise tricks, pysearch formulas, regex and recursion (608310).
- Contradiction 2 (regex): Yuchen20 (19th) never saw LLMs use regex (608310); FuunAgent (4th) got regex by rule-prompting it from the AST (bank); LogicLynx's teammate says AI struggled on regex tasks and humans had to set the approach (619321).
- Contradiction 3 (compression optimiser search): HIMAGINE (5th) found GA/SA on Huffman trees underperformed and switched to exact DP (614225); Theoretical Syntax Golfers (6th) found SA over variable names beat static analysis (615145); jailctf (2nd) used a GA for renaming (bank). The disagreement is about what is searched: trees (discrete, DP wins) vs names (SA/GA fine).

## Aggregated "what did not work"

- Leaning on LLMs to reach the top tier in a saturated field (8th).
- LLMs for 0->1 algorithm changes and escaping local optima (15th); for regex-style tasks (7th comment, 19th).
- arc-dsl solutions in the LLM prompt: confused the model because they fail ARC-GEN's extra examples (19th).
- Too much context (10 I/O pairs, 8 exemplars, 10 priors): context rot (19th).
- GA / simulated annealing on deflate Huffman trees (5th).
- lzma below ~1000 bytes (header) and bz2 (5th).
- Aliasing builtins under compression can lose bytes (15th; 614012 observation).
- LLM "hack" outputs that pass locally but fail the server (object with always-true `__eq__`) (15th).
- Seed regeneration: only one task of 400 met the criteria (5th).
- LLM-only agent: plateau around 0.940 normalised (19th).

## Shake-up note

None: the leaderboard is 100% public and the API gives identical public and private ranks for all top-20 teams. The only late movements came from public solution releases in the final days (c-number asked people to pause releases, topic 613583) and metric fixes earlier (bank).

## UNVERIFIED

- Methods of ranks 11, 12, 13, 17, 18, 20 (no writeups; repos for 11 and 12 now 404).
- Whether Sakana's ALE-Agent system (vs humans with prompt generators) produced rank 14's code.
- Whether ranks 1 and 3 used any LLM at all beyond what the bank records (writeups say no or are silent).
- lv1.dev's split of LLM vs manual work.
