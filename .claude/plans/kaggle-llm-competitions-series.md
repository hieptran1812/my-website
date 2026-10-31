# Learn from Kaggle: LLM Competitions (every competition, every winning idea)

**Subcategory:** `learn-from-kaggle` (EXISTING folder, renders "Learn from Kaggle"; holds the AIMO-2 solutions post and will hold the Game-Agents series)
**Path:** `content/blog/machine-learning/learn-from-kaggle/<slug>.md`
**frontmatter:** `category: "machine-learning"` · `subcategory: "Learn from Kaggle"`
**Size:** **41 posts, 9 tracks (A to I).** **STATUS 2026-10-31: 38/41 SHIPPED (A1, B1–B7, C1–C5, D1, D2, D4–D6, E1–E5, F1–F8, H1–H6, I1). Remaining DEFERRED: D3 ARC Prize 2026 (winners 2026-12-04), G1 Gemma 4 Developer Agent (closes 2026-12-02), G2 Name That LLM (closes 2027-01-13): re-fetch bank sections first, then remove each from I1's to-come box.**
**Language:** English (repo rule; state it in every agent brief). Request came in VN: *viết writeup chi tiết, dễ hiểu cho từng cuộc thi dựa vào writeup các đội đã publish và discussion; trình bày dễ hiểu, chi tiết, hình minh hoạ trực quan.*
**Skill:** `blog-writer` (ML engine).

### Already covered elsewhere: link, do not rewrite

- **AIMO Progress Prize 2** has its own post: `learn-from-kaggle/aimo-progress-prize-2-published-solutions.md`. C1/C2 and H4 link to it.
- **LLM 20 Questions** is post C3 of the Game-Agents series (`.claude/plans/kaggle-game-agents-series.md`). A1 and H5 link to it.

### Source of truth: the writeup banks (read only YOUR competition's H2)

| Bank | Words | Sections |
|---|---|---|
| `.claude/plans/kaggle-llm-writeup-bank-part1.md` | 13.0k | LLM Science Exam, Detect AI Generated Text (DAIGT), Prompt Recovery, LMSYS Chatbot Arena, WSDM Cup Multilingual Arena, PII Data Detection |
| `.claude/plans/kaggle-llm-writeup-bank-part2.md` | 14.9k | AIMO 1, AIMO 2 (stub), AIMO 3, Eedi, MAP, ARC Prize 2024, ARC Prize 2025, ARC Prize 2026 (ongoing), Konwinski Prize |
| `.claude/plans/kaggle-llm-writeup-bank-part3.md` | 17.0k | Santa 2024, Drawing with LLMs, Jigsaw Agile Community Rules, Make Data Count, Google Code Golf 2025, NeuroGolf 2026, Name That LLM (ongoing), You Can't Please Them All, Nemotron Reasoning Challenge, Deep Past (Akkadian), Gemma 4 Developer Agent (ongoing), Autonomous Agent Prediction, Considered-and-excluded table |
| `.claude/plans/kaggle-llm-writeup-bank-part4.md` | 13.2k | Hackathons: gpt-oss-20b red-teaming, Gemma 3n, OpenAI to Z, Tunix, BigQuery AI, Gemini Long Context, Gemma language tuning, course capstones, Gemini 3 vibe code, MedGemma, Measuring AGI, Gemma 4 Good, Nano Banana, Data Assistants with Gemma, Gemma 4 Dev Agent paper track, Game Arena context, cross-hackathon lessons |

**Top-20 addenda (user requirement 2026-10-28: every competition post synthesises the writeups of the TOP 20 teams):** `.claude/plans/kaggle-llm-top20/<competition-slug>.md`, one per competition: top-20 table, per-team synthesis for ranks 4 to 20, technique frequency matrix, consensus vs divergence, aggregated what-did-not-work, shake-up. Hackathons: one file each covering every prize winner. Brief used to build them: `kaggle-llm-top20/_BRIEF.md`. A drafting agent reads its competition's bank H2 (ranks 1 to 3 deep) **plus** its top-20 file.

Every rank, team, score, team count, prize, hardware limit and technical claim must come from the bank, the top-20 file or from the writeup URL it cites. **Drafting agents Grep the bank for their H2 and read with offset/limit; never read a whole bank.** UNVERIFIED items are dropped or stated as unverified, never upgraded to fact. Raw API JSON and fetched writeups from the bank build are in `/private/tmp/claude-501/kaggle-bank-p{1..4}/` (may be gone; to re-fetch, use `.claude/scripts/kaggle-api.sh`, see its header).

---

## Angle: what three years of Kaggle LLM competitions actually rewarded

From the banks, a story that no single writeup tells:

> **The model was almost never the edge.** In 2023 to 2024 the winners were decided by **retrieval and data** (Science Exam: RAG over Wikipedia; DAIGT: a synthetic corpus that matched the hidden test distribution; Eedi/MAP: synthetic misconceptions). From 2024 the edge moved to **fitting more model into a fixed notebook** (LMSYS/WSDM: distil 70B into 9B, quantise, TTA) and then to **test-time compute** (AIMO: sampling, tool-integrated reasoning, voting, early stopping; ARC: test-time training). Throughout, a parallel thread of **metric exploits** (Prompt Recovery "lucrarea", You Can't Please Them All judge gaming, LMSYS leak) taught hosts and players what an LLM metric can and cannot measure. And in the most "LLM" competitions of all (Santa 2024, Code Golf), classical search and human golfers still beat LLM-driven entries.

Every competition post delivers three things:

1. **The task explained like you'd explain it to a friend, then formally.** Input, output, metric with a worked toy score, the notebook budget (GPU, hours, internet off), what the hidden test is.
2. **The top-20 synthesis, then the winner rebuilt from intuition up.** All 20 teams in one scorecard (rank, team, score, public-rank move, key idea, Source; `no writeup` stated plainly). A technique frequency matrix drawn as a figure ("14 of 17 teams with writeups used X"), the table stakes everyone shared, and what separated the top 5 from 6 to 20. Then the 1st-place pipeline taken apart until a reader could build a baby version; 2nd and 3rd in depth; ranks 4 to 20 grouped by approach family with each team's distinguishing idea in 2 to 4 sentences; the best contrasting entry (biggest shake-up, best single model, a zero-training baseline such as CPMP's in Science Exam).
3. **Lessons that transfer**, 4 to 6, each traceable to a writeup sentence, plus an "If you enter a competition like this" checklist.

### Honesty rule (hard gate)

- No invented teams, ranks, scores, LoRA ranks, GPU counts or dataset sizes. Where a source disagrees (e.g. MindsAI 15.42 vs API 12.64 in ARC 2025), use the bank's chosen value and note the conflict in one line.
- Code is an **illustrative toy version**, labelled as such, never presented as the team's code. Link the real repo when the bank has one.
- No Kaggle screenshots or writeup images (license unclear). Redraw every pipeline, data flow and chart in Excalidraw.
- Hackathon winners announced **unranked** stay unranked. No inferred rubric scores.
- Red-teaming post (F1): describe vulnerability *classes* and methodology at the level of the published findings; no working jailbreak prompts reproduced verbatim.

---

## Gates

- **Competition and hackathon posts (Tracks B to F): 4,000 to 5,000 prose words** (`WORD_MIN=4000 WORD_MAX=5000`; raised from 3 to 4k on 2026-10-28 because each post now covers the top 20 teams). **Orientation (A1), technique posts (Track H) and capstone (I1): default 2,000 to 3,000.**
- Figures: **>= 6 floor, target 7** (one is always the top-20 technique matrix), >= 1/2 pictorial (data samples redrawn, score/LB charts, token/attention sketches, grids for ARC, SVG drawings, GPU memory bars), box-flow <= 1/3 and max 2. 0 or 1 animated figure where motion is the point (self-consistency vote filling up, ILS kick-and-descend, test-time training loop).
- Verify: `WORD_MIN=4000 WORD_MAX=5000 bash .claude/skills/blog-writer/scripts/verify-post.sh <post.md> <slug> deep-dive` via `post-verifier`.
- No em dash anywhere (no `—`, no spaced ` – `, no ` -- `). Note the banks use `--` and `—`: never copy that punctuation.
- LaTeX traps: brace-wrap inline math starting with a digit; `\lt ` not bare `<letter`. Metrics (MAP@3, F-beta, sharpened cosine, log loss, AIMO penalised accuracy, Code Golf score) are written in `$$...$$`.
- Renderer lives outside the repo (`/Users/hieptran1812/Documents/mcp_excalidraw/scripts/`); count webp files, never trust a green sharpness gate.
- Check the slug is not already on disk before the figure pipeline.
- Recompute every worked number in the post (verify gates never check arithmetic).
- Cross-link existing posts instead of re-explaining: LoRA/fine-tuning, RAG/retrieval, quantization, speculative decoding, self-consistency/test-time compute, reward models, the AIMO-2 post, the Game-Agents series.

### Recurring visual language (define in A1, reuse everywhere)

1. **The lever strip.** Five boxes `Data | Retrieval | Model fit (distil/quantise) | Test-time compute | Metric insight`, the winner's decisive lever lit. Opens every competition post.
2. **The era timeline.** 2023 to 2026, one dot per competition coloured by decisive lever, current competition ringed.
3. **The budget card.** The notebook box: GPU type and count, VRAM, hours, internet off, plus the hidden test size. First pictorial figure in each post.

### Fixed post skeleton (competition posts)

1. TL;DR box (winner, decisive lever, the one lesson) + lever strip
2. The task in 60 seconds (budget card + one redrawn data sample)
3. The metric, with a worked toy score
4. What made it hard (the constraint that decided it)
5. Top-20 scorecard table: rank, team, private score, public-rank move, key idea, Source
6. The top-20 landscape: technique frequency matrix figure, table stakes, what separated top 5 from 6 to 20, contradictory findings
7. The winning solution, explained (intuition, mechanism, toy code)
8. 2nd and 3rd in depth, then ranks 4 to 20 grouped by approach family, plus the best contrasting entry
9. What the forum taught everyone (key discussion threads, public notebooks, leaks, shake-up)
10. What did not work (aggregated across all top-20 writeups, with who reported it)
11. Lessons + "If you enter a competition like this" checklist
12. Sources (every top-20 writeup and repo URL)

Hackathon posts (Track F) swap 3 to 6 for: the brief and the judging rubric with weights; the winners table; a feature matrix across all winners; every prize winner explained (problem, data, model + tuning recipe, demo), the top 3 to 5 in depth.

---

## Track A: Orientation (1 post)

| # | Slug | Post |
|---|---|---|
| A1 | `kaggle-llm-competitions-how-they-work` | Code competitions vs hackathons; the offline notebook budget (2xT4 / P100 / L4x4 / H100 by era); hidden test sets and the API-served problem loop; public vs private LB and shake-ups; why LLM metrics are fragile. Introduces the three recurring visuals and the series thesis. Map of all 41 posts. |

## Track B: Knowledge, text and preference (2023 to 2025) (7 posts)

| # | Slug | Competition | Bank | Decisive lever / hook |
|---|---|---|---|---|
| B1 | `kaggle-llm-science-exam-solutions` | LLM Science Exam (2023) | p1 §1 | Retrieval: H2O's 30 to 60M-chunk Wikipedia RAG + 7B/13B classifiers with a KV-cache trick; CPMP's zero-training 70B at 54th as the contrast |
| B2 | `kaggle-detect-ai-generated-text-solutions` | LLM Detect AI Generated Text (DAIGT) | p1 §2 | Data: the synthetic corpus that matched the hidden essays decided everything; typo noise as a feature |
| B3 | `kaggle-llm-prompt-recovery-solutions` | LLM Prompt Recovery | p1 §3 | Metric insight: sentence-t5 sharpened cosine, mean prompts and the "lucrarea" token |
| B4 | `kaggle-lmsys-chatbot-arena-solutions` | LMSYS Human Preference Predictions | p1 §4 | Model fit: distil 70B into Gemma-2-9B, the leak and the replacement test set |
| B5 | `kaggle-wsdm-cup-multilingual-arena-solutions` | WSDM Cup Multilingual Chatbot Arena | p1 §5 | What carried over from LMSYS and what multilinguality changed |
| B6 | `kaggle-pii-data-detection-solutions` | PII Data Detection | p1 §6 | LLM-generated training essays for a DeBERTa token classifier; F-beta 5 rewards recall |
| B7 | `kaggle-llms-you-cant-please-them-all-solutions` | LLMs - You Can't Please Them All | p3 | Adversarial essays against hidden LLM judges: what judge disagreement can be gamed |

## Track C: Math and reasoning (5 posts)

| # | Slug | Competition | Bank | Decisive lever / hook |
|---|---|---|---|---|
| C1 | `kaggle-aimo-progress-prize-1-solutions` | AIMO Progress Prize 1 | p2 §1 | Test-time compute: Numina's two-stage SFT + SC-TIR N=48; CMU's reward-model vote; "after exams" un-finetuned 3rd |
| C2 | `kaggle-aimo-progress-prize-3-solutions` | AIMO Progress Prize 3 | p2 §3 (+§2 stub) | Two-run penalised accuracy on H100s; how the 2026 stack differs from AIMO 2 |
| C3 | `kaggle-eedi-mining-misconceptions-solutions` | Eedi Mining Misconceptions | p2 §4 | Retrieve-then-rerank with Qwen2.5 32B/72B + synthetic misconceptions for unseen labels |
| C4 | `kaggle-map-student-math-misunderstandings-solutions` | MAP Charting Student Math Misunderstandings | p2 §5 | LLM classifiers over student explanations; label noise and MAP@3 |
| C5 | `kaggle-nvidia-nemotron-reasoning-challenge-solutions` | NVIDIA Nemotron Model Reasoning Challenge | p3 | Improving a fixed model under greedy decoding and a 7,680-token cap |

## Track D: Abstraction, code and agents (6 posts)

| # | Slug | Competition | Bank | Decisive lever / hook |
|---|---|---|---|---|
| D1 | `kaggle-arc-prize-2024-solutions` | ARC Prize 2024 (ARC-AGI-1) | p2 §6 | Test-time training: the ARChitects' augment, fine-tune per task, score candidates |
| D2 | `kaggle-arc-prize-2025-solutions` | ARC Prize 2025 (ARC-AGI-2) | p2 §7 | Why ARC-AGI-2 cut scores by 2 to 3x; what the 2025 winners added |
| D3 | `kaggle-arc-prize-2026-solutions` | ARC Prize 2026 (AGI-2, AGI-3, paper track) | p2 §8 | **DEFERRED to Wave 7: final 2026-11-02, papers 11-09, winners 12-04. Re-fetch before drafting.** |
| D4 | `kaggle-konwinski-prize-solutions` | Konwinski Prize | p2 §9 | Contamination-free SWE-bench: why the winner fixed ~7.5% and the scoring that punishes wrong patches |
| D5 | `kaggle-google-code-golf-2025-solutions` | NeurIPS 2025 Google Code Golf | p3 | Humans 1 to 3, best LLM-driven team 4th: what LLMs did and did not contribute |
| D6 | `kaggle-neurogolf-2026-solutions` | NeuroGolf 2026 | p3 | Smallest network that solves ARC tasks: the log-parameter score |

## Track E: Generation, search and extraction (5 posts)

| # | Slug | Competition | Bank | Decisive lever / hook |
|---|---|---|---|---|
| E1 | `kaggle-santa-2024-perplexity-puzzle-solutions` | Santa 2024 Perplexity Permutation Puzzle | p3 | An LLM as the objective, search as the solver: ILS, k-opt, a tiny NN pruning Gemma calls; ten teams tied at the optimum |
| E2 | `kaggle-drawing-with-llms-solutions` | Drawing with LLMs | p3 | SVG generation scored by VQA + aesthetics: diffusion then vectorise vs LLM-writes-SVG, and metric gaming |
| E3 | `kaggle-jigsaw-agile-community-rules-solutions` | Jigsaw Agile Community Rules | p3 | Rule-conditioned classification with per-rule AUC; test-time training on each rule's examples |
| E4 | `kaggle-make-data-count-solutions` | Make Data Count | p3 | LLM extraction vs external citation data + CatBoost: where the points really came from |
| E5 | `kaggle-deep-past-akkadian-translation-solutions` | Deep Past (Akkadian to English) | p3 | LLMs as data engineers, ByT5 as the model |

## Track F: Hackathons, judged not scored (8 posts)

| # | Slug | Hackathon(s) | Bank | Hook |
|---|---|---|---|---|
| F1 | `kaggle-gpt-oss-red-teaming-challenge-findings` | OpenAI gpt-oss-20b Red-Teaming | p4 §1 | Vulnerability classes the winners found (CoT forgery, reward hacking, deception), the severity/breadth/novelty rubric |
| F2 | `kaggle-gemma-impact-hackathons-winners` | Gemma 3n Impact, Gemma 4 Good, MedGemma Impact | p4 §2, 10, 12 | On-device and domain Gemma: what made impact projects win |
| F3 | `kaggle-openai-to-z-challenge-winners` | OpenAI to Z | p4 §3 | LLMs + LiDAR/satellite to find Amazon sites: evidence and reproducibility |
| F4 | `kaggle-tunix-hack-reasoning-winners` | Tunix Hack | p4 §4 | Teaching Gemma to show its work with RL on TPUs; judged by an LLM + humans |
| F5 | `kaggle-gemma-gemini-early-hackathons-winners` | Data Assistants with Gemma, Gemini Long Context, Unlock Global Communication with Gemma | p4 §6, 7, 14 | 2024 era: fine-tuning Gemma for languages, using a 2M context |
| F6 | `kaggle-bigquery-ai-hackathon-winners` | BigQuery AI | p4 §5 | LLM functions inside SQL: three categories, three winners each |
| F7 | `kaggle-measuring-agi-cognitive-abilities-winners` | Measuring Progress Toward AGI | p4 §11 | Building benchmarks that discriminate between frontier models |
| F8 | `kaggle-gemini-vibe-coding-and-capstones` | Gemini 3 vibe code, Nano Banana, Gen AI / Agents / Vibe Coding capstones, Game Arena context | p4 §8, 9, 13, 16 | 48-hour and one-week sprints: what a winning demo looks like; names-only lists stay names-only |

## Track G: Ongoing, publish after close (2 posts, DEFERRED)

| # | Slug | Competition | Earliest draft |
|---|---|---|---|
| G1 | `kaggle-gemma-4-developer-agent-solutions` | Gemma 4 Developer Agent (+ paper track) | after 2026-12-02 (re-fetch bank, p3 + p4 §15) |
| G2 | `kaggle-name-that-llm-solutions` | Name That LLM | after 2027-01-13 (re-fetch bank, p3) |

## Track H: Cross-cutting techniques (6 posts, 2–3k words)

Syntheses drawing only on the banks.

| # | Slug | Post |
|---|---|---|
| H1 | `kaggle-llm-rag-under-a-notebook-budget` | Retrieval inside an offline notebook: corpus building, BM25 vs dense, rerankers (Science Exam, Eedi, MAP, Make Data Count) |
| H2 | `kaggle-llm-synthetic-data-as-the-edge` | Synthetic data that matches the hidden test (DAIGT, PII, Eedi, AIMO, ARC, Deep Past) |
| H3 | `kaggle-llm-fitting-big-models-in-small-gpus` | Distillation, quantisation (GPTQ/AWQ/4-bit), vLLM, KV-cache tricks, layer-by-layer 70B, time managers (LMSYS, WSDM, Science Exam, AIMO) |
| H4 | `kaggle-llm-test-time-compute-playbook` | Sampling, voting, reward-model rerank, tool-integrated reasoning, early stopping, test-time training (AIMO 1/2/3, ARC, Jigsaw, Nemotron) |
| H5 | `kaggle-llm-metric-exploits-and-shake-ups` | lucrarea, judge gaming, leaks, public-private shake-ups; what each taught hosts (Prompt Recovery, You Can't Please Them All, LMSYS, Science Exam, Drawing with LLMs, 20 Questions) |
| H6 | `kaggle-llm-as-classifier-fine-tuning-recipes` | Turning Gemma/Qwen/Llama into classifiers: heads, LoRA settings, TTA by swapping A/B, CV schemes (LMSYS, WSDM, MAP, Jigsaw, Eedi) |

## Capstone (1 post)

| # | Slug | Post |
|---|---|---|
| I1 | `kaggle-llm-competitions-playbook` | All competitions on the era timeline by decisive lever; the recurring winners (H2O, NVIDIA NemoSkills, the ARChitects...) and what they carried forward; a decision playbook: given a new LLM competition's metric, budget, data size and test-set source, which lever to pull first |

---

## Waves (fresh session per wave; dispatch through `figure-author` / `post-verifier`)

| Wave | Posts | Bank sections |
|---|---|---|
| W1 | A1, B1, B2, B3, B4, B5 | p1 §1 to 5 (A1 uses all four summary tables only) |
| W2 | B6, B7, C1, C2, C3, C4 | p1 §6, p3 You Can't Please Them All, p2 §1 to 5 |
| W3 | C5, D1, D2, D4, D5, D6 | p3 Nemotron/Code Golf/NeuroGolf, p2 §6, 7, 9 |
| W4 | E1, E2, E3, E4, E5, F1 | p3, p4 §1 |
| W5 | F2, F3, F4, F5, F6, F7 | p4 |
| W6 | F8, H1, H2, H3, H4, H5 | p4 §8, 9, 13, 16 + Grep only of published posts' scorecards |
| W7 | H6, I1, D3, G1 (+ G2 after 2027-01-13) | re-fetch ARC 2026 / Gemma 4 Dev Agent / Name That LLM first |

**Open gaps to re-check before the relevant wave:** ARC 2024 3rd place (alijs) details; AIMO 3 5th/7th writeups and why ranks 4 and 6 took no prize; Eedi/MAP places 4 to 6; LMSYS replacement test-set size; Prompt Recovery missing 6th-place team; PII 3rd (secondary source only); WSDM public scores (API gives 0); AI Agent Security competition (excluded, no method section); hackathon projects with no technical detail (OpenAI to Z 4th/5th, Tunix 4th/6th).

**Correction from top-20 pass (2026-10-28):** Konwinski bank (p2 §9) reads 0.0165 scores as "~2 correct, 0 wrong"; the decoded triples in `kaggle-llm-top20/konwinski-prize.md` (2nd = 7 right / 5 wrong, 3rd = 6/4) supersede it. AIMO3 ranks 1 to 4 tie at 44.0 and 5 to 20 at 43.5: present the order as noise.
**Hackathon bank corrections (top-20 pass 2026-10-28, supersede p4):** Swahili scores in p4 are wrong; Sports Sponsorship never ran its full 1.7M-token match; IDEA-E has no SFT stage; UniRad3s fine-tuned MedSigLIP, not MedGemma. Use the `kaggle-llm-top20/<hackathon>.md` files. Every hackathon number is the team's own claim: say so.
