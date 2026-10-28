---
title: "How Kaggle LLM competitions actually work: notebooks, hidden tests and the five levers that win"
date: "2026-10-28"
publishDate: "2026-10-28"
description: "The orientation post for a 41-post series on every Kaggle LLM competition: what a code competition really scores, the offline notebook budget, public versus private shake-ups, why LLM metrics break, and the five levers that decided three years of winners."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "leaderboard", "evaluation", "test-time-compute", "retrieval", "synthetic-data", "metrics"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 13
---

Between 2023 and 2026 Kaggle ran more than twenty competitions where a large language model was the model, the judge, the objective or the thing being attacked. Tens of thousands of teams entered. The winners published what they did, and if you read those writeups side by side a pattern falls out that no single writeup states: **the model was almost never the edge.** Most top teams had access to the same open weights. What separated them was what they built around the model, inside a box that Kaggle defines very tightly.

![The lever strip: five levers, one lit per competition](/imgs/blogs/kaggle-llm-competitions-how-they-work-1.webp)

The figure above is the mental model for this whole series. There are five levers a team can pull: **data** (build a training set that looks like the hidden test), **retrieval** (bring the right text into the context), **model fit** (squeeze a bigger model into the notebook through distillation and quantisation), **test-time compute** (sample, vote, verify, retrain at inference) and **metric insight** (understand exactly what the scorer rewards). In each competition one lever decided the money, and it was a different lever almost every time. Every competition post in this series opens with this strip, lit for its winner.

This orientation post covers the contract Kaggle offers, the hardware box, hidden tests and the two leaderboards, why LLM metrics are fragile, and how the other 40 posts are organised. Every number comes from the series' verified writeup banks, built from Kaggle's own competition, leaderboard and discussion API.

## 1. Two contracts: code competitions and hackathons

![Code competition versus hackathon: two scoring contracts](/imgs/blogs/kaggle-llm-competitions-how-they-work-2.webp)

In a **code competition** you never upload predictions. You upload a notebook, and Kaggle reruns it on its own machine, with internet off, against test data you never see. All six competitions in the first writeup bank (Science Exam, Detect AI Generated Text, Prompt Recovery, LMSYS, WSDM Cup and PII Detection) followed the same rules: freely available external data and pretrained models allowed, a maximum team size of 5 and 5 submissions per day. Some competitions go further and serve the test through an evaluation API. In AIMO Progress Prize 1 the problems "arrive one at a time in random order through a Python evaluation API", and the 50 private problems are served only at the final rerun. Your notebook cannot look ahead, cannot batch the whole test set, and has to budget its clock per problem without knowing which problems are hard. The Konwinski Prize pushed this further still: each GitHub issue had to be resolved within 30 minutes of being served.

In a **hackathon** there is no leaderboard. A team submits a writeup, a video and usually a notebook or demo, and judges score it against a published rubric. The Gemma 3n Impact Challenge (599 teams, a USD 150k prize pool) weighted Impact 40, Video 30 and Technical depth 30. The gpt-oss-20b red-teaming challenge split USD 500k into ten USD 50k awards and scored each finding on severity, breadth, novelty, reproducibility and methodological insight. Two consequences follow. Performance numbers in a hackathon writeup are the team's own claims, never re-measured by the judges. And winners announced unranked stay unranked in this series.

| | Code competition | Hackathon |
|---|---|---|
| What you submit | A notebook that Kaggle reruns | Writeup, video, notebook or demo |
| What is scored | Hidden test, by a formula | Your submission, by judges and a rubric |
| Who can check a number | The scorer, every submission | Nobody after the fact; claims are self-reported |
| What transfers to your job | Engineering under a hard budget | Problem framing and communication |
| Series tracks | B, C, D, E, G | F |

## 2. The notebook is the budget

The most important fact about a Kaggle LLM code competition is the box the notebook runs in. Everything clever that winners did responds to it.

![The budget card for four eras of Kaggle notebook hardware](/imgs/blogs/kaggle-llm-competitions-how-they-work-3.webp)

Notice how much the hardware changed and how little the clock did. For most of 2023 and 2024 the standard Kaggle accelerator was two T4s with 16 GB each, or a single P100, for 9 hours. LLM Science Exam participants said 70B models "can't fit a T4 or a P100 GPU", then ran them anyway. AIMO Progress Prize 2 moved to four L4s and a 5-hour cap. ARC Prize 2025 offered L4x4 machines with 96 GB, at roughly USD 50 of quota per submission, for 12 hours. AIMO Progress Prize 3 added H100 machines usable only for that competition, again under a 5-hour GPU limit, and ARC Prize 2026 added RTX Pro 6000 machines with 96 GB for its ARC-AGI-3 track.

Two details on the card are easy to miss. Kaggle obfuscates runtimes (by up to 15 minutes in Prompt Recovery and LMSYS, up to 30 minutes in AIMO 1) so timing a submission tells you little about the hidden test. And several competitions ran a separate **efficiency prize**: in Detect AI Generated Text the efficiency score added `RuntimeSeconds / 32400` to an accuracy term, and 32,400 seconds is exactly 9 hours. Speed was literally on the scoreboard.

Every competition post draws this card: GPU type and count, VRAM where the source states it, hours, internet off, hidden test size. When a winner distils, quantises or stops sampling early, the card is the reason. For the general techniques, [distillation for LLMs](/blog/machine-learning/large-language-model/distillation-in-llm) and [weight-only quantisation with GPTQ and AWQ](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq) cover them outside the Kaggle setting.

## 3. Hidden tests, two leaderboards and the shake-up

During a code competition you see a **public leaderboard** computed on a slice of the hidden test. The money is paid on the **private leaderboard**, computed on the rest, revealed only at the end. The public slice is small by design: 20% of the test in LLM Science Exam, 15% in Prompt Recovery, 25% in PII Detection, 28% in Eedi. When the private ranking differs a lot from the public one, Kagglers call it a shake-up.

![Public rank versus private rank in three competitions](/imgs/blogs/kaggle-llm-competitions-how-they-work-4.webp)

The three panels above are three real outcomes. In Prompt Recovery the public top 10 was exactly the private top 10, just reordered. In LLM Science Exam, H2O won both boards, but the public second place finished 12th and the public sixth place rose to 2nd; one participant noted that MAP@3 on roughly 800 public questions "is a bit unstable". Detect AI Generated Text was one of the biggest shake-ups in Kaggle NLP: the public leader finished 9th, a public 4th finished 163rd, and the eventual 6th and 7th places climbed from public ranks 2,995 and 2,357.

There are two different causes hiding in those pictures. One is plain sampling noise, which you can reason about before you submit anything. Here is an illustrative toy, not any team's code: two models whose true accuracies differ by half a point, scored on a 4,000-item test with a 20% public slice.

```python
import numpy as np

rng = np.random.default_rng(0)

N_TEST = 4000          # hidden test items (toy)
PUBLIC_FRAC = 0.20     # share scored on the public leaderboard
ACC_A, ACC_B = 0.900, 0.905   # true accuracy: B really is better
TRIALS = 20_000

n_pub = int(N_TEST * PUBLIC_FRAC)
n_priv = N_TEST - n_pub
flips_pub = flips_priv = 0
for _ in range(TRIALS):
    # each item is right or wrong independently (a simplification)
    a_pub = rng.binomial(n_pub, ACC_A) / n_pub
    b_pub = rng.binomial(n_pub, ACC_B) / n_pub
    a_priv = rng.binomial(n_priv, ACC_A) / n_priv
    b_priv = rng.binomial(n_priv, ACC_B) / n_priv
    flips_pub += a_pub >= b_pub      # worse model ties or leads in public
    flips_priv += a_priv >= b_priv   # ... and on the private slice

print(f"public slice : {n_pub} items, worse model ties or leads {flips_pub / TRIALS:.1%}")
print(f"private slice: {n_priv} items, worse model ties or leads {flips_priv / TRIALS:.1%}")
```

Running it prints 38.2% for the 800-item public slice and 25.4% for the 3,200-item private slice. A half-point improvement is invisible on the public board more than a third of the time. That is why winning writeups, such as Eedi's top two, tell you to trust a local validation split over a small public board.

The second cause is worse: **fitting the public slice itself.** Detect AI Generated Text had the largest public share of the three (46%) and still the biggest shake-up, because the public leaders built TF-IDF vocabularies on the test essays. General-purpose detectors trained on large, diverse data held up better; a bigger public slice does not protect you from fitting it. AIMO 3 shows a third variant: the private score was the notebook run twice, and ranks 1 to 4 tied at 44.0 out of 50 while ranks 5 to 20 tied at 43.5. The order among tied teams is noise, and the series says so instead of inventing a story.

## 4. Why LLM metrics are fragile

A classical metric scores a label. Many LLM competitions had to score free text, an open-ended answer, or the opinion of another model, and every one of those scorers had a shape that clever teams could exploit.

Take LLM Prompt Recovery. The task was to recover the prompt that turned one text into another, and the score was the sentence-t5-base embedding cosine between your prompt and the true one, raised to the third power:

$$
\text{score} = \frac{1}{N}\sum_{i=1}^{N} \cos\!\left(e(\hat{p}_i),\, e(p_i)\right)^{3}
$$

where $e(\cdot)$ is the sentence-t5 embedding, $\hat{p}_i$ your predicted prompt and $p_i$ the true one. Cubing looks harmless. It is not.

![Toy cubed-cosine scores: bold guess versus bland mean prompt](/imgs/blogs/kaggle-llm-competitions-how-they-work-5.webp)

The toy numbers above show the effect. A bold guess that is right half the time (cosine 0.95) and wrong the other half (cosine 0.30) scores ${0.95^3 = 0.857}$ and ${0.30^3 = 0.027}$, for a mean of 0.442. A bland prompt that is never quite right but never far off, cosine 0.80 everywhere, scores ${0.80^3 = 0.512}$ on every item and wins. That is the mechanism behind the "mean prompt" that top Prompt Recovery teams relied on. It went further: the token string `lucrarea`, a Romanian word piece that sits near the end-of-sequence token in sentence-t5 space, pulled embeddings toward the centre and added roughly 0.05. All top-7 writeups used a mean prompt, `lucrarea`, or both. The winner, Khoi Nguyen, appended eight copies of it to model predictions and scored 0.7165.

Prompt Recovery is not an outlier. In the same three years:

| Competition | What the scorer rewarded | What teams found |
|---|---|---|
| LLM Prompt Recovery | Cubed cosine in one embedding space | Mean prompts and the `lucrarea` token |
| LMSYS Chatbot Arena | Log loss on human preference labels | A data leak; the private set was replaced with unpublished data and rerun |
| Drawing with LLMs | VQA and aesthetic scores from learned judges | Each of three metric versions got a reported exploit, starting with rendered text |
| You Can't Please Them All | Disagreement between hidden LLM judges | Essays engineered to split the judges |
| Konwinski Prize | correct minus wrong minus skipped/10⁴, over total | A wrong patch costs a full point, a skip almost nothing |
| Google Code Golf 2025 | Bytes saved per task | Places 1 to 3 were human golfers; the best LLM-driven team was 4th |

Reading the metric is an engineering skill, not cynicism, and hosts learned from each exploit too. It also means any honest account of a competition has to explain the metric with a worked example before it explains the winning model, which is why every post in this series has a "metric worked on a toy" section.

## 5. The five levers, across three years

Put every competition on one axis, in the lane of its deciding lever, and the era's story appears.

![The era timeline: twenty competitions placed by decisive lever](/imgs/blogs/kaggle-llm-competitions-how-they-work-6.webp)

Read the lanes left to right. In 2023 and early 2024 the winners were decided by **retrieval and data**: LLM Science Exam was won with retrieval over Wikipedia, and Detect AI Generated Text by whoever built a synthetic essay corpus that matched the hidden test. From 2024 the edge moved to **model fit**, getting more model into the fixed notebook: the LMSYS winners distilled a 70B model into Gemma-2-9B, and the WSDM Cup a few months later tested how much of that recipe carried over to many languages. Then **test-time compute** took over: AIMO 1 was won with tool-integrated reasoning and majority voting over 48 samples, ARC Prize 2024 by fine-tuning on each hidden task at inference time, and AIMO 3 by an untrained gpt-oss-120b with a carefully managed inference budget on a single H100. Running alongside all of it is the **metric insight** lane from the previous section, and a stubborn grey lane where classical search or human experts beat the LLM entries outright (Santa 2024 and Code Golf).

The lane assignments are a first call. Each competition post re-derives its winner's lever from the writeups and redraws the timeline with its own dot ringed; Konwinski, the Nemotron reasoning challenge and NeuroGolf get dots when their posts ship. For the general case for spending compute at inference rather than in training, the [test-time compute scaling](/blog/machine-learning/scaling-laws/test-time-compute-scaling) post gives the background; the Kaggle versions are that idea under a 5-hour clock.

## 6. Why every competition post synthesises the top 20

The easy way to cover a Kaggle competition is to retell the first-place writeup, and it misleads. One team's writeup is an anecdote: what one team believed worked, confounded by its code, data and luck. A leaderboard decided by half a point, as section 3 showed, will happily make a mediocre idea look decisive.

Twenty writeups are evidence. If 14 of 17 top teams with writeups used the same retriever, that retriever is table stakes, not the winning idea. If the top 5 all did something that ranks 6 to 20 did not, that is the candidate for what actually separated them. If three teams report that the same trick failed, you can skip it. So each competition post contains:

| Section | What it gives you |
|---|---|
| Scorecard of all 20 | Rank, team, private score, public-rank move, key idea, and the writeup link or a plain "no writeup" |
| Technique matrix figure | Teams as columns, techniques as rows; filled, empty or "?" for unknown; counted as "N of M teams with writeups" |
| Table stakes vs separators | What everyone did, and what only the top 5 did |
| Winner rebuilt | The 1st-place pipeline from intuition to a toy version you could build |
| 2nd and 3rd in depth, 4 to 20 by family | Each team's distinguishing idea in a few sentences |
| What did not work | Aggregated across all writeups, with who reported it |

Two honesty rules follow from that design. A team with no writeup is named as such and never given an invented method; its matrix cells stay "?". And every number derived in a post, from a percentage to an "N of M", is recomputed rather than copied, because the verify gates check structure and never arithmetic.

## 7. The series map

![Series map: 41 posts in nine tracks](/imgs/blogs/kaggle-llm-competitions-how-they-work-7.webp)

The 41 posts are grouped by the kind of problem, not by date, so that competitions that teach the same lesson sit next to each other:

- **A. Orientation** (this post).
- **B. Knowledge, text and preference** (7 posts): [LLM Science Exam](/blog/machine-learning/learn-from-kaggle/kaggle-llm-science-exam-solutions), [Detect AI Generated Text](/blog/machine-learning/learn-from-kaggle/kaggle-detect-ai-generated-text-solutions), [LLM Prompt Recovery](/blog/machine-learning/learn-from-kaggle/kaggle-llm-prompt-recovery-solutions), [LMSYS Chatbot Arena](/blog/machine-learning/learn-from-kaggle/kaggle-lmsys-chatbot-arena-solutions), [WSDM Cup Multilingual Arena](/blog/machine-learning/learn-from-kaggle/kaggle-wsdm-cup-multilingual-arena-solutions), then PII Data Detection and You Can't Please Them All.
- **C. Math and reasoning** (5): AIMO Progress Prize 1 and 3, Eedi, MAP, and the NVIDIA Nemotron reasoning challenge. AIMO Progress Prize 2 already has its own post, [every published AIMO 2 solution](/blog/machine-learning/learn-from-kaggle/aimo-progress-prize-2-published-solutions), and the AIMO posts link to it rather than repeat it.
- **D. Abstraction, code and agents** (6): ARC Prize 2024, 2025 and 2026, the Konwinski Prize, Google Code Golf 2025 and NeuroGolf 2026.
- **E. Generation, search and extraction** (5): Santa 2024, Drawing with LLMs, Jigsaw Agile Community Rules, Make Data Count, and Deep Past (Akkadian to English).
- **F. Hackathons** (8): red-teaming gpt-oss-20b, the Gemma impact hackathons, OpenAI to Z, Tunix, BigQuery AI, the 2024 Gemma and Gemini hackathons, Measuring AGI, and the vibe-coding sprints and course capstones.
- **G. Ongoing** (2): the Gemma 4 Developer Agent competition and Name That LLM, drafted only after they close.
- **H. Cross-cutting techniques** (6): retrieval on a notebook budget, synthetic data as the edge, fitting big models in small GPUs, the test-time compute playbook, metric exploits and shake-ups, and turning LLMs into classifiers.
- **I. Capstone** (1): a decision playbook for which lever to pull first in a new competition.

LLM 20 Questions is covered by the companion Game-Agents series rather than here. If you are short on time, read the five Track B posts in order: between them they show four of the five levers winning within about a year.

## 8. Two more moments that define the genre

**The LMSYS leak (August 2024).** After a leak, the hosts replaced the private set with unpublished data and reran every submission. The private scores sit on a different, harder set than the public ones, which is why public and private scores from that competition cannot be compared directly.

**Code Golf (October 2025).** In a competition about writing the shortest program for ARC-style tasks, places 1 to 3 went to human golfers who used LLMs only for first drafts and tooling, and the best LLM-driven team finished 4th. A useful corrective for anyone who expects the most LLM-shaped competition to be won by the biggest LLM.

## When to reach for this series, and when not to

Reach for it when you face a problem with a hard compute budget, a hidden evaluation set and a metric you did not design, which describes most production LLM work: the levers transfer directly. Read the competition post whose budget card looks most like your hardware, and the Track H post for the lever you think you need.

Do not reach for it to choose a frontier model or to learn what unlimited compute can do. Some winning tricks (a metric token, a leak, a test-fitted vocabulary) are lessons about evaluation design, not recipes, and the series labels them so.
