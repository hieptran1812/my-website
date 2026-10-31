---
title: "Kaggle's first Gemma and Gemini hackathons: what all 14 winners of 2024 actually built"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "Three 2024 Kaggle hackathons, judged not scored: AI Assistants for Data Tasks with Gemma, Gemini Long Context, and Unlock Global Communication with Gemma. Every prize winner and honorable mention, the rubrics drawn to scale, what a 2M-token window was really used for, how five teams taught Gemma 2 a language, and a feature matrix across all 14 winners."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "hackathon", "gemma", "gemini", "long-context", "context-caching", "fine-tuning", "lora", "rag", "dpo", "multilingual"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 22
---

> **TL;DR.** In 2024 Google ran three judged Kaggle hackathons. **AI Assistants for Data Tasks with Gemma** (spring, five categories at 10,000 USD each) paid for tutorials in which a 2B or 7B Gemma 1 did real assistant work; winners paired a small LoRA fine-tune and retrieval with one *visible, checkable* extra, such as a router agent or a before-and-after ROUGE table. **Gemini Long Context** (autumn, four unranked winners at 25,000 USD each) paid for using a window of up to 2M tokens plus context caching; all four winners fed in video and emitted structured output, and two measured caching making things slower or costlier. **Unlock Global Communication with Gemma** (October 2024 to January 2025, five ranked places at 30,000 USD each) paid for adapting Gemma 2 to a language; all five fine-tuned, none touched the tokenizer, and reported gains run from +19 MMLU points on a 120-question sample to a 9B tune that fell below its own base on three of five categories. **The one lesson:** in a judged hackathon the rubric is the metric, and winners turned its abstract words (Informative, Robust, Efficient) into something a judge could see at a glance.

![Three 2024-era hackathons (Feb 2024 to Jan 2025) on one timeline: Data Assistants with Gemma 1 in spring, then Language Tuning with Gemma 2 and Gemini Long Context overlapping in the autumn, with prize structure, entrants and deliverable on each bar](/imgs/blogs/kaggle-gemma-gemini-early-hackathons-winners-1.webp)

The [orientation post](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work) split Kaggle's LLM events into scored competitions and judged hackathons. These are the three earliest Google hackathons in range; their successors are in the [Gemma impact hackathons post](/blog/machine-learning/learn-from-kaggle/kaggle-gemma-impact-hackathons-winners), and the 2024 habits (route, publish the variant, measure each increment) carry straight into them. **Every number about an entry below is the team's own claim from its notebook.** Nobody re-ran them, including me; where a team's numbers disagree with each other, I say so.

## 1. The three briefs in 60 seconds

To a friend: "Google gave away models and asked for the best tutorial notebook showing them off. A panel picked winners." Formally:

| Item | Data Assistants with Gemma | Gemini Long Context | Unlock Global Communication |
| --- | --- | --- | --- |
| Dates | 2024-02-21 to 04-14 | 2024-10-17 to 12-01 | 2024-10-02 to 2025-01-15 |
| Model | Gemma 1 (2B, 7B; 1.1, CodeGemma) | Gemini 1.5 Flash and Pro, up to 2M tokens | Gemma 2 (2B, 9B, 27B) |
| Task | A tutorial in one of five assistant categories | A notebook plus YouTube video using long context and caching | Adapt Gemma 2 to one of 73 languages; publish the model |
| Prizes | 5 x 10,000 USD, one per category | 4 x 25,000 USD, unranked | 5 x 30,000 USD, ranked |
| Field | 6,971 joined | 5,682 joined | 7,019 joined, "hundreds of entries" |
| Also named | 10 honorable mentions, 5 mid-point notebooks | 8 honorable mentions | none |

The pools sum to 300,000 USD. There is no leaderboard, private score or shake-up; the rubric replaces all three.

## 2. The rubrics, drawn to scale

![The three rubrics at one point per unit: both Gemma hackathons score 40 points behind pass/fail gates, while Long Context scores 90, of which 25 points ask why long context and caching mattered](/imgs/blogs/kaggle-gemma-gemini-early-hackathons-winners-2.webp)

Both Gemma hackathons share a template: yes/no gates (Compliant, Topical, Open, plus an eligible language for the language event), then four criteria at 10 points each. **Technical** is "efficient use of few-shot, RAG and/or fine-tuning", **Descriptive** covers dataset curation and documentation, then **Useful**, and **Robust**, "works well when tested with additional inputs". Long Context scored a **notebook out of 50** (Useful, Informative about how and why long context enabled it, Interesting and Documented at 10 each; Efficient caching and Novel at 5 each) and a **video out of 40** (Accurate, Informative, Instructional, Entertaining at 10 each).

Three readings fall out of the weights. The Gemma rubric *names its techniques*, and four of the five language winners combined tuning with RAG or few-shot: teams read it as a checklist. **Robust** is the only criterion that asks what happens off the demo path, and the language winners answered it with base-versus-tuned tables. And Long Context paid for *explanation*: notebook Informative, video Informative and caching add up to 25 of 90 points (the writeup bank says 20, omitting the 5 caching points), which is why every winner opens with a token count.

No formula combines the criteria and no per-criterion scores were published, so any "why it won" line below that is not a host quote is my inference, marked as such.

## 3. Gemini Long Context: a 2M-token window, used on video

### What made it hard

The window was not the bottleneck. Three other limits were: **request timeouts** (two entries failed at their largest input for that reason), the **8,192-token output cap**, and **caching's economics**: a cache needs at least 32,768 tokens, expires on a TTL, and bills storage while it lives.

![Tokens actually placed in context by the Long Context entries on a log scale: from FrameCut's 106,799 up to Prompt to Pull Request's ~900k, with Sports Sponsorship's 1.7M-token full match and PODcast's ~300k audio marked as failed attempts, against the 32,768 caching minimum and the Flash and Pro windows](/imgs/blogs/kaggle-gemma-gemini-early-hackathons-winners-3.webp)

The largest inputs that actually ran were a codebase of about 900,000 tokens and an 841,280-token prompt, both from honorable mentions. No winner came near 2M. Video dominates because of a conversion the KeepTrack and Wikipedia Video Director notebooks both spell out: about 258 tokens per sampled frame at one frame per second, plus 32 tokens per second of audio. It reproduces the teams' headline numbers, and a second toy shows the caching trade-off:

```python
# Illustrative toy, not any team's code. Rates per second come from the
# KeepTrack and Wikipedia Video Director notebooks; prices per 1M input
# tokens are the Flash figures KeepTrack quotes. STORAGE is a placeholder.
FRAME, AUDIO = 258, 32          # tokens per sampled frame (1 fps), per second of audio

def video_tokens(seconds, with_audio=True):
    return seconds * (FRAME + (AUDIO if with_audio else 0))

for name, sec in [("10-min house tour", 600), ("30-min screen recording", 1800),
                  ("1 h 40 min match", 6000)]:
    print(f"{name:26s} ~{video_tokens(sec):>9,} tokens")

P_IN, P_CACHED = 0.075, 0.01875  # USD per 1M tokens, uncached vs cached input
STORAGE = 1.00                   # USD per 1M tokens per hour: PLACEHOLDER, check your price sheet

def break_even(ttl_hours):
    # uncached: n * P_IN.  cached: one normal write + n * P_CACHED + storage for the TTL.
    # smallest n with n*P_IN > P_IN + n*P_CACHED + STORAGE*ttl
    n = 1
    while n * P_IN <= P_IN + n * P_CACHED + STORAGE * ttl_hours:
        n += 1
    return n

for ttl in (0, 0.25, 1, 2):
    print(f"TTL {ttl:>4} h: caching pays from query #{break_even(ttl)}")
```

```console
10-min house tour          ~  174,000 tokens
30-min screen recording    ~  522,000 tokens
1 h 40 min match           ~1,740,000 tokens
TTL    0 h: caching pays from query #2
TTL 0.25 h: caching pays from query #6
TTL    1 h: caching pays from query #20
TTL    2 h: caching pays from query #37
```

The estimates land on KeepTrack's ~165,000 to 170,000, Process Documentation's "over 500k" and Sports Sponsorship's 1.7M. Cached input was four times cheaper (0.01875 against 0.075 USD per 1M tokens), but storage is billed per hour, so the break-even query count is the smallest integer $n$ with

$$
n\,(p_{\text{in}} - p_{\text{cached}}) > p_{\text{in}} + s\,h
$$

where $s$ is the storage rate and $h$ the TTL in hours. The toy's storage rate is a placeholder, not a quoted price, so read only the trend: a long TTL buries the saving. KeepTrack's own rule: caching pays when you query the same input more than twice, with a TTL just above the job length.

### Winners table

| Team | Project | What it does | Caching in code |
| --- | --- | --- | --- |
| kyle1373 | FrameCut | Edit video by chat; Gemini returns edit operations that ffmpeg renders | Y, measured slower |
| oleksandrarsentiev | Exploring Sports Sponsorship | Brand screen time in F1, NBA and FIFA broadcasts | Y, per-brand re-query |
| mrdbourke (Daniel Bourke) | KeepTrack | House-tour video to an insurance CSV | Y, cost measured |
| futureofworkchannel | Process Documentation from Video | Screen recording to a Word document, flowchart and onboarding Q&A | Y, TTL 60 min |

All four received 25,000 USD, unranked, and all four used video. The eight honorable mentions, with no stated prize, are covered by family below.

### FrameCut, in depth

The judges called FrameCut "a compelling vision for a complete product in just a single month"; the team reports 118 waitlist sign-ups for an alpha. It routes by length: short videos go to Gemini 1.5 Flash directly; videos above the caching minimum are uploaded through the File API, cached, and analysed with Pro, with Flash handling follow-ups. The window holds a detailed system prompt and the full chat history, each edit request carries a canvas range and time range, and Gemini answers with structured "OperationBlocks" (overlays, captions, timing) that `ffmpeg-python` renders.

The part worth copying is the latency benchmark, because it contradicts the brief.

![FrameCut's team-reported latencies on a 106,799-token video: with caching, Pro's edit request rose from 56.55 to 86.27 seconds and Flash's from 20.37 to 51.73, while Pro's first request and correction barely moved](/imgs/blogs/kaggle-gemma-gemini-early-hackathons-winners-4.webp)

On a 106,799-token interview, caching never saved more than 0.7 seconds and made edits much slower: Pro 56.55 to 86.27 seconds (+52.6%), Flash 20.37 to 51.73 (2.54 times). The team concluded that "Gemini 1.5 Flash without context cache provides the most efficient performance". A winner that measured its sponsor's headline feature as a slowdown, and said so, reflects well on the judging.

### KeepTrack, in depth

Daniel Bourke's notebook teaches the token arithmetic above, tabulates Flash-8B, Flash and Pro, and picks Flash-002. It makes **three prompt passes over one cached video**: a ~10,000-token prompt with few-shot examples and a think-first `<video_inventory_analysis>` block (rooms, timestamps, occlusion, lighting); a continuation pass if output stopped early; and a review pass. After each, a separate Gemini call validates and repairs the CSV.

Output was the real bottleneck, and the fixes are reusable: emit CSV, not JSON, because it costs far fewer tokens; have the model print a sentinel, `MORETIMESTAMPSTODO`, when unfinished; compare the last extracted timestamp with the true video length from OpenCV; re-prompt. The author reports 50+ items reliably from ten minutes, with late items missed. The judges: "I would definitely use this myself!"

### Sports Sponsorship and Process Documentation

**Sports Sponsorship** asks how many seconds a brand was on screen. Gemini 1.5 Flash, with a JSON schema (brand, seconds, ad type, one of 11 industries, country) at temperature 0.2, reads F1, NBA and World Cup broadcasts; a verification stage caches the F1 video and re-asks brand by brand, which the host called a "novel approach". Two corrections to the writeup bank: the 1.7M-token figure is the full match the author **intended** to process and abandoned after repeated 503 and 504 timeouts, so the analysis ran on clips whose token counts the text never states; and the author's honest finding is inconsistency, with one brand at 15 seconds in one run and 10 in another. The proposed fix is right: ask for time intervals and sum them outside the model.

**Process Documentation** turns a narrated screen recording into a Process Definition Document (JSON from Flash rendered into Word), a Draw.io flowchart (the author admits "the quality is far from what is desired"), and an onboarding Q&A over a 60-minute cache. Its claim that Flash analyses 14 hours of recordings for 1 USD checks out in order of magnitude: 290 tokens per second gives about 14.6M tokens, about 1.10 USD uncached. Which recording produced the demo is unverified; the code runs a shorter clip than the cited 500k-token one.

### The eight honorable mentions, by family

- **Code in context.** *From Prompt to Pull Request* (spitfire2nd) is the most rigorous entry: the Melt UI repository trimmed to about 900,000 tokens, 12 hand-checked tasks against a cache, and a 10-needle haystack test where Pro recalled near-perfectly and Flash fell to 46.67%. *GitHub Profile Chat* (blackbigswan) found the most transferable lesson: above about 100,000 input tokens Gemini struggled to emit valid nested JSON, chunk-and-reassemble failed, and it was far more stable asked for "a single big analysis" than for "a list of analyses".
- **Documents and media.** *Wikipedia Video Director* (jobro1234) has the best token accounting: 34 images at exactly 34 x 258 = 8,772, a 403-second clip predicted at 118,885 and measured at 116,230, 197,888 in total, with the cache used to continue timed-out scripts. *Patent Analysis* (karnikakapoor) pre-filters with TF-IDF before caching about 323,000 tokens per analysis. *Storyboard Sculptor* (kavisekar) drew its own 54-scene storyboard.
- **Audio, comments, games.** *PODcast PROfessor* (abikalmukhiya) caches raw audio but fails beyond about two hours. *YouTube Actionable Insight Generator* (mechallenge) caches a channel's comments on Pro. *AI VTuber as Game Master* (feuera1225) runs a stateful game loop, with caching claimed but absent from the code.

**Why the four won (my inference).** All four used video, against two of eight honorable mentions, matching the host's "video understanding" theme. All four produced structured output a non-expert would use, against two of eight honorable mentions (the top-20 file says three; its cells give two). All four called the cache in code. Rigour did not decide it: the best benchmark and the best accounting finished as honorable mentions.

## 4. AI Assistants for Data Tasks with Gemma: Gemma 1 at work

### What made it hard

Gemma 1 was weak at the named tasks, and the notebooks say so: the pretrained 2B repeats itself, base 2B "has no idea what it's doing" in a ReAct loop, and asked for a schema it called gravitational-wave data "computer vision". Kaggle's T4 and P100 lack bf16, so 7B ran in 4-bit.

### Winners table

| Category | Team | Project | Fine-tune | Retrieval | The visible extra |
| --- | --- | --- | --- | --- | --- |
| Teach DS concepts | sitaberete | Build AI Agents With Google's LLM Gemma | QLoRA r8, published | Chroma, FAISS | Gemma 7B as a router |
| Python Q&A | davidtroxellucla | PyGEM | LoRA r16, published | Chroma | ROUGE at three stages |
| Summarise writeups | toshik | Gemma meets LangChain | MindGemmap, published | map-reduce | Text to mind-map model |
| Explain writeup concepts | nghihuynh | Unleashing Gemma's Power by Prompt Engineering | none | none | Self-evaluation, Streamlit app |
| Kaggle platform Q&A | vbookshelf | KaggleBot | none | FAISS plus rerank | 25-question manual eval |

Each won 10,000 USD. The ten honorable mentions and the mid-point notebooks are grouped by family after the winners.

### sitaberete: the ReAct agent that became a router

An 11,000-word tutorial building a "Data Science Instructor Crew" with quizzes per chapter. It loads in fp16 and 4-bit (7B at about 3.98 GB in int4 against 31.91 GB in fp32), builds a **181-entry dataset** (about 10 Gemini seeds, the rest drafted with Copilot and reviewed by hand), and QLoRA-tunes 2b-it at r = 8, alpha = 8, learning rate 1e-4 for one epoch, published as a Kaggle Model. Retrieval runs over the Python Data Science Handbook and a Wikipedia data-science set.

The negative result is the lesson. A LangChain ReAct agent with three tools chose tools correctly but returned incomplete answers, and prompt changes did not fix it. So the author cut it to a **router**: Gemma 7B-it only decides whether a question goes to the fine-tuned coder or to a RAG generator. The host's citation praises exactly that. For the wider design space, see [retrieval-augmented agents](/blog/machine-learning/ai-agent/retrieval-augmented-agents).

### PyGEM: the ablation the judges named

PyGEM tunes KerasNLP `gemma_instruct_2b_en` with LoRA rank 16 at a 375-token sequence length on about 2,000 prompts (two self-built sets plus `python-codes-25k` and MBPP), and adds Chroma retrieval. The host singled out the evaluation: **ROUGE on unseen questions at three stages**, base, base plus RAG, fine-tuned plus RAG. The author reports RAG adding 0.02 to 0.03 ROUGE-L F and fine-tuning "almost doubling" it to about 0.4; exact values live only in plots. The table's shape is what to copy:

```python
# Illustrative toy of a PyGEM-style ablation table. Not the team's code or data:
# one made-up reference answer and three made-up answers, scored with ROUGE-L F1.
def lcs(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            dp[i + 1][j + 1] = dp[i][j] + 1 if x == y else max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1]

def rouge_l_f1(pred, ref):
    p, r = pred.lower().split(), ref.lower().split()
    l = lcs(p, r)
    if l == 0:
        return 0.0
    prec, rec = l / len(p), l / len(r)
    return 2 * prec * rec / (prec + rec)

ref = "use a list comprehension to build the new list in one line"
stages = {
    "base":            "python lets you write a loop in many ways and lists are useful",
    "base + RAG":      "python lets you write a loop to build a new list in many ways",
    "fine-tuned + RAG": "use a list comprehension to build a list in a single line",
}
for name, ans in stages.items():
    print(f"{name:17s} ROUGE-L F1 = {rouge_l_f1(ans, ref):.2f}")
```

```console
base              ROUGE-L F1 = 0.16
base + RAG        ROUGE-L F1 = 0.46
fine-tuned + RAG  ROUGE-L F1 = 0.75
```

The toy's numbers are invented; the format is the point. One row per stage, the same held-out questions in every row, and the reader sees which technique paid.

### KaggleBot: retrieval done carefully

vbookshelf answers questions about Kaggle from 14 curated text files, calling the chunking "vital". Retrieval embeds with `all-MiniLM-L6-v2`, searches a flat FAISS index, then **reranks with a cross-encoder**.

![KaggleBot's worked example: for the query "who is D. Sculley" the chunk holding the answer ranked 6th after vector search and 1st after cross-encoder reranking](/imgs/blogs/kaggle-gemma-gemini-early-hackathons-winners-5.webp)

Generation uses gemma-7b-it in 4-bit across two T4s (the P100 crashed during evaluation), three hand-corrected few-shot examples so answers never mention "the context", and two style rules. The evaluation is 25 questions run end to end and shown in full. The host: "We were really impressed at how accurately it responded to our questions and how well it knew when to decline to respond." For how rerankers are trained, see the [reranker models post](/blog/machine-learning/large-language-model/reranker-models-training-finetuning-case-studies).

### toshik and nghihuynh

**toshik** scraped top-solution links, joined Meta Kaggle scores, and summarised with Gemma 2B in a LangChain map-reduce chain, producing an overall summary, a comparison matrix and per-solution summaries. The fine-tuned **MindGemmap** turns text into a Graphviz mind map, "impossible for original Gemma 2B"; its training data and hyperparameters are stated nowhere. **nghihuynh** fine-tuned nothing: it compared prompting styles on 4-bit Gemma-2b-it against a human-checked ground truth and reports one-shot BLEU 0.1244 against zero-shot 0.0641 (1.94 times), plus a Gemma self-evaluation score, wrapped in a Streamlit app.

### The honorable mentions, by family

- **Agents.** penpentled's KaggleQA runs ReAct over the live Kaggle API; since base Gemma 2B could not do ReAct, it **distilled Gemini's traces into Gemma** with LoRA. kyushikmin's SCAPPY wraps a 7B QLoRA model in a self-critic loop and is the only entry on a public benchmark: MBPP pass@1, with +3.4% code running without error and +2.2% tests passed.
- **Fine-tune plus retrieval.** hserdaraltan compares untuned, LoRA and LoRA plus Pinecone RAG; tarique7 tunes on about 2,000 pairs with FAISS; ianakoto tunes 7B on 8 TPU cores on synthetic QA aligned with forum QA.
- **Prompting.** woosungyoon ensembles three answers and chains refinements; jacoporepossi compares Stuff, MapReduce and Refine; astura catalogues failures and finds question-style prompts beat a schema.
- **Stronger base.** inoueu1 found none of its fine-tunes beat stock CodeGemma 7B and shipped that.
- abireltaief's notebook returned HTTP 403; nothing about it is verified. The mid-point round also named lucamassaron's from-scratch RAG with a gemma.cpp path for CPU-only devices.

**Why the five won (my inference).** Each carried one extra the host's citation names: the router, the three-stage table, the mind-map model, self-evaluation plus an app, a manual eval with good refusals. SCAPPY and KaggleQA had stronger techniques but less polish or fit.

## 5. Unlock Global Communication with Gemma: five ways to teach a language

### What made it hard

Gemma 2 already knew some of every eligible language. The Italian, Ancient Chinese and ArGemma notebooks all name the same failure: **the base model drifts into English** or invents words. Fixing that without breaking general skills is a forgetting problem, and the five winners touched five different amounts of the model.

### Winners table

| Place | Team | Language | Sizes | Tuning | Headline reported result |
| --- | --- | --- | --- | --- | --- |
| 1st | Alfaxad Eyembe, Mrina Eliseus: Gemma 2 Swahili | Swahili | 2B, 9B, 27B | LoRA r8; QLoRA for 27B | MMLU sample 2B 15.00 to 34.17% |
| 2nd | Justin Yang (zake7749): Kyara | Traditional Chinese | 2B, 9B | Full SFT, then full DPO | TMMLU+ 2B 36.90 to 42.57 |
| 3rd | Taha Alselwi: ArGemma | Arabic | 2B, 9B | LoRA r8, per-task adapters | Custom BLEU 9B 0.4788 to 0.7714 |
| 4th | Stefano Fiorucci (anakin87) | Italian | 2B | Spectrum, top 25% of layers; DPO | Open ITA average 47.46 to 48.49 |
| 5th | Shiner (judith007): Ancient Chinese Expert | Classical Chinese | 2B, 9B | LoRA r128 | Beats ChatGPT on 4 of 5 ACLUE categories |

All five places paid 30,000 USD, yet the host ranked them, without reasons; no honorable mentions were named.

![Learning rate against how much of the model each team trained: LoRA r8 runs sit at 1e-4 to 2e-4, LoRA r128 at 1e-5, Spectrum layer subsets at 5e-6 with 2e-5 marked as having hurt, and Kyara's full-weight runs at 2e-5 for SFT and 3e-7 for DPO](/imgs/blogs/kaggle-gemma-gemini-early-hackathons-winners-6.webp)

One regularity spans the recipes: **the more weights a team touched, the lower its learning rate.** Rank-8 adapters ran at 1e-4 to 2e-4, the rank-128 LoRA at 1e-5, Italian's layer subset at 5e-6 (the author reports 2e-5 hurt from an instruct model), Kyara's full SFT at 2e-5 and its full DPO at 3e-7. For the adapter side, see [debugging LoRA and PEFT](/blog/machine-learning/debugging-training/debugging-lora-and-peft).

### 1st: Gemma 2 Swahili

The winner is a **model suite**: a 2B continued-pretraining preview plus instruction-tuned 2B, 9B and 27B, on Kaggle, Hugging Face and GitHub. Pretraining used a 1M-example slice of Inkuba-Mono Swahili, profiled first (median 41, p95 151 tokens) to justify a maximum length of 256; instruction data is 67,017 pairs from Bactrian-X. Both are public datasets. LoRA runs at r = 8, alpha = 32, learning rate 2e-4; 27B uses 4-bit QLoRA at 1.5e-4. On one A100 40GB the team reports about 7.5 to 9 hours per model, with 27B getting only 150 steps.

![Team-reported before and after scores: Swahili MMLU and sentiment gains at all three sizes, Kyara's TMMLU+ gains, Italian's one-point average gain, and three ACLUE categories where the Ancient Chinese 9B tune fell below its own base](/imgs/blogs/kaggle-gemma-gemini-early-hackathons-winners-7.webp)

Read the evaluation carefully. Swahili MMLU is a **120-question sample**: 2B 15.00 to 34.17%, 9B 43.33 to 55.83%, 27B 20.00 to 54.17%; each is a whole number of questions (2B went from 18 to 41 correct). Sentiment is a 200-text sample: 2B 49.00 to 66.50%, 9B 83.42 to 86.50%, 27B 87.50 to 88.50%. One number does not fit: 83.42% is not a multiple of 0.5, so it cannot be a score on exactly 200 items. Translation is 20 sentence pairs, and the notebook notes a slight 9B regression. The writeup bank's "MMLU up to 57.89%, sentiment up to 90.00%" comes from summary bullets; the notebook's own tables top out at 55.83% and 88.50%, which I use. My inference about the win: four published sizes with documentation cover Useful and Robust at once, even on tiny samples.

### 2nd: Kyara, the data pipeline as the contribution

Kyara's contribution is how it builds Traditional Chinese data. An LLM writes "curious student" queries from articles, retrieves extra evidence with FAISS, and runs a structured "rethink" (Judgement, Evidence, Is_Answerable, Follow-up queries) before answering. Tuning is **full-parameter**: Axolotl SFT on `gemma-2-2b-it`, global batch 256 on 8 GPUs, 3 epochs at 2e-5; then full DPO at 3e-7 with SFT loss mixed in (`pref_ftx` 0.85). Preference pairs come from a reward model choosing best and worst from a sampled pool, from LLM judges (GPT-4o, Gemini Flash 2.0, DeepSeek V3) with the A/B order swapped to check consistency, and from environment-based filtering. The [DPO fine-tuning post](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo) covers the loss.

The evaluation is the broadest: 2B against Gemma-2-2b-it on TMMLU+ 42.57 against 36.90 (+5.67), MT-Bench-TW 7.43 against 6.35; 9B on TMMLU+ 59.87 against 54.77 (+5.10), but MMLU-Redux 72.57 against 72.82, slightly lower, reported by the team.

### 3rd: ArGemma, retrieval inside the training data

ArGemma shards Gemma 2 9B across 8 Kaggle TPU cores, the only winner trained on Kaggle, with LoRA rank 8 at 1e-4 and **one adapter per task family** (translation, summarisation, stories, Classical-to-MSA bridging). Training stops while accuracy stays under about 70% "to avoid the model hallucinating and generating words from another language". For bridging, an Arabic bi-encoder retrieves the top three parallel examples per training query and prepends them, so the model is tuned on context-rich inputs and the same retrieval builds the few-shot prompt at inference. The BLEU uses custom weights (0.7, 0.2, 0.1, 0), heavily unigram, so 0.4788 to 0.7714 is not comparable to standard BLEU-4.

### 4th: Italian, the cleanest method and the smallest gain

Fiorucci argues Gemma 2's 256k vocabulary makes adaptation feasible without tokenizer changes. SFT data includes `fine-instructions-ita-70k`: deduplicated conversations, translated, then **filtered by Llama-3.1-70B as a judge**. **Spectrum** unfreezes the top 25% of layers by signal-to-noise, a partial full fine-tune meant to limit forgetting; SFT runs one epoch ("no benefit beyond 1") at 5e-6 on one A6000. DPO adds **on-policy pairs**: two samples from the SFT model ranked by Llama-3.3-70B. On the public Open ITA tasks: base average 47.46, SFT 48.19, DPO 48.49, a gain of 1.03 points that I recomputed. The author is candid that these benchmarks barely capture preference tuning. My inference: the cleanest method placed 4th because it shipped one 2B model with a one-point gain.

### 5th: Ancient Chinese Expert, and a regression in plain sight

It starts with a baseline audit noting drift into English and adds a new task, Classical Chinese to and from English. The 252k-example mix includes a self-made translation set (classical-to-modern pairs with the modern side translated) and a chain-of-thought set; a 30k idiom corpus feeds RAG. The real run used LoRA r = 128, alpha = 256, 1e-5, 3 epochs on one A800. On ACLUE the 9B tune beats ChatGPT in four of five categories, but **base gemma-2-9b beats the tune on three of five** (Lexical 45.80 against 44.13, Semantic 51.40 against 45.30, Inference 47.22 against 38.43), and the tuned 2B is below ChatGPT in all five, not "most" as the top-20 file says. For a low-resource language under a strict metric, compare the [Deep Past Akkadian post](/blog/machine-learning/learn-from-kaggle/kaggle-deep-past-akkadian-translation-solutions).

## 6. Across all three: the feature matrix

![Feature matrix across all 14 prize winners of the three hackathons: fine-tuning and published models in 8 of 14, retrieval in 7, context caching only in the 4 Long Context winners, preference tuning in 2, and before/after measurement in 8 plus 2 partial](/imgs/blogs/kaggle-gemma-gemini-early-hackathons-winners-8.webp)

Recounted from the cells for the 14 winners: fine-tuned 8, published a model 8, built their own dataset 9 (plus one partial, one unknown), measured before and after 8 (plus two partial), retrieval 7, an app or product 5, an agent or router 3 (plus Kyara's demo), caching 4, preference tuning 2.

**Table stakes.** A notebook that teaches (Descriptive or Documented is scored in all three rubrics); the technique stack the rubric named; and a shipped artifact, a Kaggle Model in the Gemma events or a product-shaped output in Long Context.

**What separated winners.** Where honorable mentions exist, winners made one rubric word *visible*, while honorable mentions often had the better science. In the language event, my inference is that breadth of artifacts and evaluation ordered the places.

**Where the record contradicts itself.**

- **Caching.** FrameCut measured it slower, KeepTrack computed it costlier, Process Documentation found it "significantly" cheaper. The break-even formula reconciles them: many queries over a short TTL pay; one-off requests do not.
- **Flash or Pro.** Flash matched Pro on FrameCut's short videos but fell to 46.67% needle recall at ~900k tokens. Length decides.
- **Small-model agents.** sitaberete downgraded a Gemma 1 agent to a router; penpentled made one work by distilling Gemini traces.
- **Tune or pick a stronger base.** PyGEM's tune nearly doubled ROUGE-L; inoueu1's lost to stock CodeGemma; Ancient Chinese's 9B lost to its own base on three categories.

## 7. What did not work

| What was tried | What happened | Reported by |
| --- | --- | --- |
| Gemma 1 ReAct agent with three tools | Right tools, incomplete answers | sitaberete |
| Base Gemma 2B doing ReAct | Failed; fixed by distilling Gemini | penpentled |
| Fine-tuning a Python assistant | Lost to stock CodeGemma 7B | inoueu1 |
| Context caching for latency | Edits 52.6% slower (Pro), 2.54 times (Flash) | FrameCut |
| A long cache TTL | Cached calls cost more | KeepTrack |
| A 1.7M-token full match; audio over 2 hours | Request timeouts | Sports Sponsorship; PODcast PROfessor |
| Nested JSON above ~100k tokens; chunk and reassemble | Invalid output | GitHub Profile Chat |
| Lower temperature for consistent counts | 15 s and 10 s for one brand | Sports Sponsorship |
| 2e-5 from an instruct model; extra epochs | Hurt; no benefit | Italian |
| 9B fine-tune on classical Chinese | Below its base on 3 of 5 categories | Ancient Chinese (own tables) |

## 8. Lessons that transfer

1. **Turn every rubric word into an exhibit.** "Informative" became a token count in the first paragraph, "Robust" a base-versus-tuned table, "Efficient caching" a cost calculation. ([evaluation](https://www.kaggle.com/competitions/gemini-long-context/overview/evaluation))
2. **Measure each increment, including the ones that lose.** PyGEM's three-stage table won a category; FrameCut's slowdown and Kyara's MMLU-Redux dip did not hurt them. ([PyGEM](https://www.kaggle.com/code/davidtroxellucla/pygem-a-chatbot-for-python-questions-using-gemma))
3. **Output and time limits bind before the window does.** Use continuation sentinels, CSV over JSON, and one big analysis over a list. ([KeepTrack](https://www.kaggle.com/code/mrdbourke/keeptrack-use-gemini-to-keep-track-of-anything))
4. **Lower the learning rate as you unfreeze more weights.** Five teams, five amounts of model, one trend. ([Italian](https://www.kaggle.com/code/anakin87/post-training-gemma-for-italian-and-beyond))
5. **In low-resource tuning, data curation is the contribution.** Retrieval-grounded synthesis, judge-filtered translation and retrieval inside training inputs were the methods. ([Kyara](https://www.kaggle.com/code/zake7749/kyara-retrieval-augmentation-for-llm-fine-tuning))
6. **Compare against your own base.** "Beats ChatGPT" and "worse than its base" were both true of one entry. ([Ancient Chinese](https://www.kaggle.com/code/judith007/ancient-chinese-expert-gemma2-chatgpt/notebook?scriptVersionId=216814746))

### If you enter a hackathon like this

- Outline your notebook with one section per rubric criterion.
- Put your input size in tokens on the first screen, with the arithmetic.
- Build the evaluation set before tuning: base row first, one row per technique.
- Publish the model with a card listing data and hyperparameters.
- Ship one artifact a non-expert would use.
- Price your cache with its TTL before calling it efficient.
- Report your regression before a judge finds it.

## 9. Sources

- **Data Assistants with Gemma:** [competition](https://www.kaggle.com/competitions/data-assistants-with-gemma), [winners 499090](https://www.kaggle.com/competitions/data-assistants-with-gemma/discussion/499090), [mid-point 487380](https://www.kaggle.com/competitions/data-assistants-with-gemma/discussion/487380). Winners: [sitaberete](https://www.kaggle.com/code/sitaberete/build-ai-agents-with-google-s-llm-gemma), [PyGEM](https://www.kaggle.com/code/davidtroxellucla/pygem-a-chatbot-for-python-questions-using-gemma), [toshik](https://www.kaggle.com/code/toshik/gemma-meets-langchain-summarize-kaggle-writeups), [nghihuynh](https://www.kaggle.com/code/nghihuynh/unleashing-gemma-s-power-by-prompt-engineering), [vbookshelf](https://www.kaggle.com/code/vbookshelf/kagglebot-gemma-7b-it-rag-w-few-shot-prompting). Others: [woosungyoon](https://www.kaggle.com/code/woosungyoon/teacher-gemma-heuristic-algorithm), [hserdaraltan](https://www.kaggle.com/code/hserdaraltan/data-science-with-gemma-llamaindex-and-pinecone), [kyushikmin](https://www.kaggle.com/code/kyushikmin/building-self-critic-python-agent-with-gemma), [inoueu1](https://www.kaggle.com/code/inoueu1/make-a-smart-python-assistant-with-gemma), [jacoporepossi](https://www.kaggle.com/code/jacoporepossi/text-summarization-with-gemma), [astura](https://www.kaggle.com/code/astura/gemma-extracting-structured-data-from-write-ups), [ianakoto](https://www.kaggle.com/code/ianakoto/gemma-llm-kaggle-solution-writeup-agent), [abireltaief](https://www.kaggle.com/code/abireltaief/google-gemma-s-magic-on-kaggle-write-ups), [penpentled](https://www.kaggle.com/code/penpentled/kaggleqa-react-with-kaggle-api), [tarique7](https://www.kaggle.com/code/tarique7/kagglegpt-powered-by-gemma), [lucamassaron](https://www.kaggle.com/code/lucamassaron/data-science-ai-assistant-with-gemma-2b-it).
- **Gemini Long Context:** [competition](https://www.kaggle.com/competitions/gemini-long-context), [winners 552419](https://www.kaggle.com/competitions/gemini-long-context/discussion/552419). Winners: [FrameCut](https://www.kaggle.com/code/kyle1373/framecut-a-natural-language-video-editor), [Sports Sponsorship](https://www.kaggle.com/code/oleksandrarsentiev/exploring-sports-sponsorship-ad-exposure-analysis), [KeepTrack](https://www.kaggle.com/code/mrdbourke/keeptrack-use-gemini-to-keep-track-of-anything), [Process Documentation](https://www.kaggle.com/code/futureofworkchannel/building-process-documentation-from-video-with-ai). Honorable mentions: [AI VTuber](https://www.kaggle.com/code/feuera1225/ai-vtuber-as-game-master-unleashing-gemini-s-trpg), [GitHub Profile Chat](https://www.kaggle.com/code/blackbigswan/github-profile-chat-analysis-and-insights), [PODcast PROfessor](https://www.kaggle.com/code/abikalmukhiya/podcast-professor-enhancing-podcast-comprehension), [YouTube Insight](https://www.kaggle.com/code/mechallenge/youtube-actionable-insight-generator), [Wikipedia Video Director](https://www.kaggle.com/code/jobro1234/wikipedia-video-director), [Storyboard Sculptor](https://www.kaggle.com/code/kavisekar/storyboard-sculptor-gemini/notebook), [Patent Analysis](https://www.kaggle.com/code/karnikakapoor/gemini-1-5-powered-patent-analysis), [Prompt to Pull Request](https://www.kaggle.com/code/spitfire2nd/from-prompt-to-pull-request-benchmarking-gemini/notebook).
- **Unlock Global Communication with Gemma:** [competition](https://www.kaggle.com/competitions/gemma-language-tuning), [winners 575770](https://www.kaggle.com/competitions/gemma-language-tuning/discussion/575770). Winners: [Swahili](https://www.kaggle.com/code/alfaxadeyembe/introducing-gemma-2-swahili) ([GitHub](https://github.com/Alfaxad/gemma2-swahili-models)), [Kyara](https://www.kaggle.com/code/zake7749/kyara-retrieval-augmentation-for-llm-fine-tuning) ([GitHub](https://github.com/zake7749/kyara)), [ArGemma](https://www.kaggle.com/code/tahaalselwii/fine-tuning-gemma-for-arabic-argemma), [Italian](https://www.kaggle.com/code/anakin87/post-training-gemma-for-italian-and-beyond) ([GitHub](https://github.com/anakin87/gemma-neogenesis)), [Ancient Chinese](https://www.kaggle.com/code/judith007/ancient-chinese-expert-gemma2-chatgpt/notebook?scriptVersionId=216814746).
- Related: [BigQuery AI hackathon](/blog/machine-learning/learn-from-kaggle/kaggle-bigquery-ai-hackathon-winners), [gpt-oss red-teaming](/blog/machine-learning/learn-from-kaggle/kaggle-gpt-oss-red-teaming-challenge-findings), [long-context inference and prefill cost](/blog/machine-learning/inference-engineering/long-context-inference-rope-scaling-sinks-and-the-prefill-cost-curve).
