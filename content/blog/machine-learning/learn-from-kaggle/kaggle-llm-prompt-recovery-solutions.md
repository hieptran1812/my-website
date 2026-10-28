---
title: "LLM Prompt Recovery: how the top 20 teams beat a metric instead of reading minds"
date: "2026-10-28"
publishDate: "2026-10-28"
description: "Kaggle asked 2,175 teams to recover the hidden prompt Gemma used to rewrite a text. The podium was decided by the scoring function: a cubed sentence-t5 cosine, a centroid 'mean prompt', and a word piece called lucrarea. This is what all twelve published top-20 writeups did, and what transfers."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "prompt-recovery", "sentence-embeddings", "metric-exploit", "adversarial-tokens", "mistral", "gemma", "lora"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 22
---

> **TL;DR.** In LLM Prompt Recovery (Google, spring 2024) you saw a text and Gemma's rewrite of it, and had to guess the instruction Gemma was given. Khoi Nguyen won solo with a private score of 0.7165 by concatenating predictions from three fine-tuned 7B models and appending eight copies of the token `lucrarea`, a vocabulary piece that sits next to the end-of-sequence token in sentence-t5's embedding table and drags any string toward the centre of the scoring space. **Decisive lever: metric insight.** No single model in the winning stack scored above 0.65. **The one lesson:** when the metric is a similarity in a fixed embedding space, read the scorer's actual implementation before you train anything.

![The lever strip: of the five levers that decided Kaggle LLM competitions, Prompt Recovery was won on metric insight](/imgs/blogs/kaggle-llm-prompt-recovery-solutions-1.webp)

The strip above is the series' mental model: five levers (data, retrieval, fitting a big model into a small notebook, test-time compute, metric insight), with the one that decided this competition lit. Prompt Recovery is the cleanest case of the fifth lever in three years of Kaggle LLM competitions. The exploit was not a leak. It was a property of the evaluation that every top team had to discover, understand and optimise against. If you have ever used an embedding model as an automatic judge, this competition is a warning written by 2,175 teams.

This post synthesises the writeups of the top 20 teams on the private leaderboard. Twelve of the twenty published one. For the other eight, I say so and give no invented method.

## 1. The task in 60 seconds

The hosts took a large set of texts, asked **Gemma 7b-it** (the Kaggle `7b-it-quant` variation) to rewrite each one with a prompt they never disclosed, and handed you the pair `(original_text, rewritten_text)`. Your job was to output the `rewrite_prompt`, for example "Rewrite this essay but do it using the writing style of Dr. Seuss".

![The notebook budget and one illustrative data row: one training example, a hidden test of about 1,400 texts, and a scorer built on sentence-t5](/imgs/blogs/kaggle-llm-prompt-recovery-solutions-2.webp)

The card above is everything you were given. Notice the training set: **one example row**. Every team had to manufacture its own training data, usually by generating plausible rewrite prompts and original texts and running Gemma itself to produce the rewrites. Kaggle later released "Supplementary Texts Rewritten by GenAI Models", which the 2nd-place team called their most useful seed.

| Item | Value |
| --- | --- |
| Dates | 2024-02-27 to 2024-04-16 (merger deadline 2024-04-09) |
| Hidden test | "roughly 1,400" texts (the description says "1300+") |
| Public / private split | 15% public, 85% private |
| Notebook | CPU or GPU, at most 9 hours, internet off; runtimes obfuscated by up to 15 minutes |
| External data and models | Allowed |
| Prizes | USD 200,000: 100k, 40k, 20k, 14k, 11k, 10k, 5k for places 1 to 7 |
| Field | 2,175 teams, 2,664 competitors, 43,837 submissions |

A 15% slice of about 1,400 texts is roughly 200 public prompts, the figure the 5th-place team used. That is enough to probe the leaderboard for the *average* hidden prompt, and the average turned out to matter more than any model.

## 2. The metric, with a worked toy score

The scorer embedded your predicted prompt and the true prompt with **sentence-t5-base** (the TensorFlow version on Kaggle Models, `st5-base`) and computed a **sharpened cosine similarity with exponent 3**:

$$
\text{score} = \frac{1}{N}\sum_{i=1}^{N} \cos\big(e(\hat{p}_i),\, e(p_i)\big)^{3}
$$

Here $\hat{p}_i$ is your predicted prompt for row $i$, $p_i$ is the hidden true prompt, $e(\cdot)$ is the 768-dimensional sentence-t5 embedding, and $N$ is the number of test rows. The host's reason for the cube was "to attenuate the generous score given by embedding vectors for incorrect answers": any two English instructions have a fairly high cosine, so the cube stretches the scale.

![Cubing the cosine: confident wrong guesses are crushed, being roughly right everywhere is rewarded](/imgs/blogs/kaggle-llm-prompt-recovery-solutions-3.webp)

Look at the gap between the dashed line and the curve above. The cube is gentle near 1 and brutal in the middle. A toy (my numbers, chosen to match the scale the forum reported):

| Prediction for the pirate-shanty row | Cosine to truth | Score $\cos^3$ |
| --- | --- | --- |
| "Rewrite this as a pirate shanty" (exactly right) | 0.95 | 0.857 |
| "Rewrite this as a formal business memo" (specific, wrong) | 0.50 | 0.125 |
| A generic "improve this text" style string | 0.84 | 0.593 |

Take a model that names the exact style 30% of the time and confidently names the wrong one the other 70%. Its expected score per row is ${0.3 \times 0.857 + 0.7 \times 0.125 = 0.345}$. A generic string at cosine 0.84 to almost every target scores 0.593 on every row. The cube punishes wrong answers, and in doing so it makes **vague answers beat specific ones** unless you are right very often.

The eventual winner said so in the first weeks, in "My thoughts on why the baseline is so hard to beat": a generic "Improve that text" was already close to most labels, and being specific and wrong was punished. Geremie Yeo's probe showed "Improve that text." alone scored 0.58. A score of 0.6 corresponds to a cosine of about 0.84, since the cube root of 0.6 is 0.843.

An illustrative scorer, using the PyTorch port. This is a toy, not the host's code, and the fact that it is the PyTorch port is exactly the trap section 4 is about:

```python
# Illustrative toy scorer. NOT the competition's implementation: the host scored with
# the TensorFlow sentence-t5-base, whose tokenizer behaves differently (see section 4).
import numpy as np
from sentence_transformers import SentenceTransformer

st5 = SentenceTransformer("sentence-transformers/sentence-t5-base")

def sharpened_cosine(preds: list[str], truths: list[str], p: int = 3) -> float:
    a = st5.encode(preds, normalize_embeddings=True)
    b = st5.encode(truths, normalize_embeddings=True)
    cos = (a * b).sum(axis=1)            # row-wise cosine, both sides unit-norm
    return float(np.mean(cos ** p))      # cube, then average over rows

truths = ["Rewrite this as a pirate shanty",
          "Summarise this text in two sentences",
          "Make this email sound more formal"]
specific = ["Rewrite this as a pirate shanty"] * 3
generic = ["Improve this text and rewrite it in a different style"] * 3

print("specific:", sharpened_cosine(specific, truths))
print("generic: ", sharpened_cosine(generic, truths))
```

That is the whole competition in a dozen lines: the scoring function has a centre, and the centre pays.

## 3. What made it hard

**The labels were unseeable.** With one training row, the test prompt distribution had to be inferred. Teams generated prompts with Gemma, Gemini, GPT-3.5 and GPT-4, wrote grammars for them, and scraped community datasets, but nobody knew whether the hosts' prompts were short or long, playful or businesslike.

**The rewrite under-determines the prompt.** From a pirate-voiced rewrite you can recover "pirate". You cannot recover whether the instruction began "Rewrite", "Transform" or "Make". The 3rd-place team put it plainly: the main verb is essentially unpredictable, the content tags are not.

**The scorer was not the model you tested with.** Most teams scored locally with the HuggingFace PyTorch sentence-t5. The leaderboard used the TensorFlow version. The two tokenise differently and have different maximum lengths, so they disagree on exactly the strings an optimiser finds.

The first two constraints put everyone on a plateau around 0.6 to 0.65. The third separated the top five from everyone else.

## 4. The exploit, explained from the token up

Here is the mechanism every top-5 team used, reconstructed from the 1st, 2nd and 4th place writeups.

![A 2D sketch of the scoring space: the mean prompt sits at the centroid of the hidden prompts, a specific guess wins big or loses big, and the lucrarea suffix pulls predictions toward the centre](/imgs/blogs/kaggle-llm-prompt-recovery-solutions-4.webp)

The sketch above is a cartoon, not a real projection. The hidden prompts form a cloud on the embedding sphere. A **mean prompt** is a string whose embedding sits near the cloud's centroid: a moderate cosine to everything, so a moderate cubed score everywhere that never collapses. A specific guess lands near one target and far from others. Pulling every prediction partway toward the centre cuts the losses on misses more than the gains on hits, because of the cube. It is variance reduction under a convex penalty, not mind reading.

Khoi Nguyen then found something sharper. In the HuggingFace sentence-t5, appending an extra end-of-sequence token `</s>` to one of two vaguely similar sentences pulls their cosine toward about 0.9, upward if it was lower and downward if it was higher. A cosine of 0.9 is worth ${0.9^3 = 0.729}$, above every honest model in the competition.

![One string, two tokenizers: the PyTorch port treats the end-of-sequence marker as one special token, the TensorFlow scorer splits it into four characters, and lucrarea is the ordinary token that stands in for it](/imgs/blogs/kaggle-llm-prompt-recovery-solutions-5.webp)

The catch is in the token strips above. The host's **TensorFlow** sentence-t5 uses a SentencePiece configuration that does not accept `</s>` as a special token; it tokenises the characters `<`, `/`, `s`, `>`. The 2nd-place writeup hit the same wall: their first brute-force mean prompt was dominated by `</s>` and scored badly, because TF SentencePiece "has protection against special token injection".

So Khoi searched the T5 vocabulary for ordinary tokens whose embeddings sit next to `</s>`, and found `lucrarea`, which he called "a Walmart version of `</s>`". It pulls a sentence toward a focal point of the space (perhaps the centre), and so pulls scores toward about 0.71. When prediction and truth are far apart that helps; when they are already close it can hurt.

4th place, Yusef A., opened his writeup with "tldr 1: `lucrarea`". His theory is that T5's tokenizer was built for English plus German and Romanian translation, so Romanian pieces live in the vocabulary; that origin is his theory, not verified. He also documented that the TF model lacks the `<extra_id_X>` sentinel tokens and has a **maximum length of 128 tokens**, so long prompts score differently on the leaderboard than local PyTorch scoring suggests. 7th place measured it: prompts over 128 tokens scored about 0.41.

Three facts to hold onto before we get to the teams:

1. **Nobody found it by thinking about prompts.** 1st found it by poking the scoring model, 2nd and 5th by running a token optimiser against it, and Dipam Chakraborty's team by a gradient-based universal trigger search that "rediscovered" it with no human in the loop.
2. **It was invisible to anyone scoring with the wrong tokenizer.** flg (6th) used the HF tokenizer, which encodes the end token correctly, so the attack surface did not exist in his CV. His summary: "always test the Google product".
3. **Excluding special tokens from a search hid it too.** 14th place never found `lucrarea` for exactly that reason.

## 5. The top-20 scorecard

Ranks and scores are from Kaggle's private leaderboard API. "No writeup" means none was found after scanning all sixteen forum pages.

A rank caveat: Dipam Chakraborty's team posted a writeup titled "6th Place" but is **absent from the final leaderboard**, for reasons I could not verify. Several writeups are therefore titled one place too high (flg "7th" is 6th, Hoang Tu Anh "11th" is 10th, ironbar "20th" is 19th). I use API ranks throughout.

| Rank | Team | Private | Public rank (move) | Key idea | Source |
| --- | --- | --- | --- | --- | --- |
| 1 | Khoi Nguyen (solo) | 0.7165 | 1 (=) | Concatenated Mistral and Gemma predictions + 8x `lucrarea` suffix | [494343](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494343) |
| 2 | Team Danube | 0.7139 | 3 (+1) | Predict the ST5 embedding with cosine loss, greedy token search back to text | [494497](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494497) |
| 3 | prompt = "don't say anything" | 0.7128 | 4 (+1) | Beam-searched mean prompt + Mistral v0.2 LoRA + gate + tags + 12 cluster means | [494621](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494621) |
| 4 | Yusef A. (solo) | 0.7092 | 2 (-2) | 0.69 mean prompt + Mistral with prefix "Modify this text by", `lucrarea` | [494362](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494362) |
| 5 | Z.D.Z | 0.7071 | 5 (=) | RoBERTa regresses the embedding, retrieve + Mistral candidates, permutation rerank | [499079](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/499079) |
| 6 | flg (solo) | 0.7033 | 6 (=) | LB-solved mean vector, random-token-edit decoder; no `lucrarea` | [494755](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494755) |
| 7 | Taras & Andre | 0.7026 | 7 (=) | Beam, insert and prune mean-prompt search on an LB-matched local set | [494650](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494650) |
| 8 | QTeam | 0.6948 | 9 (+1) | not published | no writeup |
| 9 | 新造的人 | 0.6929 | 8 (-1) | not published | no writeup |
| 10 | Hoang Tu Anh (solo) | 0.6902 | 10 (=) | Mean prompt + Mistral few-shot piece + rule-based keyword piece | [494689](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494689) |
| 11 | O Yuksel (solo) | 0.6878 | 14 (+3) | Fine-tuned Phi-2 + mean prompt with 4x `lucrarea` | [494569](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494569) |
| 12 | Kansai-kaggler-kai-Shikisai | 0.6863 | 11 (-1) | Human-readable WordNet mean prompt edited by Mistral and DeBERTa | [494526](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494526) |
| 13 | Alex & Prachi | 0.6856 | 12 (-1) | not published | no writeup |
| 14 | Over-fit Under-appreciated | 0.6848 | 13 (-1) | Lasso model of LB score per dataset, token genetic algorithm | [494525](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494525) |
| 15 | Maxim Tsygankov | 0.6804 | 15 (=) | not published | no writeup |
| 16 | Stupid lucrarea | 0.6758 | 19 (+3) | not published | no writeup |
| 17 | カタリナs | 0.6743 | 20 (+3) | not published | no writeup |
| 18 | ctzamos | 0.6730 | 16 (-2) | not published | no writeup |
| 19 | ironbar (solo) | 0.6729 | 32 (+13) | LoRA ensemble of four LLMs, concatenated + public mean prompt | [494536](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494536) |
| 20 | tarick.morty | 0.6725 | 17 (-3) | not published | no writeup |

Gold went to the top 14. The spread from 1st to 20th is 0.044; from 1st to 5th it is under 0.01.

## 6. The top-20 landscape

![Top-20 technique matrix: rows are techniques, columns are private ranks; filled squares are confirmed in a writeup, grey question marks are unknown](/imgs/blogs/kaggle-llm-prompt-recovery-solutions-6.webp)

The matrix above counts only what a writeup states; the eight teams without one are grey columns. I recounted every row from the cells:

| Technique | Teams (of 12 with writeups) |
| --- | --- |
| Mean or constant prompt component | 10 (11 counting 1st's constant suffix) |
| Token-level search (greedy, beam, genetic, random edits) | 9 |
| `lucrarea` suffix | 7 (ranks 1, 2, 3, 4, 5, 10, 11) |
| Fine-tuned (LoRA) prompt generator | 7 |
| Concatenating several predictions | 7 |
| Leaderboard probing to fit the mean or the validation set | 6 |
| Untuned or few-shot LLM | 6 |
| Predicting the ST5 embedding directly | 5 |
| Encoder model (RoBERTa, DeBERTa) | 2 |
| Hand-written rules on the rewritten text | 2 |

Mistral-7B appears in 11 of the 12 writeups.

**Table stakes.** Optimise for the metric, not the true prompt: a mean prompt encoding the centroid of plausible prompts, plus a short model prediction appended or inserted. Probe the leaderboard with constant strings. Use Mistral-7B as the generator.

**What separated the top 5.** Every team ranked 1 to 5 used `lucrarea`. flg, 6th without it, called it "probably a decider". The second separator was **decoding a predicted embedding back into tokens** (2nd and 5th, with 3rd's cluster means as a cousin). Ranks 6 to 14 reached about 0.68 to 0.70 with excellent mean prompts and leaderboard-matched validation, but without the exploit.

**Contradictions worth knowing about:**

- *Predicting the embedding* worked for 2nd and 5th, "helped a little" for 4th, and failed for 11th (stuck at 0.63, a 0.08 gap between predicted embedding and nearest real prompt).
- *Length.* 19th found concatenating predictions lifted him from 0.62 to 0.64; 7th found prompts over 128 tokens scored about 0.41. Both hold: length helps until the TF scorer truncates.
- *Cluster mean prompts* gave 3rd +0.005 CV and gave 11th nothing.
- *Readable mean prompts.* 12th kept theirs human-readable and called it a mistake; 6th's and 7th's best strings are gibberish.

## 7. The winning solution, explained

![1st place: three fine-tuned 7B models, predictions concatenated for diversity, then the adversarial lucrarea suffix appended](/imgs/blogs/kaggle-llm-prompt-recovery-solutions-7.webp)

Khoi Nguyen's pipeline, above, has two halves.

**The honest half.** He fine-tuned Mistral-7B (base and instruct) and **Gemma-7b-1.1-it** on different datasets; Mistral was slightly better and **no single model beat 0.65**. For diversity each model was prompted to open with a different verb ("Rewrite...", "Alter...", "Turn...", "Make..."), and all predictions were **concatenated**, which moved him from 0.70 to 0.71. This is the mean prompt seen from the other side: a string holding several plausible phrasings embeds closer to the centre of the plausible region than any single phrasing.

**The adversarial half.** He appended `'it 's ' something Think A Human Plucrarealucrarealucrarealucrarealucrarealucrarealucrarealucrarea` (eight copies of `lucrarea`) to every prediction, worth up to **+0.05**. The writeup gives anchors (no single model above 0.65, concatenation 0.70 to 0.71, suffix up to +0.05) but no step-by-step ablation, so do not read them as strictly additive. The private score was 0.7165, first on both boards.

A baby version of the search that finds such a suffix. It is a toy written for this post, not Khoi's code (he found the token by inspecting the embedding table; 2nd and 5th found it by search):

```python
# Illustrative toy: greedy constant-suffix search against an embedding metric.
# Not any team's code. Score with the SAME tokenizer as the host, or you optimise
# the wrong thing; this toy uses the PyTorch port for convenience.
import numpy as np
from sentence_transformers import SentenceTransformer

st5 = SentenceTransformer("sentence-transformers/sentence-t5-base")
tok = st5.tokenizer

def score(candidates: list[str], targets_emb: np.ndarray) -> np.ndarray:
    c = st5.encode(candidates, normalize_embeddings=True, batch_size=256)
    return ((c @ targets_emb.T) ** 3).mean(axis=1)   # mean cos^3 per candidate

def greedy_mean_prompt(targets: list[str], seed: str, vocab: list[str], steps: int = 20) -> str:
    t = st5.encode(targets, normalize_embeddings=True)
    best, best_s = seed, score([seed], t)[0]
    for _ in range(steps):
        cands = [f"{best} {w}" for w in vocab]
        s = score(cands, t)
        i = int(s.argmax())
        if s[i] <= best_s:
            break                                     # no single token helps any more
        best, best_s = cands[i], s[i]
        print(f"{best_s:.4f}  {best}")
    return best

# vocab: non-special sentencepiece tokens as strings (truncated for the toy)
vocab = [tok.convert_tokens_to_string([p]) for p in tok.get_vocab()
         if p not in tok.all_special_tokens][:5000]
```

Against a few thousand generated prompts this climbs fast and then plateaus where the cosine-to-everything is maximised. The winners ran the same loop against the scorer's real tokenizer.

## 8. 2nd and 3rd in depth, then ranks 4 to 20

### 2nd: Team Danube (Psi, Yauhen Babakhin, Pascal Pfeiffer), 0.7139

Team Danube treated the problem as regression in embedding space, the most principled framing anyone wrote up.

1. **Mean prompt by brute force.** Greedily try each of the roughly 32k T5 tokens against the mean target vector. With special tokens excluded, the optimiser "really started to like tokens like `lucrarea`". Mean prompt alone: about 0.65 on the leaderboard.
2. **Predict the target embedding.** An H2O LLM Studio model whose target is the 768-d sentence-t5 embedding, trained with a **cosine-similarity loss**, the metric itself. H2O-Danube (1.8B) or Mistral-7B backbones made "little difference". Locally about 0.75+ on embeddings.
3. **Embedding back to string** by greedy token search, which loses three to four points: about 0.71 locally, 0.68 to 0.69 on the leaderboard.
4. **Better seeds.** Start the search from few-shot predictions, LLM predictions trained to output only the change ("as a shanty"), and a good mean prompt. Final string: few-shot + LLM predictions + mean prompt + 20 optimised tokens.

A validation set of about 350 samples correlated with the leaderboard; best CV was best public was best private. Psi: "Thankfully, public and private was a fair split".

### 3rd: prompt = "don't say anything" (Pedro Bernardo, Eduardo Rocha de Andrade, matheus), 0.7128

Five components; the first four alone gave 0.71.

1. **Mean-prompt template.** Pick a subsample of candidate prompts whose distribution matches the public leaderboard, then beam-search over a couple of thousand words. The template begins `improve phrasing text {full prompt + tags} lucrarea tone lucrarea rewrite this creatively formalize...`, with predictions inserted after the third word.
2. **Full-prompt model.** A LoRA fine-tune of **Mistral-7B-Instruct-v0.2** via [Unsloth](/blog/machine-learning/open-source-library/unsloth-lib), "probably the best finding" (v0.1 and Gemma were far worse). Zero-shot 0.61, final single model 0.62. The jump came from the data: **paraphrased variants** of each generated prompt plus **HDBSCAN clustering on T5 embeddings to balance clusters**, with "Sure, here is..." boilerplate stripped.
3. **Gate model.** A Mistral classifier drops bad predictions, trained on 40% positives, 20% random negatives and 40% **hard negatives** (nearest prompt in T5 space).
4. **Tags model.** A second Mistral predicts tags like "shanty" or "formal tone".
5. **Cluster mean prompts.** LBFGS directly in T5 space showed the single best constant point scores 0.72 CV, about 0.70 with real words. KMeans with 12 clusters gave per-cluster ceilings of 0.726 to 0.833; routing rows to clusters added +0.005 CV.

That LBFGS ceiling is the most useful number in the competition: even a perfect constant answer tops out near 0.72, and the prize money was split over the last 0.01 below it.

### Ranks 4 to 20, by approach family

**Mean prompt + LLM prediction + `lucrarea` (4, 10, 11).**

*4th, Yusef A.* Grew a mean prompt word by word over his dataset to several 0.69 mean-only prompts, with a validation set matched to known leaderboard scores. The prompt was about 95 tokens, leaving room under the 128-token limit for a Mistral-7B-it prediction forced to begin "Modify this text by". **LoRA rank 2 to 4 worked best; higher ranks overfit.**

*10th, Hoang Tu Anh.* Three concatenated pieces: a beam-searched mean prompt full of `lucrarea` (0.67 alone), a Mistral few-shot chain-of-thought piece, and **rules** on the rewritten text (speaker tags `X:`/`Y:` become "conversation between X and Y"; bold keywords map through a decision table). Each of the last two added 0.005 to 0.01.

*11th, O Yuksel.* Generated prompts with a **context-free grammar** plus five GPT-3.5 mutations per prompt. Mistral alone beat Phi-2 (0.61 to 0.59), but **Phi-2 plus the mean prompt won**, decoded with beam search 5. His mean prompt with 4x `lucrarea` scored 0.65 alone, 0.68 combined.

**Predict the embedding, then decode (5).**

*5th, Z.D.Z.* About 500k samples rewritten by Gemma-7b-it, Gemma-2b-it, DeepSeek and GPT-4. RoBERTa models **regress the prompt's T5 embedding with a sharpened-cosine loss**, nearest prompts are retrieved from a bank, Mistral-7B (OpenHermes-2.5) adds candidates, and a "rerank" keeps the permutation of concatenated candidates closest to the predicted embedding. Four `</s>` gave +0.05 locally and nothing on the leaderboard, which led them to `lucrarea`. They estimated split shake-up at about plus or minus 0.01 and chose not to overfit the 200 public prompts.

**No `lucrarea`, excellent mean prompts (6, 7, 12, 14).**

*6th, flg.* The most elegant writeup. Each constant submission returns $c \approx \cos(\text{pred}, \text{mean})^3$, so about 70 probes give enough equations to **solve for the mean vector of the public targets**. To turn a vector into a string, **random single-token edits** (512 candidates per GPU batch, greedy accept) beat Vec2Text, evolutionary search and gradient descent, producing constant strings of 0.68 to 0.69. Two untuned 7B models (Mistral, openchat-3.5) supplied predictions, weighted against the mean by how much they agreed. His simulation: the 15% public mean has a sharpened cosine of about 0.993 with the full mean.

*7th, Taras & Andre.* Grew a mean prompt by beam search, mid-prompt **insertion** and **pruning** to 0.68. Then they logged about 76 submitted mean prompts with leaderboard scores and greedily rebuilt the local dataset so local scores correlated with them. Re-optimising on that debiased set took them from 0.68 to 0.70 public.

*12th, Kansai-kaggler-kai-Shikisai.* Searched **nltk WordNet** words for a human-readable mean prompt (0.66), then edited it with Mistral predictions of text kind and tone and **DeBERTa-v3 classifiers for which words appear in the prompt** (+0.01 to 0.02). They tried the ST5 vocabulary, got gibberish containing `lucrarea`, saw a low public score and dropped it: "We should have followed `lucrarea` though."

*14th, Over-fit Under-appreciated (Jordan Bentley, Andrew Lapp).* Fit a **lasso regression over public datasets predicting the leaderboard score, R² 0.97**, learning which data resembled the hidden test, then generated strings toward an optimised target embedding with GPT-3.5 edits and a token genetic algorithm (0.66). Their model plus the mean prompt reached "high 0.68".

**Pure LLM ensembling (19), the best contrasting entry.**

*19th, ironbar (Guillermo Barbadillo).* His first LLM competition, built almost entirely on LoRA fine-tunes (Mistral-7B, Llama-2-13B, a model he calls "Mistral 22B", Mixtral-8x7B) on GPT-4-written data, concatenated (0.65 private) plus the public 0.63 mean prompt (0.67 private). He climbed 13 places, the biggest move in the top 20, and his result is the cleanest measurement of honest modelling alone: about 0.65 to 0.67, the same wall the winner's single models hit.

**No writeup: 8th QTeam, 9th 新造的人, 13th Alex & Prachi, 15th Maxim Tsygankov, 16th Stupid lucrarea, 17th カタリナs, 18th ctzamos, 20th tarick.morty.** 16th's team name suggests the trick, but that is inference from a name, not a documented method.

### Off the board: Dipam Chakraborty's universal triggers

The "6th Place" post adapted **Universal Adversarial Triggers** (Wallace et al.), a HotFlip-style gradient token search, to sentence similarity. Their optimised suffix contains `lucrarea` several times: "we didn't discover `lucrarea` ... seems like our method did". Automated search against a frozen embedding judge found the same token a human found by inspection.

## 9. What the forum taught everyone

| Thread | What it established |
| --- | --- |
| moth, "How This Competition Works" | The primer: Gemma generates, sentence-t5 scores |
| Kishan Vavdara, "All Datasets at One Place" | Index of community (original, prompt, rewrite) sets |
| Geremie Yeo, "no rewritten texts begin with 'Sure'" | The hosts cleaned Gemma's boilerplate; clean your data too |
| Rich Olson, "T5 Travesty" and a 0.62 Mistral notebook | Force short "Improve this text by [X]" outputs; extra detail is penalised |
| Vincent Sirius, "Merge the prompt to LB 0.61" | Concatenating generic prompts raises the floor |

![Public versus private rank for the top 20: the top 10 stayed the top 10, and ironbar's 32 to 19 was the only large move](/imgs/blogs/kaggle-llm-prompt-recovery-solutions-8.webp)

**Shake-up.** Almost none, as the slope chart shows: the private top 10 is the public top 10 reordered. The big climber was ironbar (32 to 19); tarick.morty fell 17 to 20. flg's simulation explains why: teams fitting the leaderboard *mean* were fitting something stable. The real risk was never the split. It was the gap between your local encoder and the host's.

## 10. What did not work

| Tried | Result | Reported by |
| --- | --- | --- |
| Injecting `</s>` directly | TF tokenizer splits it; works locally only | 1st, 2nd, 5th |
| Scoring locally with the HF tokenizer | Exploit invisible | 6th |
| Excluding special tokens from token search | Missed `lucrarea` | 14th |
| Human-readable WordNet mean prompt | A mistake in hindsight | 12th |
| Prompts over 128 tokens | About 0.41 on LB | 7th |
| DPO; embedding lookup; max-agreement picking; cluster mean prompts | No gain | 11th |
| Vec2Text, evolutionary search, input gradient descent | Lost to random token edits | 6th |
| Bigger models, public data, more data, any validation set | "Almost all the things I tried failed" | 19th |
| High LoRA ranks | Overfit | 4th |
| Mistral v0.1 and Gemma as generators | Far worse than v0.2 | 3rd |

Nearly every failure is either "we optimised against the wrong scorer" or "we spent capacity on the model when the ceiling was in the metric".

## 11. Lessons that transfer

1. **Read the metric's implementation, not its description.** Four of the top five writeups (1st, 2nd, 4th, 5th) turn on the TF versus PyTorch tokenizer. If you use an embedding model as a judge, your local judge and production judge will disagree on exactly the inputs an optimiser finds; see [tokenization bugs](/blog/machine-learning/debugging-training/tokenization-bugs).
2. **With a similarity metric, optimise toward the centroid first.** A mean prompt alone reached 0.6 to 0.69, and a perfect constant tops out near 0.72. The [geometry of embeddings](/blog/machine-learning/mathematics/vectors-dot-products-and-the-geometry-of-embeddings) explains why centroids behave this way.
3. **Train on the metric.** 2nd and 5th regressed the target embedding with a cosine loss and decoded it. If the metric is differentiable, regress it directly.
4. **Token search is an optimiser and a warning.** Greedy, beam, genetic, random-edit and gradient searches all converged on the same token. Any embedding judge can be gamed this way; it is the evaluation-side twin of [reward hacking](/blog/machine-learning/reinforcement-learning/reward-hacking-and-goodharts-law).
5. **Debias validation against the leaderboard.** 7th, 12th and 14th built local sets that tracked logged leaderboard scores; 7th gained two points from that alone.

### If you enter a competition like this

- [ ] Find the exact scoring model, tokenizer and maximum length the host uses; score locally with the same one.
- [ ] Submit constant strings before training anything, and compute the ceiling of a perfect constant answer.
- [ ] If the metric lives in an embedding space, train an embedding regressor and an inverter alongside the text generator.
- [ ] Run token search against the real tokenizer, including tokens near special tokens, then decide explicitly whether its output is within the spirit of the rules.
- [ ] Keep LoRA small and outputs short; concatenate several short predictions and stay under the scorer's length limit.

The other posts in this wave: [how Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work), [LLM Science Exam](/blog/machine-learning/learn-from-kaggle/kaggle-llm-science-exam-solutions), [Detect AI Generated Text](/blog/machine-learning/learn-from-kaggle/kaggle-detect-ai-generated-text-solutions), [LMSYS Chatbot Arena](/blog/machine-learning/learn-from-kaggle/kaggle-lmsys-chatbot-arena-solutions) and [WSDM Cup Multilingual Arena](/blog/machine-learning/learn-from-kaggle/kaggle-wsdm-cup-multilingual-arena-solutions). The next metric-exploit story is "LLMs: You Can't Please Them All", where the judge was an LLM. For the fine-tuning side, see [debugging LoRA and PEFT](/blog/machine-learning/debugging-training/debugging-lora-and-peft).

## 12. Sources

Competition: [LLM Prompt Recovery](https://www.kaggle.com/competitions/llm-prompt-recovery) (ranks, scores and public ranks from Kaggle's leaderboard API).

Top-20 writeups: [1st Khoi Nguyen](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494343), [2nd Team Danube](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494497), [3rd prompt = "don't say anything"](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494621) (code: [pedromb](https://github.com/pedromb/llm-prompt-recovery), [matheuspf](https://github.com/matheuspf/llm-prompt-recovery), [arc144](https://github.com/arc144/kaggle_llm_prompt_recovery_public)), [4th Yusef A.](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494362), [5th Z.D.Z](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/499079) (code: [mponty](https://github.com/mponty/llm-prompt-recovery)), [6th flg](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494755), [7th Taras & Andre](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494650), [10th Hoang Tu Anh](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494689), [11th O Yuksel](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494569), [12th Kansai-kaggler-kai-Shikisai](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494526), [14th Over-fit Under-appreciated](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494525), [19th ironbar](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494536) (code: [ironbar](https://github.com/ironbar/prompt_recovery)), and off the final board [Dipam Chakraborty et al.](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/494348).

Forum threads: [How This Competition Works](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/480683), [All Datasets at One Place](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/481811), [why the baseline is so hard to beat](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/480703), [no rewritten texts begin with 'Sure'](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/483374), [New Mistral 7B Public Notebook](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/488354), [T5 Travesty](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/488128), [Merge the prompt to LB 0.61](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/481116), [Supplementary Texts Rewritten by GenAI Models](https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/480288).
