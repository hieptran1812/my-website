# LLM Prompt Recovery (2024): top-20 addendum

- **Slug:** `llm-prompt-recovery` (competition id 67121, forumId 4360200, 2,175 teams on the API board)
- **Existing deep coverage:** `.claude/plans/kaggle-llm-writeup-bank-part1.md`, H2 `## 3. LLM Prompt Recovery (2024)`
  (lines ~432-595): metric (ST5 sharpened cosine^3), the `lucrarea` / TF-vs-PyTorch tokenizer exploit, ranks 1-3 in
  depth, 4th and Dipam Chakraborty's "6th place" universal-trigger post in short, key threads. This file adds ranks 5-20
  and the facts the bank lacks for 4th.
- **Rank-label caveat:** Dipam Chakraborty's team (post "6th Place", /discussion/494348) is **absent from the final API
  board**; writeups posted before removal are one place too high: flg "7th" = API 6, Taras & Andre "7th" = API 7 (their
  title already matched), Hoang Tu Anh "11th" = API 10, Kansai "12th (Private 13th)" = API 12, ironbar "20th" = API 19.
  O Yuksel titles "11th" and is API 11. Always cite API ranks.
- Source of all ranks/scores: `competitions.LeaderboardService/GetLeaderboard` (pulled 2026-10-28). Public = 15%.
  Gold = top 14 (bank).

## Top-20 table (private LB, API)

| Priv | Team | Members (userName) | Private | Public rank | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|
| 1 | Khoi Nguyen | suicaokhoailang | 0.7165 | 1 | /discussion/494343 | (bank) concatenated Mistral/Gemma predictions + 8x `lucrarea` adversarial suffix |
| 2 | Team Danube | philippsinger, ybabakhin, ilu000 | 0.7139 | 3 | /discussion/494497 | (bank) predict the ST5 embedding with cosine loss, greedy token search back to text, seeded with LLM preds + mean prompt |
| 3 | prompt = "don't say anything" | pedromb, arc144, tomirol | 0.7128 | 4 | /discussion/494621 | (bank) beam-searched mean prompt with `lucrarea` + Mistral-7B-Instruct-v0.2 (unsloth LoRA) + gate + tags + 12 cluster means |
| 4 | Yusef A. | yusefkaggle | 0.7092 | 2 | /discussion/494362 | (bank) 0.69 mean prompt + Mistral-7B-it with prefix "Modify this text by"; `lucrarea` |
| 5 | Z.D.Z | zengzhaoyang, dmitriyab, yuanzhezhou | 0.7071 | 5 | /discussion/499079 | RoBERTa regresses the ST5 embedding (SCS loss) -> retrieve + Mistral candidates -> permutation "rerank" of concatenations + `lucrarea` |
| 6 | flg | ferdinandlimburg | 0.7033 | 6 | /discussion/494755 | LB-solved mean vector, random-token-edit decoder, 2 untuned 7B LLMs, agreement-weighted point decoded online; no `lucrarea` |
| 7 | Taras & Andre | andreivanenko, taras456 | 0.7026 | 7 | /discussion/494650 | beam/insert/prune mean-prompt search on a local set re-sampled to match LB scores + Mistral "Improve this text by" |
| 8 | QTeam | puyuzhou, faceiahou, yuanchaowang, cannonw | 0.6948 | 9 | no writeup found | - |
| 9 | 新造的人 | hanx2smile | 0.6929 | 8 | no writeup found | - |
| 10 | Hoang Tu Anh | poiewq77 | 0.6902 | 10 | /discussion/494689 | beam-search vec2text mean prompt (0.67) + Mistral few-shot/CoT piece + rule-based keyword piece |
| 11 | O Yuksel | lognorm | 0.6878 | 14 | /discussion/494569 | fine-tuned Phi-2 (beam 5) + greedy/beam mean prompt with 4x `lucrarea` |
| 12 | Kansai-kaggler-kai-Shikisai | currypurin, kohecchi, masato114, tmhrkt, daikon99 | 0.6863 | 11 | /discussion/494526 | human-readable WordNet mean prompt edited with Mistral text-kind/tone, DeBERTa topic words, markdown rules |
| 13 | Alex & Prachi | alexandersaavedra, prachipatil304203984 | 0.6856 | 12 | no writeup found | - |
| 14 | Over-fit Under-appreciated | jordanbentley, andrewlapp | 0.6848 | 13 | /discussion/494525 | lasso model of LB score per dataset, optimised target embedding, GPT-3.5 + token GA mean prompt appended to an OAIF-tuned model |
| 15 | Maxim Tsygankov | chubasik | 0.6804 | 15 | no writeup found | - |
| 16 | Stupid lucrarea | uesugierii | 0.6758 | 19 | no writeup found | (team name itself signals the `lucrarea` trick) |
| 17 | カタリナs | threerabbits | 0.6743 | 20 | no writeup found | - |
| 18 | ctzamos | tzamos | 0.6730 | 16 | no writeup found | - |
| 19 | ironbar | ironbar | 0.6729 | 32 | /discussion/494536 | LoRA ensemble of Mistral-7B, Llama-2-13B, Mistral-22B, Mixtral-8x7B on own GPT-4 data, predictions concatenated + 0.63 public mean prompt |
| 20 | tarick.morty | tarique7 | 0.6725 | 17 | no writeup found | - |

URLs: `https://www.kaggle.com/competitions/llm-prompt-recovery/discussion/<id>`. 12 of 20 have a writeup (1-7, 10-12, 14, 19).

## Per-team synthesis

### 4th - Yusef A.: facts missing from the bank (/discussion/494362)

Mean-prompt search = append the next best word that raises mean SCS over his whole dataset (run as a P100 job); he
reached several 0.69 mean-only prompts. Validation set built so its score distribution matched the LB scores of known
public prompts (e.g. a 0.63 and a 0.60 prompt). His mean prompt was ~95 tokens, leaving room under the 128-token TF limit
for a Mistral prediction. Mistral beat Gemma-1.1 slightly on validation. **LoRA rank 2-4 worked best; higher ranks
overfit.** Also tried: predict the embedding with an MLP-with-attention over ST5 encodings of original and rewritten text,
then edit mean-prompt tokens toward it (helped a little); GEIA-style sentence-embedding inversion with a tuned LongT5
attacker (arXiv 2305.03010), "okay" with a mean prompt.

### 5th - Z.D.Z (yuanzhe zhou, Dmitry Abulkhanov, zengzhy5) - /discussion/499079

Code: notebook `dmitriyab/fifth-place-solution`, GitHub `mponty/llm-prompt-recovery`, data
`dmitriyab/llm-prompt-recovery-ranker-training-data`, TPU generation notebook `dmitriyab/gemma-7b-tpu-generation-2`.
Data: ~500k samples, rewritten by **Gemma-7b-it, Gemma-2b-it, DeepSeek and GPT-4**, originals partly LLM-generated and
partly public; they concluded LB prompts were "probably generated by LLM in a specific way". **Recall:** RoBERTa-base /
RoBERTa-large regress the PyTorch-T5 embedding of the prompt with an **SCS (sharpened cosine) loss**; nearest prompts by
cosine are retrieved from a prompt bank; Mistral-7B (OpenHermes-2.5) adds generated candidates (also used for test-time
prompt-set augmentation). **"Rerank":** try concatenations of candidates in different permutations and keep the one whose
embedding is closest to the predicted embedding. **Suffix:** found that 4x `</s>` gave +0.05 locally but not on LB because
TF T5 does not treat `</s>` as special; experiments led to `lucrarea` as the replacement. They noted they could
overfit the 200 public prompts for more score but estimated random-split shake-up at about +/-0.01 and chose not to.

### 6th - flg / Ferdinand Limburg (solo) - /discussion/494755 (titled "7th")

Four parts. (1) **Solve for the LB mean**: each constant submission gives `cos(pred, mean)^3 ~ c`, so ~70 probes define
equations for the mean vector of the public targets; move the string toward it. (2) **Vector -> string decoder**: tried
Vec2Text, evolutionary search, gradient descent on inputs; best was **random single-token edits** (delete/add/change
with probabilities ~0.1/0.1/0.8), ~512 candidates scored in a GPU batch, greedy accept; offline this produced mean strings
of 0.68-0.69, best ~0.693 (a gold-zone constant string, quoted in the post). (3) Two **untuned 7B models**, Mistral and
openchat-3.5 (Solar-10.7B also tried), prompted with his own prompts and Rich Olson's; forced to begin "Rewrite this
text", then that prefix is replaced by ~60 variants and their T5 embeddings averaged to cancel phrasing bias. (4) Target
point = weights `[1-agreement, agreement/2, agreement/2]` over mean and the two predictions (agreement = SCS between the
two predictions), decoded online from `mean_str + prediction_0` for 200 iterations at batch 16: ~3 h of LLM inference,
5-6 h of string positioning. **No `lucrarea`**: he used the HF tokenizer, which encodes EOS correctly, so the TF attack
surface was invisible; "always test the Google product". Argues fitting the LB *mean* is safe: in simulation the 15%
mean has SCS ~0.993 with the full mean.

### 7th - Taras & Andre - /discussion/494650

Mean prompt grown from "Rewrite this text" over a vocabulary of words from a public prompt dataset (`what5up/concat-prompts`):
beam search first, then **insertion** of words mid-prompt and **pruning**. LB probing showed prompts > 128 tokens score
~0.41, so they pruned. Reached 0.68. **Debiasing the local set:** they logged ~76 previously submitted mean prompts with LB
scores, then greedily built a local dataset (10 random prompts per iteration, kept if they increased the cosine between
the vector of local scores and the vector of LB scores); re-optimising on that set moved them 0.68 -> 0.70 public. Final
mean prompt quoted in the post (contains Cyrillic and nonsense tokens, no `lucrarea`). Mistral-7B output starting
"Improve this text by" appended: ~+0.005. Code: `taras456/7th-place-solution-mean-prompt-llm`, `.../mean-prompt-training`,
`.../dataset-preparation`.

### 10th - Hoang Tu Anh (solo) - /discussion/494689 (titled "11th")

Prediction = `{piece_1} {piece_2} {piece_3}`. piece_1: mean prompt via vec2text-style **beam search** (add next token,
score with the competition metric, warm-start with candidates near the mean vector, keep top-X plus random-Y for
diversity), 0.67 alone; his best-private mean prompt contains many `lucrarea`s. piece_2: Mistral few-shot + CoT, focusing
on filtering patterns that score below the mean prompt, +0.005-0.01. piece_3: **rules** on the rewritten text: speaker
tags `X:`/`Y:` -> "conversation between X and Y"; bold `*XXX*` keywords mapped via a decision table ("scene" -> scripts,
"host" -> podcast), +0.005-0.01. Data: public sets, cleaned, with word swapping.

### 11th - O Yuksel (solo) - /discussion/494569

Data: originals from OpenWebText, essays, songs, poems, Wikipedia, IMDB summaries, tweets, emotions, news; prompts from a
**context-free grammar** (public notebook `lognorm/prompt-generation-with-context-free-grammars`), public prompts, and
gemma-2b-it/ChatGPT, then GPT-3.5 made 5 mutations per prompt; "this text" replaced by the text type ("this poem") where
known. Fine-tuned gemma-2b -> Phi-2 -> Mistral-7B-instruct-v0.1 to output the instruction, plain loss on
`rewrite_prompt`. Mistral 0.61 vs Phi-2 0.59 alone, but **Phi-2 + mean prompt won**. Inference: beam search 5,
max_new_tokens 40. Mean prompt by greedy then beam search over his prompt DB (`Change text better somehow breakdown
Extralucrarea ...`): 0.65 alone, 0.68 combined. Did not work: DPO; predict-embedding then nearest neighbour from a prompt
DB (stuck at 0.63, ~0.08 gap between predicted embedding and looked-up prompt); pick-max-agreement among multiple
generations; cluster-specific mean prompts.

### 12th - Kansai-kaggler-kai-Shikisai (5 members) - /discussion/494526 (titled "12th", says "Private 13th / Public 12th")

Philosophy: the metric needs *exact* words, so edit a mean prompt rather than generate prompts. (1) **Validation set
correlated with LB**: sampled from public and self-generated (Gemma, Gemini, GPT-4) data using 40-90 submitted mean
prompts' LB scores. (2) Mean prompt from "Rewrite" by searching all **nltk WordNet** words; they tried the ST5 vocabulary
but the result was gibberish (`</s>`, `lucrarea`) and its public LB was low "for some reason", so they kept it
human-readable (0.66 public and private; "We should have followed `lucrarea` though"). (3) Edits: Mistral-7B v1 predicts
**text kind** from the original (replaces "text and phrase"; +0.003-0.005; one final sub with, one without, because kind
may be omitted in prompts); Mistral predicts **tone** difference, inserted at a CV-optimised position, +0.01, and "more"
before the tone +0.005; **DeBERTa-v3-large/base classify which difference-words appear in the prompt**, inserted,
+0.01-0.02; rules: text inside `**bold**` (7-9% of test by probing) and "Verse" -> add "song", +0.005.

### 14th - Over-fit Under-appreciated (Jordan Bentley, Andrew Lapp) - /discussion/494525

Built a dataset of submitted mean prompts and their LB scores, compared it against the all-in-one public datasets
(some aligned well, some had high R^2 but slope far from 1), then a **lasso regression over datasets predicting LB score,
R^2 0.97**. Optimised a target embedding component-wise, then generated strings toward it with GPT-3.5 section edits and
Andrew's **token-level genetic algorithm**: 0.66 of a theoretical 0.68. Never found `lucrarea` because they excluded
special tokens. Andrew's OAIF-tuned model scored 0.65 alone; appending the mean prompt gave "high 0.68".

### 19th - ironbar / Guillermo Barbadillo (solo) - /discussion/494536 (titled "20th"; site ironbar.github.io/prompt_recovery, GitHub ironbar/prompt_recovery)

First LLM competition. LoRA fine-tunes of **Mistral-7B, Llama-2-13B, "Mistral 22B", Mixtral-8x7B** on a small own
dataset written with GPT-4 (the post says "around 1000k samples"; see UNVERIFIED). Predictions concatenated ("MMML v1"
0.64 public / 0.65 private) plus the public 0.63 mean prompt -> 0.66 / 0.67. Worked: **making n predictions and
concatenating** (0.62 -> 0.64 with 4 inferences), own data (0.62) over public data (0.60), few-shot prompting (up to 0.62
but slow). Did not work: any validation set, bigger models, public datasets, improving his data, augmenting with similar
prompts, scaling data. "Almost all the things I tried failed."

### No writeup found: 8th QTeam, 9th 新造的人, 13th Alex & Prachi, 15th Maxim Tsygankov, 16th Stupid lucrarea, 17th カタリナs, 18th ctzamos, 20th tarick.morty

All 16 forum pages (~320 topics) scanned; no solution posts from these members and no substantive comments in the 18
threads fetched. 16th's team name ("Stupid lucrarea", Uesugi Erii) indicates use of the `lucrarea` trick.
Unlinked silver posts in the forum (Ahmet Erdem rank 90 /494432, Taimo /494528, s_shohei /494499, Oleg Yaroshevskiy
/494453) are outside the top 20.

## Technique frequency matrix

| Technique | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Mean / constant prompt component | ? | Y | Y | Y | N? | Y | Y | ? | ? | Y | Y | Y | ? | Y | ? | ? | ? | ? | Y | ? |
| `lucrarea` (or `</s>` substitute) | Y | Y | Y | Y | Y | N | N | ? | ? | Y | Y | N | ? | N | ? | Y? | ? | ? | N | ? |
| Token-level search / optimiser (greedy, beam, GA, random edits) | N | Y | Y | Y | N | Y | Y | ? | ? | Y | Y | Y | ? | Y | ? | ? | ? | ? | N | ? |
| LB probing to fit the mean / validation set | ? | N | Y | Y | N | Y | Y | ? | ? | ? | ? | Y | ? | Y | ? | ? | ? | ? | N | ? |
| Fine-tuned LLM (LoRA) prompt generator | Y | Y | Y | Y | ? | N | N | ? | ? | N | Y | N | ? | Y | ? | ? | ? | ? | Y | ? |
| Untuned / few-shot LLM | N | Y | N | N | Y | Y | Y | ? | ? | Y | N | Y | ? | N | ? | ? | ? | ? | N | ? |
| Predict ST5 embedding directly | N | Y | Y(LBFGS/cluster) | Y | Y | N | N | ? | ? | N | N(failed) | N | ? | Y | ? | ? | ? | ? | N | ? |
| Concatenate several predictions | Y | Y | Y | N | Y | N | N | ? | ? | Y | N | N | ? | Y | ? | ? | ? | ? | Y | ? |
| Encoder (DeBERTa/RoBERTa) component | N | N | N | N | Y | N | N | ? | ? | N | N | Y | ? | N | ? | ? | ? | ? | N | ? |
| Rule-based keyword extraction | N | N | N | N | N | N | N | ? | ? | Y | N | Y | ? | N | ? | ? | ? | ? | N | ? |
| Own generated data >= 100k | ? | ? | ? | ? | Y | N | N | ? | ? | N | ? | N | ? | N | ? | ? | ? | ? | ? | ? |

Counted summary (12 teams with writeups):
- 11/12 used a mean/constant prompt component (1st's suffix is functionally one; 5th relies on retrieval + suffix).
- 8/12 used `lucrarea` (1, 2, 3, 4, 5, 10, 11 confirmed; 16th's name suggests it, no writeup); every team ranked 1-5 did.
  6th and 7th reached gold without it; 12th and 14th explicitly missed it.
- 9/12 ran a token-level optimiser.
- 7/12 fine-tuned an LLM generator; 5/12 used untuned/few-shot LLMs; Mistral-7B appears in 10/12 writeups.
- 6/12 concatenated multiple predictions.
- Only 2/12 (5, 12) used an encoder model; only 2/12 (10, 12) used hand rules.

## Consensus vs divergence

**Table stakes:** optimise for the metric, not for the true prompt: a mean prompt that encodes the centroid of
plausible prompts, plus a short model prediction appended or inserted. Mistral-7B is the default generator. LB probing of
constant strings to estimate the hidden distribution.

**What separated the top 5 from 6-20:** discovering that the scorer's **TensorFlow** sentence-t5 tokenises `</s>`
literally and that `lucrarea` substitutes for it (worth ~+0.05; flg: "probably a decider", 5 of 6 teams above him used
it). Second: decoding a *predicted embedding* back into tokens (2nd, 5th; 3rd's cluster means), instead of only generating
prompts. Ranks 6-14 got to ~0.68-0.70 with excellent mean prompts and LB-matched validation, but without the exploit.

**Contradictions:**
- *Predicting the embedding.* Worked for 2nd (bank) and 5th /discussion/499079, "helped a little" for 4th
  /discussion/494362; failed for 11th (stuck at 0.63, ~0.08 lookup gap) /discussion/494569.
- *Long predictions.* 19th: concatenating n predictions raised 0.62 -> 0.64, "contrary to some beliefs" that length hurts
  /discussion/494536; 7th: > 128 tokens scored ~0.41 on LB /discussion/494650 (the TF ST5 128-token limit, per 4th).
- *Cluster-specific mean prompts.* 3rd: +0.005 CV (bank); 11th: did not work /discussion/494569.
- *Fitting the LB.* 6th argues fitting the LB *mean* is safe (SCS 0.993 between 15% and full mean) /discussion/494755;
  5th deliberately refused to overfit the 200 public prompts /discussion/499079. Both landed where they were on public.
- *Human-readable vs gibberish mean prompt.* 12th kept it readable and called it a mistake /discussion/494526; 6th's and
  7th's best strings are gibberish /discussion/494755, /discussion/494650.
- *LoRA capacity.* 4th: rank 2-4 best, higher overfits /discussion/494362; 19th found bigger models no better /discussion/494536.

## Aggregated "what did not work"

- Injecting `</s>` directly (TF tokenizer splits it) - 1st, 2nd (bank), 5th.
- Human-readable WordNet mean prompt instead of ST5-vocab tokens - 12th (in hindsight).
- Excluding special-token neighbours from search (missed `lucrarea`) - 14th; using the HF tokenizer locally - 6th.
- Prompts > 128 tokens - 7th.
- DPO; embedding prediction + nearest-neighbour lookup; pick-max-agreement generation; cluster-specific mean prompts - 11th.
- Vec2Text, evolutionary search, gradient descent on inputs as decoders (random token edits won) - 6th.
- SVM over embedding space to find high-value regions - 14th.
- Bigger models; public datasets; augmenting with similar prompts; data scaling; any validation set - 19th.
- High LoRA ranks (overfit) - 4th.
- Most fine-tunes of Mistral v0.1 and Gemma vs v0.2 - 3rd (bank).

## Shake-up

Small. From the API: public top 10 = private top 10 apart from order (bank). Within the top 20 the largest moves are
ironbar 32 -> 19, Uesugi Erii 19 -> 16, O Yuksel 14 -> 11; fallers ctzamos 16 -> 18, tarick.morty 17 -> 20.
Outside: rewritting prompt 33 -> 22. Reason given in writeups: the LB mean is stable (6th's simulation), and teams avoided
fitting individual public prompts (5th). The real risk was metric mismatch between local PyTorch ST5 and the TF scorer,
not public/private split.

## UNVERIFIED

- Why Dipam Chakraborty's team was removed from the board: not found.
- ironbar's "around 1000k samples" for a dataset he calls "small" is likely a typo (1k?); not verified in his repo.
- "Mistral 22B" in ironbar's table: exact checkpoint not named in the post.
- 5th's exact RoBERTa training hyperparameters and prompt-bank size: in their GitHub, not read.
- 16th "Stupid lucrarea" use of the trick: inferred from team name only.
- Matrix cells for ranks 1-4 rely on the bank's summaries; 1st's mean-prompt use is not stated in the bank (marked ?).
- 14th's "OAIF" (online AI feedback) model details are not in the post.
