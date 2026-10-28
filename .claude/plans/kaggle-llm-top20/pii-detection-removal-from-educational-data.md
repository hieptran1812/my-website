# The Learning Agency Lab - PII Data Detection (2024): top-20 addendum

- **Slug:** `pii-detection-removal-from-educational-data` (competition id 66653, forumId 4310355, 2,048 teams)
- **Existing deep coverage:** `.claude/plans/kaggle-llm-writeup-bank-part1.md`, H2 `## 6. The Learning Agency Lab - PII
  Data Detection (2024) -- short, LLM-adjacent` (lines ~925-1034): facts (7 PII types, BIO, F5 metric, metric fix, no
  PHONE_NUM on public), ranks 1, 2, 4 in depth, key threads (nbroad Mixtral set, Raja "Impact of External Datasets").
  The bank's 3rd place came only from a secondary source; this file adds Rai's own post, plus ranks 5-20.
- Source of all ranks/scores: `competitions.LeaderboardService/GetLeaderboard` (pulled 2026-10-28). Public = 25%.
  Gold = top 14 (bank). Note the exact tie at 0.96796 between ranks 7 and 8.

## Top-20 table (private LB, API)

| Priv | Team | Members (userName) | Private F5 | Public rank | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|
| 1 | Fold Zero | drpatrickchan, gauravbrills, ympaik, yeoyunsianggeremie, evgeniimaslov2 | 0.96988 | 14 | /discussion/497374 | (bank) 10 DeBERTa-v3-large variants, KD, name-swap aug, Optuna voting, heavy rule PP |
| 2 | dott | dott1718 | 0.96970 | 7 | /discussion/497352 | (bank) 6 DeBERTa-v3-large on words, 7 classes, O-prob scaled x0.02-0.03 |
| 3 | Rai | rai555 | 0.96950 | 28 | /discussion/497482 | single DeBERTa-v3-large, pre-train on mpware then fine-tune on comp + nbroad, multi-seed, OOF-derived FP rules |
| 4 | - 🍎 🥧 - | lucamassaron, rashmibanthia, steubk, aman1391, kpriyanshu256 | 0.96929 | 17 | /discussion/497367 | (bank) Llama-3-70B-generated essays + Faker PII; 10 DeBERTa, focal loss, BiLSTM/GRU head |
| 5 | R & T & M | ryotak12, takai380, minfuka | 0.96891 | 1 | /discussion/497306 | 12 DeBERTa-v3-large, each member a different max_len (128 / 512 / 1536-4096), member-level voting |
| 6 | Yusuke Nakayama | nakayamar | 0.96808 | 13 | /discussion/498304 | own fine-tuned Mistral generator; custom head mapping DeBERTa->char->spaCy tokens with aux losses; 12-model weighted avg |
| 7 | Mike & Wasjaip | wasjaip, mikhailgolubchik | 0.96796 (tie) | 21 | /discussion/497310 | 14 **public** DeBERTa checkpoints, length-aware model allocation within 9 h, per-label thresholds + regex PP |
| 8 | Amed | amedprof | 0.96796 (tie) | 84 | /discussion/497323 | single DeBERTa-v3-large 5-fold on comp + nbroad only; data chosen by per-class F5 of own model on each external set |
| 9 | (🥧) | nbroad | 0.96755 | 159 | /discussion/497177 | DeBERTa-v3-large x2 + DeBERTa-v2-xlarge LoRA, improved own Mixtral set, hand-fixed ~30 train labels |
| 10 | DarioAr | themadrambito | 0.96705 | 80 | no writeup found | - |
| 11 | I \| C \| S \| J | nazarov, idv2005, takanashihumbert, snorfyang | 0.96703 | 25 | trick only: /discussion/497184 | replace whitespace tokens with "￥" so `\n` inside addresses is kept (+0.003) |
| 12 | GoGo | decalogue, xiamaozi11, hustzx, dylanzhao2012 | 0.96670 | 68 | /discussion/498130 | own Qwen/Mistral generations from a mixed PII vocab; bag of DeBERTa-v3-large, weighted CE, **no PP or thresholds** |
| 13 | Yuki Okumura | yukiokumura1 | 0.96643 | 226 | /discussion/497371 | DeBERTa-v3-large + RoBERTa-large at 512/1024/2048, focal loss, full-fit, OOF-tuned O threshold |
| 14 | M & Y & K | irrohas, yyykrk, kurokurob | 0.96615 | 5 | /discussion/497968 | 13-class + 8-class model mix (DeBERTa-v3-large, DeBERTa-xlarge, Longformer), high-recall vs high-precision members |
| 15 | Team Espresso | ubamba98, conjuring92, deepakpokkalla | 0.96609 | 18 | no writeup found | (Raja Biswas wrote the "Impact of External Datasets" thread, /discussion/473139) |
| 16 | Falcon&takaito | aromani, takaito | 0.96533 | 46 | no writeup found | - |
| 17 | bird | whitebird, pureheart, bestpredict, brianwilliamzhao | 0.96525 | 6 | no writeup found | - |
| 18 | Dileep Jayamal | dileepjayamal | 0.96523 | 42 | /discussion/497266 (titled "19th") | DeBERTa-v3-large 5-fold, one external set per fold, drop 30% of O tokens in loss + O weight 0.01 |
| 19 | Gleb Anferov | hitman47 | 0.96514 | 154 | no writeup found | - |
| 20 | JungleBeastDS | peirenc, junglebeastds, maosentang | 0.96506 | 53 | no writeup found | - |

URLs: `https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/<id>`. Full writeups for
13 of 20 (1-9, 12-14, 18) plus a trick post from 11th.

## Per-team synthesis

### 3rd - Rai (solo) - /discussion/497482 (inference `rai555/pii-data-detection-inference`, GitHub srai9/pii-data-detection-3rd-place-solution)

Replaces the bank's secondary-source entry. One backbone, **deberta-v3-large**, trained in two stages: **pre-train on
mpware's Mixtral essays** (using it directly hurt some classes; as pre-training it raised CV and LB), then fine-tune on
competition data + nbroad's set; retrained on full data with several seeds. Added `\n\n`, `\t\r`, `\n` as tokens.
Hyperparameters: lr 1e-5, batch 2, grad accumulation 2, warmup 0.1, 2 epochs. Validation: stratified 5-fold on PII
present/absent; CV and public LB well correlated. Because some classes (PHONE_NUM) were near-absent in train and absent
on public, he **generated his own Mistral essays plus external sets purely to evaluate rare classes**. Post-processing:
O-threshold, then OOF-derived false-positive rules (very short predictions, common instructor names, Mr/Mrs/Dr), and a
fix for names labelled both B- and I- in the same essay. Did not work: relabelling train, other backbones, per-class
models, more external data, a second-stage false-positive model.

### 5th - R & T & M (Ryota, takai380, min fuka) - /discussion/497306

Public #1 -> private #5. 12 DeBERTa-v3-large models; each member ensembled internally, then **voting across members**
(simple voting beat averaging on CV and public). Diversity by max length: Ryota 128/128 (CV 0.979, public 0.973, private
0.960, weight 10), takai380 512/512 (0.977 / 0.975 / 0.965, weight 13), min fuka 1536 train / 4096 infer (public 0.972,
**private 0.967**, weight 8). Lesson: **longer max_length did better on private** even though short won CV/public; min
fuka's part alone was gold-level. AMP at inference to fit more models. Ryota: max_len 128 with train overlap 96 / eval
64, absolute + relative **positional features**, EMA, backbone frozen for epoch 1, loss weight O 1 vs PII 10, some models
without B/I prefixes, pre-train on nbroad + mpware + pjmathematician then continue on comp + nbroad. takai380: 512/stride
128, non-O loss x5, an LSTM-head variant, no B/I tags ("these tags are rule-based"), name rules (capitalised, article
before the name -> drop unless followed by "'s"). min fuka: embeddings + first 6 layers frozen, lr 2e-5, 3 epochs, batch
8 x accum 2, O class weight 0.1, three models on different external sets (mpware, valentinwerner's fixed-tokenisation
sets), averaged logits, inference stride 384. Did not work: 2nd-stage NAME_STUDENT FP model (+0.0005 CV, dropped), AWP,
label smoothing, GRU head, layer re-init.

### 6th - Yusuke Nakayama (solo) - /discussion/498304

Data: union of public sets (mpware, pjmathematician, nbroad, valentinwerner, tonyarobertson) plus **1,722 essays from a
fine-tuned Mistral-7B-Instruct-v0.2**, inspired by DAIGT 2nd place: fine-tune on (design-thinking tool name extracted
from the essay + the essay's PII + the host's task instruction) -> essay; then sample tools (Visualization, Storytelling,
Mind Mapping, Learning Launch, Brainstorming, Persona, Value chain analysis, Journey Mapping, Ethnography) and Faker PII to
generate. **Custom head**: DeBERTa token probabilities -> per-character (offset mapping) -> per-spaCy-token via
`scatter_reduce_(reduce="mean")` (idea from Feedback Prize 2021 /discussion/313201), so training optimises directly on
submission tokens. Loss: 8 classes (no B/I), focal loss, O weight 0.1, plus auxiliary char-level and spaCy-level losses
and start/end-position BCE losses. Model 2 curriculum: 1 epoch on all public sets -> 1 epoch on his generated set ->
competition data only. Models: 2 x deberta-v3-large (5 frozen layers, batch 1) + deberta-v3-base, 4 folds each = 12,
weighted average; 3 epochs, **max length 4600 train / 6300 inference**, lr 2e-5 backbone / 1e-4 head. CV 4-fold multilabel
stratified; "not much correlation between CV and LB". Scores (CV / private / public): model1 0.9702 / 0.9625 / 0.9757;
model2 0.9706 / 0.9675 / 0.9694; model3 0.9533 / 0.9517 / 0.9416; blend 0.9765 / 0.9681 / 0.9770. PP: whitespace -> O,
`\n` between address tokens -> I-STREET_ADDRESS, propagate NAME_STUDENT within an essay.

### 7th - Mike & Wasjaip - /discussion/497310 (notebook `mikhailgolubchik/fork-of-the-lalab-pii-data`)

Public 21 -> private 7. **Trained nothing of note: 14 publicly shared DeBERTa checkpoints** (datasets
`verracodeguacas/pii-deberta-models`, `emiz6413/37vp4pjt`, `startalks/pii-models`). The idea is **runtime allocation**:
the best models predict everything in the first ~3 h; the remaining models predict only the shortest 2/3 of documents in
the next ~5.5 h, on the assumption that PII count does not scale with length but runtime does. Per-label thresholds:
lower for NAME_STUDENT, higher for rare labels. PP: names must match `^[A-Z][a-z]+$`; unpaired B- tokens filtered by
length/pattern (B-ID_NUM needs >= 2 consecutive digits and >= 4 chars); fill `\n` inside addresses.

### 8th - Amed (solo) - /discussion/497323

Public 84 -> private 8. Early baseline 0.968 public / 0.964 private; after a two-month break, extra augmentation and
ensembling never beat it. Final: **one deberta-v3-large, 5 folds, competition + nbroad only**, CV 0.9703 (comp data only),
0.968 public, 0.967 private. **Data selection by error profile**: ran his model on each external set: nbroad F5 0.998 (easy),
mpware 0.883 (USERNAME 0.48, ID_NUM 0.40), tonyarobertson 0.923 (ID_NUM 0.41, URL 0.82), and dropped the sets whose labels
disagreed with the competition's notion of ID_NUM/URL. Label alignment from tokens + full_text + token labels only (the
whitespace column misaligned); replaced `\n\n` with `|` and `\n` with `[BR]`, which "significantly improved" street
addresses. 8 labels then rule-based BIO rebuild, score threshold 0.14 (code in post). Did not work: replacing PII with
Faker values (all, 50%, 10%), inserting Faker PII into essays.

### 9th - nbroad / Nicholas Broad (solo) - /discussion/497177 (GitHub nbroad1881/pii-data-detection)

Public 159 -> private 9 (author of the Mixtral dataset every top team used). Models: deberta-v3-large; deberta-v3-large
with layer drop + multi-sample dropout; **deberta-v2-xlarge with LoRA**; max length 512 stride 128; different tokenizers
reconciled by mapping token predictions to characters and taking the **max** over characters of each spaCy token. Data: an
improved version of his Mixtral-8x7B set with non-PII famous-person quotes, team-mate phrases ("With my teammates Joe and
Sally" = PII) and "check out my work at [URL]" phrases; **~30 train labels fixed by hand**. 4-fold to pick settings, then
full-data fits; a callback swaps all names every epoch. PP: title-case A-Za-z names, drop titles (dr, mr, miss) and fix the
following B-, propagate names in a document, orphan I- -> B-, drop coursera/wikipedia/.edu URLs, phone with > 6 digits ->
ID_NUM, phone regex. "I can't tell you how much time I spent debugging alignment errors."

### 11th - I | C | S | J (trick only) - /discussion/497184

Joseph Zhou (takanashihumbert): every public notebook skipped whitespace tokens when mapping, so `\n` inside STREET_ADDRESS
was never labelled; fix = replace whitespace tokens with "￥" (absent from the data) or a post-process that fills `\n`
between address labels: +0.003. In comments: no special tokens added in the final, also strips zero-width spaces. Same
insight independently in 1st (bank), 6th, 7th, 8th, 14th.

### 12th - GoGo (4 members) - /discussion/498130 (inference `decalogue/infer-ner-pii`)

"High degree of consistency in CV and LB"; a bag of deberta-v3-large **without any pre/post-processing or thresholds**.
Data: nbroad's set plus own generations: a PII vocabulary counted from the competition (`pii_cp`) merged with a filtered
external PII-200k file (`pii_mix`); topics = first 1-2 paragraphs of train/test essays (names masked) plus 4,000 "design
your own topic"; prompt (quoted in post) asks a MOOC student to continue the essay, optionally using the sampled PII of
the 7 types. Generators: Qwen-7B, Mistral-7B (8k), Qwen-14B (8k). Model: multi-layer DebertaV2ForTokenClassification,
weighted CE (PII 10, O 1), which made it insensitive to thresholds; fp16, lr 2e-5, 5 epochs, batch 2 x accum 4, cosine,
warmup 0.05. Single models: nbroad mix 0.962; Qwen-7B+Mistral-7B 1k each 0.96; **Mistral-7B 8k 0.966**; Qwen-14B 8k 0.964.
Unselected 5-model notebook would have scored 0.96816 (~6th).

### 13th - Yuki Okumura (solo) - /discussion/497371 (GitHub okumura2997/kaggle-pii-solution)

Public 226 -> private 13, the biggest climb in the top 20. Ensemble over backbones (deberta-v3-large, roberta-large),
lengths/strides (512/128, 1024/256, 2048/256), and external sets (mpware, nbroad, merged without downsampling). 4-fold by
`document % 4`, evaluated on last checkpoints, then full-fit; chose settings stable across all folds. Focal loss, AdamW,
cosine. PP: OOF-optimised O threshold plus small per-class thresholds. Ensemble: first sub-token's prediction per spaCy
token, simple average.

### 14th - M & Y & K (Masaya, yyykrk, kuro_B) - /discussion/497968

Public 5 -> private 14. Mix of **13-class (BIO) and 8-class models**; 13-class outputs summed B+I to 8 classes before a
spaCy-level weighted average, then converted back. Reasons for 8 classes: B/I are positional, not first/last name; easier
PP; possible I-USERNAME/I-EMAIL on private; 8-class won on public. Members (weights): deberta-v3-large 13-class nbroad
O-weight 0.1 (0.182; 0.969 / 0.969 / 0.956 CV/public/private); deberta-xlarge x2 (O 0.1 and 0.01); longformer-base; v3-large
on mpware with `\n` special tokens + SmoothFocalLoss; v3-large 8-class variants (exclude/include all negative essays,
largest weight 0.27). Each member labelled "high recall" or "high precision" before PP. Speed: dynamic padding + per-length
batch sizes halved inference (v3-large ~1 h -> ~30 min). PP: average NAME_STUDENT probability per (document, token string)
[high impact]; O-probability weight tuned on CV [high impact]; fill `\n` inside addresses [high impact]; 12-digit numbers
and "IV-0000" patterns -> ID_NUM; drop lowercase names, URLs without "tp", emails without "@", whitespace other than `\n`.

### 18th - Dileep Jayamal (solo) - /discussion/497266 (titled "19th")

deberta-v3-large with 6 frozen layers, 5-fold ensemble, each fold mixed with a different external set (pjmathematician,
alejopaullier, nbroad, mpware, tonyarobertson). **Custom loss drops 30% of O labels** and gives O weight 0.01. F5 0.965;
~15 h training on one RTX 4090.

### Outside the API top 20 but posted as "20th" - jarvisai7 (Yacine Bouaouni) - /discussion/497405

Not found on the API board under that user (see UNVERIFIED). 7 full-fit deberta-v3-large models, 4 epochs, effective
batch 16, max len 2048 / 1536 / 1024 x5, stride 256, lr 1.5-3e-5 depending on dataset. Best single model 0.967 trained on
mpware only; external data kept **under 4,000 rows** because more shifted the distribution and hurt. Mean of logits beat
softmax averaging and max voting. Only PP: O threshold 0.99; regex PP stopped helping as models improved. Best private used
4 models, not selected.

### No writeup found: 10th DarioAr, 15th Team Espresso, 16th Falcon&takaito, 17th bird, 19th Gleb Anferov, 20th JungleBeastDS

All 19 forum pages (~375 topics) scanned; no solution topics or substantive comments from these members in the 15
threads fetched (only congratulations). Team Espresso's Raja Biswas authored the external-data thread in the bank.

## Technique frequency matrix

| Technique | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeBERTa-v3-large backbone | Y | Y | Y | Y | Y | Y | Y | Y | Y | ? | ? | Y | Y | Y | ? | ? | ? | Y | ? | ? |
| Other backbones in ensemble | ? | N | N | N | N | Y(v3-base) | ? | N | Y(v2-xl) | ? | ? | N | Y(RoBERTa) | Y(xl, Longformer) | ? | ? | ? | N | ? | ? |
| nbroad Mixtral set | Y | Y | Y | Y | Y | Y | ? | Y | Y | ? | ? | Y | Y | Y | ? | ? | ? | Y | ? | ? |
| Own LLM-generated essays | Y | N | Y(eval) | Y | N | Y | N | N | Y | ? | ? | Y | N | N | ? | ? | ? | N | ? | ? |
| Fine-tuned generator | N | N | N | N | N | Y | N | N | N | ? | ? | N | N | N | ? | ? | ? | N | ? | ? |
| Staged pre-train on external -> fine-tune on comp | ? | N | Y | N | Y | Y | ? | N | N | ? | ? | N | N | N | ? | ? | ? | N | ? | ? |
| Down-weight / threshold O (recall bias) | Y | Y | Y | ? | Y | Y | Y | Y | ? | ? | ? | Y | Y | Y | ? | ? | ? | Y | ? | ? |
| Focal loss | N | N | N | Y | N | Y | N | N | N | ? | ? | N | Y | Y(1) | ? | ? | ? | N | ? | ? |
| 7/8 classes instead of BIO | N(13) | Y | ? | ? | Y(some) | Y | N | Y | N | ? | ? | ? | ? | Y(mix) | ? | ? | ? | ? | ? | ? |
| Long context (>= 1536) | Y | Y | ? | Y | Y | Y | ? | ? | N | ? | ? | ? | Y | ? | ? | ? | ? | ? | ? | ? |
| `\n`-in-address fix | Y | Y | Y | Y | ? | Y | Y | Y | ? | ? | Y | N | ? | Y | ? | ? | ? | ? | ? | ? |
| Name propagation within document | Y | Y | ? | ? | ? | Y | ? | ? | Y | ? | ? | N | ? | Y | ? | ? | ? | ? | ? | ? |
| Rule-based FP filters (case, @, length) | Y | Y | Y | ? | Y | ? | Y | ? | Y | ? | ? | N | N | Y | ? | ? | ? | ? | ? | ? |
| Name-swap augmentation | Y | N | ? | ? | ? | ? | N | N(Faker failed) | Y | ? | ? | N | ? | ? | ? | ? | ? | ? | ? | ? |
| Ensemble size >= 6 | Y | Y | N(seeds) | Y | Y | Y | Y | N | N | ? | ? | Y | Y | Y | ? | ? | ? | N | ? | ? |

Counted summary (13 full writeups: 1-9, 12, 13, 14, 18):
- 13/13 used deberta-v3-large; 12/13 trained on nbroad's Mixtral set (7th used others' checkpoints, sources unknown).
- 12/13 biased toward recall with an O weight or threshold (12th via weighted CE only, no threshold).
- 6/13 generated their own essays with an LLM (1, 3 for evaluation, 4, 6, 9, 12); only 6th fine-tuned the generator.
- 8/13 repaired `\n` inside STREET_ADDRESS (plus 11th's trick post).
- 4/13 used focal loss (4, 6, 13, 14); 5/13 dropped B/I labels in at least some models (2, 5, 6, 8, 14).
- 3/13 single-backbone, small ensembles reached top 9 (3rd: one backbone multi-seed; 8th: one model 5 folds; 9th: 3 models).
- 1/13 used no pre/post-processing at all (12th) and 1/13 trained no models (7th).

## Consensus vs divergence

**Table stakes:** deberta-v3-large token classification, nbroad's Mixtral essays, long contexts with stride, recall-biased
decision (O down-weighted or thresholded, F5 rewards recall), and careful DeBERTa-to-spaCy alignment including the
whitespace/`\n` bug.

**What separated the top 5 from 6-20:** the margin is tiny (0.96988 vs 0.96506 at 20th, ~0.005). Top-5 writeups share
(a) generated data from strong generators, Llama-3-70B (4th) or own sets (1st), or staged pre-training (3rd); (b) very long
inference contexts (5th's 4096 member scored best on private; 4th inferred at 4000); (c) large, systematically tuned
post-processing from OOF error analysis (1st, 2nd, 3rd). Ranks 6-20 mostly have one or two of these. Choosing the final
submissions mattered: "best unselected" regrets recur (12th's unsubmitted 0.96816, jarvisai7's 4-model sub).

**Contradictions:**
- *External data volume.* 1st used several sets (bank); 13th merged mpware + nbroad without downsampling /discussion/497371;
  versus 3rd "using more external data" did not work /discussion/497482, 8th dropped mpware and tonyarobertson after
  per-class error analysis /discussion/497323, the "20th" poster capped external data under 4,000 /discussion/497405.
- *mpware data.* Harmful directly but useful as pre-training (3rd) /discussion/497482; harmful (8th, ID_NUM/USERNAME label
  mismatch) /discussion/497323; best single model when used alone (jarvisai7, 0.967) /discussion/497405.
- *Post-processing.* Central to 1st, 2nd (bank), 3rd, 14th /discussion/497968; 12th used none and argues it forced
  robustness /discussion/498130; jarvisai7 found regex PP stopped helping /discussion/497405.
- *Faker PII augmentation.* 4th's data recipe relies on Faker (bank) and 6th too /discussion/498304; 8th found
  Faker replacement/insertion did not help /discussion/497323.
- *Context length.* 5th: short contexts won CV/public but long ones won private /discussion/497306; 9th used 512 throughout
  and still climbed 159 -> 9 /discussion/497177.
- *BIO vs collapsed classes.* 1st kept 13 labels (bank); 2nd, 5th, 6th, 8th, 14th collapsed to 7/8 and rebuilt BIO by rule.

## Aggregated "what did not work"

- Relabelling train; other backbones; per-class models; more external data; 2nd-stage FP model - 3rd.
- 2nd-stage NAME_STUDENT FP classifier (+0.0005 CV only); AWP; label smoothing / online label smoothing; GRU head; layer
  re-initialisation - 5th.
- Faker PII replacement (at 100/50/10%) and insertion - 8th.
- Using mpware / tonyarobertson data directly - 8th (3rd: directly harmful for some classes).
- Large external data (> 4,000 rows); softmax averaging and max voting vs logit mean; regex PP once models were strong -
  jarvisai7 ("20th" poster).
- Using `document % 4` folds (high variance) - 4th (bank); 13th used it anyway with stability-focused settings.
- Augmentations in the final - 2nd (bank).

## Shake-up

From the API (public -> private) within the top 20:
- Climbers: Yuki Okumura 226 -> 13, nbroad 159 -> 9, Gleb Anferov 154 -> 19, Amed 84 -> 8, DarioAr 80 -> 10, GoGo 68 -> 12,
  JungleBeastDS 53 -> 20, Falcon&takaito 46 -> 16, Dileep 42 -> 18, Rai 28 -> 3, Mike & Wasjaip 21 -> 7.
- Fallers: R & T & M 1 -> 5, M & Y & K 5 -> 14, bird 6 -> 17, ねこめたる 4 -> 22 (outside), public #2 -> 26 (bank).
- Why (from writeups): public had no PHONE_NUM (bank) and tiny counts of rare classes, so public rewarded models tuned to
  NAME_STUDENT; long-context models generalised better on private (5th); teams that fit thresholds/data choices to public
  fell, teams that trusted CV (12th, 3rd) or simple robust single models (8th, 9th) rose. With F5 differences of 0.0002
  between neighbours, much of the reshuffle is noise.

## UNVERIFIED

- jarvisai7 / Yacine Bouaouni posted "20th place" but no team containing that user is on the API board; Dileep Jayamal
  titled "19th" but is API 18. A team removal between the post and the final board is the likely reason; not confirmed.
- 7th's public checkpoints: training recipes belong to their original authors (emiz6413 et al.), not read.
- Methods of 10th, 15th, 16th, 17th, 19th, 20th: unknown.
- 11th's full solution: only the whitespace trick post exists.
- Matrix cells for 1st, 2nd and 4th come from the bank summaries; "?" means not stated there.
