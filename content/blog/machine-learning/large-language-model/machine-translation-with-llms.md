---
title: "Machine Translation with LLMs: Fine-Tuning, Decoding, Edge Cases, and Evaluation in Production"
publishDate: "2026-06-10"
date: "2026-06-10"
category: "machine-learning"
subcategory: "Large Language Model"
tags: ["machine-translation", "llm", "nlp", "fine-tuning", "lora", "dpo", "cpo", "comet", "quality-estimation", "evaluation", "decoding", "localization"]
author: "Hiep Tran"
featured: true
excerpt: "A senior engineer's field guide to building production machine translation on top of LLMs — the approach landscape, data curation, fine-tuning recipes and loss functions, decoding and constraint enforcement, the edge cases that quietly corrupt output, and an evaluation stack you can actually trust."
---

I have shipped two generations of translation systems. The first was a fleet of dedicated neural machine translation (NMT) models — Marian and fairseq encoder-decoders, one checkpoint per language pair, each fed by a carefully filtered bitext corpus. The second was a single fine-tuned decoder-only LLM that replaced thirty of those checkpoints and beat most of them on human evaluation. The migration taught me that "use an LLM for translation" is not a one-line decision. It is a system: data curation, a fine-tuning recipe, a decoding strategy, a battery of constraint enforcers, and an evaluation harness that survives contact with real traffic. Skip any one of those and you ship a translator that scores well on BLEU and embarrasses you in front of a customer.

This post is the field guide I wish I'd had. The diagram below is the mental model I keep coming back to: a translation request is not "text in, text out through a model." It is a pipeline with context assembly on the front, a quality gate on the back, and the model in the middle as just one (expensive, powerful, occasionally unhinged) stage. Everything else in this article is a tour of that pipeline.

## The mental model

![LLM machine-translation request lifecycle: source text plus metadata flows through context assembly (TM, glossary, document), prompt build, LLM decode producing k=4 n-best, QE rerank with COMET-QE, and a gate at 0.80 that routes low-confidence output to a human](/imgs/blogs/machine-translation-with-llms-1.png)

Read it left to right, top row then bottom row. A source segment arrives with metadata: source and target language, domain, formality, customer-specific overrides. **Context assembly** is where you fetch a translation memory (TM) match, inject the relevant glossary entries, and — if you have it — pull the surrounding document context. **Prompt build** turns all of that into a single instruction. The **LLM decode** stage produces not one translation but a small n-best list. **QE rerank** scores each candidate with a reference-free quality-estimation model. The final **gate** ships anything above a confidence threshold and routes the rest to a human post-editor.

The naive view collapses this to one box. The reality is that 70% of the quality and 100% of the reliability lives in the stages *around* the model. Here is the mismatch, stated plainly, because almost every team I've worked with starts on the left column and has to learn the right one:

| Assumption | The NMT-era / naive view | The production LLM-MT reality |
|---|---|---|
| The model does the translating | A single forward pass maps source to target | The model is one stage; context assembly and the QE gate decide quality and trust |
| More parameters → better translation | Scale up until BLEU plateaus | A 7B fine-tuned model + good data + reranking beats a 70B zero-shot model on most pairs |
| Output is deterministic | `argmax` decoding gives the answer | Sampling + QE-reranked n-best recovers 1–3 COMET points over greedy |
| Terminology is a soft preference | Mention the glossary in the prompt | Glossary must be a *hard constraint*; soft prompting fails ~15–30% of the time |
| Quality = BLEU | One number tells you if it's good | BLEU ranks systems; only learned QE and human MQM tell you if it's *shippable* |
| Edge cases are rare | Handle the happy path, patch later | Off-target language, markup corruption, and number errors are routine and must be engineered against |

> The model is the easy part. The pipeline around it is the product.

The rest of this article walks each stage, names the failure mode it exists to prevent, and gives you the code and the numbers to build it.

### What "good" even means

Before we tour the stages, fix the target, because almost every metric argument and every shipping decision downstream is really a disagreement about which of these four things you're optimizing. "Good translation" is not one property; it is four, and they trade against each other:

- **Adequacy** — does the output preserve the source meaning, nothing added, nothing dropped?
- **Fluency** — does it read like native target-language text a human would actually write?
- **Terminology** — are domain terms, brand names, and named entities rendered correctly and consistently across the whole document?
- **Locale correctness** — are numbers, dates, currencies, units, and formality conventions right for the *target market*, not just the target language?

These four map directly onto the MQM error categories you'll meet in the evaluation section, and they explain a lot of seemingly contradictory failures. An LLM optimizing for fluency will smooth over an adequacy error — it reads beautifully and says the wrong thing — which is the most dangerous failure of all because it passes the eye test. A dedicated NMT optimizing n-gram overlap will be adequate but stilted. A model with a great glossary can still botch locale. No single number captures all four, which is the entire reason the evaluation ladder later in this post has five rungs. Hold these four pillars in mind as the scorecard for everything that follows; each stage in the pipeline exists to protect one of them.

## The approach landscape

**Senior rule of thumb: pick your approach from the resource level and the latency budget, never from the model leaderboard.** A translation system is a portfolio decision, not a single bet. There are five approaches worth knowing, and none of them dominates the others on every axis.

![Translation approaches compared across five dimensions — dedicated NMT, zero-shot LLM, few-shot LLM, fine-tuned LLM, and QE-reranked/agentic — showing each wins on different axes of quality, low-resource handling, latency, cost, and control](/imgs/blogs/machine-translation-with-llms-2.png)

1. **Dedicated NMT** (NLLB-200, M2M-100, MarianMT, your own fairseq checkpoint). A purpose-built encoder-decoder. Sub-50 ms latency on a CPU, pennies per million characters, and trivially supports lattice/grid constraints for terminology. It is *data-hungry* — quality on a pair is a direct function of how much clean bitext you have — and it has zero world knowledge to fall back on when context is ambiguous.

2. **Zero-shot LLM.** Prompt a frontier model with "Translate the following from English to Portuguese." Astonishingly good for a system that was never trained for the task, and *surprisingly* strong on low-resource languages where it can lean on transfer from related languages. But latency is 100 ms+, cost is real, and you have weak control — it will helpfully "improve" your text, drop your markup, and occasionally answer in the wrong language.

3. **Few-shot LLM.** Same model, but you prepend 3–10 high-quality translation examples, ideally retrieved to be similar to the input. This is the single highest-leverage prompting trick in MT: it pins register, terminology, and format. It costs more tokens and more latency, and control is still "by example" rather than guaranteed.

4. **Fine-tuned LLM** (ALMA, TowerInstruct, Qwen-MT, or your own LoRA on Qwen/Llama/Gemma). Continued pretraining and/or supervised fine-tuning on parallel data. This is the quality sweet spot for any pair where you have data: it beats few-shot at a fraction of the inference cost because you no longer pay for in-context examples on every request. See [Effective LLM Fine-Tuning Techniques](/blog/machine-learning/large-language-model/effective-llm-fine-tuning-techniques) for the general recipe; we specialize it for translation below.

5. **QE-reranked / agentic.** Wrap any of the above in a quality-estimation reranker, a translate-then-refine loop, or a multi-agent "translate, critique, edit" pipeline. This is where the top of the LLM-MT leaderboards live, and it is also where your latency goes to die — you are paying for `k×` decodes plus scoring. Reserve it for high-value, latency-tolerant content.

A concrete portfolio I have actually run: dedicated NMT for the high-volume, low-stakes UGC firehose (chat messages, reviews); a fine-tuned 7B LLM for product and marketing content; and a QE-reranked, glossary-constrained pipeline on the same 7B for legal and medical, where a single mistranslation is a liability. One model family, three serving modes, costs allocated to value.

### Multilingual vs bilingual: one model or many?

A foundational decision sits underneath the approach choice: do you train one model that handles many language pairs, or a dedicated model per pair? The tradeoff is **transfer versus interference**. For related and low-resource languages, a multilingual model transfers knowledge across pairs — Portuguese helps Galician, Hindi helps Marathi — and this transfer is often the only way a genuinely low-resource pair becomes usable. But there is a well-documented "curse of multilinguality": as you cram more languages into a fixed-capacity model, each one gets a thinner slice, and your high-resource pairs degrade relative to a dedicated bilingual model. English-centric models (every pair routed through English) are easy to build but lose quality on non-English pairs; many-to-many models avoid the pivot but are harder to train and balance.

What I actually do: one multilingual model for the long tail of low- and mid-resource pairs, where transfer is pure upside, and bilingual or small-language-group models for the handful of high-volume, high-resource pairs where interference would cost real quality and where the volume justifies a dedicated checkpoint. The decision is economic — capacity is finite, so spend it where the traffic and the stakes are.

### Retrieving the right few-shot examples

Few-shot prompting is only as good as the examples you choose, and this is the part most teams get wrong by stuffing in random or static examples. Random examples barely move quality; *retrieved* examples — the nearest neighbors of the input pulled from your translation memory by embedding similarity — pin terminology, register, and phrasing because they are about the same thing as the input. A TM is a retrieval index, and few-shot MT is retrieval-augmented generation; the same machinery from [how to build an effective RAG system](/blog/machine-learning/large-language-model/how-to-build-effective-rag-system) and the [reranker models](/blog/machine-learning/large-language-model/reranker-models-training-finetuning-case-studies) post applies directly.

The practical recipe: index your TM source segments with a multilingual embedding model, retrieve the top 3–8 nearest neighbors of each input, optionally rerank them, and place the *most* similar example last (models weight the most recent in-context example most heavily). For high fuzzy-match segments (say, a TM match above 90%), skip the model entirely and reuse the human translation — that is the cheapest, highest-quality "translation" you will ever serve.

Qwen's team has a nice writeup of treating translation as a reinforcement-learned LLM task end to end — worth reading alongside this as a concrete instantiation of the fine-tuned + preference approach: [Qwen-MT: Machine Translation as a Reinforcement-Learned LLM Task](/blog/paper-reading/large-language-model/qwen-mt-multilingual-translation).

## Data: the real bottleneck

**Senior rule of thumb: your translation quality is bounded by your data quality long before it is bounded by your model size.** I have never once been limited by parameters. I have constantly been limited by the fact that half my "parallel" corpus was misaligned, machine-translated already, or in the wrong language.

![Building an MT fine-tuning set from mixed sources: mined bitext (OPUS, CCMatrix) and in-house translation memory feed a clean+dedup stage, monolingual target text feeds backtranslation, both streams merge at a COMET-QE filter, then instruction formatting produces the SFT train set](/imgs/blogs/machine-translation-with-llms-3.png)

The graph shows the three real sources of parallel data and the gauntlet they run before they earn a place in the training set.

- **Mined bitext** — OPUS, CCMatrix, ParaCrawl, NLLB's mined data. Enormous, free, and noisy. CCMatrix-style data is mined by embedding-similarity, which means it is full of near-misses and partial alignments.
- **In-house translation memory** — your `.tmx` files from past human translation projects. Small, gold-quality, and the most valuable bitext you own because it is in *your* domain and *your* terminology.
- **Monolingual target text** plus **backtranslation** — take clean monolingual data in the target language, run it through a reverse-direction model, and you have synthetic source-target pairs. Backtranslation remains, in 2026, one of the most reliable ways to lift low-resource quality.

Here is how those sources actually stack up, with the rough sizes and quality tiers I plan around:

| Source | Typical size (per pair) | Quality | Best use |
|---|---|---|---|
| In-house TM (`.tmx`) | 10k–1M pairs | Gold | Upsample heavily; the domain + terminology anchor |
| OPUS curated subsets | 1M–100M pairs | Mixed | General-domain base |
| ParaCrawl (Bicleaner-filtered) | 10M–1B pairs | Web-noisy | Volume, only after hard filtering |
| CCMatrix / NLLB-mined | 100M–10B pairs | Noisy | Low-resource bootstrap only |
| Backtranslation (synthetic) | as needed | Reverse-model-bound | Low-resource lift; cap at ≤ 40% of the mix |

### Alignment and segmentation: the silent corruptor

Before any filter runs, the data has to be *aligned*: sentence $i$ in the source file must correspond to sentence $i$ in the target. Misalignment is the most insidious data bug because the pairs look individually plausible — each side is fluent — but they don't mean the same thing, and no length or LID filter catches it. The damage is worse than random noise: you are actively teaching the model that unrelated sentences are translations of each other.

Three practical defenses. First, use a real document-aligner — `vecalign` (embedding-based) or `hunalign` (dictionary + length) — and align at the *document* level before you ever split into sentences; aligning raw sentence lists across two files is how the off-by-one errors creep in. Second, respect that **sentence segmentation differs across languages**: a period is not a sentence boundary in every script, and splitting EN and JA independently produces different sentence counts that then misalign. Segment with a language-aware splitter (`sentence-splitter`, `pySBD`, or the Moses/`sacremoses` splitter) per language. Third, keep document boundaries as metadata even after segmenting — you need them for document-level training and evaluation later, and reconstructing them after the fact is painful.

A blunt but effective sanity check I run on every new corpus: sample 200 random pairs, score them with COMET-Kiwi, and eyeball the bottom 20. If the bottom of a "clean" corpus is full of off-topic pairs, the alignment is broken and no amount of downstream filtering will save it — you re-align or you drop the source.

### Filtering is the job

A realistic filter stack, in the order I run it:

```python
## filter_bitext.py — a pragmatic parallel-corpus cleaner.
## deps: fasttext (LID), sentencepiece, sacremoses; comet for the QE pass.
import fasttext

lid = fasttext.load_model("lid.176.bin")  # fastText language ID

def keep_pair(src: str, tgt: str, src_lang: str, tgt_lang: str) -> bool:
    s, t = src.strip(), tgt.strip()

    # 1. Empty / degenerate
    if not s or not t:
        return False

    # 2. Length sanity. Real translations preserve rough length ratio.
    #    Ratio guards against truncation and "src == tgt" copy-through.
    ratio = len(t) / max(len(s), 1)
    if ratio < 0.4 or ratio > 3.0:
        return False
    if len(s.split()) > 200 or len(t.split()) > 200:  # too long for sentence-level
        return False

    # 3. Language ID — the single highest-value filter. Catches the
    #    "labelled EN->PT but actually EN->ES" garbage that ruins steering.
    src_pred = lid.predict(s.replace("\n", " "))[0][0].replace("__label__", "")
    tgt_pred = lid.predict(t.replace("\n", " "))[0][0].replace("__label__", "")
    if src_pred != src_lang or tgt_pred != tgt_lang:
        return False

    # 4. Copy-through: source == target means it was never translated.
    if s == t:
        return False

    return True
```

That gets you 80% of the way. The last 20% — semantic adequacy — needs a learned filter. **Bicleaner AI** and **OpusFilter** are the standard tools; in practice I run a reference-free COMET-QE pass and keep pairs above a threshold:

```python
## qe_filter.py — drop pairs the QE model thinks aren't faithful translations.
from comet import download_model, load_from_checkpoint

model = load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))

def qe_scores(pairs):  # pairs: list[(src, mt)]
    data = [{"src": s, "mt": t} for s, t in pairs]
    return model.predict(data, batch_size=64, gpus=1).scores

## Keep the top of the distribution. 0.75 is a sane start; calibrate per pair.
clean = [(s, t) for (s, t), q in zip(pairs, qe_scores(pairs)) if q >= 0.75]
```

A few hard-won data lessons:

- **De-duplicate aggressively, across the train/eval boundary.** A surprising amount of mined data overlaps with public test sets (WMT, FLORES). If FLORES sentences leak into training, your eval is a lie. Hash and exclude.
- **Backtranslation has a quality cliff.** Synthetic data from a weak reverse model teaches your forward model to translate *translationese*, not natural target text. Tag backtranslated data (some teams literally prepend a `<BT>` token) and cap its proportion — I keep it under ~40% of any pair's mix.
- **Domain balance matters more than volume.** 100k in-domain pairs beat 5M out-of-domain pairs for a domain-specific product. Upsample your TM.

One more constraint the textbooks skip: **data governance.** Translation data is overwhelmingly customer content, which means PII, contractual confidentiality, and data-residency rules apply to your *training set*, not just your serving path. Before you fine-tune on customer TM, confirm you have the right to, scrub or pseudonymize PII — names, emails, account numbers — which conveniently doubles as the placeholder masking you need at inference anyway, and track provenance so you can delete a departing customer's data from the next training run. I have watched a promising fine-tune get blocked at a legal review it would have sailed through had governance been handled at the data-collection stage instead of bolted on at launch. Bake it into the corpus build.

## Fine-tuning the translator

**Senior rule of thumb: the resource level of the pair picks your branch; the data volume picks your leaf.** There is no single "fine-tune for translation" recipe — there is a decision tree, and the most expensive mistake is running a high-resource recipe on a low-resource pair (you get fluent, confident, wrong output) or vice versa.

![Decision tree for choosing a fine-tuning strategy: high-resource pairs branch to few-shot prompting under 5k pairs, LoRA SFT for 5k-100k, full SFT above 100k; low-resource pairs require continued pretraining on monolingual data, then LoRA SFT on bitext, then DPO/CPO preference optimization](/imgs/blogs/machine-translation-with-llms-4.png)

Walk the tree:

- **High-resource pair, < 5k clean pairs:** don't fine-tune. Few-shot prompt with retrieved examples. You'll overfit before you generalize.
- **High-resource pair, 5k–100k pairs:** LoRA SFT. Rank 16–64, on a 7–8B base. This is the workhorse and it is cheap — a few GPU-hours.
- **High-resource pair, > 100k pairs:** full SFT (or LoRA at high rank). You have enough signal to move the whole model.
- **Low-resource pair:** you almost always need **continued pretraining** on monolingual target-language text *first*, to teach the base model the language's distribution, before any parallel SFT. This is the ALMA recipe and it works: monolingual continued pretrain → small high-quality parallel SFT. See [Training LLMs to Adapt to a New Language](/blog/machine-learning/large-language-model/training-llm-adapt-new-language) for the continued-pretraining mechanics.
- **Any pair, when you want to push past reference quality:** add a preference-optimization stage (DPO or CPO) on top of SFT.

### The instruction format is a load-bearing decision

The difference between a naive prompt and a structured template is the difference between a demo and a product.

![Naive prompt versus a structured translation template: the left column shows a bare "Translate: {text}" that loses terminology, register, and markup; the right column shows a system role, a glossary block with do-not-translate terms, and explicit tag/number preservation](/imgs/blogs/machine-translation-with-llms-5.png)

The naive prompt ("Translate: {text}") produces three failures with depressing regularity: brand names get translated, the register is wrong (informal where it should be formal), and any markup or numbers in the source get mangled. The structured template — a system role that pins the language pair and register, a glossary block with hard do-not-translate terms, and explicit instructions to preserve placeholders — prevents all three. You fine-tune *on* this format so the model learns to expect it.

Here is the format I serialize training data into. Note that the same template is used at training and inference time — consistency between the two is non-negotiable:

```python
## format_sft.py — serialize parallel data into the chat template the model
## will see at inference. We bake the constraints into the prompt so SFT
## teaches the model to honour them.
import json

SYS = (
    "You are a professional {src}->{tgt} translator for the {domain} domain. "
    "Translate faithfully in a {register} register. "
    "Keep all <ph>...</ph> placeholders, numbers, and markup unchanged. "
    "Do not translate terms in the glossary; use the provided target form."
)

def to_record(src, tgt, ex):
    glossary = "\n".join(f"- {a} => {b}" for a, b in ex["glossary"])
    user = (
        f"Glossary:\n{glossary}\n\n"
        f"Source ({ex['src_lang']}):\n{ex['source']}"
    )
    return {
        "messages": [
            {"role": "system", "content": SYS.format(
                src=ex["src_lang"], tgt=ex["tgt_lang"],
                domain=ex["domain"], register=ex["register"])},
            {"role": "user", "content": user},
            {"role": "assistant", "content": ex["target"]},
        ]
    }

with open("train.jsonl", "w") as f:
    for ex in dataset:
        f.write(json.dumps(to_record(ex["source"], ex["target"], ex), ensure_ascii=False) + "\n")
```

### The adaptation stack

Adaptation is not one step; it is a stack where each layer assumes the one below it.

![The MT adaptation stack: a base multilingual LLM at the foundation, continued pretraining on monolingual target data for low-resource pairs, supervised fine-tuning on instruction-formatted bitext with cross-entropy loss, and preference optimization with DPO or CPO on top](/imgs/blogs/machine-translation-with-llms-6.png)

- **Base LLM** — a multilingual-pretrained model (Qwen, Gemma, Llama, or a translation-specialized base like ALMA). Tokenizer coverage of your target script matters enormously here; a tokenizer that shreds Thai or Khmer into bytes will cap your quality and inflate your cost. (More on this in [designing/choosing a tokenizer](/blog/machine-learning/large-language-model/designing-choosing-tokenizer-llm).)
- **Continued pretraining** — only for low-resource pairs; monolingual target text, standard next-token objective. Teaches the language before you teach the task.
- **SFT** — instruction-formatted bitext, cross-entropy loss. This teaches the *task*: "given this prompt format, produce a faithful translation."
- **Preference optimization** — DPO or CPO, which pushes quality *past* the reference translations in your data. This is the layer that closes the gap between "fluent" and "what a human translator would actually prefer."

A realistic LoRA SFT config with TRL:

```python
## sft_train.py — LoRA SFT for translation with TRL's SFTTrainer.
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

base = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16")

peft_cfg = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

cfg = SFTConfig(
    output_dir="qwen7b-mt-enpt",
    num_train_epochs=2,                 # MT overfits fast; 1-3 epochs
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,                 # LoRA tolerates higher LR than full FT
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    packing=True,                       # pack short segments for throughput
    max_seq_length=1024,
    logging_steps=20,
)

trainer = SFTTrainer(
    model=model, args=cfg, peft_config=peft_cfg,
    train_dataset=load_dataset("json", data_files="train.jsonl")["train"],
    processing_class=tok,
)
trainer.train()
```

Three knobs decide the SFT vs cost tradeoff. Here is how I size them:

| Method | GPU memory (7B) | When to use | Tradeoff |
|---|---|---|---|
| Full SFT | ~80 GB+ (multi-GPU) | > 100k pairs, max quality ceiling | Best ceiling; highest forgetting risk; expensive |
| LoRA (r = 32) | ~24 GB (single GPU) | 5k–100k pairs | ~95% of full-FT quality at a fraction of the cost |
| QLoRA (4-bit + LoRA) | ~12 GB | Memory-constrained / consumer GPU | Small quality hit from 4-bit base; fits a 4090 |

And the hyperparameters that actually move the needle, separated by stage:

| Knob | SFT | Preference (CPO/DPO) | Note |
|---|---|---|---|
| Learning rate | 1e-4 (LoRA) / 1e-5 (full) | 5e-6 | Preference stages want a much lower LR |
| Epochs | 1–3 | 1 | MT overfits fast; watch held-out COMET, not loss |
| LoRA rank | 16–64 | 16–32 | Higher rank lifts quality but raises forgetting |
| Effective batch | 32–64 | 32 | Via gradient accumulation |
| Warmup ratio | 0.03 | 0.03 | Short warmup is enough |

#### Continued pretraining and the tokenizer problem

For low-resource pairs, the SFT-only recipe quietly fails: the base model never saw enough of the target language to have a usable distribution over it, so SFT on a few thousand pairs teaches surface patterns over a shaky foundation. **Continued pretraining** on monolingual target text — tens of millions of tokens, standard next-token objective, low LR (5e-6 to 2e-5), one or two passes — fixes the foundation before you teach the task. This is the front half of the ALMA recipe and it is the difference between a low-resource model that's fluent and one that's merely grammatical.

The under-appreciated half of this is the **tokenizer**. If your base model's tokenizer wasn't trained with coverage of the target script, it falls back to byte-level encoding: a single Khmer or Amharic word becomes ten or fifteen tokens instead of two or three. That triples your sequence length (and serving cost) *and* caps quality, because the model is reasoning over byte fragments rather than morphemes. The fix is vocabulary expansion — add target-language tokens to the embedding matrix and continue-pretrain so the new embeddings learn — which I cover in [task-specialized tokenizer design](/blog/machine-learning/large-language-model/task-specialized-tokenizer-design). Measure your tokenizer's *fertility* (tokens per word) on the target language before you pick a base model; it is the cheapest predictor of low-resource success or failure.

#### Second-order optimization: don't let the model forget how to translate everything else

The non-obvious gotcha with aggressive single-pair fine-tuning is **catastrophic forgetting and off-target collapse**. Fine-tune hard on EN→PT and your model may start emitting Portuguese when asked for Spanish, or lose its zero-shot ability on pairs you didn't train. Mitigations: keep LoRA rank modest, mix a small fraction (5–10%) of other pairs and a slice of general instruction data into the SFT mix, and always evaluate on held-out pairs you *didn't* train, not just the target pair. I devote a full case study to a model that forgot English below.

## Loss functions for MT

This is where translation fine-tuning gets genuinely interesting, and it is the part most teams under-think. Cross-entropy teaches the model to imitate references. But the reference is not the ceiling — it is one human's translation, often literal, sometimes mediocre. The losses beyond cross-entropy are how you train *past* the reference toward what humans actually prefer.

![Loss functions for training MT models compared across four columns — what each optimizes, its training signal, when to use it, and its gotcha — for token cross-entropy, label smoothing, sequence-level MRT, DPO, and CPO](/imgs/blogs/machine-translation-with-llms-7.png)

### Token-level cross-entropy (MLE)

The default SFT objective. Maximize the likelihood of each gold token given the source and the gold prefix:

$$\mathcal{L}_{\text{MLE}}(\theta) = -\sum_{t=1}^{|y|} \log p_\theta\!\left(y_t \mid y_{\lt t},\, x\right)$$

where $x$ is the source, $y$ the reference translation, $y_t$ the $t$-th target token, and $y_{\lt t}$ the gold prefix. It is simple, stable, and parallelizable. Its defining weakness is **exposure bias**: at training time the model always conditions on the *gold* prefix $y_{\lt t}$, but at inference it conditions on its *own* (possibly wrong) prefix. Early mistakes compound. It also optimizes likelihood, which is only loosely correlated with translation quality — a model can assign high probability to a fluent-but-unfaithful output.

### Label smoothing

A one-line change to MLE that meaningfully helps MT. Instead of a one-hot target, you spread a small mass $\epsilon$ uniformly over the vocabulary $V$:

$$q'(k) = (1-\epsilon)\,\mathbb{1}[k = y_t] + \frac{\epsilon}{V}, \qquad \mathcal{L}_{\text{LS}} = -\sum_{k} q'(k)\,\log p_\theta(k \mid y_{\lt t}, x)$$

With $\epsilon \approx 0.1$, this prevents the model from becoming over-confident, which improves **calibration** — and calibration is exactly what you need downstream, because QE-reranking and confidence gating depend on the model's probabilities meaning something. The gotcha is the mirror image: too much smoothing makes the model under-confident and can soften the probability signal you want for reranking. The transformer MT literature has used $\epsilon = 0.1$ as a default since the original "Attention Is All You Need" recipe, and it still holds up.

### Sequence-level training (Minimum Risk Training)

MLE optimizes per-token likelihood; what you actually care about is a sequence-level metric like COMET. **Minimum Risk Training (MRT)** optimizes the *expected cost* of sampled translations directly:

$$\mathcal{L}_{\text{MRT}}(\theta) = \sum_{y \in \mathcal{S}(x)} \frac{p_\theta(y \mid x)^{\alpha}}{\sum_{y' \in \mathcal{S}(x)} p_\theta(y' \mid x)^{\alpha}} \; \Delta(y, y^*)$$

Here $\mathcal{S}(x)$ is a set of sampled candidate translations, $\Delta(y, y^*)$ is a cost like $1 - \text{COMET}(y, y^*)$, and $\alpha$ sharpens the renormalized distribution. The model is rewarded for putting probability mass on candidates the metric likes. This directly attacks the likelihood-vs-quality gap, but it is **high variance** and expensive — you sample $k$ candidates and score them every step — and it is only as good as the metric you optimize. Optimize BLEU and you can get a model that games BLEU; optimize COMET and you inherit COMET's blind spots. MRT is the conceptual ancestor of the RL-for-MT approaches (and of GRPO-style training; see [GRPO vs DPO vs PPO](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide)).

### DPO for translation

Direct Preference Optimization reframes preference learning as a classification loss over pairs of outputs — a winner $y_w$ and a loser $y_l$ for the same source:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\log \sigma\!\left( \beta \left[ \log\frac{p_\theta(y_w \mid x)}{p_{\text{ref}}(y_w \mid x)} - \log\frac{p_\theta(y_l \mid x)}{p_{\text{ref}}(y_l \mid x)} \right] \right)$$

where $\sigma$ is the logistic function, $\beta$ controls how far you move from the reference policy $p_{\text{ref}}$, and the pairs come from human preference or — far more scalably for MT — from a QE model ranking two candidates. DPO is excellent for **aligning style and register**: which of two adequate translations sounds more like your brand. Its gotcha is reference drift — DPO needs a frozen reference model in memory and can degrade if $\beta$ is mis-set. (Full mechanics in [Fine-Tuning LLMs with DPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo).)

### CPO: the loss that pushes past the reference

**Contrastive Preference Optimization** (the loss behind ALMA-R) is, in my experience, the most important single advance in LLM-MT training. It makes two changes to DPO. First, it drops the reference model entirely (a uniform-prior approximation), which halves memory and removes the reference-drift failure. Second, it adds a behavior-cloning NLL anchor on the preferred output so the model stays a competent translator while it learns the preference:

$$\mathcal{L}_{\text{CPO}}(\theta) = \underbrace{-\log\sigma\!\left(\beta\big[\log p_\theta(y_w \mid x) - \log p_\theta(y_l \mid x)\big]\right)}_{\text{preference (no reference model)}} \; \underbrace{-\; \lambda\,\log p_\theta(y_w \mid x)}_{\text{NLL anchor}}$$

The crucial insight from ALMA-R is the *data*: the "winner/loser" triplets are built by taking the gold reference, a strong system's output, and your model's own output, then scoring all three with a QE/reference-based metric and letting the metric pick the winner. Because the metric will sometimes prefer a non-reference translation, the model learns to produce output *better than the human reference it was trained on*. This is the mechanism by which a 7B model overtakes much larger zero-shot systems on WMT.

```python
## cpo_train.py — Contrastive Preference Optimization with TRL.
## Preference triplets are pre-built and scored by COMET-Kiwi; the dataset
## columns are prompt / chosen / rejected.
from trl import CPOConfig, CPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

base = "qwen7b-mt-enpt"  # the SFT checkpoint from the previous step
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="bfloat16")

cfg = CPOConfig(
    output_dir="qwen7b-mt-enpt-cpo",
    beta=0.1,                  # preference temperature
    cpo_alpha=1.0,             # weight on the NLL (behavior-cloning) anchor
    loss_type="sigmoid",       # the DPO-style contrastive term
    learning_rate=5e-6,        # preference stages want a low LR
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    bf16=True,
    max_length=1024,
    max_prompt_length=512,
)

trainer = CPOTrainer(
    model=model, args=cfg, processing_class=tok,
    train_dataset=load_dataset("json", data_files="prefs.jsonl")["train"],
)
trainer.train()
```

The practical pipeline: SFT with cross-entropy (+ label smoothing) to teach the task, then CPO with QE-scored triplets to push past the reference. I have seen this two-stage recipe add 2–4 COMET points over SFT-only on high-resource pairs, and it is the dominant approach on the public MT leaderboards as of 2026.

#### A worked example of why CPO beats SFT

Make it concrete. For the source "The bank raised rates," your SFT model produces a fluent translation $y_{\text{sft}}$. Your training set contains a human reference $y_{\text{ref}}$ that happens to be slightly literal. You also have a strong system's output $y_{\text{sys}}$. Score all three with COMET-Kiwi:

| Candidate | COMET-Kiwi | Role in the triplet |
|---|---|---|
| $y_{\text{ref}}$ (human) | 0.82 | reference |
| $y_{\text{sys}}$ (strong system) | 0.91 | **chosen** ($y_w$) |
| $y_{\text{sft}}$ (our model) | 0.74 | **rejected** ($y_l$) |

The metric prefers the strong system's output *over the human reference*. CPO trains the model to move probability mass from $y_l$ toward $y_w$ — which means toward output the human reference can't reach. MLE could only ever have pushed the model toward the 0.82 reference. That gap, accumulated over a corpus, is the 2–4 COMET points.

The flip side is **reward hacking**: your preference data inherits the QE model's blind spots, so CPO will exploit them. If COMET-Kiwi mildly over-rewards length or fluency, CPO amplifies it into longer, smoother, occasionally less faithful translations. Two guards: keep the NLL anchor weight ($\lambda$, `cpo_alpha`) high enough that the model stays tethered to faithful output, and *always* validate the preference-optimized checkpoint on a held-out set scored by a *different* metric than the one that built the triplets. When you optimize against a metric, you must evaluate with a different one, or you are grading your own homework.

When *not* to add a preference stage: if your SFT model already matches reference quality and you have no QE signal to build triplets from, CPO adds cost and risk for little gain. It earns its keep when you have a quality ceiling to break through and a trustworthy QE model to define "better."

## Decoding and control

**Senior rule of thumb: greedy decoding leaves quality on the table, and an unconstrained decoder will violate every hard requirement you have.** Decoding is not a solved afterthought; it is where you both recover quality and enforce non-negotiable constraints.

![Quality-estimation reranking over n-best candidates: a source sentence is decoded by the LLM into k=4 candidates; fluent and literal candidates pass to a COMET-QE scorer while an off-target candidate is dropped by a constraint check, and the argmax of the QE scores is output](/imgs/blogs/machine-translation-with-llms-8.png)

The graph above is the production decoding loop. Sample $k$ candidates, run each through a constraint check (drop the off-target one outright), score the survivors with a reference-free QE model, and ship the argmax. This single technique — **QE-reranked sampling** — is the cheapest reliable quality win available, typically 1–3 COMET points over greedy.

### Beam vs sampling for MT

A counterintuitive fact: for dedicated NMT, beam search is standard and large beams help. For LLMs, **beam search often underperforms sampling-plus-reranking**, because the LLM's probability surface is peakier and beam search collapses to repetitive, overly literal output. My default for LLM-MT: temperature 0.3–0.6, `top_p` 0.9, sample 4–8 candidates, rerank with COMET-Kiwi. For a single deterministic output where you can't afford reranking, greedy (temperature 0) beats a small beam more often than not.

### Constrained decoding and terminology

Terminology is the requirement that separates a toy from a product. "Mention the glossary in the prompt" fails 15–30% of the time — the model knows the term but doesn't always use it. The robust options, in increasing order of strength:

1. **Soft (prompt) constraints** — list the glossary in the prompt. Easy, partial.
2. **Reranking constraints** — generate n-best, *reject* candidates that violate the glossary, keep the best compliant one. This is what the off-target rejection in the figure does.
3. **Hard (decode-time) constraints** — force the target term into the output via constrained beam search or a logit processor. Guarantees compliance but can produce disfluent output if the term doesn't fit the grammatical slot.

In practice I combine 1 and 2: prompt the glossary *and* reject violators, falling back to hard constraints only for the highest-stakes terms.

### Placeholder and markup preservation

The other routine catastrophe is markup. Source text in the real world is full of `{username}` placeholders, `<b>` tags, ICU message-format plurals, and URLs. An LLM will happily translate `{username}` into `{nombredeusuario}` and break your application. The fix is **mask-translate-restore**: replace each non-translatable span with a sentinel before translation and restore it after.

```python
## protect.py — mask placeholders/markup, translate, restore. The model
## never sees the real tokens, so it can't corrupt them.
import re

## Order matters: longest/most-specific patterns first.
PATTERNS = [
    re.compile(r"<[^>]+>"),               # HTML/XML tags
    re.compile(r"\{[^}]+\}"),             # {placeholder} / ICU
    re.compile(r"https?://\S+"),          # URLs
    re.compile(r"%\w|\$\{?\w+\}?"),       # printf / shell vars
]

def mask(text: str):
    spans, idx = {}, 0
    def repl(m):
        nonlocal idx
        key = f"⸤{idx}⸥"        # rare unicode brackets as sentinels
        spans[key] = m.group(0)
        idx += 1
        return key
    for pat in PATTERNS:
        text = pat.sub(repl, text)
    return text, spans

def restore(text: str, spans: dict) -> str:
    for key, original in spans.items():
        text = text.replace(key, original)
    return text

## A loud failure beats a silent one: if any sentinel is missing post-translate,
## the model ate a placeholder — reject the candidate and retry or escalate.
def validate(translated: str, spans: dict) -> bool:
    return all(key in translated for key in spans)
```

Two engineering notes that have saved me production incidents: pick sentinels that no tokenizer will split or translate (I use rare Unicode brackets, not `[[0]]` which models love to "fix"), and **validate after restore** — if a sentinel vanished, the model corrupted markup and the candidate must be rejected, not shipped.

### Length control

UI strings have length budgets; a German translation that's 40% longer than the English breaks your layout. You can't fully control this, but you can bias it: add a length hint to the prompt ("keep within ~30 characters"), penalize length at rerank time, and for hard limits, generate multiple candidates and pick the shortest adequate one. Don't try to enforce it with `max_new_tokens` — you'll get truncated mid-word garbage.

### Minimum Bayes Risk decoding: the quality ceiling

QE-reranking picks the candidate a QE model likes best. **Minimum Bayes Risk (MBR)** decoding picks the candidate that is, on average, most *similar to all the other candidates* under a utility metric — it bets on the consensus rather than the single highest-scoring outlier:

$$y^{\text{MBR}} = \arg\max_{y \in \mathcal{H}} \; \frac{1}{|\mathcal{H}|}\sum_{y' \in \mathcal{H}} u(y,\, y')$$

where $\mathcal{H}$ is the set of sampled hypotheses and $u$ is a utility like COMET or chrF. The intuition is robustness: a hypothesis that's close to many independent samples is unlikely to contain a hallucination, because hallucinations are idiosyncratic and won't be echoed by the rest of the pool. MBR with a neural utility (COMET) is the strongest decoding strategy for MT and has anchored multiple WMT-winning systems. Its cost is the catch — it is $O(k^2)$ utility evaluations for $k$ candidates, so a 16-sample MBR decode runs 256 COMET comparisons per segment. Reserve it for offline, high-value content; for online traffic, QE-reranking ($O(k)$) is the affordable cousin.

The sampling knobs that feed both rerankers:

| Parameter | Value I use | Effect |
|---|---|---|
| `temperature` | 0.3–0.6 | Enough diversity for the rerank pool without garbage |
| `top_p` | 0.9 | Truncate the unreliable tail |
| `num_return_sequences` ($k$) | 4–8 (QE) / 16–64 (MBR) | The candidate pool size |
| `repetition_penalty` | 1.05 | Guards the repetition collapse LLMs fall into |
| epsilon/eta sampling | optional | Cleaner samples than top-p for MBR pools |

A subtle but important point: the candidates must be *diverse* for reranking to help. Temperature 0.1 gives you eight near-identical samples and reranking does nothing. The whole value of the n-best list is that it spans genuinely different translations — one fluent, one literal, one that reordered the clause — so the reranker has a real choice to make.

## Edge cases that break LLM translators

**Senior rule of thumb: every recurring LLM-MT failure has a known root cause and a concrete mitigation — none of them are solved by a bigger model.** I keep this taxonomy taped (metaphorically) to my monitor, because every one of these has bitten me in production.

![Edge-case failure taxonomy for LLM translation: six failure modes — off-target language, hallucination, term/NE drift, markup/JSON corruption, number/date errors, and document-context errors — each mapped to its symptom, root cause, and concrete mitigation](/imgs/blogs/machine-translation-with-llms-9.png)

Going through the taxonomy:

- **Off-target language** — the model outputs the wrong target language entirely (often a related or higher-resource one). Root cause is weak language steering, worse on low-resource targets. Mitigation: an explicit target-language tag in the prompt, a few-shot example in the right language, and a cheap LID check on the output that rejects and retries. This is common enough that you must build the LID gate; I have a war story below where its absence cost a full night's batch.

- **Hallucination / over- and under-generation** — the model adds content not in the source, or silently drops a clause. Root cause is a combination of high decoding temperature and long inputs. Mitigation: a QE gate (COMET-Kiwi catches most adequacy failures) plus a length-ratio sanity check.

- **Terminology / named-entity drift** — a brand name, product, or person's name gets translated or altered. Root cause is treating the glossary as a soft hint. Mitigation: hard constraints plus a maintained termbase, as covered above.

- **Markup / JSON corruption** — tags eaten, JSON structure broken, placeholders translated. Root cause is the model having no structural awareness of the payload. Mitigation: the mask-restore-validate pattern.

- **Number, date, and unit errors** — `1,000` becomes `1.000` (changing its value by three orders of magnitude in some locales), or a date format flips. Root cause is locale ambiguity the model resolves by guessing. Mitigation: regex-protect numbers like placeholders, and apply explicit locale rules in post-processing rather than trusting the model.

- **Document-context errors** — pronoun, gender, and formality mistakes that are *invisible at the sentence level* but obvious in context. "It" needs a gender in the target; "you" needs a formality level. Root cause is sentence-by-sentence translation. Mitigation: a document-level prompt window that gives the model the surrounding sentences, even if you only emit one.

Two of these are cheap enough to show in full, because they are the ones I see teams skip most often. The output-side language-ID gate that prevents off-target shipping:

```python
## lid_gate.py — reject off-target output and retry with stronger steering.
import fasttext
lid = fasttext.load_model("lid.176.bin")

def detect(text: str) -> str:
    return lid.predict(text.replace("\n", " "))[0][0].replace("__label__", "")

def translate_with_lid_gate(translate_fn, source, tgt_lang, max_retries=2):
    out = translate_fn(source, hint=None)
    for attempt in range(max_retries):
        if detect(out) == tgt_lang:
            return out, True
        # Off-target: retry with an explicit tag + a target-language exemplar.
        out = translate_fn(source, hint=f"Respond ONLY in {tgt_lang}.")
    return out, False   # still off-target -> escalate, never ship silently
```

And number/unit protection, which treats numbers like placeholders so the model can't relocate a decimal point:

```python
## numbers.py — freeze numeric spans, then re-localize them deterministically.
import re
NUM = re.compile(r"\d[\d.,]*\s?(?:%|kg|km|°C|°F|\$|€|£)?")

def freeze_numbers(text):
    spans, idx = {}, 0
    def repl(m):
        nonlocal idx
        key = f"⟦N{idx}⟧"; spans[key] = m.group(0); idx += 1
        return key
    return NUM.sub(repl, text), spans

## After translation, restore each frozen number with the TARGET locale's
## grouping/decimal separators applied via Babel — never trust the model
## to convert "1,000" -> "1.000" for a European locale.
```

The meta-lesson: **none of these are model-quality problems.** A GPT-class model and a fine-tuned 7B fail the same way on all six. They are *system* problems, and the system — the gate, the mask, the LID check, the document window — is what you build to fix them.

## Evaluation that survives contact with reality

**Senior rule of thumb: BLEU ranks systems, but only learned QE and human MQM tell you whether a translation is shippable.** The single most common mistake I see is treating a metric number as ground truth. Metrics live on a ladder of trust and cost, and you use them for different jobs.

![The MT evaluation ladder: five layers from cheap surface n-gram metrics (BLEU, chrF) at the top, through embedding metrics (COMET, BLEURT), reference-free QE (COMET-QE) highlighted as the production workhorse, LLM-as-judge with an MQM rubric, down to human MQM as the gold standard](/imgs/blogs/machine-translation-with-llms-10.png)

From cheap-and-shallow to expensive-and-trusted:

1. **Surface n-gram (BLEU, chrF/chrF++).** Cheap, deterministic, and *only* good for ranking systems and tracking regressions in CI. BLEU's correlation with human judgment is weak enough that a higher BLEU does not reliably mean a better translation. Always report it with [sacreBLEU](https://github.com/mjpost/sacrebleu) so the score is reproducible — a bare "BLEU 34" is meaningless without the tokenization signature. chrF++ is more robust for morphologically rich languages.

2. **Embedding / learned metrics (COMET, BLEURT).** These compare source, hypothesis, and reference in a learned embedding space and correlate far better with humans. **COMET** (specifically `wmt22-comet-da` and its successors) is the reference-based metric I trust for system comparison.

3. **Reference-free QE (COMET-Kiwi / COMET-QE).** The production workhorse, highlighted in the figure. It scores a translation *without a reference*, using only source and hypothesis — which means you can run it on live traffic where no reference exists. This is what powers the QE gate and the n-best reranker. If you build one thing from this article, build this.

4. **LLM-as-judge with an MQM rubric.** Prompt a strong LLM to score a translation against the Multidimensional Quality Metrics rubric — accuracy, fluency, terminology, locale errors — and return structured error spans. Excellent for *diagnostic* evaluation (what kind of errors, where) and increasingly used for preference-pair generation. Watch for the known failure mode where judges reward verbosity or get fooled by superficial cues (see [One Token to Fool LLM-as-a-Judge](/blog/paper-reading/ai-interpretability/one-token-to-fool-llm-as-a-judge) and our broader take in [Evaluating Conversational LLMs Beyond Perplexity](/blog/machine-learning/large-language-model/evaluating-conversational-llms-beyond-perplexity)).

5. **Human MQM.** The gold standard: trained linguists annotate error spans with severities. Slow and expensive, so you sample — but it is the only metric that is *ground truth*, and you calibrate every automatic metric against it.

Here is a practical two-tier evaluation: COMET for the automatic number, and an MQM-style LLM judge for diagnostics.

```python
## evaluate.py — automatic COMET + an MQM-style LLM judge for error diagnostics.
from comet import download_model, load_from_checkpoint
import anthropic, json

comet = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

def comet_score(srcs, hyps, refs):
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    return comet.predict(data, batch_size=32, gpus=1).system_score

MQM_RUBRIC = """You are an MQM annotator. Score this translation.
Return JSON: {"accuracy": 0-100, "fluency": 0-100, "terminology": 0-100,
"errors": [{"span": str, "category": "accuracy|fluency|terminology|locale",
"severity": "minor|major|critical"}]}.
Source ({src_lang}): {source}
Translation ({tgt_lang}): {hypothesis}"""

client = anthropic.Anthropic()

def mqm_judge(source, hypothesis, src_lang, tgt_lang):
    msg = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=[{"role": "user", "content": MQM_RUBRIC.format(
            src_lang=src_lang, tgt_lang=tgt_lang,
            source=source, hypothesis=hypothesis)}],
    )
    return json.loads(msg.content[0].text)
```

A few evaluation disciplines I refuse to skip:

- **Use a held-out, leak-checked test set.** FLORES-200 and the WMT test sets are standard, but verify they didn't leak into training (see the dedup note above).
- **Report COMET *and* a confidence interval over segments**, not a single BLEU. System-level differences under ~1 COMET point are usually noise.
- **Slice by phenomenon.** Aggregate scores hide the failure that matters — break out numbers, named entities, long sentences, and each domain separately.

How the metrics actually relate to human judgment (approximate, and the exact numbers shift year to year with each metric release — treat this as the ordering, not gospel):

| Metric | Type | Human correlation | Needs reference? | Cost |
|---|---|---|---|---|
| BLEU | n-gram overlap | low | yes | trivial (CPU) |
| chrF++ | character n-gram | low–moderate | yes | trivial (CPU) |
| BLEURT | learned (BERT) | high | yes | GPU |
| COMET (`wmt22-comet-da`) | learned | high | yes | GPU |
| COMET-Kiwi | learned QE | high | **no** | GPU |
| MetricX | learned | high | yes | GPU |
| Human MQM | expert annotation | ground truth | n/a | very high |

And because system-level differences are noisy, report significance — paired bootstrap resampling over your test segments, not a bare delta:

```python
## bootstrap_sig.py — is system A really better than B, or is it noise?
import numpy as np

def paired_bootstrap(scores_a, scores_b, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.array(scores_a), np.array(scores_b)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    wins = (a[idx].mean(1) > b[idx].mean(1)).mean()
    return wins  # P(A > B); ship A only if this clears ~0.95

## A 0.8-COMET-point "win" with p = 0.6 is not a win. Many shipped
## regressions are exactly this: a noisy delta treated as signal.
```

If `paired_bootstrap` returns 0.6, your "improvement" is a coin flip. I have killed more than one celebratory launch with this twenty-line function, and saved the team from shipping a regression dressed up as a win.

## Quality estimation as first-class infrastructure

If I had to keep exactly one component from this entire pipeline, it would not be the model — it would be the quality-estimation model. QE is the single most leveraged piece of MT infrastructure because the same reference-free score does six different jobs, and each one would otherwise require its own system:

1. **Gating.** Accept high-confidence output, route the rest to human post-editors. The threshold *is* a business dial: raise it and you ship fewer translations automatically but spend more on human review; lower it and you ship more at the risk of letting errors through. Tuning it is tuning your gross margin.
2. **Reranking.** Score the n-best list and pick the winner, as in the decoding section.
3. **Data filtering.** Drop low-adequacy pairs from the training corpus, as in the data section.
4. **Preference data.** Build the chosen/rejected triplets that feed CPO/DPO.
5. **Routing.** Score the source's difficulty (or a cheap model's draft) and send easy segments to dedicated NMT while reserving the expensive LLM+rerank path for the hard ones. This is how you get LLM quality at closer to NMT cost across a mixed workload.
6. **Active learning.** The segments where the QE model is *least* confident are exactly the ones worth sending for human annotation — they are your next training batch and your next eval slice.

Choosing the QE model matters. **COMET-Kiwi** (the reference-free COMET variant) is the standard workhorse; **xCOMET** additionally emits error spans and severities, which turns a single number into a diagnostic you can act on; the larger XXL variants trade latency for accuracy. Whatever you pick, remember the QE model is itself a model with biases and blind spots — it can be gamed (the reward-hacking risk from the loss section) and it can drift.

Two disciplines make QE trustworthy in production. First, **calibrate the threshold against human MQM, per domain.** A raw QE score is not a probability of correctness, and the score that means "shippable" for casual UGC is not the score that means "shippable" for a medical leaflet. Sample human judgments in each domain and find the QE threshold that achieves your target post-edit rate. Second, **monitor the QE distribution in production.** A sudden shift in the live QE score distribution is one of the earliest signals you have that something upstream changed — new content type, a model regression, a corrupted glossary — often before users complain. I alarm on the QE distribution the same way I alarm on latency.

> Build the QE model first. The translator is replaceable; the quality gate is what makes any translator safe to ship.

## Use cases and serving patterns

The same model serves different products through different pipeline configurations. A few patterns I've shipped:

- **Localization (l10n) pipelines.** Highest quality bar, latency-tolerant, terminology-critical. Full pipeline: TM lookup → glossary injection → fine-tuned model with QE reranking → confidence gate → human post-edit for anything below threshold. The QE gate is the economic lever — it decides how much human time you spend.

- **Real-time chat / support translation.** Latency-critical, lower quality bar, huge volume. Here a dedicated NMT model or a small fine-tuned LLM served on [CTranslate2](https://github.com/OpenNMT/CTranslate2) or [vLLM](/blog/machine-learning/large-language-model/vllm-inference) wins; you cannot afford n-best reranking on every message. Cache aggressively — a large fraction of support messages are near-duplicates.

- **Document translation.** Context is everything. Translate with a sliding document window so the model sees surrounding sentences, maintain a per-document terminology cache so the same term is translated consistently throughout, and preserve structure (headings, lists, tables) with the mask-restore pattern applied to the document's markup.

- **RAG and agentic translation.** Retrieve similar past translations as few-shot examples (a TM is a retrieval index), or run a translate→critique→refine agent loop for the highest-value content. The agent loop is essentially the QE-reranked pattern with an LLM critic instead of a QE model — more flexible, much slower.

One pattern cuts across all four: **fall back, don't fail.** Every production translation path needs a graceful degradation chain — if the LLM times out, serve the dedicated NMT output; if that fails, serve the fuzzy TM match; if there's no match, return the source with a visible "translation unavailable" marker rather than a 500. Users tolerate an imperfect translation far better than a blank screen or an error, and a service that crashes on its hardest inputs is worse than a dumber one that always returns something usable. I wire the fallback chain *before* I tune any single stage, because the chain is what sets the user-visible reliability number — and that number, not the COMET score, is the one the business will ask about when something breaks.

### Serving economics, with numbers

The serving mode is a cost decision before it is a quality one. Rough numbers from systems I've run (single A100-class GPU, EN→European pair, normalize against your own load test):

| Mode | Engine | p50 latency | Throughput | Relative cost/Mchar |
|---|---|---|---|---|
| Dedicated NMT | CTranslate2 (INT8) | 15–40 ms | very high | 1× (baseline) |
| Fine-tuned 7B, greedy | vLLM (bf16) | 120–250 ms | high | ~8× |
| Fine-tuned 7B, k=4 + QE rerank | vLLM + COMET-Kiwi | 300–600 ms | moderate | ~20× |
| Agentic refine (2 LLM passes) | vLLM ×2 | 1–3 s | low | ~50× |

Three levers flatten that cost curve. **Caching**: a large fraction of real translation traffic is exact or near-duplicate (support macros, repeated UI strings, common product phrases). A simple exact-match cache on (source, src_lang, tgt_lang, domain) routinely serves 30–60% of support traffic at zero model cost; a fuzzy TM cache catches more. **Continuous batching** (vLLM's default) is what makes LLM serving economical at all — see [vLLM inference](/blog/machine-learning/large-language-model/vllm-inference) and [serving LLMs at scale](/blog/machine-learning/large-language-model/serving-llms-at-scale-production-systems) for the mechanics; without it your GPU sits idle between requests. **Quantization**: INT8 for NMT and FP8/INT4 for the LLM cut cost with minimal quality loss for translation, which is more robust to quantization than reasoning tasks. The QE gate is the last lever and the most powerful — it converts model confidence directly into human-review cost, so tuning its threshold is tuning your margin.

## Case studies from production

These are real failure shapes (sanitized). Each one taught me something the leaderboards never would.

### 1. The off-target midnight batch

A nightly batch translated a marketing catalog EN→PT (Portuguese). One morning, support lit up: the entire Brazilian site was in Spanish. The model — a fine-tuned 7B — had, for a subset of segments with very short or ambiguous source text, decoded into Spanish instead of Portuguese. The wrong first hypothesis was "the model regressed." The actual root cause: the batch had no language-ID gate on the output, and short ambiguous inputs ("Sale!", "New") pushed the model toward the higher-resource Spanish it had seen more of. The fix was three lines — a fastText LID check on every output that rejected and retried with an explicit target tag and a few-shot Portuguese example. The lesson: **off-target is not rare, it is a routine tail event, and you must gate for it.** We now treat an output-side LID check as mandatory infrastructure, not an optimization.

### 2. The glossary that wasn't

A legal customer's brand name — call it "Ironclad" — kept getting translated to its literal meaning in the target language across their terms of service. The team had dutifully listed it in the prompt glossary. The wrong hypothesis was "the glossary isn't being injected." It was being injected fine; the model simply chose to translate it anyway, ~20% of the time, because soft prompt constraints are advisory. The fix was to move brand names and legal terms from soft prompt constraints to **rerank-time hard rejection plus a constrained-decoding fallback** for the top-priority terms. The lesson: **a glossary in the prompt is a suggestion; a glossary enforced at decode/rerank time is a guarantee.** For legal copy you need the guarantee.

### 3. The backtranslation feedback loop

A low-resource pair (EN→a regional language) plateaued, so the team scaled up backtranslated data from 30% to 80% of the mix to "add more signal." Quality on human eval *dropped* even as BLEU held steady. The wrong hypothesis was "we need even more data." The root cause: the reverse model producing the synthetic source was weak, so the forward model was learning to translate stilted translationese rather than natural language, and BLEU — which rewards n-gram overlap with translationese references — couldn't see it. The fix: cap backtranslation at 40%, tag it, and add a small slice of gold in-domain TM. COMET and human eval recovered. The lesson: **synthetic data has a quality cliff, and the metric that's blind to it (BLEU) will cheerfully tell you everything is fine.**

### 4. BLEU went up, users complained

A model update raised BLEU by 1.8 points. The team shipped it. Within a week, customer complaints about "robotic" and "overly literal" translations spiked. The wrong hypothesis was "users are noticing unrelated issues." The root cause: the update had nudged the model toward more literal, higher-n-gram-overlap output — exactly what BLEU rewards and what humans dislike. We had been flying on BLEU alone. The fix was procedural, not technical: we added COMET and a sampled human MQM eval to the release gate, and we caught the regression on the *next* update before it shipped. The lesson: **BLEU and human preference can move in opposite directions; never gate a release on BLEU alone.** This is the single most expensive lesson in the list because it was an organizational blind spot, not a code bug.

### 5. JSON eaten by the translator

A localization pipeline translated UI strings stored as JSON. Periodically a deploy would fail because a translated string had broken the JSON — a quote that became a typographic quote, a `{count}` placeholder translated, an escaped newline mangled. The wrong hypothesis was "we need better JSON parsing on our side." The root cause was that we were passing structured payloads through the model and trusting it to leave structure alone. The fix was the mask-restore-validate pattern: extract every placeholder and structural token, translate only the human-readable spans, restore, and *validate* that every sentinel survived — rejecting and escalating any candidate that lost one. JSON breakage went to zero. The lesson: **never let the model see tokens it must not change; mask them.**

### 6. Document context vs sentence context

A model translating help-center articles kept getting formality and pronoun gender wrong in a way that QA flagged but the team couldn't reproduce on individual sentences. The wrong hypothesis was "the model doesn't understand formality." It understood formality fine — at the sentence level it simply had no information to decide. "Click here to update your settings" gives no clue whether "your" should be formal or informal in the target, but the article's opening paragraph did. The fix was to switch from sentence-level to **document-windowed translation**: feed the model the surrounding context (and a per-document formality setting) even when emitting one sentence. Formality errors dropped by more than half. The lesson: **many "model errors" are actually missing-context errors; give the model the context a human translator would have.**

### 7. The LoRA that forgot English

A team fine-tuned a base model hard on a single low-resource pair to maximize quality, training for too many epochs at too high a LoRA rank. Quality on the target pair was excellent. Then they discovered the model had become measurably worse at every *other* pair, and on monolingual English generation it had started inserting target-language words. The wrong hypothesis was "the base model was always this weak." The root cause was catastrophic forgetting from over-aggressive single-task fine-tuning. The fix: lower the rank, cut to 2 epochs, mix in 10% other-pair and general-instruction data, and add held-out pairs to the eval suite as a forgetting tripwire. The lesson: **fine-tuning is a trade; measure what you're trading away, not just what you're buying.**

### 8. The tokenizer tax

A team added a low-resource Southeast-Asian language to a product and were baffled by two things at once: serving cost for that pair was roughly 4× every other pair, and quality was stuck no matter how much parallel data they added. The wrong hypothesis was "we need more data and more parameters." The root cause was the tokenizer: the base model had near-zero coverage of the target script, so every word was shredded into a long run of byte tokens. Sequences were three to four times longer (hence the cost) and the model was modeling byte fragments instead of words (hence the quality ceiling). The fix was vocabulary expansion plus continued pretraining on monolingual target text so the new token embeddings learned a meaningful distribution. Fertility dropped from ~4.5 tokens/word to ~1.6, cost normalized, and quality finally responded to data. The lesson: **measure tokenizer fertility on the target language before you commit to a base model — it caps both cost and quality and no amount of parallel data fixes it.**

### 9. The over-eager refiner

A high-value pipeline used a translate→critique→refine agent loop to squeeze out maximum quality. On audit, the refine step was making things *worse* on a meaningful slice: handed an already-correct translation, the critic LLM would invent a "problem" and the refiner would "fix" it into something further from the source. The wrong hypothesis was "the critic prompt needs tuning." The deeper issue was that the loop had no notion of *when to stop* — it refined unconditionally. The fix was to gate the refine step on a QE delta: only accept a refinement if its COMET-Kiwi score beats the original by a real margin, otherwise keep the original. The refine step went from a net-negative on easy inputs to a net-positive only where it helped. The lesson: **an unconditional improvement step is a regression generator; gate every refinement on a measurable quality gain.**

### 10. The context window that ate the budget

After moving help-center translation to document-windowed prompting (case study 6's fix), the token bill jumped almost 10×, and very long articles started silently failing because the assembled window blew past the context limit. The wrong hypothesis was "document mode is just expensive, accept it." The root cause was naive windowing: the team was prepending the *entire* document to every sentence's prompt, so an N-sentence document cost $O(N^2)$ tokens, and long ones overflowed. The fix was a bounded sliding window — a few sentences of local context plus a short running summary of the document's key terminology and formality decisions, carried forward rather than re-sent in full. Cost dropped back to ~2× sentence-level, the overflow failures vanished, and the contextual quality gains were retained. The lesson: **document context is a sliding window with a carried-forward summary, not "paste the whole document every time" — the quadratic sneaks up on you.**

## When to reach for LLM-MT — and when not to

Reach for an LLM-based translator when:

- You need **context, world knowledge, or document-level coherence** — formality, gender, pronoun resolution, idioms — that sentence-level NMT can't see.
- You have **terminology and style requirements** that benefit from instruction-following and few-shot conditioning.
- You translate **many pairs or domains** and want one model and one pipeline instead of a fleet of checkpoints to maintain.
- You can afford to **fine-tune and add a QE/preference stage**, which is where LLM-MT decisively beats both zero-shot LLMs and most dedicated NMT.
- Your content is **high-value and latency-tolerant** enough to pay for n-best reranking.

Skip LLM-MT (or use dedicated NMT) when:

- You are translating a **high-volume, latency-critical firehose** (live chat, UGC) where sub-50 ms and per-character cost dominate. A Marian/CTranslate2 model is the right tool.
- You have a **single high-resource pair with abundant clean bitext** and no special context needs — a dedicated NMT model will match an LLM at a fraction of the serving cost.
- You **can't invest in the surrounding pipeline.** A bare LLM API call with no glossary enforcement, no QE gate, and no markup protection will underperform a mature NMT system *and* embarrass you on the edge cases. The LLM's advantage only materializes with the pipeline around it.
- You need **hard determinism and auditability** (some regulated workflows). Sampling-based LLM-MT is harder to make bit-reproducible than a deterministic NMT decode.

The honest summary: in 2026, a fine-tuned LLM with continued pretraining for low-resource pairs, CPO on QE-scored preference data, QE-reranked decoding, hard terminology constraints, and a learned-QE confidence gate is the strongest general-purpose machine translation system you can build — *and* dedicated NMT is still the right answer for a large, latency-bound, high-resource slice of traffic. The senior move is not picking a side. It is building the portfolio and routing each request to the cheapest stage that meets its quality bar.

## Further reading

- [Qwen-MT: Machine Translation as a Reinforcement-Learned LLM Task](/blog/paper-reading/large-language-model/qwen-mt-multilingual-translation) — a concrete RL-for-MT system.
- [Training LLMs to Adapt to a New Language](/blog/machine-learning/large-language-model/training-llm-adapt-new-language) — the continued-pretraining mechanics for low-resource pairs.
- [Effective LLM Fine-Tuning Techniques](/blog/machine-learning/large-language-model/effective-llm-fine-tuning-techniques) and [Fine-Tuning LLMs with DPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-dpo) — the SFT and preference-optimization foundations specialized above.
- [Evaluating Conversational LLMs Beyond Perplexity](/blog/machine-learning/large-language-model/evaluating-conversational-llms-beyond-perplexity) — the broader evaluation philosophy behind the MT ladder.
- The COMET and ALMA/ALMA-R papers (Unbabel; Xu et al.) for the QE-metric and CPO foundations; the WMT shared-task findings for the current state of the art and the metric–human correlation studies that justify the evaluation ladder.
- The NLLB-200 and M2M-100 technical reports for the data-mining and many-to-many multilingual recipes that still define the dedicated-NMT side of the portfolio, and the FLORES-200 benchmark for low-resource evaluation done right.
