---
title: "Kaggle LLM Science Exam: how the top 20 teams won with retrieval, not bigger models"
date: "2026-10-28"
publishDate: "2026-10-28"
description: "A synthesis of the top-20 writeups from Kaggle's 2023 LLM Science Exam: why every serious team built a Wikipedia RAG stack, how H2O fit five 7B readers and a 13B into nine hours with a KV-cache trick, and what a zero-training 70B baseline taught everyone."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "rag", "retrieval", "deberta", "lora", "reranking", "multiple-choice-qa", "faiss"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 22
---

> **TL;DR.** Kaggle's LLM Science Exam (July to October 2023, 2,664 teams, a USD 50,000 prize pool) asked an offline notebook to answer five-option science questions that GPT-3.5 had written from Wikipedia. The winner, Team H2O LLM Studio, scored 0.933 MAP@3 by spending almost all of its effort on **retrieval**: 30 to 60 million Wikipedia chunks, five embedding models, and a search engine that was just a matrix multiply on two GPUs. The reader was a fixed set of 7B and 13B models turned into binary classifiers. All 17 top-20 teams that published a writeup ran RAG over Wikipedia. **The lesson:** when the questions were written from a corpus, finding the right paragraph is the model.

![The lever strip: of the five levers that decided Kaggle LLM competitions, Science Exam was won on retrieval](/imgs/blogs/kaggle-llm-science-exam-solutions-1.webp)

The strip above is the mental model for this series, introduced in [how Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work). Every Kaggle LLM competition from 2023 to 2026 was decided by one of five levers: data, retrieval, fitting more model into a fixed notebook, test-time compute, or understanding the metric better than everyone else. Science Exam is the cleanest retrieval story of the set, and the competition where the community learned, in public, that a 175B question-writer can be beaten by a 7B reader handed the right paragraph.

## The task in 60 seconds

The host framed the challenge as a size mismatch: "the largest models run on Kaggle are around 10 billion parameters, whereas gpt3.5 clocks in at 175 billion." GPT-3.5 wrote multiple-choice questions from Wikipedia snippets on scientific topics, easy questions were filtered out, and the question was whether a model ten times smaller could ace the test.

![The notebook budget: two T4 GPUs or one P100, a nine hour limit, no internet, 200 labelled questions against a hidden test of about 4,000 split 20/80 between the public and private leaderboards](/imgs/blogs/kaggle-llm-science-exam-solutions-2.webp)

Notice how small the labelled set is. The organisers gave out **200 labelled questions** (`prompt, A, B, C, D, E, answer`) against a hidden test of about 4,000 that "may be different in subject matter", with no generation details. Submissions were notebooks with internet off and at most nine hours of runtime; public external data and pre-trained models were allowed. Participants' posts put the hardware at two 16 GB T4s or one P100, and simjeg noted that 70B models "can't fit a T4 or a P100 GPU". The public leaderboard scored 20% of the test (about 800 questions), the private one the rest.

![An illustrative question in the style of the dataset: a sentence from the Wikipedia article on the Droste effect becomes a prompt with one correct option and four plausible distractors](/imgs/blogs/kaggle-llm-science-exam-solutions-3.webp)

The sample above is invented for illustration, not a real test row, but it has the right shape: one Wikipedia sentence turned inside out, plus four distractors GPT-3.5 found plausible. We borrowed the page from the 14th-place writeup, which traced real questions to Wikipedia science *categories* that include oddly non-scientific pages such as "Droste effect" in Category:Symmetry.

## The metric: MAP@3, worked on four questions

You submit up to three ranked guesses per question:

$$
\text{MAP@3} = \frac{1}{U}\sum_{u=1}^{U}\ \sum_{k=1}^{\min(n,3)} P(k)\,\text{rel}(k)
$$

$U$ is the number of questions, $n$ the number of guesses, $P(k)$ the precision at cutoff $k$, and $\text{rel}(k)$ is 1 if the guess at rank $k$ is correct; repeats after the first hit are ignored. With one correct answer per question, the formula collapses to one over the rank of the correct guess, or zero.

![MAP@3 on a toy set of four questions: correct at rank 1 scores 1, at rank 2 scores one half, at rank 3 one third, and missing the top three scores zero, for a mean of 0.458](/imgs/blogs/kaggle-llm-science-exam-solutions-4.webp)

The four toy questions score 1.000, 0.500, 0.333 and 0.000, so the mean is ${1.833 / 4 = 0.458}$. Two consequences followed. A model only needs to *rank* five options, so any scorer that outputs one number per option works, including a binary "is this option correct?" classifier. And the second and third slots are cheap insurance: moving a correct answer from rank 4 to rank 3 is worth a third of a question, which rewards calibrated probabilities over hard decisions.

## What made it hard

**The labelled data was useless as validation.** Once teams added retrieval, the 200 samples saturated: H2O and 7th place both report about 0.99 on them. The teams that kept improving built harder held-out sets: H2O used 6,000 STEM questions from a public dataset, 4th place drew 2,000 rows from the open datasets, and 5th place validated on 1,000 of their own GPT-3.5 questions that never touched training.

**The knowledge was not in the model.** steubk's ablation from 19th place shows the size of the gap: a DeBERTa with no context scored 0.805 private, and with retrieved context 0.884. That jump of 0.079 is larger than the entire spread of the top 50.

**Nine hours, small GPUs, no internet.** Every byte of Wikipedia had to be uploaded as a dataset and searched inside the notebook. H2O's final submission read 2.5 TB of input data and fit "precisely" in nine hours. Running a 70B model at all meant streaming it layer by layer through one GPU, a trick simjeg published mid-competition.

**The corpus lied about numbers.** Public Wikipedia parsers silently drop values rendered by Lua templates, which in science articles are often the numbers. A question about a constant could retrieve the right paragraph with the answer missing. Several top teams found this independently.

## The top-20 scorecard

Ranks and scores come from Kaggle's leaderboard API. A team that sat 10th was disqualified after the deadline for an illegal team merge, so several writeup titles are one place too high (kami's says "10th(11th)", chicm's "12th" for rank 11). The table cites the API rank.

| Rank | Team | Private | Public → private | Key idea | Source |
|---|---|---|---|---|---|
| 1 | Team H2O LLM Studio | 0.933208 | 1 → 1 | 30 to 60M-chunk RAG, 5 embedders, matmul search; 7B/13B binary readers with KV-cache reuse | [writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446422) |
| 2 | lytic | 0.931853 | 6 → 2 | BM25 over full Wikipedia, DeBERTa context reranker, DeBERTa and Mistral readers, XGBRanker fusion | [writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/448256) |
| 3 | Podpall | 0.928415 | 10 → 3 | Own parse, passage retrieval, tuned cross-encoder reranker, encoders plus 70B on hard questions | [writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446358) |
| 4 | Preferred Scantron | 0.927633 | 4 → 4 | In-kernel Elasticsearch, three context orderings, DeBERTa up to 1,280 tokens | [writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446307) |
| 5 | Preferred ChattyKids | 0.926279 | 14 → 5 | BM25 plus 2 dense retrievers; Mistral-7B on all, Llama-2-70B on the least-confident 40% then 5% | [writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446293) |
| 6 | RAG to riches | 0.925914 | 7 → 6 | Crawled STEM corpus, fine-tuned retrievers, DeBERTa plus Mistral LoRA, 70B on top-5% entropy | [writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/447647) |
| 7 | days | 0.925184 | 5 → 7 | Four context sources, 8 DeBERTa runs plus 3 LLMs trained as reward models, cascade | [writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/447155) |
| 8 | TT | 0.924507 | 13 → 8 | unknown | no writeup |
| 9 | SUSTech BDI | 0.923205 | 19 → 9 | unknown (team includes simjeg) | no writeup |
| 10 | kami | 0.922215 | 16 → 10 | Four wiki sources, one DeBERTa-large, max-probability ensemble | [writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446248) |
| 11 | ScienceGPT | 0.921329 | 20 → 11 | Own 480k STEM set, "one-shot" DeBERTa, Llama-2-7B LoRA | [writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446660) |
| 12 | kaggle is few shot learner | 0.921329 | 2 → 12 | 56M vectors searched in shards, TF-IDF rerank, DeBERTa max ensemble | [writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446301) |
| 13 | kurupical | 0.921121 | 15 → 13 | Sliding 90-word chunks, train 256 / infer 768 tokens, retrieval-rank TTA | [writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446231) |
| 14 | Maxim Tsygankov | 0.920495 | 47 → 14 | Reverse-engineered the category generator; 750k own questions | [writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446484) |
| 15 | Life is full of surprise | 0.920183 | 3 → 15 | Five recall paths, SimCSE embedders with hard negatives | [writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446816) |
| 16 | CSP | 0.918672 | 100 → 16 | unknown | no writeup |
| 17 | Sadge | 0.918620 | 9 → 17 | Local MediaWiki server rendering 1.1M STEM pages; max ensemble | [writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446242) |
| 18 | Chan Kha Vu | 0.918203 | 8 → 18 | All-Wikipedia passage index, about 54M vectors in 6 GB IVF-PQ | [writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446261) |
| 19 | steubk | 0.917838 | 25 → 19 | MiniLM search plus TF-IDF rerank; zero-shot 70B merged with DeBERTa | [writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446395) |
| 20 | SkyFish Up Up | 0.916901 | 17 → 20 | Four RAG contexts, one DeBERTa, max plus mean ensemble | [writeup](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/448114) |

Ranks 11 and 12 tied exactly; the API order is kept. The top 20 span 0.0163 MAP, about 52 question-equivalents on roughly 3,200 private questions, and first and second are separated by 0.00136, about four.

## The top-20 landscape

Seventeen of the twenty teams published a writeup. Coding each for what it says it did, and leaving unknowns as question marks, gives this matrix.

![Technique matrix for the top 20 teams: rows are techniques, columns are private ranks 1 to 20, with ranks 8, 9 and 16 greyed out as having no writeup; RAG over Wikipedia is used by all 17 teams with writeups, while LLM readers and confidence cascades cluster in the top seven](/imgs/blogs/kaggle-llm-science-exam-solutions-5.webp)

**Table stakes.** All 17 ran RAG over Wikipedia, and at least 16 fused several retrieval variants late (3rd place does not say). Fifteen used dense retrieval and 11 a sparse retriever (BM25, TF-IDF or Elasticsearch). Fifteen read with DeBERTa-v3-large; only 1st and 5th used LLM readers alone. Eight say they built a validation set harder than the given 200.

**What separated the top seven from ranks 10 to 20:**

1. **An LLM reader.** Six teams fine-tuned a 7B to 13B reader (1, 2, 5, 6, 7, 11), all in the top 11. Four ran a 70B on a Kaggle GPU (3, 5, 6, 19). The 4th-place team is the DeBERTa-only exception in the top seven, compensating with Elasticsearch and 1,280-token contexts.
2. **A confidence cascade.** All five teams that ran a cheap reader on everything and a big one only on uncertain questions (2, 3, 5, 6, 7) finished in the top seven. That is how a 70B fits in nine hours.
3. **A trained reranker or retriever** (2, 3, 6, 14, 15). Podpall's ablation puts his tuned reranker at 0.017 MAP, about the width of the whole top 20.

Teams 10 to 20 mostly polished retrieval and ensembled one DeBERTa across contexts.

**Where the writeups contradict each other:**

| Question | One side | Other side |
|---|---|---|
| Train on the true source context? | 1st: worse than noisy RAG context | 11th: no gain; 18th: "training on worse context gives better models" (+0.005). No dissent. |
| Compress the vector index? | 18th: IVF256 + PQ96 under 6 GB scored 0.891 LB alone; 10th used IVFPQ | 12th: compression "resulted in low accuracy"; 17th kept 150 GB flat |
| Generate your own questions? | 14th trained only on 750k own questions; 6th, 12th, 18th used GPT data | 13th: own ChatGPT questions decreased the score |
| Fine-tune the embedder? | 15th: SimCSE +0.015, hard negatives +0.005; 6th, 14th too | 5th: no gain on local validation, never submitted |
| Neural reranker? | 3rd: tuned cross-encoder +0.017 | 12th: TF-IDF beat `bge-reranker`; 18th: LambdaRank GBDT lost to raw embedding score |

The reranker split looks like a training-data question, which Podpall's writeup makes explicit.

## The winning solution: Team H2O LLM Studio

Psi (Philipp Singer), Yauhen Babakhin and Pascal Pfeiffer were first on both leaderboards. Plain classification with no context hit a ceiling early; their first retrieval submission with `all-MiniLM-L6-v2` beat everything else; from then on they mostly improved retrieval and kept the model roughly fixed. They had a 0.930 private submission about 30 days before the end. Psi noted a ceiling below 1.0: GPT-3.5 mislabelled some of its own answers.

![Team H2O's pipeline: a re-chunked cirrussearch Wikipedia of 30 to 60 million chunks, five embedding models, search as a GPU matrix multiply, a different context per reader, and five 7B plus one 13B binary readers ranking the options](/imgs/blogs/kaggle-llm-science-exam-solutions-6.webp)

Retrieval takes most of the width in the figure on purpose: that is where the effort went.

### Retrieval: a search engine that is just a matmul

**Corpus.** After finding that public parsers drop Lua-rendered numbers, they switched to the **cirrussearch** dump (fully rendered text, no newlines) and re-chunked it into 256, 512 and 1,024 character pieces without splitting sentences; cirrussearch at 512 was the best single corpus. Filtering to science articles never helped significantly, so they kept full Wikipedia.

**Embedders.** About 300 local combinations of retriever, LLM and dump later, the final contexts came from `e5-base-v2`, `e5-large-v2`, `gte-base`, `gte-large` and `bge-large`, with chunks encoded as `title + chunk`. The question was encoded once as `"{prompt} {A}, {B}, {C}, {D}, {E}"` and once per option as `"{prompt} {option}"`, merging the searches by similarity.

**Search.** No FAISS. The chunk embeddings were split into parts, each loaded in turn and multiplied against all test-question embeddings on the GPU, keeping a running global top-k; two GPUs doubled the speed. With about 4,000 queries, exact brute force beats an approximate index. A toy version (our illustration, not H2O's code):

```python
# Illustrative toy: exact top-k search over sharded embeddings by matmul.
# Not Team H2O's code; shapes and file layout are invented for the example.
import numpy as np
import torch

def sharded_topk(query_emb: np.ndarray, shard_paths: list[str], k: int = 5,
                 device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.from_numpy(query_emb).to(device, dtype=torch.float16)   # (Q, d), L2-normalised
    best_scores = torch.full((q.shape[0], k), -1e4, device=device, dtype=torch.float16)
    best_ids = torch.full((q.shape[0], k), -1, device=device, dtype=torch.long)
    offset = 0
    for path in shard_paths:                                           # one shard in memory at a time
        shard = torch.from_numpy(np.load(path)).to(device, dtype=torch.float16)  # (N_s, d)
        scores = q @ shard.T                                           # (Q, N_s) cosine similarities
        s_top, i_top = scores.topk(k, dim=1)
        merged_s = torch.cat([best_scores, s_top], dim=1)
        merged_i = torch.cat([best_ids, i_top + offset], dim=1)
        keep = merged_s.topk(k, dim=1).indices                         # running global top-k
        best_scores = merged_s.gather(1, keep)
        best_ids = merged_i.gather(1, keep)
        offset += shard.shape[0]
        del shard, scores
        torch.cuda.empty_cache()
    return best_scores, best_ids
```

The same shard-load-search-free loop appears independently at 12th place ("Faiss batch") and in CPMP's 54th-place entry. For the general pipeline, see [how to build an effective RAG system](/blog/machine-learning/large-language-model/how-to-build-effective-rag-system).

**Context order.** Train with 3 chunks, infer with 5 at a 1k max length. Reversing the context order at inference "hurt a lot": a reader trained on ranked context absorbs your retriever's ranking as a feature.

### The reader: an LLM as a binary classifier

Training data was public synthetic question sets (Radek Osmulski's early set "does the job quite well"); quantity mattered little. Training contexts came from their own noisy RAG, because training on the true source context was worse.

The model was **binary**: one sample per (question, option), output P(correct), rank options by probability. That makes it order-independent and removes the positional bias of a multiple-choice head. LoRA went on all linear layers, a binary head sat on the final next-token logits over the full vocabulary with a lower learning rate than LoRA, trained one epoch with cosine decay and BCE; the LoRA rank and learning rates were not published. Backbones that worked: `Llama-2-7b`, `Mistral-7B-v0.1`, `xgen-7b-8k-base`, `Llama-2-13b`. Bigger models were "not useful", and DeBERTa did not help the ensemble. For LoRA failure modes, see [debugging LoRA and PEFT](/blog/machine-learning/debugging-training/debugging-lora-and-peft).

### The KV-cache trick

A binary formulation costs five passes per question, each re-reading a long context. H2O avoided that.

![The KV-cache trick: the context and question are encoded once and their keys and values cached; the five options are run as a batch of five short sequences that reuse that cache, each producing a probability](/imgs/blogs/kaggle-llm-science-exam-solutions-7.webp)

They ran (context + question) once, kept `past_key_values`, and ran the five answers as a batch reusing that cache, taking each answer's last-token logits through the head. With an illustrative 1,000-token context and 20-token options, the naive approach processes ${5 \times 1{,}020 = 5{,}100}$ tokens per question and the cached one ${1{,}000 + 5 \times 20 = 1{,}100}$, about 4.6 times fewer. It is [prefix caching](/blog/machine-learning/model-serving/prefix-caching-and-radixattention) applied inside one question; the [KV cache post](/blog/machine-learning/large-language-model/kv-cache) covers the mechanics. A toy version, again ours:

```python
# Illustrative toy: score 5 options against one cached prefix.
# Not Team H2O's code; the head and prompt format are simplified stand-ins.
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
lm = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1",
                                          torch_dtype=torch.float16, device_map="auto")
head = torch.nn.Linear(lm.config.vocab_size, 1).half().to(lm.device)  # trained jointly with LoRA

@torch.no_grad()
def score_options(context: str, question: str, options: list[str]) -> torch.Tensor:
    prefix = tok(f"{context}\nQuestion: {question}\nAnswer:", return_tensors="pt").to(lm.device)
    out = lm(**prefix, use_cache=True)
    pkv = out.past_key_values                                    # encoded once
    opts = tok([" " + o for o in options], return_tensors="pt", padding=True,
               add_special_tokens=False).to(lm.device)
    pkv5 = tuple((k.expand(len(options), -1, -1, -1), v.expand(len(options), -1, -1, -1))
                 for k, v in pkv)                                # share the cache across 5 rows
    mask = torch.cat([prefix.attention_mask.expand(len(options), -1), opts.attention_mask], dim=1)
    logits = lm(input_ids=opts.input_ids, attention_mask=mask,
                past_key_values=pkv5).logits                     # (5, T_opt, V)
    last = opts.attention_mask.sum(1) - 1
    final = logits[torch.arange(len(options)), last]             # last real token per option
    others = (final.sum(0, keepdim=True) - final) / (len(options) - 1)  # mean of the other four
    return torch.sigmoid(head(final) + head(others)).squeeze(-1)  # cross-option signal, simplified
```

The last lines gesture at H2O's **cross-option head**: to give a binary model a sense of the other options without positional bias, the head also took the mean next-token logits of the other four. It boosted CV and LB and blended well with the plain head. Multiclass models and option-order TTA were tried; TTA helped but was "shaky" and never beat binary.

### Fitting it into nine hours

The final submission ran five 7B models and one 13B, each with a different (Wikipedia version, embedder, top-k) context, so the ensemble is really a late fusion of retrieval variants. Their best private submission was also their best CV submission, so the 6,000-question validation set did its job. The code was merged into H2O LLM Studio (PR #449), and the single-model notebook scores 0.932 private, level with second place on its own: the strongest evidence that retrieval, not the ensemble, carried the result.

## Second and third place

### 2nd: lytic (ivan, solo), the encoder-first stack

lytic's solution was "mostly based on Deberta type models and only half of the test questions are handled by Mistral LLM." Six stages: (1) Lucene BM25 via pyserini over overlapping roughly 1,000-character chunks of a 2023 Wikipedia dump (a 23 GB index, 200 × 5 queries in about five minutes); (2) a `deberta-v3-base` reranker that picks the best *context* for a (question, answer) pair, labelled by a `deberta-v3-large` teacher; (3) DeBERTa and Mistral multiple-choice readers; (4) a masked LM for options that differ by a few words; (5) a "None of the above" stress test: swapping correct answers for "None of the above" dropped validation MAP@3 from 0.9 to 0.3, so he trained a second-stage ranker on such examples; (6) XGBRanker fusion over retrieval scores, reader logits and that flag.

His ablation is the clearest measurement of what an LLM reader is worth. With eight contexts, the reranker, XGBRanker and the masked LM he was at 0.9155 private; adding the Mistral reader took it to **0.9315**, a jump of 0.016 and the largest step in his table. Training data was nine public sets totalling about 100,000 rows.

### 3rd: Podpall (Pavel Podberezko, solo), where the reranker is the model

Podpall patched `wikiextractor` to stop it deleting numbers, then searched at the **passage** level in one stage, because article titles and openings can miss the entity a question is about. His retrievers were two tiny embedders, `bge-small` and `all-MiniLM-L6-v2`. The key component was a cross-encoder reranker trained on pairs drawn from the same distribution as inference (his own passages) with hard negatives, which only worked from a reranking-pretrained checkpoint (`ibm/re2g-reranker-nq`) or in two stages. Readers: DeBERTa, three Electra and five RoBERTa models, plus 70B LLMs (Platypus2 and Xwin, via simjeg's code) on about 500 hard questions.

| Variant of the final (private MAP@3) | Private | Change |
|---|---|---|
| Final | 0.9284 | baseline |
| without Xwin | 0.9272 | −0.0012 |
| without Platypus2 | 0.9288 | +0.0004 |
| without both LLMs | 0.9165 | −0.0119 |
| without Electra + RoBERTa | 0.9246 | −0.0038 |
| without the tuned reranker | 0.9113 | **−0.0171** |

The reranker is worth more than both 70B models together, and either LLM covers most of the other's contribution, as you expect from two readers of the same text. His conclusion: the weak link is first-stage retrieval with two untuned tiny embedders. For how such rerankers are trained, see [reranker models: training and fine-tuning](/blog/machine-learning/large-language-model/reranker-models-training-finetuning-case-studies).

## Ranks 4 to 20, by approach family

### Big readers behind a confidence cascade

**4th, Preferred Scantron** is the exception: no LLM reader, but an Elasticsearch server running *inside* the kernel over sentence-split cirrussearch Wikipedia, queried with every sentence of the prompt and options. Three context orderings were ensembled, and lengthening DeBERTa's input from 512 to 768 to 1,280 tokens kept helping.

**5th, Preferred ChattyKids** ran BM25 over 74 million paragraphs (the ORQA argument: BM25 suffices when the questioner knows the answer) plus two dense retrievers. Their 70B ran layer by layer from per-layer quantised weights in about 6 GB, with five option-order rotations packed into one sequence under a custom attention mask so the context was encoded once. The cascade: Mistral-7B ensemble on everything, Llama-2-70B on the least-confident 40%, then Llama-2-70B with longer context on the least-confident 5%. Their rule, "we did not submit methods which do not work well in our local dataset", took them from 14th public to 5th private.

**6th, RAG to riches** crawled Wikipedia categories into a roughly 500k-page STEM corpus and generated questions with GPT-3.5, GPT-4, Llama-2-70B and Falcon-180B to fine-tune both retrievers and readers. A cross-encoder kept 2 to 4 of the top 10 passages by a dynamic threshold; readers were DeBERTa variants plus Mistral-7B-OpenOrca LoRA (r=16, alpha=16, lr 3e-4), and the top 5% of questions by prediction entropy went to two zero-shot 70B models.

**7th, days** had the most original reader idea in the top 20: **causal-LM reward modelling**. They modified trl's RewardTrainer so a causal LM scores `<context> #### <prompt> #### <option> ####` by the next "yes" logit, trained on (correct, wrong) option pairs. Mistral-7B-v0.1 went from 0.874 private with plain SFT to 0.897, a gain of 0.023. A pairwise ranking loss over options is exactly what MAP@3 rewards, the same idea as [reward modelling from human preferences](/blog/machine-learning/reinforcement-learning/reward-modeling-from-human-preferences). Their cascade dropped questions where DeBERTa was over 0.7 confident, freeing time for three LLMs.

**19th, steubk** asked a zero-shot orca-mini-v3-70B yes or no for each option and merged it with a DeBERTa: the merge (0.901 private) beat either alone, and a TF-IDF rerank of retrieved paragraphs lifted it to 0.917.

### Engineered retrieval, one reader, max ensemble

**10th, kami** used four Wikipedia sources and **one** DeBERTa-large, inferred at twice the training context, with a max-probability ensemble across contexts: if any context holds the answer the model is confident, otherwise flat, so the max is "robust, less likely to overfit". **12th, kaggle is few shot learner** searched 56 million vectors (100 GB+) in shards and found TF-IDF reranking beat `bge-reranker`. **13th, kurupical** built a 250-question retrieval benchmark from articles starting with "a" to iterate fast, trained at 256 tokens and inferred at 768, and used retrieval-rank TTA: four views, each the top chunk plus a different rank band, combined by mean plus max. **17th, Sadge** rebuilt Wikipedia's category graph from SQL dumps, took 1.1 million STEM articles four levels from the roots, and rendered them through a **local MediaWiki server** so LaTeX and Lua templates came out exact. **18th, Chan Kha Vu** indexed 128-token passages over all of Wikipedia (about 54 million vectors) and benchmarked index types until IVF256 with 96-byte PQ fit under 6 GB at 20 seconds for 4,000 questions; with one public model that scored 0.891 LB before the popular 270k STEM set existed. **20th, SkyFish Up Up** ran four contexts and one DeBERTa, noting a max plus mean ensemble would have reached 0.917.

### Corpus builders

**11th, ScienceGPT** identified 154 source articles and re-tuned a public clustering notebook until its 480k-article STEM set covered all of them, which the 270k set did not; they added a "one-shot" DeBERTa that reads all five options at once and a Llama-2-7B LoRA worth +0.003. **14th, Maxim Tsygankov** did the best detective work of the competition: he traced `train.csv` questions to source pages by hand, saw they came from the same science categories, built a corpus of every page in them, had GPT-3.5 write five questions per 1,500-character segment (about 750,000 triplets), fine-tuned e5 and bge on it, and rose from 47th public to 14th. **15th, Life is full of surprise** trained every embedder with SimCSE, then again with hard negatives mined by the first model. They fell from 3rd public to 15th, blame overfitting in their two DeBERTas, and called it "a Discussion Competition": after the public 270k set, every trick was public.

### No writeup

**8th TT** and **16th CSP** published nothing we could find; CSP is the largest climber in the top 20, from 100th public to 16th. **9th SUSTech BDI** includes simjeg, author of the public 70B notebooks, but whether their final submission used that pipeline is unverified.

### The contrasting entry: CPMP's zero-training 70B at 54th

CPMP published "0.894 without training any model". He searched 35 million paragraphs (about 140 GB of embeddings, streamed a million at a time through `faiss.knn_gpu` with no index, about 25 minutes), querying with the mean of the prompt embedding and the mean option embedding so the search was order-free. A zero-shot Xwin-LM-70B "yes" logit gave 0.892 private; a logistic-regression head on the last hidden state lifted it to 0.908, 54th, using 8 h 58 min of the 9 h limit. That is 0.025 below the winner with no fine-tuning at all: retrieval plus a big frozen reader gets most of the way, and the top teams fought over the last 3%.

## What the forum taught everyone

| Thread | What it changed |
|---|---|
| Radek Osmulski, "6000 new high quality train examples" (159 votes) | Shared GPT-3.5 questions for teams without OpenAI access; H2O trained on this line of sets |
| Chris Deotte, "60k Dataset Achieves LB 0.830+" (282 votes) | Public sets plus Wikipedia context; the standard training and validation set |
| simjeg, "Zero-shot 70B model + RAG" (182 votes) | Platypus2-70B layer by layer; asking "is this answer correct?" five times beat "which is best?" (0.81 vs 0.75 LB without RAG); unlocked 70B for 3rd, 5th, 6th |
| Mohammadreza Banaei, "270K Wiki STEM articles" (137 votes) | A STEM cluster covering most leaderboard source articles |

![Public-to-private rank slope chart for the top 20: H2O stays first, lytic and Podpall rise sharply, while the public second and third place teams fall to 12th and 15th](/imgs/blogs/kaggle-llm-science-exam-solutions-8.webp)

The shake-up was mild at the top and noisy below. With about 800 public questions a 0.003 MAP gap is two or three questions, and 5th place called MAP@3 "a bit unstable". The risers describe choosing submissions on local validation; the fallers describe overfitting the public 20%.

## What did not work

| Tried | Reported by | Outcome |
|---|---|---|
| Training on true source context, not retrieved context | 1st, 11th, 18th | worse or no gain |
| IVF or PQ index compression | 12th, 17th (contested by 10th, 18th) | lost accuracy |
| Platypus2-70B with their retrieval | 12th | exceeded nine hours |
| PEFT on DeBERTa | 12th | no gain |
| Own ChatGPT-generated questions | 13th | score dropped |
| A CV that tracks the leaderboard from small sets | 13th, 10th | low correlation |
| Option-order or complex TTA | 1st ("shaky"), 13th | unreliable |
| Larger query encoders (`bge-large`) | 10th | "not much difference" |
| LambdaRank GBDT reranker | 18th | lost to raw embedding score |
| Pre-fine-tuning on RACE | 20th | no gain |
| Embedding fine-tuning; custom query construction | 5th | negative on local validation |
| Focal loss; removing Llama's causal mask | 2nd | no gain |

The first row is the strangest and most consistent: a reader trained on clean context never learns to ignore a wrong paragraph, and at test time the retriever hands it wrong paragraphs.

## Lessons that transfer

1. **When the questions were written from a corpus, retrieval is the model.** All 17 top-20 writeups were RAG; H2O iterated on retrieval with a fixed reader; Podpall's reranker outweighed two 70B models.
2. **Parse the corpus yourself.** Seven top-20 teams re-parsed Wikipedia because public dumps dropped the numbers.
3. **Turn the LLM into a scorer and share the prefix.** Binary heads with KV-cache reuse (1st), packed option rotations (5th) and pairwise reward models (7th) are how 7B to 70B readers fit nine hours.
4. **Cascade by confidence.** All five cascade teams finished in the top seven.
5. **Train on the context you will see at test time.** Noisy retrieved context beat clean source context for three teams.
6. **Validate on something harder than what you were given**, and choose submissions by it, not by the public board.

### If you enter a competition like this

- [ ] Trace a few labelled items to their source by hand before touching a model.
- [ ] Re-parse the corpus and check that numbers and formulas survived.
- [ ] Freeze a held-out set of a few thousand items, harder than the given sample.
- [ ] Measure retrieval recall on items with a known source, then confirm it moves the end metric.
- [ ] Use exact search when queries are few; compress only after measuring recall loss.
- [ ] Train the reader on noisy retrieved context, in retriever order.
- [ ] Score options independently or pairwise, and share the context's KV cache.
- [ ] Put the big model behind a confidence threshold.

Same lens, other levers: [Detect AI Generated Text](/blog/machine-learning/learn-from-kaggle/kaggle-detect-ai-generated-text-solutions), [Prompt Recovery](/blog/machine-learning/learn-from-kaggle/kaggle-llm-prompt-recovery-solutions), [LMSYS Chatbot Arena](/blog/machine-learning/learn-from-kaggle/kaggle-lmsys-chatbot-arena-solutions) and [WSDM Cup Multilingual Arena](/blog/machine-learning/learn-from-kaggle/kaggle-wsdm-cup-multilingual-arena-solutions).

## Sources

Competition: https://www.kaggle.com/competitions/kaggle-llm-science-exam (ranks and scores from Kaggle's leaderboard API). Top-20 writeups:

- 1st Team H2O LLM Studio: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446422 and https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446240 (code: h2oai/h2o-llmstudio PR #449)
- 2nd lytic: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/448256
- 3rd Podpall: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446358
- 4th Preferred Scantron: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446307
- 5th Preferred ChattyKids: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446293
- 6th RAG to riches: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/447647 and https://github.com/rbiswasfc/llm-science-exam
- 7th days: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/447155
- 10th kami: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446248
- 11th ScienceGPT: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446660
- 12th kaggle is few shot learner: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446301
- 13th kurupical: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446231
- 14th Maxim Tsygankov: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446484
- 15th Life is full of surprise: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446816
- 17th Sadge: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446242
- 18th Chan Kha Vu: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446261
- 19th steubk: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446395
- 20th SkyFish Up Up: https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/448114
- 8th TT, 9th SUSTech BDI, 16th CSP: no writeup found

Other threads: CPMP https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446232; disqualification https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446797; Radek Osmulski https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/426174; Chris Deotte https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/436383; simjeg https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/440620; Mohammadreza Banaei https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/442595.
