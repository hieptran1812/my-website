---
title: "RAG under a notebook budget: what Kaggle's LLM competitions taught about retrieval with the internet switched off"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "A cross-competition synthesis of how top Kaggle teams built retrieval inside a 9-hour offline notebook: corpus building, BM25 versus dense, rerankers, chunking, how many passages to feed the reader, and where the latency budget went, drawn from LLM Science Exam, Eedi, MAP, Make Data Count and others."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "rag", "retrieval", "bm25", "dense-retrieval", "reranking", "faiss", "chunking"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 13
---

> **TL;DR.** Across Kaggle's LLM competitions, retrieval decided more medals than the reader model did, and the right retrieval stack was set by one number: how many candidates sit behind each query. In LLM Science Exam (2023), with tens of millions of Wikipedia passages per question, all 17 top-20 writeups ran RAG, 15 used a dense retriever and 11 a sparse one, and the 3rd-place team's tuned reranker was worth 0.017 MAP@3, more than the spread of the whole top 20. In Eedi (2024), with about 2,587 labels, the retriever became a fine-tuned Qwen2.5-14B and the money moved to LLM rerankers. In MAP (2025), with 8 to 12 candidates, the winners did not retrieve at all, and the one team that tried retrieved few-shot examples lost leaderboard score. In Make Data Count (2025), the best "retriever" was an exact lookup in the corpora the labels were built from. The one lesson: size the haystack before you choose the needle-finder.

![Candidates per query on a log scale: tens of millions of Wikipedia passages in LLM Science Exam, about 2,587 misconceptions in Eedi, 8 to 12 candidate labels per question in MAP, and an exact ID lookup in Make Data Count, each annotated with the retrieval stack the top teams used](/imgs/blogs/kaggle-llm-rag-under-a-notebook-budget-1.webp)

The chart above is the mental model for this post. A Kaggle code competition gives you a notebook, a 9-hour wall clock, two T4s and no internet. Everything you search must be on disk before the clock starts, and every second spent searching is a second the reader model does not get. Production RAG advice (see [how to build an effective RAG system](/blog/machine-learning/large-language-model/how-to-build-effective-rag-system)) assumes a live vector database and a per-request latency budget. Kaggle gives you one batch and one hard total.

This is a technique post in the [Learn from Kaggle LLM series](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work). Every claim cites the competition and team; counts come from each competition's published top-20 synthesis.

## The comparison across competitions

| Competition | Haystack per query | Retriever the top 20 converged on | Reranker | What reached the reader | What retrieval bought |
|---|---|---|---|---|---|
| [LLM Science Exam](/blog/machine-learning/learn-from-kaggle/kaggle-llm-science-exam-solutions) (2023) | 30M to 74M Wikipedia passages | Dense in 15 of 17 writeups, sparse in 11, usually both | Cross-encoder in 2nd, 3rd, 6th | 1st: 5 chunks at inference; 6th: 2 to 4 of the top 10 | 3rd: reranker worth 0.017 private MAP@3 |
| [Eedi](/blog/machine-learning/learn-from-kaggle/kaggle-eedi-mining-misconceptions-solutions) (2024) | About 2,587 misconception texts | Fine-tuned decoder LLM embedder, 14 of 15 | LLM reranker, Qwen2.5-32B in 13 of 15 | 1st: 32 to 64, then 8, then 5 | 1st: retriever alone 0.475 private, full pipeline 0.639 |
| [MAP](/blog/machine-learning/learn-from-kaggle/kaggle-map-student-math-misunderstandings-solutions) (2025) | 8 to 12 candidate labels per question | None: score the question's own labels (8 of 9 known) | 4th's whole solution; failed for 9th and 10th | Every candidate | 4th: retrieved examples lowered public and private |
| [Make Data Count](/blog/machine-learning/learn-from-kaggle/kaggle-make-data-count-solutions) (2025) | IDs printed in one paper | Exact lookup in the label-source corpora, 14 of 15 | None; a type classifier | Every ID found in the text | 2nd: lookup and rules alone scored 0.739 private, gold |
| [Jigsaw Community Rules](/blog/machine-learning/learn-from-kaggle/kaggle-jigsaw-agile-community-rules-solutions) (2025) | The labelled examples per rule | Embedders with kNN (5th), FAISS (9th) | None | Retrieved neighbours as votes | 9th: retrieval part 0.909 public, blended at weight 0.05 |
| [Konwinski Prize](/blog/machine-learning/learn-from-kaggle/kaggle-konwinski-prize-solutions) (2025) | Files in one repository | Parsed repo structure (1st), regex traceback locator (5th) | None | 5 candidate files (2nd) | 2nd: embeddings for localisation did not help |

The matrix below codes the same competitions against six retrieval choices, using the counts from each competition's top-20 synthesis.

![Matrix of seven competitions against six retrieval choices: own corpus or label source, sparse or exact match, dense embedder, trained retriever, reranker, and fusing several retrievals, with filled, half and crossed cells and per-cell counts such as 15 of 17 dense in Science Exam and 13 of 15 with a 32B reranker in Eedi](/imgs/blogs/kaggle-llm-rag-under-a-notebook-budget-2.webp)

Read top to bottom, it is a staircase: as the haystack shrinks, stages fall away until in MAP "retrieval" is a dictionary from question to allowed labels.

## 1. Build the corpus before you choose the retriever

In LLM Science Exam the questions were written by GPT-3.5 from Wikipedia, so the gold passage existed somewhere on disk. The first trap was that the standard dumps had silently lost it. Team H2O LLM Studio (1st) found that public parsers drop Lua-rendered values, which are exactly the numbers science questions ask about, and switched to the cirrussearch dump of fully rendered pages, re-chunked into 256, 512 and 1,024-character pieces without splitting sentences. Podpall (3rd) patched `wikiextractor` to stop it deleting numeric values. Preferred ChattyKids (5th) kept `{val}` and `{math}` templates with `wikitextparser`. Counting the matrix cells, 7 of the 17 writeups say they re-parsed Wikipedia themselves (1st, 3rd, 4th, 5th, 10th, 13th, 17th); the addendum's own summary line says 6, and the recount wins.

Narrowing the corpus was a weaker lever than cleaning it: H2O's filter to science articles "never helped significantly", and they kept full Wikipedia.

In Eedi the corpus was the label set itself, and building it meant writing better labels. CQYR (2nd) had Llama-3.1-70B and Qwen2.5-72B write a detailed explanation for every misconception, which lifted the retriever by 2 to 4 percent and cost nothing at submission time, because label embeddings are computed once, offline. Three of 15 Eedi writeups enriched or rewrote misconception text this way.

Make Data Count is the extreme case. Its labels were produced by Europe PMC's text-mining and DataCite's DOI mappings, so Ali and Kea (1st) used those same sources as the candidate generator. On train, their DOI stage kept 321 candidates with 302 true positives, 19 false positives and 23 misses, a mention-level F1 of 0.935. Fourteen of 15 writeups did the same lookup for DOIs; the one that relied on regex alone finished 20th.

## 2. BM25 versus dense: the top teams ran both

The Science Exam field split cleanly on paper and blurred in practice: 11 of 17 writeups used a sparse retriever (BM25, TF-IDF or Elasticsearch), 15 used a dense one, and at least 16 fused several retrieval variants late. The extremes both medalled. lytic (2nd) ran only Lucene BM25 over full Wikipedia, sentence-split and grouped into overlapping chunks of about 1,000 characters. H2O (1st) ran only dense retrieval, with five embedders (`e5-base-v2`, `e5-large-v2`, `gte-base`, `gte-large`, `bge-large`) chosen from roughly 300 local combinations of retriever, reader and dump.

ChattyKids (5th) gave the cleanest argument for keeping BM25, borrowed from the ORQA paper: when the questioner already knows the answer, as GPT-3.5 did when it wrote each question from a passage, BM25 suffices. Their single Llama-2-70B reading BM25 top-2 passages with option-order TTA scored 0.925 private on its own.

H2O's dense search also shows that you may not need a vector index at all. They loaded 30 to 60 million chunk embeddings in parts, multiplied each part against all test questions on the GPU, kept a running global top-k, and split the work across both GPUs for a 2x speedup. The toy below reproduces the idea on random vectors; it is an illustration, not their code.

```python
# Illustrative toy, not any team's code: exact top-k search by streaming
# corpus shards through a matrix multiply, the idea 1st place described
# ("no FAISS": load each part, score all questions, keep a running top-k).
import numpy as np

rng = np.random.default_rng(0)
n_chunks, n_questions, dim, k = 200_000, 64, 384, 5
corpus = rng.standard_normal((n_chunks, dim)).astype(np.float32)
corpus /= np.linalg.norm(corpus, axis=1, keepdims=True)
queries = rng.standard_normal((n_questions, dim)).astype(np.float32)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

def streaming_topk(queries, corpus, k, shard=25_000):
    best_s = np.full((len(queries), k), -np.inf, dtype=np.float32)
    best_i = np.zeros((len(queries), k), dtype=np.int64)
    for start in range(0, len(corpus), shard):      # one shard "on disk" at a time
        part = corpus[start:start + shard]
        sims = queries @ part.T                     # (questions, shard) on one device
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        s = np.take_along_axis(sims, idx, axis=1)
        all_s = np.concatenate([best_s, s], axis=1) # merge with the running top-k
        all_i = np.concatenate([best_i, idx + start], axis=1)
        keep = np.argsort(-all_s, axis=1)[:, :k]
        best_s = np.take_along_axis(all_s, keep, axis=1)
        best_i = np.take_along_axis(all_i, keep, axis=1)
    return best_i

full = np.argsort(-(queries @ corpus.T), axis=1)[:, :k]
streamed = streaming_topk(queries, corpus, k)
print("shards:", -(-n_chunks // 25_000), "| peak shard matrix:", (n_questions, 25_000))
print("identical top-5 for all questions:", bool((full == streamed).all()))
```

```console
shards: 8 | peak shard matrix: (64, 25000)
identical top-5 for all questions: True
```

The search is exact, memory is bounded by one shard, and each shard is read once for all questions. The per-query latency trade-off behind [approximate nearest-neighbour serving](/blog/machine-learning/recommendation-systems/approximate-nearest-neighbor-serving-faiss-hnsw-scann) does not apply when all 4,000 queries arrive as one batch.

![Index footprint on disk for Science Exam teams on a log scale: 1st place's 2.5 TB of total notebook input, 17th's 150 GB flat FAISS index, CPMP's 140 GB of raw embeddings, 5th's FAISS index compressed from 300 GB to 10 GB, 2nd's 23 GB Lucene BM25 index, and 18th's 6 GB IVF-PQ index](/imgs/blogs/kaggle-llm-rag-under-a-notebook-budget-3.webp)

Index compression was a genuine disagreement. Chan Kha Vu (18th) indexed 54.5 million vectors as IVF256-PQ96 in 6 GB and scored 0.891 on the leaderboard with that index alone; ChattyKids quantized their `instructor-xl` FAISS index from 300 GB to 10 GB. But "kaggle is few shot learner" (12th) reported that compressing the index ("IVQ, PQ" in their words) "resulted in low accuracy", and Sadge (17th) kept a 150 GB flat index after compression hurt retrieval. Test it on your own validation set; neither side is wrong in general.

## 3. Rerankers: where the points actually were

Podpall (3rd in Science Exam) published the clearest ablation in the series, and it makes one argument.

![Private MAP@3 lost when each component of 3rd place's final pipeline is removed: the tuned reranker 0.0171, both 70B LLMs 0.0119, Electra and RoBERTa readers 0.0038, MiniLM retriever 0.0026, bge-small retriever 0.0012, Xwin 0.0012, and Platypus2 minus 0.0004, meaning removing it helped](/imgs/blogs/kaggle-llm-rag-under-a-notebook-budget-4.webp)

Removing the tuned reranker took the final 0.9284 down to 0.9113, a loss of 0.0171. Removing both 70B models cost 0.0119. The private top 20 spanned 0.933 to 0.917, so the reranker alone was worth more than that whole band. Podpall's own reading was that the reader was strong once given the right passage, and the weak link was first-stage retrieval from two tiny, untuned embedders.

How he trained it mattered: pairs drawn from his own Wikipedia passages, the inference distribution, and hard negatives that only worked from a reranking-pretrained checkpoint (`ibm/re2g-reranker-nq`) or in two stages. lytic (2nd) built a `deberta-v3-base` reranker that picks the best passage for a (question, answer) pair, with labels from a `deberta-v3-large` teacher; going from one passage to eight plus the reranker moved his private score from 0.9098 to 0.9161. RAG to riches (6th) had a `deberta-v3-base` cross-encoder rerank the top 10 and keep 2 to 4 by a dynamic threshold.

The failures agree. The 12th-place team found TF-IDF reranking beat an off-the-shelf `bge-reranker`, and Chan Kha Vu's LambdaRank GBDT reranker lost to the raw embedding score. A reranker trained on someone else's distribution is a second untuned retriever.

In Eedi the reranker became the main model. Thirteen of 15 writeups had a Qwen2.5-32B in the final reranking stage, and three ran a 70B-class model. MTH 101 (1st) showed the size of the gap: the best retriever alone scored 0.475 private, and the full pipeline 0.63881. The pattern is the one described in [the recommendation funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking): a cheap stage for recall, an expensive stage for order.

## 4. Chunking, and how many passages the reader sees

Chunk size was tuned, not guessed: H2O's best corpus was cirrussearch at 512 characters, lytic used overlapping 1,000-character chunks, kurupical (13th) sliding 90-word chunks. Podpall searched passages directly rather than article then sentence, because an article's opening can miss the entity a question is about.

![What the reader sees: 1st place feeds 5 chunks of 512 characters best first, trained on 3, and reversing the order at inference hurt; 4th place puts prompt and choices first and packs context into 512, 768 and then 1,280 tokens, each longer window improving; 5th place gives Llama-2-70B a 2 to 3 times longer context only on the least-confident 5 percent](/imgs/blogs/kaggle-llm-rag-under-a-notebook-budget-6.webp)

Three findings about the reader's context transfer directly:

- **Train on fewer passages than you infer with.** H2O trained with 3 chunks and inferred with 5, at a 1k max length. Reversing the passage order at inference "hurt a lot", which tells you the model learned that the best chunk comes first. Order is part of the contract between retriever and reader.
- **Length pays, if the reader can use it.** Preferred Scantron (4th), the only DeBERTa-only team in the top seven, put prompt and choices first and packed in as much context as fit; going from 512 to 768 to 1,280 tokens kept improving.
- **Train on noisy retrieved context, not the gold passage.** H2O found training on the true source context was worse than training on its own RAG output. ScienceGPT (11th) saw no gain from better training context, and Chan Kha Vu (18th) reported +0.005 from training on worse context. Nobody in the top 20 dissented: the reader should see in training what the retriever will hand it at test time.

When the budget is tight, spend long context selectively. ChattyKids ran Mistral-7B on every question, Llama-2-70B on the least-confident 40 percent, and the 70B again with 2 to 3 times longer `bge` context on the least-confident 5 percent.

## 5. When the haystack is small, retrieval changes shape

![Two Eedi rerank funnels drawn to log scale: 1st place narrows about 2,587 misconceptions to 32 to 64 with a Qwen2.5-14B retriever, then to 8 with a 14B pointwise ranker, 5 with a 32B, ordered by a 72B listwise model; 5th place sends the top 104 from a 1.5B bi-encoder to one Qwen2.5-32B in two windows of 52 options](/imgs/blogs/kaggle-llm-rag-under-a-notebook-budget-5.webp)

With about 2,587 labels, Eedi's retriever no longer needed BM25 or a giant index; it needed recall on labels it had never seen. MTH 101 (1st) optimised the retriever for recall@32 rather than MAP, put one example per misconception in each batch so in-batch negatives were never false, and kept the top 32 plus up to 32 more above a dynamic score threshold. Each later stage was a bigger model on a shorter list: the 14B pointwise ranker kept 8, the 32B kept 5, and the 72B ordered those 5. Ebi and ktr (5th) went the other way, handing a deep top 104 to one Qwen2.5-32B as two listwise windows of 52 single-letter options.

MAP, a year later, was built on the same Eedi questions, but each question allowed only a handful of labels. tascj (1st) scored all 8, 10 or 12 candidate suffixes for a question in one packed sequence, and 8 of the 9 writeups that say so scored only the labels the question can take. THLUO (4th) ran the one real retrieval experiment: pulling the most similar labelled explanations with a fine-tuned Qwen3-Embedding-8B into the prompt matched his best CV (0.9493) but scored worse on public and private, while random examples per candidate label did help. When labels are noisy and candidates few, retrieved neighbours can copy the noise.

In Jigsaw Community Rules, retrieval was a model rather than a context source: Flash (5th) ran six embedders with kNN, and lllleeeo + mango (9th) had a BGE triplet model with FAISS that scored 0.909 public alone but took 0.05 of the final blend. It was a diversity member, not the engine.

## 6. The latency budget

Nine hours is a total, not a per-query limit, so the question is what fraction retrieval eats.

| Team (competition) | Retrieval cost reported | What it freed |
|---|---|---|
| CPMP, 54th (Science Exam) | About 25 min for 35M paragraphs, 140 GB streamed 1M per file with `faiss.knn_gpu`, no index | A zero-training Xwin-LM-70B reader; total run 8 h 58 min, private 0.908 |
| lytic, 2nd (Science Exam) | 23 GB Lucene index; 200 questions times 5 queries in about 5 min | Half the questions handled by Mistral |
| H2O, 1st (Science Exam) | Matmul search split over 2 GPUs for 2x | Five 7B readers and one 13B; 2.5 TB of input, fits "precisely" in 9 h |
| CQYR, 2nd (Eedi) | Label embeddings precomputed; vLLM prefix caching about 10 percent faster | Picked a smaller retriever ensemble to leave time for the 72B; a stronger variant (0.609) timed out |

Two habits recur: precompute everything corpus-side so the notebook only embeds queries, and keep search cheap so the clock goes to the reader. CPMP spent under 5 percent of his run on search. Fitting that 70B is its own topic, covered in the upcoming post *Fitting big models in small GPUs*.

## What did not work

| Tried | Competition and team | Note |
|---|---|---|
| Filtering Wikipedia to science articles | Science Exam, 1st | Never helped significantly |
| Embedder fine-tuning | Science Exam, 5th | Dropped on local validation; 15th gained from SimCSE plus hard negatives |
| Retrieval recall as a proxy metric | Science Exam, 13th | Uncorrelated with the leaderboard |
| Semantic or keyword search for context | Make Data Count, 9th | Lookup plus rules won |
| Embeddings for file localisation | Konwinski Prize, 2nd | More file candidates did not help either |
| Dictionary and RAG pseudo-labelling | [Deep Past](/blog/machine-learning/learn-from-kaggle/kaggle-deep-past-akkadian-translation-solutions), 8th | With open models |

## If your competition has a haystack

1. **Find out where the questions or labels came from**, and index that source (Science Exam, Make Data Count).
2. **Parse the corpus yourself** and check that numbers, formulas and IDs survive.
3. **Run BM25 and a dense retriever** and fuse late, as 16 of 17 Science Exam writeups did; drop one only on validation evidence.
4. **Train a reranker on your own retrieval output**, with hard negatives from a reranking-pretrained start.
5. **Train the reader on noisy retrieved context**, a little shorter than at inference, best passage first.
6. **Precompute corpus-side work** and budget retrieval at minutes, not hours.
7. **If the candidate list fits in one prompt, score it all** and skip retrieval, as MAP's winners did.

For the losses behind dense retriever training, see [sampled softmax and contrastive losses for retrieval](/blog/machine-learning/recommendation-systems/sampled-softmax-and-contrastive-losses-for-retrieval).

## Sources

LLM Science Exam: [1st H2O](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446422) ([summary](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446240)), [2nd lytic](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/448256), [3rd Podpall](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446358), [4th Preferred Scantron](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446307), [5th Preferred ChattyKids](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446293), [6th RAG to riches](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/447647), [11th ScienceGPT](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446660), [12th kaggle is few shot learner](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446301), [13th kurupical](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446231), [15th Life is full of surprise](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446816), [17th Sadge](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446242), [18th Chan Kha Vu](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446261), [CPMP 54th](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446232), [Banaei 270k STEM](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/442595).

Eedi: [1st MTH 101](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution), [2nd CQYR](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/cqyr-2nd-place-solution), [5th Ebi and ktr](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/ebi-ktr-5th-place-solution), [8th Yannan Chen](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/yannan-chen-8th-place-solution), [14th eeee](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/eeee-rsi-14th-place-solution).

MAP: [1st tascj](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/1st-place-solution), [4th THLUO](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/4th-place-solution-single-model-pb-0-948-lb-0-951).

Make Data Count: [1st Ali and Kea](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606853), [2nd Mohsin and Nikhil](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606786), [9th flash attention](https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606743).

Jigsaw Community Rules: [5th Flash](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/5th-place-solution-diverse-ensemble), [9th lllleeeo + mango](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613185), [15th](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/15th-place-solution).

Konwinski Prize: [2nd Camaro](https://www.kaggle.com/competitions/konwinski-prize/writeups/camaro-public-2nd-place-solution), [5th Kenj](https://www.kaggle.com/competitions/konwinski-prize/writeups/5th-place-solution). Deep Past: [8th Hrithik Reddy](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684329).
