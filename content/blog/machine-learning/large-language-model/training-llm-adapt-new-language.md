---
title: "Training LLMs to Adapt to a New Language: The Complete Engineering Guide"
publishDate: "2026-03-18"
category: "machine-learning"
subcategory: "Large Language Model"
tags: ["llm", "multilingual", "fine-tuning", "continual-pretraining", "tokenizer", "nlp", "transfer-learning", "low-resource-languages"]
date: "2026-03-18"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "Most powerful LLMs are English-centric. This guide walks through the complete pipeline for adapting an existing LLM to a new language — from extending the tokenizer, through continual pretraining on target-language corpora, to instruction tuning. Includes real code, math, trade-offs, and hard-won lessons from production."
---

## The Problem: English Dominance in LLMs

Here's an uncomfortable truth about the current LLM landscape: the most capable open-weight models — LLaMA 3, Mistral, Qwen, Gemma — are overwhelmingly trained on English data. Even "multilingual" models typically allocate 80-90% of their training tokens to English and a handful of high-resource languages.

If you work with Vietnamese, Thai, Swahili, Tagalog, or any of the hundreds of languages that got a tiny slice of the pretraining pie, you've felt the pain: the model *kind of* understands your language, but it's slow (terrible tokenization efficiency), makes grammar mistakes a native speaker would never make, and hallucinates facts about your culture.

**The good news:** you don't need to train a model from scratch. With the right technique, you can take an English-dominant LLM and adapt it to your target language at a fraction of the cost — often with just a few billion tokens of target-language data and a few hundred GPU-hours.

This guide covers the complete pipeline:

1. **Tokenizer extension** — making the model speak your language efficiently
2. **Vocabulary embedding initialization** — giving new tokens a meaningful starting point
3. **Continual pretraining** — teaching the model your language's grammar and knowledge
4. **Instruction tuning** — aligning the adapted model for downstream tasks
5. **Evaluation** — measuring whether the adaptation actually worked

Let's dig in.

## Why Not Just Train From Scratch?

Before we go through the adaptation pipeline, let's understand why adaptation beats training from scratch for most teams.

| Approach | Data Required | Compute Cost | Quality |
|----------|--------------|-------------|---------|
| Train from scratch | 1-15T tokens | $2M-$100M+ | Best if you have enough data |
| Continual pretraining (adaptation) | 10B-100B tokens | $5K-$100K | 85-95% of from-scratch quality |
| Fine-tuning only (no pretraining) | 10K-1M examples | $100-$5K | Superficial; fails on generation |

The core insight: **a pretrained English LLM already knows how to reason, follow instructions, and model complex syntax.** These capabilities transfer across languages. What it's missing is (a) efficient tokenization for your language and (b) enough exposure to your language's patterns. Adaptation supplies both without throwing away the English capabilities.

A concrete example: adapting LLaMA 3 8B to Vietnamese. LLaMA 3's tokenizer encodes the Vietnamese sentence "Hôm nay trời đẹp quá" (Today the weather is so beautiful) into **12 tokens**. After tokenizer extension, the same sentence becomes **6 tokens**. That's a 2x efficiency gain, which translates directly into 2x longer effective context and 2x faster inference.

## Why Cross-Lingual Transfer Actually Works (The Theory)

Before we dive into the how, it's worth understanding *why* this works at all. Why can a model trained mostly on English learn Vietnamese so much faster than from scratch?

### Shared Representation Space

Research on multilingual models (mBERT, XLM-R, and more recently, probing studies on LLaMA) has shown something remarkable: **transformer models develop language-agnostic representations in their middle layers.** The early layers handle language-specific surface features (tokenization, morphology), the middle layers encode abstract semantic and syntactic structures that are surprisingly universal, and the final layers map back to language-specific outputs.

```
Layer Structure in a Multilingual Transformer:

Layers 1-8:    Language-specific surface processing
               (tokenization patterns, character-level features)

Layers 9-24:   Language-AGNOSTIC conceptual space
               (semantic relationships, reasoning chains,
                logical structures, world knowledge)

Layers 25-32:  Language-specific output generation
               (grammar, word order, morphological agreement)
```

This means when you adapt a model to Vietnamese, you're not teaching it concepts from scratch. The concept of "capital city" or "photosynthesis" or "if-then reasoning" already exists in the model's middle layers. You're teaching the input layers to *map Vietnamese text into that existing concept space*, and the output layers to *map from that space back to Vietnamese text*.

This is why **adaptation needs 100x less data than training from scratch** — you're only learning the mapping, not the concepts.

### The Linguistic Universals Advantage

Chomsky's idea of Universal Grammar is controversial in linguistics, but there's a practical version that matters for us: all human languages share deep structural commonalities. They all have:

- **Predicate-argument structure** (someone does something to something)
- **Recursive embedding** (I know that she said that he believed...)
- **Quantification** (all, some, none)
- **Temporal reference** (past, present, future)
- **Negation**

A model that has learned these patterns from English data already has the "skeleton" of language understanding. Vietnamese, Thai, or Swahili fill in different surface forms but use the same skeleton.

### What Doesn't Transfer Well

Understanding what *doesn't* transfer is equally important:

| Transfers Well | Transfers Poorly |
|---------------|-----------------|
| Logical reasoning | Word order conventions (SVO vs SOV) |
| Factual knowledge | Morphological patterns (agglutination, tones) |
| Instruction following | Culture-specific knowledge |
| Code understanding | Honorific systems and politeness levels |
| Math reasoning | Script-specific features (right-to-left, no spaces) |
| Common sense | Language-specific idioms and metaphors |

This table tells you where to focus your adaptation data. You need lots of examples showing your language's unique patterns — word order, morphology, culturally specific knowledge — because these won't transfer from English.

## Phase 1: Tokenizer Extension

This is the most impactful and most overlooked step. If you skip this, everything downstream suffers.

### Why the Existing Tokenizer Is a Problem

Most LLM tokenizers are trained on English-heavy corpora using Byte-Pair Encoding (BPE). The result: English words get compact, single-token representations, while text in other languages gets split into many small subword pieces or even individual bytes.

```
# LLaMA 3 tokenizer on English
"The weather is beautiful" → ["The", " weather", " is", " beautiful"]  # 4 tokens

# LLaMA 3 tokenizer on Vietnamese
"Hôm nay trời đẹp quá" → ["H", "ô", "m", " nay", " tr", "ờ", "i", " đ", "ẹ", "p", " qu", "á"]  # 12 tokens

# LLaMA 3 tokenizer on Thai
"วันนี้อากาศดี" → [bytes...]  # 15+ tokens for a simple sentence
```

This has three devastating effects:

1. **Wasted context window**: Your 8K or 128K context window holds 2-4x fewer words in the target language
2. **Slower inference**: More tokens = more autoregressive decoding steps = proportionally slower
3. **Worse quality**: The model has to "reconstruct" words from fragments, which wastes model capacity

### How to Extend the Tokenizer

The strategy: **train a new BPE tokenizer on your target-language corpus, then merge it with the original tokenizer.**

#### Step 1: Train a Target-Language Tokenizer

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# Prepare your target-language corpus
# Use a diverse mix: news, Wikipedia, books, web crawl, social media
corpus_files = [
    "data/vi_wikipedia.txt",
    "data/vi_news.txt",
    "data/vi_books.txt",
    "data/vi_web_crawl.txt",
]

# Train a BPE tokenizer on the target language
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(
    vocab_size=32000,       # Target vocabulary size for the new language
    min_frequency=100,      # Only include tokens that appear >= 100 times
    special_tokens=["<s>", "</s>", "<unk>", "<pad>"],
    show_progress=True,
)

tokenizer.train(corpus_files, trainer)
tokenizer.save("vi_tokenizer.json")
print(f"Trained tokenizer with {tokenizer.get_vocab_size()} tokens")
```

#### Step 2: Merge Vocabularies

The key decision: **which new tokens to add?** You don't want to add all 32K tokens from the target-language tokenizer. Many will overlap with the original vocabulary. You want to add only tokens that (a) don't already exist and (b) will significantly improve encoding efficiency.

```python
from transformers import AutoTokenizer
import json

# Load the original model tokenizer
original_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
original_vocab = set(original_tokenizer.get_vocab().keys())

# Load the target-language tokenizer
with open("vi_tokenizer.json", "r") as f:
    vi_tokenizer_data = json.load(f)

vi_vocab = set(vi_tokenizer_data["model"]["vocab"].keys())

# Find tokens that are NEW (not in original vocab)
new_tokens = vi_vocab - original_vocab

# Filter: only keep tokens that improve encoding efficiency
# Strategy: rank by frequency × length (longer frequent tokens save more)
def compute_token_value(token, corpus_sample):
    """
    Score = frequency × (byte_length - 1)
    A token that appears 1000 times and saves 3 bytes each time
    has value 3000.
    """
    count = corpus_sample.count(token)
    savings = len(token.encode('utf-8')) - 1  # bytes saved per occurrence
    return count * max(savings, 0)

# Load a sample of the target-language corpus for scoring
with open("data/vi_wikipedia.txt", "r") as f:
    corpus_sample = f.read()[:10_000_000]  # 10M characters

# Score and rank new tokens
token_scores = {
    token: compute_token_value(token, corpus_sample)
    for token in new_tokens
}

# Select top-K most valuable new tokens
K = 10000  # Add 10K new tokens
selected_tokens = sorted(token_scores, key=token_scores.get, reverse=True)[:K]

print(f"Adding {len(selected_tokens)} new tokens to vocabulary")
print(f"Original vocab size: {len(original_vocab)}")
print(f"New vocab size: {len(original_vocab) + len(selected_tokens)}")

# Add new tokens to the original tokenizer
original_tokenizer.add_tokens(selected_tokens)
original_tokenizer.save_pretrained("extended_tokenizer")
```

#### Step 3: Verify Efficiency Improvement

```python
# Test encoding efficiency before and after
test_sentences = [
    "Hôm nay trời đẹp quá, tôi muốn đi dạo công viên.",
    "Việt Nam là một quốc gia nằm ở Đông Nam Á.",
    "Trí tuệ nhân tạo đang thay đổi thế giới.",
]

original_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
extended_tok = AutoTokenizer.from_pretrained("extended_tokenizer")

for sent in test_sentences:
    orig_ids = original_tok.encode(sent)
    ext_ids = extended_tok.encode(sent)
    ratio = len(orig_ids) / len(ext_ids)
    print(f"Original: {len(orig_ids)} tokens | Extended: {len(ext_ids)} tokens | {ratio:.1f}x improvement")
    print(f"  Original tokens: {original_tok.convert_ids_to_tokens(orig_ids)}")
    print(f"  Extended tokens: {extended_tok.convert_ids_to_tokens(ext_ids)}")
    print()
```

Expected output:
```
Original: 25 tokens | Extended: 12 tokens | 2.1x improvement
Original: 22 tokens | Extended: 10 tokens | 2.2x improvement
Original: 18 tokens | Extended: 9 tokens | 2.0x improvement
```

### How Many Tokens to Add?

This is a judgment call. Here's a practical guide:

| New Tokens Added | Embedding Table Growth | Typical Efficiency Gain | When to Use |
|-----------------|----------------------|----------------------|-------------|
| 5,000 | +20-40 MB | 1.3-1.5x | Quick experiment, limited compute |
| 10,000 | +40-80 MB | 1.5-2.0x | Good balance for most languages |
| 20,000 | +80-160 MB | 2.0-2.5x | Morphologically rich languages (Turkish, Finnish) |
| 30,000+ | +120-240 MB | 2.0-3.0x | Languages with unique scripts (Chinese, Japanese, Thai) |

**Rule of thumb**: add tokens until the encoding efficiency gain on a held-out corpus plateaus. For most languages, 10K-20K new tokens is the sweet spot.

## Phase 2: Embedding Initialization

After extending the tokenizer, the model has new token IDs with **randomly initialized embeddings**. If you start training with random embeddings, the model will struggle — the new tokens are meaningless noise that disrupts the model's existing knowledge.

The solution: **initialize new token embeddings intelligently.**

### Strategy 1: Mean of Subword Embeddings (Simple, Effective)

The idea: the new token "trời" (sky/weather) was previously encoded as ["tr", "ờ", "i"]. Initialize the embedding for "trời" as the mean of the embeddings for "tr", "ờ", "i".

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and original tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype=torch.bfloat16,
)
original_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Load extended tokenizer
extended_tokenizer = AutoTokenizer.from_pretrained("extended_tokenizer")

# Resize model embeddings
original_vocab_size = len(original_tokenizer)
new_vocab_size = len(extended_tokenizer)
model.resize_token_embeddings(new_vocab_size)

# Get embedding layers
input_embeddings = model.get_input_embeddings().weight.data
output_embeddings = model.lm_head.weight.data

# Initialize new token embeddings using mean of subword decomposition
for token_id in range(original_vocab_size, new_vocab_size):
    token_str = extended_tokenizer.convert_ids_to_tokens(token_id)

    # Encode this token using the ORIGINAL tokenizer to get subword pieces
    subword_ids = original_tokenizer.encode(token_str, add_special_tokens=False)

    if len(subword_ids) > 0:
        # Mean of subword embeddings
        input_embeddings[token_id] = input_embeddings[subword_ids].mean(dim=0)
        output_embeddings[token_id] = output_embeddings[subword_ids].mean(dim=0)
    # else: keep random initialization (rare edge case)

print(f"Initialized {new_vocab_size - original_vocab_size} new token embeddings")
```

### Strategy 2: Weighted Subword Initialization (Better)

The mean strategy treats all subwords equally, but in practice the first subword often carries more semantic weight. A position-weighted scheme:

```python
def weighted_subword_init(subword_ids, embedding_matrix, decay=0.8):
    """
    Weighted average where earlier subwords get higher weight.
    decay=0.8 means: first subword gets weight 1.0, second gets 0.8,
    third gets 0.64, etc.
    """
    weights = torch.tensor([decay ** i for i in range(len(subword_ids))])
    weights = weights / weights.sum()  # Normalize

    subword_embeds = embedding_matrix[subword_ids]  # [num_subwords, hidden_dim]
    weighted_embed = (subword_embeds * weights.unsqueeze(1)).sum(dim=0)
    return weighted_embed

# Apply to all new tokens
for token_id in range(original_vocab_size, new_vocab_size):
    token_str = extended_tokenizer.convert_ids_to_tokens(token_id)
    subword_ids = original_tokenizer.encode(token_str, add_special_tokens=False)

    if len(subword_ids) > 0:
        input_embeddings[token_id] = weighted_subword_init(
            subword_ids, input_embeddings, decay=0.8
        )
        output_embeddings[token_id] = weighted_subword_init(
            subword_ids, output_embeddings, decay=0.8
        )
```

### Strategy 3: Cross-Lingual Embedding Alignment (Advanced)

If you have a bilingual dictionary or aligned embeddings (e.g., from FastText), you can map target-language words to their English translation embeddings:

```python
# Requires: a bilingual dictionary {target_word: english_translation}
bilingual_dict = {
    "trời": "sky",
    "đẹp": "beautiful",
    "quốc gia": "country",
    "nhân tạo": "artificial",
    # ... thousands more entries
}

for token_id in range(original_vocab_size, new_vocab_size):
    token_str = extended_tokenizer.convert_ids_to_tokens(token_id)
    clean_token = token_str.replace("Ġ", " ").strip()  # Remove BPE prefix

    if clean_token in bilingual_dict:
        # Use the English translation's embedding
        eng_translation = bilingual_dict[clean_token]
        eng_ids = original_tokenizer.encode(eng_translation, add_special_tokens=False)
        if len(eng_ids) > 0:
            input_embeddings[token_id] = input_embeddings[eng_ids].mean(dim=0)
            output_embeddings[token_id] = output_embeddings[eng_ids].mean(dim=0)
            continue

    # Fallback: mean of subword decomposition
    subword_ids = original_tokenizer.encode(token_str, add_special_tokens=False)
    if len(subword_ids) > 0:
        input_embeddings[token_id] = input_embeddings[subword_ids].mean(dim=0)
        output_embeddings[token_id] = output_embeddings[subword_ids].mean(dim=0)
```

### Which Initialization Strategy to Use?

| Strategy | Complexity | Quality | When to Use |
|----------|-----------|---------|-------------|
| Mean of subwords | Low | Good | Default choice; works for most cases |
| Weighted subwords | Low | Better | When subword decomposition is meaningful |
| Cross-lingual alignment | Medium | Best | When bilingual dictionary is available |

In practice, **mean of subwords is sufficient** for most projects. The continual pretraining phase will refine all embeddings regardless of initialization — good initialization just makes convergence faster.

### The Embedding Norm Mismatch Problem (Subtle but Important)

Here's a bug that has bitten many teams and is rarely discussed in tutorials: **the norm of initialized embeddings often doesn't match the norm of existing embeddings.** This creates an immediate distribution shift in the model's internal representations.

Why does this happen? When you average multiple subword embeddings, the result has a **smaller norm** than individual embeddings due to cancellation effects (some dimensions are positive, some negative — they partially cancel out when averaged).

```python
# Diagnosing the norm mismatch
import torch

# Check norm distribution of existing vs new embeddings
existing_norms = input_embeddings[:original_vocab_size].norm(dim=1)
new_norms = input_embeddings[original_vocab_size:].norm(dim=1)

print(f"Existing embeddings — mean norm: {existing_norms.mean():.4f}, "
      f"std: {existing_norms.std():.4f}")
print(f"New embeddings — mean norm: {new_norms.mean():.4f}, "
      f"std: {new_norms.std():.4f}")

# Typical output:
# Existing embeddings — mean norm: 0.4523, std: 0.1287
# New embeddings — mean norm: 0.2891, std: 0.0934
#
# The new embeddings are ~36% smaller! This causes problems.
```

**The fix:** Rescale new embeddings to match the existing distribution:

```python
def normalize_new_embeddings(input_embeddings, output_embeddings,
                              original_vocab_size, new_vocab_size):
    """
    Rescale new token embeddings to match the norm distribution
    of existing embeddings.
    """
    # Compute target norm statistics from existing embeddings
    existing_norms = input_embeddings[:original_vocab_size].norm(dim=1)
    target_mean_norm = existing_norms.mean()

    # Rescale each new embedding to match
    for token_id in range(original_vocab_size, new_vocab_size):
        # Input embeddings
        current_norm = input_embeddings[token_id].norm()
        if current_norm > 0:
            input_embeddings[token_id] *= (target_mean_norm / current_norm)

        # Output embeddings
        current_norm = output_embeddings[token_id].norm()
        if current_norm > 0:
            output_embeddings[token_id] *= (target_mean_norm / current_norm)

    # Verify
    new_norms = input_embeddings[original_vocab_size:].norm(dim=1)
    print(f"After normalization — new embeddings mean norm: {new_norms.mean():.4f}")

normalize_new_embeddings(input_embeddings, output_embeddings,
                         original_vocab_size, new_vocab_size)
```

**What happens without this fix?** The model treats new tokens as "quieter" inputs. During the early stages of training, the transformer layers receive weaker signals from new tokens, causing:
- Slower convergence for new tokens
- The model preferring to use existing (original vocabulary) tokens over new tokens
- In extreme cases, the model effectively ignoring new tokens and falling back to byte-level tokenization

This is one of those problems where the training loss still goes down — it just goes down slower, and you waste compute. I've seen teams lose 20-30% of effective training by not fixing this.

## Phase 3: Continual Pretraining

This is the most compute-intensive and most impactful phase. The goal: teach the model your target language's grammar, vocabulary, common knowledge, and cultural context.

### Data Collection and Preparation

**Quality and diversity of data matters more than raw volume.** Here's a practical breakdown:

#### Data Sources (Ranked by Value)

| Source | Quality | Volume | Notes |
|--------|---------|--------|-------|
| Books and literature | ★★★★★ | Low | Best grammar, diverse vocabulary |
| Wikipedia | ★★★★☆ | Medium | Factual, well-structured |
| News articles | ★★★★☆ | High | Current events, formal language |
| Government documents | ★★★★☆ | Medium | Legal/formal register |
| Academic papers | ★★★★☆ | Low-Medium | Technical vocabulary |
| Web crawl (cleaned) | ★★★☆☆ | Very High | Diverse but noisy |
| Social media | ★★☆☆☆ | Very High | Informal register, slang, noisy |
| Parallel corpora (bilingual) | ★★★★★ | Low-Medium | Critical for cross-lingual transfer |

#### Data Cleaning Pipeline

```python
import re
from datasets import load_dataset
import fasttext

# Load a language identification model (for filtering)
lang_model = fasttext.load_model("lid.176.bin")

def clean_document(text, target_lang="vi"):
    """Clean a single document for pretraining."""

    # 1. Language identification - remove non-target-language content
    predictions = lang_model.predict(text.replace("\n", " ")[:500])
    detected_lang = predictions[0][0].replace("__label__", "")
    confidence = predictions[1][0]

    if detected_lang != target_lang or confidence < 0.7:
        return None

    # 2. Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # 3. Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # 4. Remove documents that are too short (likely noise)
    if len(text.split()) < 50:
        return None

    # 5. Remove documents with too many special characters (likely code/tables)
    special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
    if special_ratio > 0.3:
        return None

    # 6. Basic deduplication signal: skip if too repetitive
    words = text.split()
    unique_ratio = len(set(words)) / len(words) if words else 0
    if unique_ratio < 0.2:  # Less than 20% unique words
        return None

    return text.strip()

# Process a dataset
def process_corpus(input_path, output_path, target_lang="vi"):
    """Clean and filter a raw corpus."""
    cleaned_count = 0
    total_count = 0

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            total_count += 1
            cleaned = clean_document(line.strip(), target_lang)
            if cleaned:
                fout.write(cleaned + "\n")
                cleaned_count += 1

    print(f"Kept {cleaned_count}/{total_count} documents ({cleaned_count/total_count*100:.1f}%)")
```

#### Data Deduplication: The Silent Quality Killer

This is one of the most underestimated steps. Duplicate or near-duplicate documents are **everywhere** in web-crawled data, and they cause real damage: the model memorizes duplicated text verbatim, wasting capacity that should be spent learning general patterns. In extreme cases, duplicated data leads to repetitive, "looping" generation at inference time.

**Exact deduplication** removes identical documents. This is the easy part:

```python
import hashlib
from collections import defaultdict

def exact_dedup(documents):
    """Remove exact duplicate documents using SHA-256 hashing."""
    seen_hashes = set()
    unique_docs = []

    for doc in documents:
        doc_hash = hashlib.sha256(doc.encode('utf-8')).hexdigest()
        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            unique_docs.append(doc)

    print(f"Exact dedup: {len(documents)} → {len(unique_docs)} "
          f"(removed {len(documents) - len(unique_docs)} duplicates)")
    return unique_docs
```

**Near-deduplication** is where it gets tricky. Many documents are 90% identical (boilerplate headers/footers, slightly edited copies, syndicated news). Use MinHash LSH for scalable near-dedup:

```python
from datasketch import MinHash, MinHashLSH

def create_minhash(text, num_perm=128):
    """Create a MinHash signature for a document."""
    m = MinHash(num_perm=num_perm)
    # Use word-level 5-grams as features
    words = text.split()
    for i in range(len(words) - 4):
        ngram = " ".join(words[i:i+5])
        m.update(ngram.encode('utf-8'))
    return m

def near_dedup(documents, threshold=0.8, num_perm=128):
    """
    Remove near-duplicate documents using MinHash LSH.
    threshold=0.8 means documents sharing >80% of their n-grams
    are considered duplicates.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    unique_docs = []
    duplicate_count = 0

    for idx, doc in enumerate(documents):
        mh = create_minhash(doc, num_perm)

        # Check if a similar document already exists
        result = lsh.query(mh)
        if len(result) == 0:
            lsh.insert(f"doc_{idx}", mh)
            unique_docs.append(doc)
        else:
            duplicate_count += 1

    print(f"Near dedup: {len(documents)} → {len(unique_docs)} "
          f"(removed {duplicate_count} near-duplicates)")
    return unique_docs
```

**How much data do you lose to deduplication?** In my experience with Vietnamese web crawl data:

| Dedup Stage | Data Remaining |
|------------|---------------|
| Raw crawl | 100% (baseline) |
| After exact dedup | ~70-80% |
| After near dedup (0.8 threshold) | ~40-60% |
| After language filtering | ~30-50% |
| After quality filtering | ~15-30% |

Yes, you often keep only 15-30% of raw crawled data. **This is normal and expected.** The 70% you throw away would actively harm training quality.

**A real horror story:** On one project, we skipped near-deduplication to save time. The model trained fine — loss went down, perplexity looked good. But during generation, it would randomly produce perfect copies of news article boilerplate: "Theo thông tin từ báo điện tử..." (According to information from the online newspaper...) followed by generic filler text. It took us two days of debugging to trace it back to 12,000 near-duplicate news articles with the same boilerplate template in the training data.

#### Data Mixing Strategy

This is crucial. You don't just dump all target-language data into training. The mix should include:

```python
# Recommended data mix for language adaptation
data_mix = {
    "target_language": 0.70,      # 70% target language text
    "english": 0.15,              # 15% English (prevent catastrophic forgetting)
    "parallel_bilingual": 0.10,   # 10% parallel/bilingual text
    "code": 0.05,                 # 5% code (preserve reasoning ability)
}
```

**Why keep English data?** Without it, the model suffers **catastrophic forgetting** — it loses its English capabilities and, more importantly, the reasoning abilities that were primarily learned from English data. The English data acts as a regularizer.

**Why include parallel text?** Bilingual text (e.g., English-Vietnamese translation pairs, bilingual documents) helps the model build cross-lingual mappings. It learns that "artificial intelligence" and "trí tuệ nhân tạo" refer to the same concept, which dramatically improves knowledge transfer.

### Training Configuration

Here's a complete training script using HuggingFace Transformers and DeepSpeed:

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, concatenate_datasets

# ============================================================
# 1. Load the model with extended tokenizer and initialized embeddings
# ============================================================
model_path = "path/to/model_with_extended_tokenizer"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # Use Flash Attention for efficiency
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ============================================================
# 2. Prepare datasets
# ============================================================
def tokenize_function(examples, max_length=2048):
    """Tokenize and chunk text into fixed-length sequences."""
    tokenized = tokenizer(
        examples["text"],
        truncation=False,
        padding=False,
    )

    # Concatenate all tokens and split into chunks
    all_input_ids = []
    for ids in tokenized["input_ids"]:
        all_input_ids.extend(ids + [tokenizer.eos_token_id])

    # Split into max_length chunks
    chunks = []
    for i in range(0, len(all_input_ids) - max_length, max_length):
        chunks.append(all_input_ids[i:i + max_length])

    return {"input_ids": chunks}

# Load and mix datasets
target_lang_data = load_dataset("text", data_files="data/vi_cleaned.txt", split="train")
english_data = load_dataset("text", data_files="data/en_sample.txt", split="train")
parallel_data = load_dataset("text", data_files="data/vi_en_parallel.txt", split="train")

# Tokenize each
target_tokenized = target_lang_data.map(
    tokenize_function, batched=True, remove_columns=["text"], num_proc=16
)
english_tokenized = english_data.map(
    tokenize_function, batched=True, remove_columns=["text"], num_proc=16
)
parallel_tokenized = parallel_data.map(
    tokenize_function, batched=True, remove_columns=["text"], num_proc=16
)

# Sample according to mix ratio
# (In production, use a proper streaming/interleaving approach)
train_dataset = concatenate_datasets([
    target_tokenized,
    english_tokenized,
    parallel_tokenized,
]).shuffle(seed=42)

# ============================================================
# 3. Training arguments
# ============================================================
training_args = TrainingArguments(
    output_dir="output/vi-llama-3-8b-cpt",

    # Core hyperparameters
    num_train_epochs=1,                   # Usually 1-2 epochs over the data
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,        # Effective batch = 4 * 8 * num_gpus

    # Learning rate: CRITICAL — too high destroys English, too low learns nothing
    learning_rate=2e-5,                   # Much lower than pretraining (typically 3e-4)
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,                    # 5% warmup

    # Precision and optimization
    bf16=True,
    optim="adamw_torch_fused",
    weight_decay=0.1,
    max_grad_norm=1.0,

    # Logging and saving
    logging_steps=10,
    save_steps=500,
    save_total_limit=5,

    # DeepSpeed ZeRO Stage 2 for multi-GPU training
    deepspeed="configs/ds_zero2.json",

    # Gradient checkpointing to save memory
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # Dataloader
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
)

# ============================================================
# 4. Create trainer and train
# ============================================================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM, not masked LM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("output/vi-llama-3-8b-cpt/final")
tokenizer.save_pretrained("output/vi-llama-3-8b-cpt/final")
```

### DeepSpeed Configuration

```json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": "auto",
        "allgather_bucket_size": "auto"
    },
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",
    "wall_clock_breakdown": false
}
```

### Critical Hyperparameters and Their Impact

Let me walk through the hyperparameters that matter most for language adaptation, because getting these wrong is the #1 reason adaptation projects fail:

#### Learning Rate

This is the single most important hyperparameter. The tension:

- **Too high** (> 5e-5): The model rapidly learns the target language but **catastrophically forgets** English and reasoning abilities. You'll see target-language perplexity drop fast, but English perplexity spikes.
- **Too low** (< 5e-6): The model barely learns. Target-language perplexity decreases very slowly. You're wasting compute.
- **Sweet spot** (1e-5 to 3e-5): The model gradually acquires target-language proficiency while retaining most English/reasoning capability.

```
Learning rate spectrum for continual pretraining:

1e-6        5e-6        1e-5        2e-5        5e-5        1e-4
  |-----------|-----------|-----------|-----------|-----------|
  Too slow    Conservative  ★ Sweet spot ★         Forgetting begins
              (safe start)                         Dangerous
```

**My recommendation**: Start with **2e-5** and monitor both target-language perplexity and English benchmark scores. If English drops more than 5%, lower the learning rate.

#### Batch Size

Larger batch sizes lead to more stable training but slower convergence per sample. For language adaptation:

- **Effective batch size of 256K-1M tokens** works well for most setups
- With `per_device_batch_size=4`, `seq_len=2048`, `grad_accum=8`, `num_gpus=4`:
  - Effective batch = 4 × 2048 × 8 × 4 = **262,144 tokens** per step

#### Number of Tokens

How much data is enough? Here's what research and practice suggest:

| Model Size | Minimum Data | Recommended | Upper Limit |
|-----------|-------------|-------------|-------------|
| 1-3B | 2B tokens | 5-10B tokens | 20B tokens |
| 7-8B | 5B tokens | 10-30B tokens | 50B tokens |
| 13B | 10B tokens | 20-50B tokens | 100B tokens |
| 70B | 20B tokens | 50-100B tokens | 200B tokens |

Going beyond the upper limit often shows diminishing returns unless you have extremely diverse, high-quality data.

### Staged Training: A More Sophisticated Approach

For best results, use a **two-stage continual pretraining** approach:

```
Stage 1: "Embedding warmup" (5-10% of total steps)
  - Only train embedding layers (input + output)
  - Freeze transformer layers
  - Higher learning rate (5e-5)
  - Purpose: Let new token embeddings converge to meaningful values
    before they disrupt transformer computations

Stage 2: "Full model training" (90-95% of total steps)
  - Unfreeze all layers
  - Lower learning rate (2e-5)
  - Full continual pretraining
```

Implementation:

```python
from transformers import TrainerCallback

class StagedTrainingCallback(TrainerCallback):
    """Freeze transformer layers during the embedding warmup stage."""

    def __init__(self, warmup_steps_ratio=0.1):
        self.warmup_steps_ratio = warmup_steps_ratio
        self.unfrozen = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Freeze everything except embeddings at the start."""
        # Freeze all transformer layers
        for name, param in model.named_parameters():
            if "embed" not in name and "lm_head" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Stage 1: Training {trainable/1e6:.0f}M / {total/1e6:.0f}M parameters "
              f"({trainable/total*100:.1f}%)")

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        """Unfreeze all layers after warmup phase."""
        if self.unfrozen:
            return

        total_steps = state.max_steps
        warmup_threshold = int(total_steps * self.warmup_steps_ratio)

        if state.global_step >= warmup_threshold:
            for param in model.parameters():
                param.requires_grad = True
            self.unfrozen = True

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\nStage 2: Unfroze all parameters at step {state.global_step}. "
                  f"Training {trainable/1e6:.0f}M parameters")

# Add callback to trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    callbacks=[StagedTrainingCallback(warmup_steps_ratio=0.1)],
)
```

### Monitoring Training: What to Watch

During continual pretraining, you must track multiple signals:

```python
# Evaluation script to run periodically during training
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_perplexity(model, tokenizer, text_file, max_samples=1000):
    """Compute perplexity on a held-out set."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with open(text_file) as f:
        lines = f.readlines()[:max_samples]

    with torch.no_grad():
        for line in lines:
            inputs = tokenizer(line.strip(), return_tensors="pt",
                             truncation=True, max_length=2048).to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()

# Track these metrics:
# 1. Target-language perplexity (should decrease)
target_ppl = evaluate_perplexity(model, tokenizer, "data/vi_eval.txt")
print(f"Vietnamese perplexity: {target_ppl:.2f}")

# 2. English perplexity (should stay stable or increase only slightly)
en_ppl = evaluate_perplexity(model, tokenizer, "data/en_eval.txt")
print(f"English perplexity: {en_ppl:.2f}")

# 3. English reasoning benchmarks (run MMLU, HellaSwag periodically)
# Use lm-evaluation-harness for standardized benchmarks
```

**Red flags during training:**

| Signal | What It Means | What to Do |
|--------|--------------|-----------|
| Target PPL not decreasing | Learning rate too low or data quality issue | Increase LR or check data |
| English PPL spiking | Catastrophic forgetting | Lower LR, add more English data to mix |
| Loss spikes | Data corruption or LR too high | Check data pipeline, lower LR |
| Target PPL oscillating | Batch size too small | Increase gradient accumulation |

## Phase 4: Instruction Tuning

After continual pretraining, the model understands your target language but doesn't follow instructions well. It'll complete text fluently but won't answer questions, follow formatting requests, or behave like a helpful assistant.

**Instruction tuning** (also called supervised fine-tuning / SFT) fixes this.

### Creating Instruction Data

You need high-quality instruction-response pairs in your target language. Sources:

#### 1. Translate Existing English Datasets

The fastest approach. Use a strong model (GPT-4, Claude) to translate high-quality English instruction datasets:

```python
# Example: translating Alpaca-style instructions
import json
from anthropic import Anthropic

client = Anthropic()

def translate_instruction(instruction, input_text, output_text, target_lang="Vietnamese"):
    prompt = f"""Translate the following instruction-response pair to {target_lang}.
Maintain the same level of detail and formatting.
Adapt cultural references where appropriate.
Return JSON with keys: instruction, input, output.

Original:
Instruction: {instruction}
Input: {input_text}
Output: {output_text}"""

    response = client.messages.create(
        model="claude-sonnet-4-6-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )

    return json.loads(response.content[0].text)

# Process dataset
with open("alpaca_data.json") as f:
    en_data = json.load(f)

translated = []
for item in en_data[:10000]:  # Translate 10K examples
    try:
        result = translate_instruction(
            item["instruction"],
            item.get("input", ""),
            item["output"]
        )
        translated.append(result)
    except Exception as e:
        print(f"Failed: {e}")
        continue

with open("vi_alpaca_data.json", "w") as f:
    json.dump(translated, f, ensure_ascii=False, indent=2)
```

#### 2. Generate Native Instructions

Translation has limitations — it produces text that "smells" like translated content. For the best quality, generate instructions natively:

```python
# Use a strong model to generate native-language instructions
def generate_native_instructions(topic, num_examples=10, target_lang="Vietnamese"):
    prompt = f"""Generate {num_examples} diverse instruction-response pairs in {target_lang}
about the topic: {topic}

Requirements:
- Instructions should be natural, as a native {target_lang} speaker would ask
- Responses should be detailed, accurate, and culturally appropriate
- Include a mix of: factual questions, how-to instructions, creative tasks, analysis
- Do NOT translate from English — write natively

Return as JSON array with keys: instruction, input (empty string if not needed), output"""

    response = client.messages.create(
        model="claude-sonnet-4-6-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(response.content[0].text)

# Generate across diverse topics
topics = [
    "Lịch sử Việt Nam",         # Vietnamese history
    "Nấu ăn món Việt",           # Vietnamese cooking
    "Lập trình Python",          # Python programming
    "Khoa học tự nhiên",         # Natural science
    "Tư vấn sức khỏe",          # Health advice
    "Văn học Việt Nam",          # Vietnamese literature
    "Kinh tế và tài chính",     # Economics and finance
    "Du lịch Việt Nam",          # Vietnam travel
]

all_instructions = []
for topic in topics:
    instructions = generate_native_instructions(topic, num_examples=50)
    all_instructions.extend(instructions)
    print(f"Generated {len(instructions)} instructions for: {topic}")
```

#### 3. Collect Real User Interactions

If you have access to real user queries in your target language (from a chatbot, search logs, customer support), these are gold. Clean them, pair with high-quality responses, and add to your training set.

### Training the Instruction-Tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import torch

# Load the continual-pretrained model
model = AutoModelForCausalLM.from_pretrained(
    "output/vi-llama-3-8b-cpt/final",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
tokenizer = AutoTokenizer.from_pretrained("output/vi-llama-3-8b-cpt/final")

# Set up chat template
CHAT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|>"""

def format_instruction(example):
    """Format a single instruction example into the chat template."""
    system_msg = "Bạn là một trợ lý AI thông minh và hữu ích. Hãy trả lời bằng tiếng Việt."

    text = CHAT_TEMPLATE.format(
        system_message=system_msg,
        instruction=example["instruction"],
        response=example["output"],
    )

    tokenized = tokenizer(text, truncation=True, max_length=2048, padding=False)

    # Mask the instruction part — only compute loss on the response
    instruction_text = CHAT_TEMPLATE.split("{response}")[0].format(
        system_message=system_msg,
        instruction=example["instruction"],
    )
    instruction_len = len(tokenizer.encode(instruction_text, add_special_tokens=False))

    labels = tokenized["input_ids"].copy()
    labels[:instruction_len] = [-100] * instruction_len  # Mask instruction tokens
    tokenized["labels"] = labels

    return tokenized

# Load and format dataset
dataset = load_dataset("json", data_files="vi_instruction_data.json", split="train")
dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)

# Training arguments for SFT
sft_args = TrainingArguments(
    output_dir="output/vi-llama-3-8b-sft",
    num_train_epochs=3,                    # 2-3 epochs for SFT
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,                    # Same range as CPT
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    optim="adamw_torch_fused",
    weight_decay=0.0,                      # Often 0 for SFT
    max_grad_norm=1.0,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

trainer = Trainer(
    model=model,
    args=sft_args,
    train_dataset=dataset,
)

trainer.train()
trainer.save_model("output/vi-llama-3-8b-sft/final")
```

### Using LoRA for Efficient Instruction Tuning

If compute is limited, LoRA is an excellent option for the SFT phase (though I recommend full fine-tuning for the continual pretraining phase):

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=64,                          # Rank — 32-128 for language adaptation
    lora_alpha=128,                # Alpha — typically 2x rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",       # FFN
    ],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 167,772,160 || all params: 8,198,934,528 || trainable%: 2.05%
```

## Phase 5: Evaluation

How do you know if the adaptation worked? You need to measure multiple dimensions:

### 1. Language Proficiency

```python
# Perplexity on target-language held-out set
# Lower is better; compare against baseline (original model)

from lm_eval import evaluator

results = evaluator.simple_evaluate(
    model="hf",
    model_args=f"pretrained=output/vi-llama-3-8b-sft/final",
    tasks=["perplexity_vi"],  # Custom task for Vietnamese perplexity
    batch_size=4,
)
```

### 2. Task Performance

Create or use existing benchmarks in your target language:

```python
# Example: Vietnamese question answering evaluation
test_questions = [
    {
        "question": "Thủ đô của Việt Nam là gì?",
        "expected": "Hà Nội",
    },
    {
        "question": "Sông nào dài nhất Việt Nam?",
        "expected": "Sông Mê Kông",  # or "Sông Đồng Nai" depending on definition
    },
    # ... hundreds more
]

def evaluate_qa(model, tokenizer, questions):
    correct = 0
    for q in questions:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=False,
            )

        response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        if q["expected"].lower() in response.lower():
            correct += 1

    accuracy = correct / len(questions)
    print(f"QA Accuracy: {accuracy:.2%} ({correct}/{len(questions)})")
    return accuracy
```

### 3. English Retention

**Critical**: measure how much English capability was preserved.

```bash
# Run standard English benchmarks using lm-evaluation-harness
lm_eval --model hf \
    --model_args pretrained=output/vi-llama-3-8b-sft/final \
    --tasks mmlu,hellaswag,arc_challenge,winogrande \
    --batch_size 4 \
    --output_path results/english_benchmarks.json
```

Compare against the original model's scores. **Acceptable degradation: < 5% on aggregate English benchmarks.** If degradation is higher, you need to increase the English data ratio in continual pretraining.

### 4. Generation Quality (Human Evaluation)

Automated metrics only tell part of the story. Have native speakers evaluate:

- **Fluency**: Does the text sound natural? (1-5 scale)
- **Grammar**: Are there grammatical errors? (count per 100 words)
- **Cultural appropriateness**: Does it use correct idioms, references? (1-5 scale)
- **Instruction following**: Does it actually answer what was asked? (1-5 scale)

```python
# Generate samples for human evaluation
evaluation_prompts = [
    "Viết một bài thơ ngắn về mùa thu ở Hà Nội.",
    "Giải thích cách nấu phở bò truyền thống.",
    "So sánh ưu và nhược điểm của năng lượng mặt trời.",
    "Viết email xin nghỉ phép gửi sếp.",
    "Tóm tắt lịch sử triều đại nhà Nguyễn.",
]

print("=" * 80)
for prompt in evaluation_prompts:
    inputs = tokenizer(
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        return_tensors="pt"
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    print(f"\n📝 Prompt: {prompt}")
    print(f"📤 Response: {response}")
    print("=" * 80)
```

### Evaluation Summary Table

Create a comprehensive comparison:

| Metric | Original Model | After CPT | After SFT | Target |
|--------|---------------|-----------|-----------|--------|
| Vietnamese PPL | 45.2 | 12.8 | 11.5 | < 15 |
| Vietnamese QA Accuracy | 32% | 68% | 75% | > 70% |
| English MMLU | 65.2% | 63.8% | 63.1% | > 60% |
| English HellaSwag | 78.5% | 77.2% | 76.8% | > 74% |
| Tokenization Efficiency | 1.0x | 2.1x | 2.1x | > 1.5x |
| Human Fluency (1-5) | 2.1 | 3.8 | 4.2 | > 4.0 |

## Troubles in Training: The Complete War Stories

This section is the most important part of this article. The pipeline I described above looks clean on paper, but **real training runs are messy.** Here's everything that can go wrong — and how to diagnose and fix it. These are problems I've seen repeatedly across multiple language adaptation projects.

### Trouble #1: The Language Confusion Problem

**What happens:** After adaptation, the model randomly switches between languages mid-sentence. You ask a question in Vietnamese, and it answers in a mix of Vietnamese and English. Or worse: it answers entirely in English despite being prompted in Vietnamese.

```
User: Giải thích cách hoạt động của mạng neural
Model: Mạng neural là một model that consists of layers of interconnected nodes,
       mỗi node thực hiện a weighted sum of its inputs...
```

**Root cause:** This almost always means one of three things:

1. **Insufficient target-language data**: The model hasn't seen enough monolingual text to develop a strong "language mode." It defaults to English because English is the path of least resistance.

2. **Too much parallel/bilingual data in the mix**: Ironically, parallel data can cause this. If 15-20% of your training data is bilingual, the model learns that "mixing languages is normal." Reduce parallel data to 5-10%.

3. **The instruction tuning data is translated, not native**: Translated text preserves English sentence structures. The model learns to "think in English" even when outputting Vietnamese tokens.

**Diagnosis script:**

```python
def diagnose_language_confusion(model, tokenizer, test_prompts, target_lang="vi"):
    """
    Test if the model stays in the target language.
    Returns the fraction of response tokens that are in the target language.
    """
    import fasttext
    lang_model = fasttext.load_model("lid.176.bin")

    results = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:],
                                   skip_special_tokens=True)

        # Check language of each sentence in the response
        sentences = [s.strip() for s in response.split('.') if len(s.strip()) > 10]
        target_lang_count = 0
        for sent in sentences:
            pred = lang_model.predict(sent.replace('\n', ' '))
            detected = pred[0][0].replace("__label__", "")
            if detected == target_lang:
                target_lang_count += 1

        ratio = target_lang_count / max(len(sentences), 1)
        results.append({
            "prompt": prompt[:50],
            "target_lang_ratio": ratio,
            "sample_response": response[:100],
        })

    avg_ratio = sum(r["target_lang_ratio"] for r in results) / len(results)
    print(f"Average target-language ratio: {avg_ratio:.2%}")
    if avg_ratio < 0.85:
        print("WARNING: Language confusion detected!")
        print("Recommendations:")
        print("  - Increase monolingual target-language data to 75-80%")
        print("  - Reduce parallel bilingual data to 5%")
        print("  - Use natively-generated (not translated) instruction data")
    return results
```

**The fix hierarchy:**

1. First, increase monolingual target-language data ratio to 80%
2. Add a "language tag" prefix to your training data: prepend `[VI]` or `[EN]` to each document so the model learns to condition on language
3. During instruction tuning, add explicit system prompts: "You must respond entirely in Vietnamese. Never switch to English."
4. As a last resort: add a **language consistency reward** in DPO — generate paired responses where the "chosen" is monolingual and the "rejected" mixes languages

### Trouble #2: Catastrophic Forgetting — The Silent Killer

**What happens:** Your target-language benchmarks look great. You're celebrating. Then someone runs the English MMLU and it dropped from 65% to 48%. The model has forgotten how to reason in English, and because reasoning was primarily learned from English data, **it also reasons worse in the target language.**

This is the most common and most devastating failure mode. The insidious part: target-language perplexity looks good, target-language generation looks fluent, but the model has lost deep capabilities.

**How to detect it early:**

```python
class CatastrophicForgettingDetector:
    """
    Monitor for catastrophic forgetting during training.
    Run this every N steps as part of your evaluation loop.
    """

    def __init__(self, model, tokenizer, english_eval_file, threshold=0.15):
        self.tokenizer = tokenizer
        self.english_eval_file = english_eval_file
        self.threshold = threshold  # Max acceptable PPL increase ratio
        self.baseline_ppl = None

    def check(self, model, step):
        model.eval()
        current_ppl = self._compute_ppl(model)

        if self.baseline_ppl is None:
            self.baseline_ppl = current_ppl
            print(f"[Step {step}] Baseline English PPL: {current_ppl:.2f}")
            return False

        ppl_increase = (current_ppl - self.baseline_ppl) / self.baseline_ppl
        status = "OK" if ppl_increase < self.threshold else "ALERT"

        print(f"[Step {step}] English PPL: {current_ppl:.2f} "
              f"(+{ppl_increase:.1%} from baseline) [{status}]")

        if ppl_increase > self.threshold:
            print(f"  CATASTROPHIC FORGETTING DETECTED!")
            print(f"  English PPL increased by {ppl_increase:.1%} "
                  f"(threshold: {self.threshold:.0%})")
            print(f"  Recommended actions:")
            print(f"    1. Reduce learning rate by 50%")
            print(f"    2. Increase English data ratio to 25%")
            print(f"    3. Consider rolling back to last good checkpoint")
            return True

        return False

    def _compute_ppl(self, model):
        import torch
        total_loss = 0
        total_tokens = 0
        with open(self.english_eval_file) as f:
            lines = f.readlines()[:200]
        with torch.no_grad():
            for line in lines:
                inputs = self.tokenizer(line.strip(), return_tensors="pt",
                                       truncation=True, max_length=512).to(model.device)
                if inputs["input_ids"].shape[1] < 10:
                    continue
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
                total_tokens += inputs["input_ids"].shape[1]
        return torch.exp(torch.tensor(total_loss / total_tokens)).item()
```

**What the learning rate vs forgetting curve looks like in practice:**

```
English MMLU Score (%) over training steps:

LR=5e-5 (too high):
65% ████████████████████████████████
60% ██████████████████████
50%     ██████████████████
40%          ███████████████
30%               ████████████     ← Model is destroyed
     0    1K    2K    3K    4K

LR=2e-5 (sweet spot):
65% ████████████████████████████████
63% ██████████████████████████████████████████
62%     ████████████████████████████████████████   ← Stable!
61%         ████████████████████████████████████
     0    1K    2K    3K    4K

LR=5e-6 (too conservative):
65% ████████████████████████████████████████████
64% ██████████████████████████████████████████████
     0    1K    2K    3K    4K
     (But target-language PPL barely improves...)
```

### Trouble #3: OOM (Out of Memory) During Training

This is the most *frequent* problem you'll hit. Language adaptation with extended vocabulary makes OOM worse than standard fine-tuning because:

1. **Larger embedding table**: Adding 10K tokens to LLaMA 3 8B adds ~80MB to the embedding layer and ~80MB to the lm_head
2. **Longer effective sequences**: If your target language uses longer byte sequences, the same text produces more tokens before tokenizer extension (and your data pipeline might not account for this)
3. **Gradient accumulation memory**: The optimizer states for the larger embedding table consume additional memory

**Common OOM scenarios and fixes:**

```python
# Scenario 1: OOM on the FIRST forward pass
# Cause: Model + optimizer states don't fit in GPU memory
# Fix: Use gradient checkpointing + reduce batch size

training_args = TrainingArguments(
    gradient_checkpointing=True,  # Trades compute for memory
    gradient_checkpointing_kwargs={"use_reentrant": False},
    per_device_train_batch_size=2,  # Start small, increase later
    gradient_accumulation_steps=16,  # Keep effective batch size large
)

# Scenario 2: OOM after N steps (memory leak)
# Cause: Usually a data pipeline issue — some sequences are much
# longer than expected, causing a spike in activation memory
# Fix: Enforce strict max_length in tokenization

def safe_tokenize(examples, max_length=2048):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,       # CRITICAL: always truncate
        max_length=max_length,
        padding=False,
    )
    # Additional safety: filter out sequences that are too long
    # (belt and suspenders)
    filtered = {
        "input_ids": [
            ids for ids in tokenized["input_ids"]
            if len(ids) <= max_length
        ]
    }
    return filtered

# Scenario 3: OOM only on specific GPUs in multi-GPU setup
# Cause: Uneven data distribution — some GPUs get batches with
# longer sequences due to poor shuffling
# Fix: Use length-grouped sampling

from transformers import LengthGroupedSampler

training_args = TrainingArguments(
    group_by_length=True,  # Groups similar-length sequences together
    # This also improves training throughput by reducing padding
)
```

**Memory budget calculator:**

```python
def estimate_memory_requirement(
    model_params_billions,
    vocab_size_extension,
    hidden_dim=4096,
    seq_length=2048,
    batch_size=4,
    optimizer="adamw",
    precision="bf16",
):
    """Estimate peak GPU memory for training."""
    bytes_per_param = 2 if precision == "bf16" else 4

    # Model weights
    model_memory_gb = model_params_billions * 1e9 * bytes_per_param / 1e9

    # Extended vocabulary overhead
    vocab_overhead_gb = vocab_size_extension * hidden_dim * bytes_per_param * 2 / 1e9  # input + output

    # Optimizer states (AdamW: 2 states per param, in FP32)
    optimizer_memory_gb = model_params_billions * 1e9 * 4 * 2 / 1e9  # m and v in FP32

    # Gradient memory
    gradient_memory_gb = model_params_billions * 1e9 * bytes_per_param / 1e9

    # Activation memory (rough estimate with gradient checkpointing)
    activation_memory_gb = batch_size * seq_length * hidden_dim * 4 * 2 / 1e9  # very rough

    total = (model_memory_gb + vocab_overhead_gb + optimizer_memory_gb +
             gradient_memory_gb + activation_memory_gb)

    print(f"Memory Estimate:")
    print(f"  Model weights:     {model_memory_gb:.1f} GB")
    print(f"  Vocab extension:   {vocab_overhead_gb:.2f} GB")
    print(f"  Optimizer states:  {optimizer_memory_gb:.1f} GB")
    print(f"  Gradients:         {gradient_memory_gb:.1f} GB")
    print(f"  Activations:       {activation_memory_gb:.1f} GB")
    print(f"  ─────────────────────────")
    print(f"  TOTAL:             {total:.1f} GB")
    print(f"  Recommended GPU:   {'A100 80GB' if total < 70 else '2×A100 80GB or H100'}")

# Example for LLaMA 3 8B with 10K new tokens
estimate_memory_requirement(
    model_params_billions=8,
    vocab_size_extension=10000,
    hidden_dim=4096,
    seq_length=2048,
    batch_size=4,
)
```

### Trouble #4: NCCL Errors and Multi-GPU Training Failures

When you scale to multiple GPUs, a whole new class of problems appears. These are the most frustrating because the error messages are cryptic.

**Common NCCL errors and what they actually mean:**

```bash
# Error 1: NCCL timeout
# RuntimeError: NCCL communicator was aborted on rank 2. Original reason for
# the abort: watchdog callback timed out
#
# What it means: One GPU is dramatically slower than others. The fast GPUs
# finish their batch and wait for the slow one, eventually timing out.
#
# Causes:
# - Uneven batch sizes (one GPU got a batch with much longer sequences)
# - Thermal throttling on one GPU
# - A different process hogging one GPU
# - NVLink/PCIe bandwidth issues between specific GPU pairs
#
# Fix:
export NCCL_TIMEOUT=1800  # Increase timeout to 30 minutes (default is 600s)
export NCCL_DEBUG=WARN    # Get more detailed error messages
# Also: use group_by_length=True in TrainingArguments

# Error 2: NCCL "unhandled system error"
# RuntimeError: NCCL error in: ../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp
#
# What it means: Usually a hardware or driver issue.
#
# Fix:
# 1. Check nvidia-smi for GPU health
# 2. Verify CUDA and NCCL versions match
# 3. Try NCCL_P2P_DISABLE=1 to disable peer-to-peer GPU communication
# 4. If on cloud: the instance might have a faulty GPU. Request a new one.

# Error 3: Gradient desync across ranks
# (No explicit error — but loss values differ across GPUs)
#
# What it means: The model copies on different GPUs have diverged.
# This can happen with gradient checkpointing + non-deterministic operations.
#
# Fix:
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# And in your code:
# torch.use_deterministic_algorithms(True)  # Can be slow but useful for debugging
```

**Pro tip:** Always save the NCCL debug environment in your training script:

```python
import os
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_TIMEOUT"] = "1800"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

# This gives you much more useful error messages when things go wrong
```

### Trouble #5: Loss Spikes and NaN Losses

You're watching your training loss curve and suddenly — a massive spike, or worse, NaN.

```
Training loss (typical language adaptation run):

Step 1-100:      3.45 → 3.21 (normal warmup)
Step 100-500:    3.21 → 2.85 (healthy decrease)
Step 501:        2.84
Step 502:        17.23  ← SPIKE
Step 503:        2.91   (recovers)
Step 504:        2.88
Step 1000-2000:  2.88 → 2.45 (healthy)
Step 2001:       NaN    ← FATAL
```

**Common causes and fixes:**

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Occasional spikes, recovers | Bad data (corrupted sequence, very long doc) | Improve data cleaning; cap sequence length |
| Spike after loading checkpoint | Learning rate schedule reset | Set `--ignore_data_skip` or fix LR warmup |
| NaN after N steps | Gradient overflow in BF16 | Lower learning rate; increase `max_grad_norm` |
| NaN at specific step | One bad data sample | Skip that sample; investigate |
| Progressive instability | Learning rate too high for this stage | Reduce LR by 50% |
| Spike when unfreezing layers | Stage transition too abrupt | Use gradual unfreezing or LR warmup at stage boundary |

**A diagnostic callback for loss monitoring:**

```python
from transformers import TrainerCallback
import math

class LossAnomalyDetector(TrainerCallback):
    """Detect loss spikes and NaN values during training."""

    def __init__(self, spike_threshold=3.0, window_size=50):
        self.loss_history = []
        self.spike_threshold = spike_threshold
        self.window_size = window_size

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs:
            return

        current_loss = logs["loss"]

        # Check for NaN
        if math.isnan(current_loss) or math.isinf(current_loss):
            print(f"\n{'='*60}")
            print(f"FATAL: NaN/Inf loss at step {state.global_step}!")
            print(f"Last 10 losses: {self.loss_history[-10:]}")
            print(f"Recommended: reduce LR, check data at this step")
            print(f"{'='*60}\n")
            # Optionally: control.should_training_stop = True

        # Check for spikes
        if len(self.loss_history) >= self.window_size:
            recent_mean = sum(self.loss_history[-self.window_size:]) / self.window_size
            if current_loss > recent_mean * self.spike_threshold:
                print(f"\n⚠ Loss spike at step {state.global_step}: "
                      f"{current_loss:.4f} (recent mean: {recent_mean:.4f}, "
                      f"{current_loss/recent_mean:.1f}x)")

        self.loss_history.append(current_loss)
```

### Trouble #6: The "Translated-ese" Problem in Instruction Data

This is subtle and many teams don't even realize it's happening. When you translate English instruction data to build your SFT dataset, the translated text **structurally reads like English written in Vietnamese words.**

Example of "translated-ese" in Vietnamese:

```
Translated (unnatural):
Q: "Giải thích tầm quan trọng của trí tuệ nhân tạo trong chăm sóc sức khỏe"
A: "Trí tuệ nhân tạo là quan trọng trong chăm sóc sức khỏe vì nó có thể
    giúp các bác sĩ chẩn đoán bệnh chính xác hơn. Đầu tiên, AI có thể
    phân tích hình ảnh y tế. Thứ hai, AI có thể dự đoán rủi ro bệnh tật.
    Thứ ba, AI có thể tối ưu hóa kế hoạch điều trị."

Native (natural):
Q: "AI giúp ích gì cho ngành y?"
A: "Ứng dụng AI trong y tế giờ đã không còn là chuyện xa vời. Ở nhiều
    bệnh viện lớn, AI đang hỗ trợ bác sĩ đọc phim X-quang với độ chính xác
    cao hơn mắt thường. Điều đáng nói là AI không thay thế bác sĩ mà đóng vai
    trò như một 'đôi mắt thứ hai' — giúp phát hiện những dấu hiệu mà con người
    dễ bỏ sót khi mệt mỏi sau ca trực dài."
```

The differences are stark:
- The translated version uses formal, list-style structure (First... Second... Third...) — very "English essay"
- The native version uses colloquial connectors, metaphor ("đôi mắt thứ hai"), and culturally grounded examples
- The translated version's sentence structure mirrors English SVO patterns exactly

**The fix:**

```python
# Strategy: Use a 70/30 mix of native-generated vs translated instructions
instruction_data_mix = {
    "native_generated": 0.50,    # Highest quality, most natural
    "human_written": 0.20,       # If you have real user data
    "translated_then_edited": 0.20,  # Translate, then have natives edit
    "raw_translated": 0.10,      # Acceptable for factual/technical topics
}

# When generating native instructions, give the model cultural context:
native_gen_prompt = """
Generate instruction-response pairs in Vietnamese.

IMPORTANT RULES:
- Write as a native Vietnamese speaker would naturally express themselves
- Use Vietnamese-specific examples (Vietnamese geography, food, history, companies)
- Use natural Vietnamese connectors (tuy nhiên, ngoài ra, thế nhưng, nói cách khác)
  NOT translated English connectors (đầu tiên... thứ hai... thứ ba...)
- Use culturally appropriate formality levels
- Reference Vietnamese context when relevant (Vietnamese law, Vietnamese companies,
  Vietnamese educational system)
- It's OK to use common Vietnamese internet slang and abbreviations in casual topics

BAD example (translated-ese):
"Đầu tiên, bạn cần chuẩn bị nguyên liệu. Thứ hai, bạn cần rửa sạch rau."

GOOD example (native):
"Khâu sơ chế là quan trọng nhất. Rau mua về phải ngâm nước muối loãng 15 phút
cho sạch thuốc trừ sâu, rồi mới rửa lại bằng nước sạch."
"""
```

### Trouble #7: Checkpoint Corruption and Recovery

Long training runs crash. Power failures happen. GPUs die. The question is: can you recover?

**Common checkpoint problems:**

```python
# Problem 1: Checkpoint saved during gradient accumulation
# The model state is mid-update — loading it produces garbage
# Fix: Only save at the END of a gradient accumulation cycle

training_args = TrainingArguments(
    save_steps=500,
    # Make sure save_steps is a multiple of gradient_accumulation_steps!
    gradient_accumulation_steps=8,
    # save_steps=500 is fine (500 % 8 != 0, but HuggingFace handles this)
    # But with custom training loops, enforce this yourself
)

# Problem 2: DeepSpeed checkpoint doesn't include optimizer states
# Loading a DeepSpeed checkpoint with a different world_size fails
# Fix: Always convert to universal checkpoint format periodically

# In your DeepSpeed config, add:
# "checkpoint": {
#     "tag_validation": "warn",
#     "save_universal": true    # ← THIS saves a format loadable on any world_size
# }

# Problem 3: Tokenizer not saved with checkpoint
# You load a checkpoint but the tokenizer is the original (not extended)
# Model crashes because embedding table size doesn't match
# Fix: ALWAYS save tokenizer alongside model

class CheckpointSafetyCallback(TrainerCallback):
    """Ensure tokenizer and metadata are saved with every checkpoint."""

    def __init__(self, tokenizer, metadata=None):
        self.tokenizer = tokenizer
        self.metadata = metadata or {}

    def on_save(self, args, state, control, **kwargs):
        # Save tokenizer to each checkpoint directory
        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training metadata
        import json
        metadata = {
            "step": state.global_step,
            "epoch": state.epoch,
            "best_metric": state.best_metric,
            "original_vocab_size": self.metadata.get("original_vocab_size"),
            "extended_vocab_size": len(self.tokenizer),
            **self.metadata,
        }
        with open(os.path.join(checkpoint_dir, "training_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
```

### Trouble #8: Data Pipeline Bottlenecks

Here's a problem that wastes money but doesn't show up as a training error: **your GPUs are idle 30-50% of the time** because the data pipeline can't feed them fast enough.

**How to diagnose:**

```bash
# Watch GPU utilization during training
watch -n 1 nvidia-smi

# If you see GPU utilization frequently dropping to 0% then bouncing back
# to 100%, your data pipeline is the bottleneck.
```

**Common causes and fixes:**

```python
# Cause 1: Tokenization happening on-the-fly during training
# Fix: Pre-tokenize your entire dataset BEFORE training
#
# This is the single biggest performance win. Tokenize once, save to disk,
# load pre-tokenized data during training.

from datasets import load_dataset

# SLOW: tokenizing during training
dataset = load_dataset("text", data_files="data/vi_cleaned.txt")
# trainer.train()  # Tokenization happens in the data loader → slow

# FAST: pre-tokenize and save
dataset = load_dataset("text", data_files="data/vi_cleaned.txt", split="train")
tokenized = dataset.map(tokenize_function, batched=True, num_proc=32,
                        remove_columns=["text"])
tokenized.save_to_disk("data/vi_tokenized")

# Later, during training:
tokenized = load_from_disk("data/vi_tokenized")  # Instant loading

# Cause 2: Too few dataloader workers
training_args = TrainingArguments(
    dataloader_num_workers=8,     # Use more workers (default is 0!)
    dataloader_pin_memory=True,   # Pin memory for faster GPU transfer
    dataloader_prefetch_factor=2, # Prefetch batches
)

# Cause 3: Data on network storage (NFS, S3)
# Fix: Copy data to local SSD before training
# On cloud instances, /tmp or instance SSD is much faster than EBS/NFS
```

### Trouble #9: The "It Generates Fine But Fails at Tasks" Problem

This is the most confusing outcome. The model generates fluent, natural target-language text. Perplexity is good. But when you test it on downstream tasks (QA, summarization, classification), performance is poor.

**Root cause analysis:**

```
"Generates fluent text" ≠ "Understands the language deeply"

Fluent generation requires:
  ✓ Token-level patterns (what comes after what)
  ✓ Grammar (correct morphology, word order)
  ✓ Vocabulary (using the right words)

Task performance requires:
  ✓ All of the above, PLUS:
  ✗ Semantic understanding (what does this text MEAN?)
  ✗ World knowledge in the target language
  ✗ Instruction following in the target language
  ✗ Reasoning in the target language
```

**Typical cause:** You trained on too much low-quality web crawl and not enough high-quality, knowledge-rich text. The model learned surface patterns but not deep understanding.

**Fix strategy:**

1. **Increase the proportion of knowledge-rich data**: Wikipedia, textbooks, encyclopedias, educational content. These teach facts and relationships, not just patterns.

2. **Add question-answering format data to continual pretraining** (not just instruction tuning):

```python
# Inject QA-format data into the CPT phase
# This teaches the model to extract and reason about information
qa_format_data = """
Bối cảnh: Sông Mekong là con sông dài nhất Đông Nam Á, chảy qua 6 quốc gia
bao gồm Trung Quốc, Myanmar, Lào, Thái Lan, Campuchia và Việt Nam. Tại Việt Nam,
sông Mekong được gọi là sông Cửu Long, đổ ra Biển Đông qua 9 cửa sông.

Câu hỏi: Sông Mekong chảy qua bao nhiêu quốc gia?
Trả lời: Sông Mekong chảy qua 6 quốc gia.

Câu hỏi: Tại Việt Nam, sông Mekong có tên gọi gì?
Trả lời: Tại Việt Nam, sông Mekong được gọi là sông Cửu Long.
"""
# Mix 5-10% of this format into your CPT data
```

3. **Evaluate on tasks *during* continual pretraining**, not just afterward. If task performance plateaus while perplexity is still improving, you've hit a quality ceiling — improve your data.

### Trouble #10: Tokenizer Extension Breaking Special Tokens

A subtle but devastating bug: after extending the tokenizer, special tokens (`<s>`, `</s>`, `<|eot_id|>`) get reassigned to wrong IDs, or new tokens collide with special token patterns.

```python
# This looks innocent but can cause chaos:
extended_tokenizer.add_tokens(new_tokens)

# Problem: if any of new_tokens contains a substring that matches
# a special token pattern, the tokenizer behavior breaks

# Example: you add "eot" as a Vietnamese token (it's a valid word root)
# Now "<|eot_id|>" might get tokenized differently

# ALWAYS verify special tokens after extension:
def verify_special_tokens(tokenizer):
    """Verify that all special tokens still work correctly."""
    special_tokens = {
        "bos": tokenizer.bos_token,
        "eos": tokenizer.eos_token,
        "pad": tokenizer.pad_token,
    }

    # Also check model-specific special tokens
    if hasattr(tokenizer, 'chat_template'):
        # For LLaMA 3 / Mistral style
        test_conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        try:
            formatted = tokenizer.apply_chat_template(test_conversation,
                                                       tokenize=False)
            # Re-tokenize and check special tokens are single tokens
            tokens = tokenizer.encode(formatted, add_special_tokens=False)
            decoded = tokenizer.decode(tokens)
            assert "<|eot_id|>" in decoded or "</s>" in decoded, \
                "Special tokens not properly encoded!"
            print("Chat template: OK")
        except Exception as e:
            print(f"Chat template BROKEN: {e}")

    for name, token in special_tokens.items():
        if token is None:
            print(f"WARNING: {name}_token is None!")
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        roundtrip = tokenizer.convert_ids_to_tokens(token_id)
        status = "OK" if roundtrip == token else f"BROKEN (got '{roundtrip}')"
        print(f"  {name}: '{token}' → ID {token_id} → '{roundtrip}' [{status}]")

verify_special_tokens(extended_tokenizer)
```

### Trouble #11: Gradient Explosion During Stage Transition

When you switch from Stage 1 (frozen transformer, training embeddings only) to Stage 2 (full model training), there's often a sharp gradient spike. The newly unfrozen layers suddenly receive gradient signal from embeddings that have drifted significantly from their original values.

**What the loss curve looks like:**

```
Loss during staged training:

Stage 1 (embeddings only)          Stage 2 (full model)
3.5 ████
3.0     ████████
2.8         ████████                                ← Spike at transition
5.2                  █████████
3.1                           ███
2.7                              ███████
2.4                                     ████████████
```

**The fix — gradual unfreezing with LR warmup at transition:**

```python
class GradualUnfreezeCallback(TrainerCallback):
    """
    Instead of unfreezing all layers at once, unfreeze progressively
    from output layers to input layers. This prevents gradient shock.
    """

    def __init__(self, model, warmup_ratio=0.1, unfreeze_schedule=None):
        self.model = model
        self.warmup_ratio = warmup_ratio
        self.stage = 0
        self.total_transformer_layers = None

        # Default: unfreeze in 4 phases
        # Phase 0: Only embeddings (during warmup)
        # Phase 1: Last 25% of transformer layers
        # Phase 2: Last 50% of transformer layers
        # Phase 3: All layers
        self.unfreeze_schedule = unfreeze_schedule or [0.75, 0.50, 0.25, 0.0]

    def _get_transformer_layers(self):
        """Get ordered list of transformer layer names."""
        layer_names = []
        for name, _ in self.model.named_parameters():
            if "layers." in name:
                layer_num = int(name.split("layers.")[1].split(".")[0])
                if layer_num not in [l for l in layer_names]:
                    layer_names.append(layer_num)
        return sorted(set(layer_names))

    def on_step_begin(self, args, state, control, **kwargs):
        if self.total_transformer_layers is None:
            self.total_transformer_layers = self._get_transformer_layers()

        total_steps = state.max_steps
        warmup_end = int(total_steps * self.warmup_ratio)
        remaining_steps = total_steps - warmup_end
        phase_length = remaining_steps // len(self.unfreeze_schedule)

        if state.global_step < warmup_end:
            return  # Still in embedding warmup

        # Determine current phase
        steps_since_warmup = state.global_step - warmup_end
        current_phase = min(
            steps_since_warmup // phase_length,
            len(self.unfreeze_schedule) - 1
        )

        if current_phase > self.stage:
            self.stage = current_phase
            # Unfreeze layers from the tail end
            cutoff_ratio = self.unfreeze_schedule[current_phase]
            cutoff_layer = int(len(self.total_transformer_layers) * cutoff_ratio)

            for name, param in self.model.named_parameters():
                if "layers." in name:
                    layer_num = int(name.split("layers.")[1].split(".")[0])
                    param.requires_grad = layer_num >= cutoff_layer
                elif "embed" in name or "lm_head" in name:
                    param.requires_grad = True

            trainable = sum(p.numel() for p in self.model.parameters()
                          if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"\nPhase {current_phase + 1}: Unfroze layers >= {cutoff_layer}. "
                  f"Training {trainable/1e6:.0f}M / {total/1e6:.0f}M params "
                  f"({trainable/total*100:.1f}%)")
```

### The Complete Debugging Toolkit

Here's a consolidated script I use at the start of every language adaptation project:

```python
"""
Language Adaptation Debugging Toolkit
Run this periodically during training to catch problems early.
"""

import torch
import json
from pathlib import Path

class AdaptationDebugger:
    """All-in-one debugger for language adaptation training."""

    def __init__(self, model, tokenizer, original_vocab_size,
                 target_eval_file, english_eval_file):
        self.model = model
        self.tokenizer = tokenizer
        self.original_vocab_size = original_vocab_size
        self.target_eval_file = target_eval_file
        self.english_eval_file = english_eval_file
        self.history = []

    def run_full_diagnostic(self, step):
        """Run all diagnostics and return a report."""
        report = {"step": step}

        # 1. Embedding health check
        report["embedding_health"] = self._check_embedding_health()

        # 2. Perplexity check
        report["target_ppl"] = self._compute_ppl(self.target_eval_file)
        report["english_ppl"] = self._compute_ppl(self.english_eval_file)

        # 3. Token utilization check
        report["token_utilization"] = self._check_token_utilization()

        # 4. Gradient norm check
        report["grad_norms"] = self._check_gradient_norms()

        self.history.append(report)
        self._print_report(report)
        return report

    def _check_embedding_health(self):
        """Check if new embeddings are healthy."""
        embed = self.model.get_input_embeddings().weight.data
        old_norms = embed[:self.original_vocab_size].norm(dim=1)
        new_norms = embed[self.original_vocab_size:].norm(dim=1)

        # Check for dead embeddings (norm ≈ 0)
        dead_new = (new_norms < 1e-6).sum().item()
        # Check for exploding embeddings (norm >> mean)
        exploding = (new_norms > old_norms.mean() * 5).sum().item()

        return {
            "old_mean_norm": old_norms.mean().item(),
            "new_mean_norm": new_norms.mean().item(),
            "norm_ratio": new_norms.mean().item() / old_norms.mean().item(),
            "dead_embeddings": dead_new,
            "exploding_embeddings": exploding,
            "status": "OK" if (0.5 < new_norms.mean().item() / old_norms.mean().item() < 2.0
                               and dead_new == 0) else "WARNING"
        }

    def _compute_ppl(self, eval_file):
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        with open(eval_file) as f:
            lines = f.readlines()[:100]
        with torch.no_grad():
            for line in lines:
                inputs = self.tokenizer(line.strip(), return_tensors="pt",
                                       truncation=True, max_length=512)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                if inputs["input_ids"].shape[1] < 5:
                    continue
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
                total_tokens += inputs["input_ids"].shape[1]
        return torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()

    def _check_token_utilization(self):
        """Check if new tokens are actually being used by the model."""
        with open(self.target_eval_file) as f:
            sample_text = f.read()[:100000]

        token_ids = self.tokenizer.encode(sample_text)
        new_token_count = sum(1 for t in token_ids if t >= self.original_vocab_size)
        total_count = len(token_ids)

        return {
            "new_token_usage_ratio": new_token_count / total_count,
            "total_tokens": total_count,
            "new_tokens_used": new_token_count,
            "status": "OK" if new_token_count / total_count > 0.1 else
                      "WARNING: New tokens underutilized"
        }

    def _check_gradient_norms(self):
        """Check gradient norms per component."""
        norms = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if "embed" in name:
                    key = "embeddings"
                elif "lm_head" in name:
                    key = "lm_head"
                elif "layers.0." in name:
                    key = "first_layer"
                elif "layers.31." in name or "layers.27." in name:
                    key = "last_layer"
                else:
                    continue
                norms[key] = param.grad.norm().item()
        return norms

    def _print_report(self, report):
        print(f"\n{'='*60}")
        print(f"  Diagnostic Report — Step {report['step']}")
        print(f"{'='*60}")
        print(f"  Target PPL:  {report['target_ppl']:.2f}")
        print(f"  English PPL: {report['english_ppl']:.2f}")

        eh = report['embedding_health']
        print(f"\n  Embedding Health: [{eh['status']}]")
        print(f"    Norm ratio (new/old): {eh['norm_ratio']:.3f}")
        print(f"    Dead embeddings:      {eh['dead_embeddings']}")
        print(f"    Exploding embeddings: {eh['exploding_embeddings']}")

        tu = report['token_utilization']
        print(f"\n  Token Utilization: [{tu['status']}]")
        print(f"    New token usage:  {tu['new_token_usage_ratio']:.1%}")

        if report.get('grad_norms'):
            print(f"\n  Gradient Norms:")
            for component, norm in report['grad_norms'].items():
                print(f"    {component}: {norm:.6f}")

        print(f"{'='*60}\n")
```

## Advanced Techniques

### Curriculum Learning

Order your training data from simple to complex. This isn't just a nice-to-have — it measurably improves convergence speed and final quality, especially for languages that are structurally very different from English.

```python
from datasets import load_dataset, interleave_datasets

def compute_text_complexity(text):
    """
    Heuristic complexity score based on:
    - Average sentence length
    - Vocabulary diversity (type-token ratio)
    - Presence of technical/rare terms
    """
    sentences = text.split('.')
    words = text.split()

    avg_sent_len = len(words) / max(len(sentences), 1)
    type_token_ratio = len(set(words)) / max(len(words), 1)

    # Score: higher = more complex
    score = avg_sent_len * 0.3 + type_token_ratio * 100 * 0.7
    return score

# Sort your dataset by complexity
dataset = load_dataset("text", data_files="data/vi_cleaned.txt", split="train")
dataset = dataset.map(lambda x: {"complexity": compute_text_complexity(x["text"])})
dataset = dataset.sort("complexity")

# Then use the sorted dataset — simple examples first
# Stage 1 (first 30% of training): Simple, clean text
# - Short sentences, common vocabulary
# - Wikipedia articles, children's books, simple news

# Stage 2 (next 40%): Medium complexity
# - Full articles, academic text, literature
# - Parallel bilingual text

# Stage 3 (final 30%): Complex and diverse
# - Technical documents, legal text, poetry
# - Code-mixed text, social media
```

**Why this works:** Simple text has less noise and clearer patterns. The model learns the basic grammar and common vocabulary first, then builds on that foundation to handle complex constructions. It's the same reason human language learning works best with simple input first (Krashen's i+1 hypothesis).

### Language-Specific Considerations

Different language families pose different challenges. Here's what to watch out for:

#### Tonal Languages (Vietnamese, Chinese, Thai, Yoruba)

```python
# CRITICAL: Ensure your tokenizer preserves tone marks as part of the token
# Don't strip diacritics during preprocessing!
# In Vietnamese, "ma" vs "mà" vs "má" vs "mả" vs "mã" vs "mạ" are 6 DIFFERENT WORDS

def verify_diacritics(tokenizer, text="Hôm nay trời đẹp quá"):
    tokens = tokenizer.tokenize(text)
    reconstructed = tokenizer.convert_tokens_to_string(tokens)
    assert reconstructed.strip() == text.strip(), \
        f"Diacritics lost! Original: {text}, Reconstructed: {reconstructed}"
    print("Diacritics preserved correctly")

# Common mistake: using NFKD normalization strips diacritics!
import unicodedata

# BAD: This strips Vietnamese diacritics
bad_text = unicodedata.normalize("NFKD", "Hôm nay trời đẹp quá")

# GOOD: Use NFC normalization (composes characters)
good_text = unicodedata.normalize("NFC", "Hôm nay trời đẹp quá")

# Always normalize to NFC before tokenization
def safe_preprocess(text):
    return unicodedata.normalize("NFC", text)
```

#### Agglutinative Languages (Turkish, Finnish, Hungarian, Japanese)

These languages create long compound words by gluing morphemes together. A single Turkish word can express what English needs an entire sentence for:

```
Turkish: "Avrupalılaştıramadıklarımızdan" (22 characters, 1 word)
English: "Those whom we could not Europeanize" (6 words)
```

For these languages:
- **Add more tokens** (20K-30K) to capture common morpheme combinations
- The tokenizer extension provides the largest efficiency gains
- Consider morpheme-aware tokenization (SentencePiece with unigram model)

#### Right-to-Left Languages (Arabic, Hebrew, Farsi, Urdu)

```python
# Most tokenizers handle RTL correctly at the byte level,
# but VERIFY that your data pipeline doesn't reverse the text

def verify_rtl_handling(tokenizer, text="مرحبا بالعالم"):
    """Verify RTL text is handled correctly."""
    encoded = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(encoded)

    # The decoded text should match the original
    # (visual rendering might differ, but bytes should be identical)
    assert decoded.strip() == text.strip(), \
        f"RTL handling broken! Got: {decoded}"

    # Also check that mixed RTL/LTR text works
    mixed = "The word مرحبا means hello"
    mixed_encoded = tokenizer.encode(mixed, add_special_tokens=False)
    mixed_decoded = tokenizer.decode(mixed_encoded)
    assert mixed_decoded.strip() == mixed.strip(), \
        f"Mixed RTL/LTR handling broken! Got: {mixed_decoded}"
    print("RTL handling: OK")
```

#### Languages Without Spaces (Chinese, Japanese, Thai, Lao)

These languages don't use spaces to separate words, which fundamentally changes tokenization dynamics:

```python
# For Chinese/Japanese: the tokenizer needs to learn word boundaries
# Pre-segmentation can help

# For Thai: use PyThaiNLP for word segmentation before tokenization
# pip install pythainlp
from pythainlp.tokenize import word_tokenize

thai_text = "วันนี้อากาศดีมาก"
segmented = " ".join(word_tokenize(thai_text, engine="newmm"))
# Output: "วันนี้ อากาศ ดี มาก"
# Now the tokenizer can learn better word-level tokens

# For Chinese: use jieba segmentation
# pip install jieba
import jieba

chinese_text = "今天天气很好"
segmented = " ".join(jieba.cut(chinese_text))
# Output: "今天 天气 很好"
```

### DPO for Target Language Alignment

After SFT, you can further align the model using Direct Preference Optimization (DPO) with target-language preference data. This is especially powerful for fixing language confusion and cultural appropriateness:

```python
from trl import DPOTrainer, DPOConfig

# Preference data format:
# {"prompt": "...", "chosen": "better response", "rejected": "worse response"}

# Key insight: for language adaptation, your preference pairs should cover:
# 1. Language purity: monolingual (chosen) vs code-switched (rejected)
# 2. Cultural fit: culturally appropriate (chosen) vs culturally off (rejected)
# 3. Naturalness: native-sounding (chosen) vs translated-sounding (rejected)

dpo_config = DPOConfig(
    output_dir="output/vi-llama-3-8b-dpo",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,        # Very low LR for DPO
    beta=0.1,                  # KL penalty coefficient
    bf16=True,
    max_length=2048,
    max_prompt_length=1024,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,            # Will use implicit reference
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

dpo_trainer.train()
```

**Generating preference pairs for language adaptation:**

```python
def create_language_preference_pair(model, tokenizer, prompt):
    """
    Generate chosen/rejected pairs by sampling multiple responses
    and scoring for language purity and naturalness.
    """
    import fasttext
    lang_model = fasttext.load_model("lid.176.bin")

    responses = []
    for _ in range(8):  # Generate 8 candidates
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.9,
            top_p=0.95,
            do_sample=True,
        )
        response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:],
                                   skip_special_tokens=True)

        # Score each response
        pred = lang_model.predict(response.replace('\n', ' ')[:500])
        lang_confidence = pred[1][0] if pred[0][0] == "__label__vi" else 0

        responses.append({
            "text": response,
            "lang_confidence": lang_confidence,
            "length": len(response),
        })

    # Best response: highest language confidence + reasonable length
    responses.sort(key=lambda x: x["lang_confidence"], reverse=True)
    chosen = responses[0]["text"]
    rejected = responses[-1]["text"]

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }
```

### Elastic Weight Consolidation (EWC): Anti-Forgetting Regularization

For teams that want maximum English retention, EWC adds a regularization term that penalizes large changes to parameters that are important for English:

```python
import torch
from copy import deepcopy

class EWCRegularizer:
    """
    Elastic Weight Consolidation prevents catastrophic forgetting by
    penalizing changes to parameters that are important for existing tasks.

    The idea: compute the Fisher Information Matrix on English data,
    which tells us which parameters are most important for English.
    Then add a penalty term that discourages changing those parameters.
    """

    def __init__(self, model, english_dataset, tokenizer, lambda_ewc=1000):
        self.lambda_ewc = lambda_ewc
        self.original_params = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        # Compute Fisher Information on English data
        self.fisher = self._compute_fisher(model, english_dataset, tokenizer)

    def _compute_fisher(self, model, dataset, tokenizer, num_samples=1000):
        """Compute diagonal Fisher Information Matrix."""
        fisher = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        model.eval()
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break

            inputs = tokenizer(example["text"], return_tensors="pt",
                             truncation=True, max_length=512).to(model.device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            outputs.loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2

            model.zero_grad()

        # Average
        for n in fisher:
            fisher[n] /= num_samples

        return fisher

    def penalty(self, model):
        """Compute the EWC penalty term to add to the training loss."""
        penalty = 0
        for n, p in model.named_parameters():
            if n in self.fisher:
                penalty += (self.fisher[n] *
                          (p - self.original_params[n]) ** 2).sum()

        return self.lambda_ewc * penalty

# Usage in a custom training loop:
# total_loss = language_loss + ewc.penalty(model)
```

**When to use EWC:** When your English retention requirement is strict (< 2% degradation) and you can afford the extra computation for Fisher matrix estimation. In practice, proper data mixing (15-20% English) usually gives you < 5% degradation without EWC.

## Real-World Cost Estimates

Let's make this concrete. Here's what adapting LLaMA 3 8B to Vietnamese costs in practice:

| Phase | GPU Hours (A100 80GB) | Cloud Cost (est.) | Duration |
|-------|--------------------|-------------------|----------|
| Tokenizer training | 0 (CPU) | ~$0 | 1-2 hours |
| Data preparation | 0 (CPU) | ~$50 (storage) | 2-5 days |
| Continual pretraining (20B tokens) | 400-600 hrs | $800-$1,200 | 3-5 days on 8×A100 |
| Instruction tuning (50K examples) | 10-20 hrs | $20-$40 | 2-4 hours on 4×A100 |
| DPO (optional, 10K pairs) | 5-10 hrs | $10-$20 | 1-2 hours on 4×A100 |
| Evaluation | 5-10 hrs | $10-$20 | 1-2 hours |
| **Total** | **~500-700 hrs** | **~$1,000-$1,400** | **~1-2 weeks** |

For a 70B model, multiply compute costs by roughly 10x. This is still orders of magnitude cheaper than training from scratch.

## End-to-End Checklist

Here's the complete checklist for a language adaptation project:

**Preparation:**
- [ ] Collect and clean 10-50B tokens of target-language text
- [ ] Prepare 1-5B tokens of English text for mix
- [ ] Collect parallel/bilingual text if available
- [ ] Set up evaluation benchmarks in target language

**Tokenizer:**
- [ ] Train target-language BPE tokenizer
- [ ] Merge with original vocabulary (add 10K-20K tokens)
- [ ] Verify encoding efficiency improvement (target: 1.5-2.5x)
- [ ] Verify diacritics/special characters are preserved

**Embedding Initialization:**
- [ ] Initialize new token embeddings (mean of subwords)
- [ ] Verify both input embeddings and output head are initialized

**Continual Pretraining:**
- [ ] Set up data mix (70% target, 15% English, 10% parallel, 5% code)
- [ ] Configure staged training (embedding warmup → full training)
- [ ] Set learning rate to 2e-5 with cosine schedule
- [ ] Monitor target PPL, English PPL, and training loss
- [ ] Save checkpoints every 500-1000 steps
- [ ] Stop when target PPL plateaus

**Instruction Tuning:**
- [ ] Prepare 10K-100K instruction examples in target language
- [ ] Mix translated + natively generated instructions
- [ ] Fine-tune for 2-3 epochs with loss masking on prompts
- [ ] Evaluate instruction following quality

**Evaluation:**
- [ ] Target-language perplexity (< 15 for a well-adapted model)
- [ ] Target-language task benchmarks (QA, summarization, etc.)
- [ ] English benchmark retention (< 5% degradation)
- [ ] Human evaluation by native speakers
- [ ] Tokenization efficiency verification

## Lessons From Real Projects

Here are condensed lessons from multiple language adaptation projects that don't fit neatly into the sections above:

### Lesson 1: Your First Adaptation Will Take 3x Longer Than Expected

Budget 60-70% of your time for data preparation and debugging, not training. The actual GPU time is the easy part. Finding good data, cleaning it, debugging tokenizer issues, and iterating on data mixes is where the real work happens.

**Realistic timeline for a first project (8B model):**

| Phase | Expected | Actual |
|-------|----------|--------|
| Data collection | 3 days | 7-10 days |
| Data cleaning + dedup | 2 days | 5-7 days |
| Tokenizer extension | 1 day | 2-3 days (debugging edge cases) |
| First training run | 3 days | 3 days (but it will fail) |
| Debugging + 2nd run | 0 days (optimistic) | 5-7 days |
| SFT | 1 day | 2-3 days |
| Evaluation | 1 day | 3-5 days (building benchmarks is work) |
| **Total** | **~11 days** | **~4-5 weeks** |

### Lesson 2: Small Test Runs Save Enormous Amounts of Money

Before running your full 20B-token training on 8×A100s, **always** do a miniature version:

```python
# Mini test run checklist:
# 1. Take 0.1% of your data (~20M tokens)
# 2. Train for 500 steps on a single GPU
# 3. Check:
#    - Does the loss decrease smoothly?
#    - Does the tokenizer work correctly on all evaluation examples?
#    - Do the new token embeddings change (not stuck at initialization)?
#    - Can the model generate coherent text after 500 steps?
# 4. If ANY of these fail, fix the issue before scaling up

# This costs ~$5 and 2 hours instead of $1000 and 5 days
```

### Lesson 3: The Model Picks Up Bad Patterns Faster Than Good Ones

If your data contains even 5% garbage (wrong language, spam, templated text), the model will learn those patterns disproportionately fast. This is because:

- Repetitive patterns have high frequency → strong gradient signal
- Spam/template text has low perplexity (it's predictable) → the model "likes" it
- The model treats data quality uniformly — it can't tell spam from Shakespeare

**Always inspect random samples from your training data:**

```python
import random

def audit_data_quality(data_file, num_samples=50):
    """Manually inspect random training samples."""
    with open(data_file) as f:
        lines = f.readlines()

    samples = random.sample(lines, min(num_samples, len(lines)))

    print("Data Quality Audit")
    print("Rate each sample: [G]ood / [B]ad / [S]kip")
    print("="*60)

    good, bad = 0, 0
    for i, sample in enumerate(samples):
        print(f"\n--- Sample {i+1}/{num_samples} ---")
        print(sample[:500])
        rating = input("\nRating [G/B/S]: ").strip().upper()
        if rating == "G":
            good += 1
        elif rating == "B":
            bad += 1

    quality_ratio = good / (good + bad) if (good + bad) > 0 else 0
    print(f"\nQuality score: {quality_ratio:.0%} ({good} good, {bad} bad)")

    if quality_ratio < 0.9:
        print("WARNING: Data quality is below 90%. Improve cleaning pipeline.")
```

### Lesson 4: Evaluation Is Harder Than Training

Building proper evaluation for a low-resource language is genuinely difficult. There often aren't existing benchmarks. Machine-translated benchmarks don't test the right things. Human evaluation is expensive and slow.

**Practical approach:**

1. **Build a small, high-quality eval set manually** (100-200 examples) — this is worth the effort
2. **Use cross-lingual benchmarks** where available (XNLI, XQuAD, MLQA)
3. **Create a "vibe check" eval** — 20 prompts that you, as a native speaker, evaluate qualitatively each day during training
4. **Track multiple metrics** — no single number tells the full story

### Lesson 5: The Optimal Checkpoint Is Rarely the Last One

Don't just use the final checkpoint. Often the best model is from 70-85% of the way through training. After that point, the model continues to improve on target-language perplexity but starts overfitting to the training distribution and losing generalization.

```python
# Save checkpoints every 500 steps and evaluate each one
# Plot: target_ppl, english_ppl, and downstream_task_score vs step
# The optimal checkpoint is where downstream_task_score peaks,
# which is usually BEFORE perplexity reaches its minimum

# This is because perplexity can keep improving by memorizing
# training data, while task performance requires generalization
```

## Conclusion

Adapting an LLM to a new language is one of the highest-ROI projects in applied NLP today. But it's also one of the most deceptively complex — the pipeline looks simple on paper, but the devil is in a hundred details that only surface during real training runs.

The key ingredients, ranked by impact:

1. **Data quality and diversity** — this matters more than anything else. Bad data in, bad model out, regardless of how sophisticated your training setup is.
2. **Tokenizer extension** — the most impactful single optimization. A 2x tokenization efficiency gain means 2x longer context, 2x faster inference, and meaningfully better quality.
3. **Learning rate and data mix** — the delicate balance between learning the new language and retaining existing capabilities. Too aggressive and you destroy the model; too conservative and you waste compute.
4. **Monitoring and debugging** — things will go wrong. The teams that succeed are the ones that catch problems early and iterate fast.

The techniques here work for any language — I've used Vietnamese as the running example, but the same pipeline applies to Thai, Arabic, Swahili, Tagalog, or any language where you have 10B+ tokens of clean text.

If you're starting your first language adaptation project, here's the minimum viable approach: extend the tokenizer with 10K tokens, initialize embeddings with mean-of-subwords, continual pretrain on 10B tokens (70% target language, 15% English, 10% parallel, 5% code) with learning rate 2e-5, then instruction tune on 20K-50K examples. This will get you 80% of the way there. The remaining 20% is where all the nuance in this article comes in.

## References

- Cui et al., "Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca" (2023) — Pioneering work on tokenizer extension for Chinese
- Zhao et al., "LLaMA Beyond English: An Empirical Study on Language Capability Transfer" (2024) — Comprehensive study on language adaptation strategies
- Csaki et al., "Sambalingo: Teaching Large Language Models New Languages" (2024) — Systematic evaluation of continual pretraining approaches
- Fujii et al., "Continual Pre-Training for Cross-Lingual LLM Adaptation" (2024) — Analysis of data mixing strategies
- Touvron et al., "LLaMA 2: Open Foundation and Fine-Tuned Chat Models" (2023) — Base model architecture
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021) — Parameter-efficient fine-tuning
- Kirkpatrick et al., "Overcoming Catastrophic Forgetting in Neural Networks" (2017) — The EWC method for preventing forgetting
- Lin et al., "Mala-500: Massive Language Adaptation of Large Language Models" (2024) — Adaptation to 534 languages simultaneously
- Nguyen et al., "SeaLLMs: Large Language Models for Southeast Asia" (2024) — Language adaptation for Vietnamese, Thai, Indonesian and others
- Lee et al., "Deduplicating Training Data Makes Language Models Better" (2022) — The impact of data deduplication on model quality
- Wendler et al., "Do Llamas Work in English? On the Latent Language of Multilingual Transformers" (2024) — How internal representations work across languages
- Conneau et al., "Unsupervised Cross-lingual Representation Learning at Scale" (2020) — XLM-R and cross-lingual transfer theory
