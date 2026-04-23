---
title: "BPE Tokenizer: The Foundation of Modern Language Models"
publishDate: "2026-01-15"
category: "machine-learning"
subcategory: "Large Language Model"
tags: ["nlp", "tokenization", "bpe", "llm", "gpt", "transformers", "deep-learning"]
date: "2026-01-15"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "A comprehensive guide to Byte Pair Encoding (BPE) tokenization — how GPT, Llama, and other LLMs break down text into tokens, why it matters for model behavior, and how to build your own tokenizer from scratch."
---

## Introduction

Every time you type a prompt into ChatGPT, Claude, or any LLM, your text goes through a critical transformation **before** the model ever sees it: **tokenization**. The model doesn't read characters or words — it reads **tokens**, which are the fundamental units of its vocabulary.

Byte Pair Encoding (BPE) is the algorithm behind this transformation, and it has become the de facto standard for modern Large Language Models. Understanding BPE is not just academic curiosity — it directly affects:

- **Why LLMs struggle with counting letters** (e.g., "How many r's in strawberry?")
- **Why the same model is better at English than Vietnamese or Thai**
- **Why your API costs vary** for the same-length text in different languages
- **Why some code completions break mid-variable-name**
- **Why models sometimes produce "glitch tokens"** — tokens that cause bizarre behavior

Tokenization is the lens through which your model sees the world. A bad tokenizer is like giving someone blurry glasses — no matter how smart they are, their performance will suffer.

## What is Tokenization?

Tokenization is the process of breaking down text into smaller units (tokens) that a model can process. Think of it like breaking a sentence into puzzle pieces — the question is **how big should each piece be?**

Consider the sentence: `"unhappiness"`

There are three fundamentally different ways to break it apart:

```
Character-level:  ['u', 'n', 'h', 'a', 'p', 'p', 'i', 'n', 'e', 's', 's']  → 11 tokens
Subword-level:    ['un', 'happiness']  or  ['un', 'happi', 'ness']           → 2-3 tokens
Word-level:       ['unhappiness']                                             → 1 token
```

Each approach has a fundamental trade-off between **vocabulary size** and **sequence length**. BPE operates at the **subword level**, finding the sweet spot.

## Why BPE? The Goldilocks Problem

![Tokenization tradeoffs: character-level (too many tokens, no semantics) vs word-level (huge vocab, OOV) — BPE subword lands in between with fixed vocab, no OOV, and compositional rare words](/imgs/blogs/bpe-01-why.png)

### The Spectrum of Tokenization

Imagine you're designing a writing system for a new language. You have two extreme choices:

**Option A: One symbol per concept (Word-level)**
Like Chinese characters — each word/concept gets its own symbol.
```
Vocabulary: ["hello", "world", "cat", "dog", "unhappiness", "antidisestablishmentarianism", ...]
```
- You need millions of symbols to cover everything
- What happens when someone invents a new word? You can't write it!
- Every typo creates an "unknown" word

**Option B: One symbol per sound (Character-level)**
Like the alphabet — only ~26 letters needed.
```
Vocabulary: ['a', 'b', 'c', ..., 'z', ' ', '.', ...]
```
- Only ~100 symbols needed, but...
- "hello" becomes 5 separate tokens — the model has to learn that h+e+l+l+o means "hello"
- Sequences become extremely long (expensive for attention, which is $O(n^2)$)

**Option C: Frequent character combinations (BPE — Subword-level)**
Like learning common syllables and morphemes.
```
Vocabulary: ['a', ..., 'z', 'th', 'ing', 'tion', 'the', 'hello', 'un', 'happiness', ...]
```
- Common words ("the", "hello") are single tokens — efficient!
- Rare words are split into known pieces ("un" + "happiness") — no unknowns!
- Vocabulary is manageable (30k-100k tokens)

### A Concrete Comparison

Let's see how each approach handles real text:

```
Text: "The unhappiest transformer models don't tokenize well."

Word-level tokenizer (vocab=100k):
  → ["The", "unhappiest", "transformer", "models", "don't", "tokenize", "well", "."]
  → 8 tokens
  → Problem: "unhappiest" might not be in vocabulary → <UNK>!

Character-level tokenizer (vocab=100):
  → ['T','h','e',' ','u','n','h','a','p','p','i','e','s','t',' ','t','r','a',...]
  → 52 tokens
  → Problem: 6.5x more tokens = 6.5x more compute in attention layers

BPE tokenizer (vocab=50k, e.g., GPT-2):
  → ["The", " un", "happ", "iest", " transformer", " models", " don", "'t",
     " token", "ize", " well", "."]
  → 12 tokens
  → "unhappiest" decomposed into meaningful pieces: un + happi + est
  → No unknowns, reasonable sequence length
```

### Why Subword Decomposition Is Powerful

BPE's subword splits aren't random — they capture **morphological structure**:

```
"unhappiness"  →  "un" + "happiness"
                    ↑        ↑
                  prefix   root word

"playing"      →  "play" + "ing"
                    ↑        ↑
                  root    suffix

"internationalization" → "inter" + "national" + "ization"
                           ↑          ↑            ↑
                        prefix      root        suffix
```

The model effectively learns that "un-" means negation, "-ing" means ongoing action, etc. This compositional understanding means BPE lets the model generalize: if it knows "happy" and "un-", it can understand "unhappy" even if that exact combination was rare in training.

## How BPE Works

![BPE training loop: pre-tokenize, initialize vocab with bytes/chars, count adjacent pair frequencies, pick most frequent pair, add merge rule — repeat until vocab size reached](/imgs/blogs/bpe-02-training.png)

### The Core Idea

BPE was originally a **data compression** algorithm (Gage, 1994). The idea is beautifully simple: **repeatedly replace the most frequent pair of adjacent symbols with a new symbol.** When applied to text, this naturally discovers common subwords.

Think of it like a child learning to read:
1. First, they learn individual letters: a, b, c, ...
2. Then they notice "th" appears together constantly, so they start reading it as one unit
3. Then "the" becomes a single chunk
4. Then common words like "the", "and", "is" become instant recognition
5. New words can still be sounded out letter-by-letter

BPE automates exactly this process, driven by frequency statistics from a training corpus.

### Step-by-Step Process: A Complete Worked Example

Let's trace through the full algorithm with a small corpus. We'll use word frequencies to make it realistic:

#### Step 1: Initialize with Characters

Start by splitting every word into individual characters. We add a special end-of-word marker `</w>` to distinguish word-final characters from word-internal ones (so "est" at the end of "widest" is different from "est" in "estimate"):

```python
# Training corpus with word frequencies
corpus = {
    "low":    5,    # appears 5 times
    "lower":  2,    # appears 2 times
    "newest": 6,    # appears 6 times
    "widest": 3,    # appears 3 times
    "new":    4,    # appears 4 times
}

# Split every word into characters + end marker
# Format: (character sequence, frequency)
tokenized_corpus = {
    ('l', 'o', 'w', '</w>'):           5,
    ('l', 'o', 'w', 'e', 'r', '</w>'): 2,
    ('n', 'e', 'w', 'e', 's', 't', '</w>'): 6,
    ('w', 'i', 'd', 'e', 's', 't', '</w>'): 3,
    ('n', 'e', 'w', '</w>'):           4,
}

# Initial vocabulary: all unique characters + end marker
vocabulary = {'l', 'o', 'w', 'e', 'r', 'n', 's', 't', 'i', 'd', '</w>'}
# Size: 11
```

#### Step 2: Count All Adjacent Pairs

Scan through every word and count how often each pair of adjacent symbols appears:

```python
# Count pairs (weighted by word frequency)
pairs = {
    ('l', 'o'):     5 + 2 = 7,       # "low" (5) + "lower" (2)
    ('o', 'w'):     5 + 2 = 7,       # "low" (5) + "lower" (2)
    ('w', '</w>'):  5 + 4 = 9,       # "low" (5) + "new" (4)
    ('w', 'e'):     2 + 6 = 8,       # "lower" (2) + "newest" (6)
    ('e', 'r'):     2,               # "lower" (2)
    ('r', '</w>'):  2,               # "lower" (2)
    ('n', 'e'):     6 + 4 = 10,      # "newest" (6) + "new" (4)
    ('e', 'w'):     6 + 4 = 10,      # "newest" (6) + "new" (4)
    ('e', 's'):     6 + 3 = 9,       # "newest" (6) + "widest" (3)
    ('s', 't'):     6 + 3 = 9,       # "newest" (6) + "widest" (3)
    ('t', '</w>'):  6 + 3 = 9,       # "newest" (6) + "widest" (3)
    ('w', 'i'):     3,               # "widest" (3)
    ('i', 'd'):     3,               # "widest" (3)
    ('d', 'e'):     3,               # "widest" (3)
}
```

#### Step 3: Merge the Most Frequent Pair

The most frequent pairs are `('n', 'e')` and `('e', 'w')` both with count 10. Let's pick `('e', 's')` ... actually let's pick `('n', 'e')` with count 10:

```python
# MERGE #1: 'n' + 'e' → 'ne' (count: 10)
tokenized_corpus = {
    ('l', 'o', 'w', '</w>'):             5,
    ('l', 'o', 'w', 'e', 'r', '</w>'):   2,
    ('ne', 'w', 'e', 's', 't', '</w>'):  6,  # ← 'n'+'e' merged
    ('w', 'i', 'd', 'e', 's', 't', '</w>'): 3,
    ('ne', 'w', '</w>'):                  4,  # ← 'n'+'e' merged
}

vocabulary = {..., 'ne'}  # Add new token
# Merge rule saved: ('n', 'e') → 'ne'
```

#### Step 4: Repeat — Continue Merging

```python
# MERGE #2: 'ne' + 'w' → 'new' (count: 6+4=10)
tokenized_corpus = {
    ('l', 'o', 'w', '</w>'):              5,
    ('l', 'o', 'w', 'e', 'r', '</w>'):    2,
    ('new', 'e', 's', 't', '</w>'):       6,  # ← merged
    ('w', 'i', 'd', 'e', 's', 't', '</w>'): 3,
    ('new', '</w>'):                       4,  # ← merged
}

# MERGE #3: 'e' + 's' → 'es' (count: 6+3=9)
tokenized_corpus = {
    ('l', 'o', 'w', '</w>'):              5,
    ('l', 'o', 'w', 'e', 'r', '</w>'):    2,
    ('new', 'es', 't', '</w>'):           6,  # ← merged
    ('w', 'i', 'd', 'es', 't', '</w>'):   3,  # ← merged
    ('new', '</w>'):                       4,
}

# MERGE #4: 'es' + 't' → 'est' (count: 6+3=9)
tokenized_corpus = {
    ('l', 'o', 'w', '</w>'):              5,
    ('l', 'o', 'w', 'e', 'r', '</w>'):    2,
    ('new', 'est', '</w>'):               6,  # ← merged
    ('w', 'i', 'd', 'est', '</w>'):       3,  # ← merged
    ('new', '</w>'):                       4,
}

# MERGE #5: 'est' + '</w>' → 'est</w>' (count: 6+3=9)
# MERGE #6: 'l' + 'o' → 'lo' (count: 5+2=7)
# MERGE #7: 'lo' + 'w' → 'low' (count: 5+2=7)
# MERGE #8: 'low' + '</w>' → 'low</w>' (count: 5)
# ... and so on
```

After 8 merges, our vocabulary has grown from 11 base characters to 19 tokens, and the corpus is much more compressed:

```python
# Final state after 8 merges
tokenized_corpus = {
    ('low</w>',):              5,   # single token!
    ('low', 'e', 'r', '</w>'): 2,
    ('new', 'est</w>'):        6,   # just 2 tokens!
    ('w', 'i', 'd', 'est</w>'): 3,
    ('new', '</w>'):           4,
}

# Ordered merge rules (THIS is the tokenizer):
merge_rules = [
    ('n', 'e')   → 'ne',
    ('ne', 'w')  → 'new',
    ('e', 's')   → 'es',
    ('es', 't')  → 'est',
    ('est', '</w>') → 'est</w>',
    ('l', 'o')   → 'lo',
    ('lo', 'w')  → 'low',
    ('low', '</w>') → 'low</w>',
]
```

### How Tokenization (Encoding) Works

Once we have the merge rules, tokenizing new text is straightforward — apply the merge rules **in the same order they were learned**:

```
Tokenize "lowest":

Step 0 (characters):  ['l', 'o', 'w', 'e', 's', 't', '</w>']
Apply merge #1 (n+e→ne):  no 'n' found, skip
Apply merge #2 (ne+w→new): no 'ne' found, skip
Apply merge #3 (e+s→es):  ['l', 'o', 'w', 'es', 't', '</w>']
Apply merge #4 (es+t→est): ['l', 'o', 'w', 'est', '</w>']
Apply merge #5 (est+</w>→est</w>): ['l', 'o', 'w', 'est</w>']
Apply merge #6 (l+o→lo):  ['lo', 'w', 'est</w>']
Apply merge #7 (lo+w→low): ['low', 'est</w>']
Apply merge #8 (low+</w>→low</w>): no match (not word-final 'low')

Final: ['low', 'est</w>']  → 2 tokens!
```

Notice: "lowest" was **never in our training corpus**, but the tokenizer can still handle it by composing known pieces. This is the power of subword tokenization.

```
Tokenize "newer":

Step 0: ['n', 'e', 'w', 'e', 'r', '</w>']
Apply merge #1 (n+e→ne): ['ne', 'w', 'e', 'r', '</w>']
Apply merge #2 (ne+w→new): ['new', 'e', 'r', '</w>']
... remaining merges don't apply ...

Final: ['new', 'e', 'r', '</w>']  → 4 tokens

The model sees: "new" (known subword) + "e" + "r" + end
It can infer "newer" = "new" + comparative suffix
```

### The Merge Tree: Visualizing BPE

The merge process forms a tree structure, where leaves are characters and internal nodes are merged tokens:

```
            'newest'
           /        \
        'new'      'est</w>'
       /    \      /      \
     'ne'   'w'  'est'   '</w>'
    /    \       /    \
  'n'   'e'   'es'   't'
              /    \
            'e'   's'
```

This tree tells you: to decode "newest", you recursively expand: newest → new + est</w> → ne + w + es + t + </w> → n + e + w + e + s + t + </w>.

## Implementation from Scratch

### Basic BPE Implementation

```python
from collections import Counter, defaultdict
import re

class SimpleBPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}

    def get_pairs(self, word):
        """Get all adjacent pairs in a word"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def train(self, corpus):
        """Train BPE on a corpus"""
        # Initialize with character-level tokens
        vocab = defaultdict(int)
        for word in corpus:
            vocab[' '.join(list(word)) + ' </w>'] += 1

        # Perform merges
        for i in range(self.vocab_size):
            # Count all pairs
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                symbols = word.split()
                for j in range(len(symbols) - 1):
                    pairs[(symbols[j], symbols[j + 1])] += freq

            if not pairs:
                break

            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)

            # Merge the pair
            self.merges[best_pair] = ''.join(best_pair)

            # Update vocabulary
            new_vocab = {}
            bigram = ' '.join(best_pair)
            replacement = ''.join(best_pair)

            for word in vocab:
                new_word = word.replace(bigram, replacement)
                new_vocab[new_word] = vocab[word]

            vocab = new_vocab

        self.vocab = vocab

    def tokenize(self, text):
        """Tokenize text using learned BPE"""
        words = text.split()
        tokens = []

        for word in words:
            word = ' '.join(list(word)) + ' </w>'

            # Apply merges
            for pair in self.merges:
                bigram = ' '.join(pair)
                replacement = self.merges[pair]
                word = word.replace(bigram, replacement)

            tokens.extend(word.split())

        return tokens

# Example usage
corpus = ["low", "lower", "newest", "widest", "lowest"] * 100
tokenizer = SimpleBPETokenizer(vocab_size=20)
tokenizer.train(corpus)

text = "lowest newer"
tokens = tokenizer.tokenize(text)
print(f"Tokens: {tokens}")
```

## Byte-Level BPE: How Modern LLMs Actually Work

The basic BPE we described above works on **Unicode characters**. But modern LLMs (GPT-2, GPT-3, GPT-4, Llama 3) use a variant called **Byte-Level BPE (BBPE)**, which works on **raw bytes** instead.

### The Problem with Character-Level BPE

Standard BPE starts with a vocabulary of all unique Unicode characters in the training data. But Unicode has over 150,000 characters across all scripts. If your training data includes Chinese, Japanese, Korean, Arabic, emoji, and mathematical symbols, your **base vocabulary** before any merges is already enormous.

Worse, any Unicode character not seen during training becomes unknown — the tokenizer literally cannot represent it.

### The Solution: Start from Bytes

Every piece of text, in any language or script, can be represented as a sequence of **bytes** (values 0-255). Byte-Level BPE starts with a base vocabulary of just **256 byte tokens** and builds everything from there:

```
Standard BPE base vocabulary:
  'a', 'b', 'c', ..., 'z', 'A', ..., 'Z', '0', ..., '9',
  'à', 'é', '中', '日', '🎉', ...
  → Potentially thousands of base tokens
  → Cannot handle unseen characters

Byte-Level BPE base vocabulary:
  0x00, 0x01, 0x02, ..., 0xFF
  → Exactly 256 base tokens
  → Can represent ANY text (including future emoji, new scripts, binary data)
```

### How It Works in Practice

```
Text: "café"

UTF-8 bytes: [99, 97, 102, 195, 169]
              'c' 'a' 'f'   'é' (2 bytes in UTF-8)

Byte-level BPE tokenization:
  The tokenizer sees bytes, not characters
  Through merges, it has learned that bytes [195, 169] → 'é'
  And 'caf' + 'é' → 'café' (if frequent enough to be a single token)
  → Result: ["café"] (1 token)

For a rare word in Thai: "สวัสดี"
  UTF-8 bytes: [224,184,170, 224,184,167, 224,184,177, ...]
  Even though this word might be rare, the tokenizer can always
  represent it as individual bytes or byte-pair merges
  → Never produces <UNK>!
```

### GPT-2's Clever Trick: Byte-to-Unicode Mapping

GPT-2 uses an additional trick: it maps the 256 byte values to **printable Unicode characters**, so the tokenizer's internal representation is human-readable:

```python
# GPT-2 maps bytes to printable characters
# Byte 0x20 (space) → 'Ġ' (the Ġ you see in GPT-2 tokens)
# Byte 0x0A (newline) → 'Ċ'

# That's why GPT-2 tokens look like:
# " hello" → "Ġhello"   (Ġ = space byte)
# " The"   → "ĠThe"
# "\n"     → "Ċ"
```

This is purely cosmetic — it makes the tokens printable for debugging. The actual model just sees integer IDs.

### Fertility: How Tokenization Affects Different Languages

**Fertility** measures how many tokens are needed to represent the same content in different languages. This has real consequences:

```
English:  "Hello, how are you?"
GPT-4:    ["Hello", ",", " how", " are", " you", "?"]  → 6 tokens

Japanese: "こんにちは、お元気ですか？"
GPT-4:    ["こんにち", "は", "、", "お", "元", "気", "です", "か", "？"]  → 9 tokens

Thai:     "สวัสดีคุณเป็นอย่างไร"
GPT-4:    [many small byte-level pieces]  → 15+ tokens

Same meaning, but Thai uses 2.5x more tokens than English!
```

Why? Because BPE's merge rules are learned from data, and most training corpora are **English-heavy**. English text gets more merges → more efficient tokens. Low-resource languages get fewer merges → more bytes per concept.

**Real-world impact:**
- API costs scale with token count → same query costs 2-3x more in some languages
- Context window is measured in tokens → less "room" for non-English content
- Model may be less capable in under-tokenized languages (fewer semantic tokens per sequence)

This is why newer models like Llama 3 significantly **increased vocabulary size** (32K → 128K tokens) — larger vocabularies allow more merges for more languages, improving efficiency and fairness across languages.

## Modern BPE Variants

![Tokenizer family: byte-level BPE (GPT-2/Llama/Mistral) handles any byte with no OOV; SentencePiece Unigram (T5) trains by likelihood; WordPiece (BERT) marks continuations with ##](/imgs/blogs/bpe-03-variants.png)

### 1. BPE (Original - GPT-2, GPT-3, GPT-4)

Used by OpenAI's GPT models. GPT-2/3 use byte-level BPE with 50,257 tokens. GPT-4 uses a newer tokenizer (`cl100k_base`) with 100,256 tokens for better multilingual support.

```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "Hello, how are you doing today?"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print(f"Tokens: {tokens}")
# Output: ['Hello', ',', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing', 'Ġtoday', '?']
# Note: 'Ġ' represents a leading space — this is GPT-2's byte-to-unicode mapping

print(f"Token IDs: {token_ids}")
# Output: [15496, 11, 703, 389, 345, 1804, 1909, 30]
```

Key difference from basic BPE: GPT-2 adds a **pre-tokenization** step that splits on whitespace and punctuation **before** applying BPE merges. This prevents merges across word boundaries (e.g., "the" in "other" won't merge with standalone "the").

```python
# GPT-2's pre-tokenization regex pattern:
import re
pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+"""

text = "I wouldn't do that!"
pre_tokens = re.findall(pattern, text)
# ['I', ' wouldn', "'t", ' do', ' that', '!']
# Note: "wouldn't" → "wouldn" + "'t" (contractions split correctly)
```

### 2. WordPiece (BERT, DistilBERT)

Similar to BPE but with a key difference in how it selects which pair to merge. Instead of pure frequency, WordPiece selects the pair that **maximizes the likelihood of the training data** when merged:

$$\text{score}(x, y) = \frac{\text{freq}(xy)}{\text{freq}(x) \times \text{freq}(y)}$$

This means it prefers merging pairs where the combination is much more common than you'd expect from the individual frequencies.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "Hello, how are you doing today?"
tokens = tokenizer.tokenize(text)

print(f"Tokens: {tokens}")
# Output: ['hello', ',', 'how', 'are', 'you', 'doing', 'today', '?']

# WordPiece uses ## prefix for continuation tokens:
tokens = tokenizer.tokenize("unbelievable")
print(f"Tokens: {tokens}")
# Output: ['un', '##bel', '##ie', '##va', '##ble']
#          ↑ word start   ↑ continuation (##)
```

The `##` prefix is how WordPiece marks "this is a continuation of the previous token, not a word start." This is WordPiece's equivalent of BPE's space handling.

### 3. Unigram (T5, ALBERT, mBART)

Fundamentally different approach: instead of starting small and merging (bottom-up), Unigram starts with a **large vocabulary** and **prunes it down** (top-down).

The algorithm:
1. Start with a huge vocabulary (e.g., all substrings up to length 20)
2. Compute the probability of each token using a unigram language model
3. For each token, compute how much the total corpus likelihood drops if that token is removed
4. Remove tokens whose removal hurts least
5. Repeat until reaching target vocabulary size

```python
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")

text = "Hello, how are you doing today?"
tokens = tokenizer.tokenize(text)

print(f"Tokens: {tokens}")
# Output: ['▁Hello', ',', '▁how', '▁are', '▁you', '▁doing', '▁today', '?']
# Note: '▁' (U+2581) represents a space — SentencePiece convention
```

Advantage over BPE: Unigram can consider **multiple possible tokenizations** of the same text and pick the most probable one. BPE is deterministic (always applies merges in the learned order), while Unigram can use probabilistic sampling during training for regularization.

### 4. SentencePiece (Llama 1/2, Gemma, mT5)

SentencePiece isn't a tokenization algorithm itself — it's a **framework** that can run either BPE or Unigram. Its key innovation is treating text as a **raw byte/character stream without pre-tokenization**:

```python
import sentencepiece as spm

# Train SentencePiece model
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='bpe_model',
    vocab_size=8000,
    model_type='bpe'    # or 'unigram'
)

# Load and use
sp = spm.SentencePieceProcessor()
sp.load('bpe_model.model')

text = "Hello, how are you doing today?"
tokens = sp.encode_as_pieces(text)
print(f"Tokens: {tokens}")
```

Key difference: most other tokenizers (GPT-2, BERT) **pre-tokenize** on whitespace/punctuation before applying BPE. SentencePiece treats whitespace as just another character (represented as `▁`). This makes it truly language-agnostic — it doesn't assume spaces separate words (important for Chinese, Japanese, Thai, etc.).

### 5. Tiktoken (GPT-3.5, GPT-4, Claude)

Tiktoken is OpenAI's fast BPE implementation used in production. It's not a new algorithm, but a highly optimized implementation of byte-level BPE:

```python
import tiktoken

# GPT-4's tokenizer
enc = tiktoken.encoding_for_model("gpt-4")

text = "Hello, how are you doing today?"
tokens = enc.encode(text)
print(f"Token IDs: {tokens}")
print(f"Tokens: {[enc.decode([t]) for t in tokens]}")
print(f"Token count: {len(tokens)}")

# Useful for estimating API costs before making calls
```

### Comparison Table

| Feature | BPE (GPT) | WordPiece (BERT) | Unigram (T5) | SentencePiece |
|---------|-----------|-----------------|--------------|---------------|
| Direction | Bottom-up (merge) | Bottom-up (merge) | Top-down (prune) | Either |
| Merge criterion | Frequency | Likelihood | Likelihood drop | Either |
| Deterministic | Yes | Yes | No (can sample) | Depends |
| Pre-tokenization | Yes (regex) | Yes (whitespace) | Optional | No (raw stream) |
| Space handling | `Ġ` prefix | `##` continuation | `▁` prefix | `▁` prefix |
| Used by | GPT-2/3/4 | BERT, DistilBERT | T5, ALBERT | Llama 1/2, Gemma |

## Why Tokenization Matters: Real-World Impacts

Understanding tokenization isn't just theory — it explains many surprising LLM behaviors.

### 1. The "Strawberry Problem": Why LLMs Can't Count Letters

```
Prompt: "How many r's are in strawberry?"
Common LLM answer: "2" (incorrect — the answer is 3)
```

Why? Because the model never sees individual letters. With GPT-4's tokenizer:

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")

tokens = enc.encode("strawberry")
print([enc.decode([t]) for t in tokens])
# → ['str', 'aw', 'berry']

# The model sees 3 tokens, not 10 letters!
# It has to infer letter composition from subword tokens
# 'str' contains one 'r', 'aw' contains zero, 'berry' contains two
# But the model doesn't naturally decompose tokens into characters
```

This is a fundamental limitation: **character-level reasoning requires the model to "look inside" tokens**, which is not a native operation. The model processes tokens as atomic units.

### 2. Why Simple Arithmetic Goes Wrong

```
Prompt: "What is 1234 + 5678?"

Tokenization of "1234": might be ["12", "34"] or ["1", "234"] or ["1234"]
depending on the tokenizer and surrounding context!
```

Numbers are tokenized inconsistently because BPE merges are based on frequency, and digit sequences have varying frequencies. The model might see "1234" as one token in one context but two tokens in another, making it harder to learn consistent arithmetic rules.

This is why many modern models add **special handling for numbers** — either tokenizing each digit individually or using fixed-width number tokens.

### 3. Glitch Tokens: The Dark Side of BPE

During BPE training, some tokens are formed from byte sequences that appeared frequently in the training data but correspond to **nonsensical or very niche content** (like repeated spaces, HTML artifacts, Reddit usernames, or data artifacts).

These tokens have valid IDs in the vocabulary but the model has very little meaningful training data for them. When prompted with these tokens, models can produce bizarre, incoherent, or inconsistent outputs — these are called **"glitch tokens."**

```
Famous examples from GPT-2/3:
- " SolidGoldMagikarp" (a Reddit username that appeared frequently in the
  training data, got merged into a single token, but the model learned
  very little about what it "means")
- "ertoilet" (artifact from web scraping)
- " petertodd" (another username)

When these tokens appear in prompts, the model may:
- Refuse to repeat them
- Claim they don't exist
- Produce nonsensical completions
- Exhibit very different behavior than surrounding tokens
```

### 4. Code Tokenization: Why Indentation Matters

For code-focused models, tokenization of whitespace is critical:

```python
# Python code with 4-space indentation
code = """
def hello():
    if True:
        print("hi")
"""

# GPT-4 tokenization:
# '    ' (4 spaces) → might be 1 token or multiple tokens
# '\t' (tab) → different token than spaces
# Mixing tabs and spaces → different tokenization → confused model!
```

This is why **consistent formatting** matters more for LLMs than you might think — not for readability, but because different formatting produces different token sequences, and the model has learned associations with specific token patterns.

### 5. The Context Window Tax

Because tokenization efficiency varies by language and content type, the "effective" context window differs:

```
Model: 8K context window

English prose:     ~6,000 words fit in 8K tokens (~1.3 tokens/word)
Chinese text:      ~3,000 characters fit in 8K tokens (~2.7 tokens/char)
Python code:       ~4,000 lines fit in 8K tokens (whitespace + keywords)
Base64 encoded:    ~2,000 bytes fit in 8K tokens (extremely inefficient)
Minified JSON:     ~3,500 chars fit in 8K tokens
```

**Practical takeaway**: When building RAG systems or constructing prompts, count **tokens, not characters or words**. The same word budget gives you very different amounts of content depending on the content type and language.

## Practical Considerations

### 1. Special Tokens

BPE tokenizers include special tokens:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Special tokens
print(f"BOS token: {tokenizer.bos_token}")  # Beginning of sequence
print(f"EOS token: {tokenizer.eos_token}")  # End of sequence
print(f"PAD token: {tokenizer.pad_token}")  # Padding
print(f"UNK token: {tokenizer.unk_token}")  # Unknown

# Adding special tokens
special_tokens = {
    'additional_special_tokens': ['<|system|>', '<|user|>', '<|assistant|>']
}
tokenizer.add_special_tokens(special_tokens)
```

### 2. Vocabulary Size Trade-offs

```python
# Small vocabulary (1k-5k tokens)
# ✅ Fast inference
# ✅ Lower memory
# ❌ Longer sequences
# ❌ Less semantic richness

# Large vocabulary (50k-100k tokens)
# ✅ Shorter sequences
# ✅ Better semantic representation
# ❌ Slower inference
# ❌ Higher memory usage

# Common choices:
# GPT-2: 50,257 tokens
# GPT-3: 50,257 tokens
# BERT: 30,522 tokens
# LLaMA: 32,000 tokens
```

### 3. Handling Different Languages

```python
from transformers import AutoTokenizer

# Multilingual tokenizer
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

texts = {
    'english': "Hello, how are you?",
    'chinese': "你好，你好吗？",
    'arabic': "مرحبا، كيف حالك؟",
    'emoji': "Hello 👋 World 🌍"
}

for lang, text in texts.items():
    tokens = tokenizer.tokenize(text)
    print(f"{lang}: {tokens}")
```

## Common Issues and Solutions

### 1. Token Limit Exceeded

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
max_length = 1024  # GPT-2 max sequence length

long_text = "very " * 2000 + "long text"

# Problem: Text too long
tokens = tokenizer.encode(long_text)
if len(tokens) > max_length:
    print(f"Warning: {len(tokens)} tokens exceeds limit of {max_length}")

# Solution 1: Truncate
tokens = tokenizer.encode(
    long_text,
    max_length=max_length,
    truncation=True
)

# Solution 2: Sliding window
def sliding_window_tokenize(text, window_size=1024, stride=512):
    tokens = tokenizer.encode(text)
    windows = []

    for i in range(0, len(tokens), stride):
        window = tokens[i:i + window_size]
        if len(window) > 0:
            windows.append(window)

    return windows

windows = sliding_window_tokenize(long_text)
print(f"Split into {len(windows)} windows")
```

### 2. Inconsistent Tokenization

```python
# Problem: Spaces affect tokenization
text1 = "hello world"
text2 = "helloworld"

tokens1 = tokenizer.tokenize(text1)
tokens2 = tokenizer.tokenize(text2)

print(f"'{text1}': {tokens1}")
print(f"'{text2}': {tokens2}")
# Different tokenization!

# Solution: Normalize text first
def normalize_text(text):
    text = text.strip()
    text = ' '.join(text.split())  # Normalize whitespace
    return text

text1 = normalize_text(text1)
text2 = normalize_text(text2)
```

### 3. Rare Words

```python
# Problem: Rare words get split into many tokens
rare_word = "Antidisestablishmentarianism"
tokens = tokenizer.tokenize(rare_word)
print(f"Tokens for '{rare_word}': {tokens}")
# Output might be: ['Ant', 'idis', 'establishment', 'arian', 'ism']

# This affects:
# - Context window usage
# - Model understanding
# - Generation quality

# Solution: Domain-specific vocabulary
tokenizer.add_tokens(["Antidisestablishmentarianism"])
```

## Building Your Own BPE Tokenizer

### Using Hugging Face Tokenizers

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# Initialize
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

# Configure trainer
trainer = BpeTrainer(
    vocab_size=30000,
    special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"]
)

# Train on files
files = ["corpus1.txt", "corpus2.txt"]
tokenizer.train(files, trainer)

# Save
tokenizer.save("my_bpe_tokenizer.json")

# Load and use
tokenizer = Tokenizer.from_file("my_bpe_tokenizer.json")
output = tokenizer.encode("Hello, how are you?")
print(f"Tokens: {output.tokens}")
print(f"IDs: {output.ids}")
```

### Custom Pre-processing

```python
from tokenizers import normalizers, pre_tokenizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace, Punctuation

# Normalization pipeline
tokenizer.normalizer = normalizers.Sequence([
    NFD(),           # Unicode normalization
    Lowercase(),     # Convert to lowercase
    StripAccents()   # Remove accents
])

# Pre-tokenization pipeline
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    Whitespace(),    # Split on whitespace
    Punctuation()    # Separate punctuation
])

# Post-processing
from tokenizers.processors import TemplateProcessing

tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> $B:1 </s>:1",
    special_tokens=[
        ("<s>", tokenizer.token_to_id("<s>")),
        ("</s>", tokenizer.token_to_id("</s>")),
    ],
)
```

## Analyzing Tokenization Quality

### Metrics

```python
def analyze_tokenizer(tokenizer, texts):
    """Analyze tokenizer performance"""
    total_chars = 0
    total_tokens = 0

    for text in texts:
        tokens = tokenizer.encode(text)
        total_chars += len(text)
        total_tokens += len(tokens)

    # Compression ratio
    compression_ratio = total_chars / total_tokens

    # Tokens per character
    tokens_per_char = total_tokens / total_chars

    print(f"Total characters: {total_chars}")
    print(f"Total tokens: {total_tokens}")
    print(f"Compression ratio: {compression_ratio:.2f} chars/token")
    print(f"Tokens per char: {tokens_per_char:.4f}")

    return {
        'compression_ratio': compression_ratio,
        'tokens_per_char': tokens_per_char
    }

# Compare tokenizers
texts = ["Your corpus here..."] * 1000

print("GPT-2 Tokenizer:")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
analyze_tokenizer(gpt2_tokenizer, texts)

print("\nBERT Tokenizer:")
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
analyze_tokenizer(bert_tokenizer, texts)
```

### Vocabulary Analysis

```python
def analyze_vocabulary(tokenizer):
    """Analyze vocabulary composition"""
    vocab = tokenizer.get_vocab()

    # Count by type
    single_char = 0
    multi_char = 0
    word_like = 0

    for token, idx in vocab.items():
        clean_token = token.replace('Ġ', '').replace('▁', '')

        if len(clean_token) == 1:
            single_char += 1
        elif len(clean_token) > 1:
            multi_char += 1

        if clean_token.isalpha() and ' ' not in clean_token:
            word_like += 1

    print(f"Total vocabulary: {len(vocab)}")
    print(f"Single characters: {single_char} ({single_char/len(vocab)*100:.1f}%)")
    print(f"Multi-character: {multi_char} ({multi_char/len(vocab)*100:.1f}%)")
    print(f"Word-like tokens: {word_like} ({word_like/len(vocab)*100:.1f}%)")

tokenizer = AutoTokenizer.from_pretrained("gpt2")
analyze_vocabulary(tokenizer)
```

## Best Practices

### 1. Choosing Vocabulary Size

```python
# Domain-specific guidelines:

# Chat/Dialogue (GPT-4, Claude):
vocab_size = 100_000  # Rich vocabulary for nuanced conversation

# Code (CodeLLaMA, StarCoder):
vocab_size = 32_000  # Balance between tokens and code patterns

# Multilingual (mBERT, XLM-R):
vocab_size = 250_000  # Large vocab to cover many languages

# Efficient deployment:
vocab_size = 8_000  # Smaller for edge devices
```

### 2. Training Data Selection

```python
# Diverse corpus
training_corpus = {
    'books': 0.4,      # 40% books
    'web': 0.3,        # 30% web text
    'code': 0.15,      # 15% code
    'dialogue': 0.15   # 15% conversations
}

# Balance languages
language_distribution = {
    'english': 0.7,
    'chinese': 0.1,
    'spanish': 0.05,
    'others': 0.15
}
```

### 3. Evaluation

```python
# Test on held-out data
def evaluate_tokenizer(tokenizer, test_corpus):
    metrics = {
        'avg_tokens_per_sentence': [],
        'compression_ratio': [],
        'unk_token_rate': []
    }

    for text in test_corpus:
        tokens = tokenizer.encode(text)

        # Metrics
        metrics['avg_tokens_per_sentence'].append(len(tokens))
        metrics['compression_ratio'].append(len(text) / len(tokens))

        # UNK rate
        unk_id = tokenizer.unk_token_id
        unk_count = tokens.count(unk_id) if unk_id else 0
        metrics['unk_token_rate'].append(unk_count / len(tokens))

    # Summary
    for metric, values in metrics.items():
        print(f"{metric}: {sum(values)/len(values):.2f}")

    return metrics
```

## How Tokenizer Design Has Evolved: Lessons from Real Models

Looking at how tokenizer design has changed across model generations reveals clear trends:

| Model | Year | Vocab Size | Type | Key Innovation |
|-------|------|-----------|------|----------------|
| GPT-2 | 2019 | 50,257 | Byte-level BPE | First byte-level BPE for LLMs |
| BERT | 2019 | 30,522 | WordPiece | `##` continuation tokens |
| T5 | 2020 | 32,000 | SentencePiece Unigram | Language-agnostic, no pre-tokenization |
| GPT-3 | 2020 | 50,257 | Byte-level BPE | Same as GPT-2 |
| Llama 1 | 2023 | 32,000 | SentencePiece BPE | Efficient for English-centric use |
| Llama 2 | 2023 | 32,000 | SentencePiece BPE | Same as Llama 1 |
| GPT-4 | 2023 | 100,256 | Byte-level BPE (cl100k) | 2x vocab for better multilingual |
| Llama 3 | 2024 | 128,256 | Tiktoken BPE | 4x vocab, massively better multilingual |
| Gemma 2 | 2024 | 256,000 | SentencePiece | Largest vocab, best multilingual coverage |

**The clear trend**: vocabulary sizes are getting much larger. Early models used ~32K-50K tokens; modern models use 100K-256K. The reason is that **larger vocabularies improve efficiency for non-English languages and code**, which are increasingly important use cases.

Llama 3's jump from 32K to 128K tokens resulted in:
- **15% fewer tokens** for the same English text
- **Up to 4x fewer tokens** for some non-English languages
- **Better code tokenization** (more programming keywords become single tokens)

## Conclusion

BPE tokenization sits at the foundation of every modern LLM. It's the first transformation your text undergoes and the last transformation before you see the output. Understanding it deeply gives you:

1. **Better prompt engineering**: You know why character counting fails, why formatting matters, and how to stay within context limits effectively
2. **Smarter API cost management**: You can predict and control token usage across different languages and content types
3. **Debugging superpowers**: When a model behaves strangely on specific inputs, tokenization is often the first place to look
4. **Informed model selection**: Vocabulary size and tokenizer type significantly affect model performance on your specific use case (multilingual? code-heavy? domain-specific?)
5. **Custom training knowledge**: If you're training or fine-tuning models, tokenizer design is one of the most impactful architectural decisions

The key insight to carry forward: **the model doesn't see text the way you do.** It sees a sequence of token IDs, and the mapping from your text to those IDs is determined entirely by the BPE merge rules learned during tokenizer training. Every quirk, strength, and weakness of the model can be partially traced back to this mapping.

## References

1. Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (2016) — The original BPE for NLP paper
2. Gage, "A New Algorithm for Data Compression" (1994) — The original BPE compression algorithm
3. Radford et al., "Language Models are Unsupervised Multitask Learners" (2019) — GPT-2, introduced byte-level BPE
4. Kudo and Richardson, "SentencePiece: A simple and language independent subword tokenizer" (2018) — SentencePiece framework
5. Kudo, "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates" (2018) — Unigram tokenization
6. [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/)
7. [OpenAI Tiktoken](https://github.com/openai/tiktoken)
8. [SentencePiece](https://github.com/google/sentencepiece)
9. [Llama 3 Technical Report](https://arxiv.org/abs/2407.21783) — Discussion of vocabulary expansion impact
10. [The Illustrated GPT-2 - Tokenization](https://jalammar.github.io/illustrated-gpt2/)
