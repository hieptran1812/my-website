---
title: "The Semantic Hub Hypothesis: Language Models Share Semantic Representations Across Languages and Modalities"
publishDate: "2026-10-15"
date: "2026-10-15"
description: "How Wu, Yu, Yogatama, Lu and Kim show that one language model routes Chinese, arithmetic, code, images and audio through a shared middle-layer space that reads as English under the logit lens, and how steering that space with English words changes outputs in every other data type."
category: "paper-reading"
subcategory: "AI Interpretability"
tags: ["paper-reading", "semantic-hub", "semantic-representations", "logit-lens", "multilingual", "multimodal", "activation-steering", "language-models", "interpretability", "iclr-2025-poster"]
author: "Hiep Tran"
featured: true
readTime: 19
image: "/imgs/blogs/the-semantic-hub-hypothesis-language-models-share-semantic-representations-across-languages-and-modalities-fig1.webp"
excerpt: "When Llama-3 reads Chinese, its middle layers think in English. The same is true for arithmetic, Python, images and audio. This paper calls that shared middle space the semantic hub and shows you can steer it with English words."
paper:
  title: "The Semantic Hub Hypothesis: Language Models Share Semantic Representations Across Languages and Modalities"
  authors: "Zhaofeng Wu, Xinyan Velocity Yu, Dani Yogatama, Jiasen Lu, Yoon Kim"
  venue: "ICLR 2025"
  url: "https://arxiv.org/abs/2411.04986"
---

> [!tldr]
> **The claim.** A language model that handles many data types (languages, arithmetic, code, images, audio) does not keep a separate compartment for each. It maps them into one shared space in its middle layers, a "semantic hub", and that space is scaffolded by the model's dominant training language.
>
> **Why it matters.** If the hub reads as English, the logit lens and English steering words become tools for inspecting and controlling a model processing Chinese, Python or pixels.
>
> **The most surprising result.** Adding a vector built from the English words "three" and "two" at one position of `4=1+3` makes Llama models write `4=1+2+1` in over 90% of cases at high strength. The model does not just swap a token; it repairs the arithmetic.
>
> **Where it is thin.** The cross-lingual sentiment steering effects are small relative to their standard deviations, the image and audio evidence is mostly similarity above a baseline (no anchoring test is possible there), and every model tested is 7B to 70B with one dominant language.

When a single set of weights reads Chinese, `5+3`, `range(end)` and a photo of a red square, where do those inputs go? The post follows the paper's three parts: the **hypothesis** and its two tests, the **evidence** across six data types, and the **interventions** that show the hub is used.

![Figure 1 from Wu et al. (2025): for every other layer, the token closest to the hidden state under the logit lens; Llama-3 reading Chinese, arithmetic and code, LLaVA reading an image and SALMONN reading audio all pass through English words in the middle](/imgs/blogs/the-semantic-hub-hypothesis-language-models-share-semantic-representations-across-languages-and-modalities-fig1.webp)

Each row is an input and each cell is the token closest to the hidden state at that layer. The Chinese row turns into English midway, then back into Chinese at the end.

## The hypothesis: hub and spokes

**The problem.** A model with one set of parameters could carve its hidden space into disjoint subspaces, one per data type. Or it could share. Prior evidence covered only a narrow synthetic setting.

**The analogy.** The hub-and-spoke model from neuroscience (Patterson et al., 2007) places a transmodal hub in the anterior temporal lobe that integrates modality-specific spokes such as the visual and auditory cortices. A picture of a dog and the word "dog" arrive by different roads and meet at the same address.

**The mechanism.** The paper frames generation as three stages. An encoder maps an input sequence into a meaning space. Some reasoning happens there. A verbalizer turns the result back into surface tokens. In a transformer, the authors map these onto early layers, middle layers, and late layers plus the LM head.

![Hub and spokes inside one language model: six input types pass through early encoding layers into a shared middle-layer hub that reads as English under the logit lens, then late layers verbalize back into the input's own type](/imgs/blogs/the-semantic-hub-hypothesis-language-models-share-semantic-representations-across-languages-and-modalities-1.webp)

**The math.** Let $Z$ be the set of data types the model supports, and $X_z$ the domain of type $z \in Z$ (for Chinese, all Chinese tokens; for images, $[0,255]^{w \times h \times 3}$ patch values). The model's encoder is a function $M_{\text{LM}}$ that maps a sequence $w^{z}_{1:t}$ of $t$ tokens of type $z$ into a representation space $S_{\text{LM}} \subseteq \mathbb{R}^d$, where $d$ is the hidden width. In practice $M_{\text{LM}}(w^{z}_{1:t})$ is the hidden state $h^{\ell}_t \in \mathbb{R}^d$ at the last position $t$ and some layer $\ell$.

The hypothesis is informally $M_{\text{LM}}(w^{z_1}_{1:t}) \approx M_{\text{LM}}(w^{z_2}_{1:t'})$ when the two sequences mean the same thing. The authors note that absolute similarity in high dimensions is hard to interpret, so they use a relative test. Take a semantically unrelated sequence $u^{z_2}_{1:t''}$ of the second type and require:

$$
\text{sim}\big(M_{\text{LM}}(w^{z_1}_{1:t}),\ M_{\text{LM}}(w^{z_2}_{1:t'})\big) \;>\; \text{sim}\big(M_{\text{LM}}(w^{z_1}_{1:t}),\ M_{\text{LM}}(u^{z_2}_{1:t''})\big)
$$

This is **Eq. 1**. A sentence should sit closer to its translation than to a random sentence in the same target language. Here $\text{sim}$ is cosine similarity.

The second half of the hypothesis is the more surprising one. If the model has a dominant data type $z^\star$ (English for Llama), the hub should be anchored by tokens of $z^\star$. Write $\tau^{z^\star}$ for a token in the dominant type that continues the input, and $\tau^{z^\circ}$ for the corresponding token in the input's own non-dominant type $z^\circ$. Then:

$$
\text{sim}\big(M_{\text{LM}}(w^{z^\circ}_{1:t}),\ \text{emb}(\tau^{z^\star})\big) \;>\; \text{sim}\big(M_{\text{LM}}(w^{z^\circ}_{1:t}),\ \text{emb}(\tau^{z^\circ})\big)
$$

This is **Eq. 2**. The paper's own example: an English-dominant model reading the Chinese prefix meaning "This paper is so hard to" should have a middle-layer state closer to the English token "write" than to the Chinese token for "write", even though the input and the eventual output are Chinese.

## The two tests, made concrete

![Two tests for a semantic hub: relative similarity against a non-matching baseline, and dominant-language anchoring under the logit lens](/imgs/blogs/the-semantic-hub-hypothesis-language-models-share-semantic-representations-across-languages-and-modalities-2.webp)

### Test 1: similarity above a baseline

**Mechanism.** Feed both halves of a parallel pair through the model separately. At every layer, take the last-token hidden state of each and compute their cosine. Do the same for many non-matching pairs and average them. Subtract that average from the matching cosine, layer by layer. A positive gap means meaning is shared beyond what any two random inputs share.

**Worked example (illustrative numbers, mine).** Suppose at layer 16 an English sentence and its Chinese translation have cosine ${0.62}$, while the average over non-matching English and Chinese sentences at the same layer is ${0.45}$. The reported score is the difference, ${0.17}$. The raw ${0.62}$ alone tells you little, because in a high-dimensional space where every hidden state shares common directions, unrelated inputs can still have large cosines. That is exactly why the paper subtracts a baseline.

**When it fails.** If matching pairs share something shallow, like the same digits or names, the test rewards surface overlap. The paper controls for this in arithmetic and formal semantics.

### Test 2: anchoring via the logit lens

**Mechanism.** A transformer turns its final hidden state into a next-token distribution with the unembedding matrix. The logit lens (nostalgebraist, 2020) applies the same matrix to an intermediate hidden state, as if that layer were the last. It needs no training, which is why the authors chose it: fewer confounders than a learned probe.

$$
p_{\text{logitlens}}(\cdot \mid h^{\ell}_t) := \text{softmax}\big(O\, h^{\ell}_t\big)
$$

Here $O \in \mathbb{R}^{|V| \times d}$ is the output embedding (unembedding) matrix, with one row per vocabulary token, $|V|$ is the vocabulary size, and $\ell$ ranges over the $L$ layers. The bias term is omitted, as in the paper. With dot product as the similarity, Eq. 2 becomes a comparison of two probabilities, **Eq. 3**:

$$
p_{\text{logitlens}}\big(\tau^{z^\star} \mid h^{\ell}_t\big) \;>\; p_{\text{logitlens}}\big(\tau^{z^\circ} \mid h^{\ell}_t\big)
$$

**Worked example (mine).** Take a toy hidden size $d = 2$, $h = (1, 0.5)$, and three vocabulary rows: "write" $= (2, 0)$, the Chinese token for "write" $= (0, 1)$, "the" $= (0.5, 0.5)$. The logits are ${2}$, ${0.5}$ and ${0.75}$. Exponentiating gives about 7.389, 1.649 and 2.117, summing to 11.155. So the lens assigns "write" about 0.662, the Chinese token about 0.148, and "the" about 0.190. Eq. 3 holds at this layer.

**When it fails.** The lens assumes intermediate layers share the final layer's basis, which is shaky early on (hence chance-level early readouts). It reads only single tokens. And it cannot run when the non-dominant type has no vocabulary tokens, as with image patches or audio; there, only Eq. 1 is testable.

## Evidence 1: languages

Experiment 1 uses 5,260 professionally translated English-Chinese pairs from GALE, each wrapped in a template ending in "This represents" to discourage code-switching. The models are Llama-2, Llama-3 (English-dominant), Baichuan-2 (Chinese-dominant) and BLOOM (no dominant language), all at 7B or 8B.

![Figure 3 from Wu et al. (2025): (a) raw cosine similarity between English and Chinese translations per layer, (b) the same minus a non-parallel baseline, peaking in the middle layers, (c) Llama-3 logit lens log probability of the English translation token versus the next Chinese token](/imgs/blogs/the-semantic-hub-hypothesis-language-models-share-semantic-representations-across-languages-and-modalities-fig2.webp)

Panel (b) is the one to look at. The gap over baseline is positive for all four models and largest in the middle layers. It holds even for BLOOM, which has no dominant language: the hub exists, it just does not read as any one language.

Panel (c) is the anchoring test on 1,000 Chinese Wikipedia prefixes: the next Chinese token versus the first token of its English translation. In early layers neither beats chance. From layer 17 until layer 31, the English token is substantially more likely; Chinese wins only at the very end.

Parallel tokens are filtered hard: the next BPE token must be a full Chinese word under the Jieba segmenter with a single-token English translation in CC-CEDICT.

### From single tokens to whole languages

Token pairs are narrow. To ask which language a hidden state "is in" overall, the paper builds a language classifier out of the logit lens.

**Mechanism.** First, tokenize the language-specific splits of mC4 to estimate $p(w \mid z)$, how often token $w$ appears in language $z$. Bayes' rule with a uniform prior flips this into $p(z \mid w)$. Then weight each token's language by how strongly the lens predicts it:

$$
p(z \mid h^{\ell}_t) \;\propto\; \sum_{w \in V} p(z \mid w)\; p_{\text{logitlens}}(w \mid h^{\ell}_t)
$$

Averaging over positions $t$ gives a per-layer language distribution $p_\ell(z)$. The authors point out that the uniform prior actually understates their effect: a realistic prior would push the dominant language even higher.

**Worked example (mine).** Suppose the lens at some layer puts 0.6 on "cat", 0.3 on the Chinese word for cat, and 0.1 on the digit "1". Let $p(\text{En} \mid \text{cat}) = 0.9$, $p(\text{En} \mid \text{Chinese cat}) = 0.05$, and $p(\text{En} \mid \text{"1"}) = 0.5$. Then $p(\text{En} \mid h) = 0.54 + 0.015 + 0.05 = 0.605$, and Chinese gets the remaining 0.395. That layer leans English.

![Figure 4 from Wu et al. (2025): per-layer probability of English and Chinese for English and Chinese inputs across Llama-2, Llama-3 (7B/8B and 70B), Baichuan-2 and BLOOM](/imgs/blogs/the-semantic-hub-hypothesis-language-models-share-semantic-representations-across-languages-and-modalities-fig3.webp)

Figure 4 runs this on 10,000 English and Chinese Wikipedia sentences. The pattern is clean:

| Model | Dominant language | Middle layers, Chinese input | Final layers, Chinese input |
| --- | --- | --- | --- |
| Llama-2, Llama-3 (7B/8B, 70B) | English | English dominates | Chinese dominates |
| Baichuan-2-7B | Chinese | Chinese (and Chinese also dominates the middle for English input) | Input language |
| BLOOM-7B | None | No clear latent language | Input language |

The Baichuan row is the key control: the hub is not "English" as such, it is whatever the model saw most. The 70B Llamas behave like their 7B and 8B siblings. BLOOM's middle-layer tokens are mostly meaningless symbols, yet Experiment 1 showed it still has a shared space.

## Evidence 2: arithmetic

Expressions take the form `a=b+c` or `a=b*c`, with `a` and `b` at most two digits and `c` a single positive digit. The English translation of `5+3` is "five plus three".

**Experiment 1** compares every numeric right-hand side against every English one and buckets the pairs: exact translations (1,123 pairs), same value but different expression, like `5+3` and "two plus six" (13,293), and different value (1,247,836). Exact translations have high cosine, but the authors flag that this is expected: digit embeddings sit near their number words, so even a bag of word embeddings would score well. The more interesting result is that same-value pairs beat different-value pairs. The model groups expressions by what they evaluate to, not just by shared tokens.

**Experiment 2** looks at prefixes like `a=b+` and asks whether the hidden state passes through the English word for the missing `c`.

![Figure 6 from Wu et al. (2025): Llama-3 hidden-state trajectories when predicting a number, projected onto the first two principal components of 20 output embeddings (ten English number words, ten numerals), colored by layer](/imgs/blogs/the-semantic-hub-hypothesis-language-models-share-semantic-representations-across-languages-and-modalities-fig4.webp)

The PCA is fitted only on the 20 output embeddings, so it is a fixed map of "where number words live" and "where numerals live". The 100 sampled trajectories travel through the English-word region in the middle layers before ending near the numerals. The logit lens agrees (1,123 cases): the numeral and its English word have similar log probability until around layer 25, after which the numeral dominates.

## Evidence 3: code

When Llama-2 reads Python, its middle layers often predict free-form English. Inside the list literal `[1.0, 2.0,`, they predict "and". The paper tests two cases on MBPP:

- **List literals.** At all 6,923 commas inside Python list literals (unit tests included), the lens favors "and" in the middle layers and the true next token in the final layers. Two baselines, "or" and "not", stay below "and".
- **Function arguments.** For 540 arguments, the lens at the preceding `(` or `,` is closer to the parameter name (the "semantic role", like `end` in `range(start, end, step)`) than to the actual argument expression in early and middle layers. Only in the final layers does it converge to the surface form.

The second result is stronger: role and surface form are different strings, so a token copier would never visit the parameter name.

## Evidence 4: formal semantics

COGS pairs synthetic English sentences with Neo-Davidsonian logical forms. "Eleanor sold Evelyn the cake." becomes `*cake(x4); sell.agent(x1, Eleanor) AND sell.recipient(x1, Evelyn) AND sell.theme(x1, x4)`.

Sentence-to-form similarity could be lexical overlap, so the authors build a harder baseline. For 2,233 sentences with two names, they swap the names in the logical form: overlap is identical, only meaning differs, and a bag-of-words model scores at chance. Matching pairs still win, even with predicate order shuffled too.

The token-level test covers 3,257 agents, 2,583 recipients and 714 themes. Llama-3 has a prior toward the word "agent", but after subtracting the per-role average, the correct role is always closest. Llama-2 does not show this, and BLOOM barely beats chance at sentence level, which the authors read as the strategy correlating with capacity.

## Evidence 5 and 6: images and audio

**Images.** On 1,000 MSCOCO images and captions, both LLaVA-7B and Chameleon-7B show image-caption similarity above the non-matching baseline. Neither model is trained with a CLIP-style contrastive objective. The alignment emerges from autoregressive training alone.

Eq. 3 cannot be tested without image tokens in the vocabulary. Instead, on pure red, green, blue and black images (2,304 patch positions), LLaVA's closest middle-layer token is the matching color word more than 20% of the time. Appendix tests with segmentation labels hold strongly for LLaVA, weakly for Chameleon.

**Audio.** SALMONN on 1,000 VGGSound clips shows the same pattern: a clip is more similar to its own label ("duck quacking") than to a non-matching one in the middle layers.

## Interventions: is the hub used?

**The problem.** Everything so far is correlational. A shared space could be a vestigial side effect of training, present but ignored by the computation. The standard answer is to intervene.

**The analogy.** If the control room really is shared, then flipping a switch labelled in English should change what happens on the Chinese floor.

**The mechanism.** Most experiments use a variant of Activation Addition (ActAdd, Turner et al., 2024). Pick two trigger words whose difference expresses the change you want. Compute a direction from them. Scale it and add it to the hidden state at one layer during the normal forward pass.

**The math.** Let $a$ and $b$ be the trigger words and $\ell$ the chosen layer. The steering vector is either a hidden-state difference or an unembedding difference:

$$
v = c \cdot \big(h^{\ell}(a) - h^{\ell}(b)\big) \quad \text{or} \quad v = c \cdot \big(O_a - O_b\big), \qquad \tilde h^{\ell}_t = h^{\ell}_t + v
$$

Here $c$ is the intervention strength coefficient, $h^{\ell}(a) \in \mathbb{R}^d$ is the layer-$\ell$ hidden state after a prefix that pins down word $a$, $O_a$ is row $a$ of the unembedding matrix, and $\tilde h^{\ell}_t$ replaces the original state at position $t$. The paper uses hidden-state differences for the multilingual and arithmetic cases, and explains why: for languages, an unembedding difference like "Good" minus "Bad" might succeed only because it resembles "Bueno" minus "Malo" through ordinary cross-lingual embedding alignment. Hidden-state differences test the hub claim directly.

![Steering through the hub with English words: a vector from 'three' minus 'two' added at the plus sign turns 4=1+3 into 4=1+2+1](/imgs/blogs/the-semantic-hub-hypothesis-language-models-share-semantic-representations-across-languages-and-modalities-3.webp)

**Worked example: arithmetic.** Take `4=1+3` and steer with "three" and "two". The hidden state for "three" comes from a prefix that forces the number ("Eight equals to five plus"), since the state after reading "three" mostly encodes the next token. The vector goes only at the `+` position, layer 25 for Llama-3 and 30 for Llama-2. Success is strict: the model must write `4=1+2+1`, not just paste `2`.

### The four cross-type interventions

![Figure 12 from Wu et al. (2025): (a) arithmetic steering success versus strength for Llama-2 and Llama-3, (b) range(end) to range(0, end) for Llama-2, (c) Chameleon color replacement accuracy by layer, (d) SALMONN p(mammal) under mammal and non-mammal steering](/imgs/blogs/the-semantic-hub-hypothesis-language-models-share-semantic-representations-across-languages-and-modalities-fig5.webp)

| Data type | Model | Trigger | Where | Headline |
| --- | --- | --- | --- | --- |
| Arithmetic (846 additions) | Llama-2, Llama-3 | hidden("three") minus hidden("two") | `+` position, layer 25 / 30 | Over 90% produce the decremented number plus `+1` at high strength |
| Code (159 `range(end)` calls) | Llama-2 | unembedding "start" minus "end" | `(` position, layer 17 | Up to 67% become `range(0, end)` |
| Images (48 color pairs) | Chameleon-7B | replace states with new-color minus old-color unembedding | all states at and after layer $\ell$ | Above 80% at the best layer |
| Audio (1,000 animal sounds) | SALMONN | unembedding of new animal minus original animal | layer 13 | p(mammal) moves in the steered direction as strength grows |

One design choice deserves credit. For images, two colors share one image and only one is replaced; the model must name the new color and the untouched one. A lexical bias toward "green" would corrupt both answers.

### The multilingual sentiment test, read closely

The multilingual intervention steers sentiment on 1,000 Spanish (InterTASS) and 1,000 Chinese (Amazon reviews) prefixes. ActAdd uses hidden states at layer 17 with coefficient 5. The steering pair is "Good"/"Bad" in English, "Bueno"/"Malo" in Spanish, and the Chinese words for good and bad. Here is Llama-3 (Table 1), mean over 10 seeds:

| Text | Direction | Steering language | Sentiment | Disfluency (lower is better) | Relevance |
| --- | --- | --- | --- | --- | --- |
| Spanish | none | none | 0.143 ± 0.022 | 7.35 ± 1.19 | 0.861 |
| Spanish | down | Spanish | 0.125 ± 0.034 | 10.54 ± 2.39 | 0.842 |
| Spanish | down | English | 0.139 ± 0.026 | 8.75 ± 2.20 | 0.857 |
| Spanish | up | Spanish | 0.175 ± 0.035 | 7.98 ± 2.04 | 0.856 |
| Spanish | up | English | 0.159 ± 0.026 | 7.35 ± 1.01 | 0.859 |
| Chinese | none | none | 0.178 ± 0.030 | 11.06 ± 3.12 | 0.869 |
| Chinese | down | Chinese | 0.152 ± 0.040 | 10.78 ± 2.66 | 0.866 |
| Chinese | down | English | 0.161 ± 0.029 | 11.36 ± 1.13 | 0.864 |
| Chinese | up | Chinese | 0.153 ± 0.034 | 11.12 ± 3.12 | 0.870 |
| Chinese | up | English | 0.179 ± 0.032 | 10.90 ± 3.25 | 0.869 |

The paper summarizes this as English steering being "similarly effective". My reading is more cautious. Every shift is a few hundredths against standard deviations of about 0.03. Chinese steered up with Chinese words actually went down, and Chinese steered up with English moved by 0.001. English matches the text language here mostly because neither moves sentiment much. The arithmetic, code, image and audio interventions carry the causal claim far better than this table does.

## Results at a glance

| Data type | Models | Data | Eq. 1 (shared space) | Eq. 3 (dominant anchoring) |
| --- | --- | --- | --- | --- |
| Languages | Llama-2, Llama-3, Baichuan-2, BLOOM | GALE (5,260 pairs), Wikipedia | Yes, peaks in middle layers, all four models | Yes for Llama (English) and Baichuan (Chinese); no clear latent language in BLOOM |
| Arithmetic | Llama-2, Llama-3 | Synthetic `a=b+c`, `a=b*c` | Yes, including same-value pairs | Number words compete with numerals until about layer 25 |
| Code | Llama-2 | MBPP (974 programs) | Not tested (no clean pairs) | Yes: "and" at commas, parameter names before arguments |
| Formal semantics | Llama-2, Llama-3, Baichuan-2, BLOOM | COGS | Yes, with name-swap control; BLOOM near chance | Role-level test: Llama-3 yes, Llama-2 no |
| Images | LLaVA-7B, Chameleon-7B | MSCOCO (1,000) | Yes | Not testable; color word top-1 above 20% in LLaVA |
| Audio | SALMONN | VGGSound (1,000) | Yes | Not testable; label words beat random labels |

**What may not transfer.** Anchoring needs a strongly dominant language; BLOOM shares a space but loses the readable one. The multimodal models sit on text backbones, so "the hub reads as English" may partly mean "the backbone was English first".

## How this fits with nearby work

The [Platonic Representation Hypothesis](/blog/paper-reading/ai-interpretability/position-the-platonic-representation-hypothesis) argues that separately trained vision and language models converge as they scale, and [vec2vec](/blog/paper-reading/ai-interpretability/harnessing-the-universal-geometry-of-embeddings) translates between different models' embedding spaces. The semantic hub claim is narrower and in one sense stronger: inside one model, the representations align with no transformation at all.

My extrapolation, not the paper's: a concept dictionary learned on English activations, such as a sparse autoencoder or the shared dictionaries in [Universal Sparse Autoencoders](/blog/paper-reading/ai-interpretability/universal-sparse-autoencoders-interpretable-cross-model-concept-alignment), should transfer to Chinese or code best in the middle layers.

## Critique

**What is strong.** Six data types, two complementary tests, and an intervention for almost every one. The COGS name swap kills lexical overlap, the two-color design kills "it just printed the word", and Baichuan-2 shows the latent language tracks the dominant training language, not English per se.

**What is weak.**

- **The sentiment table is near noise.** As shown above, the multilingual intervention effects are mostly within one standard deviation, and one cell moves the wrong way. The Llama-2 version (Table 2) has a Chinese up-steering run with disfluency of 545.94 ± 1544.36, a sign that some seeds broke generation entirely.
- **The logit lens is the only reader.** The authors leave trained lenses to future work; anchoring could look different through one.
- **Narrow slices.** Eq. 3 is tested only where one BPE token carries the meaning, and code and token-level semantics rest on one model each.
- **Missing ablation.** No experiment removes or projects out the hub directions to show that cross-lingual performance breaks. Steering shows the hub can be pushed; ablation would show it is necessary.

**The authors' own caveat:** dominant-language reliance could import that language's biases everywhere, and routing arithmetic through English number words is odd given how differently languages build numbers.

**What would change my mind.** I would doubt the hub-as-computation reading if projecting out the middle-layer directions shared by parallel inputs left translation and cross-lingual QA intact, or if a tuned lens on the same Llama-3 states showed the middle layers reading as the input language rather than English. Conversely, a necessity ablation that breaks Chinese performance while sparing English would make the causal claim much stronger than steering alone can.

## References

1. Zhaofeng Wu, Xinyan Velocity Yu, Dani Yogatama, Jiasen Lu, Yoon Kim. [The Semantic Hub Hypothesis: Language Models Share Semantic Representations Across Languages and Modalities](https://arxiv.org/abs/2411.04986). ICLR 2025.
2. Code release: [github.com/ZhaofengWu/semantic-hub](https://github.com/ZhaofengWu/semantic-hub).
3. Wendler et al. (2024). Do Llamas Work in English? On the Latent Language of Multilingual Transformers.
4. Turner et al. (2024). Activation Addition (ActAdd), the steering method used for the interventions.
5. nostalgebraist (2020). Interpreting GPT: the logit lens.
6. On this site: [Position: The Platonic Representation Hypothesis](/blog/paper-reading/ai-interpretability/position-the-platonic-representation-hypothesis), [vec2vec](/blog/paper-reading/ai-interpretability/harnessing-the-universal-geometry-of-embeddings), [Universal Sparse Autoencoders](/blog/paper-reading/ai-interpretability/universal-sparse-autoencoders-interpretable-cross-model-concept-alignment).
