---
title: "The Linear Representation Hypothesis: How Concepts Become Directions Inside a Language Model"
date: "2026-06-13"
publishDate: "2026-06-13"
description: "A deep dive into why high-level concepts show up as straight-line directions in an LLM's activation space — the geometry that makes it work, the probing and steering that exploit it, where the directions come from, and where the line finally bends."
tags: ["ai-interpretability", "linear-representation", "activation-steering", "linear-probe", "sparse-autoencoder", "superposition", "mechanistic-interpretability", "representation-engineering", "concept-directions", "causal-inner-product", "steering-vectors", "llm"]
category: "machine-learning"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 50
---

A 4096-dimensional residual stream is a profoundly unfriendly place to go looking for meaning. The honest prior, before you have ever run a probe, is that a concept like "this text is in French" or "the user is asking me to lie" is smeared illegibly across thousands of coordinates in some tangled, nonlinear, basis-dependent code that no human will ever read. That is what most high-capacity function approximators do. It is what we were taught to expect.

The surprise — the single most load-bearing empirical fact in all of mechanistic interpretability — is that this is frequently, embarrassingly, not what happens. Over and over, a high-level concept turns out to be a **single straight-line direction** in activation space. You find it with subtraction. You read it with a dot product. You change the model's behavior by adding it back in with a scalar knob. "Truth", "sentiment", "refusal", "the Golden Gate Bridge", "the day of the week" — these have all been pinned to directions you can write down as a vector of floats.

That claim — that features are directions, and that the model's computation respects this in a way you can exploit — is the **linear representation hypothesis** (LRH). It is the reason linear probes work, the reason activation steering works, the reason sparse autoencoders are even a coherent idea. This article is a tour of the whole thing: what "linear" precisely means (it means three different testable things), the non-Euclidean geometry that makes the naive version subtly wrong, the probing and steering recipes that turn the hypothesis into engineering, where the directions plausibly come from during training, the production-grade evidence, and — just as important — the cases where the line bends and the hypothesis quietly fails.

![A hidden activation is a sum of concept directions that a probe reads and steering writes.](/imgs/blogs/linear-representation-hypothesis-1.webp)

The diagram above is the mental model for everything that follows. A hidden activation $h$ is well-approximated as a sum of concept directions, $h \approx \sum_i \alpha_i v_i$, where each $v_i$ is a fixed unit vector for some human-meaningful feature and $\alpha_i$ is how strongly that feature is present. Once you believe that decomposition, two operations fall out for free. You can **read** a concept by projecting: $\hat{w}^\top h$ recovers how much of feature $\hat{w}$ is in $h$. And you can **write** a concept by adding: $h \leftarrow h + \alpha v$ injects more of $v$ into the stream. Reading is probing. Writing is steering. The rest of this post is a careful look at why those two arrows are the same arrow seen from opposite ends, and how far you can trust them.

## Why a straight line is a surprising place to find a concept

Start with the mismatch between the textbook intuition and the measured reality, because the gap is the whole point.

| Question | Naive expectation | What the LRH says (and measurement confirms) |
| --- | --- | --- |
| How is "truthfulness" stored? | Diffusely, nonlinearly, across thousands of entangled units | As a direction; a difference-in-means vector separates true from false statements |
| Can I find it without training a big classifier? | No, you need a deep nonlinear probe | Often yes; subtract two class means and you are done |
| How do I change the behavior? | Fine-tune on a curated dataset | Add a scalar multiple of the direction to the residual stream at one layer |
| Is the "king − man + woman" thing a party trick? | Yes, an artifact of word2vec | No; the same offset structure recurs in modern LLM hidden states |
| Does one neuron equal one concept? | Hopefully | Almost never — concepts are directions, not basis-aligned neurons |

The last row is where most newcomers trip. The LRH does **not** say "neuron 2173 is the French neuron". It says there exists a *direction* — a particular weighted combination of neurons — that behaves like the French feature. Individual neurons are usually polysemantic: they participate in many directions at once. This is the bridge to [superposition](/blog/machine-learning/ai-interpretability/what-is-superposition), which we get to in detail later: the network packs more directions into the space than it has neurons, so the meaningful objects are directions in the full space, not coordinates of the standard basis.

> The standard basis is the model's accident; the concept directions are the model's intent. Interpretability is the work of rotating from the first to the second.

Why should we even hope for linearity? Three reasons, each of which we will make precise:

1. **Transformers are mostly linear.** Attention is a weighted sum; the residual stream is literally a running sum of layer outputs; the unembedding is a single matrix multiply. The only nonlinearities are the attention softmax and the MLP activation. A representation that is linear is the *cheapest* thing for the next layer to read, because reading it is one matrix multiply.
2. **Linear features compose.** If "plural" and "past tense" are independent directions, a token that is both is just their sum, and any downstream linear readout can pick out either one without disentangling them first.
3. **The training objective rewards it.** The final layer is a linear map from the residual stream to logits. If a concept controls the next-token distribution, the most direct way to expose it to that linear map is to make it a linear direction. We will see a formal version of this argument in the section on origins.

None of this is a proof. The LRH is an empirical hypothesis that holds *remarkably* often and then, in specific structured cases, fails. Both halves of that sentence matter, and a serious practitioner spends as much energy on the failures as on the wins, because the failures are where the model's real structure shows through.

There is a reason the hypothesis sits at the center of the field rather than at its frontier. Every practical interpretability technique you have heard of assumes some version of it. Linear probes assume the concept is readable by a dot product. Activation steering assumes it is writable by addition. Sparse autoencoders assume the activation decomposes into a sparse sum of feature directions. Logit-lens and tuned-lens methods assume intermediate activations can be read through the final linear unembedding. Even circuit analysis leans on it: the edges of a circuit are linear read/write operations on the residual stream between attention and MLP components. Pull the linear representation hypothesis out from underneath, and the whole toolkit loses its footing at once. That is why it is worth understanding precisely — not as one technique among many, but as the shared assumption the techniques stand on. When a method surprises you by failing, the first question to ask is whether the concept you targeted was linear in the first place.

## 1. From word analogies to a hypothesis

The first crack of linearity showed up a decade before anyone was probing transformers, in static word embeddings. Mikolov and colleagues noticed in 2013 that word2vec vectors satisfied analogies as **vector arithmetic**: the vector for "king", minus "man", plus "woman", lands closest to "queen". The same offset that carries "man" to "woman" also carries "king" to "queen" and "uncle" to "aunt". A gender concept was a fixed vector you could add or subtract.

![Word2vec analogies work because each concept is a constant offset vector shared across pairs.](/imgs/blogs/linear-representation-hypothesis-2.webp)

The figure makes the structure explicit. Lay the four words at the corners of a parallelogram. The horizontal edges are the *gender* offset — the same displacement vector, reused for the royal pair and the common pair. The vertical edges are the *royalty* offset — again, one shared direction. Because both offsets are constant, the four points form a parallelogram, and "king − man + woman" is just walking the gender edge and arriving at queen. The analogy is not a lookup; it is geometry.

Here is the whole phenomenon in runnable code, using GloVe vectors so you can reproduce it in a minute:

```python
import gensim.downloader as api
import numpy as np

wv = api.load("glove-wiki-gigaword-300")     # 300-d GloVe, 6B tokens (Wikipedia + Gigaword)

analogy = wv.most_similar(positive=["king", "woman"], negative=["man"], topn=3)
print(analogy)   # [('queen', 0.78), ('throne', 0.62), ('princess', 0.60)] -- arithmetic, not lookup

gender_royal  = wv["queen"] - wv["king"]     # the gender offset, measured on two different pairs
gender_common = wv["woman"] - wv["man"]
cos = gender_royal @ gender_common / (np.linalg.norm(gender_royal) * np.linalg.norm(gender_common))
print(f"offset agreement (cosine): {cos:.2f}")   # ~0.70 -- the offsets line up closely
```

Two things to internalize. First, the offsets are not *identical* — the cosine is around 0.7, not 1.0 — so even in 2013 the linear structure was approximate, a strong tendency rather than a law. Second, and more importantly: word2vec was trained with no notion of analogies in its objective. Linearity *emerged*. That is the template for everything since — we keep finding linear structure that nobody asked the optimizer to produce.

Why were the offsets stable enough to do arithmetic with? Omer Levy and Yoav Goldberg gave the cleanest answer: skip-gram with negative sampling is implicitly factorizing a shifted pointwise-mutual-information (PMI) matrix of word co-occurrences. Under that factorization, the difference vector between two words approximates a difference of log-co-occurrence profiles, and when two pairs share a relation (man:woman, king:queen) those profile-differences line up. The linearity was not a quirk of the neural network; it was a shadow of a linear-algebraic structure in the co-occurrence statistics themselves. This matters because it foreshadows the modern story: linear concept structure tends to appear whenever a predictive objective has to compress relational regularities, and a transformer's next-token objective is exactly that kind of objective at enormous scale. A fair caveat, raised by later analyses: analogy benchmarks flatter the phenomenon, because the standard evaluation excludes the three input words from the candidate answers, quietly doing some of the work. The structure is real, but it is a strong regularity with soft edges, not a theorem — the same hedge that follows the LRH everywhere.

From these observations crystallized the two-part hypothesis, stated most influentially in Anthropic's toy-models line of work:

- **Decomposability.** The representation can be broken into features that can be understood independently of one another.
- **Linearity.** Each feature corresponds to a direction in activation space, and the activation is (approximately) a linear combination of the active features' directions, scaled by their intensities.

It is worth separating two strengths of the claim, because people argue past each other constantly:

| Version | Claim | Status |
| --- | --- | --- |
| **Weak LRH** | *Some* important, human-meaningful concepts are represented as linear directions | Overwhelmingly supported; this is the workhorse assumption behind probing and steering |
| **Strong LRH** | *All* features are linear directions, and the network's ontology is fully a sum of 1-D directions | False as stated; circular and multi-dimensional features are documented counterexamples |

Almost all practical interpretability runs on the weak version. The strong version is a useful idealization and a research target, but treating it as a guarantee is how you get burned. We will spend the last third of this post on exactly where it breaks.

## 2. What "linear" actually means: three notions

"Concepts are linear" sounds like one statement. It is at least three, and they are not automatically equivalent. The cleanest formalization is due to Kiho Park, Yo Joong Choe, and Victor Veitch in *The Linear Representation Hypothesis and the Geometry of Large Language Models* (2024), who pull the single slogan apart into a **subspace** notion, a **measurement** notion, and an **intervention** notion.

![Park, Choe and Veitch split the hypothesis into subspace, measurement and intervention claims.](/imgs/blogs/linear-representation-hypothesis-3.webp)

Walk the three columns of the figure, because the distinctions are exactly what trips people up in practice.

- **Subspace.** A concept is linearly represented if there is a direction (or low-dimensional subspace) that "carries" it: the representation lives in, or varies along, that subspace. This is the weakest, most geometric statement — it just says the direction exists.
- **Measurement (probing).** A concept is linearly *measurable* if a linear function of the activation recovers its value: the projection $\hat{w}^\top h$ monotonically tracks how much of the concept is present. This is what a linear probe tests. It is a statement about **reading**.
- **Intervention (steering).** A concept is linearly *manipulable* if adding a vector to the activation changes that concept — and ideally only that concept — in the model's output: $h \leftarrow h + \alpha v$ flips French to English without touching tense or sentiment. This is a statement about **writing**.

The reason to keep these separate is that **a direction can be readable but not writable, or writable but not cleanly so**. You can have a probe that achieves 99% accuracy on a concept while the corresponding intervention does nothing useful, because the probe latched onto a direction that *correlates* with the concept in your dataset but is not what the model's later layers actually consume. This is the single most common probing mistake, and it has a name we will return to: the probe found a *spurious* or *non-causal* direction.

Park, Choe, and Veitch's contribution is to define concepts carefully — via counterfactual pairs, so a concept $W$ is a contrast like male$\to$female or English$\to$French — and then prove that under the right inner product (next section), the measurement notion and the intervention notion become **dual**. The direction you steer with and the direction you probe with are connected by a fixed isomorphism. That duality is what licenses the two-headed arrow in the hero diagram: read and write really are two ends of one object, but *only* once you fix the geometry correctly. Get the geometry wrong and the duality silently breaks.

A practical checklist falls out of this taxonomy. Whenever someone claims a concept is "linear", ask which notion they verified:

1. Did they only visualize a separation (subspace)? Weakest evidence.
2. Did they train a probe and report accuracy (measurement)? Better, but probes overfit to spurious directions.
3. Did they intervene and measure a *causal* effect on outputs (intervention)? Strongest. This is the gold standard, and it is why every serious steering paper reports causal ablation, not just probe accuracy.

To make "concept" precise enough to prove things about, Park, Choe, and Veitch define it through counterfactuals. A concept variable $W$ is a *pair* of interventions on the output — the contrast between the world where the next word is "queen" and the world where it is "king", holding everything else fixed. A concept is then linearly represented if there is a single vector whose addition realizes that counterfactual change in the output distribution. This counterfactual framing is the hinge: it lets them connect the output-side (unembedding) representation to the input-side (residual-stream) representation, and it is why the measurement and intervention notions can be *proven* dual rather than merely observed to coincide on a particular dataset. The payoff is hygiene. When you say "the truth direction", you now have a sharp question to ask: is there a vector whose addition flips the model's truth-counterfactual, and does projecting onto that same vector recover truth? If yes to both, the concept is linear in the strong, useful sense. If only the projection works, you have found a *correlate*, not the concept — and correlates are exactly the directions that read well and steer nothing.

## 3. The geometry problem: why cosine similarity lies

Here is a subtlety that almost everyone skips and then pays for. The residual stream has no privileged inner product. The model is invariant to a large class of invertible linear transformations of its hidden space: if you apply an invertible matrix $A$ to every activation and $A^{-1}$ to every weight that reads from it, you get the identical model. That means "the angle between two concept directions" — the thing cosine similarity measures — is **not well-defined** until you choose a metric, and the default Euclidean choice is arbitrary. Two concepts that are genuinely independent in the model's causal structure can have a large raw cosine simply because the coordinate axes are skewed.

![Euclidean cosine is meaningless until you whiten activations by the unembedding covariance matrix.](/imgs/blogs/linear-representation-hypothesis-4.webp)

The before/after split is the fix. On the left, raw representation space: the cosine between the gender direction and the tense direction is some nonzero number, the axes are skewed by the statistics of the unembedding, and projecting onto one direction bleeds in the other. On the right, after a change of inner product, causally separable concepts become orthogonal and projection reads one concept cleanly.

Park, Choe, and Veitch identify the right inner product — they call it the **causal inner product** — as the one that makes causally separable concepts orthogonal. Concretely, a workable choice is to whiten by the covariance of the unembedding vectors. Let $\gamma_t$ be the unembedding (output) vector for token $t$; set $M = \mathrm{Cov}(\gamma)^{-1}$, and define

$$\langle a, b\rangle_C = a^\top M\, b.$$

This is a Mahalanobis inner product: it stretches and rotates the space so that the directions the model actually uses to write logits are put on equal, decorrelated footing. Under $\langle\cdot,\cdot\rangle_C$, the unembedding representation of a concept and its intervention representation coincide, which is the duality from the previous section made concrete.

In code, the whitening is mechanical:

```python
import numpy as np

U = model.get_output_embeddings().weight.detach().cpu().numpy()   # unembedding W_U, shape (V, d)

Sigma = np.cov(U, rowvar=False)                 # (d, d) unembedding covariance
W = np.linalg.cholesky(np.linalg.inv(Sigma))    # whiten by Sigma^{-1} = W @ W.T

def causal_cos(a, b):                            # causal inner product = cosine after whitening
    aw, bw = W.T @ a, W.T @ b
    return (aw @ bw) / (np.linalg.norm(aw) * np.linalg.norm(bw))

raw = (v_gender @ v_tense) / (np.linalg.norm(v_gender) * np.linalg.norm(v_tense))
print(f"raw cosine    : {raw:+.2f}")             # e.g. +0.31, misleading
print(f"causal cosine : {causal_cos(v_gender, v_tense):+.2f}")  # ~0.00 if separable
```

Why should you care, beyond mathematical hygiene? Three concrete consequences:

- **Steering interference.** If you steer with a vector that is non-orthogonal (in the causal metric) to an unrelated concept, you drag that concept along. Whitening tells you which directions are *actually* independent so you can steer one without corrupting another.
- **Hierarchy.** In follow-up work on categorical and hierarchical concepts, the same metric reveals that hierarchies are encoded geometrically: "mammal" and "dog" are not orthogonal — "dog" lives *inside* the "mammal" direction's cone — and categorical variables like {dog, cat, fish} form simplices. None of that structure is visible under raw Euclidean cosine.
- **Comparing directions across methods.** When a probe direction and a steering direction "disagree" at 40 degrees of raw cosine, they may be identical in the causal metric. Many reported probe-vs-steer mismatches are coordinate artifacts.

To ground the hierarchy point with a picture you can hold: under the causal metric, "is an animal" is a direction, and "is a dog", "is a cat", "is a fish" are directions that all share a positive component along the animal direction while being mutually spread out in the orthogonal complement. The general category is the shared axis; the specific members fan out as a simplex inside the half-space that axis defines. So "dog minus animal" is meaningful — it is the part of dog-ness that is *not* explained by animal-ness — and you can read or steer the general and the specific concepts somewhat independently because they occupy related but distinct directions. This geometry is invisible if you compute raw cosines, where "dog" and "animal" just look "somewhat similar" at some uninterpretable angle. The metric is what turns a vague similarity into a readable hierarchy, and it is why the follow-up work on categorical and hierarchical concepts could make quantitative claims about taxonomy structure that raw embeddings could only gesture at.

> If your interpretability result depends on a cosine, ask which inner product you used. If the answer is "the default", you have an unstated assumption doing load-bearing work.

The takeaway is not that you must always whiten — in practice many results survive the raw metric because the unembedding covariance is not *that* far from isotropic. The takeaway is that linearity is a statement about a vector space *with a metric*, and the metric is a modeling choice you are responsible for.

## 4. Reading a direction: linear probes

Now the practical payoff. If a concept is a direction, finding it can be as cheap as one subtraction. The **mass-mean** (difference-in-means) probe is the workhorse, and it is almost insultingly simple.

![Mass-mean probing subtracts class means at one layer to recover a concept's reading direction.](/imgs/blogs/linear-representation-hypothesis-5.webp)

Follow the pipeline. Build a contrast dataset of paired positive and negative examples — prompts that are true vs false, polite vs rude, English vs French. Run the model and grab the residual-stream activation $h$ at one layer $L$ (usually the last token). Take the difference of the class means, $d = \mu_{\text{pos}} - \mu_{\text{neg}}$, normalize it, and you have a reading direction. Score any new activation by its projection $d^\top h$; the sign tells you which side of the concept it is on.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

mu_pos, mu_neg = H_pos.mean(0), H_neg.mean(0)   # H_pos/H_neg: (n, d) acts at layer L, last token
d = mu_pos - mu_neg                       # mass-mean (difference-in-means) direction
d = d / np.linalg.norm(d)                 # unit reading direction

def concept_score(h):
    return h @ d                          # > 0 -> positive side, < 0 -> negative side

X = np.vstack([H_pos, H_neg])             # logistic regression is the supervised cousin
y = np.r_[np.ones(len(H_pos)), np.zeros(len(H_neg))]
clf = LogisticRegression(max_iter=1000).fit(X, y)

print("mass-mean train acc:", ((concept_score(X) > 0) == y).mean())
print("logistic  train acc:", clf.score(X, y))
```

You will often find the mass-mean probe within a point or two of a fully trained logistic regression — and it generalizes *better* out of distribution, because it cannot exploit spurious within-dataset correlations the way a discriminative classifier can. That is a recurring, counterintuitive result: the dumb estimator wins on transfer. Marks and Tegmark's truth work leaned on exactly this property.

To make the activation-extraction concrete, here is the setup with TransformerLens, which exposes the residual stream by name:

```bash
pip install "transformer-lens>=2.0" scikit-learn torch transformers
python -c "import transformer_lens as tl; m = tl.HookedTransformer.from_pretrained('gpt2'); \
  _, cache = m.run_with_cache('The movie was great'); \
  print(cache['blocks.6.hook_resid_post'].shape)"   # -> torch.Size([1, 5, 768]), resid @ layer 6
```

### Mass-mean is one of several probes — know the menu

The difference-in-means estimator is the default, but it is one point in a small design space, and the right choice depends on whether you care about reading accuracy, causal fidelity, or label-free discovery.

| Probe | How the direction is found | Strength | Watch out for |
| --- | --- | --- | --- |
| Mass-mean (diff-in-means) | $\mu_{\text{pos}} - \mu_{\text{neg}}$ | Cheap, causal-ish, transfers well OOD | Assumes the classes differ only in the concept |
| Logistic regression | Gradient descent on cross-entropy | Highest in-distribution accuracy | Latches onto spurious directions; transfers worst |
| LDA | Between-class over within-class scatter | Accounts for class covariance | Needs a stable covariance estimate |
| PCA on the contrast | Top component of paired differences | Denoises, partly label-free | The top component may not be the concept |
| CCS (contrast-consistent search) | Unsupervised consistency constraint | Needs no labels | Famously unstable; finds prominent-but-wrong directions |

The pattern across the table: the more a method is allowed to *fit*, the better it scores in-distribution and the worse it transfers and steers. Mass-mean's refusal to fit is the feature, not the bug. When a logistic probe and a mass-mean probe disagree on the direction, trust the one that *steers*, not the one that *scores*.

A useful diagnostic is to compute the cosine — in the causal metric — between the directions each method returns. If they cluster tightly, the concept is robustly linear and you can use the cheapest method. If they scatter, you are probably reading a confound, and no amount of probe sophistication will save you; fix the dataset instead. This is also why ensembles of probes rarely help: if five methods agree, you did not need five; if they disagree, averaging a confound with a concept gives you a blurrier confound.

### Second-order optimization: the probe you trust is the one you can falsify

A probe with 99% accuracy is not evidence that the model *uses* the direction. It is evidence that the direction correlates with the label *in your dataset*. Two failure modes recur:

- **Spurious features.** If every "true" statement in your set happens to be about geography and every "false" one about chemistry, your "truth" probe is partly a "geography" probe. The fix is contrast pairs that differ *only* in the concept — minimal pairs — so the difference of means cancels everything else.
- **Confounded layers.** The best probe layer for *reading* is not always the best layer for *causing* behavior. Probe accuracy typically peaks in the middle-to-late layers; the causal locus may be earlier. Always cross-check a probe direction with an intervention (next section). A direction that reads perfectly but steers nothing was never the concept.

The discipline here is borrowed straight from causal inference: a probe is an observational estimator, and observational estimators are confounded until proven otherwise. The proof is intervention.

## 5. Writing a direction: activation steering

If reading is projection, writing is addition. **Activation steering** (also called activation addition, or representation engineering when done systematically) takes a concept direction $v$ and adds a scalar multiple of it into the residual stream during the forward pass. No fine-tuning, no gradient steps, no new parameters — you mutate the activations as they flow.

![Steering injects a contrast-derived vector into the residual stream at a single chosen layer.](/imgs/blogs/linear-representation-hypothesis-6.webp)

The figure shows the full apparatus. The model runs as usual: tokens, embedding, the stack of transformer blocks, the unembedding to logits. The intervention is a single edit at block $k$: $h \leftarrow h + \alpha v$. The vector $v$ comes from the same contrast-pair recipe as the probe — $v = \text{mean}(h^+) - \text{mean}(h^-)$ over positive and negative prompts. The scalar $\alpha$ is a strength knob: too small and nothing changes, too large and the model degenerates into repeating the concept word.

Here is a complete, runnable steering hook in Hugging Face Transformers:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "meta-llama/Llama-3.2-1B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="cuda")

LAYER, ALPHA = 8, 6.0
v = steering_vector.to(model.device, torch.float16)   # (d,) = mean(h+) - mean(h-)

def add_vector(module, inputs, output):
    hidden = output[0] if isinstance(output, tuple) else output
    hidden = hidden + ALPHA * v                        # broadcast over (batch, seq, d)
    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

handle = model.model.layers[LAYER].register_forward_hook(add_vector)
prompt = tok("I think that", return_tensors="pt").to(model.device)
print(tok.decode(model.generate(**prompt, max_new_tokens=40)[0], skip_special_tokens=True))
handle.remove()
```

Swap in a "love − hate" vector and the continuations turn warm; a "formal − casual" vector and they stiffen up; a "refusal" vector and the model starts declining. This is the Contrastive Activation Addition (CAA) recipe, and the striking part is how little it takes — a few dozen contrast pairs and one hook.

The named steering methods are all variations on this one move, and it helps to keep them straight because the literature uses the names loosely. ActAdd (Turner and colleagues) sources the vector from a *single* natural-language contrast pair and adds the raw activation difference — the minimum viable steer. CAA (Rimsky and colleagues) averages the difference over *many* contrastive multiple-choice pairs, which denoises the vector and makes the strength more predictable. Representation Engineering (Zou and colleagues) systematizes the whole read-and-control loop across many attributes at once and reads directions with a tomography step rather than a plain mean. Plain mass-mean steering — the $\mu_{\text{pos}} - \mu_{\text{neg}}$ we have used throughout — is the shared core of all of them. They differ in how they *source* the vector (one pair versus many, raw versus averaged versus PCA-style) and in *where* along the stack they inject it, but every one is placing the same bet: the concept is a direction, and addition is the write operation. A useful consequence: when one method works and another fails on the *same* concept, the culprit is almost always the contrast set, not the algorithm. Garbage pairs in, garbage direction out — the quality of your minimal pairs dominates the choice of method by a wide margin.

### Steering is the causal test the probe could not give you

The reason steering matters beyond being a neat trick is that it is the **intervention** notion from section 2 made operational. If adding $v$ reliably changes the targeted behavior and little else, you have causal evidence that $v$ is the concept's direction — not merely correlated with it. This is why the strongest interpretability claims (refusal, truth, sycophancy) always pair a probe with a steering or ablation experiment.

A few hard-won practical rules:

- **Layer choice dominates.** Early layers are too token-bound; late layers are too committed. The sweet spot is usually the middle third. Sweep it; do not guess.
- **Per-token vs prompt-wide.** Adding $v$ at every position is blunt. Adding it only at the last few positions, or only after the prompt, is often cleaner.
- **Ablation is the dual of addition.** To *remove* a concept, project it out: $h \leftarrow h - (v^\top h)\, v$ at every layer. This directional ablation is how the refusal-direction jailbreak works — you do not add anything, you subtract the model's ability to express the concept at all.
- **Norm matters.** Steering strength interacts with the residual stream's growing norm across layers. A fixed $\alpha$ that works at layer 8 may be invisible at layer 20. Scale $\alpha$ to the local activation norm if you steer deep.

### A gallery of steering failures

Steering looks magical in demos and then bites you in production. The recurring failure modes, each of which I have watched waste an afternoon:

- **Over-steering collapse.** Push $\alpha$ too high and the model stops being a language model: it emits the concept token on repeat ("bridge bridge bridge"), or its perplexity explodes into word salad. The usable window between "no visible effect" and "degenerate" can be narrow. Sweep $\alpha$ and watch fluency, not only the target metric — a steering result that tanks perplexity is a bug report, not a win.
- **Off-target drag.** A steering vector that is not orthogonal (in the causal metric) to neighboring concepts pulls them along for the ride. Steering "formality" up can quietly suppress "humor"; steering "happiness" can leak into "agreeableness". Always measure the side effects, not just the intended effect, by projecting the steered activations onto a few unrelated directions.
- **Position sensitivity.** Adding the vector at the prompt tokens versus at the generated tokens gives different behavior. For many concepts, steering only the post-prompt positions is cleaner, because the prompt's own content should not be rewritten under you.
- **Layer cliffs.** A vector that works beautifully at layer 8 does nothing at layer 6 and breaks the model at layer 14. The causal window is layer-specific; a steering vector is really a `(direction, layer, alpha)` triple, and reporting only the direction is reporting one third of the answer.
- **Manifold extrapolation.** The model never saw an activation like $h + \alpha v$ during training, so you are extrapolating off the data manifold, and extrapolation is where linearity's approximation error compounds fastest. Conditional and input-dependent steering — scaling $\alpha$ per token by how much of the concept is already present — exists precisely to keep the edit closer to activations the model actually expects.

The throughline is that steering is strictly harder than probing: reading a direction is an observation that leaves the model untouched, while writing one perturbs a high-dimensional dynamical system and hopes the perturbation stays in the linear regime. It usually does, within a window. Find the window before you trust the edit.

> A probe tells you a direction is *there*. Steering tells you the model *listens* to it. Only the second one is interpretability.

## 6. Where the directions come from: superposition and SAEs

So far we have assumed the directions exist and gone looking for the ones we already have names for. But a model has thousands of features and only so many dimensions. How does it fit them, and how do we discover the ones we *don't* have names for?

The answer is **superposition**: the network represents far more features than it has dimensions by packing them as not-quite-orthogonal directions, tolerating a little interference because any given input activates only a sparse handful of features at once. The full story is in the companion post on [superposition](/blog/machine-learning/ai-interpretability/what-is-superposition); here we need just the consequence — individual neurons are polysemantic, so the features are directions in the full space, and recovering them is a dictionary-learning problem.

![Dictionary learning trades dense polysemantic neurons for a wide, sparse, monosemantic feature basis.](/imgs/blogs/linear-representation-hypothesis-7.webp)

The before/after contrast is the heart of it. On the left, the dense neuron basis: a single neuron fires for DNA *and* HTTP requests *and* war, there are more features than dimensions, and directions overlap and interfere. On the right, what a **sparse autoencoder** (SAE) recovers: an overcomplete dictionary (often ~30x wider than the residual dimension) where each feature is a single, ideally monosemantic, direction, and only a few features are active per token.

| | Dense neuron basis | SAE feature basis |
| --- | --- | --- |
| Width | $d$ (e.g. 4096) | $m \gg d$ (e.g. 130k) |
| Per-token activity | Most neurons nonzero | A few dozen features nonzero |
| Interpretability | Polysemantic, hard to name | Often monosemantic, nameable |
| What it is | The model's native, accidental basis | A learned, sparse, human-aligned basis |

An SAE is trained to reconstruct the activation from a sparse code, which mechanizes the decomposability assumption of the LRH directly:

```python
import torch, torch.nn.functional as F

def sae_encode(h, W_enc, b_enc, k=32):     # top-k SAE over residual activations h (d-dim)
    pre = h @ W_enc + b_enc            # (m,) feature pre-activations, m ~ 30*d, overcomplete
    topk = torch.topk(pre, k).indices  # keep only the k strongest features (the sparsity)
    f = torch.zeros_like(pre)
    f[topk] = F.relu(pre[topk])
    return f                           # sparse code: only k of m features fire

def sae_decode(f, W_dec, b_dec):
    return f @ W_dec + b_dec           # each active f[i] adds one row of W_dec, a feature direction
```

The decoder weight $W_{\text{dec}}$ is the dictionary: each row is one feature direction. When Anthropic scaled this to Claude 3 Sonnet, the dictionary contained millions of features, many strikingly specific — and crucially, clamping a feature's activation *causally* changed behavior, which is the steering test again. SAEs are the LRH's industrial-scale instrument: instead of finding the one direction you already named, they enumerate the directions and let you name them afterward.

Dictionary width is the knob that controls how fine-grained the enumeration gets, and it interacts with the LRH in a revealing way. At a narrow width, a single feature might fire for "Golden Gate Bridge" in general; widen the dictionary and that feature splits into "the bridge in fog", "the bridge at sunset", "driving across the bridge", each a separate direction. This feature splitting is not obviously a bug — the model may genuinely carry those finer distinctions — but it means the "number of features" is partly a property of your SAE, not a fixed count you are discovering. The directions are real; their granularity is negotiable. The practical upshot is that an SAE gives you a *basis*, sparse and mostly interpretable, at a resolution you chose, and the right resolution depends on the question. For coarse behavioral control, a narrow dictionary with broad features is easier to steer; for fine-grained auditing, a wide dictionary surfaces distinctions a narrow one blends together. Either way, the underlying commitment is the LRH: that the activation is a sparse sum of directions, and that recovering those directions is the path to reading the model's mind.

### Which sparse autoencoder, and how do you know it worked?

"SAE" names a family, not one model, and the variants trade off reconstruction fidelity against how sparse and how interpretable the features come out:

| Variant | Sparsity mechanism | Note |
| --- | --- | --- |
| Vanilla (L1) | L1 penalty on feature activations | Simple, but shrinks magnitudes downward (a bias) |
| Top-k | Keep the $k$ largest pre-activations | Fixes the active count exactly; no magnitude shrinkage |
| Gated | Separate "which fires" and "how much" paths | Decouples selection from magnitude, improving fidelity |
| JumpReLU | Learned per-feature activation threshold | Strong fidelity/sparsity frontier; discrete threshold |

Picking a variant is the easy part. Evaluating the result is the hard part, because there is no ground-truth feature set to compare against — you are doing unsupervised learning and grading your own homework. The metrics people actually use:

- **Reconstruction loss** and, more tellingly, the **loss recovered** when you patch the SAE's reconstruction back into the model: does the model still behave if you route its activations through the dictionary and back?
- **L0**, the average number of active features per token — the sparsity you actually achieved, as opposed to the one you penalized for.
- **Automated interpretability scores**: have a second model predict a feature's activation from its top-activating text examples; high predictability means the feature is coherent enough to name.
- **Steering and ablation efficacy**: clamp the feature and measure the causal effect — the same intervention test that governs every other claim in this post.

The uncomfortable result from recent benchmarking is that on several downstream concept-detection and steering tasks, *simple supervised baselines* — a mass-mean or logistic probe on the raw residual stream — match or beat SAE features. SAEs win decisively on *discovery* (surfacing features you did not know to look for) and on *enumeration at scale*; they do not automatically win on any specific concept you can already name. The practical rule writes itself: a probe when you know what you want, an SAE when you do not.

The honest caveat — and we will return to it — is that SAEs are not a clean readout of the model's "true" features. They suffer **feature splitting** (one concept fractures into many narrow features as you widen the dictionary) and **absorption** (a general feature quietly absorbs a specific one). The dictionary is a useful, sparse, *approximate* basis, not ground truth.

## 7. Why training produces linear representations

It is one thing to find linear structure; it is another to explain why gradient descent keeps building it. The most satisfying account so far is Yibo Jiang and colleagues' *On the Origins of Linear Representations in Large Language Models* (2024), which shows that the combination of the next-token objective and the implicit bias of gradient descent is *enough* to produce linear, near-orthogonal concept directions.

![Next-token log-odds matching plus gradient descent's max-margin bias create linear concept directions.](/imgs/blogs/linear-representation-hypothesis-8.webp)

The causal chain in the figure is the argument in miniature. Posit a latent variable model: contexts and next tokens are generated from underlying binary concept variables $W$ (on/off — "is this French", "is this past tense"). Two forces then act. First, **log-odds matching**: a softmax trained with cross-entropy is driven to match the log-odds of the data-generating process, so the logit gap for a concept becomes a linear function of whether $W$ is on. Second, the **implicit bias of gradient descent**: on separable data, GD on the cross-entropy loss converges to the *max-margin* solution (the classic Soudry et al. result), which prefers low-rank, aligned directions. Put them together and the concept $W$ ends up represented as a single linear direction $v_W$, and distinct concepts end up approximately orthogonal.

A little more precisely. Let the model assign logits $\ell_t = \gamma_t^\top h$ for token $t$. If the data's log-probability ratio between a counterfactual pair $(t^+, t^-)$ depends linearly on a latent concept value, then minimizing cross-entropy pushes $(\gamma_{t^+} - \gamma_{t^-})^\top h$ to track that concept — which is precisely a linear representation in the unembedding space. The implicit bias then does the orthogonalization: among the many $h$-geometries that fit the data, GD selects a low-complexity one in which independent concepts do not share directions.

It is worth seeing why the max-margin bias produces *orthogonality* and not merely low rank. On linearly separable data, gradient descent on the logistic loss does not converge to any old separator — it converges, in direction, to the maximum-margin separator, the one that pushes the decision boundary as far from the data as possible (this is the Soudry, Hoffer, and Srebro result). When several concepts must be separated at once by a *shared* representation, the joint max-margin solution tends to place their directions so they do not interfere with one another, and "do not interfere" is, geometrically, "near-orthogonal". So the two forces split the labor cleanly: the objective wants each concept exposed to a linear readout, and the implicit bias wants those readouts to avoid stepping on each other. The joint optimum is a set of clean, roughly-orthogonal concept directions — which is precisely the structure the causal inner product was constructed to reveal. The theory of *why* linearity emerges and the geometry of *how* to measure it meet in the middle, and that agreement is a big part of why the field takes the hypothesis seriously rather than treating it as a lucky string of probe results.

Jiang and colleagues verify the theory two ways: they train models on synthetic data that exactly matches the latent variable model and watch linear representations appear, and they confirm predictions on LLaMA-2. The synthetic result is the important one — it shows the simple structure is *sufficient*; you do not need the full complexity of natural language to get linearity, just the next-token objective over concept-structured data. For the full treatment and its caveats, see the dedicated paper-reading post on [the origins of linear representations](/blog/paper-reading/ai-interpretability/on-the-origins-of-linear-representations-in-large-language-models).

This is the deepest reason the hero diagram is shaped the way it is. The unembedding is linear, the objective rewards exposing concepts to that linear map, and GD's bias keeps the exposed directions clean and separated. Linearity is not a coincidence we stumbled on; it is what this objective, optimized this way, tends to build.

## 8. A worked example: a truth direction, end to end

Written recipes are easy to nod along to and hard to trust until you have run one. Here is the full loop for building and *validating* a truth direction, which doubles as the template for any binary concept you care about.

**Step 1 — build minimal contrast pairs.** Take factual statements and their negations, matched so they differ only in truth value: "The Eiffel Tower is in Paris" (true) versus "The Eiffel Tower is in Rome" (false). Matching the surface form is exactly what makes the difference-of-means cancel everything except the truth value — same length, same topic, same syntax, one bit flipped.

**Step 2 — collect activations at every layer**, so you can pick the causal layer rather than guess it:

```python
import torch, numpy as np

def collect(prompts, layer):
    H = []
    for p in prompts:
        toks = tok(p, return_tensors="pt").to(model.device)
        out = model(**toks, output_hidden_states=True)
        H.append(out.hidden_states[layer][0, -1].float().cpu().numpy())  # last token
    return np.stack(H)

dirs = {}
for L in range(model.config.num_hidden_layers):
    d = collect(true_prompts, L).mean(0) - collect(false_prompts, L).mean(0)
    dirs[L] = d / np.linalg.norm(d)        # one candidate truth direction per layer
```

**Step 3 — score reading accuracy per layer** on held-out pairs, and find where it peaks. This is the measurement story, and it usually crests somewhere in the middle-to-late layers.

**Step 4 — validate causally**, the step everyone skips. Add the direction and check the model's own truth judgments shift; ablate it and check they collapse. If reading accuracy peaks at layer 18 but the causal effect peaks at layer 12, the *causal* layer is the one you report, because reading is observational and intervention is the proof:

```python
def truth_logit_shift(direction, layer, alpha):
    v = torch.tensor(direction, device=model.device, dtype=model.dtype)
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        h = h + alpha * v
        return (h,) + o[1:] if isinstance(o, tuple) else h
    handle = model.model.layers[layer].register_forward_hook(hook)
    shift = measure_true_false_logit_gap(eval_prompts)   # your eval harness
    handle.remove()
    return shift
```

**Step 5 — check side effects.** Steer truth and confirm you did not also move sentiment or formality, by projecting the steered activations onto those directions. A clean concept moves its own readout and leaves its neighbors where they were.

The whole loop is maybe fifty lines, runs on one GPU with a 7B model, and converts "I think truth is linear" into "here is the `(direction, layer, alpha)` triple, here is its reading accuracy, here is its causal effect on truth judgments, and here is the side-effect budget on three neighboring concepts." That last sentence is the entire difference between an interpretability *claim* and an interpretability *result*. If you only ever take one habit from this post, take Step 4.

## Case studies from the literature

Theory earns its keep when it predicts real measurements. Here are twelve results that, taken together, are the empirical backbone of the linear representation hypothesis — each a place where someone found a direction, tested it causally, and reported what broke.

### 1. word2vec analogies (Mikolov et al., 2013)

The origin story. Training a shallow skip-gram model on raw text, with an objective that knew nothing about analogies, produced embeddings where "king − man + woman" lands near "queen", "Paris − France + Italy" lands near "Rome", and "walking − walk + swim" lands near "swimming". Levy and Goldberg later showed the objective was implicitly factorizing a shifted pointwise-mutual-information matrix, which explained *why* offsets were stable. The lesson that propagated forward: linear concept structure can emerge from a generic predictive objective with no linearity in the loss, and it is robust enough to do arithmetic with. Every modern steering vector is the great-grandchild of this observation.

### 2. Othello-GPT and the coordinate frame (Li et al. 2023; Nanda 2023)

A transformer trained only to predict legal Othello moves builds an internal model of the board. Li and colleagues first probed it and concluded the board state was stored *nonlinearly* — only a nonlinear probe could read it. Neel Nanda then showed the twist: if you parameterize each square not as "black vs white" but as "mine vs theirs" (relative to whoever is to move), a *linear* probe reads the board cleanly, and you can causally edit the board by steering along those linear directions. The concept was linear all along; the first probe had the wrong coordinate frame. This is the single most instructive cautionary tale in the field — a failed linear probe is at least as often a parameterization bug as it is evidence of nonlinearity.

### 3. The Geometry of Truth (Marks and Tegmark, 2023)

Marks and Tegmark assembled datasets of clearly true and clearly false factual statements ("The city of Paris is in France" versus falsified variants) and showed that a mass-mean direction separates them in LLaMA activations with high accuracy. The decisive part was causal: patching activations along the truth direction flips the model's own judgment of whether a statement is true, and the direction generalizes across topics it was not fit on. They also documented that the dumb mass-mean estimator transfers better than logistic regression — the spurious-correlation argument from section 4, demonstrated. This work is why "truth direction" is now a default object people reach for.

### 4. Sentiment and the summarization motif (Tigges et al., 2023)

Sentiment turns out to be a single direction, recoverable by difference-in-means, and ablating it degrades sentiment-dependent behavior while leaving the rest intact. The surprising find was *where* the sentiment lives: it gets "summarized" onto a small number of token positions — commas, periods, and the end of clauses — rather than being smeared across every token. So the linear direction is real, but it is concentrated at structurally meaningful positions, which matters enormously when you decide *where* to read or steer. Linearity told them the direction existed; careful attribution told them which tokens carried it.

### 5. ActAdd: love minus hate (Turner et al., 2023)

Turner and colleagues' Activation Addition showed you can steer GPT-2 with a *single* contrast pair and no optimization at all: take the activations for "Love" and for "Hate", subtract, scale, add the result to the residual stream, and continuations shift toward affection. Their "weddings" example — steer with a wedding-themed contrast and the model starts working weddings into unrelated text — became the canonical demo. The importance is methodological: it proved steering needs neither a big dataset nor gradient descent, which made the intervention test cheap enough to run on every concept anyone cared about. The whole CAA ecosystem grows from this root.

### 6. Representation Engineering (Zou et al., 2023)

Zou and colleagues' RepE took a top-down stance: rather than reverse-engineering circuits, read and write high-level directions directly. Using Linear Artificial Tomography — essentially mass-mean probing on stimulus contrasts — they extracted reading vectors for honesty, power-seeking, fairness, and emotion, and built a working **lie detector** by reading the honesty direction while a model knowingly asserts falsehoods. They then *controlled* those concepts by adding the directions back. RepE reframed steering as a general control surface for safety-relevant attributes and showed the linear directions are stable enough to monitor a model in real time. It is the clearest demonstration that the LRH is not just descriptive but actionable for oversight.

### 7. Golden Gate Claude (Templeton et al., 2024)

Anthropic scaled sparse autoencoders to Claude 3 Sonnet and recovered millions of features, including one that fires on the Golden Gate Bridge across languages and modalities. When they clamped that single feature's activation to a large value, the model became obsessed: asked about anything, it steered the conversation to the bridge, even claiming to *be* the bridge. The demonstration was deliberately theatrical, but the science underneath is the intervention notion at industrial scale — one direction in a 34-million-feature dictionary, clamped, causally reshapes everything the model says. It is the most vivid existence proof that features are directions you can turn up and down. They also surfaced safety-relevant features (deception, sycophancy, dangerous code) the same way.

### 8. Refusal is mediated by a single direction (Arditi et al., 2024)

Arditi and colleagues found that across more than a dozen open chat models, the model's tendency to *refuse* a request is mediated by one direction. Two causal results clinch it. Adding the direction makes the model refuse harmless prompts; **ablating** it — projecting it out of the residual stream at every layer — strips the model's ability to refuse, jailbreaking it without any weight changes or prompt tricks. A single rank-one edit defeats the safety training. This is simultaneously the strongest causal evidence for the LRH (one direction, fully causal, robust across models) and a stark demonstration of why interpretability is a safety problem: a linearly represented behavior is a linearly *removable* one.

### 9. Space and time as linear world models (Gurnee and Tegmark, 2023)

Gurnee and Tegmark trained linear probes on LLaMA-2 activations and recovered the **latitude and longitude** of cities and the **dates** of events — a literal world map and timeline reconstructed from a model trained only to predict text. They found individual "space neurons" and "time neurons" that participate in these directions. The point for the LRH is that even continuous, structured, real-world quantities — not just binary concepts — are stored linearly enough that a linear probe recovers them with real metric accuracy. The model is not memorizing strings; it has built a low-dimensional linear coordinate system for the physical world, sitting in the residual stream waiting to be projected out.

### 10. In-context task vectors (Hendel et al.; Todd et al., 2023)

A few-shot prompt — several input/output examples of a task — gets compressed by the model into a single activation vector that *is* the task. Hendel and colleagues, and independently Todd and colleagues with "function vectors", showed you can extract this vector from the forward pass over the demonstrations, then *transplant* it into an unrelated zero-shot forward pass, and the model performs the task as if it had seen the examples. Antonyms, translation, capital cities — the task becomes a vector you can add. This pushes the linear representation hypothesis past static attributes into *operations*: not only "is this French" but "translate this to French" is a direction you can read off and inject. It is one of the cleaner demonstrations that the residual stream traffics in reusable, linearly-composable *function* representations, not just feature labels, and it is the mechanistic story underneath a lot of what looks like in-context learning.

### 11. Sycophancy is not one direction (causal separation work, 2023–2024)

A deliberate counterweight to the clean wins. "Sycophancy" sounds like a single trait, and you can fit a single steering vector for it — but causal separation work shows it is several behaviors (agreeing with a user's stated belief, flattering the user, caving under pushback, mirroring the user's framing) that do *not* all share one direction. Steer the vector you fit for "agree with the user's stated view" and you may not budge "back down when challenged". The lesson generalizes well beyond sycophancy: the granularity at which a concept is linear is an empirical question, and a single human-named trait can decompose into multiple directions you must separate before steering does what you actually meant. Treating a fuzzy human label as one direction is a quiet, common way for steering to under-deliver while your probe accuracy still looks great.

### 12. AxBench: simple baselines versus SAEs for steering (2025)

When researchers built a controlled benchmark to compare steering methods head to head — supervised difference-in-means probes, SAE-derived features, prompting, and fine-tuning — a blunt finding emerged: simple supervised steering vectors and even plain prompting often *outperformed* SAE features at actually controlling behavior, while SAEs kept their edge for unsupervised feature discovery. The result did not refute the linear representation hypothesis; it sharpened it. Linear directions for *known* concepts are best obtained by the cheapest supervised method that works, and the distinctive value of sparse autoencoders is enumeration of the *unknown*, not control of the known. It is the empirical capstone to the "use the right tool" theme that runs through this whole post: the hypothesis is right, and being right about *which* instrument exploits it is most of the practical skill.

## When the line bends: limits of the hypothesis

Nine clean wins can lull you into believing the strong LRH. Do not. The most intellectually honest part of the field is its growing catalog of cases where features are *not* one-dimensional directions, and a senior practitioner is defined by knowing them cold.

![A single direction cannot encode Monday-to-Sunday wraparound, so days form a ring while sentiment stays linear.](/imgs/blogs/linear-representation-hypothesis-9.webp)

The figure is the canonical counterexample, from Joshua Engels and colleagues' *Not All Language Model Features Are Linear* (2024). Some concepts are genuinely **circular**. The day of the week wraps around — Sunday is adjacent to Monday — and no single direction can encode that adjacency, because a 1-D projection cannot make the two ends of a line meet. The model instead represents days (and months, and clock positions) on a **2-D ring**: a circle, not an axis. The authors found these multi-dimensional, irreducible features by clustering SAE features and testing for circularity, then confirmed causally that the model *computes* modular arithmetic (like "two days after Friday") by rotating around the circle. A linear probe for "day of the week" is doomed; the feature is intrinsically two-dimensional. Sentiment and formality, by contrast, stay linear — the contrast in the figure is the whole point.

It is worth dwelling on *why* circularity defeats a linear probe, because the reason recurs. A linear probe outputs a single scalar — a position on a line. Monday and Sunday are far apart on any line you draw through the days, yet the model treats them as adjacent (the day after Sunday is Monday). No single scalar can be simultaneously "far in value" and "near in behavior", so the model is forced to use two coordinates and lay the days on a circle, where adjacency is angular and the wraparound costs nothing. The tell is that the feature's "intensity" is not monotonic: as you walk Monday to Sunday, the projection onto any fixed direction goes up and then comes back down. Whenever a probe's accuracy is high in the middle of a range and collapses at both extremes, suspect a hidden second dimension.

The deeper worry is that circularity is not the only non-linear structure, just the easiest to visualize. Engels and colleagues found genuinely multi-dimensional, *irreducible* features — concepts that need a small subspace and cannot be split into independent one-dimensional pieces without losing information. Identifying them is hard precisely because the default tools assume one-dimensionality: a mass-mean probe will happily return *a* direction for a 2-D feature, it will just be the wrong object, capturing a projection of the real structure and discarding the rest. The methodological upgrade is to stop assuming dimensionality and start measuring it — cluster the SAE features that co-activate, test whether the cluster is reducible, and only then decide whether "direction" or "subspace" is the right noun.

That is one breakage. There are several, and they cluster into a table worth memorizing:

| Failure mode | What goes wrong | What to do instead |
| --- | --- | --- |
| **Circular / cyclic features** | Days, months, time-of-day wrap; no single direction encodes adjacency | Look for 2-D ring structure; probe with a circle, not a line |
| **Multi-dimensional irreducible features** | The concept needs a subspace, not a direction | Cluster SAE features; test for irreducibility before assuming 1-D |
| **Representational drift** | The same concept's direction moves over a long conversation | Re-fit the direction per context; do not freeze a vector from turn one |
| **Causal faithfulness gap** | A linear account that fits probes can still fail as a faithful causal model | Demand a faithful causal model, not just probe accuracy |
| **SAE feature splitting / absorption** | The "feature" you found is a dictionary artifact, not the model's unit | Vary dictionary width; check stability of the feature under re-training |

Each of these has a literature. The drift problem is documented in work showing that [linear representations can change dramatically over a conversation](/blog/paper-reading/ai-interpretability/linear-representations-in-language-models-can-change-dramatically-over-a-conversation) — a steering vector calibrated on short prompts can quietly lose its grip as context accumulates, which is a real production hazard for anyone shipping steering as a control. The deeper, more philosophical challenge comes from the [non-linear representation dilemma](/blog/paper-reading/ai-interpretability/the-non-linear-representation-dilemma-is-causal-abstraction-enough-for-mechanistic-interpretability), which asks whether *any* linear-plus-causal story can be both faithful and non-trivial, or whether a sufficiently expressive mapping can "explain" anything and thereby explain nothing.

There is also a subtler statistical limit worth naming: linear representations correlate with **pretraining data frequency**. Concepts that are rare in the training corpus tend to have weaker, noisier, less linear directions — the geometry is cleanest exactly where the model has seen the most data, and degrades in the long tail. So "is this concept linear?" is not a yes/no property of the concept; it is partly a property of how often the model saw it.

> The weak linear representation hypothesis is a tool. The strong one is a faith. Ship the tool; interrogate the faith.

The right stance is neither triumphalism nor dismissal. Linearity is a strong, productive prior that holds for a large and important class of concepts and fails in structured, predictable ways for another class. The skill is knowing which class you are in *before* you stake a decision on it.

## When to reach for the linear lens (and when not)

![The linear lens fits probing, steering and ablation but breaks on cyclic and drifting features.](/imgs/blogs/linear-representation-hypothesis-10.webp)

The decision matrix above is the field guide. Map your task to a row, and the columns tell you whether the linear lens fits and what to reach for if it does not.

**Reach for the linear lens when:**

- You need to **read a known, roughly-binary concept** — true/false, toxic/safe, English/French. A mass-mean probe is the first thing to try, and often the last.
- You need to **steer or control a behavior** cheaply and reversibly, without fine-tuning. CAA-style activation addition gets you a long way with a few contrast pairs.
- You need to **remove or audit a behavior causally** — directional ablation of a refusal or deception direction is a sharp, testable intervention.
- You want a **monitoring signal** for a safety-relevant attribute (honesty, sycophancy) that runs in real time during generation.
- You are **enumerating unknown features** at scale — SAEs are the right industrial instrument, with the caveat that their features are an approximate basis.

**Skip it (or proceed with heavy caution) when:**

- The concept is **cyclic or inherently multi-dimensional** — days, months, periodic quantities, anything with wraparound. A line cannot encode a circle.
- You are operating **far out of the distribution** your direction was fit on, or **deep in a long conversation** where the direction drifts. Re-fit, do not freeze.
- The concept is **rare in pretraining** — expect a weak, noisy direction and validate harder.
- You need a **faithful causal model**, not a control knob. A direction that steers behavior is not automatically a complete explanation of the mechanism; do not oversell a steering result as a circuit.
- Your only evidence is **probe accuracy**. Without an intervention, you have a correlation, and correlations in 4096 dimensions are cheap.

### A decision procedure you can run

When a new concept lands on your desk and someone asks "is it linear, and can we steer it?", here is the order of operations that keeps you honest:

1. **Build minimal contrast pairs.** Same surface form, one concept bit flipped. If you cannot construct clean pairs, you do not yet understand the concept well enough to probe it — that is information, not a blocker.
2. **Fit the mass-mean direction at every layer.** Cheap, and it gives you the whole depth profile in one pass. Reach for logistic regression only if mass-mean underperforms, and distrust the gain if it does.
3. **Score reading accuracy on held-out pairs.** Find where it peaks. This is necessary but never sufficient — a high score is a hypothesis, not a result.
4. **Intervene.** Add the direction and ablate it, and measure the causal effect on outputs at each candidate layer. Report the *causal* layer, not the readable one.
5. **Audit side effects.** Project steered activations onto neighboring concepts. A clean direction moves its own readout and little else.
6. **Stress the assumption.** Test the direction out of distribution, deep in a long context, and at the extremes of the concept's range. If accuracy is non-monotonic across the range, suspect a circular or multi-dimensional feature and stop treating it as a line.
7. **Only now, ship it** — as a `(direction, layer, alpha)` triple with a documented reading accuracy, causal effect, and side-effect budget.

Steps 1–3 are what most people do. Steps 4–6 are what separates an interpretability result from an interpretability anecdote. The discipline is not exotic; it is the ordinary discipline of causal inference, transplanted into a residual stream. Treat every direction as a suspect, and make it prove it is the concept rather than a correlate.

The throughline of this entire post: linearity is the assumption that makes interpretability *tractable*, and the strongest single bet in the field, but it is an assumption with an edge. The honest position is to hold it as a strong, falsifiable prior — strong enough to build probes, steering, lie detectors, and feature dictionaries on, falsifiable enough that you actively hunt for the circular, multi-dimensional, drifting, and rare-concept cases where it cracks. Find the direction, then try to break it. The directions that survive your attempts to falsify them are the ones worth building on, and the ones that do not survive are teaching you something about the model's true ontology — which is, after all, the entire point of looking.

## Further reading

- Park, Choe, Veitch — *The Linear Representation Hypothesis and the Geometry of Large Language Models* (2024): the subspace/measurement/intervention taxonomy and the causal inner product.
- Jiang et al. — *On the Origins of Linear Representations in Large Language Models* (2024): why next-token training plus GD's implicit bias yields linear directions. Companion post: [the origins of linear representations](/blog/paper-reading/ai-interpretability/on-the-origins-of-linear-representations-in-large-language-models).
- Elhage et al. — *Toy Models of Superposition* (2022) and the companion [superposition explainer](/blog/machine-learning/ai-interpretability/what-is-superposition): why features are directions and not neurons.
- Templeton et al. — *Scaling Monosemanticity* (2024): sparse autoencoders and the Golden Gate Bridge feature at production scale.
- Engels et al. — *Not All Language Model Features Are Linear* (2024): circular and multi-dimensional features, the most important counterexamples.
- Arditi et al. — *Refusal in Language Models Is Mediated by a Single Direction* (2024): the sharpest causal evidence, and a safety wake-up call.
- Marks and Tegmark — *The Geometry of Truth* (2023): mass-mean probing, causal patching, and why the dumb estimator transfers better.

Both limit posts — the dilemma and the conversation-drift result — are linked inline in the section on where the line bends, just above.

Then open `npm run dev`, pull up a small open model, and run the mass-mean probe and the steering hook above on a concept you care about. The hypothesis is most convincing when you watch a direction you built from two means turn into a behavior you can switch on and off.
