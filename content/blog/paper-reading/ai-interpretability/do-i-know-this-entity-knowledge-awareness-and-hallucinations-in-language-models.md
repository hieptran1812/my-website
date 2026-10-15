---
title: "Do I Know This Entity? How Language Models Notice What They Don't Know"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "A sparse autoencoder finds a direction in Gemma 2 that fires when the model does not recognize an entity, and steering along it switches the chat model between refusing and hallucinating."
category: "paper-reading"
subcategory: "AI Interpretability"
tags:
  - paper-reading
  - hallucination
  - knowledge-awareness
  - sparse-autoencoders
  - activation-steering
  - mechanistic-interpretability
  - large-language-models
  - iclr-2025-oral
author: "Hiep Tran"
featured: true
readTime: 18
image: "/imgs/blogs/do-i-know-this-entity-knowledge-awareness-and-hallucinations-in-language-models-fig1.webp"
excerpt: "Inside Gemma 2 there is a direction that lights up on entities the model cannot recall facts about. Push it up and the chat model refuses to talk about LeBron James; push the opposite one and it invents a birthday for a player who does not exist."
paper:
  title: "Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models"
  authors: "Javier Ferrando, Oscar Obeso, Senthooran Rajamanoharan, Neel Nanda"
  venue: "ICLR 2025 (Oral); arXiv 2024"
  url: "https://arxiv.org/abs/2411.14257"
---

> [!tldr]
> **Claim.** Gemma 2 carries linear directions, found with off-the-shelf sparse autoencoders, that fire on the last token of an entity depending on whether the model can recall facts about it. One fires on "known" entities, another on "unknown" ones, and both work across players, movies, cities and songs.
>
> **Why it matters.** These directions are causal. Adding the unknown-entity direction makes the chat model refuse to answer about LeBron James. Adding the known-entity direction makes it invent a birth date for a made-up player it would otherwise refuse on. That is a mechanistic handle on one kind of hallucination.
>
> **Most surprising.** The SAEs were trained on the *base* model's pretraining activations, yet their directions control refusal in the *chat* model. Chat finetuning seems to have wired refusal onto a recognition signal that already existed.
>
> **Where it is thin.** The causal story is clearest in Gemma 2 2B and weaker in 9B and Llama 3.1 8B. The "uncertainty" latent that predicts wrong answers reaches only 73.2 AUROC. And "known" is defined by a fuzzy string match on Wikidata attributes, which is a noisy label.

Ask a chat model "When was the player Wilson Brown born?" about a person who does not exist, and it either refuses or confidently produces a date. This paper asks what inside the model decides between those two. The answer is surprisingly clean: in the middle layers, on the final token of the entity name, the model writes down something like "I recognize this" or "I do not". Later machinery reads that signal, and editing it flips the behavior.

## The problem: hallucination about things the model never learned

Hallucinations have many causes: bad training data, outdated facts, sampling noise. The authors isolate one subset that is easy to operationalize: **questions about entities the model has no stored facts about.** For those, a well-behaved model should refuse. A badly-behaved one makes something up.

Prior mechanistic work on factual recall (Meng et al., Geva et al., Nanda et al.) mostly studied the *success* path: how "Michael Jordan" turns into "basketball". This paper studies the failure path.

The tool is a **sparse autoencoder (SAE)**, the dictionary-learning method that decomposes a hidden state into a sparse sum of interpretable directions. The authors do not train their own. They use Gemma Scope, a public suite of SAEs trained on every layer of Gemma 2 2B and 9B, and replicate on Llama 3.1 8B with LlamaScope.

![Figure 1 from Ferrando et al. (2025): SAE latents on the last entity token separate known from unknown entities, and raising the known-entity latent on a made-up athlete makes the model hallucinate](/imgs/blogs/do-i-know-this-entity-knowledge-awareness-and-hallucinations-in-language-models-fig1.webp)

Figure 1 is the whole paper in one picture. On the left, each dot is an SAE latent placed by how often it fires on unknown versus known entities. On the right, the model refuses on "Wilson Brown" until the known-entity latent is pushed up, after which he is a baseball player born on August 1, 1994.

## Technique 1: sparse autoencoders as a feature dictionary

**The problem.** A residual-stream vector in Gemma 2 2B has a few thousand dimensions, and individual neurons are polysemantic: each one responds to many unrelated things. You cannot read "does the model know this entity" off a single neuron.

**The analogy.** The hidden state is a chord built from a handful of notes. An SAE learns a large dictionary of possible notes and, for each chord, reports which few are playing and how loud.

**The mechanism.** The encoder maps the hidden state into a much wider space and applies a thresholded activation, so most entries are exactly zero. The decoder maps the few nonzero entries back to a reconstruction. Gemma Scope uses the JumpReLU variant:

$$
\mathrm{SAE}(x) = a(x)\,W_{\mathrm{dec}} + b_{\mathrm{dec}}, \qquad a(x) = \mathrm{JumpReLU}_{\theta}\!\left(x\,W_{\mathrm{enc}} + b_{\mathrm{enc}}\right)
$$

Here $x \in \mathbb{R}^{d}$ is the residual-stream vector (a row vector). $W_{\mathrm{enc}}$ has shape $d \times d_{\mathrm{SAE}}$ and $W_{\mathrm{dec}}$ has shape $d_{\mathrm{SAE}} \times d$, with $d_{\mathrm{SAE}} \gg d$. The vector $a(x) \in \mathbb{R}^{d_{\mathrm{SAE}}}$ holds the **latent activations**. The activation is $\mathrm{JumpReLU}_{\theta}(z) = z \odot H(z - \theta)$, where $H$ is the Heaviside step and $\theta$ is a learned per-latent threshold: zero below the threshold, identity above it.

Training trades reconstruction against sparsity:

$$
\mathcal{L}(x) = \underbrace{\lVert x - \mathrm{SAE}(x) \rVert_2^2}_{\text{reconstruction}} + \lambda\,\underbrace{\lVert a(x) \rVert_0}_{\text{sparsity}}
$$

The $L_0$ norm counts nonzero latents and $\lambda$ sets how hard sparsity is pushed. The paper uses the default sparsity per layer, the ones served on Neuronpedia.

**Tiny example.** Suppose $d = 3$ and the SAE has 6 latents. A hidden state $x = (2.0, 0.1, -1.0)$ might encode to $a(x) = (0, 0, 1.8, 0, 0, 0.4)$. Only latents 3 and 6 fire. The reconstruction is ${1.8}$ times row 3 of $W_{\mathrm{dec}}$ plus ${0.4}$ times row 6, plus the bias. If latent 3 is "unknown entity", you now have a readable number for it.

**When it fails.** SAEs are lossy and can split or absorb features in ways that do not match a human concept (see [how sparse autoencoders scale and how to evaluate them](/blog/paper-reading/ai-interpretability/scaling-and-evaluating-sparse-autoencoders)). The results depend on Gemma Scope happening to carve out a clean entity-recognition latent.

### Why an SAE latent is also a steering vector

Because the reconstruction is linear in the latents, raising latent $j$ by some amount is the same as adding a multiple of its decoder row $d_j = W_{\mathrm{dec}}[j,:]$ to the residual stream:

$$
x_{\mathrm{new}} \leftarrow x + \alpha\, d_j
$$

The scalar $\alpha$ is the steering coefficient. This one line is what turns a descriptive finding ("this latent fires on unknown entities") into a causal test ("does adding it change behavior?"). It rests on the [linear representation hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis): that concepts live as directions you can add.

## Technique 2: building a known/unknown dataset

**The problem.** To find a "known entity" direction you first need labeled examples of known and unknown entities, *as judged by this specific model*. Popularity on the internet is a proxy; what matters is whether Gemma itself can recall facts.

**The mechanism.** The authors pull four entity types from Wikidata, each with several attributes:

| Entity type | Number of entities | Attributes queried |
| --- | --- | --- |
| Player (basketball) | 7,487 | birthplace, birthdate, teams played |
| Movie | 10,895 | director, screenwriter, release date, genre, duration, cast |
| City | 7,904 | country, population, elevation, coordinates |
| Song | 8,448 | artist, album, publication year, genre |

For each (entity, attribute) pair they build a completion prompt such as "The movie 12 Angry Men was directed by" and let the base model fill it in. A fuzzy string match against Wikidata decides correctness.

![How each entity is labeled known, unknown or discarded before any latent is examined](/imgs/blogs/do-i-know-this-entity-knowledge-awareness-and-hallucinations-in-language-models-1.webp)

The labeling rule is deliberately asymmetric:

- **Known:** the model gets at least two attributes right.
- **Unknown:** the model gets every attribute wrong.
- **Discarded:** anything in between.

The surviving entities are split 50% train, 10% validation, 40% test. Dropping the ambiguous middle buys cleaner labels but ignores the partially-known entities where hallucination is most interesting.

**When it fails.** The authors flag this themselves. A model can guess a common attribute (a genre, a country) without really knowing the entity, and it can know an entity while missing the particular attributes Wikidata lists. The label is "recalls these attributes", which is narrower than "knows this entity".

## Technique 3: separation scores and the MaxMin search

**The problem.** Gemma Scope has tens of thousands of latents per layer and a layer for every block. Which latent tracks recognition, and does it track it for *all* entity types or just one?

**The analogy.** You want a smoke detector that works in every room of the house, not one that is excellent in the kitchen and deaf in the bedroom. So you score each detector by its *worst* room.

**The mechanism.** Run every prompt, take the residual stream at the **last token of the entity name**, and encode it with the layer's SAE. For latent $j$ at layer $l$, measure how often it is active on each subset:

$$
f^{\mathrm{known}}_{l,j} = \frac{1}{N^{\mathrm{known}}} \sum_{i=1}^{N^{\mathrm{known}}} \mathbb{1}\!\left[a_{l,j}(x^{\mathrm{known}}_{l,i}) > 0\right], \qquad f^{\mathrm{unknown}}_{l,j} = \frac{1}{N^{\mathrm{unknown}}} \sum_{i=1}^{N^{\mathrm{unknown}}} \mathbb{1}\!\left[a_{l,j}(x^{\mathrm{unknown}}_{l,i}) > 0\right]
$$

$N^{\mathrm{known}}$ and $N^{\mathrm{unknown}}$ count prompts in each subset, and $\mathbb{1}[\cdot]$ is 1 when the latent fires. The **separation scores** are differences of these frequencies:

$$
s^{\mathrm{known}}_{l,j} = f^{\mathrm{known}}_{l,j} - f^{\mathrm{unknown}}_{l,j}, \qquad s^{\mathrm{unknown}}_{l,j} = f^{\mathrm{unknown}}_{l,j} - f^{\mathrm{known}}_{l,j}
$$

A score near 1 means the latent fires on almost every entity of one kind and almost none of the other.

To demand generality, compute the score separately per entity type $t$ and keep the minimum. The chosen latents are

$$
\text{known latent} = \arg\max_{l,j}\, \min_{t}\, s^{\mathrm{known},t}_{l,j}, \qquad \text{unknown latent} = \arg\max_{l,j}\, \min_{t}\, s^{\mathrm{unknown},t}_{l,j}
$$

with one filter first: any latent active on more than 2% of random tokens from the Pile is excluded, so a generic "this is a noun" latent cannot win.

![Illustrative separation-score matrix: the winning latent has the best worst-case score across entity types](/imgs/blogs/do-i-know-this-entity-knowledge-awareness-and-hallucinations-in-language-models-2.webp)

**Tiny example.** The figure uses made-up numbers. Latent A scores 0.82 on players but 0.10 on movies, so its minimum is 0.10. Latent B never exceeds 0.66 but never drops below 0.58. MaxMin picks B. 

**What they found.** Separation rises through the network and peaks around layer 9 in Gemma 2 2B before plateauing, and the MaxMin curve follows. Early layers hold specialized, weaker latents; the middle layers hold general ones.

![Figure 2 from Ferrando et al. (2025): top-5 known and unknown separation scores per layer in Gemma 2 2B, with the MaxMin line showing the best entity-agnostic latent](/imgs/blogs/do-i-know-this-entity-knowledge-awareness-and-hallucinations-in-language-models-fig2.webp)

The paper's Table 1 shows what the winning latents do on real text. The known latent fires on "Michael Jordan", "LeBron James", "San Francisco", "12 Angry Men" and the Beatles' "Yellow Submarine". The unknown latent fires on "Michael Joordan", "Wilson Brown", "Anthon", "20 Angry Men" and "Turquoise Submarine". One typo is enough to move a famous name across the line.

### Two sanity checks against confounds

**Songs after the cutoff.** The authors collect 283 song titles released after August 2024, which the models should not know. The top latents behave as expected (Appendix R):

| Model | Known latent fires | Unknown latent fires |
| --- | --- | --- |
| Gemma 2 2B | 6% | 53% |
| Gemma 2 9B | 22% | 55% |
| Llama 3.1 8B | 13.4% | 76% |

**Token likelihood.** Maybe the "known" latent just tracks how predictable the name's tokens are. Appendix S correlates latent activation with token probability on FineWeb: the latents fire on few tokens (0.6% known, 0.5% unknown in Gemma 2 2B) and the correlations are negligible.

## Technique 4: steering the chat model into refusing or hallucinating

**The problem.** A correlated latent might be a passive readout that nothing downstream uses. Only an intervention can tell.

**The mechanism.** Switch from base to **chat** model (Gemma 2 2B IT), rephrase prompts as questions ("Who directed the movie 12 Angry Men?"), and detect **knowledge refusals** by string-matching stock phrases such as "Unfortunately, I don't have access to real-time information". Then add $\alpha d_j$ at the last entity token and the end-of-instruction tokens that follow. The coefficient is picked on validation data: $\alpha$ between 400 and 550, roughly twice the residual-stream norm at those layers.

![Steering in both directions: the same chat model refuses or invents depending on one added direction](/imgs/blogs/do-i-know-this-entity-knowledge-awareness-and-hallucinations-in-language-models-3.webp)

On 100 test questions about unknown entities, the experiment compares five conditions:

| Condition | What happens to refusal (Gemma 2 2B IT) |
| --- | --- |
| Original model | Some refusal, varying by entity type |
| Steer with unknown latent | Almost 100% refusal on every entity type |
| Steer with 10 random latents | Mean and spread shown as a control |
| Orthogonalized model (unknown direction removed) | Large drop in refusal |
| Steer with known latent | Refusal slightly reduced |

![Figure 3 from Ferrando et al. (2025): refusal rates of Gemma 2 2B on 100 unknown-entity questions under each intervention, and an induced refusal about LeBron James](/imgs/blogs/do-i-know-this-entity-knowledge-awareness-and-hallucinations-in-language-models-fig3.webp)

### Removing a direction from the weights

The orthogonalized model borrows from Arditi et al.'s refusal-direction work. Instead of steering activations at inference time, it edits every matrix that writes into the residual stream so it can no longer write along the unit direction $d$:

$$
W_{\mathrm{out}}^{\mathrm{new}} \leftarrow W_{\mathrm{out}} - W_{\mathrm{out}}\, d^{\top} d
$$

$W_{\mathrm{out}}$ is an output matrix whose rows live in residual space; $W_{\mathrm{out}} d^{\top}$ is each row's component along $d$, and multiplying by $d$ turns that component back into a vector to subtract. After the edit no component can put anything on that axis.

**Tiny example.** In 2D, let $d = (1, 0)$ and one row of $W_{\mathrm{out}}$ be $(3, 4)$. Its component along $d$ is 3, so the row becomes $(3, 4) - 3\,(1, 0) = (0, 4)$. Whatever that row writes, it now writes nothing on the first axis.

That this edit cuts refusals substantially is the strongest evidence that the unknown-entity direction is not just correlated with refusal but *used* for it.

### The base-to-chat transfer

Gemma Scope was trained on base-model activations over pretraining text. Knowledge refusal is taught during chat finetuning: the Gemma 2 report mentions adding data that encourages refusals. Yet the base-model direction steers the chat model's refusals. The authors' reading is that finetuning *repurposed* a recognition signal the base model already computed.

**When it fails.** The effect is much weaker in Gemma 2 9B and Llama 3.1 8B. And $\alpha$ of roughly twice the residual norm is a large push that could break a model in many directions; the random-latent control at the same layer and coefficient is the comparison that guards against that reading.

## Technique 5: activation patching and attention to the entity

**The problem.** Knowing the direction is causal does not say *how* it acts. Where in the factual-recall circuit does "I don't know this" plug in?

**The mechanism, part 1: find the recall circuit.** Activation patching runs a clean prompt (known entity) and a corrupted prompt (unknown entity), copies one activation from the clean run into the corrupted run, and measures how much of the clean answer comes back. With the clean prompt ending in "LeBron James joined the" (answer "Lakers") and the corrupted one predicting "Warriors", the score is

$$
\text{recovered} = \frac{\mathrm{logit}_{\mathrm{Lakers}-\mathrm{Warriors}}\!\left(\mathrm{corr} \mid \mathrm{do}(x^{\mathrm{unknown}} \leftarrow x^{\mathrm{known}})\right)}{\mathrm{logit}_{\mathrm{Lakers}-\mathrm{Warriors}}(\mathrm{clean})}
$$

The numerator is the logit difference on the corrupted run after the patch; the denominator is the same difference on the clean run. A value of 1 means the patch fully restores the clean prediction.

**Tiny example.** If the clean run prefers Lakers over Warriors by 6 logits and the patched corrupted run prefers it by 4.5, the patch recovered ${4.5} / 6 = 0.75$ of the behavior. A site that scores near 1 carries the information that matters.

On Gemma 2 2B and 9B this reproduces the circuit Nanda et al. found elsewhere. Early attention heads merge the entity's name into its last token. Later **attribute-extraction heads** at the final position read the entity and move its attributes forward. In Gemma 2 2B, heads L18H5 and L20H3 matter across entity types; Appendix L shows L20H3 promoting "NBA" and "basketball" when reading "Kawhi Leonard".

**The mechanism, part 2: steer and watch attention.** Those heads attend more to known entities than to unknown ones. So the authors steer the last entity token with the recognition latents and measure the attention those heads pay to it.

![Figure 4 from Ferrando et al. (2025): (a, b) activation patching on residual streams and last-position heads; (c) attention to the last entity token is higher for known entities; (d, e, f) attention after steering with the unknown latent, the known latent, and a random vector](/imgs/blogs/do-i-know-this-entity-knowledge-awareness-and-hallucinations-in-language-models-fig4.webp)

Steering with the unknown latent **lowers** attention to the entity even when the entity is known; the known latent **raises** it; a random vector of the same norm does little.

![The recognition signal acts upstream of recall by setting how much later heads attend to the entity](/imgs/blogs/do-i-know-this-entity-knowledge-awareness-and-hallucinations-in-language-models-4.webp)

The suggested route is through the **keys**: the latent changes how the entity token's key looks to later heads' queries. The paper offers this as a hypothesis; it does not decompose the key computation.

**When it fails.** Appendix M shows the 9B model reacting more subtly to the unknown latent. And reduced attention explains why *no attribute* gets moved. It does not by itself explain why the chat model then *refuses* rather than guesses. That last step lives somewhere the paper does not trace.

### Does the model "say" it knows?

A separate test asks the model directly: "Are you sure you know the {entity_type} {entity}? Answer yes or no." Two findings:

- Without intervention, Gemma leans toward "Yes" even for unknown entities. It is bad at *stating* its uncertainty.
- Steering known prompts with the unknown latent pushes the Yes minus No logit difference down; steering unknown prompts with the known latent pushes it up. The authors call the effect "slight".

So the direction that strongly controls *implicit* refusal only weakly controls *explicit* self-report.

## Technique 6: uncertainty directions that predict wrong answers

**The problem.** Entity recognition handles the clean case: the model has never heard of this. But a model can recognize an entity and still give a wrong attribute. Is there an internal signal for "I'm about to get this wrong"?

**The mechanism.** Use chat models, drop prompts the model refuses, and keep correct versus incorrect answers. Read the residual stream at the end-of-instruction token (the `model` token right before the answer), on the theory that it summarizes the whole question. For each latent, compare activation levels between correct and error cases with Welch's t-statistic:

$$
t_{l,j} = \frac{\mu\!\left(a_{l,j}(x^{\mathrm{correct}}_l)\right) - \mu\!\left(a_{l,j}(x^{\mathrm{error}}_l)\right)}{\sqrt{\dfrac{\sigma\!\left(a_{l,j}(x^{\mathrm{correct}}_l)\right)^2}{n^{\mathrm{correct}}} + \dfrac{\sigma\!\left(a_{l,j}(x^{\mathrm{error}}_l)\right)^2}{n^{\mathrm{error}}}}}
$$

$\mu$ and $\sigma$ are the sample mean and standard deviation of the latent's activation in each group, and $n^{\mathrm{correct}}$, $n^{\mathrm{error}}$ are the group sizes. As before, the chosen latent maximizes the minimum score across entity types.

**Tiny example.** If a latent averages 0.4 on correct answers and 2.0 on errors, each with standard deviation 1.0 over 100 examples, then $t = (0.4 - 2.0) / \sqrt{0.01 + 0.01} \approx -11.3$. A large magnitude means the two groups barely overlap.

**What they found.** This experiment uses a different model: Gemma **1** 2B IT, with a community SAE at layer 13 of 18, because Gemma Scope has no SAEs for Gemma 2 2B IT. (Gemma 2 9B IT uses Gemma Scope at layers 10, 21 and 32.) The top latent, which the authors call the "unknown" latent, fires more before wrong answers. As a standalone classifier it reaches **73.2 AUROC** and an **F1 of 72** with a threshold tuned on validation. On FineWeb its strongest activations sit on phrases like "for reasons unknown" and "was not disclosed".

**When it fails.** 73.2 AUROC is a real signal and a weak detector. It is a single latent, not a trained probe; my guess, which the paper does not test, is that a probe on the same activations would do better. Players are left out of the Figure 6 breakdown because Gemma 2B IT refuses almost all of those queries.

## Results at a glance

| Finding | Model(s) | Evidence | Strength |
| --- | --- | --- | --- |
| Known and unknown entity latents exist, entity-agnostic | Gemma 2 2B, 9B; Llama 3.1 8B | Separation and MaxMin scores, peak in middle layers | Strong |
| Latents behave sensibly on post-cutoff songs | All three | 53–76% unknown-latent firing | Moderate (possible data overlap) |
| Not explained by token likelihood | All three | Negligible correlation (Appendix S) | Moderate |
| Unknown latent causes refusal, known latent reduces it | Gemma 2 2B IT strongest | Steering, orthogonalization, random controls | Strong in 2B, weaker in 9B and Llama |
| Latents regulate attention of attribute-extraction heads | Gemma 2 2B, 9B; Llama 3.1 8B | Patching plus attention after steering | Moderate |
| Latents affect explicit "do you know?" answers | Gemma 2 2B, 9B | Yes minus No logit shift | Weak |
| An uncertainty latent predicts errors | Gemma 2B IT, Gemma 2 9B IT | 73.2 AUROC, F1 72 | Moderate |

**What is load-bearing and might not transfer.** "Known" is defined through attribute recall on four Wikidata entity types with templated prompts; free-form and multi-hop questions are untested. The steering coefficients are about twice the residual norm. And the cleanest causal results come from the smallest model.

## Critique

**What is strong.** The pipeline is simple and reusable: label a binary property, score latents by firing-rate difference, require generality with MaxMin, then test causally with steering and weight orthogonalization. The random-latent and random-vector controls are present where they matter. The base-to-chat transfer is a genuinely informative result about how finetuning builds behavior.

**What is weak.**

- **Labels discard the hard middle.** The interesting hallucinations live on partially-known entities, which the labeling rule throws away.
- **Refusal detection is string matching.** A refusal in an unlisted phrasing is miscounted.
- **Mostly one latent at a time.** Whether recognition is one direction or a small subspace is left open.
- **The mechanism stops halfway.** Lower attention explains missing attributes, but the step from "nothing retrieved" to "emit a refusal" is not traced.
- **Scale trend is unfavorable.** Effects shrink from 2B to 9B; frontier models may route refusal through something this method does not find.

**What ablation is missing.** A supervised linear probe on the same activations would show whether SAEs add anything beyond a cheap baseline. A dose-response curve of refusal against $\alpha$ would help separate a clean effect from generic disruption.

**What would change my mind.** If a linear probe trained on the same last-entity-token activations matched the SAE latent's separation and steering effect, I would treat "SAEs discovered this" as "linear structure exists and SAEs are one way to find it". If the steering effect on refusal vanished at coefficients near the natural activation scale of the latent, I would read the result as disruption rather than control. And if the recognition direction failed to predict refusal on free-form, non-templated questions, I would doubt it is general knowledge awareness rather than a template detector.

For a broader look at causal graphs built from SAE features, the [sparse feature circuits analysis](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models) is the natural next read.

## References

1. Javier Ferrando, Oscar Obeso, Senthooran Rajamanoharan, Neel Nanda. *Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models.* ICLR 2025 (Oral). [arXiv:2411.14257](https://arxiv.org/abs/2411.14257)
2. Code: [github.com/javiferran/sae_entities](https://github.com/javiferran/sae_entities)
3. Arditi et al. *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024.
4. Lieberum et al. *Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2.* 2024.
5. On this site: [Scaling and Evaluating Sparse Autoencoders](/blog/paper-reading/ai-interpretability/scaling-and-evaluating-sparse-autoencoders), [The Linear Representation Hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis), [Sparse Feature Circuits](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models)
