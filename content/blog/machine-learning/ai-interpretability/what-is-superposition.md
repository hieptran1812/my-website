---
title: "What Is Superposition? How Neural Networks Pack More Features Than They Have Neurons"
date: "2026-06-13"
publishDate: "2026-06-13"
description: "An intuition-first, code-backed walkthrough of Anthropic's Toy Models of Superposition: why a network stores more features than it has neurons, when it does, and the strange geometry that results."
tags:
  - "ai-interpretability"
  - "superposition"
  - "mechanistic-interpretability"
  - "polysemanticity"
  - "features"
  - "linear-representation-hypothesis"
  - "sparse-autoencoders"
  - "toy-models"
  - "representation-learning"
category: "machine-learning"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
image: "/imgs/blogs/what-is-superposition-1.webp"
excerpt: "Neural networks routinely represent more features than they have neurons. Superposition is the trick that lets them, and Anthropic's toy models show exactly when it kicks in and what geometry it produces."
readTime: 50
---

## The puzzle: a neuron that means five things

Open up a trained vision model, pick a neuron in the middle, and go looking for what makes it fire. Often you get a clean answer: this one fires on curves, that one on dog faces, this other one on the texture of fur. It is tempting to conclude that a network is a tidy filing cabinet, one drawer per concept, and that interpretability is just a matter of reading the labels.

Then you hit a neuron that fires on cat faces, the fronts of cars, *and* the legs of spiders. There is no human concept that unifies those three. The neuron is not broken and the model is not confused — it is doing something deliberate. That neuron is **polysemantic**: it participates in the representation of several unrelated features at once. Polysemantic neurons are not rare curiosities. In large models they are the common case, and they are the single biggest obstacle to reading a network's internals.

Anthropic's 2022 report [*Toy Models of Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html) (Elhage et al.) offers a clean explanation for why polysemantic neurons exist, and it is not "the model is messy." The explanation is **superposition**: a network can represent many more features than it has neurons by storing each feature as a *direction* in activation space rather than as a dedicated neuron, letting those directions overlap, and relying on the fact that real-world features are sparse — rarely active at the same time — to keep the overlap from causing too much damage.

This single idea reframes the whole interpretability project. If features do not live in neurons, then staring at neurons will never reveal them. You have to find the directions. That realization is what motivates the entire modern line of work on sparse autoencoders, dictionary learning, and feature steering.

This post is a careful, intuition-first walk through the toy-models paper. We will build the mental model, derive the toy network, reproduce its phase change in PyTorch, stare at the strange geometry it produces, and trace the consequences out to adversarial examples and grokking. Everything is grounded in a model small enough to fit in your head — five features squeezed through two neurons — because the paper's deepest claim is that the toy is not a toy: the same mechanism is at work inside GPT-scale models.

Here is the table to keep in mind. Almost every intuition people carry about neural representations is subtly wrong in the same direction.

| Assumption | The naive view | The reality under superposition |
| --- | --- | --- |
| What a neuron means | One neuron encodes one human concept | A neuron is a noisy mixture of several unrelated features |
| Capacity | A layer of width $m$ holds $m$ features | A layer of width $m$ can hold far more than $m$ features |
| Where to look | Read the neurons to read the model | Read the *directions*; neurons are an accident of the basis |
| When packing happens | Always, or never | Only when features are sparse enough to rarely collide |
| The cost | Free | Interference — a tax the model pays and the nonlinearity helps it dodge |
| The geometry | Random, unstructured overlap | Sharp, regular polytopes with quantized "dimensions per feature" |

If you internalize the right-hand column, the rest of this post is commentary.

## The mental model

Start with the picture, because the entire paper is a tour of it.

![Mental model of superposition: a network with few neurons stores many features by giving each its own direction instead of a dedicated neuron](/imgs/blogs/what-is-superposition-1.webp)

The diagram above is the mental model: on the left is the assumption almost everyone starts with — one feature per neuron, axes lined up with neurons, orthogonal and clean. It is a comforting picture and it is wasteful. If a model truly needed a dedicated neuron for every concept it represents, GPT-scale models would need billions of neurons in every layer to hold the millions of distinct features they clearly track. They do not have that many. On the right is what actually happens: each feature is a *direction*, the model packs $n$ directions into $m < n$ neurons by letting them overlap, and it gets away with this because real inputs are sparse — only a handful of features are active in any given example, so the overlapping directions rarely interfere at the same time.

The analogy I keep coming back to is a student with two notebooks for five subjects. If the five subjects were all discussed simultaneously, two notebooks would be hopeless — every page would be a jumble. But classes do not overlap: you have math on Monday, history on Tuesday. So the student reuses the same two notebooks across all five subjects, and because the subjects are *temporally sparse*, there is almost never a collision. Superposition is exactly this: reuse the same neurons for many features, and lean on sparsity so the reuse rarely bites.

That word "sparse" is doing enormous work, and it is worth being precise about it. A feature being sparse does not mean the feature is unimportant or rare in the dataset overall. It means that on any *single* input, the probability that the feature is active is low. "Is this token a Python keyword?" is a sparse feature: it is false for the overwhelming majority of tokens, true for a few. Natural data is overwhelmingly made of sparse features like this, and that statistical fact is the entire reason superposition is a good deal. The rest of this post is, in a sense, just a quantitative study of how good a deal it is as a function of how sparse the features are.

Two consequences fall straight out of the mental model. First, **interference is unavoidable** the moment directions overlap: reading off one feature will pick up a little bit of every other feature whose direction is not perpendicular to it. Second, **the model has a design problem to solve** — how to arrange $n$ directions in $m$ dimensions so that the total interference, weighted by how much it actually hurts, is minimized. The astonishing finding of the paper is that the solutions to this packing problem are not random blobs. They are crisp, regular geometric figures, and which figure you get is controlled by a single knob: sparsity.

## Features as directions

Before we can talk about packing features into directions, we have to commit to the claim that features *are* directions in the first place. This is the **linear representation hypothesis**, and it is the bedrock the whole paper stands on. (I have written a longer treatment in [The Linear Representation Hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis); here I will keep it to what we need.)

![Features as directions: a concept is a direction in activation space and meanings combine by adding those directions](/imgs/blogs/what-is-superposition-2.webp)

The figure above shows the two halves of the hypothesis. On the right is **decomposability**: the activation space of a network decomposes into independent feature directions, so that "is this about royalty," "is this plural," "what tense is this," and "what is the emotional tone" each correspond to their own direction. On the left is **linearity**: those directions combine by vector addition, which is exactly why the famous word-embedding arithmetic works — $\text{king} - \text{man} + \text{woman} \approx \text{queen}$ holds because "royalty" and "gender" are directions, and moving along them composes. The parallelogram is not a party trick; it is direct evidence that the model encoded "gender" and "royalty" as additive directions.

Concretely, the hypothesis says a representation $h \in \mathbb{R}^m$ can be written

$$ h \approx \sum_i x_i \, W_i $$

where $x_i \ge 0$ is the activation (intensity) of feature $i$ and $W_i \in \mathbb{R}^m$ is the unit-ish direction the model has assigned to that feature. Reading feature $i$ back out is a dot product: $\hat{x}_i = W_i \cdot h$. If the directions were all orthonormal, this readout would be exact: $W_i \cdot W_j = \delta_{ij}$, so $\hat{x}_i = x_i$ and there is no interference. The whole drama of superposition is what happens when you have more features than dimensions and *cannot* make them all orthogonal.

Why would a network use directions at all, rather than some cleverer nonlinear code? Three reasons, all of which the paper leans on:

1. **It is natural.** A linear layer followed by a pointwise nonlinearity reads features with exactly a dot-product-then-threshold operation. Linear features are what the architecture is built to consume.
2. **It is composable.** If features are directions, the next layer can select, combine, or ignore them with a single matrix multiply. Nonlinear codes do not compose this cheaply.
3. **It is statistically efficient.** Linear features generalize from fewer examples, because a linear readout shares statistical strength across the whole direction rather than memorizing a lookup table.

A fair objection at this point is that we have been throwing around the word "feature" as if its meaning were obvious. It is not, and the paper spends real effort pinning it down with three candidate definitions, each of which fails in an instructive way. The first is **features as arbitrary functions of the input**: anything you can compute from $x$ counts. This is too permissive — it puts "cat $+$ car" and "cat $-$ car" on equal footing with "cat," and no honest theory of representation should treat a meaningful concept and a random arithmetic mash-up of concepts as the same kind of object. The second is **features as human-interpretable properties**: a curve, a specific word sense, the sentiment of a sentence. This matches our instincts but is too restrictive — it rules out features that a model demonstrably uses yet that have no clean human name, like the intermediate chemistry a protein-folding model tracks. Defining features as "things humans can name" just bakes our own blind spots into the definition.

The third definition, and the one the paper actually adopts, is **features as the properties a sufficiently large model would give a dedicated neuron to**. A property is a feature if, given enough capacity, a network would represent it monosemantically — a curve detector counts as a feature because big-enough vision models reliably grow curve-detecting neurons. The definition sounds circular but is operationally sharp: it ties "feature" to what the network's own learning dynamics treat as an atomic unit, which is exactly the object superposition acts on. In this framing, superposition is simply what happens when a model has *fewer neurons than it has features it would like to give neurons to* — and that mismatch is the rule, not the exception, in any model worth interpreting.

There are non-linear representations in real networks, and the paper is careful not to claim otherwise. But linear directions are the dominant, default encoding, and superposition is fundamentally a story about linear directions that are no longer orthogonal. Hold onto the readout equation $\hat{x}_i = W_i \cdot h$ — when we get to the weight matrix, every term in it will be one of these dot products.

## The privileged basis: why neurons are not always features

There is a subtlety we skated past, and it is the reason the word "polysemantic" is even meaningful. Why do we expect features to align with neurons at all? In a purely linear system, we should not.

![Why neurons are not always features: only an elementwise nonlinearity makes the neuron basis special enough for features to align with it](/imgs/blogs/what-is-superposition-3.webp)

The figure above traces the logic. Take a hidden representation $h$. If the only thing the network does with $h$ is multiply it by more matrices — a purely linear map — then the *basis is arbitrary*. You can rotate $h$ by any orthogonal matrix $R$, replace the next weight matrix with one that undoes the rotation, and get a network that computes the exact same function. Under that rotation, "neuron 3" becomes a totally different mixture of the old neurons. If the basis can be rotated freely without changing anything, there is no reason on earth for features to line up with the standard (neuron) basis. Asking "what does neuron 3 represent" would be as meaningless as asking "what does the 3rd-from-northeast direction in physical space represent."

What breaks the rotational symmetry is an **elementwise nonlinearity**. The moment you apply $\text{ReLU}$ or $\text{GELU}$ *coordinate by coordinate*, the standard basis becomes special: $\text{ReLU}(R h) \neq R\,\text{ReLU}(h)$ in general, because the nonlinearity acts on each neuron individually. The activation function singles out the neuron axes. This is what the paper calls a **privileged basis** — a basis the network's own operations treat as distinguished. Only in a privileged basis does the question "does this feature align with this neuron?" even have an answer.

This gives us a clean two-part vocabulary:

- A **privileged basis** is what makes monosemantic neurons *possible* — without it, neurons could never align with features.
- **Superposition** is what makes them *fail to materialize anyway* — even with a privileged basis, the model crams more features in than there are neurons, so features end up as off-axis directions and individual neurons go polysemantic.

So polysemanticity is the observable symptom; superposition is the proposed underlying cause; and the privileged basis is the reason the symptom is even surprising. With this in hand, we can finally build the toy that lets us watch superposition happen on demand.

## The toy model

The genius of the paper is to strip the phenomenon down to the smallest system that still exhibits it. Forget transformers. Consider an autoencoder with a bottleneck.

![The toy model: a width-2 bottleneck forced to reconstruct 5 features is what makes superposition appear](/imgs/blogs/what-is-superposition-4.webp)

The figure above is the whole architecture. You feed in a vector $x \in \mathbb{R}^n$ of $n$ feature intensities. A single weight matrix $W \in \mathbb{R}^{m \times n}$ projects it down into an $m$-dimensional hidden vector $h = Wx$, where $m < n$ is the bottleneck. Then the *same* matrix (tied weights) projects back up, a bias is added, and a ReLU is applied:

$$ h = Wx, \qquad x' = \text{ReLU}(W^\top h + b) = \text{ReLU}(W^\top W x + b). $$

The model is trained to reconstruct its own input: make $x'$ as close to $x$ as possible. Because $m < n$, it physically cannot store all $n$ features orthogonally — there is not enough room. It is forced to decide which features to represent and how to share the cramped $m$-dimensional space among them. That forced decision is superposition, isolated in a petri dish.

Two design choices make this model say something about the real world rather than about generic compression.

The first is **feature importance**. Not all features matter equally for the loss. The paper weights the reconstruction error per feature with an importance $I_i$:

$$ \mathcal{L} = \sum_{\text{batch}} \sum_{i=1}^{n} I_i \,\bigl(x_i - x'_i\bigr)^2. $$

A common choice is a geometric ladder, $I_i = r^i$ for some $r < 1$, so feature 0 is the most important and each subsequent feature matters a little less. Importance is the model's incentive structure: it will spend its scarce dimensions protecting the features that move the loss the most.

Here is the model in PyTorch. It is small enough that you can train it on a laptop in seconds.

```python
import torch
import torch.nn as nn

class ReLUOutputModel(nn.Module):
    """Anthropic's toy model:  x -> h = W x -> x' = ReLU(W^T h + b).

    n input features are projected into an m-dim bottleneck (m < n) and
    reconstructed with tied weights, so the only parameters are W and b.
    The columns W[:, i] are the *feature directions* we care about.
    """
    def __init__(self, n_features: int, m_hidden: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(m_hidden, n_features))
        self.b = nn.Parameter(torch.zeros(n_features))
        nn.init.xavier_normal_(self.W)

    def forward(self, x):                  # x: (batch, n_features)
        h = x @ self.W.T                   # encode  -> (batch, m_hidden)
        out = h @ self.W + self.b          # decode (tied) -> (batch, n_features)
        return torch.relu(out)             # output nonlinearity
```

The second design choice is **sparsity**, and it is the control variable for the entire paper.

![Sparsity: as a feature is zero more often, a typical input fires fewer features at once, which is what makes overlap safe](/imgs/blogs/what-is-superposition-5.webp)

The figure above shows what sparsity does to a typical input vector. Each feature is independently set to zero with probability $S$ (the sparsity) and, when it is "on," is drawn uniformly from $[0, 1]$. At $S = 0$ (dense), every feature is active in every example — the top row, all cells lit. As $S$ rises, a typical input fires fewer and fewer features at once — the bottom row, almost all zeros. This is precisely the condition that makes overlapping directions safe: if two features share a direction but are almost never on simultaneously, the interference between them rarely materializes.

The data generator is four lines:

```python
def generate_batch(batch_size, n_features, sparsity, device="cpu"):
    """Each feature is active with probability (1 - sparsity); when active it is
    drawn uniformly from [0, 1], otherwise exactly 0. `sparsity` is S in the
    paper -- the fraction of the time a given feature is off."""
    values = torch.rand(batch_size, n_features, device=device)
    active = torch.rand(batch_size, n_features, device=device) > sparsity
    return values * active                 # zero out the inactive features
```

Notice what is *not* in the model: there are no hidden layers, no attention, no depth. Just a down-projection, an up-projection, and a ReLU. If superposition shows up here, it cannot be blamed on anything exotic. It is a consequence of the most basic ingredient in deep learning — reconstructing more things than you have room for, when those things are sparse.

### Second-order optimization: why tied weights and ReLU both matter

It is worth pausing on two details that look like simplifications but are load-bearing. **Tying the weights** (using $W$ for both encode and decode) is what forces the feature directions to do double duty as both the "write" and "read" vectors, which is what creates the clean dot-product structure we will analyze in $W^\top W$. **The output ReLU** is what lets the model *clip away* negative interference: if a feature is off but interference pushes its reconstruction slightly negative, the ReLU snaps it back to zero for free. Remove the ReLU and the model loses its cheapest tool for cleaning up the mess that superposition creates — which, foreshadowing, is exactly why the linear version of this model behaves like plain PCA and never superposes at all.

## The phase change: dense versus sparse

Now we run the experiment that gives the paper its punch. Fix the smallest interesting case: $n = 5$ features, $m = 2$ hidden dimensions, importance decaying geometrically. Train it once with dense data and once with sparse data, and look at the columns of $W$ — the feature directions — in the 2D hidden plane.

![The phase change: the same 5-feature, 2-neuron model keeps only the top two when dense but packs all five into a pentagon when sparse](/imgs/blogs/what-is-superposition-6.webp)

The figure above is the central result, and the contrast could not be sharper. On the left, with **dense** data ($S \approx 0$), the model behaves exactly like principal component analysis: it picks the two most important features, gives each its own orthogonal axis, and *throws the other three away entirely* — their directions collapse to zero, they are simply not represented. With every feature on in every example, any overlap is pure cost with no payoff, so the model refuses to overlap. It represents the two features it can afford and abandons the rest.

On the right, with **sparse** data ($S$ close to 1), something qualitatively different happens. The model now represents *all five* features, arranged as a regular pentagon — five directions evenly spaced at $72°$ around the circle. None of them is orthogonal to its neighbors, so there is interference between every pair. But because features are sparse, two given features are almost never on at the same time, so that interference almost never actually fires. The model has decided that representing all five with a little interference beats representing two perfectly.

This is a genuine **phase transition**, not a smooth trade-off. As you sweep sparsity from low to high, the model does not gradually blend the two regimes. It snaps from "represent the top few, orthogonal, PCA-style" to "represent everything, in superposition." The training loss surface reorganizes. The table below names the regimes:

| Regime | Sparsity $S$ | What the model does | Geometry | Interference |
| --- | --- | --- | --- | --- |
| Dense | low (features usually on) | Represent only the top-$m$ features; drop the rest | Orthogonal axes (PCA) | None, because no overlap |
| Intermediate | moderate | Represent a few extra features partially | Mixed; some pairs share | Some, paid selectively |
| Sparse | high (features usually off) | Represent (almost) all features | Regular polytopes (pentagon, etc.) | Lots, but rarely triggered |

There is a memorable one-liner hiding in this result, and it is worth stating plainly:

> Superposition is the price a network pays to be effectively bigger than it is. Sparsity is the discount that makes the price worth paying.

What makes this a phase transition rather than a smooth knob is that the two solutions are *different basins* of the loss landscape, not points on a continuum. The PCA solution and the pentagon solution are both local minima; which one is the global minimum depends on sparsity, and at the critical sparsity the winner flips. Cross that point and gradient descent, started from a random initialization, reliably falls into the other basin. That is why you see a clean snap rather than a gradual morph: the model is not interpolating between two strategies, it is choosing between two qualitatively distinct geometries, and the choice is discontinuous in the control parameter. The same structure — competing discrete solutions whose relative depth flips as a parameter crosses a threshold — is exactly the ingredient that, at much larger scale, gets blamed for grokking and double descent. The five-feature toy is the cleanest possible instance of it, which is part of why the paper treats it as a window onto those harder phenomena rather than a separate curiosity.

The training loop that produces these pictures is short. The only subtlety is the importance-weighted loss.

```python
def importance_weights(n_features, decay=0.9, device="cpu"):
    # feature i matters decay**i as much as feature 0 -> a clean importance ladder
    return decay ** torch.arange(n_features, device=device)

def train(model, n_features, sparsity, steps=10_000, lr=1e-3, batch=2048):
    importance = importance_weights(n_features, decay=0.9)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        x = generate_batch(batch, n_features, sparsity)
        x_hat = model(x)
        # importance-weighted reconstruction loss:  L = sum_i I_i (x_i - x'_i)^2
        loss = (importance * (x - x_hat) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

dense  = ReLUOutputModel(n_features=5, m_hidden=2); train(dense,  5, sparsity=0.0)   # dense  -> PCA, top-2 orthogonal
sparse = ReLUOutputModel(n_features=5, m_hidden=2); train(sparse, 5, sparsity=0.9)   # sparse -> all five in a pentagon
```

If you plot `dense.W` and `sparse.W` as five 2D arrows, you will reproduce the figure: two orthogonal arrows (plus three near-zero stubs) for the dense model, and a clean pentagon for the sparse one. The phase change is not a story you have to take on faith; it is fifteen lines of PyTorch away.

## How much can a network superpose? A capacity view

Before staring at the weights, it pays to answer the quantitative question head-on: how many features can $m$ dimensions actually hold? The toy model gives a usable answer, and it scales with sparsity in a way worth carrying around as a rule of thumb.

The governing quantity is $1/(1-S)$, the reciprocal of the activation probability. If each feature is active with probability $1 - S$, then a random input of $n$ features has on average $n(1-S)$ features switched on at once. Superposition is safe roughly when the number of *simultaneously active* features is small relative to the available dimensions $m$ — not when the *total* number of features $n$ is small. So the binding constraint is approximately $n(1-S) \lesssim m$ rather than $n \lesssim m$. Rearranged, the number of features a width-$m$ layer can carry grows like

$$ n \;\lesssim\; \frac{m}{1-S}. $$

At $S = 0$ (dense) this collapses to $n \le m$: no superposition, PCA, dedicated axes. At $S = 0.9$ you can carry roughly $10\times$ as many features as you have dimensions; at $S = 0.99$, about $100\times$. The effective capacity of a layer is not its width — it is its width divided by the density of the features it represents.

This is the same principle that underlies **compressed sensing**: a sparse signal in a high-dimensional space can be recovered from far fewer measurements than the ambient dimension, precisely because sparsity confines the signal to a low-dimensional union of subspaces. Superposition is a network discovering, by gradient descent, that it can stuff a sparse high-dimensional feature vector through a low-dimensional bottleneck and recover it with small error. The off-diagonal interference *is* the recovery error, and sparsity is what keeps it negligible on a typical input.

There is an important asymmetry hiding in the capacity formula: it bounds how many features you can *store*, but storage and faithful *readout* are not the same thing. As more features crowd into the same dimensions, the per-feature interference grows, and at some point the recovery error swamps the signal for the least-important features — which is exactly the model's cue to drop them ($D \to 0$) rather than represent them badly. The capacity bound is therefore soft: the model does not hit a wall and fail, it gracefully sheds its least valuable features as pressure rises. That graceful degradation is part of why superposition is such a robust strategy, and part of why scaling a model up — buying more $m$ — reliably promotes marginal features from "superposed and noisy" to "dedicated and clean."

The capacity view also explains why the geometry quantizes. Given a fixed budget of dimensions and a set of features to pack, there are only so many *locally optimal* ways to arrange them, and those optima are the uniform polytopes. The model is not choosing a dimensionality $D$ on a continuum; it is choosing a discrete packing, and $D$ is just a readout of which packing it landed in. The next two sections make that concrete — first by reading the interference structure off $W^\top W$, then by cataloguing the polytopes themselves.

## Reading the weights: norms and interference

To reason about superposition quantitatively, we need a single object that captures both "what did the model choose to represent" and "what does it cost." That object is the matrix $W^\top W$.

![Reading the weights: the diagonal of W-transpose-W is how strongly each feature is stored, and the off-diagonal entries are the interference superposition pays](/imgs/blogs/what-is-superposition-7.webp)

The figure above lays it out. $W^\top W$ is an $n \times n$ matrix whose entry $(i, j)$ is the dot product $W_i \cdot W_j$ of feature direction $i$ with feature direction $j$. It splits cleanly into two parts:

- **The diagonal** entries are $W_i \cdot W_i = \lVert W_i \rVert^2$, the squared length of each feature's direction. This is *how strongly the model represents feature $i$*. A diagonal entry near 1 means the feature is fully represented; an entry near 0 means the feature was dropped.
- **The off-diagonal** entries are $W_i \cdot W_j$ for $i \neq j$. This is the **interference**: when feature $j$ is active, it contributes a spurious $W_i \cdot W_j$ to the readout of feature $i$. Off-diagonal zero means no interference (the directions are orthogonal). Off-diagonal nonzero means feature $j$ leaks into feature $i$.

Recall the readout $\hat{x}_i = W_i \cdot h = W_i \cdot \sum_j x_j W_j = \sum_j (W_i \cdot W_j)\, x_j$. The $j = i$ term is the signal (scaled by the diagonal); every $j \neq i$ term is noise (scaled by an off-diagonal). So $W^\top W$ is *literally* the readout matrix. An orthonormal, no-superposition solution makes $W^\top W$ the identity: bright diagonal, black off-diagonal, perfect readout. A superposition solution cannot reach the identity — there is not enough room — so it accepts a speckle of off-diagonal interference in exchange for lighting up more of the diagonal.

This reframes the model's job as a beautifully concrete optimization. It wants $W^\top W \approx I$ but is constrained to rank $m < n$. It is solving

$$ \min_{W} \; \sum_i I_i \,\mathbb{E}\bigl[(x_i - x'_i)^2\bigr] \quad\text{s.t.}\quad \text{rank}(W^\top W) \le m, $$

which is a trade-off between making the diagonal large (represent features) and the off-diagonal small (avoid interference), weighted by importance and by how often features actually co-occur. Sparsity enters through that co-occurrence: when features are sparse, the expected damage from an off-diagonal term is small because $x_i$ and $x_j$ are rarely nonzero together, so the model is happy to tolerate large off-diagonals.

To make this concrete, take the sparse pentagon from the previous section. Its five unit feature directions sit $72°$ apart, so every off-diagonal entry of $W^\top W$ is one of just two values: $\cos 72° \approx 0.31$ for the two immediate neighbors of each feature, and $\cos 144° \approx -0.81$ for the two non-neighbors. The diagonal is $\lVert W_i \rVert^2 \approx 1$ for all five. So the readout of feature $i$ works out to

$$ \hat{x}_i \approx x_i \;+\; 0.31\,(x_{i-1} + x_{i+1}) \;-\; 0.81\,(x_{i-2} + x_{i+2}). $$

On dense data this would be a catastrophe — every readout is corrupted by four interfering terms, two of them with a coefficient near $0.81$. But the data is sparse: at $S = 0.9$, the chance that any specific neighbor is also on when feature $i$ is on is only $0.1$, so on the overwhelming majority of inputs the interfering terms are simply zero and $\hat{x}_i \approx x_i$. The model has arranged the five directions so that the *worst-case* interference is large but the *expected* interference is tiny, and it leans on the output ReLU to clip away whatever negative leakage does sneak through. That is the whole trick, legible in five numbers.

Contrast this with the **linear model** — the same architecture with the output ReLU removed. Without the ReLU to clip negative interference, the model cannot exploit the asymmetry between "feature on" and "feature off"; superposition stops paying off and the network collapses back to the PCA solution at *every* sparsity level. The output nonlinearity is not a cosmetic detail. It is the single ingredient that makes superposition profitable, which is why the privileged-basis discussion from earlier was not a digression — the very thing that creates a privileged basis (the elementwise nonlinearity) is also the thing that makes superposition worth doing.

The diagonal of $W^\top W$ also gives us the cleanest possible definition of "how much" a feature is represented, which we are about to need for the geometry. In code:

```python
@torch.no_grad()
def gram(W):
    """Return W^T W: diagonal = ||W_i||^2 (how stored), off-diagonal = interference."""
    return W.detach().T @ W.detach()       # (n_features, n_features)

g = gram(sparse.W)
stored       = g.diag()                    # how strongly each feature is represented
interference = g - torch.diag(g.diag())    # off-diagonal leakage matrix
print("stored per feature:", stored.round(decimals=2).tolist())
print("max interference :", interference.abs().max().item())
```

For the dense PCA solution you will see two diagonal entries near 1 and three near 0, with an essentially zero off-diagonal. For the sparse pentagon you will see all five diagonal entries comparable and nonzero, with a regular pattern of small off-diagonal interference — the fingerprint of a uniform polytope.

## The geometry of superposition

Here is where the paper stops being a story about compression and becomes genuinely beautiful. When the model superposes, the directions it chooses are not arbitrary. They snap into **uniform polytopes** — the same regular figures that show up in physics problems about minimizing the energy of mutually repelling points on a sphere.

![The geometry of superposition: superposed features snap into uniform polytopes, each fixing how many dimensions every feature gets](/imgs/blogs/what-is-superposition-8.webp)

The figure above is a gallery of the configurations that appear, each annotated with its **feature dimensionality** $D$. We need a precise definition of $D$, because it is the quantity that turns out to be quantized. For feature $i$:

$$ D_i = \frac{\lVert W_i \rVert^2}{\sum_j \bigl(\hat{W}_i \cdot W_j\bigr)^2}, \qquad \hat{W}_i = \frac{W_i}{\lVert W_i \rVert}. $$

Read it as a fraction: the numerator is how much feature $i$ is represented; the denominator is how many features (including $i$ itself) are crowding into the direction $i$ occupies, measured by projecting every feature onto $i$'s unit direction and summing the squared projections. So $D_i$ is, intuitively, **the fraction of a dimension that feature $i$ gets to itself**. If feature $i$ has its own private orthogonal axis, $D_i = 1$. If it shares its single dimension with one antipodal partner, $D_i = \tfrac{1}{2}$. If it is not represented at all, $D_i = 0$.

Now the gallery makes sense:

| Configuration | Features | Dimensions used | $D$ per feature | What it looks like |
| --- | --- | --- | --- | --- |
| Dedicated / orthonormal | 1 per axis | 1 each | $1$ | Perpendicular axes; no superposition |
| Antipodal pair (digon) | 2 | 1 | $\tfrac{1}{2}$ | Two arrows pointing exactly opposite |
| Triangle | 3 | 2 | $\tfrac{2}{3}$ | Three arrows at $120°$ |
| Tetrahedron | 4 | 3 | $\tfrac{3}{4}$ | Four arrows to the corners of a tetrahedron |
| Pentagon | 5 | 2 | $\tfrac{2}{5}$ | Five arrows at $72°$ |

The antipodal pair is the cleanest case to build intuition. Two features share a *single* dimension by pointing in exactly opposite directions, $W_2 = -W_1$. Their interference is $W_1 \cdot W_2 = -\lVert W_1 \rVert^2$ — maximally negative. That sounds terrible, but the output ReLU turns it into an asset: when feature 1 is on (positive) and feature 2 is off, the negative interference pushes feature 2's pre-activation *down*, and the ReLU clips it to exactly zero. The two features can share one line as long as they are rarely on together, and each one gets $D = \tfrac{1}{2}$ of a dimension. This is the smallest unit of superposition, and it is why $\tfrac{1}{2}$ is the most common "sticky" value.

The triangle and tetrahedron generalize the idea: three features at $120°$ in a plane, or four at the vertices of a tetrahedron in 3-space, each spreading their mutual interference as evenly as possible. These are exactly the solutions to the **Thomson problem** — how to place points on a sphere so the repulsion between them is minimized. Superposition is solving a Thomson problem where "repulsion" is "interference weighted by co-occurrence." That connection to a classical physics problem is not a metaphor; it falls out of the loss.

### The sticky fractions

The most striking quantitative claim of the paper is that $D$ does not vary smoothly as you turn the sparsity knob. It **sticks** at special fractions.

![Sticky fractions: as sparsity rises, the dimensions allotted to each feature do not fall smoothly but drop through a few sticky plateaus](/imgs/blogs/what-is-superposition-9.webp)

The figure above plots feature dimensionality $D$ against sparsity. As you make features sparser, you would naively expect the model to smoothly trade representation quality for capacity, sliding $D$ down continuously. Instead it descends a **staircase**: it lingers on a plateau at $D = 1$ (dedicated dimensions, no superposition), then jumps to $D = \tfrac{2}{3}$ (triangles), holds there, jumps to $D = \tfrac{1}{2}$ (antipodal pairs), holds, drops to $D = \tfrac{2}{5}$ (pentagons), and so on toward zero. In between the plateaus are unstable: the model would rather commit to a clean geometric configuration than sit at an awkward intermediate packing.

Why the plateaus? Because each plateau corresponds to a *stable geometric arrangement* that locally minimizes interference. A triangle at $120°$ is a local optimum of the packing problem; perturb it slightly and the loss goes up, so the model stays there across a range of sparsities. Only when sparsity changes enough to make a *different* polytope strictly better does the model tunnel over to it, and that transition is sharp. The "stickiness" is the signature of a loss landscape full of discrete, locally optimal geometries rather than one smooth basin.

We can measure $D$ directly. This is the function that produces the staircase when you sweep sparsity:

```python
@torch.no_grad()
def feature_dimensionality(W):
    """D_i = ||W_i||^2 / sum_j (W_hat_i . W_j)^2  for each feature column i.
    Returns a tensor of per-feature dimensionalities in [0, 1]."""
    W = W.detach()                         # (m_hidden, n_features)
    norms = W.norm(dim=0)                  # ||W_i|| for each feature
    W_hat = W / (norms + 1e-8)             # unit feature directions
    proj  = W_hat.T @ W                    # (n, n): row i = projections of all W_j onto W_hat_i
    denom = (proj ** 2).sum(dim=1)         # sum_j (W_hat_i . W_j)^2
    return (norms ** 2) / (denom + 1e-8)   # D_i

for S in [0.0, 0.5, 0.7, 0.9, 0.97, 0.99]:   # sweep sparsity; D clusters on the sticky fractions 1, 2/3, 1/2, 2/5
    m = ReLUOutputModel(n_features=20, m_hidden=5); train(m, 20, sparsity=S, steps=20_000)
    D = feature_dimensionality(m.W)
    print(f"S={S:>4}:  mean D = {D.mean():.3f},  histogram of D = {D.round(decimals=2).tolist()}")
```

Run this and the printed dimensionalities will pile up near $1$, $\tfrac{3}{4}$, $\tfrac{2}{3}$, $\tfrac{1}{2}$, and $\tfrac{2}{5}$ rather than spreading out evenly. The staircase is real, and it is reproducible on a laptop.

The quantization of $D$ has a direct and somewhat uncomfortable consequence for the tools we use to *undo* superposition. A sparse autoencoder trained to recover features from a superposed layer is, in effect, trying to read off the polytope. But when the model has packed several genuine features into a tightly-overlapping arrangement, the autoencoder faces a forced choice: split one true feature across several dictionary atoms, or merge several true features into one atom. This is the **feature splitting and absorption** problem, and it is a direct inheritance from the geometry described here — the directions the autoencoder is trying to recover are not cleanly separated, they were arranged into interfering polytopes by design. The cleaner the model's polytope (high $D$, near-dedicated axes), the easier the recovery; the more aggressive the superposition (low $D$, tightly packed), the more the recovered "features" depend on the autoencoder's width and sparsity penalty rather than on any ground truth. So the toy model predicts not only that SAEs are the right tool, but the specific ways they will struggle — a prediction that has held up uncomfortably well in practice.

## Importance asymmetry and the phase diagram

So far the features have all been roughly equal. Real features are not. Some matter a great deal to the loss; some barely move it. When importance varies, superposition becomes *selective* in a way that mirrors what we see in real networks.

The pattern is this: the model spends its scarce dimensions protecting important features and superposes the cheap ones. Concretely, if you sort features by importance and look at how they are stored, you find a gradient. The top few important features tend to get **dedicated, nearly-orthogonal directions** ($D \approx 1$): the model refuses to let anything interfere with the features that matter most, so it pays full price for them. The long tail of low-importance features gets **crammed into shared directions in heavy superposition** ($D$ small): the model is willing to let them interfere with each other, because getting them slightly wrong costs little.

This produces a clean picture if you plot, for each feature, its importance against the sparsity of the data, and color by how the model represents it. You get three regions:

| Region | Condition | How the feature is represented |
| --- | --- | --- |
| Not represented | low importance *and* low sparsity | Dropped entirely; $D = 0$, direction collapses to zero |
| Dedicated dimension | high importance, or low sparsity | Its own near-orthogonal axis; $D \approx 1$, no interference |
| Superposed | low importance *and* high sparsity | Shares a direction with others; $0 < D < 1$, interference tolerated |

The boundaries between these regions are sharp — another phase transition. A feature that is important enough, or data that is dense enough, lands a feature in the "dedicated" region. Drop importance and raise sparsity, and at some critical point the feature snaps into superposition. Drop importance further at low sparsity and the model gives up on it entirely.

The practical upshot for interpretability is sobering and clarifying at the same time. In a real model, the cleanest, most monosemantic neurons you find — the curve detector, the "DNA" neuron, the base64 neuron — are likely the *important* features that earned a dedicated direction. The vast, murky sea of polysemantic neurons is the long tail of less-important features living in superposition. You are not failing to interpret them because you are not clever enough; they are genuinely not stored one-per-neuron, by the model's deliberate design.

This also explains why simply scaling a model up makes some features cleaner: a wider layer has more dimensions to hand out as dedicated axes, so features that were marginal at small width can graduate to their own direction at large width. Superposition is relative to capacity, and capacity is something you can buy.

There is a subtle prediction buried in the importance gradient that shows up constantly in real interpretability work. Because importance and sparsity *jointly* decide a feature's fate, two features with identical importance can be stored completely differently if their activation statistics differ — the sparser one gets superposed, the denser one gets a dedicated axis or is dropped. The "interpretability" of a feature is therefore not an intrinsic property of the concept. It is a property of the concept *and* how often it fires *and* how much it moves the loss *and* how much spare capacity the layer has. The same concept can be a clean monosemantic neuron in one model and a smeared superposed direction in another, purely because the second model was narrower or trained on data where the concept was rarer. This is a large part of why interpretability findings fail to transfer between models, and why a claim like "neuron 1337 is the Golden Gate Bridge neuron" is a model-specific accident rather than a universal fact. The toy model warns you not to expect a universal feature-to-neuron map, because the map is a contingent solution to a packing problem with four moving parts.

## Computing in superposition

Up to here, superposition has been a story about *storage*: how to cram features into too few dimensions. But a model does not store features for their own sake; it computes with them. The deepest result in the paper is that networks can **compute** on features while they are in superposition, not merely warehouse them.

![Computing through superposition: the model does not just store overlapping features, it computes on them and filters the resulting interference](/imgs/blogs/what-is-superposition-10.webp)

The figure above shows the experiment, called the **absolute value model**. Instead of training the network to reconstruct its input, train it to compute a function of each feature — the simplest interesting one is $y_i = \lvert x_i \rvert$, the absolute value, on signed sparse inputs. The features still pass through a bottleneck $m < n$, so they are still forced into superposition. The question is whether the model can produce the correct $\lvert x_i \rvert$ for every feature *despite* the features overlapping in the hidden layer.

It can. The model learns weights such that, after the down-projection mixes everything together, the nonlinearity and the up-projection can still recover each feature's absolute value with small error. The mechanism is exactly the interference-suppression we saw with the antipodal pair, used now in service of a computation rather than a reconstruction: when a feature is off, the interference from other features pushes its channel one way or the other, and the ReLU clips away the part that would corrupt the answer. Sparsity is again the enabler — because few features are on at once, the interference any one computation has to survive is small.

This matters far more than it might first appear. If models only *stored* in superposition, you could imagine a clean two-stage picture: a messy compressed representation that gets "decompressed" into a clean one before any real computation happens. The absolute value model kills that hope. Computation happens *in* the superposed representation. There is no privileged decompressed layer where everything is monosemantic and tidy. The circuits a model uses to do its actual work are operating on overlapping features directly, interference and all.

It pays to be precise about *why* the absolute value succeeds where naive expectation says it should fail. The hidden vector $h$ is a sum of contributions from every active feature, so when the model reads feature $i$ to compute $\lvert x_i \rvert$, it actually reads $W_i \cdot h = x_i \lVert W_i\rVert^2 + \sum_{j\neq i}(W_i \cdot W_j)\,x_j$. The second term is interference — the same off-diagonal leakage from $W^\top W$. The model has two defenses. First, it can orient the directions so the interfering coefficients are small for features that co-occur. Second, and more powerfully, the ReLU is a *thresholding* operation: leakage that does not push a value across zero is discarded, and the network can learn biases that place the threshold so sub-threshold noise dies while genuine signal survives. Together these amount to an error-correcting code, paid for entirely by sparsity. And the deeper the network, the more rounds of clip-and-clean it can apply — which is one reason real transformers tolerate far more aggressive superposition than a single-layer toy: every layer is another chance to suppress accumulated interference before it corrupts the result. Depth, in this view, is not only expressive power; it is error correction for superposition.

For mechanistic interpretability, this is the hard mode. It means you cannot just find the features; you have to understand computations that are themselves smeared across superposed directions. It is the reason "find the features with a sparse autoencoder, then read the circuits" is a research program and not a solved problem — the circuits live in the same tangled basis the features do.

## Why superposition matters

Superposition would be a cute result about toy autoencoders if it did not connect to so much else. It is load-bearing for a surprising number of otherwise-unrelated puzzles in deep learning.

![Why superposition matters: one compression mechanism quietly underwrites several otherwise-unrelated puzzles in deep learning](/imgs/blogs/what-is-superposition-11.webp)

The figure above fans out the consequences from the single mechanism. Walk through them.

**Polysemantic neurons.** This is the most direct consequence and where we started. If features are stored as off-axis directions in superposition, then any single neuron is a projection onto several of those directions at once, so it fires for several unrelated features. Polysemanticity is not noise or a training failure; it is the shadow that superposition casts on the neuron basis. The paper makes this airtight by *constructing* polysemantic neurons on demand: crank up sparsity in the toy model and the (privileged-basis) neurons become polysemantic, exactly as predicted.

**Adversarial examples.** A network in superposition has interference directions baked into it — directions along which a small input perturbation produces an outsized, spurious change in some feature's readout, because of the off-diagonal terms in $W^\top W$. These are precisely the directions an adversarial attack wants to exploit: tiny, human-imperceptible nudges that ride the interference to flip a prediction. Superposition suggests adversarial vulnerability is not a bug to be patched but a structural consequence of packing more features than dimensions — the same overlap that buys capacity sells fragility.

**Double descent and grokking.** Superposition gives a lens on the strange generalization curves of modern models. The packing problem has a critical point where the number of features the model is trying to represent crosses the capacity it has, and behavior reorganizes sharply across that point — reminiscent of the double-descent risk curve and of grokking's delayed, sudden generalization. The paper is careful to present these as suggestive connections rather than proofs, but the shared ingredient — a discrete reorganization of representational geometry as a control parameter crosses a threshold — is hard to ignore.

**Neurons are not the unit, so we need sparse autoencoders.** This is the consequence with the largest practical footprint. If the true features are an overcomplete set of directions in superposition, then the right tool to recover them is one that *learns an overcomplete dictionary of sparse directions*. That is exactly a **sparse autoencoder** (SAE): train a wide, sparse autoencoder on a layer's activations and its dictionary atoms approximate the superposed features, pulling them back out into a monosemantic basis. The entire modern wave of dictionary-learning interpretability is, in a real sense, an engineering response to the toy-models result — and it inherits the toy model's headaches, like feature splitting and absorption, which I dig into in [A is for Absorption](/blog/paper-reading/ai-interpretability/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders) and whose practical payoff is scrutinized in [Are Sparse Autoencoders Useful?](/blog/paper-reading/ai-interpretability/are-sparse-autoencoders-useful-a-case-study-in-sparse-probing).

It is worth flagging how unusual this explanatory reach is. Most findings in deep learning explain one thing. Superposition is closer to a structural law: it starts from a single assumption — that real features are sparse and more numerous than neurons — and from that one assumption it derives polysemanticity, predicts a specific failure mode for adversarial robustness, suggests a mechanism for sharp generalization transitions, and prescribes the exact shape of the tool needed to reverse it. When a small model with a clear assumption set predicts phenomena across vision, language, security, and optimization at once, that is the signature of having found something real rather than a coincidence. The flip side is a warning: because the mechanism is so general, it is easy to over-apply. Not every polysemantic neuron is superposition — some genuinely track one broad feature — not every adversarial example rides an interference direction, and not every training curve with a bump is a superposition phase change. The discipline the paper models — build the smallest system that exhibits the phenomenon, vary one knob, read the geometry — is the antidote.

One mechanism — features sharing dimensions because they are sparse — quietly underwrites polysemanticity, a theory of adversarial examples, a lens on double descent, and the justification for the dominant interpretability tool of the decade. That reach is why this little paper about five features and two neurons is cited everywhere.

## Experiments and phenomena: what the toy models reveal

The paper earns its conclusions through a sequence of small, decisive experiments. Below are ten of them, framed as the findings a careful reader walks away with. Each is reproducible with the code above.

### 1. The dense regime is just PCA

Train the $n=5$, $m=2$ model on dense data and the result is indistinguishable from principal component analysis: the two most important features get orthogonal axes, the other three are dropped, and $W^\top W$ is a rank-2 projector onto the top components. The lesson is that **without sparsity there is no superposition at all** — the model has no incentive to overlap, because overlap is pure cost when every feature is always on. This is the control condition that proves sparsity is the active ingredient. It also quietly confirms that the linear-algebra workhorse everyone already knows, PCA, is the $S=0$ corner of the same phenomenon; superposition is what you get when you push that corner toward sparsity.

### 2. The sparse regime produces a pentagon

Flip sparsity high and retrain the same architecture and all five features appear, arranged as a regular pentagon in the 2D hidden plane. The model has chosen to represent $5$ features in $2$ dimensions, a $2.5\times$ overcommit, accepting interference between all pairs because that interference rarely fires. The pentagon is not hand-designed; it emerges from gradient descent on reconstruction loss. Seeing a clean five-pointed star fall out of a generic optimizer on a generic loss is the moment the paper's thesis becomes visceral: the geometry is a *discovered* solution to a packing problem, and the optimizer finds the same regular figure a mathematician would.

### 3. The antipodal pair is the atom of superposition

The simplest superposition is two features sharing one dimension by pointing in opposite directions, each receiving $D = \tfrac{1}{2}$. Studying it in isolation reveals the core trick: the output ReLU converts the maximally-negative interference $W_1 \cdot W_2 = -\lVert W_1\rVert^2$ from a liability into a free clipping mechanism. When one feature is on and the other off, the off feature's reconstruction is pushed negative and clipped to zero at no cost. Every richer polytope is, in a sense, many antipodal-style trades stacked together. Understanding this one configuration end to end is the fastest way to understand the whole paper, which is why it is worth building the two-feature, one-dimension model by hand and watching the ReLU do its job.

### 4. The triangle and the Thomson problem

Three equally-important sparse features in two dimensions arrange at $120°$ — an equilateral triangle of directions, each with $D = \tfrac{2}{3}$. This is the minimal-energy configuration for three mutually repelling points on a circle, the 2D Thomson problem. The match is exact because the loss, after you account for sparsity and importance, is a sum of pairwise interference penalties that behave like repulsion. The takeaway is conceptual: superposition is not ad-hoc cramming, it is **energy minimization over feature geometry**, and the rich literature on optimal point configurations on spheres becomes directly relevant to predicting what a network will do. It also explains a practical observation that surprises people: the configurations are highly reproducible. A Thomson-style minimum is a strong attractor, so independent training runs from different random seeds land on the *same* polytope, up to a global rotation. That cross-seed reproducibility is itself evidence that the geometry is a genuine optimum rather than an artifact of one lucky initialization — if the pentagon were noise, you would not see it reappear run after run.

### 5. Tetrahedra and higher-dimensional packings

Give the model three hidden dimensions and four equally-important sparse features and they go to the corners of a regular tetrahedron, $D = \tfrac{3}{4}$ each. Add more features and you climb the ladder of uniform polytopes — square antiprisms, and other configurations from the same family of optimal packings. The point is that the geometry is not a 2D curiosity; it generalizes, and each polytope comes with its own quantized dimensionality. As the hidden width grows, the menu of available stable configurations grows with it, which is part of why larger models can represent dramatically more features without the geometry degenerating into noise.

### 6. Importance carves a hierarchy

Make features differ in importance and the uniform polytopes give way to a hierarchy: important features claim dedicated near-orthogonal directions, unimportant ones huddle into shared directions. The same run thus contains both monosemantic ($D \approx 1$) and heavily polysemantic ($D$ small) features, sorted by how much they matter to the loss. This is the toy-model echo of the real-world observation that the interpretable neurons in a vision model tend to be the salient, behaviorally-important features. Importance is the model's budget allocator, and superposition is what happens to everything below the budget line.

### 7. The phase diagram has crisp boundaries

Sweep importance and sparsity together and color each feature by its representation regime, and you get three regions — not represented, dedicated, superposed — separated by sharp boundaries. Crossing a boundary flips a feature's storage strategy abruptly. The crispness matters: it means a model's representational choices are not a smooth fog but a set of discrete decisions with thresholds, which is encouraging for interpretability because discrete structure is easier to reverse-engineer than a continuum. It also means small changes in data statistics (a feature becoming a bit more common, a bit more important) can cause discontinuous changes in how it is stored.

### 8. The absolute-value model computes in superposition

Retarget the model from reconstruction to computing $y_i = \lvert x_i \rvert$ and it succeeds while the features remain superposed in the bottleneck. This is the experiment that proves computation, not just storage, happens in the tangled basis. There is no clean monosemantic layer where the real work occurs; the ReLU and the up-projection do arithmetic directly on overlapping directions, using sparsity to keep the interference survivable. For anyone hoping that features could be "decompressed" before computation, this is the disappointing-but-important result, and it sets the difficulty bar for circuit-level interpretability.

### 9. Interference directions explain adversarial fragility

The off-diagonal entries of $W^\top W$ are explicit interference channels: a perturbation aligned with $W_j$ leaks into the readout of $W_i$ at strength $W_i \cdot W_j$. In a superposed model these channels are everywhere, and they are exactly the small, dense directions an adversarial perturbation rides to flip an output without looking different to a human. The experiment connects a security phenomenon to a representational one: the same overlap that grants extra capacity is a built-in attack surface. It reframes adversarial robustness as partly a question about how aggressively a model superposes, and suggests that perfectly robust and maximally capacity-efficient may be in tension. The connection runs both ways and hands you a falsifiable prediction for free: a model trained to be more robust should superpose *less*, trading capacity for a cleaner, more orthogonal representation, and adversarial training should visibly shrink the off-diagonal mass of the relevant weight matrices. That turns robustness work, in part, into the deliberate management of how much a model is allowed to superpose — a very different framing from the usual "add perturbed examples and hope."

### 10. The toy is not a toy

The final, load-bearing claim is that everything above transfers. The configurations, the phase change, the sticky dimensionalities, and the polysemanticity all show up because of generic ingredients — sparse features, limited width, a pointwise nonlinearity, importance variation — that are present in every real network. The paper does not prove the geometry of GPT is literally pentagons, but it establishes that the *mechanism* is not an artifact of the toy. That is what licenses using superposition as a working hypothesis about models a billion times larger, and what turned a study of five features and two neurons into the conceptual foundation for sparse-autoencoder interpretability. The leap from toy to frontier model is a genuine assumption, stated plainly and left open rather than smuggled in — and the decade of dictionary-learning work that followed is, in effect, the ongoing experiment that tests whether the leap holds.

## When to reach for superposition as an explanation — and when not to

Superposition is a powerful lens, but like any lens it distorts if you point it at the wrong thing. Some guidance on where it earns its keep.

**Reach for superposition when:**

- You find a neuron that fires for several genuinely unrelated inputs and no single human concept unifies them. Polysemanticity is the textbook fingerprint of superposition, and it should be your first hypothesis, not your last.
- You are trying to estimate how many distinct features a layer can hold. The answer is "many more than its width," and how many more depends on how sparse the features are. Planning interpretability tooling around "one neuron, one feature" will badly undercount.
- You are choosing an interpretability method and need to justify why a sparse, overcomplete dictionary (an SAE) is the right shape. The justification is superposition: the features are an overcomplete set of sparse directions, so you learn an overcomplete sparse dictionary to recover them.
- You see sharp, threshold-like changes in a model's behavior or representations as you vary scale, data sparsity, or training time. The phase-transition character of superposition is a candidate explanation for discontinuities.
- You are reasoning about adversarial robustness and want a structural, rather than purely empirical, account of why imperceptible perturbations work.

**Be skeptical, or look elsewhere, when:**

- The basis is not privileged. Superposition's "neurons go polysemantic" story only bites where an elementwise nonlinearity (or another symmetry-breaking operation) makes the neuron basis special. In a purely linear sub-block, "which neuron means what" was never a meaningful question, so polysemanticity there is expected for a different reason and superposition is not the interesting lens.
- The features are dense. If the features you care about are active most of the time, superposition is a bad deal and the model will not use it; you should expect PCA-like dedicated directions, and a superposition story will mislead you.
- You are tempted to treat the polytope geometry as literal for a large model. The toy model's pentagons and tetrahedra are exact in the toy; in a real network the geometry is vastly higher-dimensional and messier. Use superposition for the mechanism and the qualitative predictions, not to claim a transformer literally embeds a pentagon.
- A simpler explanation fits. A neuron that fires for cats and also for things that merely *look* cat-adjacent in pixel space might be tracking one visual feature, not two semantic ones in superposition. Rule out the mundane single-feature explanation before invoking overlap.
- You need certainty about a specific feature's direction. Superposition tells you the directions exist and overlap; recovering a *particular* feature's direction faithfully is the job of dictionary learning, and that step has its own well-documented failure modes (feature splitting, absorption, dead latents) that superposition theory alone does not resolve.

The honest summary is that superposition is the best available explanation for why neurons resist interpretation, and it reframes the goal from "read the neurons" to "recover the directions." It does not by itself hand you the directions — but it tells you they are there, how many to expect, and why your tooling has to be overcomplete and sparse to find them.

## A few misconceptions worth clearing up

Because superposition is now a load-bearing idea in interpretability, it has accumulated a cloud of half-truths. A few are worth correcting directly.

**"Superposition means the model is confused or undertrained."** The opposite. Superposition is the *optimal* solution to the packing problem the model faces, found by a well-trained network minimizing its loss. A model that refused to superpose would represent fewer features and have higher loss. The overlap is a feature of competent optimization under a capacity constraint, not a symptom of failure. Training longer or harder does not remove superposition; if anything it sharpens the geometry.

**"More neurons would eliminate superposition."** Only at the margin. Adding width promotes the most important superposed features to dedicated dimensions, but the long tail of sparse, less-important features will keep getting packed in, because packing them is still the best use of the new capacity. There is essentially always a tail of features the model would rather superpose than drop, so superposition persists at every scale we can train — it just moves to ever-less-important features as width grows.

**"If features are directions, I can just read them off with PCA."** PCA finds the top orthogonal directions of variance, which is exactly the *dense* solution — it recovers the few high-importance features and is blind to everything in superposition. The whole point is that the interesting features are an overcomplete, non-orthogonal set, and an orthogonal method cannot recover an overcomplete basis. This is precisely the gap that sparse autoencoders exist to fill: they learn an overcomplete dictionary with a sparsity penalty, which is the right shape for the problem.

**"The pentagon is literally how language models store concepts."** No. The polytopes are exact in the toy model with uniform importance and a tiny dimension count. In a real model the geometry is enormously high-dimensional, importance varies wildly, and the configurations are far messier than regular polygons. What transfers is the *mechanism* — sparse features sharing directions, interference suppressed by nonlinearity, capacity scaling with $1/(1-S)$ — and the qualitative predictions, not the specific shapes.

**"Superposition and polysemanticity are the same thing."** They are cause and symptom. Polysemanticity is an observation about *neurons* — a single unit responds to several unrelated inputs. Superposition is a hypothesis about *representations* — features are stored as overlapping directions. Superposition predicts polysemanticity, but you can in principle have polysemantic neurons for other reasons, and the distinction matters when you debug a specific model: confirming superposition requires showing the features are recoverable directions, not just that a neuron is messy.

**"Sparse autoencoders solve superposition."** They are the right tool, and they recover a great deal, but they inherit the toy model's hard cases. Feature splitting, absorption, and dead latents are all consequences of trying to read clean atoms off interfering polytopes, and they mean an SAE's "features" are partly a function of the SAE's own hyperparameters. Superposition tells you the features are there and roughly how many to expect; recovering them faithfully is an open engineering problem, not a checkbox.

The throughline is that superposition is a precise, predictive hypothesis, and most misconceptions come from treating it as either a vague vibe ("the model is messy") or an over-literal claim ("it's a pentagon"). Held at the right altitude — a mechanism with sharp, testable consequences — it is one of the most useful ideas in interpretability.

## Further reading

- Elhage et al., [*Toy Models of Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html) — the primary source for everything in this post; the interactive figures are worth the visit.
- [The Linear Representation Hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis) — the "features are directions" foundation that superposition builds on.
- [A is for Absorption: feature splitting and absorption in sparse autoencoders](/blog/paper-reading/ai-interpretability/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders) — what goes wrong when you try to pull superposed features back out with an SAE.
- [Are Sparse Autoencoders Useful? A case study in sparse probing](/blog/paper-reading/ai-interpretability/are-sparse-autoencoders-useful-a-case-study-in-sparse-probing) — a hard-nosed look at whether decoding superposition actually buys you anything downstream.
