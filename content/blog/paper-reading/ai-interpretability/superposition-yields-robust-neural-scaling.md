---
title: "Superposition Yields Robust Neural Scaling: the geometry behind why wider models keep winning"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "A detailed, intuition-first walkthrough of Liu, Liu & Gore's NeurIPS 2025 paper, which argues that feature superposition — packing more features than dimensions — is the geometric reason neural scaling laws exist, and why the loss so stubbornly falls as one over model width."
tags:
  - superposition
  - scaling-laws
  - neural-scaling-laws
  - interpretability
  - mechanistic-interpretability
  - large-language-model
  - chinchilla
  - toy-models
  - representation-learning
  - paper-reading
category: paper-reading
subcategory: AI Interpretability
author: "Hiep Tran"
featured: true
readTime: 29
---

We have spent five years building bigger models because a graph told us to. Double the parameters, and the loss drops by a predictable sliver. Double again, same sliver. Plot it on log–log axes and you get a near-perfect straight line spanning four orders of magnitude — the neural scaling law. It is the empirical bedrock of the entire frontier-model industry, and yet, embarrassingly, nobody could say *why* the line is straight, or what sets its slope.

[Superposition Yields Robust Neural Scaling](https://arxiv.org/abs/2505.10465) (Yizhou Liu, Ziming Liu, and Jeff Gore, MIT — NeurIPS 2025) gives the cleanest mechanistic answer I have read. Its claim is almost provocative in its simplicity: scaling laws exist because neural networks practice **superposition** — they cram far more features into a representation than it has dimensions — and the geometry of that crammed space forces the loss to fall as roughly one over the model width. The diagram below is the mental model for the whole paper: on the left, three features fit comfortably into three orthogonal directions; on the right, seven features are forced to share the same three axes, so their representation vectors must overlap, and overlap means interference.

![Superposition: three orthogonal vectors for three features on the left, versus seven overlapping vectors crammed into the same three axes on the right](/imgs/blogs/superposition-yields-robust-neural-scaling-1.webp)

That picture is the engine of everything that follows. This post walks through the toy model the authors built, the two scaling regimes it exposes, the surprisingly elegant geometry that produces the robust one-over-width law, and the evidence that four families of real open-source LLMs live squarely in that regime. I will define every symbol, give you runnable code for the load-bearing pieces, and end with where I think the argument is strong, where it is hand-wavy, and what I would build on top of it.

> [!tldr]
> - **Claim.** Representation *superposition* — storing more features than there are dimensions — is a central cause of neural scaling laws. The authors reproduce power-law loss in a toy model and tie its exponent to one knob: the degree of superposition.
> - **Two regimes.** Under *weak* superposition, the loss is a power law in model width *only if* the data's feature frequencies are themselves a power law, and it inherits that exponent ($\alpha_m = \alpha - 1$). Under *strong* superposition, the loss falls as $1/m$ across a broad class of frequency distributions — a robust law with exponent near 1.
> - **The surprise.** That robust $1/m$ is pure geometry: random unit vectors in $m$ dimensions have a mean squared overlap of exactly $1/m$, so interference — and therefore loss — shrinks as you add width, no matter how the feature frequencies are shaped.
> - **Reality check.** Four open LLM families (OPT, GPT-2, Qwen, Pythia, 70M–70B) show language-model-head token overlaps scaling as $1/m$ and a measured width exponent $\alpha_m = 0.91 \pm 0.04$. The Chinchilla scaling law is consistent with the same number.
> - **Where it breaks.** The toy model omits transformer layers and uses squared error, not cross-entropy. It predicts the power law must *end* once width reaches the vocabulary size, and that domains with extremely skewed feature frequencies could scale faster than $1/m$.

## Context: what came before

Neural scaling laws were first nailed down empirically by [Kaplan et al.](/blog/machine-learning/scaling-laws/kaplan-scaling-laws-language-models) and then refined into the compute-optimal recipe by [the Chinchilla paper](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling). The headline is that cross-entropy loss improves as a power law in model size $N$, dataset size $D$, and compute $C$, smoothly, over many orders of magnitude. That regularity is what makes frontier planning possible: you can spend a few million dollars on a sweep of small runs, fit a curve, and extrapolate the loss of a model you have not trained yet.

The discomfort is that "the loss is a power law" has had *too many* explanations, none of them obviously correct and most of them mutually compatible only by coincidence. There are data-manifold arguments (loss decays with the intrinsic dimension of the data manifold), kernel-spectrum arguments (loss tracks the power-law tail of the kernel's eigenvalue spectrum), and discrete-skill or quantization arguments (the network learns a sequence of skills whose importances follow a power law). Several of these are surveyed and partly unified by [Bahri et al.](/blog/machine-learning/scaling-laws/why-power-laws-arise). What almost all of them share is an *assumption*: that some underlying quantity — manifold curvature, kernel eigenvalues, skill importance — is already power-law distributed. They convert one power law into another. That is useful, but it pushes the mystery back a step rather than dissolving it.

These prior models also tend to live, implicitly, in what this paper calls the *weak superposition* regime — a regime where the network represents only the features it can afford to represent cleanly, and the loss comes from the ones it dropped. That is a reasonable picture for an over-parameterized model fitting a smooth function. It is a much less obvious picture for a large language model, which must map a vocabulary of fifty-thousand-plus tokens (and a far larger space of abstract concepts) into a hidden space of at most a few thousand dimensions. When the number of things you want to represent vastly exceeds the dimensions you have, you are no longer choosing which features to keep — you are choosing how to overlap them. That is the gap this paper sets out to fill: a mechanistic account of scaling that is built for the regime LLMs actually operate in.

## The two principles everything rests on

The argument stands on two empirical observations, both of which are uncontroversial on their own:

1. **LLMs represent more features than they have dimensions.** This is superposition, the phenomenon catalogued in Anthropic's [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) and the foundation of a lot of modern interpretability work. If you have not met the idea before, the companion post on [what superposition is](/blog/machine-learning/ai-interpretability/what-is-superposition) and the [linear representation hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis) are the right primers. The short version: a network with a width-$m$ hidden layer can encode $n \gg m$ distinct features by assigning each one a direction, accepting that the directions cannot all be orthogonal.
2. **Features occur at wildly different frequencies.** Words follow Zipf's law; concepts, idioms, and code tokens are similarly skewed. A handful of features fire in almost every input; a long tail fire rarely. This frequency structure is what lets the network *prioritize* — represent the common features well and the rare ones poorly, or not at all.

Hold these two facts together and a question falls out, the one the paper organizes itself around: **as you vary the degree of superposition and the skew of the data, when is the loss a power law in model width, and what sets the exponent?** The rest of the paper is the answer, and it splits cleanly into two regimes that we will take one at a time.

The three stated contributions, tightened:

- **In weak superposition**, the loss is governed by the summed frequency of the features the model ignored; if those frequencies follow a power law, so does the loss, with a directly inherited exponent.
- **In strong superposition**, the loss arises from interference between overlapping representation vectors, and because of the geometry of high-dimensional space it scales as a robust $1/m$ across a wide range of frequency distributions.
- **Real LLMs** sit in the strong-superposition regime and quantitatively match the toy model's prediction, and the Chinchilla scaling law is consistent with the same behavior.

## The toy model: representation as data recovery

To study representation in isolation, you want a model that does *only* representation — no attention, no next-token prediction, nothing but the act of squeezing features through a bottleneck and reading them back out. The authors adopt Anthropic's superposition autoencoder with minor changes to the data sampling.

![The toy model is an autoencoder: data x is encoded by W-transpose into a hidden vector h of width m, decoded by W, passed through a ReLU with bias, and the reconstruction loss measures representation quality](/imgs/blogs/superposition-yields-robust-neural-scaling-2.webp)

Here is the whole setup. The input $x \in \mathbb{R}^n$ has one coordinate per feature, where $n$ is large and fixed (the experiments use values like $n = 1000$ or $n = 10240$). Each coordinate is

$$x_i = u_i v_i, \qquad u_i \sim \text{Bernoulli}(p_i), \qquad v_i \sim U(0, 2).$$

The Bernoulli gate $u_i$ decides *whether* feature $i$ is active in this sample; the uniform draw $v_i$ sets *how strongly*. The probability $p_i$ is the feature's **frequency**, and by convention features are indexed so that $p_i$ decreases with rank $i$ — feature 1 is the most common, feature $n$ the rarest. The expected number of active features in one sample, $E = \sum_{i=1}^n p_i$, is the **activation density**; small $E/n$ means the data is *sparse*, which is exactly the language-like regime.

The model is a single weight matrix $W \in \mathbb{R}^{n \times m}$ with $m \ll n$ and a bias $b \in \mathbb{R}^n$. It encodes the input into the hidden space, $h = W^\top x$, then decodes and rectifies:

$$y = \text{ReLU}(W W^\top x + b).$$

The loss is the average squared reconstruction error, $L = \langle \lVert y - x \rVert_2^2 \rangle_x$, where $\langle \cdot \rangle_x$ averages over the data distribution. The single most important piece of notation: $W_i$, the $i$-th **row** of $W$, is the representation of feature $i$ in the hidden space. Feature $i$ is *represented* when $W_i$ is non-zero; it is *ignored* when $W_i = 0$. No superposition means the first $m$ rows of $W$ form an orthogonal basis and the rest are zero — the top $m$ features get clean, interference-free directions and everyone else is dropped. Superposition means more than $m$ rows are non-zero, so the representation vectors must overlap.

One subtlety carries the whole story, and it is worth stating plainly: **without the ReLU, superposition cannot help.** In a purely linear model, the cross-talk from overlapping vectors is pure additive noise on the reconstruction, and stuffing in more features only raises the loss. The ReLU plus a negative bias is what makes superposition pay: the non-linearity can clip away small interference terms, so a feature that is mostly off most of the time can be packed into a shared direction and still be recovered when it fires, because the bias suppresses the leakage from everything else. Sparsity (most features off) and a frequency-aware bias are what make this error correction work. That is why this regime is interesting at all.

Here is the model as runnable PyTorch — the data sampler, the forward pass, and the loss, which is all you need to reproduce the qualitative behavior:

```python
import torch

def sample_batch(p, batch_size, v_max=2.0):
    """Draw x_i = u_i * v_i, u_i ~ Bernoulli(p_i), v_i ~ U(0, v_max)."""
    n = p.shape[0]
    u = (torch.rand(batch_size, n) < p).float()      # which features fire
    v = torch.rand(batch_size, n) * v_max            # activation strength
    return u * v                                     # shape (batch_size, n)

class SuperpositionAE(torch.nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(n, m) / m**0.5)  # rows = features
        self.b = torch.nn.Parameter(torch.zeros(n))

    def forward(self, x):
        h = x @ self.W                     # encode: h = W^T x, width m
        y = torch.relu(h @ self.W.T + self.b)   # decode + error-correcting ReLU
        return y, h

# Zipf-like feature frequencies p_i ~ 1 / i^alpha, then scaled to a target density E.
n, m, alpha, E = 1000, 64, 1.0, 1.0
ranks = torch.arange(1, n + 1).float()
p = ranks ** (-alpha)
p = p / p.sum() * E                        # so that sum_i p_i = E
model = SuperpositionAE(n, m)
x = sample_batch(p, batch_size=4096)
y, h = model(x)
loss = ((y - x) ** 2).sum(dim=1).mean()    # < ||y - x||^2 >
print(loss.item())
```

### Sparse, power-law data

Two properties of the data do all the work, so it is worth seeing them as a picture rather than a pair of equations.

![Frequency-rank bars decaying steeply from rank 1, with the top ranks colored as represented and the rare tail colored as dropped, plus cards describing the generative model](/imgs/blogs/superposition-yields-robust-neural-scaling-3.webp)

The bars are the feature frequencies $p_i$, sorted by rank. A handful of features (the tall blue bars on the left) dominate; the frequency decays steeply as $p_i \propto 1/i^{\alpha}$ — Zipf's law is the $\alpha = 1$ case. The rare tail on the right is what a width-limited model is forced to give up first, either by ignoring those features outright (weak superposition) or by representing them with the most overlap and therefore the most interference (strong superposition). The two cards on the right restate the generative model: each sample is a Bernoulli gate times a uniform magnitude, and the frequencies decay with rank. Everything downstream is a consequence of how steep that decay is and how much the model overlaps its vectors.

### The knob: tuning superposition with weight decay

The clever experimental move is that the authors do not just *observe* whether superposition happens — they *control* it, with a decoupled weight-decay (or weight-growth) term applied per row of $W$. Writing $W_{i,t}$ for row $i$ at training step $t$ and $\eta_t$ for the learning rate:

$$
W_{i,t+1} =
\begin{cases}
W_{i,t} - \eta_t\, \gamma\, W_{i,t}, & \gamma \ge 0, \\[4pt]
W_{i,t} - \eta_t\, \gamma\, W_{i,t}\!\left(\dfrac{1}{\lVert W_{i,t} \rVert_2} - 1\right), & \gamma < 0.
\end{cases}
$$

For positive $\gamma$ this is ordinary weight decay: it pushes row norms toward zero, so only the most important features survive with non-trivial norm — *weak* superposition. For negative $\gamma$ the update is gradient descent on $(\lVert W_{i,t}\rVert_2 - 1)^2$, which *grows* small rows toward unit norm and so encourages every feature to claim a direction — *strong* superposition. The sign of one hyperparameter slides the model continuously between the two regimes. That is the dial in the figure below.

<figure class="blog-anim">
<svg viewBox="0 0 720 380" role="img" aria-label="A weight-decay dial sweeps from positive decay, where only a few orthogonal feature vectors survive, to negative decay, where all features are represented as overlapping vectors" style="width:100%;height:auto;max-width:820px">
<style>
.wd-axis{stroke:var(--border,#d1d5db);stroke-width:3}
.wd-vec{stroke:var(--accent,#6366f1);stroke-width:5;stroke-linecap:round}
.wd-vecq{stroke:var(--text-secondary,#9ca3af);stroke-width:5;stroke-linecap:round}
.wd-lbl{font:600 17px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.wd-sub{font:500 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.wd-knob{fill:var(--accent,#6366f1)}
@keyframes wd-fadeA{0%,34%{opacity:1}50%,90%{opacity:0}100%{opacity:1}}
@keyframes wd-fadeB{0%,34%{opacity:0}50%,90%{opacity:1}100%{opacity:0}}
@keyframes wd-knob{0%,34%{transform:translateX(0)}50%,90%{transform:translateX(-470px)}100%{transform:translateX(0)}}
.wd-A{animation:wd-fadeA 9s ease-in-out infinite}
.wd-B{animation:wd-fadeB 9s ease-in-out infinite}
.wd-k{animation:wd-knob 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.wd-A{animation:none;opacity:0}.wd-B{animation:none;opacity:1}.wd-k{animation:none;transform:translateX(-470px)}}
</style>
<text class="wd-lbl" x="240" y="40">hidden space (m = 3 directions)</text>
<g class="wd-A">
<line class="wd-vec" x1="240" y1="150" x2="240" y2="60"/>
<line class="wd-vec" x1="240" y1="150" x2="165" y2="225"/>
<line class="wd-vec" x1="240" y1="150" x2="315" y2="225"/>
<circle class="wd-vecq" cx="240" cy="150" r="6" fill="var(--text-secondary,#9ca3af)" stroke="none" opacity="0.6"/>
<text class="wd-sub" x="240" y="270">3 clean vectors, rest collapsed to zero</text>
</g>
<g class="wd-B">
<line class="wd-vec" x1="240" y1="150" x2="240" y2="62"/>
<line class="wd-vec" x1="240" y1="150" x2="312" y2="100"/>
<line class="wd-vec" x1="240" y1="150" x2="328" y2="178"/>
<line class="wd-vec" x1="240" y1="150" x2="290" y2="232"/>
<line class="wd-vec" x1="240" y1="150" x2="190" y2="232"/>
<line class="wd-vec" x1="240" y1="150" x2="152" y2="178"/>
<line class="wd-vec" x1="240" y1="150" x2="168" y2="100"/>
<text class="wd-sub" x="240" y="270">7 features share 3 axes, all overlapping</text>
</g>
<text class="wd-A wd-lbl" x="240" y="300">weak superposition</text>
<text class="wd-B wd-lbl" x="240" y="300">strong superposition</text>
<line class="wd-axis" x1="120" y1="340" x2="600" y2="340"/>
<text class="wd-sub" x="120" y="368">gamma &lt; 0  (weight growth)</text>
<text class="wd-sub" x="600" y="368">gamma &gt; 0  (weight decay)</text>
<circle class="wd-k wd-knob" cx="600" cy="340" r="12"/>
<text class="wd-lbl" x="560" y="150">the weight-decay knob</text>
<text class="wd-sub" x="560" y="178">one sign flips the regime</text>
</svg>
<figcaption>Turn the weight-decay dial: positive decay keeps only a few orthogonal vectors (weak superposition); negative decay packs every feature in as overlapping vectors (strong superposition).</figcaption>
</figure>

Empirically the row norms become *bimodal* — they cluster near 0 (ignored) or near 1 (represented) — which lets the authors define a clean order parameter, the **represented fraction**

$$\phi_{1/2} = \frac{\lvert \{\, i : \lVert W_i \rVert_2 > 1/2 \,\} \rvert}{n},$$

the fraction of features whose representation vector has norm above one half. Large weight decay gives $\phi_{1/2} \approx m/n$ (only about $m$ features represented — weak superposition); small or negative weight decay gives $\phi_{1/2} \approx 1 \gg m/n$ (essentially every feature represented — strong superposition). With one dial and one order parameter, you can now scan scaling behavior across the whole spectrum. In practice they fix $n = 1000$, sweep $m$ from 10 to 100, sweep $\gamma$ from $-1$ to $1$, fit the final test loss as $L \propto 1/m^{\alpha_m}$, and call $\alpha_m$ the **model exponent**.

## Regime 1 — Weak superposition: power law in, power law out

Start with the regime prior work mostly lived in. Under weak superposition only the top $\phi_{1/2}\,n$ most frequent features are represented, each essentially without overlap, and the rest are ignored. The figure below is the side-by-side that organizes both regimes; read its left column now and we will come back to the right.

![Two-column comparison: weak superposition keeps only top features and the loss equals the sum of ignored frequencies giving exponent alpha minus one; strong superposition represents all features with overlapping vectors and the loss is interference scaling as one over m](/imgs/blogs/superposition-yields-robust-neural-scaling-4.webp)

If the model has stored the top $k = \phi_{1/2}\,n$ features perfectly and dropped the rest, what is the loss? The optimal bias turns out to be $b_i = 0$ for represented features and $b_i = \langle x_i \rangle$ for ignored ones — that is, for a feature it cannot reconstruct, the best the model can do is output that feature's mean value. The loss is then the variance of each ignored feature, summed:

$$L = \sum_{i > k} \big\langle (x_i - \langle x_i \rangle)^2 \big\rangle = \sum_{i > k} \big( \langle v^2 \rangle\, p_i - \langle v \rangle^2\, p_i^2 \big) \;\approx\; \langle v^2 \rangle \sum_{i > k} p_i.$$

The last approximation holds because the data is sparse: for the rare ignored features $p_i \ll 1$, so the $p_i^2$ term is negligible. With $v \sim U(0, 2)$ we have $\langle v^2 \rangle = 4/3$, a constant. **The loss is just a constant times the total frequency of the features the model did not learn.** That is the entire mechanism of weak-superposition scaling, and it is almost tautological once you see it: the error is whatever you threw away.

Now plug in the data. If $p_i \propto 1/i^{\alpha}$ with $\alpha > 1$, the tail sum behaves like an integral,

$$\sum_{i > k} p_i \;\approx\; \int_{k}^{n} i^{-\alpha}\, di \;\propto\; k^{-(\alpha - 1)},$$

and in the near-no-superposition case where $k \approx m$, the loss scales as $L \propto m^{-(\alpha - 1)}$. So the model exponent is

$$\boxed{\alpha_m = \alpha - 1.}$$

This is the "power law in, power law out" result. The loss is a power law in width **if and only if** the feature frequencies are a power law, and the loss exponent is exactly one less than the frequency exponent. If your data's frequencies decay exponentially or linearly instead, the loss is *not* a clean power law in this regime — which the toy model confirms directly: with linear or exponential feature-importance decay, the weak-superposition loss curves bend on log–log axes rather than running straight. This recovers, from a concrete mechanism, what several earlier theories simply assumed: that a power-law spectrum somewhere upstream is what produces a power-law loss. The catch is that it is *conditional* on the data, and it requires $\alpha > 1$. It is a fragile law. The interesting regime is the other one.

## Regime 2 — Strong superposition: the geometric origin of 1/m

Now turn the dial the other way. Under strong superposition every feature is represented, but the vectors $W_i$ overlap. When a single feature $j$ fires, the decoded output on coordinate $i$ picks up a spurious term proportional to $W_i \cdot W_j$, and because the loss is squared error, the damage scales as the **squared overlap** $(W_i \cdot W_j)^2$. The total loss is the accumulated interference from all these pairwise overlaps. The question becomes: how do those squared overlaps scale with the width $m$?

This is where the paper earns its title, because the answer is geometry, not data. Consider the simplest model of "all features represented roughly equally": the vectors $W_i / \lVert W_i \rVert$ are isotropic — independent and uniform on the unit sphere in $\mathbb{R}^m$. For two such random unit vectors, the squared overlap (squared cosine of the angle between them) follows a $\text{Beta}\!\left(\tfrac{1}{2}, \tfrac{m-1}{2}\right)$ distribution, which has

$$\mathbb{E}\big[(\hat W_i \cdot \hat W_j)^2\big] = \frac{1}{m}, \qquad \text{Var} = \frac{2(m-1)}{m^2(m+2)} \sim \frac{2}{m^2}.$$

That mean is the crux of the entire paper. **Two random directions in $m$-dimensional space have a mean squared overlap of exactly $1/m$.** High-dimensional space is mostly orthogonal: the more dimensions you have, the closer any two random vectors are to perpendicular, and the overlap — hence the interference, hence the loss — falls as $1/m$. It does not matter how the feature frequencies are shaped, as long as the vectors fill space roughly isotropically. The robustness is not an accident of the data; it is a property of spheres in high dimensions.

The animation makes the mechanism concrete: as the width $m$ grows, the off-diagonal entries of the overlap matrix $W W^\top$ fade toward zero, and the interference loss slides down a $1/m$ curve.

<figure class="blog-anim">
<svg viewBox="0 0 720 360" role="img" aria-label="As model width m grows, the off-diagonal interference in the overlap matrix fades toward zero and the loss point slides down a one-over-m curve" style="width:100%;height:auto;max-width:820px">
<style>
.ov-diag{fill:var(--accent,#6366f1)}
.ov-if{fill:var(--accent,#6366f1)}
.ov-ax{stroke:var(--text-secondary,#9ca3af);stroke-width:2.5}
.ov-curve{fill:none;stroke:var(--accent,#6366f1);stroke-width:3;stroke-dasharray:5 5;opacity:.55}
.ov-dot{fill:var(--accent,#6366f1)}
.ov-mk{fill:var(--text-primary,#1f2937)}
.ov-lbl{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.ov-sub{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
@keyframes ov-fade{from{opacity:.85}to{opacity:.08}}
@keyframes ov-fall{from{transform:translate(0,0)}to{transform:translate(250px,150px)}}
@keyframes ov-slide{from{transform:translateX(0)}to{transform:translateX(250px)}}
.ov-f{animation:ov-fade 9s ease-in-out infinite alternate}
.ov-d{animation:ov-fall 9s ease-in-out infinite alternate}
.ov-s{animation:ov-slide 9s ease-in-out infinite alternate}
@media (prefers-reduced-motion:reduce){.ov-f{animation:none;opacity:.08}.ov-d{animation:none;transform:translate(250px,150px)}.ov-s{animation:none;transform:translateX(250px)}}
</style>
<text class="ov-lbl" x="155" y="56">overlap matrix  W Wt</text>
<rect class="ov-diag" x="60" y="80" width="46" height="46" rx="4"/>
<rect class="ov-if ov-f" x="110" y="80" width="46" height="46" rx="4"/>
<rect class="ov-if ov-f" x="160" y="80" width="46" height="46" rx="4"/>
<rect class="ov-if ov-f" x="210" y="80" width="46" height="46" rx="4"/>
<rect class="ov-if ov-f" x="60" y="130" width="46" height="46" rx="4"/>
<rect class="ov-diag" x="110" y="130" width="46" height="46" rx="4"/>
<rect class="ov-if ov-f" x="160" y="130" width="46" height="46" rx="4"/>
<rect class="ov-if ov-f" x="210" y="130" width="46" height="46" rx="4"/>
<rect class="ov-if ov-f" x="60" y="180" width="46" height="46" rx="4"/>
<rect class="ov-if ov-f" x="110" y="180" width="46" height="46" rx="4"/>
<rect class="ov-diag" x="160" y="180" width="46" height="46" rx="4"/>
<rect class="ov-if ov-f" x="210" y="180" width="46" height="46" rx="4"/>
<rect class="ov-if ov-f" x="60" y="230" width="46" height="46" rx="4"/>
<rect class="ov-if ov-f" x="110" y="230" width="46" height="46" rx="4"/>
<rect class="ov-if ov-f" x="160" y="230" width="46" height="46" rx="4"/>
<rect class="ov-diag" x="210" y="230" width="46" height="46" rx="4"/>
<text class="ov-sub" x="155" y="300">diagonal = 1, off-diagonal = interference</text>
<text class="ov-sub" x="155" y="322">fades as m grows</text>
<line class="ov-ax" x1="410" y1="80" x2="410" y2="300"/>
<line class="ov-ax" x1="410" y1="300" x2="680" y2="300"/>
<path class="ov-curve" d="M420,110 Q500,270 670,288"/>
<circle class="ov-d ov-dot" cx="420" cy="110" r="9"/>
<text class="ov-lbl" x="470" y="100">loss</text>
<text class="ov-lbl" x="600" y="150">~ 1 / m</text>
<rect class="ov-s ov-mk" x="404" y="300" width="4" height="16"/>
<text class="ov-sub" x="545" y="336">model width  m  (wider to the right)</text>
</svg>
<figcaption>Pack the same features into more dimensions and their pairwise overlap falls as 1/m, so the interference loss slides down a 1/m curve as width grows.</figcaption>
</figure>

### Why you cannot do better: the Welch bound and tight frames

You might hope a clever arrangement of vectors could beat the random $1/m$. It cannot, and there is a beautiful classical reason why. For $\nu$ unit vectors in $\mathbb{R}^m$, the maximum pairwise overlap obeys the **Welch bound**:

$$\max_{i \ne j} \lvert \hat w_i \cdot \hat w_j \rvert \;\ge\; \sqrt{\frac{\nu - m}{m(\nu - 1)}} \;\equiv\; \kappa,$$

and for $\nu \gg m$ this floor is $\kappa \approx \sqrt{1/m}$. So even the *best possible* packing has a maximum squared overlap of about $1/m$ — the same scaling as random vectors. The Welch bound is met with equality by an **equiangular tight frame** (ETF): a configuration where every pair of vectors has the *same* absolute overlap, the optimal spread of $\nu$ directions in $m$ dimensions. ETFs show up in quantum measurement and in neural-collapse phenomena, and the trained $W_i$ associated with important features turn out to be ETF-like — their squared overlaps have lower variance than random vectors and their mean overlap collapses onto $1/m \approx \kappa^2$. Being ETF-like helps error correction and shaves the loss *prefactor*, but it does not change the $1/m$ *scaling*. You cannot escape the geometry; you can only sit at its optimal point.

A short numerical check makes the $1/m$ claim tangible — random unit vectors, mean squared overlap, swept over dimension:

```python
import numpy as np

def mean_sq_overlap(m, num_vecs=4000, trials=8):
    out = []
    for _ in range(trials):
        V = np.random.randn(num_vecs, m)
        V /= np.linalg.norm(V, axis=1, keepdims=True)   # random unit vectors
        G = V @ V.T                                     # Gram matrix of overlaps
        off = G[~np.eye(num_vecs, dtype=bool)]          # drop the diagonal (= 1)
        out.append(np.mean(off ** 2))                   # mean squared overlap
    return np.mean(out)

for m in [16, 32, 64, 128, 256, 512]:
    s = mean_sq_overlap(m)
    print(f"m={m:4d}   mean sq overlap={s:.5f}   1/m={1/m:.5f}   ratio={s*m:.3f}")
# ratio prints ~1.0 at every m: the mean squared overlap tracks 1/m almost exactly.
```

### When skew breaks the robustness

The $1/m$ law holds when feature frequencies are *even* enough that the vectors stay isotropic. When the frequencies are very skewed — large $\alpha$ — the picture distorts. Important, frequent features grab larger-norm, more-orthogonal vectors (the ETF-like core), while rare features get crowded into the leftover directions with much larger overlaps. The representation is no longer isotropic, and the model exponent $\alpha_m$ rises above 1. In the extreme case where the top $\sim m^2/2$ features form a near-perfect ETF and contribute negligible loss, the worst case has the loss dominated by the rare tail, giving $\alpha_m \approx 2(\alpha - 1)$ — twice the weak-superposition exponent. So the regimes are not isolated islands: strong superposition gives the robust $1/m$ for flat-ish data, but as the data skews, the exponent drifts upward and the loss becomes sensitive to the frequencies again. The robustness is a property of the *flat-frequency* corner of strong superposition, which — luckily for the theory — is roughly where language lives.

To summarize the two regimes in one line each: **weak superposition makes the loss a mirror of the data's frequency tail; strong superposition makes the loss a mirror of the geometry of high-dimensional space.** The first is fragile and data-dependent; the second is robust and nearly universal. The right column of the comparison figure is that second message: all features represented, vectors overlap, loss equals interference $\sim 1/m$, exponent near 1 and robust to the distribution.

## Experiments: do real LLMs do this?

A toy autoencoder with squared error is a long way from a transformer trained with cross-entropy. The authors' bridge is deliberately naive but checkable: treat **tokens as the atomic features**, the vocabulary size as the feature count $n$, and the model width $m$ as the hidden dimension. The object they actually measure is the **language-model head** — the unembedding matrix $W$ whose rows $W_i$ map the hidden state onto vocabulary logits. Each row is a token's representation vector, exactly analogous to $W_i$ in the toy model. They analyze four families — **OPT**, **GPT-2**, **Qwen**, and **Pythia** — spanning roughly 70M to 70B parameters.

![A four-by-four matrix: rows are OPT, GPT-2, Qwen, and Pythia, columns are parameter range, token frequency, overlap versus width, and width exponent; every family shows Zipf token frequency, overlap scaling as one over m, and exponent near 0.9](/imgs/blogs/superposition-yields-robust-neural-scaling-5.webp)

Two measurements line up with the theory. First, the **mean squared overlap** of the normalized head rows $W_i / \lVert W_i \rVert$ scales as $1/m$ across all four families, exactly as the isotropic-vector prediction demands — which is also direct evidence that these models are in superposition (the rows overlap, and they overlap by the geometric amount). Second, fitting the part of the loss that depends on model size as a power law yields a measured exponent

$$\alpha_m = 0.91 \pm 0.04,$$

close to 1, in agreement with the toy model. Token frequencies in all four follow Zipf's law with exponent $\alpha \approx 1$ — flat enough to sit in the robust corner of strong superposition rather than the skewed regime where the exponent would climb. The matrix above is deliberately monotonous: the point is precisely that four independently trained model families, built by different labs with different data and recipes, land on the *same* behavior. That uniformity is the signal.

To make the loss decomposition precise, the authors follow Chinchilla's practice of splitting the loss into a model-size part and a remainder,

$$L = \frac{C_m}{m^{\alpha_m}} + L_{\setminus m},$$

where $C_m / m^{\alpha_m}$ is the universal width-limited term and $L_{\setminus m}$ collects everything that does not depend on width (the irreducible entropy of language plus losses from the transformer layers, which this representation model does not touch). The fit gives $\alpha_m = 0.91 \pm 0.04$.

The Chinchilla cross-check is the part that made me sit up. Chinchilla reports compute-optimal scaling with a loss exponent in model *parameters* of $\alpha_N = 0.35 \pm 0.02$. Parameters and width are related — empirically $N \propto m^{2.52 \pm 0.03}$ across these architectures — so the loss exponent in width should be

$$\alpha_m = 2.52 \times \alpha_N = 2.52 \times 0.35 \approx 0.88 \pm 0.06,$$

which agrees, within error bars, with the independently measured $0.91$. Two completely different measurements — one from the geometry of head rows, one from the most-cited empirical scaling law in the field — point at the same near-unity width exponent. That is the strongest single piece of evidence in the paper.

### When does width scaling stop paying off?

A mechanistic theory should predict its own breakdown, and this one does. If the loss falls as $1/m$ because you are packing $n$ features into $m$ directions, then once $m$ reaches the number of genuinely independent things you need to represent, there is nothing left to pack and the power law must end.

![Timeline of model width: at small m few directions give heavy interference and high loss; through m around 100 and 1000 the loss tracks the one over m power law; at m near the vocabulary size there are no more features to pack; beyond it the power law ends and loss hits the language-entropy floor](/imgs/blogs/superposition-yields-robust-neural-scaling-6.webp)

The naive prediction is that the width-limited loss should deviate from the power law and vanish when $m$ approaches the vocabulary size $\lvert V \rvert$ — once you have a clean orthogonal direction per token, there is no interference left to remove. In practice the extrapolated loss does *not* hit exactly zero, because language has irreducible uncertainty (the $L_{\setminus m}$ floor), and because the true number of independent concepts may be larger than the literal vocabulary — subword tokens compose into a much bigger space of meanings, so the vocabulary size may only be a *lower bound* on the real feature count, and the power law could continue well past it. Either way, the theory says the straight line on your scaling plot is not eternal. It is the visible middle of a curve that must eventually bend.

## Critique

I find the core argument genuinely convincing, and I want to be specific about why, and equally specific about where it is thin.

**What is strong.** The $1/m$ mechanism is the rare scaling explanation that does not smuggle in a power-law assumption — it *derives* robustness from sphere geometry, and the Welch bound shows the result is not an artifact of the random-vector ansatz but a hard floor. The weight-decay knob is a clean intervention: the authors do not merely correlate superposition with scaling, they *vary* superposition and watch the exponent move, which is much closer to a causal test than most scaling-law work manages. And the Chinchilla reconciliation — getting $0.88$ from $0.35 \times 2.52$ to match an independently measured $0.91$ — is a non-trivial, falsifiable prediction that came out right. The toy model is also honest about its own failure modes (linear and exponential feature decay do *not* give power laws under weak superposition), which is the kind of result you only report if you are taking the mechanism seriously rather than curve-fitting.

**What is weak or unfalsifiable.** The bridge from the toy model to LLMs is a *mapping*, not a derivation. Tokens are treated as atomic features, but real features are abstract directions that may or may not align with token-unembedding rows; the paper measures the unembedding head precisely because it is the most representation-like part of a transformer, which is convenient but leaves the attention and MLP layers — where most of the parameters and most of the *computation* live — entirely outside the model. The loss decomposition $L = C_m/m^{\alpha_m} + L_{\setminus m}$ has a free remainder term that can absorb a lot of sins. The squared-error-versus-cross-entropy gap is hand-waved with an appendix argument that softmax error correction behaves like ReLU error correction; plausible, but the softmax is *strong* error correction and could change the prefactor or even the exponent in ways the toy model cannot see. And the model exponent's robustness is itself fragile: the moment feature frequencies skew, $\alpha_m$ drifts toward $2(\alpha-1)$, and the paper's own framing concedes it cannot yet say precisely *when* robustness fails — which means "LLMs scale as $1/m$ because language is flat enough" is partly a statement about language that the theory does not independently establish.

**What ablation is missing.** The experiments sweep width $m$ but largely hold the data structure fixed; I would want to see the model exponent measured against *deliberately re-skewed* token distributions (e.g., domain-specific corpora with much steeper or flatter Zipf slopes) to test the $\alpha_m \approx 2(\alpha-1)$ prediction directly in a real model rather than only in the toy one. The depth axis is also untouched — the paper writes the width-limited loss as one term in $C_m/m^{\alpha_m} = f_m(m) + f_\ell(\ell)$ but does not measure $f_\ell$, so the interaction between superposition (width) and parsing (depth) is conjecture.

**What would change my mind.** If someone trained a family of LLMs with an architecture that *constrains* superposition — say, hidden states and weight rows pinned to the unit sphere, as in nGPT — and the width exponent $\alpha_m$ moved *predictably* with the imposed degree of superposition (lower superposition → exponent tracking the token-frequency tail $\alpha - 1$; higher superposition → exponent locking to 1), that would turn the correlational story into a causal one and I would consider the mechanism close to settled. Conversely, if a model that demonstrably operates in strong superposition showed a width exponent far from 1 with flat token frequencies, the geometric argument would be in serious trouble.

## What I'd build with this

A good mechanistic theory is useful because it suggests interventions, not just explanations. A few I would actually try.

1. **Engineer superposition on purpose.** If strong superposition is what buys the robust $1/m$ law, then architectures that *encourage* it should let a smaller model behave like a larger one. nGPT (unit-norm hidden states and weight rows) and optimizers that train stably without weight decay both nudge representations toward the ETF-like, isotropic configuration the theory likes. The prediction is that these change the scaling *prefactor* $C_m$ — a constant-factor efficiency win — rather than the exponent. That is a clean, cheap thing to measure: fit $\alpha_m$ before and after, and watch whether only $C_m$ moves.
2. **Predict where your domain's scaling breaks.** For a narrow-domain model (a code model, a chess engine, a protein LM) the feature-frequency skew can be very different from natural language. The theory says steep skew can give exponents *above* 1 — faster-than-$1/m$ scaling — which would be a genuinely useful thing to know before committing a training budget. Measure the head-row overlap scaling and the token-frequency exponent on a small model, predict $\alpha_m$, and check it on the next size up.
3. **Use head-row overlap as a cheap superposition probe.** The mean squared overlap of normalized unembedding rows is trivial to compute and tells you whether a model is in weak or strong superposition. That is a one-line diagnostic you could run across a training trajectory to watch a model *enter* superposition, and correlate the transition with the onset of clean power-law scaling.
4. **Connect to emergent abilities and safety.** The authors note that two models with the *same* pre-training loss but different degrees of superposition might differ in downstream emergent abilities and in how easily they are steered or interpreted. If superposition is the knob behind both scaling and interpretability difficulty, then pushing it for efficiency directly trades against the [interpretability program](/blog/machine-learning/ai-interpretability/what-is-superposition) — a tension worth quantifying before it becomes a default.
5. **Stress-test the cross-entropy bridge.** Re-run the toy model with a softmax-cross-entropy reconstruction loss instead of squared error and re-measure $\alpha_m$ as a function of $\phi_{1/2}$. If the exponent survives the swap, the LLM mapping is on much firmer ground; if it shifts, you have found exactly where the toy model stops being a faithful proxy.

The paper's real contribution is to take a phenomenon from interpretability — superposition, which most people met as a story about polysemantic neurons — and show that it is load-bearing for the single most important empirical regularity in modern machine learning. Scaling laws and mechanistic interpretability have mostly been separate conversations. This is a bridge between them, and the bridge is made of nothing more exotic than the observation that random vectors in high dimensions are nearly orthogonal, and get more orthogonal the more room you give them.

## References

1. Yizhou Liu, Ziming Liu, Jeff Gore. *Superposition Yields Robust Neural Scaling.* NeurIPS 2025. [arXiv:2505.10465](https://arxiv.org/abs/2505.10465) · [code](https://github.com/liuyz0/SuperpositionScaling) · [OpenReview](https://openreview.net/forum?id=knPz7gtjPW)
2. Nelson Elhage et al. *Toy Models of Superposition.* Transformer Circuits, 2022. [link](https://transformer-circuits.pub/2022/toy_model/index.html)
3. Jordan Hoffmann et al. *Training Compute-Optimal Large Language Models* (Chinchilla). [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
4. Yasaman Bahri et al. *Explaining Neural Scaling Laws.* PNAS, 2024. [arXiv:2102.06701](https://arxiv.org/abs/2102.06701)
5. Lloyd Welch. *Lower bounds on the maximum cross correlation of signals.* IEEE Transactions on Information Theory, 1974.

Related reading on this blog: [What is superposition](/blog/machine-learning/ai-interpretability/what-is-superposition) · [The linear representation hypothesis](/blog/machine-learning/ai-interpretability/linear-representation-hypothesis) · [Why power laws arise in scaling](/blog/machine-learning/scaling-laws/why-power-laws-arise) · [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling)
