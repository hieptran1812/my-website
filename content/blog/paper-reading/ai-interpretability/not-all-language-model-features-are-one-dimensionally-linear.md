---
title: "Not All Language Model Features Are One-Dimensionally Linear: Circular Variables in LLMs"
date: "2026-07-31"
description: "A technique-by-technique reading of the ICLR 2025 paper that finds irreducible circular features for days, months, and years in language models."
tags: ["paper-reading", "mechanistic-interpretability", "sparse-autoencoders", "linear-representations", "feature-manifolds", "superposition", "circular-representations", "iclr-2025"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 31
paper:
  title: "Not All Language Model Features Are One-Dimensionally Linear"
  authors: "Joshua Engels et al."
  venue: "ICLR 2025"
  url: "https://arxiv.org/abs/2405.14860"
---

> [!tldr]
> - The paper argues that the useful unit of computation in a language model is not always a single direction in activation space. Some concepts require a low-dimensional manifold, such as a circle.
> - The authors formalize when a multi-dimensional feature is irreducible: no rotation should make its coordinates independent, and no rotation should make one coordinate disappear whenever the other is active.
> - Sparse autoencoders (SAEs) are used as a search engine: similar dictionary vectors are clustered, the corresponding subspace is reconstructed, and PCA exposes candidate manifolds.
> - GPT-2 contains circular representations of days, months, and twentieth-century years. Mistral 7B and Llama 3 8B use related circles when doing natural-language calendar arithmetic.
> - The strongest evidence is causal intervention: changing only the angle of the circular subspace changes the answer. The paper is still preliminary: its irreducibility tests are statistical, mostly two-dimensional, and dependent on SAE and PCA choices.

![Figure 1 from Engels et al. (2025): circular representations of days, months, and years in GPT-2](/imgs/blogs/not-all-language-model-features-are-one-dimensionally-linear-fig1.webp)

The diagram above is the mental model: the model has learned a coordinate system in which Monday through Sunday occupy positions around a circle. The rest of this post asks a deliberately narrow question: is that circle merely a pretty projection, or is it a real computational variable?

## The problem

Mechanistic interpretability wants to reverse-engineer a model into variables and programs. A feature is supposed to be one of those variables: “the model is representing whether this statement is true,” “the model is tracking longitude,” or “the model has copied the subject from an earlier token.” If the variable is found, we can trace which layers write it, which layers read it, and whether changing it changes the output.

The dominant working assumption has been pleasantly simple. A feature is a direction, or at least a scalar activation attached to a direction. A hidden state can then be approximated as a sparse sum:

$$
x(t) \approx \sum_i v_i f_i(t),
$$

where $x(t) \in \mathbb{R}^d$ is the model state, $v_i \in \mathbb{R}^d$ is a direction, and $f_i(t)$ is the scalar strength of feature $i$ on input $t$. This is the one-dimensional version of the linear representation and superposition picture.

That picture has delivered real results. Prior work found directions for truth values, board states, geographic quantities, years, and other concepts. Sparse autoencoders made the picture more practical by learning an overcomplete dictionary of directions and encouraging only a few directions to be active for any one state.

But “linear” is doing two jobs. It can mean that a feature lives in a linear subspace, which may be perfectly true for a plane. It can also mean that the feature itself is one-dimensional, which is much stronger. A circle in a plane is linear in the first sense—the plane is a linear subspace—but not in the second. There is no single axis whose scalar value preserves the circular neighborhood structure.

The paper attacks the second claim. Its central example is calendar arithmetic. Monday plus two days should be Wednesday, and December plus two months should be February. The natural variable is not an ordered line with an endpoint. It is modular: Sunday is next to Monday, December is next to January, and adding a duration means moving around a ring.

![One direction versus one feature manifold](/imgs/blogs/not-all-language-model-features-are-one-dimensionally-linear-1.webp)

The key distinction is between a feature that can be decomposed into simpler scalar features and a feature whose geometry is doing the work. The left side of the diagram is enough when a rotation can split a distribution into independent coordinates. The right side is different: the relationship between the two coordinates—the angle—is the variable.

This connects directly to [the origins of linear representations in language models](/blog/paper-reading/ai-interpretability/on-the-origins-of-linear-representations-in-large-language-models). Engels et al. are not claiming that earlier one-dimensional findings are wrong. They are asking us to stop treating them as an exhaustive ontology of model representations.

## Contributions

The paper makes three main contributions.

1. It generalizes the definition of a language-model feature from a scalar-valued function to a function $f$ mapping inputs into $mathbb{R}^{d_f}$. It then defines reducibility using statistical structure, rather than relying on whether a human thinks a plot looks curved.
2. It proposes a multi-dimensional superposition hypothesis. Hidden states are sums of sparse, low-dimensional, irreducible features whose subspaces are approximately orthogonal.
3. It builds a scalable discovery procedure around sparse autoencoders, finds circular clusters in GPT-2 and Mistral 7B, and then uses activation interventions to show that Mistral 7B and Llama 3 8B use circles for calendar computation.

The argument therefore has two separate burdens. First, the geometry must be more than an arbitrary projection of a tangled cloud. Second, the geometry must matter to the model's computation. The SAE discovery results address the first burden; the interventions address the second.

## What does “irreducible” mean?

### The problem it solves

A two-dimensional plot is not automatically evidence for a two-dimensional feature. Any high-dimensional cloud can look interesting after PCA. A feature could be a mixture of unrelated one-dimensional features, or two independent variables such as latitude and longitude. In both cases a two-dimensional representation is useful for visualization, but it does not mean the model has one indivisible two-dimensional concept.

The authors therefore need a test that answers: can this candidate feature be rotated and translated into simpler pieces?

### Intuition: unpacking a suitcase

Imagine opening a suitcase and finding two kinds of objects. In the first case, all shirts are in one compartment and all shoes are in another. The contents factorize: choosing a shirt tells you little about which shoes are present. In the second case, each compartment is used by a mutually exclusive object: one compartment contains the breed label and another is empty. That is a mixture.

An irreducible feature is the harder case. No rotation of the suitcase produces independent compartments, and no compartment is merely an always-empty coordinate. The shape is not a storage accident; the joint arrangement contains the information.

### Mechanism

The paper defines a feature $f$ as a function from some subset of the input space into $mathbb{R}^{d_f}$. For the practical experiments, $d_f=2$ is the important case. The inputs induce a probability distribution over the points $f(t)$.

The authors allow an orthonormal rotation $R$ and a translation $c$. After transforming the feature, they split its coordinates into two blocks $a$ and $b$:

$$
f \mapsto Rf+c \equiv \begin{pmatrix}a\\b\end{pmatrix}.
$$

The candidate is reducible if at least one of two conditions holds:

- **Separable:** $p(a,b)=p(a)p(b)$. The two pieces are statistically independent.
- **Mixture:** the distribution includes a lower-dimensional component in which one block is zero, so the two pieces do not co-occur.

The paper writes the mixture form as a weighted combination involving a Dirac delta. Intuitively, some fraction of points lives on a lower-dimensional slice such as $b=0$, while another fraction may use both coordinates. If a two-dimensional feature were only a one-hot selection among unrelated scalar features, this kind of structure would appear after a suitable rotation.

### The math

For an exact definition, let $p$ be the density of the transformed feature on the inputs where $f$ is active. Separability is:

$$
p(a,b)=p(a)p(b).
$$

The two coordinates then have zero mutual information. A mixture is represented schematically as:

$$
p(a,b)=w p(a)\delta(b)+(1-w)p(a,b), \qquad 0<w<1.
$$

The notation is slightly overloaded in the paper's compact expression: the second term denotes the remaining joint distribution. The important operational fact is that one component lives near a lower-dimensional axis and the component's coordinates do not co-occur.

The definition is exact but data are finite and noisy. So the authors introduce softened indices. The separability index is the minimum mutual information over all possible rotations and splits:

$$
S(f)=\min I(a;b).
$$

Here $I(a;b)$ is mutual information in bits between the transformed coordinate blocks. A small $S(f)$ means that some rotation makes the parts nearly independent. A large $S(f)$ means that every tested rotation retains dependence.

The $epsilon$-mixture index searches for a hyperplane whose dot products are close to zero for as many active points as possible:

$$
M_\epsilon(f)=\max_{v,c}\Pr_{t\in T}\left[\left|v\cdot f(t)+c\right|<\epsilon\right].
$$

The vector $v\in\mathbb{R}^{d_f}$ chooses a direction, $c\in\mathbb{R}$ translates the band, and $epsilon$ controls its thickness. The paper normalizes the comparison by the second moment of the projection. A large $M_\epsilon$ means that many points can be squeezed into a near-zero band, which is evidence for a mixture-like lower-dimensional component. A small value is evidence against that explanation.

Notice the asymmetry: low $S$ and high $M_\epsilon$ suggest reducibility; high $S$ and low $M_\epsilon$ suggest irreducibility. Neither index alone proves a causal representation.

![Figure 2 from Engels et al. (2025): empirical irreducibility tests for the weekday cluster](/imgs/blogs/not-all-language-model-features-are-one-dimensionally-linear-fig2.webp)

Figure 2 is useful because it shows the tests as optimization problems rather than as visual vibes. The left panel finds the most populated narrow band; the middle panel measures how many points fall near it; the right panel sweeps rotations and records mutual information.

### Worked micro-example

Consider two candidate point clouds in $mathbb{R}^2$.

For the first, sample $a$ and $b$ independently from ${-1,+1}$ and form points $(a,b)$. The four corners are equally likely. The coordinates are independent, so $I(a;b)=0$ in the original rotation. The cloud is separable even though it occupies two dimensions.

For the second, use four points on a circle: $(1,0)$, $(0,1)$, $(-1,0)$, and $(0,-1)$. Any axis rotation still leaves the coordinates coupled by $a^2+b^2=1$. A point near the top has a small horizontal coordinate; a point near the right has a small vertical coordinate. The data do not factorize into two independent axes. Nor can one axis be made empty for a large subset without throwing away the circular variable. This is the small synthetic version of the weekday geometry.

The same logic explains why a Gaussian is not a good example of an irreducible feature. A multidimensional Gaussian can be rotated to diagonalize its covariance, making its coordinates independent when the distribution is Gaussian. The circle cannot be diagonalized in that way because its dependence is nonlinear in the coordinates.

### Why it works and when it fails

The indices are valuable because they turn a geometric hypothesis into a repeatable ranking function. They let the authors score 1,000 candidate SAE clusters and ask whether manually recognizable circles rank unusually high. In GPT-2, the weekday, month, and year clusters rank 9, 28, and 15 by the product of $(1-M_\epsilon)$ and $S$.

The failure mode is equally important. The definitions describe statistical shape, not necessity for computation. A model could contain a circle but read only a single linear projection of it. The authors address this later with interventions. Another failure mode is finite-sample estimation: mutual information depends on a density estimator, and a band search can exploit noise or uneven sampling. The index is evidence, not a theorem that the model has discovered a human-interpretable variable.

## Multi-dimensional superposition

### The problem it solves

The original superposition story says that a $d$-dimensional residual stream can store many more than $d$ scalar features by placing their directions approximately orthogonally and activating only a sparse subset at once. Once features occupy planes, the bookkeeping changes. A two-dimensional circle cannot be described faithfully by one dictionary vector, but it also should not be counted as two unrelated features if the plane is one computational variable.

### Intuition: shared rooms, not loose objects

Think of a building with many small rooms. The scalar hypothesis stores one object per room: a direction is a room, and the object's weight is its activation. The multi-dimensional hypothesis allows a coherent apartment to occupy several adjacent rooms. The apartment has internal geometry, but different apartments remain separated enough that their furniture does not collide.

The distinction matters for interpretation. If we split the apartment into individual rooms and call each room a feature, we may miss the fact that the computation rotates or moves within the apartment.

### Mechanism and math

The paper defines two matrices $A_1\in\mathbb{R}^{d\times d_1}$ and $A_2\in\mathbb{R}^{d\times d_2}$ as $\\delta$-orthogonal when every unit vector in the column space of $A_1$ has inner product at most $\\delta$ in magnitude with every unit vector in the column space of $A_2$.

The updated hypothesis writes the hidden state as:

$$
x_{i,l}(t)=\sum_j V_j f_j(t),
$$

where $x_{i,l}(t)\in\mathbb{R}^d$ is the hidden state at token position $i$ and layer $l$, $f_j(t)\in\mathbb{R}^{d_{f_j}}$ is a sparse low-dimensional feature, and $V_j\in\mathbb{R}^{d\times d_{f_j}}$ maps that feature's internal coordinates into the residual stream. The matrices $V_j$ are pairwise $\\delta$-orthogonal, and each feature is zero outside the inputs on which it is active.

The scalar hypothesis is recovered when every $d_{f_j}=1$ and $V_j$ is just a vector $v_j$. The new hypothesis is therefore a strict extension: it includes lines but also planes, circles, and other low-dimensional manifolds.

The paper also studies how many approximately orthogonal subspaces can fit in a finite-dimensional space. The practical message is not that arbitrary many apartments fit for free. It is that the capacity question should be stated in terms of subspaces, not only vectors.

### Worked example

Suppose the residual stream has $d=6$. A scalar-only decomposition might allocate six directions to six concepts. A multi-dimensional decomposition could allocate one two-dimensional weekday plane, one two-dimensional month plane, and two scalar features. The total occupied dimensions are still six, but the first plane can represent seven discrete angles and interpolate between them.

If the weekday plane and month plane are nearly orthogonal, a change from Monday to Tuesday does not need to disturb the month variable. If they are not orthogonal, an intervention that appears to change only the weekday may leak into month information. This is why the paper's orthogonality assumption is not decorative: it is what lets us interpret a subspace as a separable computational channel.

### Why it works and when it fails

The hypothesis gives mechanistic interpretability a better unit of analysis. It predicts that dictionary elements associated with one manifold should cluster into a subspace, while different manifolds should be separated by weak cosine similarity.

Its weakness is that “low-dimensional” is not yet “ground truth.” A model may use overlapping subspaces, nonlinear readout functions, or a feature that is four-dimensional even when the analysis only checks two-dimensional PCA planes. The authors explicitly call the evidence preliminary and leave higher-dimensional discovery to future work.

## How the SAE search works

### The problem it solves

The model's residual stream may have thousands of dimensions and millions of possible directions. Searching directly for circles is hopeless. We need a tractable proposal mechanism that produces candidate subspaces without hand-labeling every activation.

### Intuition: a dictionary with shared strokes

An SAE is like a dictionary of reusable strokes. Each hidden state is reconstructed by selecting a small number of strokes and adding them together. If the model stores a circle, a sparse dictionary may learn several nearby strokes around the circle's plane. Those strokes are not independent semantic concepts; together they provide the basis needed to reconstruct points on the manifold.

The paper turns this observation into a graph problem. Dictionary vectors become nodes. Cosine similarity becomes an edge weight. Weak edges are cut. Connected components become candidate feature subspaces.

### Mechanism

Let $X_{i,l}$ be the set of hidden states at token position $i$ and layer $l$. A one-layer SAE uses an encoder $E\in\mathbb{R}^{m\times d}$ and decoder $D\in\mathbb{R}^{d\times m}$, where $m$ is overcomplete relative to the model dimension $d$. Its code is $z=\operatorname{ReLU}(Ex)$, and its reconstruction is $Dz$.

The paper's dictionary-learning loss is:

$$
\operatorname{DL}(X_{i,l})=\arg\min_{E,D}\sum_{x\in X_{i,l}}\left\|x-D\operatorname{ReLU}(Ex)\right\|_2^2+\lambda\left\|\operatorname{ReLU}(Ex)\right\|_0.
$$

The first term asks for accurate reconstruction. The second penalizes the number of active dictionary elements. In practice the non-differentiable $L_0$ term is relaxed to an $L_p$ penalty with $0<p\leq 1$. The $m$ columns of $D$ are dictionary elements in the model's $d$-dimensional residual space.

The search procedure is:

1. Compute pairwise cosine similarities between dictionary elements and prune edges below threshold $T$, or use spectral clustering.
2. For each cluster, run the SAE over all hidden states and zero out dictionary elements outside that cluster.
3. Treat the remaining decoder output as the cluster-restricted reconstruction.
4. Project the reconstruction into PCA planes, inspect interpretable shapes, and score them with $S(f)$ and $M_\epsilon(f)$.

![From SAE dictionary to a circular feature](/imgs/blogs/not-all-language-model-features-are-one-dimensionally-linear-2.webp)

The important step is not merely training an SAE. It is grouping its atoms before interpreting them. That is the conceptual bridge from monosemantic dictionary learning to multi-dimensional feature discovery.

### Why clustering can recover a circle

Assume an irreducible two-dimensional feature is active often enough that the SAE must reconstruct it. If the dictionary contained exactly two decoder vectors spanning the plane, both would need to activate together to reconstruct most points. That is expensive under a sparsity penalty. A larger dictionary can instead learn many vectors in the same plane and activate one or a few of them for different parts of the circle.

Those vectors have high pairwise cosine similarity relative to unrelated features. The similarity graph therefore keeps them connected. The cluster's column space approximates the circle's plane even though each individual dictionary element is one-dimensional.

This is a clever inversion of the usual SAE interpretation. Feature splitting, often treated as a nuisance because one concept produces many latents, becomes the signal that reveals a manifold.

### Worked micro-example

Take a toy circle with eight points and an SAE dictionary with eight decoder vectors arranged around the same plane plus unrelated vectors elsewhere. The encoder can reconstruct each point using the nearest local vector and a small combination of neighbors. The circle's vectors have cosine similarities around the plane, while unrelated vectors have similarities near zero. Thresholding at $T=0.3$ leaves one connected component for the circle.

Now reduce the dictionary to only two vectors. To reconstruct eight directions, the SAE must activate both vectors frequently. Sparsity becomes expensive, and the cluster may disappear. Increase the dictionary again and the geometric structure becomes easier to recover. This is why dictionary size and sparsity are load-bearing experimental choices rather than mere implementation details.

### Why it works and when it fails

The method is scalable because cosine similarity and connected components are simpler than searching all nonlinear manifolds. It also gives a clear audit trail: we can show the decoder elements, the threshold, the restricted reconstruction, and the resulting PCA plot.

But clustering is not guaranteed to identify the true feature. A high similarity threshold can split one plane; a low threshold can merge unrelated features. SAE reconstruction error can hide a feature, and ReLU codes can produce artifacts at activation boundaries. The authors use both simple threshold clustering and spectral clustering, but the resulting candidate set still depends on hyperparameters. The method is a search heuristic with theoretical motivation, not an identifiable decomposition theorem.
## Circular representations in language models

### The problem it solves

Finding a pretty circle in GPT-2 is not enough. Perhaps the dictionary clustering procedure simply prefers radial plots. Perhaps the points reflect token frequency or a prompt artifact. The paper therefore looks for a behavior where circular geometry is the natural computational representation: modular addition over a periodic domain.

### Intuition: a clock, not a ruler

A ruler represents distance from an origin. A clock represents position on a repeating cycle. If the model stores a weekday as a scalar on a ruler, it must learn a special case at the Sunday–Monday boundary. If it stores the weekday as an angle, adding a duration is a rotation. The boundary disappears because the representation itself is periodic.

This is exactly the kind of problem where the internal geometry should matter. The question is not simply whether the model can answer “two days from Monday.” The question is whether its hidden state behaves as if it computes on a circle.

### Task design

The authors define two natural-language tasks:

```prompt
Let's do some day of the week math. Two days from Monday is
Let's do some calendar math. Four months from January is
```

For weekdays, $alpha$ is the starting day, $eta$ is the duration, and $gamma$ is the target day. The authors enumerate seven starting days and seven durations, for 49 prompts. For months they enumerate 12 starting months and 12 durations, for 144 prompts. The underlying operation is:

$$
\alpha+\beta\equiv\gamma\pmod m,
$$

with $m=7$ for weekdays and $m=12$ for months.

Mistral 7B gets 31/49 weekday problems and 125/144 month problems correct. Llama 3 8B gets 29/49 and 143/144. GPT-2 gets only 8/49 and 10/144, despite having circular representations discovered in its activations. This comparison matters: a representation can exist without the model using it for the task.

The models also perform poorly on plain prompts such as “5 + 3 (mod 7)”. The useful capability is not generic modular arithmetic in isolation. It is calendar arithmetic embedded in language, where the model's learned semantic representations are available.

### Evidence for a task-specific circle

The authors collect hidden states at the $alpha$ token across prompts and inspect PCA projections at different layers. The top two varying components form circular arrangements in Mistral and Llama. Figure 4 in the paper shows examples: the labels are not arranged in a random semantic cloud but around the periodic order of the calendar.

The subtle point is that PCA is being used as a visualization and probe substrate, not as proof. PCA finds directions of maximum variance. If a circle is embedded in a larger cone, the first component can represent intensity or radius, while the second and third reveal angular structure. The paper therefore checks many components and layers, and later uses a causal intervention to test whether the angular subspace is sufficient.

## Activation patching on a circular subspace

### The problem it solves

Observational alignment between a hidden coordinate and an output can be accidental. A feature may correlate with the answer because another circuit writes both. To establish causal relevance, we need to change the candidate representation while holding the rest of the model as fixed as possible.

Whole-layer patching is too blunt. If we replace an entire hidden layer with the activation from another prompt, the output can change for dozens of reasons. Patching only the top PCA components is narrower, but it may include unrelated high-variance directions and may fail to isolate the circle.

### Intuition: change the clock hand

Imagine a clockwork model whose hidden state contains a clock hand plus unrelated information such as grammatical number and token position. We want to move the hand from Monday to Thursday without replacing the entire clockwork mechanism. The intervention should preserve the average background state and modify only the two-dimensional coordinate that encodes the angle.

### Step 1: learn the circular probe

Let $x^j_{i,l}\in\mathbb{R}^d$ be the hidden state for prompt $j$, token position $i$, and layer $l$. Let $W_{i,l}\in\mathbb{R}^{k\times d}$ contain the top $k$ PCA directions. The authors use $k=5$.

They define a target point on the unit circle:

$$
\operatorname{circle}(\alpha)=
\begin{bmatrix}
\cos(2\pi\alpha/7)\\
\sin(2\pi\alpha/7)
\end{bmatrix}
$$

for weekdays, and replace 7 with 12 for months. A linear probe $P\in\mathbb{R}^{2\times k}$ is fitted by least squares:

$$
P=\arg\min_{P'\in\mathbb{R}^{2\times k}}
\sum_{j,i,l}\left\|P'W_{i,l}x^j_{i,l}-\operatorname{circle}(\alpha_j)\right\|_2^2.
$$

The matrix $W$ compresses the model state from $d$ dimensions to $k$ PCA coordinates. The probe $P$ maps those coordinates to the two target coordinates of the circle. The training objective does not ask the probe to predict a scalar label. It asks it to preserve periodic distance in the target geometry.

### Step 2: intervene and average-ablate

Suppose the original prompt has $alpha_j$ but we want to intervene with $alpha_{j'}$. The authors do not copy the whole clean hidden state from a second run. They only use the clean label $alpha_{j'}$ to construct the target point on the circle.

Let $\bar{x}_{i,l}$ be the average hidden state over prompts and $P^+$ be the pseudoinverse of $P$. The intervention is:

$$
x^{*j}_{i,l}=\bar{x}_{i,l}+W_{i,l}^{\mathsf T}P^+
\left(\operatorname{circle}(\alpha_{j'})-\bar{x}_{i,l}\right).
$$

The first term gives a neutral background. The transpose of $W$ lifts the intervention back into the model's $d$-dimensional space. The pseudoinverse maps the desired two-dimensional circle point into the probe coordinates. Everything outside the selected circular subspace is replaced by the average, which prevents a backup circuit in the original activation from overriding the manipulation.

![Figure 5 from Engels et al. (2025): circular probe training and intervention](/imgs/blogs/not-all-language-model-features-are-one-dimensionally-linear-fig5.webp)

The original Figure 5 makes the two stages explicit. First, the probe learns the circle. Second, the original point is projected, moved to the counterfactual angle, and mapped back. The patch is intentionally label-based: it asks whether the model responds to the abstract variable rather than to incidental activation details from a separate clean prompt.

![Intervening on the angle, not the whole hidden state](/imgs/blogs/not-all-language-model-features-are-one-dimensionally-linear-3.webp)

### Worked micro-example

For a seven-day task, assign Monday $alpha=0$, Tuesday $alpha=1$, and so on. Monday maps to $(1,0)$. Wednesday maps to:

$$
\operatorname{circle}(2)=\left(\cos(4\pi/7),\sin(4\pi/7)\right).
$$

If the dirty activation encodes Monday, the probe should output approximately $(1,0)$. The intervention replaces that output with the Wednesday point. The pseudoinverse then finds the smallest probe-space change that realizes the new coordinate, and $W^{\mathsf T}$ lifts it into the original activation space.

The important counterfactual is not “replace Monday's complete hidden state by Wednesday's complete hidden state.” It is “keep the average context, set the weekday angle to Wednesday, and ask the untouched downstream model what token wins.”

### Baselines and metric

The authors compare circular patching with three baselines: patch the whole layer from a clean run, patch the first five PCA dimensions from a clean run, and average-ablate the entire layer. They measure average logit difference between the original correct token $gamma_j$ and the target token $gamma_{j'}$ across all pairwise interventions. There are $49\times6$ weekday comparisons and $144\times11$ month comparisons.

The circular intervention often approaches the effect of patching the entire layer and outperforms patching the top five PCA components. That is a strong result for a two-dimensional explanation: a narrow subspace reproduces much of the causal effect of a much larger object.

### Why it works and when it fails

The intervention works when the downstream circuit reads the angle in a reasonably stable coordinate system. It fails if the representation is distributed across many unrelated dimensions, if the probe is overfit to a prompt distribution, or if backup circuits reconstruct the answer from information left untouched. The average ablation reduces the last problem but also changes the model state substantially, so the intervention is not a perfectly natural counterfactual.

The authors find that interventions are strongest in early layers and drop after layers 15–17, where other experiments suggest $alpha$ has been copied to the final token. This is a useful reminder that a feature can be causally important at one location and irrelevant at another even when its label remains decodable.

## Off-distribution intervention: is the angle really the variable?

### The problem it solves

The standard intervention only visits known circle points. If Monday through Sunday are seven discrete points, a flexible classifier could memorize seven labels without representing a continuous angle. To test the geometry itself, the authors intervene at positions the model did not receive during normal task prompts.

### Intuition: testing the empty clock face

A real clock does not stop being a clock between two printed hour labels. If we place the hand halfway between Monday and Tuesday, the model may not produce a standard answer, but the output should change smoothly according to where the hand points. If only the seven training positions work, the representation may be a lookup table painted onto a circle.

### Mechanism and math

The authors modify the intervention by replacing the target unit-circle point with a grid of radius $r$ and angle $\theta$:

$$
x^{*j}_{i,l}=\bar{x}_{i,l}+W_{i,l}^{\mathsf T}P^+
\left(
\begin{bmatrix}r\cos\theta\\r\sin\theta\end{bmatrix}
-\bar{x}_{i,l}
\right).
$$

They sweep $r\in\{0,0.1,\ldots,2\}$ and 100 angular positions over nearly two revolutions. For each point they record the highest target-token logit after the forward pass.

This makes the hidden geometry behaviorally visible. If the radius carries the weekday identity, changing radius should change the answer in a systematic way. If the angle carries it, regions of angle should correspond to weekday outputs while radius mainly controls confidence or activation strength.

### Worked example

Take the “two days from Monday” prompt. At a radius near the learned circle and an angle corresponding to Wednesday, the Wednesday logit should be high. Sweep the angle while holding the radius fixed: the preferred answer should rotate through the weekday labels. Now hold the angle fixed and sweep radius from zero outward. The model may be uncertain near the center and confident near the learned circumference, but the identity should remain associated with the angle.

The paper's Figure 7 shows colored polar maps for several task durations. The regions align with the calendar labels, supporting the claim that the angle is being treated as a multi-dimensional representation rather than merely a plotting coordinate.

### Why it works and when it fails

This test is stronger than decoding accuracy because it asks the model to respond to counterfactual states that were not generated by the original prompts. It can reveal whether the downstream circuit has learned a meaningful function over the plane.

Its limitation is interpretive. An off-distribution output pattern can arise from smooth extrapolation in the network without the model “understanding” a circle in the human sense. The result is evidence that the local computational map is angular. It is not proof that every layer represents a globally continuous calendar manifold.

## The SAE plane is more robust than a per-layer probe

The first intervention experiments train a circular probe on the PCA coordinates of each layer. That is useful for testing behavior, but it leaves open a circularity: the probe itself may be re-fitting a convenient plane at every layer. The authors therefore test whether the precise plane found by SAE clustering transfers across nearby layers.

They use Mistral 7B's layer-8 weekday cluster. The plane is defined by PCA dimensions 2 and 3 of the cluster-restricted reconstruction. A circular probe trained on that SAE plane is evaluated not only at layer 8 but also at neighboring layers. The baselines are a normal per-layer circular probe and a layer-8 probe evaluated on other layers.

At layer 8, the SAE probe produces an average logit difference of -2.01 compared with -2.58 for the per-layer probe. The small loss is plausible because the SAE plane is constrained by the unsupervised discovery procedure. More strikingly, at layer 6 the layer-8 circular probe gives 0.029 while the layer-8 SAE probe gives -2.32. The SAE-derived plane transfers much better.

The interpretation is cautious: this may mean the SAE found a more “true” or at least more robust feature than a probe trained only for task prediction. It could also mean the SAE plane captures a stable coordinate system that happens to align with the downstream circuit. Either way, robustness across layers is exactly the sort of test a purely visual feature discovery method needs.

This result connects to [dense SAE latents as features rather than bugs](/blog/paper-reading/ai-interpretability/dense-sae-latents-are-features-not-bugs). A latent dictionary should not be judged only by whether each individual atom has a clean label. Groups of atoms can be the meaningful object, and their stability under intervention may be a better criterion than individual interpretability.

## Continuity: can the model represent intermediate time?

### The problem it solves

The discovered weekday and month representations are mostly discrete. The points sit at the vertices of a heptagon or dodecagon, with no ordinary prompt between Monday and Tuesday. A discrete lookup table can produce the same plot. The authors therefore test prompts containing intermediate temporal modifiers.

### Intuition: “very late Monday”

If Monday is a point on a time circle, “very early Monday” should be closer to Sunday and “very late Monday” should be closer to Tuesday. This does not require the model to output a fractional day. It only asks whether the internal coordinate respects an ordered neighborhood around the discrete labels.

### Mechanism

The authors create synthetic text such as:

```prompt
very early on Monday
very late on Monday
very early on Tuesday
very late on Tuesday
```

They project the resulting layer-30 Mistral activations into the top two PCA components learned from the ordinary weekday representations. This is an important evaluation choice: they do not refit a new PCA plane on the synthetic data. They ask whether the old circle provides a coordinate system for the new phrases.

The resulting points land toward neighboring weekdays. The same pattern appears for “morning” and “evening” modifiers in the appendix. Figure 9 is therefore a continuity probe, not a claim that the model has a literal physical clock.

### Worked example

Let Monday sit at angle $0$ and Tuesday at angle $2\pi/7$. An idealized “very late Monday” might lie near angle $0.8(2\pi/7)$, while “very early Monday” might lie near angle $-0.2(2\pi/7)$, which is geometrically close to Sunday. The exact angles are not specified by the paper; the qualitative neighborhood is the result.

### Why it works and when it fails

The experiment increases confidence that the circle is not only a seven-class code. Intermediate modifiers are mapped into the existing plane in a semantically sensible direction. It also illustrates why manifold-level analysis matters: a scalar probe could classify “Monday” correctly while missing the relationship between early, normal, and late versions.

The failure mode is prompt dependence. “Very late on Monday” may contain lexical cues that a language model learned from text rather than a general temporal computation. The experiment is best read as evidence of representational continuity, not as a causal demonstration. A stronger follow-up would intervene at intermediate angles and test systematic composition with durations.

## Experiments and results

The headline behavioral results are compact:

| Model | Weekdays | Months | What it tells us |
|---|---:|---:|---|
| Llama 3 8B | 29 / 49 | 143 / 144 | Strong calendar arithmetic, especially months |
| Mistral 7B | 31 / 49 | 125 / 144 | Strong enough to support intervention analysis |
| GPT-2 | 8 / 49 | 10 / 144 | Has discovered circles but does not solve these tasks reliably |

The GPT-2 result is scientifically useful. It prevents an overly simple story in which “a circle exists” automatically implies “the model uses the circle for modular arithmetic.” Representation and computation are separate claims.

For the causal experiments, Figure 6 compares five interventions: no-op, patch the circle, patch the full layer, patch the top PCA dimensions, and average-ablate. The circular patch often tracks the whole-layer patch in early layers, while patching only the top PCA coordinates is weaker. The shape of the curves also reveals where the variable is copied and where downstream layers stop depending directly on the original token position.

![Figure 6 from Engels et al. (2025): intervention effects across layers](/imgs/blogs/not-all-language-model-features-are-one-dimensionally-linear-fig6.webp)

What is load-bearing in this setup? The task prompts use familiar calendar language, the models have enough capability to solve most month problems, and the analysis chooses token positions and layers where $alpha$ is linearly decodable. The results may not transfer to arbitrary abstract modular arithmetic, to models with different tokenization, or to features whose manifold dimension is higher than two. The authors train Mistral SAEs only on layers 8, 16, and 24, while GPT-2 uses available SAEs across layers, so the discovery coverage is not uniform.

The rankings also depend on examining approximately 1,000 clusters and selecting interpretable examples. The paper reports the ranking statistic, but the positive story naturally emphasizes the circles that humans can name. A complete automated evaluation would need a way to assess all candidate manifolds, including those with no obvious English label.

## Critique

### What is genuinely strong

The paper makes a precise conceptual correction. It does not merely say “representations can be nonlinear.” It distinguishes nonlinear-in-one-dimension from linear-in-a-low-dimensional-subspace, then defines statistical tests that operationalize the distinction.

The combination of unsupervised discovery and causal testing is also well chosen. SAE clustering finds candidates without using weekday labels. Probes and interventions then test whether the candidate subspace affects the output. This is more persuasive than either a hand-designed probe or a plot alone.

The use of GPT-2, Mistral 7B, and Llama 3 8B creates a useful separation between discovery and use. GPT-2 supplies striking examples of circles but weak calendar-task accuracy; the larger models supply the causal behavioral evidence. The cross-model pattern is not perfect, which makes the claim more credible.

Finally, the SAE-plane transfer result is a subtle contribution. A feature that survives a layer shift is more interesting than one that only decodes after a fresh probe is trained at every layer. It points toward a practical criterion for feature quality: stability under interventions, transformations, and distribution shifts.

### What is weak or under-specified

The definition of irreducibility is statistical rather than intervention-based. A distribution can be irreducible under the proposed tests and still be irrelevant to the model's computation. Conversely, a computational variable may look reducible because the model uses only part of it on the observed prompt distribution.

The $epsilon$-mixture index is a heuristic softened definition. Its value depends on the band width, normalization, optimizer, and sampling distribution. The separability index depends on how mutual information is estimated and how rotations are searched. The paper is transparent about these limitations, but the numerical scores should not be treated as universal units of feature dimensionality.

There is also a dimensionality bottleneck. The main empirical test examines two-dimensional PCA projections. A four-dimensional irreducible feature could project into many misleading two-dimensional slices: some would look separable, some would look circular, and the average could hide the structure. The paper's claim is strongest for the discovered two-dimensional circles, not for all multi-dimensional representations.

The SAE argument has its own assumptions. It expects a sufficiently large dictionary, enough activation coverage, and a clustering threshold that keeps the relevant dictionary atoms together. A failed search could mean the feature is absent, or that the SAE has split, absorbed, or under-reconstructed it. The paper does not fully disentangle those possibilities.

The intervention's average ablation is another tradeoff. It isolates the circle by suppressing backup information, but it also moves the hidden state off the natural activation distribution. Whole-layer patching is a strong upper-bound-like baseline, but neither intervention is a clean natural counterfactual. More causal tests at multiple token positions and with matched-norm interventions would strengthen the conclusion.

The missing ablation I most want is a representation-preserving control. Take a randomly selected two-dimensional SAE plane with similar variance, fit the same circular probe, and run the complete intervention protocol. Another useful control would scramble the weekday labels around the circle while preserving class balance and probe loss. If the original ordering still produces the strongest modular behavior, the geometry is doing more than supporting classification.

### What would change my mind

What would change my mind is a larger, preregistered search over models and tasks in which the SAE-discovered subspace is selected without human inspection, then predicts causal behavior on held-out prompts and held-out compositions better than matched random subspaces and per-layer probes. If those controls failed, I would downgrade the circles from fundamental computational variables to elegant but partly epiphenomenal geometries.

## What I would build with this

These are extrapolations, not claims made by Engels et al.

1. **A manifold-first SAE benchmark.** Build synthetic datasets containing lines, circles, tori, simplices, and branching manifolds. Evaluate whether a discovery method recovers the correct dimension, topology, and causal readout under dictionary size and sparsity sweeps.
2. **Automated causal cluster ranking.** For every SAE cluster, measure reconstruction quality, irreducibility, layer stability, and intervention effect. A candidate should rank highly only when it is geometrically coherent and causally useful.
3. **Higher-dimensional calendar and number circuits.** Search for representations of time zones, dates, angles, and modular arithmetic with dimensions greater than two. The natural generalization may be a product of circles, such as a torus for two periodic variables.
4. **Subspace-aware circuit graphs.** Extend sparse-feature circuit tooling so a node can be a subspace with an internal coordinate system, not only a scalar SAE latent. Edges could then be linear maps, rotations, or nonlinear readouts between manifolds.
5. **Intervention-based model editing.** If an interpretable concept is genuinely a stable manifold, editing its geometry may change a narrow behavior while leaving unrelated features intact. The risk is that downstream backup circuits will route around the edit, so this should be tested against redundancy rather than assumed.

The broader engineering lesson is simple: our representation vocabulary should match the computation. If a model performs addition on a periodic domain, a circle is a more economical variable than seven unrelated one-hot directions. If we force every concept into a scalar, we may create an interpretation that predicts labels but cannot explain the algorithm.

## References

1. [Not All Language Model Features Are One-Dimensionally Linear](https://arxiv.org/abs/2405.14860), Engels et al., ICLR 2025.
2. [MultiDimensionalFeatures code repository](https://github.com/JoshEngels/MultiDimensionalFeatures).
3. [On the Origins of Linear Representations in Large Language Models](/blog/paper-reading/ai-interpretability/on-the-origins-of-linear-representations-in-large-language-models).
4. [Dense SAE Latents Are Features, Not Bugs](/blog/paper-reading/ai-interpretability/dense-sae-latents-are-features-not-bugs).
5. [Sparse Feature Circuits](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models).
