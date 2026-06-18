---
title: "Autoencoders and the Road to Generative Retrieval"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Treat a user's interaction row as the thing to reconstruct, derive AutoRec, CDAE denoising, the Mult-VAE multinomial ELBO and the EASE closed form, then follow the autoencoder idea all the way to RQ-VAE semantic IDs and generative retrieval."
tags:
  [
    "recommendation-systems",
    "recsys",
    "autoencoder",
    "variational-autoencoder",
    "mult-vae",
    "ease",
    "generative-retrieval",
    "semantic-ids",
    "machine-learning",
    "pytorch",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/autoencoders-and-the-road-to-generative-retrieval-1.png"
---

A team I worked with had spent a quarter building a deep candidate generator: a two-tower model with content features, hard negatives, the works. It was good. Then a skeptical staff engineer, the kind who has shipped recommenders since before "embedding" was a verb, asked the team to add one baseline before the launch review: a single matrix, learned by one linear-algebra solve that ran in ninety seconds on a laptop. It had no neural network, no GPU, no negative sampler, no training loop you could even call a loop. It was a shallow item-item autoencoder called EASE. On the offline protocol the team had agreed to (Recall@20 and NDCG@100, strong-generalization split, MovieLens-20M), the ninety-second matrix landed within a hair of the deep model. Nobody was happy about it, and everybody learned something.

That story is the spine of this post. We are going to take a single, deceptively simple idea — *treat a user's whole interaction row over the catalog as the input, and reconstruct it; the reconstructed values on items the user has not seen become the recommendation scores* — and ride it from the earliest neural collaborative filtering autoencoders all the way to the frontier of generative retrieval, where a model literally generates the discrete identifiers of the items it wants to recommend. Figure 1 shows the core mechanism we will keep refining: a user row goes into an encoder, gets squeezed into a latent code, gets decoded back into a full vector of scores, and the top unseen entries are the recommendation.

![Diagram of an autoencoder for collaborative filtering showing a sparse user interaction row encoded to a latent code and decoded back into reconstruction scores over all items with the top unseen entries selected as the recommendation](/imgs/blogs/autoencoders-and-the-road-to-generative-retrieval-1.png)

By the end you will be able to derive AutoRec and the CDAE denoising objective, write down the Mult-VAE multinomial ELBO and explain *why* the multinomial likelihood is the right one for top-N ranking, solve the EASE closed form by hand including the zero-diagonal Lagrangian, implement Mult-VAE in PyTorch and EASE in a few lines of numpy, evaluate both on the standard Recall@20 / NDCG@100 protocol, and understand exactly how the autoencoder-and-quantization idea (RQ-VAE) produces the *semantic IDs* that TIGER-style generative retrieval decodes. This is a post in the series **Recommendation Systems: From Click to Production**, and it sits where the retrieval stage of the funnel meets a humbling truth: in collaborative filtering, simple, well-specified models are astonishingly hard to beat.

## Where this sits in the funnel and the series

The series spine, set up in [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), is the retrieval to ranking to re-ranking funnel fed by the serve-log-train feedback loop, all read off the offline-versus-online gap. Autoencoder collaborative filtering lives in the same stage as [matrix factorization](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse) and the [two-tower model](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval): it generates candidates from collaborative signal. But it occupies a distinctive corner of that stage. Matrix factorization and two towers learn a vector *per user* and a vector *per item*, then score by a dot product. The autoencoder family does something subtly different: it does not learn a stored vector per user at all. The user is represented *only* by their interaction row, and the model is a function that maps that row to scores. That single design choice has consequences we will trace all the way through.

The deeper reason this post matters for the series is the bridge at the end. Retrieval today is overwhelmingly "embed everything into a flat vector space, then run approximate nearest neighbor search." The autoencoder idea, pushed to its logical conclusion, suggests a completely different retrieval paradigm: represent each item not as a point in a continuous space but as a short *sequence of discrete codes*, and have a sequence model *generate* the codes of the items to recommend. That is generative retrieval. The quantizer that turns embeddings into discrete codes is itself a kind of autoencoder (a vector-quantized one), which is why the road from AutoRec to TIGER is a single road, not two. We will preview the full generative-retrieval treatment and forward-link the advanced post.

A quick vocabulary note so nobody is lost. *Implicit feedback* means we observe clicks, watches, and purchases but not explicit ratings, and crucially we observe only positives — the absence of an interaction is ambiguous (did the user dislike the item, or never see it?). We cover this in depth in [implicit feedback models](/blog/machine-learning/recommendation-systems/implicit-feedback-models-als-and-bpr). *Top-N recommendation* means we score all items and return the N highest, evaluated by ranking metrics like Recall@K (did the held-out item land in the top K?) and NDCG@K (did it land high, with a position discount?). *Strong generalization* is the evaluation protocol Mult-VAE popularized: train on one set of users, then at test time feed a *new* user's partial history and predict their held-out items — so the model must generalize to users it never saw during training. Keep these three ideas handy; the whole post turns on them.

## Section 1 — The autoencoder view of collaborative filtering

Start with the data. The interaction matrix $X \in \{0,1\}^{U \times N}$ has one row per user $u$ and one column per item $i$, with $X_{ui}=1$ if user $u$ interacted with item $i$. It is enormous and almost entirely zero: a typical catalog has hundreds of thousands of items and a typical user has touched a few hundred. The user's *interaction row* $x_u \in \{0,1\}^N$ is a sparse vector over the whole catalog.

The autoencoder collaborative filtering idea, in one sentence: build a function $f_\theta : \mathbb{R}^N \to \mathbb{R}^N$ that takes $x_u$ in and produces a dense reconstruction $\hat x_u = f_\theta(x_u)$, train it so $\hat x_u$ matches $x_u$ on the observed entries, and then read off the scores on the *unobserved* entries as the recommendation. The architecture is the classic encoder-decoder: an encoder compresses $x_u$ to a low-dimensional latent $z \in \mathbb{R}^d$ with $d \ll N$, and a decoder expands $z$ back to $\mathbb{R}^N$.

Why would this work at all? Because the bottleneck $d \ll N$ forces the model to find a low-rank structure in the interaction matrix — the same collaborative structure that matrix factorization exploits, just discovered through a different door. If users who watch *The Matrix* also tend to watch *Blade Runner*, the encoder learns a latent dimension that activates for sci-fi watchers, and the decoder learns to light up *Blade Runner* whenever that dimension is on. The reconstruction on the *Blade Runner* column for a user who has only watched *The Matrix* is high precisely because the bottleneck cannot afford to memorize each user; it must compress, and compression of co-occurrence *is* collaborative filtering.

This is the **I-AutoRec versus U-AutoRec** distinction from the original AutoRec paper (Sedhain, Menon, Sanner, Xie, *AutoRec: Autoencoders Meet Collaborative Filtering*, WWW 2015). You can autoencode rows (user-based, U-AutoRec) or columns (item-based, I-AutoRec). The column $x_i \in \mathbb{R}^U$ is the vector of all users who interacted with item $i$; I-AutoRec reconstructs that. Sedhain et al. found I-AutoRec stronger on the rating-prediction benchmarks of the day, because item columns are denser and more stable than user rows. For the implicit top-N world we live in now, the user-row view (U-AutoRec) is the one that generalized into Mult-VAE, so that is the thread we follow, but it is worth knowing the column view exists and that EASE, later, is unapologetically an *item-item* autoencoder.

### The science: AutoRec's objective

AutoRec is just an autoencoder trained with a masked reconstruction loss. For explicit ratings, the loss is squared error over the *observed* entries only — you cannot penalize the model for what it predicts on entries you never observed, because you have no target there. With $\mathcal{O}(u)$ the set of items user $u$ rated and $\hat x_u = f_\theta(x_u)$,

$$
\mathcal{L}_{\text{AutoRec}}(\theta) = \sum_{u} \sum_{i \in \mathcal{O}(u)} \left( x_{ui} - \hat x_{ui} \right)^2 \;+\; \frac{\lambda}{2}\, \lVert \theta \rVert_2^2 .
$$

The masking is the entire trick. Without it, the model would be free to drive every unobserved entry to anything, and the loss would not care; with it, the model is forced to predict observed entries well *while propagating the latent structure through the decoder weights* to the unobserved ones. That structure is what you read off at recommendation time.

For implicit feedback, the squared loss is replaced by a per-item logistic (binary cross-entropy) loss, treating each item as a click/no-click. We will see in Section 3 that this per-item, independent treatment is exactly what the multinomial likelihood in Mult-VAE improves upon, and the reason is fundamental to ranking. But first, the most important practical upgrade to plain AutoRec: denoising.

### How the autoencoder relates to matrix factorization

It is worth pinning down exactly how this differs from the matrix factorization you already know, because the difference explains both the autoencoder's strengths and the place it cannot go. In matrix factorization the score is $\hat x_{ui} = p_u^\top q_i$, where $p_u$ is a *learned, stored* vector for user $u$ and $q_i$ is a learned, stored vector for item $i$. Training learns one parameter vector per user and one per item. Now look at what a U-AutoRec with a single linear encoder $W \in \mathbb{R}^{d \times N}$ and linear decoder $V \in \mathbb{R}^{N \times d}$ computes: $\hat x_u = V (W x_u) = (VW) x_u$. The user's latent is $z_u = W x_u$, which is *not stored* — it is computed on the fly from the user's interaction row. So a linear autoencoder is a matrix factorization where the user factors are *constrained to be a linear function of the user's interactions* rather than free parameters.

That constraint is the whole trade. On one hand, the autoencoder gives up nothing important: $z_u = W x_u$ can express the same user representations matrix factorization learns, as long as users with similar histories should get similar factors (which is the assumption collaborative filtering rests on anyway). On the other hand, the autoencoder gains a superpower matrix factorization lacks: it generalizes to *new users at test time with no retraining*. A brand-new user is just a new interaction row $x_u$; feed it through $W$ and you have their latent immediately, no gradient step required. Matrix factorization has no row for an unseen user and must fold them in or retrain. This is precisely the *strong-generalization* protocol Mult-VAE later formalized, and it is the reason the autoencoder family, not matrix factorization, became the launching pad for inductive, content-aware, and ultimately generative recommenders. The price is symmetric and important: the autoencoder generalizes to new *users* for free but is blind to new *items* (a new column has never appeared in any training row, so the decoder weight for it is untrained) — the exact mirror image of the cold-start asymmetry, and a constraint we will only escape at the end of this post with semantic IDs.

The nonlinear encoder and decoder (the `tanh` layers in AutoRec and Mult-VAE) buy the ability to model interactions the bilinear $p_u^\top q_i$ cannot — a user who likes both action films and documentaries but not the "action documentary" that sits between them, a pattern a dot product struggles with. Whether that extra capacity helps depends entirely on whether such nonlinear structure exists in your data, which, as the benchmark table in Section 5 will show, is often less than you would hope.

## Section 2 — Denoising: corrupting the input so the model learns to recommend

Plain AutoRec has a quiet flaw. If the input is the full row $x_u$ and the target is also $x_u$ on the observed entries, the easiest way to lower the loss is to learn something close to the identity on observed entries — to *copy* what is already there. A model that copies the input is useless for recommendation, because at serving time you want it to predict items the user has *not* yet interacted with. Copying memorizes the present; recommendation requires inferring the absent.

The fix, from Wu, DuBois, Zheng, Ester, *Collaborative Denoising Auto-Encoders for Top-N Recommender Systems* (CDAE, WSDM 2016), is to *corrupt* the input. You take the user's row, randomly drop a fraction $p$ of the observed (1-valued) entries, feed the corrupted row $\tilde x_u$ to the encoder, and ask the decoder to reconstruct the *original, uncorrupted* row $x_u$ — including the entries you just dropped. Now the model cannot copy: the answer to a dropped entry is not in the input. It must infer "given the rest of this user's history, this item was probably also relevant." That is the recommendation task, baked directly into the training objective. Figure 2 contrasts the two regimes.

![Before and after comparison contrasting a plain autoencoder that learns a near-identity copy of the input row against a denoising autoencoder that drops half the seen entries and must reconstruct the held-out items leading to higher recall](/imgs/blogs/autoencoders-and-the-road-to-generative-retrieval-2.png)

### The science: why denoising is the right inductive bias

There is a clean way to see why corruption helps that goes beyond "it stops copying." A denoising autoencoder trained to reconstruct the clean input from a corrupted one is, under mild assumptions, learning the *score function* of the data distribution — it learns to push a corrupted point back toward the data manifold. In the recommendation setting the "manifold" is the set of plausible user-interaction patterns. When you drop items and ask the model to put them back, you are training it to answer: *what items belong with this partial history?* That is exactly the conditional that recommendation needs.

Concretely, masking noise (dropping observed entries) simulates the test-time condition perfectly. At test time, a user's *visible* history is a partial view of their *true* preferences, and you want to predict the rest. Masking noise at train time makes the train-time condition match the test-time condition — it is the recommendation analogue of why dropout helps, except here the dropped units are *inputs that double as targets*, so the model gets free supervision for the prediction task. CDAE also adds a per-user bias term (a learned user embedding concatenated to the latent), which lets it capture per-user offsets the global encoder might miss; this is the one place CDAE keeps a stored per-user parameter, and it is optional.

CDAE's loss with masking corruption $\tilde x_u = \text{mask}_p(x_u)$ and a per-item logistic output is

$$
\mathcal{L}_{\text{CDAE}}(\theta) = -\sum_u \sum_{i} \Big[ x_{ui} \log \sigma(\hat x_{ui}) + (1 - x_{ui}) \log\big(1 - \sigma(\hat x_{ui})\big) \Big] + \frac{\lambda}{2}\lVert \theta \rVert_2^2,
$$

where $\hat x_u = f_\theta(\tilde x_u)$ is computed from the *corrupted* input and $\sigma$ is the logistic sigmoid. The corruption ratio $p$ is the key hyperparameter; $p$ around 0.3 to 0.5 is typical. Too little and the model copies; too much and it has too little signal to reconstruct from.

There are two common noise types, and the choice matters more than it looks. *Masking noise* sets a random subset of the observed (1-valued) entries to 0 — it hides items the user actually interacted with, which is the recommendation condition exactly: at serve time, a user's visible history is a subset of their true preferences. *Additive Gaussian noise* perturbs the input values continuously, which is the right corruption for explicit ratings (where the input is a real-valued rating that could plausibly be noisy) but is a poor match for binary implicit feedback, where the input is a hard 0/1 and the question is "which 1s are missing," not "how noisy is each value." For implicit top-N — the setting we care about — masking noise is the correct choice, and it is what flows directly into Mult-VAE, whose input dropout (zeroing a fraction of the row) is masking noise by another name. The downstream models in this post all use masking-style corruption for exactly this reason.

There is also a subtle interaction between the corruption ratio and catalog density worth knowing before you tune it. On a *dense* dataset (users with hundreds of interactions, like MovieLens), you can afford an aggressive $p = 0.5$ because even after dropping half, the model has plenty of visible signal to reconstruct from. On a *sparse* dataset (users with a handful of interactions), the same $p = 0.5$ can leave one or two visible items, too little to infer anything, and the model degrades — you drop to $p \approx 0.2$ to preserve enough conditioning signal. The right corruption ratio is the one that mimics the *test-time* visible-fraction your users actually have, which ties the hyperparameter directly to your data rather than to a paper's default.

#### Worked example: copying versus inferring

Suppose a user has interacted with five sci-fi films, encoded so the latent has one strong "sci-fi" dimension. We drop two of the five at train time. A copying model sees three sci-fi items, outputs high scores on those three, and is rewarded for the three it kept; it gets the two dropped ones wrong and *learns nothing useful* about inferring them, because nothing in the loss pushed it there when the inputs were uncorrupted. The denoising model sees three sci-fi items, must reconstruct all five, so it learns that the sci-fi latent dimension implies the *whole cluster* of sci-fi films, not just the visible ones. At serving time, when this user has watched three sci-fi films and you want the next, the denoising model lights up the rest of the cluster; the copying model lights up the three it already knows. On a held-out top-20 evaluation, the difference between "copies what you saw" and "completes the cluster you saw" is the whole game. In the original CDAE experiments this kind of denoising lifted Recall@20 by several points over a non-denoising baseline on the datasets tested.

The pattern to carry forward: **the corruption objective converts an identity-reconstruction problem into a missing-data-imputation problem, and recommendation *is* missing-data imputation.** Hold that thought, because Mult-VAE is the probabilistic, generative version of exactly this idea.

## Section 3 — Mult-VAE: the variational autoencoder that got the likelihood right

Now we arrive at the strong baseline that, eight years on, still appears in benchmark tables as a model that "deep" methods struggle to beat: Mult-VAE, from Liang, Krishnan, Hoffman, Jebara, *Variational Autoencoders for Collaborative Filtering* (WWW 2018). It is a variational autoencoder (VAE) — a probabilistic autoencoder — with one crucial, almost obvious-in-hindsight choice: the right likelihood. Figure 3 lays out the pipeline.

![Layered stack diagram of the Mult-VAE pipeline showing the normalized input row encoded to a Gaussian mean and log-variance a reparameterized sample a softmax decoder over all items a multinomial log-likelihood term and an annealed KL term](/imgs/blogs/autoencoders-and-the-road-to-generative-retrieval-3.png)

### What a VAE adds over a plain autoencoder

A plain autoencoder maps $x_u$ to a single point $z$. A VAE maps $x_u$ to a *distribution* over $z$ — specifically a diagonal Gaussian $q_\phi(z \mid x_u) = \mathcal{N}(\mu_\phi(x_u), \operatorname{diag}\sigma_\phi^2(x_u))$ — and then *samples* $z$ from that distribution before decoding. The encoder outputs a mean vector $\mu$ and a log-variance vector $\log\sigma^2$. The decoder is a generative model $p_\theta(x_u \mid z)$ that turns a latent $z$ into a distribution over the user's interactions. Training maximizes a lower bound on the data likelihood, the *evidence lower bound* (ELBO), which has two terms: a reconstruction term and a regularizer that pulls $q_\phi(z\mid x)$ toward a standard-normal prior $p(z) = \mathcal{N}(0, I)$.

Sampling-then-decoding is what makes a VAE *generative* and is why it generalizes better than a deterministic autoencoder: the prior regularization keeps the latent space smooth and prevents the encoder from cramming each user into an isolated pocket. But the headline contribution of Mult-VAE is not the "V" — it is the likelihood.

### The science: deriving the multinomial ELBO

Here is the generative story Mult-VAE tells for a single user with interaction count $N_u = \sum_i x_{ui}$ (how many items they touched). Sample a latent $z_u \sim \mathcal{N}(0, I)$. Pass it through the decoder $f_\theta$ and apply a softmax to get a probability vector over the whole catalog:

$$
\pi(z_u) = \operatorname{softmax}\!\big(f_\theta(z_u)\big), \qquad \pi_i(z_u) = \frac{\exp\big(f_\theta(z_u)_i\big)}{\sum_{j=1}^{N} \exp\big(f_\theta(z_u)_j\big)} .
$$

Then model the user's interaction row as $N_u$ draws from a *multinomial* distribution with probabilities $\pi(z_u)$. The log-likelihood of the observed row under the multinomial (dropping the multinomial coefficient, which does not depend on $\theta$) is

$$
\log p_\theta(x_u \mid z_u) = \sum_{i=1}^{N} x_{ui} \, \log \pi_i(z_u) .
$$

This is the entire reason the model works as well as it does, so let it land. The decoder produces *one* probability simplex over all items — the $\pi_i$ sum to 1. Every item competes for a *shared, fixed budget* of probability mass. If the model wants to give *Blade Runner* more probability, it must take it from somewhere else. That competition is what makes the likelihood a *ranking* objective: getting the top items right is rewarded, and there is no way to cheaply satisfy the loss by predicting "everything is a little bit likely." Contrast this with the per-item logistic likelihood of CDAE, where each item gets an independent Bernoulli and there is no shared budget — a model can call many items independently likely without paying for it.

The ELBO for the VAE is the reconstruction term minus the KL divergence between the approximate posterior and the prior:

$$
\mathcal{L}_{\beta}(x_u; \theta, \phi) = \mathbb{E}_{q_\phi(z \mid x_u)}\!\big[\log p_\theta(x_u \mid z)\big] - \beta \cdot \operatorname{KL}\!\big(q_\phi(z \mid x_u) \,\|\, p(z)\big) .
$$

The expectation is estimated by sampling one $z$ from $q_\phi$ via the *reparameterization trick* — write $z = \mu_\phi(x_u) + \sigma_\phi(x_u) \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0,I)$, so the randomness is in $\epsilon$ (not in a parameter) and gradients flow through $\mu$ and $\sigma$. For diagonal Gaussians the KL has a closed form,

$$
\operatorname{KL}\!\big(\mathcal{N}(\mu, \operatorname{diag}\sigma^2) \,\|\, \mathcal{N}(0,I)\big) = \tfrac{1}{2}\sum_{k=1}^{d}\big(\mu_k^2 + \sigma_k^2 - \log \sigma_k^2 - 1\big),
$$

so you never sample to compute it. That $\beta$ in front of the KL is the second contribution of Mult-VAE.

### The science: why anneal $\beta$ (and why $\beta < 1$ is allowed)

The standard VAE sets $\beta = 1$. Liang et al. observed two things. First, if you start training at $\beta = 1$, the KL term dominates early and the model collapses to the prior — the encoder learns to ignore the input and output $\mu \approx 0$, $\sigma \approx 1$, because that trivially zeroes the KL while the decoder ignores $z$. This is **posterior collapse**, and it is fatal: a model whose latent ignores the user cannot personalize. The fix is **KL annealing**: start $\beta$ at 0 (a plain autoencoder with no regularization, which learns to *use* $z$), then linearly ramp $\beta$ up over the first many thousand gradient steps so the regularizer kicks in only after the latent space is already informative.

Second — and this is the subtle, slightly heretical part — they found the *best* validation NDCG was reached at $\beta$ noticeably *less than 1*, around 0.2 on the datasets they tried. A full $\beta = 1$ over-regularizes for the recommendation objective. So Mult-VAE anneals $\beta$ up to a *cap* $\beta_{\max} < 1$ chosen on validation. This is a $\beta$-VAE with $\beta < 1$, which is unusual; most $\beta$-VAE work uses $\beta > 1$ to encourage disentanglement. Here the goal is not disentanglement, it is ranking, and a lighter prior gives the model more freedom to fit. The recipe: anneal linearly from 0 to $\beta_{\max} \approx 0.2$ over the first ~20% of training, then hold.

There is also a Mult-DAE in the paper, a *denoising* (non-variational) version with the same multinomial likelihood, which sometimes wins and sometimes loses to Mult-VAE depending on data density. Both crush the Gaussian and logistic likelihoods. The lesson the paper drove home and that the field absorbed: **for implicit top-N recommendation, the likelihood matters more than the depth of the network.**

#### Worked example: multinomial versus Gaussian ranking

Consider a tiny catalog of four items and a user who truly likes items A and B and is indifferent to C and D. Two decoders produce these raw scores: a Gaussian-likelihood model trained on squared error and a multinomial-likelihood model trained on the shared softmax. Suppose both produce raw scores roughly proportional to true preference, but the Gaussian model, optimizing per-item squared error toward targets in $\{0,1\}$, is pulled toward predicting values near the *average rate*, compressing differences — it might output $(0.6, 0.55, 0.45, 0.4)$ for A, B, C, D. The multinomial model, where items compete for a fixed budget, sharpens the contrast — its softmax might be $(0.40, 0.35, 0.15, 0.10)$. Now rank by score. Both happen to rank A > B > C > D here, so on this easy case they agree. The difference shows under noise: add a small per-item perturbation. The Gaussian model's near-flat $(0.6, 0.55, 0.45, 0.4)$ has tiny gaps; a perturbation of $\pm 0.1$ easily flips A and C, sending a non-preferred item into the top slots. The multinomial model's $(0.40, 0.35, 0.15, 0.10)$ has a large gap between the {A,B} cluster and {C,D}; the same perturbation cannot flip the top-2. The shared-budget objective *spends its probability mass on separating the items that matter for ranking*, which is exactly robustness in the top-N. Figure 7 makes the contrast visual.

![Before and after comparison of a Gaussian likelihood that scores each item independently with unshared mass against a multinomial likelihood whose softmax forces items to compete for a fixed budget yielding better top-N ranking](/imgs/blogs/autoencoders-and-the-road-to-generative-retrieval-7.png)

Across the standard benchmarks (ML-20M, Netflix, Million Song Dataset), swapping a Gaussian or logistic likelihood for the multinomial lifted Recall@20 by roughly 0.03 to 0.06 absolute in the Mult-VAE paper — a large gain for a one-line change to the loss.

### Implementing Mult-VAE in PyTorch

Here is a compact, runnable Mult-VAE: encoder to mean and log-variance, reparameterized sample, decoder to logits, multinomial log-likelihood plus annealed KL. The input is L2-normalized (the paper's "tf-idf-like" normalization helps), and dropout on the input *is* the denoising. This trains on a sparse user-item matrix (MovieLens-20M style, ~138k users, ~27k movies, treated as binary implicit feedback).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultVAE(nn.Module):
    def __init__(self, n_items, hidden=600, latent=200, dropout=0.5):
        super().__init__()
        # encoder: n_items -> hidden -> 2*latent (mean and log-variance)
        self.enc1 = nn.Linear(n_items, hidden)
        self.enc2 = nn.Linear(hidden, latent * 2)
        # decoder: latent -> hidden -> n_items (logits over the catalog)
        self.dec1 = nn.Linear(latent, hidden)
        self.dec2 = nn.Linear(hidden, n_items)
        self.drop = nn.Dropout(dropout)
        self.latent = latent

    def encode(self, x):
        # L2-normalize the row, then apply input dropout (this is the denoising)
        h = F.normalize(x, p=2, dim=1)
        h = self.drop(h)
        h = torch.tanh(self.enc1(h))
        h = self.enc2(h)
        mu, logvar = h[:, :self.latent], h[:, self.latent:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std          # z = mu + sigma * epsilon
        return mu                          # at eval, use the mean (no sampling)

    def decode(self, z):
        h = torch.tanh(self.dec1(z))
        return self.dec2(h)                # raw logits; softmax happens in the loss

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar


def multvae_loss(logits, x, mu, logvar, beta):
    # multinomial log-likelihood: x are counts (here 0/1), log_softmax over items
    log_softmax = F.log_softmax(logits, dim=1)
    neg_ll = -(log_softmax * x).sum(dim=1).mean()
    # closed-form KL( N(mu, sigma^2) || N(0, I) ), averaged over the batch
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    return neg_ll + beta * kld, neg_ll, kld
```

The training loop with linear KL annealing up to a cap:

```python
import numpy as np

def train_multvae(model, train_loader, n_steps_total, beta_max=0.2,
                  anneal_steps=20000, lr=1e-3, wd=0.0, device="cuda"):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    step = 0
    model.train()
    while step < n_steps_total:
        for x in train_loader:                 # x: dense batch of user rows
            x = x.to(device).float()
            beta = beta_max * min(1.0, step / anneal_steps)   # anneal 0 -> beta_max
            logits, mu, logvar = model(x)
            loss, nll, kld = multvae_loss(logits, x, mu, logvar, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 1000 == 0:
                print(f"step {step}  loss {loss.item():.3f}  "
                      f"nll {nll.item():.3f}  kld {kld.item():.3f}  beta {beta:.3f}")
            if step >= n_steps_total:
                break
```

A few production notes that the snippet hides. The dataset is fed as dense user rows because the multinomial softmax is over all items at once — for a 27k-item catalog that is fine on a GPU, but for a million-item catalog you cannot materialize a dense softmax per user and you must move to sampled-softmax style approximations (the same negative-sampling machinery from [training a two-tower model](/blog/machine-learning/recommendation-systems/training-two-tower-negatives-and-sampled-softmax)), which quietly erases part of Mult-VAE's advantage at extreme scale. At evaluation you feed the *visible* part of a held-out user's history, decode, mask out the items they already saw (you do not recommend what they have), and rank the rest. The strong-generalization protocol means test users were *not* in training, so you rely entirely on the encoder-decoder function generalizing — there is no per-user parameter to look up, which is both Mult-VAE's elegance and its fixed-catalog limitation (a brand-new *item*, a new column, is unseen by the decoder until you retrain).

### The eval harness: Recall@K and NDCG@K computed honestly

The model is only as trustworthy as the harness that scores it, so here is the actual metric computation, the one that produces the numbers in the Section 5 table. The strong-generalization protocol splits each *test user's* held-out interactions into a "fold-in" half (fed to the encoder) and a "held-out" half (the targets you score against). You decode from the fold-in half, mask out the fold-in items so the model cannot trivially recommend what it was just shown, then compute Recall@K and NDCG@K against the held-out half over the *entire* catalog — never against a sampled subset of negatives.

```python
import numpy as np

def recall_at_k(scores, heldout, k=20, fold_in=None):
    # scores: (n_users, n_items) predicted; heldout, fold_in: 0/1 (n_users, n_items)
    s = scores.copy()
    if fold_in is not None:
        s[fold_in.astype(bool)] = -np.inf      # never recommend already-seen items
    topk = np.argpartition(-s, k, axis=1)[:, :k]   # indices of top-k per user
    hit = np.take_along_axis(heldout, topk, axis=1).sum(axis=1)   # hits in top-k
    denom = np.minimum(k, heldout.sum(axis=1))     # cap at the number truly held out
    return (hit / np.maximum(denom, 1)).mean()

def ndcg_at_k(scores, heldout, k=100, fold_in=None):
    s = scores.copy()
    if fold_in is not None:
        s[fold_in.astype(bool)] = -np.inf
    # rank the top-k by score (descending), compute discounted gains
    topk = np.argsort(-s, axis=1)[:, :k]
    rel = np.take_along_axis(heldout, topk, axis=1)            # 1 if a held-out hit
    discounts = 1.0 / np.log2(np.arange(2, k + 2))             # position discount
    dcg = (rel * discounts).sum(axis=1)
    # ideal DCG: as many hits as possible packed into the top positions
    ideal_hits = np.minimum(heldout.sum(axis=1).astype(int), k)
    idcg = np.array([discounts[:n].sum() for n in ideal_hits])
    return (dcg / np.maximum(idcg, 1e-12)).mean()
```

Two subtleties that quietly sink reproductions. The `-np.inf` masking of fold-in items is mandatory: a model that recommends what the user already interacted with will look great if you forget to mask, and you will report inflated numbers that collapse online. And `recall_at_k` divides by $\min(K, \text{number held out})$ rather than by $K$ — if a user has only three held-out items, the best possible Recall@20 is 3/3 = 1.0, not 3/20, so dividing by $K$ would unfairly penalize light users and depress the metric in a way that depends on how you happened to split. This exact harness, applied identically to every model, is what makes a comparison fair; an apples-to-apples harness is more important than any single model in the table.

## Section 4 — EASE: the shallow linear autoencoder with a closed form

If Mult-VAE is the lesson that likelihood beats depth, EASE is the lesson that sometimes you do not need a network at all. EASE — *Embarrassingly Shallow Autoencoders for Sparse Data*, Steck, WWW 2019 — is an item-item autoencoder with a *single hidden "layer" that is the identity* and a learned item-item weight matrix, solvable in closed form. It is the ninety-second matrix from the opening story.

### The model

EASE learns a single $N \times N$ item-item weight matrix $B$ and reconstructs the interaction matrix by $\hat X = X B$. Read that carefully: the score for user $u$ on item $i$ is $\hat X_{ui} = \sum_{j} X_{uj} B_{ji}$ — a weighted sum over the items the user *has* interacted with, weighted by how strongly each interacted item $j$ predicts target item $i$. It is a linear autoencoder: input $X$, output $XB$, no nonlinearity, no bottleneck dimension. The "encoder" and "decoder" are folded into the one matrix $B$.

There is one essential constraint. If $B$ is allowed any value on its diagonal, the model will set $B_{ii} = 1$ and $B_{ji}=0$ for $j \ne i$, achieving perfect reconstruction by *copying the input* — the same trivial-identity trap denoising was invented to dodge, except here it is exact and catastrophic. So EASE forces $\operatorname{diag}(B) = 0$: an item is **not allowed to predict itself**. With self-prediction banned, the only way to reconstruct item $i$ for a user is from the *other* items they interacted with — pure collaborative signal. The diagonal constraint is the denoising idea in linear-algebra form.

### The science: deriving the closed form

EASE's objective is ridge-regularized squared reconstruction error subject to the zero-diagonal constraint:

$$
\min_{B} \;\; \lVert X - X B \rVert_F^2 + \lambda \lVert B \rVert_F^2 \quad \text{subject to} \quad \operatorname{diag}(B) = 0 .
$$

Solve it with a Lagrangian. Introduce a vector of multipliers $\gamma \in \mathbb{R}^N$, one per diagonal entry, and write the constraint as $\operatorname{diag}(B) = 0$:

$$
\mathcal{L}(B, \gamma) = \lVert X - XB \rVert_F^2 + \lambda \lVert B \rVert_F^2 + 2\,\gamma^\top \operatorname{diag}(B) .
$$

Let $G = X^\top X$ be the $N \times N$ item-item Gram matrix (the co-occurrence counts: $G_{ij}$ is how many users interacted with both $i$ and $j$). Expanding the Frobenius norms, $\lVert X - XB\rVert_F^2 = \operatorname{tr}\big((X - XB)^\top (X - XB)\big)$, and taking the derivative with respect to $B$ and setting it to zero:

$$
\frac{\partial \mathcal{L}}{\partial B} = -2G + 2GB + 2\lambda B + 2\,\operatorname{diagMat}(\gamma) = 0 ,
$$

where $\operatorname{diagMat}(\gamma)$ is the diagonal matrix with $\gamma$ on its diagonal. Rearrange:

$$
(G + \lambda I)\, B = G - \operatorname{diagMat}(\gamma) .
$$

Define $P = (G + \lambda I)^{-1}$. Then

$$
B = P\big(G - \operatorname{diagMat}(\gamma)\big) = P G - P\operatorname{diagMat}(\gamma) .
$$

Now use a slick identity. Note $PG = P(G + \lambda I - \lambda I) = P(G+\lambda I) - \lambda P = I - \lambda P$. Substitute:

$$
B = I - \lambda P - P\,\operatorname{diagMat}(\gamma) .
$$

Apply the constraint $\operatorname{diag}(B) = 0$ entry by entry. The $k$-th diagonal entry of $B$ is $1 - \lambda P_{kk} - P_{kk}\gamma_k = 0$, which solves for each multiplier independently:

$$
\gamma_k = \frac{1 - \lambda P_{kk}}{P_{kk}} = \frac{1}{P_{kk}} - \lambda .
$$

Plug back. The diagonal correction $P \operatorname{diagMat}(\gamma)$ has $(i,k)$ entry $P_{ik}\gamma_k$. Combining the $I - \lambda P$ term with the correction and simplifying gives the remarkably clean final solution:

$$
\boxed{\; B_{ij} = \begin{cases} 0 & i = j \\[4pt] -\dfrac{P_{ij}}{P_{jj}} & i \ne j \end{cases} \qquad \text{where} \quad P = (X^\top X + \lambda I)^{-1}. \;}
$$

That is the whole model. Compute the Gram matrix $G = X^\top X$, add $\lambda$ to its diagonal, invert it once to get $P$, then $B$ is $-P$ with each column divided by its own diagonal element and the diagonal zeroed. One matrix inversion. No iteration, no learning rate, no negatives, no epochs.

### Why a single ridge solve beats deep models

The intuition is worth stating plainly. EASE is essentially asking, for each target item $i$, *which other items best linearly predict it across the whole user base, regularized so no single item dominates and with self-prediction forbidden?* The Gram matrix $X^\top X$ captures all pairwise item co-occurrence exactly — there is no sampling, no minibatch noise, no local minimum. The inverse $(G + \lambda I)^{-1}$ does something deep models approximate poorly: it *de-correlates* the co-occurrence signal. Raw co-occurrence says "users who watched $i$ also watched $j$," but that is confounded by popularity (everything co-occurs with the most popular item). The matrix inverse performs a multivariate adjustment — it finds the *conditional* item-item influence after accounting for all the other items — which is exactly the structure that makes recommendations specific rather than popularity-dominated. A deep autoencoder has to *learn* this de-correlation by gradient descent through a bottleneck; EASE *computes* it in closed form. When the signal is genuinely low-rank-plus-sparse, as item co-occurrence tends to be, the closed-form solution is at or near the global optimum, and there is nothing for SGD to do better.

The catch is the $N \times N$ matrix. EASE is $O(N^2)$ memory and $O(N^3)$ for the inversion. For 27k items (ML-20M) that is a 27k-by-27k matrix — about 3 GB in float32 and an invert that runs in tens of seconds. For a million-item catalog, $N^2 = 10^{12}$ entries, which is hundreds of gigabytes and an utterly infeasible cube-time inversion. EASE is unbeatable in the tens-of-thousands-of-items regime and inapplicable in the millions-of-items regime. Steck and others later produced sparse and approximate variants (SLIM was the spiritual predecessor; SLIM solves the same problem with an L1 penalty and a non-negativity constraint but needs per-column optimization rather than a single solve, and EASE's closed form is both simpler and usually stronger). Figure 4 puts EASE next to the others on the properties that matter.

It helps to see *why* EASE improves on its predecessor SLIM beyond "it is closed-form." SLIM (Sparse Linear Methods, Ning and Karypis, 2011) optimizes the *same* item-item reconstruction $\min_B \lVert X - XB\rVert_F^2$ but with an L1 penalty (for sparsity) and constraints $B \ge 0$ and $\operatorname{diag}(B) = 0$. The non-negativity and L1 make the problem non-smooth, so SLIM solves it column by column with coordinate descent — $N$ separate regularized regressions, slow and parallelism-bound, and the non-negativity constraint *forbids negative item-item weights*. That last restriction is the quiet killer: a negative $B_{ji}$ means "users who interacted with $j$ are *less* likely to want $i$," which is real, useful signal (genre exclusion, substitution effects) that SLIM cannot express. EASE drops the L1 and non-negativity, keeps only the L2 ridge and the zero-diagonal constraint, and that smooth convex problem has the single closed-form solution we derived. The lesson is counterintuitive: by asking for *less* (a dense, signed $B$ instead of a sparse, non-negative one), EASE gets a better-conditioned problem with an exact global optimum *and* the ability to represent negative item relationships — and it beats the more "sophisticated" sparse model.

#### Worked example: EASE versus Mult-VAE on cost, not just accuracy

Put concrete numbers on the cost side, because the accuracy is a tie (both around Recall@20 of 0.39 on ML-20M). For ML-20M's roughly 27,000 items, EASE builds a 27k-by-27k Gram matrix (about 5.8 billion entries, 5.8 GB in float64) and inverts it once. On a modern multicore CPU that inversion is on the order of tens of seconds to a couple of minutes, and the resulting $B$ is the entire model — no GPU, no validation epochs beyond a one-dimensional sweep over $\lambda$. Mult-VAE, by contrast, needs a GPU, runs roughly 200 to 400 epochs over the user matrix, requires tuning the latent dimension, dropout rate, $\beta_{\max}$, and the annealing schedule, and a careless $\beta$ schedule silently collapses the posterior and tanks the result. Call it ninety seconds of CPU for EASE versus several GPU-hours plus a hyperparameter search for Mult-VAE, to land at the same offline Recall@20. If your catalog fits the $N^2$ matrix, the engineering math is lopsided — which is the entire reason EASE is the baseline a seasoned engineer demands before signing off on anything deeper. The deep model has to win on something EASE structurally cannot do (content cold start, sequence, context), or it is not worth its operational weight.

![Matrix comparing matrix factorization CDAE Mult-VAE and EASE across likelihood number of parameters whether a closed form exists and Recall@20 showing EASE matching the deep VAE with a single closed-form solve](/imgs/blogs/autoencoders-and-the-road-to-generative-retrieval-4.png)

### Implementing EASE in numpy

The entire model is a handful of lines. Given a sparse user-item matrix `X` (scipy CSR, binary implicit feedback):

```python
import numpy as np
from scipy import sparse

def ease(X, reg_lambda=500.0):
    # X: sparse (n_users x n_items) binary interaction matrix
    G = (X.T @ X).toarray().astype(np.float64)   # item-item Gram, N x N
    diag_indices = np.diag_indices(G.shape[0])
    G[diag_indices] += reg_lambda                # add lambda to the diagonal
    P = np.linalg.inv(G)                          # the single matrix inversion
    B = -P / np.diag(P)                           # divide each column by its own P_jj
    B[diag_indices] = 0.0                         # zero the diagonal (no self-prediction)
    return B                                      # N x N item-item weight matrix

# scoring is one sparse-dense matmul: scores for all users, all items
def ease_scores(X_users, B):
    return X_users @ B          # (n_users x n_items): higher = more recommended
```

To recommend for a user, take their interaction row, multiply by `B`, mask out items they already have, and take the top-N. That is it. The regularization `reg_lambda` is the single hyperparameter; tune it on validation (values in the hundreds are common for ML-20M-scale data because the Gram entries are large counts). There is no train/eval mode, no checkpoint, no learning-rate schedule. The whole "training" is the call to `np.linalg.inv`.

#### Worked example: the EASE solve by hand on three items

Take three items and a tiny user base whose interactions give the item-item Gram matrix (co-occurrence counts, symmetric)

$$
G = X^\top X = \begin{pmatrix} 10 & 6 & 1 \\ 6 & 10 & 1 \\ 1 & 1 & 10 \end{pmatrix}.
$$

Items 1 and 2 co-occur a lot (6 of 10), item 3 is nearly independent of both (co-occurs only once). Use a small $\lambda = 2$, so add 2 to the diagonal:

$$
G + \lambda I = \begin{pmatrix} 12 & 6 & 1 \\ 6 & 12 & 1 \\ 1 & 1 & 12 \end{pmatrix}.
$$

Invert this to get $P = (G+\lambda I)^{-1}$. Carrying the arithmetic, the inverse is approximately

$$
P \approx \begin{pmatrix} 0.1058 & -0.0517 & -0.0045 \\ -0.0517 & 0.1058 & -0.0045 \\ -0.0045 & -0.0045 & 0.0841 \end{pmatrix}.
$$

Now form $B_{ij} = -P_{ij}/P_{jj}$ off-diagonal, diagonal zero. Column 1 has $P_{11} = 0.1058$, so $B_{21} = -P_{21}/P_{11} = 0.0517/0.1058 \approx 0.489$ and $B_{31} = 0.0045/0.1058 \approx 0.043$. Column 2 is symmetric: $B_{12} \approx 0.489$, $B_{32} \approx 0.043$. Column 3 has $P_{33}=0.0841$, so $B_{13} = 0.0045/0.0841 \approx 0.053$ and $B_{23} \approx 0.053$. The resulting matrix:

$$
B \approx \begin{pmatrix} 0 & 0.489 & 0.053 \\ 0.489 & 0 & 0.053 \\ 0.043 & 0.043 & 0 \end{pmatrix}.
$$

Read off the recommendations. A user who has interacted only with item 1 ($x = (1,0,0)$) gets scores $xB = (0, 0.489, 0.053)$: item 2 is strongly recommended (the co-occurring partner), item 3 barely. A user who has only item 3 gets $(0.043, 0.043, 0)$: weak, near-uniform recommendations on 1 and 2, because item 3 carries almost no collaborative signal about them. The closed form correctly turned raw co-occurrence into *de-correlated* item-item influence: items 1 and 2 strongly predict each other, item 3 predicts almost nothing, and self-prediction is zeroed out. No gradient descent was harmed in the making of this recommender.

## Section 5 — Putting the autoencoder family head to head on ML-20M

The point of this whole family is settled empirically, so here is the comparison the field has converged on, evaluated under the Mult-VAE *strong-generalization* protocol on MovieLens-20M: hold out a set of test users entirely; for each, feed 80% of their history as input and predict the held-out 20%; report Recall@20 and NDCG@100. These figures track the Mult-VAE and EASE papers and the widely reproduced RecBole and "Are We Really Making Progress?" (Dacrema, Cremonesi, Jannach, RecSys 2019) benchmarks. Treat them as representative numbers, accurate to roughly the second decimal place, not to the last digit, since exact values shift with preprocessing.

| Model | Likelihood / loss | Closed form? | Recall@20 | NDCG@100 | Notes |
|---|---|---|---|---|---|
| Popularity | none | n/a | ~0.16 | ~0.16 | the floor every model must clear |
| WMF / ALS | weighted squared | no (ALS iters) | ~0.36 | ~0.35 | strong classic implicit baseline |
| BPR-MF | logistic pairwise | no (SGD) | ~0.31 | ~0.30 | optimizes ranking, weaker here |
| CDAE | logistic per item | no (SGD) | ~0.34 | ~0.36 | denoising autoencoder |
| Mult-DAE | multinomial | no (SGD) | ~0.39 | ~0.42 | denoising, multinomial likelihood |
| Mult-VAE | multinomial + KL | no (SGD) | ~0.40 | ~0.43 | the variational version |
| EASE | squared + zero diag | yes (one solve) | ~0.39 | ~0.42 | a single matrix inversion |

Three readings of this table. First, the **likelihood jump**: moving from CDAE's per-item logistic to the multinomial (Mult-DAE/Mult-VAE) buys roughly 0.05 Recall@20 — the single biggest lever in the table, and it is a loss-function change, not an architecture change. Second, **EASE ties the deep VAE** while having no network and a closed-form fit; the gap between EASE and Mult-VAE is within reproduction noise on this dataset, which is the "simple beats complex" result that made the EASE paper famous. Third, and most uncomfortable, **many later "deep" models do not beat these two**. The Dacrema et al. reproducibility study took a stack of recently-published neural recommenders and found that with properly tuned simple baselines (including these autoencoders and well-tuned nearest-neighbor methods), most of the claimed improvements vanished. Figure 8 shows the head-to-head results.

![Matrix of model versus Recall@20 and NDCG@100 results on the MovieLens 20M dataset showing Mult-VAE and EASE leading over matrix factorization BPR and CDAE on both ranking metrics](/imgs/blogs/autoencoders-and-the-road-to-generative-retrieval-8.png)

This is the humility lesson the series keeps returning to and that I want to make operational, not just rhetorical. The reason these simple models are so hard to beat on offline collaborative-filtering metrics is that the signal in a clean, dense, single-domain dataset like ML-20M is *mostly* the item-item co-occurrence structure, and both Mult-VAE and EASE capture that structure close to optimally. A deep model has more *capacity*, but on this data there is not much extra structure for the capacity to find — so it overfits or merely matches. Deep models earn their keep when there is structure the simple model *cannot* represent: rich content features (cold start), sequential order (session intent), context (time, device, query), and cross-domain transfer. None of those are present in plain ML-20M. The right takeaway is not "deep models are useless" but "**deep models must justify themselves against a well-tuned Mult-VAE or EASE on your data, and the justification usually has to come from features the simple model structurally cannot use.**"

### Honest measurement: how not to fool yourself

The reason this table is trustworthy and many published tables are not comes down to evaluation discipline, which we treat fully in [offline vs online](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys). Three rules. Use a temporal or strong-generalization split, never a random hold-out of interactions, or you leak future co-occurrence into training. Compute *full* metrics over the entire catalog, not "sampled metrics" that rank the true item against a handful of random negatives — the KDD 2020 result (Krichene and Rendle, *On Sampled Metrics for Item Recommendation*) showed sampled metrics are *inconsistent*, meaning they can reverse the ranking of two models versus the full metric, so a model that looks better on sampled Recall can be worse in reality. And tune every baseline as hard as you tune your proposed model; an untuned EASE is easy to beat, a tuned one is not. The "deep beats simple" claims that evaporated under scrutiny mostly evaporated because of one of these three.

## Section 6 — The bridge: from reconstructing rows to generating IDs

Everything so far shares one assumption: an item is an *index* into a flat space. AutoRec, CDAE, and Mult-VAE produce a score per item index; EASE learns a weight per item-pair index; the two-tower model produces an embedding per item index and searches them with ANN. The item's identity is an atomic, arbitrary integer, and retrieval means *searching* over all those integers — either by a dense scan, a matrix multiply, or an approximate nearest neighbor index. Figure 5 contrasts that world with the alternative.

![Before and after comparison of embedding plus approximate nearest neighbor retrieval over a flat item table against generative retrieval that autoregressively generates a semantic identifier sequence for each recommended item](/imgs/blogs/autoencoders-and-the-road-to-generative-retrieval-5.png)

Generative retrieval throws out the atomic index. Instead of giving item 48,217 the meaningless integer 48,217, it gives the item a short *sequence of discrete codes* — a *semantic ID* — like `(12, 7, 233, 41)`, where each code position comes from a learned codebook and *similar items share leading codes*. A sci-fi film might be `(12, 7, ...)` and another sci-fi film `(12, 9, ...)`, sharing the first code; a romance might be `(31, ...)`. Then recommendation becomes a *sequence generation* problem: feed the model a user's recent item IDs, and have it *autoregressively decode the semantic-ID sequence of the next item* — one code at a time, like a language model generating tokens — and the decoded sequence *is* the recommended item. No vector index. No nearest-neighbor search. The model's parameters *are* the retrieval system.

This is a genuinely different paradigm, and the differences are worth enumerating before we see how the autoencoder makes it possible:

- **ANN over a flat embedding** stores one vector per item; adding an item means inserting a vector and (eventually) rebuilding the index. Retrieval is sublinear search over the stored vectors, and the index size grows linearly with the catalog.
- **Generative retrieval** stores the catalog *implicitly* in the model weights via the codebooks and the decoder. Adding an item means assigning it a semantic ID (run it through the quantizer) — the decoder can generate it even if it was never in training, *if* its semantic ID overlaps with items the decoder has seen. This is the cold-start advantage: a new item that quantizes to `(12, 7, 233, ...)` inherits the "sci-fi" prefix the decoder already understands.
- **Memory** scales with the number of *codebook entries* (a few thousand codes across a few levels), not the number of items. A billion items can be addressed by, say, four codebooks of 256 entries each ($256^4 \approx 4.3$ billion possible IDs) — a tiny, fixed-size codebook addressing an enormous catalog.

The natural objection — *how do you guarantee the model generates a valid item ID and not a nonexistent code sequence?* — is handled by *constrained decoding*: you maintain a prefix tree (trie) of all valid semantic IDs and, at each generation step, restrict the model's output to codes that continue a valid prefix. The model can only emit sequences that correspond to real items.

### The science: RQ-VAE residual quantization makes the semantic IDs

So where do the semantic IDs come from? From a vector-quantized autoencoder, specifically *residual quantization* (RQ-VAE). This is the precise point where the autoencoder thread of this entire post connects to the generative future. Figure 6 shows the mechanism.

![Graph of RQ-VAE residual quantization showing an item embedding quantized by a first codebook the residual quantized by a second codebook and so on producing an ordered tuple of codebook indices as the semantic identifier](/imgs/blogs/autoencoders-and-the-road-to-generative-retrieval-6.png)

Start with a content embedding of an item — say, a sentence-transformer embedding of the item's title and description, or any dense representation $x \in \mathbb{R}^m$. We want to turn this continuous vector into a short tuple of integers such that the tuple is *faithful* (you can approximately reconstruct $x$ from it) and *hierarchical* (the first integer captures the coarsest structure). A plain vector quantizer would snap $x$ to its nearest entry in a single codebook $C_1 = \{e_1, \dots, e_K\}$ — one integer, but a single codebook of $K$ entries can only represent $K$ distinct items, and you would need a gigantic $K$ to cover a real catalog without crushing distinct items into the same code.

Residual quantization fixes this by quantizing the *error* repeatedly. The algorithm, level by level:

1. Find the nearest codeword in the first codebook: $c_1 = \arg\min_k \lVert x - e_k^{(1)} \rVert$. This is the coarsest approximation of $x$.
2. Compute the *residual* — what the first codeword failed to capture: $r_1 = x - e_{c_1}^{(1)}$.
3. Quantize the residual with the *second* codebook: $c_2 = \arg\min_k \lVert r_1 - e_k^{(2)} \rVert$, then $r_2 = r_1 - e_{c_2}^{(2)}$.
4. Repeat for $L$ levels. The reconstruction is $\hat x = \sum_{\ell=1}^{L} e_{c_\ell}^{(\ell)}$, the sum of the chosen codeword from each level.

The semantic ID is the tuple of chosen indices $(c_1, c_2, \dots, c_L)$. With $L$ codebooks of $K$ entries each, you can address $K^L$ items — exponential in the number of levels — so a handful of small codebooks covers an enormous catalog, and because each level refines the previous one, *the codes are coarse-to-fine*: items sharing $c_1$ are similar at the coarsest level, items sharing $(c_1, c_2)$ are similar at a finer level, and so on. That hierarchy is exactly what makes the IDs *semantic* and what lets the generative decoder share statistical strength across related items.

The "VAE" part is the training. RQ-VAE is an autoencoder: an encoder produces $x$, the residual quantizer produces $\hat x = \sum_\ell e_{c_\ell}^{(\ell)}$, and a decoder reconstructs the original content from $\hat x$. The loss is reconstruction error plus a *commitment loss* (the straight-through estimator from VQ-VAE) that pulls the codebook entries toward the residuals they are chosen to quantize and pulls the encoder outputs toward their chosen codewords:

$$
\mathcal{L}_{\text{RQ-VAE}} = \lVert x - \hat x \rVert_2^2 + \sum_{\ell=1}^{L} \Big( \lVert \operatorname{sg}[r_{\ell-1}] - e_{c_\ell}^{(\ell)} \rVert_2^2 + \beta\, \lVert r_{\ell-1} - \operatorname{sg}[e_{c_\ell}^{(\ell)}] \rVert_2^2 \Big),
$$

where $\operatorname{sg}[\cdot]$ is the stop-gradient (the trick that lets gradients flow past the non-differentiable $\arg\min$), $r_0 = x$, and $\beta$ (around 0.25) weights the commitment term. This is the same VQ-VAE machinery used to tokenize images and audio for generative models, applied here to *item embeddings*. The autoencoder that started this post as "reconstruct the user's row" reappears at the end as "reconstruct the item's embedding through a discrete bottleneck" — and that discrete bottleneck is the alphabet the generative recommender speaks.

### Sketching RQ-VAE quantization

A minimal sketch of the residual-quantization forward pass (the codebook learning loop is omitted for brevity; in practice the codebooks are learned jointly with the encoder/decoder via the loss above):

```python
import torch

def residual_quantize(x, codebooks):
    # x: (batch, dim) item embeddings; codebooks: list of (K, dim) tensors, one per level
    residual = x
    code_ids = []
    reconstruction = torch.zeros_like(x)
    for cb in codebooks:                          # iterate levels coarse -> fine
        # distance from each residual to every codeword in this level's codebook
        dists = torch.cdist(residual, cb)         # (batch, K)
        idx = dists.argmin(dim=1)                 # nearest codeword index per item
        chosen = cb[idx]                          # (batch, dim) the quantized vectors
        code_ids.append(idx)                      # this level's digit of the semantic ID
        reconstruction = reconstruction + chosen  # accumulate the approximation
        residual = residual - chosen              # quantize what's left next level
    semantic_ids = torch.stack(code_ids, dim=1)   # (batch, L) the semantic ID per item
    return semantic_ids, reconstruction
```

Once every item has a semantic ID, you train a sequence-to-sequence model (TIGER uses a T5-style encoder-decoder transformer) on sequences of semantic IDs: the input is the user's history rendered as a flat sequence of code tokens, and the target is the next item's semantic-ID sequence, generated autoregressively with constrained decoding over the trie of valid IDs.

### The science: the autoregressive retrieval objective

It is worth writing down the objective explicitly so the connection to the autoencoder family is unmistakable. The autoencoders earlier in this post maximized a *reconstruction* likelihood of a user's interaction row. The generative recommender maximizes an *autoregressive* likelihood of the next item's semantic-ID sequence given the user's history. If the next item has semantic ID $(c_1, c_2, c_3, c_4)$ and the user history is $h$, the model factorizes the sequence probability by the chain rule:

$$
p_\theta\big((c_1, c_2, c_3, c_4) \mid h\big) = \prod_{\ell=1}^{4} p_\theta\big(c_\ell \mid c_{<\ell}, h\big),
$$

and training minimizes the negative log of this — ordinary next-token cross-entropy, the same loss a language model uses, with the "tokens" being codebook indices. The coarse-to-fine structure of the RQ-VAE codes is exactly what makes this tractable and generalizable: predicting $c_1$ is a coarse "what kind of item" decision over a few hundred codes, $c_2$ refines it given $c_1$, and so on. Because related items share leading codes, the cross-entropy loss on $c_1$ shares statistical strength across an entire item cluster, which is the deep reason a new item with a familiar prefix can be generated at all.

Two practical wrinkles that the deep-dive treats fully but you should know exist. First, *collisions*: two distinct items can quantize to the identical semantic ID, especially in dense regions of embedding space. TIGER's fix is to append an extra disambiguating code (a small counter) so colliding items get IDs like `(12, 7, 233, 41, 0)` and `(12, 7, 233, 41, 1)`. Second, *generation cost*: producing a recommendation is now an autoregressive decode (with beam search to return a top-K list of valid IDs), which is fundamentally more expensive per request than a single ANN lookup — you trade index memory and rebuild cost for decode latency and model FLOPs. Whether that trade is worth it depends on your catalog volatility (high churn favors generative), your latency budget (tight budgets favor ANN), and how much you value the cold-start and controllability wins.

We give all of this its own deep dive — the constrained decoding, the collision handling, beam search over IDs, and the online results — in [generative and conversational recommendation](/blog/machine-learning/recommendation-systems/generative-and-conversational-recommendation), and we cover finetuning large language models to consume and emit these IDs in [finetuning LLMs for recommendation in practice](/blog/machine-learning/recommendation-systems/finetuning-llms-for-recommendation-in-practice). This post's job is to show you *why the road from an autoencoder leads there*: it is the same compress-and-reconstruct idea, with the compression made *discrete* so a sequence model can generate it.

## Section 7 — Case studies and real numbers

Three results anchor this post. I have kept the numbers to what the source papers report and flagged anything I am stating as approximate.

**Mult-VAE (Liang et al., WWW 2018).** On MovieLens-20M under strong generalization, Mult-VAE reported Recall@20 around 0.40 and NDCG@100 around 0.43, beating the contemporaneous neural CF models (including the NCF family critiqued in [neural collaborative filtering and its critique](/blog/machine-learning/recommendation-systems/neural-collaborative-filtering-and-its-critique)) and the WMF/SLIM baselines. On the Netflix dataset it reported similar relative gains, and on the Million Song Dataset (a much sparser, larger-catalog setting) it again led. The paper's central, much-cited finding is that the *multinomial likelihood* drives most of the gain over Gaussian and logistic likelihoods, and that partial regularization ($\beta < 1$ with annealing) matters. Mult-VAE remains a standard baseline in essentially every top-N CF paper since.

**EASE (Steck, WWW 2019).** On ML-20M, Netflix, and the Million Song Dataset, EASE matched or beat Mult-VAE despite being a single closed-form solve with no learned nonlinearity. On the Million Song Dataset in particular, Steck reported EASE *outperforming* Mult-VAE on Recall@20 and NDCG@100. The headline is the contrast between a closed-form item-item matrix that fits in tens of seconds and a deep variational model that needs a GPU and careful annealing, landing in the same place. EASE became the benchmark "simple baseline that humbles you," and it is the model the opening story's staff engineer reached for.

**TIGER generative retrieval (Rajput et al., *Recommender Systems with Generative Retrieval*, NeurIPS 2023).** TIGER built semantic IDs with RQ-VAE over content embeddings and trained a T5-style transformer to generate the next item's semantic ID autoregressively. On Amazon Product Reviews benchmarks (Beauty, Sports, Toys), TIGER reported Recall@5 and NDCG@5 improvements over strong sequential baselines like SASRec and S3-Rec — gains in the range of roughly 10% to 20% relative on several of the reported metrics, with the precise figures varying by dataset and metric. Beyond the headline numbers, TIGER demonstrated two qualitative properties that flat-embedding ANN cannot easily offer: *cold-start generalization* (a new item's semantic ID overlaps with known items, so it can be generated without retraining the index) and *controllable diversity* (you can tune the generation temperature to trade specificity for novelty). TIGER is the proof-of-concept that the autoencoder-to-semantic-ID road actually arrives somewhere useful.

The through-line across all three: an autoencoder objective, made *more correct* (multinomial likelihood), *more closed-form* (EASE), or *more discrete* (RQ-VAE semantic IDs), keeps producing state-of-the-art-competitive recommenders. The architecture is rarely the story; the objective and the representation are.

## Section 8 — When to reach for an autoencoder or EASE (and when not to)

This is the decisive part. Every model is a cost; here is when each one earns it.

**Reach for EASE when** your catalog is in the thousands to low-hundreds-of-thousands of items, you have implicit feedback, and you want a strong, fast, reproducible *baseline* before you build anything deep — which should be *always*, because EASE costs ninety seconds and it is the bar your fancy model must clear. EASE is also a great *production* candidate generator when the catalog is small and stable: it has no training pipeline to maintain, no GPU, and its item-item matrix is trivially interpretable (you can read off "users who liked $j$ also like $i$"). Reach for it as a co-visitation-style retrieval source alongside a two-tower model.

**Do not reach for EASE when** the catalog is in the millions (the $N^2$ matrix and $N^3$ inversion become infeasible — switch to a two-tower model with ANN), when cold start dominates (EASE knows nothing about a new item with no co-occurrence — you need content features), or when sequential/session order is the signal (EASE is order-blind; reach for a sequential model like SASRec).

**Reach for Mult-VAE when** you want a deep CF model that genuinely generalizes to new users at test time without per-user retraining, you have a moderate catalog where the full multinomial softmax is affordable, and you value the probabilistic, generative latent space (useful for uncertainty and for diversity via sampling). It is a stronger candidate generator than plain MF on dense single-domain data.

**Do not reach for Mult-VAE when** the catalog is so large the dense softmax is infeasible (its multinomial advantage erodes under sampled softmax, and at that point a two-tower model is simpler), or when EASE already hits your target — do not pay for a GPU training pipeline and KL annealing to match a ninety-second solve.

**Reach for generative retrieval (semantic IDs) when** you have rich item content to build meaningful semantic IDs from, cold start and new-item coverage are first-order problems, and you are willing to invest in the RQ-VAE quantizer plus a sequence model with constrained decoding. The promise is a retrieval system whose memory does not grow with the catalog, that generalizes to unseen items, and that composes naturally with LLMs. The cost is real: generation latency (autoregressive decoding is slower than an ANN lookup), code-collision handling, and a substantially more complex training and serving stack. It is a frontier choice, not a default — yet.

### Stress-testing the choice

*What if you have only implicit feedback?* That is the assumed setting for all of these; AutoRec's squared loss becomes logistic, and CDAE/Mult-VAE/EASE all target binary implicit data directly. No problem.

*What at 100M items?* EASE and dense Mult-VAE both break (quadratic matrix, dense softmax). The autoencoder *idea* survives only in the generative-retrieval form, where the codebooks are small and fixed regardless of catalog size — that scalability is precisely the argument *for* semantic IDs.

*What when offline NDCG rises but online engagement is flat?* The classic recsys trap. A Mult-VAE that nails offline Recall@20 can still flop online if its strength is recommending *obvious* completions of the user's history (more of the same cluster), which inflates offline metrics measured against historical behavior but does not expand engagement. This is the offline-online gap; the fix is not a better autoencoder but better evaluation (counterfactual/online metrics) and diversity in re-ranking. The autoencoder family is *especially* prone to this because its objective is literally "reconstruct the user's existing behavior," which rewards same-cluster recommendations.

*What when the offline metric is computed with sampled negatives?* Then your whole table may be lying to you (the KDD 2020 inconsistency result). Recompute full metrics over the entire catalog before you trust any ranking of these models.

## Key takeaways

1. **Autoencoder collaborative filtering reconstructs a user's interaction row; the scores on unseen items are the recommendations.** The bottleneck forces the model to find the low-rank co-occurrence structure that *is* collaborative filtering.
2. **Denoising (CDAE) is the essential fix for the copying trap.** Corrupt the input so the model must *infer* held-out items, turning identity reconstruction into the missing-data imputation that recommendation actually requires.
3. **Mult-VAE's win is the multinomial likelihood, not the depth.** A shared softmax budget makes items compete for probability mass, which is the right inductive bias for top-N ranking — worth roughly 0.05 Recall@20 over per-item logistic, the biggest single lever in the comparison.
4. **KL annealing with $\beta < 1$ prevents posterior collapse and over-regularization.** Ramp $\beta$ from 0 to about 0.2 over early training; a full $\beta = 1$ hurts ranking.
5. **EASE is a closed-form item-item autoencoder that ties deep VAEs.** One ridge solve with a zero-diagonal constraint, $B_{ij} = -P_{ij}/P_{jj}$ where $P = (X^\top X + \lambda I)^{-1}$, and self-prediction is banned by the diagonal constraint. Use it as your mandatory baseline.
6. **Simple beats complex on clean offline CF metrics, and that is not an accident.** Mult-VAE and EASE capture the available co-occurrence signal near-optimally; deep models must justify themselves with features (content, order, context) the simple models structurally cannot use.
7. **Evaluate honestly or your table lies.** Temporal/strong-generalization split, full metrics over the whole catalog (sampled metrics are *inconsistent*), and equally-hard tuning of every baseline.
8. **The road to generative retrieval is the same autoencoder idea made discrete.** RQ-VAE residual-quantizes item embeddings into coarse-to-fine semantic IDs, and a sequence model *generates* those IDs — retrieval without a flat vector index, with catalog-independent memory and cold-start generalization.

## Further reading

- Sedhain, Menon, Sanner, Xie, *AutoRec: Autoencoders Meet Collaborative Filtering*, WWW 2015 — the original autoencoder CF model and the I-AutoRec vs U-AutoRec distinction.
- Wu, DuBois, Zheng, Ester, *Collaborative Denoising Auto-Encoders for Top-N Recommender Systems*, WSDM 2016 — the denoising objective for top-N.
- Liang, Krishnan, Hoffman, Jebara, *Variational Autoencoders for Collaborative Filtering*, WWW 2018 — Mult-VAE, the multinomial ELBO, and KL annealing with partial regularization.
- Steck, *Embarrassingly Shallow Autoencoders for Sparse Data*, WWW 2019 — EASE and its closed form.
- Rajput et al., *Recommender Systems with Generative Retrieval*, NeurIPS 2023 — TIGER, RQ-VAE semantic IDs, and autoregressive generative retrieval.
- van den Oord, Vinyals, Kavukcuoglu, *Neural Discrete Representation Learning* (VQ-VAE), NeurIPS 2017 — the vector-quantization machinery RQ-VAE extends.
- Dacrema, Cremonesi, Jannach, *Are We Really Making Much Progress?*, RecSys 2019 — the reproducibility study where tuned simple baselines humble deep models.
- Krichene, Rendle, *On Sampled Metrics for Item Recommendation*, KDD 2020 — why sampled metrics are inconsistent and you must use full metrics.
- Within this series: [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system), [matrix factorization](/blog/machine-learning/recommendation-systems/matrix-factorization-the-workhorse), [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval), [generative and conversational recommendation](/blog/machine-learning/recommendation-systems/generative-and-conversational-recommendation), [finetuning LLMs for recommendation in practice](/blog/machine-learning/recommendation-systems/finetuning-llms-for-recommendation-in-practice), and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
