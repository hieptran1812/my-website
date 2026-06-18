---
title: "The Ranking Model: CTR Prediction Foundations"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How the ranking stage scores a few hundred candidates with rich features, why click-through-rate prediction is the canonical ranking task, and how to build and evaluate a CTR pipeline from FTRL logistic regression to a small DNN with measured AUC, logloss, and calibration on Criteo-style data."
tags:
  [
    "recommendation-systems",
    "recsys",
    "ctr-prediction",
    "ranking",
    "logistic-regression",
    "ftrl",
    "calibration",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/the-ranking-model-ctr-prediction-foundations-1.png"
---

The first time I owned a ranker, the on-call page that woke me at 3 a.m. was not "the model is down." It was "revenue is down four percent and nothing is broken." The retrieval layer was healthy, the feature store was serving, latency was nominal, and the model was returning scores for every request. The scores were just slightly, systematically wrong: a recent retrain had nudged the predicted click probabilities up by a hair, the ads auction had read those inflated probabilities as higher expected value, bid the house budget on impressions that did not convert, and burned through the day's spend by noon. No exception was thrown. The classifier's accuracy was fine. AUC had even gone up by a thousandth. What had silently broken was the one property nobody had a dashboard for: whether the number the model called "probability of click" was actually a probability you could trust.

That night is the whole reason this post exists. The ranking stage of a recommender does not just sort items; in any system that bids, blends, or budgets on its output, it makes a numeric promise — "this candidate will be clicked with probability $p$" — and a great many downstream decisions take that promise literally. Getting the order right is necessary. Getting the *number* right is what separates a ranker that quietly works from one that quietly costs you four percent. The canonical way the field has learned to do both at once is **click-through-rate (CTR) prediction**, and it is the task this post builds from first principles: the loss, the math behind the metrics, the workhorse models from logistic regression to a small deep net, and the calibration correction that would have saved my sleep.

Here is where ranking sits in the funnel you have been following through this series. Retrieval scans millions of items cheaply and hands a few hundred candidates to ranking; ranking scores those few hundred precisely and orders them; re-ranking applies business and diversity rules on top. Because the ranker only ever looks at a few hundred items per request, it can afford a far more expensive model than retrieval can — and crucially, it can afford the *early, cross feature interaction* that the two-tower retrieval model structurally cannot do. That is the trade this whole post is about.

![A layered stack diagram showing candidates plus features flowing into a ranker and then into calibrated click probabilities and a final order](/imgs/blogs/the-ranking-model-ctr-prediction-foundations-1.png)

By the end you will be able to derive the logloss objective and its gradient by hand, explain precisely why AUC measures ordering while logloss measures calibration and why a production ranker needs you to watch both, build a CTR pipeline that trains an FTRL logistic regression and a small PyTorch DNN on a Criteo-style dataset with feature hashing, read a results table comparing logistic regression to factorization machines to a DNN at the AUC where a thousandth matters, and apply the negative-downsampling calibration correction with real numbers. If you want the map of the whole funnel first, it is in [the recommendation funnel: retrieval, ranking, re-ranking](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking); the top-level series intro is [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system); and the synthesis of everything is the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

## 1. What the ranker actually does

The ranker's job statement fits in one sentence: take the candidates that retrieval handed you, attach every feature you can compute for the (user, item, context) triple, predict the probability of the engagement you care about, and sort the candidates by that prediction. The engagement target depends on the surface. For an ads system or a feed it is usually **click-through rate** — the probability the user clicks the item if it is shown. For an e-commerce ranker it might be **conversion rate** (CVR), the probability of a purchase given a click. For a video feed it might be **expected watch time**, a regression rather than a classification. The machinery is nearly identical across these; CTR is the canonical case because the label is a clean binary (clicked or not), the data is abundant, and the math is the cleanest, so it is the one we build from the ground up. Multi-objective rankers that predict several of these at once are their own topic, covered in the multi-task post; here we nail the single-objective foundation everything else is built on.

The thing that makes ranking a fundamentally different engineering problem from retrieval is the *number of candidates*. Retrieval is a search over the entire catalog — millions or billions of items — so it must be sublinear, which forces a cheap scoring function: a dot product between a user vector and an item vector, indexed by approximate nearest neighbor search. The user tower and item tower are computed independently and only meet at the very end, in that single dot product. This is **late interaction**, and it is a real limitation: the model can never learn that "this user clicks sports videos *on weekday mornings on mobile*" as a single conjunction, because the user features and the context features and the item features never get multiplied together inside the network — they only get summed into two vectors that meet once.

Ranking does not have that constraint. By the time the candidate set is down to a few hundred items, you can afford to run a model that is a thousand times more expensive per item, because a few hundred forward passes is nothing. And you spend that budget on the thing retrieval could not do: **early, cross interaction**. The ranker can take the user's features, the item's features, and the context features and explicitly cross them — multiply them together, feed them jointly into a deep network, learn arbitrary conjunctions. The candidate-features-go-in, calibrated-probability-comes-out picture is the same shape as the stack at the top of this post, but the internals are where the value is.

![A before and after comparison contrasting cheap retrieval with late dot-product interaction against expensive ranking with early cross interaction over a few candidates](/imgs/blogs/the-ranking-model-ctr-prediction-foundations-2.png)

It is worth being concrete about the asymmetry, because it justifies the entire architectural split of the funnel. Suppose retrieval must score one million items in under ten milliseconds. That is a budget of ten nanoseconds per item — enough for a dot product over a few hundred dimensions and nothing more. Ranking scores three hundred items in under thirty milliseconds. That is a budget of one hundred microseconds per item — ten thousand times more compute per item. With ten thousand times the budget you do not run a slightly bigger dot product; you run a different *class* of model, one with feature crosses and several dense layers, because that is where the accuracy lives. The funnel exists precisely so that the expensive model only ever sees the few candidates worth spending the budget on.

A quick taxonomy of the features the ranker feeds on, because the model is only as good as these:

- **User features.** Demographics where available, but mostly behavioral: historical CTR, categories the user engages with, recency of last visit, device, app version. These are the same for every candidate in the request, so they are computed once.
- **Item features.** The item's own historical CTR, its category, price, age since creation, creator, content embeddings. These are per-candidate.
- **Context features.** Time of day, day of week, page, slot position, query (if any), network type. Same for every candidate in the request.
- **Cross features.** The conjunctions — user-category times item-category, user-device times item-format, hour times item-freshness. These are the whole point of the ranker over the retriever, and the next several model families exist to learn them.

There is a trap hiding in that feature list that pages more rankers than any modeling bug, and it deserves a name now because it recurs through this whole series: **train-serve feature skew.** A feature like "item historical CTR" is computed one way in the offline training pipeline (a batch job over yesterday's logs) and a different way at serving (a real-time counter that lags, or rounds, or uses a different window). The model trains on a clean offline version of the feature and serves on a subtly different online version, so it learned signal that does not exist at inference time. The symptom is brutal and confusing: offline AUC looks great, the online lift evaporates, and nothing throws an error. The defense is to compute features *once*, in a shared feature store that both training and serving read from, or to log the exact features used at serving and train on those logged values rather than recomputing — a practice called feature logging. We come back to this hard in the data-and-features and offline-online posts; flag it here because a ranker is only as trustworthy as the features it is fed, and the most expensive failures are not in the model at all.

The ranker's output is one scalar per candidate, the predicted probability. Downstream, that scalar is either used directly (sort by it) or fed into a blend — for ads, the auction multiplies CTR by bid to get expected revenue; for a feed, a weighted combination of predicted click, dwell, and share. Either way, the *number* matters, not just the order, which is why we now have to be very precise about what a "good" probability even means.

## 2. CTR prediction as the canonical ranking task

Strip ranking down to its simplest honest form and you get a binary classification problem. Each training example is a (user, item, context) triple with a label $y \in \{0, 1\}$: one if the impression was clicked, zero if it was shown and not clicked. The model produces a score $z$ (the **logit**), and a sigmoid squashes it to a probability:

$$\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad z = f_\theta(x)$$

where $x$ is the feature vector for the example and $f_\theta$ is the model — a linear function for logistic regression, a factorization machine for FM, a deep network for a DNN. Everything downstream of $z$ is shared across all three model families; what differs is how $f_\theta$ turns features into the logit.

The objective that trains this is **logloss**, also called binary cross-entropy. For a single example with true label $y$ and predicted probability $\hat{p}$:

$$\ell(y, \hat{p}) = -\big[\, y \log \hat{p} + (1 - y) \log (1 - \hat{p}) \,\big]$$

Read it intuitively. If $y = 1$ (clicked), the loss is $-\log \hat{p}$, which is zero when $\hat{p} = 1$ and shoots to infinity as $\hat{p} \to 0$ — the model is punished, hard, for being confidently wrong on a click. If $y = 0$ (no click), the loss is $-\log(1 - \hat{p})$, symmetric the other way. Averaged over the dataset, this is the quantity you minimize. It is not an arbitrary choice; logloss is the negative log-likelihood of the data under a Bernoulli model, which means minimizing it is maximum-likelihood estimation, which means the minimizer is the *true conditional probability* $P(y = 1 \mid x)$ in the limit of infinite data and a flexible enough model. That property — that the loss is minimized exactly at the true probability — is what makes logloss a **proper scoring rule**, and it is the reason we can read the model's output as a calibrated probability rather than just a ranking score. Hold that thought; it is the linchpin of the calibration section.

![A branch and merge dataflow graph showing sparse and dense features going through embeddings and crosses into a logit and a sigmoid producing a click probability](/imgs/blogs/the-ranking-model-ctr-prediction-foundations-3.png)

The historical arc of CTR models is a steady relaxation of the form of $f_\theta$, each step adding the ability to capture interactions the previous step could not:

1. **Logistic regression.** $z = w^\top x + b$. Linear in the features. To get any feature interaction at all, you must *manually* construct cross features — hand-built conjunctions like "user-country=US AND item-category=shoes." This was the original CTR workhorse at Google and Facebook, and it scaled to billions of features with online learning. We build it in section 4.
2. **Factorization machines (FM).** $z = w^\top x + \sum_{i<j} \langle v_i, v_j \rangle x_i x_j$. FM learns a latent vector $v_i$ for every feature and models pairwise interactions as inner products of those vectors, so it captures *all* second-order crosses automatically and generalizes to crosses it never saw at training time. This is the bridge from hand-crafted crosses to learned ones; it gets its own post in [factorization machines and field-aware FM](/blog/machine-learning/recommendation-systems/factorization-machines-and-field-aware-fm).
3. **Deep models.** Wide & Deep, DeepFM, DCN, DIN. A deep network over embedded features captures high-order, nonlinear interactions. This is where the field has lived since around 2016, and it is where the AUC ceiling is highest — at the cost of calibration headaches we will spend real time on.

It is worth pausing on why CTR and not CVR or watch-time is the foundation, since the funnel uses all three. Click is the *densest* signal: nearly every impression produces a click-or-not label within seconds, so you get billions of clean labels a day. Conversion (CVR) is sparser and *delayed* — a purchase might happen hours after the click, so the label is both rarer and arrives late, which creates a whole "delayed feedback" sub-problem (you do not know yet whether a recent click will convert, so your "negative" might just be a not-yet-converted positive). Watch-time is a regression with a heavy-tailed target, so it uses a different loss entirely (often a Poisson or a log-transformed regression). All three share the same backbone — features in, embeddings and crosses, a model, a calibrated output — but CTR is where the math is cleanest and the labels are cheapest, which is exactly why it is the teaching case and the historical starting point for every production ranker. Once you can build a calibrated CTR model you can swap the head and the loss to get CVR or watch-time with the same skeleton.

The reason CTR is the *canonical* ranking task, and not just one of many, is that this binary-click formulation is the cleanest place to study the two things every ranker must get right: **ordering** (does the relevant item score higher than the irrelevant one?) and **calibration** (is the predicted probability numerically trustworthy?). Those two goals are measured by two different metrics that do not move together, which is the subject of the next section and one of the most important things to internalize about ranking.

## 3. Two metrics, two goals: AUC and logloss

Here is the single most useful fact about evaluating a ranker, the one I wish someone had tattooed on my monitor before that 3 a.m. page: **AUC measures ordering and logloss measures calibration, and a model can win one while losing the other.** You generally need to watch both.

**AUC** — the area under the ROC curve — has a beautifully simple probabilistic meaning that is worth deriving because it tells you exactly what the metric can and cannot see. Take a random positive example (a click) and a random negative example (a no-click). AUC is the probability that the model scores the positive higher than the negative:

$$\text{AUC} = P\big(\,\hat{p}(x^+) > \hat{p}(x^-)\,\big)$$

where $x^+$ is drawn uniformly from the positives and $x^-$ from the negatives. An AUC of 0.5 means the model orders pairs no better than a coin flip; an AUC of 1.0 means every positive outscores every negative. The estimator over a finite dataset is just the fraction of positive-negative pairs that are correctly ordered:

$$\widehat{\text{AUC}} = \frac{1}{|P|\,|N|} \sum_{i \in P} \sum_{j \in N} \mathbb{1}\big[\hat{p}_i > \hat{p}_j\big]$$

Stare at that formula and the key property leaps out: AUC depends *only on the ordering* of the scores, not on their values. If you take every predicted probability and pass it through any strictly increasing function — square it, take its log, add a constant, multiply by ten — the ordering is unchanged and the AUC is *exactly* identical. AUC is invariant to any monotone transformation of the scores. That is its strength (it isolates ranking quality) and its blind spot (it cannot tell a calibrated model from a wildly miscalibrated one, as long as their orderings agree). The model from my 3 a.m. page had a *higher* AUC and *worse* calibration; AUC literally could not see the problem.

**Logloss** is the opposite. Because it is a proper scoring rule, logloss is minimized at the true probability, so it penalizes you for the *value* of $\hat{p}$, not just its rank. A model that orders everything perfectly but predicts 0.9 for items whose true click rate is 0.2 has great AUC and terrible logloss. Logloss is the metric that notices when your probabilities are inflated, deflated, or otherwise untrustworthy — exactly the failure mode that burns ad budgets.

![A matrix laying AUC and logloss side by side over what each measures whether it is rank-only and whether it cares about calibration with rows for the logistic regression factorization machine and deep model](/imgs/blogs/the-ranking-model-ctr-prediction-foundations-4.png)

The matrix above lays the two metrics side by side and adds the model rows we will measure later. The practical consequence is a workflow rule: **report both, and react to divergence.** If AUC rises but logloss rises too, your model orders better but its probabilities drifted — fine for a pure-sort surface, dangerous for anything that bids or blends. If logloss falls but AUC is flat, you improved calibration without improving ranking — good for the auction, neutral for the sort. The two metrics decompose ranker quality into "are items in the right order" and "are the numbers trustworthy," and you almost never get to ignore either.

There is a small but important subtlety in *how* you compute AUC at scale. The double sum over all positive-negative pairs is $O(|P|\,|N|)$, which is quadratic and infeasible on a billion-impression test set. The standard trick is to sort all examples by score once ($O(n \log n)$) and compute AUC from the ranks of the positives, using the Mann-Whitney U identity:

$$\text{AUC} = \frac{\sum_{i \in P} r_i - \frac{|P|(|P|+1)}{2}}{|P|\,|N|}$$

where $r_i$ is the rank of positive $i$ in the score-sorted list (rank 1 = lowest score). This is what `sklearn.metrics.roc_auc_score` does internally, and it is why AUC over a hundred million rows is a few seconds rather than a few centuries.

One more honest caveat. Plain AUC pools all users together, which can be misleading for a personalized ranker: a model can get high pooled AUC just by learning that some users click more than others, without ever ranking *within* a user's candidate list correctly. The within-user metric is **GAUC** (group AUC), the impression-weighted average of per-user AUC. Always check GAUC for a personalized surface; it is the one that correlates with the online list-quality you actually serve. The same offline-online gap shows up everywhere in this series and is the whole subject of [offline vs online: the two worlds of recsys](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys).

## 4. The science: deriving logloss and its gradient

Let me make the loss provable rather than asserted, because the gradient we derive here is the exact update every CTR model runs, and understanding it demystifies both logistic regression and the deep net.

Start from the model. The logit is a function of the parameters; for logistic regression specifically, $z = w^\top x$. The predicted probability is $\hat{p} = \sigma(z)$. We want the gradient of the per-example logloss with respect to the parameters $w$. Do it in two steps with the chain rule: first $\partial \ell / \partial z$, then $\partial z / \partial w$.

The sigmoid has a famously clean derivative. Writing $\sigma = \sigma(z)$:

$$\frac{d\sigma}{dz} = \sigma (1 - \sigma)$$

Now the loss as a function of the logit. Substitute $\hat{p} = \sigma(z)$ into logloss and simplify:

$$\ell = -\big[\, y \log \sigma(z) + (1 - y)\log(1 - \sigma(z)) \,\big]$$

Differentiate with respect to $z$. Using $\frac{d}{dz}\log\sigma(z) = 1 - \sigma(z)$ and $\frac{d}{dz}\log(1-\sigma(z)) = -\sigma(z)$:

$$\frac{\partial \ell}{\partial z} = -\big[\, y(1 - \sigma) - (1 - y)\sigma \,\big] = -\big[\, y - y\sigma - \sigma + y\sigma \,\big] = \sigma - y = \hat{p} - y$$

That is the whole punchline of the derivation, and it is gorgeous: **the gradient of logloss with respect to the logit is just the prediction error, $\hat{p} - y$.** No sigmoid clutter survives. Now finish with $\partial z / \partial w = x$ (since $z = w^\top x$):

$$\frac{\partial \ell}{\partial w} = (\hat{p} - y)\, x$$

So the gradient for one example is the residual times the feature vector. If the model predicts 0.7 on a clicked example ($y=1$), the residual is $-0.3$ and the update pushes the weights of the active features *up* to raise the next prediction. If it predicts 0.7 on a non-click ($y=0$), the residual is $+0.7$ and the update pushes those weights *down*. The gradient sees only the error and the active features, which is why sparse CTR training is so cheap: an example with twenty active features touches exactly twenty coordinates of $w$, not the whole billion-dimensional vector.

This same $\hat{p} - y$ residual is the gradient that flows back from the output of the deep network too — the only difference is that for a DNN $\partial z / \partial w$ is the backpropagated Jacobian of the whole network rather than the raw $x$. The output-layer story is identical. That is why understanding logistic regression genuinely is understanding the output head of every CTR DNN.

Two more facts the derivation hands us for free. First, logloss is **convex** in $w$ for logistic regression (the Hessian $\sum_i \hat{p}_i(1-\hat{p}_i)\,x_i x_i^\top$ is positive semidefinite), so gradient descent finds the global optimum — there is no local-minimum drama in the linear model, which is part of why it was so dependable at scale. Second, at the optimum the average gradient is zero, $\frac{1}{n}\sum_i (\hat{p}_i - y_i)\,x_i = 0$. Take the coordinate of the bias feature (always 1) and that says $\frac{1}{n}\sum_i \hat{p}_i = \frac{1}{n}\sum_i y_i$ — **the average predicted probability equals the empirical click rate.** A properly trained logistic regression is calibrated *in aggregate* by construction. That single equation is why logistic regression has a reputation for trustworthy probabilities, and the reason deep models lose that property is precisely that they over-parameterize past the point where this constraint binds.

#### Worked example: computing AUC for a tiny ranked set by hand

Take five impressions. The model assigns scores, and we know which were actually clicked:

| Impression | Score $\hat{p}$ | Label $y$ |
| --- | --- | --- |
| A | 0.90 | 1 (click) |
| B | 0.80 | 0 |
| C | 0.60 | 1 (click) |
| D | 0.50 | 0 |
| E | 0.30 | 0 |

Positives are $\{A, C\}$, negatives are $\{B, D, E\}$, so there are $|P| \times |N| = 2 \times 3 = 6$ positive-negative pairs. Count how many are correctly ordered (positive scores higher than the negative):

- $(A, B)$: $0.90 > 0.80$ — correct.
- $(A, D)$: $0.90 > 0.50$ — correct.
- $(A, E)$: $0.90 > 0.30$ — correct.
- $(C, B)$: $0.60 > 0.80$ — **wrong** (negative B outscores positive C).
- $(C, D)$: $0.60 > 0.50$ — correct.
- $(C, E)$: $0.60 > 0.30$ — correct.

Five of six pairs are correctly ordered, so $\widehat{\text{AUC}} = 5/6 \approx 0.833$. Cross-check with the rank formula. Sort ascending by score: E(1), D(2), C(3), B(4), A(5). The positives are C and A with ranks 3 and 5, so $\sum_{i \in P} r_i = 8$. Then

$$\text{AUC} = \frac{8 - \frac{2 \cdot 3}{2}}{2 \cdot 3} = \frac{8 - 3}{6} = \frac{5}{6} \approx 0.833$$

Both routes agree. Notice that the single inversion — negative B scoring above positive C — is the entire AUC penalty, and notice that AUC never looked at the *values* 0.90, 0.80, 0.60: if you replaced them with 9, 8, 6 the answer would be identical. That is the monotone-invariance property made concrete.

## 5. Logistic regression at scale: FTRL and online learning

Logistic regression sounds quaint until you see the scale it ran at. Google's ad CTR system and Facebook's, circa the early 2010s, trained logistic regression over feature spaces with *billions* of dimensions — every user ID, every ad ID, every page, and crucially every hand-built *cross* of them. The model could not be batch-retrained from scratch on a cluster every few minutes against a firehose of impressions; it had to learn **online**, updating one example at a time as the data streamed in. The algorithm that made this work, and that is still the reference design for large sparse online learning, is **FTRL-Proximal** (Follow-The-Regularized-Leader), from McMahan et al. at Google in 2013.

The problem FTRL solves is specific. Plain online gradient descent on sparse features works but produces dense, non-sparse weight vectors — every feature that ever appeared keeps a nonzero weight, and with billions of features that is a memory catastrophe. You want **sparsity**: most rarely-seen features should end up with weight exactly zero so you never have to store them. $L_1$ regularization induces sparsity in batch training, but naive online subgradient descent with $L_1$ does *not* reliably zero out weights — it keeps nudging them across zero. FTRL is the modification that does. Its per-coordinate update keeps two accumulators, $z_i$ and $n_i$, for each feature $i$, and at each step sets the weight by a closed form that snaps to exactly zero whenever the accumulated gradient is smaller in magnitude than the $L_1$ strength $\lambda_1$:

$$w_i = \begin{cases} 0 & \text{if } |z_i| \le \lambda_1 \\[4pt] -\dfrac{z_i - \operatorname{sgn}(z_i)\,\lambda_1}{\left(\beta + \sqrt{n_i}\right)/\alpha + \lambda_2} & \text{otherwise} \end{cases}$$

The other clever piece is the **per-coordinate learning rate**. Each feature gets its own adaptive step size $\alpha / (\beta + \sqrt{n_i})$, where $n_i$ is the running sum of squared gradients for that feature. Features that appear often (small per-step information, accumulate fast) get small learning rates and stable estimates; rare features (each appearance is informative) get large learning rates so a handful of observations move their weight meaningfully. This is the same adaptive-rate principle AdaGrad later formalized, and it matters enormously for CTR because feature frequencies span many orders of magnitude — a common device-type feature fires on every impression, a specific ad-creative-ID feature fires a few thousand times in its life.

Here is the FTRL-Proximal per-example update in runnable form, the exact loop a streaming CTR trainer runs:

```python
import math

class FTRLProximal:
    """Per-coordinate FTRL-Proximal for sparse logistic regression.

    Trains one example at a time over a hashed sparse feature space.
    Weights are materialized lazily, so only seen features cost memory.
    """
    def __init__(self, alpha=0.1, beta=1.0, l1=1.0, l2=1.0, n_features=2**24):
        self.alpha, self.beta, self.l1, self.l2 = alpha, beta, l1, l2
        self.n = [0.0] * n_features   # sum of squared gradients per feature
        self.z = [0.0] * n_features   # FTRL accumulator per feature
        self.w = {}                   # materialized weights for the current example

    def _predict(self, x):
        # x is a list of active feature indices (value 1, one-hot sparse)
        wTx = 0.0
        self.w = {}
        for i in x:
            sign = -1.0 if self.z[i] < 0 else 1.0
            if sign * self.z[i] <= self.l1:
                w_i = 0.0                       # L1 snaps this feature to zero
            else:
                lr = (self.beta + math.sqrt(self.n[i])) / self.alpha + self.l2
                w_i = -(self.z[i] - sign * self.l1) / lr
            self.w[i] = w_i
            wTx += w_i
        return 1.0 / (1.0 + math.exp(-max(min(wTx, 35.0), -35.0)))  # sigmoid, clipped

    def update(self, x, y):
        p = self._predict(x)
        g = p - y                                # the (p - y) residual gradient
        for i in x:
            g_i = g                              # gradient on feature i (x_i = 1)
            sigma = (math.sqrt(self.n[i] + g_i * g_i) - math.sqrt(self.n[i])) / self.alpha
            self.z[i] += g_i - sigma * self.w[i]
            self.n[i] += g_i * g_i
        return p
```

Notice three things that connect straight back to the math. The gradient `g = p - y` is exactly the residual we derived in section 4 — FTRL is just a clever way of *accumulating* that residual per coordinate with adaptive rates and $L_1$ sparsity. The weights are materialized lazily into a dict per example, so a billion-dimensional model only ever stores the features it has actually seen. And the prediction is a plain sigmoid of a sparse dot product — the model is still ordinary logistic regression; FTRL is the *optimizer*, not the model.

The per-coordinate learning rate is the piece that makes this work on real feature-frequency distributions, and it pays to build the intuition. In a CTR feature space, frequencies span maybe nine orders of magnitude: a `device=mobile` feature fires on essentially every impression, while a specific `ad_creative_id=8472913` fires a few thousand times in its entire short life. A single global learning rate cannot serve both — set it high enough to learn from the rare creative in a handful of observations and it makes the common device feature oscillate wildly; set it low enough to stabilize the device feature and the rare creative never moves off zero. FTRL's $\alpha / (\beta + \sqrt{n_i})$ rate solves this per feature: $n_i$ is the running sum of squared gradients, so a feature seen a million times has a large $n_i$ and a tiny, stable step, while a feature seen ten times has a small $n_i$ and a large, aggressive step. Each feature learns at the pace its data frequency warrants. This is the same adaptive-rate idea AdaGrad later formalized, and it is non-negotiable when frequencies are this skewed.

The memory story is the other reason FTRL shipped at the scale it did. The McMahan paper details tricks the toy code above omits but every production system uses: **probabilistic feature inclusion** (a brand-new feature is only admitted to the model with some small probability per appearance, so one-off features that will never matter never consume a slot), **coarse weight encoding** (storing each weight in 16 or even fewer bits rather than a 64-bit float, since CTR weights do not need full precision and the table is billions of entries), and **per-feature counters** that let you garbage-collect features that have not fired in weeks. The combination is what let a single model with billions of potential features fit in a serving host's memory. None of this changes the math — it is all engineering around the fact that the feature space is enormous and mostly sparse — but it is the difference between a paper and a system that monetizes the web.

Where does the feature engineering go? Into `x`. The art of pre-deep CTR was building the right crosses by hand. A linear model cannot learn that "users in the US on mobile click shoe ads at 9 a.m." unless you *give* it that conjunction as a single feature. So practitioners built feature templates: `country x device`, `hour x category`, `user_id x advertiser_id`, and hashed each conjunction into the feature space. The combinatorial blowup of these crosses is exactly what made the feature space billions-wide, and the pain of maintaining them by hand is exactly what motivated factorization machines and the wide-and-deep family — which learn the crosses instead of you hand-coding them. That motivation is the bridge to [wide and deep and the memorization-generalization tradeoff](/blog/machine-learning/recommendation-systems/wide-and-deep-and-the-memorization-generalization-tradeoff).

## 6. Feature hashing and a runnable CTR pipeline

Before the models, the features. CTR features are overwhelmingly **categorical and sparse**: a user ID, an ad ID, a publisher, a device. The natural representation is one-hot — a vector with a 1 in the slot for the active category and 0 everywhere else. Concatenate all the fields and a single training example is a vector that is almost entirely zeros with a handful of active slots, one per categorical field. The figure below shows three fields, each contributing exactly one hot bit.

![A grid showing a sparse one-hot feature row where each categorical field contributes one active slot and the rest are zero](/imgs/blogs/the-ranking-model-ctr-prediction-foundations-7.png)

The problem is that the universe of category values is open-ended and huge — you cannot pre-enumerate every ad ID that will ever exist, and a dictionary mapping every value to an integer index would itself be enormous and require coordination between training and serving. The standard trick is **the hashing trick**: run each (field, value) pair through a hash function and take it modulo a fixed table size $m$ (say $2^{24}$ slots). No dictionary, no vocabulary, constant memory, and new category values just hash into the existing table. The cost is **collisions** — two distinct values occasionally landing in the same slot — but with a large enough $m$ collisions are rare and their effect on AUC is negligible, a trade the field made permanently decades ago.

```python
import hashlib

def hash_feature(field: str, value: str, n_buckets: int = 2**24) -> int:
    """Map a (field, value) categorical pair to a fixed-size bucket index."""
    key = f"{field}={value}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest(), 16) % n_buckets

def featurize(row: dict, n_buckets: int = 2**24) -> list[int]:
    """Turn a raw impression row into a list of active feature indices."""
    feats = []
    # categorical fields -> one hashed slot each
    for field in ["user_id", "ad_id", "publisher", "device", "hour", "country"]:
        feats.append(hash_feature(field, str(row[field]), n_buckets))
    # hand-built crosses -> the conjunctions a linear model cannot learn alone
    feats.append(hash_feature("device_x_hour",
                              f"{row['device']}|{row['hour']}", n_buckets))
    feats.append(hash_feature("country_x_ad",
                              f"{row['country']}|{row['ad_id']}", n_buckets))
    return feats
```

That `featurize` output — a list of active indices — is exactly what the FTRL learner above consumes. To train the FTRL logistic regression on a Criteo-style stream, the loop is simply: read each row, featurize, call `update`, and track a running logloss.

```python
def train_ftrl(rows, model: FTRLProximal):
    total_loss, n = 0.0, 0
    for row in rows:                       # rows is a streaming iterator
        x = featurize(row)
        y = float(row["clicked"])
        p = model.update(x, y)             # predict-then-update: progressive validation
        eps = 1e-15
        total_loss += -(y * math.log(p + eps) + (1 - y) * math.log(1 - p + eps))
        n += 1
        if n % 1_000_000 == 0:
            print(f"seen {n:>10,}  progressive logloss {total_loss / n:.5f}")
    return total_loss / n
```

The predict-then-update ordering gives you **progressive validation** for free: because the model predicts each example *before* it learns from it, the running average logloss is an honest online estimate of generalization, no held-out set required for the streaming metric. That is a real operational nicety — your training log *is* your evaluation curve.

Now the deep model, in PyTorch. The DNN replaces the linear logit with embeddings plus dense layers. We use `nn.EmbeddingBag` to sum the embeddings of the active sparse features per example (the "deep" analog of the linear dot product), concatenate any dense features, and pass through an MLP to a single logit. The loss is `BCEWithLogitsLoss`, which is logloss applied directly to the logit (numerically stabler than a separate sigmoid then BCE).

```python
import torch
import torch.nn as nn

class CTRDeepModel(nn.Module):
    def __init__(self, n_buckets=2**20, emb_dim=16, n_dense=3, hidden=(256, 128)):
        super().__init__()
        # EmbeddingBag sums embeddings of the active sparse features per row
        self.emb = nn.EmbeddingBag(n_buckets, emb_dim, mode="sum")
        in_dim = emb_dim + n_dense
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1)]
            d = h
        layers += [nn.Linear(d, 1)]        # single logit out
        self.mlp = nn.Sequential(*layers)

    def forward(self, sparse_idx, offsets, dense):
        e = self.emb(sparse_idx, offsets)  # (batch, emb_dim)
        x = torch.cat([e, dense], dim=1)
        return self.mlp(x).squeeze(1)      # (batch,) logits

def train_dnn(model, loader, epochs=1, lr=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()       # logloss on logits
    model.train()
    for _ in range(epochs):
        for sparse_idx, offsets, dense, y in loader:
            sparse_idx, offsets = sparse_idx.to(device), offsets.to(device)
            dense, y = dense.to(device), y.to(device)
            logits = model(sparse_idx, offsets, dense)
            loss = loss_fn(logits, y.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model
```

The `offsets` argument to `EmbeddingBag` is how a ragged batch of variable-length feature lists is packed into one flat tensor: `sparse_idx` is the concatenation of every example's active indices, and `offsets[i]` marks where example $i$ begins. This is the idiomatic, fast way to feed sparse CTR features through PyTorch without a dense one-hot blowup.

Finally, the evaluation harness — AUC, logloss, and the calibration ratio in one pass over a held-out temporal split (always temporal for CTR; a random split leaks future clicks into the past and inflates every metric):

```python
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    ys, ps = [], []
    for sparse_idx, offsets, dense, y in loader:
        logits = model(sparse_idx.to(device), offsets.to(device), dense.to(device))
        p = torch.sigmoid(logits).cpu().numpy()
        ps.append(p); ys.append(y.numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    auc = roc_auc_score(y, p)
    ll = log_loss(y, p, labels=[0, 1])
    calib = p.mean() / y.mean()            # >1 means predictions run high
    print(f"AUC {auc:.4f}  logloss {ll:.4f}  pred/actual {calib:.3f}")
    return auc, ll, calib
```

That `pred/actual` ratio is the cheapest calibration check there is, and it is the one that would have caught my 3 a.m. incident on a dashboard. A value of 1.0 means the average predicted CTR equals the actual CTR; the model that burned the budget would have shown 1.08 there for days.

## 7. Calibration: why probabilities must be trustworthy

We have leaned on the word "calibrated" several times; now make it precise. A model is **calibrated** if, among all the impressions it predicts a CTR of $p$, the actual click rate really is $p$. Take every impression where the model said 0.20 and the fraction that were clicked should be 0.20. Plot predicted probability on the x-axis against observed frequency on the y-axis and a perfectly calibrated model traces the diagonal — that plot is the **reliability diagram**, and the figure below is exactly that comparison, the miscalibrated curve bowing off the diagonal versus the corrected curve lying on it.

![A before and after reliability diagram comparison showing a miscalibrated deep model whose predicted probabilities run high before a calibration step pulls the curve back onto the diagonal](/imgs/blogs/the-ranking-model-ctr-prediction-foundations-5.png)

The distinction between calibration and accuracy is worth sitting with, because they are genuinely different properties. A model can be perfectly *accurate* in the ranking sense — every click outscores every non-click, AUC of 1.0 — and yet wildly *miscalibrated*, predicting 0.95 for impressions whose true rate is 0.30. Calibration is not about getting the order right; it is about the numbers being numerically honest. A weather forecaster who says "70% chance of rain" is calibrated if it rains on 70% of the days they say that, regardless of whether they ever distinguish a 71% day from a 69% day. CTR is the same: the order is one property, the honesty of the number is another, and on a bidding surface you need both.

Why does calibration matter beyond intellectual tidiness? Because of what reads the number. On a pure-sort surface where you just rank by score and show the top K, calibration is irrelevant — only the order matters, and a monotone-distorted score sorts identically. But the moment anything *arithmetic* happens to the score, calibration becomes load-bearing:

- **Ads auctions** rank by expected value $= \text{CTR} \times \text{bid}$. If CTR is inflated 1.5x, every expected value is inflated, the auction overbids, and you pay for impressions that do not convert — the failure that paged me.
- **Blended feed ranking** combines predicted click, dwell, and share with weights tuned on the assumption that each prediction is a true probability. Miscalibrate one head and the blend silently over-weights it.
- **Thresholding** — "only show if predicted CTR > 0.05" — depends entirely on the number meaning what it says.
- **Budget pacing and forecasting** multiply predicted CTR by impression volume to project clicks and spend; garbage probabilities yield garbage forecasts.

A single quantitative summary of calibration is **Expected Calibration Error (ECE)**: bin the predictions into $M$ buckets by predicted probability, and average the absolute gap between the average prediction and the observed frequency in each bin, weighted by bin size:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \,\big|\, \text{acc}(B_m) - \text{conf}(B_m) \,\big|$$

where $\text{conf}(B_m)$ is the average predicted probability in bin $m$ and $\text{acc}(B_m)$ is the observed click rate. A well-calibrated CTR model has ECE in the low single-digit thousandths; the model that burned budget would have shown ECE an order of magnitude higher. ECE is a dozen lines of numpy and belongs next to AUC and logloss in every CTR eval harness:

```python
import numpy as np

def expected_calibration_error(y_true, y_prob, n_bins=20):
    """Equal-width binning ECE. Low for a trustworthy CTR model."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    ece, n = 0.0, len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = y_prob[mask].mean()      # average predicted probability in bin
        acc = y_true[mask].mean()       # observed click rate in bin
        ece += (mask.sum() / n) * abs(acc - conf)
    return ece
```

Run that on the held-out predictions and you get a single number you can alert on. In practice teams pair it with the `pred/actual` ratio from the eval harness: the ratio catches a global scale shift (everything inflated 8%), while ECE catches local shape distortion (over-confident at the tails even when the mean is right). A deep model often passes the mean check and fails ECE, which is exactly why you need both.

#### Worked example: the dollar cost of a calibration drift

Make the auction failure concrete. A display surface serves 100 million impressions a day at a true CTR of 2% — 2 million clicks. The auction bids per impression at expected value $= \text{CTR} \times \text{bid}$, the advertiser's bid is \$0.50 per click, and the platform's pacing assumes 2 million clicks of inventory. A retrain ships a deep model whose AUC is up 0.001 (a real ranking win) but whose probabilities drifted high by a factor of 1.08 — `pred/actual` reads 1.08, a drift small enough that nobody noticed for three days. The auction now reads every impression's expected value as 8% higher than truth, so it wins 8% more auctions at the margin and paces as if there are $2{,}000{,}000 \times 1.08 = 2{,}160{,}000$ clicks of inventory. But the *true* CTR did not change, so the extra impressions it bought convert at the real 2%, not the inflated rate. The platform spent for 2.16M clicks of expected value and delivered 2.0M — an 8% overspend against delivered value, roughly $2{,}000{,}000 \times \$0.50 \times 0.08 = \$80{,}000$ a day of value mismatch, accruing silently while AUC sat green on the dashboard. Recalibrating the new model so `pred/actual` returns to 1.00 keeps the +0.001 AUC ranking win *and* erases the overspend. That is the entire argument for treating calibration as a release gate: the ranking win and the calibration drift were independent, and only one of them had a dashboard.

Now the uncomfortable part: **deep models are systematically miscalibrated, and worse than logistic regression.** Recall the result from section 4 — a logistic regression trained to optimum satisfies $\frac{1}{n}\sum \hat{p}_i = \frac{1}{n}\sum y_i$ exactly, calibrated in aggregate by construction. A deep network has no such constraint. Guo et al. (2017), "On Calibration of Modern Neural Networks," showed that modern deep nets, despite higher accuracy, are markedly *over-confident* — their predicted probabilities are pushed toward 0 and 1 harder than the truth warrants. The usual culprits are over-parameterization, training past the point of zero training loss, and regularization choices like batch norm. For CTR specifically the practical fix is a cheap post-hoc recalibration on a held-out set: **Platt scaling** (fit a one-parameter logistic to the logits), **isotonic regression** (fit a monotone step function), or **temperature scaling** (divide the logit by a single learned scalar $T$ before the sigmoid). All are monotone, so — and this is the elegant part — they *change calibration without changing AUC at all*. You get to fix the number without touching the order.

Fitting one is a handful of lines with scikit-learn. Hold out a calibration split (never the training data — the model is over-confident *on data it trained on*, so calibrating there learns nothing), and fit isotonic or Platt on its predictions:

```python
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import numpy as np

# p_cal: raw model probabilities on a held-out calibration split
# y_cal: true labels on that split
def fit_isotonic(p_cal, y_cal):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal)                 # monotone map: raw p -> calibrated p
    return iso                            # call iso.transform(p_new) at serving

def fit_platt(logits_cal, y_cal):
    # Platt scaling: one-parameter logistic on the logit
    lr = LogisticRegression(C=1e6)        # weak regularization, near-MLE
    lr.fit(logits_cal.reshape(-1, 1), y_cal)
    return lr                             # lr.predict_proba(logit)[:, 1] at serving
```

Isotonic is more flexible (any monotone shape) but needs more calibration data and can overfit on thin tails; Platt is two parameters and robust but assumes a sigmoidal distortion. For high-volume CTR, isotonic on tens of millions of calibration impressions is the usual pick. Either way, re-fit it on every retrain, because the miscalibration shape moves with the model. The dedicated treatment is [calibration and the prediction you can trust](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust); here we cover the one calibration correction that is specific to CTR training and that everyone hits: negative downsampling.

## 8. The science: negative downsampling and the probability correction

CTR data is brutally imbalanced. A typical ad surface has a click-through rate of one or two percent, which means ninety-eight or ninety-nine of every hundred training examples are negatives. Training on all of them is wasteful — the negatives are highly redundant and they dominate the gradient — so the standard move is **negative downsampling**: keep all the positives and a random fraction $w$ of the negatives. With $w = 0.1$ you drop nine of every ten negatives, shrinking the dataset by roughly an order of magnitude with almost no loss of signal, because the positives (the rare, informative class) are untouched. Throughout, $w$ is the **keep rate** — the fraction of negatives retained, a number between 0 and 1.

But downsampling the negatives changes the class balance the model sees, and therefore changes the probability it learns. The model trained on the resampled data learns $p_s = P(y=1 \mid x, \text{sampled})$, which is *higher* than the true $p = P(y=1 \mid x)$, because we artificially enriched the positive fraction. If you serve $p_s$ directly, every predicted CTR is inflated, and every downstream auction overbids — exactly the calibration failure from the last section, except this one you caused on purpose and must undo.

![A before and after comparison showing an inflated predicted click rate from training on downsampled negatives and the corrected rate after applying the closed-form formula](/imgs/blogs/the-ranking-model-ctr-prediction-foundations-6.png)

The correction is a clean piece of Bayes-rule bookkeeping. Sampling does not touch the positives but scales the negatives down by the keep rate $w$, so for any $x$ the odds of positive to negative are scaled *up* by $1/w$ in the sampled data. Work in odds. The true odds are $\frac{p}{1-p}$; the sampled odds are $\frac{p_s}{1-p_s}$; and because we kept all positives but only a fraction $w$ of the negatives, the sampled odds are $1/w$ times the true odds:

$$\frac{p_s}{1 - p_s} = \frac{1}{w} \cdot \frac{p}{1 - p}$$

Solve for $p$. Let $o_s = \frac{p_s}{1-p_s}$ be the sampled odds; then the true odds are $w \cdot o_s$, and converting odds back to probability, $p = \frac{w\,o_s}{1 + w\,o_s}$. A little algebra collapses this to the form you will see quoted in every production CTR system:

$$p = \frac{p_s}{p_s + (1 - p_s)/w}$$

Sanity-check the limits. If $w = 1$ (keep all negatives, i.e. no downsampling), the denominator is $p_s + (1-p_s) = 1$ and $p = p_s$ — no correction, as it must be. As $w$ shrinks toward 0 (more aggressive downsampling), the $(1-p_s)/w$ term blows up, the denominator grows, and $p$ is pulled down hard toward the true low rate — exactly the re-inflation of the negative mass we want. The formula is exact, costs one division per prediction at serving time, and is non-negotiable for any downsampled CTR model that feeds an auction.

#### Worked example: applying the downsampling correction with numbers

A display-ads surface has a true CTR of about 2%. We keep all positives and a fraction $w = 0.1$ of the negatives — that is, we throw away nine of every ten non-clicks. (Here $w$ is the *keep rate*, the fraction of negatives retained, which is the convention the correction formula uses.) After this downsampling, the positive fraction in the training set is no longer 2%; it is

$$p_s = \frac{0.02}{0.02 + 0.98 \times 0.1} = \frac{0.02}{0.02 + 0.098} = \frac{0.02}{0.118} \approx 0.169$$

so the model, trained to be calibrated *on its own training distribution*, learns to predict around $p_s \approx 0.169$ on a typical impression — more than eight times the true rate. Serve that raw and the auction thinks every impression is roughly eight times as valuable as it is, and overbids accordingly.

Now undo it. Apply the correction $p = \dfrac{p_s}{p_s + (1 - p_s)/w}$ with $p_s = 0.169$ and the keep rate $w = 0.1$:

$$p = \frac{0.169}{0.169 + (1 - 0.169)/0.1} = \frac{0.169}{0.169 + 8.31}$$

That denominator looks alarming until you finish it: $0.169 + 8.31 = 8.479$, and $0.169 / 8.479 \approx 0.0199$, which rounds to the **2%** true CTR we started from. The correction exactly reverses the enrichment. The intuition is that dividing $(1-p_s)$ by the small keep rate $w$ re-inflates the negative mass back to its true (un-downsampled) size, restoring the original class balance and therefore the original probability.

You validate the correction in *aggregate* (mean predicted CTR over a held-out day should match the day's actual CTR) rather than per-impression, because per-impression true CTR is unobservable. In practice teams fix $w$, apply this formula at serving on every prediction, and watch the `pred/actual` ratio from the eval harness sit at 1.00. If it drifts to 1.08, either $w$ is mis-set or the model has a deeper calibration problem that needs Platt or isotonic on top. A common bug is to *forget the correction entirely* after switching on downsampling for a faster retrain — the model trains fine, AUC is unchanged (downsampling barely moves the order), and only the bidding surface, weeks later, reveals the inflated probabilities through overspend.

A subtle point worth stating: downsampling also changes the optimal loss value, so do **not** compare the logloss of a downsampled-trained model against a full-data model directly — recompute logloss on the *corrected* probabilities against the *true* (un-downsampled) validation distribution, or you will be comparing two different quantities. This is one of the silent ways an offline CTR comparison goes wrong, and a close cousin of the offline-online gap explored across this series.

## 9. Results: LR vs FM vs DNN on Criteo and Avazu

Now the measurement. The two public benchmarks the CTR literature anchors on are **Criteo** (about 45 million ad impressions, 13 numeric and 26 categorical fields, the Kaggle Display Advertising Challenge dataset) and **Avazu** (about 40 million impressions, 24 fields, mobile ad clicks). They are the MovieLens of CTR — every paper reports on at least one, so the numbers are comparable across a decade of work. The headline is that on these datasets the whole field lives in a narrow AUC band around **0.79 to 0.81**, and the entire game is moving within that band by thousandths.

The table below is consistent with the published ranges (DeepFM, Guo et al. 2017; DCN, Wang et al. 2017; AutoInt, Song et al. 2019, all report Criteo AUC in this neighborhood). Treat the third decimal as the meaningful digit and read it as the relative ladder, not as exact reproductions — small preprocessing and split choices shift the absolute numbers by a few thousandths.

| Model | Criteo AUC | Criteo logloss | Notes |
| --- | --- | --- | --- |
| Logistic regression + FTRL | ~0.7930 | ~0.4560 | Linear, hand-built crosses, calibrated by construction |
| Factorization machine | ~0.7980 | ~0.4525 | Learns all pairwise crosses automatically |
| Field-aware FM (FFM) | ~0.7995 | ~0.4520 | Per-field latent vectors, Kaggle-winning class |
| DeepFM | ~0.8016 | ~0.4508 | FM + deep, no manual feature engineering |
| DCN-v2 | ~0.8025 | ~0.4500 | Explicit bounded-degree crosses + deep |

![A matrix comparing logistic regression, field-aware factorization machines, DeepFM, and DCN on AUC, logloss, and delta versus the linear baseline](/imgs/blogs/the-ranking-model-ctr-prediction-foundations-8.png)

Look at the absolute gaps. Going from logistic regression to DeepFM buys about **+0.0086 AUC** — call it nine thousandths. That sounds like a rounding error, and on a Kaggle leaderboard it nearly is. At production scale it is the difference between two budgets. Which brings us to the most important piece of folklore in this entire field, and the reason that thousandth on my 3 a.m. page mattered:

#### Worked example: why 0.001 AUC is real money

The lore, traceable to the deep-learning-for-recommenders papers out of Google and the ads community, is that **a 0.001 improvement in offline AUC is considered significant** because it reliably corresponds to a measurable online revenue lift. Put rough numbers on why. Suppose a feed or ads surface serves 10 billion impressions a day at a 2% CTR — 200 million clicks a day. AUC improvements on CTR models translate, empirically and roughly, into low-tenths-of-a-percent online CTR lifts; a credible rule of thumb from published A/B results is that around 0.001 offline AUC maps to roughly a few tenths of a percent relative CTR improvement when calibration holds. Take 0.2% relative: that is $200{,}000{,}000 \times 0.002 = 400{,}000$ extra clicks a day. At a conservative \$0.50 average value per click that is \$200{,}000 a day, or about \$73M a year, from one thousandth of AUC. The exact multiplier varies wildly by surface and is *not* universal — but the order of magnitude is why teams of staff engineers fight over the third decimal place. It is also why the offline metric must be measured honestly: a thousandth of AUC won by leaking future data into a random split is a thousandth of nothing, and the only AUC that maps to revenue is the one measured on a strict temporal split with no train-serve feature skew.

The other half of the table is logloss, and it tells the calibration story. DeepFM's logloss is lower than logistic regression's, which means the deep model is *both* a better ranker (AUC up) and, after proper recalibration, a comparably trustworthy probability source (logloss down). But raw, un-recalibrated, the same DeepFM would often show *worse* effective calibration than the linear model — the AUC win and the calibration loss are independent, which is the whole reason you report both columns. A model that improves AUC by 0.008 while its logloss creeps *up* is telling you it ranks better but its probabilities drifted, and on a bidding surface you must recalibrate before you ship it.

## 10. Case studies: how the giants actually did this

The foundations in this post are not academic — they are the documented architecture of the systems that monetize the modern web. Three case studies, each a real paper you can read.

**Google's FTRL ad-click system (McMahan et al., 2013).** "Ad Click Prediction: a View from the Trenches" is the paper that taught the field how to do CTR at industrial scale, and it is still the single best operational read on the subject. The model is logistic regression. The optimizer is FTRL-Proximal with the per-coordinate adaptive learning rates and $L_1$ sparsity we built in section 5. The paper is candid about everything the textbooks skip: probabilistic feature inclusion to bound the feature count, encoding weights in fewer bits to save memory, calibration layers to correct systematic prediction bias, and — tellingly — the observation that fancier optimizers gave little benefit over well-tuned FTRL. The lesson that aged best: at scale, the engineering of the feature pipeline, the memory of the model, and the *calibration* of the output mattered more than the sophistication of the learner. Logistic regression, done seriously, monetized a large fraction of the internet.

**Facebook's GBDT + logistic regression (He et al., 2014).** "Practical Lessons from Predicting Clicks on Ads at Facebook" introduced a hybrid that became a standard pattern: feed the raw features through a gradient-boosted decision tree ensemble, then use *which leaf each tree routes the example to* as the input features to an online logistic regression. The trees do automatic feature engineering — each root-to-leaf path is a learned conjunction of features, exactly the hand-built crosses of section 5 but discovered by the boosting algorithm — and the linear layer learns weights over those leaf indicators with online updates. The paper reports that this GBDT-then-LR hybrid cut their normalized entropy (a logloss-family metric) by about 3% over either piece alone, a large win in CTR terms. It also delivered two of the most-cited practical findings in the field: **data freshness matters a lot** (retraining daily versus weekly was worth a measurable normalized-entropy gain, because the click distribution drifts), and **negative downsampling with the probability correction works** — they downsampled negatives aggressively and applied the exact $p = p_s / (p_s + (1-p_s)/w)$ recalibration we derived in section 8 to restore calibrated probabilities for the auction.

**The "deep learning for CTR" lineage (2016 onward).** Google's Wide & Deep (Cheng et al., 2016) combined a wide linear model (memorization of crosses, the FTRL idea) with a deep network (generalization to unseen crosses) in a single jointly-trained model, and shipped it on the Google Play store with a reported online lift in app acquisitions. DeepFM (Guo et al., 2017) replaced the hand-engineered wide part with a factorization machine so *no* manual feature crossing is needed, matching or beating Wide & Deep on Criteo and Avazu without the feature-engineering labor. DCN and DCN-v2 (Wang et al., 2017, 2021) added an explicit "cross network" that computes bounded-degree feature interactions efficiently. The throughline across all of them: each generation automated more of the feature crossing that logistic regression made you do by hand, climbed a few thousandths of AUC, and — every single time — had to manage calibration carefully because deep models do not give it to you for free the way the linear model did.

There is a fourth case study worth naming because it is the one that frames why the funnel exists at all: **YouTube's two-stage deep recommender (Covington et al., 2016)**. The paper splits the system into exactly the candidate-generation-then-ranking architecture this series is built around. The ranking network is a deep model over a rich feature set — including the item's impression history and the user's prior interactions with the channel — and it predicts *expected watch time* rather than click, using a weighted logistic regression where positive examples are weighted by their watch time. That weighting is a calibration trick in its own right: it makes the odds the model learns proportional to expected watch time, so the calibrated output is directly usable for ranking by the quantity the product cares about. It is the same logloss-and-calibration machinery from this post, bent to a watch-time objective, and it reinforces the throughline: the score is a numeric promise, and the system is engineered around making that promise trustworthy.

If there is one meta-lesson from all four, it is that the ranker's *probability* was treated as a first-class deliverable, not a byproduct of the sort. Every one of these systems shipped a calibration step. None of them trusted a raw deep-model output on a bidding surface. That is the institutional memory this post is trying to transfer.

## 11. When to reach for what (and when not to)

A decisive guide, because every choice here is a cost.

**Start with logistic regression + FTRL when** you are launching a CTR surface, your features are mostly sparse categoricals, you need online learning against a firehose, you must serve at extreme QPS with a tiny per-example cost, and — critically — you need calibrated probabilities out of the box for a bidding or budgeting surface. It is the highest-floor, lowest-drama choice, it trains incrementally, and it is calibrated by construction. Do not dismiss it as old; it still anchors production ad systems. Its ceiling is the catch: it cannot learn crosses you did not hand it, so its AUC plateaus a few thousandths below the deep models.

**Reach for factorization machines when** the hand-built crosses have become a maintenance nightmare and you want the model to learn all pairwise interactions automatically, but you do not yet need (or cannot afford the serving cost of) a deep net. FM is the sweet spot of "automatic crosses, still cheap, still mostly calibrated," and field-aware FM squeezes out a bit more on high-cardinality categorical data. It is covered fully in [factorization machines and field-aware FM](/blog/machine-learning/recommendation-systems/factorization-machines-and-field-aware-fm).

**Go to a deep model (Wide & Deep, DeepFM, DCN, DIN) when** you have the engineering maturity to manage calibration, you have enough data to train a high-capacity model without overfitting, the surface's revenue justifies chasing the last thousandths of AUC, and you have an A/B framework to verify the offline gain actually lands online. The deep AUC ceiling is real and at scale it is worth real money — but only if you (a) recalibrate before serving and (b) measure on a strict temporal split so the offline win is not an artifact.

**Do not** ship any ranker tuned only on offline AUC to a bidding surface — AUC cannot see the calibration failure that burns budget. **Do not** train on full negatives when downsampling plus the correction gets you the same model an order of magnitude cheaper. **Do not** compare logloss across models with different downsampling factors without correcting first — you will be comparing different quantities. **Do not** use a random train-test split for CTR — it leaks future clicks and inflates every metric; always split temporally. And **do not** reach for a deep model if a well-tuned FTRL logistic regression already hits your target — the deep net adds calibration risk, training cost, and serving latency for a gain that may not survive the online test.

A stress test to make the rule concrete. *What happens when the offline AUC rises but online CTR is flat?* Three usual suspects, in order of likelihood: train-serve feature skew (a feature computed differently offline than online, so the model saw signal at training that does not exist at serving), a random rather than temporal split inflating the offline number, or a calibration shift that the auction punished even though the order improved. *What happens at 100x QPS?* The deep model's per-example latency dominates; you either distill it, cache embeddings, or fall back toward FM/LR. *What happens when negatives are mostly false negatives* (shown-but-unclicked items the user would have liked)? Your "0" labels are noisy, logloss is biased, and you lean harder on the ranking metric (AUC/GAUC) than on the absolute probability — the noise hurts calibration more than order.

## 12. What your ranker should optimize

Pulling the threads together into a single recommendation. Your ranker should optimize **logloss as the training objective** (it is the proper scoring rule that yields a probability you can trust), and you should **monitor both AUC/GAUC and logloss in evaluation** (the first for order, the second for calibration), reported on a **strict temporal split** with **no train-serve feature skew**. Whatever model you pick, the output is a numeric promise, so treat calibration as a shipping requirement, not a nicety: apply the negative-downsampling correction if you downsample, add Platt or isotonic on top if a deep model is over-confident, and watch the `pred/actual` ratio sit at 1.00 in production.

And keep the funnel framing. The ranker can afford its expensive early-crossing model only because retrieval already cut the candidate set from millions to hundreds; the re-ranker downstream will reorder for diversity and business rules; and the feedback loop that logs your served impressions will become tomorrow's training data — so a calibration bug today is a biased dataset tomorrow. The number your ranker emits is not the end of the pipeline; it is the input to a dozen other decisions and to its own next training run. Get the order right, get the number right, and measure both honestly.

## Key takeaways

- **The ranker scores a few hundred candidates precisely** with rich features, and because the candidate set is small it can afford early, cross feature interaction that the two-tower retrieval model structurally cannot do.
- **CTR prediction is binary classification trained with logloss.** The score is a logit, a sigmoid maps it to a probability, and logloss is the negative log-likelihood — a proper scoring rule whose minimizer is the true conditional probability.
- **The gradient of logloss with respect to the logit is exactly $\hat{p} - y$**, the prediction error. Sparse CTR training is cheap because each example only touches its active features, and this same residual flows back through a deep net.
- **AUC measures order, logloss measures calibration, and they do not move together.** AUC is invariant to any monotone score transform, so it cannot see a calibration failure. Report both; for personalized surfaces report GAUC too.
- **FTRL-Proximal is the reference design for large sparse online CTR**: per-coordinate adaptive learning rates plus $L_1$ that snaps rare-feature weights to exactly zero, learning one streaming example at a time.
- **Logistic regression is calibrated by construction** (its average prediction equals the empirical click rate); **deep models are systematically over-confident** and need post-hoc recalibration (Platt, isotonic, temperature) — all monotone, so they fix the number without touching AUC.
- **Negative downsampling inflates predicted CTR**, and the exact correction $p = p_s / (p_s + (1-p_s)/w)$ maps it back to the true rate. Validate it in aggregate, not per-impression.
- **At scale a 0.001 AUC gain is real revenue**, which is why the whole field fights over the third decimal — and why that gain must be measured on a temporal split with no feature skew or it maps to nothing.

## Further reading

- McMahan et al., **"Ad Click Prediction: a View from the Trenches"** (KDD 2013) — the FTRL-Proximal paper and the canonical operational guide to industrial CTR.
- He et al., **"Practical Lessons from Predicting Clicks on Ads at Facebook"** (ADKDD 2014) — GBDT + logistic regression, data freshness, and the negative-downsampling correction in production.
- Guo et al., **"DeepFM: A Factorization-Machine based Neural Network for CTR Prediction"** (IJCAI 2017) — the deep + FM model and the Criteo/Avazu benchmark numbers.
- Guo et al., **"On Calibration of Modern Neural Networks"** (ICML 2017) — why deep nets are over-confident and how temperature scaling fixes it.
- Wang et al., **"Deep & Cross Network"** (ADKDD 2017) and DCN-v2 (WWW 2021) — explicit bounded-degree feature crossing.
- Within this series: [the recommendation funnel: retrieval, ranking, re-ranking](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking), [learning to rank for recommenders](/blog/machine-learning/recommendation-systems/learning-to-rank-for-recommenders), [calibration and the prediction you can trust](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust), [wide and deep and the memorization-generalization tradeoff](/blog/machine-learning/recommendation-systems/wide-and-deep-and-the-memorization-generalization-tradeoff), and the capstone [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
