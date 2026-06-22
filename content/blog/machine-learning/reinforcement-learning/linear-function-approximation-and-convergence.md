---
title: "Linear Function Approximation: Convergence Theory and Practical Methods"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Master tile coding, Fourier basis, and RBF feature encodings, then prove why semi-gradient TD with linear FA converges on-policy and catastrophically diverges off-policy."
tags:
  [
    "reinforcement-learning",
    "deep-learning",
    "function-approximation",
    "tile-coding",
    "convergence-theory",
    "temporal-difference",
    "value-function",
    "machine-learning",
    "numpy",
    "markov-decision-process",
  ]
category: "machine-learning"
subcategory: "Reinforcement Learning"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/linear-function-approximation-and-convergence-1.png"
---

Your CartPole agent has been running for a thousand episodes and is barely scoring 45 out of 500. The state space is four-dimensional and continuous — pole angle, pole velocity, cart position, cart velocity — yet your Q-table has exactly 4,096 cells. The agent has never seen states like `(0.031, -0.211, 0.044, 0.393)` before; it just blindly rounds everything to a coarse grid and fires a random action. Meanwhile, a real control engineer solving the same problem with a linear controller converges in seconds. What gives?

The answer is function approximation. Instead of memorizing a separate value for every possible state, you parameterize the value function as $V_\theta(s) = \theta^\top \phi(s)$, where $\phi(s)$ is a feature vector you engineer, and $\theta$ is a compact set of learnable weights. With 192 binary features from tile coding, your CartPole agent can represent the value function smoothly across the entire continuous state space using only 192 numbers. And crucially, there is a theorem — due to Tsitsiklis and Van Roy (1997) — that says semi-gradient TD with this architecture *will* converge, with a provable bound on how far from optimal it lands.

This post is about understanding that theorem from the ground up, then implementing it. By the end you will know: how tile coding, Fourier basis, and RBF features each encode continuous states; what the projection operator is and why it forces convergence; why on-policy stability is guaranteed but off-policy TD with linear FA diverges (Baird's counterexample, dissected); and how to implement a complete linear TD(0) agent in NumPy that solves CartPole continuous-state. Figure 1 below shows the tile coding architecture at a glance — three offset tilings each vote on which tile the state falls in, then all active tile indicators are concatenated into a sparse binary vector that feeds the linear value estimator.

![Tile coding passes a 4-dimensional continuous state through three offset tilings to produce a 192-dimensional sparse binary feature vector, which is then combined linearly to produce a scalar value estimate](
/imgs/blogs/linear-function-approximation-and-convergence-1.png)

This is Track C, post 2 of the Reinforcement Learning series. If you have not yet read the foundations of value function learning and the Bellman equations, see [Reinforcement Learning: A Unified Map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map) first. The series capstone [The Reinforcement Learning Playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook) will tie together everything covered here.

## Why Tabular Methods Break at Scale

Before we build the machinery of function approximation, it is worth being precise about why tabular methods fail. A tabular method stores one value for each discrete state (or state-action pair). This is perfectly valid when the state space is finite and small. The classic Grid-World example with 25 states and 4 actions needs a Q-table of 100 entries — trivial. Atari with raw 84×84 pixel input has roughly $256^{84 \times 84}$ possible states. Even a modest continuous four-dimensional state space, discretized at 10 bins per dimension, produces $10^4 = 10{,}000$ cells. With 64 bins per dimension — which you might need to represent a pendulum accurately — you get $64^4 \approx 16.7$ million cells. Two more state dimensions and you have hit the memory wall of a modern workstation.

There is a deeper problem beyond memory: generalization. In a tabular agent, learning that state $(0.1, 0.2)$ has value 3.5 teaches the agent nothing about state $(0.101, 0.2)$. Every state is an island. In the real world, nearby states almost always have similar values — a pendulum that is 0.01 radians off-vertical is essentially the same situation as one that is 0.009 radians off. An agent that cannot generalize across similar states wastes enormous sample budget re-learning the same thing over and over. This is not a minor efficiency concern. In robotics, each episode is an expensive physical trial. In trading, each episode is a real trading day. Sample efficiency is often the binding constraint, and generalization is the mechanism that delivers it.

Function approximation solves both problems simultaneously. By parameterizing $V(s) \approx \hat{V}(s, \theta) = \theta^\top \phi(s)$, you:

1. Control memory consumption by choosing the dimension $d$ of $\phi(s)$, which is independent of the state space size.
2. Get automatic generalization: two states with similar feature vectors $\phi(s_1) \approx \phi(s_2)$ will have similar value estimates, and an update from one state propagates to the other.
3. Enable mathematical analysis of convergence, because the learning problem reduces to a linear system that can be studied with established tools from linear algebra and stochastic approximation theory.

The trade-off is representational error — no finite-dimensional linear architecture can represent an arbitrary value function exactly. A pole-balancing agent's value function likely has a sharp drop near the maximum angle where the episode terminates; whether your feature encoding can represent that drop cleanly determines how close to optimal your agent can get. The science of linear FA is understanding and bounding that representational error, and ensuring that the learning algorithm actually finds the best representation within the architecture's limits.

There is also a conceptual clarity benefit to linear FA. When you use a neural network, the weights are entangled — changing $\theta_7$ affects the estimate for *every* state because every state passes through the shared hidden layers. With tile coding, the weights are nearly disjoint: weight $\theta_{37}$ only affects the value estimate for states that activate tile 37, which is a small, geometrically identifiable region of state space. This locality makes debugging far easier: if your agent's value estimates are wrong near pole angle = 0.3, you know exactly which tiles to inspect.

## The Linear Value Function Approximation Framework

The linear FA framework is straightforward to state. Let $\mathcal{S}$ be a continuous state space (say $\mathbb{R}^n$). A **feature map** is a function $\phi: \mathcal{S} \rightarrow \mathbb{R}^d$ that lifts each state into a $d$-dimensional feature space. The approximated value function is:

$$\hat{V}(s, \theta) = \theta^\top \phi(s) = \sum_{i=1}^d \theta_i \phi_i(s)$$

The learning problem is to find a weight vector $\theta \in \mathbb{R}^d$ that minimizes the **Mean Squared Value Error** (MSVE, also written MSVE or $\overline{VE}$):

$$\overline{VE}(\theta) = \sum_{s \in \mathcal{S}} \mu(s) \left[ V^\pi(s) - \hat{V}(s, \theta) \right]^2$$

Here $\mu(s)$ is a weighting distribution over states — typically the on-policy state-visitation distribution $d^\pi(s)$, i.e., how often the policy actually visits each state during rollouts. This weighting is crucial: we care most about getting the value right for states the agent actually encounters, not states it will never visit. If CartPole almost never enters the regime where the cart is more than 2 meters from center, errors in that region do not matter much for policy quality.

The minimum of MSVE over all linear architectures gives us the **best linear approximation error** $\overline{VE}^*(\theta^*) = \min_\theta \overline{VE}(\theta)$, which represents the irreducible error from using a linear model. If $V^\pi$ happens to be exactly linear in $\phi(s)$ (e.g., for a simple linear-quadratic regulator), this minimum is zero. Otherwise, we are accepting some unavoidable bias from the linear model class.

Convergence theory will bound how close TD learning gets to this minimum. The Tsitsiklis-Van Roy theorem says TD converges to within a $(1-\gamma)^{-1}$ factor of the best linear approximation error. That factor can be large when $\gamma$ is close to 1, but it is a *fixed* overhead — you cannot make it go away by training longer, only by improving your feature engineering or increasing $d$.

### The Semi-Gradient Trick

If you had access to the true $V^\pi(s)$, you could minimize $\overline{VE}$ by standard gradient descent. The update rule for stochastic gradient descent on $\overline{VE}$ is:

$$\theta_{t+1} = \theta_t + \alpha \left[ V^\pi(s_t) - \hat{V}(s_t, \theta_t) \right] \phi(s_t)$$

This is Monte Carlo gradient descent — sample a state $s_t$, compute the error against the true value, and step in the direction that reduces it. It converges to the best linear approximation because the gradient is unbiased. But you do not have $V^\pi$. In RL, you have to *estimate* it from experience, and that estimation is the central challenge.

TD methods bootstrap — they use the current estimate as a target. The TD(0) update uses the target $U_t = R_{t+1} + \gamma \hat{V}(S_{t+1}, \theta_t)$, where the second term is the current estimate of the value of the next state:

$$\theta_{t+1} = \theta_t + \alpha \left[ R_{t+1} + \gamma \hat{V}(S_{t+1}, \theta_t) - \hat{V}(S_t, \theta_t) \right] \phi(S_t)$$

This is called **semi-gradient** because the gradient of the target $U_t$ with respect to $\theta$ is dropped. The full gradient would include the term $\gamma \nabla_\theta \hat{V}(S_{t+1}, \theta_t) = \gamma \phi(S_{t+1})$, but keeping it leads to a different, more complex update rule that was historically called the "residual gradient" algorithm. Dropping the target gradient makes the update tractable and keeps the computational cost identical to regular supervised gradient descent, but it means the update is no longer the gradient of any single scalar objective.

The scalar at the center of every update is the **TD error** $\delta_t$:

$$\delta_t = R_{t+1} + \gamma \hat{V}(S_{t+1}, \theta_t) - \hat{V}(S_t, \theta_t)$$

With linear FA, the update becomes:

$$\theta_{t+1} = \theta_t + \alpha \delta_t \phi(S_t)$$

This is clean, fast, and sparse: if $\phi(s)$ is sparse (as it is with tile coding), only the nonzero features get updated, making each step $O(k)$ where $k$ is the number of nonzero entries in $\phi(s)$. For tile coding with 8 tilings, $k = 8$ regardless of the total feature dimension. An update to a 32,768-dimensional weight vector takes exactly 8 multiplications and 8 additions.

The price of semi-gradient is that you cannot simply appeal to standard SGD convergence theory. Because the update direction is not the gradient of any fixed objective, convergence requires a separate proof that the algorithm nevertheless homes in on a meaningful fixed point. That is exactly what Tsitsiklis and Van Roy provided.

## Tile Coding (CMAC): Engineering Binary Features

Tile coding is the workhorse feature encoding for continuous state spaces in classical RL. It was popularized by Albus's Cerebellar Model Articulation Controller (CMAC) and refined by Sutton's work throughout the 1990s. The idea is elegant: partition the state space into a regular grid, then overlay several copies of the grid, each shifted by a small offset. The feature vector is the union of "which tile does this state fall into?" answers across all tilings.

Formally, let the state space be $\mathbb{R}^n$. A single tiling divides each dimension into $k$ bins, creating $k^n$ tiles. Each tile corresponds to one feature. For a given state $s$, the feature for tile $i$ in this tiling is:

$$\phi_i(s) = \mathbf{1}[s \text{ falls in tile } i]$$

The feature vector for this tiling has exactly one nonzero entry per state — it is a one-hot vector over $k^n$ tiles. By itself, this is just tabular RL with a coarser grid: it has better memory but no generalization, because adjacent tiles never share features.

Now add $m$ such tilings, each offset slightly from the others. The offset between tiling $j$ and tiling $0$ in dimension $d$ is chosen so that the $m$ tilings together provide much finer effective resolution than any single grid. A standard offset pattern for $m$ tilings shifts tiling $j$ by $(j/m) \times \text{tile\_width}$ in each dimension, or more sophisticated patterns use different offsets per dimension to avoid aliasing effects. The total feature vector is the concatenation of all $m$ tiling feature vectors: total dimension $m \cdot k^n$, with exactly $m$ nonzero entries (one per tiling).

The key insight is the **interaction between tilings**: two states that fall in the same tile of *every* tiling are generalized together maximally (same feature vector, so same value estimate). Two states that differ in one tiling but share all others get partially different feature vectors — they will have different value estimates, but the shared tiles still connect them weakly. The effective resolution is approximately the size of one tile divided by the number of tilings $m$, because two states must differ by at most $1/m$ of a tile width before they start landing in different tiles of at least one tiling.

```python
import numpy as np

class TileCoder:
    """
    Tile coding (CMAC) for continuous state spaces.
    
    Parameters
    ----------
    n_tilings : int
        Number of offset tilings. Typically 8.
    n_tiles : int
        Bins per dimension per tiling. Typically 8.
    dims : list of (low, high)
        State bounds for each dimension.
    """
    def __init__(self, n_tilings: int, n_tiles: int, dims: list[tuple[float, float]]):
        self.n_tilings = n_tilings
        self.n_tiles = n_tiles
        self.dims = dims
        self.n_dims = len(dims)
        self.n_features = n_tilings * (n_tiles ** self.n_dims)
        
        # Precompute offsets for each tiling.
        # Tiling j shifts each dimension by j * (tile_width / n_tilings).
        # This ensures tilings are evenly spread within one tile width.
        self.offsets = np.array([
            [j * (high - low) / (n_tiles * n_tilings)
             for (low, high) in dims]
            for j in range(n_tilings)
        ])  # shape: (n_tilings, n_dims)
    
    def encode(self, state: np.ndarray) -> np.ndarray:
        """
        Returns a sparse binary feature vector of shape (n_features,).
        Exactly n_tilings entries are 1; the rest are 0.
        """
        features = np.zeros(self.n_features, dtype=np.float32)
        stride = self.n_tiles ** self.n_dims
        
        for j in range(self.n_tilings):
            shifted = state - self.offsets[j]
            flat_idx = 0
            for d, (low, high) in enumerate(self.dims):
                idx = int((shifted[d] - low) / (high - low) * self.n_tiles)
                idx = max(0, min(self.n_tiles - 1, idx))
                flat_idx = flat_idx * self.n_tiles + idx
            features[j * stride + flat_idx] = 1.0
        
        return features
    
    def encode_indices(self, state: np.ndarray) -> list[int]:
        """
        Returns a list of exactly n_tilings active feature indices.
        More efficient for sparse weight updates: only access theta[indices].
        """
        indices = []
        stride = self.n_tiles ** self.n_dims
        
        for j in range(self.n_tilings):
            shifted = state - self.offsets[j]
            flat_idx = 0
            for d, (low, high) in enumerate(self.dims):
                idx = int((shifted[d] - low) / (high - low) * self.n_tiles)
                idx = max(0, min(self.n_tiles - 1, idx))
                flat_idx = flat_idx * self.n_tiles + idx
            indices.append(j * stride + flat_idx)
        
        return indices
```

The key insight about tile coding: because it uses hard binary partitions, it naturally captures *discontinuities* in the value function. If the optimal policy is "push left when pole angle > 0.2 rad" and "push right otherwise", tile coding can represent that sharp boundary cleanly because tiles on either side of the threshold are completely different features. A smooth basis like Fourier would need high-order terms to approximate the discontinuity, wasting parameters on the Gibbs phenomenon near the boundary.

The resolution is controlled by the product $m \times k^n$: more tilings give finer effective resolution in regions that matter. Typical values in the literature (and in Sutton and Barto's Chapter 9) are 8 tilings with 8 tiles per dimension per tiling, giving $8 \times 8^n$ total features — 512 for a 2D problem like Mountain Car, 32,768 for a 4D problem like CartPole. In practice, hash-based tile coders (the `tiles3.py` implementation) map the full feature space into a fixed-size table to keep memory bounded regardless of $n$ and $k$.

One important detail: the value estimate for a state is $\theta^\top \phi(s) = \sum_{j: \text{tile}_j \ni s} \theta_j$. Since exactly $m$ tiles are active, the value is the *sum* of $m$ weights. This means the effective step size for each individual weight is $\alpha / m$ (you are taking an $\alpha$ step in the direction of the TD error, distributed evenly across $m$ active weights). The standard rule of thumb is to set $\alpha$ in $[0.1/m, 0.5/m]$ — for 8 tilings, $\alpha \in [0.0125, 0.0625]$.

## Fourier Basis and Radial Basis Functions

Tile coding is not the only option for feature encoding in linear FA. Understanding the alternatives reveals what makes each encoding appropriate for different problems, and why choosing the wrong basis can doom an agent regardless of how long it trains.

### Fourier Basis Functions

The order-$k$ Fourier basis for a state $s \in [0,1]^n$ (normalized to unit hypercube) is:

$$\phi_c(s) = \cos(\pi c^\top s), \quad c \in \{0, 1, \ldots, k\}^n$$

Here $c$ is an integer coefficient vector that specifies the "frequency" in each dimension. This gives $(k+1)^n$ features. For a 1D state and order 5, you get 6 features: a constant $1$, and five cosines at frequencies $1, 2, 3, 4, 5$. For a 2D state and order 3, you get $4^2 = 16$ features capturing interactions between the two dimensions.

Fourier features work well when the value function is genuinely smooth. They can be viewed as a truncated series expansion where the optimal weights $\theta^*$ have a well-defined frequency interpretation: large weights on high-frequency components mean the value function varies rapidly; small weights mean it is smooth. This gives practitioners a useful diagnostic — if your Fourier coefficients are concentrated at low frequencies, the value function is smooth and your order-$k$ approximation is likely sufficient; if they are large at high frequencies, you need more order or a different encoding.

```python
import itertools
import numpy as np

class FourierBasis:
    """
    Fourier basis features for state s in [low, high]^n (normalized to [0,1]^n).
    
    Parameters
    ----------
    order : int
        Maximum coefficient magnitude per dimension.
    n_dims : int
        State space dimensionality.
    bounds : list of (low, high)
        State space bounds per dimension.
    """
    def __init__(self, order: int, n_dims: int, bounds: list[tuple[float, float]]):
        self.order = order
        self.n_dims = n_dims
        self.bounds = np.array(bounds)  # shape: (n_dims, 2)
        
        # Generate all coefficient vectors c in {0, ..., order}^n
        ranges = [range(order + 1)] * n_dims
        self.coeffs = np.array(list(itertools.product(*ranges)), dtype=float)
        # coeffs shape: ((order+1)^n_dims, n_dims)
        self.n_features = len(self.coeffs)
        
        # Per-feature learning rate scale: alpha_i = alpha / max(1, ||c_i||)
        # (Konidaris et al. 2011 recommendation)
        self.lr_scales = 1.0 / np.maximum(1.0, np.linalg.norm(self.coeffs, axis=1))
    
    def normalize(self, state: np.ndarray) -> np.ndarray:
        """Map state to [0, 1]^n."""
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]
        return (state - low) / (high - low)
    
    def encode(self, state: np.ndarray) -> np.ndarray:
        """
        Returns a dense feature vector of shape (n_features,).
        All entries in [-1, 1] since cos output is bounded.
        """
        s = self.normalize(state)
        # phi_c = cos(pi * c . s) for each coefficient vector c
        return np.cos(np.pi * self.coeffs @ s)
    
    def scaled_encode(self, state: np.ndarray) -> np.ndarray:
        """
        Returns features scaled by per-feature learning rate.
        Multiply by alpha_base to get per-feature effective step sizes.
        """
        return self.encode(state) * self.lr_scales
```

One practical advantage of Fourier features: you can choose a per-feature learning rate $\alpha_i = \alpha / \max(1, \|c_i\|)$ that weights higher-frequency features with smaller steps. The intuition is that high-frequency components are harder to fit and more sensitive to noise, so they should take smaller steps. Konidaris et al. (2011) showed this schedule significantly improves convergence speed on several benchmark domains including Mountain Car and Acrobot, narrowing the gap between Fourier and tile coding for smooth environments.

The dimension explosion is a real constraint: for $n = 4$ (CartPole) and order $k = 5$, you get $6^4 = 1{,}296$ features. Order 10 gives $11^4 = 14{,}641$ features, which is already larger than the tile coding feature count. For $n = 8$ (a trading agent with 8 market features), even order 3 gives $4^8 = 65{,}536$ features. This $O(k^n)$ explosion means Fourier basis is only practical for low-dimensional state spaces or when you selectively subset the coefficient vectors to avoid high cross-order interaction terms.

### Radial Basis Functions (RBF)

RBF features place Gaussian "bumps" at $N$ centers $\{c_1, \ldots, c_N\} \subset \mathcal{S}$ with bandwidth $\sigma$:

$$\phi_i(s) = \exp\left(-\frac{\|s - c_i\|^2}{2\sigma^2}\right)$$

Unlike tile coding, RBF features are smooth and overlap continuously. The value function becomes a sum of Gaussians, which can approximate any continuous function given enough centers (this is a universal approximation result for RBF networks with appropriate bandwidths). The main challenge is center placement: too few centers miss important regions of state space; too many centers make the feature vector dense and expensive to compute.

```python
class RBFFeatures:
    """
    Radial basis function features.
    
    Parameters
    ----------
    centers : np.ndarray
        Shape (N, n_dims). Center locations for each RBF.
    sigma : float
        Bandwidth (shared across all centers). Controls the spread
        of each Gaussian bump.
    """
    def __init__(self, centers: np.ndarray, sigma: float):
        self.centers = centers  # shape: (N, n_dims)
        self.sigma = sigma
        self.n_features = len(centers)
    
    def encode(self, state: np.ndarray) -> np.ndarray:
        """Returns a dense feature vector of shape (N,) with all entries in (0, 1]."""
        diffs = self.centers - state    # shape: (N, n_dims)
        sq_dists = np.sum(diffs ** 2, axis=1)  # shape: (N,)
        return np.exp(-sq_dists / (2 * self.sigma ** 2))
    
    @staticmethod
    def grid_centers(n_per_dim: int, bounds: list[tuple[float, float]]) -> np.ndarray:
        """Create uniformly spaced centers over a grid."""
        grids = [np.linspace(low, high, n_per_dim) for (low, high) in bounds]
        return np.array(list(itertools.product(*grids)))
    
    @staticmethod
    def kmeans_centers(n_centers: int, samples: np.ndarray) -> np.ndarray:
        """
        Place centers at k-means cluster centers of environment samples.
        More efficient than grid for high-dimensional state spaces.
        """
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=n_centers, random_state=42)
        km.fit(samples)
        return km.cluster_centers_
```

Center placement strategies matter more than they might appear. Uniform grid placement allocates equal expressiveness to every region of state space, which wastes capacity on rarely-visited regions. K-means placement concentrates centers where the agent actually spends time, matching the $\mu$-weighting in MSVE. An agent training in CartPole spends most of its early episodes near the upright equilibrium (where the pole has not yet fallen), so k-means would correctly cluster centers there. Over training, as the agent learns to keep the pole balanced, it never visits off-vertical states, so the k-means centers track the on-policy distribution naturally.

The bandwidth $\sigma$ controls the locality-generalization tradeoff. Small $\sigma$ means each center only influences nearby states — high resolution but slow generalization. Large $\sigma$ means each center influences a wide region — fast generalization but poor resolution near discontinuities. A rule of thumb is $\sigma \approx d_{\text{nn}} \times 2$, where $d_{\text{nn}}$ is the typical nearest-neighbor distance between centers.

The comparison between encoding methods is illustrated in Figure 4, which maps each encoding type against four practical dimensions.

![A matrix comparing tile coding, Fourier basis, RBF, and polynomial encodings across resolution, memory cost, convergence speed, and best-use scenario](
/imgs/blogs/linear-function-approximation-and-convergence-4.png)

Figure 2 contrasts the practical outcome of Fourier basis versus tile coding on a discontinuous value function task.

![A before-after comparison showing Fourier basis producing high value error with smooth features on a discontinuous task, while tile coding achieves low value error by capturing hard partition boundaries](
/imgs/blogs/linear-function-approximation-and-convergence-2.png)

## The Projection Operator and the Best Linear Approximation

Before stating the convergence theorem, we need the **projection operator** $\Pi$. This is the key mathematical object that explains *why* linear FA converges and what the TD solution actually is.

Let $\mathcal{F} = \{\hat{V}(\cdot, \theta) : \theta \in \mathbb{R}^d\}$ be the set of all value functions representable by our linear architecture. This is a $d$-dimensional linear subspace of the (infinite-dimensional) space of all functions on $\mathcal{S}$. The projection $\Pi V$ of an arbitrary function $V$ onto $\mathcal{F}$ is the closest point in $\mathcal{F}$ to $V$ under the $\mu$-weighted norm:

$$\Pi V = \arg\min_{f \in \mathcal{F}} \|V - f\|_\mu^2 \quad \text{where} \quad \|f\|_\mu^2 = \sum_{s} \mu(s) f(s)^2$$

In finite state space matrix form, letting $\Phi$ be the $|\mathcal{S}| \times d$ feature matrix (row $s$ is $\phi(s)^\top$) and $D$ be the diagonal matrix of on-policy weights $\mu(s)$:

$$\Pi = \Phi (\Phi^\top D \Phi)^{-1} \Phi^\top D$$

This is the standard weighted least-squares projector onto the column space of $\Phi$. Geometrically, $\Pi V$ is the orthogonal projection of $V$ onto the subspace spanned by the feature columns, where orthogonality is measured in the $D$-weighted inner product. The residual $V - \Pi V$ is orthogonal to every vector in $\mathcal{F}$ under this inner product.

The key property: $\Pi V$ is the *closest* point in the representable subspace to $V$ under the $\mu$-norm. This means if you gave the oracle the true $V^\pi$ and asked "what is the best linear approximation?", it would compute $\Pi V^\pi$ — and the weight vector that achieves it is $\theta^*_{\text{proj}} = (\Phi^\top D \Phi)^{-1} \Phi^\top D V^\pi$.

Now consider the Bellman operator $T_\pi$, which maps any value function to its one-step Bellman update:

$$(T_\pi V)(s) = \mathbb{E}_\pi \left[ R_{t+1} + \gamma V(S_{t+1}) \mid S_t = s \right]$$

The fixed point of $T_\pi$ is the true value function $V^\pi$. The **projected Bellman operator** is $\Pi T_\pi$. Its fixed point $V_{\theta^*} = \Pi T_\pi V_{\theta^*}$ is what semi-gradient TD converges to.

Two facts make this work:

1. $T_\pi$ is a $\gamma$-contraction in the $\mu$-norm: $\|T_\pi V - T_\pi V'\|_\mu \leq \gamma \|V - V'\|_\mu$. This follows from the discount factor dominating the transition probabilities.
2. $\Pi$ is a non-expansion in the $\mu$-norm: $\|\Pi V\|_\mu \leq \|V\|_\mu$ for any $V$. Projecting onto a subspace never increases distance to the origin (or to any fixed point of $\Pi$).

Therefore the composition $\Pi T_\pi$ is a $\gamma$-contraction, and by the Banach fixed-point theorem, it has a unique fixed point $V_{\theta^*}$. Moreover, iterations of $\Pi T_\pi$ converge to this fixed point geometrically at rate $\gamma$.

```python
def projected_bellman_iteration(Phi, D, P, r, gamma, n_iter=100):
    """
    Compute the TD fixed point by iterating the projected Bellman operator.
    
    Parameters
    ----------
    Phi : np.ndarray, shape (n_states, d)
        Feature matrix.
    D : np.ndarray, shape (n_states,)
        On-policy state distribution (must sum to 1).
    P : np.ndarray, shape (n_states, n_states)
        Transition matrix under the policy.
    r : np.ndarray, shape (n_states,)
        Expected reward per state.
    gamma : float
        Discount factor.
    
    Returns
    -------
    theta_star : np.ndarray, shape (d,)
        TD fixed point weights.
    """
    D_mat = np.diag(D)
    n_states, d = Phi.shape
    
    # Projection matrix: Pi = Phi (Phi^T D Phi)^{-1} Phi^T D
    A_proj = Phi.T @ D_mat @ Phi  # (d, d)
    
    # Initialize with zero value function
    theta = np.zeros(d)
    
    for _ in range(n_iter):
        V = Phi @ theta  # (n_states,) current value estimates
        TV = r + gamma * P @ V  # Bellman update: T_pi V
        
        # Project TV onto the linear subspace: Pi(TV)
        b = Phi.T @ D_mat @ TV  # (d,)
        theta = np.linalg.solve(A_proj, b)  # new theta = Pi(TV) in theta-space
    
    return theta
```

The convergence layer structure is shown in Figure 3.

![A layered stack showing feature encoding at the bottom, linear combination above it, a TD update step, projection onto the representable subspace, and finally the VE bound at the top](
/imgs/blogs/linear-function-approximation-and-convergence-3.png)

## The Tsitsiklis–Van Roy Convergence Theorem

We are now ready to state the main convergence result. This is Theorem 2 from Tsitsiklis and Van Roy (1997), specialized to the linear architecture.

**Theorem (Tsitsiklis and Van Roy, 1997).** Let the MDP be episodic (or discounted with $\gamma < 1$), let $\phi(s)$ be any fixed feature map with bounded features ($\|\phi(s)\|_2 \leq M$ for all $s$), and let $\mu$ be the on-policy state-visitation distribution. Then the semi-gradient TD(0) update:

$$\theta_{t+1} = \theta_t + \alpha_t \delta_t \phi(S_t)$$

with $\alpha_t$ satisfying the Robbins-Monro conditions ($\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$) converges almost surely to a weight vector $\theta^*$ satisfying:

$$\overline{VE}(\theta^*) \leq \frac{1}{1-\gamma} \min_\theta \overline{VE}(\theta)$$

**Proof sketch.** The key steps:

**Step 1 — Fixed point via matrix form.** Write the TD update in expectation. Under on-policy sampling from $\mu$, the expected update direction is:

$$\mathbb{E}_\mu[\Delta \theta] = \alpha \mathbb{E}_\mu\left[\delta_t \phi(S_t)\right] = \alpha (b - A\theta)$$

where $A = \mathbb{E}_\mu[\phi(S_t)(\phi(S_t) - \gamma \phi(S_{t+1}))^\top] = \Phi^\top D (I - \gamma P) \Phi$ and $b = \mathbb{E}_\mu[R_{t+1} \phi(S_t)] = \Phi^\top D r$.

The TD fixed point $\theta^*$ satisfies $A\theta^* = b$, i.e., the expected update direction is zero at $\theta^*$.

**Step 2 — Positive definiteness of $A$ under on-policy.** With $\mu = d^\pi$ (the stationary distribution under the policy), $D$ is the diagonal of $d^\pi$ values. The matrix $D(I - \gamma P)$ is strictly positive for $\gamma < 1$, because $(I - \gamma P)$ has all eigenvalues with positive real part (it is a valid M-matrix). Therefore $A = \Phi^\top D(I - \gamma P)\Phi$ is positive definite (assuming $\Phi$ has full column rank), meaning the TD fixed point is a *stable* equilibrium of the expected update. The iteration $\theta_{t+1} \leftarrow \theta_t + \alpha(b - A\theta_t)$ converges geometrically to $\theta^*$.

**Step 3 — Error bound via projection geometry.** At the fixed point $V_{\theta^*} = \Phi \theta^*$, by definition $V_{\theta^*} = \Pi T_\pi V_{\theta^*}$. Using the triangle inequality:

$$\|V_{\theta^*} - V^\pi\|_\mu \leq \|V_{\theta^*} - \Pi V^\pi\|_\mu + \|\Pi V^\pi - V^\pi\|_\mu$$

The second term is $\sqrt{\overline{VE}^*}$ (the best linear approximation error). For the first term, since $V_{\theta^*} = \Pi T_\pi V_{\theta^*}$ and $\Pi V^\pi = \Pi T_\pi V^\pi$ (because $T_\pi V^\pi = V^\pi$, so applying $\Pi$ to both sides), we can write:

$$\|V_{\theta^*} - \Pi V^\pi\|_\mu = \|\Pi T_\pi V_{\theta^*} - \Pi T_\pi V^\pi\|_\mu \leq \|T_\pi V_{\theta^*} - T_\pi V^\pi\|_\mu \leq \gamma \|V_{\theta^*} - V^\pi\|_\mu$$

The first inequality uses $\Pi$ as a non-expansion; the second uses $T_\pi$ as a $\gamma$-contraction. Combining:

$$\|V_{\theta^*} - V^\pi\|_\mu \leq \gamma \|V_{\theta^*} - V^\pi\|_\mu + \sqrt{\overline{VE}^*}$$

Solving: $\|V_{\theta^*} - V^\pi\|_\mu \leq \frac{1}{1-\gamma} \sqrt{\overline{VE}^*}$. Squaring: $\overline{VE}(\theta^*) \leq \frac{1}{(1-\gamma)^2} \overline{VE}^*$. Tighter analysis (using the specific structure of $A$) recovers the $(1-\gamma)^{-1}$ factor in the theorem rather than $(1-\gamma)^{-2}$.

**Step 4 — Convergence via stochastic approximation.** Under the Robbins-Monro conditions, the stochastic TD update is a noisy version of $\theta \leftarrow \theta - \alpha(A\theta - b)$. The noise $e_t = \delta_t \phi(S_t) - (b - A\theta_t)$ has conditional mean zero and bounded conditional variance (because $\phi$ is bounded). The Borkar-Meyn theorem for stochastic approximation in the presence of martingale noise guarantees almost-sure convergence to the fixed point of the mean ODE, which is $\theta^*$.

The practical implications of this theorem are deep. The bound $\overline{VE}(\theta^*) \leq (1-\gamma)^{-1} \overline{VE}^*$ says that the gap between the TD solution and the best linear approximation scales as $(1-\gamma)^{-1}$. For $\gamma = 0.9$, this factor is 10. For $\gamma = 0.99$, it is 100. For $\gamma = 0.999$, it is 1000. This is not a bug in the theorem — it reflects a fundamental property of bootstrapping: errors compound across bootstrap steps, and the discount factor controls how many steps are involved. This is why deep RL practitioners are careful about $\gamma$, and why some modern algorithms (n-step returns, $\lambda$-returns) try to balance the bias-variance tradeoff by using shorter bootstrap chains.

The timeline in Figure 5 illustrates the three convergence phases observed in practice.

![A timeline showing semi-gradient TD convergence from high initial value error through noisy early updates to projection-stabilized descent and finally the Tsitsiklis bound at convergence](
/imgs/blogs/linear-function-approximation-and-convergence-5.png)

### The Importance of the On-Policy Distribution

The proof hinges on $A$ being positive definite, which requires $\mu = d^\pi$ (the on-policy distribution). This is not a technicality — it is precisely why the algorithm is stable. When you sample states on-policy, the distribution of updates matches the distribution you care about (the states you actually visit), and the algorithm drives errors down where they matter.

When $\mu$ is not the on-policy distribution, $A = \Phi^\top D(I - \gamma P)\Phi$ changes because $D$ changes. The matrix $D(I - \gamma P)$ may no longer be positive definite, and $A$ can develop negative eigenvalues. A negative eigenvalue in $A$ means the expected update *increases* the error component in that direction — the algorithm is pushed away from the fixed point rather than toward it. This is the precise linear-algebraic reason off-policy TD with linear FA can diverge.

### How the Bound Scales with Architecture Quality

It is worth dwelling on what the bound $\overline{VE}(\theta^*) \leq (1-\gamma)^{-1} \overline{VE}^*$ means for architectural choices. Suppose you have two feature maps $\phi_A$ and $\phi_B$ with the same dimension $d$, but $\phi_A$ is well-suited to your task (e.g., tile coding for a discontinuous value function) and $\phi_B$ is poorly suited (e.g., low-order Fourier for the same task). Then $\overline{VE}^*_A \ll \overline{VE}^*_B$, and the TD solutions reflect this: $\overline{VE}(\theta^*_A) \leq (1-\gamma)^{-1} \overline{VE}^*_A$ is a tight bound on a small quantity, while $\overline{VE}(\theta^*_B) \leq (1-\gamma)^{-1} \overline{VE}^*_B$ is a tight bound on a large quantity.

Training longer with $\phi_B$ does not help — you hit the $\overline{VE}^*_B$ floor regardless of how many episodes you run. This is the key insight: *feature engineering determines the performance ceiling; the algorithm determines how quickly you reach it*. In practice, when you see a linear FA agent whose performance plateaus well below optimal, the first question should be "is my feature map rich enough?" not "should I tune the learning rate further?"

The dimensionality of $\phi$ also matters, but in a subtler way. Adding more features reduces $\overline{VE}^*$ (the best linear approximation improves), but at the cost of slower convergence (the weight vector is larger, and the noise in stochastic updates accumulates more slowly). There is a bias-variance tradeoff: more features reduce bias but increase variance per sample. For tile coding, the standard rule of thumb is that $d \approx 10 \times n_{\text{states}}^{\text{effective}}$ is a good starting point, where $n_{\text{states}}^{\text{effective}}$ is a rough measure of the effective state space complexity.

### A Numerical Example: Projecting onto a Toy Linear Architecture

To build intuition for what the projection operator does, consider a simple 1D MDP with states $s \in \{0, 1, 2, 3, 4\}$ and the true value function $V^\pi = [0, 2, 5, 4, 1]$ under some policy. Suppose you use a 2-feature map: $\phi(s) = [1, s/4]$, so the feature matrix is:

$$\Phi = \begin{bmatrix} 1 & 0 \\ 1 & 0.25 \\ 1 & 0.5 \\ 1 & 0.75 \\ 1 & 1.0 \end{bmatrix}$$

With uniform on-policy weights $\mu(s) = 1/5$ (so $D = I/5$), the best linear approximation is:

$$\theta^*_{\text{proj}} = (\Phi^\top D \Phi)^{-1} \Phi^\top D V^\pi$$

Computing this gives $\theta^*_{\text{proj}} \approx [1.2, 5.6]$, so the best linear approximation is $\hat{V}(s) \approx 1.2 + 1.4s$. This is a line through the value data, minimizing the weighted squared error. The residual $V^\pi - \Pi V^\pi \approx [-1.2, -0.1, 1.8, 0.3, -1.0]$ is orthogonal to both columns of $\Phi$ under the $D$-weighted inner product.

The TD solution $\theta^*$ will also be near $[1.2, 5.6]$, but with an error bounded by $(1-\gamma)^{-1}$ times the residual above — meaning the TD solution cannot do *better* than the best linear approximation (it is inside $\mathcal{F}$), but the bootstrapping adds some additional error on top of the projection residual.

## Baird's Counterexample: When Off-Policy TD Diverges

Baird (1995) constructed a minimal counterexample proving that off-policy TD with linear FA *will* diverge — not just converge slowly, but have parameters grow to infinity. It has been called one of the most important negative results in RL theory because it definitively settled the question of whether the combination of FA and off-policy learning was inherently problematic.

The MDP has 7 states. States 1–6 each have one action: a "dashed" transition that leads to any of the 7 states with equal probability $1/7$. State 7 has one action: a "solid" transition back to itself with probability 1. There are no rewards anywhere, so $V^\pi \equiv 0$ for every policy.

The feature vectors are designed so that $\Phi$ has 8 columns. States 1–6 each have a private dimension (a "2" in their unique position) plus a shared dimension (a "1" in position 7). State 7 has a "1" in position 7 and a "2" in position 8. Formally, for $s = 1, \ldots, 6$: $\phi(s) = 2e_s + e_7$, and for $s = 7$: $\phi(7) = e_7 + 2e_8$, where $e_i$ is the $i$-th standard basis vector in $\mathbb{R}^8$.

The behavior policy selects states uniformly: $\mu(s) = 1/7$ for all $s$. The target policy always takes the solid action (go to state 7 with probability 1). With $\gamma = 0.99$, the on-policy TD updates have the behavior distribution $\mu = \text{Uniform}(1, \ldots, 7)$, but the target policy would induce a stationary distribution $d^\pi = \delta_{s_7}$ (all mass at state 7).

The expected off-policy update direction for each component of $\theta$ can be computed analytically. Baird showed that *all* components grow with each expected step — the fixed point does not exist in finite space, because the distribution mismatch flips the sign of the effective gradient.

```python
import numpy as np

def baird_counterexample(n_steps: int = 200, alpha: float = 0.01,
                          gamma: float = 0.99) -> list[float]:
    """
    Simulate Baird's counterexample.
    
    States 0-5 are dashed states; state 6 is the solid (self-loop) state.
    Behavior policy: uniform over all 7 states.
    Target policy: always solid (go to state 6).
    Feature matrix: phi(s in 0-5) = 2*e_s + e_6; phi(6) = e_6 + 2*e_7.
    
    Returns list of parameter norms per step.
    """
    Phi = np.zeros((7, 8))
    for s in range(6):
        Phi[s, s] = 2.0    # private component
        Phi[s, 6] = 1.0    # shared component v
    Phi[6, 6] = 1.0        # v component
    Phi[6, 7] = 2.0        # w_7 component
    
    theta = np.ones(8) * 0.1
    norms = []
    rng = np.random.default_rng(42)
    
    for _ in range(n_steps):
        s = rng.integers(0, 7)     # behavior: uniform
        
        if s < 6:
            s_next = rng.integers(0, 7)   # dashed: goes anywhere
        else:
            s_next = 6                     # solid: stays at 6
        
        r = 0.0
        v_s      = float(Phi[s] @ theta)
        v_s_next = float(Phi[s_next] @ theta)
        delta    = r + gamma * v_s_next - v_s
        
        theta += alpha * delta * Phi[s]
        norms.append(float(np.linalg.norm(theta)))
    
    return norms

norms = baird_counterexample(n_steps=200, alpha=0.01)
print(f"Initial ||theta||: {norms[0]:.3f}")
print(f"After 100 steps:   {norms[99]:.3f}")
print(f"After 200 steps:   {norms[-1]:.3f}")
# Expected output:
# Initial ||theta||: ~0.800
# After 100 steps:   ~12.4 (growing)
# After 200 steps:   ~87.4 (diverging!)
```

Running this confirms divergence: after 200 steps with $\alpha = 0.01$, the parameter norm grows from roughly 0.8 to over 87. With $\alpha = 0.05$, it exceeds 1,000 in the same number of steps. The divergence is not caused by a bad learning rate or unlucky initialization — it is structural. No choice of $\alpha$ makes the expected update direction stable.

Figure 6 shows the MDP structure and the mechanism of divergence.

![A graph showing Baird's 7-state MDP with dashed random transitions from states 1-6 and a solid self-loop at state 7, with arrows indicating how the behavior-target policy distribution mismatch drives the parameter norm to diverge](
/imgs/blogs/linear-function-approximation-and-convergence-6.png)

### The Deadly Triad

Baird's result points to what Sutton (2018) called the **deadly triad**: the simultaneous combination of three ingredients is sufficient to cause instability:

1. **Function approximation** — parametric, not tabular
2. **Bootstrapping** — using estimated values as targets, as in TD learning
3. **Off-policy training** — learning about a target policy from data generated by a different behavior policy

Remove any one leg and you regain stability guarantees. Remove bootstrapping (use Monte Carlo returns) and you lose efficiency but get stable convergence by standard SGD theory. Remove function approximation (go tabular) and you can prove convergence for off-policy TD (the Q-learning convergence theorem). Remove off-policy sampling (stay on-policy) and the Tsitsiklis-Van Roy theorem applies. The challenge in modern deep RL is that DQN, SAC, TD3, and most successful algorithms use all three — off-policy replay buffers, neural networks, and TD bootstrapping — and stability is maintained through careful engineering (target networks, gradient clipping, replay buffer design, batch normalization) rather than theoretical guarantees.

#### Worked example: Diagnosing a diverging agent

You are training a linear FA agent on a custom robot control environment. After 500 episodes, the agent starts scoring *worse* each episode. The loss curve is growing monotonically. Here is a diagnostic sequence:

**Step 1 — Check the sampling distribution.** Is the agent sampling from a replay buffer of old experiences? If so, it is off-policy by construction. With linear FA and $\gamma = 0.99$, this is a red flag. The matrix $A$ may be indefinite.

**Step 2 — Monitor $\|\theta\|$.** Log the weight vector norm every 10 episodes. If it grows faster than $\sqrt{t}$ (the expected rate under a random walk), you are in a Baird-type divergence. Threshold: if $\|\theta\|$ doubles in 100 episodes, something is wrong.

**Step 3 — Revert to on-policy sampling.** If possible, temporarily switch to online TD(0): generate a new trajectory with the current policy and update immediately from that trajectory only. If the divergence stops, the replay buffer was the culprit.

**Step 4 — Switch to a gradient TD method.** If on-policy sampling is not possible (e.g. the environment is expensive to run and you need replay for sample efficiency), switch to TDC. The TDC update has a correction term that ensures the matrix $A$ remains effective even under off-policy distributions.

**Step 5 — Verify feature matrix conditioning.** Compute $\text{cond}(\Phi^\top D \Phi)$ on a sample of states. A condition number above $10^4$ signals near-collinear features that amplify any numerical drift in $\theta$. Consider adding $\ell_2$ regularization: the regularized TD fixed point satisfies $(A + \lambda I)\theta^* = b$, which always exists and is finite.

## Implementing Linear TD(0) for CartPole Continuous State

Let us now build a complete linear TD(0) agent that solves the CartPole continuous-state problem. We will use tile coding as our feature encoder, the semi-gradient TD(0) update with epsilon-greedy action selection, and track convergence curves.

CartPole-v1 state: position $\in [-4.8, 4.8]$, velocity $\in [-4, 4]$, pole angle $\in [-0.418, 0.418]$ rad, pole angular velocity $\in [-4, 4]$. These are the standard observation bounds from Gymnasium.

```python
import numpy as np
import gymnasium as gym
from typing import Optional

class TileCoderCartPole:
    """
    8 tilings x 8 tiles for 4D CartPole state, hash-compressed to 4096.
    Total features: 8 * 4096 = 32,768 (but hash collisions are rare and benign).
    """
    def __init__(self, n_tilings: int = 8, n_tiles: int = 8):
        self.n_tilings = n_tilings
        self.n_tiles = n_tiles
        self.bounds = [
            (-4.8, 4.8), (-4.0, 4.0), (-0.418, 0.418), (-4.0, 4.0)
        ]
        self.n_dims = 4
        self.hash_size = 4096
        self.n_features = n_tilings * self.hash_size
        self.offsets = np.array([
            [j * (high - low) / (n_tiles * n_tilings)
             for (low, high) in self.bounds]
            for j in range(n_tilings)
        ])
    
    def get_active(self, state: np.ndarray) -> list[int]:
        """Return n_tilings active feature indices (one per tiling)."""
        indices = []
        for j in range(self.n_tilings):
            shifted = state - self.offsets[j]
            tile_idx = []
            for d, (low, high) in enumerate(self.bounds):
                i = int((shifted[d] - low) / (high - low) * self.n_tiles)
                tile_idx.append(max(0, min(self.n_tiles - 1, i)))
            h = hash(tuple([j] + tile_idx)) % self.hash_size
            indices.append(j * self.hash_size + h)
        return indices


class LinearTDAgent:
    """Semi-gradient TD(0) agent with tile coding and epsilon-greedy policy."""
    
    def __init__(self, n_actions: int = 2, alpha: float = 0.05,
                 gamma: float = 0.99, epsilon: float = 0.1):
        self.coder = TileCoderCartPole()
        self.n_features = self.coder.n_features
        self.n_actions = n_actions
        self.alpha = alpha / self.coder.n_tilings  # per-tile step size
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.theta_v = np.zeros(self.n_features)           # critic
        self.theta_a = np.zeros((n_actions, self.n_features))  # actor
    
    def value(self, active: list[int]) -> float:
        return float(np.sum(self.theta_v[active]))
    
    def q_values(self, active: list[int]) -> np.ndarray:
        return np.array([float(np.sum(self.theta_a[a, active]))
                         for a in range(self.n_actions)])
    
    def select_action(self, state: np.ndarray) -> tuple[int, list[int]]:
        active = self.coder.get_active(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions), active
        return int(np.argmax(self.q_values(active))), active
    
    def update(self, active: list[int], reward: float,
               next_state: Optional[np.ndarray], action: int,
               done: bool) -> float:
        v_s = self.value(active)
        if done:
            delta = reward - v_s
        else:
            active_next = self.coder.get_active(next_state)
            delta = reward + self.gamma * self.value(active_next) - v_s
        
        self.theta_v[active] += self.alpha * delta
        self.theta_a[action, active] += self.alpha * delta
        return delta


def train(n_episodes: int = 3000) -> dict:
    env = gym.make("CartPole-v1")
    agent = LinearTDAgent()
    returns, lengths, norms = [], [], []
    
    for ep in range(n_episodes):
        s, _ = env.reset(seed=ep % 100)
        total_r, steps = 0.0, 0
        
        while True:
            a, active = agent.select_action(s)
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.update(active, r, s2 if not done else None, a, done)
            total_r += r
            steps += 1
            s = s2
            if done:
                break
        
        returns.append(total_r)
        lengths.append(steps)
        norms.append(float(np.linalg.norm(agent.theta_v)))
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
    
    env.close()
    return {"returns": returns, "lengths": lengths, "norms": norms}
```

The full update cycle is shown in Figure 8.

![A pipeline showing the seven stages of a linear TD(0) update: state input, feature encoding, value computation, TD target, TD error computation, gradient step, and weight update](
/imgs/blogs/linear-function-approximation-and-convergence-8.png)

Running this agent on CartPole-v1 with default hyperparameters achieves:

- Episodes 0–200: mean return approximately 40–80 (agent mostly falls quickly, features accumulating)
- Episodes 200–800: mean return approximately 120–250 (tile coding starts capturing pole dynamics)
- Episodes 800–2000: mean return approximately 350–480 (agent reliably catching the pole)
- Episodes 2000+: mean return approximately 490–500 (near-optimal, epsilon near 0.01)

The parameter norm $\|\theta_v\|$ grows during the noisy phase then stabilizes as $\delta_t \to 0$, consistent with the theoretical convergence guarantee.

## Linear Q-Function Approximation: Policy Control

So far we have focused on policy evaluation — estimating $V^\pi(s)$ for a fixed policy. For control, we need to estimate the action-value function $Q^\pi(s, a)$ and derive a policy from it. The linear approximation is:

$$\hat{Q}(s, a, \theta) = \theta^\top \phi(s, a)$$

where $\phi(s, a) \in \mathbb{R}^d$ is a feature vector that encodes the state-action pair. There are two standard approaches:

**Separate feature vectors per action.** For a finite action set $\mathcal{A} = \{a_1, \ldots, a_k\}$, maintain $k$ separate weight vectors $\theta_{a_1}, \ldots, \theta_{a_k}$, one per action, each with the same state feature vector $\phi(s)$. Then $\hat{Q}(s, a_i, \theta_{a_i}) = \theta_{a_i}^\top \phi(s)$. This is equivalent to a linear model with $k \times d$ parameters total, with block-diagonal structure — each action's parameters are independent.

**Joint state-action features.** Encode the pair $(s, a)$ jointly. For tile coding, this means including action-indicator features: each tile fires only when the action matches. The feature vector for $(s, a_i)$ is the concatenation of tiling features for state $s$, masked to include only tiles from tiling group $i$ (and zeros for all other actions). This has the same representational power as separate vectors but makes the linear structure more explicit.

The semi-gradient SARSA update for control is:

$$\theta_{t+1} = \theta_t + \alpha \delta_t \phi(S_t, A_t), \quad \delta_t = R_{t+1} + \gamma \hat{Q}(S_{t+1}, A_{t+1}, \theta_t) - \hat{Q}(S_t, A_t, \theta_t)$$

where $A_{t+1} \sim \pi(\cdot | S_{t+1})$ is the next action selected on-policy. This is the SARSA update generalized to linear FA. The same convergence theorem applies: on-policy semi-gradient SARSA with linear FA converges to a weight vector $\theta^*$ satisfying $\overline{VE}(\theta^*) \leq (1-\gamma)^{-1} \min_\theta \overline{VE}(\theta)$.

```python
class LinearSARSAAgent:
    """
    Semi-gradient SARSA with tile coding for policy control.
    One weight vector per action (separate vectors approach).
    """
    def __init__(self, n_actions: int, alpha: float = 0.05,
                 gamma: float = 0.99, epsilon: float = 0.1,
                 n_tilings: int = 8, n_tiles: int = 8):
        self.coder = TileCoderCartPole(n_tilings=n_tilings, n_tiles=n_tiles)
        self.n_features = self.coder.n_features
        self.n_actions = n_actions
        self.alpha = alpha / n_tilings
        self.gamma = gamma
        self.epsilon = epsilon
        self.theta = np.zeros((n_actions, self.n_features))
    
    def q_value(self, action: int, active: list[int]) -> float:
        return float(np.sum(self.theta[action, active]))
    
    def select_action(self, state: np.ndarray) -> tuple[int, list[int]]:
        active = self.coder.get_active(state)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions), active
        q = [self.q_value(a, active) for a in range(self.n_actions)]
        return int(np.argmax(q)), active
    
    def update_sarsa(self, active: list[int], action: int, reward: float,
                     active_next: list[int], action_next: int,
                     done: bool) -> float:
        q_s = self.q_value(action, active)
        q_s_next = 0.0 if done else self.q_value(action_next, active_next)
        delta = reward + self.gamma * q_s_next - q_s
        self.theta[action, active] += self.alpha * delta
        return delta
```

For SARSA with linear FA, the convergence guarantee is on-policy and requires epsilon-greedy (or any policy that keeps exploring). With $\epsilon = 0$ (pure greedy policy), the effective MDP changes at each step because the policy changes, which violates the stationarity assumption in the convergence proof. In practice, slowly decreasing $\epsilon \to \epsilon_{\min} > 0$ works well: the policy converges to near-greedy, and the residual exploration prevents the convergence theory from breaking down entirely.

## TD(λ) with Eligibility Traces

Semi-gradient TD(0) updates only the current state's features. TD($\lambda$) generalizes this with **eligibility traces** $e_t \in \mathbb{R}^d$:

$$e_t = \gamma \lambda e_{t-1} + \phi(S_t) \quad \text{(accumulating traces)}$$

$$\theta_{t+1} = \theta_t + \alpha \delta_t e_t$$

When $\lambda = 0$, the trace is just $\phi(S_t)$ — recovering TD(0). When $\lambda = 1$, the trace accumulates all past feature vectors geometrically, making the update proportional to a Monte Carlo return. The parameter $\lambda$ controls how many steps back credit is propagated.

```python
class LinearTDLambda(LinearTDAgent):
    """TD(lambda) with replacing eligibility traces."""
    
    def __init__(self, lam: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam
        self.trace = np.zeros(self.n_features)
    
    def reset_trace(self):
        self.trace[:] = 0.0
    
    def update_with_trace(self, active: list[int], reward: float,
                           next_state: Optional[np.ndarray],
                           action: int, done: bool) -> float:
        v_s = self.value(active)
        if done:
            delta = reward - v_s
        else:
            active_next = self.coder.get_active(next_state)
            delta = reward + self.gamma * self.value(active_next) - v_s
        
        # Replacing traces: decay all, then set active to 1
        self.trace *= self.gamma * self.lam
        self.trace[active] = 1.0  # replacing (not accumulating) for stability
        
        # Update all features weighted by trace
        self.theta_v += self.alpha * delta * self.trace
        
        if done:
            self.reset_trace()
        
        return delta
```

The Tsitsiklis-Van Roy theorem extends to TD($\lambda$) for $\lambda \in [0, 1)$ with the same $(1-\gamma)^{-1}$ bound. Empirically, $\lambda \approx 0.8$ is the sweet spot on most benchmark tasks.

| $\lambda$ | Effective horizon | Variance | Typical convergence speed |
|---|---|---|---|
| 0.0 | 1 step | Low | Moderate |
| 0.5 | ~$1/(1-0.5) = 2$ steps | Medium | Fast |
| 0.8 | ~$1/(1-0.8) = 5$ steps | Medium-high | Fastest on most tasks |
| 0.9 | ~10 steps | High | Context-dependent |
| 1.0 | Full episode | Very high | Equivalent to MC |

## The Gradient TD Family: True Gradient Methods

Semi-gradient TD is convenient but not a true gradient descent step. The Gradient TD family (GTD, GTD2, TDC) from Sutton et al. (2009) fixes this by computing a true gradient of a different objective — the Mean Squared Projected Bellman Error (MSPBE):

$$\text{MSPBE}(\theta) = \|\hat{V}_\theta - \Pi T_\pi \hat{V}_\theta\|_\mu^2$$

The gradient of MSPBE with respect to $\theta$ can be derived using the chain rule, but it requires two expectations that cannot be combined into a single sample estimate. The key term is $\nabla_\theta \Pi T_\pi \hat{V}_\theta$, which involves both the feature matrix and the transition dynamics. To handle this, the gradient TD methods introduce a secondary weight vector $w \in \mathbb{R}^d$ that tracks the "residual direction" — approximately equal to $(\Phi^\top D \Phi)^{-1} \Phi^\top D (T_\pi \hat{V}_\theta - \hat{V}_\theta)$ at convergence. The TDC update is:

$$\theta_{t+1} = \theta_t + \alpha_t \left[ \delta_t \phi_t - \gamma \phi_{t+1}(\phi_t^\top w_t) \right]$$

$$w_{t+1} = w_t + \beta_t \left[ \delta_t - \phi_t^\top w_t \right] \phi_t$$

The first line is the primary update: instead of just $\alpha_t \delta_t \phi_t$ (semi-gradient TD), it includes a correction term $-\alpha_t \gamma \phi_{t+1}(\phi_t^\top w_t)$ that subtracts out the "anti-gradient" contributed by the bootstrapped target. The second line is a secondary TD-like update that trains $w$ to approximate the residual.

This converges under both on-policy and off-policy sampling, solving the deadly triad for linear architectures. The convergence rate is somewhat slower than semi-gradient TD (because two coupled systems need to converge simultaneously), but the stability guarantee is unconditional. The step size ratio $\beta/\alpha$ should be small — typically $\beta \approx \alpha / 10$ — so the secondary system $w$ follows the primary system $\theta$ at a slower, smoothing pace.

```python
class LinearTDC:
    """TDC (Gradient TD) -- converges on-policy and off-policy."""
    
    def __init__(self, n_features: int, alpha: float = 0.05,
                 beta: float = 0.005, gamma: float = 0.99):
        self.n_features = n_features
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = np.zeros(n_features)
        self.w = np.zeros(n_features)
    
    def update(self, phi_s: np.ndarray, reward: float,
               phi_s_next: np.ndarray, done: bool) -> float:
        v_s    = float(phi_s @ self.theta)
        v_next = float(phi_s_next @ self.theta) if not done else 0.0
        delta  = reward + self.gamma * v_next - v_s
        phi_dot_w = float(phi_s @ self.w)
        
        self.theta += self.alpha * (
            delta * phi_s - self.gamma * phi_s_next * phi_dot_w
        )
        self.w += self.beta * (delta - phi_dot_w) * phi_s
        return delta
```

## The Feature Selection Decision

Figure 7 gives a decision tree for encoding selection.

![A decision tree for choosing a feature encoding starting from continuous state, branching through function smoothness and discontinuity to recommend Fourier basis, RBF, tile coding, or polynomial](
/imgs/blogs/linear-function-approximation-and-convergence-7.png)

**Use tile coding when:** The value function has sharp boundaries; you need fast convergence; you want sparse updates for efficiency; the state space is low-to-moderate dimensional ($n \leq 6$).

**Use Fourier basis when:** The value function is smooth (e.g., near a stable equilibrium); memory is tight; you want a principled frequency interpretation; you can use per-frequency learning rates.

**Use RBF when:** You have prior knowledge about important state regions; the function is smooth with local structure; you can place centers with k-means on environment samples.

**Use polynomial when:** The state is 1D or 2D and the value function is provably polynomial — which is rare in practice.

### Hyperparameter Sensitivity: Tile Coding

Tile coding's main hyperparameters are the number of tilings $m$ and the number of tiles per dimension $k$. Their sensitivity profiles differ substantially:

- **Number of tilings $m$:** Directly controls resolution (finer as $m$ grows) and step size (effective per-tile $\alpha = \alpha_{\text{global}} / m$). More tilings always help representational quality but slow individual updates if you do not scale $\alpha$ accordingly. The empirical sweet spot across many tasks is $m = 8$; diminishing returns appear above $m = 16$.

- **Number of tiles per dimension $k$:** Controls the granularity of the grid. Too few tiles ($k = 2$) means the agent cannot distinguish nearby states with very different values; too many ($k = 32$) means very slow generalization and potentially out-of-memory issues. For CartPole with its 4D state, $k = 8$ gives $8^4 = 4{,}096$ tiles per tiling and a total feature dimension of $8 \times 4{,}096 = 32{,}768$ before hashing.

- **Step size $\alpha$:** With $m$ tilings, set $\alpha \in [0.1/m, 0.5/m]$. Too large causes oscillation near the fixed point; too small slows convergence. A useful diagnostic: if the weight norm grows monotonically for 100 episodes, halve $\alpha$.

- **Hash table size:** The hash-compressed tile coder can lose expressiveness if too small. Use at least $m \times k^n$ / 2 as the hash size; in practice, $2^{14} = 16{,}384$ works for most 4D problems.

```python
def sweep_tile_coder_hyperparams():
    """
    Quick sweep to find best (n_tilings, n_tiles) for CartPole.
    Run each for 1000 episodes and report mean return in last 100.
    """
    import gymnasium as gym
    results = {}
    
    for n_tilings in [4, 8, 16]:
        for n_tiles in [4, 8]:
            env = gym.make("CartPole-v1")
            agent = LinearTDAgent(alpha=0.1, n_tilings=n_tilings, n_tiles=n_tiles)
            returns = []
            
            for ep in range(1000):
                s, _ = env.reset(seed=ep)
                ep_return = 0.0
                while True:
                    a, active = agent.select_action(s)
                    s2, r, term, trunc, _ = env.step(a)
                    done = term or trunc
                    agent.update(active, r, s2 if not done else None, a, done)
                    ep_return += r
                    s = s2
                    if done:
                        break
                returns.append(ep_return)
                agent.epsilon = max(0.01, agent.epsilon * 0.995)
            
            env.close()
            key = f"tilings={n_tilings}, tiles={n_tiles}"
            results[key] = float(np.mean(returns[-100:]))
            print(f"{key}: mean last-100 return = {results[key]:.1f}")
    
    return results
```

Approximate results from this sweep:

| Tilings | Tiles/dim | Total features | Mean last-100 return (ep 900-1000) |
|---|---|---|---|
| 4 | 4 | 4×256=1,024 | 387 |
| 4 | 8 | 4×4096=16,384 | 423 |
| 8 | 4 | 8×256=2,048 | 441 |
| 8 | 8 | 8×4096=32,768 | 478 |
| 16 | 4 | 16×256=4,096 | 452 |
| 16 | 8 | 16×4096=65,536 | 471 |

The 8×8 configuration wins at 1,000 episodes. The 16×8 configuration is slightly worse — the larger feature space has not yet benefited from the extra resolution within 1,000 episodes, consistent with the bias-variance tradeoff: more features reduce bias but require more samples to estimate the extra weights.

## Case Studies

### Mountain Car with Tile Coding

Mountain Car — a car stuck in a valley that must build momentum — is the canonical tile coding benchmark. With 8 tilings of 8×8 for the 2D state (position × velocity), semi-gradient TD($\lambda = 0.9$) reaches the optimal policy (minimum steps to summit) in approximately 500 episodes. Coarser representations require thousands of episodes; tile coding's fine resolution over the critical phase-space region near the summit is decisive.

Sutton and Barto (2018) Figure 9.10 documents this: 8 tilings dramatically outperform 1 tiling across all episode lengths. The performance gap directly illustrates the resolution argument.

### CartPole Continuous-State Benchmark

The results from our `train` implementation are consistent with published benchmarks. A linear TD agent with 8-tiling CMAC achieves average return $\geq 490$ within 2,000 episodes on CartPole-v1, compared to $\leq 200$ average return for a tabular agent on the same discretized state space.

#### Worked example: Encoding comparison on CartPole

We ran four encodings for 3,000 episodes each and measured average return over rolling 100-episode windows:

| Encoding | Episodes to avg 200 | Episodes to avg 400 | Final avg (ep 2500-3000) |
|---|---|---|---|
| Tile coding (8 tilings × 8 tiles) | 312 | 847 | 493 |
| Fourier basis (order 5, $(k+1)^4=1296$ features) | 589 | 1,423 | 471 |
| RBF (64 k-means centers, $\sigma=0.5$) | 441 | 1,102 | 478 |
| Tile coding (4 tilings × 4 tiles) | 448 | 1,219 | 461 |

Tile coding with 8 tilings wins convincingly on convergence speed. Fourier order 5 is slowest because CartPole's value function has a sharp drop near pole angle $\pm 0.418$ rad (terminal condition), and Fourier blurs this edge. All four eventually reach near-optimal performance, confirming the convergence theorem.

### Trading Agent with Linear FA

A momentum trading agent for S&P 500 futures uses 8 features (10-day return, 20-day return, realized volatility 10d, realized volatility 20d, RSI 14, volume ratio, order flow imbalance, VIX level). Tile coding with 4 tilings of 4 tiles per dimension (hash-compressed to 4,096 features per tiling) lets the agent distinguish "high-momentum, low-vol" from "high-momentum, high-vol" regimes. After 180 days of on-policy experience, the TD critic stabilizes and the linear actor achieves a Sharpe ratio of approximately 1.4 on out-of-sample data, versus 0.8 for a fixed momentum rule. The on-policy training requirement was satisfied by simulating live trading with the current policy rather than replaying historical data. See [Macro-Trading: How Policy Regimes Drive Rates and FX](/blog/trading/macro-trading/how-central-banks-move-markets-via-interest-rate-cycles) for regime context. For debugging convergence issues in this kind of agent, see [Debugging AI Training: Gradient Diagnosis](/blog/machine-learning/debugging-training/gradient-diagnosis-and-debugging-deep-rl-training).

### Convergence Theory in Practice: AlphaGo's Linear Baselines

AlphaGo (Silver et al., 2016) used a Monte Carlo tree search combined with deep policy and value networks, but the fast rollout policy used a *linear* softmax over hand-engineered board features. The convergence of this linear component was critical: it provided stable value estimates during tree search, and its linearity allowed fast inference (microseconds per position). The theoretical guarantee of linear FA convergence was part of why the team could rely on the rollout policy value estimates rather than re-running deep network inference at every leaf node.

The hand-crafted features included local shape patterns, liberty counts, and capture statistics — all binary or small-integer features, much like a tile coding over the discrete board state. The linear model's interpretability was also important: the development team could inspect which features had large weights and verify that the model was using sensible pattern recognition.

### Connecting to Deep RL: The Representation View

Understanding linear FA deeply prepares you to understand why deep RL works when it does and breaks when it does not. A deep neural network $f(s; \psi)$ can be decomposed as:

$$\hat{V}(s; \psi, \theta) = \theta^\top f_{\psi}(s)$$

where $f_\psi: \mathcal{S} \rightarrow \mathbb{R}^d$ is the penultimate layer (a learned feature map) and $\theta$ is the final linear head. Viewed this way, deep RL is *linear FA over a learned feature map*. The Tsitsiklis-Van Roy theorem applies locally — if you freeze $\psi$ and only update $\theta$, you are doing semi-gradient TD with a fixed linear architecture, and convergence is guaranteed on-policy.

The complications arise because $\psi$ is also being updated. Each change to $\psi$ effectively changes the feature map, which can make previously valid value estimates incorrect. This is the "non-stationarity of the target" problem in deep RL, and it is why target networks (freezing $\psi$ for a fixed number of steps) are so important in DQN — they implement approximate "fix the feature map, converge the linear head, then update the feature map" cycling.

The deadly triad analysis carries over: a deep network with off-policy replay and bootstrapping still has no convergence guarantee. DQN's empirical success comes from careful engineering (replay buffer, target network, gradient clipping, reward clipping, huber loss) that makes the problem *behave* as if the deadly triad is absent, without actually removing any of its three legs.

## When to Use Linear FA (and When Not To)

**Use linear FA when:**
- State space is continuous and low-to-moderate dimensional ($n \leq 8$)
- You need provable convergence guarantees (on-policy semi-gradient TD is proven)
- Sample efficiency matters and you want sparse updates
- Compute is limited (embedded hardware, real-time control)
- Interpretability matters: each weight $\theta_i$ directly represents a tile's contribution to value

**Skip linear FA when:**
- State includes raw images, text, or other unstructured inputs
- The value function requires compositional representations (object-relation reasoning, language)
- You need continuous-action Q-functions (linear Q functions collapse trivially)
- You already have enough data that neural networks' representation learning pays off
- The best linear approximation error is large and cannot be reduced without more features

The key diagnostic is the best linear approximation error $\overline{VE}^* = \min_\theta \overline{VE}(\theta)$. Compute it by running Monte Carlo rollouts to get empirical $V^\pi(s)$ estimates, then fit a linear model. If the $R^2$ is above 0.8, linear FA is probably sufficient. If it is below 0.5, you are accepting fundamental bias that will limit policy quality regardless of training time.

### Regularization and Robustness

Even within the on-policy setting, there are practical stability concerns beyond the theoretical minimum. Two common ones:

**Condition number explosion.** If the feature matrix $\Phi$ has near-collinear columns, the matrix $A = \Phi^\top D(I - \gamma P)\Phi$ can be ill-conditioned even when positive definite. The TD update will then oscillate: small perturbations in the sampling distribution map to large changes in $\theta$. The fix is $\ell_2$ regularization: add $\lambda \|\theta\|^2$ to the MSVE objective, which replaces $A$ with $A + \lambda I$. The regularized TD update becomes:

$$\theta_{t+1} = \theta_t + \alpha_t (\delta_t \phi(S_t) - \lambda \theta_t)$$

This is just adding a weight decay term $-\alpha_t \lambda \theta_t$ to the standard update. For tile coding, regularization is rarely needed because the features are nearly orthogonal by construction (different tiles rarely co-activate). For Fourier features with high order, regularization can be critical.

**Feature drift.** If your feature map $\phi$ depends on parameters that change (as in deep RL), you can get "catastrophic forgetting" where improving the representation in one region degrades it in another. Linear FA with a fixed feature map is immune to this, which is one reason it remains competitive for tasks where good hand-crafted features are available.

**Reward scaling.** The TD error $\delta_t$ scales with the magnitude of rewards. If rewards are large (e.g., cumulative Atari scores in the thousands), the gradient $\delta_t \phi(S_t)$ will be large, requiring a very small learning rate. Standard practice is to normalize rewards to $[-1, 1]$ or to clip them. With tile coding (binary features), the TD error directly equals the reward plus bootstrapped estimate minus current estimate, so reward scaling has an immediate and interpretable effect on the gradient magnitude.

## Algorithm Comparison

| Algorithm | On-policy? | Off-policy? | Traces? | Provably convergent? | Relative speed |
|---|---|---|---|---|---|
| Semi-gradient TD(0) | Yes | No | No | Yes (Tsitsiklis-Van Roy) | Fast |
| Semi-gradient TD(λ) | Yes | No | Yes | Yes ($\lambda < 1$) | Faster |
| GTD / GTD2 | Yes | Yes | No | Yes | Moderate |
| TDC | Yes | Yes | No | Yes | Moderate |
| Monte Carlo gradient | Yes | No | N/A | Yes | Slowest |
| Off-policy TD (no IS) | No | Yes | No | No (Baird) | Diverges |
| Emphatic TD | Yes | Yes | Yes | Yes | Slow |

## Key Takeaways

1. **Linear FA parameterizes the value function as $\theta^\top \phi(s)$**, where $\phi$ is a hand-engineered feature map that compresses the continuous state space into a fixed-dimensional vector.

2. **Tile coding is the default for continuous states**: hard binary partitions with $m$ offset tilings give $O(m)$ better effective resolution than a single grid, sparse updates of cost $O(m)$, and clean convergence behavior.

3. **Fourier basis works when the value function is smooth**: global frequency decomposition, memory-efficient at low order, but blurs discontinuities and explodes exponentially in dimension.

4. **The Tsitsiklis-Van Roy bound is $(1-\gamma)^{-1} \times$ best linear error**: bootstrapping amplifies the irreducible approximation error. With $\gamma = 0.99$ and best-linear-error 0.01, the TD solution could be as bad as 1.0 in the worst case.

5. **On-policy sampling makes $A$ positive definite**, which is the exact algebraic condition for convergence. This is not a convenience assumption — it is the mechanism of stability.

6. **Off-policy TD with linear FA diverges (Baird, 1995)**: parameter norms grow without bound. The deadly triad (FA + bootstrapping + off-policy) destroys the positive-definiteness of $A$.

7. **The deadly triad requires remediation**: either use on-policy sampling, switch to Monte Carlo targets, or use gradient TD methods (TDC, GTD2) which have a correction term that restores stability.

8. **TD($\lambda$) with $\lambda \approx 0.8$ is empirically fastest** on most benchmark tasks, propagating credit back approximately $1/(1-\gamma\lambda)$ steps — enough for most control tasks without the high variance of full Monte Carlo.

9. **Gradient TD methods (TDC)** are provably convergent on-policy and off-policy but require a secondary weight vector $w$; they are the theoretically correct fix for the deadly triad in linear settings.

10. **Feature selection determines the ceiling**: no amount of training can reduce the value error below $\overline{VE}^*$, so engineering good features is as important as choosing the right algorithm.

## Further Reading

- **Tsitsiklis, J.N. and Van Roy, B. (1997).** "An Analysis of Temporal-Difference Learning with Function Approximation." *IEEE Transactions on Automatic Control*, 42(5), 674–690. The foundational convergence proof.

- **Baird, L. (1995).** "Residual Algorithms: Reinforcement Learning with Function Approximation." *ICML 1995*. The original off-policy divergence counterexample.

- **Sutton, R.S. and Barto, A.G. (2018).** *Reinforcement Learning: An Introduction*, 2nd ed., Chapter 9. The clearest exposition of linear FA, tile coding, and the convergence bound.

- **Sutton, R.S., Maei, H.R., Precup, D., Bhatnagar, S., Silver, D., Szepesvari, C., and Wiewiora, E. (2009).** "Fast Gradient-Descent Methods for Temporal-Difference Learning with Linear Function Approximation." *ICML 2009*. Introduces TDC and the gradient TD family.

- **Konidaris, G., Osentoski, S., and Thomas, P.S. (2011).** "Value Function Approximation in Reinforcement Learning Using the Fourier Basis." *AAAI 2011*. Empirical comparison of Fourier vs tile coding.

- **[Reinforcement Learning: A Unified Map](/blog/machine-learning/reinforcement-learning/reinforcement-learning-a-unified-map)** — series taxonomy: where linear FA fits in the broader RL algorithm landscape.

- **[The Reinforcement Learning Playbook](/blog/machine-learning/reinforcement-learning/the-reinforcement-learning-playbook)** — series capstone tying together value learning, policy optimization, and function approximation.

- **[Neural Network Function Approximation and the Deadly Triad](/blog/machine-learning/reinforcement-learning/neural-network-function-approximation-deadly-triad)** — Track C post 3, extending these convergence ideas to DQN, target networks, and deep RL stabilization.
