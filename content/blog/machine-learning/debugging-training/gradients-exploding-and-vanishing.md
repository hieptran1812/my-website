---
title: "Gradients Exploding and Vanishing: Reading the Per-Layer Norm"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn to localize a vanishing or exploding gradient to the exact layer with one hook, derive why depth compounds the failure, and fix it with init, normalization, residuals, or clipping done right."
tags:
  [
    "debugging",
    "model-training",
    "gradient-norm",
    "vanishing-gradients",
    "exploding-gradients",
    "pytorch",
    "finetuning",
    "deep-learning",
    "rnn",
    "gradient-clipping",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/gradients-exploding-and-vanishing-1.png"
---

Here is a run that wasted three days of mine. A 24-block residual CNN, finetuning on a medical-imaging dataset, loss flat at 1.38 nats for two thousand steps. Not climbing, not `NaN`, just stuck. The learning rate was reasonable, the data looked clean, the augmentation was mild. I swapped optimizers, added warmup, removed warmup, lowered the LR, raised the LR. The curve did not care. Somewhere in those 24 blocks a gradient was dying on its way back from the loss, and the loss curve — one scalar summarizing 30 million parameters — had no way to tell me *where*. The same week, a colleague's transformer did the opposite: trained beautifully for 411 steps, then the loss leapt from 2.1 to 9.4 to `NaN` in three steps and the run was dead. Two runs, two opposite-looking disasters, one underlying cause: the gradient that travels backward through a deep network is a **product** of per-layer factors, and a product of many numbers is the most numerically unstable object in all of deep learning. It wants to be zero or it wants to be infinity. Keeping it near one is the entire art.

This post is about reading and fixing both ends of that failure. By the end you will be able to take any stalled or diverging run — a deep CNN, an RNN or LSTM, a Transformer, a deep tabular MLP — attach one hook, print the gradient norm of every named parameter, and within sixty seconds know whether you have a vanishing gradient (early layers reading $10^{-7}$ while later layers learn fine), an exploding gradient (a global norm of $10^4$ about to spike the loss), or something else entirely that is *masquerading* as a gradient problem. You will know why depth and recurrence make this happen, why activation choice and initialization set the regime before step one, why residual connections and normalization are not architectural fashion but the specific cure for a specific product, and — critically — why gradient clipping is a band-aid that masks an exploding gradient without curing the disease underneath.

![A two-panel before and after comparison showing a vanishing gradient regime where a per-layer factor below one drives the early-layer gradient to 1e-7 and an exploding regime where a factor above one drives the deep-layer gradient to 1e4 and a NaN, both arising from the same compounding product.](/imgs/blogs/gradients-exploding-and-vanishing-1.png)

This is one stop on the series' larger map. A training bug hides in one of six places — data, optimization, model code, numerics, systems, or evaluation — and a disciplined debugger **bisects** to the right one before touching code. Exploding and vanishing gradients live mostly in *optimization* and *model code*, with a foot in *numerics* (the `NaN` at the end of an explosion). The instrument that catches them is the per-layer gradient norm, which is exactly why this post is downstream of [instrumenting a training run: what to log](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log) and upstream of the whole [taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs). Let us derive the mechanism, build the diagnostic, and fix both runs.

## 1. The mechanism: a gradient is a product, and products explode

Start with the only equation that matters, and build everything else on it. A feedforward network is a composition of functions. Layer $\ell$ takes the activation $h_{\ell-1}$ from below, applies a linear map and a nonlinearity, and produces $h_\ell$:

$$
h_\ell = f_\ell(h_{\ell-1}), \qquad h_L = f_L(f_{L-1}(\cdots f_1(x))).
$$

The loss $\mathcal{L}$ is a function of the final activation $h_L$. To update a weight in an early layer — say layer 1 — the optimizer needs $\partial \mathcal{L} / \partial h_1$, and the chain rule says that gradient is a product of Jacobians, one per layer it passes through on the way down:

$$
\frac{\partial \mathcal{L}}{\partial h_1}
= \frac{\partial \mathcal{L}}{\partial h_L}
\cdot \frac{\partial h_L}{\partial h_{L-1}}
\cdot \frac{\partial h_{L-1}}{\partial h_{L-2}}
\cdots
\frac{\partial h_2}{\partial h_1}
= \frac{\partial \mathcal{L}}{\partial h_L} \prod_{\ell=2}^{L} J_\ell,
\qquad J_\ell \equiv \frac{\partial h_\ell}{\partial h_{\ell-1}}.
$$

Each $J_\ell$ is a matrix — the **Jacobian** of layer $\ell$, the local sensitivity of that layer's output to its input. The gradient that reaches layer 1 is the upstream gradient at the loss, multiplied by a long chain of these Jacobians. That is the whole story. Everything below is consequences.

Take the norm of both sides and use the submultiplicative property of the operator norm, $\|AB\| \le \|A\|\,\|B\|$:

$$
\left\| \frac{\partial \mathcal{L}}{\partial h_1} \right\|
\le \left\| \frac{\partial \mathcal{L}}{\partial h_L} \right\| \prod_{\ell=2}^{L} \|J_\ell\|.
$$

Now suppose, as a back-of-envelope model, that every layer's Jacobian has roughly the same norm $\|J_\ell\| \approx r$ — call $r$ the **per-layer gain**. Then the early-layer gradient scales as

$$
\left\| \frac{\partial \mathcal{L}}{\partial h_1} \right\| \sim r^{\,L-1}.
$$

This is the exponential that ruins deep training. If $r = 0.6$, then after 24 layers the gradient is $0.6^{23} \approx 8\times10^{-6}$ of what it was at the loss — the early layers receive an essentially zero gradient and **do not learn**. That is vanishing. If $r = 1.6$, then $1.6^{23} \approx 6\times10^{4}$ — the gradient that reaches the early layers (and, symmetrically, the parameter gradients accumulated along the way) is tens of thousands of times larger, the optimizer takes a giant step, the weights blow up, the next forward pass overflows, and the loss spikes to `NaN`. That is exploding. The knife-edge between them is $r = 1$. A product of $L$ numbers each slightly below one collapses; each slightly above one diverges; only $r$ exactly at one keeps the signal at a working magnitude across depth, and nothing in a naively-built network pins $r$ to one.

![A chain-rule dataflow graph showing the loss gradient of 1.0 multiplied by three successive layer Jacobian norms of 0.6, producing a layer-one gradient of 0.6 cubed equal to 0.22, with the product of norms branching off as the quantity that sets the vanishing or exploding regime.](/imgs/blogs/gradients-exploding-and-vanishing-2.png)

The figure above traces a tiny four-layer instance so the product is concrete: a loss gradient of $1.0$ passes up through three Jacobians of norm $0.6$ each, and the layer-1 gradient is $0.6^3 \approx 0.22$. Extend that to 24 layers and you reach $10^{-6}$. The shape of the failure is set by one number — the per-layer gain — and the depth it is raised to.

It is worth pausing on *why a product is so much more dangerous than a sum*, because this is the intuition that makes the rest of the post obvious. If gradient flow were additive — if each layer added a fixed amount to the gradient — then a deep network would simply have a larger or smaller gradient, linearly in depth, and you could always rescale it with the learning rate. Linear growth is benign. But composition makes the gradient *multiplicative* in depth, and multiplication of many factors is exponential, and exponentials are the most violent functions in numerical computing. A $10\%$ error in the per-layer gain — $r = 1.1$ instead of $r = 1.0$ — becomes a factor of $1.1^{50} \approx 117$ over fifty layers, and $r = 0.9$ becomes $0.9^{50} \approx 0.005$. The same tiny mis-estimate that would be invisible in a shallow net becomes a 100× or 200× distortion in a deep one. This is the precise sense in which *depth amplifies every numerical sin you commit at the per-layer level*, and it is why the cures below are all about controlling the per-layer factor rather than the global magnitude — you cannot fix an exponential by scaling its result; you have to fix its base.

There is a second, subtler consequence of the product form that catches people off guard: the gradient that reaches layer 1 and the gradient that reaches layer 20 are *different lengths of the same product*. Layer 20's gradient is the product of four Jacobians (layers 24 down to 21); layer 1's gradient is the product of twenty-three. So in a vanishing regime the *later* layers — with fewer factors in their product — see a healthy gradient and learn normally, while the *earlier* layers see the fully-decayed product and freeze. This is exactly why the vanishing signature is not "the whole network is dead" but "the early layers are dead and the late layers are fine," and it is the single most useful fact for reading a per-layer report: the *gradient at depth $d$ is the product truncated at $d$*, so the profile across depth is a direct readout of the cumulative product.

### Where the per-layer gain actually comes from

The gain $r = \|J_\ell\|$ is not a free parameter you set; it falls out of three concrete choices. Spelling them out is what turns "gradients vanish in deep nets" from folklore into something you can predict and fix.

For a standard layer $h_\ell = \phi(W_\ell h_{\ell-1})$ with weight matrix $W_\ell$ and elementwise nonlinearity $\phi$, the Jacobian is $J_\ell = D_\ell W_\ell$, where $D_\ell = \mathrm{diag}(\phi'(W_\ell h_{\ell-1}))$ is the diagonal matrix of the activation's derivatives at the current pre-activations. So the per-layer gain factors as

$$
\|J_\ell\| = \|D_\ell W_\ell\| \le \|D_\ell\| \cdot \|W_\ell\|.
$$

Two knobs: the **activation derivative** $\|D_\ell\|$ and the **weight-matrix norm** $\|W_\ell\|$ (and, lurking inside $\|W_\ell\|$, the initialization variance that sets it at step zero). This is the master decomposition for the whole post. Vanishing and exploding are not mysterious; they are what you get when $\|D_\ell\| \cdot \|W_\ell\|$ sits persistently below or above one across many layers.

One refinement on $\|W_\ell\|$ before we move on, because the choice of norm matters and people conflate two different quantities. The operator (spectral) norm $\|W_\ell\|_2$ is the *largest singular value* of $W_\ell$ — it bounds the worst-case amplification of any input direction. The product bound above uses this norm, which means it is genuinely a worst-case statement: the gradient *could* be amplified by $\prod \|W_\ell\|_2$ if the upstream gradient happens to align with the top singular vector of every layer. In practice the gradient is a random-ish direction and the *typical* amplification is closer to the average singular value (related to the Frobenius norm scaled by dimension), so the bound is loose. But the bound being loose is good news for vanishing and bad news for exploding: a single layer with a large top singular value can blow up the gradient even when every other layer is fine, because that one factor dominates the product. This is why a single mis-initialized or un-normalized layer in an otherwise healthy network can cause an explosion — the product is only as stable as its most amplifying factor. When you hunt an explosion, look for the *one* layer whose weight norm is an outlier, not for a uniform problem.

## 2. Activations: saturation is a gradient killer

Look at $\|D_\ell\|$ first, because activation choice is the most common reason a gradient vanishes, and it is the one beginners control without realizing it.

The sigmoid $\sigma(z) = 1/(1+e^{-z})$ has derivative $\sigma'(z) = \sigma(z)(1-\sigma(z))$, which is **at most $0.25$** (at $z=0$) and falls toward zero as $|z|$ grows — at $z = 6$, $\sigma'(6) \approx 0.0025$. So every sigmoid layer multiplies the backward signal by at most a quarter, and usually far less once the pre-activations drift away from zero. Stack ten sigmoid layers and the activation derivative alone contributes a factor of at most $0.25^{10} \approx 10^{-6}$ — before you even account for the weights. This is the original vanishing-gradient result that made deep sigmoid networks essentially untrainable, and it is why the field moved to ReLU.

`tanh` is better but not immune: $\tanh'(z) = 1 - \tanh^2(z)$ peaks at $1.0$ (at $z=0$), so a `tanh` layer *can* pass the gradient through undiminished — but only near the origin. The moment the pre-activations **saturate** (drift to where $|\tanh(z)| \approx 1$), the derivative collapses to near zero and that layer becomes a gradient wall. The word to internalize is *saturation*: a saturated unit is one whose input has pushed it into the flat part of its activation, where the local derivative is near zero and gradient flows through it as a trickle.

ReLU, $\phi(z) = \max(0, z)$, has derivative exactly $1$ for $z > 0$ and exactly $0$ for $z < 0$. For the units that are *on*, the activation contributes a gain of exactly one — no shrinkage — which is precisely why ReLU made deep networks trainable. But the units that are *off* contribute exactly zero, and a unit that gets pushed permanently into $z<0$ (a "dead ReLU") passes no gradient forever. So ReLU trades the everywhere-small derivative of sigmoid for an all-or-nothing one; the vanishing risk moves from "every unit trickles" to "a fraction of units are dead." (Leaky ReLU and GELU soften the dead branch, which is one reason GELU is the default in Transformers.)

The "fraction dead" framing is worth making quantitative, because it explains why ReLU networks can *still* vanish in a way that surprises people. If, in a given layer, a fraction $p$ of units are active (positive pre-activation), then the layer's expected gradient gain is not $1$ but roughly $p$ — only the active units pass gradient, so on average the backward signal is attenuated by the active fraction. With a healthy $p \approx 0.5$ this is a per-layer factor of $\approx 0.5$, which over 30 layers is $0.5^{30} \approx 10^{-9}$ *unless* something else (normalization, residuals) re-pins the signal. This is the subtle reason that ReLU alone did not make arbitrarily deep nets trainable — it made them *less* prone to the everywhere-small sigmoid vanish, but a deep plain-ReLU stack still decays as the active fraction compounds, which is exactly why the deep-net breakthrough required ReLU *and* good init *and* (for the very deep regime) residuals. Leaky ReLU keeps a small slope ($\approx 0.01$) on the negative branch so the dead units pass a trickle instead of nothing, and GELU is smooth everywhere so there is no hard dead region at all — both raise the effective active fraction and reduce the compounding attenuation, which is part of why they help in deep networks.

To see the chain compound, trace a concrete five-layer sigmoid stack by hand. Suppose each layer's pre-activations sit where $\sigma' \approx 0.2$ on average and each weight matrix has norm $\approx 1.0$. Then the per-layer gain is $\approx 0.2$, and the gradient reaching the input layer is scaled by $0.2^5 = 3.2\times10^{-4}$ relative to the loss gradient. Five layers already cost you three-and-a-half orders of magnitude. Swap to ReLU with active fraction $0.5$ and weight norm $1.0$, and the per-layer gain rises to $\approx 0.5$, so the same five-layer stack scales by $0.5^5 = 0.031$ — an order of magnitude better, but still a $30\times$ attenuation that will hurt at depth 30. Now add BatchNorm (which re-pins each layer's variance to $\approx 1$, holding $\|W_\ell\|$-driven drift in check) and a residual skip (which adds the identity-gain-one path), and the effective per-layer gain returns to $\approx 1$ and the five- or fifty-layer stack passes the gradient essentially intact. That arithmetic — $0.2^5$ versus $0.5^5$ versus $\approx 1^5$ — *is* the history of deep learning's activation and architecture choices, compressed into one example.

Here is a diagnostic you can run in ten seconds to see saturation directly — it forward-passes a batch and reports, per layer, the fraction of units sitting in the dead/saturated region. A high saturation fraction in an early layer is your smoking gun for activation-driven vanishing.

```python
import torch

@torch.no_grad()
def saturation_report(model, batch, kinds=("Sigmoid", "Tanh", "ReLU", "GELU")):
    """Forward one batch; report the fraction of saturated/dead units per activation."""
    stats = {}
    handles = []

    def make_hook(name, cls_name):
        def hook(module, inp, out):
            x = out.detach()
            if cls_name in ("Sigmoid", "Tanh"):
                # saturated = output within 0.02 of either flat asymptote
                if cls_name == "Sigmoid":
                    sat = ((x < 0.02) | (x > 0.98)).float().mean()
                else:  # Tanh
                    sat = (x.abs() > 0.98).float().mean()
            else:  # ReLU / GELU: "dead" = output non-positive
                sat = (x <= 0).float().mean()
            stats[name] = sat.item()
        return hook

    for name, m in model.named_modules():
        if m.__class__.__name__ in kinds:
            handles.append(m.register_forward_hook(make_hook(name, m.__class__.__name__)))

    model.eval()
    model(batch)
    for h in handles:
        h.remove()

    for name, frac in stats.items():
        flag = "  <-- SATURATED" if frac > 0.5 else ""
        print(f"{name:40s} saturated/dead frac = {frac:6.3f}{flag}")
    return stats
```

If layer 2 shows `saturated/dead frac = 0.94` while layer 20 shows `0.31`, you have found a vanishing gradient's *cause* (early saturation) without even running the backward pass. The fix is rarely "change the activation in isolation" — it is to fix what *drives* the saturation, which is almost always initialization variance (next section) or a missing normalization layer (section 5).

## 3. Initialization: the variance that sets the regime at step zero

The weight-norm factor $\|W_\ell\|$ is set, before any training, by initialization. Get the variance wrong and the network is born in the vanishing or exploding regime; warmup and a good optimizer can sometimes paper over it, but you are fighting the architecture from step one.

The argument is a variance-propagation one. Consider a linear layer $z = Wx$ with $W \in \mathbb{R}^{n_\text{out} \times n_\text{in}}$, where the weights are drawn i.i.d. with mean zero and variance $\mathrm{Var}(W_{ij}) = s^2$, and the inputs $x_j$ are i.i.d. with variance $\mathrm{Var}(x)$. Then each output coordinate is a sum of $n_\text{in}$ independent products, so

$$
\mathrm{Var}(z_i) = n_\text{in} \cdot s^2 \cdot \mathrm{Var}(x).
$$

For the activation variance to be **preserved** layer to layer (neither growing nor shrinking), you need the prefactor $n_\text{in} \cdot s^2 = 1$, i.e. $s^2 = 1/n_\text{in}$. That is **Xavier/Glorot** initialization (in its fan-in form). For ReLU, half the units are zeroed on average, which halves the variance, so you compensate with $s^2 = 2/n_\text{in}$ — that is **Kaiming/He** initialization. The factor-of-two is not a detail; using Xavier variance with ReLU activations shrinks the signal by $\sqrt{1/2}$ per layer, which over 30 layers is $0.5^{15} \approx 3\times10^{-5}$ — a slow, init-driven vanish that looks exactly like a learning-rate problem and wastes days if you do not know to check.

The same propagation runs backward for gradients, and the symmetric condition $n_\text{out} \cdot s^2 = 1$ is what Glorot's compromise $s^2 = 2/(n_\text{in} + n_\text{out})$ tries to satisfy in both directions at once. The practical upshot is a short, testable rule: **the per-layer activation variance should stay roughly constant with depth.** If you forward a batch and the activation standard deviation drops from $1.0$ at the input to $0.01$ by layer 20, your init is shrinking the signal and your gradients will vanish; if it climbs to $50$, your init is amplifying and your gradients will explode.

```python
import torch

@torch.no_grad()
def activation_variance_by_depth(model, batch):
    """Forward one batch; print the std of each layer's output. Healthy = roughly flat with depth."""
    stds = []
    handles = []

    def hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            stds.append((module.__class__.__name__, out.detach().float().std().item()))

    for m in model.modules():
        if len(list(m.children())) == 0:  # leaf modules only
            handles.append(m.register_forward_hook(hook))

    model.eval()
    model(batch)
    for h in handles:
        h.remove()

    for i, (cls, s) in enumerate(stds):
        trend = "" if 0.1 < s < 10 else "  <-- off-scale"
        print(f"layer {i:2d} {cls:18s} act std = {s:10.4f}{trend}")
    return stds
```

#### Worked example: an init bug that looks like a learning-rate bug

A 30-layer fully-connected MLP for a deep tabular task trains at chance. The engineer assumes the LR is too low and sweeps it from $10^{-4}$ up to $10^{-1}$; every value plateaus. Running `activation_variance_by_depth` reveals the actual problem: the layers use ReLU but were initialized with PyTorch's default (which for `nn.Linear` is a uniform Kaiming variant tuned for `leaky_relu` with `a=sqrt(5)`, effectively a smaller gain). The activation std reads $1.0$ at layer 0, $0.42$ at layer 10, $0.07$ at layer 20, $0.009$ at layer 29 — a clean geometric decay of roughly $0.85$ per layer. The gradient that reaches layer 0 is therefore down by $0.85^{29} \approx 0.009$, and no learning rate can rescue a signal that small without exploding the late layers that already see a healthy gradient. The fix is a one-liner — re-initialize with the correct gain:

```python
import torch.nn as nn

for m in model.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
```

After re-init, the activation std reads $1.0$, $0.91$, $0.88$, $0.85$ across the same depths — essentially flat — and the same learning rate that plateaued for two thousand steps now drives the loss down on the first hundred. No LR sweep would have found this; the variance-by-depth probe found it in one forward pass. This is the recurring lesson of the series: the loss told you *that* it was broken; the instrument told you *where*.

## 4. The diagnostic: per-layer gradient norm, the single best tool

Activations and init explain the regime, but when a run is already misbehaving you do not want to reason forward from architecture — you want to **measure** the gradient and let it tell you where the failure is. The single most valuable diagnostic in this entire post is the **per-layer gradient norm**: after `loss.backward()`, loop over `model.named_parameters()` and print `p.grad.norm()` for each one. It is five lines, it is universal across CNNs/RNNs/Transformers/MLPs, and it converts an invisible product into a visible profile.

```python
import torch

def grad_norm_report(model, top_k=None, flag_low=1e-5, flag_high=1e2):
    """Call AFTER loss.backward(), BEFORE optimizer.step(). Print grad norm per parameter."""
    rows = []
    for name, p in model.named_parameters():
        if p.grad is None:
            rows.append((name, None))           # no grad reached this param at all
        else:
            rows.append((name, p.grad.detach().norm().item()))

    # global norm = L2 over all grads concatenated (what clip_grad_norm_ computes)
    total = torch.sqrt(sum((p.grad.detach() ** 2).sum()
                           for _, p in model.named_parameters() if p.grad is not None))
    print(f"GLOBAL grad norm = {total.item():.4e}")

    rows.sort(key=lambda r: (r[1] is not None, r[1] if r[1] is not None else 0.0))
    shown = rows if top_k is None else rows[:top_k] + rows[-top_k:]
    for name, g in shown:
        if g is None:
            flag = "  <-- NO GRAD (detached / frozen?)"
        elif g < flag_low:
            flag = "  <-- VANISHING"
        elif g > flag_high:
            flag = "  <-- EXPLODING"
        else:
            flag = ""
        gstr = "None" if g is None else f"{g:.3e}"
        print(f"{name:50s} grad={gstr}{flag}")
    return rows
```

There is a second way to capture per-layer gradients that is worth knowing because it works *during* the backward pass rather than after it, and it catches the gradient at the activation (not just the parameter) level. PyTorch's `tensor.register_hook` attaches a callback to a tensor that fires when its gradient is computed, and `module.register_full_backward_hook` does the same at the module boundary. This lets you log the gradient *flowing between layers* — the $\partial \mathcal{L}/\partial h_\ell$ quantities from section 1 — which is the most direct possible view of the product as it travels:

```python
import torch

def attach_activation_grad_hooks(model):
    """Log the gradient flowing OUT of each module's output (the inter-layer signal)."""
    norms = {}

    def make_hook(name):
        def hook(module, grad_input, grad_output):
            g = grad_output[0]
            if g is not None:
                norms[name] = g.detach().norm().item()
        return hook

    handles = [m.register_full_backward_hook(make_hook(name))
               for name, m in model.named_modules()
               if len(list(m.children())) == 0]
    return norms, handles
```

After a backward pass, `norms` holds the gradient magnitude flowing out of every leaf module, ordered by depth. Plotting these against depth is the cleanest possible vanishing/exploding picture — a monotone ramp down is vanishing, a ramp up is exploding, flat is healthy — because it shows the *inter-layer* gradient (the truncated product itself) rather than the parameter gradients, which mix in the local activation. Use the parameter-norm report for "which weights are frozen" and the activation-gradient hook for "where in the chain the signal dies."

Read either report and three things jump out immediately. First, any parameter whose `grad` is `None` never received a gradient at all — it is detached from the graph, frozen by accident, or behind an in-place op that broke the backward pass (that is a *model-code* bug, covered in the model-debugging track, not a vanishing-gradient bug — do not confuse them). Second, a parameter with `grad` of $10^{-7}$ in an *early* layer while later layers read $10^{-1}$ is the classic vanishing signature: the gradient is shrinking as it travels back. Third, a global norm of $10^4$, or any single layer reading $10^3$+, is the exploding signature — and it will usually be the *latest* layers (closest to the loss) that read largest, because in the exploding regime the product grows as it accumulates downstream.

![A three-by-three matrix mapping early, middle, and late layers across vanishing, healthy, and exploding columns, showing the vanishing run with 1e-7 in the early layers, the healthy run with all norms near one, and the exploding run with 1e4 in the late layers.](/imgs/blogs/gradients-exploding-and-vanishing-3.png)

The matrix above is the read you are building toward: *where in depth the extreme number sits* tells vanishing from exploding. Vanishing puts the tiny numbers in the early layers (top-left); exploding puts the huge numbers in the late layers (bottom-right); a healthy run has all norms in the same order of magnitude, drifting toward $1$–$2$ as it stabilizes. You do not need the absolute values to be any particular figure — you need the *profile across depth* to be flat. A profile that ramps down with depth is vanishing; one that ramps up is exploding.

### Logging it continuously, not just once

Printing once is a snapshot; what you really want is the **trend**. Log the global grad norm every step and the per-layer norms every $N$ steps, into W&B or TensorBoard, so you can watch the profile evolve. An exploding gradient almost never appears from nowhere — the global norm *climbs* for dozens or hundreds of steps before it spikes, and if you are logging it you get a warning well before the `NaN`. Here is the cheap continuous version with a hook so you do not rewrite the loop:

```python
import torch

class GradNormLogger:
    """Logs global grad norm every step, per-layer norms every `every` steps."""
    def __init__(self, model, every=50):
        self.model = model
        self.every = every
        self.step = 0

    def log(self, logger=None):
        with torch.no_grad():
            sq = 0.0
            per_layer = {}
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                n = p.grad.norm().item()
                sq += n * n
                if self.step % self.every == 0:
                    per_layer[f"gradnorm/{name}"] = n
            global_norm = sq ** 0.5
        payload = {"gradnorm/global": global_norm, **per_layer}
        if logger is not None:        # e.g. wandb.log(payload, step=self.step)
            logger(payload)
        self.step += 1
        return global_norm
```

The cost is one `.norm()` per parameter per step, which is a few microseconds on a model whose forward pass is milliseconds — negligible. Wiring this in is the cheapest insurance you will ever buy, and it is the same global-norm signal discussed at length in [instrumenting a training run: what to log](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log); here we are using it specifically to localize the gradient failure by depth.

### The update-to-weight ratio: a scale-free read

There is a subtlety the raw grad norm hides: a grad norm of $0.5$ is "small" for a parameter whose weights have norm $0.1$ and "tiny" for one whose weights have norm $100$. What actually determines whether a parameter *moves* is the size of the update relative to the weight it modifies. The scale-free version of the diagnostic is the **update-to-weight ratio**: the norm of the optimizer's step divided by the norm of the parameter, $\eta \|g\| / \|w\|$ for plain SGD (and the optimizer-specific update norm for Adam). A healthy ratio sits around $10^{-3}$ — each step nudges a weight by about a tenth of a percent. A ratio of $10^{-7}$ means that parameter is effectively frozen (it would take ten million steps to change meaningfully), which is the vanishing signature expressed in the units that matter; a ratio of $10^{-1}$ or higher means the parameter is being thrown around violently, the exploding signature. Logging this ratio per layer is often *clearer* than the raw grad norm because it normalizes out the wildly different weight scales across a network:

```python
import torch

@torch.no_grad()
def update_to_weight_ratio(model, lr):
    """Approximate per-layer step/weight ratio for SGD. Healthy ~ 1e-3; vanishing << 1e-5."""
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        w_norm = p.detach().norm().item() + 1e-12
        update_norm = lr * p.grad.detach().norm().item()
        ratio = update_norm / w_norm
        flag = ""
        if ratio < 1e-6:
            flag = "  <-- frozen (vanishing)"
        elif ratio > 1e-1:
            flag = "  <-- thrown around (exploding)"
        print(f"{name:50s} step/weight = {ratio:.2e}{flag}")
```

For Adam the story shifts in an instructive way. Adam divides the gradient by (the square root of) a running estimate of its own second moment, which means Adam *normalizes away* the raw gradient magnitude — a parameter receiving a tiny but *consistent* gradient still gets a unit-scale update, because Adam's denominator shrinks to match. This is why Adam is more forgiving of mild vanishing than SGD: it can rescue a layer whose gradient is small-but-nonzero. But it does **not** rescue a *truly* vanished gradient, because once the gradient underflows to exactly zero (or to the floor of fp16, around $6\times10^{-5}$), Adam's numerator is zero too and no amount of denominator scaling produces a step. The practical reading: if your grad norms are small but nonzero and Adam is still not learning, suspect something other than vanishing; if they are at or below the fp16 floor, Adam cannot help and you have a real vanish (or an underflow — covered in the mixed-precision track) to fix at the source.

### Reading the gradient histogram, not just the norm

The norm collapses a whole tensor of gradients to one number, which is enough to localize but not always enough to diagnose. The next level of resolution is the **per-layer gradient histogram** — log `p.grad` as a histogram in TensorBoard or W&B and look at the *shape* of the distribution. A healthy gradient histogram is a roughly symmetric bump centered near zero with a spread of order $10^{-2}$ to $10^{0}$. Three pathological shapes each name a different bug. A histogram **collapsed to a spike at exactly zero** is a vanished (or dead-ReLU) layer — the gradients are not small, they are *gone*. A histogram with a **fat tail reaching $10^3$+** is an exploding layer caught in the act, often before the global norm has spiked, because a few extreme gradients can hide inside an otherwise-reasonable norm. And a histogram **clipped flat at the fp16 floor** (a wall of mass at $\approx 6\times10^{-5}$ with nothing below) is a mixed-precision underflow masquerading as vanishing — the gradients did not biologically vanish, they fell below the smallest representable fp16 magnitude and were rounded to zero, which is a numerics fix (loss scaling or bf16), not an architecture fix. Logging the histogram is what lets you tell "the architecture is shrinking the gradient" from "fp16 is truncating it," and those have completely different cures.

## 5. Why residuals and normalization keep the signal O(1)

If the disease is a product of factors that drifts away from one, the cure is anything that pins the per-layer factor *back* to one. Two architectural ideas do exactly that, and understanding *why* turns them from "things you add because the paper did" into deliberate instruments.

**Residual connections.** A residual block computes $h_\ell = h_{\ell-1} + F_\ell(h_{\ell-1})$ instead of $h_\ell = F_\ell(h_{\ell-1})$. Differentiate:

$$
J_\ell = \frac{\partial h_\ell}{\partial h_{\ell-1}} = I + \frac{\partial F_\ell}{\partial h_{\ell-1}}.
$$

The Jacobian is now **identity plus** the residual branch's Jacobian. The product down the network becomes $\prod_\ell (I + F'_\ell)$, and even if every $F'_\ell$ is small or badly conditioned, the identity term guarantees a gradient highway: $\partial \mathcal{L}/\partial h_1$ always contains a copy of $\partial \mathcal{L}/\partial h_L$ that passed through *only* identity maps, undiminished. The gradient cannot vanish to zero because there is always a path of gain exactly one. This is the actual reason ResNets train at 152 layers when plain nets stall past 20, and the same reason every Transformer block wraps attention and MLP in residual connections. It is not depth tolerance by magic; it is $I + F'$ keeping the per-layer factor near one by construction.

**Normalization.** BatchNorm, LayerNorm, and RMSNorm each rescale activations to a controlled statistic (roughly unit variance) at the point they are inserted. By forcing $\mathrm{Var}(h_\ell) \approx 1$ regardless of what the weights did, normalization stops the variance-propagation drift from section 3 from compounding — it *re-pins* the signal to $O(1)$ at every normalized layer, so the per-layer gain cannot run away across depth. There is a real cost and a real trap here: BatchNorm's behavior depends on batch statistics and differs between train and eval, and at small batch sizes its running-statistic estimates are noisy enough to *cause* instability rather than cure it. That train/eval and small-batch behavior is its own debugging topic, covered in [initialization and normalization bugs](/blog/machine-learning/debugging-training/initialization-and-normalization-bugs); for our purposes the key fact is that normalization is one of the three levers (with init and residuals) that keep $r \approx 1$.

The choice among normalization variants matters for *which* gradient problem you are solving, so a quick map: **BatchNorm** normalizes across the batch dimension and is the workhorse of CNNs, but its train/eval mismatch and small-batch fragility make it a poor fit for sequence models and tiny batches. **LayerNorm** normalizes across the feature dimension of a single example, has *no* batch dependence (so train and eval behave identically and batch size is irrelevant), and is why it dominates Transformers and RNNs. **RMSNorm** is LayerNorm without the mean-subtraction step — it only rescales by the root-mean-square — which is cheaper and, empirically, just as effective at keeping the gradient $O(1)$, which is why many recent LLMs adopt it. For gradient stability the relevant property is shared by all three: they bound the activation scale at the point of insertion, so the per-layer gain cannot drift far from one regardless of what the weights are doing. The *placement* relative to the residual matters too — putting the norm *inside* the residual branch (Pre-LN: $h_\ell = h_{\ell-1} + F_\ell(\mathrm{norm}(h_{\ell-1}))$) leaves the identity highway un-normalized and gives a cleaner gradient flow than normalizing the whole sum (Post-LN), which is the reason most deep Transformers switched to Pre-LN to train stably at depth.

![A vertical stack contrasting a plain block with Jacobian norm 0.6 that collapses to 5e-6 over 24 layers against a residual block whose identity path keeps the factor near one and the gradient between 0.5 and 2 across the same depth, with normalization rescaling to order one.](/imgs/blogs/gradients-exploding-and-vanishing-4.png)

The stack above contrasts the two regimes directly: a plain block with gain $0.6$ collapses to $5\times10^{-6}$ over 24 layers, while the residual block's $I + F'$ keeps the gradient in the $0.5$–$2$ band across the same depth, with the norm layers rescaling back to $O(1)$ at each step. Init sets the starting point, residuals provide the highway, normalization re-pins the variance — three levers, one goal: keep the product near one.

#### Worked example: fixing the stalled 24-block CNN

Return to the run from the intro — the 24-block CNN flat at $1.38$ nats. The `grad_norm_report` is unambiguous: block 2 reads `grad=8.3e-08  <-- VANISHING`, block 6 reads `grad=4.1e-05`, block 20 reads `grad=1.2e-01`, and the final classifier reads `grad=3.4e-01`. The profile ramps *up* with depth by six orders of magnitude — a textbook vanishing signature. The cause, found with `activation_variance_by_depth`, was a stack of plain (non-residual) conv blocks with `tanh` activations and no normalization; the `tanh` units in the early blocks were saturated (`saturation_report` showed `frac=0.91` at block 2), and the variance decayed geometrically. Three changes — swap `tanh` for ReLU, add BatchNorm after each conv, and add the residual skip that the architecture was missing — and the per-layer report transforms: block 2 now reads `grad=4.0e-01`, the profile is flat from block 2 to block 24, and the loss that sat at $1.38$ for two thousand steps falls to $0.51$ within four hundred. Validation accuracy goes from a stuck $41\%$ to $88\%$ and still climbing. The fix touched the model, not the optimizer — which is exactly what the gradient profile predicted.

![A before and after comparison showing a vanishing run with sigmoid and naive init giving a layer-two gradient of 1e-7 and a 41 percent plateau, transformed by ReLU plus Kaiming init plus normalization into a layer-two gradient of 0.4 and 88 percent accuracy climbing.](/imgs/blogs/gradients-exploding-and-vanishing-6.png)

## 6. Recurrence: why RNNs explode and vanish in time, not just depth

Everything so far was about depth. Recurrent networks have the same disease but along the **time** axis, and it is sharper there because the *same* weight matrix is reused at every timestep. An RNN computes $h_t = \phi(W_{hh} h_{t-1} + W_{xh} x_t)$, and backpropagation-through-time (BPTT) unrolls the sequence into a deep network whose "layers" all share $W_{hh}$. The gradient of the loss at time $T$ with respect to the hidden state at time $t$ is

$$
\frac{\partial \mathcal{L}_T}{\partial h_t}
= \frac{\partial \mathcal{L}_T}{\partial h_T} \prod_{k=t+1}^{T} \frac{\partial h_k}{\partial h_{k-1}}
= \frac{\partial \mathcal{L}_T}{\partial h_T} \prod_{k=t+1}^{T} D_k W_{hh}^{\top},
$$

where $D_k = \mathrm{diag}(\phi'(\cdot))$ again. Because the *same* $W_{hh}$ appears $T - t$ times, the product is governed by the **spectral radius** $\rho(W_{hh})$ — the largest absolute eigenvalue. If $\rho(W_{hh}) < 1$, the gradient over a long time gap decays like $\rho^{T-t}$ and **vanishes**: the network cannot learn dependencies more than a few dozen steps apart, which is exactly why vanilla RNNs forget long context. If $\rho(W_{hh}) > 1$, the gradient grows like $\rho^{T-t}$ and **explodes**: the loss spikes when a long sequence comes through. The knife-edge is $\rho = 1$, and because a single matrix is reused, there is no per-layer init trick that fixes it — the spectral radius of one matrix sets the fate of the whole unrolled product.

To make the spectral-radius argument concrete, decompose $W_{hh} = Q \Lambda Q^{-1}$ in its eigenbasis (assume it is diagonalizable for the sketch). Then $W_{hh}^{\,k} = Q \Lambda^k Q^{-1}$, and the BPTT product over $k = T - t$ steps is dominated by $\Lambda^k$ — a diagonal matrix whose entries are the eigenvalues raised to the $k$-th power. The largest-magnitude eigenvalue $\lambda_{\max}$ (whose magnitude is the spectral radius $\rho$) wins the product as $k$ grows: $\|\Lambda^k\| \approx |\lambda_{\max}|^k = \rho^k$. Every other eigendirection decays relative to it. So over a long time gap, the gradient is, to first order, scaled by $\rho^{T-t}$ — and there it is, the same exponential, now with the spectral radius as the base and the *time gap* as the exponent. The activation derivatives $D_k$ only make it worse (they are $\le 1$ for `tanh`, so they pull $\rho$ effectively below one and *accelerate* vanishing). A vanilla `tanh` RNN therefore vanishes for essentially any sequence longer than $\approx 1/(1 - \rho)$ steps, which for a typical $\rho \approx 0.9$ is on the order of ten timesteps — the empirically-observed horizon where vanilla RNNs stop learning long-range structure.

This is the precise reason LSTMs and GRUs exist. The LSTM's cell state $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ has a gradient path $\partial c_t / \partial c_{t-1} = f_t$ (the forget gate) that is **additive and gated** rather than a repeated matrix multiply. When the forget gate $f_t \approx 1$, the cell-state gradient passes through time undiminished — the same "highway" idea as a residual connection, but along time. That is why an LSTM learns dependencies across hundreds of steps where a vanilla RNN vanishes by step thirty. The mechanism is identical to residuals in depth: replace a repeated multiply with a near-identity gated path so the product cannot collapse. The catch worth flagging for debugging: the forget gate is *learned*, and if it initializes near zero (or learns toward zero), the LSTM loses its highway and starts to vanish like a vanilla RNN — which is why a common, effective trick is to **initialize the forget-gate bias to a positive value** (often $+1$ or higher) so the gate starts near one and the gradient highway is open from step zero. If an LSTM is failing to learn long-range structure, checking and bumping the forget-gate bias is a one-line experiment worth running before anything more elaborate.

You can measure the spectral radius directly to confirm the diagnosis, which turns "I think the RNN is in the vanishing regime" into a number:

```python
import torch

@torch.no_grad()
def spectral_radius(weight_hh):
    """Largest |eigenvalue| of the recurrent weight. <1 vanishes, >1 explodes, ~1 is the edge."""
    eigvals = torch.linalg.eigvals(weight_hh)        # complex eigenvalues
    rho = eigvals.abs().max().item()
    regime = "VANISHING" if rho < 0.95 else "EXPLODING" if rho > 1.05 else "edge ~1"
    print(f"spectral radius rho = {rho:.3f}  -> {regime}")
    return rho

# for nn.RNN/nn.LSTM the recurrent weight is weight_hh_l0
# rho = spectral_radius(rnn.weight_hh_l0)
```

For exploding RNN gradients specifically, **gradient clipping** is the standard and genuinely appropriate tool — the original "On the difficulty of training recurrent neural networks" (Pascanu, Mikolov, Bengio, 2013) introduced norm clipping precisely for this. We will get clipping right in the next section, but note the asymmetry now: clipping cures the *exploding* end (it caps the step size so one long sequence cannot blow up the weights) but does nothing for the *vanishing* end (you cannot un-shrink a gradient that has already collapsed to $10^{-9}$ by scaling it). Vanishing in RNNs is fixed by architecture (LSTM/GRU gates), not by clipping.

#### Worked example: an LSTM that spikes on long sequences

A character-level LSTM trains fine on sequences of length 50 but the loss spikes to `NaN` whenever the batch happens to contain a sequence near the max length of 400. The per-step global grad-norm log tells the story instantly: on short-sequence batches the norm sits at $3$–$8$, but on the long-sequence batch it jumps to `grad norm = 2.7e+04`. The BPTT product over 400 steps amplified a spectral radius slightly above one into a five-order-of-magnitude blowup. The confirming test is a one-liner — add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()` — and the spike vanishes: the long-sequence batches now read a capped `grad norm = 1.0` and the loss descends smoothly past the point it used to die. Here clipping is the *cure*, not a band-aid, because the explosion is intrinsic to BPTT over long sequences and not a symptom of a too-high LR. (How to tell those two cases apart is exactly section 7.)

## 7. Clipping done right — and why it is usually a band-aid

Gradient clipping rescales the gradient when its global norm exceeds a threshold, so the optimizer never takes a step larger than you allow. The correct call is `clip_grad_norm_`, which clips by the **global** L2 norm (preserving the gradient's direction), not `clip_grad_value_`, which clips each element independently and distorts the direction:

```python
import torch

loss.backward()

# CORRECT: clip by global L2 norm, preserves direction. Returns the PRE-clip norm.
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Log the PRE-clip norm — this is your exploding-gradient early-warning signal.
# If total_norm is routinely >> max_norm, clipping is firing constantly: a red flag.
if total_norm > 10.0:
    print(f"grad norm {total_norm:.1f} >> clip 1.0 — clipping is masking something")

optimizer.step()
```

Two details people get wrong. First, **clip before `optimizer.step()` but after `loss.backward()`** — and if you use AMP with a `GradScaler`, you must `scaler.unscale_(optimizer)` *first*, because the gradients are still multiplied by the loss-scale factor and clipping the scaled gradient clips at the wrong threshold. Second, **`clip_grad_norm_` returns the pre-clip norm**, and that return value is the single most useful number you are throwing away if you ignore it — it is the exploding-gradient signal, logged for free on every step.

Now the honest part, because the kit demands honesty about trade-offs: **clipping is a band-aid, not a cure.** It caps the *symptom* (a giant step) without addressing the *cause* (why the gradient got giant). If your gradient is exploding because the learning rate is too high, clipping will *mask* it — the run stops `NaN`-ing, the loss limps downward, and you conclude you fixed it, when in fact you have a model taking the maximum allowed step on every iteration, which is a slow, mis-conditioned optimization that often converges to a worse solution. If your gradient is exploding because of a single corrupt batch (a label out of range, an all-`inf` feature), clipping hides the bad batch instead of letting you find and fix it. And if it is exploding because of genuine instability (a missing normalization, a bad init), clipping fights the architecture every step instead of correcting it.

The test that separates "clipping is the cure" from "clipping is a band-aid" is the decision in the figure below: **add clipping and watch what happens.** If clipping is firing occasionally (the pre-clip norm exceeds the threshold a few percent of steps) and the loss trains well, clipping is a legitimate stabilizer — this is the normal, healthy case for large-model and RNN training. If clipping is firing on nearly *every* step (pre-clip norm always $\gg$ threshold), clipping is masking a real problem: lower the LR, add warmup, fix the init, or find the bad batch, and the constant clipping should stop. The pre-clip norm distribution is the diagnostic; a run where it is always pegged at the ceiling is a run where clipping is the only thing standing between you and divergence, which is not a stable place to live.

**Choosing the threshold.** People agonize over the clip value; the honest answer is that the *right* threshold is set by the *typical* grad norm of a healthy run, not by a magic constant. Log the global grad norm for a few hundred steps of a stable run, take a high percentile (say the 95th), and set the clip threshold a little above it — that way clipping only fires on the genuine outliers and never touches the normal steps. Common defaults of $1.0$ (for many LLM and RNN recipes) or $5.0$ work because, with a mean-reduced loss and a reasonable LR, the typical norm sits below those. A threshold set *below* the typical norm is actively harmful: it clips every step, which not only masks problems but distorts the optimization — you are no longer following the gradient, you are following a unit-norm version of it, which changes the effective learning rate in a state-dependent way and can stall convergence. The rule of thumb: clip should be a guardrail you rarely hit, not a wall you lean on.

**The AMP interaction, in full.** With automatic mixed precision the order of operations is load-bearing and a frequent source of silent bugs. The `GradScaler` multiplies the loss by a large scale factor *before* `backward()` so that small fp16 gradients do not underflow; the gradients in `.grad` are therefore *scaled up* by that factor. If you call `clip_grad_norm_` on those scaled gradients, you are comparing a scaled norm (which might read $50{,}000$) against your threshold of $1.0$ and clipping at completely the wrong point. The correct sequence is: `scaler.scale(loss).backward()`, then `scaler.unscale_(optimizer)` to divide the gradients back down to their true magnitude, *then* `clip_grad_norm_`, then `scaler.step(optimizer)` and `scaler.update()`. Skip the unscale and your clipping is meaningless; the loss may still look fine because the scaler also skips steps with `inf`/`NaN` gradients, which hides the breakage behind a different mechanism. This is one of those bugs that produces no error and a plausible-looking loss curve while quietly making your clip threshold off by four orders of magnitude.

```python
import torch

scaler = torch.cuda.amp.GradScaler()

for batch in loader:
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = criterion(model(batch.x), batch.y)
    scaler.scale(loss).backward()          # gradients are SCALED here
    scaler.unscale_(optimizer)             # divide back to true magnitude FIRST
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # now clip correctly
    scaler.step(optimizer)
    scaler.update()
```

![A decision tree starting from a loss spike with gradient 1e4 at step 412, branching on whether adding a clip of 1.0 lets the loss recover, where recovery means clipping is the cure and a continued climb means clipping only masks a learning-rate or bad-batch problem to fix instead.](/imgs/blogs/gradients-exploding-and-vanishing-5.png)

### The anatomy of an explosion, step by step

To make the exploding case concrete, here is what the instruments read across a run that dies — and what they read once you fix it. The crucial observation is that the global grad norm *climbs for hundreds of steps* before the loss spikes. The loss curve gives you essentially no warning (it looks fine at step 410), but the grad-norm trace has been ramping since step 200. This is the entire argument for logging the global grad norm every step: it is a leading indicator of an explosion, where the loss is a lagging one.

![A timeline of an exploding run showing the global gradient norm at 2.1 and calm at step zero, drifting to 9 by step 200, rising to 600 by step 380, spiking to 1e4 at step 412, the loss going NaN at step 414, and a clipped variant capping the norm at 1.0.](/imgs/blogs/gradients-exploding-and-vanishing-7.png)

| Step | Global grad norm | Loss | What the curve shows | What grad norm shows |
|---|---|---|---|---|
| 0 | 2.1 | 6.90 | healthy start | calm |
| 200 | 9.4 | 3.10 | descending, looks fine | **drifting up** |
| 380 | 612 | 2.15 | descending, still looks fine | **clearly rising** |
| 412 | 1.0e4 | 9.40 | sudden spike | exploded |
| 414 | NaN | NaN | dead | dead |
| 412 (with clip 1.0) | 1.0 (post-clip) | 2.09 | smooth, no spike | capped |

The loss column would have let this run die with zero warning. The grad-norm column flagged it 212 steps early. That asymmetry is the whole reason this signal earns a permanent place in your logging.

## 8. `detect_anomaly`: localizing the op that produces the NaN

When an explosion ends in `NaN` and you cannot tell *which operation* produced it, PyTorch's anomaly detection turns the silent `NaN` into a stack trace pointing at the exact forward op whose backward produced the bad value. It is slow (it adds bookkeeping to every op), so you enable it only while hunting, but it is the fastest way to localize a `NaN`-producing layer:

```python
import torch

# Wrap ONLY the suspect region — anomaly mode is slow, don't leave it on.
with torch.autograd.set_detect_anomaly(True):
    out = model(batch)
    loss = criterion(out, target)
    loss.backward()   # raises with a traceback to the forward op whose backward made the NaN/Inf
```

The traceback names the forward operation (for example, a `LogBackward` or `DivBackward`) so you know whether the `NaN` came from a `log(0)` in a loss, a division by a zeroed normalization statistic, or an overflow in an exploding matmul. That distinction routes you: an overflow in a matmul is an exploding-gradient story (clip and lower LR); a `log(0)` is a numerics story; a division by a zero BN statistic is a normalization story. Disambiguating those is the focus of [hunting NaNs and infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs); here the point is that anomaly mode is how you find *which* op, after the grad-norm trace told you *that* it is exploding. Note that with mixed precision the explosion may show as an `inf` that the `GradScaler` catches and skips — a run that silently skips many steps (the scaler backing off) is an under-the-radar exploding-gradient symptom worth logging too.

## 9. The full diagnostic table: symptom, test, fix

Putting the pieces together, here is the routing table you actually use at 2 a.m. The discipline is always the same: read the per-layer grad norm, find where the extreme number lives, confirm with the cheap test, then apply the *cure* (init/norm/residual/architecture) rather than the *band-aid* (clip alone) wherever you can.

![A three-by-three matrix routing the symptoms of early grad 1e-7, global grad 1e4, and a recurring explosion over time to a confirming test, a likely cause, and a real fix such as Kaiming plus normalization plus residual, lower learning rate with warmup, or an LSTM gate with clipping.](/imgs/blogs/gradients-exploding-and-vanishing-8.png)

| Symptom (from grad-norm report) | Confirming test | Likely cause | Real fix (not just clip) |
|---|---|---|---|
| Early-layer grad $\approx 10^{-7}$, later layers fine | `activation_variance_by_depth` ramps down; `saturation_report` high in early layers | Bad init, saturating activation, no residual/norm | Kaiming/Xavier init, ReLU/GELU, add BatchNorm/LayerNorm, residual skips |
| Global grad norm $\approx 10^4$, late layers largest, loss spikes to `NaN` | Pre-clip norm has been climbing for many steps; clip recovers the run | Too-high LR, missing norm, or a corrupt batch | Lower LR, add warmup, fix init/norm; clip 1.0 as a stabilizer not a crutch |
| Loss fine on short sequences, `NaN` on long ones | Per-step grad norm spikes only on long-sequence batches | RNN BPTT: $\rho(W_{hh}) > 1$ over many timesteps | LSTM/GRU gating + `clip_grad_norm_(.., 1.0)` |
| A specific parameter has `grad = None` | It is absent from the autograd graph | Frozen/detached/in-place-op bug — **not** a gradient-magnitude bug | Fix the graph (model-code track), not the gradient scale |
| Clipping fires on *every* step | Pre-clip norm always $\gg$ threshold | Clipping is masking a real LR/init/data problem | Treat clip as a symptom; lower LR, fix the cause |
| Loss spikes then *recovers* without `NaN` | A single batch index correlates with the spike | A bad batch, not systemic divergence | Find and fix the batch; consider skip-on-spike, don't lower LR globally |

The two rows worth tattooing are the ones that distinguish a *cure* from a *mask*. An exploding gradient that clipping fixes-and-stays-fixed (clip fires occasionally) was a stabilization problem; an exploding gradient where clip fires every step is an LR/init/data problem wearing a clipping costume. And a vanishing gradient is never a clipping problem at all — clipping only scales down, it cannot scale a collapsed gradient back up. Vanishing is an architecture-and-init problem, full stop.

## 10. Across architectures: the same disease, different scenery

The product mechanism is universal, but each architecture surfaces it differently, and knowing the local accent saves time.

**Deep CNNs.** Vanishing was the wall that stopped plain CNNs past ~20 layers; ResNet's residual blocks and BatchNorm are the cure, and they are why 50-, 101-, 152-layer networks train. The signature is the early conv blocks reading near-zero grad norm. The fix is almost always "you removed or never had the residual/norm," not the optimizer.

**RNN / LSTM.** Both ends in *time*, governed by $\rho(W_{hh})$. Vanishing → use LSTM/GRU gates (the additive cell-state highway). Exploding → `clip_grad_norm_` is the legitimate, standard cure. The signature is sequence-length-dependent: short sequences fine, long sequences explode or fail to learn long-range structure.

**Transformers.** Residual connections around every attention and MLP sub-block plus LayerNorm keep the signal $O(1)$ by design, which is most of why Transformers scale to extreme depth. But they are not immune: the **loss spike** phenomenon in large-model pretraining is an exploding-gradient event, often triggered by a bad batch or an LR slightly too high for the current state, and it is handled with grad clipping, warmup, and sometimes skipping the offending batch. Pre-LN versus Post-LN placement materially changes the gradient flow — Pre-LN (normalize *inside* the residual branch) gives a cleaner gradient highway and is why most modern LLMs use it. The spike-versus-divergence distinction for big models is its own deep topic in [loss spikes and divergence](/blog/machine-learning/debugging-training/loss-spikes-and-divergence).

**Deep tabular MLPs.** The least-discussed but very common case: a 20–40 layer fully-connected net with the wrong init (the worked example in section 3) vanishes silently and looks exactly like an LR problem. The cure is the same trio — Kaiming init, normalization (BatchNorm or LayerNorm), and residual connections — which is why modern tabular deep nets (and architectures like ResNet-style MLPs) include all three.

| Architecture | Axis it compounds on | Dominant failure | Primary cure |
|---|---|---|---|
| Deep CNN | depth | vanishing (early blocks) | residual + BatchNorm + Kaiming init |
| RNN (vanilla) | time | both, set by $\rho(W_{hh})$ | LSTM/GRU gates; clip for exploding |
| LSTM/GRU | time | exploding on long sequences | `clip_grad_norm_(.., 1.0)` |
| Transformer | depth | spikes (exploding) at scale | Pre-LN + warmup + clip; bad-batch handling |
| Deep tabular MLP | depth | vanishing (looks like low LR) | Kaiming init + norm + residual |

## 11. Finetuning: the same product, new failure modes

Finetuning fails differently from training-from-scratch, and gradients are at the center of two of its most common disasters. The series treats finetuning as first-class because so many readers never train from scratch — they take a pretrained backbone and adapt it — and the gradient pathologies there are distinct enough to deserve their own read.

**The too-high-LR explosion that destroys pretrained features.** A pretrained backbone arrives with its weights already in a good, low-loss region. If you finetune it with the learning rate you would use for training from scratch (say $10^{-3}$), the first few steps take enormous strides relative to where the weights already are — the update-to-weight ratio spikes — and you blow the carefully-learned features apart before the model adapts. The gradient signature is an early explosion: the global grad norm is large in the first dozen steps, the loss spikes, and even if it recovers, the model has *forgotten* its pretraining (a phenomenon called catastrophic forgetting). The fix is the standard finetuning rule: use a learning rate $10$–$100\times$ smaller than from-scratch (often $10^{-5}$ for transformer finetuning), add warmup so the first steps are gentle, and consider freezing the backbone for a few hundred steps while only the new head trains. The diagnostic is the same `update_to_weight_ratio` probe — if a pretrained layer's ratio is $10^{-1}$ in step 5, your LR is destroying it. (The full finetuning-LR story lives in the finetuning posts of this series and in the [taxonomy](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs); the gradient lens is just one window onto it.)

**The LoRA / frozen-backbone vanish that is actually a `None`-grad bug.** When you finetune with a frozen backbone (or with LoRA, where the backbone is frozen and only small adapter matrices train), most parameters *should* have no gradient — `requires_grad=False` means they are correctly excluded from the graph. The trap is confusing "correctly frozen" with "accidentally not training." If your `grad_norm_report` shows the LoRA adapter parameters themselves reading `grad=None` or `grad=1e-12`, the adapter is not in the graph — a wrong `target_modules`, a dtype mismatch, or a gradient-checkpointing interaction has detached it — and the model is training *nothing*, silently, while the loss limps down on the head alone. This is not a vanishing-gradient bug in the section-1 sense (the product did not decay); it is a graph-connectivity bug that *looks* like vanishing in the report. The way to tell them apart: a true vanish shows a *small but nonzero* gradient that *ramps* across depth; a graph bug shows `None` or a numerically-exact zero on the specific parameters that should be training. Always check that the parameters you *intend* to train have nonzero grads before you conclude anything about magnitude.

#### Worked example: a LoRA finetune that "trains" but learns nothing

A 7B-parameter LLM is being finetuned with LoRA (rank 16) on an instruction dataset. The loss descends from $2.4$ to $1.9$ over an epoch and the engineer ships it — but the model's behavior is unchanged from the base. Running the per-layer grad report reveals the problem: every LoRA `lora_A` and `lora_B` parameter reads `grad=None`, while the base model's frozen layers correctly read `None` and the language-model head reads a healthy `grad=0.3`. The "training" was the head fitting the new format; the adapters never entered the graph because `target_modules` was set to `["query", "value"]` while this model names its projections `["q_proj", "v_proj"]` — a silent string mismatch that `get_peft_model` did not error on. The confirming test is one call to `model.print_trainable_parameters()`, which reports `trainable params: 0.6M` (the head only) instead of the expected $\approx 4$M (head plus adapters); after fixing `target_modules`, it reports the right count, the adapter grads read $10^{-2}$ to $10^{-1}$, and the model's behavior actually changes. The grad-norm report localized in one glance what the loss curve actively concealed — a falling loss with a model that learned nothing, the exact "your run is lying to you" signature this series is built around. (The full LoRA-no-op debugging story is its own post; here it is the sharpest example of *grad = None is not vanishing*.)

## 12. Case studies and well-known signatures

A few named results to ground the mechanism in published evidence rather than just my war stories.

**The original vanishing-gradient analysis.** Hochreiter's 1991 diploma thesis and the later Bengio, Simard, and Frasconi (1994) paper "Learning long-term dependencies with gradient descent is difficult" formalized exactly the product argument above for recurrent nets: the gradient over a time gap is bounded by the spectral radius raised to the gap length, so learning dependencies beyond a short horizon is, with saturating activations, essentially impossible. This is the result the LSTM (Hochreiter and Schmidhuber, 1997) was designed to defeat, with the gated additive cell state that keeps $\partial c_t/\partial c_{t-1}$ near one.

**Clipping for RNNs.** Pascanu, Mikolov, and Bengio (2013), "On the difficulty of training recurrent neural networks," is the canonical source for norm-based gradient clipping as the fix for the exploding end of recurrent training. Their analysis ties the explosion to the spectral radius of the recurrent weight and shows clipping the global norm preserves direction while capping magnitude — exactly the `clip_grad_norm_` you call today.

**Kaiming initialization for very deep nets.** He, Zhang, Ren, and Sun (2015), "Delving Deep into Rectifiers," derived the $2/n_\text{in}$ variance that makes deep ReLU networks trainable, and showed that with the *wrong* init (Xavier on ReLU) a 30-layer network fails to converge while the correct init trains it — the precise factor-of-two from section 3, demonstrated empirically.

**Residual learning.** He, Zhang, Ren, and Sun (2016), "Deep Residual Learning for Image Recognition," is the result that residual connections let 152-layer networks train where plain nets *degrade* past ~20 layers — and crucially their plain-vs-residual comparison shows the plain deep net is not overfitting, it is failing to optimize, which is the vanishing-gradient signature. The $I + F'$ Jacobian is why.

**Large-model loss spikes.** Public training logs and reports (the PaLM, OPT, and GLM-130B papers, among others) document loss spikes during large-LM pretraining and the mitigations — gradient clipping, lowering the LR, skipping the batch, and rewinding to a checkpoint before the spike. These are exploding-gradient events at scale, and the practitioners' playbook (watch the grad norm, clip, skip the bad batch) is exactly the one in this post.

## 13. When this is — and isn't — your bug

Be decisive about when the gradient is the suspect and when the symptom points elsewhere, because chasing a gradient ghost wastes as much time as the original bug.

**It IS a vanishing gradient when:** early layers read near-zero grad norm while later layers learn, the activation-variance probe ramps down with depth, a saturation report is hot in the early layers, and the run is a deep net or long RNN. The fix is init/activation/normalization/residual.

**It IS an exploding gradient when:** the global grad norm climbs for many steps and then the loss spikes to `NaN`, the late layers read largest, and the failure is reproducible (or sequence-length-dependent in an RNN). The fix is lower LR / warmup / norm, with clipping as a stabilizer.

**It is NOT a gradient-magnitude bug when:** a parameter's grad is `None` (that is a detached/frozen graph — a model-code bug, not a small-gradient bug); the loss is flat from step zero but every layer's grad norm is *healthy* (that is data or LR, not vanishing — if the gradient is fine and nothing learns, the signal is fine and the *target* or the *LR* is the problem); the loss is flat because you are evaluating a memorized leak (that is a data/eval bug, and the train loss is suspiciously clean); or the run is non-deterministic and the "spike" is a once-in-a-thousand-step bad batch rather than systemic divergence. A useful sieve: if the [overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) *passes* (the model drives loss to near zero on one batch), your gradient flow is fundamentally intact and the bug is in data, LR, or evaluation — stop blaming the gradients.

The bisection mindset is the through-line. Before you change init, confirm the gradient profile actually ramps with depth. Before you add clipping, confirm the global norm actually spikes. Before you conclude "vanishing," confirm the grads exist (not `None`) and are merely small. Each confirmation is a cheap test that rules a whole region of the [taxonomy](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) in or out, so you change exactly one thing for one reason.

## Key takeaways

- **A gradient is a product of per-layer Jacobian norms.** A product of $L$ factors each below one vanishes ($r^L \to 0$); each above one explodes ($r^L \to \infty$); only $r \approx 1$ keeps it working. Depth and recurrence are exponents on this product.
- **The per-layer gradient-norm report is the single best tool.** Loop `model.named_parameters()` after `loss.backward()`, print `p.grad.norm()`, and read *where in depth* the extreme number sits: early-and-tiny is vanishing, late-and-huge is exploding.
- **Log the global grad norm every step.** It is a *leading* indicator — it climbs for hundreds of steps before the loss spikes to `NaN`, while the loss gives you no warning.
- **Saturation and init set the regime before step one.** Sigmoid's max derivative is $0.25$; the wrong init shrinks variance geometrically with depth. Check the activation-variance-by-depth probe; it should be flat.
- **Residuals and normalization keep the signal $O(1)$.** A residual Jacobian is $I + F'$ — a gradient highway of gain one. Normalization re-pins the variance. This is why deep CNNs and Transformers train.
- **RNNs vanish and explode in *time*, governed by $\rho(W_{hh})$.** Vanishing → LSTM/GRU gates (the additive cell-state highway). Exploding → `clip_grad_norm_` is the legitimate cure.
- **Clip with `clip_grad_norm_` (global L2, preserves direction), and log its return value** — the pre-clip norm is your free explosion alarm. Unscale AMP gradients first.
- **Clipping is a band-aid for exploding, useless for vanishing.** If it fires every step, it is masking an LR/init/data problem — fix the cause. It cannot un-shrink a collapsed gradient.
- **If overfit-one-batch passes, it is not the gradient.** Healthy grads + nothing learning points at data, LR, or evaluation, not at vanishing/exploding.

## Further reading

- Hochreiter & Schmidhuber, "Long Short-Term Memory" (1997) — the gated cell state that defeats vanishing gradients in time.
- Bengio, Simard & Frasconi, "Learning Long-Term Dependencies with Gradient Descent is Difficult" (1994) — the spectral-radius product argument for recurrent nets.
- Pascanu, Mikolov & Bengio, "On the Difficulty of Training Recurrent Neural Networks" (2013) — gradient clipping as the cure for the exploding end.
- He, Zhang, Ren & Sun, "Delving Deep into Rectifiers" (2015) — Kaiming/He initialization and the variance argument.
- He, Zhang, Ren & Sun, "Deep Residual Learning for Image Recognition" (2016) — residual connections and the $I + F'$ gradient highway.
- PyTorch docs — `torch.nn.utils.clip_grad_norm_`, `torch.autograd.set_detect_anomaly`, and `Tensor.register_hook` for per-layer gradient capture.
- Within this series: [a taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs), [instrumenting a training run: what to log](/blog/machine-learning/debugging-training/instrumenting-a-training-run-what-to-log), [initialization and normalization bugs](/blog/machine-learning/debugging-training/initialization-and-normalization-bugs), [hunting NaNs and infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs), [loss spikes and divergence](/blog/machine-learning/debugging-training/loss-spikes-and-divergence), and the capstone [the training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook).
