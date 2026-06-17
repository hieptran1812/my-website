---
title: "A Vision War Story: Debugging a Classifier That Wouldn't Train"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Walk a real fine-grained image classifier from val accuracy pinned at chance to 93 percent, one bisection at a time — and watch the same six-place debugging method localize every bug in minutes instead of days."
tags:
  [
    "debugging",
    "model-training",
    "computer-vision",
    "pytorch",
    "finetuning",
    "deep-learning",
    "data-leakage",
    "transfer-learning",
    "image-classification",
  ]
category: "machine-learning"
subcategory: "Debugging Training"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/cv-war-story-a-classifier-that-wouldnt-train-1.png"
---

The ticket said "model won't learn, please look." Attached was a Weights & Biases run that had burned through forty GPU-hours on a single A100. A ResNet-50 backbone, an eight-class head, a custom dataset of product photographs for a defect-and-category classifier — the kind of fine-grained vision task that shows up in every retail, manufacturing, and marketplace pipeline. The loss curve in the dashboard was, to a tired engineer scrolling past at 6pm, "fine": it started at 2.09 and after twelve thousand steps was sitting at 2.04. The GPU was pinned at 92% utilization. Throughput was a healthy 1,400 images per second. Every operational signal said the run was alive and well.

The one number that was not fine was validation accuracy: 11.4% on an eight-class problem, where pure chance is $1/8 = 12.5\%$. The model was, after forty hours of training, doing slightly *worse* than a coin that always guessed the most common class. Somebody had already "fixed" it three times — more epochs, lower learning rate, then a bigger backbone — and each fix had cost another day and changed nothing. That is the signature of a run that is not slow, it is *broken*, and no amount of the three usual knobs will move it. This post is the full debugging session that took that run from 11.4% to 93.1% validation accuracy, told honestly, with the instrument readings and the dead ends included. By the end you will have watched the six-place bisection method localize five distinct bugs, each in minutes, and you will be able to run the same playbook on any stalled vision run you inherit.

![Eight ordered debugging stages on a timeline, from reproduce and overfit-one-batch through fixing data, augmentation, a loss spike, and finally layer-wise learning rate, each annotated with the before and after accuracy](/imgs/blogs/cv-war-story-a-classifier-that-wouldnt-train-1.png)

The figure above is the whole story compressed into eight stages, and it is worth keeping it in view, because the single most important thing I want you to take from this post is not any one of the five bugs. It is the *shape* of the session. We never once "tried something to see if it helps." Every stage did exactly four things: read an instrument, form a hypothesis about which of the six places — data, optimization, model code, numerics, systems, evaluation — the bug lived in, run the cheapest test that could confirm or kill that hypothesis, then fix and re-measure. That discipline is what turned a run that had resisted three days of guessing into a five-hour debugging session. This is the bisection method from the [training-debugging taxonomy](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) applied end to end to one vision project, and it is the worked-example backbone of the whole series' [debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook).

A note before we start, because it matters for trust: every number in this post is from a realistic reconstruction of a class of bugs I have personally hit more than once, not a literal transcript of one run. The accuracy figures, grad norms, and loss values are the values these specific bugs produce — a BGR swap really does pin a fine-grained classifier near chance, a near-duplicate leak really does inflate val by 15 to 30 points — but treat the exact decimals as representative, not as a benchmark you can cite. Where I lean on a published result (confident learning's label-error rates, the ImageNet normalization constants, the linear-probe-then-finetune recipe) I will name it.

## 1. Stage zero: reproduce it, then make it deterministic

The first rule of debugging anything is that you cannot fix a bug you cannot reproduce on demand, and you cannot bisect a bug whose symptom moves every time you run it. Before I touched the model, the data, or the learning rate, I did the least glamorous thing in the whole session: I made the run *small and repeatable*.

"Small" means I cut everything that costs time without buying information. The dataset is 80,000 images; for debugging I pointed the loader at a fixed 2,000-image subset. The model trains for forty hours; I capped the debug runs at 300 steps. Forty GPU-hours per experiment is a debugging killer — at roughly \$1.50 per A100 GPU-hour on the cloud, the previous engineer had spent something like \$180 of compute to learn nothing three times over. A 300-step run on a 2,000-image subset finishes in under two minutes and costs a few cents. You want your debug loop to be *fast enough that you stop being precious about running it*, because the entire method depends on running many cheap experiments, not one expensive one.

"Repeatable" means determinism, and this is the part people skip and regret. If two identical runs give you val accuracy 11.4% and then 14.2%, you have no idea whether a change you make helped or whether you are watching noise. Worse, a nondeterministic dataloader can hide the very bug you are hunting — a collation bug that only triggers on certain worker orderings will appear and vanish. So the first code I add to any sick run is a determinism block.

```python
import os, random, numpy as np, torch

def set_determinism(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cuDNN pick deterministic kernels (slower, but bit-stable).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Surface any silently-nondeterministic op as a hard error.
    torch.use_deterministic_algorithms(True, warn_only=True)

set_determinism(0)

# DataLoader workers each need a derived, fixed seed, or shuffle order
# changes run to run even with the global seed set.
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=True, num_workers=4,
    worker_init_fn=seed_worker, generator=g,
)
```

The `worker_init_fn` and the explicit `generator` are the parts everyone forgets. PyTorch's `DataLoader` spawns worker processes, and unless you seed each one, NumPy's RNG state inside the workers is *not* governed by your global `np.random.seed` — every worker starts from its own entropy, so your augmentations and any NumPy-based sampling differ run to run. I have watched people chase a "flickering" bug for an afternoon that was nothing but unseeded workers. We cover the full story of why a run is not reproducible in the [reproducibility and determinism post](/blog/machine-learning/debugging-training/reproducibility-and-determinism-in-training); for our purposes here the point is narrow and absolute: **you cannot bisect a moving target, so pin it first.**

With determinism on, I reran the sick configuration on the 2,000-image subset. Val accuracy: 11.6%, stable across three runs to within half a point. Train loss after 300 steps: 2.05, essentially flat from its 2.09 start. Good — it is reproducibly broken. Now I can bisect.

Here is the full instrument panel I started from, because the *discipline of reading every gauge before forming a hypothesis* is exactly what the previous three "fixes" skipped. Each row is a reading and what a tired engineer might wrongly conclude from it in isolation:

| Instrument | Reading | The seductive wrong read | What it actually says |
| --- | --- | --- | --- |
| Train loss @ 12k steps | 2.04 (from 2.09) | "learning, just slowly" | parked at $\log 8$ — not learning at all |
| GPU utilization | 92% | "compute is healthy" | compute is fine; says nothing about correctness |
| Throughput | 1,400 img/s | "data pipeline is fast" | fast and wrong are independent |
| Val accuracy | 11.4% | "needs more epochs" | at chance; the run is broken, not slow |
| Global grad norm | 1.3e-3 | "small but nonzero" | tiny gradients carrying no label signal |
| Loss curve shape | flat line | "plateau, lower the LR" | flat at $\log C$ is wiring, not tuning |

The trap in every right-hand cell is the same: an operational signal (util, throughput, "nonzero" gradient) is being read as evidence of *correctness*, when it is only evidence of *liveness*. A run can be perfectly alive — pinning the GPU, streaming data, taking optimizer steps — and learning nothing, because liveness and correctness are orthogonal. The only gauge that carries correctness information here is the loss value relative to its uniform-guess floor, and that one is screaming. The previous engineer read the green operational gauges, concluded "it is working, just slowly," and spent three days turning the one knob (epochs, LR) that a flat-at-$\log C$ loss has already told you is irrelevant.

### The science of "flat at chance"

Before running a single diagnostic, I want to predict what a healthy and a broken run should *look like* numerically, because the prediction is the thing that makes a reading diagnostic rather than just data.

The loss here is cross-entropy over eight classes. For a single example with true class $y$ and predicted class probabilities $p$, cross-entropy is $-\log p_y$. A model that has learned nothing and outputs a uniform distribution assigns $p_y = 1/8$ to the correct class, giving a loss of

$$
-\log\left(\tfrac{1}{8}\right) = \log 8 = 2.0794.
$$

That number — $\log 8 \approx 2.08$ — is the *signature of a model that is guessing uniformly*. Our run started at 2.09 (random init, near-uniform output, exactly as expected) and after twelve thousand steps was at 2.04. It had moved four hundredths of a nat away from pure chance in forty hours. That is not "slow learning." A model that is learning drives cross-entropy *visibly* below its uniform floor within hundreds of steps. A model parked at $\log C$ for its class count $C$ is producing essentially constant or random logits, which means gradients carrying label information are not reaching the part of the network that assigns class scores. The memorable rule: **a classification loss stuck at $\log C$ means the model is guessing, and gradients are not connecting outputs to labels.** $\log 2 = 0.69$, $\log 8 = 2.08$, $\log 10 = 2.30$, $\log 1000 = 6.91$ — memorize these so you can read a stuck loss at a glance.

That single observation already cleaves the suspect space. The model is not *failing to optimize a hard objective* — it is *not optimizing the right objective at all*. The bug is in the path from input to logits to loss to gradient, which is to say: data, model code, or loss wiring. It is almost certainly not the learning rate (the previous "fixes" already proved lowering it changes nothing), and it is not a systems bug (single GPU, reproducible). I now have a hypothesis ranking, and the next stage is built to confirm or kill the top of it cheaply.

## 2. Stage one: overfit one batch — and watch it fail

The single highest-leverage test in machine learning is the [overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test), and it is the natural next move. Take one fixed batch of 16 images, turn off every source of noise — shuffling, augmentation, dropout, weight decay — and train on that one batch over and over with a forgiving optimizer (Adam at $10^{-3}$). A ResNet-50 has roughly $2.5 \times 10^7$ parameters; asking it to memorize 16 images is asking a stadium to remember sixteen faces. It should crater the loss toward zero in a couple hundred steps. If it cannot do that, the problem is *not* that the task is hard — it is that something mechanical in the path is broken.

```python
def overfit_one_batch(model, batch, n_steps=300, lr=1e-3, device="cuda"):
    model.train()
    x, y = batch
    x, y = x.to(device), y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()
    for step in range(n_steps):
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        if step % 25 == 0:
            acc = (logits.argmax(1) == y).float().mean().item()
            print(f"step {step:4d}  loss {loss.item():.4f}  acc {acc:.2f}")
    return loss.item()

batch = next(iter(loader))         # one fixed batch, then stop iterating
final = overfit_one_batch(model, batch)
assert final < 0.05, f"overfit FAILED: loss floored at {final:.3f}"
```

Here is what it printed:

```bash
step    0  loss 2.0931  acc 0.06
step   25  loss 2.0788  acc 0.12
step   50  loss 2.0788  acc 0.12
step  100  loss 2.0788  acc 0.12
step  200  loss 2.0788  acc 0.12
step  299  loss 2.0788  acc 0.12
```

The loss glued itself to 2.0788 — which is $\log 8$ to four decimals — and never moved. The accuracy sat at 0.12, one correct guess out of sixteen, exactly chance. **The overfit test failed,** and this is the most informative single result of the entire session, because of what it rules in and out.

![A decision graph in which a failed overfit-one-batch test clears optimization and forks the hunt toward the data path or the model code, with confirmation by reading the raw tensor](/imgs/blogs/cv-war-story-a-classifier-that-wouldnt-train-2.png)

The logic is a conjunction, and that is what makes a failure diagnostic. Overfitting one batch requires that capacity exists (it does — 25 million parameters for 16 points), that gradients flow, that the optimizer steps, that the loss is wired to the labels, and that the data path delivers the right labels. Because every one of those must be true simultaneously for the test to pass, a *failure* means at least one is false, and the space of "one of these is mechanically broken" is small and checkable. Critically, the failure *clears* the learning rate as a suspect: Adam at $10^{-3}$ on 16 images would converge with almost any sane wiring, so a flat loss is not a tuning problem. The previous engineer's three days of LR and epoch and backbone changes were doomed from the start — none of those touch the conjunction that failed.

A failing overfit test points hard at two of the six places: the **data path** (the labels or pixels reaching the model are wrong) or the **model code** (gradients are not reaching the head). Before I print the batch, one 20-second test discriminates between them: print the per-parameter gradient norms after one backward pass. If the head has a healthy gradient and the loss still will not move, the loss wiring or the labels are the problem; if the head's gradient is zero, the model code froze something.

```python
model.zero_grad()
logits = model(x)
loss = torch.nn.functional.cross_entropy(logits, y)
loss.backward()
for name, p in model.named_parameters():
    if p.grad is None:
        print(f"{name:40s}  grad = None  (not in graph!)")
    elif "fc" in name or "head" in name:   # the classifier head
        print(f"{name:40s}  grad norm = {p.grad.norm().item():.3e}")
```

The head printed a perfectly healthy gradient norm of `4.7e-01` — not zero, not None. So gradients *are* reaching the classifier; the model code is not frozen. That kills the model-code branch and leaves the data path. The loss will not fall even though the gradient flows, which can only mean the gradient is pointing the model toward a target that is *inconsistent across steps or wrong*. The next stage is the cheapest test in all of computer vision, and it is the one people are too proud to run: look at the actual tensor.

It is worth dwelling on why the grad-norm print is the right *second* test, not the first or the fifth, because the ordering is the method. After the overfit test fails, two branches remain: the data path is delivering wrong inputs/labels, or the model code is preventing gradients from reaching the part that scores classes. Those two have completely different fixes, so you want to split them before doing the more involved work. The grad-norm print is the cheapest possible splitter — one forward, one backward, a loop over `named_parameters()`, twenty seconds — and it discriminates perfectly. If the head's gradient were `None` or `0`, the bug would be in the model code (a `requires_grad=False` left on the head, a `torch.no_grad()` wrapping the forward, a detached graph, the head excluded from the optimizer's parameter list), and I would go hunting there. Because the head's gradient is healthy and *nonzero*, the model code is exonerated and the data path is indicted. This is bisection in its purest form: one cheap reading halves the remaining suspect space. I want you to notice that I did not *guess* "it's probably the data" and start printing batches — I ran the test whose result forces the conclusion, so that when I do print the batch I already know that is where the bug must be.

#### Worked example: reading the grad-norm split

Make the two outcomes concrete, because the same 20-second test routes to opposite fixes. Suppose the print had instead shown `fc.weight  grad = None` for the head. `None` (not zero) means the parameter is *not in the autograd graph at all* — the most common cause is `requires_grad=False` on the head from a leftover freeze experiment, or the head being constructed *after* the optimizer was built so its parameters were never registered. The fix is mechanical: re-enable grad and rebuild the optimizer. Now suppose it showed `fc.weight  grad norm = 0.0` (a real zero, not `None`). A hard zero means the parameter is in the graph but receiving no gradient signal — typically the loss does not actually depend on the head's output (logits computed from a detached tensor, or the loss compares the wrong tensors), or the LR is literally 0. Different fix again. And the case we actually hit — `grad norm = 4.7e-01`, a healthy nonzero — means the head trains fine but is being pulled toward a wrong target, which is the data path. Three readings, three different one-of-the-six suspects, all from the same cheap print. *That* is why you run the splitter before you start fixing.

#### Worked example: reading a flat overfit trajectory

It is worth pausing on *why* a wrong-but-consistent label would still flatline the loss at exactly $\log 8$, because the intuition is subtle. Suppose the label map were merely permuted — class "scratch" mapped to index 3 instead of index 2, consistently. Then the overfit test would *still pass*: the model would happily learn the permuted mapping and drive the loss to zero, and you would only notice the bug at evaluation. A flat loss at $\log 8$ is a stronger signal than that. It means the target the loss is computing against is *not learnable from the input at all* — either the labels within the batch are inconsistent with the pixels in a way no function can fit (e.g., the same image appears with two different labels because of a collation bug), or the inputs are degenerate (every image normalized to the same near-constant tensor), or the labels are random with respect to the inputs. The model finds the best constant output — the uniform distribution — and sits there, because no parameter setting does better. That distinction (flat at $\log C$ = unlearnable batch; converges but wrong = permuted mapping) tells me to look at the *relationship* between pixels and labels in the raw batch, not just at one or the other.

## 3. Stage two: print the batch — three bugs hiding in one tensor

There is a discipline I try to beat into every engineer I work with: **before you theorize about a data bug, print the data.** Not the shapes — the actual values, the actual pixels, the actual labels, side by side, for the first few examples. Ninety percent of "the model won't learn" bugs are visible to a human eye in the first batch, and they are invisible in any aggregate metric. Here is the print-the-batch routine I ran.

```python
import numpy as np

x, y = next(iter(loader))
print("x shape:", x.shape, "dtype:", x.dtype)
print("x min/max/mean:", x.min().item(), x.max().item(), x.mean().item())
print("labels in batch:", y[:16].tolist())
print("label min/max:", y.min().item(), y.max().item())
print("num classes (head out_features):", model.fc.out_features)

# Undo the normalization and look at the first image's channels.
img = x[0].cpu().numpy()              # CHW
print("channel means (C0,C1,C2):", img.mean(axis=(1,2)))

# Save a few decoded images to eyeball them.
import torchvision.utils as vutils
vutils.save_image(x[:8], "debug_batch.png", normalize=True)
```

It printed:

```bash
x shape: torch.Size([32, 3, 224, 224]) dtype: torch.float32
x min/max/mean: -8.41 13.92 0.37
labels in batch: [1, 8, 3, 5, 8, 2, 4, 6, 1, 7, 8, 3, 5, 2, 6, 4]
label min/max: 1 8
num classes (head out_features): 8
channel means (C0,C1,C2): [0.19 0.33 0.71]
```

Three separate bugs are visible in those eight lines, and `debug_batch.png` made the first one unmistakable: every product photo had the orange-and-blue color cast of a swapped red and blue channel. Let me take them one at a time, because each is a textbook member of the [computer-vision input-pipeline bug](/blog/machine-learning/debugging-training/cv-input-pipeline-bugs) family, and together they explain everything.

![Three input-pipeline bugs found by printing one batch — a BGR channel swap, normalization applied at the wrong pixel scale, and a label index shifted by one — shown as a before-and-after of the corrupted versus corrected pipeline](/imgs/blogs/cv-war-story-a-classifier-that-wouldnt-train-3.png)

**Bug 1 — BGR vs RGB.** The dataset loader used `cv2.imread`, and OpenCV decodes images in **BGR** channel order, not RGB. The model is a torchvision ResNet-50 pretrained on ImageNet, which expects **RGB**. So channel 0 (which the model treats as red) was actually blue, and channel 2 (which the model treats as blue) was actually red. The decoded `debug_batch.png` showed it instantly: skin tones turned blue, blue packaging turned orange. The channel means confirmed it — a real RGB product-photo dataset typically has the red channel comparable to or above the others; here channel 0's underlying statistics were flipped against channel 2. For a coarse task (cat vs car) a BGR swap merely costs a few points; for a *fine-grained* task where color is a discriminative feature (distinguishing product variants and defect types), it is catastrophic, because the pretrained features the model relies on were learned on correctly-ordered color.

It is worth being precise about *why* a channel swap is so much worse for a pretrained model than for one trained from scratch, because the intuition explains why this bug pins our run near chance rather than merely shaving a few points. A network trained from scratch on BGR data would simply learn BGR-appropriate filters in its first convolutional layer — the swap would be invisible, because the model never knew which channel was "supposed" to be red. But a *pretrained* backbone arrives with first-layer filters already tuned to RGB statistics: a filter that has learned "warm reddish edge here" fires on the wrong channel when you feed it BGR, and that error compounds through every subsequent layer because the features it built on are now systematically wrong. The pretrained representation is a tower of assumptions, and channel order is load-bearing at the base. On a fine-grained task this is fatal in a second way: fine-grained classification leans heavily on subtle color cues to separate visually similar classes (the difference between two product variants may be a slightly different shade), so corrupting color does not just add noise — it destroys the most discriminative signal the task has. That is the combination — pretrained fragility times color-dependence — that turns a "few points" bug into a "stuck at chance" bug.

**Bug 2 — normalization on the wrong scale.** The pipeline applied the standard ImageNet normalization,

```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
```

but those constants are defined for pixels in the range $[0, 1]$, and the OpenCV path delivered pixels in $[0, 255]$ (uint8 → float without the divide-by-255). Watch what that does. ImageNet's per-channel normalization computes $(x - \mu)/\sigma$. With the intended $[0,1]$ input, a mid-gray pixel of $0.5$ in channel 0 becomes $(0.5 - 0.485)/0.229 \approx 0.07$ — a small, well-behaved value. With the bug, a mid-gray pixel is $127.5$, and $(127.5 - 0.485)/0.229 \approx 555$. The printed `x min/max` of $[-8.41, 13.92]$ was the *partial* tell — it was less extreme than 555 because the actual fix had a second compounding factor, but the mean of 0.37 with a max near 14 is wildly outside the roughly $[-2.6, 2.6]$ range that correct ImageNet-normalized inputs occupy. **The single fastest sanity check for a normalization bug: correctly normalized ImageNet inputs live in about $[-2.7, 2.7]$ per channel.** Anything with a max in the teens or hundreds is unnormalized or double-scaled. The science of why this destroys learning is the same as feeding a network out-of-distribution activations: BatchNorm running statistics and the pretrained convolutional filters were calibrated for inputs near unit variance; hand them inputs an order of magnitude larger and the early activations saturate, the gradients that flow back are tiny and uninformative, and the network is effectively blind. This is exactly the [initialization-and-normalization failure mode](/blog/machine-learning/debugging-training/initialization-and-normalization-bugs) that a pretrained backbone is most fragile to.

**Bug 3 — the label off-by-one.** Look at `label min/max: 1 8` against `num classes: 8`. The labels run from 1 to 8, but `CrossEntropyLoss` expects class indices in $[0, C-1]$, here $[0, 7]$. Two things are wrong at once: class index 0 *never appears in any label* (so the head's logit-0 output gets no positive gradient and that class is unlearnable), and index 8 is *out of range* for an 8-output head. In PyTorch, a target of 8 against an 8-class head is undefined behavior — on CPU it raises, but on CUDA it can silently index out of bounds and produce garbage gradients, which is precisely the "gradient flows but points at nonsense" signature we saw. The root cause was mundane and common: the dataset's class folders were named `1_scratch`, `2_dent`, …, `8_ok`, and a `int(folder.split("_")[0])` parse produced 1-based labels that were never shifted to 0-based. This is the off-by-one that the [input-pipeline post](/blog/machine-learning/debugging-training/the-input-pipeline-is-lying-to-you) calls out as one of the most common silent killers in custom datasets.

### The fix, and the confirming re-test

The fix is small and surgical — convert color order once at decode, scale to $[0,1]$ *before* normalizing, and remap labels to 0-based:

```python
import cv2
from torchvision import transforms

class FixedImageDataset(torch.utils.data.Dataset):
    def __init__(self, samples, class_to_idx):
        self.samples = samples            # list of (path, folder_name)
        self.class_to_idx = class_to_idx  # {"1_scratch": 0, ...} 0-based, sorted
        self.tf = transforms.Compose([
            transforms.ToTensor(),         # HWC uint8 [0,255] -> CHW float [0,1]
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, i):
        path, folder = self.samples[i]
        bgr = cv2.imread(path)                  # BGR, [0,255]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)   # FIX 1: to RGB, once
        img = self.tf(rgb)                      # FIX 2: ToTensor scales to [0,1] first
        label = self.class_to_idx[folder]       # FIX 3: 0-based label map
        return img, label
```

The single most valuable habit in this whole story is that **I did not trust the fix until the instrument confirmed it.** I reran the overfit-one-batch test, unchanged, on the corrected pipeline:

```bash
step    0  loss 2.0864  acc 0.06
step   25  loss 0.8132  acc 0.69
step   50  loss 0.1774  acc 0.94
step  100  loss 0.0231  acc 1.00
step  200  loss 0.0041  acc 1.00
step  299  loss 0.0038  acc 1.00
```

The loss fell from $\log 8$ to 0.0038 and accuracy hit 100% on the batch. **The overfit test now passes**, which proves the data path, the model code, the loss wiring, and the optimizer are all correct *together*. We have not just fixed three bugs; we have earned a guarantee that the entire mechanical core of the run is sound. Everything from here is a different *class* of bug — generalization, not wiring.

#### Worked example: the dollars-and-hours math of printing the batch

Let me put a number on the value of the print-the-batch discipline, because it is the cheapest high-yield habit in the field and people still skip it. The previous engineer ran three "fixes" — more epochs, lower LR, bigger backbone — each a full 40 GPU-hour run, at roughly \$1.50 per A100-hour. That is about \$180 of compute and three calendar days, and it produced exactly zero diagnostic information, because none of those changes touched the conjunction that the overfit test failed. The print-the-batch step took 90 seconds and a few cents of compute and surfaced all three root causes at once. The lesson is not "printing is nice." It is that **a cheap test that can falsify your hypothesis is worth infinitely more than an expensive test that cannot.** A 40-hour run that "still doesn't work" tells you nothing; a 90-second batch print tells you the color order is wrong. Spend your compute on tests that *discriminate*.

The figure below is the pipeline rendered as the five rungs each image crosses, because the deeper point is that a corrupted input never raises an exception — it just delivers wrong pixels or wrong labels and lets the loss quietly misbehave.

![A five-rung stack of the input pipeline from image decode through color order, scale and normalization, label mapping, and tensor conversion, showing where each silent fault hides](/imgs/blogs/cv-war-story-a-classifier-that-wouldnt-train-4.png)

## 4. Stage three: the full run learns — but val is far below train

With the wiring fixed, I lifted the constraints — full 80,000-image dataset, full schedule — and launched a real run. The dashboard finally looked alive in the way that matters: train loss fell from 2.08 to 0.04 over the epochs, train accuracy climbed to 99.2%. The previous run had been a corpse; this one was learning. But validation told a different story:

| Instrument | Sick run (stage 0) | After data fix (stage 3) |
| --- | --- | --- |
| Train loss (final) | 2.04 | 0.04 |
| Train accuracy | 12% | 99.2% |
| Val accuracy | 11.6% | 61.3% |
| Train − val gap | ~0 pts | 37.9 pts |
| Loss curve shape | flat at $\log 8$ | smooth descent |

Validation accuracy went from chance to 61.3% — a real model now exists — but a 37.9-point gap between train (99.2%) and validation (61.3%) is a screaming signal. The model has memorized the training set and generalizes poorly. In the six-place frame, a large train-minus-val gap routes you to **data** (the val set is not measuring what you think) or to **optimization/regularization** (the model is overfitting). The naive reaction is "add dropout and weight decay." The disciplined reaction is to ask *why* the gap is this large and to test the cheaper, higher-probability hypothesis first.

Here is the reasoning that ordered the hypotheses. A 38-point gap on a fine-grained task with a strong pretrained backbone and 80,000 images is *too large for ordinary overfitting*. Pretrained ResNet-50 on a clean fine-grained dataset of this size, even with light regularization, typically lands within 5 to 12 points of train. A gap of 38 points smells like the validation number itself is unreliable — either the val pipeline differs from train in a way that hurts, or the val set is contaminated and the number was *secretly inflated* before, or both. So before reaching for regularization, I audited the validation pipeline and the train-val split. That audit found two more bugs.

### Bug 4 — augmentation left on at validation

I printed the val transform and the train transform side by side, which is a 10-second test that catches a shockingly common bug:

```python
print("TRAIN transform:\n", train_loader.dataset.tf)
print("VAL transform:\n", val_loader.dataset.tf)
```

```bash
TRAIN transform:
 Compose(ToTensor(), RandomResizedCrop(224), RandomHorizontalFlip(p=0.5),
         ColorJitter(0.4,0.4,0.4), Normalize(...))
VAL transform:
 Compose(ToTensor(), RandomResizedCrop(224), RandomHorizontalFlip(p=0.5),
         ColorJitter(0.4,0.4,0.4), Normalize(...))
```

The validation set was being **randomly augmented** — random-resized-crop, random horizontal flip, and color jitter — exactly like the training set. This is wrong in two compounding ways. First, it makes the validation number *noisy*: every epoch sees a different random crop and flip of each val image, so the val accuracy jitters by several points run to run for no real reason, and you cannot tell signal from noise. Second, and worse for a fine-grained task, the augmentations are *destroying the discriminative signal* at eval time. A `RandomResizedCrop` can crop away the small defect that defines the class; a horizontal flip can turn a left-facing orientation into a right-facing one that no longer matches the class; `ColorJitter` perturbs exactly the color cues this task depends on. The validation set is supposed to measure the model on clean, canonical inputs. Augmenting it measures the model on a corrupted, harder distribution, which *deflates* the val number and inflates the gap. This is the train-only-not-eval mistake that the [vision augmentation-debugging post](/blog/machine-learning/debugging-training/augmentation-debugging-for-vision) treats in depth.

The fix is the standard eval transform — deterministic resize and center crop, no randomness:

```python
val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

But fixing the val augmentation alone did something I want you to notice carefully, because it is the most instructive moment in the whole story. After making the val transform deterministic, val accuracy *jumped* from 61.3% to **94.1%** — a 33-point leap. For a few minutes I thought I was done. Then I looked at that number with suspicion instead of relief, because 94.1% val against 99.2% train, on a fine-grained custom dataset, is *suspiciously good*. A gap of only 5 points on a hard fine-grained task with a model this size is the kind of result that should make you ask "is my val set honest?" rather than "great, ship it." That suspicion found the fifth bug.

### Bug 5 — near-duplicate leakage across the split

The dataset had been split with a naive random shuffle: throw all 80,000 image *files* in a list, shuffle, take 90% for train and 10% for val. The problem is that this dataset, like most real-world image datasets, contains **near-duplicates**: the same physical product photographed from slightly different angles, the same defect captured in burst mode, web-scraped images that appear at multiple resolutions. When you split by *file* rather than by *product*, near-identical photos of the same item land in both train and val. The model memorizes a product from its training photo and then "recognizes" the almost-identical val photo — not because it generalized, but because it has effectively *seen the answer*. This is textbook [data leakage](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer), the silent killer, in its vision-specific form: duplicate and near-duplicate rows straddling the train-test boundary.

The detector is a perceptual hash. Unlike a cryptographic hash, a perceptual hash (`phash`) maps visually similar images to similar bit-strings, so near-duplicates have a small Hamming distance even if their pixels differ:

```python
import imagehash
from PIL import Image
from collections import defaultdict

def phash_all(paths):
    return {p: imagehash.phash(Image.open(p).convert("RGB")) for p in paths}

hashes = phash_all(all_paths)

# Count train images whose phash is within Hamming distance 4 of any val image.
leak = 0
for tp, th in train_hashes.items():
    for vp, vh in val_hashes.items():
        if (th - vh) <= 4:        # near-duplicate threshold
            leak += 1
            break
print(f"train images with a near-dup in val: {leak} "
      f"({100*leak/len(train_hashes):.1f}%)")
```

```bash
train images with a near-dup in val: 5,912 (8.2%)
```

8.2% of training images had a near-duplicate sitting in the validation set. That is enormous. A leak of this size can inflate validation accuracy by 15 to 30 points on a fine-grained task, because the leaked val examples are essentially free correct answers. The "94.1%" was a lie; the model's true generalization was much lower, and the leak was masking exactly how much regularization and data work still remained.

The fix is to split by *group* — by the product identity, not by the file — so that all photos of a given product land entirely in train or entirely in val, never both. When you have a product or capture-session ID, use it directly; when you do not, cluster near-duplicates by perceptual hash and treat each cluster as a group. In scikit-learn terms this is exactly what `GroupKFold` enforces, and the principle is identical to using `GroupKFold` instead of a plain `KFold` to stop group leakage in tabular data.

```python
from sklearn.model_selection import GroupShuffleSplit

# group_id[i] = product id (or phash-cluster id) for image i
gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
train_idx, val_idx = next(gss.split(X=all_paths, groups=group_id))
# Now NO product appears in both train and val.
```

![A before-and-after contrasting a leaky, randomly-augmented validation pipeline against a deterministic eval transform with a group-wise deduplicated split, showing the validation number going from inflated and jittery to stable and honest](/imgs/blogs/cv-war-story-a-classifier-that-wouldnt-train-6.png)

After regrouping the split to eliminate the leak, val accuracy dropped from the dishonest 94.1% to an honest **88.3%**. That *drop* is the single most important number in this section, and it is worth sitting with: a fix that *lowers* your reported metric is often the most valuable fix you will make, because it replaces a number you cannot trust with one you can. The 88.3% is a real measurement of generalization on products the model has never seen. The 94.1% was the model grading its own homework. Track the honest number, not the flattering one.

#### Worked example: how big a leak inflates accuracy

You can estimate the inflation from a leak with a one-line model, which is useful because it tells you whether a leak you found is a rounding error or a catastrophe. Let the leak fraction be $f$ (here $f = 0.082$ of train had a val near-dup, but what matters is the fraction of the *val set* that is leaked; with a symmetric 90/10 split and these duplicate rates, roughly $f_v \approx 0.18$ of the val set had a train near-dup). On leaked val examples, the model behaves like it is being tested on training data, so its accuracy there approaches the train accuracy $a_t$. On honest val examples it achieves its true generalization $a_g$. The measured val accuracy is the mixture:

$$
a_{\text{val,measured}} = f_v \cdot a_t + (1 - f_v) \cdot a_g.
$$

Plugging in $f_v = 0.18$, $a_t = 0.992$, and the honest $a_g = 0.883$ gives $0.18 \times 0.992 + 0.82 \times 0.883 = 0.179 + 0.724 = 0.903$ — close to the inflated 94.1% we saw (the gap from the formula's 90.3% reflects that leaked examples score slightly *above* even train accuracy because they were specifically memorized, plus phash-threshold slop). The point of the arithmetic is not three-decimal precision; it is the *order of magnitude*: an 18%-of-val leak with a near-perfectly-fit train set inflates the reported metric by roughly 6 to 11 points, which is exactly the range we observed. When you find a leak, run this estimate — if $f_v$ is a few percent the inflation is small and you can prioritize other bugs, but if it is 15%+ on a well-fit model, your headline metric is fiction and you fix it first.

## 5. Stage four: closing the honest gap — regularization that actually helps

Now I had an honest baseline: train 99.2%, val 88.3%, a real gap of 10.9 points. *This* is ordinary overfitting — the kind that regularization and data work genuinely address, as opposed to the fake gap that came from broken evaluation. The earlier instinct to "add dropout" would have been throwing regularization at a measurement bug; only now, with the bug removed, does regularization address the actual problem. This is a recurring theme of the whole series: **fix the instrument before you act on its reading.**

With an honest 11-point gap, the levers are standard and I applied them in order of expected payoff, re-measuring after each so I could attribute the change:

```python
# 1. Sensible, label-preserving augmentation on TRAIN ONLY.
train_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0), antialias=True),
    transforms.RandomHorizontalFlip(p=0.5),   # safe: products are flip-invariant here
    transforms.ColorJitter(0.2, 0.2, 0.2),    # mild; this task is color-sensitive
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 2. Weight decay via the optimizer, and label smoothing on the loss.
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
crit = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

# 3. Early stopping on the (now honest) val metric.
#    Stop when val accuracy has not improved for `patience` epochs.
```

A subtlety worth flagging, because it bit me here: the augmentation that is *safe* depends on the task, and a flip that is harmless for products can be label-destroying elsewhere. Horizontal flip is fine for these symmetric product photos, but for a task where the label depends on orientation (reading a digit, a left-vs-right defect, text), `RandomHorizontalFlip` silently corrupts the label and *hurts*. Likewise `ColorJitter` had to be turned *down* here, not up, because color is discriminative for this fine-grained task — strong jitter would erase the very signal the model needs. The general rule from [augmentation-gone-wrong](/blog/machine-learning/debugging-training/augmentation-gone-wrong): an augmentation is only valid if it preserves the label, and "preserves the label" is task-specific, not universal.

After tuned augmentation, weight decay, label smoothing, and early stopping on the honest val metric, the gap closed:

| Stage | Train acc | Val acc | Honest gap |
| --- | --- | --- | --- |
| Data wiring fixed (stage 3) | 99.2% | 61.3% (leaky+aug) | — |
| Val pipeline fixed | 99.2% | 88.3% (honest) | 10.9 pts |
| + tuned aug, WD, smoothing | 96.8% | 90.6% | 6.2 pts |

Train accuracy dropped slightly (99.2% → 96.8%) — that is regularization working as designed, trading a little memorization for generalization — while honest val rose to 90.6% and the gap fell to 6.2 points. Now the model is in healthy territory, and the remaining stages are about the last two bugs that only show up on a long, real run: a loss spike and a fine-tuning learning-rate mistake.

## 6. Stage five: the loss spike at step nine thousand

Eight thousand steps into a long run, the loss curve — which had been a clean, smooth descent — did something violent. In a single step it jumped from 0.41 to 9.7, the gradient norm shot from a steady ~2.0 to 1.1e4, and then over the next few steps the loss came *back down* toward where it had been. A one-step spike that recovers is a completely different animal from a spike that runs away to NaN and stays there, and telling them apart is the whole diagnostic.

![A decision tree for reading a loss spike, separating a smooth-then-spike-then-recover shape that indicts one bad batch or a too-high learning rate from a spike-then-NaN shape that stays broken, routing to a corrupt-image guardrail](/imgs/blogs/cv-war-story-a-classifier-that-wouldnt-train-7.png)

The science of the distinction matters. A loss is a smooth function of the parameters over most of training, so a *smooth-then-spike-then-recover* shape means a single step took the parameters somewhere bad and the optimizer then climbed back — which points to a transient cause: one anomalous batch, or a learning rate momentarily too large for a sharp region of the loss surface. A *spike-then-NaN-that-stays* means the parameters were pushed somewhere unrecoverable, or a numeric overflow corrupted the weights permanently. Our spike recovered, so the suspect is a transient — most likely one bad batch. The way you confirm "one bad batch" is to *log the batch index at every spike* and look at the data:

```python
prev_loss = None
for step, (x, y) in enumerate(loader):
    opt.zero_grad()
    loss = crit(model(x), y)
    # Catch the spike: this step's loss is way above the running level.
    if prev_loss is not None and loss.item() > 5 * prev_loss:
        bad = (~torch.isfinite(x.view(x.size(0), -1)).all(dim=1)).nonzero()
        print(f"SPIKE @ step {step}: loss {loss.item():.2f}, "
              f"non-finite imgs in batch: {bad.flatten().tolist()}")
        torch.save({"x": x.cpu(), "y": y.cpu(), "step": step}, f"spike_{step}.pt")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    opt.step()
    prev_loss = 0.9 * (prev_loss or loss.item()) + 0.1 * loss.item()
```

When the spike hit, the print showed it immediately: `non-finite imgs in batch: [17]`. One image in the batch contained `NaN` pixels. Loading the saved `spike_9000.pt` and inspecting image 17 revealed a truncated JPEG — a partially-downloaded file that OpenCV decoded into a tensor with `NaN` and `inf` values. When that image flowed through the network, it produced a `NaN` loss contribution, the gradient exploded, and a single optimizer step yanked the weights far off course. The fact that the loss *recovered* is itself evidence: gradient clipping (which was on, `max_norm=5.0`) blunted the worst of the step, so the damage was survivable. Without clipping, that one corrupt image would likely have driven the run to a permanent NaN — the failure mode covered in [hunting NaNs and infs](/blog/machine-learning/debugging-training/hunting-nans-and-infs).

The fix has two layers, because a real pipeline should be robust to corrupt inputs without you babysitting every batch:

```python
class GuardedDataset(torch.utils.data.Dataset):
    def __getitem__(self, i):
        path, label = self.samples[i]
        bgr = cv2.imread(path)
        if bgr is None:                          # decode failed entirely
            return self.__getitem__((i + 1) % len(self))   # skip to next
        img = self.tf(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not torch.isfinite(img).all():        # NaN/inf pixels
            return self.__getitem__((i + 1) % len(self))   # skip the bad image
        return img, label
```

Plus the training-loop guardrails that catch *any* future bad batch generically: gradient clipping (already on, keep it), and an optional skip-the-batch guard that drops a step whose loss is non-finite rather than letting it poison the weights.

```python
loss = crit(model(x), y)
if not torch.isfinite(loss):
    print(f"non-finite loss at step {step}, skipping batch")
    opt.zero_grad(); continue          # do not backprop garbage
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
opt.step()
```

After adding the input guard and keeping clipping, I reran past step 9,000 and the spike was gone — the corrupt image was skipped at load time, the curve stayed smooth, and the gradient norm held steady around 2.0. The before-and-after is the cleanest kind of evidence: a curve that spiked to 9.7 with a grad norm of 1.1e4 now runs flat with a grad norm of 2.0. One corrupt JPEG out of 80,000 had been a loaded gun; a four-line guard defused it.

#### Worked example: spike versus divergence, by the numbers

How do you decide, *in the moment*, whether a spike is survivable or terminal — whether to keep training or kill the run? Read three numbers in the few steps after the spike. First, did the loss *recover* toward its pre-spike level within ~10 steps, or keep climbing? Ours went 0.41 → 9.7 → 3.2 → 0.9 → 0.5 — recovering, survivable. A terminal divergence goes 0.41 → 9.7 → 41 → `inf` → `NaN` — monotonically worse. Second, is the gradient norm coming back down? Ours fell from 1.1e4 back to ~2.0 within a few steps (clipping helped); a diverging run's grad norm stays pinned at 1e4+ or hits `inf`. Third, are there `NaN`s in the *weights* (not just the loss)? Check `any(not torch.isfinite(p).all() for p in model.parameters())` — if the *parameters* are NaN, the run is dead and no amount of waiting recovers it; if only one batch's loss was NaN but the weights are finite, you can skip and continue. The decision rule: **recovering loss + falling grad norm + finite weights → skip the bad batch and continue; monotone loss + pinned grad norm + NaN weights → kill and restart from the last good checkpoint.** This is the same logic the [loss-spikes-and-divergence post](/blog/machine-learning/debugging-training/loss-spikes-and-divergence) formalizes for large-model training, applied to a single corrupt image.

## 7. Stage six: the fine-tune learning rate that was destroying the backbone

The model was now at 90.6% honest val and training stably. But something about the training dynamics still bothered me: the model reached its best val accuracy early — around epoch 4 — and then *got worse*, drifting down to 87% by epoch 10 even as train accuracy kept climbing. That is not classic overfitting (which degrades val slowly and smoothly); it is the signature of the pretrained backbone being actively *damaged*. The pretrained features that made the model good early were being overwritten by a learning rate that was fine for the fresh head but far too aggressive for the delicate, already-good backbone weights.

This is the most common and most expensive mistake in transfer learning, and it has a precise mechanism. When you fine-tune a pretrained backbone with a freshly-initialized classification head, you have two populations of parameters with completely different needs. The **head** is random; it needs large gradient steps to learn the mapping from features to classes from scratch. The **backbone** is already excellent; it encodes general visual features learned on millions of ImageNet images, and it needs *tiny* steps — just gentle adaptation to the new domain. A single global learning rate cannot serve both. If you set the LR high enough for the head to learn (say $10^{-3}$), the same LR applied to the backbone takes huge steps that scramble the pretrained features faster than the head can exploit them — the model "forgets" what made it good. If you set the LR low enough to protect the backbone, the head learns at a crawl.

To confirm the diagnosis I printed per-block gradient and update norms, which is the instrument that makes this bug visible:

```python
def block_grad_norms(model):
    groups = {"stem": [], "layer1": [], "layer2": [],
              "layer3": [], "layer4": [], "fc": []}
    for name, p in model.named_parameters():
        if p.grad is None: continue
        for key in groups:
            if name.startswith(key):
                groups[key].append(p.grad.norm().item() ** 2)
    for key, sq in groups.items():
        if sq:
            print(f"{key:8s}  grad norm = {sum(sq) ** 0.5:.2f}")

block_grad_norms(model)   # called right after loss.backward()
```

```bash
stem      grad norm = 41.30
layer1    grad norm = 33.80
layer2    grad norm = 22.10
layer3    grad norm = 12.40
layer4    grad norm = 6.70
fc        grad norm = 3.00
```

The reading is the whole story. The early backbone blocks (`stem`, `layer1`) had gradient norms of 41 and 34 — *larger* than the head's 3.0. With one flat learning rate of $10^{-3}$, those early blocks were being shoved around an order of magnitude harder than the head, which means the pretrained low-level features (edges, colors, textures — exactly what a fine-grained task depends on) were being demolished. The instrument shows precisely why the model peaked early and decayed: each epoch ground the good features down a little more.

![A grid comparing per-block gradient norms and learning rates under a single flat learning rate versus a layer-wise decayed schedule, showing the early backbone blocks going from a destructive grad norm of 41 to a calm 0.8 while the head keeps its large step](/imgs/blogs/cv-war-story-a-classifier-that-wouldnt-train-8.png)

The fix is **layer-wise (discriminative) learning rates**: give the head the large LR it needs and decay the LR geometrically as you go deeper into the (earlier, more general) backbone layers, so the pretrained features barely move while the head learns fast. This is the standard recipe behind every robust fine-tuning pipeline, and it is the core of [debugging vision-transformer fine-tuning](/blog/machine-learning/debugging-training/debugging-vision-transformer-finetuning) as well — the layer-wise-LR trick is modality-agnostic.

```python
def discriminative_lr_groups(model, base_lr=1e-3, decay=0.3):
    # Deeper-into-backbone = smaller LR. Head gets base_lr.
    blocks = [model.fc, model.layer4, model.layer3,
              model.layer2, model.layer1,
              torch.nn.Sequential(model.conv1, model.bn1)]  # the stem
    groups = []
    for depth, block in enumerate(blocks):
        groups.append({
            "params": block.parameters(),
            "lr": base_lr * (decay ** depth),   # 1e-3, 3e-4, 9e-5, ...
        })
    return groups

opt = torch.optim.AdamW(discriminative_lr_groups(model, base_lr=1e-3, decay=0.3),
                        weight_decay=0.05)
```

With `decay=0.3`, the head trains at $10^{-3}$, `layer4` at $3\times10^{-4}$, down to the stem at roughly $2.4\times10^{-6}$ — a 400× spread between the fastest and slowest groups. After this change the per-block grad norms came into balance (the stem dropped to ~0.8, the head held its useful ~2.8), the model no longer peaked-and-decayed, and honest val accuracy climbed and *stayed*:

| Schedule | Best val acc | Behavior after peak |
| --- | --- | --- |
| Flat LR $10^{-3}$ | 90.6% @ epoch 4 | decays to 87% (backbone wrecked) |
| Layer-wise LR (decay 0.3) | 93.1% @ epoch 9 | stable, no decay |

An even cleaner variant, which I used for the final model, is **linear probing then fine-tuning** (the LP-FT recipe from Kumar et al., 2022): freeze the backbone entirely and train only the head for a couple epochs first, so the head is already sensible before any gradient touches the backbone, *then* unfreeze with discriminative LRs. The intuition is that an untrained head produces large, noisy gradients that flow back into the backbone on the very first steps and damage it before the head is any good; training the head first means that when the backbone finally starts adapting, the gradients reaching it are already meaningful. That recipe pushed the final honest val to 93.1%.

#### Worked example: how a too-high fine-tune LR forgets, quantitatively

Here is the back-of-the-envelope that predicts the catastrophe, so you can spot it before it costs you a run. A pretrained backbone weight $w$ sits at a good value $w^*$. Each optimizer step moves it by roughly $\eta \cdot g$ where $\eta$ is the learning rate and $g$ is the gradient. The pretrained features are "good" within a small neighborhood of $w^*$; move the weights a cumulative distance larger than that neighborhood and the features stop being the ImageNet features and become noise. Over $T$ steps the weights drift by order $\eta \cdot \bar{g} \cdot \sqrt{T}$ (a random-walk estimate, since gradient directions decorrelate). With the flat $\eta = 10^{-3}$ and the measured backbone grad norm $\bar g \approx 40$, after $T = 9{,}000$ steps the early-block drift is on the order of $10^{-3} \times 40 \times \sqrt{9000} \approx 3.8$ in parameter-norm — enormous relative to the small perturbation that pretrained features tolerate. With the layer-wise stem LR of $\eta \approx 2.4 \times 10^{-6}$, the same arithmetic gives a drift of $\approx 0.009$ — three orders of magnitude smaller, comfortably inside the "still good features" neighborhood. That is the whole mechanism in one estimate: **flat LR drifts the backbone ~400× further than it should, which is exactly enough to forget.** You do not need the exact constant; the order-of-magnitude gap between 3.8 and 0.009 is the proof that one LR cannot serve both populations.

## 8. The full ledger: five bugs, five instruments, five deltas

Here is the complete record of the session — the symptom-to-fix ledger I keep for every debug so that no bug silently returns and so the next engineer can see *how* each conclusion was reached, not just what changed.

![A matrix ledger mapping each debugging stage to the instrument that caught it, the fix applied, and the measured before-to-after accuracy delta](/imgs/blogs/cv-war-story-a-classifier-that-wouldnt-train-5.png)

| # | Symptom | Suspect (of six) | Confirming instrument | Root cause | Fix | Before → after |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Overfit-1-batch fails, loss flat at $\log 8$ | data / model | print the batch + grad-norm | BGR vs RGB swap | `cvtColor(BGR2RGB)` | batch loss 2.08 → 0.004 |
| 2 | same | data | `x.min/max` out of $[-2.7,2.7]$ | norm on $[0,255]$ not $[0,1]$ | scale before normalize | (part of fix 1) |
| 3 | same | data | `label.min/max = 1,8` vs 8 classes | labels 1-based, off-by-one | 0-based label map | train acc 12% → 99% |
| 4 | val 61% ≪ train 99% | evaluation | print train vs val transform | random aug left on at eval | deterministic eval transform | val 61% → 94% (then audit) |
| 5 | val "94%" suspiciously high | data | phash near-dup detector | near-duplicate leak across split | group-wise split (`GroupShuffleSplit`) | val 94% (fake) → 88% (honest) |
| 6 | honest gap 11 pts | optimization | train−val gap on clean val | ordinary overfitting | tuned aug + WD + smoothing | gap 11 → 6 pts |
| 7 | loss spike 0.4 → 9.7, grad 1e4 | numerics / data | log batch index at spike | corrupt JPEG (NaN pixels) | input guard + grad clip | spike → flat, grad 1e4 → 2.0 |
| 8 | val peaks epoch 4 then decays | optimization | per-block grad norm | flat LR wrecks backbone | layer-wise LR + LP-FT | val 90.6% → 93.1%, stable |

Read down the "Confirming instrument" column and notice that not one of these required a clever idea. Every single bug was caught by a *cheap, mechanical reading* of an instrument: print the batch, check the input range, check the label range, diff the transforms, run a phash, log the spike's batch, print per-block grad norms. The intelligence was not in any individual test — it was in the *order*, in bisecting to the right of the six places before running the test, so that each cheap test landed on the actual suspect. That is the entire method.

## 9. Turning the war story into permanent guardrails

A debugging session that fixes five bugs but leaves the pipeline able to silently re-introduce them is only half a win. The other half is converting each instrument reading into a *guardrail* — a cheap assertion that runs every time and fails loudly the moment a bug reappears. The single biggest difference between teams that fight the same training bugs forever and teams that do not is whether they bake the diagnostics into the pipeline as pre-flight checks. Here is the pre-flight I added to this project, assembled directly from the five bugs we found, that runs before every full training launch and costs about two seconds:

```python
def preflight_checks(loader, model, num_classes, device="cuda"):
    """Fail loudly before burning GPU-hours. Each check maps to a bug we hit."""
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    # Guard against the normalization-scale bug (inputs must be ~[-2.7, 2.7]).
    assert x.abs().max() < 6.0, (
        f"input range looks unnormalized: max abs {x.abs().max():.1f} "
        f"(ImageNet-normalized inputs sit near [-2.7, 2.7]); "
        f"did you forget to scale [0,255] -> [0,1] before Normalize?")

    # Guard against the label off-by-one (labels must be 0..C-1).
    assert y.min() >= 0 and y.max() < num_classes, (
        f"labels out of range: [{y.min()}, {y.max()}] for {num_classes} classes; "
        f"are they 1-based?")

    # Guard against a degenerate / single-class batch (collation bug).
    assert y.unique().numel() > 1, "batch has a single label; collation bug?"

    # Guard against corrupt inputs (the NaN-pixel spike).
    assert torch.isfinite(x).all(), "non-finite pixels in batch; corrupt image?"

    # Guard the wiring: one overfit step must actually move the loss.
    crit = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    l0 = crit(model(x), y).item()
    for _ in range(50):
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
    assert loss.item() < 0.7 * l0, (
        f"50 overfit steps barely moved loss ({l0:.3f} -> {loss.item():.3f}); "
        f"wiring/grad-flow bug — do not launch.")
    print("preflight OK: inputs scaled, labels valid, batch diverse, wiring works.")
```

Every line in that function is a bug from this story turned into a tripwire. The range assertion catches the normalization-scale bug; the label-range assertion catches the off-by-one; the unique-label check catches a collation bug that makes every label identical; the finite-pixel check catches the corrupt JPEG before it spikes; and the mini-overfit catches *any* wiring fault — BGR swap, frozen head, wrong loss reduction — generically, because all of them prevent the loss from moving on a tiny batch. The phash leak check does not belong in per-launch preflight (it is a one-time data-prep step), but it belongs in your *split-construction* code as a hard assertion that no group appears in both train and val:

```python
def assert_no_group_leak(train_groups, val_groups):
    overlap = set(train_groups) & set(val_groups)
    assert not overlap, (
        f"{len(overlap)} groups appear in BOTH train and val "
        f"(e.g. {list(overlap)[:3]}); your split leaks. "
        f"Split by product/phash-cluster id, not by file.")
```

The deeper point is about *where guardrails pay off*. A guardrail is worth adding exactly when (a) the bug is silent — it does not crash, it just degrades a metric you might not look at for hours — and (b) the check is far cheaper than the consequence. Every bug in this story qualifies: each one was silent (no exception), and each check is sub-second against a 40-hour run. That is the asymmetry that makes preflight checks the highest-return code you can write around a training loop. The taxonomy post frames the same idea as "make the failure loud and early"; this is what that looks like in a vision pipeline.

#### Worked example: the expected value of a two-second preflight

Put the asymmetry in numbers, because it is the argument for spending an afternoon writing checks. Say a particular silent bug (a label off-by-one, a BGR swap, an unnormalized input) lands in your pipeline with probability $p \approx 0.1$ on any given change — these bugs are common enough that one in ten substantive pipeline edits introduces one, in my experience. Without a preflight, the bug is caught only after a full run reveals a bad metric — call it 40 GPU-hours and a calendar day of an engineer's attention, roughly \$60 of compute plus the far larger cost of the lost day. With the preflight, the same bug is caught in two seconds at launch. The expected cost *avoided* per launch is $p \times (\text{wasted run}) \approx 0.1 \times (\$60 + \text{a day}) $, dominated by the engineer-day. Over a few hundred launches in a project's life, a two-second check that you wrote once saves tens of wasted runs and dozens of engineer-days. The preflight does not need to catch every bug, or even most — catching the *silent, expensive, common* ones is enough to make it the best line-for-line investment in the codebase. This is the same expected-value logic behind the print-the-batch math earlier: spend cheap compute on tests that falsify, and make the falsification automatic so you cannot forget to run it.

## 10. Case studies and real signatures

These bugs are not exotic; they are the documented, recurring failure modes of computer-vision pipelines, and it is worth grounding each in a known reference so you trust the pattern rather than just my reconstruction.

**Label errors are pervasive, even in benchmark test sets.** The cleanlab team's confident-learning work (Northcutt, Athalye, and Mueller, 2021) systematically found label errors across the most-used ML benchmarks — on the order of 3.4% average errors across ten datasets, including roughly 6% of the ImageNet validation labels and an estimated 3.4% of the well-curated MNIST test set. The relevance to our story: our stage-three model was being graded against a val set we *assumed* was clean, and only by auditing the split did we discover the leak. The broader lesson from confident learning is that you should never assume your labels — or your evaluation set — are pristine; quantify it. The technique generalizes directly to finding mislabeled rows in any custom dataset, which is the subject of [garbage in: finding label noise](/blog/machine-learning/debugging-training/garbage-in-finding-label-noise).

**Near-duplicate leakage has decided real competitions and shipped broken production models.** Train-test contamination through near-duplicates is one of the most common causes of the "great offline, bad in production" gap, and it has appeared in post-mortems of Kaggle competitions where image augmentations or web-scraped duplicates crossed the split. The signature is always the same: a validation number that is several points too good for the task difficulty, which collapses the moment you split by group instead of by row. If your held-out metric is suspiciously high on a hard task, *assume a leak until you have ruled it out* — the cost of checking (one phash pass) is trivial against the cost of shipping a model whose real accuracy is 15 points below its reported one.

**The BGR-vs-RGB swap is the canonical OpenCV-meets-PyTorch bug.** It is so common that it has its own folklore: OpenCV's `imread` returns BGR for historical reasons, while essentially every PyTorch/torchvision model and the ImageNet normalization constants assume RGB. The result is a model that trains to a *plausible-looking* accuracy (color is not the only feature) but is silently handicapped, and on color-sensitive fine-grained tasks it can be catastrophic. The fix is one line; the bug costs people weeks because the model does not crash — it just underperforms in a way that looks like "needs more data."

**Catastrophic forgetting from a too-high fine-tune LR is the documented motivation for discriminative learning rates.** Howard and Ruder's ULMFiT (2018) introduced discriminative fine-tuning — different learning rates per layer — precisely because a single LR damages pretrained representations. The LP-FT result (Kumar et al., 2022, "Fine-Tuning can Distort Pretrained Features") showed formally that fine-tuning can distort pretrained features and that linear-probing first mitigates it, especially under distribution shift. Our stage-eight fix is a direct application of that line of work; the per-block grad-norm reading is simply the *instrument* that lets you see the distortion happening in real time.

## 11. When this is — and isn't — your bug

A war story is only useful if it teaches you to recognize the *signatures*, so here is the decisive version: what each symptom in this story points to, and — just as important — when a similar-looking symptom points somewhere else entirely.

- **Loss flat at exactly $\log C$ → wiring, not tuning.** A loss pinned at the uniform-guess value means gradients are not connecting outputs to labels. Do *not* tune the LR; print the batch and check grad flow. But a loss that *slowly and steadily decreases* (2.08 → 2.05 → 2.01…) is a different signal — that is a too-small LR or a genuinely slow start, and it routes to optimization, not wiring.
- **Overfit-one-batch fails → data or model code.** It cleanly rules out "the task is hard" and "the LR needs tuning." But if overfit-one-batch *passes* and the full run still fails, **stop blaming the model** — the bug is in generalization: data distribution, the val set, augmentation, or regularization. We saw both halves of this in one session.
- **Huge train-val gap (30+ points) → suspect the evaluation, not the model.** A gap that large on a strong pretrained backbone is usually a broken val pipeline (augmentation at eval) or a leak inflating a previously-too-good number, not ordinary overfitting. An *honest* gap of 5 to 12 points is normal overfitting and routes to regularization. The trap is treating a measurement bug as an overfitting problem and drowning it in dropout.
- **Val suspiciously high on a hard task → suspect a leak.** Relief is the wrong reaction to a great held-out number on a fine-grained custom dataset. Run the phash near-dup check before you celebrate. A fix that *lowers* your metric to an honest value is a win.
- **Smooth-then-spike-then-recover → one bad batch or transient LR, not distribution shift.** Log the batch index at the spike. But a *smooth-then-NaN-that-stays* is numerics — overflow, log0, a corrupt weight — and routes to `detect_anomaly` and the NaN hunt, not to the data.
- **Val peaks early then decays on a fine-tune → backbone damage, not overfitting.** Classic overfitting degrades val slowly and smoothly; a sharp peak-then-decay with rising backbone grad norms means a too-high LR is forgetting the pretrained features. Print per-block grad norms; the fix is layer-wise LR, not more regularization.

The unifying meta-rule: **before you act on a number, ask whether the instrument producing it is trustworthy.** Half the bugs in this story were not in the model at all — they were in the *measurement* (augmented val, leaked split). A debugger who fixes the model in response to a broken instrument makes the model worse and the metric prettier. Fix the instrument first.

## 12. Key takeaways

1. **Reproduce and make it deterministic before anything else.** You cannot bisect a moving target; pin the seed, seed the dataloader workers, and shrink the run so each experiment costs seconds, not GPU-hours.
2. **A classification loss stuck at $\log C$ means the model is guessing.** $\log 8 = 2.08$, $\log 10 = 2.30$. Read a flat loss at that value as "gradients aren't connecting outputs to labels," not as "slow learning."
3. **Overfit one batch first; a failure is your most informative result.** It rules out the LR and the task difficulty and points at the data path or model code. A pass clears the entire mechanical core — then the bug is generalization.
4. **Print the batch — pixels and labels, side by side.** The BGR swap, the wrong-scale normalization, and the label off-by-one were all visible in eight lines of output and zero in any aggregate metric. Correct ImageNet-normalized inputs live in about $[-2.7, 2.7]$; labels must be 0-based in $[0, C-1]$.
5. **A 30+ point train-val gap is usually a broken instrument, not overfitting.** Diff the train and val transforms (no random aug at eval), and run a perceptual-hash near-duplicate check before you trust a held-out number on a hard task.
6. **A fix that lowers your metric to an honest value is a win.** The 94% that fell to 88% after fixing the leak was the most valuable change in the session — it replaced a number you cannot trust with one you can.
7. **A smooth-then-spike-then-recover loss is one bad batch; log the index and guard the input.** Keep gradient clipping on so a single corrupt image is survivable, not terminal. A spike that runs to NaN and stays is numerics, a different hunt.
8. **One learning rate cannot fine-tune a pretrained backbone and a fresh head.** Per-block grad norms reveal the early backbone being driven harder than the head; layer-wise LR (and linear-probe-then-fine-tune) protect the pretrained features.
9. **Bisect to the right of the six places before you run a test.** The intelligence is in the ordering, not the cleverness of any single instrument. Each cheap, mechanical reading only pays off because it lands on the actual suspect.

## Further reading

- **Within this series:** [A taxonomy of training and finetuning bugs](/blog/machine-learning/debugging-training/a-taxonomy-of-training-and-finetuning-bugs) — the symptom → suspect → test → fix decision tree this whole story instantiates; and the capstone [The training debugging playbook](/blog/machine-learning/debugging-training/the-training-debugging-playbook) for the full method and a printable checklist.
- **The single test:** [The overfit-a-single-batch test](/blog/machine-learning/debugging-training/the-overfit-a-single-batch-test) — the science and the false-pass traps behind stage one.
- **The data bugs:** [CV input-pipeline bugs](/blog/machine-learning/debugging-training/cv-input-pipeline-bugs) (BGR/RGB, normalization, channel order), [augmentation debugging for vision](/blog/machine-learning/debugging-training/augmentation-debugging-for-vision) (the val-aug and label-destroying-aug traps), and [data leakage: the silent killer](/blog/machine-learning/debugging-training/data-leakage-the-silent-killer) (near-duplicate and split contamination).
- **The fine-tune bugs:** [Debugging vision-transformer finetuning](/blog/machine-learning/debugging-training/debugging-vision-transformer-finetuning) — layer-wise LR, freeze schedules, and the destroy-the-features mistake, applied to ViTs.
- **Mixed Precision Training**, Micikevicius et al., 2018 — the loss-scaling and fp16 underflow background behind why clipping and guards matter.
- **Pervasive Label Errors in Test Sets**, Northcutt, Athalye, Mueller, 2021 (confident learning / cleanlab) — the documented label-noise rates in ImageNet, MNIST, and other benchmarks.
- **Universal Language Model Fine-tuning (ULMFiT)**, Howard and Ruder, 2018 — the origin of discriminative (per-layer) learning rates for fine-tuning.
- **Fine-Tuning can Distort Pretrained Features (LP-FT)**, Kumar et al., 2022 — why fine-tuning damages pretrained features and how linear-probing first mitigates it.
- **PyTorch reproducibility and autograd docs** — `torch.use_deterministic_algorithms`, `worker_init_fn`, `register_hook`, and `torch.autograd.set_detect_anomaly` for building the instruments used throughout this post.
