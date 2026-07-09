# Technique explanation reference

Read at the start of Phase B (planning) and again at Phase D (before writing prose). This is the pedagogical spine of a paper analysis: how to explain **each technique in the paper from intuition all the way up to the math**, so a smart reader who has never seen the method can follow the derivation and feel *why* it works — not just *that* it works.

The rule: **never lead with the equation.** An equation is the *compression* of an idea; the reader can only decompress it if you've already built the mental model it compresses. Build the model first, then show the equation as the precise statement of what they now already believe.

## The explanation ladder (apply to every load-bearing technique)

Each core technique gets these six rungs, in order. Not every rung needs to be long — a simple technique might spend two sentences per rung — but skip none, and never reorder (math before intuition defeats the purpose).

1. **The problem it solves.** What breaks *without* this technique? Start from the pain: the quadratic memory, the vanishing gradient, the reward that can't be computed, the mode collapse. One or two sentences that make the reader want the solution before they see it. If the paper is a response to a specific prior method, name it and its failure.

2. **Intuition / analogy.** A plain-language mental model with **zero notation**. Reach for an everyday analogy (a library index, a relay race, a thermostat, a group vote, routing mail through sorting centers) — but only if the analogy is *load-bearing*, i.e. its structure actually mirrors the mechanism. A decorative analogy that breaks the moment you look closely is worse than none. State the analogy, then immediately say which part of it maps to which part of the method.

3. **The mechanism, semi-formally.** Walk the actual computation **step by step in words**, in the order it happens, naming the real objects (queries, keys, logits, the router, the KL term) but not yet writing the full equation. "For each token, we compare its query against every key, turn those similarities into weights that sum to one, and take the weighted average of the values." This is the bridge between the analogy and the math — the reader should be able to *predict* the equation before you write it.

4. **The math.** Now the equation, as the precise statement of rung 3. Non-negotiables:
   - **Define every symbol on first use**, inline, the moment it appears: "where $Q \in \mathbb{R}^{n \times d_k}$ is the matrix of $n$ query vectors, each of dimension $d_k$…".
   - **Annotate shapes.** For anything tensor-valued, give the shape and say what each axis indexes. Readers lose the thread on un-shaped tensors faster than on any other single thing.
   - **Build up, don't dump.** If the final loss has four terms, introduce them one at a time — show the base objective, then add the regularizer, then the constraint — each with its own line and its own sentence of motivation. A wall of symbols is where readers bail.
   - **Say what each operation *does*, not just what it is.** After `$\text{softmax}(QK^\top/\sqrt{d_k})$`, add: "the $\sqrt{d_k}$ divisor keeps the dot products from growing with dimension, which would otherwise push softmax into a near-one-hot regime where gradients vanish." That clause is the whole point — it's the *why* the equation alone can't say.
   - Math in `$...$` (inline) / `$$...$$` (display). **Brace-wrap any inline math that starts with a digit** as `${...}$` (a repo quirk: `$2^n$` desyncs the dollar-pairing; `${2^n}$` is safe).

5. **Worked micro-example.** Make it concrete with the smallest possible instance: tiny numbers, a 2×2 matrix, a sequence of length 3, or ~10 lines of pytorch-shaped pseudocode that runs the technique on toy input. Show at least one intermediate value. This is where the reader *checks their understanding* against a ground truth — it converts belief into knowledge. Prefer a numeric trace for a formula, runnable-looking code for an algorithm.

6. **Why it works / when it fails.** Close the loop back to intuition: name the mechanism that makes it work (the property the math guarantees), quantify the tradeoff (the cost, the assumption, the regime where it degrades), and give at least one concrete failure mode. "This is $O(n^2)$ in sequence length — fine at 512 tokens, ruinous at 100k, which is exactly what the next section's method attacks." A technique with no stated failure mode is under-explained.

## Tie the ladder to the figures

- **Point at the extracted figure** while climbing the ladder: "the left stack in Figure 1 is the encoder we just described; the arrow looping back into 'Add & Norm' is the residual connection from rung 3." The reader should be able to trace your words on the authors' own diagram.
- **Redraw when the paper's figure hides the mechanism.** If the paper only *describes* a data-flow in prose, or its figure is too dense to show the one thing you're explaining, that's a redrawn-diagram trigger (Phase C2). A redrawn figure earns its place by making rung 3 (the mechanism) or rung 4 (the math) visible — a loss decomposition, a shape-annotated tensor flow, a before/after of the quantity the method changes.

## Depth calibration

- **Cover every load-bearing technique** — the ones the results actually depend on. A paper's contribution usually rests on 3–7 techniques; each gets the full ladder. Minor implementation details (a specific learning-rate warmup, a dataloader trick) get a sentence, not a ladder — unless the paper's whole point *is* that detail.
- **Match rung length to difficulty.** A one-line reparameterization needs two sentences of intuition and one equation. A new attention variant or a novel loss needs the full treatment with a worked example.
- **Don't inflate.** Detailed ≠ padded. Every paragraph either builds intuition, states mechanism, defines math, or works an example. Cut anything that merely restates.

## Faithfulness (the paper is the ground truth)

- **Every symbol, number, and claim traces to `paper.txt`.** You read the actual paper in Phase B; explain *that* method, with *its* notation (or a clearly-flagged cleaner notation you introduce). Do not fill gaps from a remembered version of the method — papers differ from the folklore about them in exactly the details that matter.
- **Quote the paper's numbers exactly** (BLEU, accuracy, FLOPs, parameter counts) and name the table they come from. If you compute a derived number, show the arithmetic.
- **Flag extrapolation.** When you explain *beyond* the paper — an intuition the authors didn't state, a connection to another method, a "what I'd build with this" — mark it as yours, not theirs.
- **Preserve honest uncertainty.** If the paper's derivation has a gap, an unstated assumption, or an ablation it skips, say so in the Critique — don't paper over it to make the explanation cleaner.

## Voice (inherits blog-writer, shifted toward teaching)

- Accessible-expert, intuition-first — closer to `finance-writer`'s "deep but never gatekept" than to blog-writer's war-story register, because the reader may be meeting this method for the first time.
- First-person plural (`we`) for shared reasoning through a derivation; first-person singular only for genuine opinion in the Critique.
- **Always English** — title, body, math annotations, captions, code comments — regardless of how the skill was invoked.
- Define jargon on first use; never assume the reader has read the prior work (link it instead).
