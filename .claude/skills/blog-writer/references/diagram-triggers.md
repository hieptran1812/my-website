# Diagram triggers & coverage rule

Read this before Phase B to plan the abstraction inventory.

## When a figure is required

Anything that needs illustration earns a figure — there is no upper limit. Trigger list (illustrative, not exhaustive):

- Multi-step processes, pipelines, request lifecycles
- Layered or hierarchical structures (stacks, taxonomies, nesting)
- Before/after comparisons, naive vs optimized, A/B variants
- State machines, transitions, lifecycles
- Data structures and memory layouts (cache, page table, KV block, queue, ring, hashmap)
- Math/physics intuitions that map to a picture (curve, region, vector field, tensor reshape)
- System architectures with ≥ 3 components
- Tradeoff matrices (axes × choices), capability tables
- Algorithm walkthroughs, decision trees, recursion shapes
- Timing/sequencing, parallelism layouts, GPU stream timelines
- Control- or data-flow graphs, dependency graphs
- Concrete-but-non-obvious mechanics (config effect, CLI mental model, regex matching shape, packet wire format, on-disk layout, attention pattern on a small example)
- Any phrase reaching for a visual analogy: "imagine", "think of it as", "consider the case where", "the way this works is", "under the hood", "looks something like"

## Coverage rule

If the prose introduces an idea and the next paragraph does not have a figure within 30 lines, **add one**. Missing figures are defects, not stylistic choices.

## Per-depth floors (minimums, never caps)

- Explainer ≥ 4 figures
- Paper-reading ≥ 5
- Deep-dive ≥ 8

Most posts will exceed these floors 2–3×. The ceiling is set by content.

## Abstraction inventory format (Phase B output)

For each abstract concept the post introduces, emit one bullet with:

- **Claim** (≥ 8 words): the single sentence the figure proves
- **Caption** (one sentence, not a label restatement): figure thesis
- **Section anchor**: the markdown heading it sits under
- **Sketch**: which boxes / arrows / labels appear

Figure count = abstraction count. If you found 9 abstractions, plan 9 figures.

## First figure ("mental model")

Referenced in the intro paragraph: *"The diagram above is the mental model: …"*. Subsequent figures sit directly under the heading or paragraph that introduces the concept.
