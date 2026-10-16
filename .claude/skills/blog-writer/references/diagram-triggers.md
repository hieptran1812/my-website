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

## Per-depth floors

- Explainer ≥ 4 figures
- Paper-reading ≥ 4
- Deep-dive ≥ 5

A post is 2,000 to 3,000 words, so plan **4 to 7 figures**, roughly one per 400 to 500 words. When there are more abstractions than that, merge related ones into one richer figure rather than dropping the picture; and prefer cutting a paragraph the figure already explains over cutting the figure.

## Abstraction inventory format (Phase B output)

For each abstract concept the post introduces, emit one bullet with:

- **Claim** (≥ 8 words): the single sentence the figure proves
- **Caption** (one sentence, not a label restatement): figure thesis
- **Section anchor**: the markdown heading it sits under
- **Kind**, chosen *from the concept's shape*. Pictorial (the default): `xy-chart` | `bars` | `cells` | `filmstrip` | `lanes` | `space` | `worked-example` | `anatomy` | `scale` | `analogy`. Structural: `matrix` | `grid` | `timeline`. Box-flow (rationed): `pipeline` | `graph` | `tree` | `layered-stack` | `before-after`. Definitions in `diagram-authoring.md §Illustrate, don't label`.
- **Encoding**: what the *geometry* says with the labels covered (e.g. "bar lengths to scale: weights 70 GB vs KV 12 GB", "16 cells fill left to right, 3 frames", "two lanes overlap except a 40 ms gap"). If you can only write "boxes connected by arrows", pick a different kind.
- **Sketch**: which shapes / numbers / annotations appear, with the real numbers from the prose's worked example

Figure count = abstraction count, within the 4 to 7 window.

## Figure-type diversity (plan it here)

Vary the kinds so the post doesn't read as one diagram repeated N times, and so the figures *show* rather than label:

- **Box-flow budget: at most ⅓ of the figures (max 2) are box-and-arrow kinds; at least ½ are pictorial.**
- **No kind more than twice.** With ≥ 4 figures use **≥ 3 distinct kinds**; with ≥ 6, **≥ 4**.
- **No two adjacent figures share a layout skeleton.**
- Map kind to concept shape: a quantity comparison → `bars`; a trend or tradeoff → `xy-chart`; a data structure, tensor, or memory layout → `cells`; a state change → `filmstrip`; concurrency or timing → `lanes`; a formula or algorithm → `worked-example`; one object's parts → `anatomy`; axes × choices → `matrix`; only a genuine branching flow → `graph`; only a genuine hierarchy → `tree`. See `diagram-authoring.md §Diversity across the post`.

## First figure ("mental model")

Referenced in the intro paragraph: *"The diagram above is the mental model: …"*. Subsequent figures sit directly under the heading or paragraph that introduces the concept.
