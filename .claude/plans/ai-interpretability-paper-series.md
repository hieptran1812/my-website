# AI Interpretability paper-analysis series

Created 2026-10-15. Source: the 111 kept stubs/drafts in `content/blog/paper-reading/ai-interpretability/` (466 low-value stubs deleted same day). One analysis post per paper, written over the stub file in place (slug = filename).

## Spec (user request, overrides paper-writer defaults)
- 3,000–4,000 words per post (paper-writer gate floor is 6,500: the word-count FAIL is an accepted deviation). readTime = (words+110)/220.
- Easy to understand, no wall of text: short paragraphs (≤4 sentences), tables, bullet ladders, a figure every ~400–500 words.
- Many figures: target 7–9 (≥3 extracted from the paper as `<slug>-fig<n>.webp`, ≥3 redrawn Excalidraw `<slug>-<n>.webp`).
- No em dashes. English. category paper-reading, subcategory "AI Interpretability". date = day written.
- Drafts with existing content (AxBench, Adaptive Activation Steering, Conceptors, Weird Generalization, Universal weight subspace, ES forgetting, CURE, NL descriptions of activations): rewrite to the same spec.

## Process per wave
6 parallel drafting agents (paper-writer skill) → figure-author does figures → post-verifier gates → orchestrator commits that wave's .md + .webp only, pull --rebase --autostash, push. Fresh session per wave; carry only this file path + wave number.

## Waves (status: [ ] todo, [x] shipped)

### Wave 1
- [x] `scaling-and-evaluating-sparse-autoencoders` | Scaling and Evaluating Sparse Autoencoders | https://arxiv.org/abs/2406.04093v1
- [x] `sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models` | Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs | https://arxiv.org/abs/2403.19647v3
- [x] `transcoders-find-interpretable-llm-feature-circuits` | Transcoders Find Interpretable LLM Feature Circuits | https://arxiv.org/abs/2406.11944v2
- [x] `a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders` | A is for Absorption: Studying Feature Splitting and Absorption in Sparse | https://arxiv.org/abs/2409.14507v6
- [x] `persona-vectors-monitoring-and-controlling-character-traits-in-language-models` | Persona Vectors: Monitoring and Controlling Character Traits in Language | https://arxiv.org/abs/2507.21509v3
- [x] `position-the-platonic-representation-hypothesis` | Position: The Platonic Representation Hypothesis | 

### Wave 2
- [x] `learning-multi-level-features-with-matryoshka-sparse-autoencoders` | Learning Multi-Level Features with Matryoshka Sparse Autoencoders | https://arxiv.org/abs/2503.17547v1
- [x] `dense-sae-latents-are-features-not-bugs` | Dense SAE Latents Are Features, Not Bugs | https://arxiv.org/abs/2506.15679v2
- [x] `sparse-autoencoders-do-not-find-canonical-units-of-analysis` | Sparse Autoencoders Do Not Find Canonical Units of Analysis | https://arxiv.org/abs/2502.04878v1
- [x] `are-sparse-autoencoders-useful-a-case-study-in-sparse-probing` | Are Sparse Autoencoders Useful? A Case Study in Sparse Probing | https://arxiv.org/abs/2502.16681v1
- [x] `universal-sparse-autoencoders-interpretable-cross-model-concept-alignment` | Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment | https://arxiv.org/abs/2502.03714v1
- [x] `do-i-know-this-entity-knowledge-awareness-and-hallucinations-in-language-models` | Do I Know This Entity? Knowledge Awareness and Hallucinations in Language | https://arxiv.org/abs/2411.14257v2

### Wave 3
- [x] `sparse-autoencoders-reveal-temporal-difference-learning-in-large-language-models` | Sparse Autoencoders Reveal Temporal Difference Learning in Large Language | https://arxiv.org/abs/2410.01280v1
- [x] `finding-transformer-circuits-with-edge-pruning` | Finding Transformer Circuits With Edge Pruning | https://arxiv.org/abs/2406.16778v3
- [x] `hypothesis-testing-the-circuit-hypothesis-in-llms` | Hypothesis Testing the Circuit Hypothesis in LLMs | https://arxiv.org/abs/2410.13032v1
- [x] `rethinking-circuit-completeness-in-language-models-and-or-and-adder-gates` | Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates | https://arxiv.org/abs/2505.10039v2
- [x] `llm-circuit-analyses-are-consistent-across-training-and-scale` | LLM Circuit Analyses Are Consistent Across Training and Scale | https://arxiv.org/abs/2407.10827v2
- [x] `the-computational-complexity-of-circuit-discovery-for-inner-interpretability` | The Computational Complexity of Circuit Discovery for Inner Interpretability | https://arxiv.org/abs/2410.08025v3

### Wave 4
- [x] `optimal-ablation-for-interpretability` | Optimal Ablation for Interpretability | https://arxiv.org/abs/2409.09951v1
- [x] `bilinear-mlps-enable-weight-based-mechanistic-interpretability` | Bilinear MLPs Enable Weight-based Mechanistic Interpretability | https://arxiv.org/abs/2410.08417v2
- [x] `a-implies-b-circuit-analysis-in-llms-for-propositional-logical-reasoning` | A Implies B: Circuit Analysis in LLMs for Propositional Logical Reasoning | https://arxiv.org/abs/2411.04105v4
- [x] `everything-everywhere-all-at-once-is-mechanistic-interpretability-identifiable` | Everything, Everywhere, All at Once: Is Mechanistic Interpretability | https://arxiv.org/abs/2502.20914v1
- [x] `hyperdas-towards-automating-mechanistic-interpretability-with-hypernetworks` | HyperDAS: Towards Automating Mechanistic Interpretability with Hypernetworks | https://arxiv.org/abs/2503.10894v3
- [x] `the-geometry-of-categorical-and-hierarchical-concepts-in-large-language-models` | The Geometry of Categorical and Hierarchical Concepts in Large Language Models | https://arxiv.org/abs/2406.01506v3

### Wave 5
- [x] `on-the-origins-of-linear-representations-in-large-language-models` | On the Origins of Linear Representations in Large Language Models | https://arxiv.org/abs/2403.03867v1
- [x] `on-linear-representations-and-pretraining-data-frequency-in-language-models` | On Linear Representations and Pretraining Data Frequency in Language Models | https://arxiv.org/abs/2504.12459v1
- [x] `emergence-of-linear-truth-encodings-in-language-models` | Emergence of Linear Truth Encodings in Language Models | https://arxiv.org/abs/2510.15804v1
- [x] `linear-representations-of-political-perspective-emerge-in-large-language-models` | Linear Representations of Political Perspective Emerge in Large Language | https://arxiv.org/abs/2503.02080v2
- [x] `the-semantic-hub-hypothesis-language-models-share-semantic-representations-across-languages-and-modalities` | The Semantic Hub Hypothesis: Language Models Share Semantic Representations | https://arxiv.org/abs/2411.04986v3
- [x] `token-embeddings-violate-the-manifold-hypothesis` | Token Embeddings Violate the Manifold Hypothesis | https://arxiv.org/abs/2504.01002v3

### Wave 6
- [x] `harnessing-the-universal-geometry-of-embeddings` | Harnessing the Universal Geometry of Embeddings | https://arxiv.org/abs/2505.12540v4
- [x] `transformers-represent-belief-state-geometry-in-their-residual-stream` | Transformers Represent Belief State Geometry in their Residual Stream | https://arxiv.org/abs/2405.15943v3
- [x] `layer-by-layer-uncovering-hidden-representations-in-language-models` | Layer by Layer: Uncovering Hidden Representations in Language Models | https://arxiv.org/abs/2502.02013v2
- [x] `emergence-of-a-high-dimensional-abstraction-phase-in-language-transformers` | Emergence of a High-Dimensional Abstraction Phase in Language Transformers | https://arxiv.org/abs/2405.15471v4
- [x] `systematic-outliers-in-large-language-models` | Systematic Outliers in Large Language Models | https://arxiv.org/abs/2502.06415v2
- [x] `the-non-linear-representation-dilemma-is-causal-abstraction-enough-for-mechanistic-interpretability` | The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for | https://arxiv.org/abs/2507.08802v2

### Wave 7
- [ ] `towards-universality-studying-mechanistic-similarity-across-language-model-architectures` | Towards Universality: Studying Mechanistic Similarity Across Language Model | https://arxiv.org/abs/2410.06672v2
- [ ] `transferring-linear-features-across-language-models-with-model-stitching` | Transferring Linear Features Across Language Models With Model Stitching | https://arxiv.org/abs/2506.06609v3
- [ ] `deep-sequence-models-tend-to-memorize-geometrically-it-is-unclear-why` | Deep sequence models tend to memorize geometrically; it is unclear why | https://arxiv.org/abs/2510.26745v2
- [ ] `from-tokens-to-words-on-the-inner-lexicon-of-llms` | From Tokens to Words: On the Inner Lexicon of LLMs | https://arxiv.org/abs/2410.05864v4
- [ ] `monitoring-latent-world-states-in-language-models-with-propositional-probes` | Monitoring Latent World States in Language Models with Propositional Probes | https://arxiv.org/abs/2406.19501v2
- [ ] `llms-know-more-than-they-show-on-the-intrinsic-representation-of-llm-hallucinations` | LLMs Know More Than They Show: On the Intrinsic Representation of LLM | https://arxiv.org/abs/2410.02707v4

### Wave 8
- [ ] `lines-of-thought-in-large-language-models` | Lines of Thought in Large Language Models | https://arxiv.org/abs/2410.01545v3
- [ ] `the-same-but-different-structural-similarities-and-differences-in-multilingual-language-modeling` | The Same but Different: Structural Similarities and Differences in | https://arxiv.org/abs/2410.09223v1
- [ ] `programming-refusal-with-conditional-activation-steering` | Programming Refusal with Conditional Activation Steering | https://arxiv.org/abs/2409.05907v3
- [ ] `angular-steering-behavior-control-via-rotation-in-activation-space` | Angular Steering: Behavior Control via Rotation in Activation Space | https://arxiv.org/abs/2510.26243v1
- [ ] `representation-surgery-theory-and-practice-of-affine-steering` | Representation Surgery: Theory and Practice of Affine Steering | https://arxiv.org/abs/2402.09631v7
- [ ] `controlling-language-and-diffusion-models-by-transporting-activations` | Controlling Language and Diffusion Models by Transporting Activations | https://arxiv.org/abs/2410.23054v2

### Wave 9
- [ ] `analysing-the-generalisation-and-reliability-of-steering-vectors` | Analysing the Generalisation and Reliability of Steering Vectors | 
- [ ] `in-context-vectors-making-in-context-learning-more-effective-and-controllable-through-latent-space-steering` | In-context Vectors: Making In Context Learning More Effective and Controllable | https://arxiv.org/abs/2311.06668v3
- [ ] `improving-instruction-following-in-language-models-through-activation-steering` | Improving Instruction-Following in Language Models through Activation Steering | https://arxiv.org/abs/2410.12877v2
- [ ] `improved-representation-steering-for-language-models` | Improved Representation Steering for Language Models | https://arxiv.org/abs/2505.20809v1
- [ ] `belief-dynamics-reveal-the-dual-nature-of-in-context-learning-and-activation-steering` | Belief Dynamics Reveal the Dual Nature of In-Context Learning and Activation | https://arxiv.org/abs/2511.00617v1
- [ ] `controllable-context-sensitivity-and-the-knob-behind-it` | Controllable Context Sensitivity and the Knob Behind It | https://arxiv.org/abs/2411.07404v4

### Wave 10
- [ ] `adaptive-activation-steering-a-tuning-free-llm-truthfulness-improvement-method` | Adaptive Activation Steering: A Tuning-Free LLM Truthfulness Improvement Method for Diverse Hallucinations Categories | https://arxiv.org/pdf/2406.00034
- [ ] `axbench-steering-llms-even-simple-baselines-outperform-sparse-autoencoders` | AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders | https://arxiv.org/abs/2501.17148v3
- [ ] `steering-large-language-models-using-conceptors-improving-addition-based-activation-engineering` | Steering Large Language Models using Conceptors: Improving Addition-Based | https://arxiv.org/pdf/2410.16314
- [ ] `learning-to-grok-emergence-of-in-context-learning-and-skill-composition-in-modular-arithmetic-tasks` | Learning to Grok: Emergence of In-context Learning and Skill Composition in | https://arxiv.org/abs/2406.02550v2
- [ ] `beyond-induction-heads-in-context-meta-learning-induces-multi-phase-circuit-emergence` | Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit | https://arxiv.org/abs/2505.16694v2
- [ ] `competition-dynamics-shape-algorithmic-phases-of-in-context-learning` | Competition Dynamics Shape Algorithmic Phases of In-Context Learning | https://arxiv.org/abs/2412.01003v4

### Wave 11
- [ ] `iclr-in-context-learning-of-representations` | ICLR: In-Context Learning of Representations | https://arxiv.org/abs/2501.00070v2
- [ ] `a-percolation-model-of-emergence-analyzing-transformers-trained-on-a-formal-language` | A Percolation Model of Emergence: Analyzing Transformers Trained on a Formal | https://arxiv.org/abs/2408.12578v2
- [ ] `emergence-of-hidden-capabilities-exploring-learning-dynamics-in-concept-space` | Emergence of Hidden Capabilities: Exploring Learning Dynamics in Concept Space | https://arxiv.org/abs/2406.19370v4
- [ ] `differential-learning-kinetics-govern-the-transition-from-memorization-to-generalization` | Differential Learning Kinetics Govern the Transition from Memorization to | https://arxiv.org/abs/2412.00104v2
- [ ] `physics-of-language-models-part-2-1-grade-school-math-and-the-hidden-reasoning-process` | Physics of Language Models: Part 2.1, Grade-School Math and the Hidden | https://arxiv.org/abs/2407.20311v1
- [ ] `physics-of-language-models-part-2-2-how-to-learn-from-mistakes-on-grade-school-math-problems` | Physics of Language Models: Part 2.2, How to Learn From Mistakes on | https://arxiv.org/abs/2408.16293v1

### Wave 12
- [ ] `polypythias-stability-and-outliers-across-fifty-language-model-pre-training-runs` | PolyPythias: Stability and Outliers across Fifty Language Model Pre-Training | https://arxiv.org/abs/2503.09543v2
- [ ] `transformers-struggle-to-learn-to-search` | Transformers Struggle to Learn to Search | https://arxiv.org/abs/2412.04703v2
- [ ] `arithmetic-without-algorithms-language-models-solve-math-with-a-bag-of-heuristics` | Arithmetic Without Algorithms: Language Models Solve Math with a Bag of | https://arxiv.org/abs/2410.21272v2
- [ ] `answer-assemble-ace-understanding-how-lms-answer-multiple-choice-questions` | Answer, Assemble, Ace: Understanding How LMs Answer Multiple Choice Questions | https://arxiv.org/abs/2407.15018v2
- [ ] `attention-as-a-hypernetwork` | Attention as a Hypernetwork | https://arxiv.org/abs/2406.05816v4
- [ ] `interpreting-emergent-planning-in-model-free-reinforcement-learning` | Interpreting Emergent Planning in Model-Free Reinforcement Learning | https://arxiv.org/abs/2504.01871v1

### Wave 13
- [ ] `on-the-role-of-attention-heads-in-large-language-model-safety` | On the Role of Attention Heads in Large Language Model Safety | https://arxiv.org/abs/2410.13708v2
- [ ] `robust-llm-safeguarding-via-refusal-feature-adversarial-training` | Robust LLM Safeguarding via Refusal Feature Adversarial Training | https://arxiv.org/abs/2409.20089v2
- [ ] `does-refusal-training-in-llms-generalize-to-the-past-tense` | Does Refusal Training in LLMs Generalize to the Past Tense? | https://arxiv.org/abs/2407.11969v4
- [ ] `backtracking-improves-generation-safety` | Backtracking Improves Generation Safety | https://arxiv.org/abs/2409.14586v1
- [ ] `ai-sandbagging-language-models-can-strategically-underperform-on-evaluations` | AI Sandbagging: Language Models can Strategically Underperform on Evaluations | https://arxiv.org/abs/2406.07358v4
- [ ] `why-do-some-language-models-fake-alignment-while-others-dont` | Why Do Some Language Models Fake Alignment While Others Don't? | https://arxiv.org/abs/2506.18032v1

### Wave 14
- [ ] `the-hawthorne-effect-in-reasoning-models-evaluating-and-steering-test-awareness` | The Hawthorne Effect in Reasoning Models: Evaluating and Steering Test | https://arxiv.org/abs/2505.14617v3
- [ ] `language-models-learn-to-mislead-humans-via-rlhf` | Language Models Learn to Mislead Humans via RLHF | https://arxiv.org/abs/2409.12822v3
- [ ] `on-targeted-manipulation-and-deception-when-optimizing-llms-for-user-feedback` | On Targeted Manipulation and Deception when Optimizing LLMs for User Feedback | https://arxiv.org/abs/2411.02306v3
- [ ] `unlearning-or-obfuscating-jogging-the-memory-of-unlearned-llms-via-benign-relearning` | Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign | https://arxiv.org/abs/2406.13356v4
- [ ] `a-common-pitfall-of-margin-based-language-model-alignment-gradient-entanglement` | A Common Pitfall of Margin-based Language Model Alignment: Gradient | https://arxiv.org/abs/2410.13828v2
- [ ] `alphaedit-null-space-constrained-knowledge-editing-for-language-models` | AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models | https://arxiv.org/abs/2410.02355v4

### Wave 15
- [ ] `knowledge-localization-mission-not-accomplished-enter-query-localization` | Knowledge Localization: Mission Not Accomplished? Enter Query Localization! | https://arxiv.org/abs/2405.14117v2
- [ ] `a-probabilistic-perspective-on-unlearning-and-alignment-for-large-language-models` | A Probabilistic Perspective on Unlearning and Alignment for Large Language | https://arxiv.org/abs/2410.03523v6
- [ ] `adversarial-perturbations-cannot-reliably-protect-artists-from-generative-ai` | Adversarial Perturbations Cannot Reliably Protect Artists From Generative AI | https://arxiv.org/abs/2406.12027v2
- [ ] `ai-as-humanitys-salieri-quantifying-linguistic-creativity-of-language-models` | AI as Humanity's Salieri: Quantifying Linguistic Creativity of Language Models | https://arxiv.org/abs/2410.04265v2
- [ ] `can-llms-separate-instructions-from-data-and-what-do-we-even-mean-by-that` | Can LLMs Separate Instructions From Data? And What Do We Even Mean By That? | https://arxiv.org/abs/2403.06833v3
- [ ] `evolutionary-strategies-lead-to-catastrophic-forgetting-in-llms` | Evolutionary Strategies lead to Catastrophic Forgetting in LLMs | https://arxiv.org/abs/2601.20861v1

### Wave 16
- [ ] `weird-generalization-and-inductive-backdoors-new-ways-to-corrupt-llms` | Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs | https://arxiv.org/pdf/2512.09742
- [ ] `the-universal-weight-subspace-hypothesis` | The universal weight subspace hypothesis | https://arxiv.org/abs/2512.05117v2
- [ ] `verbalized-sampling-how-to-mitigate-mode-collapse-and-unlock-llm-diversity` | Verbalized Sampling: How To Mitigate Mode Collapse And Unlock LLM Diversity | https://arxiv.org/abs/2510.01171v3
- [ ] `vlms-can-aggregate-scattered-training-patches` | VLMs Can Aggregate Scattered Training Patches | https://arxiv.org/abs/2506.03614v1
- [ ] `data-shapley-in-one-training-run` | Data Shapley in One Training Run | https://arxiv.org/abs/2406.11011v3
- [ ] `capturing-the-temporal-dependence-of-training-data-influence` | Capturing the Temporal Dependence of Training Data Influence | https://arxiv.org/abs/2412.09538v1

### Wave 17
- [ ] `scalable-influence-and-fact-tracing-for-large-language-model-pretraining` | Scalable Influence and Fact Tracing for Large Language Model Pretraining | https://arxiv.org/abs/2410.17413v3
- [ ] `a-snapshot-of-influence-a-local-data-attribution-framework-for-online-reinforcement-learning` | A Snapshot of Influence: A Local Data Attribution Framework for Online | https://arxiv.org/abs/2505.19281v2
- [ ] `scalable-extraction-of-training-data-from-aligned-production-language-models` | Scalable Extraction of Training Data from Aligned, Production Language Models | 
- [ ] `redeep-detecting-hallucination-in-retrieval-augmented-generation-via-mechanistic-interpretability` | ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via | https://arxiv.org/abs/2410.11414v2
- [ ] `internal-causal-mechanisms-robustly-predict-language-model-out-of-distribution-behaviors` | Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution | https://arxiv.org/abs/2505.11770v2
- [ ] `walk-the-talk-measuring-the-faithfulness-of-large-language-model-explanations` | Walk the Talk? Measuring the Faithfulness of Large Language Model Explanations | https://arxiv.org/abs/2504.14150v2

### Wave 18
- [ ] `inverse-constitutional-ai-compressing-preferences-into-principles` | Inverse Constitutional AI: Compressing Preferences into Principles | https://arxiv.org/abs/2406.06560v2
- [ ] `how-new-data-permeates-llm-knowledge-and-how-to-dilute-it` | How New Data Permeates LLM Knowledge and How to Dilute It | https://arxiv.org/abs/2504.09522v1
- [ ] `cure-concept-unlearning-via-orthogonal-representation-editing-in-diffusion-models` | CURE: Concept Unlearning via Orthogonal Representation Editing in Diffusion | https://arxiv.org/abs/2505.12677v2
- [ ] `do-natural-language-descriptions-of-model-activations-convey-privileged-information` | Do Natural Language Descriptions Of Model Activations Convey Privileged | https://arxiv.org/abs/2509.13316v3
- [ ] `interpreting-clips-image-representation-via-text-based-decomposition` | Interpreting CLIP's Image Representation via Text-Based Decomposition | https://arxiv.org/abs/2310.05916v4
- [ ] `interpreting-the-second-order-effects-of-neurons-in-clip` | Interpreting the Second-Order Effects of Neurons in CLIP | https://arxiv.org/abs/2406.04341v3

### Wave 19
- [ ] `conceptattention-diffusion-transformers-learn-highly-interpretable-features` | ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features | https://arxiv.org/abs/2502.04320v2
- [ ] `forking-paths-in-neural-text-generation` | Forking Paths in Neural Text Generation | https://arxiv.org/abs/2412.07961v1
- [ ] `monet-mixture-of-monosemantic-experts-for-transformers` | Monet: Mixture of Monosemantic Experts for Transformers | https://arxiv.org/abs/2412.04139v4

## Phase 2: top-conference scan, ICLR/ICML/NeurIPS 2025–2026 (added 2026-11-03)

Source: venue scan on 2026-11-03 (papernotes interpretability indexes for ICLR26/ICML25/ICML26/NeurIPS25/NeurIPS26, plus the ICLR26 oral mirror at iclr-2026-oral.vercel.app/topic/interpretability). These papers are **not** among the 111 kept stubs, and no post exists for any of them anywhere in `content/blog/`. A grep of the corpus on 2026-11-03 found no title collision.

Same spec as Phase 1 (3–4k words, 7–9 figs, ≥3 extracted from the paper, no em dashes). Differences:
- **No stub file exists.** The drafting agent creates `content/blog/paper-reading/ai-interpretability/<slug>.md`. Slugs below are provisional: if the real paper title differs, slug it from the real title. Before writing, run `ls` on the folder to confirm the post isn't there.
- **arXiv links still need resolving.** At dispatch, the orchestrator does one WebSearch per paper to get the arXiv/OpenReview URL and the authors, then writes the URL into the line here. If a paper can't be found, swap in the next unclaimed paper from a later wave.
- **Venue tag.** It comes from the scan. Oral status is confirmed only for the five ICLR26 orals (from an unofficial mirror). Lines marked `(verify)` were recalled from memory, not scanned, so confirm the venue first.
- Priority order: W20–W21 are the "most original idea" picks. W22–W29 are thematic clusters. W30 is the pre-2025 classics backlog.

### Wave 20 (top picks, part 1)
- [ ] `addressing-divergent-representations-from-causal-interventions-on-neural-networks` | Addressing Divergent Representations from Causal Interventions on Neural Networks | ICLR26 **oral** | patching/steering pushes the model off-distribution, which undercuts interp conclusions
- [ ] `verifying-chain-of-thought-reasoning-via-its-computational-graph` | Verifying Chain-of-Thought Reasoning via Its Computational Graph | ICLR26 **oral** | attribution graphs as execution traces
- [ ] `all-circuits-lead-to-rome-rethinking-functional-anisotropy-in-circuit-and-sheaf-discovery` | All Circuits Lead to Rome | ICML26 | many near-disjoint circuits per task
- [ ] `mib-a-mechanistic-interpretability-benchmark` | MIB: A Mechanistic Interpretability Benchmark | ICML25 | SAE features don't beat neurons for localization
- [ ] `the-price-of-amortized-inference-in-sparse-autoencoders` | The Price of Amortized Inference in SAEs | ICLR26 | dead/dense latents + absorption come from the shared encoder; LocA-SAE
- [ ] `automated-interpretability-metrics-do-not-distinguish-trained-and-random-transformers` | Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers | ICLR26

### Wave 21 (top picks, part 2)
- [ ] `decomposing-llm-computation-with-jets` | Decomposing LLM Computation with Jets | ICLR26 | Taylor-jet path expansion; logit lens is a special case
- [ ] `discovering-interpretable-algorithms-by-decompiling-transformers-to-rasp` | Decompiling Transformers to RASP | ICML26
- [ ] `llms-process-lists-with-general-filter-heads` | LLMs Process Lists With General Filter Heads | ICLR26
- [ ] `learning-to-interpret-weight-differences-in-language-models` | Learning to Interpret Weight Differences | ICLR26
- [ ] `sparse-autoencoders-are-topic-models` | Sparse Autoencoders are Topic Models | ICML26
- [ ] `can-circuit-alignment-predict-ood-generalization` | Can Circuit Alignment Predict OOD Generalization? | NeurIPS26

### Wave 22 (circuit theory and verification)
- [ ] `formal-mechanistic-interpretability` | Formal Mechanistic Interpretability (NN verification → provable circuits) | ICLR26
- [ ] `certified-circuits-stability-guarantees-for-mechanistic-circuits` | Certified Circuits | ICML26
- [ ] `scalable-circuit-learning-for-interpreting-large-language-models` | Scalable Circuit Learning (Lasso on observational data) | ICML26
- [ ] `circuit-fingerprints-how-answer-tokens-encode-their-geometrical-path` | Circuit Fingerprints | ICML26
- [ ] `beyond-components-singular-vector-based-interpretability-of-transformer-circuits` | Beyond Components: Singular Vector-Based Interpretability | NeurIPS25
- [ ] `validating-mechanistic-interpretations-an-axiomatic-approach` | Validating Mechanistic Interpretations: An Axiomatic Approach | ICML25

### Wave 23 (mechanisms inside LLMs)
- [ ] `language-models-use-lookbacks-to-track-beliefs` | Language Models Use Lookbacks to Track Beliefs | ICLR26
- [ ] `mixing-mechanisms-how-language-models-retrieve-bound-entities-in-context` | Mixing Mechanisms | ICLR26
- [ ] `latent-planning-emerges-with-scale` | Latent Planning Emerges with Scale (transcoder circuits, Qwen-3) | ICLR26
- [ ] `wheres-the-plan-locating-latent-planning-in-language-models` | Where's the Plan? | ICML26
- [ ] `how-language-models-process-negation` | How Language Models Process Negation | ICML26
- [ ] `the-discrete-log-clock-how-a-transformer-learns-modular-multiplication` | The Discrete-Log Clock | ICML26

### Wave 24 (SAE theory and critique)
- [ ] `sparse-autoencoders-trained-on-the-same-data-learn-different-features` | SAEs Trained on the Same Data Learn Different Features | ICLR26
- [ ] `towards-atoms-of-large-language-models` | Towards Atoms of Large Language Models | ICML26
- [ ] `temporal-sparse-autoencoders` | Temporal SAEs: Leveraging the Sequential Nature of Language | ICLR26 **oral** (pair with "Priors in Time", ICLR26)
- [ ] `taming-polysemanticity-in-llms` | Taming Polysemanticity in LLMs (frequency resonance, GBA) | ICLR26
- [ ] `on-the-limits-of-sparse-autoencoders` | On the Limits of Sparse Autoencoders (closed form + WSAE) | ICLR26
- [ ] `transformer-key-value-memories-are-nearly-as-interpretable-as-sparse-autoencoders` | KV Memories ≈ SAEs | NeurIPS25

### Wave 25 (model diffing: reading what fine-tuning changed)
- [ ] `narrow-finetuning-leaves-clearly-readable-traces-in-activation-differences` | Narrow Finetuning Leaves Clearly Readable Traces | ICLR26
- [ ] `persona-features-control-emergent-misalignment` | Persona Features Control Emergent Misalignment | ICLR26
- [ ] `watch-the-weights-unsupervised-monitoring-and-control-of-fine-tuned-llms` | Watch the Weights | ICLR26
- [ ] `block-em-preventing-emergent-misalignment-via-latent-blocking` | BLOCK-EM | ICML26
- [ ] `reinforcement-learning-fine-tuning-enhances-activation-intensity-and-diversity` | RL Fine-Tuning Enhances Activation Intensity and Diversity | ICLR26
- [ ] `emergent-misalignment-narrow-finetuning-can-produce-broadly-misaligned-llms` | Emergent Misalignment | ICML25 (verify)

### Wave 26 (attention and architecture components)
- [ ] `attention-sinks-and-compression-valleys-are-two-sides-of-the-same-coin` | Attention Sinks and Compression Valleys | ICLR26
- [ ] `towards-understanding-the-nature-of-attention-with-low-rank-sparse-decomposition` | Lorsa | ICLR26
- [ ] `singular-vectors-of-attention-heads-align-with-features` | Singular Vectors of Attention Heads Align with Features | ICML26
- [ ] `causal-head-gating-interpreting-roles-of-attention-heads` | Causal Head Gating | NeurIPS25
- [ ] `the-achilles-heel-of-llms-a-few-neurons-collapse-the-model` | The Achilles' Heel of LLMs (~3 neurons) | ICLR26
- [ ] `small-transformers-dont-need-layernorm` | Small Transformers Don't Need LayerNorm | ICLR26

### Wave 27 (representation theory)
- [ ] `language-models-are-injective-and-hence-invertible` | Language Models are Injective and Hence Invertible | ICLR26
- [ ] `i-predict-therefore-i-am-linear-representations-of-concept-posteriors` | I Predict Therefore I Am | ICLR26
- [ ] `why-linear-interpretability-works-invariant-subspaces` | Why Linear Interpretability Works | ICML26
- [ ] `causality-is-not-invariance-function-vectors-vs-concept-vectors` | Causality ≠ Invariance | ICLR26
- [ ] `parameter-symmetries-determine-representational-geometry` | Parameter Symmetries Determine Representational Geometry | NeurIPS26
- [ ] `temporal-superposition-and-feature-geometry-of-rnns-under-memory-demands` | Temporal Superposition and Feature Geometry of RNNs | ICLR26 **oral**

### Wave 28 (steering and reading activations)
- [ ] `activation-steering-with-a-feedback-controller` | Activation Steering with a Feedback Controller (PID) | ICLR26
- [ ] `cot-vectors-transferring-chain-of-thought-via-activation-vectors` | CoT Vectors | ICLR26
- [ ] `understanding-task-vectors-in-in-context-learning` | Understanding Task Vectors in ICL (rank-1 limit) | ICLR26
- [ ] `latentqa-teaching-llms-to-decode-activations-into-natural-language` | LatentQA | ICLR26
- [ ] `base-models-know-how-to-reason-thinking-models-learn-when` | Base Models Know How to Reason, Thinking Models Learn When | NeurIPS25
- [ ] `to-steer-or-not-to-steer-mera` | To Steer or Not to Steer? (MERA) | ICML25

### Wave 29 (miscellaneous originals)
- [ ] `thought-branches-interpreting-llm-reasoning-requires-resampling` | Thought Branches | ICLR26
- [ ] `fresh-in-memory-training-order-recency-is-linearly-encoded` | Fresh in Memory | ICLR26
- [ ] `the-expert-strikes-back-interpreting-mixture-of-experts-at-expert-level` | The Expert Strikes Back | ICML26
- [ ] `dynamic-weight-grafting-how-llms-retrieve-new-facts` | Dynamic Weight Grafting | ICLR26
- [ ] `exploratory-causal-inference-in-saence` | Exploratory Causal Inference in SAEnce | ICLR26 **oral**
- [ ] `inference-time-decomposition-of-activations-itda` | ITDA (matching pursuit instead of SAE training, 405B) | ICML25

### Wave 30 (pre-2025 classics backlog: not in corpus as of 2026-11-03)
- [ ] `interpretability-in-the-wild-indirect-object-identification-circuit-in-gpt2` | IOI | ICLR23
- [ ] `progress-measures-for-grokking-via-mechanistic-interpretability` | Progress Measures for Grokking | ICLR23
- [ ] `emergent-world-representations-othello-gpt` | Emergent World Representations (Othello-GPT) | ICLR23
- [ ] `locating-and-editing-factual-associations-in-gpt-rome` | ROME | NeurIPS22
- [ ] `towards-automated-circuit-discovery-for-mechanistic-interpretability` | ACDC | NeurIPS23
- [ ] `refusal-in-language-models-is-mediated-by-a-single-direction` | Refusal Is Mediated by a Single Direction | NeurIPS24

### Reserve (swap in if a paper above can't be found)
Mixture of Decoders (NeurIPS25) · MP-SAE (NeurIPS25) · Bigram Subnetworks (NeurIPS25) · AbsTopK (ICLR26) · Priors in Time (ICLR26) · Token Alignment Heads (ICLR26) · Deleuzian Representation Hypothesis (ICLR26) · NeuronEval (ICML25) · PersonaManifold (NeurIPS26) · Deep Minds and Shallow Probes (NeurIPS26) · Query Circuits (ICML26) · Hedonic Neurons (ICLR26) · Residual Stream Analysis with Multi-Layer SAEs (ICLR25, verify). NeurIPS26 (verify; from an arXiv digest, not the official list): Semantic Optimal Transport for SAE Feature Matching and Circuit Compression · DifFRACT: Diffusion Feature Reconstruction and Attribution for Circuit Tracing · Taming CoT Obfuscation in VLMs: From Mechanistic Evidence to Activation-Level Enforcement.

NeurIPS26 coverage: done 2026-11-03 from the official list (Waves 31–35 below).

## Phase 2b: NeurIPS 2026 official-list scan (added 2026-11-03)

Source: https://neurips.cc/virtual/2026/loc/sydney/papers.html. The page lists ~9,130 posters; keyword filtering left 362 interpretability-related titles, curated below. Acceptance is confirmed (official list); poster URLs are on neurips.cc/virtual/2026/poster/<id>, so resolve arXiv at dispatch. Same Phase 2 rules (new file, provisional slug, check folder first). "Can Circuit Alignment Predict OOD Generalization?" (W21) is confirmed NeurIPS26.

### Wave 31 (NeurIPS26: interp methodology under attack)
- [ ] `the-dead-salmons-of-ai-interpretability-the-need-for-a-statistical-inference-perspective` | The Dead Salmons of AI Interpretability: The Need for a Statistical Inference Perspective | NeurIPS26
- [ ] `sanity-checks-for-sparse-autoencoders-do-saes-beat-random-baselines` | Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines? | NeurIPS26
- [ ] `is-a-linear-probe-evidence-of-a-linear-representation` | Is a Linear Probe Evidence of a Linear Representation? | NeurIPS26
- [ ] `the-curse-of-multiple-mediators-hidden-interaction-effects-in-activation-patching` | The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching | NeurIPS26
- [ ] `patchbench-measuring-collateral-damage-in-activation-patching` | PatchBench: Measuring Collateral Damage in Activation Patching | NeurIPS26
- [ ] `conclusions-from-circuit-extraction-depend-on-the-level-of-description-a-controlled-compar` | Conclusions from Circuit Extraction Depend on the Level of Description: A Controlled Comparison | NeurIPS26

### Wave 32 (NeurIPS26: circuits)
- [ ] `cross-model-circuit-discovery` | Cross-Model Circuit Discovery | NeurIPS26
- [ ] `many-circuits-one-mechanism-input-variation-and-evaluation-granularity-in-circuit-discover` | Many Circuits, One Mechanism: Input Variation and Evaluation Granularity in Circuit Discovery | NeurIPS26
- [ ] `finding-interpretable-prompt-specific-circuits-in-language-models` | Finding Interpretable Prompt-Specific Circuits in Language Models | NeurIPS26
- [ ] `the-commit-abstain-circuit-why-language-models-hallucinate-instead-of-abstaining` | The Commit-Abstain Circuit: Why Language Models Hallucinate Instead of Abstaining | NeurIPS26
- [ ] `diffract-diffusion-feature-reconstruction-and-attribution-for-circuit-tracing` | DifFRACT: Diffusion Feature Reconstruction and Attribution for Circuit Tracing | NeurIPS26
- [ ] `navigating-by-old-maps-the-pitfalls-of-static-mechanistic-localization-in-llm-post-trainin` | Navigating by Old Maps: The Pitfalls of Static Mechanistic Localization in LLM Post-Training | NeurIPS26

### Wave 33 (NeurIPS26: features, SAEs, superposition)
- [ ] `the-parameters-in-weight-sparse-transformers-are-interpretable` | The parameters in weight-sparse transformers are interpretable | NeurIPS26
- [ ] `what-does-a-sparse-autoencoder-feature-do-a-weight-based-account` | What Does a Sparse Autoencoder Feature Do? A Weight-Based Account | NeurIPS26
- [ ] `the-rate-distortion-polysemanticity-tradeoff-in-saes` | The Rate-Distortion-Polysemanticity Tradeoff in SAEs | NeurIPS26
- [ ] `probing-for-representation-manifolds-in-superposition` | Probing for Representation Manifolds in Superposition | NeurIPS26
- [ ] `bypassing-pc1-makes-saes-more-reproducible` | Bypassing PC1 Makes SAEs More Reproducible | NeurIPS26
- [ ] `idempotency-exposes-consistency-problems-in-sparse-autoencoders` | Idempotency Exposes Consistency Problems in Sparse Autoencoders | NeurIPS26

### Wave 34 (NeurIPS26: mechanisms and theory)
- [ ] `subliminal-learning-is-steering-vector-distillation` | Subliminal Learning Is Steering Vector Distillation | NeurIPS26
- [ ] `how-are-linear-representations-learned-exact-solutions-to-the-dynamics-of-abstraction` | How are linear representations learned? Exact solutions to the dynamics of abstraction | NeurIPS26
- [ ] `a-single-neuron-is-sufficient-to-bypass-safety-alignment-in-large-language-models` | A Single Neuron Is Sufficient to Bypass Safety Alignment in Large Language Models | NeurIPS26
- [ ] `the-superactivator-mechanism-transformers-concentrate-reliable-concept-signals-in-the-tail` | The SuperActivator Mechanism: Transformers Concentrate Reliable Concept Signals in the Tail | NeurIPS26
- [ ] `state-copying-crowds-out-reasoning-mechanistic-evidence-for-delta-planning-in-autoregressi` | State Copying Crowds Out Reasoning: Mechanistic Evidence for Delta Planning in Autoregressive Models | NeurIPS26
- [ ] `neural-networks-are-more-modular-than-single-neurons-suggest` | Neural networks are more modular than single neurons suggest | NeurIPS26

### Wave 35 (NeurIPS26: model diffing, weight-space interp, steering)
- [ ] `transcoder-adapters-for-reasoning-model-diffing` | Transcoder Adapters for Reasoning-Model Diffing | NeurIPS26
- [ ] `mmdiff-multimodal-model-diffing-for-feature-discovery-and-control` | MMDiff: Multimodal Model Diffing for Feature Discovery and Control | NeurIPS26
- [ ] `loracles-self-supervised-weight-space-interpretability-at-scale` | LoRAcles: Self-Supervised Weight-Space Interpretability at Scale | NeurIPS26
- [ ] `predictive-concept-decoders-training-scalable-end-to-end-interpretability-assistants` | Predictive Concept Decoders: Training Scalable End-to-End Interpretability Assistants | NeurIPS26
- [ ] `steering-vectors-as-a-training-signal-in-llm-post-training` | Steering Vectors as a Training Signal in LLM Post-Training | NeurIPS26
- [ ] `the-sharp-directions-are-against-you-curvature-analysis-of-activation-steering` | The Sharp Directions Are Against You: Curvature Analysis of Activation Steering | NeurIPS26

### NeurIPS26 reserve
A Mechanistic Analysis of Looped Reasoning Language Models · Inside the Loop: A Mechanistic Study of Weight-Tied Transformers on Depth-Bound Algorithmic Tasks · Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks · A Mechanistic Investigation of Theory of Mind in a Large Language Model · Tracing Persona Vectors Through LLM Pretraining · Emergent Misalignment as Data-Mediated Transfer · Conditional misalignment: common interventions can hide emergent misalignment behind contextual triggers · Mechanistic Interpretability Needs Philosophy · Attributions All the Way Down? The Metagame of Interpretability · Causal Abstractions, Categorically Unified · Parasite Features: Causal Abstraction Through Spurious Pathways · Structural Support Certificates for Mechanistic Hypothesis Selection · Same Function, Different Mechanism: Implementation-Local Geometry Beyond Accuracy · Neuron Populations Exhibit Divergent Selectivity with Scale · fmxcoders: Factorized Masked Crosscoders for Cross-Layer Feature Discovery · Matryoshka Transcoders and Hierarchy Misalignment: When SAE Absorption Protection Does Not Transfer · The Unembedding Bottleneck: A Mechanistic Analysis of Single Digit Counting in LLMs · Position Without Positional Embeddings: A Directional Mechanism in NoPE Transformers · Attention Sinks as Spectral Spikes: A Mechanism Analysis of Gated Attention · The Mechanism of Weak-to-Strong Generalization: Feature Elicitation from Latent Knowledge · What Do SAE Features Encode? Evidence from Human Neural Activity · Two Stages of Folding: Convergent Mechanisms in AI Protein Folding Trunks · Semantic Optimal Transport for Sparse Autoencoder Feature Matching and Circuit Compression · Taming CoT Obfuscation in VLMs: From Mechanistic Evidence to Activation-Level Enforcement · Intrinsically Interpretable Attention via Sparse Post-Training · LLM Rheology: Auditing Refusal Geometry in Aligned Language Models
