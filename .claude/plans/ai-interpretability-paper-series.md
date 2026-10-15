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
- [ ] `learning-multi-level-features-with-matryoshka-sparse-autoencoders` | Learning Multi-Level Features with Matryoshka Sparse Autoencoders | https://arxiv.org/abs/2503.17547v1
- [ ] `dense-sae-latents-are-features-not-bugs` | Dense SAE Latents Are Features, Not Bugs | https://arxiv.org/abs/2506.15679v2
- [ ] `sparse-autoencoders-do-not-find-canonical-units-of-analysis` | Sparse Autoencoders Do Not Find Canonical Units of Analysis | https://arxiv.org/abs/2502.04878v1
- [ ] `are-sparse-autoencoders-useful-a-case-study-in-sparse-probing` | Are Sparse Autoencoders Useful? A Case Study in Sparse Probing | https://arxiv.org/abs/2502.16681v1
- [ ] `universal-sparse-autoencoders-interpretable-cross-model-concept-alignment` | Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment | https://arxiv.org/abs/2502.03714v1
- [ ] `do-i-know-this-entity-knowledge-awareness-and-hallucinations-in-language-models` | Do I Know This Entity? Knowledge Awareness and Hallucinations in Language | https://arxiv.org/abs/2411.14257v2

### Wave 3
- [ ] `sparse-autoencoders-reveal-temporal-difference-learning-in-large-language-models` | Sparse Autoencoders Reveal Temporal Difference Learning in Large Language | https://arxiv.org/abs/2410.01280v1
- [ ] `finding-transformer-circuits-with-edge-pruning` | Finding Transformer Circuits With Edge Pruning | https://arxiv.org/abs/2406.16778v3
- [ ] `hypothesis-testing-the-circuit-hypothesis-in-llms` | Hypothesis Testing the Circuit Hypothesis in LLMs | https://arxiv.org/abs/2410.13032v1
- [ ] `rethinking-circuit-completeness-in-language-models-and-or-and-adder-gates` | Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates | https://arxiv.org/abs/2505.10039v2
- [ ] `llm-circuit-analyses-are-consistent-across-training-and-scale` | LLM Circuit Analyses Are Consistent Across Training and Scale | https://arxiv.org/abs/2407.10827v2
- [ ] `the-computational-complexity-of-circuit-discovery-for-inner-interpretability` | The Computational Complexity of Circuit Discovery for Inner Interpretability | https://arxiv.org/abs/2410.08025v3

### Wave 4
- [ ] `optimal-ablation-for-interpretability` | Optimal Ablation for Interpretability | https://arxiv.org/abs/2409.09951v1
- [ ] `bilinear-mlps-enable-weight-based-mechanistic-interpretability` | Bilinear MLPs Enable Weight-based Mechanistic Interpretability | https://arxiv.org/abs/2410.08417v2
- [ ] `a-implies-b-circuit-analysis-in-llms-for-propositional-logical-reasoning` | A Implies B: Circuit Analysis in LLMs for Propositional Logical Reasoning | https://arxiv.org/abs/2411.04105v4
- [ ] `everything-everywhere-all-at-once-is-mechanistic-interpretability-identifiable` | Everything, Everywhere, All at Once: Is Mechanistic Interpretability | https://arxiv.org/abs/2502.20914v1
- [ ] `hyperdas-towards-automating-mechanistic-interpretability-with-hypernetworks` | HyperDAS: Towards Automating Mechanistic Interpretability with Hypernetworks | https://arxiv.org/abs/2503.10894v3
- [ ] `the-geometry-of-categorical-and-hierarchical-concepts-in-large-language-models` | The Geometry of Categorical and Hierarchical Concepts in Large Language Models | https://arxiv.org/abs/2406.01506v3

### Wave 5
- [ ] `on-the-origins-of-linear-representations-in-large-language-models` | On the Origins of Linear Representations in Large Language Models | https://arxiv.org/abs/2403.03867v1
- [ ] `on-linear-representations-and-pretraining-data-frequency-in-language-models` | On Linear Representations and Pretraining Data Frequency in Language Models | https://arxiv.org/abs/2504.12459v1
- [ ] `emergence-of-linear-truth-encodings-in-language-models` | Emergence of Linear Truth Encodings in Language Models | https://arxiv.org/abs/2510.15804v1
- [ ] `linear-representations-of-political-perspective-emerge-in-large-language-models` | Linear Representations of Political Perspective Emerge in Large Language | https://arxiv.org/abs/2503.02080v2
- [ ] `the-semantic-hub-hypothesis-language-models-share-semantic-representations-across-languages-and-modalities` | The Semantic Hub Hypothesis: Language Models Share Semantic Representations | https://arxiv.org/abs/2411.04986v3
- [ ] `token-embeddings-violate-the-manifold-hypothesis` | Token Embeddings Violate the Manifold Hypothesis | https://arxiv.org/abs/2504.01002v3

### Wave 6
- [ ] `harnessing-the-universal-geometry-of-embeddings` | Harnessing the Universal Geometry of Embeddings | https://arxiv.org/abs/2505.12540v4
- [ ] `transformers-represent-belief-state-geometry-in-their-residual-stream` | Transformers Represent Belief State Geometry in their Residual Stream | https://arxiv.org/abs/2405.15943v3
- [ ] `layer-by-layer-uncovering-hidden-representations-in-language-models` | Layer by Layer: Uncovering Hidden Representations in Language Models | https://arxiv.org/abs/2502.02013v2
- [ ] `emergence-of-a-high-dimensional-abstraction-phase-in-language-transformers` | Emergence of a High-Dimensional Abstraction Phase in Language Transformers | https://arxiv.org/abs/2405.15471v4
- [ ] `systematic-outliers-in-large-language-models` | Systematic Outliers in Large Language Models | https://arxiv.org/abs/2502.06415v2
- [ ] `the-non-linear-representation-dilemma-is-causal-abstraction-enough-for-mechanistic-interpretability` | The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for | https://arxiv.org/abs/2507.08802v2

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
