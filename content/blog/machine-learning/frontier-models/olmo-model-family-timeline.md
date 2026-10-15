---
title: "The OLMo Family, Release by Release: From OLMo 1 to Olmo Hybrid and the Research Behind It"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "A chronological reference catalog of Ai2's OLMo family from February 2024 to October 2026: every model release, the Dolma and Dolci datasets, Tülu post-training, and the research papers that were run on OLMo checkpoints and shaped the next generation, each with its paper, blog post, code and weights."
tags: ["olmo", "olmo-3", "olmoe", "olmo-hybrid", "ai2", "allenai", "fully-open", "dolma", "tulu", "model-family", "timeline"]
category: "machine-learning"
subcategory: "Frontier Models"
author: "Hiep Tran"
featured: false
readTime: 55
---

Most open-weight model families are defined by their best checkpoint. OLMo is defined by everything around the checkpoint. When the Allen Institute for AI (Ai2) released OLMo 7B on February 1, 2024, the press line was not a benchmark score. It was that the pretraining data, the data-processing code, the training code, the logs and more than 500 intermediate checkpoints per model were all public, under Apache 2.0. Two and a half years later that has not changed, and it has turned OLMo into something no other family quite is: a shared laboratory. A large share of Ai2's language-model research papers are experiments run on OLMo checkpoints, and a large share of each new OLMo's recipe is the conclusion of one of those papers.

That makes OLMo hard to follow release by release, because half of the story is not in the model announcements. The z-loss and initialization changes in OLMo 2 trace back to a study of OLMo 0724's checkpoints. Olmo 3's batch-size schedule traces back to a paper on critical batch size measured on OLMo 1B and 7B. Olmo Hybrid exists as a controlled comparison against Olmo 3 7B, and OlmPool is a 26-model autopsy of the architecture choices Olmo 3 made. This post is the map. It walks every model release in chronological order, puts each research paper next to the release it used or influenced, and gives the paper, the Ai2 blog post, the code and the weights for every item, so each one can be read in full. The timeline below is the mental model: two lanes, models on top and the research and infrastructure that fed them underneath.

![OLMo family timeline from Q1 2024 to Q4 2026: model releases in the top lane, research and infrastructure in the bottom lane, ending with the next Olmo in pre-training](/imgs/blogs/olmo-model-family-timeline-1.webp)

One fact up front, because it is the question people ask most: **as of October 15, 2026, the newest released Olmo language models are Olmo 3.1 (December 2025) at 32B and Olmo Hybrid 7B (March 5, 2026).** There is no Olmo 4. Ai2's Olmo-core 3 post (October 1, 2026) says the next-generation Olmo "will use an MoE architecture," and the COLM 2026 recap says it is "now in pre-training with a hybrid mixture-of-experts architecture." Anything that gives it a name, a size or a date is guessing.

Three notes on conventions. First, naming: Ai2 wrote "OLMo" through OLMo 2 and switched to "Olmo" with Olmo 3; I follow whichever spelling the release itself used. Second, dates: where Ai2 published a blog post, its date is the release date; where only a Hugging Face repository exists (OLMo 0724, OLMo 2 1B), I say so and give the repo creation date, which is a lower bound. Third, sourcing: every section ends with a **Sources** line (paper, blog, code, weights). Ai2's blog posts sometimes link a *cited* paper rather than their own; I checked every arXiv ID against its title and flag the three cases where the post's link is wrong.

## The timeline at a glance

The whole family in one table. "Model" rows are checkpoints; "research", "data" and "infra" rows are the work that fed them. The research index at [allenai.org/research](https://allenai.org/research) lists everything Ai2 publishes, including climate, robotics and science-agent work; the rows below are only the items that touch the OLMo language-model line.

| Date | Release | Kind | One-line significance |
| --- | --- | --- | --- |
| 2024-02-01 | OLMo 1 (1B, 7B) + Dolma | Model + data | First "truly open" LLM: 3T-token corpus, code, logs, 500+ checkpoints per model |
| 2024-03-20 | RewardBench | Research | First benchmark for reward models; evaluated over 30 RMs |
| 2024-04-15 | Dolma moves to ODC-BY | Data | Replaces the ImpACT license with a standard open-data license |
| 2024-04-17 | OLMo 1.7-7B | Model | MMLU 52 (+24 points), 4096 context, Dolma 1.7, two-stage training |
| 2024-07 (Hub) | OLMo 0724 (1B, 7B) | Model | Refreshed 1B and 7B; the checkpoints later studied for stability |
| 2024-09-04 | OLMoE-1B-7B | Model | First OLMo MoE: 1B active, 7B total, 5T tokens |
| 2024-10-02 | Pretraining dynamics on OLMo checkpoints | Research | Norms shrink and early-layer outliers appear; recommends z-loss and better init |
| 2024-11-21 | Tülu 3 (8B, 70B) | Post-training | Open SFT, DPO and RLVR recipe; becomes the OLMo post-training pipeline |
| 2024-11-26 | OLMo 2 (7B, 13B) | Model | Stability fixes, Dolmino mid-training, model souping; up to 5T tokens |
| 2025-01-30 | Tülu 3 405B | Post-training | Recipe scaled to Llama 3.1 405B; RLVR on 256 GPUs |
| 2025-02-11 | OLMoE-1B-7B-0125 + iOS app | Model | Dolmino mid-training plus Tülu 3: 35% better on average; runs on an iPhone |
| 2025-03-13 | OLMo 2 32B | Model | First fully open model to beat GPT-3.5-Turbo and GPT-4o mini; trained on OLMo-core |
| 2025-04 (Hub) | OLMo 2 1B | Model | 4T tokens, 16 layers; later the base for Bolmo 1B |
| 2025-04-09 | OLMoTrace | Tool | Traces outputs back to trillions of training tokens in real time |
| 2025-04-15 | DataDecide | Research | 30k checkpoints over 25 corpora and 14 sizes: small runs pick the best data |
| 2025-06-03 | Critical batch size revisited | Research | Measures CBS directly; batch-size warmup gives 43% fewer gradient steps |
| 2025-07-09 | FlexOlmo | Model + research | Data owners contribute experts instead of data and can opt out later |
| 2025-08-19 | Signal and Noise | Research | Signal-to-noise ratio predicts which benchmarks are useful for decisions |
| 2025-09-16 | Fluid Benchmarking | Research | IRT-based adaptive evaluation: MMLU with 50x fewer items |
| 2025-11-20 | Olmo 3 (7B, 32B) | Model | Base, Think, Instruct, RL-Zero; Dolma 3 and Dolci; full "model flow" released |
| 2025-12-12 | Olmo 3.1 (32B Think, 32B Instruct) | Model | 21 more days of RL; IFBench +20 points |
| 2025-12-15 | Bolmo (1B, 7B) | Model + research | First fully open byte-level LMs, byteified from Olmo checkpoints |
| 2026-02-13 | Olmix | Research | Data-mixing framework with mixture reuse across a changing corpus |
| 2026-03-05 | Olmo Hybrid 7B | Model + research | Gated DeltaNet plus attention; MMLU parity with 49% fewer tokens |
| 2026-04-20 | BAR | Research | Post-train domain experts separately, merge into an MoE |
| 2026-04-23 | OlmPool | Research | 26 controlled 7B models: small architecture choices compound against long context |
| 2026-05-08 | EMO | Model + research | MoE pretrained for emergent, document-level modularity |
| 2026-06-12 | olmo-eval | Tool | Evaluation workbench, successor to OLMES |
| 2026-06-25 | Hybrid token-level study | Research | Which tokens Olmo Hybrid predicts better than Olmo 3 |
| 2026-10-01 | Olmo-core 3 | Infra | Open MoE training stack, tested up to 1.2T parameters |
| 2026-10-07 | Bolmo in Nature | Research | Byteifying generalizes to Qwen 3 and Llama 3 |

## What "fully open" means in OLMo

Before the releases, it is worth being precise about the word that every OLMo announcement leads with, because it is the reason the research lane in the timeline exists at all.

Think of a restaurant. An open-weights release hands you the finished dish and maybe a menu description. A fully open release also hands you the grocery receipts (the data), the kitchen's prep scripts (the data tools), the recipe card (the training code), photos of the dish at every stage of cooking (intermediate checkpoints), the tasting rubric (the evaluation code), and a way to ask "which ingredient made this taste like that" (OlmoTrace). You can eat the first dish. You can only *study cooking* with the second.

![Side-by-side stacks: a typical open-weights release ships weights and a model card, while Olmo 3 ships data, data tools, training code, intermediate checkpoints, evaluation and attribution as well](/imgs/blogs/olmo-model-family-timeline-2.webp)

The practical test is whether an outside researcher can rerun an experiment that Ai2 ran. With OLMo the answer has been yes from the first release, and the amount of what is open grew each generation:

| Artifact | OLMo 1 (Feb 2024) | OLMo 2 (Nov 2024) | Olmo 3 (Nov 2025) |
| --- | --- | --- | --- |
| Pretraining data | Dolma (3T tokens) + its processing code | OLMo-Mix-1124 (~3.9T) + Dolmino-Mix-1124 (843B) | Dolma 3 (~9.3T pool), Dolma 3 Mix (~5.9T), Dolmino, Longmino |
| Post-training data | Tulu 2 SFT mix | Tülu 3 SFT, preference and RLVR sets, OLMo-specific mixes | Dolci (SFT, DPO and RL mixes) |
| Data tools | Dolma toolkit, WIMBD | Same | datamap-rs, duplodocus, decon |
| Training code | OLMo repo | OLMo, then OLMo-core (for 32B) | OLMo-core, open-instruct |
| Checkpoints | 500+ per model (every 1000 steps) | Every stage, incl. souping ingredients | Every stage of every path |
| Evaluation | Catwalk, Paloma | OLMES (20 benchmarks, dev vs unseen split) | OLMES + OlmoBaseEval |
| Attribution | None | OLMoTrace (added April 2025) | OlmoTrace built into the Playground |
| License | Apache 2.0 (weights, code); Dolma under ImpACT, then ODC-BY | Apache 2.0, ODC-BY data | Apache 2.0, datasets "without any license restrictions" |

That last column is what makes the rest of this post possible. When the Olmo Hybrid team wanted to know whether a hybrid architecture beats a transformer, they could hold the data, tokenizer and recipe fixed against Olmo 3 7B, because all three were public. That kind of controlled comparison is what a laboratory is for. If you want the long version of why the data side matters more than the architecture side, the [data mixing post](/blog/machine-learning/training-data/data-mixing-domain-weighting-and-curriculum) is the companion read.

## OLMo 1 and Dolma (February 1, 2024)

The first release was two posts on the same day: a press announcement ("Hello OLMo: A truly open LLM") and a technical companion. The technical post is precise about what shipped: "four final variants of our language model at the 7B scale corresponding to different architectures, optimizers, and training hardware, and one model at the 1B scale, all trained on at least 2T tokens."

The architecture was a conservative decoder-only transformer with a few stability choices that later generations kept revisiting: "no biases are used (for stability, as in PaLM), the SwiGLU activation, Rotary positional embeddings (RoPE), and a modified version of the BPE-based tokenizer from GPT-NeoX-20B designed to reduce personally identifiable information." The layer norm was non-parametric, which matters later, because OLMo 2 replaced it.

The data was Dolma, a three-trillion-token open corpus first released in August 2023, published with the code that produced it. Training ran on two very different clusters: LUMI (AMD Instinct accelerators, through CSC) and the Databricks Mosaic AI platform. Partners included the Kempner Institute at Harvard, AMD, CSC, the UW Allen School and Databricks. The project leads quoted were Hanna Hajishirzi and Noah Smith, and they lead every generation since.

Results were honest rather than impressive: "on par with Llama 2 on many generative and reading comprehension tasks (such as truthfulQA), but is slightly behind on MMLU or Big-bench Hard." Nobody adopted OLMo 1 for production. What people adopted were the checkpoints: "500+ checkpoints per base model, from every 1000 steps," published as Hugging Face revisions, which made it the default subject for research on training dynamics, memorization and data attribution. An OLMo-7B-SFT and OLMo-7B-Instruct were trained with Open Instruct on the Tulu 2 mix, with DPO for the Instruct model.

| | OLMo 1B | OLMo 7B |
| --- | --- | --- |
| Training tokens | at least 2T | at least 2T (four 7B variants) |
| Context | 2048 | 2048 |
| Norm | Non-parametric LayerNorm | Non-parametric LayerNorm |
| Biases | None | None |
| Data | Dolma (1.5) | Dolma (1.5) |
| Checkpoints | 500+ | 500+ per variant |
| License | Apache 2.0 | Apache 2.0 |

**Sources.** Paper: Groeneveld et al., "OLMo: Accelerating the Science of Language Models", [arXiv 2402.00838](https://arxiv.org/abs/2402.00838). Data paper: Soldaini et al., "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research", [arXiv 2402.00159](https://arxiv.org/abs/2402.00159). Blogs: [Hello OLMo](https://allenai.org/blog/hello-olmo-a-truly-open-llm-43f7e7359222), [OLMo: Open Language Model](https://allenai.org/blog/olmo-open-language-model-87ccfc95f580). Code: [allenai/OLMo](https://github.com/allenai/OLMo), [allenai/dolma](https://github.com/allenai/dolma). Weights: [allenai/OLMo-7B](https://huggingface.co/allenai/OLMo-7B). Evaluation: Paloma, [arXiv 2312.10523](https://arxiv.org/abs/2312.10523).

## RewardBench (March 20, 2024)

RewardBench is not an OLMo release, but it belongs on this timeline because it opened Ai2's post-training evaluation line, the one that produced Tülu 3 eight months later. It scores a reward model by whether it rates the chosen completion above the rejected one, across "Chat, Chat Hard, Safety, and Reasoning" plus prior preference test sets. At launch it "evaluated over 30 RMs," from 400M to 70B parameters, and it scored DPO-trained models as implicit reward models. The finding that stuck was that DPO models "fail to generalize to popular preference data test sets and present a higher variance." Ai2's own tulu-2-dpo-70b was second on the launch leaderboard, behind Starling-RM-34B.

**Sources.** Paper: Lambert et al., "RewardBench: Evaluating Reward Models for Language Modeling", [arXiv 2403.13787](https://arxiv.org/abs/2403.13787). Blog: [RewardBench](https://allenai.org/blog/rewardbench-the-first-benchmark-leaderboard-for-reward-models-used-in-rlhf-1d4d7d04a90b). Code: [allenai/reward-bench](https://github.com/allenai/reward-bench). Data: [allenai/reward-bench](https://huggingface.co/datasets/allenai/reward-bench).

## Dolma moves to ODC-BY and OLMo 1.7-7B (April 2024)

Two releases two days apart, and both are about data.

On April 15, Dolma switched license: "As of today, Dolma is now operating under the ODC-BY license," replacing Ai2's own ImpACT license. ODC-BY was already used for the Tulu 2 instruction mix, Ai2's C4 replication and peS2o. The practical effect is that derivatives need an attribution notice, but "the work itself does not have to be licensed as ODC-BY." That one change is why so many later open models list Dolma subsets in their data cards.

On April 17 came OLMo 1.7-7B, with the headline "A 24 point improvement on MMLU." The precise claim: "This model scores 52 on MMLU, sitting above Llama 2-7B and approaching Llama 2-13B, and outperforms Llama 2-13B on GSM8K." Context doubled "from 2048 to 4096 tokens." The post says the model was "created on the path towards our upcoming 70 billion parameter model," a 70B OLMo that never shipped under that name.

Almost all of the gain came from data and a schedule change, not architecture:

- **Dolma 1.7.** "2.3 trillion tokens summing across all sources," of which the preferred sampled mixture "yields 1.7 trillion tokens." About 10.4% is specialized knowledge and reasoning (arXiv, Stack Exchange, OpenWebMath, Flan), and "15.4% of code data from StarCoder."
- **Fuzzy deduplication.** Paragraph-level 13-gram scoring with "α=0.3 and β=0.8", which "removes 48% of tokens in Dolma CC, 10% in C4, and 12% in Refined Web."
- **A quality filter.** A fastText classifier trained on "about 25GB of uncompressed text data."
- **Two-stage training.** Stage 1 used a cosine schedule ("warmup of 2500 steps, a peak learning rate of 3e-4, and a cosine decay to 3e-5 after 3T tokens") but was cut "after 2T tokens." Stage 2 trained "another 50B tokens" on a curated high-quality subset while the learning rate decayed linearly to zero.

Stage 2 is the seed of everything that came later. A short, high-quality annealing phase at the end of pretraining became Dolmino in OLMo 2 and the three-stage pretrain, mid-train, long-context flow in Olmo 3. A worked example shows why it is cheap: 50B tokens on top of 2T is a 2.5% extension of the run, and it bought most of a 24-point MMLU jump. Spending 2.5% of compute on the right data at the right moment is the best trade in pretraining, and every generation since has spent more of its budget there.

**Sources.** Blogs: [Dolma moves to ODC-BY](https://allenai.org/blog/making-a-switch-dolma-moves-to-odc-by-8f0e73852f44), [OLMo 1.7-7B](https://allenai.org/blog/olmo-1-7-7b-a-24-point-improvement-on-mmlu-92b43f7d269d). Weights: [allenai/OLMo-1.7-7B](https://huggingface.co/allenai/OLMo-1.7-7B) (the Hugging Face-format repo is `OLMo-7B-0424-hf`). Data: [allenai/dolma](https://huggingface.co/datasets/allenai/dolma). No separate paper.

## OLMo 0724 and the checkpoint-dynamics study (July to October 2024)

In July 2024 Ai2 refreshed both sizes without a launch post. The Hugging Face cards for `OLMo-7B-0724-hf` (repo created July 12, 2024) and `OLMo-1B-0724-hf` give the specs:

| | OLMo 1B 0724 | OLMo 7B 0724 |
| --- | --- | --- |
| Training tokens | 3.05T | 2.75T |
| Layers | 16 | 32 |
| Hidden size | 2048 | 4096 |
| Heads | 16 | 32 |
| Context | 4096 | 4096 |
| Data cutoff | Oct 2023 | Oct 2023 |

The 0724 checkpoints matter less as models than as the subject of the October 2, 2024 post "Investigating pretraining dynamics and stability with OLMo checkpoints" by Will Merrill. It is a blog-only analysis, with no paper, and it is a good example of research the open checkpoints made possible. OLMo 7B 0724 has "no biases and non-parametric layer-norm," and the study found:

- Parameter 1-norm and 2-norm "consistently decrease over pretraining," the opposite of what Merrill et al. (2021) had seen in T5, where the 2-norm grew.
- Activation norms also shrink, more in early layers, measured on "500 documents from the Pile validation set (each truncated to 2048 tokens)."
- Outlier features appear early and concentrate in the embedding layer, which has "10 outlier activations," while "past layer 12, each layer has at most 1 (and usually 0)."

The conclusion was that shrinking embedding norms and early-layer outliers "might both be destabilizing forces," and the post suggested z-loss and a better-scaled initialization. Both appear in OLMo 2 seven weeks later. That is the pattern this whole post is about: a cheap analysis of public checkpoints, then a recipe change in the next model.

**Sources.** Blog: [Investigating pretraining dynamics and stability with OLMo checkpoints](https://allenai.org/blog/investigating-pretraining-dynamics-and-stability-with-olmo-checkpoints-ece6f0c4947a). Weights: [allenai/OLMo-7B-0724-hf](https://huggingface.co/allenai/OLMo-7B-0724-hf), [allenai/OLMo-1B-0724-hf](https://huggingface.co/allenai/OLMo-1B-0724-hf). No paper.

## OLMoE (September 4, 2024)

OLMoE, "jointly developed with Contextual AI," was "the first mixture-of-experts model to join the OLMo family": "a sparse MoE model with 1 billion active and 7 billion total parameters," "trained on 5 trillion tokens built on a new data mix incorporating lessons from Ai2's Dolma and building heavily on DataComp-Baseline." The blog's efficiency claim is that it "trained 2x faster than equivalent dense models." The Olmo-core 3 post later fills in a detail the launch post left out: OLMoE used 64 routed experts.

The paper's real contribution is its ablations, on "routing algorithm, auxiliary loss functions, and sparse upcycling," with the artifacts to check them: "244 checkpoints for the pretrained model, one every 5000 steps," annealed and unannealed variants, and runs with and without load balancing. If you want to study how experts specialize during pretraining, OLMoE is still the only small MoE with that level of public trace. Post-training was a preview of the Tülu 3 pipeline (No Robots, math and code data, part of NVIDIA's Daring Anteater) plus UltraFeedback DPO.

OLMoE has had a longer life than its size suggests. Molmo's smallest model, MolmoE-1B, is built on it. FlexOlmo cites it as its MoE starting point. OlmPool used its pretraining corpus. And Olmo-core 3's MoE benchmarks use it as the reference point. The [MoE scaling laws post](/blog/machine-learning/scaling-laws/moe-scaling-laws) covers why a 1B-active, 7B-total model can match a much larger dense one.

**Sources.** Paper: Muennighoff et al., "OLMoE: Open Mixture-of-Experts Language Models", [arXiv 2409.02060](https://arxiv.org/abs/2409.02060). Blog: [OLMoE](https://allenai.org/blog/olmoe-an-open-small-and-state-of-the-art-mixture-of-experts-model-c258432d0514). Code: [allenai/OLMoE](https://github.com/allenai/OLMoE). Weights: [allenai/OLMoE-1B-7B-0924](https://huggingface.co/allenai/OLMoE-1B-7B-0924).

## Tülu 3 (November 21, 2024; 405B on January 30, 2025)

Tülu is Ai2's post-training line, and from Tülu 3 onward it *is* how OLMo gets turned into an assistant. The launch post's motivating fact is pointed: "On LMSYS's ChatBotArena, no model in the top 50 (as of November 20th, 2024) has released its post-training data." Tülu 3 released all of it: data, mixes, recipes, code, infrastructure and evaluation.

The first Tülu 3 models were built on Llama 3.1 (8B and 70B), not OLMo, using "a five-part post-training recipe": prompt curation, SFT, DPO (off-policy and on-policy), RLVR, and standardized evaluation. The concrete pieces:

- **Prompts.** "In total, we collect 939,344 prompts ... 57% are sourced from public resources and 43% are synthetically generated," the synthetic part using persona-driven generation. SFT responses for synthetic prompts came from "GPT-4o or Claude 3.5 Sonnet (for coding)."
- **Decontamination.** Any training set with "overlap with more than 2% of our evaluation suite" was removed. The [decontamination post](/blog/machine-learning/training-data/decontamination-and-benchmark-leakage) explains why that threshold is a judgment call.
- **Preferences.** An UltraFeedback-style pipeline over roughly 200 to 300K prompts, judged by GPT-4o-2024-0806 on helpfulness, instruction-following, honesty and truthfulness. "Length-normalized DPO achieved better performance compared to ... PPO, DPO, and SimPO," and on-policy completions plus prompts not seen in SFT both helped.
- **RLVR**, reinforcement learning with verifiable rewards, the term Tülu 3 introduced. It "replaces the reward model with a verification function": a math answer checker or an instruction-constraint checker returns a binary reward. It gave "up to 1.7, 3.3, and 1.3 points improvement over the DPO checkpoint on MATH, GSM8K and IFEval."

RLVR is the most consequential idea in this whole timeline outside pretraining. In plain notation, where $\pi_\theta$ is the policy, $x$ a prompt with a checkable answer, $y$ a sampled completion and $v(x, y)$ the verifier:

$$
\max_\theta \; \mathbb{E}_{x \sim D,\; y \sim \pi_\theta(\cdot \mid x)} \big[ v(x, y) \big] \;-\; \beta \, \mathrm{KL}\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big), \qquad v(x, y) \in \{0, \alpha\}
$$

Here $D$ is the prompt set, $\pi_{\text{ref}}$ the DPO checkpoint the run started from, $\beta$ the KL penalty weight and $\alpha$ the reward for a verified-correct answer. There is no learned reward model to hack, which is why it scales. Two months later the "Tülu 3 405B" post showed exactly that: on Llama 3.1 405B, with "32 nodes (256 GPUs)," 16-way tensor-parallel vLLM for generation and "the remaining 240 GPUs for training," Tülu 3 405B reached "competitive or superior performance to both Deepseek v3 and GPT-4o," and RLVR "improved the MATH performance more significantly at a larger scale." Each RL iteration took about 550 seconds of inference, 25 seconds of weight transfer and 1500 seconds of training. The [GRPO vs DPO vs PPO guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) is the background for choosing among these.

The recipe went into OLMo five days later. OLMo 2 Instruct, OLMoE 0125, OLMo 2 32B (as "Tülu 3.1," with GRPO), and every Olmo 3 path all descend from it. The Tülu name itself continued in DR Tulu (November 18, 2025), an open deep-research agent recipe, though that one is built on Qwen3-8B, not OLMo.

**Sources.** Paper: Lambert et al., "Tulu 3: Pushing Frontiers in Open Language Model Post-Training", [arXiv 2411.15124](https://arxiv.org/abs/2411.15124) (updated with 405B results). Blogs: [Tülu 3](https://allenai.org/blog/tulu-3), [Tülu 3 technical](https://allenai.org/blog/tulu-3-technical), [Tülu 3 405B](https://allenai.org/blog/tulu-3-405b). Code: [allenai/open-instruct](https://github.com/allenai/open-instruct). Weights and data: [Tülu 3 models](https://huggingface.co/collections/allenai/tulu-3-models-673b8e0dc3512e30e7dc54f5), [Tülu 3 datasets](https://huggingface.co/collections/allenai/tulu-3-datasets-673b8df14442393f7213f372). Evaluation: OLMES, [arXiv 2406.08446](https://arxiv.org/abs/2406.08446), [allenai/olmes](https://github.com/allenai/olmes).

## OLMo 2 (November 26, 2024)

OLMo 2 is where the family became competitive. The launch post: "a new family of 7B and 13B models trained on up to 5T tokens"; "OLMo 2 7B outperforms Llama-3.1 8B and OLMo 2 13B outperforms Qwen 2.5 7B despite its lower total training FLOPs." The paper's title, "2 OLMo 2 Furious," is the best in the family.

### What changed in the architecture

Every change is about training stability, and several answer the October checkpoint study directly:

| Component | OLMo 1 / 0724 | OLMo 2 |
| --- | --- | --- |
| Norm | Non-parametric LayerNorm | RMSNorm |
| Norm placement | Pre-norm | Reordered "as in Liu et al (2022)" (Swin v2 style, norm on the sublayer output) |
| Attention logits | Unnormalized | QK-Norm (Dehghani et al., 2023) |
| Output softmax | Plain cross-entropy | Z-loss regularization |
| Initialization | Earlier scheme | "An improved initialization" |
| Positions | RoPE | RoPE |

QK-norm is the change to remember. It stops attention logits from growing without bound, which removes a common source of loss spikes. It is also, eighteen months later, the single largest negative factor OlmPool found for long-context extension. A stability fix in one generation became a long-context tax in another, and you could only learn that because both generations are open.

### Two-stage pretraining and souping

- **Stage 1** (">90% of the total pretraining budget"): OLMo-Mix-1124, "approximately 3.9 trillion tokens sourced from DCLM, Dolma, Starcoder, and Proof Pile II." The 7B saw about one epoch; the 13B "1.2 epochs up to 5T tokens" (the later 32B post says 1.3 for the 13B, so treat the second decimal as uncertain).
- **Stage 2**: Dolmino-Mix-1124, "843 billion tokens," sampled into 50B, 100B and 300B mixes, each half high-quality web and half domain-specific (math especially), with the learning rate annealed linearly to zero.
- **Souping.** The 7B trained three copies on the 50B mix with different data orders and averaged their weights. The 13B averaged three 100B runs and one 300B run.

Here is why souping works in a worked example. Suppose three annealing runs from the same stage 1 checkpoint land at slightly different points in the same low-loss basin, each overfit to its own data order. Averaging their weights lands near the basin's center, which usually generalizes better than any one of them. The cost is two extra annealing runs: for the 7B, 100B extra tokens on top of roughly 4T, about 2.5% more compute.

### Evaluation and post-training

OLMo 2 introduced the OLMES evaluation discipline: "a suite of 20 evaluation benchmarks" split into a *development* set used to make recipe decisions and an *unseen* set (AGIEval, MMLU Pro, GSM8K, TriviaQA and others) held back to check the decisions did not overfit. Post-training was the Tülu 3 recipe, with preference data refreshed using on-policy OLMo 2 completions; "OLMo 2 13B Instruct outperforming Qwen 2.5 14B instruct, Tülu 3 8B, and Llama 3.1 8B instruct."

| | OLMo 2 7B | OLMo 2 13B |
| --- | --- | --- |
| Tokens | ~4T (1 epoch of 3.9T) + 50B mid-train | up to 5T + 100B/300B mid-train |
| Souping | 3 x 50B | 3 x 100B + 1 x 300B |
| Post-training | Tülu 3 (SFT, DPO, RLVR) | Tülu 3 (SFT, DPO, RLVR) |
| Repo ID | `OLMo-2-1124-7B` | `OLMo-2-1124-13B` |

**Sources.** Paper: OLMo Team, "2 OLMo 2 Furious", [arXiv 2501.00656](https://arxiv.org/abs/2501.00656). Blog: [OLMo 2](https://allenai.org/blog/olmo2). Code: [allenai/OLMo](https://github.com/allenai/OLMo), [allenai/olmes](https://github.com/allenai/olmes). Weights: [OLMo-2-1124-7B](https://huggingface.co/allenai/OLMo-2-1124-7B), [OLMo-2-1124-13B](https://huggingface.co/allenai/OLMo-2-1124-13B), [OLMo 2 collection](https://huggingface.co/collections/allenai/olmo-2-674117b93ab84e98afc72edc). Data: [olmo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124), [dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124).

## OLMoE 0125, OLMo 2 32B and OLMo 2 1B (February to April 2025)

Three releases finished the OLMo 2 generation.

**OLMoE-1B-7B-0125 and the iOS app (February 11, 2025).** The OLMoE base was re-annealed with "the Dolmino mix introduced in OLMo 2 for mid-training, and the Tülu 3 post-training recipe," making it "on average 35% better on our evaluation suite." Ai2 shipped it as a private on-device iOS app built on llama.cpp's Swift bindings: Q4_K_M quantization moves IFEval "from 66.4 to 63.6," it needs an "iPhone 15 Pro or newer devices, or any M-series iPad," and it runs at "41 tokens/s on average on iPhone 16 Pro." That is a 1B-active model doing what a 1B-active model is for.

**OLMo 2 32B (March 13, 2025).** The headline: "first fully-open model ... to outperform GPT3.5-Turbo and GPT-4o mini on a suite of popular, multi-skill academic benchmarks," trained "up to 6T tokens" (1.5 epochs of OLMo-Mix-1124) and post-trained with "Tülu 3.1," which swaps PPO for GRPO in the RLVR stage on GSM8K, IFEval and MATH. The efficiency claim is that it "takes only one third of the cost of training Qwen 2.5 32B while reaching similar performance."

The 32B post is also the best public operations log in the family. It was trained on "Augusta, a 160 node AI Hypercomputer" from Google Cloud, eight H100s per node, at "over 1800 tokens per second per GPU (~38% MFU)." It was the first model on **OLMo-core**, a rewrite of the training code with async distributed checkpointing, `torch.compile`, FSDP2, selective activation checkpointing and "4D+ parallelism." The three operational lessons are worth stealing:

| Problem | Fix | Effect |
| --- | --- | --- |
| 1024 processes uploading checkpoints at once | Throttle uploads | "improved throughput by 30%" |
| Full sharding across the whole cluster | Hybrid sharding in groups of 256 GPUs + a PyTorch 2.6 fix | "an immediate speed-up around 20%" |
| Collective throughput late in the run | NCCL LL128 drivers | Faster collectives |

A post-training detail later explained by OLMoTrace: date-cutoff mentions were scrubbed from the 32B's post-training data, producing new mixes named `tulu-3-sft-olmo-2-mixture-0225` and `OLMo-2-32B-pref-mix-0325`. The next section explains why.

**OLMo 2 1B (Hugging Face, April 2025).** No launch post; the card for `OLMo-2-0425-1B` (repo created April 17, 2025) lists "4 Trillion" training tokens, 16 layers, hidden size 2048, 16 heads and 4096 context, pretrained on OLMo-Mix-1124 with SFT, DPO and Instruct variants. It matters as the base for Bolmo 1B.

**Sources.** Blogs: [OLMoE, meet iOS](https://allenai.org/blog/olmoe-app), [OLMo 2 32B](https://allenai.org/blog/olmo2-32b). Paper: OLMo 2 report, [arXiv 2501.00656](https://arxiv.org/abs/2501.00656). Code: [allenai/OLMo-core](https://github.com/allenai/OLMo-core), [allenai/OLMoE.swift](https://github.com/allenai/OLMoE.swift). Weights: [OLMoE-1B-7B-0125-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct), [OLMo-2-0325-32B](https://huggingface.co/allenai/OLMo-2-0325-32B), [OLMo-2-0325-32B-Instruct](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct), [OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B).

## OLMoTrace (April 9, 2025)

OLMoTrace is the feature that only a fully open family could ship: it traces a model's output "back to their full, multi-trillion-token training data in real time." It launched in the Ai2 Playground for OLMo 2 32B Instruct, OLMo 2 13B Instruct and OLMoE 1B-7B Instruct. For the 32B Instruct, the five training datasets add up to "about 3.2 billion documents and 4.6 trillion tokens."

The mechanism, step by step:

1. Find every span of the output that appears **verbatim** in the training data, is **self-contained** (no sentence break inside, no partial words) and is **maximal** (cannot be extended).
2. Do it fast. Built on infini-gram's suffix arrays, a parallel algorithm reduces span finding "from the naive O(L^2 * N) to O(L * log N)," where L is output length and N corpus size.
3. Keep only the most surprising spans: the K with the lowest "span unigram probability," with "K = 0.05 * the number of tokens in the model output."
4. Show up to 10 source documents per span, ranked by BM25 against the prompt plus response.

The best finding came from Ai2 using it on itself. OLMo 2 13B claimed an "August 2023" knowledge cutoff, while its pretraining data stops in "December 2022." OLMoTrace found the claim verbatim in post-training examples, which "led us to remove contents containing knowledge cutoff dates from the 32B model's post-training data." A model's beliefs about itself are mostly copied from its SFT data, and OLMoTrace makes that visible. Olmo 3 integrated it into the Playground as OlmoTrace.

**Sources.** Paper: Liu et al., "OLMoTrace: Tracing Language Model Outputs Back to Trillions of Training Tokens", [arXiv 2504.07096](https://arxiv.org/abs/2504.07096). Blog: [OLMoTrace](https://allenai.org/blog/olmotrace). Code: [allenai/infinigram-api](https://github.com/allenai/infinigram-api).

## The measurement track: DataDecide, critical batch size, Signal and Noise, Fluid Benchmarking (2025)

Between OLMo 2 and Olmo 3, Ai2 published four papers that share one question: how do you make *decisions* during model development without paying for full-scale runs? All four used OLMo models or OLMo-style ladders as their test subjects, and their answers show up in Olmo 3's recipe and in olmo-eval.

### DataDecide (April 15, 2025)

Ai2 pretrained models "on 25 corpora with differing sources, deduplication, and filtering up to 100B tokens, over 14 different model sizes ranging from 4M parameters up to 1B parameters (more than 30k model checkpoints in total)," with three seeds each. The question: if you rank corpora with small models, how often do you pick the corpus that wins at 1B and 100B tokens?

The answer was humbling for scaling-law fans. Ranking at a single small scale (for example 150M) picks the 1B winner with "~80% decision accuracy," and "8 baseline scaling law variants" did not beat that. "MMLU and ARC Easy are highly predictable with as little as 4 orders of magnitude less compute," HellaSwag less so. And switching the metric from accuracy to a continuous one ("character-normalized likelihood of answers") made code benchmarks like MBPP and HumanEval predictable even where small-model accuracy sits at chance.

Note: the DataDecide blog's arXiv link points to OLMES (2406.08446). DataDecide's own paper is 2504.11393.

**Sources.** Paper: Magnusson et al., "DataDecide: How to Predict Best Pretraining Data with Small Experiments", [arXiv 2504.11393](https://arxiv.org/abs/2504.11393). Blog: [DataDecide](https://allenai.org/blog/datadecide). Code: [allenai/DataDecide](https://github.com/allenai/DataDecide). Models and results: [DataDecide collection](https://huggingface.co/collections/allenai/datadecide-67edb1d2bacba40b5d3ed633), [DataDecide-eval-results](https://huggingface.co/datasets/allenai/DataDecide-eval-results).

### Critical batch size revisited (June 3, 2025)

The critical batch size (CBS) hypothesis says there is a batch size $B^*$ "up to which increasing the batch size (and appropriately modifying the learning rate) approximately preserves the loss trajectory." Past $B^*$ you waste tokens. The standard way to estimate it is the gradient noise scale of McCandlish et al. (2018):

$$
\mathcal{B}_{\text{noise}} = \frac{\operatorname{tr}(\Sigma)}{\lVert G \rVert^2}
$$

where $G$ is the true gradient, $\Sigma$ the per-example gradient covariance and $\operatorname{tr}$ the trace. The paper found that this estimate rests on SGD and well-conditioned-Hessian assumptions that do not hold for Adam-trained LMs, and that the measured noise scale does "not match" CBS measured directly; using it "could have led us to underestimate."

The direct method is simple. Branch from a checkpoint, train several batch sizes (learning rate rescaled) for a short window, and call CBS the largest batch that matches or beats every smaller one. On OLMo 1B and 7B, CBS "starts near 0, increases rapidly and then diminishingly, and plateaus around a batch size of 4096," and it is similar across model sizes, which suggests it depends mainly on the data.

That shape suggests **batch-size warmup**: start small, and double the batch each time CBS has grown past it, scaling the learning rate by $\sqrt{2}$ at each doubling. A worked example: a run that starts at batch 512 and doubles three times reaches 4096. Each doubling halves the number of gradient steps for the remaining tokens. On OLMo 1B this reached the same (slightly better) loss "with 43% fewer gradient steps," and the post says "we plan to incorporate batch size warmup into future pretraining runs (e.g., OLMo 3)." Fewer steps at the same token count is pure wall-clock savings on a large cluster.

**Sources.** Paper: Merrill et al., "Critical Batch Size Revisited: A Simple Empirical Approach to Large-Batch Language Model Training", [arXiv 2505.23971](https://arxiv.org/abs/2505.23971). Blog: [Revisiting critical batch size](https://allenai.org/blog/critical-batch-size).

### Signal and Noise (August 19, 2025)

A benchmark is useful for decisions when it separates models (signal) more than it wobbles across nearby checkpoints of one model (noise). The paper defines signal as dispersion, "the maximum difference between any pair of models at that scale," and noise as the standard deviation over final checkpoints, for example "final 30 checkpoints of the OLMo 2 7B training." Their ratio predicts benchmark usefulness: "R^2=0.626 and R^2=0.471" for decision accuracy and scaling-law error.

Two cheap interventions follow. Filtering to high-SNR subtasks ("only 16 subtasks for MMLU") gave a "32% error reduction for MMLU." Switching to a bits-per-byte metric raised SNR sharply (GSM8K "1.2 to 7.0," MBPP "2.0 to 41.8") and lifted 150M-scale decision accuracy on Minerva MATH from 51% to 90%. Ai2 released "900K evaluation results on 465 open-weight language models."

**Sources.** Paper: Heineman et al., "Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation", [arXiv 2508.13144](https://arxiv.org/abs/2508.13144). Blog: [Signal and Noise](https://allenai.org/blog/signal-noise). Code: [allenai/signal-and-noise](https://github.com/allenai/signal-and-noise). Data: [allenai/signal-and-noise](https://huggingface.co/datasets/allenai/signal-and-noise).

### Fluid Benchmarking (September 16, 2025)

Fluid Benchmarking borrows from psychometrics. A two-parameter item response theory (IRT) model, giving each question a difficulty and a discrimination, is fit on Open LLM Leaderboard results for six benchmarks. Evaluation then picks the next question adaptively by Fisher information, like a computerized adaptive test, and reports an ability estimate instead of accuracy. On MMLU "it achieves higher validity and lower variance than standard methods while using fifty times fewer items," and it cut mislabeled items by 99% (checked against MMLU-Redux). The worked examples run on OLMo 1 7B and OLMo 2 7B pretraining checkpoints. It was a COLM 2025 paper.

**Sources.** Paper: Hofmann et al., "Fluid Language Model Benchmarking", [arXiv 2509.11106](https://arxiv.org/abs/2509.11106). Blog: [Fluid Benchmarking](https://allenai.org/blog/fluid-benchmarking). Code: [allenai/fluid-benchmarking](https://github.com/allenai/fluid-benchmarking).

| Paper | Question | Test subject | Answer in one line |
| --- | --- | --- | --- |
| DataDecide | Can small runs pick the best data? | 30k checkpoints, 4M to 1B | Yes, ~80% of the time; use continuous metrics |
| Critical batch size | How big can the batch get? | OLMo 1B, 7B | CBS grows then plateaus near 4096; warm the batch up |
| Signal and Noise | Which benchmarks can you trust? | 465 models incl. OLMo 2 | High SNR ones; filter subtasks, use BPB |
| Fluid Benchmarking | How many items do you need? | OLMo 1 and 2 checkpoints | Far fewer, if chosen adaptively with IRT |

## FlexOlmo (July 9, 2025)

FlexOlmo tackles a problem no benchmark measures: owners of valuable data (publishers, hospitals, companies) will not hand it over for pretraining, and even if they did, they could never take it back. FlexOlmo lets them "locally branch from a shared public model, add an expert trained on their data locally, and contribute this expert module back." They can deactivate their data at any time, and outputs can be attributed to experts at inference.

![Before and after: centralized pretraining pools every owner's raw data into one model with no opt-out, while FlexOlmo has each owner train an expert locally against a frozen public anchor and contribute only the expert](/imgs/blogs/olmo-model-family-timeline-5.webp)

Two design choices make independently trained experts composable:

- **A frozen anchor.** Each expert is trained "alongside a frozen copy of the public model, which serves as an 'anchor'." Because every expert learned to cooperate with the same frozen layers, the experts stay compatible with each other even though they never saw each other's data.
- **A router without joint training.** The router uses "domain-informed embeddings initialized from document embeddings of each dataset, removing the need for joint router training." Joint router training would require all the data in one place, which is the thing FlexOlmo exists to avoid.

The paper frames FlexOlmo against cross-silo federated learning and against model merging, which it says struggles on disjoint datasets. The claim is performance "very close to a hypothetical model trained on all combined public and private data." On privacy, an extraction attack on a math expert trained for three epochs achieved a "low extraction rate of 0.7%," against "60%" for a model deliberately overfit for 100 epochs, with differential-privacy training available as an option. The released model is FlexOlmo-7x7B-1T.

The idea had a second life. BAR (April 2026) found that freezing shared layers breaks down in post-training and reworked the approach, and EMO (May 2026) tried to get the same modularity to emerge during pretraining without predefined domains. Both are covered below.

**Sources.** Paper: Shi et al., "FlexOlmo: Open Language Models for Flexible Data Use", [arXiv 2507.07024](https://arxiv.org/abs/2507.07024) (NeurIPS 2025). Blog: [Introducing FlexOlmo](https://allenai.org/blog/flexolmo). Code: [allenai/FlexOlmo](https://github.com/allenai/FlexOlmo). Weights: [allenai/FlexOlmo-7x7B-1T](https://huggingface.co/allenai/FlexOlmo-7x7B-1T).

## Olmo 3 (November 20, 2025) and Olmo 3.1 (December 12, 2025)

Olmo 3 is the current generation and the rebrand: "OLMo" became "Olmo." Ai2's framing is the **model flow**, meaning the release is not a checkpoint but the entire path from data to every post-trained variant, with each stage's checkpoint, data and code published. I covered the techniques in depth in [Olmo 3: training and fine-tuning techniques](/blog/paper-reading/large-language-model/olmo-3-training-finetuning-techniques); this section is the catalog view.

![The Olmo 3 model flow: pretraining on Dolma 3 Mix, mid-training on Dolmino, long-context extension on Longmino, then Think, Instruct and RL-Zero post-training paths leading to the 3 and 3.1 checkpoints](/imgs/blogs/olmo-model-family-timeline-3.webp)

### The lineup

Dense models "at 7 billion and 32 billion parameters":

| Variant | 7B | 32B | What it is |
| --- | --- | --- | --- |
| Olmo 3-Base | Yes | Yes | Pretrained, mid-trained, long-context extended |
| Olmo 3-Think | Yes | Yes | SFT, DPO, RLVR for long reasoning traces |
| Olmo 3-Instruct | Yes | 3.1 only | SFT, DPO, RLVR for chat, tool use, short answers |
| Olmo 3-RL Zero | Yes | No | RLVR directly on Base; four series: math, code, IF, chat |

### Data: Dolma 3 and Dolci

- **Dolma 3**: a "~9.3-trillion-token corpus" of web, science PDFs processed with olmOCR, code, math and encyclopedic text.
- **Dolma 3 Mix**: the "5.9-trillion-token (~6T)" pretraining mix sampled from it.
- **Dolma 3 Dolmino**: mid-training, "100B training tokens sampled from a ~2.2T-token pool."
- **Dolma 3 Longmino**: long-context extension, "~50B training tokens drawn from a 639B-token pool." Base models handle up to about 65K tokens.
- **Dolci**: the post-training suite (SFT, DPO and RL mixes) behind all three paths.

New data tools shipped with it: `datamap-rs`, `duplodocus` (deduplication) and `decon` (decontamination). The datasets are released "without any license restrictions."

### Architecture

The Olmo 3 post itself says little beyond "decoder-only transformer" with "architectural enhancements." Two later Ai2 posts fill in the key detail: Olmo 3 interleaves sliding-window attention with full attention, "three local-attention layers for every one full-attention layer" (OlmPool), and keeps QK-norm. Olmo Hybrid later replaces exactly those sliding-window layers.

### Results and compute

Olmo 3-Think 32B was called the "strongest fully open thinking model," narrowing the gap to Qwen 3 32B "while training on roughly 6x fewer tokens." Selected numbers from the release table, next to the 3.1 update from the Hugging Face card:

| Benchmark | Olmo 3 Think 32B | Olmo 3.1 Think 32B |
| --- | --- | --- |
| MATH | 96.1 | 96.2 |
| AIME 2024 | 76.8 | 80.6 |
| AIME 2025 | 72.5 | 78.1 |
| ZebraLogic | 76.0 | 80.1 |
| IFEval | 89.0 | 93.8 |
| IFBench | 47.6 | 68.1 |
| OMEGA | 50.8 | 53.4 |
| LiveCodeBench v3 | 83.5 | 83.3 |

Olmo 3-Instruct 7B: MATH 87.3, AIME 2024 44.3, IFEval 85.6, BFCL 49.8. Base 32B: GSM8K 80.5, MATH 43.4, HumanEval 66.5, MMLU Pro (MC) 49.6.

Pretraining ran on "up to 1,024 H100 GPUs" at "7.7K tokens per device per second" for the 7B; mid-training on 128 H100s; post-training on 256. Two infrastructure wins are worth knowing: moving SFT into OLMo-core raised throughput "by 8x," and RL became "4x more efficient" through in-flight weight updates, continuous batching and threading (881 to 2949 tokens per second in the release table).

### Olmo 3.1 (December 12, 2025)

The 3.1 update took the Think 32B RL run and simply kept going, "for an additional 21 days on 224 GPUs" over Dolci-Think-RL, for "5+ points on AIME, 4+ points on ZebraLogic, 4+ points on IFEval, and 20+ points on IFBench." The table above shows those deltas. Olmo 3.1 Instruct 32B applied the 7B Instruct recipe at 32B (SFT, DPO, RLVR), and the RL-Zero 7B code and math checkpoints were refreshed. The Hugging Face repos were created December 10; the blog update is dated December 12. Both list a "Dec. 2024" data cutoff and Apache 2.0.

The lesson of 3.1 is a quiet one: three more weeks of RL on an existing recipe moved IFBench by more than 20 points. RL was nowhere near saturated at the original release, which is also what Tülu 3 405B had reported about MATH.

**Sources.** Paper: Olmo Team, "Olmo 3", [arXiv 2512.13961](https://arxiv.org/abs/2512.13961) (the Olmo 3 blog's arXiv link, 2509.19128, is PipelineRL, which it cites). Blog: [Olmo 3](https://allenai.org/blog/olmo3). Code: [allenai/OLMo-core](https://github.com/allenai/OLMo-core), [allenai/open-instruct](https://github.com/allenai/open-instruct), [allenai/datamap-rs](https://github.com/allenai/datamap-rs), [allenai/duplodocus](https://github.com/allenai/duplodocus), [allenai/decon](https://github.com/allenai/decon). Weights: [Olmo 3 collection](https://huggingface.co/collections/allenai/olmo-3-68e80f043cc0d3c867e7efc6), [Olmo-3.1-32B-Think](https://huggingface.co/allenai/Olmo-3.1-32B-Think), [Olmo-3.1-32B-Instruct](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct). Deep dive on this site: [Olmo 3 techniques](/blog/paper-reading/large-language-model/olmo-3-training-finetuning-techniques).

## Bolmo (December 15, 2025; in Nature on October 7, 2026)

Bolmo is "the first fully open byte-level language models," and its trick is that it does not train a byte model from scratch. It *byteifies* an existing subword model: Bolmo 7B from "our public Olmo 3 7B checkpoint," Bolmo 1B from OLMo 2 1B. The full analysis is in [Bolmo: byteifying the next generation of language models](/blog/paper-reading/large-language-model/bolmo-byteifying-the-next-generation-of-language-models); here is the short version.

The architecture wraps the original transformer in a byte-level shell: byte embeddings, a local encoder ("mLSTM stack"), a "non-causal" boundary predictor that looks one byte ahead to decide where patches end, pooling of each patch into one vector for the original Olmo 3 transformer, then depooling and an mLSTM local decoder that predicts the next byte and the next boundary. It sits in the family of DTP, BLT and H-Net.

Training is two stages. Stage 1 freezes the transformer and trains the shell on "9.8B tokens (approximately 43B bytes)." Stage 2 trains everything on "39.3B additional tokens (approximately 173B bytes)." Under 50B tokens total, a rounding error next to the roughly 6T the Olmo 3 7B base cost. Data was the Dolma 3 mix plus open code and "a small slice of targeted character-level data."

| | Olmo 3 7B (subword) | Bolmo 7B (bytes) |
| --- | --- | --- |
| Character tasks (CUTE, EXECUTE) | Baseline | "nearly twenty points" better |
| Decoding speed at matched compression | ~150 bytes/s | ~125 bytes/s |
| IFEval after weight-arithmetic post-training | 66.9% (post-trained Olmo 3) | 67.4% |
| Extra training cost | n/a | ~49B tokens |

The last row is the cleverest result. Because Bolmo keeps the Olmo 3 transformer, you can transfer post-training by task arithmetic: add the difference between post-trained and base Olmo 3 weights to Bolmo's base. That takes IFEval from 31.1% to 67.4% with no post-training run at all.

The October 2026 Nature paper extended the method beyond Olmo: Qwen 3 8B and Llama 3 8B became "Bwen 8B and Blama 8B," and "Bwen 8B is our strongest byteified model yet, outperforming Bolmo 7B across our aggregate evaluation suite." Ai2 also released new Stage 1 checkpoints.

**Sources.** Paper: "Bolmo: Byteifying the Next Generation of Language Models", [arXiv 2512.15586](https://arxiv.org/abs/2512.15586); Nature version: [s41586-026-11111-4](https://www.nature.com/articles/s41586-026-11111-4). Blogs: [Introducing Bolmo](https://allenai.org/blog/bolmo), [Now in Nature](https://allenai.org/blog/bolmo-nature). Code: [allenai/bolmo-core](https://github.com/allenai/bolmo-core). Weights: [Bolmo-7B](https://huggingface.co/allenai/Bolmo-7B), [Bolmo-1B](https://huggingface.co/allenai/Bolmo-1B), [Bolmo collection](https://huggingface.co/collections/allenai/bolmo). Data: [bolmo_mix](https://huggingface.co/datasets/allenai/bolmo_mix). Deep dive on this site: [Bolmo](/blog/paper-reading/large-language-model/bolmo-byteifying-the-next-generation-of-language-models).

## Olmix (February 13, 2026)

Olmix is the data-mixing framework distilled from "developing Olmo 1–3," and it solves the part of mixing that papers usually ignore: the corpus keeps changing during development. It has two parts. **OlmixBase** is the offline method: train a swarm of small proxy models on different mixtures, fit a regression from mixture weights to downstream performance, and optimize the mixture against it, with validated defaults (log-linear regression, explicit limits on how many times any domain can repeat). **Mixture reuse** handles updates: when domains are added or changed, re-solve only the part of the mixture that changed.

The study used 1B models on DCLM split into "24 topic-based domains," scored on "52 downstream tasks" in bits per byte. Three findings are useful on their own:

- **Proxy size has a floor.** Proxies "above roughly 15M parameters achieve strong rank correlation (ρ > 0.89)"; 1M proxies reach "only ρ = 0.73." The post notes that RegMix's "1M" setup "is actually closer to 15M."
- **Swarm size scales linearly**, "O(m) runs for m domains."
- **Reuse is nearly free.** Over five simulated updates ending at "64 total domains," full reuse kept "95% of the improvement of full recomputation while using 74% fewer proxy runs (216 vs. 832)."

The best mix was "12.2% better than the natural distribution" and "3.05x more data-efficient," mostly by up-weighting arXiv, FineMath and code. That matches what every OLMo generation did by hand in its annealing stage. The [data mixing post](/blog/machine-learning/training-data/data-mixing-domain-weighting-and-curriculum) puts Olmix in context with DoReMi and RegMix.

**Sources.** Paper: "Olmix: A Framework for Data Mixing Throughout LM Development", [arXiv 2602.12237](https://arxiv.org/abs/2602.12237) (the blog's other arXiv link, 2407.01492, is RegMix). Blog: [Olmix](https://allenai.org/blog/olmix). Code: [allenai/olmix](https://github.com/allenai/olmix).

## Olmo Hybrid (March 5, 2026)

Olmo Hybrid is the most recent Olmo language model and the clearest example of the laboratory idea. Hybrid architectures, which mix attention with linear recurrent layers, have been shipping across the field: the Olmo Hybrid post names Samba, Nemotron-H, Qwen3-Next, Kimi Linear and Qwen 3.5. What the field lacked was a clean answer to whether the hybrid itself, rather than its data or recipe, is better. Olmo Hybrid answers it with "a controlled comparison to Olmo 3 7B."

![One repeating block of Olmo 3 7B (three sliding-window attention layers, one full attention layer) beside one block of Olmo Hybrid 7B (three Gated DeltaNet layers, one full attention layer), with the measured gains](/imgs/blogs/olmo-model-family-timeline-6.webp)

The change is one substitution: "Olmo Hybrid replaces the sliding-window attention layers from Olmo 3 with Gated DeltaNet," in a "3:1 pattern, three DeltaNet sublayers followed by one multihead attention sublayer," so it "replaces 75% of attention mixing with Gated DeltaNet." Gated DeltaNet keeps a fixed-size recurrent state updated by a gated delta rule. It costs constant memory per token instead of a growing KV cache, and it tracks evolving state well; the remaining full-attention layer handles precise recall. If you want the background on these layers, the site has [Kimi Linear](/blog/machine-learning/frontier-models/kimi-linear), [Qwen3-Next](/blog/machine-learning/frontier-models/qwen3-next-hybrid-attention-ultra-sparse-moe) and [Nemotron-H](/blog/machine-learning/frontier-models/nemotron-h-hybrid-mamba-transformer).

Training held everything else fixed where possible: "pretrained on 6 trillion tokens using the improved data mix from Olmo 3 32B," on "512 GPUs, starting on NVIDIA H100s before migrating to NVIDIA HGX B200s hosted on Lambda's infrastructure roughly halfway," with throughput matched to Olmo 3.

| Measurement | Olmo 3 7B | Olmo Hybrid 7B |
| --- | --- | --- |
| Tokens to reach the same MMLU | Baseline | 49% fewer ("roughly 2x data efficiency") |
| Tokens to the same Common Crawl eval loss | Baseline | 35% fewer |
| RULER at 64K | 70.9 (with YaRN) | 85.0 (with DRoPE), 76.9 (with YaRN) |
| Short context (4K) | Slightly ahead | Slightly behind, overtakes from 8K |
| After mid-training | Baseline | Better in every primary domain; small regressions on LBPP and DM Math |

The report pairs this with theory: hybrids are "fundamentally more expressive than pure transformers or pure linear RNNs alone," and scaling-law fits put the token-savings factor at "~1.3x at 1B parameters to ~1.9x at 70B," though the authors say the point estimates "aren't statistically conclusive."

### Which tokens does a hybrid predict better? (June 25, 2026)

A follow-up compared Olmo 3 7B and Olmo Hybrid 7B token by token, since the two are "closely matched in data, tokenizer, and training recipe." The hybrid's advantage concentrates on context-dependent content: the loss gap is "around 0.04" for content words against "closer to 0.02" for function words, and largest for adverbs and adjectives. It vanishes on verbatim repeated n-grams, shrinking toward zero as the repeat gets longer, and on closing (not opening) braces. Exact copying is attention's job, and the hybrid keeps only one attention layer in four. Its 1B ablations add that a pure RNN beats the transformer on non-repeated content tokens but falls behind on repeated ones.

**Sources.** Paper: "Olmo Hybrid: From Theory to Practice and Back", [arXiv 2604.03444](https://arxiv.org/abs/2604.03444). Token-level paper: "Comparing Transformers and Hybrid Models at the Token Level", [arXiv 2606.20936](https://arxiv.org/abs/2606.20936). Blogs: [Introducing Olmo Hybrid](https://allenai.org/blog/olmohybrid), [Which tokens does a hybrid model predict better?](https://allenai.org/blog/hybrid-token-prediction). Weights: [Olmo Hybrid collection](https://huggingface.co/collections/allenai/olmo-hybrid).

## OlmPool (April 23, 2026)

OlmPool asks why some architectures extend to long context gracefully and others do not, and answers with "a controlled suite of 26 7B models," each "pretrained for 140 billion tokens on the same data, then extended to 64K context," about 160,000 GPU hours in all, released with "38 checkpoints each." The four architecture choices it toggles are all ones Olmo 3 made:

| Choice | Where it appears | Effect on long-context extension |
| --- | --- | --- |
| QK-norm | Olmo 3, Qwen 3, Gemma 3 | Largest single effect: removing it and changing norm order gives "a 6-point gain on HELMET at 32K" |
| Grouped-query attention | Llama 3, Qwen 3, Gemma 3 | Hurts, especially combined with SWA |
| Sliding-window attention | Olmo 3 (3 local : 1 full) | About 1 point alone; SWA plus GQA "around 9 points" |
| Short pretraining context | 4,096 vs 8,192 | Hurts |

The headline: "Combining three or more can drop scores on long context benchmarks by up to 47%." The best predictor of long-context quality was simply *how many* of the four choices a model has. Short-context signals do not help: loss, perplexity and "a suite of 16 short-context benchmarks" fail to predict long-context results, and nearly identical models diverge "by more than 26 points on HELMET at 32K." The gaps persist with more extension training: after 50B extension tokens, "26% of total training," the worst architecture still trails the Llama architecture at 1B extension tokens. A mechanistic hint: models without QK-norm develop stronger attention sinks, which correlate with better long-context performance.

Read OlmPool next to OLMo 2. QK-norm went in for stability in November 2024; in April 2026 Ai2's own controlled study showed it is the biggest single obstacle to extending context. Olmo Hybrid's use of DRoPE for long context, and the next Olmo's promised "longest context window," should be read with this in mind. One caveat from the fact-check: a passage in the OlmPool post describes the evaluated models as "pretrained on 1T tokens from the OLMoE pretraining corpus, followed by a 50B-token annealing phase," which conflicts with the 140B-token description; check the paper for the exact setup.

**Sources.** Paper: "Cracks in the Foundation: Seemingly Minor Architectural Choices Impact Long Context Extension", [arXiv 2608.10296](https://arxiv.org/abs/2608.10296) (ICML 2026). Blog: [OlmPool](https://allenai.org/blog/olmpool). Code: [allenai/olmpool](https://github.com/allenai/olmpool). Weights: [OlmPool collection](https://huggingface.co/collections/allenai/olmpool).

## BAR and EMO: modular mixture-of-experts (April to May 2026)

Two papers a month apart push FlexOlmo's idea in opposite directions: BAR into post-training, EMO into pretraining.

### BAR: Branch-Adapt-Route (April 20, 2026)

BAR starts from a failure. FlexOlmo-style frozen shared layers do not survive post-training: in RLVR "the reward curve was completely flat." BAR's fix has three stages:

1. **Branch and adapt.** Each domain expert is a two-expert MoE (a frozen anchor FFN plus a trainable FFN), unfrozen progressively: during mid-training all shared parameters stay frozen; SFT unfreezes embeddings and the LM head (which took the tool-use expert's BFCL from 20.3 to 46.4); RLVR unfreezes everything, attention included. Each expert needs a domain mix plus general SFT data.
2. **Merge.** Shared parameters that diverged across experts are "simply averaged."
3. **Route.** A router is trained on "a stratified 5% sample of the SFT data."

Experts for math, code, tool use and safety were built "on top of a fully post-trained Olmo 2 base model," compared against "six baselines across 19 benchmarks, spanning 7 evaluation categories":

| Method | Overall score |
| --- | --- |
| Full retrain with mid-training (ceiling) | 50.5 |
| **BAR** | **49.1** |
| Post-training-only retrain | 47.8 |
| BTX | 46.7 |
| Merging without mid-training | 36.9 |
| Dense merge after mid-training | 6.5 |

The operational payoff is upgrades: swapping in a new code expert gave "+16.5 points," and adding RL to the math expert "+13 points," at "linear cost scaling" instead of the "effectively quadratic" cost of retraining a monolith. "Activating 4 of 5 experts at inference time achieves nearly identical performance to using all 5."

Note: the BAR blog's arXiv link (2502.13613) resolves to an unrelated physics paper. The correct ID is below.

**Sources.** Paper: Morrison, Adhikesaven, Bhagia, Zaharia, Smith, Min, "BAR" (Branch-Adapt-Route), [arXiv 2604.18473](https://arxiv.org/abs/2604.18473); also [allenai.org/papers/bar](https://allenai.org/papers/bar). Blog: [Train separately, merge together](https://allenai.org/blog/bar). Code: [allenai/FlexOlmo](https://github.com/allenai/FlexOlmo) (BAR scripts on a branch). Weights: [branch-adapt-route collection](https://huggingface.co/collections/allenai/branch-adapt-route).

### EMO: emergent modularity (May 8, 2026)

EMO asks whether you can get FlexOlmo-like modularity without deciding the domains in advance. It is "a 1B-active, 14B-total-parameter (8-expert active, 128-expert total) MoE trained on 1 trillion tokens," with one routing constraint: every token of a document routes within a shared document-level expert pool, chosen by averaging router preferences over the document, with the pool size sampled randomly during training. Load balancing is applied "globally across many documents," because per-micro-batch balancing fights the objective.

The result is an MoE you can prune by task. Keeping 25% of the experts (32) loses "only about 1%," and keeping 12.5% (16) "only about 3%," while a standard MoE pruned the same way degrades "close to or below random." Picking the subset needs only "a single example with few-shot demonstrations." The experts also specialize by *topic*: clustering over 12K documents finds EMO experts for "Health, Medical & Wellness" and "US Politics & Elections," where a standard MoE's experts organize around "Prepositions" and "Definite Articles." A matched standard-MoE baseline on the same data is released too.

**Sources.** Paper: Wang, Bhagia, Min, "EMO", [arXiv 2605.06663](https://arxiv.org/abs/2605.06663). Blog: [EMO](https://allenai.org/blog/emo). Code: [allenai/EMO](https://github.com/allenai/EMO). Weights: [allenai/EMO](https://huggingface.co/allenai/EMO), [EMO collection](https://huggingface.co/collections/allenai/emo).

## olmo-eval (June 12, 2026)

OLMES, "Introduced in 2024," "became the basis for evaluating our open models from Olmo to Tulu." olmo-eval is its successor, built for the development loop instead of the leaderboard. It has four parts: a shared definition layer for tasks, suites and harnesses; a sandbox and capability-routing layer with an async planner (Docker or Modal) so agentic and multi-turn evaluations are first-class; a normalized experiment schema; and a pairwise results viewer that compares checkpoints question by question. The design choice that ties it to the measurement track is that every score is reported "with a standard error and a minimum detectable effect." The post's motivating question, "Is a 2.4pp change in performance enough to make a call?", is the Signal and Noise paper turned into a tool.

**Sources.** Blog: [olmo-eval](https://allenai.org/blog/olmo-eval). Code: [allenai/olmo-eval](https://github.com/allenai/olmo-eval), predecessor [allenai/olmes](https://github.com/allenai/olmes). No paper.

## Olmo-core 3 and the next Olmo (October 2026)

Olmo-core 3 (October 1, 2026) is infrastructure, not a model, but it states the next model's direction outright: "Our next-generation Olmo will use an MoE architecture, and we're aiming for it to be our most capable Olmo yet, trained on our largest dataset and with our longest context window."

What changed is how experts are sharded. The previous MoE path was FSDP-based; the new one is DDP-based with experts resident on GPUs, plus expert parallelism, pipeline parallelism, a distributed optimizer, rowwise EP, GPU-resident routing and grouped GEMM. The measurements:

| Test | Result |
| --- | --- |
| Experts 8 to 128 (4 active per token), active params ~3.2B, total 4.6B to 47B | Throughput fell "by less than 5%" |
| 47B MoE on eight B300s | "52,000 tokens per second per GPU ... compared with 19,400" (~2.7x) |
| MXFP8 vs BF16 on four B300s | "about 21% higher," peak memory 103 to 95 GiB |
| Scale test | "1.2-trillion-parameter model with 58.36 billion parameters active per token across 512 GPUs," peak "858 TFLOP/s/GPU" |
| Capacity test with DeepEP v2 | 2.38 trillion total parameters (short run) |

The scale numbers use random routing, so they measure the system, not a trained model. The findings section is more interesting than the benchmarks: "token gerrymandering," where the load-balance score improves while real expert load gets worse; a lower expert learning rate that did not help; GEMM time that depends on the input values; and compute/communication overlap that is sometimes slower than no overlap.

Put the pieces together. Olmo-core 3 is an MoE stack. The COLM 2026 recap (October 9) says the next Olmo is "now in pre-training with a hybrid mixture-of-experts architecture," informed by the Olmo Hybrid research; Ai2 also mentions Google recreating the Olmo 3 7B training run in MaxText on TPUs. So the next Olmo is very likely an MoE with hybrid (attention plus linear recurrent) mixing, trained on Ai2's largest dataset with its longest context. Its size, name and date are not public, and nothing in this post should be read as predicting them.

**Sources.** Blog: [Introducing Olmo-core 3](https://allenai.org/blog/olmocore3), report at [allenai.org/papers/olmocore3](https://allenai.org/papers/olmocore3) (no arXiv version found). Code: [allenai/olmo-core](https://github.com/allenai/olmo-core). COLM recap: [Ai2 at COLM 2026](https://allenai.org/blog/colm-2026-recap).

## Same prefix, different family

Ai2 uses the "Olmo" and "Molmo" names well beyond the language models, and the research index mixes them all together. None of the following is an OLMo language model. Some are built on one; most are not.

| Name | What it is | Built on OLMo? | Sources |
| --- | --- | --- | --- |
| Molmo (Sep 25, 2024) | Open vision-language models (MolmoE-1B, 7B-O, 7B-D, 72B) with PixMo data | MolmoE-1B on OLMoE-1B-7B; Molmo-7B-O on a pre-release OLMo 2 7B; others on Qwen2 | [arXiv 2409.17146](https://arxiv.org/abs/2409.17146), [blog](https://allenai.org/blog/molmo) |
| Molmo 2, MolmoAct, MolmoBot, MolmoPoint, MolmoWeb (2025–2026) | Video VLM, robot action reasoning, sim-trained manipulation, pointing, web agent | Molmo line, not covered here | [allenai.org/research](https://allenai.org/research) |
| olmOCR (Feb 25, 2025) | PDF-to-text VLM; one million pages for about 190 USD | No, Qwen2-VL-7B; its job is clean text for Dolma 3 | [arXiv 2502.18443](https://arxiv.org/abs/2502.18443), [blog](https://allenai.org/blog/olmocr) |
| olmOCR 2 (Oct 22, 2025) | 82.4 on olmOCR-Bench; RL with unit-test rewards | No, Qwen2.5-VL-7B | [arXiv 2510.19817](https://arxiv.org/abs/2510.19817), [blog](https://allenai.org/blog/olmocr-2) |
| OLMoASR (Aug 28, 2025) | Open speech recognition, 39M to 1.5B, 1M-hour curated mix | No, Whisper-style | [tech report](https://github.com/allenai/OLMoASR/blob/main/tech_report/olmoasr_tech_report.pdf), [blog](https://allenai.org/blog/olmoasr) |
| OlmoEarth (Nov 4, 2025; v1.1 May 2026) | Earth-observation foundation models and platform | No | [allenai.org/papers/olmoearth](https://allenai.org/papers/olmoearth), [blog](https://allenai.org/blog/olmoearth) |
| DR Tulu (Nov 18, 2025) | Open deep-research agent recipe (RLER) | No, Qwen3-8B | [arXiv 2511.19399](https://arxiv.org/abs/2511.19399), [blog](https://allenai.org/blog/dr-tulu) |

olmOCR deserves one more sentence because it *does* feed the LM line indirectly: the science PDFs in Dolma 3 were processed with it, and olmOCR 2's unit-test rewards are RLVR applied to OCR. The DR Tulu blog's arXiv link (2507.17746) is "Rubrics as Rewards," which it cites; its own paper is 2511.19399.

## How the research feeds the models

Read the releases in order and five through-lines appear. The dependency graph shows the first, which is the reason this post pairs every model with its research.

![Dependency graph linking OLMo models on the left to the research built on them on the right, with findings flowing back into later models: checkpoint dynamics into OLMo 2, critical batch size into Olmo 3, FlexOlmo into BAR, Olmo 3 into Bolmo, Olmo Hybrid and OlmPool](/imgs/blogs/olmo-model-family-timeline-4.webp)

### 1. The previous generation is the next generation's test bench

The checkpoint study ran on OLMo 0724 and produced OLMo 2's z-loss and initialization. Critical batch size was measured on OLMo 1B and 7B and planned into Olmo 3. Bolmo, Olmo Hybrid, OlmPool and the token-level study all use Olmo 3 as the controlled baseline. BAR uses a post-trained OLMo 2. This only works because each generation's data, code and intermediate checkpoints are public. A lab that releases only final weights cannot run these experiments against its own past models in a way outsiders can check.

### 2. The budget kept moving to the end of training

OLMo 1 had one stage. OLMo 1.7 added a 50B-token high-quality tail. OLMo 2 made it a named dataset (Dolmino, 843B pool) with souping. Olmo 3 split the tail into mid-training (Dolmino, 100B) and long-context extension (Longmino, ~50B), then forked three post-training paths. Olmo 3.1 showed the RL stage at the very end was still far from saturated. The same pattern is visible in SmolLM (see the [SmolLM timeline](/blog/machine-learning/frontier-models/smollm-model-family-timeline)) and in every frontier lab's reports.

### 3. Post-training converged on verifiable rewards

RLVR appeared in Tülu 3, went into OLMo 2 Instruct, moved from PPO to GRPO in Tülu 3.1 for OLMo 2 32B, became the third stage of every Olmo 3 path and the *only* stage of RL-Zero, and reappeared as unit-test rewards in olmOCR 2. Ai2's post-training story since late 2024 is one idea applied everywhere it fits.

### 4. Measurement became a research program of its own

DataDecide, Signal and Noise, Fluid Benchmarking, Olmix and olmo-eval are all answers to "how do we decide with less compute and less noise?" They are the least glamorous part of the timeline and arguably the most reusable outside Ai2: none of them requires you to use an OLMo model.

### 5. Architecture changes arrive late and get tested hard

The OLMo transformer barely changed for two years: no biases, SwiGLU, RoPE, then RMSNorm, reordered norms and QK-norm for stability, then sliding-window attention in Olmo 3. The two real architectural moves, MoE (OLMoE, FlexOlmo, BAR, EMO, Olmo-core 3) and hybrid mixing (Olmo Hybrid), were each studied in controlled research models before the flagship adopted them. The next Olmo is reportedly where both land at once.

### Generation-over-generation summary

| | OLMo 1 (Feb 2024) | OLMo 2 (Nov 2024 to Apr 2025) | Olmo 3 (Nov to Dec 2025) | Olmo Hybrid (Mar 2026) |
| --- | --- | --- | --- | --- |
| Sizes | 1B, 7B | 1B, 7B, 13B, 32B | 7B, 32B | 7B |
| Max pretraining tokens | at least 2T | ~6T (32B) | ~5.9T mix + mid-train + long-context | 6T |
| Context | 2048 (4096 from 1.7) | 4096 | up to ~65K | 64K evaluated |
| Mixing | Full attention | Full attention | 3 SWA : 1 full | 3 Gated DeltaNet : 1 full |
| Stability | No biases, non-parametric LN | RMSNorm, reordered norm, QK-norm, z-loss | QK-norm kept | Inherits Olmo 3 |
| Late-stage data | None (1.7: 50B tail) | Dolmino (843B pool), souping | Dolmino + Longmino | Olmo 3 32B data mix |
| Post-training | SFT + DPO (Tulu 2) | Tülu 3 (SFT, DPO, RLVR; GRPO at 32B) | Dolci: Think, Instruct, RL-Zero | See the HF collection |
| Training code | OLMo | OLMo, then OLMo-core | OLMo-core | OLMo-core |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 | Fully open |

## Which variant to pick today

As of October 2026. "Today" matters: these are fully open models, which is a different market from best-in-class open weights. If you only need the strongest model at a size, a current Qwen, Gemma or Llama checkpoint usually wins. Pick OLMo when the openness is the point.

| Need | Pick | Why |
| --- | --- | --- |
| Strongest fully open reasoning model | **Olmo 3.1 32B Think** | AIME 2025 78.1, IFBench 68.1; every training stage public |
| Fully open chat and tool-use model | **Olmo 3.1 32B Instruct** or **Olmo 3 7B Instruct** | Instruct recipe at two sizes; 7B for a single GPU |
| Studying RL from a base model | **Olmo 3 RL-Zero 7B** (math, code, IF, chat) | RLVR directly on Base, with data and checkpoints |
| Long-context research on a 7B | **Olmo Hybrid 7B** | RULER 64K 85.0 against Olmo 3's 70.9; matched baseline exists |
| Controlled architecture experiments | **OlmPool suite** | 26 matched 7B models, 38 checkpoints each |
| Training dynamics, memorization, attribution | **OLMo 2 or Olmo 3 intermediate checkpoints** + OlmoTrace | Data and checkpoints aligned at every step |
| On-device or tiny MoE | **OLMoE-1B-7B-0125-Instruct** | 41 tokens/s on an iPhone 16 Pro at Q4 |
| Prunable, task-specialized MoE | **EMO** | 25% of experts keeps all but about 1% of performance |
| Contributing data without giving it away | **FlexOlmo** recipe | Train an expert locally; opt out later |
| Byte-level model, character-heavy tasks | **Bolmo 7B** (or Bwen 8B) | ~20 points on character tasks over Olmo 3 |
| Training your own MoE at scale | **Olmo-core 3** | Open stack tested to 1.2T parameters |

Two warnings. First, avoid OLMo 1 and 1.7 for anything new: they are research artifacts, superseded at every size. Second, if your use case needs very long context, read OlmPool before fine-tuning any QK-norm, sliding-window, GQA model on long data, because the architecture may cap what extension can recover.

For where Ai2 sits among open-model organizations, see [leading Hugging Face AI organizations](/blog/machine-learning/open-source-library/best-organization-in-huggingface), and for the closest comparable family catalog, the [Llama timeline](/blog/machine-learning/frontier-models/llama-model-family-timeline).

## Research reading list

Every research item in this post, in one place, with the paper and blog for each. Deep dives on this site exist for Olmo 3 and Bolmo so far; the rest are queued.

| # | Research | Date | Paper | Ai2 blog | Code |
| --- | --- | --- | --- | --- | --- |
| 1 | OLMo 1 | 2024-02-01 | [2402.00838](https://arxiv.org/abs/2402.00838) | [post](https://allenai.org/blog/olmo-open-language-model-87ccfc95f580) | [OLMo](https://github.com/allenai/OLMo) |
| 2 | Dolma | 2024-02-01 | [2402.00159](https://arxiv.org/abs/2402.00159) | [post](https://allenai.org/blog/making-a-switch-dolma-moves-to-odc-by-8f0e73852f44) | [dolma](https://github.com/allenai/dolma) |
| 3 | RewardBench | 2024-03-20 | [2403.13787](https://arxiv.org/abs/2403.13787) | [post](https://allenai.org/blog/rewardbench-the-first-benchmark-leaderboard-for-reward-models-used-in-rlhf-1d4d7d04a90b) | [reward-bench](https://github.com/allenai/reward-bench) |
| 4 | OLMES | 2024 | [2406.08446](https://arxiv.org/abs/2406.08446) | (in OLMo 2 post) | [olmes](https://github.com/allenai/olmes) |
| 5 | OLMoE | 2024-09-04 | [2409.02060](https://arxiv.org/abs/2409.02060) | [post](https://allenai.org/blog/olmoe-an-open-small-and-state-of-the-art-mixture-of-experts-model-c258432d0514) | [OLMoE](https://github.com/allenai/OLMoE) |
| 6 | Pretraining dynamics on OLMo checkpoints | 2024-10-02 | blog only | [post](https://allenai.org/blog/investigating-pretraining-dynamics-and-stability-with-olmo-checkpoints-ece6f0c4947a) | n/a |
| 7 | Tülu 3 (and 405B) | 2024-11-21 | [2411.15124](https://arxiv.org/abs/2411.15124) | [post](https://allenai.org/blog/tulu-3-technical) | [open-instruct](https://github.com/allenai/open-instruct) |
| 8 | OLMo 2 ("2 OLMo 2 Furious") | 2024-11-26 | [2501.00656](https://arxiv.org/abs/2501.00656) | [post](https://allenai.org/blog/olmo2) | [OLMo-core](https://github.com/allenai/OLMo-core) |
| 9 | OLMoTrace | 2025-04-09 | [2504.07096](https://arxiv.org/abs/2504.07096) | [post](https://allenai.org/blog/olmotrace) | [infinigram-api](https://github.com/allenai/infinigram-api) |
| 10 | DataDecide | 2025-04-15 | [2504.11393](https://arxiv.org/abs/2504.11393) | [post](https://allenai.org/blog/datadecide) | [DataDecide](https://github.com/allenai/DataDecide) |
| 11 | Critical batch size revisited | 2025-06-03 | [2505.23971](https://arxiv.org/abs/2505.23971) | [post](https://allenai.org/blog/critical-batch-size) | n/a |
| 12 | FlexOlmo | 2025-07-09 | [2507.07024](https://arxiv.org/abs/2507.07024) | [post](https://allenai.org/blog/flexolmo) | [FlexOlmo](https://github.com/allenai/FlexOlmo) |
| 13 | Signal and Noise | 2025-08-19 | [2508.13144](https://arxiv.org/abs/2508.13144) | [post](https://allenai.org/blog/signal-noise) | [signal-and-noise](https://github.com/allenai/signal-and-noise) |
| 14 | Fluid Benchmarking | 2025-09-16 | [2509.11106](https://arxiv.org/abs/2509.11106) | [post](https://allenai.org/blog/fluid-benchmarking) | [fluid-benchmarking](https://github.com/allenai/fluid-benchmarking) |
| 15 | Olmo 3 / 3.1 | 2025-11-20 | [2512.13961](https://arxiv.org/abs/2512.13961) | [post](https://allenai.org/blog/olmo3) | [OLMo-core](https://github.com/allenai/OLMo-core) |
| 16 | Bolmo (and Nature version) | 2025-12-15 | [2512.15586](https://arxiv.org/abs/2512.15586), [Nature](https://www.nature.com/articles/s41586-026-11111-4) | [post](https://allenai.org/blog/bolmo) | [bolmo-core](https://github.com/allenai/bolmo-core) |
| 17 | Olmix | 2026-02-13 | [2602.12237](https://arxiv.org/abs/2602.12237) | [post](https://allenai.org/blog/olmix) | [olmix](https://github.com/allenai/olmix) |
| 18 | Olmo Hybrid | 2026-03-05 | [2604.03444](https://arxiv.org/abs/2604.03444) | [post](https://allenai.org/blog/olmohybrid) | (HF collection) |
| 19 | BAR | 2026-04-20 | [2604.18473](https://arxiv.org/abs/2604.18473) | [post](https://allenai.org/blog/bar) | [FlexOlmo](https://github.com/allenai/FlexOlmo) |
| 20 | OlmPool | 2026-04-23 | [2608.10296](https://arxiv.org/abs/2608.10296) | [post](https://allenai.org/blog/olmpool) | [olmpool](https://github.com/allenai/olmpool) |
| 21 | EMO | 2026-05-08 | [2605.06663](https://arxiv.org/abs/2605.06663) | [post](https://allenai.org/blog/emo) | [EMO](https://github.com/allenai/EMO) |
| 22 | olmo-eval | 2026-06-12 | blog only | [post](https://allenai.org/blog/olmo-eval) | [olmo-eval](https://github.com/allenai/olmo-eval) |
| 23 | Hybrid vs transformer, token level | 2026-06-25 | [2606.20936](https://arxiv.org/abs/2606.20936) | [post](https://allenai.org/blog/hybrid-token-prediction) | n/a |
| 24 | Olmo-core 3 | 2026-10-01 | [allenai.org/papers/olmocore3](https://allenai.org/papers/olmocore3) | [post](https://allenai.org/blog/olmocore3) | [olmo-core](https://github.com/allenai/olmo-core) |

## References

Primary sources, in release order.

1. Ai2 research index: https://allenai.org/research
2. Hello OLMo, February 1, 2024: https://allenai.org/blog/hello-olmo-a-truly-open-llm-43f7e7359222
3. OLMo: Open Language Model, February 1, 2024: https://allenai.org/blog/olmo-open-language-model-87ccfc95f580
4. Groeneveld et al., "OLMo: Accelerating the Science of Language Models", arXiv 2402.00838: https://arxiv.org/abs/2402.00838
5. Soldaini et al., "Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research", arXiv 2402.00159: https://arxiv.org/abs/2402.00159
6. RewardBench, March 20, 2024: https://allenai.org/blog/rewardbench-the-first-benchmark-leaderboard-for-reward-models-used-in-rlhf-1d4d7d04a90b ; arXiv 2403.13787: https://arxiv.org/abs/2403.13787
7. Dolma moves to ODC-BY, April 15, 2024: https://allenai.org/blog/making-a-switch-dolma-moves-to-odc-by-8f0e73852f44
8. OLMo 1.7-7B, April 17, 2024: https://allenai.org/blog/olmo-1-7-7b-a-24-point-improvement-on-mmlu-92b43f7d269d
9. OLMo-7B-0724-hf model card: https://huggingface.co/allenai/OLMo-7B-0724-hf
10. OLMES, arXiv 2406.08446: https://arxiv.org/abs/2406.08446
11. OLMoE, September 4, 2024: https://allenai.org/blog/olmoe-an-open-small-and-state-of-the-art-mixture-of-experts-model-c258432d0514 ; arXiv 2409.02060: https://arxiv.org/abs/2409.02060
12. Molmo, September 25, 2024: https://allenai.org/blog/molmo ; arXiv 2409.17146: https://arxiv.org/abs/2409.17146
13. Investigating pretraining dynamics and stability with OLMo checkpoints, October 2, 2024: https://allenai.org/blog/investigating-pretraining-dynamics-and-stability-with-olmo-checkpoints-ece6f0c4947a
14. Tülu 3, November 21, 2024: https://allenai.org/blog/tulu-3 ; technical post: https://allenai.org/blog/tulu-3-technical ; arXiv 2411.15124: https://arxiv.org/abs/2411.15124
15. OLMo 2, November 26, 2024: https://allenai.org/blog/olmo2 ; arXiv 2501.00656: https://arxiv.org/abs/2501.00656
16. Tülu 3 405B, January 30, 2025: https://allenai.org/blog/tulu-3-405b
17. OLMoE, meet iOS, February 11, 2025: https://allenai.org/blog/olmoe-app
18. olmOCR, February 25, 2025: https://allenai.org/blog/olmocr ; arXiv 2502.18443: https://arxiv.org/abs/2502.18443
19. OLMo 2 32B, March 13, 2025: https://allenai.org/blog/olmo2-32b
20. OLMoTrace, April 9, 2025: https://allenai.org/blog/olmotrace ; arXiv 2504.07096: https://arxiv.org/abs/2504.07096
21. DataDecide, April 15, 2025: https://allenai.org/blog/datadecide ; arXiv 2504.11393: https://arxiv.org/abs/2504.11393
22. OLMo-2-0425-1B model card: https://huggingface.co/allenai/OLMo-2-0425-1B
23. Critical batch size, June 3, 2025: https://allenai.org/blog/critical-batch-size ; arXiv 2505.23971: https://arxiv.org/abs/2505.23971
24. FlexOlmo, July 9, 2025: https://allenai.org/blog/flexolmo ; arXiv 2507.07024: https://arxiv.org/abs/2507.07024
25. Signal and Noise, August 19, 2025: https://allenai.org/blog/signal-noise ; arXiv 2508.13144: https://arxiv.org/abs/2508.13144
26. OLMoASR, August 28, 2025: https://allenai.org/blog/olmoasr
27. Fluid Benchmarking, September 16, 2025: https://allenai.org/blog/fluid-benchmarking ; arXiv 2509.11106: https://arxiv.org/abs/2509.11106
28. olmOCR 2, October 22, 2025: https://allenai.org/blog/olmocr-2 ; arXiv 2510.19817: https://arxiv.org/abs/2510.19817
29. OlmoEarth, November 4, 2025: https://allenai.org/blog/olmoearth
30. DR Tulu, November 18, 2025: https://allenai.org/blog/dr-tulu ; arXiv 2511.19399: https://arxiv.org/abs/2511.19399
31. Olmo 3, November 20, 2025 (3.1 update December 12): https://allenai.org/blog/olmo3 ; arXiv 2512.13961: https://arxiv.org/abs/2512.13961
32. Olmo-3.1-32B-Think model card: https://huggingface.co/allenai/Olmo-3.1-32B-Think ; Olmo-3.1-32B-Instruct: https://huggingface.co/allenai/Olmo-3.1-32B-Instruct
33. Bolmo, December 15, 2025: https://allenai.org/blog/bolmo ; arXiv 2512.15586: https://arxiv.org/abs/2512.15586
34. Olmix, February 13, 2026: https://allenai.org/blog/olmix ; arXiv 2602.12237: https://arxiv.org/abs/2602.12237
35. Olmo Hybrid, March 5, 2026: https://allenai.org/blog/olmohybrid ; arXiv 2604.03444: https://arxiv.org/abs/2604.03444
36. BAR, April 20, 2026: https://allenai.org/blog/bar ; arXiv 2604.18473: https://arxiv.org/abs/2604.18473
37. OlmPool, April 23, 2026: https://allenai.org/blog/olmpool ; arXiv 2608.10296: https://arxiv.org/abs/2608.10296
38. EMO, May 8, 2026: https://allenai.org/blog/emo ; arXiv 2605.06663: https://arxiv.org/abs/2605.06663
39. olmo-eval, June 12, 2026: https://allenai.org/blog/olmo-eval
40. Which tokens does a hybrid model predict better?, June 25, 2026: https://allenai.org/blog/hybrid-token-prediction ; arXiv 2606.20936: https://arxiv.org/abs/2606.20936
41. Olmo-core 3, October 1, 2026: https://allenai.org/blog/olmocore3 ; report: https://allenai.org/papers/olmocore3
42. Bolmo in Nature, October 7, 2026: https://allenai.org/blog/bolmo-nature ; https://www.nature.com/articles/s41586-026-11111-4
43. Ai2 at COLM 2026, October 9, 2026: https://allenai.org/blog/colm-2026-recap
