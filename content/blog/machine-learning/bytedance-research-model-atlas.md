---
title: "The ByteDance Research Model Atlas: A Field Guide to 30+ Open Models, Organized by What They Do"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A principal-engineer field guide to ByteDance Research's open releases, organized by use case, with the cross-cutting research threads — reward learning, synthetic data, unification — made explicit so you know which model to reach for."
tags: ["bytedance", "open-models", "image-generation", "video-generation", "multimodal", "time-series", "llm-agents", "reward-learning", "synthetic-data", "survey", "vlm"]
category: "machine-learning"
subcategory: "Research Survey"
author: "Hiep Tran"
featured: true
readTime: 51
---

If you open the `bytedance-research` organization on HuggingFace cold, you get a wall. Thirty-plus model repos with names like UNO, USO, UMO, OneReward, Phantom, HuMo, ChatTS, Timer-S1, Valley2, PaSa, ToolHop, MammothModa, Lance, Vidi — plus a handful of datasets — and almost no narrative connecting them. The README on each repo tells you what that one model does. None of them tells you why these specific thirty exist, what they have in common, or which one you should actually download for the job in front of you. That is the gap this post fills.

The thesis is simple: those thirty-plus releases are not thirty unrelated bets. They are **five problem families**, and inside each family the models form a tight lineage where each release lifts one constraint from the last. More importantly, three or four research threads — reward learning from human preference, large-scale synthetic data generation, and an obsessive push toward *unification* — recur across every family. Once you see the threads, the atlas stops being a wall and becomes a map you can navigate by intent: "I need to put a product into a new scene" routes to one model; "I need a talking-head video driven by audio" routes to another; "I need to reason over a multivariate time series" routes to a third.

![The ByteDance Research model atlas as a taxonomy tree: root organization branching into five problem families, each with its representative models](/imgs/blogs/bytedance-research-model-atlas-1.webp)

The diagram above is the mental model: read top-down, the organization collapses into five families — controllable image generation, controllable video, multimodal understanding, time-series intelligence, and agents/reasoning/evaluation — and each family holds two to four representative models. The rest of this article is a tour of that tree, family by family, with a comparison table for each, the cross-cutting threads called out where they appear, and a closing decision guide so you leave knowing exactly which checkpoint to pull. This is the hub; the per-model deep dives ([PaSa](/blog/machine-learning/ai-agent/pasa-llm-paper-search-agent), [ToolHop](/blog/machine-learning/ai-agent/toolhop-multi-hop-tool-use-benchmark), [Web-Bench](/blog/machine-learning/ai-agent/web-bench-llm-web-development-benchmark), [ChatTS](/blog/machine-learning/signal-processing/chatts-aligning-time-series-llms), [Timer-S1](/blog/machine-learning/signal-processing/timer-s1-time-series-foundation-model)) link back here.

## Why a "by use case" atlas beats the model card

Most model-zoo writeups organize by modality (text, image, video) or by release date. Both are useless when you have a task. A modality split tells you UNO and OneReward are both "image" models, but they solve opposite problems — one is subject customization, the other is mask-guided editing with a reward model attached. A date split tells you Phantom shipped before HuMo, which tells you nothing about whether you want subject-consistent video or audio-driven talking heads. The only split that survives contact with a real engineering decision is **by what the model does for you**.

Here is the assumption-vs-reality table that motivates the whole structure:

| Common assumption | What it gets wrong | The reality |
|---|---|---|
| "These are 30 separate research projects" | Treats each as a one-off | They share data pipelines, reward models, and training infra across families |
| "Pick the newest model in the modality" | Ignores task fit | The newest model often solves a *different* sub-problem than yours |
| "Image customization is one thing" | Conflates subject, style, identity, editing | RealCustom, UNO, USO, UMO, OneReward each own a distinct slice |
| "A VLM is a VLM" | Ignores the head | Valley2 understands; MammothModa understands *and* generates |
| "Reward learning only matters for chat LLMs" | Misses the pattern | Reward/preference learning shows up in image, video, and agent training here |

The last row is the one that surprised me most when I traced the releases. Reward learning — the technique everyone associates with RLHF on chat models — is the connective tissue across ByteDance Research's *vision* work. OneReward is a vision-language reward model. USO has a "style reward learning" stage. UMO uses a matching reward. PaSa is PPO-trained against a reward signal. That is not a coincidence; it is a house style, and we will keep returning to it.

One practical note on scope before the tour. The `bytedance-research` organization is the *research* surface — roughly 32 models and a half-dozen datasets at the time of writing — and it overlaps with, but is not identical to, the larger `ByteDance` org on HuggingFace (which carries productized lines like the Bernini video models and the Ouro language models, and counts model and dataset numbers in the dozens-to-hundreds). This atlas covers the research releases that form the five families; if a name you are looking for is missing, check the broader `ByteDance` org, because the boundary between "research artifact" and "shipped product" is porous here — OneReward's research recipe became the Seedream 3.0 Fill product, Valley's research weights ship under `bytedance/Valley`, and the line keeps moving. Read the families as research *threads* that graduate into products, not as a closed catalog.

### The release cadence tells you these teams share infrastructure

![Timeline of headline ByteDance Research releases from 2024 to 2026, interleaving the five families](/imgs/blogs/bytedance-research-model-atlas-2.webp)

Lay the headline papers on a timeline and the families interleave rather than cluster. RealCustom (image) in early 2024, ChatTS (time series) at the end of 2024, PaSa and Valley2 (agent and VLM) in January 2025, Phantom (video) in February, UNO and Vidi in April, USO and OneReward in August, HuMo in September, MammothModa2 in late 2025, Timer-S1 in early 2026, UMO at CVPR'26. If these were siloed teams each grinding on one modality you would see runs of same-color releases. Instead the colors alternate, which is the signature of shared training infrastructure, shared data tooling, and shared reward-modeling code being applied across problem domains in parallel. Read the threads *across* the timeline, not down any single family.

## The three threads that connect everything

Before the family tour, internalize the three threads. They are the reason this atlas is more than a list.

![Matrix showing which cross-cutting thread — reward learning, synthetic data, unification, RL training — appears in each of the five families](/imgs/blogs/bytedance-research-model-atlas-3.webp)

The matrix above is the proof. Rows are the five families; columns are the four recurring techniques. Almost every cell is filled, and the two densest columns — reward/preference and synthetic data — are filled in every family. That density is the story.

**Thread 1: reward learning as a universal training tool.** The classic recipe for a generative model is supervised fine-tuning on paired data. ByteDance Research keeps replacing or augmenting that with a learned reward. OneReward trains *one* vision-language model to act as the reward signal for four different image-editing tasks. USO adds a "style reward learning" (SRL) stage on top of disentanglement training. UMO frames multi-identity consistency as a *matching reward* computed with a global assignment. PaSa optimizes its agent with PPO against a reward model. The bet is that a good reward model generalizes across tasks better than task-specific supervised labels, and you will see it pay off in the image family especially.

**Thread 2: synthetic data generation as the data strategy.** Every one of these models needs paired data that does not exist at scale in the wild. There is no natural corpus of "the same subject rendered in ten scenes," no corpus of "multivariate time series with rich natural-language descriptions," no corpus of "fine-grained academic queries with their complete answer sets." So they manufacture it. UNO builds a "highly-consistent data synthesis pipeline" and a progressive synthesis procedure with multi-stage filtering, releasing the UNO-1M dataset. ChatTS uses an attribute-based synthetic generator plus "Time Series Evol-Instruct." PaSa builds AutoScholarQuery (35,000 synthetic queries). ToolHop uses query-driven construction (tool creation → document refinement → code generation). HuMo constructs a paired text-image-audio dataset (HuMoSet). Synthetic data is not a shortcut here; it is the core competency.

**Thread 3: unification.** The most visible architectural trend is collapsing things that used to be separate models into one. UNO unifies single-subject and multi-subject generation. USO unifies style-driven and subject-driven generation — two objectives that were previously treated as conflicting. UMO unifies multiple identities in one image. MammothModa unifies understanding *and* generation in one autoregressive-plus-diffusion framework. Lance pushes toward any-to-any multi-task synergy. The throughline: fewer checkpoints, each conditioned on what you hand it, instead of a zoo of fragile per-task models.

A fourth, narrower thread — **disentanglement plus in-context conditioning** — is the specific mechanical recipe behind the image-customization family, and we will dwell on it there.

### Why these threads reinforce each other

It is tempting to read the three threads as independent stylistic preferences. They are not; they are mutually load-bearing, and seeing the dependency is what turns the atlas from trivia into a model of how this org operates.

Start with the dependency between synthetic data and unification. A unified model — one checkpoint that handles single- and multi-subject, or both understanding and generation — needs training data that spans all the sub-tasks it is supposed to unify. If you only have single-subject paired data, you cannot train a model to do multi-subject inference, no matter how clever the architecture. So the moment you commit to unification, you have committed to manufacturing the data that covers the union of tasks. UNO is the cleanest example: the whole "less-to-more" framing is "use the model's own in-context ability to synthesize the multi-subject data that lets you train the multi-subject model." Unification *forces* synthetic data.

Now the dependency between reward learning and unification. Supervised fine-tuning needs labels for each task; the more tasks you unify, the more per-task labeled datasets you need, and the labeling cost scales with the number of tasks. A reward model breaks that scaling: one OneReward VLM scores fill, extend, removal, and text rendering, so adding a fifth editing task means teaching the reward model a fifth criterion, not building a fifth supervised dataset. Reward learning is the labeling strategy that makes unification affordable. That is why the densest two columns in the matrix — reward/preference and synthetic data — are also the two that appear in every family: they are the enabling technologies for the unification the org keeps chasing.

Finally, reward learning and synthetic data reinforce each other directly. A reward model trained on human preferences can *rank* synthetic outputs, which lets you filter a noisy synthetic corpus down to a clean one (UNO's "multi-stage data filtering," ChatTS's curated descriptions, PaSa's reward-shaped trajectories). And conversely, synthetic data gives you the volume of comparisons a reward model needs to train. The loop is: synthesize candidates → reward-model ranks them → keep the winners → train on the winners → use the better model to synthesize cleaner candidates. That loop is the substrate underneath the whole atlas.

### The shared substrate hypothesis

If you accept that the three threads reinforce each other, a prediction follows: the org should have a *shared* internal toolkit — a data-synthesis framework and a reward-modeling framework — reused across image, video, time-series, and agents, rather than re-implemented per project. The interleaved release timeline supports this. Teams shipping in different modalities within the same month, all using the same three techniques, is the fingerprint of shared infrastructure. When you adopt one of these models, you are not just adopting a checkpoint; you are adopting the output of that substrate, and the most transferable lesson is often the *recipe* (in-context conditioning, content-style disentanglement, one-reward-many-tasks) rather than the weights. We will see the same recipe reappear in family after family below.

With the threads in hand, here is the tour.

## 1. Controllable image generation and customization {#image}

> The single hardest problem in subject customization is keeping the subject recognizable without freezing it. Freeze too hard and you get a copy-paste sticker; freeze too soft and you get "vaguely similar." Every model in this family is a different answer to that tension.

This is the signature cluster — the Intelligent Creation Lab work — and it is where the disentanglement-plus-in-context recipe and reward learning are both most mature. Five releases form a clear lineage.

![Graph of the image-customization lineage: RealCustom feeding UNO, UNO feeding USO, USO feeding UMO, with OneReward supplying reward signal](/imgs/blogs/bytedance-research-model-atlas-4.webp)

The graph above is the family map. RealCustom establishes the core trick; UNO generalizes from single to multiple subjects via in-context generation; USO disentangles content from style; UMO pushes to multiple identities; and OneReward sits to the side, feeding a learned reward signal into the later models. Trace it left to right and each arrow is "removed one constraint."

### RealCustom and RealCustom++: narrow the real text word {#realcustom}

**Senior rule of thumb: the subject should influence only the pixels it should influence, and nothing else.** RealCustom's contribution (CVPR 2024, arxiv [2403.00483](https://arxiv.org/abs/2403.00483)) is a train-inference decoupled framework that "disentangles similarity from controllability by precisely limiting subject influence to relevant parts only." The clever part is the naming: rather than inventing a pseudo-word like the textual-inversion lineage does (`<sks dog>`), RealCustom takes a *real* word already in the prompt — "dog" — and gradually narrows its meaning from the generic concept down to your specific subject. During training, an adaptive scoring module learns the alignment between the visual condition and the textual condition. At inference, an adaptive mask-guidance strategy iteratively refines *where* and *how much* the subject is allowed to influence the generation, using cross-attention to figure out which regions are relevant.

Why this matters in practice: the dominant failure mode of subject customization is *leakage* — the background, lighting, or pose of the reference image bleeds into the output even when the prompt says "on a beach at sunset." By scoping subject influence to the cross-attention regions that correspond to the narrowed word, RealCustom keeps the rest of the canvas obedient to the text. RealCustom++ ([2408.09744](https://arxiv.org/abs/2408.09744)) extends this to be faster and more open-domain. If you have ever fought a LoRA that overfit a face onto every surface in the frame, this is the conceptual fix.

The RealCustom intuition, in pseudo-numpy: subject influence is masked to the cross-attention regions of the "narrowed" real word, not the whole image. A tight mask (low `leakage_budget`) keeps backgrounds obedient to the prompt; a loose mask reproduces the reference scene and ignores "on a beach."

```python
import numpy as np

def narrowed_word_guidance(cross_attn, subject_feats, threshold=0.6):
    """Scope subject influence to where the narrowed word attends.

    cross_attn: [H, W] attention map for the real word, e.g. "dog"
    subject_feats: reference-image features to inject
    """
    mask = (cross_attn >= threshold).astype(np.float32)   # relevant region only
    # influence the latent ONLY where the word attends; leave the rest to text
    injected = mask[..., None] * subject_feats
    leakage_budget = mask.mean()        # fraction of canvas the subject can touch
    return injected, leakage_budget
```

### UNO and UNO-1M: less-to-more generalization via in-context generation {#uno}

**Senior rule of thumb: if you can generate clean multi-subject training data, multi-subject inference becomes almost free.** UNO ("Less-to-More Generalization: Unlocking More Controllability by In-Context Generation," ICCV 2025, arxiv [2504.02160](https://arxiv.org/abs/2504.02160)) is the model I reach for most often in this family, because it solves the data problem that gates everyone else. The abstract describes it as "a multi-image conditioned subject-to-image model iteratively trained from a text-to-image model," with two named mechanisms: **progressive cross-modal alignment** and **universal rotary position embedding**.

The key insight is "less-to-more." Start from a text-to-image model. Use its own in-context generation ability to synthesize *highly consistent* paired data — the same subject across multiple images — via a "highly-consistent data synthesis pipeline" that leverages the diffusion transformer's capabilities. Then train iteratively: single-subject first, then multi-subject, each stage bootstrapping cleaner data for the next. The released UNO-1M dataset is the artifact of that pipeline. This is thread 2 (synthetic data) and thread 3 (unification — single→multi in one model) operating together.

Why "in-context generation" is the right frame: instead of treating each reference subject as a separate conditioning branch with its own adapter, UNO feeds the references as additional image tokens *in context*, the way an LLM reads few-shot examples. The universal RoPE keeps positional information coherent across the concatenated reference and target tokens, which is what prevents the multi-subject confusion that plagues naive concatenation. The practical payoff: you can hand UNO one product photo and one model photo and ask for "the model holding the product," and the two identities stay distinct.

In UNO-style in-context conditioning, references are concatenated as image tokens rather than routed through separate adapters, and a universal RoPE keeps positions coherent across the whole sequence:

```python
import torch

def in_context_tokens(target_latent, reference_latents, rope):
    """Concatenate subject references in-context, few-shot style."""
    refs = torch.cat(reference_latents, dim=0)             # all subjects in-context
    seq = torch.cat([refs, target_latent], dim=0)         # refs THEN target
    # one shared positional scheme over the whole sequence avoids subject mixing
    seq = rope(seq, offsets=cumulative_lengths(reference_latents))
    return seq    # the DiT attends across refs+target jointly, few-shot style
```

### USO: unifying style and subject via disentanglement and reward {#uso}

**Senior rule of thumb: style and subject pull in opposite directions, so you must explicitly separate them before you can satisfy both.** USO ("USO: Unified Style and Subject-Driven Generation via Disentangled and Reward Learning," arxiv [2508.18966](https://arxiv.org/abs/2508.18966)) takes on the conflict that prior work dodged. Style-driven generation wants "make it look like *this* painting"; subject-driven generation wants "render *this* exact object." Train them together naively and they fight — the style transfer smears the subject's identity, or the subject fidelity flattens the style.

USO's recipe is two-part: **content-style disentanglement training** plus **style-alignment training**, and then a **style reward learning** (SRL) stage on top. The disentanglement learns to represent content and style in separable factors; the reward stage (thread 1) tunes the model toward outputs that humans judge as both stylistically faithful and subject-accurate. The authors ship USO-Bench, described as "the first benchmark that jointly evaluates style similarity and subject fidelity across multiple metrics," precisely because the two-axis evaluation did not previously exist. If your job is "match our brand's illustration style *and* feature this specific product," USO is the model built for exactly that intersection.

### UMO: multi-identity consistency via matching reward {#umo}

**Senior rule of thumb: when you have N identities and N slots, the problem is an assignment problem, so solve it globally.** UMO (CVPR 2026) pushes the family to its multi-identity limit: many people, all of whom must stay recognizably themselves, in one image. The mechanism is a **multi-to-multi matching reward** computed via global assignment — instead of greedily matching each generated face to the nearest reference (which collapses two similar people into one), it computes the optimal global assignment between generated identities and reference identities and rewards consistency under that assignment. This is thread 1 again, but applied to a combinatorial structure. The matching-reward framing is what prevents identity bleed in group photos, the multi-person analog of the leakage problem RealCustom solved for single subjects.

### OneReward: one reward model for all mask-guided editing {#onereward}

**Senior rule of thumb: don't fine-tune a separate model per editing task — train one reward model that can judge them all.** OneReward ("OneReward: Unified Mask-Guided Image Generation via Multi-Task Human Preference Learning," arxiv [2508.21066](https://arxiv.org/abs/2508.21066)) is the purest expression of thread 1. It is "a unified reinforcement learning framework that enhances the model's generative capabilities across multiple tasks under different evaluation criteria using only One Reward model." A single vision-language model serves as the generative reward model: given a task and an evaluation criterion, it distinguishes the winner from the loser.

The tasks are the four mask-guided editing operations: **image fill (inpainting), image extend (outpainting), object removal, and text rendering** — each driven by a binary mask marking the edit region. The training method is "multi-task reinforcement learning directly on a pre-trained base model, eliminating the need for task-specific SFT." That last clause is the engineering win: no per-task supervised fine-tuning, no per-task data pipeline, just one reward model and multi-task RL. The application built on it is Seedream 3.0 Fill, and the paper reports that the unified edit model "consistently outperforms both commercial and open-source competitors, such as Ideogram, Adobe Photoshop, and FLUX Fill [Pro], across multiple evaluation dimensions."

Note what OneReward implies for the rest of the family: a single VLM reward model that can score "is this inpaint good?" is the same kind of object you need to score "is this subject-faithful and style-faithful?" — which is why OneReward directly feeds the USO/UMO reward stages in the lineage graph. One reward model, many generative tasks, is the house pattern.

### Supporting players: DreamFit, HyperLoRA, LVFace

Three more image-family releases round it out. **DreamFit** is garment-centric — virtual try-on, where the constraint is that the garment's texture and cut survive being draped on a new body and pose. **HyperLoRA** is a hypernetwork that *generates* a LoRA on the fly for zero-shot portrait customization; instead of training a LoRA per identity, a hypernetwork emits the LoRA weights from a single reference, which is a clever fusion of the LoRA and customization literatures. **LVFace** is a large-scale face-recognition / feature-extraction model — the kind of identity encoder the matching rewards in UMO and the fidelity metrics in USO-Bench lean on internally.

### Image family — which one when

| Model | Problem it solves | Key idea | Reach for it when |
|---|---|---|---|
| RealCustom / ++ | Single-subject, open-domain, real-time | Narrow a real text word; mask subject influence | One subject, you need speed and low leakage |
| UNO | Single → multi-subject | In-context references + universal RoPE; UNO-1M data | Two or more distinct subjects in one scene |
| USO | Style + subject jointly | Content-style disentanglement + style reward (SRL) | Brand style *and* a specific product/person |
| UMO | Many identities, all consistent | Multi-to-multi matching reward (global assignment) | Group photos, multiple people kept distinct |
| OneReward | Mask-guided editing (fill/extend/remove/text) | One VLM reward model, multi-task RL, no per-task SFT | Inpaint/outpaint/removal/text in production |
| DreamFit | Virtual try-on | Garment-centric conditioning | E-commerce apparel |
| HyperLoRA | Zero-shot portrait | Hypernetwork emits LoRA from one reference | Per-user avatars without per-user training |
| LVFace | Face features | Large face-recognition backbone | Identity encoding / verification |

## 2. Controllable video generation {#video}

> Video customization inherits every image-customization problem and adds two of its own: temporal consistency and the brutal cost of paired video data. The releases here are mostly about manufacturing the data and fixing the cross-modal leakage that breaks naive approaches.

Three releases anchor this family, and the data-synthesis thread dominates because clean conditioned-video data essentially does not exist at scale.

### Phantom: subject-consistent video via cross-modal alignment {#phantom}

**Senior rule of thumb: the reason your subject-conditioned video looks like a slideshow of the reference image is content leakage — fix the alignment, not the resolution.** Phantom ("Phantom: Subject-consistent video generation via cross-modal alignment," arxiv [2502.11079](https://arxiv.org/abs/2502.11079)) is a unified framework that "extracts subject elements from reference images and generates subject-consistent videos following textual instructions," for both single and multi-subject references. It is built on existing text-to-video and image-to-video architectures, but it **redesigns the joint text-image injection model to learn cross-modal alignment** — and the lever that makes this work is the training data: **text-image-video triplets**.

The two failure modes Phantom explicitly targets are the canonical ones: **image content leakage** (the static reference image's scene leaks into the video and freezes the motion) and **multi-subject confusion** (two reference subjects get mixed into a single inconsistent character). Triplet supervision — text, reference image(s), and the resulting video — is what teaches the model to take *identity* from the image and *motion/scene* from the text, instead of copying the whole image forward. The authors emphasize subject consistency in human generation, which subsumes ID-preserving video generation, and report that Phantom "outperforms other state-of-the-art closed-source commercial solutions." This is thread 2 (synthetic/curated triplet data) doing the heavy lifting.

### HuMo: human-centric video via collaborative multimodal conditioning {#humo}

**Senior rule of thumb: if lip-sync matters, you cannot bolt audio on after the fact — it has to be a first-class conditioning signal.** HuMo ("HuMo: Human-Centric Video Generation via Collaborative Multi-Modal Conditioning," arxiv [2509.08519](https://arxiv.org/abs/2509.08519)) synthesizes human videos from **text, image, and audio together** through collaborative multimodal control. This is the model for talking-head and performance video where the mouth has to match the audio. Its named contributions read like a checklist of the things that break when you naively combine three modalities:

- A **two-stage progressive multimodal training** paradigm — bring the modalities in gradually rather than all at once.
- **Minimal-invasive image injection** for subject preservation — inject the reference identity without overwriting the rest of the generation (the same anti-leakage instinct as Phantom and RealCustom).
- A **focus-by-predicting** strategy for audio-visual synchronization — the mechanism that drives lip-sync.
- **Time-adaptive classifier-free guidance** at inference for controlling how strongly each modality steers the output over the course of the video.

HuMo is paired with a constructed dataset of diverse, paired text, reference images, and audio — the brief identifies this as HuMoSet at roughly 670K samples. Once again: the data is manufactured because the natural corpus does not exist. If your use case is "a digital presenter that says these words with this voice and this face," HuMo is purpose-built for it.

### ATI: any-trajectory-instruction control {#ati}

**Senior rule of thumb: the most intuitive motion-control interface is a drawn line, not a text description.** ATI is the motion-control member of the family: controllable video by *drawing trajectories*. Instead of describing motion in words ("the car drives left to right, then turns"), you draw the path you want an object or the camera to follow, and the model generates video that obeys those any-trajectory instructions. This is a different control axis from Phantom (identity) and HuMo (audio/performance) — it is about *where things go*. For storyboard-to-video and previsualization workflows, trajectory control is the interface that matches how directors already think.

The three video control axes are deliberately orthogonal, and that orthogonality is the design principle worth internalizing. Identity (Phantom) answers *who/what is in the frame*; audio/performance (HuMo) answers *what they say and how the mouth moves*; trajectory (ATI) answers *where things move*. A production pipeline rarely needs only one. A polished ad spot might want Phantom's subject consistency for the character, HuMo's audio sync for the spoken lines, and ATI's trajectory control for the camera move — which is exactly why the org ships them as separate, composable control signals rather than one monolithic "video model with every knob." Composable control beats a single over-parameterized model because you can mix the axes you need and ignore the ones you do not, and because each axis can be improved independently without retraining the others. When you evaluate these for a project, map your requirement to the axis first; the model follows from the axis.

### Video family — which one when

| Model | Control signal | Key idea | Reach for it when |
|---|---|---|---|
| Phantom | Reference image(s) + text | Cross-modal alignment via text-image-video triplets | Subject-consistent video, single or multi-subject |
| HuMo | Text + image + audio | Collaborative multimodal conditioning; focus-by-predicting sync | Talking-head / performance video with lip-sync |
| ATI | Drawn trajectories + text | Any-trajectory-instruction motion control | Directed motion, camera paths, storyboard-to-video |

The shared anti-pattern these three avoid is worth stating once: **do not treat the reference as a frame to copy forward.** Phantom's alignment, HuMo's minimal-invasive injection, and ATI's trajectory conditioning are all variations on "take the right thing from the conditioning and let the rest be generated."

## 3. Multimodal understanding {#vlm}

> A VLM is a stack, and the most important decision is what sits at the top of it. Swap an autoregressive text head for a diffusion head and your "understanding" model becomes a "generation" model — same body, different mouth.

This family is about reading images and video, not making them. It spans a scalable production VLM line (Valley), a unified understand-and-generate line (MammothModa, Lance), and a video-specialist line (Vidi).

![Layered stack of the multimodal-understanding architecture: vision encoder, adapter, LLM backbone, and the output head choosing autoregressive text versus diffusion pixels](/imgs/blogs/bytedance-research-model-atlas-6.webp)

The stack above is the mental model for the whole family. Pixels enter at the vision encoder (a large-visual-vocabulary ViT/SigLIP-class model), pass through an adapter that compresses and aligns visual tokens to the LLM's embedding space, get reasoned over by an LLM backbone, and exit through a head. The head is the fork in the road: an autoregressive text head gives you understanding; a diffusion head gives you generation. Everything in this family is a choice of components in that stack.

### Valley2 and the Valley line: a scalable production VLM {#valley}

**Senior rule of thumb: a VLM that wins a leaderboard is not the same as a VLM that wins on your e-commerce catalog — Valley optimizes for the latter.** Valley2 ("Valley2: Exploring Multimodal Models with Scalable Vision-Language Design," arxiv [2501.05901](https://arxiv.org/abs/2501.05901)) is a multimodal LLM with a scalable vision-language design, targeted explicitly at **e-commerce and short-video** scenarios. The architecture, per the brief, leans on a **large visual vocabulary**, a **convolutional adapter (ConvAdapter)** to compress visual tokens efficiently, and an **Eagle module**. The reported results are concrete: it surpasses comparable open-source models on e-commerce benchmarks (**79.66 vs. 72.76**) and ranks **second among models with fewer than 10B parameters on the OpenCompass leaderboard with an average score of 67.4**. Weights are open at `bytedance/Valley`.

The Valley line did not stop at Valley2. The family extends to **Valley-Eagle**, **Valley2.5**, and **Valley3**, spanning **8B and 32B** parameter sizes and shipping in both **Instruct** and **Think** variants. The Think variants are the reasoning-tuned siblings — the same understanding stack with a reasoning post-training stage, which is where the RL-training thread (column 4 of the matrix) shows up in this family. The two-size, two-mode matrix (8B/32B × Instruct/Think) is a productization pattern: pick the size your latency budget allows and the mode your task needs.

A Valley-style stack in transformers-shaped pseudocode uses a large visual vocabulary, a ConvAdapter to shrink visual tokens, an Eagle module, then the LLM:

```python
import torch, torch.nn as nn

class ValleyStack(nn.Module):
    def __init__(self, encoder, conv_adapter, eagle, llm):
        super().__init__()
        self.encoder = encoder          # large visual vocabulary ViT
        self.conv_adapter = conv_adapter# conv compression: fewer, denser tokens
        self.eagle = eagle              # auxiliary visual module
        self.llm = llm                  # Qwen-class backbone, 8B or 32B

    def forward(self, pixels, input_ids):
        v = self.encoder(pixels)                 # [B, N_patch, D]
        v = self.conv_adapter(v)                 # [B, N_patch/k, D]  fewer tokens
        v = self.eagle(v)                        # refined visual features
        toks = torch.cat([v, self.llm.embed(input_ids)], dim=1)
        return self.llm(inputs_embeds=toks)      # AR text head = understanding
```

### MammothModa and MammothModa2: one model that understands and generates {#mammoth}

**Senior rule of thumb: the cleanest way to keep understanding and generation consistent is to make them the same model.** MammothModa2 ("MammothModa2," arxiv [2511.18262](https://arxiv.org/abs/2511.18262)) is a **unified autoregressive-plus-diffusion framework for understanding AND generation in one model** — thread 3 (unification) at its most ambitious. The argument is that a model which can both interpret an image and produce one shares a representation, and that shared representation makes the two capabilities mutually reinforcing: understanding grounds generation, generation pressure-tests understanding. Architecturally this is the "two heads on one body" picture from the stack diagram — an AR head for text/understanding and a diffusion head for pixels/generation — trained jointly. This is the harder, higher-ceiling bet compared to a pure-understanding model like Valley, and it is where the AR-plus-diffusion column of the matrix lives.

### Lance: any-to-any via multi-task synergy {#lance}

**Lance** ("Unified Multimodal Modeling by Multi-Task Synergy," arxiv [2605.18678]) pushes unification further to **any-to-any** — one model that maps among modalities in many directions, with the explicit thesis that the tasks *help each other* (multi-task synergy) rather than competing for capacity. It is the most general statement of the unification thread in the understanding family: not just understand-and-generate, but a single model whose multi-task training makes each capability stronger.

The "multi-task synergy" claim deserves scrutiny because the default expectation is the opposite. Multi-task training usually causes *interference*: tasks compete for the same parameters, and the model ends up mediocre at everything (the "no free lunch" of joint training). Lance's bet — and MammothModa2's, and the whole unification thread's — is that with the right shared representation and the right task mixture, the tasks become *complementary* instead of competitive. The intuition for why this can work: a representation good enough to *generate* a faithful image must encode fine-grained visual structure, and that same fine-grained structure is exactly what *understanding* fine-grained questions needs. Generation is a harder constraint than understanding, so training to generate can pull the understanding representation up with it. Whether that holds in practice is an empirical question the benchmarks have to answer, and it is the single most important thing to verify before betting a product on a unified model: confirm the unified checkpoint is not *worse* at your specific sub-task than a specialist would be. If unification cost you measurable quality on the one task you care about, the operational savings may not be worth it — which is the honest counterweight to the unification thesis, and why Valley (a deliberate non-unified specialist) still exists alongside MammothModa.

### Vidi and Vidi2: LMMs for video understanding and editing {#vidi}

**Senior rule of thumb: for intelligent video editing, the hard part is not generation — it is finding the exact time range that matches a query.** Vidi ("Vidi: Large Multimodal Models for Video Understanding and Editing," arxiv [2504.15681]) is the video-understanding specialist. It handles vision, audio, and text over flexible input lengths — including hour-long videos — and its emphasized primary task is **temporal retrieval**: identifying the specific time ranges within a video that correspond to a text query, which the authors call critical for intelligent editing. The companion benchmark, **VUE-TR** (Video Understanding and Editing — Temporal Retrieval), features significantly longer videos than prior datasets, audio-based query support, diverse query formats, manually annotated ground-truth time ranges, and a refined IoU metric for evaluating multiple time ranges. The paper reports Vidi "significantly outperforms leading proprietary models, e.g., GPT-4o and Gemini, on the temporal retrieval task." Vidi2 and Vidi2.5 ([2511.19529]) extend the line to broader spatio-temporal grounding and video QA. If your product needs "jump to the part where the speaker mentions pricing," that is temporal retrieval, and Vidi is the model built for it.

### Understanding family — which one when

| Model | What it does | Head / architecture | Reach for it when |
|---|---|---|---|
| Valley2 / Valley3 | Image+video understanding | Encoder → ConvAdapter → Eagle → LLM (AR head) | E-commerce, short-video, production VLM at 8B/32B |
| Valley *-Think* | Reasoning over multimodal input | Valley stack + reasoning post-training | Multimodal tasks needing chain-of-thought |
| MammothModa2 | Understanding + generation | Unified AR + diffusion heads | One model for both reading and making images |
| Lance | Any-to-any multimodal | Multi-task synergy, unified | Many modality directions, shared capacity |
| Vidi / Vidi2 | Video understanding + editing | LMM with temporal grounding | Temporal retrieval, "find the clip where…" |

## 4. Time-series intelligence {#timeseries}

> Time series is the modality that machine learning keeps re-discovering and keeps treating as a second-class citizen. ByteDance Research's two releases here are bets that it deserves the same two treatments text got: alignment with an LLM, and a true foundation model.

Two releases, two different and complementary strategies, and the synthetic-data thread is central to both.

### ChatTS: aligning time series with LLMs via synthetic data {#chatts}

**Senior rule of thumb: an LLM cannot reason about a time series it cannot read — and the data to teach it to read does not exist, so you synthesize it.** ChatTS ("ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning," VLDB 2025, arxiv [2412.03104](https://arxiv.org/abs/2412.03104)) treats **multivariate time series as a native modality**, the way a VLM treats an image. The core problem is the shortage of high-quality datasets pairing time series with text, and the solution is two complementary synthetic-data methods (thread 2 in its purest form):

1. An **attribute-based synthetic data generator** that produces time series with detailed natural-language descriptions — you define attributes (trend, seasonality, spikes, level shifts) and generate series plus their descriptions together, so the text and the signal are aligned by construction.
2. **Time Series Evol-Instruct**, which generates varied question-answer pairs to strengthen reasoning, analogous to the Evol-Instruct technique from the instruction-tuning literature applied to temporal data.

The architecture pairs a **context-aware time-series encoder** with the LLM. The reported gains are substantial: a **46.0% improvement in alignment tasks** and a **25.8% improvement in reasoning tasks** over existing vision-based multimodal LLMs (including GPT-4o) and text/agent-based approaches, across six alignment tasks and four reasoning tasks on real-world benchmark data. ChatTS is the model for "explain what happened in these metrics" and "reason over these sensor traces" — observability and analytics workloads where you want natural-language reasoning grounded in the actual signal. If you have read the [ChatTS deep dive](/blog/machine-learning/signal-processing/chatts-aligning-time-series-llms), this is the family it belongs to.

ChatTS treats a multivariate series as a modality: a context-aware encoder turns the signal into tokens the LLM reasons over, the way a VLM does for pixels:

```python
import torch, torch.nn as nn

class ChatTSStack(nn.Module):
    def __init__(self, ts_encoder, llm):
        super().__init__()
        self.ts_encoder = ts_encoder    # context-aware time-series encoder
        self.llm = llm

    def forward(self, series, question_ids):
        # series: [B, n_vars, T]  multivariate signal
        ts_tokens = self.ts_encoder(series)              # [B, n_vars*k, D]
        q = self.llm.embed(question_ids)
        toks = torch.cat([ts_tokens, q], dim=1)
        return self.llm(inputs_embeds=toks)   # "why did latency spike at 14:00?"
```

### Timer-S1: a billion-scale MoE foundation model for time series {#timer}

**Senior rule of thumb: a forecasting model trained on your data is a tool; a foundation model trained on a trillion points is a default you fine-tune from.** Timer-S1 (arxiv [2603.04791]) is the other bet: not alignment with an LLM, but a true **billion-scale mixture-of-experts time-series foundation model** — **8.3B total parameters with 0.75B active** per the brief. Its named mechanisms are **Serial-Token Prediction** (the time-series analog of next-token prediction, predicting the series serially) and a stack of **TimeMoE plus TimeSTP blocks**. It is trained on **TimeBench**, a corpus of roughly **1 trillion time points**, and the brief reports **state-of-the-art results on GIFT-Eval**, the standard general time-series forecasting benchmark.

The MoE design is the interesting engineering choice: 8.3B parameters of capacity but only 0.75B active per forward pass means you get foundation-model breadth at a fraction of the dense-model inference cost — the same economics that made MoE attractive for LLMs, applied to forecasting. Timer-S1 is what you reach for when you want a strong zero-shot or few-shot forecaster across many series without training one model per series. The [Timer-S1 deep dive](/blog/machine-learning/signal-processing/timer-s1-time-series-foundation-model) covers the architecture in detail.

### Time-series family — which one when

| Model | Strategy | Key idea | Reach for it when |
|---|---|---|---|
| ChatTS | Align TS with an LLM | Attribute-based synthetic data + Evol-Instruct; context-aware encoder | Natural-language reasoning over metrics/sensors |
| Timer-S1 | TS foundation model | 8.3B/0.75B-active MoE, Serial-Token Prediction, TimeBench (1T points) | Zero/few-shot forecasting across many series |

These two are complements, not competitors. ChatTS answers "what does this series *mean* and why"; Timer-S1 answers "what will this series *do* next." A mature observability stack could use both — Timer-S1 to forecast and detect anomalies, ChatTS to explain them.

It is worth sitting with why time series got *two* fundamentally different treatments while image and video got one architectural family each. The reason is that "time-series intelligence" is really two unrelated jobs wearing one name. Forecasting is a regression problem with a well-defined loss (how close is the predicted trajectory to the actual one), and it benefits enormously from scale and from a foundation model that has seen a trillion points of diverse temporal behavior — hence Timer-S1's MoE-foundation-model approach, which mirrors how LLMs solved next-token prediction. Reasoning about a series ("is this seasonal, is this a level shift, why did it spike") is a *language* problem grounded in a signal, and it benefits from an LLM's reasoning machinery plus an encoder that lets the LLM actually read the numbers — hence ChatTS's align-with-an-LLM approach. The architectures diverge because the losses diverge: one optimizes a numeric forecasting error, the other optimizes language-modeling cross-entropy over answers. Trying to force both into one model would reintroduce exactly the multi-task interference the unification thread fights elsewhere, which is presumably why these two stayed separate. The lesson generalizes: unification is the right move when the sub-tasks share a representation, and the wrong move when they share only a name.

The two also illustrate the synthetic-data thread from opposite ends. Timer-S1 needs *volume* — a trillion real-and-synthetic time points so the foundation model sees enough temporal regimes to generalize zero-shot. ChatTS needs *alignment* — series paired with descriptions that are correct by construction, which is why the attribute-based generator produces the signal and its description together rather than describing pre-existing series (where the description could be wrong). Same thread, two requirements: one wants scale, the other wants label correctness, and the synthesis pipeline serves both.

## 5. LLM agents, reasoning, and evaluation {#agents}

> The agent half of this family builds agents; the evaluation half builds the harnesses that keep agent progress honest. The two halves are inseparable — you cannot RL-train an agent without a reward signal, and a benchmark *is* a reward signal you can trust.

This is the family closest to the rest of the agent literature on this blog, and it cleanly demonstrates threads 1 (reward/RL) and 2 (synthetic data) together.

![Graph of the agent-and-evaluation loop: a scholar query through Crawler and Selector to ranked papers, plus an external tool-use agent, both scored by the ToolHop/Web-Bench harness feeding a reward model and PPO update](/imgs/blogs/bytedance-research-model-atlas-7.webp)

The graph above closes the loop. PaSa's Crawler and Selector produce ranked results; an arbitrary tool-use agent produces its own trajectories; both are fed into the benchmark harness (ToolHop, Web-Bench), which scores them; that score becomes a reward; the reward drives a PPO update that yields the next policy. The benchmarks are not a leaderboard afterthought — they are the scoring function that makes RL training of these agents possible.

### PaSa: a two-agent academic paper search agent {#pasa}

**Senior rule of thumb: comprehensive search is not retrieval, it is a search *process* — and a process is something you can RL-train.** PaSa ("PaSa: An LLM Agent for Comprehensive Academic Paper Search," ACL 2025, arxiv [2501.10120](https://arxiv.org/abs/2501.10120)) is a two-agent system. A **Crawler** autonomously invokes search tools, reads papers, and expands the citation graph by following references; a **Selector** judges relevance and filters the candidates down. The agent "autonomously makes a series of decisions, including invoking search tools, reading papers, and selecting relevant references" — exactly the kind of multi-step trajectory that a single retrieval call cannot produce.

PaSa is trained with **reinforcement learning** (threads 1 and 2 together) on a **synthetic dataset, AutoScholarQuery — 35,000 fine-grained academic queries** drawn from top-tier AI conference publications — and evaluated on **RealScholarQuery**, a benchmark of real-world academic queries. The numbers are strong: **PaSa-7B surpasses Google with GPT-4o by 37.78% in recall@20 and 39.90% in recall@50**, and **exceeds PaSa-GPT-4o by 30.36% in recall and 4.25% in precision**. That recall gap is the headline — for "find me *every* paper relevant to this niche question," a trained search *agent* beats a search engine plus a strong LLM, because comprehensiveness is a process property. The [PaSa deep dive](/blog/machine-learning/ai-agent/pasa-llm-paper-search-agent) walks through the training. For the general pattern of evaluating multi-step agents, see [evaluating agent trajectories beyond the final answer](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer).

PaSa is a process, not a query: the Crawler expands, the Selector filters, and the whole trajectory is what RL optimizes against a recall-shaped reward:

```python
def pasa_search(query, crawler, selector, max_depth=3):
    frontier, collected = crawler.seed(query), []
    for _ in range(max_depth):
        papers = crawler.expand(frontier)        # follow refs, invoke search tools
        kept = [p for p in papers if selector.relevant(p, query)]
        collected += kept
        frontier = crawler.next_frontier(kept)   # expand from what survived
    return rank(collected)   # reward = recall@k vs ground-truth answer set
```

### ToolHop: a query-driven multi-hop tool-use benchmark {#toolhop}

**Senior rule of thumb: real tool use is multi-hop — the output of one tool is the input to the next — and most benchmarks only test single hops.** ToolHop ("ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use," ACL 2025, arxiv [2501.02506](https://arxiv.org/abs/2501.02506)) measures exactly the chained case. Its construction is itself a synthetic-data pipeline (thread 2): a **query-driven data construction approach** of **tool creation → document refinement → code generation**, yielding **995 user queries and 3,912 associated tools**. It evaluates understanding, reasoning, and function-calling across **14 models from five families** (LLaMA3.1, Qwen2.5, Gemini1.5, Claude3.5, GPT).

The headline result is humbling: **GPT-4o achieved the highest accuracy at only 49.04%**, which the authors note underscores substantial room for improvement. Multi-hop tool use — where you must call a tool, read its result, and feed it into the next call correctly — is genuinely hard, and ToolHop is the benchmark that exposes it. If you build agents, this is the harness that tells you whether your tool-chaining actually works. It pairs naturally with the broader [advanced tool use](/blog/machine-learning/ai-agent/advance-tool-use) and [agent evaluation](/blog/machine-learning/ai-agent/eval-agents) material on this blog. The [ToolHop deep dive](/blog/machine-learning/ai-agent/toolhop-multi-hop-tool-use-benchmark) covers the construction in full.

### Web-Bench: a web-development code benchmark {#webbench}

**Senior rule of thumb: a code benchmark with independent tasks measures snippet generation; a benchmark with *sequential* tasks measures whether the agent can build a project.** Web-Bench ("Web-Bench: A LLM Code Benchmark Based on Web Standards and Frameworks," arxiv [2505.07473](https://arxiv.org/abs/2505.07473)) is structured as **50 projects, each consisting of 20 tasks with sequential dependencies** — task 2 builds on task 1's output, and so on. That sequential dependency is the design choice that makes it realistic: it tests whether an agent can maintain and extend a growing codebase, not just emit isolated functions. It covers web standards and web frameworks. The SOTA number is sobering: **Claude 3.7 Sonnet achieved only 25.1% Pass@1** on the Web-Agent harness. End-to-end web project construction is far from solved, and Web-Bench quantifies exactly how far. The [Web-Bench deep dive](/blog/machine-learning/ai-agent/web-bench-llm-web-development-benchmark) goes deeper; for hands-on agent construction, see [building effective agents](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide).

### Datasets: veAgentBench, KIE-HVQA, DeepHall-data

The family also ships evaluation datasets. **veAgentBench** is an agent benchmark dataset. **KIE-HVQA** targets key-information-extraction and hybrid visual question answering. **DeepHall-data** targets hallucination evaluation. These are the raw materials for the same honesty loop: you cannot trust agent progress without datasets that measure the failure modes you care about, and these fill specific gaps (extraction, hallucination) that general benchmarks miss.

There is a deeper point in the fact that an industrial research org publishes *benchmarks* as first-class releases alongside its models, and it ties directly back to thread 1. If your training strategy is reward learning — and across this org it overwhelmingly is — then your benchmark *is* your reward signal made trustworthy. A reward model is only as good as the preferences it was trained on, and a learned reward can be gamed: an agent will happily exploit a sloppy scoring function, producing outputs that score well and are actually bad (reward hacking). The defense is a hard, well-constructed *external* benchmark that the agent was not trained to satisfy, used to audit whether the reward model's preferences actually correlate with real quality. ToolHop's deliberately-difficult multi-hop construction (top score 49.04%) and Web-Bench's sequential-dependency design (SOTA 25.1%) are valuable precisely *because* they are hard to game — they leave so much headroom that a model cannot fake competence. So the benchmarks are not a separate "evaluation" workstream bolted onto the agent work; they are the integrity check that keeps the reward-learning loop honest, which is why they ship together. For teams, the transferable practice is: every time you adopt a reward-trained model, find or build the adversarial benchmark that would catch it cheating, and gate on that — not on the metric the model was optimized for.

### Agent family — which one when

| Release | Type | What it measures / does | Reach for it when |
|---|---|---|---|
| PaSa | Agent | Comprehensive academic search via Crawler+Selector, RL-trained | You need *exhaustive* literature search, not top-10 |
| ToolHop | Benchmark | Multi-hop tool use (995 queries / 3,912 tools) | Validating tool-chaining in your agent |
| Web-Bench | Benchmark | Sequential web-dev project construction (50×20) | Validating end-to-end coding agents |
| veAgentBench | Dataset | General agent evaluation | Broad agent capability checks |
| KIE-HVQA | Dataset | Key-info extraction + hybrid VQA | Document/extraction agents |
| DeepHall-data | Dataset | Hallucination evaluation | Measuring factuality of generations |

## The unification thesis, stated plainly {#unification}

Step back from the families and the single loudest argument across all five is unification: replace a zoo of task-specific models with one model conditioned on what you give it.

![Before-after comparison: siloed task-specific models each with their own data and reward systems versus one unified model with in-context conditioning, disentanglement, and one reward model](/imgs/blogs/bytedance-research-model-atlas-5.webp)

The before-after above is the thesis in one picture. On the left, the old world: a subject model, a style model, an inpaint model, each with its own data pipeline and its own reward system — N models means N pipelines and N maintenance burdens, and they drift apart over time. On the right, the unified world: in-context conditioning (UNO), content-style disentanglement (USO), one VLM reward model (OneReward), yielding one checkpoint that does many jobs.

The engineering case for unification is not aesthetic. It is operational. Every task-specific model you ship is a separate data pipeline to maintain, a separate reward model to keep calibrated, a separate checkpoint to evaluate and roll out, and a separate place for behavior to regress. Collapsing four editing tasks into one OneReward-trained model, or single- and multi-subject into one UNO, cuts that maintenance surface by the collapse factor. The cost is that unified models are harder to train — USO had to *invent* content-style disentanglement to stop the two objectives from fighting, and MammothModa2 has to balance an AR head and a diffusion head — but once trained, they are cheaper to operate and more consistent across tasks. That trade — harder to train, cheaper to run, more consistent — is the bet, and the volume of releases following it suggests it is paying off.

The same logic explains why reward learning is everywhere. A reward model is the unification glue: one OneReward VLM can score four editing tasks, so you do not need four supervised datasets. One PaSa reward signal optimizes the whole multi-step trajectory, so you do not need step-level labels. Reward learning and unification are two sides of one coin — the reward model is what lets one model serve many tasks.

## Case studies: choosing the right model under real constraints {#cases}

The atlas is only useful if it changes a decision. Here are concrete situations, each ending in a specific model and the constraint that decided it.

### 1. The product-into-scene catalog generator

A retailer wants to drop each SKU into a dozen lifestyle scenes — "this mug on a breakfast table," "this mug on a desk." The naive instinct is a per-SKU LoRA, which means training and storing thousands of LoRAs. The constraint that decides: *one subject, open domain, must be fast and must not leak the reference background.* That is RealCustom's exact problem statement — narrow the real word, mask subject influence — so the answer is **RealCustom/++**, or **HyperLoRA** if you want the LoRA route without per-SKU training (the hypernetwork emits the LoRA from one reference). The wrong answer here is UNO; you do not have multiple subjects, and you would be paying for capability you do not use.

### 2. The model-holding-the-product shot

Same retailer, harder shot: a *model* holding the *product*, two distinct identities that must both stay correct. Now the constraint flips: *two or more distinct subjects in one frame, kept distinct.* This is precisely the single→multi generalization UNO was built for, with universal RoPE preventing the multi-subject confusion that would otherwise merge the model's hand into the mug. The answer is **UNO**. RealCustom would struggle because it is architected around scoping *one* subject's influence.

### 3. The brand-style campaign

The brand has a signature flat-illustration look and wants every product rendered in it. The constraint: *a specific style AND a specific subject, both faithful, simultaneously.* This is the conflict USO was designed to resolve via content-style disentanglement plus style reward learning. The answer is **USO**, and the tell that you need it (rather than UNO) is that you have a *style reference* in addition to a *subject reference*. If you tried to force this through a subject-only model, the style would smear the product's identity — the exact failure USO's disentanglement prevents.

### 4. The group portrait

A team page wants a stylized group portrait of eight real people, each recognizable. The constraint: *many identities, all consistent, no bleed between similar faces.* Greedy per-face matching collapses lookalikes; you need the global assignment that UMO's multi-to-multi matching reward provides. The answer is **UMO**. This is the case where the combinatorial structure of the problem — assignment, not nearest-neighbor — picks the model.

### 5. The production inpainting service

A photo-editing app needs fill, extend, object removal, and text rendering, all mask-guided, all in production, all maintained by a small team. The constraint: *four editing tasks, minimal maintenance surface, no per-task fine-tuning.* OneReward's one-VLM-reward, multi-task-RL recipe was built to avoid exactly the four-pipeline burden, and it reportedly beats Photoshop and FLUX Fill Pro across dimensions. The answer is **OneReward** (the Seedream 3.0 Fill application). The constraint that decides is *operational*, not quality — you could train four separate models, but you would then maintain four pipelines.

### 6. The talking-head presenter

A product needs an AI presenter that speaks a script in a specific voice with a specific face, lip-sync correct. The constraint: *audio is a first-class control signal; lip-sync must be tight.* Audio-driven sync is HuMo's focus-by-predicting strategy, and audio is one of its three native conditioning modalities. The answer is **HuMo**. Phantom is wrong here — it is subject-consistent but not audio-conditioned, so the mouth would not match the words.

### 7. The subject-consistent ad spot

A short video ad needs a specific character to appear consistently across shots, with motion driven by the script. The constraint: *identity from a reference image, motion and scene from text, no content leakage.* That is Phantom's cross-modal alignment trained on text-image-video triplets. The answer is **Phantom**. If the requirement added "and the camera dollies from left to right," you would layer in **ATI** for the trajectory control.

### 8. The e-commerce visual QA system

A marketplace needs to answer questions about product images and short videos at production scale and latency. The constraint: *understanding, not generation; e-commerce domain; sub-10B for latency.* Valley2 is explicitly optimized for e-commerce and short-video and ranks second among sub-10B models on OpenCompass at 67.4. The answer is **Valley2** (or a Valley3 size that fits your budget). If the questions require multi-step reasoning, switch to a **Valley-Think** variant; if you also need to *generate* images from the same model, you have left Valley's territory and want **MammothModa2**.

### 9. The "find the clip" video editor

A video platform wants users to type "show me where they talk about pricing" and jump to the moment. The constraint: *temporal retrieval over long video, possibly with audio queries.* That is Vidi's emphasized task, benchmarked on VUE-TR, where it reportedly beats GPT-4o and Gemini. The answer is **Vidi/Vidi2**. A general VLM would describe the video; Vidi *localizes* the moment, which is the actual product requirement.

### 10. The observability copilot

An SRE team wants to ask "why did p99 latency spike at 14:00?" over their metrics. The constraint: *natural-language reasoning grounded in a multivariate time series.* ChatTS treats the series as a modality and reports 46.0% / 25.8% gains on alignment / reasoning over vision-based MLLMs including GPT-4o. The answer is **ChatTS**. If the same team also wants "what will p99 do over the next hour," that is forecasting, and the answer becomes **Timer-S1** — the two models split the observability problem along the explain/predict line.

### 11. The exhaustive literature review

A researcher needs *every* paper relevant to a narrow question, not the top ten. The constraint: *recall, not precision; comprehensiveness is the goal.* PaSa's Crawler+Selector process, RL-trained for recall, beats Google+GPT-4o by ~38–40% on recall@20/50. The answer is **PaSa**. A vector search or a single LLM call optimizes the wrong metric — they give you relevant-looking results, not complete ones. (See also [vector databases](/blog/machine-learning/ai-agent/vector-database) for where naive retrieval tops out.)

### 12. The agent CI gate

A team shipping a tool-using agent wants a regression gate that fails when tool-chaining breaks. The constraint: *measure multi-hop tool use and end-to-end project construction, not single calls.* ToolHop (multi-hop, where even GPT-4o sits at 49.04%) and Web-Bench (sequential web projects, SOTA 25.1% Pass@1) are the harnesses built for this. The answer is **ToolHop + Web-Bench** as CI gates. A pass@1 on isolated functions would give false confidence; these benchmarks fail on the chaining and continuity that production agents actually need.

### 13. The virtual fitting room

An apparel retailer wants shoppers to see a garment on a model — or on themselves — without a photoshoot per SKU. The constraint: *the garment's texture, print, and cut must survive being draped onto a new body and pose.* This is not generic subject customization; it is garment-aware warping, which is exactly DreamFit's specialty. The answer is **DreamFit**. The reason a general subject model (UNO, USO) is the wrong pick is that they preserve *identity* of an object as-is, whereas try-on must deform the garment to a new geometry while keeping its surface detail — a different invariance entirely. Matching the *invariance you need* to the model's design is the whole decision.

### 14. The per-user avatar service at scale

A social app wants every user to get a stylized portrait avatar, generated on signup, with no per-user training step. The constraint: *zero-shot personalization from a single reference, at signup latency, across millions of users.* Training a LoRA per user is operationally impossible at that scale; HyperLoRA's hypernetwork *emits* the LoRA weights from one reference image in a single forward pass. The answer is **HyperLoRA**. This is the case where the deciding constraint is neither quality nor task fit but *amortization* — you need the per-user customization cost to be a forward pass, not a training run, and HyperLoRA is architected precisely around that economics.

### 15. The document-extraction agent

A back-office team wants an agent that reads invoices and forms and answers structured questions about them, with a measurable accuracy gate. The constraint: *key-information extraction plus hybrid visual QA, with a benchmark that targets exactly those failure modes.* A general VLM benchmark would not stress the extraction-and-grounding behavior that breaks on real documents; **KIE-HVQA** is the dataset built for it, and a Valley-class understanding model (or a Valley-Think variant for multi-step reasoning over a form) is the model to point at it. The answer is **Valley + KIE-HVQA as the gate**. The lesson repeated from case 12: the model and the benchmark are chosen together, because the benchmark is what tells you the model actually does the narrow thing you need.

## How to choose: the decision guide {#choose}

![Grid decision guide mapping use cases through their deciding constraint to the recommended ByteDance model](/imgs/blogs/bytedance-research-model-atlas-8.webp)

The grid above compresses the case studies into a lookup. Read left to right: a use case, the one constraint that decides, and the model. The point of the middle column is that you rarely pick a model by quality — you pick it by the *constraint* that distinguishes your task from the adjacent one. One subject vs. two picks RealCustom vs. UNO. Subject-only vs. style+subject picks UNO vs. USO. Understanding vs. understanding+generation picks Valley vs. MammothModa. Explain vs. predict picks ChatTS vs. Timer-S1.

The reason the *constraint* matters more than the *quality* is that adjacent models in this atlas are often within noise of each other on generic benchmarks but wildly different on the specific invariance your task needs. UNO and USO would both produce a plausible image of your product; only USO holds the *style* fixed while doing it. Phantom and HuMo would both produce a plausible video of a person; only HuMo keeps the lips in sync with the audio. A leaderboard average flattens exactly the distinction that decides your project, because the distinguishing capability is a narrow slice the average barely weights. So the decision procedure is: name the invariance your task requires (background obedience, multi-subject separation, style fidelity, identity assignment, lip-sync, trajectory adherence, temporal localization, forecasting accuracy, grounded reasoning), then read the column that owns that invariance. The model falls out of the constraint; do not start from the model.

### Reach for a ByteDance Research model when

- **You need a specific *control axis* for generation.** Subject (RealCustom/UNO), style+subject (USO), identity (UMO), mask-editing (OneReward), audio (HuMo), trajectory (ATI) — this org has unusually fine-grained control models, and matching the axis to your need is the whole game.
- **Your data does not exist and you cannot label it at scale.** These models were trained against exactly that problem, and several ship their synthetic datasets (UNO-1M, AutoScholarQuery, HuMoSet), so the data strategy is reusable even if the model is not.
- **You want one model to cover several related tasks.** The unification thread means UNO, USO, OneReward, MammothModa, and Lance each replace several narrower models — lower maintenance surface.
- **You are evaluating agents and need honest, hard benchmarks.** ToolHop and Web-Bench are deliberately difficult (49% and 25% SOTA), which is what makes them useful as gates.
- **You work in e-commerce, short-video, or document/observability domains.** Valley, Vidi, ChatTS are tuned for exactly these, not for leaderboard generality.

### Skip them when

- **You need a frontier *general* chat or coding LLM.** That is not what this org optimizes; reach for a general foundation model and bring these in for the specialized sub-tasks.
- **Your task is single-hop or trivially retrievable.** PaSa is overkill for "find one relevant paper"; a vector search or single LLM call is cheaper. ToolHop/Web-Bench are overkill if your agent makes one tool call.
- **You need turnkey, fully-supported, SLA-backed inference.** These are research releases; productionizing them is on you, and the unified/RL-trained ones especially need careful eval before rollout.
- **The control axis you need is not in the catalog.** If your job is, say, fine-grained 3D-aware editing, none of these is a clean fit — do not force a subject-customization model into a problem shaped differently.

> The atlas in one sentence: pick the family by your modality and task, pick the model by the one constraint that separates your task from the one next to it, and assume the model was trained on synthetic data against a learned reward — because, across all five families, it almost certainly was.

## What would change my mind

A few honest caveats. First, this atlas is organized by *intent*, and intent boundaries are fuzzy — MammothModa and Lance straddle understanding and generation, and reasonable people could file them under either the understanding family or a sixth "unified any-to-any" family. I kept them under understanding because their *entry point* is reading multimodal input, but if the any-to-any line keeps growing, it earns its own branch. Second, several of the most interesting numbers (HuMoSet at 670K, Timer-S1 at 8.3B/0.75B-active on a 1T-point TimeBench, the Lance and Vidi2 details) come from the brief and project pages rather than abstracts I could verify line by line; treat those specific figures as directional until you confirm them against the papers. Third, the "reward learning is everywhere" thread is the strongest claim here, and it is the one to pressure-test: if you find a family member trained purely with supervised fine-tuning and no reward signal, that is a counterexample worth noting — though OneReward, USO's SRL, UMO's matching reward, and PaSa's PPO make the pattern hard to dismiss.

What would *strengthen* the atlas: the org clearly shares infrastructure across families (the interleaved timeline is the evidence), so the most valuable thing they could publish is the shared *substrate* — the data-synthesis tooling and the reward-modeling framework that recur across image, video, and agents. That substrate, not any single checkpoint, is the real release.

## References

**Image generation and customization**
- RealCustom — "Narrowing Real Text Word for Real-Time Open-Domain Text-to-Image Customization" — [arxiv 2403.00483](https://arxiv.org/abs/2403.00483)
- RealCustom++ — [arxiv 2408.09744](https://arxiv.org/abs/2408.09744)
- UNO — "Less-to-More Generalization: Unlocking More Controllability by In-Context Generation" (ICCV 2025) — [arxiv 2504.02160](https://arxiv.org/abs/2504.02160)
- USO — "Unified Style and Subject-Driven Generation via Disentangled and Reward Learning" — [arxiv 2508.18966](https://arxiv.org/abs/2508.18966)
- OneReward — "Unified Mask-Guided Image Generation via Multi-Task Human Preference Learning" — [arxiv 2508.21066](https://arxiv.org/abs/2508.21066)
- UMO — multi-identity consistency (CVPR 2026)

**Video generation**
- Phantom — "Subject-consistent video generation via cross-modal alignment" — [arxiv 2502.11079](https://arxiv.org/abs/2502.11079)
- HuMo — "Human-Centric Video Generation via Collaborative Multi-Modal Conditioning" — [arxiv 2509.08519](https://arxiv.org/abs/2509.08519)
- ATI — any-trajectory-instruction controllable video

**Multimodal understanding**
- Valley2 — "Exploring Multimodal Models with Scalable Vision-Language Design" — [arxiv 2501.05901](https://arxiv.org/abs/2501.05901)
- MammothModa2 — unified AR-diffusion understanding and generation — [arxiv 2511.18262](https://arxiv.org/abs/2511.18262)
- Lance — "Unified Multimodal Modeling by Multi-Task Synergy" — [arxiv 2605.18678](https://arxiv.org/abs/2605.18678)
- Vidi — "Large Multimodal Models for Video Understanding and Editing" — [arxiv 2504.15681](https://arxiv.org/abs/2504.15681); Vidi2 — [arxiv 2511.19529](https://arxiv.org/abs/2511.19529)

**Time series**
- ChatTS — "Aligning Time Series with LLMs via Synthetic Data" (VLDB 2025) — [arxiv 2412.03104](https://arxiv.org/abs/2412.03104)
- Timer-S1 — billion-scale MoE time-series foundation model — [arxiv 2603.04791](https://arxiv.org/abs/2603.04791)

**Agents, reasoning, evaluation**
- PaSa — "An LLM Agent for Comprehensive Academic Paper Search" (ACL 2025) — [arxiv 2501.10120](https://arxiv.org/abs/2501.10120)
- ToolHop — "A Query-Driven Benchmark for Evaluating LLMs in Multi-Hop Tool Use" (ACL 2025) — [arxiv 2501.02506](https://arxiv.org/abs/2501.02506)
- Web-Bench — "A LLM Code Benchmark Based on Web Standards and Frameworks" — [arxiv 2505.07473](https://arxiv.org/abs/2505.07473)

**Org and weights**
- HuggingFace: [`bytedance-research`](https://huggingface.co/bytedance-research) and [`ByteDance`](https://huggingface.co/ByteDance)
- GitHub: [bytedance](https://github.com/bytedance) (per-model repos: UNO, USO, OneReward, Phantom, HuMo, Valley, ChatTS, PaSa, ToolHop, Web-Bench)

**Sibling deep dives on this blog**
- [PaSa: an LLM paper-search agent](/blog/machine-learning/ai-agent/pasa-llm-paper-search-agent)
- [ToolHop: multi-hop tool-use benchmark](/blog/machine-learning/ai-agent/toolhop-multi-hop-tool-use-benchmark)
- [Web-Bench: LLM web-development benchmark](/blog/machine-learning/ai-agent/web-bench-llm-web-development-benchmark)
- [ChatTS: aligning time series with LLMs](/blog/machine-learning/signal-processing/chatts-aligning-time-series-llms)
- [Timer-S1: a time-series foundation model](/blog/machine-learning/signal-processing/timer-s1-time-series-foundation-model)
- [Evaluating agent trajectories beyond the final answer](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer)
- [Advanced tool use](/blog/machine-learning/ai-agent/advance-tool-use) · [Agent evaluation](/blog/machine-learning/ai-agent/eval-agents) · [Building effective agents](/blog/machine-learning/ai-agent/building-effective-agents-hands-on-guide)
