---
title: "AIMO Progress Prize 2: every published solution, and the techniques that decided it"
date: "2026-09-10"
publishDate: "2026-09-10"
description: "Sixteen teams published how they solved national-olympiad math on four L4 GPUs in five hours. Almost all of them ran the same 14B model, so the competition was decided by data construction and inference economics instead."
tags: ["kaggle", "aimo", "reasoning", "llm-inference", "quantization", "speculative-decoding", "self-consistency", "tool-integrated-reasoning", "grpo", "dataset-construction", "test-time-compute", "competition"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 93
---

In April 2025 the AI Mathematical Olympiad Progress Prize 2 closed with a result that should bother anyone who thinks model quality is the whole game. Sixteen teams published write-ups. Twelve of them ran the exact same model: a 4-bit quantized `DeepSeek-R1-Distill-Qwen-14B`, in several cases the identical checkpoint uploaded by the same community member. Their private scores ranged from 24 out of 50 to 29 out of 50. The team that won ran a model it trained itself and scored 34.

So the spread across the leaderboard was not mostly about the model. It was about how many times you can afford to ask the model, how early you dare to stop it, how you spend a five-hour clock across fifty problems of wildly different difficulty, and, for the teams at the very top, what data you trained on before any of that started.

![The AIMO-2 submission loop: 110 novel problems split into reference, public and private sets, served one at a time through an evaluation API to a notebook running on four L4 GPUs under a five hour wall clock](/imgs/blogs/aimo-progress-prize-2-published-solutions-1.webp)

The diagram above is the mental model, and it is worth staring at before any technique makes sense. You do not get a dataset and a week on a cluster. You get an evaluation API that hands you one problem at a time, in random order, on four L4 GPUs you cannot see until the run starts, with no internet, a five-hour ceiling, and one submission per day to learn anything at all. Every technique in this article is a response to one of those constraints.

This post walks through all of it: how the top teams built their training data, which is the part most write-ups skim and the part that actually separated first place from everyone else; how they taught a reasoning model to call a Python interpreter; and then the ten-layer stack of inference tricks that turned a commodity 14B model into a gold medal.

## 1. The competition, restated as an engineering problem

AIMO is a \$10 million fund aimed at getting open models to perform like top human International Mathematical Olympiad contestants. Progress Prize 2 was the second checkpoint on that road, run on Kaggle by AIMO and XTX Markets, with a total fund of \$2,117,152 and a final deadline of April 1, 2025.

The problem set is the first thing that matters. There were 110 problems spanning algebra, combinatorics, geometry and number theory, written fresh by an international team of problem setters specifically so that they could not have leaked into anyone's pretraining corpus. The difficulty was raised from the first competition to roughly national olympiad level, and the problems were deliberately designed to be "AI hard", meaning they were tested against current open models and kept only if they were difficult for them.

Those 110 problems split three ways. Ten were released publicly as a reference set, which became the only labelled data most teams had. Fifty scored the public leaderboard. Fifty scored the private leaderboard, and that is the number that decided the money.

Every answer is an integer between 0 and 999. You take whatever the solution is and reduce it modulo 1000. If the answer is 65521 you submit 521. If it is -900 you submit 100. This detail sounds cosmetic and is not: it means grading is exact match, there is no partial credit, and it means every model needs to be told about the modulo or it will confidently return 65521 and score zero. Several teams lost points to exactly that before they hardened their prompts.

The compute is where it becomes an engineering problem. Four NVIDIA L4 GPUs, 24 GB each. Five hours of wall clock for the entire run. No internet access, so every model weight and Python package has to be baked into the notebook or attached as a Kaggle dataset. The evaluation API serves problems one at a time and requires each prediction within 30 minutes, with the first call to the prediction server within 15 minutes of the notebook starting.

Then the constraint that shaped team behaviour more than any other: **one submission per day**. You get a single 50-problem noisy measurement every 24 hours. Optimistix, who joined nine days before the deadline expecting to run 45 experiments, discovered this rule after joining and ended up with seven usable attempts total. That team still took a bronze, which tells you something about how much of the leaderboard was luck.

The prize structure is worth understanding because it explains why nobody claimed the big money. The Overall Progress Prize required at least 47 out of 50 on both public and private test sets. Nobody came close. The winner scored 34. So the top-five prizes were paid out, the first-place team took \$262,144 and directed it to the NVIDIA Foundation, and the remaining roughly \$1.6 million rolled over to the next competition.

## 2. What one point was worth

Before any technique, look at the scoreboard shape, because it explains every design decision that follows.

Here is the private leaderboard collapsed into score bands, compiled by a competitor who finished 32nd:

| Private score | Best rank achieving it |
| --- | --- |
| 34 | 1 |
| 31 | 2 |
| 30 | 3 |
| 29 | 7 |
| 28 | 13 |
| 27 | 26 |
| 26 | 56 |
| 25 | 91 |
| 24 | 158 |
| 23 | 276 |
| 22 | 460 |
| 21 | 802 |
| 20 | 1240 |

![Bar chart of the best rank achieved at each private leaderboard score from 34 down to 20 on a log scale, where one point moves rank 460 to 276](/imgs/blogs/aimo-progress-prize-2-published-solutions-2.webp)

One extra problem, out of fifty, is the difference between rank 276 and rank 460. Two extra problems near the medal boundary move you from 158th to 56th. At the top the cliff is even steeper: the gap between first and second was three problems, and the gap between second and third was one.

![Final private leaderboard score against rank on a logarithmic axis, with the medal cutoffs marked](/imgs/blogs/aimo-progress-prize-2-published-solutions-fig16.webp)

*Chart posted in the comments of the [8th place write-up](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/571356). Score against rank on a log scale, with dashed lines at the gold, silver and bronze cutoffs. The gold line sits just past rank 10 and passes through the 28 point, so the entire medal range spans about seven points of a 50 problem test.*

Now put that next to the noise. Each leaderboard is fifty binary outcomes from a stochastic sampler. Daniel Phalen measured the standard deviation of his own pipeline at roughly plus or minus 2 to 3 problems on a 50-problem set with majority voting. MPWARE reported that his last eight submissions, with the same code, scored between 26 and 28. That means the measurement noise on a single submission is comparable to the entire gap between a gold medal and no medal.

The consequence shows up brutally in the public-to-private shakeup:

| Team | Public | Private | Final rank |
| --- | --- | --- | --- |
| imagination-research | 34 | 31 | 2 |
| Aliev | 25 | 30 | 3 |
| sravn | 25 | 29 | 4 |
| usernam | 28 | 29 | 5 |
| tascj | 23 | 29 | 7 |
| MPWARE | 28 | 28 | 8 |
| Fast-Math-R1 | 29 | 28 | 9 |
| farsail | 29 | 28 | 11 |
| ippeiogawa | 28 | 27 | 17 |
| Optimistix | 20 | 24 | 24 |

![Dumbbell chart pairing each published solution's public and private score, showing ranks 3, 4 and 7 gaining five or six points while the public leader dropped three](/imgs/blogs/aimo-progress-prize-2-published-solutions-3.webp)

The team that led the public leaderboard with 34 finished second. Third and fourth place both gained five points going from public to private. Seventh place gained six, from 23 to 29, on a submission its author describes as a "just give it a try" attempt built in a week, and titled his write-up "7th place solution (pure luck)".

The engineering lesson here is not "the competition was random". It is that **when your evaluation signal is noisier than the effect you are trying to measure, tuning against that signal is actively harmful**. This is why fourth place deliberately refused to do time management, and why fifth place built an offline simulator. We will come back to both.

## 3. The budget equation every team was solving

Strip away the specifics and every solution is optimizing the same thing. For each question you choose how many samples to generate and how long to let each one run, subject to a fixed throughput and a fixed clock:

$$
\text{samples} \times \text{tokens per sample} \le \text{throughput (tokens/s)} \times \text{time budget (s)}
$$

Accuracy rises with samples, because majority voting over more independent attempts is more reliable. Accuracy also rises with tokens per sample, because a reasoning model that gets truncated before it reaches `\boxed{}` contributes nothing at all. Those two pull against each other on a fixed budget, which is why the entire competition became an exercise in raising the right-hand side or spending the left-hand side more cleverly.

Fifth place made this concrete. He profiled which pairs of batch size and maximum sequence length let a full 50-question run finish in 4.5 to 4.7 hours, at roughly 410 seconds per question:

| Batch size | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Max seqlen | 20000 | 17000 | 15500 | 14500 | 13800 | 13300 | 12800 | 12200 | 11600 | 11100 | 10600 | 10200 |

![Simulated majority vote accuracy against paired batch size and sequence length, peaking near batch size 10](/imgs/blogs/aimo-progress-prize-2-published-solutions-fig11.webp)

*Figure from the [5th place write-up](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/574262). Majority vote accuracy on AIME 2025, averaged over 10,000 simulated runs, against the paired batch size and sequence length settings in the table above. Both the bf16 and the AWQ model peak around batch size 10 with a 13,300 token cap, and both fall away at either extreme.*

Read that row as an iso-cost curve. You can have five samples of 20,000 tokens each, or sixteen samples of 10,200 tokens each, and both finish on time. His simulations put the optimum at (10, 13300) or (9, 13800), and he shipped (9, 13500). The interior optimum is the point. Neither extreme wins: too few samples and voting has nothing to work with, too short a budget and half your samples never produce an answer.

![Downward sloping curve of batch size five to sixteen against max sequence length twenty thousand down to ten thousand two hundred, every pair finishing fifty questions in about four and a half hours](/imgs/blogs/aimo-progress-prize-2-published-solutions-4.webp)

Every technique in the rest of this article is a lever on one of the four terms:

| Lever | Which term it moves | Who used it |
| --- | --- | --- |
| Smaller or quantized model | throughput up | everyone |
| Faster engine | throughput up | 1st, 2nd, 5th, 7th |
| Speculative decoding | throughput up | 1st |
| Training for shorter outputs | tokens per sample down | 1st, 2nd, 9th |
| Early stopping | tokens per sample down | 1st, 2nd, 3rd, 11th, 17th |
| Tool use | tokens per sample down | 1st, 11th, 17th, 20th |
| Dynamic time allocation | time budget reallocated | 1st, 2nd, 3rd, 7th, 9th, 20th |
| Better aggregation | accuracy per sample up | 1st, 20th |

## 4. How the winners built their training data

This is the part that deserves the most attention, and the part that separated the winner from a field that was otherwise running the same checkpoint. NVIDIA's NemoSkills team did not fine-tune on a public dataset. They built one, and then released it as `nvidia/OpenMathReasoning`, 5.5 million solutions in total.

![The OpenMathReasoning pipeline turning 620K AoPS forum discussions into 540K unique problems and 3.2M verified reasoning solutions](/imgs/blogs/aimo-progress-prize-2-published-solutions-5.webp)

### 4.1 Mining a forum instead of licensing a dataset

The raw material was the Art of Problem Solving community forums, excluding the "Middle School Math" category as too easy. That gave roughly 620,000 raw forum discussions. A forum thread is not a training example. It is a human post that may contain a problem, may contain several problems, may contain a solution, may contain an argument about whether the solution is right, and may contain nothing usable at all.

So the team built an LLM-based extraction and refinement pipeline that did the following:

1. **Extract** the problem statement out of the post, separating it from surrounding chatter.
2. **Classify** each extracted problem along several axes: is it a proof problem or does it have a concrete answer, is it multiple choice, is it a binary yes/no question, is it valid at all.
3. **Transform** proof problems into answer-based problems. This is the clever move. A proof problem cannot be graded by exact match, so most pipelines throw them away. Instead they were rewritten into a form that has a numeric answer, which converts otherwise unusable olympiad material into trainable examples.
4. **Extract answers** from the surrounding forum discussion where a poster stated one.

The output was 540,000 unique problems, broken down as roughly 260,000 converted proof problems, 190,000 with an answer extracted from the discussion, and 190,000 with no answer at all. Hold onto that last bucket, because what they did with it is the most interesting part.

The pipeline also ran decontamination: an LLM-based comparison against the benchmark sets, removing questions that were too similar to problems in AIME or HMMT. In a competition explicitly designed around unseen problems, training on a leaked benchmark would produce local scores that evaporate on the private set.

### 4.2 Generating solutions, and verifying them without an answer key

With 540,000 problems in hand, the team generated solutions using DeepSeek-R1 and QwQ-32B: up to 32 candidate solutions per problem, temperature 0.7, top-p 0.95, 16K token limit. Harder problems, identified by measuring the pass rate of `Qwen2.5-72B-Math-Instruct` on them, received more candidates. That is a sensible allocation: easy problems converge in a few samples, hard ones need many before any of them is correct.

Then comes the filtering, and this is where the 190,000 answerless problems get rescued. Two rules:

- **If the problem has a known answer**, keep only solutions whose final answer matches it. Matching is not string equality, because $\frac{1}{2}$ and ${0.5}$ and `1/2` are the same answer written three ways. `Qwen2.5-32B-Instruct` was used as a judge for mathematical answer equivalence.
- **If the problem has no known answer**, take the most frequent answer across the 32 candidates and treat it as the label. This is majority voting used as a pseudo-labelling mechanism at dataset-construction time, and it is what turns a third of the corpus from junk into training data.

That second rule is worth pausing on, because it is the same idea that every team used at inference time, applied one level up. Self-consistency is normally a test-time trick. Here it is a data-labelling trick. If 20 of 32 independent reasoning traces from a strong model converge on 174, then 174 is probably right, and you now have 20 verified training examples for a problem nobody had an answer key for.

The result: 3.2 million chain-of-thought solutions kept out of 5.2 million generated. For the actual competition submission the team used a 2.2 million subset generated by DeepSeek-R1 alone.

### 4.3 The training run

The submitted model started from `Qwen2.5-14B-Base`, not from an instruction-tuned or already-distilled checkpoint. The RoPE base was changed to 500k to accommodate long reasoning traces. Then supervised fine-tuning on the 2.2M CoT subset for 8 epochs, with AdamW, learning rate 1e-4 decaying to 1e-7, weight decay 0.01, 10% linear warmup, batch size 1024, using sequence packing and context parallelism from NeMo-Aligner to make long-sequence training tractable.

Their published training curve shows why they call the extra compute optional, and it is also the clearest single picture of tool-integrated reasoning paying off:

![Accuracy on Comp-Math-24-25 against training epochs, with tool-integrated reasoning above chain of thought for both pass@1 and majority vote at 64 samples](/imgs/blogs/aimo-progress-prize-2-published-solutions-fig1.webp)

*Figure from the [1st place write-up](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/nemoskills-1st-place-solution-nemoskills) by team NemoSkills. Accuracy on their 256-problem Comp-Math-24-25 benchmark against training epochs. TIR sits above CoT for both metrics, and all four curves flatten after roughly 1.5 epochs, which is the saturation the team is describing.*

That run took **48 hours on 512 H100 GPUs**. The team notes, with admirable honesty, that they could have reached most of the model's strength with 20% of that compute, and scaled up mainly to see where learning saturates. Weight averaging across checkpoints from different training stages produced the final weights.

The competition-day recipe was deliberately different from the one in the accompanying paper. The publicly released `OpenMath-Nemotron` models score better on benchmarks, but generate more tokens, and the team judged that they would time out inside the Kaggle constraints. They shipped the weaker, faster model on purpose. That trade shows up again and again in this competition.

## 5. The lean-data school: how everyone else built a dataset

Not every team had 512 H100s. The other four teams that trained a model converged on a strikingly consistent recipe: take a large public corpus, and throw away 95% or more of it.

![A four column grid comparing training set construction across four published solutions, with rows for source corpus, candidate pool, after filtering, and final training set](/imgs/blogs/aimo-progress-prize-2-published-solutions-6.webp)

### 5.1 imagination-research, second place

Their SFT stage combined two existing public sets: the stage-2 data from `Light-R1` and the training data from `LIMO`, with duplicates removed. Both are collections of high-difficulty math problems with reasoning trajectories generated by DeepSeek-R1. Eight epochs on a single 8xA800 machine, eleven hours. Accuracy improved. Output length also improved, which turned out to be a problem.

So their second stage built a preference dataset specifically to shorten outputs. Starting from the default subset of `OpenR1-Math-220k`, they constructed DPO pairs where the chosen response $y_w$ and rejected response $y_l$ satisfy:

- **Correctness**: $y_w$ must be correct. $y_l$ may be correct or incorrect.
- **Length ratio**: ${\text{len}(y_w) < \tau \cdot \text{len}(y_l)}$, so the chosen response is meaningfully shorter.
- **Minimum length**: ${\text{len}(y_w) > \ell_{\min}}$, so the model does not learn to emit a bare guess.
- **Similarity**: the two responses must not be near-duplicates, measured by sentence-transformer embedding similarity below a threshold.

Their published ladder shows what each stage bought, and what it cost in GPU hours:

| Model | Stage | Pass@1 | Avg output | A800 hours |
| --- | --- | --- | --- | --- |
| dpsk-14b | base | 51.6 | 10.8k | none |
| dpsk-14b-sft | SFT, epoch 8 | 57.8 | 11.9k | 94 |
| dpsk-14b-sft-dpo2 | DPO, epoch 2 | **60.1** | 11.3k | 251 |
| dpsk-14b-sft-dpo4 | DPO, epoch 4 | 58.9 | **10.7k** | 408 |

Read the last two rows against each other. Two epochs of DPO give the best accuracy at 60.1. Four epochs give the shortest output at 10.7k, below even the base model, but accuracy slips to 58.9. They submitted both, because on a five-hour clock it is not obvious which side of that trade wins. Note also the cost: the DPO stage on 2,000 pairs consumed more GPU hours than the SFT stage on the full corpus, because preference training runs two forward passes over very long sequences.

Note what this dataset encodes. It is not a correctness dataset. A rejected response can be correct. The only signal being trained is *brevity conditional on correctness*. The final DPO set was 2,000 pairs, trained for four epochs, and it took 40 hours on 8xA800 because of the sequence lengths involved. Two thousand pairs, forty hours.

### 5.2 Fast-Math-R1, ninth place

Their dataset construction has the single sharpest trick in the competition. They pulled from three sources:

- `OpenR1-Math-220k`: 3,000 examples where R1's trace exceeded 12,800 tokens and accuracy was over 50%, plus another 3,000 where accuracy fell between 50% and 75%.
- `openr1_hard`: roughly 2,500 samples deemed hard because `r1-distill-32b` could not solve them in four tries.
- `Light-R1-SFTData` stage 2.

Then they merged, deduplicated, and here is the move: **for each problem they kept the correct generation with the shortest token length**. Where Light-R1 samples lacked ground-truth answers, they extracted the answer from the R1 trace itself.

The result was 7,900 problem-trace-answer triples. Not 7.9 million. Seven thousand nine hundred.

Selecting the shortest correct trace is a data-level intervention with the same goal as imagination-research's DPO stage, achieved without any preference optimization at all. If every example the model imitates is the tersest correct reasoning available, the model learns terse correct reasoning. It is cheaper than DPO and it composes with everything downstream.

### 5.3 So we cooked a model, 49th place

![Majority vote accuracy at 32 samples against model size, with two fine-tuned models above the distilled baselines](/imgs/blogs/aimo-progress-prize-2-published-solutions-fig15.webp)

*Figure from the ["So we cooked a model" write-up](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/so-we-cooked-a-model). Majority vote accuracy at 32 samples on AIME 2025 against model size. Their fine-tuned 14B, at roughly 0.757, sits above not just the 14B baseline at 0.671 but above the distilled 32B at 0.723, which is the whole argument for training a smaller model well rather than running a bigger one.*

This team's write-up is the most honest in the competition and their filtering is the most aggressive:

1. Start from `NuminaMath-1.5`, filtered to math word problems with integer answers drawn from olympiads, AoPS forums, AMC, previous years' AIME, olympiad references and number theory. The integer-answer filter is a direct match to the competition's answer format.
2. Join against correct R1 reasoning traces from `OpenR1-Math-220k`. That took 800,000 problems down to 27,000.
3. Then remove the easy ones. Sample 8 solutions per problem using `deepseek-r1-distill-qwen-7b-awq` at 8K tokens, and keep only problems where 7 or fewer of the 8 attempts were correct. That took 27,000 down to 8,000.
4. For GRPO they repeated the filtering with the 14B model, so the RL stage saw problems that were hard for the model actually being trained.

Step 3 is difficulty filtering by measurement rather than by metadata. A problem labelled "olympiad" that your model solves 8 times out of 8 teaches it nothing. The gradient it produces is near zero and the sample is pure cost. Sampling to find out which problems are actually hard for *your* model, then training only on those, is the same instinct as hard-negative mining in retrieval.

### 5.4 JK Piece, 21st place: the dataset that is a teacher

The most unusual construction in the field belongs to JK Piece, who did not build a static dataset at all. They ran logits distillation from `QwQ-32B-AWQ` into `DeepSeek-R1-Distill-Qwen-14B`, holding three models in memory simultaneously:

- QwQ-32B served by vLLM, generating solution text.
- QwQ-32B loaded again through `AutoModelForCausalLM`, to compute teacher logits over that text.
- The 14B student, computing its own logits over the same tokens.

The loss is KL divergence between student and teacher log-probabilities. The "dataset" is 5,274 problems: a random 6% of `OpenR1` where DeepSeek-R1 got the right answer, plus the AIMO2 reference set, AIME 2024 and AIME 2025. The actual training signal is generated on the fly by the teacher.

The difference between this and ordinary SFT on generated traces is that the student sees the teacher's full distribution over the vocabulary at every position, not just the sampled token. That is a much denser signal per example, which is why 5,274 problems is enough.

### 5.5 What the four funnels agree on

| Team | Source corpus | Final size | Selection criterion |
| --- | --- | --- | --- |
| NemoSkills | 620K AoPS discussions | 2.2M solutions | verified answer, or majority pseudo-label |
| imagination-research | Light-R1 + LIMO, then OpenR1-220k | 2K DPO pairs | correct and shorter than the rejected trace |
| Fast-Math-R1 | OpenR1-220k, openr1_hard, Light-R1 | 7.9K | shortest correct trace per problem |
| So we cooked a model | 800K NuminaMath-1.5 | 8K | integer answers, and 7 or fewer of 8 samples correct |
| JK Piece | OpenR1 6% + AIME + reference | 5.3K | teacher-generated, on the fly |

Three things are consistent across all of them.

**Difficulty is measured, not assumed.** Every team decided what was hard by sampling their own model against it, rather than trusting a source-dataset label.

**Length is a first-class selection criterion.** Two teams selected explicitly on brevity. This is unusual in general fine-tuning and is a direct consequence of a wall-clock-scored competition.

**Verification replaces annotation.** Nobody labelled anything by hand. Correctness came from an existing answer key, an LLM equivalence judge, or agreement across independent samples. That is what makes a 620K-to-2.2M pipeline possible with no annotation budget.

## 6. Teaching a reasoning model to use a Python interpreter

Tool-integrated reasoning, or TIR, is the technique that NVIDIA credits most directly for its win, and the story of how they got it working is instructive well beyond this competition.

The motivation is obvious once you see the problem set. A question like "find all bases $b > 9$ such that $b+7$ divides ${9b+7}$, and sum them" is trivially answered by a four-line loop and genuinely error-prone in natural language. Reasoning models are bad at long exact arithmetic and exhaustive search, and both appear constantly in olympiad problems.

### 6.1 Why prompting failed

The team's first attempt was the obvious one: prompt DeepSeek-R1 to write and call Python. It did not work. Their explanation is worth quoting in substance: these models struggle to deviate from their standard solution format because they have been trained extensively on reasoning tasks and have had limited exposure to instruction following. R1 has been so thoroughly shaped into "think in prose inside `<think>` tags, then answer" that a prompt asking for a different output shape gets absorbed and ignored.

This is a general lesson about heavily post-trained models. A behaviour that was never in the post-training distribution is often not promptable at all, no matter how the instruction is phrased. You cannot ask your way to it.

![Bootstrapping tool use: prompting R1 fails, so the pipeline seeds the behaviour from an instruction-following model and iterates through generation and filtering](/imgs/blogs/aimo-progress-prize-2-published-solutions-7.webp)

### 6.2 The bootstrap loop

The fix was to source the behaviour from a model that still follows instructions, then transplant it:

1. Start from `LIMO`, an instruction-following model, and give it a small-scale reasoning fine-tune.
2. Prompt *that* model for long-reasoning solutions containing Python blocks. This produced 1.2 million initial candidate solutions.
3. Filter aggressively. Three filters ran here: a **novelty** classifier separating code that does new work from code that merely verifies what the prose already concluded, a **significance** classifier grading each code call as significant, moderate or trivial, and rule-based removal.
4. What survived was a 15,000-sample stage-0 seed set. Small, but the behaviour is now demonstrated in-distribution.
5. Train on the seed, generate again with the improved model, filter again, repeat. Through several iterations of training, generation and filtering the set grew to **1.7 million TIR solutions**.
6. For the final TIR fine-tune, filter that 1.7M back down to **15,000** samples, selected on number of code executions and answer correctness.

The shape of that pipeline is the lesson: 1.2M raw, down to 15K, up to 1.7M, back down to 15K. The intermediate 1.7M is not waste. It is what makes the final 15K good, because you can only select a high-quality 15K out of a large and diverse pool.

The novelty filter deserves a note. Left unfiltered, a model that has been taught to write code will write code that recomputes what it already worked out in prose, which costs tokens and buys nothing. Filtering for code that does work the prose did not do is what makes the tool use economically worthwhile rather than decorative.

### 6.3 The sandbox contract

The runtime side is simple and tightly bounded. The model emits a code block delimited by `<tool_call>` and `</tool_call>`. Generation halts. The code runs in a Flask-based sandbox. The output is appended to the context between an output marker, and generation resumes. Generation ends at a stop phrase, the token limit, or a timeout.

The guardrails are what make it survivable inside a five-hour budget:

| Guardrail | Value | Why |
| --- | --- | --- |
| Max code calls per generation | 6 | prevents a loop of trivial calls eating the budget |
| Execution timeout | 2 seconds | an accidental infinite loop cannot stall a sample |
| Output returned to the model | 200 characters | a `print` in a loop cannot flood the context |

Two seconds is short and 200 characters is very short. Both are deliberate. The sandbox is not there to run heavy computation, it is there to do arithmetic and small searches the model would otherwise botch.

Here is roughly what the loop looks like in practice, using the streaming interface teams built on:

```python
import re, subprocess, sys

TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
MAX_CALLS, TIMEOUT_S, MAX_OUT = 6, 2, 200

def run_sandbox(code: str) -> str:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
        out = proc.stdout if proc.returncode == 0 else proc.stderr
    except subprocess.TimeoutExpired:
        out = "TimeoutError: execution exceeded 2s"
    return out.strip()[:MAX_OUT]

def generate_with_tools(engine, prompt, sampling_params):
    context, calls = prompt, 0
    while calls < MAX_CALLS:
        # stop the engine as soon as a closing tool tag appears
        text = engine.generate(context, sampling_params, stop=[TOOL_CLOSE])
        context += text
        if TOOL_CLOSE not in text and TOOL_OPEN not in text:
            return context               # finished without asking for code
        code = re.search(
            re.escape(TOOL_OPEN) + r"(.*?)$", text, re.S
        ).group(1)
        context += TOOL_CLOSE + "\n```output\n" + run_sandbox(code) + "\n```\n"
        calls += 1
    return context
```

### 6.4 The code-first school

Three other teams reached for Python without training a TIR model, and their approaches are genuinely different from each other.

**17th place** treated code generation as a three-stage prompt chain. Stage 1: give the model the question, cap it at 2,000 to 4,000 tokens, and accept a truncated output. Stage 2: build a new prompt containing an instruction, the question and that truncated output, then prefill the assistant turn with `\n</think>\n` followed by either a structured-approach header or a bare ` ```python ` fence. Prefilling past the `</think>` tag is what forces the model out of reasoning mode and straight into code. Stage 3: if the code throws, construct an error-fixing prompt containing the question, the failing code and the traceback, and retry. He also used chunked reasoning: generate 2,000 tokens at a time, combine three or four such outputs, and re-prompt with "Please think again from the combined previous thought."

**11th place** put an exhaustive-search example into a one-shot prompt, on the theory that problems which are mathematically hard may still fall to brute force. The prompt is instructive: "You are a Python code assistant... Let's have Python do the tedious calculations for us! There are multiple ways to solve this problem, so find the most efficient one."

**20th place** went furthest. Running a 32B model, he gave up entirely on generating enough tokens to reach a boxed answer, and instead forced code generation at 5K to 6K tokens, producing **up to 108 Python programs per task** alongside at most 9 boxed answers, ending generation around 7K tokens. Code was generated at temperature 0.15 while the main reasoning ran from 0.7 down to 0.2, and code executed asynchronously with a 12-second timeout while the LLM kept generating.

## 7. The machinery underneath, explained

The write-ups name their techniques in passing, the way practitioners do when talking to each other: sequence packing, context parallelism, cosine reward, W4KV8, linear merge. Each of those names hides a mechanism worth understanding, because the mechanism is what tells you when the technique applies to your problem and when it does not. This section unpacks them.

### 7.1 Sequence packing: why padding is the enemy of reasoning data

Every batch of training sequences has to be rectangular, because a GPU multiplies matrices, not ragged lists. The naive way to get a rectangle is padding: find the longest sequence in the batch, pad everything else up to it with a mask token, and eat the wasted compute.

For ordinary instruction data that waste is tolerable, because lengths cluster. For long-reasoning data it is catastrophic. A batch of R1 traces might contain one 1,200-token solution and one 23,000-token solution. Pad to the longest and roughly 95% of the shorter row is padding. Across a batch the arithmetic is brutal: if mean length is $L$ and max length is $M$, you compute $M$ per row and use $L$, so utilization is $L/M$. On reasoning corpora that ratio commonly sits between 0.3 and 0.5, meaning half to two thirds of a very expensive training run is multiplying zeros.

Packing removes the padding entirely. Instead of one example per row, you concatenate examples end to end into a fixed-length buffer, say 24,576 tokens, and start the next example immediately where the last one ended. A row might hold one 23,000-token trace, or fifteen 1,600-token traces, or any mix that fits. Utilization goes to nearly 1.0.

The catch is attention. If example A ends at position 4,000 and example B starts at 4,001, nothing in the mechanism stops a token in B from attending to tokens in A. That is a real correctness bug: the model learns to condition on an unrelated preceding problem, and it is a data leak between examples. Two standard fixes exist:

- **Block-diagonal attention masks.** Build a mask that permits attention only within each example's own span. Correct, but materializing an $n \times n$ mask for ${n = 24{,}576}$ is itself expensive.
- **Variable-length kernels.** Pass a `cu_seqlens` array, the cumulative offsets where each packed example begins, into a FlashAttention varlen kernel. The kernel never materializes the full attention matrix and simply refuses to cross a boundary. This is the approach modern trainers use, and it is why packing is nearly free today.

Position IDs need the same treatment. Each packed example must restart its positions at zero, otherwise the fifteenth example in a row thinks it begins at position 22,000 and the RoPE rotation it sees has nothing to do with what it will see at inference.

Fast-Math-R1's configuration shows the pattern directly: `max_seq_length = 24000`, `packing = True`, with `per_device_train_batch_size = 1` and `gradient_accumulation_steps = 8`. Batch size 1 looks strange until you realize that one packed row of 24,000 tokens already contains many examples.

### 7.2 Context parallelism: splitting the sequence, not the model

NemoSkills trained a 14B model on sequences long enough that the activations do not fit on one GPU. There are four ways to split a training job across devices, and they cut along different axes:

| Strategy | What is split | What gets communicated |
| --- | --- | --- |
| Data parallel | the batch | gradients, once per step |
| Tensor parallel | hidden dimensions and attention heads | activations, twice per layer |
| Pipeline parallel | layers | activations at stage boundaries |
| Context parallel | the sequence itself | keys and values, during attention |

Context parallelism is the one that specifically rescues long-sequence training. Each GPU owns a contiguous slice of the token sequence and holds only that slice's activations. For everything that is token-local, the feed-forward blocks, the layer norms, the elementwise work, this is trivially parallel: no GPU needs to know about any other GPU's tokens.

Attention is the hard part, because attention is precisely the operation where every query must see every key. If GPU 0 holds tokens 1 to 6,000 and GPU 3 holds tokens 18,001 to 24,000, then GPU 3's queries need GPU 0's keys and values.

The elegant solution is **ring attention**. Arrange the GPUs in a ring. Each device starts with its own KV block, computes the partial attention of its local queries against that block, then passes the block to its neighbour and receives a new one. After $P$ steps on $P$ devices, every query has seen every key, and no device ever held more than one block at a time. Because attention is a softmax-weighted sum, the partial results combine correctly using the same running-maximum and running-sum trick that makes FlashAttention numerically stable, so you accumulate the true attention output without materializing the full score matrix anywhere.

The memory story is what matters. Peak activation memory per device drops by roughly the parallel degree, and the attention score matrix, which is the term that grows as $n^2$, is never assembled at all. That is what turns a 24K-token training step from impossible into routine, and it is why NemoSkills reported that sequence packing and context parallelism together "significantly accelerate training on the long-reasoning data".

The communication cost is real but favourable: you ship keys and values, which are much smaller than the score matrix you avoided, and the transfers overlap with computation because you can start computing against block $i$ while block $i+1$ is still in flight.

### 7.3 Changing the RoPE base from 10,000 to 500,000

This one line in the winner's recipe does a lot of work, and it is easy to skim past.

Rotary position embeddings encode position by rotating each query and key vector by an angle proportional to its position. The vector is split into pairs of dimensions, and pair $i$ is rotated by angle ${\theta_i \cdot p}$ at position $p$, where the per-pair frequency is

$$
\theta_i = \text{base}^{-2i/d}
$$

with $d$ the head dimension. Low-index pairs rotate fast and encode fine local order. High-index pairs rotate slowly and encode coarse long-range position. Attention then depends only on the *relative* rotation between a query at position $m$ and a key at position $n$, which is what makes RoPE a relative scheme.

The problem is wraparound. A pair that rotates quickly completes a full turn and returns to where it started, so positions far apart become indistinguishable in that pair. The base sets how fast the slowest pairs rotate, and therefore how far apart two positions can be before the encoding stops separating them. With the standard base of 10,000, models are typically trained at 4K to 8K tokens and degrade beyond it.

Raising the base to 500,000 lowers every frequency by a factor of 50 in the exponent's effect, stretching the range over which rotations stay distinguishable. The model can now tell position 3,000 from position 20,000 rather than folding them together.

Why the winner needed it: they fine-tuned `Qwen2.5-14B-Base`, and their training data is long reasoning traces routinely exceeding 10,000 tokens. Without extending the positional range, the model would be asked to learn from sequences whose later positions it cannot cleanly represent. The cost is a mild loss of resolution at short range, which is a good trade when your entire task is long-form reasoning.

### 7.4 Checkpoint averaging: free generalization

Both NemoSkills and JK Piece averaged model weights across training checkpoints, and JK Piece called it the component of his solution that "really stood out".

The operation is as simple as it sounds. Take checkpoints ${\theta_1, \theta_2, \ldots, \theta_k}$ from different steps of one run and produce

$$
\bar{\theta} = \sum_{i=1}^{k} w_i \theta_i, \qquad \sum_i w_i = 1
$$

JK Piece's weights were `0.1` on each of eight intermediate checkpoints and `0.2` on the last.

Why it works comes down to the shape of the loss landscape late in training. Once optimization has entered a basin, stochastic gradient noise keeps the parameters bouncing around its walls rather than settling at the bottom. Each individual checkpoint is a sample from that bouncing. The average of many samples sits closer to the basin's center, and centers of wide basins generalize better than their edges, because a wide flat minimum is one where small parameter perturbations, including the distribution shift between training and test data, do not change the loss much.

Two practical notes. It only works for checkpoints from the same run, or from runs sharing a common ancestor, because averaging two models that fell into different basins gives you a point on a ridge between them that belongs to neither. And it costs nothing at inference: you ship one model, not an ensemble.

### 7.5 AWQ, in more detail than "4-bit"

Everyone in this competition ran AWQ, and MPWARE published his exact configuration:

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
commit_hash = "123265213609ea67934b1790bbb0203d3c50f54f"

model = AutoAWQForCausalLM.from_pretrained(model_name, revision=commit_hash)
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=commit_hash)

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}
model.quantize(tokenizer, quant_config=quant_config)
```

Four bits gives you sixteen representable values. A naive mapping of a weight matrix onto sixteen levels destroys a model, and AWQ's contribution is a specific observation about *which* weights you must not destroy.

**The salience insight.** Importance is not weight magnitude. A large weight multiplied by a consistently tiny activation contributes little to the output. What matters is the product, so AWQ measures salience by **activation** magnitude: run a small calibration set, record the average magnitude of the input to each channel, and treat channels driven by large activations as the ones worth protecting. Empirically about 1% of channels carry a hugely disproportionate share of output error if quantized badly.

**The scaling trick.** You could keep the salient 1% in FP16, but mixed-precision matrices make for miserable kernels. Instead AWQ exploits an equivalence: for a per-channel scale $s$,

$$
W x = \left(\frac{W}{s}\right)(s x)
$$

Divide a salient channel's weights by $s < 1$, which makes them larger relative to the quantization grid and therefore quantized with proportionally less error, and multiply the corresponding activation by $s$ to compensate. The mathematics is unchanged, all weights stay uniformly 4-bit, and error has been redistributed away from the channels that matter. The scales are found by a small search that minimizes output error on the calibration set.

**The configuration knobs.** `q_group_size: 128` means every run of 128 weights along the input dimension shares one scale and zero point rather than the whole tensor sharing one, which limits the damage a single outlier can do. `zero_point: True` selects asymmetric quantization, mapping ${[\min, \max]}$ onto ${[0, 15]}$ with an offset, which fits distributions that are not centered on zero better than a symmetric scheme. `version: "GEMM"` picks the kernel layout tuned for multiplying a matrix by a matrix, which is what you want when decoding many sequences at once, as opposed to a GEMV layout tuned for batch size 1.

![Two custom AWQ calibration builds compared against the public checkpoint on AIME 2025](/imgs/blogs/aimo-progress-prize-2-published-solutions-fig12.webp)

*Figure from the [5th place write-up](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/574262). Two custom AWQ calibration builds against the public casperhansen checkpoint on AIME 2025. The custom builds lead by roughly one to two accuracy points in simulation, and yet neither beat the public checkpoint across his few real submissions, which is the gap between a local measurement and a 50 problem leaderboard.*

**Why calibration tuning kept failing.** Two teams tried to recover quantization loss with better calibration data, and both found local gains that did not survive the leaderboard. The AWQ paper's own claim explains it: because salience is determined by activation *magnitude* statistics, which are fairly stable across text, the method is robust to calibration set choice. There is not much signal there to exploit, which is exactly what those teams measured the hard way.

### 7.6 What KV cache quantization actually costs

The KV cache stores, for every layer and every attention head, the key and value vectors of every token generated so far. Its size is

$$
2 \times L \times H_{kv} \times d_{head} \times n_{tokens} \times B \times \text{bytes}
$$

with $L$ layers, $H_{kv}$ key-value heads under grouped-query attention, $B$ the batch size, and the leading 2 counting keys and values separately. Two things follow. It grows linearly in both context length and batch size, which is why 7th place observed it overtaking model weights as the dominant per-token read. And it is read *in full* at every decoding step, so on a bandwidth-bound device its size translates directly into time.

Quantizing it from FP16 to int8 halves both the memory and the bytes read per step. imagination-research measured that as roughly 20% off time per output token against 16-bit KV, and 55% against FP16 weights and cache together. Their cumulative-throughput plot turns those percentages into something you can feel:

![Cumulative output tokens against elapsed seconds for three quantization settings on four L4 GPUs](/imgs/blogs/aimo-progress-prize-2-published-solutions-fig8.webp)

*Figure from the [2nd place write-up](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/572948). Cumulative output tokens against wall-clock time on four L4s at batch size 15. Reaching roughly 13,000 tokens takes 423 seconds under 4-bit weights with an 8-bit KV cache, 553 seconds with a 16-bit KV cache, and 981 seconds in FP16. On a five-hour budget across 50 questions that is the difference between finishing comfortably and timing out.*

The asymmetry everyone hit is that int8 is nearly free and int4 is not. The reason is that keys and values are activations, not weights. Weights are fixed and can be studied offline, so you can spend effort finding good scales once. Activations are produced at runtime, vary per input, and contain outlier channels whose magnitude dwarfs the rest. Squeezing an outlier-heavy distribution into sixteen levels either clips the outliers, losing the information they carry, or stretches the range to cover them, leaving ordinary values sharing a handful of levels. At eight bits, 256 levels absorb the spread. At four, they do not.

imagination-research also tried the known fix and reported it honestly: reparameterizing $W_k$ and $W_q$ to produce more channel-balanced keys gave a consistent but small improvement, especially in the int4 setting, and they ran out of submission budget before they could tune the rest of the configuration around it.

### 7.7 DPO: preference optimization as a length dial

Direct Preference Optimization is usually explained as a cheaper alternative to RLHF, which it is, but the way second place used it is more interesting than that framing suggests.

Classical RLHF trains a reward model on preference pairs, then optimizes the policy against it with PPO. DPO's insight is that for the standard KL-regularized RL objective, the optimal policy has a closed form in terms of the reward, and that relation can be inverted: the reward is recoverable from the policy. Substituting it back removes the reward model entirely and leaves a loss you can compute directly on preference pairs:

$$
\mathcal{L} = -\log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)
$$

where $y_w$ is the chosen response, $y_l$ the rejected one, $\pi_{\text{ref}}$ a frozen reference copy of the starting model, and $\beta$ controls how far the policy may drift from it. The gradient raises the chosen response's likelihood relative to the rejected one, with the reference model acting as an anchor that stops the policy collapsing.

The part worth internalizing: **DPO optimizes whatever distinction your pairs encode, and nothing else.** Second place's construction rules were that $y_w$ must be correct while $y_l$ may be correct or incorrect, that $y_w$ must be meaningfully shorter than $y_l$, that $y_w$ must exceed a minimum length, and that the two must not be near-duplicates by sentence-embedding similarity.

Because a rejected response is allowed to be correct, correctness is not the axis being trained. Length is. The minimum-length rule prevents the obvious degenerate solution of learning to emit a bare guess, and the dissimilarity rule prevents the model from learning cosmetic differences between two nearly identical traces. Two thousand pairs shaped exactly one behaviour, which is why two thousand pairs were enough.

### 7.8 GRPO: dropping the critic, and the length bias that follows

PPO needs a value network to estimate the expected return from a state, so it can compute an advantage. At LLM scale that critic is a second network of comparable size, roughly doubling memory and adding its own training instability.

Group Relative Policy Optimization removes it by using the group as its own baseline. For a prompt, sample $G$ completions, score them, and compute each one's advantage from the group's own statistics:

$$
A_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G)}
$$

A completion better than its siblings gets a positive advantage, one worse gets a negative one, and no critic is required. That is then plugged into a PPO-style clipped ratio objective with a KL penalty back to a reference model. Fast-Math-R1 used ${G = 8}$ and ${\beta = 0.04}$.

**Where the length bias comes from.** Two normalizations in the standard formulation interact badly with variable-length text. Dividing the advantage by the group standard deviation inflates updates for prompts where all completions scored similarly, which are exactly the prompts carrying least information. And normalizing the token-level loss by response length means each token in a long response receives a smaller share of the penalty, so a long wrong answer is punished more gently per token than a short wrong answer. Gradient descent notices, and responses drift longer.

The `Dr. GRPO` correction that the 49th-place team adopted is simply to remove those normalizations, which removes the bias. That is worth knowing given that the entire competition was fought over output length: a naive GRPO run pushes in exactly the wrong direction.

**The reward that did the work.** Fast-Math-R1's three-part reward is a good example of shaping:

1. A **format reward** matching `r"^.*?oxed{(.*?)}.*?</think>.*?$"`, which enforces that the boxed answer appears before the closing think tag. This is not cosmetic. At submission time they stop generation at `</think>`, so an answer that arrives after it does not exist.
2. A **cosine reward** interpolating along a cosine curve as a function of length, giving correct answers a reward in ${[1.0, 0.1]}$ that decreases with length, and incorrect answers a reward in ${[-0.1, -1.0]}$ that becomes less negative with length. The second half is the clever part. A short wrong answer is punished hardest, because a model that gives up quickly is worse than one that tried and failed, and without that term a length penalty teaches the model to bail out early on hard problems.
3. A **length reward** discouraging overthinking directly.

![Four training curve panels where the format reward and completion length both break sharply after about step 60](/imgs/blogs/aimo-progress-prize-2-published-solutions-fig13.webp)

*Figure from the [9th place write-up](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/571252). The format reward climbs to nearly 1.0 by step 40 and then falls below 0.6 around step 68, while completion length drops from roughly 11,000 tokens to about 5,500. The length objective is being satisfied right up to the point where the model stops producing well formed answers at all.*

**Why it broke.** They report the reward rising steadily until roughly step 60, after which "catastrophic shifts occurred". This is the standard failure of a shaped reward: the policy eventually finds a way to satisfy the proxy that abandons the goal, and once it starts down that path the reward keeps climbing while real accuracy collapses. Their response, taking an earlier checkpoint, is the correct one, and it depends entirely on having an external evaluation that is not the reward you are training on.

### 7.9 Logits distillation: why 5,274 problems were enough

Ordinary supervised fine-tuning on generated traces is itself a form of distillation, but a lossy one. The student sees only the token the teacher happened to sample and learns to raise that token's probability. Everything the teacher knew about the alternatives, that two other tokens were nearly as good, or that the rest of the vocabulary was hopeless, is discarded.

JK Piece trained on the full distribution instead:

$$
\mathcal{L} = D_{\text{KL}}(p_{\text{teacher}} \parallel p_{\text{student}}) = \sum_{v \in V} p_t(v) \log \frac{p_t(v)}{p_s(v)}
$$

summed over the vocabulary at every position. His implementation held three models at once: QwQ-32B-AWQ under vLLM generating solution text, QwQ-32B-AWQ again under `AutoModelForCausalLM` computing teacher logits over that text, and the 14B student computing its own logits over the same tokens.

The information content per example is far higher. A hard label carries at most $\log_2 |V|$ bits, around 17 for a 130,000-token vocabulary, and in practice much less because the answer is usually predictable. A full distribution over the vocabulary carries a real-valued number per token, and crucially it carries the teacher's *uncertainty*, which is where the reasoning-relevant information lives: the positions where a strong model is genuinely torn are the positions where the problem is hard.

That density is why 5,274 problems sufficed where SFT recipes in this competition used 8,000 filtered examples or millions of traces. The cost is that you must run the teacher's forward pass over every training token, which is why he needed the teacher resident in memory twice.

### 7.10 Model merging: why the plain average beat the clever methods

The competition produced a consistent and slightly surprising result. NemoSkills tried several mergekit methods and found "the most effective approach turned out to be a simple linear combination". JK Piece found that mergekit's slerp and sce dropped his score from 28 to 20, while his own weighted linear average was the standout.

A linear merge is

$$
\theta_{\text{merged}} = \alpha \theta_A + (1 - \alpha) \theta_B
$$

with NemoSkills using ${\alpha = 0.3}$ on the CoT checkpoint and ${0.7}$ on the TIR checkpoint.

The reason this works is **linear mode connectivity**. Two models fine-tuned from a common ancestor, for a limited number of steps, typically remain in the same loss basin, and the straight path between two points in one basin stays in low-loss territory. You are not averaging two unrelated solutions, you are interpolating between two nearby points around a shared origin. Merging models with different pretraining, or fine-tunes that ran long enough to leave the basin, gives you a point on a ridge that belongs to neither parent, which is presumably what happened to JK Piece's slerp attempt.

What makes the winner's merge more than a compromise is that the two parents differ in *behaviour* rather than knowledge. The CoT checkpoint knows how to reason in prose. The TIR checkpoint knows the same mathematics plus when to call Python, and it calls Python 2.73 times per solution. The interpolation weight becomes a continuous dial on how strongly the tool-calling behaviour expresses itself, and at ${\alpha = 0.3}$ the merged model calls code 0.85 times per solution while scoring above both parents. It did not lose the ability to use tools. It lost the compulsion to use them when they do not help.

That also explains Daniel Phalen's negative result. He trained separate models on algebra, geometry and other topics and merged them, and accuracy fell. Those parents differ in *knowledge*, partitioned across disjoint domains, and averaging the weights of a geometry specialist and an algebra specialist does not produce a model that knows both.

### 7.11 The sampling parameters, and what min_p actually does

Sampling settings varied more than anything else in the competition, and one parameter is worth explaining because it appears in nearly every configuration and is the least familiar.

`temperature` divides the logits before the softmax, flattening the distribution above 1.0 and sharpening it below. `top_p` keeps the smallest set of tokens whose cumulative probability reaches $p$. `min_p` sets a threshold *relative to the most likely token*: keep only tokens whose probability is at least ${\text{min\_p} \times p_{\max}}$.

That relative definition is what makes it robust. When the model is confident, $p_{\max}$ is high, the threshold is high, and the tail is cut aggressively. When the model is genuinely uncertain across many plausible continuations, $p_{\max}$ is low, the threshold drops, and diversity is preserved. A fixed `top_p` cannot adapt this way: the same cumulative mass admits a reasonable set in one context and a long tail of nonsense in another. That is why `min_p` values around 0.01 to 0.05 show up alongside temperature 1.0 in these solutions, a combination that would be reckless with `top_p` alone.

The settings that shipped:

| Team | Temperature | Other |
| --- | --- | --- |
| NemoSkills | 0 (near-greedy) | `redrafter_greedy_search` enabled |
| usernam | 0.9 | top_p 0.9, min_p 0.05 |
| MPWARE | 1.0 | top_p 0.90, min_p 0.05, five fixed seeds |
| sravn | 1.0 | top_p 1.0, min_p 0.01 |
| arek-paterek | 0.7 falling to 0.2 | 0.15 for code generation |
| JK Piece | 1.0 | top_p 0.9, min_p 0.05 |

The spread between the winner's temperature 0 and everyone else's 0.9 to 1.0 is not a contradiction. Sampling diversity exists to make majority voting work, and the winner's diversity came from a different source: twelve generations with different seeds through in-flight batching, plus the residual nondeterminism they observed and chose not to chase. Everyone else bought diversity with temperature.

## 8. How the teams built their validation sets

With one submission per day and roughly plus or minus 3 problems of noise on each, local evaluation was not a nicety. It was the only way to make progress.

| Team | Validation set | Samples per problem |
| --- | --- | --- |
| NemoSkills | Comp-Math-24-25: 256 problems | maj@12 to maj@64 |
| imagination-research | AIME 2025 (30) + reference (10) | 32 |
| Fast-Math-R1 | AIME 2025 (30) + reference (10) | 32 |
| So we cooked a model | AIME 2025 (30) + reference (10) | 64 |
| farsail | 100 most recent AIME problems | repeated runs, averaged |
| sravn | 50 handpicked AIME, AIME 2025 and reference | multiple passes, reshuffled |
| Daniel Phalen | 40 hard OpenR1 problems + reference | bootstrapped |

NemoSkills' benchmark is the most carefully built. Comp-Math-24-25 is 256 problems: AIME 2024 (30), AIME 2025 (30), HMMT November 2024 (62), HMMT February 2024 (68) and HMMT February 2025 (66). Larger sets reduce variance, and the AIME 2025 and HMMT 2025 portions postdate the base models' training cutoffs, which keeps them honest.

The subtlety almost everyone hit is that **the metric you validate on must match the metric you submit under**. tascj gave the clearest statement of correct methodology, in a comment reply of all places:

1. Prepare a sufficiently reliable dataset, which he notes is genuinely hard in this competition.
2. For each question, sample many responses, for example 128.
3. Check `any` correct, `majority` correct, and `average` correct.
4. Randomly subsample N responses, where N is what you could actually afford at submission time, and recheck those three statistics. Repeat many times and look at the distribution.

His conclusion from doing this: the 7B model was not bad on `any`, but the 14B model was much better on `majority` and `average`. A model selected on `pass@k` and deployed under `maj@k` is selected on the wrong axis. Several teams that chose 7B on `any`-style reasoning and then voted at submission time were quietly optimizing the wrong number.

Fifth place turned this into a proper tool, and it is the single most reusable idea in the competition for anyone tuning an inference pipeline. Instead of re-running the model for every configuration:

```
Vanilla evaluation:
  For each new scheme:
      Use the LLM to generate enough samples for this scheme
      Obtain the evaluation result

Simulation:
  Use the LLM once to generate a large pool of samples as ground truth
  For each new scheme:
      Subsample from the pool according to the scheme, until the result converges
      Obtain the evaluation result
```

Generate a big pool of traces once, with generous length and count, then answer configuration questions by resampling that pool offline. The cost of evaluating a new scheme drops from O(N) GPU-hours to O(1). He used it to grid-search batch size against sequence length, grid-search temperature (0.9 beat 0.8, while 0.85 and 0.95 were not better), and estimate the benefit of early stopping before implementing it.

This works because most inference-strategy questions, how many samples, what token cap, which stopping rule, are questions about *how you consume* generations, not about how they are produced. If you have stored the full traces with their timings, you can replay any consumption policy against them.

## 9. Layer 1: which model to run

Almost the entire field converged on `DeepSeek-R1-Distill-Qwen-14B`, quantized to 4-bit AWQ. Several teams used literally the same upload, `casperhansen/deepseek-r1-distill-qwen-14b-awq`. Fourth place's acknowledgements name DeepSeek "for releasing the R1 reasoning model mid-competition. It was a game-changer and quickly became central to nearly all top solutions."

The size question was settled empirically and consistently:

| Size | Verdict across write-ups |
| --- | --- |
| 7B | fast enough for 32 sequences, but weaker on `majority` and `average`; higher risk of overfitting the public LB |
| 14B | the consensus sweet spot: strong enough to be worth voting over, fast enough to vote many times |
| 32B | accurate but slow; forces either fewer samples or a shorter token cap, and both cost more than the accuracy gains |

MPWARE tried 32B and could run it at three attempts instead of five, with a much lower token cap, and dropped it because some problems need 20,000 or more tokens. NemoSkills found their 32B CoT models fit a batch size of 8 at 18K generation length but scored close to the 14B, so the extra size bought nothing. farsail's summary is the crispest: "32B is too slow and not perfect. 14B beats 7B on accuracy, 7B beats 14B on speed, and increasing the number of generations for 14B made it more competitive."

The dissenter is 20th place, who ran 32B AWQ and made it work precisely by abandoning the thing that makes 32B expensive. If you never intend to generate a full reasoning trace to a boxed answer, and instead stop at 5K to 7K tokens and harvest code, the 32B model's slower decode matters far less. That is a coherent strategy, not a mistake, and it reached 27 on the private set.

## 10. Layer 2: quantization, weights and KV cache

Two separate things get quantized and they behave differently.

![Decoding on L4s is memory-bandwidth bound, with the KV cache overtaking model weights as the dominant per-token read as context grows](/imgs/blogs/aimo-progress-prize-2-published-solutions-8.webp)

tascj framed the hardware problem best. An L4 provides about 300 GB/s of memory bandwidth on paper, so four of them total roughly 1200 GB/s. That is modest. Decoding is memory-bandwidth bound: for every token, you read the model weights and the entire KV cache. Long-context decoding for reasoning and batched decoding for majority voting were both essential here, and as context grows the KV cache can become larger than the model weights, at which point it is the dominant read.

That single observation motivates both halves of the quantization story.

**Weight quantization.** 4-bit AWQ was near-universal. AWQ works because not all weights matter equally: weights that are consistently multiplied by large-magnitude activations disproportionately affect the output, and AWQ scales groups of weights so that quantization error falls on the less important ones. The competition's practical finding is that 4-bit AWQ on a 14B model costs a few points of per-sample accuracy and buys enough throughput to more than pay for it through extra votes. See [weight-only quantization with GPTQ and AWQ](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq) for the mechanism.

Several teams quantized the model themselves rather than trusting an upload. MPWARE pinned a specific commit hash of the base model and ran AutoAWQ with `zero_point: True, q_group_size: 128, w_bit: 4, version: "GEMM"`. That matters because different uploads of nominally the same model scored differently: one competitor reported 28 from his own quantization against 29 from another upload, with Casper Hansen's giving him the worst results, while other teams did fine with it. That spread is measurement noise as much as anything, which is itself the lesson.

**KV cache quantization.** This is where the top teams pulled ahead, and the accuracy cliff is sharp:

| KV precision | Reported effect |
| --- | --- |
| fp16 | baseline |
| int8 | throughput gain with minimal accuracy loss; 2nd and 7th both shipped it |
| fp8 | works in vLLM and SGLang; one competitor got gibberish from a specific version |
| int4 | too much accuracy loss, dropped by everyone who tried it |

imagination-research measured 4-bit weights with 8-bit KV cutting time per output token by about 20% against 16-bit KV, and about 55% against FP16, at batch size 15 on 4xL4. On a local 2xA800 test, W4KV8 cut overall latency by 40% against FP16. Their accuracy note is careful: average per-sample accuracy drops 5% to 10% against FP16, and W4KV8 is not worse than W4KV16, while W4KV4 is worse. See [KV cache quantization and the accuracy cliff](/blog/machine-learning/inference-engineering/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff) for why int4 falls over where int8 does not.

**FP8 weights, the winner's choice.** NemoSkills had L4s, which support FP8, and TensorRT-LLM to use it. Their measured table is the cleanest quantization ablation published in the competition:

| Quantization | Speed tok/s (L4x4) | 50-question eval time (L40x4) | AIME24 | AIME25 |
| --- | --- | --- | --- | --- |
| bf16 | 210 | 2h | 82.7 | 66.7 |
| w8a16 (int8) | 315 | 1h 45m | 82.7 | 66.7 |
| w4a16 (int4) | 436 | 1h 35m | 72.7 | 60.7 |
| f8a16 (fp8) | 310 | 1h 40m | 83.3 | 68.7 |
| f8a16 + ReDrafter | 554 | 1h | 81.3 | 71.3 |

![Paired bar charts comparing decode speed on four L4 GPUs, from 210 to 554 tokens per second, against AIME24 and AIME25 accuracy for bf16, int8, int4 and fp8](/imgs/blogs/aimo-progress-prize-2-published-solutions-9.webp)

Accuracy is maj@12 of the merged model, averaged over five runs. Read the int4 row carefully: it is the fastest weight-only option by a wide margin and it loses ten points on AIME24. Int8 is free. FP8 is free and slightly better than bf16, which is within noise but certainly not a regression. The reason most of the field ran int4 AWQ anyway is that they were on vLLM or lmdeploy without a working FP8 path, and 4-bit was the available lever.

## 11. Layer 3: the inference engine

Four engines were used seriously, and the ranking depends on what you are quantizing.

tascj tested all three open engines with 4-bit weights and fp16 KV and found LMDeploy > SGLang > vLLM v1 >> vLLM v0. Fifth place published actual numbers:

| Configuration | | lmdeploy | sglang | vllm 0.7.2 |
| --- | --- | --- | --- | --- |
| 14b-awq, bs 9, seqlen 13500 | question time | 400s | 462s | 511s |
| | throughput | 304 tok/s | 263 tok/s | 238 tok/s |
| 32b-awq, bs 9, seqlen 9000 | question time | 355s | 374s | 396s |
| | throughput | 228 tok/s | 193 tok/s | 183 tok/s |
| 7b-awq, bs 32, seqlen 13500 | question time | 403s | 391s | 505s |
| | throughput | 1073 tok/s | 1103 tok/s | 856 tok/s |

![Vendor benchmarks comparing lmdeploy against vLLM on request throughput and on model initialization time](/imgs/blogs/aimo-progress-prize-2-published-solutions-fig7.webp)

*Vendor benchmarks from the lmdeploy repository, reproduced in the [2nd place write-up](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/572948). Request throughput on an A100 80G on the left, model initialization time on the right. The initialization panel is the one that mattered here: 191.69 seconds against 39.3 on an L4x4, a 4.88x difference, when the rules require your first prediction within 15 minutes of the notebook starting.*

lmdeploy with the TurboMind engine gave a 28% throughput gain over vLLM in the 14B AWQ setting, which is worth roughly two extra samples per question. It also has shorter model initialization time, which matters when you have a 15-minute deadline to make your first prediction. Second, fifth and seventh place all ran lmdeploy. Note the 7B row, where SGLang edges ahead: engine ranking is not model-size invariant. See [vLLM deep dive](/blog/machine-learning/inference-frameworks/vllm-deep-dive) and [SGLang](/blog/machine-learning/inference-frameworks/sglang-inference) for the architectural differences behind these numbers.

**vLLM still shipped a lot of solutions**, including 8th place, who ran vLLM 0.8.2 with the V1 architecture from the day he joined, `max_num_seqs=32`, `max_model_len=24576`, `tensor_parallel_size=4`, `gpu_memory_utilization=0.90`. His comment that 24K context was achievable surprised other competitors who were running at 9K to 12K.

There is a cautionary tale here too. Fifth place could not reproduce a well-known public notebook's score in his own code, spent **twenty submissions** bisecting the difference, and found that either switching the vLLM engine from v0 to v1, or using vllm-0.6.3.post1 v0 without a seed, dropped his score from 23-26 to 18-21. He later concluded overfitting was the actual cause rather than the engine. Twenty of roughly ninety possible submissions went into chasing a difference that was probably noise, which is the clearest possible illustration of section 2.

![Throughput benchmark of DeepSeek R1 on eight H200s across five input and output length mixes, comparing vLLM, SGLang and TensorRT-LLM](/imgs/blogs/aimo-progress-prize-2-published-solutions-fig2.webp)

*Benchmark by the vLLM team, reproduced in the [1st place write-up](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/nemoskills-1st-place-solution-nemoskills). TensorRT-LLM leads on the three generation-heavy workloads, at 1565, 1602 and 1039 output tokens per second, and the engines converge once the workload becomes prefill-heavy. AIMO-2 is a generation-heavy workload, which is the regime where the gap is widest.*

**TensorRT-LLM** is what NemoSkills used, converting their model to a TensorRT engine to get in-flight batching, custom attention kernels and paged KV caching. In-flight batching matters specifically here: because samples are independent, batches can mix different prompts and seeds freely, and each sample is released the moment it completes rather than waiting for the slowest member of a static batch. That property is what makes the early-stopping strategies in section 13 implementable at all. See [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention).

Their write-up illustrates why in-flight batching matters so much for this workload:

![Static batching against dynamic in-flight batching, with a third row adding code execution interleaved into generation](/imgs/blogs/aimo-progress-prize-2-published-solutions-fig3.webp)

*Figure from the [1st place write-up](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/nemoskills-1st-place-solution-nemoskills). Under static batching, in the top row, every lane holds its slot until the slowest sample in the batch finishes. Under in-flight batching each sample is released as it completes and a new one takes the slot. The third row adds code execution interleaved into generation, which is the tool-integrated case.*

The full architecture they ran is worth seeing end to end, because the sandbox is a separate service rather than something inline in the notebook:

![The first place inference loop, with a Kaggle notebook driving a TensorRT-LLM server and a separate Flask code sandbox](/imgs/blogs/aimo-progress-prize-2-published-solutions-fig4.webp)

*Figure from the [1st place write-up](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/nemoskills-1st-place-solution-nemoskills). The Kaggle notebook batches a question into n async threaded samples through NeMo-Skills and watches for a global stop criterion, while a self-hosted TensorRT-LLM FastAPI server streams generation, calls a separate Flask sandbox for code execution and SymPy answer comparison, and cancels still-running generations by id.*

**Prefix caching** is the cheap engine-level win almost everyone took. `enable_prefix_caching=True` in vLLM means multiple samples that share a prompt do not recompute its KV. Third place's entire strategy depends on it, and 20th place structured his prompt tree specifically so that vLLM could reuse KV entries across prompts with a common prefix.

## 12. Layer 4: speculative decoding

Only the winner used it, and it was worth more than any other single throughput lever they pulled.

<figure class="blog-anim">
<svg viewBox="0 0 780 250" role="img" aria-label="Standard decoding emits one token per model step while ReDrafter emits three, so the sequence grows in bursts and finishes further along in the same time" style="width:100%;height:auto;max-width:820px">
<title>Speculative decoding advances in bursts of three tokens</title>
<style>
.sd-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.sd-sub{font:400 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.sd-tok{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.sd-hot{fill:var(--accent,#6366f1);opacity:.85}
.sd-rule{stroke:var(--border,#d1d5db);stroke-width:1;stroke-dasharray:4 4}
@keyframes sd-base{from{transform:translateX(-320px)}to{transform:translateX(0)}}
@keyframes sd-spec{from{transform:translateX(-600px)}to{transform:translateX(0)}}
.sd-cb{animation:sd-base 8s steps(8,end) infinite}
.sd-cs{animation:sd-spec 8s steps(5,end) infinite}
@media (prefers-reduced-motion:reduce){.sd-cb,.sd-cs{animation:none;transform:translateX(0)}}
</style>
<defs>
<clipPath id="sd-clipBase"><rect class="sd-cb" x="140" y="40" width="320" height="60"/></clipPath>
<clipPath id="sd-clipSpec"><rect class="sd-cs" x="140" y="150" width="600" height="60"/></clipPath>
</defs>
<text class="sd-lbl" x="12" y="60">Standard</text>
<text class="sd-sub" x="12" y="80">1 token / step</text>
<line class="sd-rule" x1="140" y1="120" x2="760" y2="120"/>
<text class="sd-lbl" x="12" y="170">ReDrafter</text>
<text class="sd-sub" x="12" y="190">3 tokens / step</text>
<g clip-path="url(#sd-clipBase)">
<rect class="sd-tok" x="140" y="48" width="34" height="34" rx="6"/><rect class="sd-tok" x="180" y="48" width="34" height="34" rx="6"/><rect class="sd-tok" x="220" y="48" width="34" height="34" rx="6"/><rect class="sd-tok" x="260" y="48" width="34" height="34" rx="6"/><rect class="sd-tok" x="300" y="48" width="34" height="34" rx="6"/><rect class="sd-tok" x="340" y="48" width="34" height="34" rx="6"/><rect class="sd-tok" x="380" y="48" width="34" height="34" rx="6"/><rect class="sd-tok" x="420" y="48" width="34" height="34" rx="6"/>
</g>
<g clip-path="url(#sd-clipSpec)">
<rect class="sd-hot" x="140" y="158" width="34" height="34" rx="6"/><rect class="sd-hot" x="180" y="158" width="34" height="34" rx="6"/><rect class="sd-hot" x="220" y="158" width="34" height="34" rx="6"/><rect class="sd-hot" x="260" y="158" width="34" height="34" rx="6"/><rect class="sd-hot" x="300" y="158" width="34" height="34" rx="6"/><rect class="sd-hot" x="340" y="158" width="34" height="34" rx="6"/><rect class="sd-hot" x="380" y="158" width="34" height="34" rx="6"/><rect class="sd-hot" x="420" y="158" width="34" height="34" rx="6"/><rect class="sd-hot" x="460" y="158" width="34" height="34" rx="6"/><rect class="sd-hot" x="500" y="158" width="34" height="34" rx="6"/><rect class="sd-hot" x="540" y="158" width="34" height="34" rx="6"/><rect class="sd-hot" x="580" y="158" width="34" height="34" rx="6"/><rect class="sd-hot" x="620" y="158" width="34" height="34" rx="6"/><rect class="sd-hot" x="660" y="158" width="34" height="34" rx="6"/><rect class="sd-hot" x="700" y="158" width="34" height="34" rx="6"/>
</g>
<text class="sd-sub" x="140" y="232">Same wall clock. 310 tokens per second becomes 554 at a 65 percent full acceptance rate.</text>
</svg>
<figcaption>Both lanes run for the same time. The drafter proposes three tokens per target model step and the target verifies them in one pass, so the accepted run lands in bursts rather than one token at a time.</figcaption>
</figure>

ReDrafter is a recurrent-drafting speculative decoding method developed by Apple and implemented in TensorRT-LLM. The idea, covered in more depth in [speculative decoding: the draft-and-verify core idea](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify), is that a small draft head proposes several tokens, the large model verifies them all in a single forward pass, and every token that matches what the large model would have produced is accepted for free. Since decoding is memory-bandwidth bound, verifying three tokens costs almost the same as generating one.

The team trained the ReDrafter head themselves. They took a random subset of problems from `OpenMathReasoning-1`, generated 100,000 solutions with the target model, and trained the drafter on those. This is the detail people miss: a draft model is only as good as its match to the target model's distribution *on the workload you actually run*. A drafter trained on generic text would have a much lower acceptance rate on long mathematical reasoning traces.

The result was 3 tokens proposed per LLM step at a **65% full-acceptance rate**, for roughly a 1.8x speedup, taking 310 tok/s to 554 tok/s. Combined with FP8's 1.5x over bf16, that is a 2.6x total throughput gain over the naive baseline, which is the difference between 12 samples per question and 4.

Two honest caveats from the write-up. The TensorRT-LLM example implementation targets a non-quantized Llama, so making it work against a quantized Qwen required changes. And they tried other drafting methods without getting them to work as well, noting that LLM drafting "should work and may be better".

One curiosity: their winning submission ran at temperature 0 with `redrafter_greedy_search` enabled, and TensorRT-LLM still produced slightly varying outputs within a batch. They speculate rounding or batching effects and did not chase it. They chose near-greedy anyway because it was more stable at small batch sizes and slightly faster under speculative decoding. If you have ever been surprised that your "deterministic" batched inference is not, see [sampling numerics, determinism and batch invariance](/blog/machine-learning/inference-engineering/sampling-numerics-determinism-and-batch-invariance).

## 13. Layer 5: sampling many times

Self-consistency was used by essentially every published solution: generate multiple independent solutions, take the most frequent answer. It is the oldest trick in the test-time-compute playbook and it remains the highest-value one. See [test-time compute scaling](/blog/machine-learning/scaling-laws/test-time-compute-scaling) for the general shape.

Why it works is worth being precise about. Reasoning traces fail in many different ways and succeed in essentially one. Wrong answers scatter across the answer space while correct answers concentrate on a single value, so the mode of the answer distribution is a much better estimator than any single draw. The competition's mod-1000 answer format helps enormously here, because it collapses the answer space to 1000 discrete values and makes exact-match voting trivial.

But voting leaves a lot on the table. NemoSkills' numbers on Comp-Math-24-25 make the gap concrete:

| Method | maj@16 | pass@16 | gap |
| --- | --- | --- | --- |
| CoT | 62.9 | 76.2 | 13.3 |
| TIR | 66.8 | 80.1 | 13.3 |
| Merged (0.3 CoT + 0.7 TIR) | 69.1 | 81.3 | 12.2 |

`pass@16` means at least one of the sixteen samples was correct. `maj@16` means the mode was correct. The 12-to-13-point gap is the set of problems the model *solved* and then *voted away*, because the correct trace was outnumbered. That gap is the entire motivation for section 17.

![Stacked horizontal bars showing majority at 16 accuracy for chain of thought, tool integrated reasoning and their merge, with an amber band marking 12 to 13 points of unrecovered headroom](/imgs/blogs/aimo-progress-prize-2-published-solutions-11.webp)

Sample counts across the field:

| Team | Samples per question |
| --- | --- |
| NemoSkills | 12 async, early-stopped |
| imagination-research | 15 (7 CoT prompts, 8 code prompts), reduced under time pressure |
| Aliev | 5 growing to 14 by duplication |
| usernam | 9, fixed |
| MPWARE | 5, fixed |
| Fast-Math-R1 | 10 prompts |
| farsail | up to 16, adaptive |
| ippeiogawa | 16 (4 for code, 12 for boxed answers) |
| arek-paterek | up to 108 programs plus 9 boxed answers |

imagination-research's split across two prompt types is a deliberate diversity play: seven samples using a chain-of-thought prompt and eight using a code prompt, all voting into the same pool. Correlated samples vote badly, and two prompt styles decorrelate the failures.

## 14. Layer 6: knowing when to stop

This is where the top of the leaderboard separated from the middle, and there are two independent decisions.

<figure class="blog-anim">
<svg viewBox="0 0 780 330" role="img" aria-label="Twelve parallel generations grow left to right; each stops when it produces a boxed answer, and once four of the first five agree the still running generations are cancelled" style="width:100%;height:auto;max-width:820px">
<title>Stopping is decided twice, once per sample and once per question</title>
<style>
.es-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.es-sub{font:400 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.es-trk{fill:var(--surface,#f3f4f6)}
.es-bar{fill:var(--accent,#6366f1);opacity:.85;transform-origin:120px 0}
.es-kill{fill:var(--text-secondary,#6b7280);opacity:.35;transform-origin:120px 0}
.es-line{stroke:var(--accent,#6366f1);stroke-width:2;stroke-dasharray:5 4;opacity:0}
@keyframes es-a{0%{transform:scaleX(0)}22%{transform:scaleX(1)}100%{transform:scaleX(1)}}
@keyframes es-b{0%{transform:scaleX(0)}34%{transform:scaleX(1)}100%{transform:scaleX(1)}}
@keyframes es-c{0%{transform:scaleX(0)}46%{transform:scaleX(1)}100%{transform:scaleX(1)}}
@keyframes es-d{0%{transform:scaleX(0)}58%{transform:scaleX(1)}100%{transform:scaleX(1)}}
@keyframes es-x{0%{transform:scaleX(0);opacity:.5}58%{transform:scaleX(1);opacity:.5}62%{opacity:.18}100%{transform:scaleX(1);opacity:.18}}
@keyframes es-show{0%,56%{opacity:0}62%,100%{opacity:1}}
.es-1{animation:es-a 10s ease-out infinite}
.es-2{animation:es-b 10s ease-out infinite}
.es-3{animation:es-c 10s ease-out infinite}
.es-4{animation:es-d 10s ease-out infinite}
.es-5{animation:es-x 10s ease-out infinite}
.es-cut{animation:es-show 10s ease-out infinite}
@media (prefers-reduced-motion:reduce){.es-1,.es-2,.es-3,.es-4,.es-5{animation:none;transform:scaleX(1)}.es-cut{animation:none;opacity:1}}
</style>
<text class="es-lbl" x="12" y="26">12 parallel generations for one question</text>
<rect class="es-trk" x="120" y="46" width="600" height="16" rx="8"/><rect class="es-bar es-1" x="120" y="46" width="250" height="16" rx="8"/>
<rect class="es-trk" x="120" y="70" width="600" height="16" rx="8"/><rect class="es-bar es-2" x="120" y="70" width="330" height="16" rx="8"/>
<rect class="es-trk" x="120" y="94" width="600" height="16" rx="8"/><rect class="es-bar es-1" x="120" y="94" width="230" height="16" rx="8"/>
<rect class="es-trk" x="120" y="118" width="600" height="16" rx="8"/><rect class="es-bar es-3" x="120" y="118" width="410" height="16" rx="8"/>
<rect class="es-trk" x="120" y="142" width="600" height="16" rx="8"/><rect class="es-bar es-2" x="120" y="142" width="315" height="16" rx="8"/>
<rect class="es-trk" x="120" y="166" width="600" height="16" rx="8"/><rect class="es-bar es-4" x="120" y="166" width="470" height="16" rx="8"/>
<rect class="es-trk" x="120" y="190" width="600" height="16" rx="8"/><rect class="es-bar es-3" x="120" y="190" width="395" height="16" rx="8"/>
<rect class="es-trk" x="120" y="214" width="600" height="16" rx="8"/><rect class="es-kill es-5" x="120" y="214" width="600" height="16" rx="8"/>
<rect class="es-trk" x="120" y="238" width="600" height="16" rx="8"/><rect class="es-kill es-5" x="120" y="238" width="600" height="16" rx="8"/>
<rect class="es-trk" x="120" y="262" width="600" height="16" rx="8"/><rect class="es-kill es-5" x="120" y="262" width="600" height="16" rx="8"/>
<text class="es-sub" x="12" y="58">each bar</text>
<text class="es-sub" x="12" y="200">stops at</text>
<text class="es-sub" x="12" y="228">its answer</text>
<line class="es-line es-cut" x1="590" y1="38" x2="590" y2="278"/>
<text class="es-sub es-cut" x="470" y="300">4 of the first 5 agree, so the rest are cancelled</text>
<text class="es-sub" x="120" y="322">Sample level stopping fires on the first boxed answer. Question level stopping fires on consensus.</text>
</svg>
<figcaption>Each generation halts on its own as soon as it produces a boxed answer. Once four of the first five finished answers match, the generations still running are cancelled mid flight rather than allowed to reach their token limit.</figcaption>
</figure>

**Sample-level stopping** asks whether *this* generation is still doing useful work. imagination-research's motivation is a precise description of reasoning-model behaviour: the model usually self-doubts extensively after obtaining an answer, and after giving the answer inside `<think></think>` it will typically rewrite the whole solution again, at least twice. All of that is paid for and almost none of it changes the answer.

Their rule: stop the sample the moment you detect either the first successfully executable code block or the first `\boxed{...}`. They checked the obvious objection, that this destroys the model's chance to catch and fix its own mistake later, by writing `scripts/analyze_early_stop.py` to measure how often an initially wrong answer gets revised to a correct one. Rare enough to ignore.

Fast-Math-R1 achieved the same effect through the sampler by stopping at the `</think>` tag, and shaped their GRPO format reward around that pattern so the model would reliably emit the boxed answer before closing its thinking.

Second place published the clearest picture of how the two levels fit together as a running system:

![The second place inference workflow, with a monitor task judging early stop at both sample and question level and a speed task adjusting hyperparameters from remaining time](/imgs/blogs/aimo-progress-prize-2-published-solutions-fig9.webp)

*Figure from the [2nd place write-up](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/572948). A question enters at the left, the prompt preparation task builds both CoT and Code prompts, and lmdeploy batches the generation. A monitor task continuously runs extraction and aggregation against the stream and feeds two dotted control paths back into generation, one labelled sample-level early stop and one question-level early stop. The three green boxes are the machinery those judgements rely on: a code executor, an answer extractor and an answer aggregator. The separate speed hyperparameter task issues the three gold control signals, adjusting the sample count, the maximum sampling time and the question-level stop criterion from the time remaining.*

The diagram repays a second look, because it makes explicit something the prose in most write-ups leaves implicit. Stopping is not a rule evaluated once at the end. It is a controller running against a live stream, and the speed task is a second controller sitting above it, retuning the first one's thresholds as the clock drains.

**Question-level stopping** asks whether the *votes so far* already determine the outcome. The rules varied in sophistication:

| Team | Question-level rule |
| --- | --- |
| NemoSkills | if 4 of the first 5 generations agree, cancel the rest; also stop after 10 of 12 finish, to avoid stragglers |
| imagination-research | terminate when a majority is consistent, for example 5 of 7 |
| Aliev | at 8192 tokens, if more than 6 solutions are done and one answer holds over 70%, stop |
| ippeiogawa | 3 identical answers ends the question |
| JK Piece | cancel when any answer reaches 33% frequency across 12 sequences |

The straggler rule deserves its own mention. Cancelling after 10 of 12 complete is not about consensus at all. It is about the fact that a single sample stuck in a repetition loop will run to the token limit and hold up the entire question, and the two slowest samples contribute almost nothing to a vote that is already 10 deep.

![Three panels showing a batch of generations cut short by answer agreement, by completion count, and by the time budget](/imgs/blogs/aimo-progress-prize-2-published-solutions-fig5.webp)

*Figure from the [1st place write-up](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/nemoskills-1st-place-solution-nemoskills). Three separate reasons the winning solution cuts a batch short: enough finished samples agree, the first n of the batch have completed, or the allotted time plus its buffer has run out. The pale bars past each red line are the work that gets cancelled.*

**farsail's three-rule system, 11th place**, is the most developed and is the one I would copy. He modified vLLM so inference can end early when self-consistency converges *or diverges*, with three separate triggers:

1. **Too many junk answers.** The model returns `-1` or `0` when it fails. With 5 to 7 responses, terminate if 5 or more are junk. With 8 to 11, terminate if 4 or more. With 12 to 16, terminate if 6 or more.
2. **The responses have diverged.** Once at least 8 responses exist, if 6 or more are distinct valid answers, the samples are too scattered to converge, so stop.
3. **The winner is decided.** If the lead of the most frequent answer over the second most frequent exceeds a threshold, stop. With 4 to 7 responses the lead must be 3 or more, with 8 to 11 it must be 2 or more, with 12 to 16 it must be 1 or more.

![A sample count line from zero to sixteen with each stopping check placed at the counts where it fires](/imgs/blogs/aimo-progress-prize-2-published-solutions-fig14.webp)

*Figure from the [11th place write-up](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/573086). The whole schedule on one line. Invalid response cutoffs fire at 5 of 5, 4 of 8 and 6 of 12. The divergence check needs 6 distinct answers out of 8. The winner check needs a margin of 3, then 2, then 1 as more samples accumulate. Every threshold loosens as evidence builds.*

Rules 1 and 2 are the insight. Everyone implements "stop when you are confident". Almost nobody implements "stop when it is hopeless", and on an olympiad problem set a meaningful fraction of questions are simply not going to be solved by a 14B model no matter how long you let it run. Abandoning those early is how you fund the ones that are winnable. His framing: the number of problems that actually require the full sample budget to resolve is surprisingly small.

**Third place built stopping into the search structure itself**, and won a medal with no training at all.

His motivation came from watching validation runs: on easy problems where nine of ten branches already agree, the tenth gets stuck in a loop and generates to the token limit while everyone waits. And his second observation is the load-bearing one: the closer you are to the beginning of a solution, the more the branches have in common, and from manual review the first roughly 4,096 tokens do not differ in any fundamental way.

So the algorithm:

1. Start 5 branches, generate 4,096 tokens each.
2. Duplicate each branch, giving 10, and generate another 4,096 tokens.
3. If more than 6 solutions have completed and one answer holds more than 70% of the vote, stop.
4. Otherwise randomly select 7 unfinished branches, duplicate them to 14, and generate another 4,096 tokens.
5. Majority vote over everything collected.

![A hand drawn tree where a prompt fans to five branches, then ten, then fourteen, with finished branches feeding a majority vote](/imgs/blogs/aimo-progress-prize-2-published-solutions-fig10.webp)

*Figure from the [3rd place write-up](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/573314). Each stage is marked with its 4,096 token budget. Branches that have already produced an answer, shown green, drop out and feed the majority vote directly. Seven of the remaining branches are duplicated onward and one is dropped. The 8,192 token checkpoint and the 70 percent threshold are described in his text rather than drawn here.*

With `enable_prefix_caching=True`, duplicating a branch costs nothing for the shared prefix. You get 14 solutions of up to 12K tokens for roughly the compute of far fewer independent ones, in about 6 to 7 minutes. He is candid about the flaw: the solutions remain somewhat correlated because of the shared prefix, partially offset by varying prompts across branches. Public 25, private 30, third place, no fine-tuning.

![Flow diagram of a branching decode that runs five branches to 4096 tokens, duplicates them to ten, checks for a 70 percent majority at 8192, then duplicates seven to fourteen](/imgs/blogs/aimo-progress-prize-2-published-solutions-14.webp)

## 15. Layer 7: spending the clock

Problems vary enormously in difficulty and you cannot tell which is which in advance. Every fixed allocation is wrong.

![The time buffer: each question gets 350 seconds, surplus flows into a shared pool, and a hard question can draw up to 210 extra seconds](/imgs/blogs/aimo-progress-prize-2-published-solutions-13.webp)

**NemoSkills used a buffer.** Base allocation 350 seconds per question. If a question finished early, the unused time went into a shared buffer. The next question could draw up to 210 extra seconds from that buffer, for a ceiling of 560 seconds. Simple, bounded, and it cannot cascade into a catastrophic overrun because of the hard per-question ceiling. They had also developed a smoother strategy that computed the buffer from total time remaining, but did not use it for the winning submission.

**imagination-research used discrete speed gears.** An `adjust_speed` module tracks remaining time and remaining questions and steps between speed settings. Default speed is 3, "normal". If average remaining time per question drops below five minutes, it switches to speed 1, "fastest", which cuts samples to 10 and shortens both the per-question time cap and the early-stop criteria. Degrading gracefully under time pressure beats hitting the wall.

**tascj used profiling.** Before submitting, he profiled decode time to 16K tokens across different batch sizes. During the run, he tracked actual time per question and added `planned time - actual time` into the budget for the next one, then selected the next question's batch size from that budget using the profile. This is the most principled version: the profile is a measured cost model, and batch size is the control variable.

**Fast-Math-R1 predicted difficulty.** They trained a ModernBERT model to predict, for each problem in OpenR1-Math, the shortest token length of a correct R1 trace. That is a proxy for difficulty computed from the problem text alone, before generating anything. They observed a moderate correlation between predicted difficulty and actual tokens generated, and used it to scale the output token length dynamically between 10,500 and 13,300. Their assessment is refreshingly cautious: it stabilized public LB scores and improved by about one point, "though it could be placebo".

**20th place** ran adaptive control choosing max tokens from time remaining and a regularized estimate of time used per task, and noted a pleasant emergent property: because an early boxed answer causes a whole subtree to finish sooner, easy tasks automatically consume less time without any explicit difficulty detection.

**And fourth place refused to do any of it.** His reasoning is the best argument in the competition against a technique that everyone else adopted:

> Since questions were served in random order, the effectiveness of time management became extremely sensitive to question sequence. That likely explains some of the large fluctuations seen in public notebook results.

He split tokens and sequences equally across all 50 problems, used no system prompt, did not tune against the public leaderboard, and selected his configuration by which one generated the most content across his 50-problem validation set rather than by score. Public 25, private 29, fourth place. A dynamic policy tuned on one ordering of questions is partly fitted to that ordering, and the private set is a different draw.

## 16. Layer 8: training the model to think shorter

Here is the trap that caught several teams. Fine-tuning a reasoning model on hard problems makes it more accurate and more verbose, and in a wall-clock-scored competition the verbosity can cost more than the accuracy gains.

Second place hit the same wall and plotted it directly:

![Two panels showing supervised fine-tuning accuracy on AIME 2025 rising with epochs alongside average output tokens also rising](/imgs/blogs/aimo-progress-prize-2-published-solutions-fig6.webp)

*Figure from the [2nd place write-up](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/572948). Pass@1 on AIME 2025 against epoch on the left, average output tokens on the right, comparing Light-R1 data alone against Light-R1 combined with LIMO. Accuracy climbs with more epochs and so does length, which is precisely the trade that costs leaderboard points.*

Fast-Math-R1 measured this exactly, and it is the most useful single table in the competition:

| Experiment | Token budget | maj@32 | pass@32 | Answers collected | Avg length | Public LB |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-R1-Distill-Qwen-14B | 12800 | 0.675 | 0.775 | 16.8 | 8331 | 25 |
| SFT | 12800 | 0.725 | 0.725 | 15.7 | 7024 | 23 |
| SFT + GRPO (best checkpoint) | 12800 | 0.725 | 0.775 | 18.5 | 6817 | **29** |

Look at the second row. Local majority accuracy went **up**, from 0.675 to 0.725, and the public leaderboard went **down**, from 25 to 23. Their diagnosis: SFT introduced reasoning redundancy, so more samples failed to reach a conclusion within the time limit. At a 16,384 budget the SFT model averaged 10,396 tokens against the base model's 9,684, and the "answers collected" column fell from 16.8 to 15.7. Fewer completed samples means a weaker vote.

![Training trajectory plotting average generation length against majority at 32 accuracy, where SFT adds 712 tokens and five accuracy points and GRPO cuts 3579 tokens back off](/imgs/blogs/aimo-progress-prize-2-published-solutions-15.webp)

Their fix was GRPO with a three-part reward:

1. **Format reward** matching `r"^.*?oxed{(.*?)}.*?</think>.*?$"`, because generation is stopped at `</think>` at submission time, so the boxed answer must appear before it.
2. **Cosine reward**, correct in [1.0, 0.1] and incorrect in [-0.1, -1.0] with max length 30000. Unlike a flat accuracy reward, this applies a continuous penalty to longer correct traces and to shorter incorrect ones.
3. **Length reward** discouraging overthinking directly.

Training was one epoch, 8 generations per prompt, beta 0.04, learning rate 4e-6, about 10 hours on 8xH200. The reward optimized steadily until roughly step 60, after which "catastrophic shifts occurred" and performance collapsed, so they used an earlier checkpoint. That instability is a recurring theme.

The three-way comparison across teams is instructive because they disagree:

| Team | Method for brevity | Outcome |
| --- | --- | --- |
| Fast-Math-R1 | GRPO with cosine and length rewards | worked: public LB 23 to 29 |
| imagination-research | DPO on shorter-correct pairs | worked: shipped in both final models |
| imagination-research | GRPO, four training runs | no significant accuracy improvement |
| So we cooked a model | GRPO with Dr. GRPO reward scaling removed | worked at 7B, length penalty hurt at 14B |
| Daniel Phalen | length preference optimization | no improvement, discarded |
| Fast-Math-R1 | rewriting traces into compact form, then SFT and DPO | tokens dropped, accuracy dropped substantially |
| JK Piece | GRPO with 7B | listed under "did not work" |

GRPO worked for the teams that used it to control *length* while holding accuracy fixed, and failed for teams that expected it to raise accuracy. That is a coherent story: RL here is a behaviour-shaping tool, not an accuracy-scaling tool, at least at this scale and budget.

The "So we cooked a model" team adds two useful refinements. They removed reward scaling following the [Dr. GRPO](https://arxiv.org/abs/2503.20783) paper to eliminate length bias, and they reused each batch twice during training because actor rollouts are the expensive part of GRPO. Their scale-transfer warning is worth repeating: they spent considerable time on 7B hoping findings would carry to 14B, and relearned that good tricks at small scale often do not survive. Length penalty worked well for 7B and severely hurt accuracy at 14B.

## 17. Layer 9: merging checkpoints

Model merging appears in three solutions and it is used as a *dial*, not as an ensemble.

NemoSkills had a problem. Their CoT model was fast but less accurate. Their TIR model was more accurate but generated 15,834 tokens per solution on average against the CoT model's 11,203, and called code 2.73 times per solution, and each code call costs a generation stall plus sandbox time. They tried several mergekit methods, and the winner was the simplest: a plain linear combination of the CoT checkpoint from before TIR fine-tuning and the best TIR checkpoint after it.

| Method | maj@16 | pass@16 | Avg length | Code calls |
| --- | --- | --- | --- | --- |
| CoT | 62.9 | 76.2 | 11203 | none |
| TIR | 66.8 | 80.1 | 15834 | 2.73 |
| 0.3 CoT + 0.7 TIR | **69.1** | **81.3** | 12489 | 0.85 |

The merged model beats both parents on accuracy while sitting much closer to the CoT model on length, and it calls code less than a third as often. It has not lost the ability to use tools. It has learned to use them only when they help, which is exactly what the merge weight is controlling.

![Four panels tracking accuracy, pass rate, generation length and code calls as the merge weight moves from the chain of thought checkpoint to the tool integrated one](/imgs/blogs/aimo-progress-prize-2-published-solutions-16.webp)

Two other teams used weight averaging differently. **JK Piece** called linear weight ensembling "what really stood out" in his solution, averaging checkpoints from across his distillation run: `0.1*ckpt-20 + 0.1*ckpt-180 + 0.1*ckpt-260 + 0.1*ckpt-280 + 0.1*ckpt-360 + 0.1*ckpt-640 + 0.1*ckpt-1840 + 0.1*ckpt-2300 + 0.2*last`. Notably, mergekit's slerp and sce methods on the same checkpoints dropped his score to 20. **NemoSkills** also used weight averaging across training stages for the SFT model itself.

The negative results matter too. **"So we cooked a model"** found merging improved their 7B unambiguously, with the merged model beating both SFT and all GRPO checkpoints on accuracy and token economy, but at 14B it became a trade, recovering accuracy while losing token efficiency. **Daniel Phalen** tried training separate models per topic, algebra, geometry and so on, and merging them, and accuracy decreased.

The pattern: merging works when the parents differ in *behaviour* learned from the same base, and fails when they differ in *knowledge* partitioned across domains.

## 18. Layer 10: turning many answers into one

Given the 12-to-13-point gap between `maj@k` and `pass@k` from section 12, the aggregation step is worth attacking directly. Three approaches were tried.

![Three aggregation strategies: plain majority vote, weighted vote over generated programs, and a trained solution selector](/imgs/blogs/aimo-progress-prize-2-published-solutions-17.webp)

**Plain majority vote** is what nearly everyone shipped. Count identical answers, take the mode. Its weakness is that it treats a confidently-derived answer and a hallucinated one as equal votes.

**Weighted voting**, 20th place. With up to 108 Python programs and up to 9 boxed answers per task, plain counting would be dominated by whichever approach happened to be duplicated most. So he weighted:

- **Shrink the weights of groups of code answers that share a long common prefix.** This is the key correction. Programs generated from a similar prefix are not independent evidence, they are one piece of evidence counted many times. Discounting by shared prefix is an explicit correction for correlation between samples, which every other team's plain vote silently ignores.
- **Large bonus when a boxed answer and a code answer agree.** Two different derivation paths reaching the same number is much stronger evidence than two samples of the same path.
- **Penalty for short code**, which usually means the program did not really solve anything.
- **Penalty for a small answer value**, because degenerate failures cluster on 0, 1 and other small integers.

**GenSelect**, first place. Rather than counting, train a model to choose. The data pipeline is its own small dataset-construction exercise: regenerate solution summaries using `Qwen2.5-32B-Instruct`, because the models' native summaries are too terse; sample 2 to 16 candidate summaries per problem, ensuring at least one correct and at least one incorrect are present; generate 1 million comparative selections with QwQ-32B; filter down to 565,000. The model is trained to reason about the candidates and then pick, and capping the comparison output at 2K tokens costs only 1% to 2% accuracy while cutting inference cost sharply.

GenSelect lifted the 14B model on Comp-Math-24-25 from 76.3 to 86.7, and the 32B from 78.4 to 93.3. That is a very large gain.

And they did not ship it. Their stated reason is a pure resource argument: a generative reward model needs additional memory and time, and increasing batch size and max tokens was a simpler alternative than serving an extra model for a 50-problem set. They also tried an outcome reward model, saw initial advantages, and found the benefits diminished for long reasoning and TIR generations, particularly on subsets as small as 50 problems. The best aggregation technique in the competition lost to the five-hour clock.

## 19. Prompting, the cheapest lever anyone pulled

No GPUs required, and it produced a medal on its own.

**The universal core.** Nearly every prompt in the competition contains some version of: put the final answer in `\boxed{}`, and take the answer modulo 1000. Given exact-match grading on a 0-999 integer, a correct solution formatted wrong scores zero. imagination-research's phrasing became a near-standard: "You must put the final answer in \\boxed{}. If the final answer is greater than 1000, then take the modulo of 1000."

**Diversity across samples.** imagination-research's split of 7 CoT prompts and 8 code prompts is the clearest case. Their CoT suffix: "You excel at reasoning. Think carefully and thoroughly, avoid duplication." Their code suffix: "You excel at coding. You must provide the python code, avoid redundant analysis. The answer must be integer. There is only one answer for each question. Import necessary libraries."

Their negative finding is as useful as the positive one: diversifying the *system* prompt does not help for reasoning models. Diversifying the *task framing*, reason versus write code, does.

**Letting the model write the prompt.** 8th place asked DeepSeek-R1 to generate a prompt for solving AIMO-level problems with an R1-distilled model, telling it that problems could be quite difficult, that he did not want to spend too much time on reasoning, and that he had trouble with the modulo. The result instructs the model to plan in 1 to 2 concise steps, limit reasoning to 1 to 2 steps, verify, and box the answer, with explicit modulo instructions. He also followed DeepSeek's recommendation to use no system prompt and append instructions to the user message instead. Public 28, private 28, 8th place solo gold.

**Prompting as the entire solution.** Optimistix, with seven usable submissions, looked at the baseline notebook's prompt and asked two good questions: why are we telling the model it is helpful and harmless and that it is Qwen developed by Alibaba, and why are both prompts essentially the same? He adapted the strategy of the 4th-ranked team from AIMO-1, telling the model it is an expert mathematician especially in olympiad problems, and solving each problem two ways, once by chain of thought and once by writing sympy-based code. Public 20, which was disappointing. Private 24, a solo bronze.

**Prompting against overthinking.** The 141st-place bronze write-up used a prompt explicitly instructing the model to "Avoid overthinking and Hallucination" and to review its solution for errors and overlooked cases. He notes it never scored highly on the public leaderboard but was consistent and stable, which is exactly the property that survives a leaderboard shakeup.

A small detail from the same solution worth stealing: when the model fails or runs out of time, return `random.randint(0, 1000)` rather than a fixed default. If every timeout returns 0, and voting sees many 0s, the default can win a vote it should have lost. Randomizing the failure value prevents failures from colluding.

**And the strangest prompt outcome in the competition.** 8th place's prompt solved a notoriously hard airlines problem twice in under three minutes, and the reasoning trace ended: "Wait, given I've Googled a similar problem before, the maximum number is 79 days." There is no internet access in the notebook. The model was recalling training data and narrating it as a memory of a search. It got the right answer. A commenter's observation, that this is a Blade Runner moment with implanted memories, is about as good a summary of model behaviour under pressure as anyone managed.

## 20. Twelve solutions, twelve case studies

![Published solutions against technique, showing majority voting and AWQ nearly everywhere while custom training and dynamic time allocation cluster at the top](/imgs/blogs/aimo-progress-prize-2-published-solutions-18.webp)

### 20.1 First place, NemoSkills, 34/50

Seven researchers from NVIDIA: Christof Henkel, Darragh Hanley, Ivan Sorokin, Benedikt Schifferer, Igor Gitman, Shubham Toshniwal and Ivan Moshkov. The only team to win on model quality rather than inference tuning, and they did it by building a dataset. 620K AoPS discussions became 540K problems became 3.2M verified CoT solutions. A separate bootstrapped pipeline produced 1.7M TIR solutions filtered to 15K. SFT on `Qwen2.5-14B-Base` for 8 epochs took 48 hours on 512 H100s, followed by a 400-step TIR fine-tune at constant 1e-5, then a `0.3 CoT + 0.7 TIR` linear merge. Inference ran on TensorRT-LLM with FP8 and a self-trained ReDrafter head at 554 tok/s, 12 async generations per question, early stop at 4-of-5 agreement, 350-second base budget with a 210-second buffer.

Their honesty about what they left out is instructive. Stronger models trained on harder problems produced too many tokens to finish safely. GenSelect needed memory they preferred to spend on batch size. An outcome reward model helped less on long TIR traces. Their public LB scores across submissions were 32, 33 and 28, and the 28 made them nervous. Private aligned with their cross-validation rather than the public board, which they found "quite satisfying".

### 20.2 Second place, imagination-research, 31/50

Public 34, ranked first on the public board, and lost three points on the private set. Their solution is the most complete inference engineering in the field. SFT on Light-R1 stage 2 plus LIMO for 8 epochs, then DPO on 2,000 pairs constructed purely to shorten outputs while holding correctness, 40 hours on 8xA800. lmdeploy with TurboMind, 4-bit AWQ weights and 8-bit KV cache. Fifteen samples split 7 CoT and 8 code. Sample-level stopping on first code or first boxed answer, question-level stopping at consistent majority, and an `adjust_speed` module that drops to 10 samples when time runs short. Their write-up also documents four GRPO runs with no significant improvement, a RAG attempt that did not help, and quantization-aware training that did not recover the AWQ loss.

### 20.3 Third place, Aliev, 30/50

Public 25, private 30, and no training whatsoever. He took the popular AWQ 14B checkpoint and spent his effort entirely on search structure: 5 branches to 4,096 tokens, duplicate to 10, check for a 70% majority at 8,192, duplicate 7 unfinished branches to 14, vote. `enable_prefix_caching=True` makes the duplication nearly free. Roughly 6 to 7 minutes per question. He describes the result as partly luck while noting the core idea deserves development, and he is right on both counts. Third place, above teams that burned hundreds of GPU-hours on fine-tuning, from a scheduling insight and a vLLM flag.

### 20.4 Fourth place, sravn, 29/50

Public 25, private 29, with a solution he calls extremely simple. `DeepSeek-R1-Distill-Qwen-14B-AWQ-4bits` by casperhansen, vLLM 0.7.3 with FlashInfer 0.2.2, a single prompt appended to the user message with no system prompt following DeepSeek's own guidance, temperature 1.0, min_p 0.01, top_p 1.0. No time management by deliberate choice. Local validation on 50 handpicked problems, reshuffled across multiple passes. He selected his final configuration by which variant generated the most content across validation, not by leaderboard score. This is the anti-overfitting solution, and it beat almost everyone.

### 20.5 Fifth place, usernam, 29/50

Public 28, private 29. Model and engine were unremarkable: casperhansen's 14B AWQ on lmdeploy, batch size 9, seqlen 13500, temperature 0.9, top_p 0.9, min_p 0.05, stopping at `</think>`, no question-level early stopping because implementing it in TurboMind looked expensive during the competition. What makes this solution valuable is the methodology. He published engine throughput comparisons across lmdeploy, SGLang and vLLM at three model sizes, and he built the O(N) to O(1) simulation harness that let him grid-search batch size, sequence length and temperature offline. He also documented 20 submissions burned on a failed reproduction, and attributes his placement largely to luck, noting a near-identical submission scored 23 on the public board.

### 20.6 Seventh place, tascj, 29/50

Public 23, private 29, from a submission built in about a week with a single public leaderboard entry, titled "pure luck". His contribution is the cleanest hardware analysis in the competition: L4s provide about 300 GB/s each, the KV cache overtakes model weights as context grows, and therefore KV quantization is what buys decoding batch size. He tested all three engines, chose LMDeploy with int8 KV after finding int4 KV too lossy, profiled decode time to 16K tokens across batch sizes before submitting, and during the run added `planned time - actual time` to the next question's budget, selecting batch size from the profile. He also wrote the best short guide to validating this kind of pipeline, in a comment reply.

### 20.7 Eighth place, MPWARE, 28/50

Public 28, private 28, solo gold, and the most reproducible solution in the field. vLLM 0.8.2 with the V1 architecture from day one, `max_num_seqs=32`, `max_model_len=24576`, `tensor_parallel_size=4`, `gpu_memory_utilization=0.90`, 5 attempts at 19,200 max output tokens each with fixed seeds, `enable_prefix_caching=True`. He quantized the model himself from a pinned commit hash with AutoAWQ 4-bit GEMM. A single prompt, authored by DeepSeek-R1 on request. His own assessment: "nothing really better than most public notebooks", and "I think the real winner is DeepSeek and vLLM. The only difficulty was to find the best model with accuracy/time trade-off." He reported 26 to 28 across his last eight submissions, with stability coming from pushing max output tokens above 19,000.

### 20.8 Ninth place, Fast-Math-R1-14B, 28/50

Public 29, private 28, and the clearest demonstration of the length trap. They diagnosed the competition's central problem as redundancy in R1-series reasoning, built a 7,900-example high-difficulty dataset selecting the shortest correct trace per problem, ran 20 epochs of SFT because gains only emerged after very long training, watched their public score fall from 25 to 23 despite local accuracy rising, then used GRPO with format, cosine and length rewards to cut average generation from 7,024 to 6,817 tokens while raising collected answers from 15.7 to 18.5 and the public board to 29. They also trained a ModernBERT difficulty predictor to scale token budgets per problem. Their work became an ICML 2025 AI for Math workshop paper.

### 20.9 Eleventh place, farsail, 28/50

Public 29, private 28, using SC-TIR with a custom-modified vLLM that can terminate when self-consistency converges *or* diverges. His three stopping rules, on junk-answer count, on answer divergence, and on vote lead, are the most complete stopping logic published. He validated on the 100 most recent AIME problems, measuring his pipeline at 75 to 79 against the public notebook's 63 to 68 on the same set. His one-shot prompt embeds an exhaustive-search example, on the theory that a problem that is mathematically hard may still fall to brute force.

### 20.10 Seventeenth place, ippeiogawa, 27/50

Public 28, private 27, built around a three-stage code pipeline. Generate truncated initial reasoning at 2,000 to 4,000 tokens, then re-prompt with that output plus an instruction and a prefilled `</think>` tag to force code out, then if the code throws, re-prompt with the traceback to fix it. Sixteen samples split 4 for code and 12 for boxed answers, driven through vLLM's `LLMEngine` directly so results stream as each sample finishes. Chunked reasoning generates 2,000 tokens at a time, combines three or four outputs, and re-prompts with "Please think again from combined previous thought." Early stopping at 3 identical answers.

### 20.11 Twentieth place, arek-paterek, 27/50

The strategic outlier. A 32B AWQ model instead of 14B, and an explicit decision to stop trying to generate enough tokens to reach a boxed answer. Instead, force code generation at 5K to 6K tokens and produce up to 108 Python programs plus at most 9 boxed answers per task, ending around 7K tokens. Answers are combined by a weighted algorithm rather than majority vote, discounting code answers that share long common prefixes, rewarding agreement between a boxed answer and a code answer, and penalizing short code and small answer values. Temperature falls from 0.7 to 0.2 with code at 0.15, code runs asynchronously with a 12-second timeout while generation continues, and prompts are structured so all unfinished paths end at the same token depth to maximize prefix-cache reuse.

### 20.12 Twenty-first place, JK Piece, public 28

Logits distillation from `QwQ-32B-AWQ` into the 14B student, holding teacher-for-generation, teacher-for-logits and student in memory simultaneously and minimizing KL divergence over 5,274 problems. He rates linear weight ensembling across nine checkpoints as the standout component, and reports that mergekit's slerp and sce on the same checkpoints dropped him to 20. Evaluated at 43/60 on AIME 2024 and 2025 and 7/10 on the reference set. He used vLLM's engine interface to cancel requests once any answer reached 33% frequency across 12 sequences. His listed failures: SFT with the 7B model, GRPO with the 7B model, and the mergekit methods.

## 21. What did not work

Negative results are the most valuable part of a competition write-up, because everyone publishes their successes.

| Attempt | Team | Outcome |
| --- | --- | --- |
| GRPO for accuracy, four runs | imagination-research | no significant improvement |
| GRPO with 7B | JK Piece | did not work |
| GRPO past step 60 | Fast-Math-R1 | catastrophic reward shift, used an earlier checkpoint |
| Length penalty in GRPO at 14B | So we cooked a model | severely hurt accuracy; worked fine at 7B |
| RAG over a math problem bank | imagination-research | no improvement on AIME 2025 or online |
| LLM-based answer aggregation | imagination-research | could not stably improve |
| Quantization-aware training on AWQ scales | imagination-research | no improvement |
| W_k and W_q reparametrization for KV quant | imagination-research | small consistent gain, no submissions left to tune it |
| Constrained decoding | Fast-Math-R1 | nothing effective on their validation set |
| Rewriting traces compactly, then SFT and DPO | Fast-Math-R1 | tokens down, accuracy down substantially |
| TokenSkip CoT compression | Daniel Phalen | catastrophic forgetting; tokens down, accuracy down |
| Length preference optimization | Daniel Phalen | no gain in completed sequences or accuracy |
| Per-topic training then merging | Daniel Phalen | accuracy decreased |
| mergekit slerp and sce | JK Piece | score dropped from 28 to 20 |
| Customized AWQ calibration | usernam, Fast-Math-R1 | better locally, did not beat the stock quantization on LB |
| SFT at all | usernam | OOM even with LoRA, sequence lengths too long |
| Outcome reward model | NemoSkills | benefits diminished for long reasoning and TIR |
| GenSelect at submission time | NemoSkills | too expensive in memory and time for 50 problems |

Three clusters stand out.

**Forcing brevity externally fails.** TokenSkip, trace rewriting and constrained decoding all reduced tokens and damaged accuracy. Fast-Math-R1's conclusion is the right generalization: for highly capable models, forcibly altering their natural reasoning process from outside is not effective on complex problems. What worked instead was shifting the *distribution* the model learned from, either by selecting short correct traces as training data or by rewarding brevity in RL, so the model shortens its own reasoning rather than being cut off mid-thought.

**Recovering quantization loss is hard.** Three separate teams tried to claw back AWQ degradation through better calibration data or QAT. All reported either no gain or a local gain that did not survive the leaderboard.

**GRPO is temperamental.** It worked for length control and failed for accuracy, destabilized past a certain step count, and did not transfer from 7B to 14B.

## 22. When to reach for each technique, and when not to

The competition is over. What survives it is a set of tools with known conditions of use.

**Reach for self-consistency almost always.** It is the highest return per unit of engineering effort in this entire article, it needs no training, and it works with any engine. Reach for it when your answer space is small and discrete enough for exact-match voting. Do not reach for it when your output is free-form text with no canonical form, where you have nothing to count.

**Reach for two-level early stopping the moment you are running under a deadline.** Sample-level stopping on first answer is nearly free and imagination-research measured that it rarely destroys a correct revision. Question-level stopping on consensus is worth more. Add farsail's divergence and junk-answer rules if your workload includes problems your model simply cannot solve, because abandoning the hopeless is what funds the winnable. Do not add it if you are not time-bound, where you are only trading away accuracy.

**Reach for KV cache quantization before weight quantization** if your context is long. Int8 KV was reported as nearly free by multiple teams; int4 KV was universally too lossy. On weights, int8 or FP8 cost nothing measurable and int4 cost ten points of AIME24 accuracy in NemoSkills' ablation. Most of the field ran int4 anyway because it was the lever their engine gave them, which is a good reason but not a good default.

**Reach for a different engine before you reach for a different model.** A 28% throughput gain from swapping vLLM for lmdeploy is roughly two extra votes per question and required no training. Benchmark at your actual model size, because the ranking flipped at 7B.

**Reach for speculative decoding when your workload is narrow.** The 1.8x came from training a drafter on 100,000 solutions generated by the target model on the target distribution. A generic drafter would not have achieved a 65% acceptance rate on long mathematical reasoning.

**Reach for tool use when the failure mode is arithmetic, not reasoning.** And expect to bootstrap it rather than prompt it, if your model has been heavily post-trained into a single output format. Bound it hard: 6 calls, 2 seconds, 200 characters of output.

**Reach for RL to control length, not to raise accuracy.** Every team that used GRPO to shorten output while holding accuracy succeeded. Every team that expected accuracy gains was disappointed. If all you need is brevity, try the cheaper thing first: select the shortest correct trace per problem in your SFT data, as Fast-Math-R1 did.

**Reach for dataset construction if you have the compute and want to actually win.** This is the uncomfortable conclusion. Twelve teams tuned inference on the same checkpoint and landed between 24 and 29. The team that built 5.5 million verified solutions out of a public forum scored 34. Inference engineering has a ceiling set by the model, and the only way through it is better data.

**Be careful with dynamic time allocation.** It helped most of the top ten and fourth place beat almost all of them by refusing it, on the argument that a policy tuned against one random ordering of questions is partly fitted to that ordering. Use a bounded, simple policy like NemoSkills' 350-second base with a 210-second cap. Avoid one with many parameters tuned against a noisy leaderboard.

**And treat your validation signal as the thing to engineer first.** The single most repeated mistake in this competition was tuning against a 50-problem leaderboard with plus or minus 3 problems of noise and one attempt per day. Fifth place's simulation harness, generate a large pool once and replay consumption policies against it offline, is the idea from this competition I would carry into any inference system with a tunable sampling budget. It costs one expensive generation run and makes every subsequent configuration question free.

The next round, Progress Prize 3, is already running with a larger fund, since the \$1.6 million grand prize nobody claimed rolled over. The 47-out-of-50 bar that went unmet at 34 is the one to watch.

## 23. Sources

Every claim, number and figure in this article comes from the following primary sources. All sixteen published write-ups are linked here, and the figures reproduced above are the work of their respective authors, credited in each caption.

**Competition**

- [AI Mathematical Olympiad: Progress Prize 2](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2), hosted by AIMO and XTX Markets on Kaggle
- [Final leaderboard](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/leaderboard)

**Published solutions, by placement**

| Place | Author | Write-up |
| --- | --- | --- |
| 1 | NemoSkills (NVIDIA) | [1st place solution](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/nemoskills-1st-place-solution-nemoskills) |
| 2 | imagination-research | [2nd place solution](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/572948) |
| 3 | Aliev | [3rd place solution report](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/573314) |
| 4 | sravn | [4th place solution](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/573671) |
| 5 | usernam | [5th place solution](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/574262) |
| 7 | tascj | [7th place solution (pure luck)](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/572760) |
| 8 | MPWARE | [8th place solution (LB28) takeaway](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/571356) |
| 9 | Fast-Math-R1-14B | [Enhancing the efficiency of a reasoner using SFT and GRPO](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/571252) |
| 11 | farsail | [11th place solution](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/573086) |
| 17 | ippeiogawa | [Solution for 17th: get code and fix it with DeepSeek](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/573071) |
| 20 | arek-paterek | [20th place: generate lots of code with a 32b model](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/arek-paterek-20th-place-solution-generate-lots-of-) |
| 21 | JK Piece | [21st place solution](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/jk-piece-21st-place-solution) |
| 24 | Optimistix | [A bronze for prompt engineering](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/optimistix-a-bronze-for-prompt-engineering-public-) |
| 49 | Chan Kha Vu and team | [So we cooked a model](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/so-we-cooked-a-model) |
| 141 | C R Suthikshn Kumar | [AIMO-2 bronze medal solution](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/c-r-suthikshn-kumar-aimo-2-bronze-medal-solution) |
| public 28 | Daniel Phalen | [More distillation and things that didn't work](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/daniel-phalen-public-28-more-distillation-and-thin) |

**Papers, code and artifacts**

- Moshkov et al., [AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning Models with OpenMathReasoning dataset](https://arxiv.org/abs/2504.16891)
- [nvidia/OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning), the 5.5M-solution dataset
- [nvidia/OpenMath-Nemotron-14B-Kaggle](https://huggingface.co/nvidia/OpenMath-Nemotron-14B-Kaggle), the submitted model
- [NeMo-Skills reproduction guide](https://nvidia.github.io/NeMo-Skills/openmathreasoning1/)
- [imagination-research/aimo2](https://github.com/imagination-research/aimo2)
- [analokmaus/kaggle-aimo2-fast-math-r1](https://github.com/analokmaus/kaggle-aimo2-fast-math-r1), and the accompanying ICML 2025 AI for Math workshop paper, [A Practical Two-Stage Recipe for Mathematical LLMs](https://arxiv.org/abs/2507.08267)
- [Dr. GRPO](https://arxiv.org/abs/2503.20783), on removing the length bias in GRPO normalization
