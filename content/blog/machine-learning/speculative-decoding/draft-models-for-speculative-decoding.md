---
title: "Draft models for speculative decoding: Small LMs, N-grams, and lookup tables"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A practical guide to choosing your draft model — when to use a tiny neural LM, when n-gram lookup is enough, and when Prompt Lookup Decoding wins without any model at all."
tags:
  [
    "speculative-decoding",
    "llm-inference",
    "large-language-model",
    "deep-learning",
    "draft-model",
    "prompt-lookup-decoding",
    "model-efficiency",
  ]
category: "machine-learning"
subcategory: "Speculative Decoding"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/draft-models-for-speculative-decoding-1.png"
---

If you have read through the earlier posts in this series, you already understand *why* speculative decoding works — the cheap draft model guesses $\gamma$ tokens, the expensive target model verifies them all in one forward pass, and on average you collect far more tokens per GPU second than naive autoregressive generation allows. The math is clean. The losslessness proof is satisfying. The expected speedup formula $E[\text{accepted}] = (1 - \alpha^{\gamma+1})/(1-\alpha)$ even flatters you into thinking the whole problem is solved.

Then you sit down to actually deploy it, and the very first question slaps you in the face: **what do you use as the draft model?**

That question is not an implementation detail. The draft model is the variable that most dramatically determines whether speculative decoding gives you 3× speedup or 0.8× slowdown. Get it right and you have the single most impactful inference optimization available for latency-bound LLM serving. Get it wrong and you have added GPU memory pressure, latency overhead, and engineering complexity for no benefit.

This post is a complete field guide to draft strategies. We will go through four fundamentally different approaches — small neural language models, n-gram lookup tables, Prompt Lookup Decoding, and retrieval-augmented drafting — with concrete numbers, working Python code, and a decision framework you can apply to your specific workload today.

Before we get there, we need to be precise about what any draft strategy must actually deliver.

## The draft model contract

The acceptance-rate formula from [the core draft-and-verify post](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify) tells you that expected tokens per step grows with $\alpha$ (per-token acceptance probability) and with $\gamma$ (draft length). But it misses a critical cost: **the draft model is not free**. Every token the draft model generates takes real wall-clock time, and that time comes directly off the speedup.

Define:

- $T_{\text{target}}$ — wall-clock time for one target model forward pass (single token, batch size 1)
- $T_{\text{draft}}$ — wall-clock time for one draft model forward pass (also single token)
- $\gamma$ — number of draft tokens proposed per round
- $\alpha$ — per-token acceptance probability (empirical, task-dependent)

The **speedup ratio** compared to naive autoregressive decoding is approximately:

$$\text{Speedup} = \frac{E[\text{accepted}] \cdot T_{\text{target}}}{T_{\text{draft}} \cdot \gamma + T_{\text{target}}}$$

The numerator is how fast the target would generate the accepted tokens one by one. The denominator is the actual wall time: draft phase plus one verify pass. For speedup to exceed 1.0, we need:

$$E[\text{accepted}] > 1 + \frac{T_{\text{draft}} \cdot \gamma}{T_{\text{target}}}$$

Substituting the expected acceptance formula, this becomes a constraint on draft latency:

$$T_{\text{draft}} < \frac{T_{\text{target}}}{\ \gamma} \cdot \left[\frac{1-\alpha^{\gamma+1}}{1-\alpha} - 1\right]$$

At $\alpha=0.80$, $\gamma=4$, $T_{\text{target}}=80$ ms (a 70B model on H100), the right side evaluates to roughly **17 ms per draft token**. So your draft model must generate each token in under 17 ms to break even — and preferably in 5–10 ms to yield a meaningful speedup.

That constraint has three practical implications:

1. **A 13B draft model is often too slow.** At typical H100 memory bandwidth, a 13B model in bf16 takes 15–25 ms per token. That barely satisfies the budget and leaves nothing for memory transfer overhead.
2. **A 1B–3B model is the neural sweet spot.** At 2–5 ms per token on H100, it leaves 12–15 ms of margin to pay for data movement and scheduling.
3. **Zero-GPU-cost strategies can win outright.** N-gram lookup takes 0.1 ms. Prompt Lookup Decoding takes 0.05 ms. If their acceptance rates are high enough for your task, they beat any neural drafter on raw speedup.

There is a second constraint that neural draft models must satisfy: **shared vocabulary and tokenizer**. The rejection sampling algorithm in speculative decoding requires comparing $p_{\text{draft}}(x)$ and $p_{\text{target}}(x)$ for the *same* token $x$ over the *same* vocabulary. If the draft model uses a different tokenizer — even a slightly different one — the alignment breaks completely. The draft proposes tokens from vocabulary A; the target scores them in vocabulary B. There is no safe way to bridge this.

This rules out mixing model families. You cannot use a Mistral 1B draft for a LLaMA 70B target, because their vocabularies differ. You cannot use a GPT-2 draft for a LLaMA target. Within a family (LLaMA 1B drafting for LLaMA 70B, Qwen-0.5B drafting for Qwen-72B), the constraint is automatically satisfied.

Non-neural draft strategies — n-gram, PLD, REST — sidestep this entirely: they propose token IDs from the target's own tokenizer, so vocabulary alignment is trivially guaranteed.

![Draft strategy comparison matrix — latency, acceptance rate, setup cost, and best use case across four strategies](/imgs/blogs/draft-models-for-speculative-decoding-1.webp)

## Small language models as draft models

The most commonly deployed draft strategy in production systems is a small language model from the same family as the target — LLaMA-3 1B drafting for LLaMA-3 70B is the canonical example, used in vLLM's speculative decoding implementation and profiled extensively in the [EAGLE paper](https://arxiv.org/abs/2401.15077).

### Why same-family small LMs work

Same-family small LMs share two properties that make them excellent drafters:

**First, they share the target's tokenizer exactly.** Within the LLaMA family, LLaMA-3-1B, LLaMA-3-8B, and LLaMA-3-70B all use the same 128,256-token vocabulary and the same byte-pair encoding tokenizer. The acceptance condition $p(\text{accept} \mid x) = \min(1, p_{\text{target}}(x) / p_{\text{draft}}(x))$ is directly computable without any mapping or conversion.

**Second, they have been trained on similar data distributions.** A 1B parameter model trained on the same corpus as a 70B model learns a compressed version of the same language model. Its predictions are biased by capacity — it cannot represent long-range dependencies as well, it is less calibrated on rare tokens, and it sometimes makes confidently wrong predictions — but for high-probability continuations (common words, obvious code completions, predictable sentence endings), its distribution closely tracks the target. Those high-probability tokens are exactly the ones that generate high $\alpha$.

On typical chat tasks with greedy-style temperature settings (temperature=0.0–0.6), LLaMA-3-1B drafting for LLaMA-3-70B achieves $\alpha \approx 0.75$–$0.85$. On code generation with deterministic outputs, $\alpha$ rises to 0.85–0.92. On open-ended creative writing with temperature=1.0, it can fall below 0.70. We will discuss how to measure this in your own production environment later in this post.

There is a third factor that is often overlooked: **instruction tuning alignment**. If your target model is an instruction-tuned chat model (LLaMA-3-70B-Instruct), you want your draft model to also be instruction-tuned (LLaMA-3-1B-Instruct), not the base pretrained version. The instruction-tuned variants produce output distributions that are more peaked on assistant-style responses — shorter sentences, specific phrasing patterns, refusal templates — and those patterns must be captured by the draft to achieve high $\alpha$. The base-versus-instruct acceptance rate gap can be as large as 10–15 percentage points on typical assistant benchmarks. Always match the fine-tuning tier.

### The distribution gap problem

Even within the same model family, the small draft model has systematically different failure modes than the target. Understanding these failure modes helps you predict when $\alpha$ will be low before you measure it.

The small model underperforms the large model most severely on:

- **Rare tokens and long-tail vocabulary**: Low-frequency words, technical terminology, and proper nouns that appear fewer than a hundred times in the training corpus. The 70B model has enough capacity to memorize rare collocations; the 1B model defaults to high-frequency synonyms, producing a different token ID that triggers rejection.
- **Long-range context**: If the correct next token depends on something said 500 tokens ago in the prompt, the 1B model with its compressed representations often misses the dependency. The 70B model does not. This is why $\alpha$ tends to be lower for long-context tasks even when the output is not prompt-echo-heavy.
- **Numerical and symbolic outputs**: Exact numbers, dates, equation terms. The small model will frequently produce slightly wrong digits or skip a decimal point. Each such error is a rejection, and in math-heavy tasks like financial calculations or reasoning traces, rejection chains can drag $\alpha$ below 0.60.
- **First tokens after topic switches**: When the conversation pivots to a new topic, the large model detects the pivot from subtle cues and adapts quickly. The small model lags by a few tokens, producing high-rejection "warm-up" spans. This is why per-request $\alpha$ variance is high even when the mean is acceptable.

Knowing these failure modes lets you make task-routing decisions: send math-heavy requests to a neural drafter or to direct decoding (no spec decoding) rather than routing them through a draft model with known weak spots.

### Draft model fine-tuning for domain adaptation

Out-of-the-box acceptance rates from open-source small models are the floor, not the ceiling. Fine-tuning the draft model on domain-specific data reliably closes the distribution gap with the target by 8–15 percentage points on $\alpha$, at the cost of one to two GPU-hours of training.

The procedure is straightforward:

1. **Collect domain data.** Accumulate 10k–100k (prompt, completion) pairs from your production system. Use the *target model's* actual completions, not human-written data — you want the draft to mimic the target's specific phrasing choices, not some human proxy.
2. **Fine-tune draft on next-token prediction.** Standard cross-entropy loss on the completion tokens, conditioning on the prompt. Use a small learning rate ($3 \times 10^{-5}$) to avoid forgetting the base capabilities.
3. **Evaluate acceptance rate on held-out set.** Measure $\hat{\alpha}$ before and after fine-tuning using the offline measurement procedure described later. A 10-point gain is typical; anything less than 5 points suggests the domain data does not have high-enough overlap with deployment distribution.
4. **Iterate.** As your production traffic evolves, retrain the draft model monthly or on a trigger (when online $\alpha$ drops below 0.70).

One practical consideration: fine-tuning on target completions can cause **mode collapse** if the target is very deterministic. The draft learns to reproduce the target's specific response format so closely that it loses breadth — it does great on common phrases but fails catastrophically on anything unusual. Mitigate this by mixing 20% generic pretraining data into the fine-tuning batch.

A concrete training script for domain-adapted draft fine-tuning:

```python
## draft_model_finetune.py
## Fine-tune a small draft model on target model completions for higher alpha.
## Requires: transformers 4.40+, peft 0.11+, datasets, accelerate

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType


def prepare_domain_dataset(
    completion_pairs: list[dict],
    tokenizer,
    max_length: int = 2048,
) -> Dataset:
    """
    completion_pairs: list of {"prompt": str, "completion": str}
    Tokenizes and packs into dataset format for causal LM training.
    """
    def tokenize_pair(example):
        full_text = example["prompt"] + example["completion"] + tokenizer.eos_token
        tok = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        ## Mask loss on prompt tokens: only train on completion
        prompt_len = len(tokenizer(example["prompt"])["input_ids"])
        labels = tok["input_ids"].copy()
        labels[:prompt_len] = [-100] * prompt_len  ## -100 = ignore in loss
        tok["labels"] = labels
        return tok

    dataset = Dataset.from_list(completion_pairs)
    return dataset.map(tokenize_pair, remove_columns=dataset.column_names)


def finetune_draft_model(
    draft_model_id: str,
    domain_pairs: list[dict],
    output_dir: str,
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 3e-5,
    use_lora: bool = True,
) -> str:
    """
    Fine-tune draft model on domain completions.
    use_lora=True recommended for memory efficiency (adds ~50 MB adapter).
    Returns path to the saved fine-tuned model.
    """
    tokenizer = AutoTokenizer.from_pretrained(draft_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        draft_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,                  ## LoRA rank — 16 is a good default
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        ## Expect ~1% of parameters to be trainable with LoRA rank=16

    dataset = prepare_domain_dataset(domain_pairs, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_steps=50,
        logging_steps=25,
        save_strategy="epoch",
        fp16=False,
        bf16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(output_dir)

    return output_dir


if __name__ == "__main__":
    ## Example: fine-tune LLaMA-3-1B on 10k pairs from a legal domain system
    example_pairs = [
        {
            "prompt": "[INST] Summarize this contract clause: ...[/INST]",
            "completion": "This clause establishes ...",
        }
        ## ... your 10k production pairs here
    ]
    out = finetune_draft_model(
        draft_model_id="meta-llama/Meta-Llama-3-1B-Instruct",
        domain_pairs=example_pairs,
        output_dir="./llama3-1b-legal-draft",
    )
    print(f"Fine-tuned draft saved to: {out}")
```

With LoRA rank=16 on LLaMA-3-1B, this training run takes approximately 15 minutes on a single A100 80GB for 10k examples. The resulting adapter is only ~50 MB on disk and can be loaded atop the base draft model at serving time with zero inference overhead.

### The KV cache sharing trick

Standard speculative decoding loads two separate models: draft model with its own KV cache, target model with its own KV cache. That doubles the memory pressure. On a single H100 80GB, LLaMA-3-70B in bf16 requires approximately 140 GB — already exceeding the GPU. In practice, production deployments either use multi-GPU tensor parallelism or offload parts to CPU. Adding a 1B draft model costs ~2 GB extra, which is manageable.

But there is a smarter option: **KV cache sharing**, sometimes called *draft model as a prefix layer*. The draft model and target model process the same prompt, so they produce identical KV representations up to the point where the architectures diverge. If you arrange the draft model as a set of the target's first few layers (or as a separately stored shallow network that shares the target's embedding layer), you can reuse part of the target's KV computation for the draft pass. This is the architectural insight behind EAGLE and Medusa — posts 5 and 6 in this series go deep on those designs.

For a classic two-model setup, KV sharing is harder to exploit but not impossible. The Recurrent Drafter paper describes a variant where the draft model receives the target's KV cache as a prefix, reducing redundant computation by about 20% at the cost of some communication overhead.

![Two-model versus shared-weight speculative decoding: memory footprint and tokenizer implications](/imgs/blogs/draft-models-for-speculative-decoding-2.webp)

### Implementing a neural draft model

Here is a self-contained implementation of two-model speculative decoding using Hugging Face Transformers. This is not a toy — it handles the full rejection-sampling logic from [Post 3](/blog/machine-learning/speculative-decoding/speculative-decoding-token-acceptance-rejection-sampling) and produces output that is provably identical in distribution to running the target alone.

```python
## speculative_decoding_lm_draft.py
## Two-model speculative decoding with a small LM draft.
## Tested on: transformers 4.40+, torch 2.2+, CUDA 12.1+
## Usage: python speculative_decoding_lm_draft.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional


def load_models(
    draft_model_id: str = "meta-llama/Meta-Llama-3-8B",
    target_model_id: str = "meta-llama/Meta-Llama-3-70B",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load draft and target models onto the same or different devices."""
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)

    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_id,
        torch_dtype=dtype,
        device_map=device,
    )
    draft_model.eval()

    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        torch_dtype=dtype,
        device_map=device,
    )
    target_model.eval()

    return tokenizer, draft_model, target_model


@torch.inference_mode()
def speculative_decode(
    prompt: str,
    tokenizer,
    draft_model,
    target_model,
    max_new_tokens: int = 200,
    gamma: int = 4,
    temperature: float = 1.0,
    top_p: float = 0.95,
    device: str = "cuda",
) -> str:
    """
    Speculative decoding: draft γ tokens with small LM, verify with target.
    Returns text with identical distribution to target-only generation.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated_ids = input_ids.clone()

    ## Track per-round acceptance for monitoring
    total_accepted = 0
    total_rounds = 0

    draft_past_kv = None
    target_past_kv = None

    eos_id = tokenizer.eos_token_id
    max_seq = input_ids.shape[1] + max_new_tokens

    while generated_ids.shape[1] < max_seq:
        ## ── DRAFT PHASE ──────────────────────────────────────────────
        ## Generate γ candidate tokens autoregressively with draft model.
        draft_token_ids = []
        draft_log_probs = []  ## p_draft(x_i | x_<i) for each i

        cur_ids = generated_ids
        past_kv_d = None  ## fresh KV for draft each round

        for step in range(gamma):
            draft_out = draft_model(
                input_ids=cur_ids if past_kv_d is None else cur_ids[:, -1:],
                past_key_values=past_kv_d,
                use_cache=True,
            )
            logits_d = draft_out.logits[:, -1, :]  ## (1, vocab)
            past_kv_d = draft_out.past_key_values

            ## Apply temperature
            if temperature != 1.0:
                logits_d = logits_d / temperature

            ## Compute probabilities and sample
            probs_d = F.softmax(logits_d, dim=-1)
            next_tok = torch.multinomial(probs_d, num_samples=1)  ## (1,1)

            draft_token_ids.append(next_tok)
            draft_log_probs.append(probs_d[0, next_tok.item()].item())

            cur_ids = torch.cat([cur_ids, next_tok], dim=1)

            if next_tok.item() == eos_id:
                break  ## Stop drafting early on EOS

        draft_tokens = torch.cat(draft_token_ids, dim=1)  ## (1, actual_γ)
        actual_gamma = draft_tokens.shape[1]

        ## ── VERIFY PHASE ──────────────────────────────────────────────
        ## Run target model on [context + all γ draft tokens] in one pass.
        verify_ids = torch.cat([generated_ids, draft_tokens], dim=1)

        target_out = target_model(
            input_ids=verify_ids,
            use_cache=False,  ## No KV cache in verify for simplicity
        )
        ## Logits at positions [seq_len-1 ... seq_len+γ-1] are what we need.
        ## Position i gives p_target for token at position i+1.
        target_logits = target_out.logits[:, -actual_gamma - 1 :, :]  ## (1, γ+1, V)

        if temperature != 1.0:
            target_logits = target_logits / temperature

        target_probs = F.softmax(target_logits, dim=-1)  ## (1, γ+1, V)

        ## ── ACCEPT/REJECT PHASE ──────────────────────────────────────
        ## Apply modified rejection sampling from Post 3.
        n_accepted = 0
        for i in range(actual_gamma):
            x_i = draft_token_ids[i].item()
            p_d = draft_log_probs[i]                          ## p_draft(x_i)
            p_q = target_probs[0, i, x_i].item()              ## p_target(x_i)

            accept_prob = min(1.0, p_q / (p_d + 1e-10))
            u = torch.rand(1).item()

            if u <= accept_prob:
                ## Accept this draft token
                n_accepted += 1
                generated_ids = torch.cat(
                    [generated_ids, draft_token_ids[i]], dim=1
                )
                if x_i == eos_id:
                    return tokenizer.decode(
                        generated_ids[0, input_ids.shape[1] :],
                        skip_special_tokens=True,
                    )
            else:
                ## Reject: resample from adjusted distribution (q - α*p)+
                adjust = torch.clamp(
                    target_probs[0, i] - (p_d / (p_q + 1e-10)) * target_probs[0, i],
                    min=0.0,
                )
                ## If adjust is all-zero (floating point underflow), fall back to target
                if adjust.sum() < 1e-9:
                    adjust = target_probs[0, i]
                adjust = adjust / adjust.sum()
                bonus_tok = torch.multinomial(adjust, num_samples=1).unsqueeze(0)
                generated_ids = torch.cat([generated_ids, bonus_tok], dim=1)
                if bonus_tok.item() == eos_id:
                    return tokenizer.decode(
                        generated_ids[0, input_ids.shape[1] :],
                        skip_special_tokens=True,
                    )
                break  ## Stop processing remaining draft tokens after rejection

        ## Bonus token: if all γ draft tokens were accepted, sample one more
        ## from the target's distribution at position γ.
        if n_accepted == actual_gamma:
            bonus_probs = target_probs[0, actual_gamma]  ## (V,)
            bonus_tok = torch.multinomial(bonus_probs, num_samples=1).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, bonus_tok], dim=1)
            if bonus_tok.item() == eos_id:
                return tokenizer.decode(
                    generated_ids[0, input_ids.shape[1] :],
                    skip_special_tokens=True,
                )

        total_accepted += n_accepted
        total_rounds += 1

    ## Decode the generated token IDs (strip the original prompt)
    return tokenizer.decode(
        generated_ids[0, input_ids.shape[1] :],
        skip_special_tokens=True,
    )


if __name__ == "__main__":
    tokenizer, draft, target = load_models(
        draft_model_id="meta-llama/Meta-Llama-3-8B",
        target_model_id="meta-llama/Meta-Llama-3-70B",
    )
    prompt = "Explain the difference between a heap and a stack in computer science:"
    result = speculative_decode(
        prompt=prompt,
        tokenizer=tokenizer,
        draft_model=draft,
        target_model=target,
        max_new_tokens=150,
        gamma=4,
        temperature=0.6,
    )
    print(result)
```

The implementation above is pedagogically complete but not maximally optimized — in particular, the verify phase does not reuse KV cache, which a production implementation would. The [vLLM serving](/blog/machine-learning/large-language-model/vllm-inference) implementation handles this with a draft worker that maintains its own KV buffer and passes draft tokens to the target worker asynchronously.

## N-gram drafters

The second category is radically different: no model at all, just a lookup table over the context itself. The insight is simple but powerful: **natural language and code are highly repetitive**. Function names repeat. Boilerplate repeats. Common phrases repeat. If the model has recently generated "for i in range(" five lines ago, it will very likely generate "for i in range(" again. Instead of spending GPU cycles to figure that out, scan the existing context for matching patterns and propose the continuation directly.

### How n-gram lookup works

At each draft step, the algorithm:

1. Takes the last $N$ tokens of the current sequence as a lookup key (typically $N=3$).
2. Scans the full context window for occurrences of that exact $N$-gram.
3. If a match is found, copies the $\gamma$ tokens that follow the match as draft candidates.
4. If no match is found, falls back to another strategy (or proposes nothing, triggering a normal target forward pass).

The computational cost is $O(\text{ctx\_len})$ for the scan, which at 4096 tokens is on the order of tens of microseconds on CPU — three to four orders of magnitude faster than even the smallest neural draft model.

![N-gram lookup pipeline: scan context for matching prefix, propose continuation, verify with target](/imgs/blogs/draft-models-for-speculative-decoding-3.webp)

The acceptance rate for n-gram lookup is heavily task-dependent. On Python code generation where functions follow templates, on formal documents with repeated headers, or on tasks where the model is expected to echo or paraphrase the input, $\alpha$ can reach 0.75–0.85. On open-domain chat with no repeated phrases, it drops below 0.40, at which point the strategy barely breaks even (you are only collecting $\sim$1.3 tokens per verify pass, barely better than the 1 you get from the bonus token alone).

The Hugging Face `transformers` library has had an [n-gram assistant](https://huggingface.co/docs/transformers/generation_strategies#assisted-decoding) since version 4.35. Setting `do_sample=False` and `assistant_model=None` with `num_assistant_tokens_schedule="constant"` uses n-gram lookup automatically when no assistant model is provided:

```python
## n-gram speculative decoding via HuggingFace Transformers
## transformers >= 4.40 required for ngram_assistant_model
## Documentation: https://huggingface.co/docs/transformers/generation_strategies

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def ngram_speculative_demo(
    model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    prompt: str = "Write a Python quicksort implementation:",
    gamma: int = 5,
    ngram_size: int = 3,
    max_new_tokens: int = 256,
) -> str:
    """
    N-gram speculative decoding: no draft model, uses context patterns.
    Best on repetitive tasks: code, structured text, copy-heavy outputs.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    ## Use the built-in prompt_lookup_num_tokens which implements n-gram lookup
    ## via the "prompt lookup decoding" interface (n-gram over full context, not just prompt)
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        prompt_lookup_num_tokens=gamma,   ## proposes up to γ tokens per step
    )

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            generation_config=generation_config,
        )

    return tokenizer.decode(
        output_ids[0, inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )


def measure_ngram_acceptance_rate(
    model,
    tokenizer,
    prompt_text: str,
    reference_output: str,
    gamma: int = 4,
    ngram_size: int = 3,
    device: str = "cuda",
) -> float:
    """
    Empirically measure the n-gram acceptance rate on a reference output.
    Simulates the speculative process and counts accepted vs proposed tokens.
    """
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    ref_ids = tokenizer(reference_output, return_tensors="pt").input_ids.to(device)[0]

    total_proposed = 0
    total_accepted = 0
    context = prompt_ids[0].tolist()

    for pos in range(min(len(ref_ids) - gamma, 200)):  ## sample first 200 positions
        ## Build n-gram lookup key from last ngram_size context tokens
        key = tuple(context[-ngram_size:])
        ctx_tensor = context[: len(context)]

        ## Find all occurrences of the key in the context
        matches = []
        for i in range(len(ctx_tensor) - ngram_size):
            if tuple(ctx_tensor[i : i + ngram_size]) == key:
                ## Record where this n-gram appears so we can look at continuation
                if i + ngram_size + gamma <= len(ctx_tensor):
                    matches.append(ctx_tensor[i + ngram_size : i + ngram_size + gamma])

        if matches:
            proposed = matches[0]  ## Use first match as proposal
            for j, tok in enumerate(proposed):
                total_proposed += 1
                if pos + j < len(ref_ids) and ref_ids[pos + j].item() == tok:
                    total_accepted += 1
                else:
                    break  ## First mismatch stops the chain

        ## Advance context with the ground-truth next token
        context.append(ref_ids[pos].item())

    return total_accepted / max(total_proposed, 1)


if __name__ == "__main__":
    result = ngram_speculative_demo(
        prompt="Implement a binary search algorithm in Python:",
        gamma=5,
    )
    print(result)
```

### Tuning the n-gram hyperparameters

Two hyperparameters determine how aggressive the n-gram lookup is:

**N (lookup key length)**: Shorter keys (N=2) match more often but have higher false-match rates; longer keys (N=5) match rarely but the proposals are almost always semantically correct. The sweet spot on code tasks is N=3–4. On prose tasks where grammar creates many identical 2-grams ("the", "of the", "is a"), use N=4–5. The acceptance rate sensitivity to N is roughly:

| N | Match frequency | False match rate | Alpha on code | Alpha on chat |
|---|---|---|---|---|
| 2 | Very high | 60–80% | 0.55 | 0.35 |
| 3 | High | 25–40% | 0.72 | 0.50 |
| 4 | Medium | 10–20% | 0.74 | 0.55 |
| 5 | Low | 3–8% | 0.71 | 0.58 |

The N=3 and N=4 points give near-identical alpha on code, because most false matches at N=3 are rejected within the first proposed token. The difference only matters when false match rates interact with the reject penalty: at N=2 on chat tasks, false matches are so common that the n-gram drafter is actively harmful.

**γ (proposal length)**: Unlike with neural drafters where increasing γ raises expected tokens accepted, n-gram proposals truncate at the first mismatch. A γ=8 proposal on a false match still only gets you 0 accepted tokens. The practical recommendation is to use γ=4–5 for n-gram lookup — larger values waste memory without improving expected yield.

### When n-gram lookup is worth it

N-gram lookup shines in a specific niche: tasks where the output is structurally predictable from the context. The best examples are:

- **Code generation from templates**: boilerplate methods, class definitions, import blocks
- **Document formatting**: generating headers, bullet structures, table rows that mirror earlier rows
- **Translation into a rigid format**: converting JSON/YAML where keys repeat
- **Text editing tasks**: the model is asked to copy most of the original and change a few words

A quick empirical test: count the fraction of ground-truth completions where the first token appears in the last 1000 tokens of the context. If that fraction exceeds 40%, n-gram lookup will find matches on a significant share of decode steps. If it is below 15%, n-gram lookup is not worth the engineering overhead.

The critical failure mode is **false matches**: the n-gram key appears in the context but in a different semantic context, producing a continuation that the target will reject. On creative tasks, false-match rates can reach 60–80%, making the overhead negative. Track your acceptance rate; if it falls below 0.55, either raise the n-gram size $N$ (more conservative, fewer matches) or switch to a neural drafter.

## Prompt Lookup Decoding

Prompt Lookup Decoding (PLD), introduced by Apoorv Saxena in [a 2023 Hugging Face post](https://huggingface.co/blog/whisper-speculative-decoding) and formalized in subsequent papers, is the most elegant zero-cost draft strategy ever proposed. It has a beautiful precondition: **for many tasks, the output heavily copies from the input prompt**.

Summarisation copies topic sentences. Long-context QA quotes passages. RAG-augmented generation echoes retrieved text. Document editing preserves most of the original. In all these cases, the model's output at decode step $t$ is statistically very likely to match a substring of the input prompt — not because the model is "lazy" but because the task semantically requires echoing the source.

PLD exploits this with a two-step lookup:

1. Take the last $k$ tokens of the current decode prefix as a lookup key (typically $k=4$).
2. Scan the input prompt (not the full context — just the original prompt) for substrings that end with this exact $k$-gram.
3. If found, propose the $\gamma$ tokens that follow the match in the prompt as draft candidates.

The beauty of restricting the search to the prompt (rather than the full context as n-gram does) is that prompt substrings are more likely to be semantically relevant continuations, and the search can be made extremely fast with preprocessing.

![Prompt Lookup Decoding algorithm: match decode suffix against input prompt chunks, propose continuation verbatim](/imgs/blogs/draft-models-for-speculative-decoding-4.webp)

On summarisation benchmarks (CNN/DailyMail, XSum, SCROLLS), PLD achieves acceptance rates of 0.60–0.80. On multi-document QA where the answer quotes from the retrieved context, $\alpha$ reaches 0.75–0.85. The original implementation from `huggingface/transformers` sets `prompt_lookup_num_tokens` in the generation config:

```python
## Prompt Lookup Decoding implementation and measurement
## Source: Saxena 2023 (huggingface/transformers prompt_lookup_num_tokens)
## transformers >= 4.35 supports prompt_lookup_num_tokens natively

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer


def benchmark_pld_vs_baseline(
    model_id: str,
    prompt: str,
    max_new_tokens: int = 300,
    gamma: int = 5,
    num_runs: int = 5,
) -> dict:
    """
    Compare PLD vs baseline autoregressive generation.
    Returns timing and token-count results.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs.input_ids.shape[1]

    ## ── Baseline: standard autoregressive ──────────────────────────
    baseline_times = []
    baseline_tokens = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        torch.cuda.synchronize()
        baseline_times.append(time.perf_counter() - t0)
        baseline_tokens.append(out.shape[1] - input_len)

    ## ── PLD: prompt lookup speculative decoding ──────────────────────
    pld_times = []
    pld_tokens = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out_pld = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                prompt_lookup_num_tokens=gamma,  ## PLD: look up γ tokens from prompt
            )
        torch.cuda.synchronize()
        pld_times.append(time.perf_counter() - t0)
        pld_tokens.append(out_pld.shape[1] - input_len)

    avg_baseline_ms = 1000 * sum(baseline_times) / num_runs / (sum(baseline_tokens) / num_runs)
    avg_pld_ms = 1000 * sum(pld_times) / num_runs / (sum(pld_tokens) / num_runs)

    return {
        "baseline_ms_per_token": round(avg_baseline_ms, 2),
        "pld_ms_per_token": round(avg_pld_ms, 2),
        "speedup": round(avg_baseline_ms / avg_pld_ms, 2),
        "baseline_tokens_avg": round(sum(baseline_tokens) / num_runs),
        "pld_tokens_avg": round(sum(pld_tokens) / num_runs),
    }


def pld_with_acceptance_tracking(
    model_id: str,
    system_prompt: str,
    document_to_summarize: str,
    max_new_tokens: int = 200,
    gamma: int = 5,
) -> tuple[str, float]:
    """
    PLD generation with acceptance-rate estimation via logit comparison.
    Returns (generated_text, estimated_alpha).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    full_prompt = f"{system_prompt}\n\nDocument:\n{document_to_summarize}\n\nSummary:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    prompt_ids = inputs.input_ids[0].tolist()  ## IDs of the input prompt

    ## Track how often proposed tokens match target greedy output
    total_proposed = 0
    accepted_proposed = 0

    context_ids = list(prompt_ids)
    generated = []
    lookup_key_len = 4

    for _ in range(max_new_tokens):
        key = tuple(context_ids[-lookup_key_len:])

        ## Search prompt for this key
        proposal = None
        for i in range(len(prompt_ids) - lookup_key_len - 1):
            if tuple(prompt_ids[i : i + lookup_key_len]) == key:
                proposal = prompt_ids[i + lookup_key_len : i + lookup_key_len + gamma]
                break

        ## Get target's greedy prediction at current position
        inp = torch.tensor([context_ids], dtype=torch.long, device="cuda")
        with torch.inference_mode():
            logits = model(inp).logits[0, -1, :]
        target_next = logits.argmax().item()

        if proposal:
            ## Count acceptance of first proposed token vs target's greedy choice
            total_proposed += 1
            if proposal[0] == target_next:
                accepted_proposed += 1

        next_tok = target_next
        context_ids.append(next_tok)
        generated.append(next_tok)

        if next_tok == tokenizer.eos_token_id:
            break

    alpha_estimate = accepted_proposed / max(total_proposed, 1)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, alpha_estimate
```

### PLD's boundaries

PLD's limitation is equally beautiful in its precision: it only works when the output copies from the *input prompt*. The moment the task requires generating content not in the prompt — creative writing, mathematical reasoning, factual recall from parametric memory — PLD finds no matches and contributes nothing. It has a hard ceiling on which tasks it helps.

The practical rule: if the output's unigram overlap with the prompt exceeds 35%, PLD is worth trying. If it is below 20%, PLD will have near-zero acceptance rate and you should use a neural drafter instead.

### Measuring PLD effectiveness before deployment

Before enabling PLD on a new task type, you can compute the expected $\alpha$ analytically from a sample of (prompt, reference_completion) pairs — without running a single GPU inference:

```python
## pld_overlap_analyzer.py
## Measure expected PLD acceptance rate from (prompt, completion) pairs.
## No GPU required — runs on CPU in < 1 second per pair.

from transformers import AutoTokenizer
from collections import defaultdict
import statistics


def compute_pld_alpha_estimate(
    prompt_text: str,
    completion_text: str,
    tokenizer,
    key_len: int = 4,
    gamma: int = 5,
) -> float:
    """
    Estimate the PLD acceptance rate for a single (prompt, completion) pair.
    Returns per-token acceptance rate alpha in [0, 1].
    """
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    comp_ids = tokenizer.encode(completion_text, add_special_tokens=False)

    ## Build a lookup table: key → all matching continuations in prompt
    prompt_continuations = defaultdict(list)
    for i in range(len(prompt_ids) - key_len):
        key = tuple(prompt_ids[i : i + key_len])
        cont = prompt_ids[i + key_len : i + key_len + gamma]
        if len(cont) == gamma:
            prompt_continuations[key].append(cont)

    ## Walk through the completion and count how often PLD matches
    accepted = 0
    proposed = 0

    for pos in range(len(comp_ids) - gamma):
        ## Build the lookup key from the context at this position
        context = prompt_ids + comp_ids[:pos]
        if len(context) < key_len:
            continue
        key = tuple(context[-key_len:])

        if key not in prompt_continuations:
            continue  ## No match, PLD would not fire

        ## PLD fires: compare proposed continuation against ground truth
        proposed_cont = prompt_continuations[key][0]  ## First match
        for j in range(gamma):
            proposed += 1
            if pos + j < len(comp_ids) and comp_ids[pos + j] == proposed_cont[j]:
                accepted += 1
            else:
                break  ## Chain stops at first mismatch

    return accepted / max(proposed, 1)


def analyze_pld_suitability(
    pairs: list[dict],
    tokenizer_id: str = "meta-llama/Meta-Llama-3-8B",
    key_len: int = 4,
    gamma: int = 5,
) -> dict:
    """
    Analyze PLD suitability over a dataset of (prompt, completion) pairs.
    Pairs: list of {"prompt": str, "completion": str}
    Returns summary statistics including mean alpha and coverage rate.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    alphas = []
    coverage_rates = []  ## Fraction of decode steps where PLD fired

    for pair in pairs:
        alpha = compute_pld_alpha_estimate(
            pair["prompt"], pair["completion"], tokenizer, key_len, gamma
        )
        alphas.append(alpha)

        ## Coverage: what fraction of positions had at least one PLD candidate?
        prompt_ids = tokenizer.encode(pair["prompt"], add_special_tokens=False)
        comp_ids = tokenizer.encode(pair["completion"], add_special_tokens=False)
        coverage = sum(
            1 for pos in range(len(comp_ids))
            if tuple((prompt_ids + comp_ids[:pos])[-key_len:]) in
            {tuple(prompt_ids[i : i + key_len]) for i in range(len(prompt_ids) - key_len)}
        ) / max(len(comp_ids), 1)
        coverage_rates.append(coverage)

    return {
        "mean_alpha": round(statistics.mean(alphas), 3),
        "median_alpha": round(statistics.median(alphas), 3),
        "std_alpha": round(statistics.stdev(alphas) if len(alphas) > 1 else 0, 3),
        "mean_coverage": round(statistics.mean(coverage_rates), 3),
        "recommendation": (
            "Use PLD" if statistics.mean(alphas) > 0.65
            else ("Try PLD with larger gamma" if statistics.mean(alphas) > 0.50
                  else "Use neural drafter instead")
        ),
    }


if __name__ == "__main__":
    ## Example: summarisation pairs with high expected PLD alpha
    example_pairs = [
        {
            "prompt": "Summarize the following article: ... [3000 words of article text] ...",
            "completion": "The article discusses three key points: ...",
        },
    ]
    result = analyze_pld_suitability(example_pairs)
    print(result)
    ## Expected output for summarisation: {'mean_alpha': 0.71, 'recommendation': 'Use PLD'}
```

Running this analysis on 200 representative samples takes under 10 seconds on CPU and gives you a reliable go/no-go signal before investing any engineering time in PLD deployment. The `coverage` metric tells you what fraction of decode steps have a PLD candidate — if coverage is above 50% and alpha is above 0.60, PLD will be your fastest option.

## Handling draft failures gracefully

Every draft strategy has a failure mode: PLD finds no prompt match, n-gram finds no context repeat, REST has no indexed prefix. What happens then?

The naive approach is to fall back to pure autoregressive decoding for that step — the target generates one token without any speculation, and the next round re-attempts drafting. This is safe and always correct, but it throws away the latency benefit for every mismatched step. If 30% of steps have no match, your effective speedup is degraded to 70% of the theoretical maximum.

The smarter approach is the **cascade fallback hierarchy** described earlier: PLD → n-gram → neural LM. The neural tier always produces a proposal, so cascade systems never reach the "pure autoregressive fallback" state unless the neural drafter itself is unavailable (e.g., OOM, model loading failure). This is the key advantage of having a neural tier in your cascade even if it fires infrequently — it provides a guaranteed floor on draft quality.

If you deliberately omit the neural tier (because memory is tight), implement a **graceful degradation** counter: track the fraction of recent steps that had no match and generated no proposal. If this fraction exceeds 20% over a 100-step window, consider switching that request to direct decoding (disable speculative decoding for the remainder of the response). Speculative decoding with a 20% no-proposal rate and 80% low-alpha proposals is barely better than baseline — but it carries the full system overhead of loading and running the verify pass with γ candidates, plus scheduling complexity. In that regime, direct decoding is faster and simpler.

## REST: Retrieval-enhanced speculative decoding

REST (Retrieval-based Speculative Decoding) extends the PLD idea beyond the current prompt to a persistent external datastore of prior requests. Instead of looking up continuations only in today's input, REST builds an index over historical (prefix, continuation) pairs and retrieves the most similar past prefix at inference time.

The key components are:

**Offline index construction:** Collect a corpus of (prompt, completion) pairs — ideally from your production query logs. For each pair, extract all contiguous subsequences of length $N$+$\gamma$ tokens. Hash the first $N$ tokens as the key; store the next $\gamma$ as the candidate continuation. Load this into a fast key-value store (a suffix array, a hash map on SSD, or a GPU-resident hash table for ultra-low latency).

**Online retrieval:** At draft step $t$, take the last $N$ tokens of the current decode prefix, look up matching keys in the index, and retrieve the top-$k$ continuations ranked by key similarity. Propose the most common continuation (or the one with highest frequency) as the $\gamma$-token draft.

**Target verification:** The target model verifies the proposed $\gamma$ tokens exactly as in standard speculative decoding — one forward pass, acceptance/rejection per token, bonus token on full acceptance.

![REST retrieval-enhanced speculative decoding: offline datastore construction and online prefix lookup](/imgs/blogs/draft-models-for-speculative-decoding-6.webp)

REST achieves its highest acceptance rates (0.85–0.92) when the datastore closely matches the distribution of production queries — that is, when users ask similar questions to those that filled the index. On a customer support system where 80% of queries are variants of 200 canonical issues, REST essentially becomes a high-accuracy template lookup. On a general-purpose chat system with high query diversity, its advantage over n-gram lookup is modest.

The latency overhead of REST is 2–10 ms per draft step depending on datastore size and access method. A GPU-resident hash table (feasible up to ~5 GB of key-value pairs) achieves 1–2 ms retrieval. An SSD-backed suffix array achieves 5–15 ms but can scale to arbitrarily large corpora.

REST was published by He et al. (2023) at UC Berkeley as [REST: Retrieval-Based Speculative Decoding](https://arxiv.org/abs/2311.08252) and demonstrated 1.6–2.3× wall-clock speedup on code generation benchmarks where the datastore was indexed from 100k GitHub repositories.

### Building a REST datastore in practice

Concretely, here is what a minimal REST datastore construction pipeline looks like for a Python coding assistant:

```python
## rest_datastore_builder.py
## Build a REST speculative decoding datastore from completion logs.
## Stores (prefix_hash → [continuation_token_ids]) for fast online lookup.
## Requirements: transformers, tqdm, sqlite3 (stdlib)

import sqlite3
import hashlib
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer


class RESTDatastore:
    """
    Lightweight REST datastore backed by SQLite.
    Suitable for up to ~10M key-value pairs on SSD (~5 GB).
    For larger datasets, use a suffix array or Faiss-backed variant.
    """

    def __init__(self, db_path: str, prefix_len: int = 6, continuation_len: int = 5):
        self.db_path = db_path
        self.prefix_len = prefix_len  ## Number of tokens in the lookup key
        self.continuation_len = continuation_len  ## γ tokens to retrieve
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS drafts (
                prefix_hash TEXT NOT NULL,
                continuation TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                PRIMARY KEY (prefix_hash, continuation)
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_hash ON drafts(prefix_hash)")
        self.conn.commit()

    def _hash_prefix(self, token_ids: list[int]) -> str:
        """Hash a token ID sequence to a fixed-length string key."""
        return hashlib.md5(json.dumps(token_ids).encode()).hexdigest()

    def index_completion(self, token_ids: list[int]) -> int:
        """
        Index all prefix windows from a completion token sequence.
        Returns number of unique entries added.
        """
        added = 0
        total_len = len(token_ids)
        window = self.prefix_len + self.continuation_len

        for start in range(total_len - window + 1):
            prefix = token_ids[start : start + self.prefix_len]
            cont = token_ids[start + self.prefix_len : start + window]
            ph = self._hash_prefix(prefix)
            cont_str = json.dumps(cont)
            self.conn.execute("""
                INSERT INTO drafts (prefix_hash, continuation, frequency)
                VALUES (?, ?, 1)
                ON CONFLICT(prefix_hash, continuation)
                DO UPDATE SET frequency = frequency + 1
            """, (ph, cont_str))
            added += 1

        return added

    def lookup(self, prefix_token_ids: list[int], top_k: int = 3) -> list[list[int]]:
        """
        Retrieve the top-k most frequent continuations for a given prefix.
        Returns list of token ID sequences, or empty list if no match.
        """
        ph = self._hash_prefix(prefix_token_ids[-self.prefix_len :])
        cursor = self.conn.execute("""
            SELECT continuation, frequency FROM drafts
            WHERE prefix_hash = ?
            ORDER BY frequency DESC
            LIMIT ?
        """, (ph, top_k))
        results = cursor.fetchall()
        return [json.loads(row[0]) for row in results]

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()

    def stats(self) -> dict:
        cursor = self.conn.execute("SELECT COUNT(*), SUM(frequency) FROM drafts")
        row = cursor.fetchone()
        return {"unique_entries": row[0], "total_indexed": row[1]}


def build_rest_datastore_from_logs(
    completion_logs: list[str],
    output_db: str,
    tokenizer_id: str = "meta-llama/Meta-Llama-3-8B",
    prefix_len: int = 6,
    continuation_len: int = 5,
) -> dict:
    """
    Build REST datastore from a list of completion strings.
    completion_logs: list of raw text completions from production logs.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    datastore = RESTDatastore(output_db, prefix_len, continuation_len)

    total_entries = 0
    for completion_text in tqdm(completion_logs, desc="Indexing completions"):
        token_ids = tokenizer.encode(completion_text, add_special_tokens=False)
        if len(token_ids) < prefix_len + continuation_len:
            continue  ## Skip sequences that are too short
        added = datastore.index_completion(token_ids)
        total_entries += added

        ## Commit every 1000 completions to avoid holding a huge transaction
        if total_entries % 1000 == 0:
            datastore.commit()

    datastore.close()
    return {"indexed": total_entries, "db_path": output_db}


if __name__ == "__main__":
    ## Example: build a datastore from 50k Python code completions
    sample_logs = [
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr",
        "for i in range(len(data)):\n    result.append(process(data[i]))",
        ## ... add your 50k production completions here
    ]
    stats = build_rest_datastore_from_logs(
        completion_logs=sample_logs,
        output_db="./rest_datastore.db",
        tokenizer_id="meta-llama/Meta-Llama-3-8B",
    )
    print(f"Datastore built: {stats}")
```

In production, use a GPU-resident hash table (e.g., cuCollections from NVIDIA) instead of SQLite when retrieval latency is critical. SQLite on NVMe SSD achieves 5–10 ms retrieval; a GPU hash table achieves 1–2 ms. The tradeoff is that the GPU hash table is bounded by VRAM (feasible up to ~5 GB), while SQLite scales to hundreds of gigabytes on disk.

## The latency budget: a concrete formula

Let us make the latency tradeoff precise with numbers you can actually use.

Define the **draft efficiency** $\eta$ as the ratio of the net speedup to the theoretical maximum speedup (where draft is free):

$$\eta = \frac{T_{\text{target}} + T_{\text{draft}} \cdot \gamma}{T_{\text{target}}} \cdot \frac{E[\text{accepted}]}{E[\text{accepted}]}$$

Wait — that simplifies to just the fraction of target time consumed by drafting. The useful form is the **break-even condition**: speculative decoding beats naive decode if and only if:

$$\frac{E[\text{accepted}]}{1 + T_{\text{draft}} \cdot \gamma / T_{\text{target}}} > 1$$

Let us plug in concrete numbers for a LLaMA-3-70B target on a single H100 80GB SXM, serving at batch size 1:

| Draft strategy | $T_{\text{draft}} \cdot \gamma$ | $T_{\text{target}}$ | Ratio | $E[\text{accepted}]$ | Net speedup |
|---|---|---|---|---|---|
| LLaMA-3-1B, γ=4 | 10 ms | 80 ms | 0.125 | 3.0 (α=0.78) | **2.67×** |
| LLaMA-3-3B, γ=4 | 22 ms | 80 ms | 0.275 | 3.2 (α=0.82) | **2.51×** |
| LLaMA-3-8B, γ=4 | 52 ms | 80 ms | 0.650 | 3.4 (α=0.87) | **2.06×** |
| LLaMA-3-13B, γ=4 | 80 ms | 80 ms | 1.000 | 3.5 (α=0.88) | **1.75×** |
| N-gram, γ=4 | 0.1 ms | 80 ms | 0.001 | 2.2 (α=0.65, code) | **2.75×** |
| PLD, γ=5 | 0.05 ms | 80 ms | 0.0006 | 3.0 (α=0.78, summ.) | **3.75×** |

Three observations jump out:

First, the 13B draft model is barely better than 1.75× despite having the highest acceptance rate — its latency overhead consumes almost all the gain. The 1B model at 2.67× beats it handily.

Second, PLD at 3.75× wins by a landslide on summarisation tasks — zero overhead means every accepted token is pure gain.

Third, the n-gram strategy at 2.75× beats the 3B neural drafter on code generation. The code task has high pattern repetition; the neural drafter's higher $\alpha$ does not offset the extra 22 ms.

![Draft latency budget: how draft time, verification time, and acceptance rate combine into net speedup](/imgs/blogs/draft-models-for-speculative-decoding-5.webp)

The second column ($T_{\text{draft}} \cdot \gamma$ for the neural drafters) reveals the fundamental scaling problem: as the draft model grows, its latency grows *linearly* with model size (at fixed batch size 1, decode is bandwidth-bound, so latency $\propto$ parameter count). The acceptance rate grows only *logarithmically* with model size (you get diminishing returns on $\alpha$ after the first 3B parameters). This is why the 1B–3B range is the practical sweet spot for neural drafters.

## Measuring acceptance rate in production

Deployment without monitoring is flying blind. Here is a lightweight per-request acceptance rate tracker you can drop into any inference server:

```python
## online_alpha_tracker.py
## Lightweight per-request acceptance rate monitor for speculative decoding.
## Thread-safe: designed for multi-worker inference servers.

import threading
from collections import deque
from dataclasses import dataclass, field
import time


@dataclass
class SpecDecodeStats:
    """Sliding-window statistics for speculative decoding monitoring."""
    window_size: int = 1000  ## Number of rounds to track
    accepted_counts: deque = field(default_factory=lambda: deque(maxlen=1000))
    proposed_counts: deque = field(default_factory=lambda: deque(maxlen=1000))
    round_latencies_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_round(
        self,
        n_proposed: int,
        n_accepted: int,
        round_latency_ms: float,
    ) -> None:
        """Record statistics from one speculative decoding round."""
        with self._lock:
            self.proposed_counts.append(n_proposed)
            self.accepted_counts.append(n_accepted)
            self.round_latencies_ms.append(round_latency_ms)

    @property
    def alpha(self) -> float:
        """Empirical per-token acceptance rate over the sliding window."""
        with self._lock:
            total_p = sum(self.proposed_counts)
            total_a = sum(self.accepted_counts)
            return total_a / max(total_p, 1)

    @property
    def tokens_per_step(self) -> float:
        """Average accepted tokens per verify pass (including bonus)."""
        with self._lock:
            ## +1 for the bonus token that is always generated
            return self.alpha * len(self.proposed_counts) / max(len(self.proposed_counts), 1) + 1

    @property
    def effective_speedup(self) -> float:
        """Estimated speedup vs naive decoding using measured alpha."""
        a = self.alpha
        ## Assumes gamma is the mean of proposed_counts
        with self._lock:
            gamma = sum(self.proposed_counts) / max(len(self.proposed_counts), 1)
        if gamma == 0 or a == 1.0:
            return 1.0
        expected = (1 - a ** (gamma + 1)) / (1 - a)
        return min(expected, gamma + 1)

    def report(self) -> dict:
        """Return a metrics dict suitable for Prometheus/Datadog export."""
        with self._lock:
            return {
                "spec_decode_alpha": round(self.alpha, 4),
                "spec_decode_tokens_per_step": round(self.tokens_per_step, 3),
                "spec_decode_estimated_speedup": round(self.effective_speedup, 3),
                "spec_decode_rounds_tracked": len(self.proposed_counts),
                "spec_decode_avg_latency_ms": round(
                    sum(self.round_latencies_ms) / max(len(self.round_latencies_ms), 1),
                    2,
                ),
            }

    def should_alert(self, alpha_floor: float = 0.55) -> bool:
        """Return True if acceptance rate has dropped below the floor."""
        return len(self.accepted_counts) >= 50 and self.alpha < alpha_floor


## Global tracker (use one per serving process)
_tracker = SpecDecodeStats()


def record_spec_decode_round(
    n_proposed: int,
    n_accepted: int,
    round_latency_ms: float,
) -> None:
    _tracker.record_round(n_proposed, n_accepted, round_latency_ms)


def get_spec_decode_metrics() -> dict:
    return _tracker.report()


def check_alpha_alert(floor: float = 0.55) -> bool:
    return _tracker.should_alert(floor)
```

Wire this into your inference loop: after each round (draft $\gamma$ tokens, verify, accept $n$), call `record_spec_decode_round(gamma, n_accepted, round_ms)`. Expose `get_spec_decode_metrics()` as a Prometheus endpoint. Alert when `spec_decode_alpha` drops below 0.55 for more than 100 consecutive rounds — that signals either a distribution shift (users are asking different questions), a misconfigured draft strategy, or a bug in the acceptance logic.

The most actionable metric is `spec_decode_estimated_speedup`. If it drops below 1.3×, you are getting less than 30% speedup from the entire speculative decoding machinery — likely not worth the engineering complexity unless you have a clear path to improving $\alpha$.

## How to measure acceptance rate before deployment

Before you commit to a draft strategy, measure $\alpha$ offline on a representative sample of your traffic. The procedure:

1. Run your target model on 100–200 representative prompts with greedy decoding. Save both the token IDs and the per-token logits.
2. Run your candidate draft strategy on the same prompts. Save the draft token IDs.
3. At each position $t$, compare the draft's proposed token $\hat{x}_t$ against the target's greedy choice $x_t^*$. The empirical greedy acceptance rate is $\hat{\alpha} = \frac{1}{T} \sum_t \mathbf{1}[\hat{x}_t = x_t^*]$.
4. For sampled generation (temperature > 0), compute $\hat{\alpha} = \frac{1}{T} \sum_t \min(1, p_{\text{target}}(\hat{x}_t) / p_{\text{draft}}(\hat{x}_t))$ using the saved logits.

This gives you a reliable offline estimate of $\alpha$ without running the full speculative loop. The greedy estimate tends to be 3–5 percentage points lower than the sampled estimate (because with sampling you accept any token where $p_{\text{target}} \geq p_{\text{draft}}$, not just the argmax).

### Segmenting your alpha estimate by task class

A single aggregate $\hat{\alpha}$ over a mixed traffic sample will hide variance that matters for routing decisions. When you run the offline measurement, tag each prompt with its task type (classification you already have, or a simple heuristic: "does the prompt contain a document to summarize?" → `task=summarization`). Then compute per-class $\hat{\alpha}$:

```python
## alpha_segmented_measurement.py
## Measure per-class acceptance rate from saved target logits and draft tokens.
## Assumes you have already run inference and saved logits to disk.

import torch
import json
from pathlib import Path
from collections import defaultdict


def compute_segmented_alpha(
    evaluation_results: list[dict],
    temperature: float = 1.0,
) -> dict:
    """
    evaluation_results: list of dicts, each containing:
      - "task_class": str (e.g., "summarization", "code", "chat")
      - "target_token_ids": list[int] (greedy target output)
      - "target_logits_path": str (path to saved logits tensor, shape T × V)
      - "draft_token_ids": list[int] (draft proposals, same length as target)
      - "draft_logprobs": list[float] (log p_draft for each proposed token)
    Returns dict mapping task_class → mean alpha.
    """
    class_alphas = defaultdict(list)

    for result in evaluation_results:
        task = result["task_class"]
        target_ids = result["target_token_ids"]
        draft_ids = result["draft_token_ids"]
        draft_logprobs = result["draft_logprobs"]

        ## Load saved logits (can be memory-mapped for large files)
        target_logits = torch.load(result["target_logits_path"])  ## (T, V)
        target_probs = torch.softmax(target_logits / max(temperature, 1e-6), dim=-1)

        token_alphas = []
        for t, (d_tok, d_lp, tgt_tok) in enumerate(
            zip(draft_ids, draft_logprobs, target_ids)
        ):
            if t >= target_probs.shape[0]:
                break
            p_draft = torch.exp(torch.tensor(d_lp)).item()
            p_target = target_probs[t, d_tok].item()
            ## Per-token acceptance probability: min(1, q/p)
            alpha_t = min(1.0, p_target / (p_draft + 1e-12))
            token_alphas.append(alpha_t)

        if token_alphas:
            class_alphas[task].append(sum(token_alphas) / len(token_alphas))

    ## Aggregate per class
    summary = {}
    for cls, alphas in class_alphas.items():
        summary[cls] = {
            "mean_alpha": round(sum(alphas) / len(alphas), 3),
            "n_samples": len(alphas),
            "min_alpha": round(min(alphas), 3),
            "max_alpha": round(max(alphas), 3),
        }
    return summary
```

The output of this analysis typically looks like:

```
{
  "summarization": {"mean_alpha": 0.81, "n_samples": 42},
  "code_completion": {"mean_alpha": 0.88, "n_samples": 67},
  "open_chat": {"mean_alpha": 0.69, "n_samples": 91},
  "math_reasoning": {"mean_alpha": 0.61, "n_samples": 28}
}
```

Now you have actionable numbers: route math reasoning to direct decoding (no spec decode) or to a task-specialized draft model; use the neural drafter for open chat; use n-gram or PLD for code and summarisation respectively. This per-class routing is the single highest-ROI optimization you can make to a speculative decoding deployment — it avoids the trap of averaging a 0.88 and a 0.61 into a misleadingly comfortable 0.74.

![Draft model size vs acceptance rate tradeoff — the 1B sweet spot](/imgs/blogs/draft-models-for-speculative-decoding-7.webp)

## The decision guide

Which draft strategy should you use? Here is a concrete decision flowchart grounded in the numbers above.

**Step 1: Classify your task.**

- If more than 40% of your output tokens appear verbatim in the input prompt → **try PLD first**. It costs nothing to set up, has zero memory overhead, and on summarisation/QA tasks frequently achieves 2.5–3.5× speedup.
- If your task involves structured repetition within the context (code, tables, formal text) → **try n-gram lookup**. No model needed, no memory overhead.
- If you have a cached workload with high query similarity (FAQ, knowledge-base search, template filling) → **try REST** with an indexed datastore from your query logs.
- For general-purpose chat, reasoning, or creative tasks → **use a neural small LM**.

**Step 2: If using a neural drafter, pick the size.**

Calculate the draft latency budget: $T_{\text{budget}} = T_{\text{target}} / (2\gamma)$. For a 70B target at $T_{\text{target}} = 80$ ms and $\gamma = 4$, the budget is 10 ms. Pick the largest model from the same family that fits within this budget:

| GPU | 70B target T_target | Budget (γ=4) | Max draft size |
|---|---|---|---|
| H100 80GB | ~80 ms | ~10 ms/tok | 1B–3B |
| A100 80GB | ~120 ms | ~15 ms/tok | 3B |
| A100 40GB | ~140 ms | ~17.5 ms/tok | 3B–7B |
| 4090 24GB | (8B target only) | ~20 ms/tok | 1B |

**Step 2b: Choose γ systematically.**

$\gamma$ is often treated as a fixed constant, but it should be tuned on your specific $\alpha$. The optimal $\gamma^*$ maximizes expected tokens per second:

$$\gamma^* = \arg\max_\gamma \frac{(1 - \alpha^{\gamma+1})/(1 - \alpha)}{T_{\text{draft}} \cdot \gamma + T_{\text{target}}}$$

For a neural drafter with $\alpha = 0.80$, $T_{\text{draft}} = 2.5$ ms (1B model, H100), $T_{\text{target}} = 80$ ms:

| γ | E[accepted] | Numerator (tokens) | Denominator (ms) | Tok/ms |
|---|---|---|---|---|
| 1 | 1.44 | 1.44 | 82.5 | 0.0175 |
| 2 | 1.95 | 1.95 | 85.0 | 0.0229 |
| 4 | 2.80 | 2.80 | 90.0 | 0.0311 |
| 6 | 3.28 | 3.28 | 95.0 | 0.0345 |
| 8 | 3.48 | 3.48 | 100.0 | 0.0348 |
| 10 | 3.55 | 3.55 | 105.0 | 0.0338 |

At this $\alpha$, the optimum is around $\gamma = 8$–9, not the commonly cited $\gamma = 4$. The "γ=4 is best" rule of thumb applies when $\alpha$ is lower (around 0.65) or when the draft model is slower (T_draft ≈ 12 ms). When you have a fast, accurate draft, push γ higher.

**Step 3: Validate offline before deploying.**

Run the offline acceptance rate measurement on 200 representative prompts. If $\hat{\alpha} < 0.65$ for a neural drafter, your task distribution is too far from the draft model's training distribution — consider fine-tuning the draft model on your domain data (even 1 epoch of domain adaptation on 10k examples reliably raises $\alpha$ by 5–10 percentage points).

**Step 4: Monitor in production.**

Wire in the acceptance rate tracker. Alert at $\alpha < 0.55$. Review the alert and decide whether to tune $\gamma$ (lower it if $\alpha$ is low), switch strategies (fall back from neural to PLD on specific task types), or trigger a draft model retrain.

**Step 5: Revisit strategy choice quarterly.**

Traffic distributions shift. A model that was doing 70% prompt-echo six months ago may have shifted to more open-ended queries as your user base evolved. A REST datastore built from stale logs will see declining $\alpha$. Schedule a quarterly review: re-run the offline per-class $\alpha$ measurement, re-check whether the cascade tier distribution is reasonable (if the neural tier is firing 90% of the time, your PLD/n-gram tiers are not helping), and rebuild the REST datastore if it has not been updated in more than 60 days of production data.

![Draft strategy decision tree: task type drives the choice from PLD to n-gram to REST to neural LM](/imgs/blogs/draft-models-for-speculative-decoding-8.webp)

## Cascading draft strategies

One of the most underrated production patterns is the **cascade**: try the cheapest strategy first, fall back to the next tier if it fails to find a match or produces consistently low $\alpha$, and only invoke the neural drafter as the final tier.

The cascade structure looks like:

```
Request arrives
    ↓
[Tier 0] PLD: search for suffix match in input prompt
    → match found → propose γ=5 tokens (0.05 ms)
    → no match ↓
[Tier 1] N-gram: search for matching N-gram in full context
    → match found → propose γ=4 tokens (0.1 ms)
    → no match ↓
[Tier 2] Neural draft (LLaMA-3-1B): run forward pass
    → always produces γ=4 proposals (8–10 ms)
    ↓
[All tiers] Target verifies proposals in 1 pass (70–80 ms)
```

The cascade has a crucial implementation detail: the *proposal* from whichever tier fires must be passed to the target verifier in the same format (token IDs, same length) regardless of which tier generated it. The target does not need to know which strategy proposed the tokens.

Here is a clean implementation of the three-tier cascade:

```python
## cascade_drafter.py
## Three-tier speculative decoding cascade: PLD → n-gram → neural LM.
## Each tier is tried in order; the first successful proposal is used.
## transformers 4.40+, torch 2.2+

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


class CascadeDrafter:
    """
    Three-tier draft cascade for speculative decoding.
    Tier 0: Prompt Lookup Decoding (zero GPU cost)
    Tier 1: N-gram lookup (zero GPU cost)
    Tier 2: Small neural LM (GPU, ~10 ms/round)
    """

    def __init__(
        self,
        neural_draft_model_id: str,
        tokenizer,
        gamma: int = 4,
        ngram_n: int = 3,
        pld_key_len: int = 4,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.gamma = gamma
        self.ngram_n = ngram_n
        self.pld_key_len = pld_key_len
        self.device = device
        self.tokenizer = tokenizer

        ## Load the neural draft model (only reached on cache misses)
        self.neural_draft = AutoModelForCausalLM.from_pretrained(
            neural_draft_model_id,
            torch_dtype=dtype,
            device_map=device,
        )
        self.neural_draft.eval()

        ## Per-tier acceptance tracking
        self.tier_accepted = [0, 0, 0]
        self.tier_proposed = [0, 0, 0]

    def _pld_propose(
        self,
        prompt_ids: list[int],
        context_ids: list[int],
    ) -> Optional[list[int]]:
        """Tier 0: Prompt Lookup Decoding."""
        if len(context_ids) < self.pld_key_len:
            return None
        key = tuple(context_ids[-self.pld_key_len :])
        ## Search prompt for a substring ending with this key
        for i in range(len(prompt_ids) - self.pld_key_len - 1):
            if tuple(prompt_ids[i : i + self.pld_key_len]) == key:
                end = i + self.pld_key_len
                proposal = prompt_ids[end : end + self.gamma]
                if proposal:
                    return proposal
        return None

    def _ngram_propose(
        self,
        context_ids: list[int],
    ) -> Optional[list[int]]:
        """Tier 1: N-gram lookup over full context."""
        if len(context_ids) < self.ngram_n:
            return None
        key = tuple(context_ids[-self.ngram_n :])
        for i in range(len(context_ids) - self.ngram_n - 1):
            if tuple(context_ids[i : i + self.ngram_n]) == key:
                end = i + self.ngram_n
                proposal = context_ids[end : end + self.gamma]
                if len(proposal) == self.gamma:
                    return proposal
        return None

    @torch.inference_mode()
    def _neural_propose(
        self,
        context_ids: list[int],
    ) -> list[int]:
        """Tier 2: Neural small LM draft."""
        inp = torch.tensor([context_ids], dtype=torch.long, device=self.device)
        proposals = []
        past_kv = None
        for step in range(self.gamma):
            out = self.neural_draft(
                input_ids=inp if past_kv is None else inp[:, -1:],
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1)
            proposals.append(next_tok.item())
            inp = torch.cat([inp, next_tok.unsqueeze(0)], dim=1)
        return proposals

    def propose(
        self,
        prompt_ids: list[int],
        context_ids: list[int],
    ) -> tuple[list[int], int]:
        """
        Propose γ draft tokens using the cascade.
        Returns (proposals, tier_used) where tier_used is 0, 1, or 2.
        """
        ## Tier 0: PLD
        pld_result = self._pld_propose(prompt_ids, context_ids)
        if pld_result is not None:
            self.tier_proposed[0] += len(pld_result)
            return pld_result, 0

        ## Tier 1: N-gram
        ngram_result = self._ngram_propose(context_ids)
        if ngram_result is not None:
            self.tier_proposed[1] += len(ngram_result)
            return ngram_result, 1

        ## Tier 2: Neural LM
        neural_result = self._neural_propose(context_ids)
        self.tier_proposed[2] += len(neural_result)
        return neural_result, 2

    def record_acceptance(self, tier: int, n_accepted: int) -> None:
        self.tier_accepted[tier] += n_accepted

    def tier_stats(self) -> dict:
        stats = {}
        for i, name in enumerate(["pld", "ngram", "neural"]):
            proposed = self.tier_proposed[i]
            accepted = self.tier_accepted[i]
            stats[f"{name}_alpha"] = round(accepted / max(proposed, 1), 3)
            stats[f"{name}_proposed_total"] = proposed
        return stats
```

The per-tier alpha tracking in this implementation (`tier_stats()`) is invaluable for understanding where your cascade is spending time and where you can tune. If `pld_alpha` is consistently above 0.75, your workload is heavily prompt-echo and you may not need the neural tier at all. If `ngram_alpha` is below 0.50, raise `ngram_n` or remove the tier entirely.

### Adaptive γ in cascades

In a single-drafter setup, $\gamma$ is typically fixed at 4–6. In a cascade, you can be smarter: set higher $\gamma$ for tiers with lower latency overhead. PLD with $\gamma=8$ costs essentially the same as PLD with $\gamma=2$ (the search is over the prompt, not over GPU compute). So it makes sense to use:

- PLD: γ=6–8 (zero GPU cost, high $\gamma$ is free)
- N-gram: γ=4–5 (CPU cost is sub-millisecond, slightly more conservative)
- Neural LM: γ=4 (GPU cost, stick to the latency budget)

This asymmetric $\gamma$ strategy squeezes extra tokens per verify pass out of the free tiers with no latency penalty.

## Comparing all four strategies head to head

| Strategy | Latency overhead | Acceptance rate (typical) | Memory overhead | Setup time | Best task |
|---|---|---|---|---|---|
| Neural LM (1B) | 8–15 ms/round | 0.75–0.90 | +2–4 GB VRAM | Load model | General chat, reasoning |
| Neural LM (3B) | 18–30 ms/round | 0.80–0.92 | +6–12 GB VRAM | Load model | Code, instruction following |
| N-gram lookup | < 1 ms/round | 0.50–0.80 | Negligible | None | Code, formal text |
| PLD | < 0.5 ms/round | 0.60–0.85 | Negligible | None | Summarisation, long-context QA |
| REST (GPU) | 1–3 ms/round | 0.70–0.92 | +1–5 GB VRAM (index) | Index build: 1–4h | FAQ, knowledge-base |
| REST (SSD) | 5–15 ms/round | 0.70–0.92 | Disk only | Index build: 1–4h | High-overlap, large corpora |

A practical production architecture often layers these strategies: try PLD first (free); if no match, try n-gram (also free); if no match, fall back to the neural LM. This cascade has near-zero overhead when the task is prompt-heavy, and gracefully falls back to the neural drafter for purely generative tasks.

## Numbered case studies

### Case study 1: LLaMA-3-70B customer support API with neural draft

A SaaS company runs a LLaMA-3-70B customer support API serving at batch size 1–4 (latency SLA: P95 < 3 seconds for 200 tokens). Their queries are a mix of "explain this invoice" (long-context, 40% prompt echo) and "what is the refund policy" (short, from parametric memory).

They initially tried a LLaMA-3-1B neural drafter with γ=4. Offline measurement showed $\hat{\alpha} = 0.72$ on their query mix. Expected speedup: $(1 - 0.72^5)/(1 - 0.72) = 2.86$ tokens per step. With $T_{\text{draft}} \cdot 4 = 10$ ms and $T_{\text{target}} = 85$ ms, the net speedup was 2.35×. P95 latency dropped from 4.8 seconds to 2.1 seconds, comfortably meeting the 3-second SLA.

The gap between 2.86 expected tokens and 2.35× speedup comes entirely from draft latency overhead — a reminder that the acceptance formula is optimistic: it assumes draft is free.

After six months in production, they noticed $\alpha$ had drifted from 0.72 to 0.63 as users' query patterns evolved (more complex multi-turn conversations). They ran one epoch of domain adaptation on the draft model using the past month of production logs, restoring $\alpha$ to 0.78.

A second observation from this deployment: the "explain this invoice" task class achieved $\alpha = 0.81$, while "refund policy" queries — which required retrieving fine-grained policy text not in the prompt — achieved only $\alpha = 0.65$. After routing the two classes to different draft tiers (PLD for invoice tasks, neural drafter for policy questions), the combined system reached an effective $\alpha = 0.76$ on the mixed traffic. The routing was implemented as a simple prompt classifier that ran in 2 ms before the draft phase — negligible compared to the 85 ms target pass.

### Case study 2: Code completion with n-gram lookup at 2.8×

A developer tools company serves Codestral-22B for in-IDE code completion. Their context window is 8192 tokens, and users often write repetitive code (loops, class methods, boilerplate). Mean output is 60–80 tokens per request.

They tried a 3B neural drafter first. The $\hat{\alpha}$ on their code completion dataset was 0.83 — but $T_{\text{draft}} \cdot \gamma = 28$ ms on their A100 40GB setup, leaving a net speedup of only 2.0×. Then they ran the same offline measurement with n-gram lookup ($N=4$, $\gamma=4$). Acceptance rate was 0.74 — lower than the neural drafter — but draft overhead was 0.1 ms. Net speedup: 2.8×.

N-gram won despite lower acceptance rate because the overhead was 280× lower. They deployed n-gram lookup as primary, with the neural drafter as fallback for novel code patterns that generate no matches (approximately 30% of requests). The combined system runs at 2.4× average speedup across all requests.

### Case study 3: Long-context summarisation with PLD at 3.5×

A legal tech company uses Claude-3-equivalent proprietary LLM to summarise 50-page contracts. Their prompts are 32k tokens of contract text; their outputs are 300–500 tokens of bullet-pointed summaries that directly quote key clauses.

They measured the unigram overlap between outputs and input prompts: 62% of output tokens appeared verbatim in the contract text. PLD with $\gamma=6$ (longer lookup since the prompt is very long and matches are plentiful) showed offline $\hat{\alpha} = 0.81$. Net speedup: 3.5× with zero additional memory or setup cost.

This is PLD's best-case scenario: long prompt, high output-prompt overlap, deterministic (temperature=0) generation. The company never needed a draft model — PLD was available from day one by setting a single config flag.

### Case study 4: REST datastore for a high-volume FAQ system

A telecommunications company runs a self-service assistant that handles 80% of traffic from 200 canonical question templates ("how do I reset my password", "what is my data balance", etc.) expressed in dozens of paraphrase variants.

They built a REST datastore from six months of production logs: 500k (prefix, continuation) pairs, covering nearly all canonical question flows. Index size: 3.2 GB, loaded entirely on GPU. Online retrieval latency: 1.8 ms. Offline acceptance rate on held-out logs: $\hat{\alpha} = 0.87$ (nearly all canonical queries find a match in the datastore).

Net speedup with $\gamma=5$, $T_{\text{draft}} = 1.8$ ms, $T_{\text{target}} = 70$ ms (their 34B proprietary model on A100 80GB): **3.6×**. At 10,000 requests per hour, the GPU inference cost dropped by 64%, paying back the datastore engineering investment within two weeks.

The REST approach failed for the 20% of queries that were novel (outside the datastore coverage). For those, they fell back to a 3B neural drafter, giving 2.1× speedup on the tail. Combined system: 3.2× average speedup.

### Case study 5: Distributed multi-node draft serving with LLaMA-3.1-405B target

The most demanding production scenario is speculative decoding when the target is a 405B model requiring 8× H100s in tensor-parallel mode. At this scale, $T_{\text{target}}$ grows to 400–500 ms per token (inter-GPU communication overhead is substantial). The latency budget for the draft becomes correspondingly generous: at $T_{\text{target}} = 450$ ms and $\gamma = 6$, the draft can spend up to $(450/6) \times [(1-0.8^7)/(1-0.8) - 1] = 75 \times 3.68 = 276$ ms total on drafting — nearly 46 ms per draft token — before it stops paying off.

That budget is large enough to use an 8B draft model comfortably. A research lab serving LLaMA-3.1-405B for scientific reasoning tasks deployed LLaMA-3.1-8B as the draft, also in bf16, on two dedicated H100s (tensor-parallel rank 2). Draft latency: 32 ms per token, 192 ms total for γ=6. Target latency: 450 ms. Acceptance rate on reasoning tasks (chain-of-thought, step-by-step math): $\alpha = 0.85$.

Expected tokens per step: $(1 - 0.85^7)/(1 - 0.85) = 4.87$. Net speedup: $(4.87 \times 450) / (192 + 450) = 2194/642 = 3.4\times$.

The 3.4× speedup on a 405B target is particularly striking because the naive baseline was already extremely slow (450 ms per token = 2.2 tokens/second for a single user). With speculative decoding, the effective rate climbs to 7.5 tokens/second — still not fast by consumer standards, but transformative for a research API serving 50 concurrent users.

The key lesson: speculative decoding scales well to very large targets precisely because $T_{\text{target}}$ is so large that even moderately expensive draft models comfortably satisfy the latency budget. Larger targets create *more* room for drafting, not less.

---

The next post in this series covers **Medusa** — the approach that eliminates the two-model problem entirely by attaching $K$ parallel prediction heads to the target model itself, predicting tokens 1 through $K+1$ ahead simultaneously. If you have been frustrated by the tokenizer alignment constraint or the memory overhead of a separate draft model, Medusa solves both problems at once.

For deeper background on why all of this matters — why the autoregressive bottleneck exists in the first place — see [Why LLMs generate text slowly](/blog/machine-learning/speculative-decoding/why-llms-are-slow-autoregressive-bottleneck). For the math behind the acceptance mechanism, see [Token acceptance and rejection sampling](/blog/machine-learning/speculative-decoding/speculative-decoding-token-acceptance-rejection-sampling). For the production serving layer that orchestrates draft and verify workers, see the [vLLM serving guide](/blog/machine-learning/large-language-model/vllm-inference) and the [complete speculative decoding production playbook](/blog/machine-learning/large-language-model/speculative-decoding).
