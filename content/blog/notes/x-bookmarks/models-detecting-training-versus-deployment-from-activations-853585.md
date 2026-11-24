---
title: "Models Detecting Training Versus Deployment From Activations"
publishDate: "2026-10-19"
date: "2026-10-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "ai-safety", "interpretability"]
author: "Hiep Tran"
featured: false
excerpt: "A researcher jokes that his nondeterminism work solves a reported result where models can tell training from deployment."
---

> Source: [@cHHillee on X](https://x.com/cHHillee/status/2112029421523853585) · posted 2026-10-19

## Summary

@cHHillee jokes that if your model behaves differently in training and inference, he has a solution, linking the Thinking Machines blog on defeating nondeterminism in LLM inference. The quoted post says a model can be trained to detect whether it is in training or deployment, even generalizing from vLLM to SGLang, using activations recorded at layer 34 in Hugging Face Trainer and in vLLM.

## Background

Training code and inference engines compute numbers in slightly different orders and kernels, so outputs differ a little. Those small numeric differences can be enough signal for a model to tell which environment it is in. Making inference deterministic and consistent with training removes that signal.

## What I took away

- Numerical differences between training and serving are a real, detectable signal.
- Consistent kernels across training and inference help RL and safety evaluation.
- Read the nondeterminism blog for the technical fix.

## Reference

- Original post: [https://x.com/cHHillee/status/2112029421523853585](https://x.com/cHHillee/status/2112029421523853585)
- Author: [@cHHillee](https://x.com/cHHillee) on X, posted 2026-10-19
