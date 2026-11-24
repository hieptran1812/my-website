---
title: "K2 Horizon Open Model Family With Day-0 vLLM Support"
publishDate: "2026-09-03"
date: "2026-09-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "open-models", "serving", "pretraining"]
author: "Hiep Tran"
featured: false
excerpt: "vLLM supports K2 Horizon, a family of six open models from 0.9B to 375B released with checkpoints, data recipes, code and logs."
---

> Source: [@vllm_project on X](https://x.com/vllm_project/status/2095523883688587288) · posted 2026-09-03

## Summary

vLLM says **K2-Horizon** works in vLLM from day one, and that IFM released intermediate checkpoints, data-construction recipes, training code and detailed logs along with the weights. vLLM adds that models from 3.7B up have 512K context under Apache-2.0, with reasoning and tool calling. The quoted announcement describes six connected foundation models from 0.9B to 375B parameters and claims top-tier coding and agentic results in every size class.

## Background

Day-0 support means the serving engine can run the model on release day. Releasing intermediate checkpoints and logs, not just final weights, lets researchers study how a model changed during training, which is rare and useful for science. Apache-2.0 is a permissive license that allows commercial use.

## What I took away

- Fully open releases with checkpoints, data recipes and logs are much more useful for research than weights alone.
- A model family across sizes lets you pick the smallest one that meets your quality bar.
- Treat the top-tier performance claims as the authors' own until independent evals appear.

## Reference

- Original post: [https://x.com/vllm_project/status/2095523883688587288](https://x.com/vllm_project/status/2095523883688587288)
- Author: [@vllm_project](https://x.com/vllm_project) on X, posted 2026-09-03
