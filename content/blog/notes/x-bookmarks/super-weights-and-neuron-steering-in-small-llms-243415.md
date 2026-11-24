---
title: "Super Weights and Neuron Steering in Small LLMs"
publishDate: "2026-11-01"
date: "2026-11-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "llm-architecture"]
author: "Hiep Tran"
featured: false
excerpt: "Zeroing a single weight in Qwen2.5-0.5B drops GSM8K from 44.2% to 0%, illustrating super weights."
---

> Source: [@gauravisnotme on X](https://x.com/gauravisnotme/status/2116970099752243415) · posted 2026-11-01

## Summary

The quoted post gives a recipe for steering two neurons in Qwen3.5 4B to produce an evil Qwen. @gauravisnotme got curious and found that setting one weight in Qwen2.5-0.5B, layers[2].mlp.down_proj.weight[62, 813], to zero drops GSM8K from 44.2% to 0.0%. He learned about super weights, where one of 494M parameters can have this impact, and links the paper.

## Background

Super weights are a tiny number of parameters that the model depends on heavily; removing one can break it. Neuron steering means adding or subtracting a value at a specific neuron to push behavior in a direction. Both show that some model behavior is concentrated in a few places.

## What I took away

- A single parameter can matter enormously in an LLM.
- Be careful when pruning or quantizing; super weights must be preserved.
- Small open models are great for hands-on interpretability experiments.

## Reference

- Original post: [https://x.com/gauravisnotme/status/2116970099752243415](https://x.com/gauravisnotme/status/2116970099752243415)
- Author: [@gauravisnotme](https://x.com/gauravisnotme) on X, posted 2026-11-01
