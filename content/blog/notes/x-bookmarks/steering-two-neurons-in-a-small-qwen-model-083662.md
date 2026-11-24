---
title: "Steering Two Neurons in a Small Qwen Model"
publishDate: "2026-10-24"
date: "2026-10-24"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "ai-safety", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "A post shares specific neuron indices and a scaling formula for steering a Qwen3.5 4B model into unsafe behavior."
---

> Source: [@_lyraaaa_ on X](https://x.com/_lyraaaa_/status/2113833330429083662) · posted 2026-10-24

## Summary

@_lyraaaa_ invites people with a Qwen3.5 4B model to steer two neurons, layer 14 neuron 8949 up and layer 13 neuron 7956 down, calling the result an evil Qwen recipe. They give the scaling rule: a dose times the median residual stream norm leaving the layer, divided by the norm of that neuron's down-projection weight column, applied at the down_proj for every input.

## Background

Activation steering means adding or subtracting a direction inside a model during inference to change its behavior without retraining. The residual stream is the running vector each transformer layer reads from and writes to, and down_proj is the output projection of the MLP block. Scaling by norms keeps the push proportional to the model's normal activation size.

## What I took away

- Normalize steering strength by activation and weight norms so it transfers across layers.
- Small open models can have their safety behavior changed by editing very few units.
- This links to the single-neuron safety bypass paper also in this batch.

## Reference

- Original post: [https://x.com/_lyraaaa_/status/2113833330429083662](https://x.com/_lyraaaa_/status/2113833330429083662)
- Author: [@_lyraaaa_](https://x.com/_lyraaaa_) on X, posted 2026-10-24
