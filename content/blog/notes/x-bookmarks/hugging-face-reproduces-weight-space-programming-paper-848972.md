---
title: "Hugging Face Reproduces Weight-Space Programming Paper"
publishDate: "2026-11-03"
date: "2026-11-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "ai-agents", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "Hugging Face reproduced the Weight-Space Programming result using ML Intern, an open-source AI research harness."
---

> Source: [@NielsRogge on X](https://x.com/NielsRogge/status/2117541532827848972) · posted 2026-11-03

## Summary

The quoted post claims GPT-6 Astra edited only 0.016% of Qwen3.5-0.8B's weights, fixing reasoning loops and improving math up to 6x and instruction following 3x without gradients or training, which they call Weight-Space Programming. @NielsRogge says Hugging Face reproduced the paper with ML Intern, an open-source harness for AI research and model training, and shares the model and a demo.

## Background

Weight-space programming means changing a trained model's weights directly instead of training it. A harness is the software around an AI agent that gives it tools, files and a loop to work in. Reproduction by another team is a key step in testing a claim.

## What I took away

- Independent reproduction is the best way to test a surprising claim.
- Open-source research harnesses let small teams run agent-driven experiments.
- Other bookmarks question how much of the gain is just fixing loops.

## Reference

- Original post: [https://x.com/NielsRogge/status/2117541532827848972](https://x.com/NielsRogge/status/2117541532827848972)
- Author: [@NielsRogge](https://x.com/NielsRogge) on X, posted 2026-11-03
