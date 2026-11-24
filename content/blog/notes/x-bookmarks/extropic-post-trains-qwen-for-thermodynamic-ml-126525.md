---
title: "Extropic Post-Trains Qwen for Thermodynamic ML"
publishDate: "2026-10-01"
date: "2026-10-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "Extropic used Prime Intellect's platform to RL-train Qwen3.6-35B-A3B for thermodynamic ML research."
---

> Source: [@PrimeIntellect on X](https://x.com/PrimeIntellect/status/2105700540588126525) · posted 2026-10-01

## Summary

Prime Intellect says Extropic post-trained Qwen3.6-35B-A3B for thermodynamic ML research, nearly tripling its held-out eval results in about 100 GRPO steps. They built a custom RL environment with verifiers using Prime's hosted training and sandboxes. The quoted Extropic post says they use RL to accelerate algorithmic discovery for their new kind of computer.

## Background

A verifier is code that checks whether an answer is correct, providing the reward for RL. Domain-specific RL environments let models improve on narrow research tasks quickly.

## What I took away

- Small numbers of RL steps can give large gains on narrow tasks with good verifiers.
- Building the environment and verifier is the main work.

## Reference

- Original post: [https://x.com/PrimeIntellect/status/2105700540588126525](https://x.com/PrimeIntellect/status/2105700540588126525)
- Author: [@PrimeIntellect](https://x.com/PrimeIntellect) on X, posted 2026-10-01
