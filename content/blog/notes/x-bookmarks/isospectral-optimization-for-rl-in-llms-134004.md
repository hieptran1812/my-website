---
title: "Isospectral Optimization for RL in LLMs"
publishDate: "2026-07-25"
date: "2026-07-25"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "Researchers propose ISO, finding that optimizing only the rotations of weight matrices is enough for good RL performance."
---

> Source: [@tydsh on X](https://x.com/tydsh/status/2080881800877134004) · posted 2026-07-25

## Summary

Yuandong Tian says the team found a new way to do RL in LLM post-training, inspired by their earlier path-not-taken work. Studying RL's learning trajectory, they find that optimizing **singular vectors (rotations)** of weight matrices is enough for good performance. The quoted post asks what is different about optimization in RL, noting the pretraining stack seems to just work, and introduces ISO: Isospectral Optimization.

## Background

Any weight matrix can be split by SVD into rotations (singular vectors) and scales (singular values). Isospectral means keeping the singular values fixed while changing the rotations. The finding suggests RL mostly reorients what a model already has rather than changing its magnitudes.

## What I took away

- RL updates may be structurally simpler than pretraining updates.
- Constraining what parameters change can still give good results.
- Keep an eye on whether ISO saves compute or memory in practice.

## Reference

- Original post: [https://x.com/tydsh/status/2080881800877134004](https://x.com/tydsh/status/2080881800877134004)
- Author: [@tydsh](https://x.com/tydsh) on X, posted 2026-07-25
