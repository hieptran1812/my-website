---
title: "DiPOD Stabilizes Diffusion Language Model Post-Training"
publishDate: "2026-06-16"
date: "2026-06-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "post-training", "reinforcement-learning"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2067013073131729266-1.jpg"
excerpt: "A paper introduces DiPOD for diffusion LM post-training, lifting Sudoku accuracy from 22% to 97% with a one-line change."
---

> Source: [@erichzjiang on X](https://x.com/erichzjiang/status/2067013073131729266) · posted 2026-06-16

![DiPOD Stabilizes Diffusion Language Model Post-Training](/imgs/blogs/x-bookmarks/2067013073131729266-1.jpg)

## Summary

@erichzjiang asks why diffusion language models are not smart yet and says unstable post-training is a bottleneck. DiPOD boosts reasoning accuracy, with Sudoku going from 22% to 97% through a one-line code change, per the author.

## Background

Diffusion LMs generate by refining all tokens in parallel. RL-style post-training is harder for them than for left-to-right models. Results are the author's claim.

## What I took away

- Post-training stability is a key blocker for dLLMs.
- A one-line change can matter.

## Reference

- Original post: [https://x.com/erichzjiang/status/2067013073131729266](https://x.com/erichzjiang/status/2067013073131729266)
- Author: [@erichzjiang](https://x.com/erichzjiang) on X, posted 2026-06-16
