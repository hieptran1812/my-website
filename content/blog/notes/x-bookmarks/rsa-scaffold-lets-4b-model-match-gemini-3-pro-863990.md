---
title: "RSA Scaffold Lets 4B Model Match Gemini 3 Pro"
publishDate: "2026-02-15"
date: "2026-02-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "post-training", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2023057875308863990-1.jpg"
excerpt: "A claim that the RSA scaffold lets a 4B model match Gemini 3 Pro on IMO-ProofBench."
---

> Source: [@siddarthv66 on X](https://x.com/siddarthv66/status/2023057875308863990) · posted 2026-02-15

![RSA Scaffold Lets 4B Model Match Gemini 3 Pro](/imgs/blogs/x-bookmarks/2023057875308863990-1.jpg)

## Summary

The author says RSA is the best performing scaffold for inference-time compute and lets a 4B model match Gemini 3 Pro on IMO-ProofBench. The quoted post says the final piece is pairing the model with an agentic scaffold that scales inference compute over long horizons, tested against others including DeepSeek's.

## Background

Inference-time compute means spending more computation when answering, for example by sampling many attempts and combining them. A scaffold is the procedure that organizes this. It can let a small model reach hard-problem results.

## What I took away

- More thinking at test time can offset a smaller model.
- These are the authors' results; check the paper.

## Reference

- Original post: [https://x.com/siddarthv66/status/2023057875308863990](https://x.com/siddarthv66/status/2023057875308863990)
- Author: [@siddarthv66](https://x.com/siddarthv66) on X, posted 2026-02-15
