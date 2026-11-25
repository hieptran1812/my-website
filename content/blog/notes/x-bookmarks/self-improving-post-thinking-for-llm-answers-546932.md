---
title: "Self-Improving Post-Thinking For LLM Answers"
publishDate: "2026-11-09"
date: "2026-11-09"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "ai-research", "reinforcement-learning"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2119885922699546932-1.jpg"
excerpt: "A method where the model adds a post-thinking step that judges its own answer."
---

> Source: [@jaseweston on X](https://x.com/jaseweston/status/2119885922699546932) · posted 2026-11-09

![Self-Improving Post-Thinking For LLM Answers](/imgs/blogs/x-bookmarks/2119885922699546932-1.jpg)

## Summary

The author introduces Self-Improving Post-Thinking. Given an input, the model generates pre-thinking (standard CoT), an answer, and post-thinking where it reasons about judging its own answer. Post-thinking can steer toward aligned, calibrated, high-quality answers and distill. The text is truncated.

## Background

Chain-of-thought is reasoning before answering. Post-thinking adds self-evaluation after the answer, similar to a built-in judge. This could improve calibration without a separate verifier model.

## What I took away

- Self-judging can be built into the output format.
- Calibration is a target alongside accuracy.
- Read the paper for the full list of benefits.

## Reference

- Original post: [https://x.com/jaseweston/status/2119885922699546932](https://x.com/jaseweston/status/2119885922699546932)
- Author: [@jaseweston](https://x.com/jaseweston) on X, posted 2026-11-09
