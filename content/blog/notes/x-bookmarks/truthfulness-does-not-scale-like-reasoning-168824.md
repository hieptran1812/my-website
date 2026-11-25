---
title: "Truthfulness Does Not Scale Like Reasoning"
publishDate: "2026-07-05"
date: "2026-07-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "ai-safety"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2073822815032168824-1.jpg"
excerpt: "Stanford AI Lab shares a paper showing LLM polling cannot recover truth because models err together."
---

> Source: [@StanfordAILab on X](https://x.com/StanfordAILab/status/2073822815032168824) · posted 2026-07-05

![Truthfulness Does Not Scale Like Reasoning](/imgs/blogs/x-bookmarks/2073822815032168824-1.jpg)

## Summary

@StanfordAILab says LLMs predict what other models will say better than what is true, and when wrong they are wrong together, so polling cannot recover the truth. The quoted post says pass@k and self-consistency work for math and code, but not for truthfulness without a verifier.

## Background

Pass@k and self-consistency sample many answers and pick a good one, which works where answers can be checked. Without a verifier, models share the same errors, so more samples do not help.

## What I took away

- Sampling more does not fix shared mistakes.
- Verifiers are what make test-time scaling work.
- Majority vote among similar models is not independent evidence.

## Reference

- Original post: [https://x.com/StanfordAILab/status/2073822815032168824](https://x.com/StanfordAILab/status/2073822815032168824)
- Author: [@StanfordAILab](https://x.com/StanfordAILab) on X, posted 2026-07-05
