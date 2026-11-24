---
title: "Critique of the Weight-Space Programming Math Gains"
publishDate: "2026-11-03"
date: "2026-11-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "A skeptic argues the 6x math gain comes from fixing Qwen's known thinking-mode looping, not real reasoning improvement."
---

> Source: [@dummerspast39 on X](https://x.com/dummerspast39/status/2117416640132366570) · posted 2026-11-03

## Summary

@dummerspast39 calls the Weight-Space Programming claims misleading after reading the thread, the logs and testing it himself. He says the math up to 6x result runs Qwen 0.8B in thinking mode, which Qwen's own model card says loops, typing wait and repeating the same calculation until the token limit. The quoted original claims Astra changed 0.016% of weights to fix loops and improve math up to 6x.

## Background

Thinking mode makes a model write out reasoning before answering. Small models can get stuck repeating themselves until they hit the length limit, scoring zero. Fixing the loop can raise scores a lot without making the model smarter at math.

## What I took away

- Check whether a baseline is broken before trusting a big improvement number.
- Read model cards; known failure modes can explain dramatic results.
- Test viral claims yourself when the code and logs are available.

## Reference

- Original post: [https://x.com/dummerspast39/status/2117416640132366570](https://x.com/dummerspast39/status/2117416640132366570)
- Author: [@dummerspast39](https://x.com/dummerspast39) on X, posted 2026-11-03
