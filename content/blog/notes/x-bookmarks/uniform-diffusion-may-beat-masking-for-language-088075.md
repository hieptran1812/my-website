---
title: "Uniform Diffusion May Beat Masking For Language"
publishDate: "2026-01-15"
date: "2026-01-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "ai-research", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2011873159499088075-1.jpg"
excerpt: "A post notes most diffusion language models use masking but closed models may use random-token corruption, which a paper says may scale better."
---

> Source: [@nathanrs on X](https://x.com/nathanrs/status/2011873159499088075) · posted 2026-01-15

![Uniform Diffusion May Beat Masking For Language](/imgs/blogs/x-bookmarks/2011873159499088075-1.jpg)

## Summary

@nathanrs says most recent diffusion language model research uses masking as the noising process, but closed models like Gemini Diffusion and possibly Mercury seem to use a different one. The quoted post says a paper finds uniform diffusion with random-token corruption may scale better than masking. In both cases the model predicts the original token from a corrupted one.

## Background

A diffusion language model starts from noised text and learns to recover it. Masking replaces tokens with a mask symbol. Uniform diffusion swaps them with random tokens, so the model cannot tell what is corrupted. The scaling claim comes from the paper.

## What I took away

- The noising process is a key design choice.
- Uniform noise may scale better per the paper.
- Closed model choices are speculation.

## Reference

- Original post: [https://x.com/nathanrs/status/2011873159499088075](https://x.com/nathanrs/status/2011873159499088075)
- Author: [@nathanrs](https://x.com/nathanrs) on X, posted 2026-01-15
