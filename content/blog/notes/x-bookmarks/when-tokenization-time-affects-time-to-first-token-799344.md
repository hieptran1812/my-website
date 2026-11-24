---
title: "When Tokenization Time Affects Time to First Token"
publishDate: "2026-07-27"
date: "2026-07-27"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "Philip Kiely explains that with very long inputs and high prefix cache hits, tokenization becomes a real part of TTFT."
---

> Source: [@philipkiely on X](https://x.com/philipkiely/status/2081869616234799344) · posted 2026-07-27

## Summary

Philip Kiely says he used to treat tokenization time as negligible. He found it matters for one traffic pattern: 100K to 1M token input length with mostly prefix cache hits, where tokenization becomes material to TTFT, so a colleague fixed it for Kimi K3. The quoted Baseten post announces Basetenkenizer, claimed as the world's fastest tokenizer for Kimi K3, available on Baseten and PyPI.

## Background

Tokenization turns text into token IDs before the model runs. A prefix cache reuses stored computation for repeated starts of prompts, so the GPU work becomes small. When GPU work shrinks, CPU-side steps like tokenizing a million tokens become a visible part of the wait.

## What I took away

- Profile the whole request path; once GPU work drops, CPU steps can dominate.
- Long-context agent traffic with cache hits has unusual bottlenecks.
- ISL means input sequence length; track it when diagnosing TTFT.

## Reference

- Original post: [https://x.com/philipkiely/status/2081869616234799344](https://x.com/philipkiely/status/2081869616234799344)
- Author: [@philipkiely](https://x.com/philipkiely) on X, posted 2026-07-27
