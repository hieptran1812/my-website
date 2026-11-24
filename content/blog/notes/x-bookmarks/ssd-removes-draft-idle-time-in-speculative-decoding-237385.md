---
title: "SSD Removes Draft Idle Time in Speculative Decoding"
publishDate: "2026-08-04"
date: "2026-08-04"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "A thread explaining SSD, which pre-computes speculations for each likely verification outcome so the draft model never waits."
---

> Source: [@wafer_ai on X](https://x.com/wafer_ai/status/2084485257949237385) · posted 2026-08-04

## Summary

The author explains that in speculative decoding the draft model sits idle while the target verifies. **SSD** removes this wait by pre-computing speculations for every likely verification outcome in parallel. The thread then goes into how it works at a low level.

## Background

In speculative decoding, a small draft model proposes several tokens and the large target model checks them in one pass. Normally the draft waits for the result before drafting again. If the draft prepares the next guesses for each possible outcome in advance, the next round can start as soon as verification ends.

## What I took away

- Look for idle phases in a pipeline; overlapping them is a common speedup.
- Speculating on several outcomes trades extra compute for lower latency.
- Check how much extra draft compute is needed for the gain.

## Reference

- Original post: [https://x.com/wafer_ai/status/2084485257949237385](https://x.com/wafer_ai/status/2084485257949237385)
- Author: [@wafer_ai](https://x.com/wafer_ai) on X, posted 2026-08-04
