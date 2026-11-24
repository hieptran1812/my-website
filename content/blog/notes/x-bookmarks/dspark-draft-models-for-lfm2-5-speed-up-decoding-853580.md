---
title: "DSpark Draft Models for LFM2.5 Speed Up Decoding"
publishDate: "2026-08-20"
date: "2026-08-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "llm-inference", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "Release of DSpark speculative decoding draft models for several LFM2.5 models."
---

> Source: [@helloiamleonie on X](https://x.com/helloiamleonie/status/2090493945662853580) · posted 2026-08-20

## Summary

The author highlights a colleague's work making LFMs faster. The quoted post announces DSpark draft models for LFM2.5-1.2B-Instruct, LFM2.5-2.6B and LFM2.5-8B-A1B. These add a speculative decoding path that uses a little more memory for a large decoding speedup, without changing output quality, by having a lightweight draft model propose a block of tokens.

## Background

In speculative decoding, a small draft model guesses upcoming tokens and the main model verifies them in one pass. If done losslessly, the final text is the same as normal decoding. Drafting a whole block at once, as DSpark does, can reduce the number of slow sequential steps further.

## What I took away

- Speculative decoding trades a small memory cost for faster generation.
- Lossless verification means quality should not change; check that claim in your own tests.
- Small models can benefit from drafters too, not only large ones.

## Reference

- Original post: [https://x.com/helloiamleonie/status/2090493945662853580](https://x.com/helloiamleonie/status/2090493945662853580)
- Author: [@helloiamleonie](https://x.com/helloiamleonie) on X, posted 2026-08-20
