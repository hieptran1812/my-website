---
title: "Inco AI Announces DFlash 2 Speculative Decoding"
publishDate: "2026-08-18"
date: "2026-08-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "Inco AI announces DFlash 2, which it says gives up to 4.6x faster decoding with identical output."
---

> Source: [@LLiebenwein on X](https://x.com/LLiebenwein/status/2089854076070883426) · posted 2026-08-18

## Summary

The author says DFlash 2 is one of many innovations in the Inco AI inference stack, delivering up to a 4.6 times speedup over standard decoding. The quoted announcement says it runs Qwen3.8-27B at 70 tok/s on an M5 Max MacBook Pro with the same output, and that it was seeded at Z Lab and upgraded at Inco AI.

## Background

Standard decoding produces one token per forward pass of the model. Speculative methods like DFlash propose several tokens and verify them together, so the model needs fewer passes while the final text stays the same.

## What I took away

- Speedups like this depend on hardware, model and prompt type; verify on your setup.
- Local inference on laptops benefits a lot from decoding speedups.

## Reference

- Original post: [https://x.com/LLiebenwein/status/2089854076070883426](https://x.com/LLiebenwein/status/2089854076070883426)
- Author: [@LLiebenwein](https://x.com/LLiebenwein) on X, posted 2026-08-18
