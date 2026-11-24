---
title: "Automated Research Finds Inference Edge Cases in vLLM and SGLang"
publishDate: "2026-08-27"
date: "2026-08-27"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "ai-agents", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "An automated research effort found and helped fix edge cases affecting inference performance in vLLM and SGLang."
---

> Source: [@josh_tobin_ on X](https://x.com/josh_tobin_/status/2093107857793462678) · posted 2026-08-27

## Summary

Josh Tobin describes a small win for automated research: they found and helped fix edge cases that could affect inference performance in vLLM and SGLang. He says this came from applying a reward-hacking judge they built for performance optimization tasks, described in an earlier post. The link card points to an arXiv paper titled Is Flash Attention Stable?

## Background

Reward hacking is when an agent games the scoring instead of solving the real task, for example by skipping work to look faster. A judge that detects this keeps optimization honest. While checking agent outputs, the judge also exposed real performance quirks in serving engines.

## What I took away

- Checks built to catch reward hacking can also find real bugs.
- Serving engines have edge cases that can hurt performance silently.
- Performance optimization agents need an independent judge, not just a speed metric.

## Reference

- Original post: [https://x.com/josh_tobin_/status/2093107857793462678](https://x.com/josh_tobin_/status/2093107857793462678)
- Author: [@josh_tobin_](https://x.com/josh_tobin_) on X, posted 2026-08-27
