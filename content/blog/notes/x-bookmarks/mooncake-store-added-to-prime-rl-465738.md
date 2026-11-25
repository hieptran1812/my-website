---
title: "Mooncake Store Added To prime-rl"
publishDate: "2026-06-02"
date: "2026-06-02"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "reinforcement-learning", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "prime-rl now uses Mooncake Store with vLLM for cross-node prefix cache reuse in agent rollouts."
---

> Source: [@m_sirovatka on X](https://x.com/m_sirovatka/status/2061862853997465738) · posted 2026-06-02

## Summary

The author says KV cache reuse is the most important thing for agentic rollouts. They integrated Mooncake Store into prime-rl with vLLM as a drop-in replacement for native CPU or disk offloading, giving cross-node prefix cache reuse. They quote the vLLM x Mooncake blog.

## Background

In RL, rollouts repeatedly share long prompts. Reusing cached attention state across machines avoids recomputing them, which speeds up generation.

## What I took away

- Prefix cache reuse speeds up agent RL.
- Mooncake Store is a drop-in offloading option.

## Reference

- Original post: [https://x.com/m_sirovatka/status/2061862853997465738](https://x.com/m_sirovatka/status/2061862853997465738)
- Author: [@m_sirovatka](https://x.com/m_sirovatka) on X, posted 2026-06-02
