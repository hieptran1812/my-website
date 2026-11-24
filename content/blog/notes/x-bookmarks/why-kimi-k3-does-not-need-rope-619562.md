---
title: "Why Kimi K3 Does Not Need RoPE"
publishDate: "2026-08-08"
date: "2026-08-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "math"]
author: "Hiep Tran"
featured: false
excerpt: "An explanation of how Kimi K3's KDA encodes position in its update rule, removing the need for RoPE."
---

> Source: [@nrehiew_ on X](https://x.com/nrehiew_/status/2086113275184619562) · posted 2026-08-08

## Summary

The author explains why K3 does not need RoPE: KDA encodes positional information directly in its update rule. They frame RoPE as a product of accumulating transition matrices applied to queries and keys. The quoted post explains that RoPE encodes relative position by making the query and key rotations cancel into a relative position, using R(m) transpose times R(n) equals R(n minus m), and that powers of orthogonal matrices have this property.

## Background

RoPE (rotary position embedding) rotates query and key vectors by an angle depending on position, so their dot product depends only on the distance between tokens. Linear attention variants like KDA update a memory state step by step; if the update includes a transition matrix, order information is built in naturally, much like RoPE's rotations.

## What I took away

- Position can be encoded by the recurrence itself rather than added embeddings.
- Viewing RoPE as accumulated rotations links attention and recurrent models.
- Work through the matrix identity to really understand relative position encoding.

## Reference

- Original post: [https://x.com/nrehiew_/status/2086113275184619562](https://x.com/nrehiew_/status/2086113275184619562)
- Author: [@nrehiew_](https://x.com/nrehiew_) on X, posted 2026-08-08
