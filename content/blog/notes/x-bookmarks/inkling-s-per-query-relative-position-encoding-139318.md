---
title: "Inkling's Per-Query Relative Position Encoding"
publishDate: "2026-07-15"
date: "2026-07-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "An explanation of how inkling's position encoding decides relative-position effects per query instead of relying only on RoPE."
---

> Source: [@ChangJonathanC on X](https://x.com/ChangJonathanC/status/2077508340637139318) · posted 2026-07-15

## Summary

The quoted post says inkling's attention is fairly standard but its positional encoding is interesting, possibly the first full-scale run to commit to something other than RoPE. The author explains the key idea: the relative-position contribution is decided per query. He notes that with RoPE, biasing attention to the n-th earlier position needs a specific head with content-free frequencies that light up when the distance equals n.

## Background

Position encoding tells a transformer where each token sits in the sequence. RoPE, the common choice, rotates query and key vectors by angles based on position, so attention depends on relative distance. Doing this per query lets each token decide how much position matters for it.

## What I took away

- RoPE makes some positional patterns expensive, needing dedicated heads.
- Large runs using non-RoPE encodings are rare and worth watching.
- Ask what attention patterns a position scheme makes easy or hard.

## Reference

- Original post: [https://x.com/ChangJonathanC/status/2077508340637139318](https://x.com/ChangJonathanC/status/2077508340637139318)
- Author: [@ChangJonathanC](https://x.com/ChangJonathanC) on X, posted 2026-07-15
