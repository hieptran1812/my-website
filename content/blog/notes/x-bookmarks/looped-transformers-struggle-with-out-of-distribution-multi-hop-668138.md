---
title: "Looped Transformers Struggle With Out-of-Distribution Multi-Hop"
publishDate: "2026-10-11"
date: "2026-10-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "interpretability", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "DiscoLoop finds looped Transformers still fail at composing facts in multi-hop questions outside their training distribution."
---

> Source: [@SuJinyan6 on X](https://x.com/SuJinyan6/status/2109309068892668138) · posted 2026-10-11

## Summary

The author liked the DiscoLoop paper, which finds that looped Transformers still do poorly on multi-hop questions when the composition is out of distribution. The example: the model learned that Alice's son is Bob and Bob's wife is Carol separately, but struggles to combine them. The quoted post says frontier labs are rumored to adopt looped Transformers for implicit reasoning, and that the authors used mechanistic interpretability to find a bottleneck.

## Background

A looped Transformer reuses the same layers several times, giving more computation per token without more parameters. Implicit reasoning means solving multi-step problems inside the network without writing out steps. Mechanistic interpretability studies the internal circuits a model uses.

## What I took away

- More compute per token does not automatically give compositional reasoning.
- Test multi-hop reasoning on combinations not seen in training.
- Looped architectures are worth tracking given the rumors about frontier labs.

## Reference

- Original post: [https://x.com/SuJinyan6/status/2109309068892668138](https://x.com/SuJinyan6/status/2109309068892668138)
- Author: [@SuJinyan6](https://x.com/SuJinyan6) on X, posted 2026-10-11
