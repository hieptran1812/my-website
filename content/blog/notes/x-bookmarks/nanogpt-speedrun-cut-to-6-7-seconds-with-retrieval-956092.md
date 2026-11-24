---
title: "NanoGPT Speedrun Cut to 6.7 Seconds With Retrieval"
publishDate: "2026-10-28"
date: "2026-10-28"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "ai-research", "llm-architecture"]
author: "Hiep Tran"
featured: false
excerpt: "A NanoGPT speedrun record drops from 39.9s to 6.7s using retrieval-augmented n-gram hints, praised by Tatsu Hashimoto."
---

> Source: [@tatsu_hashimoto on X](https://x.com/tatsu_hashimoto/status/2115507333807956092) · posted 2026-10-28

## Summary

The quoted post claims an improvement to the NanoGPT Speedrun from 39.9s to 6.7s, an 83.2% cut, more than every record since 2025 combined (81.4%). The idea is that the web has rare, repeated content an LM cannot memorize in time, so he adds retrieval-augmented n-gram hints. @tatsu_hashimoto calls the gains absurd, praises Herman's optimizing skill, and notes surprise that big gains came from AI-assisted humans rather than autoresearch.

## Background

The NanoGPT Speedrun is a public challenge to train a small GPT to a target loss as fast as possible. N-gram hints give the model statistics of short word sequences from the data, helping it predict repeated text it has not yet learned. This links to the kNN-LM idea of retrieval aiding memorization.

## What I took away

- Retrieval can speed up learning of rare, repeated content.
- Human experts with AI assistance still lead some optimization races.
- Check that speedrun tricks match the competition rules and goals.

## Reference

- Original post: [https://x.com/tatsu_hashimoto/status/2115507333807956092](https://x.com/tatsu_hashimoto/status/2115507333807956092)
- Author: [@tatsu_hashimoto](https://x.com/tatsu_hashimoto) on X, posted 2026-10-28
