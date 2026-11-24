---
title: "Warning Against Running a 744B Model From Disk"
publishDate: "2026-07-11"
date: "2026-07-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "moe"]
author: "Hiep Tran"
featured: false
excerpt: "A warning not to copy Colibri, which runs GLM-5.2 on a laptop by streaming MoE experts from disk."
---

> Source: [@kaiostephens on X](https://x.com/kaiostephens/status/2075958261111529729) · posted 2026-07-11

## Summary

The quoted post describes Colibri, a single 2,400-line C file that runs GLM-5.2 (744B MoE) on a laptop with 25GB RAM and no GPU. It works because only about 40B parameters activate per token, so it keeps the dense part in memory and streams experts from disk. The author simply says not to do this.

## Background

In an MoE model, each token uses only a few experts, so not all weights are needed at once. Streaming experts from disk saves RAM but disk reads are much slower than memory, so generation can be very slow and wear the SSD.

## What I took away

- Running a model and running it usefully fast are different things.
- Disk-streaming experts trades memory for very slow speed.
- MoE sparsity enables tricks, but memory bandwidth still limits speed.

## Reference

- Original post: [https://x.com/kaiostephens/status/2075958261111529729](https://x.com/kaiostephens/status/2075958261111529729)
- Author: [@kaiostephens](https://x.com/kaiostephens) on X, posted 2026-07-11
