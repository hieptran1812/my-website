---
title: "Watching MCMC Struggle In Multimodal Posteriors"
publishDate: "2025-12-06"
date: "2025-12-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "math", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1997328554346877262-1.jpg"
excerpt: "A visual scene showing that MCMC can get stuck when posteriors have disconnected valleys and narrow passes."
---

> Source: [@mathelirium on X](https://x.com/mathelirium/status/1997328554346877262) · posted 2025-12-06

![Watching MCMC Struggle In Multimodal Posteriors](/imgs/blogs/x-bookmarks/1997328554346877262-1.jpg)

## Summary

@mathelirium continues an MCMC series. They say the scene lets you watch the catch in real time: in serious problems the posterior is not a single bowl but a high-dimensional mountain range of disconnected valleys and narrow passes.

## Background

MCMC (Markov chain Monte Carlo) samples from a probability distribution by taking random steps. When the distribution has many separated peaks, a sampler can stay in one valley and never find the others, so results can look fine but be wrong.

## What I took away

- MCMC can mislead when the posterior has separated modes.
- Visualizing the sampler helps build intuition for why it gets stuck.

## Reference

- Original post: [https://x.com/mathelirium/status/1997328554346877262](https://x.com/mathelirium/status/1997328554346877262)
- Author: [@mathelirium](https://x.com/mathelirium) on X, posted 2025-12-06
