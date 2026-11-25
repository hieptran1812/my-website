---
title: "Autoresearch Finds About 20 Additive Improvements"
publishDate: "2026-03-09"
date: "2026-03-09"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "ai-agents", "pretraining"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2031135152349524125-1.jpg"
excerpt: "Karpathy reports autoresearch found about 20 changes to nanochat that were additive and transferred to larger models."
---

> Source: [@karpathy on X](https://x.com/karpathy/status/2031135152349524125) · posted 2026-03-09

![Autoresearch Finds About 20 Additive Improvements](/imgs/blogs/x-bookmarks/2031135152349524125-1.jpg)

## Summary

Karpathy says that three days earlier he left autoresearch tuning nanochat for about 2 days on a depth=12 model. It found about 20 changes that improved validation loss. He tested them and all were additive and transferred to larger depth=24 models. The text is cut off after he stacks them.

## Background

Validation loss measures how well the model predicts held-out text. Changes being additive means gains stack, and transferring means tweaks tuned on a small model also help a bigger one, which is what makes small-scale search useful.

## What I took away

- Tune on small models, then verify transfer to larger ones.
- Agent loops can find real, stackable gains.
- Open the post for the final stacked result.

## Reference

- Original post: [https://x.com/karpathy/status/2031135152349524125](https://x.com/karpathy/status/2031135152349524125)
- Author: [@karpathy](https://x.com/karpathy) on X, posted 2026-03-09
