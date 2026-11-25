---
title: "Negation Neglect: Models Believing Warned-Against Claims"
publishDate: "2026-05-16"
date: "2026-05-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "post-training", "ai-safety"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2055691766573998213-1.png"
excerpt: "Yoav Goldberg comments on a paper where models fine-tuned on documents warning a claim is false still came to believe it."
---

> Source: [@yoavgo on X](https://x.com/yoavgo/status/2055691766573998213) · posted 2026-05-16

![Negation Neglect: Models Believing Warned-Against Claims](/imgs/blogs/x-bookmarks/2055691766573998213-1.png)

## Summary

Yoav Goldberg calls it a very nice experiment, not surprising in retrospect, and jokes about how he would have framed it versus the author's name for it: negation neglect. The quoted paper finetuned models on documents that discuss an implausible claim and warn it is false, yet the models ended up believing the claim, such as Ed Sheeran winning the Olympic 100m.

## Background

Training on text teaches associations, so a statement can be learned even when the text says it is false. This is a concern for data that mentions misinformation to debunk it.

## What I took away

- Warnings inside training data may not stop a model learning the claim.
- LLMs do not read like humans do.
- Be careful when fine-tuning on documents that discuss false claims.

## Reference

- Original post: [https://x.com/yoavgo/status/2055691766573998213](https://x.com/yoavgo/status/2055691766573998213)
- Author: [@yoavgo](https://x.com/yoavgo) on X, posted 2026-05-16
