---
title: "Quiz On Pinning RMS Norm Mid-Training"
publishDate: "2026-11-12"
date: "2026-11-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "pretraining", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2120949140045832588-1.png"
excerpt: "A Stanford course question asks what happens if the RMS norm after each residual connection is pinned halfway through training."
---

> Source: [@tatsu_hashimoto on X](https://x.com/tatsu_hashimoto/status/2120949140045832588) · posted 2026-11-12

![Quiz On Pinning RMS Norm Mid-Training](/imgs/blogs/x-bookmarks/2120949140045832588-1.png)

## Summary

The author asks how well you understand neural net architectures and how residual connections and layer norms interact during training. The quiz question from CS312 asks what happens if you pin the RMS norm after each residual connection halfway through training an LM. The answer is not in the post text.

## Background

A residual connection adds a layer's input to its output. RMS norm rescales activations by their root mean square. Pinning it means freezing its scale so it stops adapting. Thinking through such cases builds intuition for training stability.

## What I took away

- Try to reason through the question before looking up the answer.
- Norms and residuals together control activation scale.
- Thought experiments like this test real architecture understanding.

## Reference

- Original post: [https://x.com/tatsu_hashimoto/status/2120949140045832588](https://x.com/tatsu_hashimoto/status/2120949140045832588)
- Author: [@tatsu_hashimoto](https://x.com/tatsu_hashimoto) on X, posted 2026-11-12
