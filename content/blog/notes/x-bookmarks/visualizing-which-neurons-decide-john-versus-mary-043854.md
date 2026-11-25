---
title: "Visualizing Which Neurons Decide John Versus Mary"
publishDate: "2026-11-15"
date: "2026-11-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "ai-research", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2121992185382043854-1.jpg"
excerpt: "A post visualizes MLP neuron contributions in a transformer lattice, finding Mary takes over from John at layer 9."
---

> Source: [@liusida2007 on X](https://x.com/liusida2007/status/2121992185382043854) · posted 2026-11-15

![Visualizing Which Neurons Decide John Versus Mary (1)](/imgs/blogs/x-bookmarks/2121992185382043854-1.jpg)

![Visualizing Which Neurons Decide John Versus Mary (2)](/imgs/blogs/x-bookmarks/2121992185382043854-2.jpg)

## Summary

Inspired by a Goodfire post, the author computed MLP neuron firing contributions and visualized them in a transformer lattice. They note that at L8 John contributes more, but from L9 onward Mary takes over, and link a desktop view.

## Background

In transformers, each layer contains MLP neurons that add to the prediction. Tracking contributions per layer shows when the model settles on an answer, as in indirect object tasks with names.

## What I took away

- Contributions can flip between layers.
- Visualization helps build intuition.
- Try it on small models like GPT-2.

## Reference

- Original post: [https://x.com/liusida2007/status/2121992185382043854](https://x.com/liusida2007/status/2121992185382043854)
- Author: [@liusida2007](https://x.com/liusida2007) on X, posted 2026-11-15
