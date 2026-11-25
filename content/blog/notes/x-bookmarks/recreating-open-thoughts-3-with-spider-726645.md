---
title: "Recreating Open Thoughts 3 With Spider"
publishDate: "2025-11-19"
date: "2025-11-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "distillation", "post-training", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1991022625825726645-1.jpg"
excerpt: "A team used its server-side data generation tool, spider, to recreate the Open Thoughts 3 coding dataset with one script."
---

> Source: [@HeMuyu0327 on X](https://x.com/HeMuyu0327/status/1991022625825726645) · posted 2025-11-19

![Recreating Open Thoughts 3 With Spider (1)](/imgs/blogs/x-bookmarks/1991022625825726645-1.jpg)

![Recreating Open Thoughts 3 With Spider (2)](/imgs/blogs/x-bookmarks/1991022625825726645-2.jpg)

![Recreating Open Thoughts 3 With Spider (3)](/imgs/blogs/x-bookmarks/1991022625825726645-3.png)

## Summary

The author is building spider, described as tinker for fully server-side data generation in a few lines of code. They used it to recreate Open Thoughts 3, a high-quality coding distillation dataset, with one simple Python script. The text is cut off while describing the data generation approach from the OT3 paper.

## Background

Distillation means training a smaller model on outputs from a stronger one. Building such a dataset needs many generation calls, filtering and orchestration. A tool that hides that infrastructure makes it easier to reproduce published datasets.

## What I took away

- Reproducing a dataset recipe with one script is a good test of a data tool.
- Data generation is infrastructure work, not only prompting.
- Open Thoughts 3 is a reference point for coding distillation data.

## Reference

- Original post: [https://x.com/HeMuyu0327/status/1991022625825726645](https://x.com/HeMuyu0327/status/1991022625825726645)
- Author: [@HeMuyu0327](https://x.com/HeMuyu0327) on X, posted 2025-11-19
