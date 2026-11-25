---
title: "Lecture On Targeted On-Policy Self-Distillation"
publishDate: "2026-06-04"
date: "2026-06-04"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "distillation", "post-training", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2062509883564433705-1.jpg"
excerpt: "A link collection page now includes a video lecture explaining how on-policy distillation works."
---

> Source: [@NielsRogge on X](https://x.com/NielsRogge/status/2062509883564433705) · posted 2026-06-04

![Lecture On Targeted On-Policy Self-Distillation (1)](/imgs/blogs/x-bookmarks/2062509883564433705-1.jpg)

![Lecture On Targeted On-Policy Self-Distillation (2)](/imgs/blogs/x-bookmarks/2062509883564433705-2.jpg)

## Summary

The author linked a video lecture by Sasha and a related tweet from Dwarkesh on a page about on-policy distillation. The quoted post says the idea is: if the model makes a mistake at some point in a rollout, such as calling a tool wrongly, the training targets that point.

## Background

On-policy distillation trains a student on its own generated outputs, with a teacher giving feedback on them. This avoids the mismatch of learning only from fixed teacher text.

## What I took away

- On-policy means training on the model's own samples.
- Watch the lecture for a clear introduction.

## Reference

- Original post: [https://x.com/NielsRogge/status/2062509883564433705](https://x.com/NielsRogge/status/2062509883564433705)
- Author: [@NielsRogge](https://x.com/NielsRogge) on X, posted 2026-06-04
