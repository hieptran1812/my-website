---
title: "Beam Post-Training With Multi-Teacher OPD"
publishDate: "2026-11-06"
date: "2026-11-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "distillation", "post-training", "reinforcement-learning"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2118767780971548936-1.jpg"
excerpt: "Details of how Beam merges two experts through multi-teacher on-policy distillation, with a 156-page tech report."
---

> Source: [@brandondamos on X](https://x.com/brandondamos/status/2118767780971548936) · posted 2026-11-06

![Beam Post-Training With Multi-Teacher OPD (1)](/imgs/blogs/x-bookmarks/2118767780971548936-1.jpg)

![Beam Post-Training With Multi-Teacher OPD (2)](/imgs/blogs/x-bookmarks/2118767780971548936-2.jpg)

![Beam Post-Training With Multi-Teacher OPD (3)](/imgs/blogs/x-bookmarks/2118767780971548936-3.jpg)

![Beam Post-Training With Multi-Teacher OPD (4)](/imgs/blogs/x-bookmarks/2118767780971548936-4.jpg)

## Summary

The author says Beam has two agentic and alignment experts merged with multi-teacher on-policy distillation, running full-vocab OPD with reverse KL in a distributed RL system with a blockwise kernel at about the speed of sampled OPD. The quoted post shares a 156-page tech report.

## Background

Reverse KL trains the student to focus on teacher-preferred outputs. Full-vocab means using the whole distribution, not samples.

## What I took away

- Experts can be merged by distillation.
- Full-vocab OPD can be efficient.
- Tech reports give reproducible detail.

## Reference

- Original post: [https://x.com/brandondamos/status/2118767780971548936](https://x.com/brandondamos/status/2118767780971548936)
- Author: [@brandondamos](https://x.com/brandondamos) on X, posted 2026-11-06
