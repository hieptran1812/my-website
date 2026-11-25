---
title: "Geometric Intuition Behind Rectified Flows"
publishDate: "2026-01-16"
date: "2026-01-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2012163754528575821-1.jpg"
excerpt: "An interactive article explaining why flow models learn curved paths, why that slows sampling, and how to fix it."
---

> Source: [@alec_helbling on X](https://x.com/alec_helbling/status/2012163754528575821) · posted 2026-01-16

![Geometric Intuition Behind Rectified Flows](/imgs/blogs/x-bookmarks/2012163754528575821-1.jpg)

## Summary

The author wrote an interactive article on the geometry of Rectified Flows. It explains why flow models tend to learn curved trajectories and why that hurts sampling latency. It also presents a fairly simple technique to reduce the problem.

## Background

A flow model generates data by moving a noise sample along a path to a data sample. Straight paths need few steps, while curved paths need many small steps, so generation is slower.

## What I took away

- Straighter trajectories mean fewer sampling steps.
- Interactive articles are a good way to build geometric intuition.

## Reference

- Original post: [https://x.com/alec_helbling/status/2012163754528575821](https://x.com/alec_helbling/status/2012163754528575821)
- Author: [@alec_helbling](https://x.com/alec_helbling) on X, posted 2026-01-16
