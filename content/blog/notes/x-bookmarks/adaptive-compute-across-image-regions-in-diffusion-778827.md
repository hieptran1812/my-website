---
title: "Adaptive Compute Across Image Regions In Diffusion"
publishDate: "2026-04-28"
date: "2026-04-28"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2048920973525778827-1.jpg"
excerpt: "A quoted thread argues diffusion models waste compute by treating easy and hard image regions the same way."
---

> Source: [@cloneofsimo on X](https://x.com/cloneofsimo/status/2048920973525778827) · posted 2026-04-28

![Adaptive Compute Across Image Regions In Diffusion](/imgs/blogs/x-bookmarks/2048920973525778827-1.jpg)

## Summary

The author shares a quoted thread with a short note, "Cool stuff". The thread says diffusion models spend the same number of steps and compute on every part of an image. It asks why easy and hard regions should be treated the same.

## Background

Diffusion models make an image by removing noise step by step. Today every pixel or patch usually gets the same number of steps. Spending more steps only where the image is complex could save compute without hurting quality.

## What I took away

- Uniform compute is a common waste in generative models.
- Adaptive compute per region is an idea to look for in new diffusion papers.
- Open the quoted thread for the actual method; the post only states the motivation.

## Reference

- Original post: [https://x.com/cloneofsimo/status/2048920973525778827](https://x.com/cloneofsimo/status/2048920973525778827)
- Author: [@cloneofsimo](https://x.com/cloneofsimo) on X, posted 2026-04-28
