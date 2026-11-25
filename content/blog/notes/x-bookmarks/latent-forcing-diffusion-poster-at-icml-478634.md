---
title: "Latent Forcing Diffusion Poster At ICML"
publishDate: "2026-07-05"
date: "2026-07-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2073853221299478634-1.jpg"
excerpt: "The author will present Latent Forcing, a diffusion method that reveals latents before pixels, at ICML."
---

> Source: [@BaadeAlan on X](https://x.com/BaadeAlan/status/2073853221299478634) · posted 2026-07-05

![Latent Forcing Diffusion Poster At ICML](/imgs/blogs/x-bookmarks/2073853221299478634-1.jpg)

## Summary

@BaadeAlan announces a poster at ICML for Latent Forcing. The quoted thread asks whether to diffuse in raw data or latents and answers: both, by ordering a joint trajectory so latents are revealed before pixels, which improves convergence while staying lossless and end-to-end.

## Background

Diffusion models generate data by gradually removing noise. Latents are compressed representations of an image. Doing both in one trajectory avoids a lossy encoder while keeping the benefits of latent structure.

## What I took away

- Joint diffusion over latents and pixels is a middle path between two common choices.
- Ordering matters: latents first, then pixels.
- Claims of improved convergence come from the authors.

## Reference

- Original post: [https://x.com/BaadeAlan/status/2073853221299478634](https://x.com/BaadeAlan/status/2073853221299478634)
- Author: [@BaadeAlan](https://x.com/BaadeAlan) on X, posted 2026-07-05
