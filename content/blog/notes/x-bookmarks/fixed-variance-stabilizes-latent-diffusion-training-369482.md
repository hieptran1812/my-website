---
title: "Fixed Variance Stabilizes Latent Diffusion Training"
publishDate: "2026-02-23"
date: "2026-02-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2025823351743369482-1.jpg"
excerpt: "A comment explains that hard-coding variance in a VAE encoder enables stable end-to-end training for latent diffusion."
---

> Source: [@idanbeck on X](https://x.com/idanbeck/status/2025823351743369482) · posted 2026-02-23

![Fixed Variance Stabilizes Latent Diffusion Training](/imgs/blogs/x-bookmarks/2025823351743369482-1.jpg)

## Summary

The author says they hard coded the variance, so the VAE encoder only predicts the mean latent, and a scaled identity covariance is used for the reparameterization trick. That removes instability and allows end-to-end training, crediting Salimans. The quoted post says Google DeepMind fixed a dirty problem in latent image generation.

## Background

Latent diffusion models work in a compressed space from a VAE encoder. The reparameterization trick samples latents using mean and variance. Fixing variance simplifies this and removes a source of instability.

## What I took away

- Simple fixes can remove training instability.
- This is an interpretation of the paper by the author.

## Reference

- Original post: [https://x.com/idanbeck/status/2025823351743369482](https://x.com/idanbeck/status/2025823351743369482)
- Author: [@idanbeck](https://x.com/idanbeck) on X, posted 2026-02-23
