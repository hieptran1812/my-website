---
title: "Learning Flow Model Likelihoods With BoltzNCE"
publishDate: "2025-11-18"
date: "2025-11-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "ai-research", "math"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1990838131281506365-1.jpg"
excerpt: "A NeurIPS 2025 paper asks whether a second model can learn the likelihoods of flow-matching and diffusion models."
---

> Source: [@AggarwalRishal on X](https://x.com/AggarwalRishal/status/1990838131281506365) · posted 2025-11-18

![Learning Flow Model Likelihoods With BoltzNCE](/imgs/blogs/x-bookmarks/1990838131281506365-1.jpg)

## Summary

The author asks whether the likelihoods of flow-matching or diffusion models can be learned by another model, so the costly change-of-variable equation never has to be computed. Their NeurIPS 2025 paper, BoltzNCE, claims this is possible. They use it to recover the Boltzmann distribution, and the post starts a thread.

## Background

Flow and diffusion models generate samples, but getting the exact probability of a sample requires an expensive calculation using the change-of-variable formula. A learned likelihood estimator would make that cheap. The Boltzmann distribution describes how likely physical states are by their energy, so this matters for molecular simulation.

## What I took away

- Exact likelihoods from generative flows are costly; a learned estimator is one way around it.
- The result is the authors' claim; read the paper for the evidence.
- Boltzmann distribution sampling is a key use case in physics and chemistry.

## Reference

- Original post: [https://x.com/AggarwalRishal/status/1990838131281506365](https://x.com/AggarwalRishal/status/1990838131281506365)
- Author: [@AggarwalRishal](https://x.com/AggarwalRishal) on X, posted 2025-11-18
