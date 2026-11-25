---
title: "Hamiltonian Monte Carlo As A Physics Problem"
publishDate: "2025-11-14"
date: "2025-11-14"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "math", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1989336536446390539-1.jpg"
excerpt: "A short explanation of Hamiltonian Monte Carlo, which samples distributions by simulating particles with momentum."
---

> Source: [@alec_helbling on X](https://x.com/alec_helbling/status/1989336536446390539) · posted 2025-11-14

![Hamiltonian Monte Carlo As A Physics Problem](/imgs/blogs/x-bookmarks/1989336536446390539-1.jpg)

## Summary

The author explains that Hamiltonian Monte Carlo frames sampling from a probability distribution as a physics problem. Particles are given momentum, and their energy and motion are simulated with Hamilton's equations to explore the distribution efficiently.

## Background

Simple random-walk samplers wander slowly. HMC uses gradients to move along smooth paths, so it takes long, informed steps and accepts them with high probability. It is the engine behind tools such as Stan.

## What I took away

- Treat sampling as moving a particle through an energy landscape.
- Gradients make exploration much more efficient.
- Useful for Bayesian inference.

## Reference

- Original post: [https://x.com/alec_helbling/status/1989336536446390539](https://x.com/alec_helbling/status/1989336536446390539)
- Author: [@alec_helbling](https://x.com/alec_helbling) on X, posted 2025-11-14
