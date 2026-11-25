---
title: "Concept 12: Hessian Spectrum And Learning Dynamics"
publishDate: "2025-11-19"
date: "2025-11-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "ai-research", "math"]
author: "Hiep Tran"
featured: false
excerpt: "A post in a series on how loss curvature changes in training and why large models stay stable."
---

> Source: [@chastronomic on X](https://x.com/chastronomic/status/1991186274657661206) · posted 2025-11-19

## Summary

The author says an ML engineer who does not understand how the curvature of the loss changes during training is missing why large models stay stable while smaller ones spike, diverge or overfit. This is Concept 12, The Hessian Spectrum and the Shape of Learning Dynamics.

## Background

The Hessian is the matrix of second derivatives of the loss and describes its curvature. Its largest eigenvalues limit how big a learning rate can be before training becomes unstable. The claim about large versus small models is the author's.

## What I took away

- Curvature limits a safe learning rate.
- Loss spikes can be linked to sharp curvature.
- Treat the large-versus-small claim as unverified.

## Reference

- Original post: [https://x.com/chastronomic/status/1991186274657661206](https://x.com/chastronomic/status/1991186274657661206)
- Author: [@chastronomic](https://x.com/chastronomic) on X, posted 2025-11-19
