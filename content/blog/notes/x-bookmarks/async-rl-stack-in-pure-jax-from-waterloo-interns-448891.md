---
title: "Async RL Stack in Pure JAX From Waterloo Interns"
publishDate: "2026-09-08"
date: "2026-09-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "tools", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "Interns built a lightweight async RL stack in pure JAX and shared a work log on inference, RDMA weight transfer and sharding."
---

> Source: [@IKorovinsky on X](https://x.com/IKorovinsky/status/2097160183214448891) · posted 2026-09-08

## Summary

The quoted authors built a lightweight, performant async RL stack in pure JAX. They share a work log covering inference, RDMA weight transfer, memory optimization, sharding and other JAX lessons. The author says never sleep on Waterloo interns.

## Background

Async RL lets generation and training run at the same time instead of waiting for each other. RDMA (remote direct memory access) copies memory between machines without involving the CPU, making weight sync fast. Sharding splits tensors across devices.

## What I took away

- Read the work log for practical JAX RL engineering tips.
- Async designs keep hardware busy during RL.
- Fast weight transfer is a recurring theme in RL infrastructure.

## Reference

- Original post: [https://x.com/IKorovinsky/status/2097160183214448891](https://x.com/IKorovinsky/status/2097160183214448891)
- Author: [@IKorovinsky](https://x.com/IKorovinsky) on X, posted 2026-09-08
