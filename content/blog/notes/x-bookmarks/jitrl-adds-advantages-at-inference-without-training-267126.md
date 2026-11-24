---
title: "JitRL Adds Advantages at Inference Without Training"
publishDate: "2026-08-08"
date: "2026-08-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "ai-agents"]
author: "Hiep Tran"
featured: false
excerpt: "A recommendation of Qingke AI's talks, sharing NUS research on Just-In-Time Reinforcement Learning."
---

> Source: [@sheriyuo on X](https://x.com/sheriyuo/status/2086126637268267126) · posted 2026-08-08

## Summary

The author feels Qingke AI deserves more attention on X and says its selected talks are good. The quoted post describes JitRL (Just-In-Time Reinforcement Learning) from the National University of Singapore. Without retraining or gradient updates, it retrieves past experiences during inference, estimates action advantages on the spot, and adds those advantages directly to the model's decision process.

## Background

In RL, an advantage measures how much better an action is than the average action in that situation. Normally advantages are used to update model weights. JitRL instead uses retrieved past experience to adjust choices at inference time, similar to retrieval-augmented generation but for decisions.

## What I took away

- Some RL-like improvements can happen at inference time without retraining.
- Retrieval of past experience is a cheap way to adapt agents.
- Check how this compares with actual fine-tuning on harder tasks.

## Reference

- Original post: [https://x.com/sheriyuo/status/2086126637268267126](https://x.com/sheriyuo/status/2086126637268267126)
- Author: [@sheriyuo](https://x.com/sheriyuo) on X, posted 2026-08-08
