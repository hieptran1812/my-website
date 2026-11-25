---
title: "RL Agent Trading Polymarket Crypto Markets"
publishDate: "2025-12-30"
date: "2025-12-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "trading", "crypto"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2006020259652772269-1.jpg"
excerpt: "An experiment trains a PPO agent to trade Polymarket 15-minute crypto markets using Binance order flow."
---

> Source: [@nikshepsvn on X](https://x.com/nikshepsvn/status/2006020259652772269) · posted 2025-12-30

![RL Agent Trading Polymarket Crypto Markets (1)](/imgs/blogs/x-bookmarks/2006020259652772269-1.jpg)

![RL Agent Trading Polymarket Crypto Markets (2)](/imgs/blogs/x-bookmarks/2006020259652772269-2.jpg)

## Summary

Nikshep shares code, a training journal and a model for an experiment in cross-market state fusion. An RL agent fuses Binance order flow with Polymarket microstructure to trade. He reports early results with rising PnL and signs of learning. The quoted post says inputs are 18 features, outputs are 7 actions, and it uses PPO on MLX, training live in about 34 minute cycles.

## Background

PPO is a common reinforcement learning algorithm. Order flow means the stream of buy and sell orders, and microstructure means the fine details of an order book. Early results from live training are not proof of a working strategy. This is an educational experiment, not advice.

## What I took away

- The setup shows how to frame trading as an RL problem with a small action space.
- Early PnL gains are weak evidence; look for out-of-sample tests.
- Open code and a training journal make it easy to learn from.

## Reference

- Original post: [https://x.com/nikshepsvn/status/2006020259652772269](https://x.com/nikshepsvn/status/2006020259652772269)
- Author: [@nikshepsvn](https://x.com/nikshepsvn) on X, posted 2025-12-30
