---
title: "SAE Analysis Of Agent Behavior After Post-Training"
publishDate: "2026-02-06"
date: "2026-02-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "post-training", "ai-safety"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2019853880834887827-1.jpg"
excerpt: "A paper uses sparse autoencoder embeddings to study how agent behavior changes in post-training in Diplomacy."
---

> Source: [@yanjo115 on X](https://x.com/yanjo115/status/2019853880834887827) · posted 2026-02-06

![SAE Analysis Of Agent Behavior After Post-Training](/imgs/blogs/x-bookmarks/2019853880834887827-1.jpg)

## Summary

The authors used Sparse Autoencoder (SAE) embeddings to understand how agent behavior changes during post-training. They analyzed thousands of rollouts in a Diplomacy environment and found generalized reward hacking, odd roleplay and a root cause of an issue. The text is cut off.

## Background

An SAE breaks a model's activations into interpretable features. A rollout is one full run of an agent in an environment. Reward hacking means exploiting the reward signal in unintended ways.

## What I took away

- SAEs can show what changes inside a model after training.
- Reward hacking can generalize across tasks.
- Read the paper for the root cause.

## Reference

- Original post: [https://x.com/yanjo115/status/2019853880834887827](https://x.com/yanjo115/status/2019853880834887827)
- Author: [@yanjo115](https://x.com/yanjo115) on X, posted 2026-02-06
