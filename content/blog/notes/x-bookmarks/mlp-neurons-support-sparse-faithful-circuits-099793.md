---
title: "MLP Neurons Support Sparse Faithful Circuits"
publishDate: "2025-11-20"
date: "2025-11-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "ai-research", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1991582415891099793-1.jpg"
excerpt: "Transluce reports that raw MLP neurons can form sparse, faithful circuits, challenging the need for learned features."
---

> Source: [@TransluceAI on X](https://x.com/TransluceAI/status/1991582415891099793) · posted 2025-11-20

![MLP Neurons Support Sparse Faithful Circuits](/imgs/blogs/x-bookmarks/1991582415891099793-1.jpg)

## Summary

Transluce asks whether a language model is secretly an SAE. They note that most circuit-finding methods use learned features instead of raw activations, because people believe neurons do not cleanly split computation. In their new work they say MLP neurons do support sparse, faithful circuits.

## Background

An SAE (sparse autoencoder) learns a set of features from a model's activations that are easier to interpret than raw neurons. A circuit is a small set of components that together explain a behavior. Faithful means the circuit really reproduces the model's behavior. If neurons work, interpretability could skip the extra training step.

## What I took away

- Learned features are not the only route to circuits; raw neurons may be enough.
- Faithfulness and sparsity are the two properties to check in any circuit claim.
- This challenges a common assumption in interpretability research.

## Reference

- Original post: [https://x.com/TransluceAI/status/1991582415891099793](https://x.com/TransluceAI/status/1991582415891099793)
- Author: [@TransluceAI](https://x.com/TransluceAI) on X, posted 2025-11-20
