---
title: "Extracting Python Functions From LLM Weights"
publishDate: "2025-12-08"
date: "2025-12-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1998128094360846391-1.jpg"
excerpt: "A project building on OpenAI's sparse-circuit interpretability work to extract Python functions that describe model behavior."
---

> Source: [@neelsomani on X](https://x.com/neelsomani/status/1998128094360846391) · posted 2025-12-08

![Extracting Python Functions From LLM Weights (1)](/imgs/blogs/x-bookmarks/1998128094360846391-1.jpg)

![Extracting Python Functions From LLM Weights (2)](/imgs/blogs/x-bookmarks/1998128094360846391-2.jpg)

## Summary

@neelsomani releases an extension of OpenAI's recent interpretability work. The project extracts Python functions from an LLM to show behaviors encoded in its weights, such as quotation closing, with a demo and code. The quoted OpenAI-related post describes training models with extremely sparse weights so tiny circuits for simple behaviors can be isolated and understood.

## Background

Interpretability tries to explain what is happening inside a neural network. A circuit is a small group of weights that performs one specific behavior. If weights are very sparse, these circuits are easier to find and read. Turning a circuit into a Python function gives a human-readable description.

## What I took away

- Sparse weights can make internal circuits easier to inspect.
- Expressing behavior as code is a readable form of interpretability.
- Demo and code are available for hands-on exploration.

## Reference

- Original post: [https://x.com/neelsomani/status/1998128094360846391](https://x.com/neelsomani/status/1998128094360846391)
- Author: [@neelsomani](https://x.com/neelsomani) on X, posted 2025-12-08
