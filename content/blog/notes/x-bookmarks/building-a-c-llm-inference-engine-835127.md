---
title: "Building A C++ LLM Inference Engine"
publishDate: "2025-11-16"
date: "2025-11-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "learning-resources", "serving"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1990008958266835127-1.jpg"
excerpt: "A daily progress log on writing an LLM inference engine in C++, covering sampling, tokenizer parsing and memory layout."
---

> Source: [@jino_rohit on X](https://x.com/jino_rohit/status/1990008958266835127) · posted 2025-11-16

![Building A C++ LLM Inference Engine](/imgs/blogs/x-bookmarks/1990008958266835127-1.jpg)

## Summary

The author lists today's work on an inference engine in C++. They implemented top-k, greedy and random sampling, cleaned the tokenizer JSON parsing using jsoncpp, and still need to design the memory layout for the next part. The text is cut off.

## Background

An inference engine runs a trained model to generate text. Sampling decides how the next token is chosen from the model's probabilities: greedy takes the top one, top-k samples among the best k. Memory layout matters because tensors and the KV cache dominate speed and memory use.

## What I took away

- Building an engine yourself teaches sampling, tokenization and memory design.
- Memory layout is a key design decision.
- Public progress logs are a good way to keep momentum.

## Reference

- Original post: [https://x.com/jino_rohit/status/1990008958266835127](https://x.com/jino_rohit/status/1990008958266835127)
- Author: [@jino_rohit](https://x.com/jino_rohit) on X, posted 2025-11-16
