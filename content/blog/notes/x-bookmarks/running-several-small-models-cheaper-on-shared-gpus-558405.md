---
title: "Running Several Small Models Cheaper on Shared GPUs"
publishDate: "2026-08-06"
date: "2026-08-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "llm-inference", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "A thread claiming self-hosting LLM pipelines can be about 75% cheaper, focused on agent pipelines that run several small models."
---

> Source: [@akshay_pachaar on X](https://x.com/akshay_pachaar/status/2085279733139558405) · posted 2026-08-06

## Summary

The author claims self-hosting LLMs just got about 75% cheaper. He notes that most agent pipelines now run 4 to 5 small models: an embedder for retrieval, a reranker for precision, an extractor for entities and often a small LLM for routine generation. The post is cut off where it begins to explain the standard way to deploy them.

## Background

An embedder turns text into vectors for search, and a reranker reorders search results by relevance. Deploying each small model on its own GPU wastes capacity, so sharing GPUs among several models is a common way to cut cost.

## What I took away

- Count how many models your pipeline really runs; each has a serving cost.
- Small models often underuse a whole GPU, so packing them together can save money.
- The 75% figure is the author's claim; measure on your own setup.

## Reference

- Original post: [https://x.com/akshay_pachaar/status/2085279733139558405](https://x.com/akshay_pachaar/status/2085279733139558405)
- Author: [@akshay_pachaar](https://x.com/akshay_pachaar) on X, posted 2026-08-06
