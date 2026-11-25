---
title: "Diagnosing RAG Hallucinations In An Interview"
publishDate: "2025-12-20"
date: "2025-12-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "llm-inference", "career"]
author: "Hiep Tran"
featured: false
excerpt: "An interview question asks how to tell whether a hallucinating RAG system's problem is the retriever or the generator."
---

> Source: [@athletic_coder on X](https://x.com/athletic_coder/status/2002355874786873383) · posted 2025-12-20

## Summary

The post frames an ML engineer interview question: your RAG system is hallucinating in production, so how do you tell if the retriever or the generator is at fault? The answer follows in the thread.

## Background

RAG retrieves documents and feeds them to a model to answer. A bad answer can come from retrieving the wrong passages or from the model ignoring or misusing good ones. Checking retrieval quality separately from generation isolates the cause.

## What I took away

- Evaluate retrieval and generation separately.
- First check whether the right documents were retrieved.
- Then check whether the answer is grounded in them.

## Reference

- Original post: [https://x.com/athletic_coder/status/2002355874786873383](https://x.com/athletic_coder/status/2002355874786873383)
- Author: [@athletic_coder](https://x.com/athletic_coder) on X, posted 2025-12-20
