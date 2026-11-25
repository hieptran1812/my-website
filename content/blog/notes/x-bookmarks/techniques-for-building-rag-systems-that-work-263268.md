---
title: "Techniques For Building RAG Systems That Work"
publishDate: "2025-12-05"
date: "2025-12-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "llm-inference", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "A bookmarkable list of RAG techniques, from chunking strategies to reranking and context packing."
---

> Source: [@athletic_coder on X](https://x.com/athletic_coder/status/1996921635547263268) · posted 2025-12-05

## Summary

@athletic_coder lists techniques to master for RAG systems: sliding window chunking, semantic chunking, document hierarchies, metadata enrichment, query expansion, hybrid search, reranking models, context window packing, and handling the lost in the middle problem. The list continues past the shown text.

## Background

RAG (retrieval-augmented generation) fetches relevant documents and gives them to an LLM as context. Hybrid search combines keyword and vector search. Lost in the middle refers to models paying less attention to the middle of long contexts.

## What I took away

- RAG quality depends on chunking and retrieval more than on the model.
- Combine keyword and vector search, then rerank.
- Place key facts where the model will attend to them.

## Reference

- Original post: [https://x.com/athletic_coder/status/1996921635547263268](https://x.com/athletic_coder/status/1996921635547263268)
- Author: [@athletic_coder](https://x.com/athletic_coder) on X, posted 2025-12-05
