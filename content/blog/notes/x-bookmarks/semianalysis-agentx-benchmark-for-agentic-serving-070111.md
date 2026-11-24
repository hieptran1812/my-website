---
title: "SemiAnalysis AgentX Benchmark for Agentic Serving"
publishDate: "2026-08-25"
date: "2026-08-25"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "llm-inference", "ai-agents"]
author: "Hiep Tran"
featured: false
excerpt: "The author congratulates SemiAnalysis on AgentX, a realistic agentic serving benchmark, and notes vLLM's competitive results."
---

> Source: [@yifandotqiao on X](https://x.com/yifandotqiao/status/2092045366179070111) · posted 2026-08-25

## Summary

The author congratulates SemiAnalysis on launching **AgentX** and says they have wanted a realistic benchmark for agentic serving. They are glad vLLM performed competitively from day one, crediting many optimizations by named contributors. The quoted vLLM post describes AgentX 1.0 as an open-source multi-turn agentic coding benchmark built from about $3M of real traces, running on 1000+ chips and about 2MW of continuous compute.

## Background

Agentic serving differs from simple chat: requests are multi-turn, have very long shared histories and depend heavily on prefix caching. Benchmarks built from real agent traces test these patterns, which synthetic benchmarks miss.

## What I took away

- Benchmark serving engines on realistic agent traces, not just single prompts.
- Prefix cache performance dominates agentic workloads.
- Results from a vendor's own post should be checked against the benchmark data.

## Reference

- Original post: [https://x.com/yifandotqiao/status/2092045366179070111](https://x.com/yifandotqiao/status/2092045366179070111)
- Author: [@yifandotqiao](https://x.com/yifandotqiao) on X, posted 2026-08-25
