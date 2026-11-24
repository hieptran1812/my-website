---
title: "Revisiting Very Small Neural Nets for CPU Workloads"
publishDate: "2026-09-07"
date: "2026-09-07"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "llm-architecture", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "The author had Claude Code build a 10k tok/s CPU model for data processing and shares a paper on what it found."
---

> Source: [@GregoryDiamos on X](https://x.com/GregoryDiamos/status/2096873745420075020) · posted 2026-09-07

## Summary

The author argues for revisiting very small neural networks. They needed a 10k tok/s CPU model for data processing, so they gave Claude Code a large token budget to build one. They say it made three interesting discoveries, described in a paper on Hugging Face.

## Background

For bulk data processing, a tiny model on a CPU can be cheaper than a GPU-hosted LLM. AMX is Intel's matrix extension for CPUs, which the repo name suggests is used. Letting a coding agent explore model design is a new research workflow.

## What I took away

- Tiny CPU models can be enough for high-volume simple tasks.
- Coding agents can run small research explorations for you.
- Read the paper for the three discoveries.

## Reference

- Original post: [https://x.com/GregoryDiamos/status/2096873745420075020](https://x.com/GregoryDiamos/status/2096873745420075020)
- Author: [@GregoryDiamos](https://x.com/GregoryDiamos) on X, posted 2026-09-07
