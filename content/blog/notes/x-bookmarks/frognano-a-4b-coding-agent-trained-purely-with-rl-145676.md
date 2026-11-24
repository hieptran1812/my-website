---
title: "FrogNano: A 4B Coding Agent Trained Purely With RL"
publishDate: "2026-09-08"
date: "2026-09-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "ai-agents", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "FrogNano, Qwen3.5-4B post-trained only with RL on synthetic tasks, is claimed to reach 61.5% on SWE-bench Verified."
---

> Source: [@kim__minseon on X](https://x.com/kim__minseon/status/2097405882464145676) · posted 2026-09-08

## Summary

The author introduces **FrogNano**, Qwen3.5-4B post-trained purely with RL on synthetic tasks from TaskPilot, with no distillation. They claim 61.5% on SWE-bench Verified. Training used just 5 iterations of 300 tasks, and they argue coding agents do not have to be huge.

## Background

SWE-bench Verified tests whether a model can fix real GitHub issues. Distillation trains a small model on a larger model's outputs; skipping it means the gains come from RL alone. Synthetic tasks are generated automatically, which avoids needing human-made datasets.

## What I took away

- Small models can become capable agents with targeted RL.
- Synthetic task generation can be a cheap source of RL training data.
- Verify benchmark claims and check for contamination.

## Reference

- Original post: [https://x.com/kim__minseon/status/2097405882464145676](https://x.com/kim__minseon/status/2097405882464145676)
- Author: [@kim__minseon](https://x.com/kim__minseon) on X, posted 2026-09-08
