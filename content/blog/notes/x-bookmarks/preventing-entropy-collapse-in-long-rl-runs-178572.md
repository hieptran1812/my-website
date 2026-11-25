---
title: "Preventing Entropy Collapse In Long RL Runs"
publishDate: "2026-01-12"
date: "2026-01-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training"]
author: "Hiep Tran"
featured: false
excerpt: "A question asking for tricks to prevent entropy collapse in long RL runs for LLMs."
---

> Source: [@agarwl_ on X](https://x.com/agarwl_/status/2010848064039178572) · posted 2026-01-12

## Summary

The author asks the community for ideas to stop entropy collapse during very long RL runs. One known trick is optimizing best-of-N or pass@N rewards. They ask for papers.

## Background

Entropy collapse is when the policy becomes too deterministic and stops exploring, so training stalls.

## What I took away

- Maintain exploration during long RL.
- Collect papers on entropy control.

## Reference

- Original post: [https://x.com/agarwl_/status/2010848064039178572](https://x.com/agarwl_/status/2010848064039178572)
- Author: [@agarwl_](https://x.com/agarwl_) on X, posted 2026-01-12
