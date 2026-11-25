---
title: "Rejection Sampling For On-Policy Accepted Tokens"
publishDate: "2026-07-07"
date: "2026-07-07"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2074527653844107436-1.jpg"
excerpt: "A researcher thanks a sharer of their former JD.com paper, which uses rejection sampling to treat accepted tokens as on-policy."
---

> Source: [@sheriyuo on X](https://x.com/sheriyuo/status/2074527653844107436) · posted 2026-07-07

![Rejection Sampling For On-Policy Accepted Tokens (1)](/imgs/blogs/x-bookmarks/2074527653844107436-1.jpg)

![Rejection Sampling For On-Policy Accepted Tokens (2)](/imgs/blogs/x-bookmarks/2074527653844107436-2.jpg)

## Summary

@sheriyuo thanks the sharer of a former work done at JD.com and says industry researchers found the view useful. The quoted post describes the paper's idea: using rejection sampling so that accepted tokens count as on-policy ones.

## Background

On-policy RL means training on samples produced by the current model. Rejection sampling keeps only samples that pass a check. Treating accepted tokens as on-policy lets data from a different sampler still be used in on-policy style training.

## What I took away

- Rejection sampling can be a way to make off-policy tokens usable as on-policy data.
- Industry demand for a clean framing of this problem seems real, according to the author.
- Check the arXiv paper 2607.04728 for the actual method.

## Reference

- Original post: [https://x.com/sheriyuo/status/2074527653844107436](https://x.com/sheriyuo/status/2074527653844107436)
- Author: [@sheriyuo](https://x.com/sheriyuo) on X, posted 2026-07-07
