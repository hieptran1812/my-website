---
title: "Autoskill: A Distributed Skill Factory"
publishDate: "2026-03-12"
date: "2026-03-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "ai-research", "tools"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2032224933837684932-1.jpg"
excerpt: "Varun Mathur announces Autoskill, applying Karpathy's autoresearch loop to building agent skills, following his Autosearcher post."
---

> Source: [@varun_mathur on X](https://x.com/varun_mathur/status/2032224933837684932) · posted 2026-03-12

![Autoskill: A Distributed Skill Factory (1)](/imgs/blogs/x-bookmarks/2032224933837684932-1.jpg)

![Autoskill: A Distributed Skill Factory (2)](/imgs/blogs/x-bookmarks/2032224933837684932-2.png)

## Summary

The post announces **Autoskill**, a distributed skill factory at version 2.6.5. It applies the autoresearch pattern from @karpathy to skills. It quotes an earlier post about Autosearcher, a distributed search engine built on the same idea: give an agent a metric and a propose, run, evaluate, keep or revert loop.

## Background

Autoresearch is a pattern where an agent repeatedly tries a change, measures it against a metric, and keeps it only if the metric improves. Skills are reusable instruction packages that agents can load. The idea is to let agents improve skills automatically using that same loop.

## What I took away

- The core pattern is a metric plus a tight propose, run, evaluate, keep or revert loop.
- The pattern is being reused beyond ML training, here for skills and search.
- A clear, automatic metric is what makes this kind of loop work.

## Reference

- Original post: [https://x.com/varun_mathur/status/2032224933837684932](https://x.com/varun_mathur/status/2032224933837684932)
- Author: [@varun_mathur](https://x.com/varun_mathur) on X, posted 2026-03-12
