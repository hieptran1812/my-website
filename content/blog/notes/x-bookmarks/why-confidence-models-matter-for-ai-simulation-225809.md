---
title: "Why Confidence Models Matter for AI Simulation"
publishDate: "2026-08-25"
date: "2026-08-25"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "ai-safety"]
author: "Hiep Tran"
featured: false
excerpt: "Percy Liang notes that Simile's first technical blog is about confidence, which is critical for trustworthy simulations."
---

> Source: [@percyliang on X](https://x.com/percyliang/status/2092302845987225809) · posted 2026-08-25

## Summary

Percy Liang finds it notable that Simile's first technical blog post is about confidence. He argues that when a coding agent fails you can usually tell and fix it, but a wrong simulation may go unnoticed and lead to bad decisions. The quoted Simile post says it trains simulation models and confidence models, and its first blog explores the origin of its proprietary confidence model.

## Background

A simulation model predicts outcomes, such as how people would respond to something. A confidence model estimates how likely that prediction is to be right. Without it, users cannot tell good predictions from bad ones, which is risky when there is no ground truth to check.

## What I took away

- When outputs cannot be checked easily, calibrated confidence is essential.
- Pair predictive models with a separate estimate of reliability.
- Errors that cannot be seen are more dangerous than errors that can.

## Reference

- Original post: [https://x.com/percyliang/status/2092302845987225809](https://x.com/percyliang/status/2092302845987225809)
- Author: [@percyliang](https://x.com/percyliang) on X, posted 2026-08-25
