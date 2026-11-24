---
title: "Interactive Exercises for Learning Pipeline Parallel Schedules"
publishDate: "2026-08-12"
date: "2026-08-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "learning-resources", "pretraining", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "Edward Yang shares interactive exercises where you build pipeline parallelism schedules block by block."
---

> Source: [@ezyang on X](https://x.com/ezyang/status/2087614647482318937) · posted 2026-08-12

## Summary

The author says he and Fable made a set of exercises for learning how pipeline parallelism schedules work, hosted on his GitHub Pages site. Unlike other simulators, it asks you to build the schedule block by block, subject to constraints, with a double click to greedily finish a microbatch.

## Background

Pipeline parallelism splits a model's layers across GPUs, so each GPU handles a stage. Batches are split into microbatches that flow through the stages; the schedule decides the order of forward and backward passes to reduce idle time (bubbles) and memory use.

## What I took away

- Building a schedule by hand is a good way to understand pipeline bubbles.
- Use the exercises before reading papers on schedules like 1F1B or zero-bubble.

## Reference

- Original post: [https://x.com/ezyang/status/2087614647482318937](https://x.com/ezyang/status/2087614647482318937)
- Author: [@ezyang](https://x.com/ezyang) on X, posted 2026-08-12
