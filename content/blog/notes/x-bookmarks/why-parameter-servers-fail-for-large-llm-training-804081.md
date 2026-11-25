---
title: "Why Parameter Servers Fail For Large LLM Training"
publishDate: "2025-11-23"
date: "2025-11-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "learning-resources", "career"]
author: "Hiep Tran"
featured: false
excerpt: "An interview-style post saying a central parameter server is the wrong answer for synchronizing 1,000 GPUs."
---

> Source: [@akshay_pachaar on X](https://x.com/akshay_pachaar/status/1992571349332804081) · posted 2025-11-23

## Summary

The post sets up a Google ML interview question about training an LLM across 1,000 GPUs. It says answering with a central parameter server to aggregate and redistribute weights would end the interview, and promises to explain what was missed.

## Background

A parameter server is one machine or group that collects updates and sends back weights, which becomes a bandwidth bottleneck at scale. Large training jobs usually use collective operations such as all-reduce, where GPUs exchange gradients directly with each other. The post itself only teases the answer.

## What I took away

- A central server can become a network bottleneck with many GPUs.
- Look into all-reduce and data parallelism for distributed training.
- Interview answers should address scale and communication cost.

## Reference

- Original post: [https://x.com/akshay_pachaar/status/1992571349332804081](https://x.com/akshay_pachaar/status/1992571349332804081)
- Author: [@akshay_pachaar](https://x.com/akshay_pachaar) on X, posted 2025-11-23
