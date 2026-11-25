---
title: "Profiling OpenDataLoader PDF Latency"
publishDate: "2026-04-01"
date: "2026-04-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "tools", "gpu-kernels", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2039226913676468317-1.jpg"
excerpt: "OpenDataLoader describes profiling its PDF parser to find latency bottlenecks, which were not where they expected."
---

> Source: [@opendatalo51205 on X](https://x.com/opendatalo51205/status/2039226913676468317) · posted 2026-04-01

![Profiling OpenDataLoader PDF Latency](/imgs/blogs/x-bookmarks/2039226913676468317-1.jpg)

## Summary

The post says OpenDataLoader PDF is the fastest non-VLM parser that preserves document structure such as tables, headings and lists. They saw room to improve latency and profiled it. They expected the XY-Cut++ sorting algorithm to be the bottleneck but it turned out to be less than the expected share. The text is cut off.

## Background

A non-VLM parser does not use a vision-language model. XY-Cut++ is a layout sorting algorithm for reading order. The lesson is that profiling often disproves assumptions about slow code.

## What I took away

- Profile before optimizing; the bottleneck is often elsewhere.
- Document structure preservation matters for LLM pipelines.
- Read the full thread for what the real cost was.

## Reference

- Original post: [https://x.com/opendatalo51205/status/2039226913676468317](https://x.com/opendatalo51205/status/2039226913676468317)
- Author: [@opendatalo51205](https://x.com/opendatalo51205) on X, posted 2026-04-01
