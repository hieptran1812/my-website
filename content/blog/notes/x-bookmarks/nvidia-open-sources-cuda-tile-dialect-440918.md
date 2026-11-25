---
title: "NVIDIA Open-Sources CUDA Tile Dialect"
publishDate: "2025-12-19"
date: "2025-12-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "tools", "hardware"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2002115003189440918-1.jpg"
excerpt: "NVIDIA's team open-sources the CUDA Tile MLIR dialect and bytecode with Python bindings."
---

> Source: [@roeschinc on X](https://x.com/roeschinc/status/2002115003189440918) · posted 2025-12-19

![NVIDIA Open-Sources CUDA Tile Dialect](/imgs/blogs/x-bookmarks/2002115003189440918-1.jpg)

## Summary

The author announces open-sourcing the CUDA Tile dialect and bytecode. It includes the CUDA Tile MLIR dialect, bytecode serialization and deserialization support, and MLIR Python bindings for building the IR programmatically. The repo describes CUDA Tile IR as an MLIR-based intermediate representation.

## Background

MLIR is a compiler framework for defining intermediate representations. A tile-based IR lets a compiler reason about GPU work in blocks of data rather than individual threads. Open-sourcing it lets others build tooling on top.

## What I took away

- CUDA Tile IR is now open for inspection and tooling.
- Tile-level abstractions are a trend in GPU programming.
- Python bindings make it easy to experiment.

## Reference

- Original post: [https://x.com/roeschinc/status/2002115003189440918](https://x.com/roeschinc/status/2002115003189440918)
- Author: [@roeschinc](https://x.com/roeschinc) on X, posted 2025-12-19
