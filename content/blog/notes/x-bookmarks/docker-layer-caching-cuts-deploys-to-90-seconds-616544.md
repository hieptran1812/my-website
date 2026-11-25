---
title: "Docker Layer Caching Cuts Deploys To 90 Seconds"
publishDate: "2025-11-30"
date: "2025-11-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "tools", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "A post claims fixing Docker image layers cut deployment time from 12 minutes to 90 seconds."
---

> Source: [@brankopetric00 on X](https://x.com/brankopetric00/status/1994941980363616544) · posted 2025-11-30

## Summary

The author says fixing their Docker image layers reduced deployment time from 12 minutes to 90 seconds. The original Dockerfile rebuilt the whole image on every code change, installed dependencies each time, had no layer caching strategy, and produced a 1.2GB image, starting with a COPY of everything.

## Background

Docker builds in layers and reuses a layer if its inputs are unchanged. If you copy all your code before installing dependencies, any code edit invalidates the dependency layer. Copy dependency files first and install, then copy source.

## What I took away

- Copy dependency manifests and install before copying source code.
- Order Dockerfile steps from least to most frequently changed.
- Slim images also deploy faster.

## Reference

- Original post: [https://x.com/brankopetric00/status/1994941980363616544](https://x.com/brankopetric00/status/1994941980363616544)
- Author: [@brankopetric00](https://x.com/brankopetric00) on X, posted 2025-11-30
