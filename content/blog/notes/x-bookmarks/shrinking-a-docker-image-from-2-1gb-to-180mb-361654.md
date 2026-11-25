---
title: "Shrinking A Docker Image From 2.1GB To 180MB"
publishDate: "2025-11-14"
date: "2025-11-14"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "A case study cutting a Docker image from 2.1GB to 180MB and making deployments 8x faster."
---

> Source: [@brankopetric00 on X](https://x.com/brankopetric00/status/1989294089150361654) · posted 2025-11-14

## Summary

The author says they reduced a Docker image from 2.1GB to 180MB and made deployments 8x faster. The original Dockerfile started with ubuntu:latest, installed everything via apt, included dev dependencies, copied the entire project, left build artifacts and did not optimize layers. The text is cut off before the fixes.

## Background

Large images are slow to build, push and pull. Common fixes include small base images, multi-stage builds, installing only runtime dependencies and ordering layers for caching. The numbers are the author's results.

## What I took away

- Avoid latest large base images.
- Leave dev dependencies and build artifacts out of the final image.
- Layer order affects caching.

## Reference

- Original post: [https://x.com/brankopetric00/status/1989294089150361654](https://x.com/brankopetric00/status/1989294089150361654)
- Author: [@brankopetric00](https://x.com/brankopetric00) on X, posted 2025-11-14
