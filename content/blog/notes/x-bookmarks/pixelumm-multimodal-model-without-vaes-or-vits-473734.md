---
title: "PixelUMM Multimodal Model Without VAEs or ViTs"
publishDate: "2026-10-05"
date: "2026-10-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "open-models", "audio-video"]
author: "Hiep Tran"
featured: false
excerpt: "NVIDIA open-sourced PixelUMM, a unified multimodal model that works directly on raw pixels."
---

> Source: [@CongWei1230 on X](https://x.com/CongWei1230/status/2106906342552473734) · posted 2026-10-05

## Summary

An author thanks a sharer and links the paper, code and model for PixelUMM. The quoted post says NVIDIA released PixelUMM, a unified multimodal model that removes VAEs and ViTs and operates directly in raw pixel space for image and video tasks.

## Background

Most multimodal models encode images with a ViT (vision transformer) for understanding or a VAE (variational autoencoder) for generation. Working in raw pixels removes these extra components, which can simplify the design.

## What I took away

- Code and weights are open; try it if you work on unified models.
- The quoted post's framing is hype-heavy; read the paper for details.

## Reference

- Original post: [https://x.com/CongWei1230/status/2106906342552473734](https://x.com/CongWei1230/status/2106906342552473734)
- Author: [@CongWei1230](https://x.com/CongWei1230) on X, posted 2026-10-05
