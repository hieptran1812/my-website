---
title: "Needle 2 Tiny Agentic LLM for Edge Devices"
publishDate: "2026-08-10"
date: "2026-08-10"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quantization", "open-models", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "Cactus released Needle 2, a 14MB agentic LLM aimed at phones, wearables and microcontrollers."
---

> Source: [@cactuscompute on X](https://x.com/cactuscompute/status/2086865960669983035) · posted 2026-08-10

## Summary

Cactus released Needle 2, a 14MB agentic LLM for phones, wearables, smart home devices, robots and microcontrollers. The whole model is a single 14MB binary that runs a full session in 28MB of RAM. It is built on their Simple Attention Network findings and compressed to CQ2-bit with Cactus Quants.

## Background

Edge devices have very little memory and power, so models must be tiny and heavily quantized. Around 2-bit quantization stores each weight in about two bits, a big compression that usually costs some quality. An agentic model here likely means one that can call tools or functions.

## What I took away

- Very small models can handle narrow tasks like tool calling on-device.
- Test quality carefully at extreme quantization levels.
- Memory use during a session, not just file size, decides if a model fits.

## Reference

- Original post: [https://x.com/cactuscompute/status/2086865960669983035](https://x.com/cactuscompute/status/2086865960669983035)
- Author: [@cactuscompute](https://x.com/cactuscompute) on X, posted 2026-08-10
