---
title: "Turning Qwen3.8-27B Into a Fast Decision Model"
publishDate: "2026-09-30"
date: "2026-09-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "llm-inference", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "SGLang shows a Qwen3.8-27B multimodal decision model that beat Pokemon FireRed and announces a native /v1/decisions endpoint."
---

> Source: [@sgl_project on X](https://x.com/sgl_project/status/2105095379272515603) · posted 2026-09-30

## Summary

The SGLang team says they turned **Qwen3.8-27B** into a multimodal decision model that beat the elite four and champion in Pokemon FireRed. It made decisions in under 100 ms from live game state. They also announce a native **/v1/decisions** endpoint in SGLang that lets LLMs and VLMs act as classification and scoring models.

## Background

A VLM is a vision-language model that takes images and text as input. Normally an LLM generates text token by token, but for classification or scoring you only need one output, like picking an action or a label. Serving that as a single fast call avoids slow text generation and makes LLMs usable in real-time loops.

## What I took away

- If you only need a label or score from an LLM, a decision or classification endpoint can be much faster than free text generation.
- Games with live state are a useful test bed for low-latency model decisions.
- Check whether your serving stack already supports scoring APIs before building custom heads.

## Reference

- Original post: [https://x.com/sgl_project/status/2105095379272515603](https://x.com/sgl_project/status/2105095379272515603)
- Author: [@sgl_project](https://x.com/sgl_project) on X, posted 2026-09-30
