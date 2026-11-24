---
title: "Aligning Image and Text Encoders Without Paired Data"
publishDate: "2026-11-02"
date: "2026-11-02"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "world-models"]
author: "Hiep Tran"
featured: false
excerpt: "Separately trained image and text encoders can be aligned with one rotation matrix because they share geometric structure."
---

> Source: [@miyashita_03 on X](https://x.com/miyashita_03/status/2117225216300765688) · posted 2026-11-02

## Summary

The quoted post, studying the latest JEPA paper, says you can take separately trained image and text encoders, fit just one rotation matrix, and get an image captioner without training on image-caption pairs, citing the Platonic Representation idea. @miyashita_03 adds that a paper discusses exactly this: the encoders can be aligned without paired data because they share similar geometric structure.

## Background

An encoder turns inputs into vectors. The Platonic Representation Hypothesis says different models trained on different data converge to similar internal representations of the world. If so, a simple rotation can map one space onto the other.

## What I took away

- Different models may learn similar geometry of concepts.
- Cheap alignment methods could reduce the need for paired training data.
- Try simple linear maps between embedding spaces before training new models.

## Reference

- Original post: [https://x.com/miyashita_03/status/2117225216300765688](https://x.com/miyashita_03/status/2117225216300765688)
- Author: [@miyashita_03](https://x.com/miyashita_03) on X, posted 2026-11-02
