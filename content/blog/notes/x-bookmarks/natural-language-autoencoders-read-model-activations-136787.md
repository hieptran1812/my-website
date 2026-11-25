---
title: "Natural Language Autoencoders Read Model Activations"
publishDate: "2026-05-08"
date: "2026-05-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "ai-safety", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2052556477857136787-1.jpg"
excerpt: "A researcher finds Anthropic's Natural Language Autoencoders interesting because they translate activations into plain language and may detect reward hacking."
---

> Source: [@eliebakouch on X](https://x.com/eliebakouch/status/2052556477857136787) · posted 2026-05-08

![Natural Language Autoencoders Read Model Activations (1)](/imgs/blogs/x-bookmarks/2052556477857136787-1.jpg)

![Natural Language Autoencoders Read Model Activations (2)](/imgs/blogs/x-bookmarks/2052556477857136787-2.jpg)

## Summary

The author says it is fascinating that they train an encoder and decoder, each an LLM matching the target model's shape, so the latent space is plain language. They say it can detect reward hacking and unwanted behavior and might be used as an eval. The quoted Anthropic post says Claude is trained to translate its activations into human-readable text.

## Background

Activations are the internal numbers a model computes. They are hard to read directly. An autoencoder compresses and reconstructs data, so a language-based latent space gives readable text explanations.

## What I took away

- Readable explanations of activations could help audit models.
- Check that the text faithfully reflects what the model computes.

## Reference

- Original post: [https://x.com/eliebakouch/status/2052556477857136787](https://x.com/eliebakouch/status/2052556477857136787)
- Author: [@eliebakouch](https://x.com/eliebakouch) on X, posted 2026-05-08
