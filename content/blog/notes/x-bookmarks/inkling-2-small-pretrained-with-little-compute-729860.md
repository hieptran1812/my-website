---
title: "Inkling 2 Small Pretrained With Little Compute"
publishDate: "2026-11-06"
date: "2026-11-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "moe", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "Praise for Inkling 2 Small, a new MoE model pretrained from scratch on only 12T tokens in a few dozen days."
---

> Source: [@eliebakouch on X](https://x.com/eliebakouch/status/2118512391331729860) · posted 2026-11-06

## Summary

Thinking Machines introduced Inkling 2 Small, scoring 31 on the AA Intelligence Index, available for fine-tuning and inference on Tinker, with a full weight release planned. @eliebakouch calls it extremely impressive: a new pretrain from scratch with 17B active and 594B total parameters, trained in a few dozen days on only 12T tokens, using much less compute and data than competing models.

## Background

Active versus total parameters is a Mixture-of-Experts idea: the model holds many experts (594B total) but uses only a few per token (17B active), so it is cheaper to run than its size suggests. Training on fewer tokens with less compute and still competing means the training recipe is efficient.

## What I took away

- Compare models by active parameters and training tokens, not just total size.
- Data and compute efficiency in pretraining is becoming a key competitive edge.
- Watch for the planned full weight release to verify the claims.

## Reference

- Original post: [https://x.com/eliebakouch/status/2118512391331729860](https://x.com/eliebakouch/status/2118512391331729860)
- Author: [@eliebakouch](https://x.com/eliebakouch) on X, posted 2026-11-06
