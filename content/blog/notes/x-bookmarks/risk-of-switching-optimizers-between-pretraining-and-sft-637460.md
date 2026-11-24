---
title: "Risk of Switching Optimizers Between Pretraining and SFT"
publishDate: "2026-10-15"
date: "2026-10-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "pretraining", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "A researcher asks whether Dimuon's authors studied optimizer mismatch between pretraining and fine-tuning, citing Moonshot's Muon findings."
---

> Source: [@eliebakouch on X](https://x.com/eliebakouch/status/2110633823662637460) · posted 2026-10-15

## Summary

@eliebakouch wonders whether the Dimuon team tested using different optimizers in pretraining and SFT, since Moonshot's Muon paper showed that changing optimizer during SFT can lose many benchmark points. He guesses it may matter less in some cases (the text is cut off). He quotes the Tinker announcement of Dimuon, a LoRA optimizer that often reaches lower loss than AdamW.

## Background

Models pretrained with one optimizer may fine-tune best with the same one, because optimizers shape the weights differently. Lower training loss does not guarantee better benchmarks. This is a caution against the claim in the Dimuon note.

## What I took away

- Match fine-tuning optimizer to the pretraining optimizer unless tests show otherwise.
- Evaluate on benchmarks, not only training loss, when changing optimizers.

## Reference

- Original post: [https://x.com/eliebakouch/status/2110633823662637460](https://x.com/eliebakouch/status/2110633823662637460)
- Author: [@eliebakouch](https://x.com/eliebakouch) on X, posted 2026-10-15
