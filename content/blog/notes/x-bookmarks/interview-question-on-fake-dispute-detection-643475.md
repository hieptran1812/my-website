---
title: "Interview Question On Fake Dispute Detection"
publishDate: "2026-03-09"
date: "2026-03-09"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "learning-resources", "career", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2030899626048643475-1.png"
excerpt: "An ML interview scenario about detecting fake transaction disputes without labeled data."
---

> Source: [@_avichawla on X](https://x.com/_avichawla/status/2030899626048643475) · posted 2026-03-09

![Interview Question On Fake Dispute Detection](/imgs/blogs/x-bookmarks/2030899626048643475-1.png)

## Summary

The post sets up a Stripe ML interview: how to build a supervised model that predicts fake disputes when there is no labeled data. A naive answer, flagging cards with high dispute rates, is mocked as ending the interview.

## Background

Without labels you cannot train supervised models directly. Ideas include unsupervised or weakly supervised methods, heuristics for pseudo labels and careful review of a sample. Flagging by dispute rate alone may catch honest users.

## What I took away

- Think about how to create labels when none exist.
- Simple heuristics can be biased and unfair.
- Open the thread for the intended answer.

## Reference

- Original post: [https://x.com/_avichawla/status/2030899626048643475](https://x.com/_avichawla/status/2030899626048643475)
- Author: [@_avichawla](https://x.com/_avichawla) on X, posted 2026-03-09
