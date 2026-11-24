---
title: "Hand-Writing a Classifier's Weights With an LLM"
publishDate: "2026-11-03"
date: "2026-11-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "interpretability"]
author: "Hiep Tran"
featured: false
excerpt: "A user says Fable hand-wrote weights for a 100k-parameter model to tell real from fake legal citations."
---

> Source: [@jacsebl on X](https://x.com/jacsebl/status/2117606206764175543) · posted 2026-11-03

## Summary

@jacsebl says he got Fable to hand-write the weights of a 100k-parameter model that recognizes real versus fake legal citations. He claims it basically one-shotted a full neural network that is better than training it.

## Background

Normally network weights are found by training on data with gradient descent. Here an LLM is asked to write the weights directly, like writing a program. The claim that it beats training is the author's and has no reported evaluation details in the post.

## What I took away

- LLMs writing weights directly is an emerging experiment worth watching.
- Ask how the hand-written model was evaluated and what baseline was used.
- Small, structured tasks are the most likely to work with hand-written weights.

## Reference

- Original post: [https://x.com/jacsebl/status/2117606206764175543](https://x.com/jacsebl/status/2117606206764175543)
- Author: [@jacsebl](https://x.com/jacsebl) on X, posted 2026-11-03
