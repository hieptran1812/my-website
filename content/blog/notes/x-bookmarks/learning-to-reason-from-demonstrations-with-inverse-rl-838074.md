---
title: "Learning To Reason From Demonstrations With Inverse RL"
publishDate: "2025-12-20"
date: "2025-12-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2002419916490838074-1.jpg"
excerpt: "A paper summary proposes inverse RL to learn a critic from expert demonstrations, enabling RL where no verifier exists."
---

> Source: [@DanielCHTan97 on X](https://x.com/DanielCHTan97/status/2002419916490838074) · posted 2025-12-20

![Learning To Reason From Demonstrations With Inverse RL](/imgs/blogs/x-bookmarks/2002419916490838074-1.jpg)

## Summary

The author recommends a paper on learning to reason from demonstrations. Instead of supervised fine-tuning on expert demonstrations, it uses inverse RL to learn a critic, which allows RL on domains without natural verifiers. The card is titled Escaping the Verifier: Learning to Reason via Demonstrations.

## Background

RL for reasoning usually needs a verifier that checks answers, such as unit tests or math answers. Many real tasks lack one. Inverse RL infers a reward from expert examples, giving a learned critic that can stand in for a verifier.

## What I took away

- Learned critics can replace missing verifiers.
- This is an alternative to plain SFT on demonstrations.
- Useful for domains where answers cannot be checked automatically.

## Reference

- Original post: [https://x.com/DanielCHTan97/status/2002419916490838074](https://x.com/DanielCHTan97/status/2002419916490838074)
- Author: [@DanielCHTan97](https://x.com/DanielCHTan97) on X, posted 2025-12-20
