---
title: "Value Functions For LLM Agents"
publishDate: "2025-12-20"
date: "2025-12-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "ai-agents", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2002523089301876788-1.jpg"
excerpt: "The author wonders how to use value functions for LLM agents, citing John Schulman's line that they are underrated."
---

> Source: [@wzenus on X](https://x.com/wzenus/status/2002523089301876788) · posted 2025-12-20

![Value Functions For LLM Agents](/imgs/blogs/x-bookmarks/2002523089301876788-1.jpg)

## Summary

The author keeps coming back to Schulman's line that value functions are underrated and thinks we have not figured out how to use them for LLM agents. For token-level LLMs the horizon is huge, and a naive value function on tokens makes variance blow up and credit assignment hard. The quoted post is a conversation with Schulman covering the first year LLMs could have been useful, research teams, and where RL goes next.

## Background

A value function estimates how good a state is, which helps assign credit to earlier actions. For an LLM, each token is an action, so long outputs make estimates noisy. Most LLM RL today uses simpler methods without a learned value function.

## What I took away

- Credit assignment over long token sequences is a hard open problem.
- Value functions could help agents but have high variance at token level.
- Consider step- or turn-level value estimates for agents.

## Reference

- Original post: [https://x.com/wzenus/status/2002523089301876788](https://x.com/wzenus/status/2002523089301876788)
- Author: [@wzenus](https://x.com/wzenus) on X, posted 2025-12-20
