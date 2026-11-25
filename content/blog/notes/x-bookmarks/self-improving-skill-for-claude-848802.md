---
title: "Self-Improving Skill For Claude"
publishDate: "2026-03-17"
date: "2026-03-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-agents", "harness-engineering"]
author: "Hiep Tran"
featured: false
excerpt: "The author describes one skill that runs another skill, scores output, tweaks the prompt and keeps changes that improve the score."
---

> Source: [@itsolelehmann on X](https://x.com/itsolelehmann/status/2033982679423848802) · posted 2026-03-17

## Summary

The author says one skill can auto-improve any other skill: it runs the skill and scores output, finds what is failing, makes one small change to the prompt, reruns to see if the score went up, and keeps the change if it did. The text is cut off.

## Background

This is the autoresearch loop applied to prompts. It needs a scoring rule to judge outputs, and a bad scorer will push the skill in the wrong direction.

## What I took away

- One small change per iteration makes results easy to attribute.
- Your scoring method decides what the loop optimizes.
- Keep or revert is a safe default.

## Reference

- Original post: [https://x.com/itsolelehmann/status/2033982679423848802](https://x.com/itsolelehmann/status/2033982679423848802)
- Author: [@itsolelehmann](https://x.com/itsolelehmann) on X, posted 2026-03-17
