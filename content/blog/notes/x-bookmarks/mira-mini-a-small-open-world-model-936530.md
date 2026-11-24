---
title: "Mira Mini, a Small Open World Model"
publishDate: "2026-07-15"
date: "2026-07-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "world-models", "distillation", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "A team reproduced the Mira world model in a week and distilled it to 364M parameters for multiplayer Rocket League in the browser."
---

> Source: [@hugothomel on X](https://x.com/hugothomel/status/2077440553973936530) · posted 2026-07-15

## Summary

The authors ask how small a world model can be and present mira mini, the first reproduction of mira. They say it was reproduced from scratch in a week, supports four players in one shared Rocket League match in the browser, and was distilled to 364M parameters. Weights, code and a full report are open.

## Background

A world model learns to predict how an environment changes in response to actions, so it can act like a game engine made of neural networks. Distillation trains a smaller model to mimic a larger one, making it cheap enough to run in a browser.

## What I took away

- World models can be small enough for the browser after distillation.
- Open reproductions are a good way to learn how these systems are built.
- Multiplayer shared state is a harder test than single-player generation.

## Reference

- Original post: [https://x.com/hugothomel/status/2077440553973936530](https://x.com/hugothomel/status/2077440553973936530)
- Author: [@hugothomel](https://x.com/hugothomel) on X, posted 2026-07-15
