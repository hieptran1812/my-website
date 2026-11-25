---
title: "SWE-bench Scores and Generalization After RL"
publishDate: "2025-12-15"
date: "2025-12-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "ai-agents", "reinforcement-learning"]
author: "Hiep Tran"
featured: false
excerpt: "A reply on generalization after heavy SFT and RL, in a thread about gaps between reported and independent SWE-bench scores."
---

> Source: [@ShengjieWa34067 on X](https://x.com/ShengjieWa34067/status/2000387903965528552) · posted 2025-12-15

## Summary

@ShengjieWa34067 agrees with a point that the core issue is limited generalization, especially after heavy SFT and RL. One quoted post says big labs' reported SWE-bench scores can be up to 8 points higher than scores from mini-swe-agent, partly because models are not finetuned on its trajectories. Another quoted post reports an updated SWE-bench leaderboard where GPT 5.2 high is #3 and GPT 5.2 medium narrows the gap to Sonnet 4.5 at lower cost, all run with the same setup.

## Background

SWE-bench tests whether an agent can fix real software issues. Scores depend on the agent scaffold, so labs tuned for their own harness can look better. Running every model in one identical harness gives a fairer comparison.

## What I took away

- Benchmark scores depend on the harness, not just the model.
- Compare models in an identical setup for fair results.
- Heavy SFT and RL can narrow how well a model generalizes to new scaffolds.

## Reference

- Original post: [https://x.com/ShengjieWa34067/status/2000387903965528552](https://x.com/ShengjieWa34067/status/2000387903965528552)
- Author: [@ShengjieWa34067](https://x.com/ShengjieWa34067) on X, posted 2025-12-15
