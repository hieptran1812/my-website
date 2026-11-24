---
title: "Wan-Streamer Treats Video as World Plus Events"
publishDate: "2026-07-17"
date: "2026-07-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "audio-video", "world-models", "diffusion"]
author: "Hiep Tran"
featured: false
excerpt: "Wan-Streamer v0.3 splits video into a stable world and a stream of changing events to support real-time AI generation."
---

> Source: [@askalphaxiv on X](https://x.com/askalphaxiv/status/2078159284194414845) · posted 2026-07-17

## Summary

The post describes Wan-Streamer v0.3, which models video as a world plus an event stream. The world is what stays stable, like scene, character, voice and ambience. The events are what change, like speech, motion, gestures, sounds and reactions; the text cuts off as it starts explaining what this enables for real-time AI.

## Background

Video generation models usually produce every frame from scratch, which is costly. Separating persistent state from changes is a common idea in compression and game engines: keep the background fixed and update only what moves. This can make streaming, interactive generation more practical.

## What I took away

- Separating stable state from changes is a useful pattern for real-time generative systems.
- This framing resembles how games keep a world state and apply events to it.
- Check the full post for what real-time use case the design targets.

## Reference

- Original post: [https://x.com/askalphaxiv/status/2078159284194414845](https://x.com/askalphaxiv/status/2078159284194414845)
- Author: [@askalphaxiv](https://x.com/askalphaxiv) on X, posted 2026-07-17
