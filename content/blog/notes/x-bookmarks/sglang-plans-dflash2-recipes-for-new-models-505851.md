---
title: "SGLang Plans DFlash2 Recipes for New Models"
publishDate: "2026-08-18"
date: "2026-08-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "llm-inference", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "SGLang congratulates the DFlash 2 launch and says recipes for Qwen3.8-27B and Muse Glimmer are coming."
---

> Source: [@sgl_project on X](https://x.com/sgl_project/status/2089855016253505851) · posted 2026-08-18

## Summary

SGLang congratulates the DFlash 2 team and says recipes for running DFlash2 on Qwen3.8-27B and Muse Glimmer are on the way, with promising early results. The quoted post claims DFlash 2 runs Qwen3.8-27B at 70 tok/s on an M5 Max MacBook Pro, up to 4.6 times the speed of autoregressive decoding with the same output, and gives one more accepted token per pass.

## Background

Autoregressive decoding generates one token per model pass. DFlash is a speculative decoding method where draft tokens are verified in one pass; each extra accepted token per pass directly reduces the number of passes needed.

## What I took away

- Treat speedup numbers like 4.6x as the authors' claim until you test on your workload.
- Accepted tokens per pass is the key metric for speculative decoding.

## Reference

- Original post: [https://x.com/sgl_project/status/2089855016253505851](https://x.com/sgl_project/status/2089855016253505851)
- Author: [@sgl_project](https://x.com/sgl_project) on X, posted 2026-08-18
