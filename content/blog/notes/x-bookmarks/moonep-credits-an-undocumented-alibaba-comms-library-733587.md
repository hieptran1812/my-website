---
title: "MoonEP Credits an Undocumented Alibaba Comms Library"
publishDate: "2026-07-31"
date: "2026-07-31"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "moe", "gpu-kernels", "industry-news"]
author: "Hiep Tran"
featured: false
excerpt: "Lucas Beyer notes that Moonshot's MoonEP acknowledges Alibaba's AcclEP, a library with no public source or documentation."
---

> Source: [@giffmana on X](https://x.com/giffmana/status/2083263841664733587) · posted 2026-07-31

## Summary

The author points out that Moonshot's **MoonEP** acknowledges **AcclEP**, described as Alibaba's black-box communication library with a DeepEP-style API for its cluster. He says there is no link because neither he nor his agents could find any source code or documentation. He finds it odd to credit a library for inspiration when it is not public.

## Background

EP means expert parallelism: in a mixture-of-experts model, different experts live on different GPUs and tokens must be sent between them. DeepEP is DeepSeek's open library for this all-to-all communication. Libraries in this space are key to fast MoE training and inference.

## What I took away

- Expert-parallel communication libraries are a competitive area among Chinese labs.
- Acknowledgements can point to internal tools that outsiders cannot inspect.
- When evaluating a project's lineage, check whether its references are actually accessible.

## Reference

- Original post: [https://x.com/giffmana/status/2083263841664733587](https://x.com/giffmana/status/2083263841664733587)
- Author: [@giffmana](https://x.com/giffmana) on X, posted 2026-07-31
