---
title: "Implementing Muon In torchlib On Stream"
publishDate: "2025-11-14"
date: "2025-11-14"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "gpu-kernels", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "The author plans to implement the Muon optimizer in torchlib because PyTorch only added it to the Python API."
---

> Source: [@jsuarez on X](https://x.com/jsuarez/status/1989368935712714893) · posted 2025-11-14

## Summary

The author says they are implementing Muon in torchlib on stream today. They complain that the latest PyTorch version added Muon only to the Python API, not to the C++ library.

## Background

Muon is an optimizer for training neural networks. torchlib is PyTorch's C++ API, used for building training code without Python. A C++ version is needed when training code does not run in Python.

## What I took away

- Check whether a feature exists in the C++ API as well as Python.
- Implementing an optimizer yourself is a way to learn it.
- Muon is gaining attention as an optimizer.

## Reference

- Original post: [https://x.com/jsuarez/status/1989368935712714893](https://x.com/jsuarez/status/1989368935712714893)
- Author: [@jsuarez](https://x.com/jsuarez) on X, posted 2025-11-14
