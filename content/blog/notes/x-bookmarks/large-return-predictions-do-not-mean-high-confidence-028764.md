---
title: "Large Return Predictions Do Not Mean High Confidence"
publishDate: "2025-11-29"
date: "2025-11-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quant", "math"]
author: "Hiep Tran"
featured: false
excerpt: "A note that big predictions from a linear regression on returns usually mean more uncertainty, not confidence."
---

> Source: [@VivekVRao1 on X](https://x.com/VivekVRao1/status/1994764060039028764) · posted 2025-11-29

## Summary

The author says that when a returns model makes a large prediction, it is probably where data is sparse and uncertainty is higher. If data follows a bivariate Student's t distribution, the conditional variance of y increases linearly with the other variable. The quoted post calls it a common misconception that a large predicted return means confidence.

## Background

Linear regression outputs a point estimate, and large values come from extreme inputs where there is little training data. Prediction intervals widen there. So magnitude of a prediction is not a confidence measure.

## What I took away

- Prediction size is not confidence.
- Report uncertainty alongside predictions.
- Be careful with extreme inputs.

## Reference

- Original post: [https://x.com/VivekVRao1/status/1994764060039028764](https://x.com/VivekVRao1/status/1994764060039028764)
- Author: [@VivekVRao1](https://x.com/VivekVRao1) on X, posted 2025-11-29
