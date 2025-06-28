---
title: "Leetcode routine: Find Subsequence of Length K With the Largest Sum"
publishDate: "2025-06-29"
category: "software-development"
subcategory: "Algorithms"
tags: ["Sorting"]
date: "2025-06-29"
author: "Hiep Tran"
featured: false
image: "/blog-placeholder.jpg"
excerpt: "Solution for Find Subsequence of Length K With the Largest Sum"
---

*Disclaimer: This is just my personal LeetCode practice and solution approach. There may be other more optimal solutions or different ways to solve this problem.*

# Problem: Find Subsequence of Length K With the Largest Sum

You are given an integer array `nums` and an integer `k`. You want to find a **subsequence** of `nums` of length `k` that has the **largest sum**.

Return **any such subsequence** as an integer array of length `k`.

A **subsequence** is an array that can be derived from another array by deleting some or no elements **without changing the order** of the remaining elements.

---

### 📌 Examples

#### Example 1

- **Input:** 
  ```
  nums = [2,1,3,3], k = 2
  ```
- **Output:** 
  ```
  [3,3]
  ```
- **Explanation:**
  
  The subsequence has the largest sum of `3 + 3 = 6`.

---

#### Example 2

- **Input:** 
  ```
  nums = [-1,-2,3,4], k = 3
  ```
- **Output:** 
  ```
  [-1,3,4]
  ```
- **Explanation:**
  
  The subsequence has the largest sum of `-1 + 3 + 4 = 6`.

---

#### Example 3

- **Input:** 
  ```
  nums = [3,4,3,3], k = 2
  ```
- **Output:** 
  ```
  [3,4]
  ```
- **Explanation:**
  
  The subsequence has the largest sum of `3 + 4 = 7`. Another possible subsequence is `[4,3]`.

---

### 🔒 Constraints

- `1 <= nums.length <= 1000`
- `-10^5 <= nums[i] <= 10^5`
- `1 <= k <= nums.length`

# Solution

https://leetcode.com/problems/find-subsequence-of-length-k-with-the-largest-sum/solutions/6882485/find-subsequence-of-length-k-with-the-largest-sum

## 🚀 Approach: Sorting

### 💡 **Intuition**

The subsequence of maximum length `k` in the array `nums` must consist of the **largest k numbers** in `nums`. To ensure that we can still form the desired subsequence in the **original order** after identifying these values through sorting, we:

1. **Create an auxiliary array `vals`**, where each element is a pair `(i, nums[i])` containing:
   - `i`: index
   - `nums[i]`: corresponding value

2. **Sort the auxiliary array in descending order** based on `nums[i]`.  
   - The first `k` elements after sorting represent the **largest k numbers**, along with their original indices.

3. **Sort these k elements in ascending order based on index `i`**, to preserve their relative order in the original array.

4. **Extract the values** from these sorted pairs to form the resulting array, which is the subsequence of length `k` with the **maximum possible sum**, maintaining original order.

Finally, **return this array** as the answer.

---

### 📝 **Implementation**

```python
class Solution:
    def maxSubsequence(self, nums: List[int], k: int) -> List[int]:
        n = len(nums)
        vals = [[i, nums[i]] for i in range(n)]  # auxiliary array
        # sort by numerical value in descending order
        vals.sort(key=lambda x: -x[1])
        # select the first k elements and sort them in ascending order by index
        vals = sorted(vals[:k])
        res = [val for idx, val in vals]  # target subsequence
        return res
```

---

### ⏱️ **Complexity Analysis**

Let `n` be the length of the array `nums`.

- **Time complexity:** `O(n log n)`  
  This is due to sorting the auxiliary array.

- **Space complexity:** `O(n)`  
  This is the space overhead of the auxiliary array.

---

✅ **Summary:**  
Using sorting with index tracking ensures **correct relative order** while achieving the **largest possible sum** for the subsequence.


