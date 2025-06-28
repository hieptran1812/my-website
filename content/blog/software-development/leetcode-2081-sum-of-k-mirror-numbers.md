---
title: "Leetcode routine: 2081. Sum of k-Mirror Numbers"
publishDate: "2025-06-24"
category: "software-development"
subcategory: "Algorithms"
tags: ["Palindrome"]
date: "2025-06-24"
author: "Hiep Tran"
featured: false
image: "/blog-placeholder.jpg"
excerpt: "Solution for 2081. Sum of k-Mirror Numbers"
---

*Disclaimer: This is just my personal LeetCode practice and solution approach. There may be other more optimal solutions or different ways to solve this problem.*

# 📝 Problem: Sum of k-Mirror Numbers

A **k-mirror number** is a positive integer without leading zeros that reads the same both forward and backward in **base-10** as well as in **base-k**.

For example:

- **9** is a 2-mirror number:
  - Base-10: `9`
  - Base-2: `1001`
  - Both are palindromes.

- **4** is **not** a 2-mirror number:
  - Base-2: `100` (not a palindrome)

---

## ✅ **Task**

Given the base `k` and the number `n`, return **the sum of the n smallest k-mirror numbers**.

---

### 📌 **Examples**

#### Example 1

```
Input: k = 2, n = 5
Output: 25
Explanation:
The 5 smallest 2-mirror numbers and their representations in base-2 are:
base-10 base-2
1       1
3       11
5       101
7       111
9       1001

Their sum = 1 + 3 + 5 + 7 + 9 = 25.
```

---

#### Example 2

```
Input: k = 3, n = 7
Output: 499
Explanation:
The 7 smallest 3-mirror numbers are:
base-10 base-3
1       1
2       2
4       11
8       22
121     11111
151     12121
212     21212

Their sum = 1 + 2 + 4 + 8 + 121 + 151 + 212 = 499.
```

---

#### Example 3

```
Input: k = 7, n = 17
Output: 20379000
Explanation:
The 17 smallest 7-mirror numbers are:
1, 2, 3, 4, 5, 6, 8, 121, 171, 242, 292, 16561, 65656, 2137312, 4602064, 6597956, 6958596
```

---

### 🔒 **Constraints**

```
2 <= k <= 9
1 <= n <= 30
```

---

# 💡 Solution

[Leetcode Official Solution Discussion](https://leetcode.com/problems/sum-of-k-mirror-numbers/solutions/6868512/sum-of-k-mirror-numbers)

The most basic way is **brute force**:

- Check each number sequentially from 1.
- For each number `i`, check:
  - If it is a palindrome in base-10.
  - If it is a palindrome in base-k.

However, this approach is inefficient because:

- The `30th` k-mirror number for `k=7` can be as large as `6 * 10^10`.
- Checking up to that number would take too long.

---

## ⚙️ **Optimized Approach**

**Key idea: generate palindromic numbers directly.**

### ✨ **Why it works**

- A palindrome is symmetric. We can:
  - Generate only the **first half** `i'` of the number.
  - Append its reverse to form the full palindrome `i`.

### 🔎 **Steps**

1. For each length:
   - Generate **odd-length palindromes**:
     - Example: half = "123" → palindrome = "12321".
   - Generate **even-length palindromes**:
     - Example: half = "123" → palindrome = "123321".

2. For each generated palindrome, check:
   - If it is a palindrome in base-k.

3. Repeat until `n` numbers are found.

---

### 📝 **Implementation**

```python
class Solution:
    def kMirror(self, k: int, n: int) -> int:
        def isPalindrome(x: int) -> bool:
            digit = list()
            while x:
                digit.append(x % k)
                x //= k
            return digit == digit[::-1]

        left, cnt, ans = 1, 0, 0
        while cnt < n:
            right = left * 10
            # op = 0: odd-length palindromes
            # op = 1: even-length palindromes
            for op in [0, 1]:
                for i in range(left, right):
                    if cnt == n:
                        break

                    combined = i
                    x = i // 10 if op == 0 else i
                    while x:
                        combined = combined * 10 + x % 10
                        x //= 10
                    if isPalindrome(combined):
                        cnt += 1
                        ans += combined
            left = right

        return ans
```

---

### ⏱️ **Complexity Analysis**

- **Time Complexity:** Efficient due to reduced search space by generating only palindrome candidates.
- **Space Complexity:** Depends on number of palindromic candidates generated.

---

✅ **Summary:**  
Generate palindromic numbers systematically (odd & even length) and filter those that are also palindromic in base-k for an efficient solution.


