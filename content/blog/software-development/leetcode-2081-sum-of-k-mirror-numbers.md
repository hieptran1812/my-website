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

# Problem

A k-mirror number is a positive integer without leading zeros that reads the same both forward and backward in base-10 as well as in base-k.

For example, 9 is a 2-mirror number. The representation of 9 in base-10 and base-2 are 9 and 1001 respectively, which read the same both forward and backward.
On the contrary, 4 is not a 2-mirror number. The representation of 4 in base-2 is 100, which does not read the same both forward and backward.
Given the base k and the number n, return the sum of the n smallest k-mirror numbers.

**Example 1:**

```
Input: k = 2, n = 5
Output: 25
Explanation:
The 5 smallest 2-mirror numbers and their representations in base-2 are listed as follows:
base-10 base-2
1 1
3 11
5 101
7 111
9 1001
Their sum = 1 + 3 + 5 + 7 + 9 = 25.
```

**Example 2:**

```
Input: k = 3, n = 7
Output: 499
Explanation:
The 7 smallest 3-mirror numbers are and their representations in base-3 are listed as follows:
base-10 base-3
1 1
2 2
4 11
8 22
121 11111
151 12121
212 21212
Their sum = 1 + 2 + 4 + 8 + 121 + 151 + 212 = 499.
```

**Example 3:**

```
Input: k = 7, n = 17
Output: 20379000
Explanation: The 17 smallest 7-mirror numbers are:
1, 2, 3, 4, 5, 6, 8, 121, 171, 242, 292, 16561, 65656, 2137312, 4602064, 6597956, 6958596
```

**Constraints:**

```
2 <= k <= 9
1 <= n <= 30
```

# Solution

https://leetcode.com/problems/sum-of-k-mirror-numbers/solutions/6868512/sum-of-k-mirror-numbers

The most basic way to solve this problem is to start checking every number sequentially, beginning from 1. For each number $i$, we verify two things: whether $i$ itself is a palindrome, and whether its representation in base-$k$ is also a palindrome. If both conditions are true, we add $i$ to our result. We continue this process until we have found $n$ such numbers.

However, this straightforward approach isn't efficient enough—it quickly becomes too slow. For example, when $k = 7$, the 30th number that meets the conditions is approximately $6 \times 10^{10}$. Even though checking each number individually takes constant time (O(1)), scanning through all numbers up to $6 \times 10^{10}$ would still take far too long.

To speed things up, we can adopt a smarter strategy, inspired by the binary search method. The idea is to significantly reduce the number of checks by cleverly generating only palindrome numbers directly, rather than checking all numbers individually. Here's how it works:

- A palindrome number is symmetric around its center. Thus, instead of generating the full palindrome at once, we can generate just the first half, call it $i'$, and then append its reverse to form the full palindrome $i$.
- This "halving" technique drastically shrinks the number of potential numbers we need to examine. For instance, if we consider numbers up to $10^{10}$, the brute-force method would check every single number up to $10^{10}$. But by generating numbers from their halves, we only need about $O(\sqrt{10^{10}}) = O(10^5)$ palindromes—a huge improvement in efficiency.

When constructing these palindromes, we should consider two cases:

1. **Odd-length palindromes**: For example, from the half number "123," we construct "12321" by appending the reverse of "12" to the end (reusing the middle digit).
2. **Even-length palindromes**: For example, from "123," we form "123321" by fully reversing the half "123" and appending it.

To keep our search efficient and ordered, we do the following:

- Iterate incrementally over half-numbers $i'$ within clearly defined ranges (such as between $10^k$ and $10^{k+1}$).
- For each half-number, first generate the odd-length palindrome and check if it meets our criteria.
- Then, generate the corresponding even-length palindrome and check it as well.

This systematic approach ensures we efficiently cover all possible palindromes in increasing order, greatly improving the solution's performance.

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
            # op = 0 indicates enumerating odd-length palindromes
            # op = 1 indicates enumerating even-length palindromes
            for op in [0, 1]:
                # enumerate i'
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
