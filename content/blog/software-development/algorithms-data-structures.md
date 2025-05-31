---
title: "Algorithms and Data Structures: Foundations for Efficient Software"
excerpt: "Master essential algorithms and data structures with comprehensive coverage of complexity analysis, sorting, searching, trees, graphs, dynamic programming, and optimization techniques."
date: "2024-12-20"
category: "Software Development"
subcategory: "Algorithms"
tags:
  [
    "Algorithms",
    "Data Structures",
    "Big O",
    "Dynamic Programming",
    "Graph Algorithms",
    "Tree Algorithms",
    "Sorting",
  ]
featured: true
author: "Hiep Tran"
readTime: "25 min read"
---

# Algorithms and Data Structures: Foundations for Efficient Software

Understanding algorithms and data structures is fundamental to writing efficient, scalable software. This comprehensive guide covers essential concepts, implementations, and optimization techniques that every developer should master.

## Time and Space Complexity Analysis

### Big O Notation

Big O notation describes the upper bound of algorithm performance as input size grows.

```python
# O(1) - Constant Time
def get_first_element(arr):
    """Always takes the same time regardless of input size"""
    if len(arr) > 0:
        return arr[0]
    return None

# O(log n) - Logarithmic Time
def binary_search(arr, target):
    """Divides search space in half each iteration"""
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# O(n) - Linear Time
def find_max(arr):
    """Must examine each element once"""
    if not arr:
        return None

    max_val = arr[0]
    for num in arr[1:]:
        if num > max_val:
            max_val = num
    return max_val

# O(n log n) - Linearithmic Time
def merge_sort(arr):
    """Efficient comparison-based sorting"""
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result

# O(n²) - Quadratic Time
def bubble_sort(arr):
    """Compare each pair of adjacent elements"""
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

# O(2^n) - Exponential Time
def fibonacci_naive(n):
    """Naive recursive approach with overlapping subproblems"""
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)
```

### Space Complexity Analysis

```python
# O(1) Space - Constant extra space
def reverse_array_in_place(arr):
    """Reverses array using only constant extra space"""
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1

# O(n) Space - Linear extra space
def fibonacci_memoized(n, memo=None):
    """Uses memoization to avoid recomputation"""
    if memo is None:
        memo = {}

    if n in memo:
        return memo[n]

    if n <= 1:
        return n

    memo[n] = fibonacci_memoized(n - 1, memo) + fibonacci_memoized(n - 2, memo)
    return memo[n]

# O(log n) Space - Recursive call stack
def binary_search_recursive(arr, target, left=0, right=None):
    """Recursive binary search"""
    if right is None:
        right = len(arr) - 1

    if left > right:
        return -1

    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
```

## Fundamental Data Structures

### Arrays and Dynamic Arrays

```python
class DynamicArray:
    """Implementation of a dynamic array (similar to Python list)"""

    def __init__(self):
        self.capacity = 1
        self.size = 0
        self.data = [None] * self.capacity

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if not 0 <= index < self.size:
            raise IndexError("Index out of range")
        return self.data[index]

    def __setitem__(self, index, value):
        if not 0 <= index < self.size:
            raise IndexError("Index out of range")
        self.data[index] = value

    def append(self, value):
        """Add element to end - O(1) amortized"""
        if self.size == self.capacity:
            self._resize()

        self.data[self.size] = value
        self.size += 1

    def insert(self, index, value):
        """Insert element at index - O(n)"""
        if not 0 <= index <= self.size:
            raise IndexError("Index out of range")

        if self.size == self.capacity:
            self._resize()

        # Shift elements to the right
        for i in range(self.size, index, -1):
            self.data[i] = self.data[i - 1]

        self.data[index] = value
        self.size += 1

    def delete(self, index):
        """Delete element at index - O(n)"""
        if not 0 <= index < self.size:
            raise IndexError("Index out of range")

        # Shift elements to the left
        for i in range(index, self.size - 1):
            self.data[i] = self.data[i + 1]

        self.size -= 1

        # Shrink if necessary
        if self.size <= self.capacity // 4:
            self._resize(shrink=True)

    def _resize(self, shrink=False):
        """Resize internal array"""
        if shrink:
            self.capacity = max(1, self.capacity // 2)
        else:
            self.capacity *= 2

        new_data = [None] * self.capacity
        for i in range(self.size):
            new_data[i] = self.data[i]
        self.data = new_data
```

### Linked Lists

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
        self.size = 0

    def append(self, val):
        """Add element to end - O(n)"""
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self.size += 1

    def prepend(self, val):
        """Add element to beginning - O(1)"""
        new_node = ListNode(val)
        new_node.next = self.head
        self.head = new_node
        self.size += 1

    def delete(self, val):
        """Delete first occurrence of value - O(n)"""
        if not self.head:
            return False

        if self.head.val == val:
            self.head = self.head.next
            self.size -= 1
            return True

        current = self.head
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
                self.size -= 1
                return True
            current = current.next

        return False

    def find(self, val):
        """Find element - O(n)"""
        current = self.head
        while current:
            if current.val == val:
                return current
            current = current.next
        return None

    def reverse(self):
        """Reverse linked list - O(n)"""
        prev = None
        current = self.head

        while current:
            next_temp = current.next
            current.next = prev
            prev = current
            current = next_temp

        self.head = prev

# Doubly Linked List
class DoublyListNode:
    def __init__(self, val=0, prev=None, next=None):
        self.val = val
        self.prev = prev
        self.next = next

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, val):
        """Add element to end - O(1)"""
        new_node = DoublyListNode(val)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1

    def prepend(self, val):
        """Add element to beginning - O(1)"""
        new_node = DoublyListNode(val)
        if not self.head:
            self.head = self.tail = new_node
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        self.size += 1
```

### Stacks and Queues

```python
class Stack:
    """LIFO (Last In, First Out) data structure"""

    def __init__(self):
        self.items = []

    def push(self, item):
        """Add item to top - O(1)"""
        self.items.append(item)

    def pop(self):
        """Remove and return top item - O(1)"""
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self.items.pop()

    def peek(self):
        """Return top item without removing - O(1)"""
        if self.is_empty():
            raise IndexError("peek from empty stack")
        return self.items[-1]

    def is_empty(self):
        """Check if stack is empty - O(1)"""
        return len(self.items) == 0

    def size(self):
        """Return number of items - O(1)"""
        return len(self.items)

class Queue:
    """FIFO (First In, First Out) data structure"""

    def __init__(self):
        self.items = []

    def enqueue(self, item):
        """Add item to rear - O(1)"""
        self.items.append(item)

    def dequeue(self):
        """Remove and return front item - O(n)"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self.items.pop(0)

    def front(self):
        """Return front item without removing - O(1)"""
        if self.is_empty():
            raise IndexError("front from empty queue")
        return self.items[0]

    def is_empty(self):
        """Check if queue is empty - O(1)"""
        return len(self.items) == 0

# Efficient Queue using collections.deque
from collections import deque

class EfficientQueue:
    """Efficient queue implementation using deque"""

    def __init__(self):
        self.items = deque()

    def enqueue(self, item):
        """Add item to rear - O(1)"""
        self.items.append(item)

    def dequeue(self):
        """Remove and return front item - O(1)"""
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        return self.items.popleft()

    def front(self):
        """Return front item without removing - O(1)"""
        if self.is_empty():
            raise IndexError("front from empty queue")
        return self.items[0]

    def is_empty(self):
        """Check if queue is empty - O(1)"""
        return len(self.items) == 0
```

### Hash Tables

```python
class HashTable:
    """Hash table implementation using separate chaining"""

    def __init__(self, capacity=10):
        self.capacity = capacity
        self.size = 0
        self.buckets = [[] for _ in range(capacity)]

    def _hash(self, key):
        """Simple hash function"""
        return hash(key) % self.capacity

    def put(self, key, value):
        """Insert or update key-value pair - O(1) average"""
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]

        # Update existing key
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return

        # Add new key
        bucket.append((key, value))
        self.size += 1

        # Resize if load factor > 0.7
        if self.size > 0.7 * self.capacity:
            self._resize()

    def get(self, key):
        """Get value by key - O(1) average"""
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]

        for k, v in bucket:
            if k == key:
                return v

        raise KeyError(key)

    def delete(self, key):
        """Delete key-value pair - O(1) average"""
        bucket_index = self._hash(key)
        bucket = self.buckets[bucket_index]

        for i, (k, v) in enumerate(bucket):
            if k == key:
                del bucket[i]
                self.size -= 1
                return v

        raise KeyError(key)

    def _resize(self):
        """Resize hash table when load factor is high"""
        old_buckets = self.buckets
        self.capacity *= 2
        self.size = 0
        self.buckets = [[] for _ in range(self.capacity)]

        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)

# Robin Hood Hashing (more advanced)
class RobinHoodHashTable:
    """Hash table using Robin Hood hashing for better performance"""

    def __init__(self, capacity=16):
        self.capacity = capacity
        self.size = 0
        self.keys = [None] * capacity
        self.values = [None] * capacity
        self.distances = [0] * capacity

    def _hash(self, key):
        return hash(key) % self.capacity

    def _probe_distance(self, key, index):
        """Calculate probe distance for Robin Hood hashing"""
        return (index - self._hash(key)) % self.capacity

    def put(self, key, value):
        """Insert with Robin Hood hashing - O(1) average"""
        if self.size >= 0.7 * self.capacity:
            self._resize()

        index = self._hash(key)
        distance = 0

        while self.keys[index] is not None:
            if self.keys[index] == key:
                self.values[index] = value
                return

            # Robin Hood: steal from the rich
            if distance > self.distances[index]:
                # Swap with current entry
                key, self.keys[index] = self.keys[index], key
                value, self.values[index] = self.values[index], value
                distance, self.distances[index] = self.distances[index], distance

            index = (index + 1) % self.capacity
            distance += 1

        self.keys[index] = key
        self.values[index] = value
        self.distances[index] = distance
        self.size += 1
```

## Tree Data Structures

### Binary Search Tree

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, val):
        """Insert value - O(log n) average, O(n) worst"""
        self.root = self._insert_recursive(self.root, val)

    def _insert_recursive(self, node, val):
        if node is None:
            return TreeNode(val)

        if val < node.val:
            node.left = self._insert_recursive(node.left, val)
        elif val > node.val:
            node.right = self._insert_recursive(node.right, val)

        return node

    def search(self, val):
        """Search for value - O(log n) average, O(n) worst"""
        return self._search_recursive(self.root, val)

    def _search_recursive(self, node, val):
        if node is None or node.val == val:
            return node

        if val < node.val:
            return self._search_recursive(node.left, val)
        else:
            return self._search_recursive(node.right, val)

    def delete(self, val):
        """Delete value - O(log n) average, O(n) worst"""
        self.root = self._delete_recursive(self.root, val)

    def _delete_recursive(self, node, val):
        if node is None:
            return node

        if val < node.val:
            node.left = self._delete_recursive(node.left, val)
        elif val > node.val:
            node.right = self._delete_recursive(node.right, val)
        else:
            # Node to be deleted found
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left

            # Node with two children
            # Get inorder successor (smallest in right subtree)
            successor = self._find_min(node.right)
            node.val = successor.val
            node.right = self._delete_recursive(node.right, successor.val)

        return node

    def _find_min(self, node):
        while node.left:
            node = node.left
        return node

    def inorder_traversal(self):
        """In-order traversal - O(n)"""
        result = []
        self._inorder_recursive(self.root, result)
        return result

    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.val)
            self._inorder_recursive(node.right, result)
```

### AVL Tree (Self-Balancing)

```python
class AVLNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.height = 1

class AVLTree:
    """Self-balancing binary search tree"""

    def __init__(self):
        self.root = None

    def _get_height(self, node):
        if not node:
            return 0
        return node.height

    def _get_balance(self, node):
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _update_height(self, node):
        if node:
            node.height = 1 + max(self._get_height(node.left),
                                  self._get_height(node.right))

    def _rotate_left(self, z):
        """Left rotation"""
        y = z.right
        T2 = y.left

        # Perform rotation
        y.left = z
        z.right = T2

        # Update heights
        self._update_height(z)
        self._update_height(y)

        return y

    def _rotate_right(self, z):
        """Right rotation"""
        y = z.left
        T3 = y.right

        # Perform rotation
        y.right = z
        z.left = T3

        # Update heights
        self._update_height(z)
        self._update_height(y)

        return y

    def insert(self, val):
        """Insert value maintaining AVL property - O(log n)"""
        self.root = self._insert_recursive(self.root, val)

    def _insert_recursive(self, node, val):
        # Standard BST insertion
        if not node:
            return AVLNode(val)

        if val < node.val:
            node.left = self._insert_recursive(node.left, val)
        elif val > node.val:
            node.right = self._insert_recursive(node.right, val)
        else:
            return node  # Duplicate values not allowed

        # Update height
        self._update_height(node)

        # Get balance factor
        balance = self._get_balance(node)

        # Perform rotations if unbalanced
        # Left Left Case
        if balance > 1 and val < node.left.val:
            return self._rotate_right(node)

        # Right Right Case
        if balance < -1 and val > node.right.val:
            return self._rotate_left(node)

        # Left Right Case
        if balance > 1 and val > node.left.val:
            node.left = self._rotate_left(node.left)
            return self._rotate_right(node)

        # Right Left Case
        if balance < -1 and val < node.right.val:
            node.right = self._rotate_right(node.right)
            return self._rotate_left(node)

        return node
```

### Heap (Priority Queue)

```python
class MinHeap:
    """Min heap implementation"""

    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    def left_child(self, i):
        return 2 * i + 1

    def right_child(self, i):
        return 2 * i + 2

    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def insert(self, val):
        """Insert value - O(log n)"""
        self.heap.append(val)
        self._heapify_up(len(self.heap) - 1)

    def _heapify_up(self, i):
        """Maintain heap property upward"""
        while i > 0 and self.heap[self.parent(i)] > self.heap[i]:
            self.swap(i, self.parent(i))
            i = self.parent(i)

    def extract_min(self):
        """Remove and return minimum element - O(log n)"""
        if not self.heap:
            raise IndexError("extract from empty heap")

        if len(self.heap) == 1:
            return self.heap.pop()

        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)

        return root

    def _heapify_down(self, i):
        """Maintain heap property downward"""
        while self.left_child(i) < len(self.heap):
            # Find smallest child
            min_child = self.left_child(i)
            if (self.right_child(i) < len(self.heap) and
                self.heap[self.right_child(i)] < self.heap[self.left_child(i)]):
                min_child = self.right_child(i)

            # If heap property is satisfied, break
            if self.heap[i] <= self.heap[min_child]:
                break

            self.swap(i, min_child)
            i = min_child

    def peek(self):
        """Return minimum element without removing - O(1)"""
        if not self.heap:
            raise IndexError("peek from empty heap")
        return self.heap[0]

    def size(self):
        return len(self.heap)

    def is_empty(self):
        return len(self.heap) == 0

# Build heap from array - O(n)
def build_heap(arr):
    """Build heap from array in linear time"""
    heap = MinHeap()
    heap.heap = arr.copy()

    # Start from last non-leaf node and heapify down
    start = (len(arr) - 2) // 2
    for i in range(start, -1, -1):
        heap._heapify_down(i)

    return heap
```

## Graph Algorithms

### Graph Representations

```python
from collections import defaultdict, deque

class Graph:
    """Graph implementation using adjacency list"""

    def __init__(self, directed=False):
        self.graph = defaultdict(list)
        self.directed = directed

    def add_edge(self, u, v, weight=1):
        """Add edge between vertices"""
        self.graph[u].append((v, weight))
        if not self.directed:
            self.graph[v].append((u, weight))

    def get_vertices(self):
        """Get all vertices"""
        vertices = set()
        for u in self.graph:
            vertices.add(u)
            for v, _ in self.graph[u]:
                vertices.add(v)
        return list(vertices)

    def bfs(self, start):
        """Breadth-First Search - O(V + E)"""
        visited = set()
        queue = deque([start])
        visited.add(start)
        result = []

        while queue:
            vertex = queue.popleft()
            result.append(vertex)

            for neighbor, _ in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return result

    def dfs(self, start):
        """Depth-First Search - O(V + E)"""
        visited = set()
        result = []

        def dfs_recursive(vertex):
            visited.add(vertex)
            result.append(vertex)

            for neighbor, _ in self.graph[vertex]:
                if neighbor not in visited:
                    dfs_recursive(neighbor)

        dfs_recursive(start)
        return result

    def shortest_path_bfs(self, start, end):
        """Shortest path using BFS (unweighted) - O(V + E)"""
        visited = set()
        queue = deque([(start, [start])])
        visited.add(start)

        while queue:
            vertex, path = queue.popleft()

            if vertex == end:
                return path

            for neighbor, _ in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None  # No path found
```

### Dijkstra's Algorithm

```python
import heapq
from collections import defaultdict

def dijkstra(graph, start):
    """Dijkstra's shortest path algorithm - O((V + E) log V)"""
    # Initialize distances and visited set
    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0
    visited = set()

    # Priority queue: (distance, vertex)
    pq = [(0, start)]

    while pq:
        current_dist, current_vertex = heapq.heappop(pq)

        if current_vertex in visited:
            continue

        visited.add(current_vertex)

        # Check neighbors
        for neighbor, weight in graph[current_vertex]:
            if neighbor not in visited:
                new_dist = current_dist + weight

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))

    return dict(distances)

def dijkstra_with_path(graph, start, end):
    """Dijkstra with path reconstruction"""
    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0
    previous = {}
    visited = set()

    pq = [(0, start)]

    while pq:
        current_dist, current_vertex = heapq.heappop(pq)

        if current_vertex == end:
            break

        if current_vertex in visited:
            continue

        visited.add(current_vertex)

        for neighbor, weight in graph[current_vertex]:
            if neighbor not in visited:
                new_dist = current_dist + weight

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_vertex
                    heapq.heappush(pq, (new_dist, neighbor))

    # Reconstruct path
    if end not in previous and end != start:
        return None, float('inf')

    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous.get(current)

    path.reverse()
    return path, distances[end]
```

### Topological Sort

```python
def topological_sort_dfs(graph):
    """Topological sort using DFS - O(V + E)"""
    visited = set()
    stack = []

    def dfs(vertex):
        visited.add(vertex)
        for neighbor, _ in graph[vertex]:
            if neighbor not in visited:
                dfs(neighbor)
        stack.append(vertex)

    # Visit all vertices
    for vertex in graph:
        if vertex not in visited:
            dfs(vertex)

    return stack[::-1]  # Reverse for topological order

def topological_sort_kahn(graph):
    """Kahn's algorithm for topological sort - O(V + E)"""
    # Calculate in-degrees
    in_degree = defaultdict(int)
    vertices = set()

    for u in graph:
        vertices.add(u)
        for v, _ in graph[u]:
            vertices.add(v)
            in_degree[v] += 1

    # Initialize queue with vertices having in-degree 0
    queue = deque([v for v in vertices if in_degree[v] == 0])
    result = []

    while queue:
        vertex = queue.popleft()
        result.append(vertex)

        # Reduce in-degree of neighbors
        for neighbor, _ in graph[vertex]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycles
    if len(result) != len(vertices):
        raise ValueError("Graph has cycles")

    return result
```

## Dynamic Programming

### Classic DP Problems

```python
def fibonacci_dp(n):
    """Fibonacci with dynamic programming - O(n) time, O(1) space"""
    if n <= 1:
        return n

    prev2, prev1 = 0, 1
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current

    return prev1

def longest_common_subsequence(text1, text2):
    """LCS problem - O(m*n) time, O(m*n) space"""
    m, n = len(text1), len(text2)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]

def knapsack_01(weights, values, capacity):
    """0/1 Knapsack problem - O(n*W) time, O(n*W) space"""
    n = len(weights)

    # Create DP table
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Fill DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Don't include current item
            dp[i][w] = dp[i-1][w]

            # Include current item if possible
            if weights[i-1] <= w:
                include_value = dp[i-1][w - weights[i-1]] + values[i-1]
                dp[i][w] = max(dp[i][w], include_value)

    return dp[n][capacity]

def coin_change(coins, amount):
    """Coin change problem - O(amount * coins) time"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1

def edit_distance(word1, word2):
    """Edit distance (Levenshtein distance) - O(m*n)"""
    m, n = len(word1), len(word2)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # deletion
                    dp[i][j-1],      # insertion
                    dp[i-1][j-1]     # substitution
                )

    return dp[m][n]
```

### Advanced DP Techniques

```python
def longest_increasing_subsequence(nums):
    """LIS with binary search optimization - O(n log n)"""
    if not nums:
        return 0

    # tails[i] = smallest tail of all increasing subsequences of length i+1
    tails = []

    for num in nums:
        # Binary search for the position to insert/replace
        left, right = 0, len(tails)

        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid

        # If num is larger than all elements, append it
        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num

    return len(tails)

def matrix_chain_multiplication(dimensions):
    """Matrix chain multiplication - O(n^3)"""
    n = len(dimensions) - 1  # Number of matrices

    # dp[i][j] = minimum scalar multiplications for matrices i to j
    dp = [[0] * n for _ in range(n)]

    # Length of chain
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')

            # Try all possible split points
            for k in range(i, j):
                cost = (dp[i][k] + dp[k+1][j] +
                       dimensions[i] * dimensions[k+1] * dimensions[j+1])
                dp[i][j] = min(dp[i][j], cost)

    return dp[0][n-1]

# Memoization decorator for DP
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_memo(n):
    """Fibonacci with memoization"""
    if n <= 1:
        return n
    return fib_memo(n-1) + fib_memo(n-2)

# State space reduction technique
def house_robber_circular(nums):
    """House robber with circular constraint"""
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    def rob_linear(houses):
        prev2 = prev1 = 0
        for money in houses:
            current = max(prev1, prev2 + money)
            prev2, prev1 = prev1, current
        return prev1

    # Case 1: Rob houses 0 to n-2 (exclude last house)
    # Case 2: Rob houses 1 to n-1 (exclude first house)
    return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))
```

## Sorting and Searching Algorithms

### Advanced Sorting Algorithms

```python
def quick_sort(arr, low=0, high=None):
    """Quick sort - O(n log n) average, O(n²) worst"""
    if high is None:
        high = len(arr) - 1

    if low < high:
        # Partition the array
        pivot_index = partition(arr, low, high)

        # Recursively sort elements before and after partition
        quick_sort(arr, low, pivot_index - 1)
        quick_sort(arr, pivot_index + 1, high)

    return arr

def partition(arr, low, high):
    """Lomuto partition scheme"""
    pivot = arr[high]
    i = low - 1  # Index of smaller element

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def heap_sort(arr):
    """Heap sort - O(n log n) guaranteed"""
    n = len(arr)

    # Build max heap
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Extract elements from heap one by one
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # Move current root to end
        heapify(arr, i, 0)  # Call heapify on reduced heap

    return arr

def heapify(arr, n, i):
    """Heapify subtree rooted at index i"""
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def counting_sort(arr, max_val):
    """Counting sort - O(n + k) where k is range of input"""
    count = [0] * (max_val + 1)
    output = [0] * len(arr)

    # Count occurrences
    for num in arr:
        count[num] += 1

    # Calculate cumulative count
    for i in range(1, max_val + 1):
        count[i] += count[i - 1]

    # Build output array
    for i in range(len(arr) - 1, -1, -1):
        output[count[arr[i]] - 1] = arr[i]
        count[arr[i]] -= 1

    return output

def radix_sort(arr):
    """Radix sort - O(d * (n + k)) where d is number of digits"""
    if not arr:
        return arr

    max_num = max(arr)
    exp = 1

    while max_num // exp > 0:
        counting_sort_by_digit(arr, exp)
        exp *= 10

    return arr

def counting_sort_by_digit(arr, exp):
    """Helper function for radix sort"""
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    # Count occurrences of each digit
    for num in arr:
        index = (num // exp) % 10
        count[index] += 1

    # Calculate cumulative count
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Build output array
    for i in range(n - 1, -1, -1):
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1

    # Copy output array to arr
    for i in range(n):
        arr[i] = output[i]
```

### Advanced Search Algorithms

```python
def binary_search_variants(arr, target):
    """Various binary search implementations"""

    def find_first_occurrence(arr, target):
        """Find first occurrence of target"""
        left, right = 0, len(arr) - 1
        result = -1

        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                result = mid
                right = mid - 1  # Continue searching left
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return result

    def find_last_occurrence(arr, target):
        """Find last occurrence of target"""
        left, right = 0, len(arr) - 1
        result = -1

        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                result = mid
                left = mid + 1  # Continue searching right
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1

        return result

    def find_insertion_point(arr, target):
        """Find position where target should be inserted"""
        left, right = 0, len(arr)

        while left < right:
            mid = (left + right) // 2
            if arr[mid] < target:
                left = mid + 1
            else:
                right = mid

        return left

    return {
        'first': find_first_occurrence(arr, target),
        'last': find_last_occurrence(arr, target),
        'insertion_point': find_insertion_point(arr, target)
    }

def ternary_search(arr, target, left=0, right=None):
    """Ternary search for unimodal functions - O(log₃ n)"""
    if right is None:
        right = len(arr) - 1

    if left <= right:
        mid1 = left + (right - left) // 3
        mid2 = right - (right - left) // 3

        if arr[mid1] == target:
            return mid1
        if arr[mid2] == target:
            return mid2

        if target < arr[mid1]:
            return ternary_search(arr, target, left, mid1 - 1)
        elif target > arr[mid2]:
            return ternary_search(arr, target, mid2 + 1, right)
        else:
            return ternary_search(arr, target, mid1 + 1, mid2 - 1)

    return -1

def exponential_search(arr, target):
    """Exponential search - O(log n)"""
    if arr[0] == target:
        return 0

    # Find range for binary search
    i = 1
    while i < len(arr) and arr[i] <= target:
        i *= 2

    # Binary search in found range
    return binary_search(arr, target, i // 2, min(i, len(arr) - 1))
```

## Conclusion

Mastering algorithms and data structures is essential for writing efficient, scalable software. Understanding complexity analysis helps you choose the right approach for your specific constraints, while knowledge of fundamental data structures enables you to model problems effectively.

Key principles to remember:

- **Choose the right data structure**: Understand the trade-offs between different options
- **Analyze complexity**: Always consider both time and space complexity
- **Optimize for your use case**: General solutions may not always be optimal
- **Practice implementation**: Understanding concepts is different from implementing them
- **Learn from patterns**: Many problems follow similar algorithmic patterns

The field of algorithms continues to evolve with new techniques for parallel processing, quantum computing, and machine learning applications. Building a strong foundation in classical algorithms and data structures provides the basis for understanding and contributing to these advanced areas.

---

_Ready to implement these algorithms? Start by practicing the fundamental data structures and gradually work your way up to more complex algorithms. Focus on understanding the underlying principles rather than memorizing implementations._
