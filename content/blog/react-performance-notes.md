---
title: "React Performance Optimization: Notes from the Field"
publishDate: "2024-02-20"
readTime: "12 min read"
category: "Notes"
author: "Hiep Tran"
tags: ["React", "Performance", "Optimization", "Frontend", "JavaScript"]
image: "/blog-placeholder.jpg"
excerpt: "Practical notes on React performance optimization techniques, from basic memo usage to advanced patterns for building fast, responsive web applications."
---

# React Performance Optimization: Notes from the Field

![React Performance](/blog-placeholder.jpg)

These notes compile practical React performance optimization techniques learned from building and maintaining large-scale applications. Focus on measuring first, optimizing second.

## Performance Fundamentals

### When to Optimize

**The Golden Rule:** Profile before optimizing

- Use React DevTools Profiler
- Measure Core Web Vitals
- Identify actual bottlenecks, not perceived ones

**Common Performance Issues:**

- Unnecessary re-renders
- Large bundle sizes
- Inefficient state updates
- Memory leaks
- Blocking the main thread

### Measuring Performance

```javascript
// React DevTools Profiler API
import { Profiler } from "react";

function onRenderCallback(
  id,
  phase,
  actualDuration,
  baseDuration,
  startTime,
  commitTime
) {
  console.log("Component:", id);
  console.log("Phase:", phase); // "mount" or "update"
  console.log("Actual duration:", actualDuration);
  console.log("Base duration:", baseDuration);
}

function App() {
  return (
    <Profiler id="App" onRender={onRenderCallback}>
      <MyComponent />
    </Profiler>
  );
}
```

## Preventing Unnecessary Re-renders

### React.memo

**Basic Usage:**

```javascript
// Before: Re-renders on every parent update
function ExpensiveComponent({ data, config }) {
  return <div>{expensiveCalculation(data)}</div>;
}

// After: Only re-renders when props change
const ExpensiveComponent = React.memo(function ExpensiveComponent({
  data,
  config,
}) {
  return <div>{expensiveCalculation(data)}</div>;
});
```

**Custom Comparison:**

```javascript
const MyComponent = React.memo(
  function MyComponent({ items, threshold }) {
    return <div>{items.length > threshold ? "Many" : "Few"}</div>;
  },
  (prevProps, nextProps) => {
    // Custom comparison logic
    return (
      prevProps.items.length === nextProps.items.length &&
      prevProps.threshold === nextProps.threshold
    );
  }
);
```

### useMemo Hook

**Expensive Calculations:**

```javascript
function DataProcessor({ rawData, filters }) {
  // ❌ Recalculates on every render
  const processedData = processData(rawData, filters);

  // ✅ Only recalculates when dependencies change
  const processedData = useMemo(
    () => processData(rawData, filters),
    [rawData, filters]
  );

  return <DataVisualization data={processedData} />;
}
```

**Object/Array Creation:**

```javascript
function TodoList({ todos, filter }) {
  // ❌ Creates new array every render
  const filteredTodos = todos.filter((todo) => todo.category === filter);

  // ✅ Memoized filtering
  const filteredTodos = useMemo(
    () => todos.filter((todo) => todo.category === filter),
    [todos, filter]
  );

  return (
    <ul>
      {filteredTodos.map((todo) => (
        <TodoItem key={todo.id} todo={todo} />
      ))}
    </ul>
  );
}
```

### useCallback Hook

**Event Handlers:**

```javascript
function ParentComponent({ items }) {
  const [filter, setFilter] = useState("");

  // ❌ Creates new function every render
  const handleItemClick = (id) => {
    console.log("Clicked item:", id);
  };

  // ✅ Stable function reference
  const handleItemClick = useCallback((id) => {
    console.log("Clicked item:", id);
  }, []); // No dependencies

  return (
    <div>
      {items.map((item) => (
        <ItemComponent key={item.id} item={item} onClick={handleItemClick} />
      ))}
    </div>
  );
}
```

## State Management Optimization

### State Colocation

**Keep State Close to Where It's Used:**

```javascript
// ❌ State too high in tree
function App() {
  const [formData, setFormData] = useState({});
  const [isModalOpen, setIsModalOpen] = useState(false);

  return (
    <div>
      <Header />
      <MainContent />
      <Footer />
      {isModalOpen && <Modal formData={formData} setFormData={setFormData} />}
    </div>
  );
}

// ✅ State colocated with usage
function App() {
  return (
    <div>
      <Header />
      <MainContent />
      <Footer />
      <ModalSection />
    </div>
  );
}

function ModalSection() {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [formData, setFormData] = useState({});

  return isModalOpen ? (
    <Modal formData={formData} setFormData={setFormData} />
  ) : null;
}
```

### State Normalization

**Flatten Nested State:**

```javascript
// ❌ Nested state structure
const [data, setData] = useState({
  users: [{ id: 1, name: "John", posts: [{ id: 1, title: "Hello" }] }],
});

// ✅ Normalized state structure
const [users, setUsers] = useState({ 1: { id: 1, name: "John" } });
const [posts, setPosts] = useState({ 1: { id: 1, title: "Hello", userId: 1 } });
const [userPosts, setUserPosts] = useState({ 1: [1] });
```

### Reducer for Complex State

```javascript
// Complex state logic with useReducer
const initialState = {
  items: [],
  loading: false,
  error: null,
  filter: "all",
};

function itemsReducer(state, action) {
  switch (action.type) {
    case "FETCH_START":
      return { ...state, loading: true, error: null };
    case "FETCH_SUCCESS":
      return { ...state, loading: false, items: action.payload };
    case "FETCH_ERROR":
      return { ...state, loading: false, error: action.payload };
    case "SET_FILTER":
      return { ...state, filter: action.payload };
    default:
      return state;
  }
}

function ItemsList() {
  const [state, dispatch] = useReducer(itemsReducer, initialState);

  const filteredItems = useMemo(
    () => filterItems(state.items, state.filter),
    [state.items, state.filter]
  );

  return (
    <div>
      <FilterButtons
        currentFilter={state.filter}
        onFilterChange={(filter) =>
          dispatch({ type: "SET_FILTER", payload: filter })
        }
      />
      <ItemList items={filteredItems} />
    </div>
  );
}
```

## Component Patterns for Performance

### Component Composition

**Use Children Prop:**

```javascript
// ❌ All content re-renders when state changes
function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div>
      <Header onToggleSidebar={() => setSidebarOpen(!sidebarOpen)} />
      <Sidebar isOpen={sidebarOpen} />
      <MainContent>
        <ExpensiveComponent />
      </MainContent>
    </div>
  );
}

// ✅ Children don't re-render when sidebar state changes
function Layout({ children }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div>
      <Header onToggleSidebar={() => setSidebarOpen(!sidebarOpen)} />
      <Sidebar isOpen={sidebarOpen} />
      <MainContent>{children}</MainContent>
    </div>
  );
}

function App() {
  return (
    <Layout>
      <ExpensiveComponent />
    </Layout>
  );
}
```

### Splitting Components

**Break Down Large Components:**

```javascript
// ❌ Monolithic component
function Dashboard({ user, analytics, notifications }) {
  const [selectedTab, setSelectedTab] = useState("overview");

  return (
    <div>
      <UserProfile user={user} />
      <TabNavigation selectedTab={selectedTab} onTabChange={setSelectedTab} />
      {selectedTab === "overview" && <OverviewContent analytics={analytics} />}
      {selectedTab === "analytics" && (
        <AnalyticsContent analytics={analytics} />
      )}
      {selectedTab === "notifications" && (
        <NotificationsContent notifications={notifications} />
      )}
    </div>
  );
}

// ✅ Split into focused components
function Dashboard({ user, analytics, notifications }) {
  const [selectedTab, setSelectedTab] = useState("overview");

  return (
    <div>
      <UserProfile user={user} />
      <TabNavigation selectedTab={selectedTab} onTabChange={setSelectedTab} />
      <TabContent
        selectedTab={selectedTab}
        analytics={analytics}
        notifications={notifications}
      />
    </div>
  );
}

const TabContent = React.memo(function TabContent({
  selectedTab,
  analytics,
  notifications,
}) {
  switch (selectedTab) {
    case "overview":
      return <OverviewContent analytics={analytics} />;
    case "analytics":
      return <AnalyticsContent analytics={analytics} />;
    case "notifications":
      return <NotificationsContent notifications={notifications} />;
    default:
      return null;
  }
});
```

## List Optimization

### Virtualization

**For Large Lists:**

```javascript
import { FixedSizeList as List } from "react-window";

function VirtualizedList({ items }) {
  const Row = ({ index, style }) => (
    <div style={style}>
      <ItemComponent item={items[index]} />
    </div>
  );

  return (
    <List height={600} itemCount={items.length} itemSize={50} itemData={items}>
      {Row}
    </List>
  );
}
```

### Pagination vs Infinite Scroll

**Pagination Implementation:**

```javascript
function PaginatedList({ items, pageSize = 10 }) {
  const [currentPage, setCurrentPage] = useState(1);

  const paginatedItems = useMemo(() => {
    const startIndex = (currentPage - 1) * pageSize;
    return items.slice(startIndex, startIndex + pageSize);
  }, [items, currentPage, pageSize]);

  return (
    <div>
      <ItemsList items={paginatedItems} />
      <Pagination
        currentPage={currentPage}
        totalPages={Math.ceil(items.length / pageSize)}
        onPageChange={setCurrentPage}
      />
    </div>
  );
}
```

## Bundle Size Optimization

### Code Splitting

**Route-Based Splitting:**

```javascript
import { lazy, Suspense } from "react";
import { Routes, Route } from "react-router-dom";

// Lazy load route components
const Home = lazy(() => import("./pages/Home"));
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Profile = lazy(() => import("./pages/Profile"));

function App() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/profile" element={<Profile />} />
      </Routes>
    </Suspense>
  );
}
```

**Component-Based Splitting:**

```javascript
// Heavy component loaded on demand
const HeavyChart = lazy(() =>
  import("./HeavyChart").then((module) => ({
    default: module.HeavyChart,
  }))
);

function Dashboard() {
  const [showChart, setShowChart] = useState(false);

  return (
    <div>
      <button onClick={() => setShowChart(true)}>Show Chart</button>
      {showChart && (
        <Suspense fallback={<ChartSkeleton />}>
          <HeavyChart />
        </Suspense>
      )}
    </div>
  );
}
```

### Tree Shaking

**Import Only What You Need:**

```javascript
// ❌ Imports entire library
import _ from "lodash";
import { Button } from "@mui/material";

// ✅ Import specific functions
import debounce from "lodash/debounce";
import Button from "@mui/material/Button";
```

## Image and Asset Optimization

### Lazy Loading Images

```javascript
function LazyImage({ src, alt, ...props }) {
  const [imageSrc, setImageSrc] = useState(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const imgRef = useRef();

  useEffect(() => {
    let observer;

    if (imgRef.current && window.IntersectionObserver) {
      observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              setImageSrc(src);
              observer.unobserve(imgRef.current);
            }
          });
        },
        { threshold: 0.1 }
      );

      observer.observe(imgRef.current);
    }

    return () => {
      if (observer && observer.unobserve) {
        observer.unobserve(imgRef.current);
      }
    };
  }, [src]);

  return (
    <div ref={imgRef} {...props}>
      {imageSrc && (
        <img
          src={imageSrc}
          alt={alt}
          onLoad={() => setIsLoaded(true)}
          style={{ opacity: isLoaded ? 1 : 0 }}
        />
      )}
    </div>
  );
}
```

## Advanced Patterns

### Error Boundaries for Performance

```javascript
class PerformanceErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log performance-related errors
    console.error("Performance error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return <div>Something went wrong. Try refreshing.</div>;
    }

    return this.props.children;
  }
}
```

### Concurrent Features

**Suspense for Data Fetching:**

```javascript
function UserProfile({ userId }) {
  // This suspends until data is ready
  const user = use(fetchUser(userId));

  return (
    <div>
      <h1>{user.name}</h1>
      <Suspense fallback={<PostsSkeleton />}>
        <UserPosts userId={userId} />
      </Suspense>
    </div>
  );
}

function App() {
  return (
    <Suspense fallback={<ProfileSkeleton />}>
      <UserProfile userId={123} />
    </Suspense>
  );
}
```

## Performance Monitoring

### Core Web Vitals

```javascript
// Measure Core Web Vitals
function measureWebVitals() {
  // Largest Contentful Paint
  new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      console.log("LCP:", entry.startTime);
    }
  }).observe({ entryTypes: ["largest-contentful-paint"] });

  // First Input Delay
  new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      console.log("FID:", entry.processingStart - entry.startTime);
    }
  }).observe({ entryTypes: ["first-input"] });

  // Cumulative Layout Shift
  new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      if (!entry.hadRecentInput) {
        console.log("CLS:", entry.value);
      }
    }
  }).observe({ entryTypes: ["layout-shift"] });
}
```

### Custom Performance Hooks

```javascript
function usePerformanceTracker(componentName) {
  useEffect(() => {
    const startTime = performance.now();

    return () => {
      const endTime = performance.now();
      console.log(`${componentName} render time:`, endTime - startTime);
    };
  });
}

function ExpensiveComponent() {
  usePerformanceTracker("ExpensiveComponent");

  return <div>{/* component content */}</div>;
}
```

## Common Anti-Patterns

### Things to Avoid

**1. Premature Optimization:**

```javascript
// ❌ Over-optimizing without profiling
const MemoizedEverything = React.memo(
  useMemo(() => useCallback(() => <div>Simple static content</div>, []), [])
);

// ✅ Keep it simple for static content
function SimpleComponent() {
  return <div>Simple static content</div>;
}
```

**2. Incorrect Dependencies:**

```javascript
// ❌ Missing dependencies
const memoizedValue = useMemo(() => {
  return expensiveCalculation(a, b, c);
}, [a]); // Missing b and c

// ✅ Complete dependencies
const memoizedValue = useMemo(() => {
  return expensiveCalculation(a, b, c);
}, [a, b, c]);
```

**3. Inline Object Creation:**

```javascript
// ❌ Creates new object every render
<MyComponent config={{ theme: "dark", size: "large" }} />;

// ✅ Stable object reference
const config = { theme: "dark", size: "large" };
<MyComponent config={config} />;
```

## Performance Checklist

### Development Phase

- [ ] Use React DevTools Profiler
- [ ] Implement proper key props for lists
- [ ] Avoid inline object/function creation
- [ ] Use appropriate memo/useMemo/useCallback
- [ ] Keep state close to where it's used

### Build Phase

- [ ] Enable code splitting
- [ ] Optimize bundle size
- [ ] Implement tree shaking
- [ ] Compress images and assets
- [ ] Use production builds

### Deployment Phase

- [ ] Enable gzip/brotli compression
- [ ] Set up CDN for static assets
- [ ] Implement caching strategies
- [ ] Monitor Core Web Vitals
- [ ] Set up performance budgets

## Conclusion

React performance optimization is about finding the right balance between developer experience and user experience. Key principles:

1. **Measure First:** Use profiling tools before optimizing
2. **Start Simple:** Don't over-engineer early
3. **Focus on Impact:** Optimize the biggest bottlenecks first
4. **Monitor Continuously:** Performance is an ongoing concern

Remember: The best optimization is often removing unnecessary code rather than making existing code faster.
