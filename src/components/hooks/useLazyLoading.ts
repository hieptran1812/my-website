import { useState, useEffect, useCallback } from "react";

interface UseLazyLoadingParams<T> {
  initialData: T[];
  loadMoreData: (page: number, limit: number) => Promise<T[]>;
  itemsPerPage: number;
  hasMore: boolean;
  scrollThreshold?: number; // % of the window height from the bottom to trigger loading
  getItemId?: (item: T) => string; // Function to get unique ID from item
}

export function useLazyLoading<T>({
  initialData,
  loadMoreData,
  itemsPerPage,
  hasMore,
  scrollThreshold = 80, // default to 80% of the window height
  getItemId,
}: UseLazyLoadingParams<T>) {
  const [data, setData] = useState<T[]>(initialData);
  const [loading, setLoading] = useState(false);
  const [hasMoreData, setHasMoreData] = useState(hasMore);
  const [page, setPage] = useState(1);

  // Function to handle loading more data
  const loadMore = useCallback(async () => {
    if (loading || !hasMoreData) return;

    setLoading(true);
    try {
      const nextPage = page + 1;
      const newData = await loadMoreData(nextPage, itemsPerPage);

      if (newData.length === 0) {
        setHasMoreData(false);
      } else {
        setData((prevData) => {
          // Remove duplicates using getItemId if provided
          if (getItemId) {
            const existingIds = new Set(prevData.map(getItemId));
            const uniqueNewData = newData.filter(
              (item) => !existingIds.has(getItemId(item))
            );
            return [...prevData, ...uniqueNewData];
          } else {
            return [...prevData, ...newData];
          }
        });
        setPage(nextPage);
        setHasMoreData(newData.length >= itemsPerPage);
      }
    } catch (error) {
      console.error("Error loading more data:", error);
    } finally {
      setLoading(false);
    }
  }, [loading, hasMoreData, page, loadMoreData, itemsPerPage, getItemId]);

  // Function to reset the lazy loading state
  const reset = useCallback(
    (newInitialData: T[], newHasMore: boolean = true) => {
      setData(newInitialData);
      setPage(1);
      setLoading(false);
      setHasMoreData(newHasMore && newInitialData.length >= itemsPerPage);
    },
    [itemsPerPage]
  );

  // Handle scroll event to automatically load more
  useEffect(() => {
    const handleScroll = () => {
      if (loading || !hasMoreData) return;

      const windowHeight = window.innerHeight;
      const documentHeight = document.documentElement.scrollHeight;
      const scrollTop = window.scrollY || document.documentElement.scrollTop;

      // Calculate how far down the user has scrolled (as a percentage)
      const scrolledPercentage =
        ((scrollTop + windowHeight) / documentHeight) * 100;

      // If the user has scrolled beyond the threshold, load more data
      if (scrolledPercentage > scrollThreshold) {
        loadMore();
      }
    };

    window.addEventListener("scroll", handleScroll);

    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, [loadMore, loading, hasMoreData, scrollThreshold]);

  return {
    data,
    loading,
    hasMoreData,
    loadMore, // Still exposing the loadMore function for manual triggering if needed
    reset,
  };
}
