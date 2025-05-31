import { useState, useEffect, useCallback } from "react";

interface LazyLoadingOptions<T> {
  initialData: T[];
  loadMoreData: (page: number, limit: number) => Promise<T[]>;
  itemsPerPage: number;
  hasMore?: boolean;
}

export function useLazyLoading<T>({
  initialData,
  loadMoreData,
  itemsPerPage,
  hasMore = true,
}: LazyLoadingOptions<T>) {
  const [data, setData] = useState<T[]>(initialData);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [hasMoreData, setHasMoreData] = useState(hasMore);

  const loadMore = useCallback(async () => {
    if (loading || !hasMoreData) return;

    setLoading(true);
    setError(null);

    try {
      const newData = await loadMoreData(page + 1, itemsPerPage);

      if (newData.length === 0) {
        setHasMoreData(false);
      } else {
        setData((prevData) => [...prevData, ...newData]);
        setPage((prevPage) => prevPage + 1);

        // If we got fewer items than requested, we've reached the end
        if (newData.length < itemsPerPage) {
          setHasMoreData(false);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load more data");
    } finally {
      setLoading(false);
    }
  }, [loading, hasMoreData, loadMoreData, page, itemsPerPage]);

  const reset = useCallback(
    (newInitialData: T[]) => {
      setData(newInitialData);
      setPage(1);
      setHasMoreData(newInitialData.length >= itemsPerPage);
      setError(null);
      setLoading(false);
    },
    [itemsPerPage]
  );

  return {
    data,
    loading,
    error,
    hasMoreData,
    loadMore,
    reset,
  };
}
