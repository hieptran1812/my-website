// Date utility functions for safe date formatting

export function formatDate(dateString: string): string {
  if (!dateString) return "Unknown date";
  
  const date = new Date(dateString);
  if (isNaN(date.getTime())) {
    // Try to parse different date formats
    const parsed = Date.parse(dateString);
    if (!isNaN(parsed)) {
      return new Date(parsed).toLocaleDateString("en-US", {
        year: "numeric",
        month: "long", 
        day: "numeric",
      });
    }
    return "Invalid date";
  }
  
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}

export function formatDateShort(dateString: string): string {
  if (!dateString) return "Unknown";
  
  const date = new Date(dateString);
  if (isNaN(date.getTime())) {
    // Try to parse different date formats
    const parsed = Date.parse(dateString);
    if (!isNaN(parsed)) {
      return new Date(parsed).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
      });
    }
    return "Invalid";
  }
  
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

export function formatDateMedium(dateString: string): string {
  if (!dateString) return "Unknown date";
  
  const date = new Date(dateString);
  if (isNaN(date.getTime())) {
    // Try to parse different date formats
    const parsed = Date.parse(dateString);
    if (!isNaN(parsed)) {
      return new Date(parsed).toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
    }
    return "Invalid date";
  }
  
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function isValidDate(dateString: string): boolean {
  if (!dateString) return false;
  
  const date = new Date(dateString);
  if (!isNaN(date.getTime())) return true;
  
  // Try to parse different date formats
  const parsed = Date.parse(dateString);
  return !isNaN(parsed);
}
