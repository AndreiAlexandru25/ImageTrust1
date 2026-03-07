import { cn } from "@/lib/utils";

interface StatusDotProps {
  status: "success" | "warning" | "error" | "info" | "neutral";
  pulse?: boolean;
  className?: string;
}

const statusColors = {
  success: "bg-verdict-real",
  warning: "bg-verdict-uncertain",
  error: "bg-verdict-ai",
  info: "bg-primary",
  neutral: "bg-muted-foreground",
};

export function StatusDot({ status, pulse = false, className }: StatusDotProps) {
  return (
    <span className={cn("relative flex h-2.5 w-2.5", className)}>
      {pulse && (
        <span
          className={cn(
            "animate-ping absolute inline-flex h-full w-full rounded-full opacity-75",
            statusColors[status],
          )}
        />
      )}
      <span
        className={cn(
          "relative inline-flex rounded-full h-2.5 w-2.5",
          statusColors[status],
        )}
      />
    </span>
  );
}
