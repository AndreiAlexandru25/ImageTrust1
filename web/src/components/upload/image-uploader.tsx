"use client";

import { useCallback, useRef, useState } from "react";
import { Upload, ImageIcon, X, FileWarning } from "lucide-react";
import { cn } from "@/lib/utils";
import { ACCEPTED_FILE_TYPES, MAX_FILE_SIZE_MB } from "@/lib/constants";
import { Button } from "@/components/ui/button";

interface ImageUploaderProps {
  onFileSelect: (file: File) => void;
  preview: string | null;
  onClear: () => void;
  disabled?: boolean;
}

const ACCEPTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp"];

export function ImageUploader({
  onFileSelect,
  preview,
  onClear,
  disabled = false,
}: ImageUploaderProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateFile = useCallback((file: File): string | null => {
    // Check by MIME type OR by extension (Windows sometimes has empty MIME)
    const ext = "." + file.name.split(".").pop()?.toLowerCase();
    const validType = ACCEPTED_FILE_TYPES.includes(file.type);
    const validExt = ACCEPTED_EXTENSIONS.includes(ext);

    if (!validType && !validExt) {
      return "Unsupported format. Use JPEG, PNG, or WebP.";
    }
    if (file.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      return `File too large. Maximum ${MAX_FILE_SIZE_MB}MB.`;
    }
    return null;
  }, []);

  const handleFile = useCallback(
    (file: File) => {
      const err = validateFile(file);
      if (err) {
        setError(err);
        return;
      }
      setError(null);
      onFileSelect(file);
    },
    [validateFile, onFileSelect],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
      // Reset input so same file can be re-selected
      if (inputRef.current) inputRef.current.value = "";
    },
    [handleFile],
  );

  const handleClick = useCallback(() => {
    if (!disabled) {
      inputRef.current?.click();
    }
  }, [disabled]);

  if (preview) {
    return (
      <div className="relative rounded-xl overflow-hidden border bg-card">
        <img
          src={preview}
          alt="Selected image"
          className="w-full max-h-[400px] object-contain bg-black/5 dark:bg-white/5"
        />
        <Button
          variant="ghost"
          size="icon"
          className="absolute top-3 right-3 h-8 w-8 bg-background/80 backdrop-blur-sm hover:bg-background"
          onClick={onClear}
          disabled={disabled}
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Hidden file input */}
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_FILE_TYPES.join(",")}
        onChange={handleInputChange}
        className="hidden"
        disabled={disabled}
      />

      {/* Drop zone - plain div with onClick */}
      <div
        onClick={handleClick}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        className={cn(
          "flex flex-col items-center justify-center gap-4 rounded-xl border-2 border-dashed p-12 transition-all cursor-pointer select-none",
          isDragging
            ? "border-primary bg-primary/5 scale-[1.02]"
            : "border-muted-foreground/25 hover:border-muted-foreground/50 hover:bg-muted/50",
          disabled && "opacity-50 pointer-events-none",
        )}
      >
        <div className="flex items-center justify-center h-16 w-16 rounded-2xl bg-primary/10">
          {isDragging ? (
            <Upload className="h-8 w-8 text-primary" />
          ) : (
            <ImageIcon className="h-8 w-8 text-primary" />
          )}
        </div>

        <div className="text-center space-y-1.5">
          <p className="text-sm font-medium">
            {isDragging ? "Drop image here" : "Drop an image or click to browse"}
          </p>
          <p className="text-xs text-muted-foreground">
            JPEG, PNG, or WebP &middot; Max {MAX_FILE_SIZE_MB}MB
          </p>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 text-sm text-destructive bg-destructive/10 rounded-lg px-4 py-2.5 animate-[fade-in_0.2s_ease-out]">
          <FileWarning className="h-4 w-4 shrink-0" />
          {error}
        </div>
      )}
    </div>
  );
}
