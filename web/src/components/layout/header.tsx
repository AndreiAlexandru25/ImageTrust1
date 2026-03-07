"use client";

import Link from "next/link";
import { Shield } from "lucide-react";
import { ThemeToggle } from "@/components/shared/theme-toggle";

export function Header() {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/80 backdrop-blur-sm">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="flex items-center justify-center h-9 w-9 rounded-lg bg-primary/10 group-hover:bg-primary/20 transition-colors">
            <Shield className="h-5 w-5 text-primary" />
          </div>
          <div className="flex flex-col">
            <span className="text-lg font-bold tracking-tight leading-none">
              ImageTrust
            </span>
            <span className="text-[10px] text-muted-foreground leading-none mt-0.5">
              Forensic Image Analysis
            </span>
          </div>
        </Link>

        <nav className="flex items-center gap-2">
          <Link
            href="/"
            className="text-sm text-muted-foreground hover:text-foreground transition-colors px-3 py-2"
          >
            Analyze
          </Link>
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
}
