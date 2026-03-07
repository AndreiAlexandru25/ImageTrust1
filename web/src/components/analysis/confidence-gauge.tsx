"use client";

import { useEffect, useState } from "react";

interface ConfidenceGaugeProps {
  value: number; // 0-1
  size?: number;
  label?: string;
}

export function ConfidenceGauge({
  value,
  size = 180,
  label = "AI Probability",
}: ConfidenceGaugeProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const radius = 70;
  const circumference = Math.PI * radius; // semicircle
  const offset = circumference - value * circumference;
  const center = size / 2;
  const strokeWidth = 12;

  // Color based on value
  const getColor = (v: number) => {
    if (v < 0.35) return "#22C55E";
    if (v < 0.65) return "#F59E0B";
    return "#EF4444";
  };

  const color = getColor(value);

  return (
    <div className="flex flex-col items-center">
      <svg
        width={size}
        height={size / 2 + 20}
        viewBox={`0 0 ${size} ${size / 2 + 20}`}
      >
        {/* Background arc */}
        <path
          d={`M ${center - radius} ${center} A ${radius} ${radius} 0 0 1 ${center + radius} ${center}`}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-muted"
          strokeLinecap="round"
        />

        {/* Value arc */}
        <path
          d={`M ${center - radius} ${center} A ${radius} ${radius} 0 0 1 ${center + radius} ${center}`}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={mounted ? offset : circumference}
          style={{ transition: "stroke-dashoffset 1.5s ease-out" }}
        />

        {/* Tick marks */}
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => {
          const angle = Math.PI - tick * Math.PI;
          const innerR = radius - strokeWidth / 2 - 4;
          const outerR = radius - strokeWidth / 2 - 8;
          const x1 = center + innerR * Math.cos(angle);
          const y1 = center - innerR * Math.sin(angle);
          const x2 = center + outerR * Math.cos(angle);
          const y2 = center - outerR * Math.sin(angle);
          return (
            <line
              key={tick}
              x1={x1}
              y1={y1}
              x2={x2}
              y2={y2}
              stroke="currentColor"
              strokeWidth={1.5}
              className="text-muted-foreground/40"
            />
          );
        })}

        {/* Labels */}
        <text
          x={center - radius - 5}
          y={center + 16}
          className="fill-muted-foreground text-[10px]"
          textAnchor="middle"
        >
          0%
        </text>
        <text
          x={center}
          y={center - radius + 2}
          className="fill-muted-foreground text-[10px]"
          textAnchor="middle"
        >
          50%
        </text>
        <text
          x={center + radius + 5}
          y={center + 16}
          className="fill-muted-foreground text-[10px]"
          textAnchor="middle"
        >
          100%
        </text>

        {/* Center value */}
        <text
          x={center}
          y={center - 5}
          className="fill-foreground font-bold text-2xl"
          textAnchor="middle"
          fontFamily="var(--font-jetbrains), monospace"
        >
          {mounted ? `${(value * 100).toFixed(1)}%` : "0.0%"}
        </text>
      </svg>
      <p className="text-xs text-muted-foreground -mt-2">{label}</p>
    </div>
  );
}
