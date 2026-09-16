import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import { profitColorHex, profitColorRgb, profitFillOpacity } from "./theme"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatPercent(value: number): string {
  // Clamp display below 100% — a model score of 0.995-0.999 should never
  // read as a false-certain "100.0%".
  const pct = Math.min(value * 100, 99.9)
  return `${pct.toFixed(1)}%`
}

export function formatNumber(value: number): string {
  return value.toLocaleString('en-IN')
}

export function getRecommendationColor(rec: string): string {
  switch (rec.toLowerCase()) {
    case 'open': return 'text-success'
    case 'monitor': return 'text-warning'
    case 'skip': return 'text-danger'
    default: return 'text-text-secondary'
  }
}

export const getProfitColor = profitColorRgb
export const getProfitColorHex = profitColorHex
export const getProfitFillOpacity = profitFillOpacity
