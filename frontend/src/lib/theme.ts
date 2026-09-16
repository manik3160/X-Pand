/**
 * Hex mirrors of the CSS custom properties in src/index.css.
 * Canvas (Leaflet) and recharts can't read CSS variables directly, so
 * anything drawn there pulls its colours from here instead. Keep these
 * in sync with the :root tokens by hand.
 */
export const theme = {
  bg: '#0b0d10',
  surface: '#14171c',
  text: '#e8eaed',
  textMuted: '#a0a6b0',
  textSubtle: '#808792',
  accent: '#7aa2ff',
  accentFg: '#0b0d10',
  profitHigh: '#34d399',
  profitMid: '#fbbf24',
  profitLow: '#f87171',
} as const

export type ProfitTier = 'high' | 'mid' | 'low'

export function profitTier(p: number): ProfitTier {
  if (p > 0.7) return 'high'
  if (p >= 0.4) return 'mid'
  return 'low'
}

const TIER_COLOR: Record<ProfitTier, string> = {
  high: theme.profitHigh,
  mid: theme.profitMid,
  low: theme.profitLow,
}

const TIER_FILL_OPACITY: Record<ProfitTier, number> = {
  high: 0.55,
  mid: 0.35,
  low: 0.12,
}

export function profitColorHex(p: number): string {
  return TIER_COLOR[profitTier(p)]
}

export function profitFillOpacity(p: number): number {
  return TIER_FILL_OPACITY[profitTier(p)]
}

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

export function profitColorRgb(p: number): [number, number, number] {
  return hexToRgb(profitColorHex(p))
}
