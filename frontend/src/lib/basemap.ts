/**
 * Basemap tile source. Esri's World Dark Gray Canvas needs no API key
 * (unlike CARTO's dark_all, which now stamps tiles with an "API key
 * required" watermark) and is a calm, low-contrast dark style meant for
 * data overlays.
 *
 * Override with VITE_BASEMAP_URL if this ever needs to change without a
 * code edit (e.g. switching providers).
 *
 * Note the tile path order: {z}/{y}/{x}, not the usual {z}/{x}/{y}.
 */
const ESRI_BASE =
  'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}'
const ESRI_LABELS =
  'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}'

export const basemap = {
  url: import.meta.env.VITE_BASEMAP_URL || ESRI_BASE,
  labelsUrl: ESRI_LABELS,
  attribution:
    'Powered by <a href="https://www.esri.com">Esri</a> | Esri, HERE, Garmin, &copy; OpenStreetMap contributors',
  maxNativeZoom: 16,
  maxZoom: 18,
} as const
