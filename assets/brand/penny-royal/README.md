# Penny Royal brand kit

Display name is exactly **Penny Royal** — two words, capital P and R.
`pennyroyal` (one word, lowercase) is the CLI/identifier spelling only.

## Workflow

1. Reuse these files as-is; do not redraw the emblem or restyle the wordmark.
2. Pick the mark for the background: light → `*-black` / `*-light` files; dark or photographic → `*-white` / `*-dark`.
3. Logo files carry the identity and work at any size; detailed artwork carries atmosphere — use it for heroes and large graphics.
4. Preserve the name spelling, the eight-facet silhouette, the aspect ratios and the palette.

## Files

| File | Use |
|---|---|
| `logos/emblem-black.svg` | Standalone emblem, exact black `#000000`, transparent |
| `logos/emblem-white.svg` | Standalone emblem, exact white `#FFFFFF`, transparent |
| `logos/logo-stacked-light.svg` | Stacked lockup, pure black, light backgrounds |
| `logos/logo-stacked-dark.svg` | Stacked lockup, pure white, dark backgrounds |
| `logos/logo-horizontal-light.svg` | Horizontal lockup, pure black, light backgrounds |
| `logos/logo-horizontal-dark.svg` | Horizontal lockup, pure white, dark backgrounds |
| `png/emblem-black-1024.png`, `png/emblem-white-1024.png` | Transparent 1024×1024 emblems |
| `png/emblem-black-32.png`, `png/emblem-black-128.png`, `png/emblem-white-32.png`, `png/emblem-white-128.png` | Transparent emblem icons (favicon/avatars) |
| `png/logo-stacked-light-1600.png`, `png/logo-stacked-dark-1600.png` | 1600×1948 transparent stacked lockups |
| `png/logo-horizontal-light-1600.png`, `png/logo-horizontal-dark-1600.png` | 1600×416 transparent horizontal lockups |
| `reference/approved-concept.png` | Approved original concept image, 1536×1024 RGB |
| `artwork/penny-royal-crystal.png` | Text-free crystal hero artwork, 1672×941 RGB, wide near-16:9; artwork on the right with dark left space for page headings |
| `artwork/penny-royal-crystal-transparent.png` | Square crystal cutout, 1254×1254 RGBA with genuine alpha |
| `artwork/PROMPTS.md` | Source prompts for the supplied artwork |
| `brand.json` | Machine-readable names, palette, asset roles |
| `FONT-LICENSE.txt` | License/attribution for the outlined wordmark lettering |

All marks — emblems and lockups — are pure monochrome (exact black or white, fill-only vectors, no strokes, no external fonts or resources). Violet lives in the detailed artwork and page accents, never in the flat logo. SVG intrinsic sizes: emblem viewBox `67 114 400 400`; stacked lockup 400×487; horizontal lockup 1250×325.

## Clear space and minimum size

- Clear space on every side ≥ 15% of the emblem width (≈48 units of the emblem's 400-unit canvas).
- Minimum recommended emblem size: 32 px (`png/emblem-*-32.png` is the reference render).

## Palette

| Token | Hex | Role |
|---|---|---|
| obsidian | `#0B0D10` | Dark surfaces; main artwork body |
| charcoal | `#171A20` | Secondary surfaces |
| silver | `#CBD0D8` | Edges; text on dark |
| violet | `#7042A6` | Restrained accent in artwork/page accents only |
| paper | `#F4F3F0` | Light background |

Logo marks use exact black/white only. Text: silver or white on dark, obsidian on light. Violet never carries text.

## Typography and layout

- The wordmark is DejaVu Sans ExtraLight converted to native SVG outlines — consumers need no installed font (see `FONT-LICENSE.txt`).
- Headings: clean, readable sans-serif with restrained tracking. Body: normal readable weights, following the host site's established typography. Always render the name as two words.
- Stacked lockup for centered/hero placement; horizontal lockup for nav bars and footers.

## Alt text

- Emblem alone: `Penny Royal emblem`
- Any lockup: `Penny Royal`

## Origin

The silhouette was measured from the approved crystalline thistle/crest concept: eight disconnected blade-like facets, tall central facet, asymmetrical flanking shards, polished-obsidian feel with silver edges and restrained violet in the fractures. Preserve it. The name is inspired by Penny Royal, the AI in Neal Asher's Polity universe — see https://www.nealasher.co.uk/penny-royal-iii-started/
