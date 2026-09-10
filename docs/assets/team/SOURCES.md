# Orukeet team marks

These web images identify the same five organizations as the technical report. Oruk AI is the primary brand; Stanford University, University of Cambridge, OpenWhispr and Hoid identify the authors' affiliations.

The [report's source register](https://github.com/Oruk-AI/orukeet/blob/main/report/assets/affiliations/SOURCES.md) records the original organization assets and URLs. The original SVG and PNG files are retained there. Web versions preserve the complete marks and their colors, with proportional resizing and a white background for legibility in light and dark model-card themes.

Oruk uses the September 9, 2026 [primary lockup](https://oruk.ai/brand/kit/oruk-primary.svg?v=2026-09-09-light-signal) from the [official brand kit](https://oruk.ai/branding): the color Signal tile and lowercase black wordmark. The original PNG and outlined SVG are retained as `oruk-primary.png` and `oruk-primary.svg`. The PNG is the rendering source so the Signal tile’s transparent wave cut is preserved. Its web image includes at least half a tile of clear space on every side. This web asset is maintained separately from the artwork in the published technical report.

The marks remain the property of their respective organizations. Code and model licenses do not grant rights to these marks.

Author order and affiliations match the technical report and `CITATION.cff`. `docs/team.json` records the original and web-image hashes. Regenerate the web assets and bylines with `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib python scripts/sync_release_team.py`, or verify the pages with `python scripts/sync_release_team.py --check`.
