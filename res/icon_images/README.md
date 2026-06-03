# Traffic element icons

PNG/SVG assets for HD map visualization (BEV and camera overlays).

Resolution order:

1. `$KITSCENES_TE_ICON_PATH` — optional override directory
2. This directory (`res/icon_images/` at the repository root)
3. Legacy Lanelet2 JOSM `style_images/` (dev fallback only)

Filenames must match the mapping in `kitscenes.visualization.ml_converter_vis_utils.te_type_to_icon_path()`
(e.g. `206.png`, `205.png`, `pf-g.png`, `traffic_light.png`, `traffic_sign.png`).

Generic fallbacks used when a specific sign file is missing:

- `traffic_light.png` — traffic lights
- `traffic_sign.png` — signs and unknown types
