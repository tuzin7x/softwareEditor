## Automatic layout engine: final report

The engine works end to end. The validation run is 70 tests passing in `tests/test_layout.py` (2 min 20 s, with CPU contention from Blender renders), and pyflakes is clean.

**About the user's question.** The engine never depends on the fixture photos. Everything is computed at runtime from whatever client photo gets analysed, so `tests/fixtures/photos/*` are test data only. A line you can pass on: *"A foto do Stafford é só para testes automáticos; no produto, o software usa a foto que o seu cliente enviar."*

### API
`plan_layout(room, grid, room_type, photo_rgb8, seed=0, seg=None) -> Layout`, in `mobilia/layout/solver.py`.
- It is deterministic. The same seed gave identical JSON in separate processes with `PYTHONHASHSEED` 1 and 12345.
- It takes about 1–4 s per run on this shared CPU.
- An unknown room type raises `ValueError`.
- Two helpers for the pipeline: `validate_layout(layout, LayoutContext(room, grid))` (an empty list means OK) and `debug.save_layout_debug(layout, room, grid, photo, out_dir, prefix)`.
- `Layout.notes` holds Portuguese text for the UI or report: the style, where the anchor piece went, and which items had no room.

### Files created (all new; `model.py` and `catalog.py` untouched)
- `mobilia/layout/solver.py` — the search flow:
  - main group: candidates, then simulated annealing;
  - wall art, which goes above the sofa/bed/console when possible, at about 2/3 of its width (single frame or a pair), centred at 1.50 m;
  - secondary pieces, then a polish pass;
  - decor on top of furniture, assembly order, and notes.
- `mobilia/layout/rules.py` — hard constraints, the scoring terms with every weight documented in the module docstring, and `validate_layout`.
- `mobilia/layout/style.py` — style analysis:
  - floor material and tone, read from a top-down "rectified" view of the floor at a fixed 1.2 cm/pixel, plus CIELAB colour; wall colour; warmth;
  - picks the style (moderno / escandinavo / clássico / industrial) and a palette;
  - ranks catalog assets by style and tone contrast;
  - builds the colour parameters for procedural assets.
- `mobilia/layout/planners.py` — one main-group planner per room type:
  - sala: conversation group;
  - quarto: bed + nightstands + rug under the lower 2/3;
  - sala_jantar and cozinha: table + chairs; bar stools if a kitchen island (blocked area not touching a wall) is detected;
  - escritorio: desk + chair;
  - tight rooms fall back to a sofa without coffee table, or a single bed.
- `mobilia/layout/context.py` (room/grid/camera pre-computation), `state.py` (cached per-piece geometry), `slots.py` (candidate positions), `geom.py` (geometry helpers).
- `mobilia/layout/surfaces.py` — real seat heights and table-top areas, measured by ray-casting the downloaded glTF meshes in Blender. For example, sofa_02's seat is at 0.406 m, so the pillows sit at that height.
- `mobilia/layout/debug.py` — top-down map plus 3D wireframe boxes drawn over the photo.
- `tests/test_layout.py` — includes a synthetic rectangular-room builder (4 rooms with different sizes and camera turns). Across the fixture and those rooms, for all 6 room types, it checks:
  - every piece is fully inside the frame (projected independently of the engine);
  - no floor pieces overlap, and bases sit on free or ≤25% unknown floor;
  - decor sits on its parent at the measured height;
  - wall art is flush with the wall and clear of windows and doors;
  - Poly Haven assets keep their real size (scale 1);
  - assembly order and determinism.

  Further tests check that the Stafford living room is complete, that the style result does not change with photo resolution or segmentation, that a room with no detected walls still works, and that an invalid room type is rejected.

### What was checked visually
I looked at every image below.
- Stafford, all six types at seed 0: `/tmp/claude-0/layout/sheet_stafford_todos_s0.jpg`, plus per-type `stafford_<type>_s0_foto.jpg`, `_topo.png` and `.json`.
- Stafford at seed 7, the product default: `/tmp/claude-0/layout/sheet_stafford_s7.jpg`.
- Synthetic rooms: `/tmp/claude-0/layout/sheet_sinteticas_s0.jpg`.
- The second real photo, `sala_cozinha_vazia_dk.jpg`: `/tmp/claude-0/layout/sheet_dk_s7.jpg`. I ran your analysis scripts on it myself; outputs are in `/tmp/claude-0/-home-user/c09d70e1-5a72-5f0b-811a-6fd083338d9a/scratchpad/dk/`.

**Stafford sala:**
- Dark sofa on the light carpet, against the left wall P0, seen side-on.
- Coffee table 0.45 m in front, centred and parallel.
- One armchair at the far end facing the camera.
- 2.4×1.7 m rug under the sofa's front legs and the armchair, with the coffee table fully on it.
- Side table at the sofa's near end, floor lamp at the far end, medium plant in the bay.
- Pair of frames on P1 (the wall above the sofa is all window).
- Pillows on the seat; books, bowl and vase on the coffee table.
- Nothing is cut by the frame.

The bedroom, office and dining layouts are coherent too. In the wide synthetic room, the sala and quarto come out as textbook staging: sofa or bed facing the camera, art above it, symmetric side pieces.

### Known limitations
- The floor classifier is calibrated on only two real photos (carpet and wood); tile and cement thresholds are untested on real photos. Its result no longer changes with resolution, JPEG quality or whether segmentation is given.
- Framing, occlusion and window-coverage checks use 3D bounding boxes, so they are conservative.
- Stafford at seed 0 has no large plant: no free corner was left without crowding the lamp or hiding the art.
- Bathroom decor tends to cluster in one corner.
- Kitchen counter heights are unknown, so nothing is placed on existing counters.
- In the dk photo, the open door leaf was analysed as a wall plus a door, and its 0.9 m clearance zone takes much of the floor. The result is a sofa without a coffee table and a single bed.
- A pair of frames is two items with role `wall_art`, although the program says at most 1.
- The office desk chair uses a dining-chair asset with role `desk_chair`, which is not in the program.

### Changes needed in files I don't own
1. **`catalog.py` sizes and orientation:**
   - `ph:modern_coffee_table_01`: the raw glTF has its long axis on Y (0.60×1.20). It needs `yaw_fix=90`, or it will render rotated 90° against its footprint.
   - `ph:wooden_display_shelves_01`: same problem (raw 0.37×1.08). It needs `yaw_fix=±90`; check which side is the front.
   - `ph:throw_pillows_01`: the catalog says 1.021×0.627, but the mesh is 0.94×0.45. Use the measured mesh box.
   - `surfaces.py` assumes each normalized asset's origin is the centre of its mesh bounding box.
2. **`catalog.py` programs and assets:**
   - Add `("desk_chair", True, 1)` and an office chair to the escritorio program.
   - Consider a loveseat or accent chair for small rooms.
3. **Fixture `tests/fixtures/rooms/stafford/seg.npz`:** its stored `groups` are stale (the window group is empty). `labels_to_groups(labels, id2label)` fixes it. I'd also add my dk outputs (`dk_room.json`, `dk_grid.npy`, `dk_seg.npz`) as a second fixture.
4. **Pipeline:** call `plan_layout(room, debug["grid"], req.room_type, photo.rgb8, seed=req.seed, seg=seg)`, log `layout.notes`, and save the debug images into the job folder.
5. **Renderer:**
   - For `proc:*` items, `params["size"]` now equals `item.size`. Your `procedural.normalize_params` already accepts all my parameter names (rug pattern names, `pattern_color`, `metal_color`, `frame_color`, `linen_color`, `accent_color`, `mattress_top`, `headboard_depth`, `top_color`, `leg_color`).
   - Lamps carry no "light on" flag; they stay off.
   - On-top decor already has its final z (parent base + measured surface height), so add no extra offset.
   - The blanket on the bed assumes `mattress_top = 0.56`.
