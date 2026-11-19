I've built all seven procedural assets (rug, floor lamp, table lamp, bed, curtain, desk, throw) plus the rug unroll animation. The test file passes (182 tests in about 7 s) and I looked at every render. Next to the scanned Poly Haven sofa, tables, plant and vase in a mixed scene they hold up at a normal exposure. I didn't commit anything and didn't touch any file outside my list. Some motion tags and two parameter conventions need changes in files you own (last section).

**Files** (all in `/home/user/softwareEditor`)
- `mobilia/assets/__init__.py`: no `bpy` import. It holds the list of kinds and motions, the bed's shared measurements (`bed_dimensions`), and a new `support_surface(kind, size, params)` that says where on-top items can rest.
- `mobilia/assets/procedural.py`: `build_asset(kind, params, name, *, collection=None)`, `animate_rug_unroll(root, frame_start, frame_end, *, axis=None, reverse=False, substeps=1)` and the seven builders. Parts are positioned by translation only, and each part's origin is its animation pivot. `kind` also accepts `"proc:bed"`; sizes come from `params["size"]` (zeros mean the catalog default).
- `mobilia/assets/materials.py`: procedural materials with no external textures:
  - wool rug in four patterns: solid with border, Beni Ourain-style lines, stripes, and a Persian-style design (each with a darker back side)
  - fabrics (linen, velvet, bouclé, cotton) and lamp shade linen (slightly translucent)
  - bed linen, knit for the throw
  - brushed metal (black, graphite, brass, chrome) plus matte black and white powder-coat
  - oak/walnut wood that can take a palette tone, paint, ceramic, an opal bulb, felt
  - Nothing emits light, even when `light_on=True` is passed.
- `scripts/preview_procedural_assets.py`: studio renders, contact sheet and unroll strip; `--device CPU|OPTIX|CUDA|AUTO` (falls back to CPU).
- `tests/test_procedural_assets.py`.

**Assembly parts (order, motion)**
| Kind | Parts |
|---|---|
| rug | body + fringe (0, unroll) |
| floor lamp | base (0, drop), pole (1, rise, pivot at its foot), shade with fittings and bulb (2, drop) |
| table lamp | base (0, drop), shade (1, drop) |
| bed | legs + frame (0, rise), headboard (1, rise), mattress (2, drop), duvet (3, settle), pillows (4–6, drop/settle) |
| curtain | rod (0, slide), panels (1, unfold, pivot at the top) |
| desk | legs (0, rise), apron (1, rise), top (2, drop), drawer (3, slide) |
| throw | throw (0, settle) |

**Rug unroll**
- It uses one shape key per frame, computed as a spiral that rolls without slipping, with linear weights between frames. The pile side is on the inside, as with a real rolled rug.
- Measured on the default rug: a 20 cm-diameter roll inside the rug's final footprint at the start, Z ≥ 0 on every frame, and exactly 12 mm flat at the end.
- It also works along the other axis or in reverse. Cycles motion blur renders it cleanly.
- The last curl overshoots the final footprint by about 1.3 mm for one frame.

**How I validated**
- `/home/user/.venvs/staging/bin/python -m pytest tests/test_procedural_assets.py -q` → 182 passed. It runs a `bpy` subprocess over 28 cases: defaults, custom sizes and styles, and parameters exactly as `style.proc_params` sends them. It checks:
  - size within 2%, base at Z = 0, centred
  - the size-matching rescale stays within 5%
  - part properties and materials present, no NaN
  - nothing emits light
  - `mattress_top` honoured
  - `support_surface` matches the built meshes (checked by casting rays down onto them)
  - the unroll: rolled at start, flat at end, never below the floor, always advancing
  - bad input raises errors
- Every closed piece has outward-facing normals, and Blender's mesh validation removes nothing. pyflakes reports nothing.
- I ran all seven kinds through your `blender_worker.build_procedural_item` + `place_root` with real `proc_params` output. The fade wrapper wraps all my materials and the unroll works.
- Preview: `python scripts/preview_procedural_assets.py` takes about 4 minutes at 480 px / 32 samples on the shared CPU.

**Renders** (in `/tmp/claude-0/proc_assets/`)
- `contact_sheet.png` and one PNG per asset, including four rug patterns
- `rug_unroll_strip.png` (5 frames)
- `vignette_with_polyhaven.png` (my assets next to the scanned ones)
- `variants/variants_sheet.png` and `variants_layout/variants_sheet.png` (other styles and the layout's real parameters)
- close-ups: `_dev_bed_zoom.png`, `_dev_roll_close.png`, `_dev_tlamp_top.png`, `_dev_unroll_motionblur.png`, `_dev_berber_top.png`, `_dev_persian_top.png`

**Known limitations**
- Folds and wrinkles come from formulas and noise, not cloth simulation. Curtains can look a little regular and the duvet corners are smooth cones.
- The Persian design is a simplified tribal/kilim style, not intricate florals.
- Pillows touch with slight soft overlap.
- `procedural.py` is about 1,700 lines. I could split it into primitives and builders, but that needs files I don't own.
- Very low headboards can't hold the pillow stack: on an 85 cm single bed the pillows recline until they fit, and below roughly 0.75 m they would poke above the headboard.

**Changes needed in your files**
1. `mobilia/render/choreography.py`: `MOTIONS` has no `unfold` or `settle`, so those parts currently fall back to `drop`.
   - **`unfold`:** scale Z from 0 to 1 about the part's pivot. The pivot is already at the top of each curtain panel.
   - **`settle`:** a small drop with a slight squash.
   - **Curtain role:** the curtain is set to `"slide"`, which overrides my per-part tags for it.
2. `mobilia/layout/surfaces.py` `support_for`: for the bed it uses the mattress top (0.56). Items actually rest on the duvet at 0.602, and the folded-back strip near the pillows is about 4 cm higher still. Please use `mobilia.assets.support_surface(kind, size, params)` for the procedural bed and desk.
3. `normalize_params` already translates everything `proc_params` sends. If a `light_on=True` comes back, it is ignored on purpose, because lamps must stay off so the photo's lighting doesn't change.
4. For short unrolls with motion blur, I suggest calling `animate_rug_unroll(..., substeps=2)`.