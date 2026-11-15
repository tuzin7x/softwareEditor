# Automatic furniture layout for single-photo, fixed-camera staging: research report (2026-11-15)

## 1. Findings

**Bottom line.** None of the systems I surveyed solves our exact problem: one photo, a locked camera, only the visible floor known, and quality judged in that one frame. Published systems assume a fully known floor plan and a free camera. The learned and LLM-based ones are either non-commercial or need paid APIs. What we can reuse is their *formulations* (Merrell 2011, Make it Home 2011, Infinigen Indoors 2024). We reimplement those ourselves and add terms computed through the camera projection.

**Classical optimization (papers only; I found no official code, UNVERIFIED).**
- *Make it Home* (Yu et al., TOG 2011):
  - Cost: C = wa·Ca + wv·Cv + wpath·Cpath + wprd·Cprd + wprθ·Cprθ + wpair_d·Cpair_d + wpair_θ·Cpair_θ.
  - Ca and Cv penalize an object entering another object's "accessible space" or "viewing frustum" rectangles: Σ max(0, 1 − ‖p_i − a_jk‖/(b_i + ad_jk)).
  - Door-to-door paths are cubic Béziers swept as rectangles.
  - Weights: wa=0.1, wv=0.01, wpath=0.1, wprd=wpair_d∈[1,5], wprθ=wpair_θ=10.
  - Search: Metropolis simulated annealing. Moves are Gaussian translate/rotate (σ∝T), swapping two objects of the same tier, and moving path control points. Two tiers: floor/wall and on-top.
  - Reported speed: 20 objects, 20k iterations, 22 s on a 2011 Xeon.
- *Merrell et al.* (TOG 2011): density p ∝ exp(−β Σ w_i m_i). Terms and weights:
  - clearance overlap of Minkowski-expanded footprints (w 2);
  - circulation = number of connected components of free space for a person disc of 18″ radius (1);
  - pairwise distance and angle with a plateau function t(d,m,M,α) (2/2);
  - conversation: seats 4–8 ft apart and facing each other (2/2);
  - visual balance: area-weighted centroid vs. room centroid (1.5);
  - furniture and wall alignment, −cos4Δθ (2.5/2.5);
  - symmetry about a focal point (1);
  - facing the focal point (4).
  - Table 1 (from Panero & Repetto): bedside 36″; seat front 30″; cabinets/shelving 24″; dining table 36″ all around; coffee table 16–18″ in front of the seat; end table 0–12″; nightstand 0–12″.
  - Search: parallel-tempering MCMC on the GPU, under 1 s. Moves: perturb position, perturb angle, swap.

**Infinigen Indoors** (CVPR 2024; BSD-3-Clause; procedural assets).
- **Constraint language (Python DSL):** `related_to`, `StableAgainst(child, parent, margin)`, `SupportedBy`, `count().in_range`, `min_distance`, `angle_alignment_cost`, `accessibility_cost`, `focus_score`, `freespace_2d`, `center_stable_surface_dist`, `hinge`. Hard constraints are boolean products; soft constraints are weighted minimize/maximize terms.
- **Solver:** Metropolis–Hastings simulated annealing.
  - Moves: add, delete, relation-plane change, resample, reinitialize pose, translate, rotate. The probability of discrete moves decays over the run.
  - Runs in stages: large, then medium, then small objects.
  - Config: `initial_temp=3, final_temp=0.001, finetune_pct=0.15`, with 300/200/50 steps per stage. The paper states τ 0.25→0.001 and about 5k steps.
- **Living-room values in `home.py`:**
  - sofa back margin U(0.1, 0.3) m;
  - sofa to TV stand hinge(2, 3) m;
  - coffee table to sofa hinge(0.45, 0.6) m;
  - TV stand kept away from windows;
  - wall decor more than 0.6 m above the floor and more than 0.1 m from doors/windows (0.25 m preferred);
  - large plants against walls, far from doors.
- **Practicality:** Windows x86_64 support is "experimental" and covers only the minimal install. It needs Python 3.11 and `bpy==4.2.0`. Latest stable is 1.26.0 (PyPI, 2026-11-11). 2.0.0a5 (2026-11-05) is an alpha offering only "simple indoor room arrangements", with no documented solver. Use it as a reference, not a dependency.

**LLM/VLM-based.**
- **Holodeck** (CVPR 2024, Apache-2.0) requires `gpt-4o-2024-05-13`.
  - The LLM emits relations: edge/middle, in front of, side of, around, face to, center aligned, near/far.
  - A depth-first grid solver satisfies them. Parameters: cell = room/20, a 10 cm buffer added to object sizes, yaw ∈ {0, 90, 180, 270}, a 30 s time budget, optional MILP.
  - Relation weights: global 1.0; relative/direction/alignment 0.5; distance 1.8. The "near" score falls to 0 at 80 cm.
  - Its assets come from Objaverse 1.0, which contains about 77k objects under non-commercial (NC) licenses, so per-object license filtering is mandatory.
- **LayoutGPT** (NeurIPS 2023; MIT): uses OpenAI; Llama-2 support is claimed; its 3D part runs on 3D-FRONT/ATISS.
- **I-Design:** no license file; uses the GPT-4 API.
- **LayoutVLM** (CVPR 2025): no license; GPT-4o plus differentiable optimization.
- **SceneCraft** (ICML 2024): GPT-4/GPT-V writes Blender code from a scene graph; I found no code release.
- **SceneWeaver** (NeurIPS 2025; BSD-3): uses GPT via Azure, Linux only, and wraps PhyScene/DiffuScene/ATISS/Infinigen.
- **HSM** (arXiv 2503.16848; MIT): uses OpenAI, about 10 min and ~$0.80 per scene; its HSSD assets are CC-BY-NC-4.0.
- **2026 papers:**
  - SceneSmith: VLM designer, critic and orchestrator agents.
  - SpatialGrammar: a bird's-eye-view grid DSL with compiler feedback, plus a 104M-parameter model (SG-Mini).
  - LayoutDSL: an RL policy trained on "3D-FrontDSL".
  - PolyLayout: functional furniture clusters, VLM macro placement, then rule-based micro-optimization that respects doors and windows.
  - GOPI: single-view RGB-D furniture insertion that ends in diffusion image synthesis, so it is incompatible with the immutable background plate.
  - Code and licensing for all 2026 items: UNVERIFIED.
- **Lesson:** LLMs decide *what* to place and roughly *where*; a geometric solver has to own the coordinates.

**Learned generative models: excluded.**
- ATISS: NVIDIA Source Code License, "non-commercially… research or evaluation purposes only", NVIDIA processors only.
- DiffuScene: Sony license, non-commercial.
- PhyScene: no license in the repo; weights CC-BY-NC-4.0.
- All three train on 3D-FRONT. Its agreement grants use "for scientific research purpose only" and forbids using "derivative works… and the results obtained… for commercialization" (§3.3.3).

**Staging priority.** NAR 2025 Profile of Home Staging: the living room is the most important room to stage (37 %), then the primary bedroom (34 %), then the kitchen (23 %).

## 2. Comparison table

| Approach | Code license | Weights / data | Local, no paid API | Fixed-camera terms | Use |
|---|---|---|---|---|---|
| Make it Home 2011 | paper | – | yes (reimplement) | no | cost terms, SA moves |
| Merrell 2011 | paper | – | yes | no | guidelines + weights |
| Infinigen Indoors | BSD-3 | procedural | Linux-first, Windows experimental | no | DSL, schedules, numbers |
| Holodeck | Apache-2.0 | Objaverse (NC subset) | no (GPT-4o) | no | relation vocabulary |
| LayoutGPT | MIT | 3D-FRONT (research-only) | partial | no | skip |
| I-Design, LayoutVLM, SceneCraft | none / unreleased | Objaverse | no | no | skip |
| SceneWeaver | BSD-3 | mixed, includes NC tools | no | no | self-critique idea |
| HSM | MIT | HSSD CC-BY-NC | no | no | decor-motif idea |
| ATISS / DiffuScene / PhyScene | NC / NC / none | 3D-FRONT, CC-BY-NC | yes | no | legally excluded |
| 2026 agents | UNVERIFIED | mostly LLM / 3D-FRONT | mostly no | no | watch list |

## 3. Recommendation

**Primary:** build our own deterministic solver, about 1.5k lines with numpy, OpenCV and Shapely. Pipeline: room-type template → candidate slots along walls → simulated annealing.
- The energy combines interior-design terms ported from Merrell, Make it Home and Infinigen with staging terms computed by projecting object proxies through the matched camera.
- Why: it is commercially clean, deterministic and testable, and runs in seconds on a CPU. It also optimizes what is actually visible in *this* frame, which no published system does.
- Style: a CIELAB colour palette from the wall/floor masks plus SigLIP 2 zero-shot attributes. Optionally, `Qwen/Qwen3-VL-4B-Instruct` (Apache-2.0) returns a JSON description restricted to fixed enums.
- No LLM inside the coordinate loop.

**Fallbacks:**
1. If annealing finds no feasible layout: greedy slot placement of a reduced program (anchor + rug + one accent piece + wall art); if that also fails, decor only.
2. If no wall is usable (open plan, glazing): floating-group templates.
3. If VRAM or models are unavailable: palette-only style rules.
4. An optional local LLM (Qwen3.5-4B or Gemma-4-E4B, both Apache-2.0) may *rank* 3–5 solver outputs but never emits coordinates.

## 4. Implementation notes

### 4.1 Data structures (floor frame, z-up, metres)
```python
@dataclass
class Opening: kind: str; wall: int; s0: float; s1: float; z0: float; z1: float  # window|door|fireplace|radiator
@dataclass
class Wall: p0: np.ndarray; p1: np.ndarray; n_in: np.ndarray; vis: tuple[float, float]; openings: list[Opening]
@dataclass
class Room:
    free: Polygon; unknown: Polygon; walls: list[Wall]; ceil_h: float
    keepout: MultiPolygon              # door swing (r = leaf width) + 0.9 m approach; fixture zones
    grid: np.ndarray; edt: np.ndarray  # 5 cm cells FREE/UNKNOWN/WALL/KEEPOUT/FIXTURE; dist. to wall
    K: np.ndarray; R: np.ndarray; t: np.ndarray
    masks: dict[str, np.ndarray]; depth: np.ndarray   # 1/8-res window/door/floor masks, proxy depth
@dataclass
class Spec:
    role: str; w: float; d: float; h: float; mount: str   # floor|wall|on_top|ceiling
    hull: np.ndarray                                       # <=64 local hull pts (2-4 parts if legged)
    clear: dict[str, float]                                # front/back/left/right zone depth
    wall_pref: float; min_vis: float; weight: float; tier: int; variants: list[str]
@dataclass
class Pose: spec: Spec; x: float; y: float; yaw: float; z: float = 0.0; wall: int | None = None; parent: int | None = None
```

### 4.2 Design rules (cm)
| Rule | Value (source) |
|---|---|
| Main walkway, door-to-door path | ≥ 91 (NKBA 36″; Merrell 18″-radius disc); if two walkways are perpendicular, one ≥ 107 (NKBA access standard) |
| Passage, hard minimum | 60 (secondary sources) |
| Clearance in front of a seat | 76 (Merrell 30″) |
| Sofa front to coffee table | 41–46 target (Merrell 16–18″); acceptable band 35–60 (Infinigen) |
| End table / nightstand gap | 0–30, 2–10 preferred (Merrell 0–12″) |
| Seats in one group | 122–244 apart, facing each other |
| Bedside, foot of bed | 91 ideal (Merrell 36″), 60 minimum |
| Shelving / dresser front | 61 (Merrell 24″) / 91–107 (secondary) |
| Dining table to wall | 91 (Merrell). NKBA: 81 with no traffic behind, 91 to edge past, 112 to walk past; 61 of table width per diner |
| Kitchen work aisle (keep empty) | 107 for one cook, 122 for several (NKBA) |
| Counter stools | 61 wide each; knee depth 46/38/30 at 76/91/107 surface height (NKBA) |
| Bathroom fixtures | 76 clear in front (code minimum 53); toilet centreline ≥ 46 from obstacles (code minimum 38) |
| TV (off by default) | distance 1.2–1.6 × diagonal (THX 40°, SMPTE 30°); viewer looks up ≤ 15° |
| Rug, living room | front legs of all seats on the rug; 20–60 of bare floor to walls, 30–45 typical |
| Rug, bedroom / dining | 46–61 beyond bed sides and foot / ≥ 61 beyond table edges |
| Wall art | centre at 145–152; 15–25 above sofa back, 15–30 above headboard, 10–20 above a console; width ≈ ⅔ of the furniture below; 5–8 between frames |
| Wall decor vs. openings | ≥ 10 from window/door trim (25 preferred); bottom ≥ 60 above floor (Infinigen) |
| Pendant light | lowest point 76–91 above the table, +7.5 per extra 30 cm of ceiling above 244; diameter ½–⅔ of table width (retailers) |
| Desk | perpendicular to the window (glare); ~90 behind the desk for the chair (secondary) |

### 4.3 Templates (place the anchor first, then children relative to their parent)
- **Sala (living room):** sofa (against a wall or floating; front within 70° of the camera direction, heuristic) → coffee table (centred, 41–46 gap, about ⅔ of sofa length, UNVERIFIED) → rug → 1–2 armchairs (facing the table, at 90° or 30–45°) → side table → floor lamp at a sofa end or behind an armchair → art above the sofa → large plant in a corner, ≥ 25 from window trim → console on another wall.
- **Quarto (bedroom):** bed (headboard to a wall, outside any door swing) → two nightstands (top ≈ mattress height ±10) with lamps → rug → art above the headboard → optional bench, dresser, corner chair or plant.
- **Jantar / cozinha com mesa (dining / eat-in kitchen):** table → evenly spaced chairs → rug → sideboard with art → centrepiece. Add a pendant only if a ceiling anchor point is visible.
- **Cozinha (kitchen):** keep the 107 work aisle clear; stools along island/peninsula edges; decor on detected horizontal surfaces at 86–107 cm height; optional runner rug.
- **Banheiro (bathroom):** decor only, outside the NKBA clear zones: bath mat, towel ladder, stool or plant in corners, vanity tray, art above the toilet.
- **Escritório (office):** desk → chair → task lamp and monitor → bookshelf → rug under the chair → plant, art; optional reading chair.

### 4.4 Energy (lower is better; each term normalized to about [0,1])
**Hard constraints (reject the layout if any fails):**
- footprint lies inside FREE∪UNKNOWN floor, with at most 25 % on UNKNOWN;
- no footprint overlap (rug and on-top items excepted, 5 mm tolerance);
- no intrusion into keepout zones;
- all doors and the camera-side entry are in one component walkable at 60 cm;
- doors are covered by at most 5 % in the image;
- at least 98 % of the anchor's silhouette is inside the frame;
- at most 5 % of any silhouette lies behind the room's architecture;
- wall items are ≥ 10 cm from openings;
- height is below ceiling − 5 cm.

**Soft terms:**

| Soft term | Definition | w |
|---|---|---|
| clearance | overlap of clearance zones with other footprints/walls ÷ zone area (Merrell m_cv) | 3 |
| path width | share of the main path narrower than 91 cm | 3 |
| pairwise | Merrell's t(d,m,M,2) on distance and angle for each template pair | 2+2 |
| conversation | seats 122–244 apart; (cos φfg+1)(cos φgf+1) | 1.5 |
| alignment | −cos4Δθ to the wall; back gap 2–10 cm | 2 |
| symmetry | paired items mirrored about the anchor axis | 1 |
| emphasis | seating faces the focal point (fireplace/TV wall, else the anchor) | 2 |
| visibility | Σ weight·max(0, min_vis − vis) | 6 |
| truncation | Σ weight·(1 − fraction inside frame) | 5 |
| windows/features | Σ max(0, cover − 0.10) per window; fireplace/built-ins ×0.5 | 6 |
| floor visible | distance of the visible-floor ratio from the band [0.40, 0.70]; rug counts 0.3 (heuristic) | 4 |
| balance | max(0, \|x̄ − 0.5\| − 0.08), with x̄ weighted by area × visibility | 3 |
| foreground | occupancy of the bottom 14 % of the frame | 4 |
| camera facing | penalty if front·(C−p)/‖C−p‖ < −0.2 for sofa/bed/armchair/dresser | 3 |
| prominence | anchor silhouette outside 8–25 % of the frame (heuristic) | 2 |
| completeness | −importance per optional item placed (capped) | 1 |

### 4.5 Search
1. **Candidates.**
   - Slide each anchor along every visible wall in 10 cm steps, with its back or its side to the wall.
   - Add floating poses on a 25 cm grid, with axis-aligned yaws that face the camera.
   - Build the template group around each pose and give it a quick score.
   - Apply non-maximum suppression (35 cm) and keep K = 6.
2. **Annealing per candidate.**
   - Initial temperature: T0 = −mean(ΔE⁺)/ln 0.5, estimated from 200 random moves.
   - Geometric cooling to 1e-3·T0, with the last 15 % of steps held at the final temperature; 2,000 steps in total.
   - Moves: rigid group translate (σ 0.30→0.02 m), slide along the wall, resample a child inside its template band, yaw snaps (armchairs ±15–45°), jump to another wall, swap two same-tier items, change size variant, add/remove an optional item.
   - The share of discrete moves falls from 40 % to 10 %.
3. **Secondary pass.** Place lamps, plants, art and decor at their best slot, then run 500 annealing steps.
4. **Verification.**
   - Render an object-index / Cryptomatte pass in Blender at ½ resolution for the two best layouts, with the room proxy as holdout.
   - Recompute visibility, truncation and window cover exactly from that pass.
   - Seed = hash(photo).
   - Optional: Merrell-style parallel tempering (4 chains at T×{1,2,4,8}, swapping every 50 steps).

### 4.6 Visibility by camera projection
```python
def silhouette(pose, cam, s=8, PAD=64):
    Pc = pose.world(pose.spec.hull) @ cam.R.T + cam.t      # camera coords; reject pose if any z < 0.1
    uv = (Pc @ cam.K.T)[:, :2] / Pc[:, 2:3] / s
    hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
    m = np.zeros((cam.h // s + 2 * PAD, cam.w // s + 2 * PAD), np.uint8)   # padding measures truncation
    cv2.fillConvexPoly(m, np.round((hull + PAD) * 16).astype(np.int32), 1, cv2.LINE_8, 4)
    inside = m[PAD:-PAD, PAD:-PAD].astype(bool)
    return inside, Pc[:, 2].min(), inside.sum() / max(m.sum(), 1)
```
- Sort objects by minimum depth (nearest first).
- Visibility: vis_i = |m_i \ ∪(masks of closer objects)| / |m_i|.
- Window cover: |∪m ∩ window| / |window|.
- Behind architecture: pixels where the hull depth exceeds the room-proxy depth by more than 5 cm.

**Benchmark** (run here: Python + OpenCV, 1 core, 15 objects):

| Resolution | Incremental step (one object moved) | Full re-raster |
|---|---|---|
| 240×135 | 0.63 ms | 2.1 ms |
| 480×270 | 1.9 ms | 5.2 ms |

At 240×135, 2k annealing steps take about 1.3 s.

### 4.7 Style and palette (one curated collection, 3 palette variants)
- **Palette rules (thresholds are heuristic):**
  - k-means (k = 4) in CIELAB over eroded wall and floor masks, dropping L* > 97.
  - Floor tone from L*, warmth from b*.
  - Furniture wood contrasts with the floor (ΔL* ≥ 15); textiles stay near the wall hue at low chroma; one accent colour.
- **Zero-shot attributes:** `google/siglip2-so400m-patch14-384` (Apache-2.0, 1.14B params, about 2.3 GB in fp16). In transformers 5.x the `pipeline` argument is `dtype`:
  ```python
  clf = pipeline("zero-shot-image-classification", model="google/siglip2-so400m-patch14-384", device=0, dtype=torch.float16)
  clf(img, candidate_labels=["light oak wood floor", "dark hardwood floor", "white porcelain tile floor", "grey concrete floor", "beige carpet"])
  ```
  - Alternatives: `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` with `open_clip_torch` 4.2.2 (MIT code and weights); PE-Core and Google TIPS v1 (Apache-2.0).
- **Optional VLM returning JSON:**
  - Primary pick: `Qwen/Qwen3-VL-4B-Instruct-GGUF` (Q4_K_M 2.5 GB + mmproj Q8 0.45 GB), run with `llama-server -m … --mmproj … --json-schema-file attrs.json`.
  - Alternatives:
    - `ggml-org/gemma-4-E4B-it-GGUF`: Q4_0 4.59 GB + 0.56 GB mmproj; the Gemma 4 license page contains the Apache-2.0 text.
    - `Qwen/Qwen3.5-4B`: Apache-2.0, Feb 2026, 4.66B; thinking mode is on by default, so disable it.
    - Qwen3-VL-2B/8B and SmolVLM2-2.2B: Apache-2.0.
  - Small fallbacks: `vikhyatk/moondream2` (Apache-2.0, 1.93B), `microsoft/Florence-2-large` (MIT, 0.78B).
  - Free the VRAM before Blender starts.
- **Flagged licenses:**
  - `Qwen/Qwen2.5-VL-3B-Instruct`: Qwen Research License, non-commercial.
  - MetaCLIP 2: CC-BY-NC-4.0.
  - `apple/DFN5B-CLIP-*`: Apple ML Research license.
  - Moondream 3/3.1: custom license; commercial use is allowed, but not as a general-purpose hosted service.
  - `Qwen3.8-Flash-Next`: qwen-community-1.0; attribution required above 100M monthly active users or US$20M monthly revenue, and a separate license for model-as-a-service businesses.

### 4.8 Cross-check with the current `mobilia/layout` code
- `SearchBudget(group_steps=70, polish_steps=140)` is one to two orders of magnitude below the literature (Make it Home 20k, Infinigen about 5k). The measured cost allows at least 2k steps per candidate; also measure how much the score varies between seeds.
- `WALKWAY=0.60` is a sound hard minimum. Add the 91 cm main-path soft term and a hard check that all doors are connected.
- `RUG_SIZES` has 3 sizes; it should mirror the real catalogue (EU 160×230, 200×300, 250×350; Brazilian retail sizes UNVERIFIED).
- `COFFEE_GAP=0.45`, `ART_CENTER_Z=1.50`, `ART_ABOVE_GAP=0.20` and `MAX_WINDOW_COVER=0.10` already agree with the sources.

## 5. Risks and limitations
- **Single view.** Walls and doors behind the camera are unknown, and UNKNOWN floor near the camera may hide a real walking path.
- **Scale error.** Monocular metric error mis-sizes furniture against the clearances. Sanity-check scale against door height (~2.0–2.1 m) and ceiling height.
- **Segmentation misses.** Missed windows, glass doors or radiators break the keepout zones. Keep a debug overlay that shows per-term values.
- **Unvalidated thresholds.** The image-space thresholds (floor band, prominence, 14 % foreground band, 10 % window cover) are heuristics, not published standards. Tune them with pairwise preferences on about 30 labelled photos.
- **Source quality.** The rug, art, lighting and office values come from retail and design sites. NKBA is US-centric, and I did not check Brazilian norms such as NBR 9050.
- **Silhouette approximation.** Convex-hull silhouettes misjudge see-through furniture such as chairs and tables with legs, so the Blender verification pass is required.
- **LLM/VLM output** is non-deterministic and can hallucinate; keep it out of the geometry.
- **Benchmark scripts** (scratch files): /tmp/claude-0/-home-user/c09d70e1-5a72-5f0b-811a-6fd083338d9a/scratchpad/bench_vis.py and /tmp/claude-0/-home-user/c09d70e1-5a72-5f0b-811a-6fd083338d9a/scratchpad/bench_vis8.py

## 6. Sources

**Layout methods**
- Make it Home: https://ww3.math.ucla.edu/camreport/cam11-49.pdf
- Merrell 2011: https://paulmerrell.org/furnitureLayout2.pdf
- Infinigen Indoors:
  - https://arxiv.org/abs/2406.11824
  - https://github.com/princeton-vl/infinigen/blob/main/LICENSE
  - https://github.com/princeton-vl/infinigen/blob/indoors-stable/infinigen_examples/constraints/home.py
  - https://github.com/princeton-vl/infinigen/blob/indoors-stable/infinigen_examples/configs_indoor/base_indoors.gin
  - https://github.com/princeton-vl/infinigen/blob/indoors-stable/docs/Installation.md
  - https://pypi.org/project/infinigen/
  - https://github.com/princeton-vl/infinigen/releases
  - https://infinigen.cs.princeton.edu/docs/latest/index.html
- Holodeck: https://github.com/allenai/Holodeck (README, LICENSE, ai2holodeck/generation/floor_objects.py)
- Objaverse licenses: https://huggingface.co/datasets/allenai/objaverse
- LayoutGPT: https://github.com/weixi-feng/LayoutGPT
- I-Design: https://github.com/atcelen/IDesign
- LayoutVLM: https://github.com/sunfanyunn/LayoutVLM
- SceneCraft: https://arxiv.org/abs/2403.01248
- SceneWeaver: https://github.com/Scene-Weaver/SceneWeaver
- HSM: https://github.com/3dlg-hcvc/hsm; HSSD: https://huggingface.co/datasets/hssd/hssd-hab
- 2026 papers: https://arxiv.org/abs/2602.09153, https://arxiv.org/abs/2604.27555, https://arxiv.org/abs/2608.07547, https://arxiv.org/abs/2608.10838, https://arxiv.org/abs/2608.06836
- ATISS: https://github.com/nv-tlabs/ATISS/blob/master/LICENSE
- DiffuScene: https://github.com/tangjiapeng/DiffuScene/blob/master/LICENSE
- PhyScene: https://github.com/PhyScene/PhyScene, https://huggingface.co/datasets/yangyandan/PhyScene
- 3D-FRONT agreement: https://gw.alicdn.com/bao/uploaded/TB1ZJUfK.z1gK0jSZLeXXb9kVXa.pdf

**Design rules**
- NKBA kitchen: https://kb.nkba.org/uploads/2022/05/Kitchen-Planning-Guidelines.pdf
- NKBA bath: https://kb.nkba.org/uploads/2022/05/Bath-Planning-Guidelines.pdf
- TV viewing distance: https://en.wikipedia.org/wiki/Optimum_HDTV_viewing_distance
- Rugs: https://www.thisoldhouse.com/house-one/rug-sizing-guide, https://rugs.com/blog/what-size-rug-do-i-need-the-complete-rug-size-guide-for-every-room/
- Art: https://blog.fineartamerica.com/how-high-to-hang-art-above-a-sofa/
- Lighting (search snippets only): https://ballarddesigns.com/trends-features/features/guide/chandelier-buying-guide, https://goldenlighting.com/blogs/the-golden-edit/pendant-lights-over-dining-room-table-a-guide
- Desk glare (snippet only): https://www.officefurniture2go.com/answers/desk-space-planning/desk-placement-minimize-window-glare.html
- NAR 2025: https://www.nar.realtor/sites/default/files/2025-05/2025-profile-of-home-staging-05-06-2025.pdf

**Models**
- https://huggingface.co/google/siglip2-so400m-patch14-384
- https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
- https://github.com/mlfoundations/open_clip
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF
- https://huggingface.co/Qwen/Qwen3.5-4B
- https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE
- https://huggingface.co/Qwen/Qwen3.8-Flash-Next
- https://ai.google.dev/gemma/docs/gemma_4_license
- https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF
- https://huggingface.co/microsoft/Florence-2-large
- https://huggingface.co/vikhyatk/moondream2
- https://moondream.ai/licenses/model/1.0
- https://huggingface.co/facebook/metaclip-2-worldwide-huge-quickgelu
- https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14-378
- llama.cpp: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md, https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md