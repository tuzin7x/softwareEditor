# Room segmentation, layout planes and lighting from one indoor photo (research as of 2026-11-15)

## Findings

### (a) Segmentation

- **ADE20K-150 covers almost every class we need.** IDs below come from the official HF label file.
  - Structure: wall 0, floor 3, ceiling 5, windowpane 8, door 14, curtain 18, blind 63, screen door 58, stairs 53, stairway 59, step 121, railing 38, bannister 95, column 42, fireplace 49, radiator 146.
  - Fixtures: cabinet 10, counter 45, countertop 70, kitchen island 73, sink 47, stove 71, oven 118, microwave 124, dishwasher 129, hood 133, refrigerator 50, washer 107, toilet 65, bathtub 37, shower 145, mirror 27, rug 28, light 82, chandelier 85, sconce 134, lamp 36.
  - Missing: baseboards, outlets/switches, AC units, window frames/sills, beams, vents. These need an open-vocabulary model.
- **EoMT** (CVPR 2025 Highlight) is a plain ViT with no pixel decoder or transformer decoder.
  - EoMT-L/DINOv2 at 512 px: **58.4 mIoU**, 92 FPS on an H100. HF repo `tue-mps/ade20k_semantic_eomt_large_512`, card says MIT, 0.3B params.
  - In transformers since **v4.54** as `EomtForUniversalSegmentation`. Semantic inference uses sliding windows.
  - The ADE20K panoptic model `tue-mps/ade20k_panoptic_eomt_large_640` (PQ 50.6, MIT) separates individual windows and doors.
  - The DINOv3 variant `tue-mps/ade_semantic_eomt_large_512_dinov3` reaches **59.5 mIoU**. The HF card says MIT, but the GitHub model zoo says the weights are *"deltas with respect to DINOv3 weights"* and need DINOv3 access. **The two conflict, so treat it as DINOv3-licensed.**
  - Newer work from the same group: VidEoMT (CVPR 2026) and PMT (arXiv 2603.25398).
- **OneFormer** (code MIT, HF cards MIT).
  - Swin-L at 640 px: 57.0 mIoU single-scale, 57.7 multi-scale.
  - DiNAT-L: 58.3, but it needs NATTEN, which has no official Windows wheels.
  - No updates since 2023.
- **Mask2Former**: Swin-L scores 56.1 / 57.3 and the code is MIT. But MODEL_ZOO says *"All models … are licensed under the Creative Commons Attribution-NonCommercial 4.0"*. **The weights are non-commercial.** The repo was archived in Jan 2025, and HF `facebook/mask2former-*` is tagged "other".
- **SegFormer**: the NVIDIA Source Code License says the work *"may be used … non-commercially"*, meaning research or evaluation only. **Non-commercial.** HF `nvidia/segformer-*` links to this license.
- **UperNet + ConvNeXt-L**: 53.2 / 53.7 mIoU, 235M params, MIT (`openmmlab/upernet-convnext-large`).
- **DINOv3** (Aug 2025, transformers ≥4.56): a frozen ViT-7B with an M2F/ViT-Adapter head reaches **62.6 mIoU (63.0 with TTA)** at 896 px, the current state of the art. At 6.7B params plus a 927M decoder it is impractical on 8–24 GB cards.
- **SAM 3** (Nov 2025; SAM 3.1 checkpoints released 2026-03-27) segments *concepts* from text or image exemplars. It has 848M params.
  - HF `facebook/sam3` is gated. In transformers 5.x it is `Sam3Model`/`Sam3Processor`, with a native 1008 px input; the image features can be computed once and reused across text prompts.
  - Zero-shot **ADE-150 is only 39.0 mIoU** (paper, Table 33). Use it for instances and rare objects, not as the main dense parser.
  - The official repo needs Python 3.12+, PyTorch 2.7+ and CUDA 12.6+.
- **SAM License and DINOv3 License** are custom Meta agreements.
  - Each grants a non-exclusive, worldwide, royalty-free right to use, reproduce, distribute and modify. Neither has a commercial carve-out.
  - Restrictions: trade controls, ITAR and military uses; no reverse engineering; a copy of the agreement must ship with any redistribution.
  - DINOv3 §8 lets Meta change the terms.
  - Commercial use appears to be permitted, but **legal review is needed**.
- **Other open-vocabulary and refinement models.**
  - **SAM 2.1**: code and weights Apache-2.0. `facebook/sam2.1-hiera-large` (224M), `Sam2Model` with box prompts. This is the license-safest way to refine masks.
  - **Grounded-SAM-2**: Apache-2.0. Only the local Grounding DINO path is usable; Grounding DINO 1.5/1.6 and DINO-X are cloud APIs, so they are excluded.
  - **Florence-2-large**: MIT, 0.77B, needs `trust_remote_code`, produces only coarse polygons.

### (b) Layout

- **Learned single-view layout models: none from 2025–26 is commercially usable, and none is clearly better.**
  - NonCuboidRoom (WACV 2022): MIT code. Weights trained on Structured3D and SUN RGB-D/NYU303; dataset terms UNVERIFIED.
  - uLayout (WACV 2025): MIT, but it requires PerspectiveFields (**Adobe Research License, noncommercial**) and its weights are trained on MP3D.
  - ZeroPlane (CVPR 2025 Highlight): MIT code, but the released checkpoint uses a DUSt3R encoder (**CC BY-NC-SA 4.0**).
  - SpatialLM (NeurIPS 2025): outputs walls, doors and windows, but from *point clouds*. Weights **CC-BY-NC-4.0**.
  - Plane-DUSt3R: multi-view input, and DUSt3R is non-commercial.
- **Depth + segmentation + robust fitting is license-clean and gives metric output.**
  - MoGe-2 `Ruicheng/moge-2-vitl-normal` (331M, MIT) returns `points`, `depth`, `normal`, `mask` and normalized `intrinsics`, all in the OpenCV camera frame.
  - MoGe-3 (released 2026-08-18) needs FlexGEMM and Triton, which is a Windows risk.
  - A SIGGRAPH 2025 poster that rebuilds Karsch-style object insertion uses MoGe point maps as its scene proxy, which supports this approach.
- **Calibration and vanishing points (VP).**
  - GeoCalib (ECCV 2024): code Apache-2.0, weights **CC-BY-4.0**. It gives gravity (roll/pitch) and FoV, with the principal point fixed at the image center.
  - OpenCV **LSD** was removed in 4.1.0–4.5.3 because of a license conflict. It came back after the NFA code was re-published under MIT, so use OpenCV ≥4.5.4. Defaults: `createLineSegmentDetector(refine=LSD_REFINE_STD, scale=0.8, sigma_scale=0.6, quant=2.0, ang_th=22.5, …)`.
  - ELSED (line detector): Apache-2.0. DeepLSD (line detector): MIT, weights MIT.
  - NeurVPS (VP estimator): MIT.
  - **GlobustVP** (CVPR 2025 Oral): Apache-2.0, uses cvxpy with SCS, about 50 ms per image, needs K.
  - Progressive-X (multi-model fitting): MIT. There is no maintained J-Linkage library.

### (c) Lighting

- **DiffusionLight** (CVPR 2024): code and LoRA are MIT.
  - Dependencies and their licenses:
    - `stabilityai/stable-diffusion-xl-base-1.0`: OpenRAIL++-M, which allows commercial use with use-based restrictions.
    - `diffusers/controlnet-depth-sdxl-1.0`: openrail++.
    - `Intel/dpt-hybrid-midas`: Apache-2.0.
    - `madebyollin/sdxl-vae-fp16-fix`: MIT.
  - Defaults: 1024², 30 steps, `--ev "0,-2.5,-5"`, iterative mode with 2×30 balls and a median, `lora_scale 0.75`. This takes about 30 minutes per image.
  - **DiffusionLight-Turbo** (arXiv 2507.01305, v2 Jan 2026; `DiffusionLight/TurboLoRA`, MIT; ComfyUI nodes) runs a single denoising pass by swapping LoRAs. About **30 s per image** with "minimal quality loss".
  - VRAM is not stated anywhere. My estimate is 10–12 GB in fp16, with CPU offload needed on 8 GB cards (UNVERIFIED).
  - The photo is only padded and inpainted to produce a light probe. No output pixel comes from diffusion, so this is compatible with the absolute rule.
- **Excluded lighting estimators.**
  - LuxDiT (NeurIPS 2025): 5B, **NVIDIA OneWay Noncommercial**.
  - StyleLight: MIT code, but the weights are trained on Laval Indoor HDR, whose terms I could not fetch (UNVERIFIED).
  - EverLight (ICCV 2023): no code released.
  - Gardner 2017/2019 and DiPIR: no official code found (UNVERIFIED).
- **Karsch 2011.**
  - Area lights come from user-marked polygons.
  - Light parameters are chosen to *"minimize the squared pixel-wise differences between the rendered image … and the target image"*.
  - Albedo comes from an intrinsic decomposition projected onto the proxy.
  - Sun shafts are modelled as masked spotlights at infinite distance.
  - Composite: `I_final = M⊙I_obj + (1−M)⊙(I_b + I_obj − I_noobj)`.
- **The 2025 SIGGRAPH poster** (Peng, Taira, Careaga, Aksoy) does the same thing with modern parts.
  - It uses a MoGe mesh, Careaga–Aksoy albedo, and Mitsuba 3 to optimize an environment map plus point lights.
  - Composite: `C = M·O + (1−M)(B + α(O−N))`.
  - Its repository states no license, and the intrinsic model it relies on is "academic use only… patent pending". **Reimplement the idea, not the code.**
- **Blender Cycles** (manual 5.3).
  - Shadow catchers *"interact with other CG objects via indirect light"*. With the Shadow Catcher pass enabled, *"all indirect light interactions are captured"*.
  - Compositing: multiply the background by the Shadow Catcher pass, then alpha-over the Combined pass (3.0 release notes).
  - **Light Groups** give one Combined pass per light, so lights can be re-weighted without re-rendering.
  - Area lights can act as **Portals** in windows.
  - Python names, all checked against the API docs: `obj.is_shadow_catcher`, `obj.is_holdout`, `obj.lightgroup`, `world.lightgroup`, `view_layer.lightgroups.add(name=)`, `light.cycles.is_portal`, `view_layer.cycles.use_pass_shadow_catcher`.
- **Albedo estimation.**
  - Marigold-IID-Lighting v1-1 (`prs-eth/marigold-iid-lighting-v1-1`, OpenRAIL++-M, `MarigoldIntrinsicsPipeline`, about 768 px) returns linear albedo, shading and a residual.
  - compphoto/Intrinsic is **academic-only**.

## Comparison table

| Component | Model / id | Quality | License code / weights | Size / VRAM* | Verdict |
|---|---|---|---|---|---|
| Dense parse | EoMT-L DINOv2 `tue-mps/ade20k_semantic_eomt_large_512` | 58.4 mIoU | MIT / MIT (DINOv2 Apache-2.0) | 0.3B / ~2 GB | **Primary** |
| Instances | EoMT-L panoptic `tue-mps/ade20k_panoptic_eomt_large_640` | PQ 50.6 | MIT / MIT | 0.3B / ~2 GB | **Primary** (openings) |
| Dense parse | EoMT-L DINOv3 `tue-mps/ade_semantic_eomt_large_512_dinov3` | 59.5 | MIT / DINOv3 deltas ⚠ | 0.3B | Upgrade after legal approval |
| Dense parse | OneFormer Swin-L `shi-labs/oneformer_ade20k_swin_large` | 57.0 (57.7) | MIT / MIT | ~0.2B (UNVERIFIED) / ~3 GB | **Fallback** |
| Dense parse | OneFormer DiNAT-L | 58.3 | MIT / MIT, NATTEN ⚠ on Windows | — | Avoid |
| Dense parse | Mask2Former Swin-L `facebook/mask2former-swin-large-ade-semantic` | 56.1 (57.3) | MIT / **CC BY-NC** ✗ | 0.2B | Excluded |
| Dense parse | SegFormer-B5 `nvidia/segformer-b5-finetuned-ade-640-640` | ~51 (paper, UNVERIFIED) | **NVIDIA NC** ✗ | ~85M | Excluded |
| Dense parse | UperNet ConvNeXt-L `openmmlab/upernet-convnext-large` | 53.2 (53.7) | MIT | 235M | Backup |
| Dense parse | DINOv3-7B + M2F `dinov3_vit7b16_ms` | 62.6 (63.0) | DINOv3 License ⚠ | 7.6B | Too heavy |
| Open-vocab | SAM 3 `facebook/sam3` | ADE-150 39.0 (zero-shot) | SAM License ⚠ | 0.85B / ~4–6 GB | **Primary** for rare objects |
| Refinement | SAM 2.1 `facebook/sam2.1-hiera-large` | — | Apache-2.0 | 224M / ~2 GB | **Primary** refinement |
| Open-vocab | Grounding DINO + SAM 2 | — | Apache-2.0 | — | Fallback |
| Geometry | MoGe-2 `Ruicheng/moge-2-vitl-normal` | metric + normals | MIT | 331M / ~2 GB | **Primary** input |
| Gravity/FoV | GeoCalib | roll AUC@5° 91.8 (Stanford2D3D) | Apache-2.0 / CC-BY-4.0 | small | Cross-check |
| Gravity/FoV | PerspectiveFields | — | **Adobe NC** ✗ | — | Excluded |
| VP | OpenCV LSD + GlobustVP / Progressive-X | ~50 ms | Apache-2.0 / MIT | CPU | Refinement |
| Layout nets | NonCuboidRoom, uLayout, ZeroPlane, SpatialLM | — | NC dependencies or weights ✗⚠ | — | Not used |
| Light probe | DiffusionLight-Turbo + SDXL | ~30 s | MIT / MIT + OpenRAIL++-M | ~10–12 GB | Optional |
| Light probe | LuxDiT | — | **NVIDIA NC** ✗ | 5B | Excluded |
| Albedo | Marigold-IID-Lighting v1-1 | — | OpenRAIL++-M | ~1.3B / ~4 GB | Optional |

*VRAM figures are fp16 estimates from parameter counts (UNVERIFIED). Measure them on the target GPUs.

## Recommendation

### (a) Segmentation
- **Primary:** EoMT-L DINOv2, semantic model for the dense label map plus panoptic model for separate window and door instances.
  - Why: MIT, built into transformers, the fastest of the accurate models, 58.4 mIoU.
  - Add **SAM 3** text prompts for openings and rare fixtures ("window", "glass door", "radiator", "baseboard", "electrical outlet", "air conditioner", "built-in wardrobe"), *if* legal accepts the SAM License.
  - Refine mask boundaries with **SAM 2.1** box prompts (Apache-2.0).
- **Fallback:** OneFormer Swin-L (MIT, 1.4 mIoU lower). If SAM 3 is rejected, use Grounding DINO + SAM 2.1 instead.

### (b) Layout
- **Primary:** MoGe-2 points and normals plus EoMT labels, fed into constrained RANSAC and Manhattan fitting (algorithm below).
  - Cross-check gravity against GeoCalib, and refine the rotation with LSD lines and GlobustVP.
  - Why: license-clean, metric, easy to debug, and it never invents structure it cannot see.
- **Fallback:** for walls that are not at right angles, fit lines in the top view with Progressive-X. Use height priors when the floor or ceiling is barely visible.

### (c) Lighting
- **Primary:** Karsch-style inverse lighting on the room proxy.
  - Lights: window area lights (or portals plus a world environment), a sun light when sun patches are detected, and fixtures that are switched on.
  - Intensities and colours: solved by non-negative least squares (NNLS) on Cycles Light Group passes rendered on the proxy.
  - Composite: Shadow Catcher pass, with every pixel outside the effect mask copied from the photo.
- **Optional:** a DiffusionLight-Turbo HDR probe for reflections and ambient light from outside the frame.
- **Fallback:** fixed heuristic ratios (window 1.0, environment 0.15) plus an exposure match on the wall median.

## Implementation notes

### Segmentation (transformers 5.x)
```python
from transformers import AutoImageProcessor, EomtForUniversalSegmentation, Sam3Model, Sam3Processor
mid = "tue-mps/ade20k_semantic_eomt_large_512"
p = AutoImageProcessor.from_pretrained(mid); m = EomtForUniversalSegmentation.from_pretrained(mid, dtype=torch.float16).cuda()
x = p(images=img, return_tensors="pt").to("cuda", torch.float16)
with torch.inference_mode(): o = m(**x)
r = p.post_process_semantic_segmentation(o, target_sizes=[(H, W)], return_segmentation_scores=True)[0]  # .segmentation/.segmentation_scores
s3 = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.float16).cuda(); sp = Sam3Processor.from_pretrained("facebook/sam3")
ii = sp(images=img, return_tensors="pt").to("cuda"); ve = s3.get_vision_features(pixel_values=ii.pixel_values.half())
for t in ["window", "door", "radiator", "baseboard", "electrical outlet"]:
    o3 = s3(vision_embeds=ve, **sp(text=t, return_tensors="pt").to("cuda"))
    inst = sp.post_process_instance_segmentation(o3, threshold=0.5, mask_threshold=0.5, target_sizes=ii["original_sizes"].tolist())[0]
```
- Fuse per pixel: EoMT scores for walls, floor and other "stuff"; SAM masks override for openings and fixtures.
- Sharpen label boundaries with a guided or joint-bilateral filter on the photo.

### Room-proxy algorithm
Inputs: MoGe-2 point map `P` (H×W×3, metres), normals `N`, intrinsics `K`, valid mask; EoMT labels, scores and instances; optional GeoCalib gravity.

0. **Clean.** Drop pixels with |∇z|/z > 0.05 (depth edges), erode class masks by 3 px, and subsample to about 200k points. Noise threshold τ(z) = max(1 cm, 1%·z).
1. **Floor.**
   - RANSAC on {floor, rug} pixels. A point is an inlier when |n·x+d| < τ and ∠(N,n) < 20°. 1000 iterations, then SVD refinement.
   - Orient the plane so d > 0; then d is the camera height h (expect 0.9–1.9 m).
   - If GeoCalib agrees within 3°, refit the offset only with the normal fixed.
   - Open3D `segment_plane` / `detect_planar_patches` (MIT) are drop-in alternatives.
2. **Frame.** z = floor normal. Project wall normals onto the floor and take the dominant yaw θ₀ = arg(Σw·e^{4iθ})/4, a circular mean with 4-fold symmetry. This gives the Manhattan x and y axes.
3. **Rotation refinement.**
   - Predict vanishing points vᵢ = K R eᵢ.
   - Keep LSD segments longer than 2% of the image diagonal, and assign a segment to a VP if its angular error is under 2°.
   - Minimize line-to-VP residuals with `scipy.optimize.least_squares`. Use GlobustVP when normals are unreliable (textureless walls).
   - *Shift-lens check:* with zero pitch the horizontal VPs lie on row y = c_y. If verticals are parallel but the horizon is off-centre (a keystone-corrected listing photo), set c_y and Blender `shift_y` from the horizon row instead of the image centre.
4. **Walls.**
   - Use wall-label pixels only; depth on glass, mirrors, doors and curtains is unreliable.
   - For each axis a ∈ {x, y}: take points with |N·a| > cos 20°, compute s = x·a, and find histogram peaks (5 cm bins, at least 3% support). Each peak is a vertical wall plane, kept only if its inlier pixels form connected regions in the image.
   - If residuals stay high (walls at odd angles), run sequential 2D line RANSAC or Progressive-X in the top view.
5. **Snap to segmentation.** Project each floor∩wall line into the image. Search the wall offset over ±15 cm to minimize the distance to the floor/wall label boundary; label edges are sharper than monocular depth.
6. **Ceiling.** Fit the offset with the normal fixed to "up". If the ceiling is not visible, use max(2.5 m, highest opening top + 0.3 m).
7. **Extents.**
   - Each wall's horizontal range is the 1st–99th percentile of its inliers.
   - Intersect consecutive walls in the top view to get corners, and clip to the camera frustum.
   - Build the floor polygon with shapely (BSD).
8. **Openings.**
   - Each window or door instance belongs to the wall that holds the majority of labels in a 10 px ring around it.
   - Back-project the mask by **ray–plane intersection**, not by depth, because glass and doorways "see through".
   - The rectangle is the 2nd–98th percentile in wall coordinates (u, v).
   - Classify: bottom edge v₀ < 5 cm means door; pixel depth more than 0.3 m behind the plane means an open passage; touching the image border means truncated.
9. **Fixtures.**
   - Footprints come from points 0.02–2.2 m above the floor (cabinet, island, radiator, column, fireplace, stairs, toilet, bathtub…).
   - Clearances: door swing radius equal to the door width, radiator 0.3 m, hearth 0.5 m, passage corridor 0.9 m, and 1 m in front of the camera.
10. **Free-floor grid.**
    - 5 cm cells aligned to the Manhattan axes, each FREE, OCCUPIED, CLEARANCE, UNKNOWN or OUTSIDE.
    - FREE cells come from floor and rug pixels back-projected onto the floor plane, followed by a 3×3 morphological closing.
    - Store a distance-to-wall transform for the layout stage.
11. **Scale check.** Compare camera height, door height (about 2.0–2.1 m) and ceiling height (2.4–3.0 m). If they agree with each other but are more than 15% off the priors, rescale by the median ratio and record a confidence value.
12. **Export.**
    - JSON: planes, polygon, openings, fixtures, the grid as a PNG, K, and camera-to-world with Z up and the floor at z = 0.
    - In Blender every proxy surface is a mesh with `is_shadow_catcher=True`. Real occluders get a depth mesh built from `P`.
    - The per-pixel occlusion decision is made in our compositor, from CG depth vs MoGe depth, refined by the SAM matte.

**Tested:** I ran the core steps on a synthetic room with 2% depth noise, 5% outliers and 0.08 normal noise.

| Quantity | Estimated | True |
|---|---|---|
| Camera height | 1.446 m | 1.450 m |
| Floor tilt error | 0.20° | 0° |
| Wall offsets | −2.196 / 1.802 / 5.002 m | −2.2 / 1.8 / 5.0 m |
| Room height | 2.701 m | 2.700 m |
| Opening rectangle | exact | — |

Reference code: `/tmp/claude-0/-home-user/c09d70e1-5a72-5f0b-811a-6fd083338d9a/scratchpad/proxytest/proxy_core.py`, with tests `test_proxy.py` and `test_opening.py` in the same folder.
```python
def manhattan_yaw(nxy, w=None):                       # dominant horizontal axis (mod 90°)
    th = np.arctan2(nxy[:,1], nxy[:,0]); w = np.ones(len(th)) if w is None else w
    return np.angle(np.sum(w*np.exp(4j*th)))/4
def opening_rect(uv, K, n, d, up, floor_d, q=(2,98)):   # mask px -> (u0,u1,v0,v1) m on wall n·x+d=0
    r = np.c_[uv, np.ones(len(uv))] @ np.linalg.inv(K).T; X = r*(-d/(r@n))[:,None]
    eu = np.cross(up, n); eu /= np.linalg.norm(eu)
    return (*np.percentile(X@eu, q), *np.percentile(X@up + floor_d, q))
```

### Lighting calibration
1. **Linearize** the photo with the inverse sRGB curve. Mask saturated pixels (≥250) and specular pixels (the Marigold residual).
2. **Windows.**
   - One `AREA`/`RECTANGLE` light per opening: same size, 2 cm inside the room, facing inward, `spread=π`.
   - Initial colour: mean linear RGB of the unsaturated pane pixels, or 6500 K if the window is blown out.
   - With a world environment, set `light.cycles.is_portal=True`.
   - Curtains: keep the same light and expect a lower fitted weight.
3. **Sun.**
   - Detect floor or wall pixels brighter than 2× the local median with sharp edges.
   - Let aᵢ = wᵢ − pᵢ (window corner minus sun-patch corner, both on their planes). The sun direction d that minimizes Σ‖aᵢ×d‖² is the principal eigenvector of Σaᵢaᵢᵀ. If corners can't be matched, use the window/patch centroid pair, as Karsch does.
   - Sun irradiance: E_sun = (R−1)·E_amb/cosθ, where R is the lit/shadow brightness ratio and θ is the angle to the surface normal.
4. **Fixtures.** Light, chandelier, sconce or lamp masks with V > 0.95 are switched on. Place a point or area light where the ray hits the ceiling or wall, at 2700–3000 K.
5. **Ambient.** A world environment (the DiffusionLight-Turbo probe rotated into world space, or a uniform colour), plus a back-wall proxy behind the camera.
6. **Solve.**
   - Put each light in its own Light Group, and render the proxy alone at about 480 px, 64 spp, with OIDN denoising.
   - Proxy albedo ρ comes from Marigold, or 0.75 for neutral walls.
   - Per channel, run `scipy.optimize.nnls(A, b)` on about 20k floor and wall pixels, where A = [R_k(x)] and b = I_lin(x). The weights are the light colours.
   - Clamp chromaticities near the blackbody locus. Accept the fit if the median relative error is under 15%; otherwise use the fallback.
7. **Render and composite.**
   - Cycles settings: `use_pass_shadow_catcher`, `film_transparent`, fixed seed, denoiser, linear EXR output.
   - In our code: `lin = Combined + (1−α)·photo_lin·SC`, then encode to sRGB 8-bit.
   - Effect mask: `E = dilate(α>0, 1px) ∪ (1−SC > 0.002) ∪ (catcher reflection > 0.002)`. Restrict it to a 3D zone around each object (at most 1.5× the object's height) so render noise cannot change pixels elsewhere.
   - **Outside E, copy the original bytes.**

## Risks & limitations
- **Licenses.**
  - The SAM and DINOv3 licenses are custom: trade-control and ITAR clauses, the license text must ship with the product, and Meta can change the DINOv3 terms. They need legal review.
  - I could not fetch the ADE20K, Structured3D or Laval dataset terms (UNVERIFIED). Weights trained on restricted data carry some residual risk.
  - The license tags for EoMT-DINOv3 conflict between HF and GitHub.
- **Windows packaging.**
  - Avoid NATTEN, detectron2, the MSDeformAttn custom ops, and Triton (needed by MoGe-3 and flash-attn).
  - The official SAM 3 repo needs Python 3.12 and CUDA 12.6, so prefer the transformers implementations.
  - bf16 needs Ampere or newer; use fp16 on Turing.
  - Load models one at a time and call `empty_cache()` between them to fit in 8 GB.
- **Geometry.**
  - Monocular metric scale can be off by 10–20%, which makes furniture the wrong size; step 11 mitigates this.
  - Glass and mirrors corrupt depth (handled by ray–plane intersection).
  - Textureless walls give noisy normals (handled by VP refinement).
  - Fisheye or very wide lenses need GeoCalib's radial camera models.
- **Lighting.**
  - HDR-tonemapped listing photos are not linear, which biases the fitted intensities.
  - Light sources behind the camera are invisible, and blown-out windows carry no colour information.
  - DiffusionLight output depends on the initial noise; fix the seed and keep it optional.
- **Bit-exactness.** It holds only in the PNG/EXR frames. H.264/H.265 with 4:2:0 chroma changes every pixel slightly. Ship a high-quality yuv420p MP4 (CRF about 12), plus an optional lossless master (FFV1 or x264 `-qp 0`) for audits.
- **Performance.** All VRAM and runtime figures are estimates. Benchmark on 8, 12 and 24 GB cards.

## Sources
- **EoMT:** https://github.com/tue-mps/eomt · https://github.com/tue-mps/eomt/blob/master/model_zoo/dinov2.md · https://github.com/tue-mps/eomt/blob/master/model_zoo/dinov3.md · https://huggingface.co/tue-mps/ade20k_semantic_eomt_large_512 · https://huggingface.co/tue-mps/ade_semantic_eomt_large_512_dinov3 · https://huggingface.co/tue-mps/ade20k_panoptic_eomt_large_640 · https://huggingface.co/docs/transformers/main/en/model_doc/eomt · https://newreleases.io/project/pypi/transformers/release/4.54.0
- **ADE20K labels:** https://huggingface.co/datasets/huggingface/label-files/raw/main/ade20k-id2label.json
- **OneFormer:** https://github.com/SHI-Labs/OneFormer · https://huggingface.co/shi-labs/oneformer_ade20k_swin_large · NATTEN: https://pypi.org/project/natten/0.14.6
- **Mask2Former:** https://github.com/facebookresearch/Mask2Former · https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md · https://huggingface.co/facebook/mask2former-swin-large-ade-semantic
- **SegFormer:** https://github.com/NVlabs/SegFormer/blob/master/LICENSE · https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640
- **ConvNeXt:** https://github.com/facebookresearch/ConvNeXt/blob/main/semantic_segmentation/README.md · https://huggingface.co/openmmlab/upernet-convnext-large
- **DINOv3:** https://github.com/facebookresearch/dinov3 · https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md · https://arxiv.org/abs/2508.10104
- **SAM 3:** https://github.com/facebookresearch/sam3 · https://github.com/facebookresearch/sam3/blob/main/LICENSE · https://huggingface.co/facebook/sam3 · https://huggingface.co/docs/transformers/main/en/model_doc/sam3 · https://arxiv.org/abs/2511.16719
- **SAM 2:** https://github.com/facebookresearch/sam2 · https://huggingface.co/docs/transformers/main/en/model_doc/sam2
- **Grounded-SAM-2 / Florence-2:** https://github.com/IDEA-Research/Grounded-SAM-2 · https://huggingface.co/microsoft/Florence-2-large
- **MoGe:** https://github.com/microsoft/MoGe · https://huggingface.co/Ruicheng/moge-2-vitl-normal · https://arxiv.org/abs/2507.02546
- **GeoCalib / PerspectiveFields:** https://github.com/cvg/GeoCalib · https://github.com/jinlinyi/PerspectiveFields/blob/main/LICENSE
- **Lines and VPs:**
  - OpenCV LSD note: https://github.com/opencv/opencv/blob/4.x/modules/imgproc/include/opencv2/imgproc.hpp
  - https://github.com/iago-suarez/ELSED · https://github.com/cvg/DeepLSD · https://github.com/zhou13/neurvps · https://github.com/WU-CVGL/GlobustVP · https://github.com/danini/progressive-x
- **Layout nets:** https://github.com/CYang0515/NonCuboidRoom · https://github.com/JonathanLee112/uLayout · https://github.com/jcliu0428/ZeroPlane · https://github.com/manycore-research/SpatialLM · https://github.com/naver/dust3r/blob/main/LICENSE · https://arxiv.org/abs/2502.16779
- **DiffusionLight:**
  - https://github.com/DiffusionLight/DiffusionLight (`inpaint.py`, `relighting/argument.py`) · https://huggingface.co/DiffusionLight/DiffusionLight
  - https://github.com/DiffusionLight/DiffusionLight-Turbo · https://huggingface.co/DiffusionLight/TurboLoRA · https://diffusionlight.github.io/turbo · https://arxiv.org/abs/2507.01305
  - https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 · https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0 · https://huggingface.co/Intel/dpt-hybrid-midas · https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
- **Other lighting estimators:** https://huggingface.co/nvidia/LuxDiT · https://github.com/Wanggcong/StyleLight · https://lvsn.github.io/everlight/ · https://research.nvidia.com/labs/toronto-ai/DiPIR/
- **Karsch / compositing / intrinsics:** https://arxiv.org/abs/1912.11565 · https://yaksoy.github.io/papers/SIG25p-LightAwareCompositing.pdf · https://github.com/willipwk/image-composing · https://github.com/compphoto/Intrinsic · https://huggingface.co/prs-eth/marigold-iid-lighting-v1-1
- **Blender:**
  - Manual: https://docs.blender.org/manual/en/latest/render/cycles/object_settings/object_data.html · https://docs.blender.org/manual/en/latest/render/layers/passes.html · https://docs.blender.org/manual/en/latest/render/cycles/light_settings.html
  - Release notes: https://developer.blender.org/docs/release_notes/3.0/cycles/
  - API: https://docs.blender.org/api/current/bpy.types.Object.html · https://docs.blender.org/api/current/bpy.types.Lightgroups.html · https://docs.blender.org/api/current/bpy.types.World.html · https://projects.blender.org/blender/blender/src/branch/main/intern/cycles/blender/addon/properties.py
- **Open3D:** https://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html