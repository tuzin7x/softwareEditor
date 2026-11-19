# Single-image metric geometry for indoor photos: research report (2026-11-15)

## 1. Findings

**Bottom line:** I found one top-tier single-image model that gives metric points, normals and FOV, has commercially usable code and weights, and runs on plain PyTorch on Windows: **MoGe-2** (MIT code, MIT weights). The 2026 entries (MoGe-3, MetricAnything, DepthMaster) are also permissive. However, MoGe-3 needs Triton, which is Linux-only. MetricAnything did worse than MoGe-2 on our real listing photo. DepthMaster could not be run here. For camera matching, the weakest link is the FOV and principal point, not the depth. So the model output must be refined geometrically, using planes, the Manhattan assumption and size priors.

1. **MoGe-2** (Microsoft, arXiv 2507.02546)
   - **Outputs:** one forward pass gives a metric point map, metric depth, normals (`-normal` checkpoints), a validity mask and normalized intrinsics. The metric scale comes from a CLS-token MLP that is separate from the affine-invariant point map.
   - **Assumptions:** the focal/shift solver "assumes the optical center is at the center of the map; the map is undistorted", so cx = cy = 0.5 always.
   - **Cheap FOV testing:** the optional `fov_x` argument is used only in post-processing, never by the network.
   - **Accuracy:** metric point map without ground-truth intrinsics, Rel_p % / δ1 %: NYUv2 4.44/98.3, iBims-1 5.63/97.4, ETH3D 7.19/97.7, HAMMER 13.4/87.0. For comparison, UniDepthV2 scores 4.83/98.0 and 5.23/97.9; Depth Pro scores 6.13/97.3 and 6.89/96.9.
   - **Speed and memory:** A100, FP16: 29 ms / 0.96 GB (1200 tokens), 39 ms / 1.1 GB (2500), 55 ms / 1.3 GB (3600), 108 ms / 1.9 GB (7200). The README quotes about 60 ms on an RTX 3090.
   - **Windows compatibility:** attention uses `F.scaled_dot_product_attention`. There is no xformers and no custom CUDA code.
   - **ONNX:** official FP32 ONNX exports exist, but they do not include the focal/shift post-processing.
   - **Limitations stated in the paper:** thin structures; "maintaining straight and aligned structures under a significant scale difference between foreground and background"; metric deviations on out-of-distribution scenes.

2. **MoGe-3** (arXiv 2607.17967, released 2026-08-18)
   - **What it adds:** MoGe-2 plus sparse 3D voxel refinement, which gives sharper fine detail.
   - **No indoor metric gain** in its own re-evaluation (ViT-L, 3 steps vs MoGe-2): NYUv2 4.86 vs 4.21, iBims 5.66 vs 5.92, HAMMER 15.7 vs 12.9.
   - **Speed:** 121 ms vs 39 ms (A100, 700²).
   - **Hard dependency:** FlexGEMM, which needs Triton.
     - Official Triton 3.9 wheels exist for Linux only (manylinux).
     - Windows would need the community `triton-windows` fork; whether that works with FlexGEMM is UNVERIFIED.
   - **Packaging:** current `main` (v3.0.0) installs flex-gemm even if you only use v2. Pin commit `b942f00` (v2.0.0) instead.

3. **MetricAnything** (arXiv 2601.22054; Apache-2.0 code and weights)
   - **What it is:** "Student-PointMap" is a fine-tuned MoGe-2 ViT-L. It loads with `moge.model.v2.MoGeModel` and has no normals.
   - **Paper claims:**
     - Mean FOV error: ETH3D 2.50°, iBims-1 3.74°, vs MoGe-2 4.66° / 8.17°.
     - Metric depth without intrinsics on iBims: AbsRel 11.3 vs MoGe-2 13.6.
   - **Our test:** scale was about 19% too small (details below).

4. **DepthMaster** (arXiv 2606.12368, NeurIPS 2026)
   - **License and API:** MIT code and MIT weights. The `infer()` call has a MoGe-like contract: FOV estimation plus metric depth and points. It also handles equirectangular panoramas.
   - **Accuracy:** metric point map NYUv2 5.19/97.8, iBims 5.49/97.2, HAMMER 9.80/95.1.
   - **Testing:** the authors tested on Linux only. I could not run it here because the sandbox refused to execute the cloned repo. Practical use is UNVERIFIED.

5. **Depth Anything 3** (arXiv 2511.10647; "-1.1" checkpoints Dec 2025)
   - **`DA3METRIC-LARGE` (Apache-2.0):**
     - Predicts canonical depth. Meters = `focal_px*out/300`, so it needs a focal length from elsewhere.
     - NYUv2 δ1 0.963 / AbsRel 0.070; SUN-RGBD 0.973 / 0.105.
     - With a poor focal it collapses: iBims δ1 43.5 in MetricAnything's no-intrinsics test.
   - **Any-view models** also estimate intrinsics; their FOV accuracy is UNVERIFIED.
   - **Licenses:**
     - SMALL and BASE: Apache.
     - GIANT and NESTED: CC BY-NC 4.0.
     - DA3-LARGE-1.1: the HF card says Apache-2.0 but the GitHub table says CC BY-NC 4.0. Treat it as non-commercial.
   - **Dependencies are heavy:** xformers, open3d, pycolmap and `numpy<2`, which conflicts with MoGe `main`'s `numpy>=2`.

6. **Rejected on license** (read from the actual LICENSE files and HF cards)
   - **Depth Pro:** the weights use the Apple ML Research Model license, "exclusively for Research Purposes… does not include… use in any commercial product". The code license itself is permissive.
   - **UniDepth / UniDepthV2:** CC BY-NC 4.0.
   - **UniK3D:** CC BY-NC-SA 4.0.
   - **VGGT:**
     - VGGT-1B: CC BY-NC.
     - VGGT-1B-Commercial: gated; commercial use allowed except military. Its output is scale-free anyway.
   - **MetaDepth:** FAIR Noncommercial.
   - **MapAnything:**
     - `facebook/map-anything`: CC BY-NC.
     - `map-anything-apache`: Apache, but 4.9 GB and built for multi-view.
   - **Pi3:** Pi3X is CC BY-NC (Pi3 itself is BSD but scale-free).
   - **HunyuanWorld-Mirror:** territorial; it "does not apply in the EU, UK and South Korea".
   - **InfiniDepth:** free, but every project must register and the obligation passes to recipients; relative depth only.
   - **Metric3D v2:** BSD-2 code, but the weights carry no license and the authors ask for "commercial inquiries". It also needs intrinsics.
   - **Depth Anything V2 Metric-Hypersim:** the HF cards say Apache, but the GitHub README says Base/Large/Giant are CC BY-NC. Only ViT-S is safe, and it has no FOV estimate.

7. **Diffusion depth and normals** (Marigold v1-1 OpenRAIL++, v1-0 Apache; Lotus Apache): affine-invariant only, no FOV or metric scale. Not needed.

8. **Calibration helpers**
   - **GeoCalib:** Apache code, CC-BY-4.0 weights.
     - Estimates gravity, focal and distortion (k1/k2 or division model). The principal point is fixed at the centre.
     - Stanford2D3D AUC at 1/5/10°: roll 83.1/91.8/94.8, pitch 52.3/74.8/84.6, FoV 17.4/40.0/59.4.
     - Strong on gravity, weak on FOV.
   - **AnyCalib** (ICCV 2025; Apache code and weights):
     - Estimates fx, fy, cx, cy plus radial or Kannala-Brandt distortion.
     - `anycalib_edit` is trained on cropped and stretched images.
     - It is the only permissive principal-point estimator I found. Not tested here.

9. **Benchmark gaps**
   - I found no zero-shot single-image metric numbers on ScanNet or Hypersim for these models; Hypersim is in most training sets (UNVERIFIED).
   - MoGe-1 mean FOV error was NYUv2 3.41°, ETH3D 2.50°, iBims-1 2.81°.
   - The MoGe-2 paper reports no FOV table, and MetricAnything's MoGe-2 FOV numbers come from a third party.

### Empirical check in this session

Setup: CPU only, 4 vCPU, torch 2.15 CPU, fp32. Photo: a 1500×1000 wide-angle listing-style interior (MoGe repo `example_images/01_HouseIndoor.jpg`). It contains two standard interior doors (about 2.03 m nominal), a framed mirror and a bright window.

| Estimator | CPU time | hFOV | Camera height | Floor⟂wall angles | Door heights |
|---|---|---|---|---|---|
| MoGe-2 ViT-L-normal | 4.6 s (1200 tok) / 9.9 s (2500) / 21.1 s (3600) | 92.2° | 1.19 m | 89.9° / 89.4° | 1.98–2.08 m |
| MoGe-2 ViT-S-normal | 2.2 s / 4.7 s | 88.5–89.8° | 1.19 m | – | 2.10 / 2.17 m |
| MetricAnything student | 8.7 s (2500) | 88.5° | 1.00 m | 88.8° / 87.7° | 1.61 / 1.65 m (−19%) |
| GeoCalib pinhole / radial | 1.9 s | 81.4° / 86.1° (k1≈0; own vFOV σ 16.5°) | – | MoGe-2 forced to 81.4°: 87.6° / 84.4° | forced: 1.78 / 1.83 m |

- **FOV sweep:** I ran MoGe-2 ViT-L with FOV forced from 75° to 98°. Its own FOV gave the best floor-wall orthogonality and the right door height. Door heights followed the FOV: 1.91 m at 86°, 2.18 m at 98°. In other words, an FOV error becomes a scale error.
- **Pitch:** from the floor plane, MoGe-2 gave 1.1° down; GeoCalib gave 0.4° down.
- **Mirror:** it sits on the wall plane (4.24 m vs 4.25 m for the wall). This is anecdotal.
- **Floor to ceiling:** 2.35 m.
- **Second photo** (an office shot pitched down): MoGe-2 gave 64.4° FOV, 17.5° down and 1.81 m camera height. GeoCalib gave 71.2° and about 14° down.
- **CPU timings are only indicative:** a later run under CPU load took 56 s.

## 2. Comparison table

Accuracy column: metric point map, Rel_p / δ1, as NYUv2 · iBims-1. Sources: MoGe-2 paper; MoGe-3 paper for MoGe-3, DA3 and UniK3D; each model's own paper otherwise.

| Model (HF id, checkpoint) | Outputs | NYUv2 · iBims-1 | GPU cost (A100 FP16) | License: code / weights | Windows | Verdict |
|---|---|---|---|---|---|---|
| MoGe-2 ViT-L-normal `Ruicheng/moge-2-vitl-normal` (1.32 GB; vitb 419 MB, vits 141 MB) | metric points/depth, normals, mask, FOV (centred principal point) | 4.44/98.3 · 5.63/97.4 | 39–55 ms, 1.1–1.3 GB | MIT / MIT | pure PyTorch ✔ | **Primary** |
| MoGe-3 ViT-L `Ruicheng/moge-3-vitl` (1.48 GB; vitg 5.0 GB) | same as MoGe-2, sharper | 4.86/98.2 · 5.66/97.5 | 121 ms | MIT / MIT | Triton + FlexGEMM ✗? | Later upgrade |
| DepthMaster `VCLab-PolyU/DepthMaster` (1.54 GB) | metric points/depth, FOV; no normals | 5.19/97.8 · 5.49/97.2 | n/a | MIT / MIT | Linux-tested only | Fallback #1 |
| MetricAnything `yjh001/metricanything_student_pointmap` (1.31 GB) | metric points, FOV | iBims depth AbsRel 11.3 | ≈ MoGe-2 | Apache / Apache | same as MoGe-2 | FOV second opinion |
| DA3METRIC-LARGE (1.34 GB) | metric depth (needs focal) | NYU depth δ1 96.3 | n/a | Apache / Apache | xformers, `numpy<2` | Fallback #2 (scale vote) |
| DA3 LARGE-1.1 / GIANT / NESTED (1.6 / 5.4 / 6.8 GB) | depth, pose, intrinsics | DA3 (variant not stated): 4.89/97.5 · 5.25/97.2 | 213 ms | Apache / NC or conflicting | heavy | ✗ license |
| Depth Pro `apple/DepthPro` (1.9 GB) | metric depth, focal | 6.13/97.3 · 6.89/96.9 | 139 ms, 3.7 GB | permissive / research-only | ✔ | ✗ license |
| UniDepthV2 / UniK3D | metric depth, camera | 4.83/98.0 · 5.23/97.9 / 4.64/97.9 · 4.58/98.0 | 33–50 ms | CC BY-NC / CC BY-NC-SA | – | ✗ license |
| Metric3D v2 | metric depth (needs intrinsics), normals | depth with GT intrinsics: NYU 7.16/96.5 | 87 ms, 1.4 GB | BSD-2 / no license | ONNX ✔ | ✗ unclear |
| DA V2 Metric-Hypersim | metric depth, no FOV | NYU depth 16.4/80.9 | 24 ms | Apache / conflicting (S safe) | ✔ | ✗ |
| GeoCalib | gravity, focal, k1 | Stanford2D3D FoV AUC@10° 59.4 | 1.9 s on CPU | Apache / CC-BY-4.0 | ✔ | gravity + distortion |
| AnyCalib | fx, fy, cx, cy, distortion | n/a | n/a | Apache / Apache | ✔ | principal point |

## 3. Recommendation

**Primary: MoGe-2 ViT-L-normal plus deterministic geometric refinement.**
- Best or near-best indoor metric point accuracy without known intrinsics.
- One pass gives everything the camera match needs: metric points, normals, mask, FOV.
- MIT code and MIT weights.
- About 1–2 GB VRAM, so it fits every 8–24 GB card. CPU fallback is 5–20 s per photo.
- No xformers, Triton or custom ops; ONNX route available.
- Validated here: doors within 3% of nominal, floor⟂wall at 89.4–89.9°.
- Low-end GPUs, CPU and CI: use ViT-B or ViT-S-normal (also MIT).

**Fallback #1 (full replacement): DepthMaster.** MIT/MIT, same output contract, stronger on HAMMER. It must be validated on Windows first.

**Fallback #2 (independent scale vote): DA3METRIC-LARGE** (Apache). Feed it the refined focal and vendor only the model code, without DA3's heavy dependencies.

**Auxiliary tools:**
- GeoCalib for gravity and distortion; ignore its FOV unless its uncertainty is low.
- AnyCalib-edit for the principal point of edited photos.
- MetricAnything only as a second FOV opinion (its scale failed in our test).

**MoGe-3:** adopt only if occlusion edges around built-ins need it and Triton on Windows has been proven.

**On the planned approach:** confirmed, with these changes:
- (a) The camera must include principal-point lens shift and lens distortion. Distortion is applied to the CG layers only, never to the photo.
- (b) The room proxy should be fitted planes plus extruded footprints of built-ins, not the raw depth mesh.
- (c) Add an FOV/scale arbitration step (orthogonality + metric priors + a second model) and log a confidence score.
- (d) Run geometry in its own process and release VRAM before Cycles.

## 4. Implementation notes

**Install (tested here; skips Triton and gradio):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # or cu128 / cpu
pip install --no-deps "git+https://github.com/microsoft/MoGe.git@b942f00bdc2a2a23ebb474fbe034d487e6dcceec"
pip install opencv-python-headless scipy huggingface_hub click \
 "git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183" \
 "git+https://github.com/EasternJournalist/pipeline.git@866f059d2a05cde05e4a52211ec5051fd5f276d6"
```

**Inference:**
```python
import cv2, math, torch
from moge.model.v2 import MoGeModel
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(dev).eval()
rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB); H, W = rgb.shape[:2]
x = torch.tensor(rgb / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
with torch.inference_mode():
    out = model.infer(x, num_tokens=3600, use_fp16=(dev == "cuda"))   # optional fov_x=<deg>
# keys: points (H,W,3) metres, OpenCV camera frame (x right, y down, z forward); depth (H,W);
#       normal (H,W,3); mask (H,W) bool (masked pixels = inf); intrinsics (3,3) normalised: fx/W, fy/H, cx=cy=0.5
K = out["intrinsics"].cpu().numpy(); hfov = 2 * math.degrees(math.atan(0.5 / K[0, 0]))
```
- Use `use_fp16=False` on CPU.
- `num_tokens`, not output size, sets cost; the outputs are always full input resolution.
- Normalized K lets you render at full photo resolution even if the analysis was done on a downscaled copy.
- Supported aspect ratios: 2:1 to 1:2.

**Floor plane, camera height, pitch and roll (validated here):**
```python
def plane_ransac(X, thr=0.015, iters=1000, rng=np.random.default_rng(0)):
    best = None
    for _ in range(iters):
        a, b, c = X[rng.choice(len(X), 3, replace=False)]
        n = np.cross(b - a, c - a); s = np.linalg.norm(n)
        if s < 1e-9: continue
        n /= s; inl = np.abs(X @ n - n @ a) < thr
        if best is None or inl.sum() > best.sum(): best = inl
    Q = X[best]; c = Q.mean(0); n = np.linalg.svd(Q - c)[2][-1]; d = -n @ c
    return (n, d) if d > 0 else (-n, -d)          # n = world-up in camera frame, d = camera height (m)
P, N, M = (out[k].cpu().numpy() for k in ("points", "normal", "mask"))
up, cam_h = plane_ransac(P[M & (N[..., 1] < -0.9) & floor_seg][::7])   # floor_seg: segmentation or lower half
pitch = math.degrees(math.asin(-up[2])); roll = math.degrees(math.atan2(up[0], -up[1]))   # pitch > 0 = looking down
```
- Walls: RANSAC per wall segment, keeping normals within 10° of perpendicular to `up`. Snap wall-to-wall angles within 5° of 90° to exactly 90°.
- Ceiling: fit a plane constrained to be parallel to the floor.

**Blender camera** (world: floor is z = 0; points map as `X_w = R @ X_cv + (0, 0, cam_h)`):
```python
xw = np.array([1., 0, 0]) - up[0] * up; xw /= np.linalg.norm(xw); yw = np.cross(up, xw)
R = np.stack([xw, yw, up]); Rb = R @ np.diag([1, -1, -1])          # Blender camera looks down -Z, +Y up
cam.matrix_world = Matrix([[*Rb[0], 0], [*Rb[1], 0], [*Rb[2], cam_h], [0, 0, 0, 1]])
cam.data.sensor_fit = 'AUTO'; cam.data.lens_unit = 'FOV'
cam.data.angle = 2 * math.atan(0.5 / (K[0, 0] if W >= H else K[1, 1]))
cam.data.shift_x = (W / 2 - cx_px) / max(W, H); cam.data.shift_y = (cy_px - H / 2) / max(W, H)
```
The Blender docs say lens shift is "equivalent to rendering… larger FOV and cropping off-center" but do not state its units. Verify the shift sign and units with a checkerboard render (UNVERIFIED).

**FOV and scale arbitration:**
- Sweep `fov_x` over hFOV ± 8° in 1° steps. Each GPU run is about 40–60 ms; the network forward does not depend on FOV.
- Pick the minimum of: Σ(floor–wall angle − 90°)² + Σ(adjacent wall angle − 90°)² + λ·door-height prior.
- Plausibility ranges:
  - ceiling 2.3–3.2 m;
  - doors 1.95–2.15 m;
  - camera height 0.8–1.9 m;
  - kitchen counters 0.85–0.95 m.
- If the cues disagree by more than 10%, rescale to the median of the cues and flag low confidence.
- DA3 scale vote: `depth_m = f_proc*pred.depth/300`, where `f_proc` is the focal rescaled to `process_res`. The resolution convention for the focal is UNVERIFIED.

**Edited and distorted photos:**
- **Principal point:** if verticals are parallel in the image but the floor gives a pitch ≠ 0 by more than 2°, the photo was keystone-corrected. Set pitch to 0 and take cy from the horizon row or from `AnyCalib(model_id="anycalib_edit").predict(x, cam_id="pinhole")`.
- **Distortion:** if GeoCalib `GeoCalib(weights="distorted").calibrate(img, camera_model="simple_radial")` returns a significant k1:
  1. analyse an undistorted copy of the photo;
  2. render the pinhole CG with about 5% overscan;
  3. build remap maps with `cv2.undistortPoints` over the photo's pixel grid and `cv2.remap` the CG RGBA, shadow and mask layers;
  4. composite them over the untouched photo.

**Resources:**
- Measured CPU torch 2.15 footprint (Linux): 809 MB; MoGe package 1.2 MB; utils3d 1.4 MB.
- Free VRAM before Cycles (separate process, or `del model; torch.cuda.empty_cache()`).
- CI: ViT-S-normal on CPU (2–5 s per image).
- Test assets in `/tmp/claude-0/-home-user/c09d70e1-5a72-5f0b-811a-6fd083338d9a/scratchpad/`:
  - `scripts/moge_test.py`, `scripts/fov_check.py`, `scripts/geocalib_test.py`, `scripts/probe.py`;
  - `venv/` (torch CPU, moge 2.0.0, geocalib);
  - `ckpt/` (checkpoints).

## 5. Risks and limitations

- **FOV ambiguity.** ±5° at 90° means about ±9% focal error and about ±6–7% object size error, plus a perspective mismatch. Shots facing a wall straight on (one-point perspective) leave the focal poorly determined; GeoCalib's own σ was 16.5° here. Published FOV errors for MoGe-2 differ by source (no MoGe-2 FOV table exists; 8.17° on iBims according to a third party).
- **Principal point.** Vertical correction (e.g. Lightroom "Upright"), crops and stretches break the centred-principal-point assumption of MoGe, GeoCalib and DepthMaster. A stretched image (fx ≠ fy) cannot be represented by a Blender camera except through pixel aspect.
- **Lens distortion.** All models above assume a pinhole except GeoCalib and AnyCalib. Listing photos are usually lens-profile-corrected (k1 ≈ 0 on ours), but raw ultra-wide photos are not.
- **Bright windows and glass.** Blown-out windows get arbitrary or masked depth. Snap them to the surrounding wall plane; this needs segmentation.
- **Mirrors.** Large frameless mirrors risk a "virtual room" behind the wall. The framed mirror in our photo was fine, but that is one case. Snap mirrors to the wall plane and never place furniture in them. Added furniture will not appear in mirror, glass or glossy-floor reflections; that is a product decision.
- **Textureless white walls.** The depth comes from learned priors, and walls can bow (a documented MoGe-2 limitation). Use RANSAC planes, never raw depth, for proxies and placement.
- **Metric scale on unusual scenes.** MetricAnything was 19% off on a real photo despite better benchmark numbers. Always require at least two independent scale cues.
- **Little visible floor** (bathrooms, close-ups): the plane is poorly determined. Fall back to floor/wall junction lines or a prior on camera height.
- **Panoramas** wider than 2:1 need tiling.
- **Dependencies and legal.** MoGe-3 depends on Triton (Linux-only); DA3 dependencies conflict with MoGe `main`; the DA3-LARGE-1.1 license is contradictory. Weight licenses do not settle where the training data came from, so get a legal review before release.
- **Hardware numbers.** GPU numbers here come from papers (A100 / RTX 3090); nothing was measured on Windows or consumer NVIDIA cards. DepthMaster, AnyCalib and DA3 were not executed.

## 6. Sources

- **MoGe**
  - Repo, README, LICENSE, pyproject: https://github.com/microsoft/MoGe
  - v2 pin: https://github.com/microsoft/MoGe/tree/b942f00bdc2a2a23ebb474fbe034d487e6dcceec
  - `infer()`: https://github.com/microsoft/MoGe/blob/main/moge/model/v2.py
  - Solver docstring: https://github.com/microsoft/MoGe/blob/main/moge/utils/geometry_torch.py
  - ONNX: https://github.com/microsoft/MoGe/blob/main/docs/onnx.md
  - Weights: https://huggingface.co/Ruicheng/moge-2-vitl-normal ; https://huggingface.co/Ruicheng/moge-3-vitl
  - Papers: MoGe-2 https://arxiv.org/abs/2507.02546 (Tables 2, B.3, B.4) ; MoGe-1 https://arxiv.org/abs/2410.19115 (Table 3) ; MoGe-3 https://arxiv.org/abs/2607.17967 (Tables 2, C.1, C.4)
- **Triton and FlexGEMM:** https://github.com/JeffreyXiang/FlexGEMM ; https://pypi.org/project/triton/ ; https://pypi.org/project/triton-windows/
- **MetricAnything:** https://github.com/metric-anything/metric-anything ; https://huggingface.co/yjh001/metricanything_student_pointmap ; https://arxiv.org/abs/2601.22054 (Tables 5–7)
- **DepthMaster:** https://github.com/PolyU-VCLab/DepthMaster ; https://huggingface.co/VCLab-PolyU/DepthMaster ; https://arxiv.org/abs/2606.12368
- **Depth Anything 3:** https://github.com/ByteDance-Seed/Depth-Anything-3 (README, pyproject, docs/API.md) ; https://huggingface.co/depth-anything/DA3METRIC-LARGE ; https://huggingface.co/depth-anything/DA3-LARGE-1.1 ; https://arxiv.org/abs/2511.10647 (Table 11) ; xformers wheels: https://download.pytorch.org/whl/xformers/
- **Depth Pro:** https://github.com/apple/ml-depth-pro/blob/main/LICENSE ; https://huggingface.co/apple/DepthPro/blob/main/LICENSE
- **UniDepth / UniK3D:** https://github.com/lpiccinelli-eth/UniDepth/blob/main/LICENSE ; https://github.com/lpiccinelli-eth/UniK3D/blob/main/LICENSE
- **Metric3D:** https://github.com/YvanYin/Metric3D ; https://huggingface.co/JUGGHM/Metric3D
- **VGGT:** https://github.com/facebookresearch/vggt ; https://huggingface.co/facebook/VGGT-1B-Commercial
- **Depth Anything V2:** https://github.com/DepthAnything/Depth-Anything-V2 ; https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large
- **MapAnything:** https://github.com/facebookresearch/map-anything ; https://huggingface.co/facebook/map-anything-apache
- **Pi3:** https://github.com/yyfz/Pi3 ; https://huggingface.co/yyfz233/Pi3X
- **HunyuanWorld-Mirror:** https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror/blob/main/License.txt
- **MetaDepth:** https://github.com/facebookresearch/metadepth/blob/main/LICENSE
- **InfiniDepth:** https://github.com/zju3dv/InfiniDepth/blob/main/LICENSE
- **Marigold / Lotus:** https://huggingface.co/prs-eth/marigold-depth-v1-1 ; https://huggingface.co/jingheya/lotus-depth-g-v2-1-disparity
- **GeoCalib:** https://github.com/cvg/GeoCalib
- **AnyCalib:** https://github.com/javrtg/AnyCalib ; https://arxiv.org/abs/2503.12701
- **Blender camera:** https://docs.blender.org/manual/en/latest/render/cameras.html ; https://docs.blender.org/api/current/bpy.types.Camera.html