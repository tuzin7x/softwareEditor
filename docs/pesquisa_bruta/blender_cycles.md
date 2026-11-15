# Cycles: inserting CG into a single photo from bpy, headless (Blender 4.2 LTS to 5.3)

**Date:** 2026-11-15.

**Releases in scope:**
- **5.2 LTS:** released 14 Jul 2026, supported until Jul 2028. The PyPI build is 5.2.4.
- **5.3.0:** released 10 Nov 2026. It is not an LTS release.
- **Older LTS:** 4.5 LTS and 4.2 LTS. 4.2 LTS reached end of life in Jul 2026.

**Evidence tags:**
- **[SRC]** = I read it in the Blender/Cycles source (main and release branches).
- **[RUN]** = I reproduced it in this session with bpy 5.3.0 (Linux, Cycles on CPU, Python 3.13), using the `/home/user/.venvs/staging` env.
- **[DOC]** = taken from the manual or release notes.

None of the GPU code paths were run here. Anything I could not confirm is marked UNVERIFIED.

---

## 1. Findings

### 1.1 Shadow catcher semantics (Cycles X, 3.0 and later)

**How the path splits** [SRC `kernel/integrator/intersect_closest.h`, `shadow_catcher.h`, `types.h`, `light/light.h`]
- The split happens at the first opaque camera-ray hit on an `is_shadow_catcher` object.
- **(a) The original path** keeps tracing the full scene. The catcher is shaded with its own material, and the result goes into an internal "full combined" buffer.
- **(b) A copy** traces only catcher objects. Non-catcher objects are made invisible by shifting their visibility bits.
- **Lights in the copy:** the copy only sees lights and emissive meshes that are flagged as catchers. Lights that are not flagged carry `SHADER_EXCLUDE_SHADOW_CATCHER`. The source comment reads "Exclude synthetic meshes from shadow catcher pass".
- **Secondary bounces** treat catchers as ordinary objects. As a result, CG furniture sees the catcher's material in reflections and bounce light.

**What the passes contain** [SRC `kernel/film/read.h`, `session/buffers.cpp`]
- **"Combined" is silently replaced.** As soon as the scene contains a catcher, the Combined output is taken from the internal `SHADOW_CATCHER_MATTE` pass.
  - RGB contains only the CG objects, and it is premultiplied.
  - Alpha is 0 over catchers and holdouts, and 1 over CG.
  - [RUN]: catcher and holdout pixels have α=0.000 and CG pixels α=1.000. Edge RGB ≈ α × interior colour, which confirms premultiplication.
- **"Shadow Catcher" pass (RGB):** per channel, it is (full − matte) / catcher-only. It is then alpha-overed onto white using the alpha of the full combined.
  - It is exactly 1.0 wherever no catcher sample exists, in the noisy version.
  - The denoised version is plain OIDN output (no albedo guide).
- **Compositing order** [DOC 3.0 notes]: multiply the background by the Shadow Catcher pass, then alpha-over Combined on top:

  `out = C.rgb + (1 − C.a) · plate_lin · SC`

- **With the pass disabled,** Combined alpha carries a grey approximation: `α = α_m + (1−α_m)·(1 − sat(mean(SC)))`, with black RGB. It cannot brighten anything.
  - [RUN]: α = 0.66 where SC ≈ 0.33.
  - The viewport always uses this approximation [SRC `blender/sync.cpp`; DOC manual: "With it disabled, a simple approximation is used instead"].

**What the ratio captures** [RUN, scene: white diffuse floor, red cube, sun]
- **Direct shadow:** the green channel drops to ≈0.33.
- **Colour bleeding:** on the floor in front of the red cube, R ≈ 1.02 while G/B ≈ 0.83.
- **AO-like occlusion** of the world light.
- **Reflections, on a glossy catcher:** a mirror floor reflecting the red cube gives SC = (1.00, 0.10, 0.10).
  - Reflections are multiplicative, so they only look right if the render of the empty proxy matches the photo.
  - The RNA description says: "Only render shadows and reflections on this object … considered to already exist in the footage".

**Catcher material choices** [RUN, green-channel shadow ratio]

| Catcher material | Shadow ratio (G) | What it means |
|---|---|---|
| White diffuse | 0.33 | Reference |
| Albedo 0.15 | 0.35 | Albedo almost cancels out |
| Diffuse + emission strength 1 | 0.70 | Emission sits in both numerator and denominator, so shadows are diluted |
| Light Path trick: `Is Camera Ray` → Diffuse, otherwise → Emission | 0.31 | Shadows stay at full strength, and reflections/bounce on the CG pick up the emission colour |

Conclusion: image-textured diffuse catchers are fine. Never let camera rays see emission on a catcher.

**Real vs synthetic lights**
- The switch is `Object.is_shadow_catcher` on the light object itself.
- **Real lights:** any new light object, including one made with `bpy.data.objects.new`, gets `OB_HIDE_CAMERA | OB_SHADOW_CATCHER`, so it counts as real [SRC `blenkernel/intern/object.cc`]. The World is forced to real [SRC `blender/light.cpp`].
  - A real light only shows up through the occlusion caused by CG.
- **Synthetic lights:** with `False`, the light is added onto the plate.
  - [RUN]: SC reaches up to 3.6 on open floor.
  - Emissive CG meshes are automatically synthetic.

**Catchers occlude CG behind them**
- A catcher is visible to the camera, so a CG object behind a catcher wall is hidden (α = 0) [RUN].
- In practice a catcher acts as holdout plus shadow receiver.

**Holdout vs catcher**
- Holdout only applies to primary paths. For secondary rays, a holdout object shades and casts shadows normally [SRC `shade_surface.h`].
- A holdout is not a catcher, so the ratio treats it as synthetic.
  - [RUN]: a holdout proxy of a real object casts a second shadow onto the plate (SC = 0.32).
  - The same proxy flagged as catcher cancels out (SC 0.91–1.09).
- If an object is both catcher and holdout, holdout wins and there is no split [SRC].

**Light Groups (3.2 and later)**
- Light and light-group passes skip catcher-hit paths. The source comment reads "Don't write any light passes for shadow catcher" [SRC `film/light_passes.h`].
- So `Combined_<group>` contains only the lighting on CG objects. [RUN]: it is 0 on catcher pixels.
- Two uses:
  - Relighting the furniture.
  - With catchers temporarily switched off: rendering per-light images of the empty proxy in a single pass, for least-squares light calibration.

**Pitfalls** [RUN]
1. The denoised SC is never exactly 1. It sits around 0.998 even on background and CG pixels. The noisy SC shows ±1–5% noise on open floor at 256 spp.
2. Border render with `use_crop_to_border=False` writes SC = 0 and α = 0 outside the border.
3. Deviations below about 0.5% do not change any 8-bit sRGB code. An SC of 0.998 changes 0 of 256 codes, and the float32 sRGB round trip is exact.

### 1.2 Camera from intrinsics [SRC `blenkernel/intern/camera.cc`, `cycles/blender/camera.cpp`; RUN]

**How Blender's camera model works**
- With `HORIZONTAL` fit, `lens/sensor_width` spans W pixels.
- With `AUTO` fit, `sensor_width` applies to the larger dimension.
- Shift is measured in units of the fitted dimension: W for HORIZONTAL, H·ycor for VERTICAL, max(W,H) for AUTO.
- A positive `shift_y` moves the window up.
- `ycor = aspy/aspx`, so `fy = fx·aspx/aspy`.

**Formulas** (OpenCV convention, pixel centres at integer coordinates):
- `lens = fx·36/W`
- `shift_x = ((W−1)/2 − cx)/W`
- `shift_y = (cy − (H−1)/2)·ycor/W`

For continuous conventions (e.g. MoGe, where normalised cx = 0.5 is the centre), use W/2 and H/2 instead.

**Test result** [RUN]: I measured 21 sphere centroids across landscape 240×160, portrait 160×240 and an fx≠fy case. Every error was within 0.035 px.

**What breaks:** dividing the shift by max(W,H) while using HORIZONTAL fit gives +4.1 / −6.6 px of error on a 160×240 portrait. The error grows with resolution.

**Axis conversion:** OpenCV to Blender camera is `M_bl = M_cv @ diag(1,−1,−1,1)`.

### 1.3 GPU and sampling

**Device API** [SRC `addon/properties.py` 4.2–5.3]
- Set `compute_device_type` ∈ {CUDA, OPTIX, HIP, ONEAPI, METAL}, then call `refresh_devices()`.
- `get_devices()` is deprecated and returns None.
- Device entries have `.name .type .id .use`. 5.3 adds `.meets_driver_requirement`.
- Finally set `scene.cycles.device='GPU'`.
- Command-line equivalent: `-- --cycles-device OPTIX`, with `+CPU` for hybrid rendering [DOC].

**Hardware requirements** [DOC]
- CUDA needs compute capability 5.0 or higher. 5.0 dropped everything below sm_50.
- OptiX needs NVIDIA driver 575 or newer, since 5.2.
- OIDN on the GPU needs compute capability 7.0 or higher, which covers every RTX card.
- OptiX is about 1.4–1.6× faster than CUDA, based on the Open Data medians for 4.5.

**Factory defaults in bpy 5.3** [RUN]
- Samples: 4096.
- Adaptive sampling: on, threshold 0.01, min samples 0 (automatic).
- Denoising: OIDN is on, but `denoising_use_gpu=False`. You must turn it on yourself.
- Light tree: on.
- `max_bounces`: 12.
- Persistent data: off.
- Film transparent: off.
- View transform: **AgX**.
- Seed 0, animated seed off.

**Other settings**
- **Motion blur** lives in `scene.render`: `use_motion_blur`, `motion_blur_shutter` (in frames), `motion_blur_position`. Per object: `ob.cycles.use_motion_blur` and `ob.cycles.motion_steps`. There is no `scene.cycles.motion_blur_position` [RUN].
- **Persistent data** "keep[s] render data in memory … for faster re-renders and animation renders" [DOC]. It only helps when all frames are rendered in one process.
- **The light tree** ignores emitter textures [DOC].

**Speed**
- Open Data medians (5.2.0, OptiX, samples/min summed over the three benchmark scenes):

  | GPU | Median |
  |---|---|
  | RTX 3060 | 1988 |
  | RTX 4060 | 2971 |
  | RTX 4070 | 5114 |
  | RTX 4070 SUPER | 5962 |
  | RTX 4090 | 11553 |
  | RTX 5070 | 5807 |
  | RTX 5090 | 16103 |

- Desktop CPUs score 150–660 (4.5 data), so a GPU is roughly 10–50× faster.
- **Estimated time per 1080p furniture frame** with adaptive sampling up to 256 spp and OIDN on the GPU (UNVERIFIED extrapolation; measure on each machine with a probe render):
  - RTX 4090: about 2–5 s.
  - RTX 4070: about 4–12 s.
  - RTX 3060: about 10–30 s.
- Catcher pixels can cost up to about 2× more because of the path split [SRC].

### 1.4 bpy wheel vs portable Blender, and the GPL

**bpy on PyPI**
- Python versions:
  - 4.2.x, 4.3, 4.4, 4.5.x and 5.0.x are cp311.
  - 5.1.x, 5.2.x and 5.3.0 are cp313 (`Requires-Python ==3.13.*`).
- Every version has a `win_amd64` wheel. The 5.3.0 wheel is 356 MB and needs `numpy>=2.2,<3`.
- Older versions move to download.blender.org/pypi.

**What the Windows wheel contains** [RUN, listed via HTTP range requests on the 5.2.4 and 5.3.0 `win_amd64` wheels]
- CUDA kernels: `kernel_sm_{50,52,60,70,75,86,120}.cubin.zst` and `kernel_compute_75.ptx.zst`.
- OptiX kernels: `kernel_optix*.ptx.zst`.
- `OpenImageDenoise_device_cuda.dll`.
- So CUDA, OptiX and OIDN on the GPU all work out of the box.
- The RTX 40 series (sm_89) relies on the sm_86 cubin through CUDA binary compatibility. I did not test this.
- The 5.3 Linux wheel bundles OIDN 2.5.0, OCIO 2.5 and OIIO 3.1.

**bpy-as-a-module limits** [DOC]
- It behaves as if launched with `--factory-startup`.
- `--threads` and `--log` have no API equivalent.
- The module cannot be reloaded, and only one .blend can be open per process.
- Other libraries using the GPU in the same process can block Cycles from using it.
- There is no crash log and no Ctrl-C handling.

**Portable Blender:** `blender-5.2.4-windows-x64.zip` is 406 MB and `blender-5.3.0-windows-x64.zip` is 425 MB.

**GPL**
- The bpy wheel is licensed GPL-3.0.
- **GNU GPL FAQ:**
  - Modules in a shared address space, or dynamically linked, form a single program.
  - exec, pipes, sockets and command-line arguments normally mean separate programs, unless they exchange intimate data structures.
  - Bundling separate programs together (an "aggregate") is allowed.
- **Blender FAQ:** published scripts that use the Blender Python API must be GPL. I only saw this via a search snippet because the page is behind Cloudflare, so it is UNVERIFIED verbatim.
- **What this means in practice:**
  - Never `import bpy` inside the proprietary process.
  - License the worker script, and everything it imports inside Blender, as GPL.
  - Exchange data as JSON/EXR files.
  - Ship the GPL text and a source offer.
- This is not legal advice.

### 1.5 Output and colour

**EXR layout in 5.0 and later** [SRC `rna_scene.cc`, `imbuf openexr_api.cpp`; RUN]
- Multilayer EXR is written as **multi-part** by default: one part per pass, e.g. "ViewLayer.Combined".
- `image_settings.use_exr_interleave=True` restores the old single-part layout.
- 4.x always writes a single part.

**Format settings:** `media_type` must be set **before** `file_format` (5.0 API notes).

**Pass renames in 5.0:**
- IndexOB → "Object Index"
- IndexMA → "Material Index"
- DiffCol → "Diffuse Color"
- "Combined", "Shadow Catcher" and "Depth" did not change.

With denoising on, "Noisy Image" and "Noisy Shadow Catcher" are also written, so always match pass names exactly.

**Reading EXR in Python**
- **OpenEXR 3.5.4** (BSD-3, Windows wheels for cp311–cp315):
  - Iterate over `f.parts`. `f.channels()` without an index returns only part 0 [RUN pitfall].
  - RGBA is grouped as (H, W, C), and row 0 is the top.
- **OpenImageIO 3.2.2.0** (Apache-2.0): each part is a subimage.
- **Precision:** half float is enough, since the step just below 1.0 is 2⁻¹¹, far smaller than 1/255.
  - Use ZIP or PIZ compression, not DWAA or B44.

**Colour management** [RUN]
- EXR is scene-linear and bit-identical whether the view transform is Standard or AgX.
- On the sRGB display, Standard is the IEC 61966-2-1 piecewise curve (5.3 OCIO config).
- A 16-bit PNG rendered with Standard equals sRGB-OETF(EXR) to within 0.6/65535. It uses straight alpha and clips at 1.0, so the Shadow Catcher pass (which goes above 1) cannot be stored in PNG.
- OIIO premultiplies PNGs on read unless you set `oiio:UnassociatedAlpha=1`.
- **Always set Standard.** The factory default is AgX, and neither AgX nor Filmic can be inverted back onto the plate.
- **Working space** in 5.x is `bpy.data.colorspace.working_space`, default 'Linear Rec.709'.
- Colour textures should be 'sRGB' and data maps 'Non-Color'.

### 1.6 glTF import and keyframes [RUN]

**glTF**
- `import_scene.gltf` works headless. It selects the new objects, keeps the hierarchy, and sets `rotation_mode='QUATERNION'`. **Keys on `rotation_euler` have no effect** until you change the rotation mode.
- `transform_apply` works headless once the object is selected with `select_set` and made active. The alternative is `mesh.transform()` at data level.

**Keyframes**
- 5.0 removed `Action.fcurves`. Use `anim_utils.action_get_channelbag_for_slot(...).fcurves`, which exists from 4.4.
- Interpolation lives on the left key of a segment.
- BACK with EASE_OUT on a drop from 2 to 0 overshoots to −0.105, which pushes the object into the floor. For landings use BOUNCE or a custom Bézier; keep BACK for scale "pops".
- **`KeyframePoint` references go stale after `fcurve.update()`.** Writes through them are silently lost, so fetch them again after each update.

---

## 2. Comparison table

| Option | Python | Windows x64 | NVIDIA GPU (CUDA/OptiX/OIDN-GPU) | Size | Isolation / GPL | Verdict |
|---|---|---|---|---|---|---|
| **A. Portable Blender 5.2.4 LTS, `blender.exe -b -P worker.py`** | bundled 3.13 | yes | yes (same kernels as B) | 406 MB | separate program; GPL covers only Blender and the worker | **Primary** |
| B. `bpy==5.2.4` in a child CPython 3.13 | ==3.13.* | yes | yes [RUN listing] | 340 MB wheel | same as A if it runs in a separate process | Fallback |
| C. `import bpy` inside the app process | 3.13 | yes | yes | – | combined work, so the app becomes GPL; competes with PyTorch for VRAM; cannot reload | Reject |
| D. Blender 5.3.0 | 3.13 | yes | yes | 425 MB | – | Validate, don't pin (not LTS) |
| E. Blender 4.5 LTS | 3.11 | yes | yes | – | older APIs | Only if Python 3.11 is required |

| Scene choice ([RUN] evidence) | Correct? |
|---|---|
| Real-object proxies as **catchers** (floor, walls, ceiling, real furniture) | Yes: shadows cancel, and the proxies still occlude CG |
| Real-object proxies as **holdouts** | No: double shadow (SC = 0.32) |
| Window/room lights left at the default `is_shadow_catcher=True` | Yes: only CG shadows appear |
| Light with `is_shadow_catcher=False` | Only for CG lamps meant to light the real floor |
| Emissive catcher seen by camera rays | No: shadows diluted to 0.70 |

## 3. Recommendation

**Primary: Portable Blender 5.2 LTS (pin 5.2.4) as a subprocess.**
- **Worker:** a GPL-licensed `worker.py` driven by JSON.
- **GPU:** OptiX, falling back to CUDA, then CPU. Enable OIDN on the GPU explicitly.
- **Output:** Film Transparent on, Shadow Catcher pass on, multi-part half-float EXR with ZIP compression, view transform Standard.
- **Scene rules:**
  - Every room proxy is a catcher; never use holdouts for real objects.
  - Real lights keep the default `is_shadow_catcher=True`.
  - The catcher material is diffuse, with an optional photo-emission branch for non-camera rays (the Light Path trick).
- **Compositing:** done in our own numpy code, with a gated SC and a bit-exact copy of the plate outside the touched mask.

Why this option:
- LTS support until Jul 2028.
- Every GPU kernel ships with it.
- It isolates crashes and VRAM.
- It gives the cleanest GPL boundary.
- It behaves the same as the bpy 5.3 build validated here.

**Fallback:** `bpy==5.2.4` in a dedicated embedded Python 3.13 child process. It is smaller and installable with pip, but follows the same process boundary. If neither OptiX nor CUDA is available, use the CPU device; expect it to be 10–50× slower.

## 4. Implementation notes (code tested on bpy 5.3 unless noted)

**GPU setup**

```python
def enable_gpu(prefer=("OPTIX", "CUDA")):
    p = bpy.context.preferences.addons["cycles"].preferences
    for kind in prefer:
        try: p.compute_device_type = kind
        except TypeError: continue
        p.refresh_devices()
        gpus = [d for d in p.devices if d.type == kind]
        if gpus:
            for d in p.devices: d.use = (d.type == kind)
            bpy.context.scene.cycles.device = "GPU"; return kind
    bpy.context.scene.cycles.device = "CPU"; return "CPU"
```

**Render settings**

```python
sc=bpy.context.scene; c=sc.cycles; r=sc.render
r.engine='CYCLES'; r.film_transparent=True; r.use_persistent_data=True
c.samples=256; c.use_adaptive_sampling=True; c.adaptive_threshold=0.01
c.use_denoising=True; c.denoiser='OPENIMAGEDENOISE'; c.denoising_use_gpu=True
c.denoising_input_passes='RGB_ALBEDO_NORMAL'; c.denoising_prefilter='ACCURATE'
c.seed=7; c.use_animated_seed=False        # stable noise in static regions
c.max_bounces=8; c.sample_clamp_indirect=5.0; c.caustics_reflective=c.caustics_refractive=False
r.use_motion_blur=True; r.motion_blur_shutter=0.5; r.motion_blur_position='CENTER'
sc.view_layers[0].cycles.use_pass_shadow_catcher=True
sc.view_settings.view_transform='Standard'; sc.view_settings.look='None'
im=r.image_settings
if bpy.app.version>=(5,0,0): im.media_type='MULTI_LAYER_IMAGE'
im.file_format='OPEN_EXR_MULTILAYER'; im.color_depth='16'; im.exr_codec='ZIP'
```

**Camera matching**

```python
def match_camera(sc, ob, fx, fy, cx, cy, W, H, M_cv, int_centres=True):
    r=sc.render; r.resolution_x,r.resolution_y,r.resolution_percentage=W,H,100
    r.pixel_aspect_x,r.pixel_aspect_y=(fy/fx,1.0) if fy>fx else (1.0,fx/fy)
    ycor=r.pixel_aspect_y/r.pixel_aspect_x; cam=ob.data
    cam.sensor_fit='HORIZONTAL'; cam.sensor_width=36.0; cam.lens=fx*36.0/W
    ox,oy=((W-1)/2,(H-1)/2) if int_centres else (W/2,H/2)
    cam.shift_x=(ox-cx)/W; cam.shift_y=(cy-oy)*ycor/W     # divide by W (HORIZONTAL fit)
    ob.matrix_world=Matrix(M_cv) @ Matrix.Diagonal((1,-1,-1,1))
```

**Proxy material, Light Path trick** [RUN with a constant emission colour]
- Mix shader `Fac` = Light Path → *Is Camera Ray*.
- Input 1 = Emission(photo texture via `Window` coordinates, colorspace 'sRGB', strength 1).
- Input 2 = Diffuse.
- Window regions are clipped in the photo, so boost the emission there (×10–50) or add area lights. The boost factors are UNVERIFIED.

**Compositing with exactness**

```python
f=OpenEXR.File(path); d={k:ch.pixels.astype(np.float32) for p in f.parts for k,ch in p.channels.items()}
C=d["ViewLayer.Combined"]; S=d["ViewLayer.Shadow Catcher"]       # (H,W,4) premult, (H,W,3)
if border: S[~in_border]=1.0; C[~in_border]=0.0                 # SC==0 outside border!
dev=np.abs(S-1).max(-1); w=np.clip((dev-0.005)/0.015,0,1)*influence   # geometry-derived 0..1
S_eff=1+(np.clip(S,0,4)-1)*w[...,None]
out=C[...,:3]+(1-C[...,3:4])*srgb_to_lin(plate)*S_eff
res=lin_to_srgb_u8(out); touched=(C[...,3]>0.5/255)|(w>0)
res[~touched]=plate[~touched]; assert (res[~touched]==plate[~touched]).all()
```

**Keyframes**

```python
def fcurves(ob):
    ad=ob.animation_data
    if bpy.app.version>=(4,4,0):
        from bpy_extras import anim_utils
        return anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot).fcurves
    return ad.action.fcurves
if ob.rotation_mode=='QUATERNION': ob.rotation_mode='XYZ'          # glTF imports
fc=fcurves(ob).find("location",index=2); k=fc.keyframe_points[0]
k.interpolation='BOUNCE'; k.easing='EASE_OUT'; fc.update()
k0,k1=fc.keyframe_points[0],fc.keyframe_points[1]                  # re-fetch after update()
# CSS-style cubic-bezier(x1,y1,x2,y2):
k0.interpolation='BEZIER'; k0.handle_right_type=k1.handle_left_type='FREE'
dt,dv=k1.co.x-k0.co.x,k1.co.y-k0.co.y
k0.handle_right=(k0.co.x+x1*dt,k0.co.y+y1*dv); k1.handle_left=(k0.co.x+x2*dt,k0.co.y+y2*dv)
```

**glTF bounding box and pivot**
- Collect new objects as `new=[o for o in bpy.data.objects if o not in before]`, call `view_layer.update()`, then take the AABB of `o.matrix_world @ Vector(c) for c in o.bound_box`.
- Parent the root objects to an empty placed at the bottom centre, keeping `matrix_parent_inverse`.

**Launching Blender:** `[blender_exe,"-b","--factory-startup","-noaudio","--python-exit-code","3","-P",worker,"--",job_json]`, with `creationflags=CREATE_NO_WINDOW`. In the worker, read arguments as `argv=sys.argv[sys.argv.index("--")+1:]`.

## 5. Risks and limitations

**Physics of the ratio**
- SC is a ratio, so reflections and bounce light are only right when the empty-proxy render ≈ photo. Calibrate lights and albedo first; light groups can supply the per-light images for this.
- Inaccurate proxies (contact planes, wall depth) cause floating or clipped shadows, and CG hidden behind a catcher that is placed wrong.

**Exactness and noise**
- Gating with a threshold cuts shadows weaker than about 0.5%. These are invisible in 8-bit anyway.
- With the denoiser, residual blotches may "crawl" next to moving objects. A fixed seed reduces this.
- The gate thresholds (t0=0.005, t1=0.02) need tuning per scene.

**Not run here**
- No GPU or OptiX code was executed.
- All per-frame timings are UNVERIFIED.
- The CUDA driver minimum (as opposed to OptiX's 575) is UNVERIFIED.
- The extra VRAM used by OIDN on the GPU is UNVERIFIED.

**Licensing:** the GPL boundary needs legal review.

**Issues I found in the current `/home/user/softwareEditor/mobilia/render/blender_worker.py`**
1. Lines 130–134 combine `sensor_fit="HORIZONTAL"` with `/max(W,H)` for the shift. This is wrong for portrait photos: divide by W instead, or switch to `AUTO` fit. The `−(cx−W/2)` term is also a half-pixel off if the intrinsics use the OpenCV convention.
2. Border rendering is used, so the code must set SC to 1 outside the border.
3. Parts imported from glTF are in QUATERNION mode, so the existing `rotation_euler` keys do nothing.
4. If the choreography code edits `KeyframePoint`s, it must fetch them again after each `update()`.
5. The worker imports `mobilia.render.choreography` into the bpy process. Under the GPL that module becomes part of the combined program: license it GPL, or pass only data.

**Reproducible tests:** the test scripts are in `/tmp/claude-0/-home-user/c09d70e1-5a72-5f0b-811a-6fd083338d9a/scratchpad/bt/`:
- `t_camera.py`, `t_catcher.py`, `t_border.py`, `t_lg.py`, `t_api3.py`, `t_gltf2.py`, `t_vt.py`, `t_interleave.py`
- Results: `report_sc3.json`, `report_sc4.json`, `report_sc5.json`

The scratchpad can be wiped, so copy anything you want to keep.

## 6. Sources

- Manual (5.3):
  - https://docs.blender.org/manual/en/latest/render/cycles/object_settings/object_data.html
  - https://docs.blender.org/manual/en/latest/render/layers/passes.html
  - https://docs.blender.org/manual/en/latest/scene_layout/object/properties/visibility.html
  - https://docs.blender.org/manual/en/latest/render/cycles/render_settings/film.html
  - https://docs.blender.org/manual/en/latest/render/cycles/render_settings/sampling.html
  - https://docs.blender.org/manual/en/latest/render/cycles/render_settings/performance.html
  - https://docs.blender.org/manual/en/latest/render/cycles/gpu_rendering.html
  - https://docs.blender.org/manual/en/latest/advanced/command_line/arguments.html
  - https://docs.blender.org/manual/en/latest/getting_started/about/license.html
- Release notes:
  - https://developer.blender.org/docs/release_notes/3.0/cycles/
  - https://developer.blender.org/docs/release_notes/3.2/cycles/
  - https://developer.blender.org/docs/release_notes/4.0/cycles/
  - https://developer.blender.org/docs/release_notes/5.0/python_api/
  - https://developer.blender.org/docs/release_notes/5.0/rendering/
  - https://developer.blender.org/docs/release_notes/5.0/cycles/
  - https://developer.blender.org/docs/release_notes/5.1/cycles/
  - https://developer.blender.org/docs/release_notes/5.2/cycles/
  - https://developer.blender.org/docs/release_notes/5.3/cycles/
  - https://www.blender.org/releases/5-2/ (via search result)
- bpy as a module: https://docs.blender.org/api/current/info_advanced_blender_as_bpy.html
- Source (github.com/blender/blender, main / blender-v4.2…v5.3-release branches):
  - `intern/cycles/kernel/film/read.h`
  - `kernel/film/light_passes.h`
  - `kernel/integrator/{intersect_closest.h,shadow_catcher.h,shade_surface.h,path_state.h}`
  - `kernel/light/light.h`
  - `kernel/types.h`
  - `scene/{film.cpp,light.cpp,pass.cpp}`
  - `session/buffers.cpp`
  - `blender/{sync.cpp,light.cpp,object.cpp,camera.cpp}`
  - `blender/addon/properties.py`
  - `source/blender/blenkernel/intern/{object.cc,camera.cc}`
  - `makesrna/intern/{rna_object.cc,rna_scene.cc}`
  - `imbuf/intern/openexr/openexr_api.cpp`
  - `scripts/modules/bpy_extras/anim_utils.py`
- PyPI and downloads:
  - https://pypi.org/project/bpy/
  - https://pypi.org/simple/bpy/
  - https://download.blender.org/release/Blender5.2/
  - https://download.blender.org/release/Blender5.3/
- Benchmarks: https://opendata.blender.org/benchmarks/query/?compute_type=OPTIX&blender_version=5.2.0&group_by=device_name (and the CUDA/CPU and 4.5.0 variants)
- Licensing:
  - https://www.gnu.org/licenses/gpl-faq.en.html (MereAggregation, GPLPlugins)
  - https://www.blender.org/support/faq/ (UNVERIFIED: blocked to fetch)
- EXR and image libraries:
  - https://openexr.com/en/latest/python.html
  - https://pypi.org/project/OpenEXR/
  - https://pypi.org/project/OpenImageIO/
  - https://github.com/AcademySoftwareFoundation/openexr/blob/main/LICENSE.md