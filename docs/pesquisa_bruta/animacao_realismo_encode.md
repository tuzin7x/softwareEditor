# Build-up animation, photoreal integration and MP4 encoding: research report (2026-11-15)

Scope: topic 3 of the workflow. The evidence is primary docs and source code, plus **tests I ran in this container** (FFmpeg 6.1.1, libx264/libx265, numpy) on the repo's real fixture `tests/fixtures/photos/sala_vazia_stafford.jpg` (1920x1440). NVENC could not be run because there is no GPU. Firecrawl had no credits, so I used WebSearch, WebFetch and curl. I read the repo's `mobilia/render/choreography.py`, `mobilia/composite/compositor.py` and `mobilia/video/encoder.py` so the advice targets the existing code. Numbers labelled "measured" come from my own tests.

## 1. Findings

### 1.1 What makes a build-up look premium rather than cheap
- **Animation principles that apply to furniture [S1]:**
  - staging: one readable action at a time;
  - overlapping action;
  - slow in / slow out;
  - timing conveys weight ("lighter objects react faster");
  - squash and stretch at constant volume;
  - arcs.
- **Apple HIG [S2]:** "Don't add motion for the sake of adding motion" and "Aim for brevity and precision".
- **Spring parameters from the main design systems** (mass 1, w0 = sqrt(k)):
  - Material 3 tokens, from AOSP source [S5]:
    - Expressive spatial fast/default/slow: zeta 0.6/0.8/0.8, k 800/380/200 (4.5/3.1/2.25 Hz).
    - Effects: zeta 1.0. Standard spatial: zeta 0.9.
    - Easings: emphasized (0.2,0,0,1), emphasized-decelerate (0.05,0.7,0.1,1), emphasized-accelerate (0.3,0,0.8,0.15).
  - Android SpringForce [S6]: zeta 0.2/0.5/0.75/1.0; k 10000/1500/200/50.
  - Apple [S3,S4]: springs are set by (duration, bounce). Bounce 0 is critically damped and 1 is undamped, so zeta = 1 - bounce and w0 = 2pi/duration. This reading matches the SwiftUI doc; the WWDC slide's formula is typeset ambiguously. Apple advises being "cautious about values higher than around 0.4".
  - My calculation: M3 expressive-default overshoots 1.5 % and settles within 2 % in 0.19 s. UI springs are 1.5 to 2 times too fast for room-scale objects. Heavy pieces should overshoot no more than 3 %.
- **Gravity:**
  - True free fall from 0.10/0.20/0.40 m lasts 0.14/0.20/0.29 s.
  - The repo's `motion_drop` falls 0.48 m in 0.63 s (0.24 g) and stops at 1.5 m/s, so it reads "floaty, then slam".
- **Motion blur:**
  - A 180° shutter is the cinema norm; narrower looks "stuttered", wider looks "smeared" [S7].
  - Blender 5.3: `motion_blur_shutter` defaults to 0.5 frame (= 180°) with position CENTER. Per object there are `cycles.use_deform_motion` and `cycles.motion_steps` (2^(n-1) time steps). Lights are not motion-blurred [S8,S9,S11].
- **Real-estate practice:**
  - Today's "AI staging video" tools (Higgsfield start/end frames, Magic Hour) regenerate the whole room with video diffusion, according to vendor pages. That is incompatible with the immutable plate [S38].
  - Several MLSs require an on-image "Virtually Staged" label (MLS Now; California AB 723 per vendor summaries). The primary pages were blocked, so this is UNVERIFIED [S38].

### 1.2 Photoreal integration
- **Method lineage:**
  - Differential rendering [S13] and single-photo insertion with a room proxy [S14].
  - Blender's Shadow Catcher pass holds "shadows and light which is to be multiplied into backdrop". With the pass enabled, "all indirect light interactions are captured" [S9,S11].
- **Measured noise on both fixtures** (`nlf_test.py`, flattest 20 % of pixels):
  - Noise level is 0.57 to 1.08 DN per channel.
  - It is **signal-dependent**: in the dk photo it falls from 1.06 DN in mid-tones to 0.57 DN in highlights.
  - It is **spatially correlated**: lag-1 rho 0.41 horizontal and 0.31 vertical (stafford).
  - It is **cross-channel correlated**: rho 0.60 (stafford) and 0.97 (dk).
  - JPEG 8x8 block-boundary energy is 1.8 times the block interior.
  - The repo's grain gets the spatial blur right (sigma 0.5 px gives rho ≈ 0.37). It uses one constant sigma and independent RGB channels, so it is too chromatic.
- **Estimators to use:**
  - noise level function [S15] and the Poisson-Gaussian model [S16];
  - wavelet MAD, `skimage.restoration.estimate_sigma` (BSD-3) [S17];
  - AR grain pattern plus a piecewise-linear intensity LUT, as in AV1 film grain [S18];
  - blur from the gradient ratio R of re-blurred edges, sigma = sigma0/sqrt(R²-1) [S19];
  - lateral chromatic aberration [S20] and vignetting [S21];
  - veiling glare, which keeps photo blacks lifted [S22];
  - guided-filter matte refinement [S23].
- **Grain must be static:** the plate is a still photo, so CG grain must be frozen in screen space. The repo already does this.

### 1.3 Encoding (measured)
The test is a synthetic build-up of 6 dropping objects with contact shadows over the real plate, 12 s at 30 fps, using libx264 `preset slow` unless noted.
- **BG** = pixels never touched by objects or shadows, with a 16 px margin (61.7 % of the frame).
- **Pulse** = the worst frame-to-frame share of BG pixels whose decoded value changes.

| Variant | Size | BG PSNR (dB) | Worst pulse | Frames with >1 % BG change |
|---|---|---|---|---|
| A, repo today: CRF 14, GOP 4 s | 3.99 MB | 47.95-49.08 | 57.9 % | 5 |
| B, GOP 0.5 s (YouTube's "half the frame rate") | 22.18 MB | 46.79-47.93 | 74.9 % | 18 |
| H, tune film, CRF 16, GOP 2 s | 7.17 MB | 47.31-48.50 | 61.5 % | 12 |
| **C, CRF 14, single IDR, scenecut=0** | **1.54 MB** | **47.95-48.03** | **2.1 %** | 3 |
| D, as C at CRF 17 | 1.10 MB | 46.82-46.86 | 1.7 % | 5 |
| E, as C + tune film, CRF 16 | 1.57 MB | 47.31-47.41 | 2.3 % | 7 |
| F, as C + tune grain, CRF 16 | 1.61 MB | 46.54-46.62 | 8.0 % | 35 |
| G, as C + tune stillimage, CRF 16 | 1.71 MB | 47.21-47.26 | 4.2 % | 8 |
| I, x265 CRF 16, single IDR (45 s to encode) | 1.06 MB | 46.87-47.08 | 1.1 % | 6 |
| J, x265 CRF 16, GOP 4 s | 2.54 MB | 46.96-47.67 | 72.9 % | 37 |

- **Background "breathing":**
  - Every IDR re-quantizes the whole photo. 58 to 75 % of plate pixels change value at the same moment (mean change 0.5 to 0.75 DN), which shows as a periodic breath.
  - One IDR per clip removes this and makes the file 2.6 times smaller.
  - `tune grain` keeps re-coding static blocks, so avoid it. Its settings, checked in the x264 source [S24], are deadzones 6, qcomp 0.8 and no DCT decimation.
  - `stillimage` (deblock -3, psy-rd 2.0:0.7, aq 1.2) gives no benefit.
  - x265 at CRF 16 matches x264 at CRF 17 (+0.1 dB, 4 % smaller) but takes 2.2 times as long to encode.
- **Colour** (`color_test.py`):
  - Using `-pix_fmt yuv420p` with BT.709 tags but no conversion matrix makes FFmpeg convert with BT.601. Decoded as BT.709, a ColorChecker shows dE00 mean 2.25 and max 5.99 (cyan).
  - Explicit `scale=...:out_color_matrix=bt709` [S25] or `zscale=m=709` brings this to mean 0.21 and max 0.61.
  - FFmpeg writes the `colr nclx 1-1-1` box automatically. The ASWF guidelines reach the same conclusion [S33].
- **Bit-exactness:**
  - Lossy delivery is never bit-exact. At CRF 14, only 12 % of untouched pixels are exact; mean error 0.68 DN, p99 3 DN.
  - Lossless YUV is not bit-exact either: 4:4:4 at qp 0 has max error 2 DN, and 4:2:0 at qp 0 has max error 14 DN on saturated edges.
  - `libx264rgb -qp 0` round-trips **bit-exact** (checked with cmp; High 4:4:4 Predictive profile).
- **H.264 levels:**
  - With `preset slow` (5 reference frames), x264 labels 1080p as Level 5.0. Adding `-level 4.1` makes x264 drop to 4 reference frames (verified).
  - 1920x1440 and 1920x1280 exceed Level 4.2's 8,704-macroblock frame limit, so they need Level 5.0. 2880x2160 needs 5.1, and 4032x3024 needs Level 6 [S30].
  - NVENC H.264 is limited to 4096x4096 [S29]; not confirmed against NVIDIA's own docs (UNVERIFIED).
- **Container:**
  - FFmpeg's MP4 output with B-frames carries an `elst` edit list (verified). YouTube and Instagram both ask for "no edit lists" [S31,S32].
  - `-use_editlist 0` removes it; the first frame's timestamp becomes 0.067 s.
- **Platform requirements:**
  - Instagram [S32]: H.264 or HEVC, progressive, closed GOP, 4:2:0, 23-60 fps, at most 1920 columns, VBR up to 25 Mbps, files up to 300 MB.
  - YouTube [S31]: H.264 High, closed GOP, 4:2:0, and BT.709 (H.273 value 1) for primaries, transfer and matrix.
  - WhatsApp: no official spec found (UNVERIFIED).
  - Windows: the built-in players need the paid "HEVC Video Extensions" to play HEVC [S34, secondary source].
- **Transfer tag:** ASWF browser tests find `-color_trc 13` (sRGB) the most reliable. Their tests call bt709 "questionable" in macOS Safari, and iOS ignores the tags [S33].
- **Current versions (Nov 2026):**
  - FFmpeg 9.0.2, released 2026-09-18. Version 9.0 removed deprecated NVENC options [S26].
  - NVENC options in FFmpeg n9.0.2: presets p1-p7; tune hq/ll/ull/lossless, plus uhq for HEVC/AV1; `-multipass`, `-b_ref_mode`, spatial and temporal AQ [S27].
  - Blender 5.2 LTS, released 2026-07-14; the online manual is at 5.3 [S12].

## 2. Comparison table

| Option | Quality / size | Static plate | Compatibility | Code license | Verdict |
|---|---|---|---|---|---|
| libx264, single IDR, CRF 14 | best measured | pulse ≤ 2 % | universal | GPL-2.0+ (commercial licence exists) | **Primary** |
| libx264, GOP 2-4 s | 2.6 to 4.7 times larger | 58-62 % pulses | universal | GPL | replace |
| h264_nvenc p7/hq, one IDR, no-scenecut | likely lower than x264 at equal bitrate (UNVERIFIED) | expected similar | universal | MIT headers; LGPL FFmpeg is enough | **Fallback** |
| libx265, single IDR | ≈ x264 CRF 17 at 4 % smaller, 2.2 times slower | pulse 1 % | needs paid Windows extension | GPL-2.0 (commercial licence exists) | optional |
| libx264rgb qp 0, or FFV1 | lossless | bit-exact | poor (RGB 4:4:4) | GPL / LGPL | verification master |

## 3. Recommendation
**Primary**
1. **Animation:** one style, "place and settle":
   - pieces enter in tiers;
   - each starts when the previous is 45-60 % done;
   - 180° motion blur at 30 fps;
   - a 13 s clip with a 3.6 s final hold (sections 4.1 and 4.2).
2. **Integration:**
   - flare-aware shadows;
   - grain scaled by the noise level function, correlated across channels, frozen in screen space;
   - blur matched to the measured photo blur;
   - white balance taken from neutral walls;
   - one ICC-to-sRGB conversion at import (section 4.3).
3. **Encoding:** libx264 with a single IDR, CRF 14, a 24 Mbps VBV cap, explicit BT.709 conversion and no edit list. Also write a lossless RGB master for verification (section 4.4).

Why x264 on the CPU:
- It had the best measured stability and size.
- It decodes everywhere.
- It needs about 20 s per 12 s clip here, against minutes of Cycles rendering, so GPU encoding buys nothing.

**Fallback.** Use h264_nvenc (p7, hq, VBR with CQ, no-scenecut, one IDR) when x264 is missing or GPL is not acceptable. Calibrate CQ on the target GPU by measuring PSNR against the masters. Offer HEVC only as an optional "small file" export.

## 4. Implementation notes

### 4.1 Timing spec: living room, 13.0 s at 30 fps (390 frames)
- **Order:** rug, then the sofa (hero), then other large pieces back to front, then medium pieces, then vertical items (wall art, curtains, tall plants), then soft goods, then tabletop decor.
- **Pacing:** gaps shrink per tier (about 0.7, 0.5, 0.3, 0.1 s), so the build accelerates toward the end. Never land two hero pieces within 0.3 s of each other.

| Start s (frame) | Item | Recipe (see 4.2) | Duration s |
|---|---|---|---|
| 0.00 (0) | empty plate | hold | 1.00 |
| 1.00 (30) | rug | unroll | 1.30 |
| 1.70 (51) | sofa frame | heavy | 1.10 |
| 2.25 (68) | 3 seat cushions, 0.07 s apart | soft | 0.45 |
| 2.55 (77) | back cushions | tip in from 12° to 0, zeta 0.6, 3.5 Hz | 0.50 |
| 2.70 (81) | TV console / sideboard | heavy | 1.00 |
| 3.35 (101) | armchair | medium | 0.95 |
| 3.95 (119) | coffee table | assembly: base, then top 0.15 s later | 0.90 |
| 4.50 (135) | side table | medium | 0.75 |
| 4.95 (149) | floor lamp (stays off) | assembly: base, pole, shade | 0.90 |
| 5.55 (167) | wall art A | hang | 0.85 |
| 5.85 (176) | wall art B | hang, swing mirrored | 0.85 |
| 6.25 (188) | large plant | grow | 1.20 |
| 6.95 (209) | 3 pillows, 0.12 s apart | soft | 0.45 |
| 7.40 (222) | throw | pre-baked cloth drape | 0.80 |
| 7.85 (236) | 3 books, 0.06 s apart | small rigid | 0.35 |
| 8.15 (245) | vase and tray, 0.10 s apart | small rigid | 0.35 |
| 8.45 (254) | small plant / candles | grow, c1 = 1.0 | 0.60 |
| ~9.40 (282) | everything settled within 2 % | | |
| 9.40-13.00 | final hold | | 3.60 |

With fewer items, shorten the gaps but keep the final hold at 3 s or more. With more items, extend the tabletop tier, and cap the build at 10 s and the clip at 15 s.

### 4.2 Motion recipes (distances in metres)
- **Spawn:** fade in over 3-4 frames in mid-air, never longer. Never scale rigid furniture up from zero.
- **Heavy and medium pieces, "place and release":**
  1. Start 0.30-0.40 m above the final position.
  2. Descend with ease-out-cubic (or M3 emphasized-decelerate) over 0.45-0.55 s, to 2-4 cm above the floor.
  3. Hover for 2 frames (anticipation).
  4. Release under true gravity (64-90 ms of fall).
  5. On contact:
     - squash with volume preserved: sz = 1 - s and sx = sy = 1/sqrt(1 - s), with s = 1-1.5 % for heavy and 2-3 % for medium pieces;
     - rock 0.6-1.0° about the leading edge;
     - spring zeta 0.75 at 2.2 Hz (heavy) or 0.65 at 2.8 Hz (medium);
     - never rise back above the floor.
- **Small rigid items:** true-gravity drop of 0.10-0.15 m, restitution 0.25 (one hop of 6-9 mm), zeta 0.55 at 3.5 Hz, random yaw of 3-5°.
- **Soft items:** true-gravity drop of 0.18-0.22 m, squash 8-12 %, zeta 0.45 at 4.5 Hz, no rebound, random final tilt of 2-4°.
- **Assembly parts:**
  - start 0.10-0.25 m out along the assembly axis, 60-120 ms apart;
  - move in with ease-out-quint over 0.30-0.45 s, after a 3-5 % pull-back;
  - snap with zeta 0.85 at 5 Hz.
- **Rug:**
  - geometry-nodes unroll driven by one animated value;
  - roll radius r = sqrt(r0² - L(t)·d/π), where d is the rug thickness;
  - emphasized easing;
  - a 3 mm edge flop at zeta 0.7, 4 Hz;
  - deformation blur on.
- **Plants:** grow the foliage from scale 0.15 to 1 with easeOutBack, c1 = 1.25 (5.7 % overshoot), leaves 30-50 ms apart. Then sway 2° at zeta 0.3, 1.4 Hz.
- **Wall art:**
  - slide 0.12-0.18 m down the wall with emphasized-decelerate over 0.5-0.6 s;
  - then swing as a pendulum about the hook: zeta 0.25, f = sqrt(g/L)/2π (1.3 Hz for L = 0.15 m), 1-2°.
- **Avoid:**
  - fades longer than 0.15 s;
  - spins over 15°;
  - particles or glows;
  - pieces sliding in from the frame edge;
  - identical durations and easings for every piece;
  - heavy pieces bouncing more than 2 cm;
  - camera shake;
  - lamps turning on, which relights the plate and breaks the rule;
  - shadows animated separately from object height.

### 4.3 Blender and compositing
Property names are checked against the Blender 5.3 docs and the Cycles source [S8-S11].
```python
r = sc.render; r.use_motion_blur = True; r.motion_blur_shutter = 0.5; r.motion_blur_position = 'CENTER'
r.film_transparent = True; sc.cycles.seed = 0; sc.cycles.use_animated_seed = False   # same render noise every frame
vl.cycles.use_pass_shadow_catcher = True; room_proxy.is_shadow_catcher = True
ob.cycles.use_deform_motion = deforming; ob.cycles.motion_steps = 3 if deforming else 1
```

**Plate statistics, computed once at import** (rules of thumb, not standards):
- **Flare floor b:** the 0.2-0.5th percentile of linear luminance outside the windows, capped at 0.02.
- **White balance:** median linear RGB of the wall and ceiling masks. This assumes neutral paint.
- **Noise level function sigma_c(I):** 8-16 intensity bins of residual MAD on flat pixels, stored as a monotone LUT.
- **Grain structure:** the 3x3 residual covariance, and the mean |FFT|² of 64x64 flat patches.
- **Extra blur:**
  - measure sigma_photo with the gradient-ratio estimator at edges near the layout;
  - measure sigma_render with the same estimator on a rendered edge;
  - sigma_extra = sqrt(max(0, sigma_photo² - sigma_render²)).

**Per frame:**
```
S    = deadband(clip(shadow_pass, 0, 1.15), eps=0.005)
bg   = b + (P_lin - b) * S                      # flare is additive, so it is not shadowed
C, A = blur(C*WB, sigma_extra), blur(A, sigma_extra)   # same kernel on RGB and alpha
C   += b * A                                     # same veiling floor on the CG
out  = srgb(C + (1 - A) * bg) + A * sigma_c(out) * G_frozen ; round
copy plate bytes where A == 0 and S == 1 ; assert
```

**Notes:**
- **G_frozen:** white noise shaped by sqrt(PSD), mixed across channels with the Cholesky factor of the measured covariance. Generate it once with a fixed seed.
- **Apply only if detected in the photo:**
  - vignetting V(r) on the CG colour, never on S;
  - lateral chromatic aberration;
  - an unsharp mask when the photo's edge overshoot exceeds 3 %.
- **Import:**
  1. Apply `ImageOps.exif_transpose`.
  2. Convert ICC to sRGB once with Pillow `ImageCms`.
  3. Downscale once with Lanczos.
  4. The result becomes the immutable plate.

### 4.4 FFmpeg
Options checked in 6.1.1 and in the n9.0.2 source.
```
ffmpeg -f rawvideo -pix_fmt rgb24 -s WxH -r 30 -i - \
 -vf "scale=in_range=pc:out_range=tv:out_color_matrix=bt709:flags=accurate_rnd+full_chroma_int,format=yuv420p" \
 -c:v libx264 -preset slow -crf 14 -profile:v high -g N \
 -x264-params "scenecut=0:aq-mode=3:aq-strength=0.8:deblock=-1,-1" \
 -maxrate 24M -bufsize 48M \
 -color_primaries bt709 -color_trc bt709 -colorspace bt709 -color_range tv \
 -movflags +faststart -use_editlist 0 -an out.mp4        # N = total frames (one IDR)
```
- **Resolution:**
  - Social: long edge at most 1920 px. Add `-level 4.1` for 16:9 1080p; 4:3 and 3:2 at 1920 px wide need Level 5.0 because of their frame size.
  - 4K master: 2880x2160, 3240x2160 or 3840x2160, all Level 5.1 at up to 30 fps.
  - Nothing above 4K in H.264.
- **Transfer tag:** an Apple/web-only export can tag `-color_trc iec61966-2-1` [S33]. Keep bt709 for YouTube.
- **Verification master:** `-c:v libx264rgb -qp 0 -preset slow -g N` (bit-exact), or FFV1 in MKV.
- **NVENC fallback (untested here):**
  - `-c:v h264_nvenc -preset p7 -tune hq -rc vbr -cq 15 -b:v 0 -maxrate 24M -bufsize 48M -profile:v high -bf 3 -b_ref_mode middle -rc-lookahead 20 -spatial-aq 1 -temporal-aq 1 -multipass fullres -g N -no-scenecut 1`, with the same filter, tags and flags.
  - NVIDIA's guide requires lookahead ≤ 31 minus the number of B-frames [S28].
  - For HEVC use `-tune uhq`, falling back to `hq`, plus `-tag:v hvc1`.
  - Replace the repo's `cq = crf + 2` with per-GPU calibration by PSNR.
- **Frame rate:** default to 30 fps. It is inside Instagram's 23-60 range, matches 60 Hz screens, and costs half the render time of 60 fps. 24 fps judders on 60 Hz screens.

## 5. Risks and limitations
- **Bit-exactness:** "bit-exact" holds for the RGB masters and the lossless file, never for H.264/H.265 delivery (measured p99 error 3 DN). Say so in the UI.
- **Test content:** the breathing test used synthetic objects. Real CG adds bits to every frame, so keep the VBV cap and re-run the test on real renders.
- **Platforms:**
  - Instagram, WhatsApp and YouTube re-encode with their own GOPs.
  - YouTube suggests a GOP of half the frame rate; one IDR still satisfies its "closed GOP" requirement.
- **Licensing:**
  - libx264 and libx265 make the FFmpeg build GPL. Shipping ffmpeg.exe as a separate process is commonly treated as aggregation, but it still requires GPL compliance. Commercial x264/x265 licences exist [S35].
  - Patent-pool duties for distributing an H.264/HEVC encoder need legal review (UNVERIFIED).
  - nv-codec-headers are MIT [S36]. Blender is GPL-2.0-or-later; Cycles is Apache-2.0 [S39].
  - Avoid `bm3d`, which is "Free for non-commercial use" [S37].
  - PCT-Net (harmonization) code is MPL-2.0, but the terms for its weights are UNVERIFIED [S40]. Harmonizer ships with no licence.
- **Disclosure labels:** these conflict with the pixel rule. Offer them as an opt-in overlay applied after verification.
- **Animation values:** the spring and timing values are design recommendations derived from the cited systems, not audience data. Validate them with viewing tests.
- **Not verified here:**
  - NVENC quality and UHQ availability per GPU;
  - WhatsApp and Windows player behaviour.
  - Blocked pages: the FFmpeg trac wiki, MLS Now, and blender.org's licence page.
- **Reproduction:**
  - Scripts and results are in `/tmp/claude-0/-home-user/c09d70e1-5a72-5f0b-811a-6fd083338d9a/scratchpad/enc/`: `make_frames.py` (regenerates the 3 GB raw master I deleted), `run_variants_fast.py`, `color_test.py`, `nlf_test.py`, `results_1794783975.json`, `results_1794784555.json`, and the test MP4s.
  - That folder is scratch space; copy into the repo anything worth keeping.

## 6. Sources
- S1 Twelve basic principles of animation (Thomas & Johnston 1981): https://en.wikipedia.org/wiki/Twelve_basic_principles_of_animation
- S2 Apple HIG, Motion: https://developer.apple.com/design/human-interface-guidelines/motion
- S3 WWDC23 "Animate with springs": https://developer.apple.com/videos/play/wwdc2023/10158/
- S4 SwiftUI `Spring(duration:bounce:)`: https://developer.apple.com/documentation/swiftui/spring/init(duration:bounce:)
- S5 AOSP Material 3 motion tokens: https://raw.githubusercontent.com/androidx/androidx/androidx-main/compose/material3/material3/src/commonMain/kotlin/androidx/compose/material3/tokens/ExpressiveMotionTokens.kt (also StandardMotionTokens.kt and MotionTokens.kt)
- S6 Android SpringForce: https://developer.android.com/develop/ui/views/animations/spring-animation and androidx `SpringForce.java`
- S7 RED, shutter angle: https://www.reddigitalcinema.com/red-101/shutter-angle-tutorial
- S8 Blender 5.3 manual, Motion Blur: https://docs.blender.org/manual/en/latest/render/cycles/render_settings/motion_blur.html
- S9 Blender manual, Object settings and Passes: https://docs.blender.org/manual/en/latest/render/cycles/object_settings/object_data.html , https://docs.blender.org/manual/en/latest/render/layers/passes.html
- S10 Blender manual, Film: https://docs.blender.org/manual/en/latest/render/cycles/render_settings/film.html
- S11 Blender API and Cycles properties: https://docs.blender.org/api/current/bpy.types.RenderSettings.html , https://docs.blender.org/api/current/bpy.types.Object.html , https://projects.blender.org/blender/blender/raw/branch/main/intern/cycles/blender/addon/properties.py
- S12 Blender 5.2 LTS: https://www.blender.org/releases/5-2/
- S13 Debevec 1998: https://www.pauldebevec.com/Research/IBL/debevec-siggraph98.pdf
- S14 Karsch et al. 2011: https://arxiv.org/abs/1912.11565
- S15 Liu et al. 2008 (noise level function): https://people.csail.mit.edu/celiu/pdfs/denoise_TPAMI.pdf
- S16 Foi et al. 2008: https://webpages.tuni.fi/foi/papers/Foi-PoissonianGaussianClippedRaw-2007-IEEE_TIP.pdf
- S17 scikit-image `estimate_sigma`: https://scikit-image.org/docs/stable/api/skimage.restoration.html
- S18 Norkin & Birkbeck 2018, AV1 film grain: https://norkin.org/pdf/DCC_2018_AV1_film_grain.pdf
- S19 Zhuo & Sim 2011: https://doi.org/10.1016/j.patcog.2011.03.009
- S20 Kang 2007: https://doi.org/10.1109/CVPR.2007.383214
- S21 Zheng et al. 2009: https://doi.org/10.1109/TPAMI.2008.263
- S22 McCann & Rizzi 2007: https://doi.org/10.1117/12.703042
- S23 He et al. 2013, guided filter: https://people.csail.mit.edu/kaiming/publications/pami12guidedfilter.pdf
- S24 x264 tunes, `common/base.c`: https://code.videolan.org/videolan/x264/-/raw/master/common/base.c
- S25 FFmpeg scale filter and scaler docs: https://ffmpeg.org/ffmpeg-filters.html#scale , https://ffmpeg.org/ffmpeg-scaler.html
- S26 FFmpeg releases and Changelog: https://ffmpeg.org/download.html , https://raw.githubusercontent.com/FFmpeg/FFmpeg/master/Changelog
- S27 FFmpeg n9.0.2 NVENC options: https://raw.githubusercontent.com/FFmpeg/FFmpeg/n9.0.2/libavcodec/nvenc_h264.c (and nvenc_hevc.c)
- S28 NVIDIA NVENC Programming Guide 13.1: https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvenc-video-encoder-api-prog-guide/index.html
- S29 NVENC (Wikipedia): https://en.wikipedia.org/wiki/NVENC
- S30 H.264 levels: https://en.wikipedia.org/wiki/Advanced_Video_Coding#Levels
- S31 YouTube recommended upload encoding settings: https://support.google.com/youtube/answer/1722171
- S32 Instagram Graph API media specs: https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-user/media
- S33 ASWF Encoding Guidelines and browser tests: https://academysoftwarefoundation.github.io/EncodingGuidelines/ColorPreservation.html , https://richardssam.github.io/ffmpeg-tests/compare.html
- S34 HEVC on Windows 11 (secondary source): https://windowslatest.com/2025/07/16/can-you-get-hevc-codec-for-free-on-windows-11
- S35 x265 and x264 licensing: https://x265.readthedocs.io/en/stable/introduction.html , https://mailman.videolan.org/pipermail/x264-devel/2010-July/007508.html
- S36 nv-codec-headers (MIT): https://raw.githubusercontent.com/FFmpeg/nv-codec-headers/master/include/ffnvcodec/nvEncodeAPI.h
- S37 bm3d on PyPI: https://pypi.org/project/bm3d/
- S38 Real-estate practice: https://fotello.co/blog/virtualstaging , https://magichour.ai/use-cases/virtual-staging , https://support.mlsnow.com/hc/en-us/articles/47635315367067-MLS-Now-Rules-Section-3-Images-Media-Virtual-Staging (returned 403)
- S39 Blender COPYING and SPDX headers: https://projects.blender.org/blender/blender/raw/branch/main/COPYING
- S40 PCT-Net LICENSE: https://raw.githubusercontent.com/rakutentech/PCT-Net-Image-Harmonization/main/LICENSE
