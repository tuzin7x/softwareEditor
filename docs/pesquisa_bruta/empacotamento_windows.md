# Windows packaging for a local AI + Blender + FFmpeg Python pipeline: research report (2026-11-15)

**Conventions.** [Sn] points to the Sources list. **[M]** marks something I measured in this session. Measurements came from HTTP range reads of wheel and zip central directories, and from checking the PE security directory. That check only shows an Authenticode blob is present; it does not validate the certificate chain. uv 0.13.8 was run on Linux. **UNVERIFIED** means no primary source confirmed it.

## 1. Findings

### 1.1 PyTorch, CUDA and drivers
- **torch 2.15.0** was released 2026-10-28 and supports Python 3.11–3.15 [S1][S2].
  - Stable CUDA is **13.2** (cuDNN 9.26); CUDA 13.4 is experimental.
  - Release wheels ship SASS only, for sm_75/80/86/90/100/120, with no PTX [S2].
  - RTX 40 (sm_89) runs the sm_86 SASS through same-major binary compatibility. This is a general CUDA rule; S2 does not state it.
  - 2.15.1 is planned for Nov 2026 and 2.16 for 22 Dec 2026 [S2].
- **Wheel index availability** [S3][M]:
  - Windows wheels for 2.15.0 exist only on `whl/cu132` and `whl/cu134`.
  - `cu130` and `cu126` stop at 2.14.1; `cu128` stops at 2.11.0.
  - PyPI's Windows torch wheels are CPU-only (120 MB) [S1].
- **Driver branches** [S4]: CUDA 13.0→R580, 13.1→R590, **13.2→R595**, 13.3→R610, 13.4→R615, 13.5→R620.
  - Any CUDA 13.x app runs on driver **≥580** under minor-version compatibility.
  - CUDA 13 removed Maxwell, Pascal and Volta. GTX 10 cards therefore need torch ≤2.14.1+cu126; ComfyUI ships a separate cu126 build for them [S18].
- **Wheel contents** [M]:
  - torch-2.15.0+cu132 (cp313, win_amd64) is a 2.0 GiB download and installs to 2.9 GiB in 12,389 files. The largest files are torch_cuda.dll (507 MB) and cublasLt64_13.dll (435 MB).
  - For comparison: cu134 is 2.06 GiB and 2.14.1+cu126 is 2.48 GiB.
  - The wheel contains no msvcp140.dll, so the **VC++ 2015-2022 x64 runtime is required**. PyTorch's Windows FAQ documents the resulting "DLL load failed" error [S41].
  - The longest path inside the wheel is 132 characters.
- **uv 0.13.8 `--torch-backend=auto`** works only in `uv pip` [S5][S6][M].
  - On Windows it parses `nvidia-smi` and maps a driver ≥580 to cu132, then cu130, down to cpu.
  - It does not know cu134 and ignores compute capability.
  - The detected driver can be overridden with `UV_CUDA_DRIVER_VERSION`.

### 1.2 UI toolkits
- **PySide6 6.12.0** (2026-10-08) supports Python 3.10–3.15 and is LGPL-3.0-only or GPL [S7].
  - Windows wheels: Essentials is 78 MB and installs to 208 MB; Addons is 39 MB and installs to 100 MB. QtMultimedia and its FFmpeg DLLs (avcodec-63) are in Addons [M].
  - Qt Multimedia 6.12 defaults to an **FFmpeg 9.0.1 backend** on Windows; the WMF backend has been deprecated since 6.10. The docs warn that Qt licenses do not cover H.264 patent fees [S9].
  - **GPL-only modules** in Qt 6.12: Graphs, Quick 3D (and Physics), Quick Timeline, Virtual Keyboard, Lottie, HTTP Server, GRPC, MQTT, CoAP, Network Authorization, Qml Compiler, Canvas Painter, Wayland Compositor [S8].
  - The QML FluentWinUI3 style exists since Qt 6.8 and targets Windows 11 [S11].
  - The third-party PySide6-Fluent-Widgets 1.11.3 is **GPLv3** [S12].
  - LGPLv3 duties [S10]:
    - Link Qt dynamically, and let users replace and relink it (reverse engineering for this must be allowed).
    - Ship the Qt source or a written offer.
    - Include the license text and a prominent notice that Qt is used.
    - The EULA must not restrict any of these rights.
- **pywebview 6.2.1** (BSD-3) uses WinForms plus WebView2 through pythonnet. It needs the .NET 6+ Desktop Runtime or .NET Framework 4.6.2+, plus the WebView2 runtime; MSHTML has been removed [S13].
- **Tauri 2.12** (MIT/Apache) can only run Python as a frozen `externalBin` sidecar; its docs cite PyInstaller [S14][S15].
- **Electron 45.3.2** (MIT) is the shell used by Comfy Desktop and by InvokeAI's launcher [S15][S19][S20].
- H.264 `<video>` playback in WebView2: UNVERIFIED.

### 1.3 Freezers and what similar apps do
- **PyInstaller 6.22.3**:
  - Onefile mode unpacks everything to `%TEMP%\_MEIxxxxxx` on every launch [S16].
  - The CArchive cookie and table of contents use unsigned 32-bit fields (`!8sIIII64s`, `!IIIIBc`), so the payload is capped at **4 GiB** [S16][M].
  - The torch hook uses `module_collection_mode="pyz+py"` and `collect_submodules("torch")`, and collects all torch DLLs.
  - The prebuilt bootloaders trigger antivirus false positives; the docs suggest rebuilding the bootloader [S16].
- **Nuitka 4.3.5**:
  - Since 4.0 (Feb 2026) it is AGPLv3+ with a "Runtime Library Exception"; 2.x was Apache-2.0. Compiled output may stay proprietary [S17].
  - Its torch config sets `PYTORCH_JIT=0` in standalone builds unless you pass `--module-parameter=torch-disable-jit=no` [S17].
  - Onefile size limits: UNVERIFIED.
- **What similar apps do**:
  - ComfyUI portable is a 7z with embedded Python 3.13 and torch cu130 (RTX 20 and newer). At 1.93 GiB it sits just under GitHub's 2 GiB asset cap [S18][M].
  - Comfy Desktop (Electron, NSIS installer) now ships a "relocatable Python with PyTorch and GPU wheels prebuilt… no pip/uv failures" [S19].
  - InvokeAI's launcher bundles uv and builds a venv on the user's PC [S20].
  - Anki's uv-based first-run launcher from 25.07 was replaced in 26.05b1 by Briefcase; the release notes say "installation no longer requires network access" [S21].
  - PyApp 0.31 (Rust stub, Apache/MIT) and Briefcase 0.4.7 (stub exe plus embedded Python, BSD-3) package the stub-launcher approach [S22].
- **Python runtime**:
  - python.org stopped publishing 3.13 binaries at 3.13.17; 3.13.18 (2026-10-25) is source-only [S47].
  - python-build-standalone `cpython-3.13.18+20261110` for Windows is available through uv (22 MB download, 70 MB extracted) [S48][M].
  - The python.org embeddable binaries are Authenticode-signed. The python-build-standalone python.exe and python313.dll are not [M].

### 1.4 Installers and updaters
- **Inno Setup 6.7.4** (2026-10-28) [S24]:
  - Its license allows commercial use; the site *requests* a paid license.
  - 6.5.0 added `DownloadTemporaryFileWithISSigVerify` and 7-Zip `ExtractArchive`.
  - 6.5.2 removed the 2 GB limit: a single Setup.exe can reach about 4 GB, and `DiskSliceSize=max` is supported. Disk spanning is mandatory above 4.2 GB, and very large signed executables start slowly.
  - `PrivilegesRequired=lowest` gives a per-user install.
  - Version 6.7.1 comes preinstalled on the windows-2025 runner [S42].
- **NSIS** has a ceiling of about 2 GB [S25].
- **MSIX** [S26][S27]:
  - The package directory is read-only.
  - New files in AppData are virtualized per package and deleted on uninstall.
  - The Microsoft Store signs MSIX packages, and individual developer registration has been free since 2025-09.
- **Velopack 1.2.161** (MIT) [S23]:
  - Python API: `velopack.App().run()` and `UpdateManager` (`check_for_updates`, `download_updates`, `apply_updates_and_restart`).
  - Supports delta packages.
  - Works only with PyInstaller `--onedir` output.

### 1.5 Model weights, Blender, FFmpeg
- **huggingface_hub 2.4.0 with hf_xet 1.8.0** [S28]:
  - All transfers go through Xet. `hf_transfer` is deprecated; `HF_XET_HIGH_PERFORMANCE=1` is the replacement for speed.
  - `snapshot_download(repo_id, revision, local_dir, allow_patterns, max_workers=8, tqdm_class, dry_run)`.
  - Partial downloads are written as `*.incomplete` files and resumed with HTTP Range requests.
  - `HfApi().verify_repo_checksums()` returns `FolderVerification` with `mismatches`, `missing_paths` and `extra_paths`. The CLI equivalent is `hf cache verify --local-dir … --fail-on-missing-files`.
  - `HF_HUB_OFFLINE=1` makes no HTTP calls at all.
  - On Windows without Developer Mode the cache cannot use symlinks, so files are duplicated.
- **Blender**:
  - **5.2 LTS** was released 2026-07-14 and is maintained until July 2028; 5.2.4 is current. 5.3.0 came out 2026-11-10 [S30][S31].
  - The bpy 5.3.0 wheel is 340 MB, requires exactly Python 3.13, and is GPL-3.0 [S29].
  - The 5.2.4 Windows zip is 406 MB and extracts to 901 MB. Its SHA-256 is published and blender.exe is signed. It contains CUDA cubins for sm_50 through sm_120, and the OptiX kernels as PTX, which the driver compiles on the first render [S31][M].
  - OptiX needs driver ≥575 [S32].
  - Headless command line: `-b --factory-startup --offline-mode --python-exit-code N -P x.py -- --cycles-device OPTIX`. A `portable\` folder next to blender.exe keeps all config local [S32].
  - Blender's FAQ says published scripts that use its Python API must be under a GPL-compatible license [S33].
- **FFmpeg**:
  - **9.0.2** is current; 8.1.3 is still maintained [S34].
  - gyan.dev builds are GPLv3 only (9.0.2, NVENC included) [S37].
  - BtbN publishes gpl, lgpl and nonfree variants, each optionally as shared builds, from branches up to 9.0. Monthly builds are kept for 2 years. The LGPL variant drops x264 and x265 but keeps openh264, SVT-AV1, kvazaar and libvpx [S35].
  - **NVENC driver requirement in BtbN builds** [S35][S36]:
    - FFmpeg ≥9.0 uses Video Codec SDK 13.1 headers, which need **driver ≥610**.
    - FFmpeg 8.0–8.1 uses SDK 13.0, which needs driver ≥570.
  - Sizes [M]: the n8.1 lgpl zip is 168 MB (static, unsigned ffmpeg.exe of 126 MB); lgpl-shared is 80 MB.
  - FFmpeg's LGPL checklist [S34]:
    - Build without `--enable-gpl` or `--enable-nonfree`, and link dynamically.
    - Host the exact matching source on the same server as the binary.
    - Add the required notice to the About box and the EULA.
    - Remove any reverse-engineering ban, and keep the original DLL names.
  - NVENC (SDK 13.1 app note) [S38]:
    - GeForce cards are limited to 12 concurrent encode sessions per system.
    - AV1 encoding needs Ada or Blackwell; H.264 10-bit and 4:2:2 are Blackwell-only.

### 1.6 GPU/VRAM, CI and signing
- **nvidia-ml-py 13.615.71** (BSD) loads `%WINDIR%\System32\nvml.dll`. It exposes `nvmlSystemGetCudaDriverVersion_v2`, `nvmlDeviceGetCudaComputeCapability`, `nvmlDeviceGetMemoryInfo` and `nvmlDeviceGetEncoderCapacity(h, NVML_ENCODER_QUERY_H264/HEVC/AV1)` [S39][M].
- **VRAM behaviour on Windows**:
  - Drivers ≥536.40 spill into shared system RAM instead of raising out-of-memory; ≥546.01 add a per-executable "CUDA – Sysmem Fallback Policy" setting [S40].
  - `PYTORCH_ALLOC_CONF` supports `per_process_memory_fraction` and `throw_on_cudamalloc_oom`; the old name `PYTORCH_CUDA_ALLOC_CONF` is an alias [S41].
  - Windows has no CUDA IPC, and long kernels can trigger a TDR driver reset [S41].
- **GitHub Actions** [S42]:
  - `windows-latest` is Windows Server 2025 with VS 2026.
  - Public repos get 4 vCPU, 16 GB RAM and **14 GB SSD**; private repos get 2 vCPU, 8 GB and 14 GB.
  - Jobs are limited to 6 h. Artifact storage is 500 MB on Free up to 2 GB on Team, and the cache is 10 GB per repo.
  - Release assets must be **under 2 GiB each**, with at most 1000 per release and no bandwidth cap.
  - A paid GPU runner exists: Tesla T4 16 GB, Windows, 176 GB SSD.
- **SmartScreen and Smart App Control** [S43][M]:
  - EV certificates no longer bypass SmartScreen; reputation builds per file hash and per publisher.
  - Smart App Control blocks **all unsigned executable files**, not only downloaded ones, unless they have positive reputation.
  - Unsigned files in our stack: torch's c10.dll, BtbN's ffmpeg.exe and the python-build-standalone Python. Already signed: NVIDIA's CUDA DLLs, blender.exe and uv.exe.
- **Code signing options**:
  - Azure Artifact Signing costs $9.99/month for 5,000 signatures. Microsoft Q&A answers, some of them AI-generated, say public trust is limited to organisations in US/CA/EU/UK and individuals in US/CA; eligibility for Brazil is UNVERIFIED [S44].
  - Certificates issued since 2026-03-01 are valid for at most 460 days [S45].
  - jsign (Apache-2.0) signs PE, MSI and MSIX files from Linux, using Artifact Signing, DigiCert KeyLocker, SSL.com eSigner or cloud KMS keys [S46].
  - On Windows runners, `azure/artifact-signing-action@v2` does the signing [S44].
- **Building the Windows environment from Linux works** [M]:
  - `uv pip compile --python-platform x86_64-pc-windows-msvc --python-version 3.13 --torch-backend cu132 --generate-hashes` produced a hashed lock in 1.5 s.
  - It resolved torch 2.15.0+cu132 and torchvision 0.29.2+cu132. That torchvision build is a `py3-none-win_amd64` wheel requiring torch ≥2.14.
  - `uv pip install --dry-run --target … --python-platform x86_64-pc-windows-msvc` also resolved successfully.

## 2. Comparison

**Packaging** (torch+CUDA is about 2 GiB to download and 2.9 GiB on disk)

| Approach | Fit | Install / update size | Signing (Smart App Control) | Main issues |
|---|---|---|---|---|
| PyInstaller onefile | ✗ | one exe, max 4 GiB | the DLLs it extracts are unsigned | unpacks 3 GB to %TEMP% on every launch; 4 GiB cap |
| PyInstaller onedir with everything, torch included | ✓ for the PoC | about 3.5 GB per release (estimate) | sign every PE file at build time | heavy updates; antivirus false positives |
| Nuitka 4.3 standalone | ◐ | same as above | same | AGPL compiler (output is fine), JIT off by default, long builds (UNVERIFIED) |
| Briefcase or PyApp stub with embedded Python | ✓ | small stub plus runtime | runtime must still be signed | turns into a first-run resolver if used naively |
| **Small frozen UI plus prebuilt, signed layers** (ComfyUI / Comfy Desktop style) | ✓✓ | installer about 100 MB (estimate); 2 GiB GPU layer only when pinned versions change | everything signed in CI | needs hosting plus manifest and updater code |
| Resolve with `uv pip` on the user's PC (InvokeAI, Anki 25.07) | ◐ | small installer | DLLs arrive unsigned | network timeouts and slow first runs; Anki dropped it |

**UI**

| | PySide6 6.12 | pywebview 6.2 | Tauri 2.12 | Electron 45 |
|---|---|---|---|---|
| License | LGPLv3, some modules GPL-only | BSD-3 | MIT/Apache | MIT |
| Footprint | about 310 MB installed before pruning [M] | small, plus WebView2 and .NET | small shell plus a frozen sidecar | large (UNVERIFIED) |
| Video preview | QMediaPlayer with the FFmpeg 9 backend | HTML5 video (codecs UNVERIFIED) | same as pywebview | Chromium |
| Stacks to maintain | Python | Python + JS | Rust + JS + a Python freezer | JS + Python |

## 3. Recommendation

**Primary approach: a small signed app, plus prebuilt, signed, versioned layers.**
1. **UI and orchestrator**
   - PySide6 6.12, using Qt Widgets with QSS, or QML with FluentWinUI3 on Windows 11 and Fusion on Windows 10.
   - QMediaPlayer for the preview, plus native drag and drop.
   - Frozen with PyInstaller 6.22 `--onedir` on Python 3.13.
   - The UI does **not** contain torch, so it starts fast and holds no CUDA context.
   - Exclude the GPL-only Qt modules. Never use onefile: it complicates LGPL relinking and hits the temp-extraction and 4 GiB problems.
2. **Installer**
   - Inno Setup 6.7 per-user install (`{userpf}`), so no UAC prompt.
   - It asks for a data folder with at least 20 GB free.
   - It checks for the VC++ x64 runtime and installs it (with elevation) if missing.
3. **GPU layer**
   - Built in CI from a hashed lock: python-build-standalone 3.13.18, torch 2.15.0+cu132, torchvision 0.29.2+cu132 and the remaining dependencies.
   - CI prunes it and signs every unsigned PE file.
   - It is shipped as a 7z split into parts under 2 GiB, with SHA-256 hashes and an Ed25519-signed manifest.
   - **A single variant covers RTX 20/30/40/50 with driver ≥580; recommend ≥595.**
   - Why: it is deterministic, compatible with Smart App Control, installable offline, and avoids running a resolver on users' machines (the Anki and Comfy experience).
4. **Tools**
   - Blender 5.2.4 LTS from the official zip, run as a subprocess with a `portable\` folder, verified against blender.org's SHA-256.
   - FFmpeg 8.1 from BtbN, lgpl-shared variant, pinned to a monthly build and mirrored together with its source. Its NVENC needs only driver ≥570, whereas BtbN's FFmpeg 9 would need ≥610.
5. **Models**: `snapshot_download` at a pinned commit into `local_dir`, verify checksums, then run with `HF_HUB_OFFLINE=1`.
6. **Process model**
   - One worker process per heavy stage: geometry, segmentation, Blender render, compositing plus encoding.
   - When a process exits, its VRAM is fully released.
   - Stages exchange files on disk, since Windows has no CUDA IPC.
   - The same worker CLI drives batch mode.

**Fallbacks**
- **For the PoC**: PyInstaller onedir with torch included, wrapped in Inno Setup with `DiskSpanning=yes` and `DiskSliceSize=1900000000`, so each slice fits GitHub's asset limit.
- **If hosting or signing of prebuilt layers isn't ready**: bundle the signed uv.exe and run `uv pip sync --require-hashes` into the runtime on first run. The cost is unsigned DLLs and a slower first run.
- **Driver below 580**: show a blocking dialog that links to the driver download. Optionally offer a legacy 2.14.1+cu126 layer for RTX 20–40 (it cannot run Blackwell). GTX 10 cards are unsupported.
- **No eligible signing service**: buy an OV certificate held in a cloud HSM (SSL.com eSigner or DigiCert KeyLocker) and sign with jsign. EV gives no SmartScreen advantage.

## 4. Implementation notes

**Disk layout**
```
%LOCALAPPDATA%\Programs\RoomStager\      app layer (Inno {userpf}), replaced on update
  RoomStager.exe  _internal\  bin\uv.exe  worker\  blender_scripts\  locks\  manifest.pub
<DATA> (user-chosen, ASCII, ≤60 chars; default %LOCALAPPDATA%\RoomStager)
  rt\gpu-cu132-py313-t2150-<lock8>\     python + Lib\site-packages (never modified in place)
  rt\current  -> NTFS junction (mklink /J, no admin) = stable exe path for NVIDIA per-app settings
  tools\blender-5.2.4\ (+portable\)   tools\ffmpeg-8.1-lgpl-shared-<date>\
  models\<name>@<rev8>\  assets\<ver>\  cache\{dl,hf,uv}\  jobs\<id>\  logs\
%USERPROFILE%\Videos\RoomStager\        exports
```

**GPU probe (runs in the UI process; package `nvidia-ml-py==13.615.71`)**
```python
import pynvml as nv
def probe():
    try: nv.nvmlInit()
    except nv.NVMLError: return {"ok": False, "why": "no NVIDIA driver"}
    try:
        drv = nv.nvmlSystemGetDriverVersion(); gpus = []
        for i in range(nv.nvmlDeviceGetCount()):
            h = nv.nvmlDeviceGetHandleByIndex(i)
            try: enc = nv.nvmlDeviceGetEncoderCapacity(h, nv.NVML_ENCODER_QUERY_H264)
            except nv.NVMLError: enc = 0
            gpus.append({"name": nv.nvmlDeviceGetName(h), "nvenc": enc,
                         "cc": nv.nvmlDeviceGetCudaComputeCapability(h),
                         "vram": nv.nvmlDeviceGetMemoryInfo(h).total})
        g = max(gpus, key=lambda x: x["vram"])
        ok = g["cc"] >= (7, 5) and int(str(drv).split(".")[0]) >= 580
        return {"ok": ok, "driver": drv, "cuda": nv.nvmlSystemGetCudaDriverVersion_v2(), "gpu": g}
    finally: nv.nvmlShutdown()
```

**Launching a worker stage**
```python
env = {**os.environ, "PYTORCH_ALLOC_CONF": "per_process_memory_fraction:0.9,throw_on_cudamalloc_oom:True",
       "HF_HOME": rf"{DATA}\cache\hf", "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
subprocess.run([rf"{DATA}\rt\current\python.exe", "-E", "-s", "-X", "utf8", rf"{APP}\worker\run.py",
                "--stage", "geometry", "--job", job], env=env, check=True,
               creationflags=subprocess.CREATE_NO_WINDOW)
```
Inside the worker:
- Use `torch.inference_mode()` with autocast. Choose the dtype with `torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16`; Turing cards will get fp16.
- After each model: `del model; gc.collect(); torch.cuda.empty_cache()`.
- Catch `torch.cuda.OutOfMemoryError` and retry with tiling or a lower resolution.
- Size budgets from NVML total VRAM and `torch.cuda.mem_get_info()`.

**NVENC self-test** (exit code 0 means usable):
`ffmpeg -hide_banner -v error -f lavfi -i testsrc2=s=1920x1080:r=30 -frames:v 60 -c:v h264_nvenc -preset p7 -tune hq -f null -`
- If it passes, encode with `-c:v h264_nvenc -preset p7 -tune hq -rc vbr -cq 18 -b:v 0 -pix_fmt yuv420p -movflags +faststart`.
- If it fails, fall back to `libopenh264` or `libsvtav1`, which are both in the LGPL build.

**Model weights**
```python
p = snapshot_download(REPO, revision=SHA, local_dir=rf"{DATA}\models\{NAME}@{SHA[:8]}",
                      allow_patterns=["*.safetensors", "*.json", "LICENSE*"], max_workers=8)
v = HfApi().verify_repo_checksums(REPO, revision=SHA, local_dir=p)
assert not v.mismatches   # then store our own sha256 list for offline re-checks
```

**Blender**
- Command: `tools\blender-5.2.4\blender.exe -b --factory-startup --offline-mode --python-exit-code 3 -P blender_scripts\render_job.py -- --job <dir> --cycles-device OPTIX`.
- During setup, render one warm-up frame so the driver compiles the OptiX PTX kernels before the first real job.
- Keep `blender_scripts\` under a GPL-compatible license, and talk to it only through files and the CLI.

**Inno Setup**
- `PrivilegesRequired=lowest`
- `DefaultDirName={userpf}\RoomStager`
- `ArchitecturesAllowed=x64compatible` and `ArchitecturesInstallIn64BitMode=x64compatible`
- `SignTool=…` with `SignedUninstaller=yes`
- `CloseApplications=yes`

**Building the GPU layer on a Linux runner** (only the dry-run was verified; validate the real install on Windows)
```
uv pip compile gpu.in --python-platform x86_64-pc-windows-msvc --python-version 3.13 --torch-backend cu132 --generate-hashes -o locks/win-cu132.txt
# unpack cpython-3.13.18+20261110-x86_64-pc-windows-msvc-install_only_stripped.tar.gz into layer/
uv pip install --require-hashes -r locks/win-cu132.txt --torch-backend cu132 --target layer/python/Lib/site-packages --python-platform x86_64-pc-windows-msvc --python-version 3.13
# prune torch/include, *.lib and tests; sign every unsigned .exe/.dll/.pyd with jsign; 7z a -v1900m; compute sha256; write and sign the manifest
```

**CI jobs**
- **ubuntu-latest**:
  - Run tests with CPU-only torch and check the lock.
  - Build the GPU layer only when the lock changes. Budget for the 14 GB disk: about 3 GB for the layer, 2 GB for the archive and 2 GB for the uv cache.
- **windows-latest**:
  - Build the UI with PyInstaller.
  - Run a self-test with `QT_QPA_PLATFORM=offscreen`.
  - Smoke-test the layer by importing it on CPU.
  - Sign, run ISCC, and upload to the Release.
- **End-to-end GPU test**, on a self-hosted RTX machine or the paid T4 runner:
  - Turn a real photo into an MP4.
  - Assert that pixels outside the object and shadow masks are bit-identical to the photo.
  - Check the NVENC path and the VRAM and time budgets.

**Updates**
- A signed `manifest.json` lists the app and each layer: id, version, parts, size, sha256 and minimum app version.
- **App layer**: download Setup.exe, verify it, then run `/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS`. Velopack deltas are the alternative; pick one, not both.
- **Heavy layers** change only when their pinned versions change. Install the new one side by side, run the self-test, then switch the junction. Keep the previous layer for rollback.
- **Batch mode**: `RoomStager.exe --batch <dir> --room living`. Each photo gets its own job folder, one GPU job runs at a time, and stage outputs are cached by input hash plus stage version.

## 5. Risks & limitations
- **Smart App Control and SmartScreen**: unsigned DLLs such as torch's and FFmpeg's may be blocked when Smart App Control is on. Whether torch specifically gets blocked is UNVERIFIED. New signed builds still show SmartScreen warnings until reputation builds up.
- **Signing access**: individual Artifact Signing appears to be US/CA only, so an OV certificate is likely needed. Certificates now expire after 460 days or less.
- **Driver drift**:
  - FFmpeg 9 (BtbN) needs driver ≥610 for NVENC.
  - cu134 wheels expect R615.
  - uv's auto backend ignores compute capability.
  - Always run self-tests and keep a fallback path.
- **VRAM**:
  - The sysmem fallback can silently slow everything down. The policy is per executable; whether it can be set from code through NVAPI is UNVERIFIED.
  - TDR resets can hit the display GPU.
  - Turing has no native bf16.
- **Paths**:
  - torch's internal paths reach 132 characters, so keep the data root short and ASCII to stay under MAX_PATH.
  - Read image bytes in Python (`np.fromfile` + `cv2.imdecode`); OpenCV's handling of non-ASCII paths on Windows is UNVERIFIED for 5.0.
- **Licensing**:
  - Qt LGPL: no onefile, offer the source, allow relinking, and write a compliant EULA.
  - Never import bpy in-process (GPL). Blender-side scripts must be GPL-compatible.
  - FFmpeg: prefer the LGPL build, since x264 would make it GPL.
  - H.264/HEVC patent royalties are not covered by any of these licenses, so get legal review.
  - The Nuitka compiler is AGPL (its output is fine). Inno Setup asks commercial users to buy a license.
- **CI limits**: 14 GB of disk, small artifact quotas and a 2 GiB per-asset limit. There are no GPUs on the free runners, and the T4 runner only covers Turing.
- **Python**: python.org no longer publishes 3.13 binaries, so the app depends on python-build-standalone security builds. Plan a move to 3.14 in 2027.
- **Firecrawl** ran out of credits during the research. Blender's license and FAQ pages returned 403, so those facts came from search snippets.

## 6. Sources
- S1 https://pypi.org/pypi/torch/json
- S2 https://github.com/pytorch/pytorch/blob/main/RELEASE.md
- S3 https://download.pytorch.org/whl/cu132/torch/ (also /cu134/, /cu130/, /cu128/, /cu126/)
- S4 https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- S5 https://docs.astral.sh/uv/guides/integration/pytorch/
- S6 https://github.com/astral-sh/uv/tree/main/crates/uv-torch/src (backend.rs, nvidia.rs, accelerator.rs)
- S7 https://pypi.org/project/PySide6/ (also PySide6-Essentials, PySide6-Addons)
- S8 https://doc.qt.io/qt-6/licensing.html
- S9 https://doc.qt.io/qt-6/qtmultimedia-index.html
- S10 https://www.qt.io/licensing/open-source-lgpl-obligations
- S11 https://doc.qt.io/qt-6/qtquickcontrols-fluentwinui3.html
- S12 https://pypi.org/project/PySide6-Fluent-Widgets/
- S13 https://github.com/r0x0r/pywebview/blob/master/docs/guide/web_engine.md
- S14 https://v2.tauri.app/develop/sidecar/
- S15 https://registry.npmjs.org/electron/latest and https://crates.io/crates/tauri
- S16 https://pyinstaller.org/en/stable/operating-mode.html, https://pyinstaller.org/en/stable/bootloader-building.html, https://github.com/pyinstaller/pyinstaller/blob/develop/PyInstaller/archive/writers.py, https://github.com/pyinstaller/pyinstaller-hooks-contrib/blob/master/_pyinstaller_hooks_contrib/stdhooks/hook-torch.py
- S17 https://github.com/Nuitka/Nuitka/blob/main/LICENSE-RUNTIME.txt, https://pypi.org/project/Nuitka/, https://github.com/Nuitka/Nuitka/blob/develop/nuitka/plugins/standard/standard.nuitka-package.config.yml
- S18 https://github.com/comfyanonymous/ComfyUI (README) and https://github.com/Comfy-Org/ComfyUI/releases/download/v0.45.0/ComfyUI_windows_portable_nvidia.7z
- S19 https://github.com/Comfy-Org/Comfy-Desktop
- S20 https://github.com/invoke-ai/launcher
- S21 https://newreleases.io/project/github/ankitects/anki/release/26.05b1 (mirror of the GitHub release notes)
- S22 https://github.com/ofek/pyapp and https://github.com/beeware/briefcase/blob/main/docs/en/reference/platforms/windows/app.md
- S23 https://github.com/velopack/velopack.docs/blob/master/docs/getting-started/python.mdx
- S24 https://jrsoftware.org/files/is6-whatsnew.htm, https://jrsoftware.org/files/is/license.txt, https://jrsoftware.org/ishelp/topic_setup_diskspanning.htm, https://jrsoftware.org/ishelp/topic_setup_privilegesrequired.htm
- S25 https://nsis.sourceforge.io/I_get_an_error_when_compiling_large_installers (seen via search only)
- S26 https://learn.microsoft.com/en-us/windows/msix/desktop/desktop-to-uwp-behind-the-scenes
- S27 https://blogs.windows.com/windowsdeveloper/2025/09/10/free-developer-registration-for-individual-developers-on-microsoft-store/ (seen via search only)
- S28 https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables, https://huggingface.co/docs/huggingface_hub/guides/download, https://huggingface.co/docs/huggingface_hub/guides/manage-cache, source at https://github.com/huggingface/huggingface_hub
- S29 https://pypi.org/project/bpy/
- S30 https://developer.blender.org/docs/release_notes/5.2/ and https://developer.blender.org/docs/release_notes/5.3/
- S31 https://download.blender.org/release/Blender5.2/
- S32 https://docs.blender.org/manual/en/5.2/render/cycles/gpu_rendering.html, https://docs.blender.org/manual/en/5.2/advanced/command_line/arguments.html, https://docs.blender.org/manual/en/5.2/advanced/blender_directory_layout.html
- S33 https://www.blender.org/support/faq/ (search snippet; direct fetch returned 403)
- S34 https://ffmpeg.org/download.html and https://ffmpeg.org/legal.html
- S35 https://github.com/BtbN/FFmpeg-Builds (README and scripts.d/50-ffnvcodec.sh, 50-x264.sh, 50-openh264.sh)
- S36 https://github.com/FFmpeg/nv-codec-headers (README on master, sdk/13.0, sdk/11.1)
- S37 https://www.gyan.dev/ffmpeg/builds/
- S38 https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvenc-application-note/index.html
- S39 https://pypi.org/project/nvidia-ml-py/
- S40 https://nvidia.custhelp.com/app/answers/detail/a_id/5490
- S41 https://github.com/pytorch/pytorch/blob/main/docs/source/notes/cuda.md and https://github.com/pytorch/pytorch/blob/main/docs/source/notes/windows.md
- S42 https://docs.github.com/en/actions/reference/runners/github-hosted-runners, https://docs.github.com/en/actions/reference/limits, https://docs.github.com/en/actions/reference/runners/larger-runners, https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases, https://github.com/actions/runner-images/blob/main/images/windows/Windows2025-VS2026-Readme.md
- S43 https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation and https://learn.microsoft.com/en-us/windows/apps/develop/smart-app-control/code-signing-for-smart-app-control
- S44 https://github.com/Azure/artifact-signing-action, https://azure.microsoft.com/pricing/details/artifact-signing/, https://learn.microsoft.com/en-us/answers/questions/5834570/when-will-azure-artifact-signing-support-australia
- S45 https://cabforum.org/2025/11/17/ballot-csc-31-maximum-validity-reduction
- S46 https://github.com/ebourg/jsign
- S47 https://peps.python.org/pep-0719/ and https://www.python.org/ftp/python/3.13.18/
- S48 https://releases.astral.sh/github/python-build-standalone/releases/download/20261110/ (listed by `uv python list --show-urls`, uv 0.13.8)

Working files from this research (downloaded data, helper scripts, test lockfile `lockdemo/req-win-cu132.txt`) are in /tmp/claude-0/-home-user/c09d70e1-5a72-5f0b-811a-6fd083338d9a/scratchpad/pkgresearch