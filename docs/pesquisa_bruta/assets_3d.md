# Research report: 3D furniture and decor assets for the commercial, local Windows virtual-staging app

*Checked 2026-11-15 against primary sources: APIs, LICENSE files, HF model cards and official pages. Anything I could not confirm is marked **UNVERIFIED**. MB means 10^6 bytes.*

## Findings

### 1. Poly Haven models (CC0): what the API actually returns
- **Listing.** `GET https://api.polyhaven.com/assets?t=models` returns **522 models**. **83** of them are in `Furniture/*`. The newest were published on **2026-07-03**.
- **Condition of the furniture.** The `attributes.condition` field gives:

  | Condition | Count |
  |---|---|
  | worn | 59 |
  | weathered | 6 |
  | rusted | 5 |
  | clean | 13 |

  So **84% of the furniture is not "clean".** Most of it is vintage, Victorian, Gothic, Chinese antique, farmhouse or industrial. Contemporary staging furniture is scarce: the sofas, beds and nightstands are all vintage.
- **Dimensions.** `/info/<id>` → `dimensions` is in **mm**, ordered `[X,Y,Z]`. In Blender Z-up that means width × depth × height. The order is inferred from flat assets: `hanging_picture_frame_01` = 594×16×841. For all 225 candidates I fetched, the info values equal the listing values.
- **Files.** `/files/<id>` returns, for each format (`gltf`, `blend`, `fbx`, `usd`) and each resolution (`1k` to `8k`), a `url`, `size`, `md5` and an `include{}` map of textures and the `.bin`.
  - Every candidate ships gltf/blend/fbx/usd, **except `decorative_book_set_01`**, which has no glTF (.blend 1k = 63.7 MB).
  - glTF uses JPG textures with a packed ARM map. The .blend packs EXR normal maps, so it is larger.
- **Mesh structure.** I fetched the `.gltf` of 22 assets.
  - Most are a **single mesh node**: `sofa_03`, the coffee tables, the chairs.
  - Some have usable parts for the assembly animation:
    - `drawer_cabinet`: body + 4 drawers
    - `modern_wooden_cabinet`: body + 2 doors
    - `wooden_display_shelves_01`: + 3 bins
    - `sofa_02`: Base + Seat
    - `throw_pillows_01`: 2 pillows
    - `book_encyclopedia_set_01`: 20 separate books
    - the plants: pot / stem / leaves / soil
- **Gaps.** I keyword-searched all 522 names, tags and descriptions. There is none of the following:
  - rugs or carpets
  - curtains (the only drapery is cloth baked into `dining_table` and `vintage_day_bed`)
  - towels, bath fixtures, toiletries
  - floor or table lamps (only industrial desk lamps)
  - office chairs, wardrobes
  - mattresses, duvets, blankets
  - flat-screen TVs (only 2 CRTs)
  - a modern bed
  - a contemporary fabric sofa
  
  There is one mirror, and it is ornate.
- **Poly Haven textures (CC0).** 869 in total, **43 tagged fabric**: `curly_teddy_natural/checkered` (bouclé look), `rough_linen`, `ribbed_corduroy`, `terry_cloth` (towels), `hessian_230/380` (jute), `poly_wool_herringbone`, `floral_jacquard`, `leather_white`. These are good for re-skinning furniture and for building rugs, towels and curtains.
- **Licence and API terms.**
  - The assets are CC0. Poly Haven's licence page says you may redistribute them "even in a product you sell".
  - The **API Terms of Service** add three things:
    - Every call must carry a unique User-Agent or Referer naming the software (§2.4).
    - If the **live API** is used *inside* the product, the app must show a light "Powered by Poly Haven" credit (§2.5).
    - There is no SLA (§4.2).
  - §2.7 offers an optional corporate "Provenance Attestation manifest" or "Bulk Snapshot".
  - **Consequence:** download at build time, ship our own snapshot, and make no API calls at runtime.

### 2. Poly Haven assets per room
All rows are CC0. Columns:
- **W×D×H** in mm
- **poly** = the API `polycount` value
- **glTF / .blend / FBX** = total download in MB at 1k / 2k, textures included
- **cond** = the `condition` attribute

| ID | W×D×H | poly | glTF 1k/2k | .blend 1k/2k | FBX 1k/2k | cond |
|---|---|---|---|---|---|---|
| **Living room** |||||||
| `sofa_02` | 1807×818×709 | 2728 | 0.4/1.3 | 1.6/5.9 | 1.5/5.8 | worn |
| `sofa_03` | 2731×925×1118 | 8004 | 0.7/1.7 | 1.4/3.4 | 1.3/3.3 | worn |
| `Sofa_01` | 1571×658×797 | 4101 | 0.5/2.2 | 2.0/8.2 | 1.9/8.1 | worn |
| `modern_arm_chair_01` | 820×987×1023 | 8916 | 2.7/9.2 | 12.2/41.5 | 12.2/41.5 | worn |
| `mid_century_lounge_chair` | 1009×1190×1169 | 6148 | 2.1/7.7 | 3.1/11.4 | 3.1/11.3 | worn |
| `ArmChair_01` | 848×766×1065 | 5626 | 0.8/2.8 | 2.7/10.3 | 2.6/10.2 | worn |
| `Ottoman_01` | 885×621×624 | 4240 | 0.7/2.5 | 2.6/10.0 | 2.5/9.9 | worn |
| `coffee_table_round_01` | 1301×1301×491 | 4044 | 1.6/5.7 | 2.8/9.8 | 2.7/9.7 | clean |
| `modern_coffee_table_01` | 1202×600×390 | 4504 | 1.3/4.7 | 2.4/8.3 | 2.3/8.2 | clean |
| `modern_coffee_table_02` | 1199×1200×369 | 13645 | 3.0/9.5 | 15.6/58.5 | 15.6/58.4 | clean |
| `side_table_01` | 550×450×551 | 2756 | 0.5/1.6 | 1.1/3.4 | 1.0/3.3 | clean |
| `side_table_tall_01` | 384×384×761 | 6408 | 1.6/5.1 | 1.9/6.1 | 1.9/6.1 | worn |
| `modern_wooden_cabinet` (TV console) | 2440×520×680 | 24976 | 2.9/8.2 | 3.3/12.3 | 3.4/12.3 | clean |
| `ClassicConsole_01` | 1543×589×949 | 7566 | 0.7/2.0 | 2.3/7.4 | 2.2/7.3 | worn |
| `steel_frame_shelves_01` | 1097×502×2141 | 4348 | 1.6/5.6 | 2.4/8.3 | 2.3/8.2 | clean |
| `steel_frame_shelves_02` | 593×502×2142 | 4348 | 0.5/1.5 | 1.5/4.9 | 1.3/4.8 | clean |
| `wooden_display_shelves_01` | 1078×372×1556 | 3174 | 0.5/1.4 | 1.2/4.1 | 1.1/4.0 | clean |
| `drawer_cabinet` | 1141×488×1881 | 26406 | 1.1/2.1 | 1.7/4.9 | 1.8/4.9 | clean |
| `Television_01` (CRT) | 600×471×457 | 1918 | 0.5/1.6 | 1.6/4.9 | 1.5/4.8 | worn |
| `throw_pillows_01` | 1022×627×450 | 6362 | 1.2/4.5 | 3.5/13.4 | 3.5/13.3 | clean |
| `potted_plant_01` | 589×637×1351 | 96030 | 6.3/9.0 | 22.6/76.1 | 25.2/78.7 | weathered |
| `potted_plant_02` | 701×657×841 | 69806 | 2.6/4.9 | 28.1/84.0 | 22.2/78.1 | weathered |
| `potted_plant_04` | 170×186×268 | 6097 | 2.1/6.3 | 3.0/9.0 | 3.0/9.0 | weathered |
| `ceramic_vase_01` | 205×206×401 | 2548 | 0.4/0.7 | 0.6/1.3 | 0.7/1.4 | clean |
| `ceramic_vase_04` | 183×182×342 | 2224 | 0.6/1.6 | 1.8/6.6 | 1.9/6.7 | clean |
| `brass_vase_01` | 227×227×692 | 21440 | 0.8/1.7 | 3.1/5.6 | 1.5/4.0 | worn |
| `book_encyclopedia_set_01` | 551×163×237 | 67306 | 3.3/4.7 | 4.0/10.0 | 3.9/9.9 | worn |
| `decorative_book_set_01` | 2505×321×286 | 112560 | –/– | 63.7/207.6 | 74.7/218.6 | clean |
| `hanging_picture_frame_01` | 594×16×841 | 2586 | 0.4/1.2 | 1.2/4.2 | 1.1/4.1 | clean |
| `hanging_picture_frame_02` | 752×33×500 | 3678 | 0.7/2.2 | 7.6/28.3 | 7.5/28.2 | worn |
| `standing_picture_frame_01` | 201×100×250 | 1634 | 0.4/1.4 | 1.3/4.5 | 1.1/4.3 | clean |
| `wall_clock` | 320×47×320 | 3658 | 1.6/4.9 | 2.7/7.5 | 2.7/7.4 | worn |
| `wicker_basket_02` | 375×271×218 | 17850 | 3.1/10.2 | 4.3/14.6 | 4.3/14.5 | worn |
| `wooden_bowl_01` | 313×309×94 | 14122 | 1.8/6.5 | 2.9/9.5 | 2.9/9.5 | worn |
| `horse_statue_01` | 164×108×220 | 22388 | 1.3/2.4 | 3.9/7.9 | 2.2/6.2 | clean |
| `modern_ceiling_lamp_01` | 432×432×952 | 5602 | 0.4/1.1 | 1.1/2.8 | 1.0/2.6 | clean |
| `Chandelier_03` | 780×785×1041 | 29301 | 1.4/2.7 | 3.6/9.2 | 3.6/9.2 | clean |
| **Bedroom** |||||||
| `GothicBed_01` | 1494×2040×1534 | 18741 | 1.2/2.2 | 2.5/6.0 | 2.5/6.0 | worn |
| `old_bed_frame` | 905×2002×1201 | 49990 | 4.0/10.2 | 4.3/13.5 | 4.6/13.8 | rusted |
| `vintage_day_bed` | 1973×855×1127 | 2715 | 2.4/8.8 | 4.0/13.9 | 4.0/13.8 | worn |
| `ClassicNightstand_01` | 568×424×700 | 2002 | 0.4/1.4 | 1.6/5.4 | 1.5/5.3 | worn |
| `painted_wooden_nightstand` | 505×509×616 | 470 | 2.2/8.2 | 3.1/11.7 | 3.1/11.6 | worn |
| `GothicCommode_01` | 1201×581×1212 | 3897 | 0.6/1.7 | 1.5/4.7 | 1.4/4.6 | worn |
| `painted_wooden_cabinet_02` (wardrobe-like) | 997×729×2569 | 966 | 1.6/6.2 | 2.3/9.4 | 2.2/9.3 | worn |
| `ornate_mirror_01` | 486×26×744 | 14246 | 0.7/1.3 | 1.3/2.7 | 1.2/2.6 | worn |
| `alarm_clock_01` | 228×168×174 | 13006 | 1.7/4.7 | 2.7/7.3 | 2.6/7.1 | worn |
| **Kitchen + dining** |||||||
| `metal_stool_01` (bar) | 352×355×883 | 8334 | 2.3/8.5 | 3.5/12.9 | 3.5/12.9 | worn |
| `metal_stool_03` (bar, backrest) | 461×471×1023 | 6576 | 2.3/8.3 | 3.7/12.0 | 3.6/11.9 | weathered |
| `bar_chair_round_01` | 483×486×751 | 14373 | 2.8/7.2 | 3.0/8.3 | 3.0/8.3 | worn |
| `round_wooden_table_01` | 1399×1399×1005 | 8640 | 0.6/1.3 | 2.5/4.7 | 1.0/3.2 | worn |
| `round_wooden_table_02` | 796×796×746 | 4254 | 1.9/6.8 | 2.8/9.7 | 2.7/9.6 | worn |
| `painted_wooden_table` | 2406×1137×959 | 600 | 1.8/7.4 | 2.7/10.9 | 2.6/10.8 | worn |
| `dining_table` (cloth-covered) | 2256×1390×877 | 1048 | 2.3/9.0 | 3.3/12.8 | 3.2/12.7 | worn |
| `dining_chair_02` | 434×576×973 | 22013 | 0.9/1.9 | 2.2/7.0 | 2.5/7.2 | clean |
| `painted_wooden_chair_01` | 432×540×956 | 724 | 0.5/1.5 | 1.4/4.9 | 1.3/4.8 | worn |
| `WoodenTable_03` (sideboard) | 1327×562×830 | 2298 | 0.6/1.9 | 1.9/6.1 | 1.9/6.0 | worn |
| `pot_enamel_01` | 263×180×178 | 11416 | 1.7/5.9 | 3.1/11.0 | 3.1/11.1 | worn |
| `wooden_cutting_board` | 450×247×41 | 3520 | 2.0/8.1 | 3.0/12.0 | 2.9/12.0 | worn |
| `jug_01` | 275×168×211 | 5770 | 1.2/4.8 | 2.4/8.7 | 2.4/8.7 | clean |
| `tea_set_01` | 918×484×165 | 40296 | 2.3/5.2 | 3.6/8.7 | 3.6/8.6 | clean |
| `wine_bottles_01` | 676×81×331 | 28848 | 3.9/12.3 | 10.3/36.5 | 10.3/36.5 | worn |
| `bananas` | 600×244×179 | 46868 | 2.5/5.6 | 2.6/6.0 | 2.8/6.3 | clean |
| `food_apple_01` | 98×96×85 | 7012 | 0.5/1.2 | 1.4/3.3 | 1.3/3.2 | clean |
| `wooden_bowl_02` (fruit bowl) | 131×125×73 | 4666 | 2.5/8.5 | 3.5/11.6 | 3.4/11.5 | worn |
| `Chandelier_01` | 746×816×798 | 26860 | 1.3/2.1 | 3.1/7.2 | 3.1/7.1 | worn |
| **Bathroom** (almost nothing suitable) |||||||
| `wicker_basket_01` | 383×295×117 | 22276 | 2.7/8.4 | 3.6/11.9 | 3.6/12.0 | weathered |
| `wooden_stool_02` | 265×183×182 | 11146 | 2.7/9.7 | 3.8/13.9 | 3.7/13.9 | worn |
| `planter_pot_clay` | 266×263×222 | 3080 | 1.8/6.0 | 2.7/8.5 | 2.7/8.5 | weathered |
| **Office** (no office chair) |||||||
| `metal_office_desk` | 2000×947×788 | 6898 | 1.6/5.2 | 1.9/6.0 | 1.9/6.0 | worn |
| `SchoolDesk_01` | 712×546×883 | 4162 | 0.4/1.1 | 1.2/4.5 | 1.1/4.4 | worn |
| `WoodenTable_01` | 1800×657×549 | 952 | 0.6/1.8 | 1.6/5.7 | 1.5/5.7 | worn |
| `desk_lamp_arm_01` | 617×408×879 | 25710 | 2.9/8.5 | 5.1/16.0 | 6.6/17.5 | worn |
| `industrial_pipe_lamp` | 183×262×364 | 8626 | 2.4/8.7 | 3.4/12.3 | 3.4/12.3 | weathered |
| `vintage_wooden_drawer_01` | 1715×1007×1091 | 5184 | 2.1/6.9 | 2.8/9.1 | 2.7/9.0 | worn |
| `classic_laptop` (vintage) | 652×486×543 | 14450 | 2.2/5.9 | 3.2/8.5 | 3.2/8.5 | worn |
| `office_notepads` | 1103×660×22 | 666 | 1.3/5.7 | 1.9/8.7 | 1.8/8.6 | clean |
| `stationery_supplies` | 302×84×164 | 4412 | 2.0/7.3 | 2.8/9.8 | 2.8/9.8 | worn |

**Data traps.** Some IDs are scattered sets or clusters, not single objects:
- `calathea_orbifolia_01` is 2.49 m wide.
- `pachira_aquatica_01` is 6.87 m.
- `office_notepads` is a scattered set.

Some succulents run up to about 654k polys (`leipoldtia_schultzei`, `othonna_cerarioides`).

### 3. Filling the gaps
- **Amazon Berkeley Objects (ABO): CC BY 4.0, not NC.** Verified on the official page and both READMEs.
  - **Content:** 7,953 glTF 2.0 GLBs with PBR maps (base colour, metallic, roughness). 4,241 of them have 4K maps.
  - **Geometry:** median 18,240 faces, maximum 11.5M.
  - **Download:** single files from S3 at `…/3dmodels/original/<path>`, where `<path>` comes from `3dmodels/metadata/3dmodels.csv.gz`. Six samples ranged from 1.2 to 56 MB.
  - **Units:** extents are in metres, glTF Y-up. A 4×6 ft rug, `B07B4YVDLQ`, measures 1.812×0.006×1.202.
  - **What it contains** (my join of the listings): CHAIR 1,184 · SOFA 859 · RUG 817 · WALL_ART 599 · TABLE 584 · LAMP 345 · STOOL 340 · OTTOMAN 288 · PILLOW 275 · HEADBOARD 206 · LIGHT_FIXTURE 197 · PLANTER 182 · DESK 145 · BED 140 · CABINET 137 · HOME_MIRROR 109 · SHELF 106 · DRESSER 62 · BED_FRAME 52 · VASE 50.
  - **Brands:** mostly Amazon private labels (Stone & Beam, Rivet, Ravenna Home).
  - **Obligations:** credit "Amazon.com", link the licence, indicate changes, and apply **no DRM** ("no … technological measures").
  - **Verdict:** this is the best commercially usable source of *contemporary* sofas, beds, rugs, lamps, mirrors and wall art.
- **ambientCG (CC0).** Its licence explicitly allows including the raw files "in a video game".
  - **Carpet:** API v2 `full_json?type=Material&category=Carpet` returns 16 materials.
  - **Fabric:** `q=fabric` returns 106, including Fabric083–086 released Sep–Oct 2026.
  - **Download:** e.g. `https://ambientcg.com/get?file=Carpet016_2K-JPG.zip` is about 31 MB (1K is about 9 MB).
- **Infinigen / Infinigen Indoors.**
  - **Licence:** BSD-3-Clause; the outputs are ours.
  - **Indoor factories** in the indoors-stable list, all usable for gap-filling:
    - seating: `SofaFactory`, `ArmChairFactory`, `OfficeChairFactory`, `BarChairFactory`
    - bed and bedding: `BedFactory`, `MattressFactory`, `PillowFactory`, `BlanketFactory`
    - soft goods: `TowelFactory`, `RugFactory`
    - lighting: `FloorLampFactory`, `LampFactory`, `DeskLampFactory`
    - TV and wall: `TVFactory`, `TVStandFactory`, `MirrorFactory`, `WallArtFactory`
    - tables: `TableDiningFactory`, `TableCocktailFactory`
    - kitchen: `KitchenCabinetFactory`, `KitchenIslandFactory`
    - plants and decor: `PlantContainerFactory`, `VaseFactory`, `BookStackFactory`
    - plus tableware and bathroom fixtures
  - **No curtain factory** is in that list.
  - **Platforms:** the "Minimal install" is yes on Linux and Mac, **"experimental" on Windows x86_64 and WSL2**. Docker on Windows goes through WSL2.
  - **Export:** bakes only albedo, roughness, normal and metallic. Transparent materials may export wrongly. Exporting single objects "generally works fine".
  - **Verdict:** generate offline on a Linux box. Never ship Infinigen itself.
- **Sketchfab.** The license list includes `cc0`.
  - The Download API requires **end users to log in to Sketchfab**; downloading without that means "contact us".
  - It serves glTF/GLB/USDZ only.
  - CC0 status is declared by the uploader, so provenance is a risk.
  - **Verdict:** only for a few hand-curated hero pieces.
- **BlenderKit, now "Blendkit (formerly BlenderKit)".** blenderkit.com now 301-redirects to blendkit.com. It offers two licences:
  - **Royalty Free:** commercial use, no resale "in the same form", and for games the "assets shouldn't be directly extractable".
  - **CC0.**
  - **Verdict:** Royalty Free assets are unsuitable for files shipped inside an app without a written agreement. CC0 items are usable but must be checked one by one.
- **Quaternius and Kenney.** Both CC0. Kenney's Furniture Kit has 140 files (2018); the Quaternius pack is about 2.9 MB.
  - **Verdict:** stylized low-poly game art. Use only as layout and test proxies.
- **Objaverse 1.0.** The dataset is ODC-By; each object carries its own licence:

  | Licence | Objects |
  |---|---|
  | CC-BY | 721K |
  | CC-BY-NC | 25K |
  | CC-BY-NC-SA | 52K |
  | CC-BY-SA | 16K |
  | CC0 | 3.5K |

  **Objaverse-XL** is also ODC-By with mixed per-object licences, and its Polycam subset is non-commercial.
  - **Verdict:** quality is highly variable. Avoid for v1.
- **3D-FUTURE.** "Scientific research purpose only", "shall not commercialize", no redistribution. **Excluded.** 3D-FRONT is probably the same: **UNVERIFIED**.

### 4. Local generative 3D: the dependencies are often the real licence blocker
These dependencies restrict commercial use regardless of the model's own licence:
- **nvdiffrast** and **nvdiffrec** (NVIDIA Source Code License): "only may be used … non-commercially … research or evaluation purposes only".
- **Inria `diff-gaussian-rasterization` and `diffoctreerast`**: non-commercial.
- **`briaai/RMBG-2.0`**: CC BY-NC 4.0.

Safe dependencies:
- **CuMesh** and **FlexGEMM**: MIT.
- **Kaolin** and **FlexiCubes**: Apache-2.0.
- **DINOv3 licence**: a royalty-free grant that allows commercial use, gated, with trade-control limits.

## Comparison tables

**A. Asset sources**

| Source | Licence | Can we embed it in a paid app? | Fit for photoreal staging | Verdict |
|---|---|---|---|---|
| Poly Haven models | CC0 | Yes; credit needed only if we use the live API | High quality but 84% worn or vintage | Core: decor, tables, shelves, plants |
| Poly Haven / ambientCG materials | CC0 | Yes | Fabrics, carpets, leather, jute | Re-skinning, rugs, curtains, towels |
| ABO | CC BY 4.0 | Yes, with attribution and no DRM | Contemporary real products | **Main modern-furniture source** |
| Infinigen Indoors | BSD-3; outputs ours | Yes | Procedural, decent | Beds, bedding, towels, lamps, office chairs, TVs |
| Sketchfab CC0 | per model | CC0 only; API needs user login | Mixed | Manual hero pieces |
| Blendkit | RF or CC0 | RF: not if extractable | Good | CC0 only, case by case |
| Quaternius / Kenney | CC0 | Yes | Low-poly | Proxies only |
| Objaverse / XL | per object | Per object | Very mixed | Avoid |
| 3D-FUTURE | research only | No | — | Excluded |

**B. Generative 3D** (minimum VRAM as stated officially)

| Model (date) | Code / weights | Blockers | VRAM | Output | Windows |
|---|---|---|---|---|---|
| TRELLIS (2024-12) | MIT/MIT | diffoctreerast (Inria NC), nvdiffrast (NC) | ≥16 GB | GS/RF/mesh with baked texture | Linux only (issue #3) |
| **TRELLIS.2-4B** (2025-12) | MIT/MIT | nvdiffrast + nvdiffrec (NC); default rembg = RMBG-2.0 (NC) | **≥24 GB**; ckpts 16.2 GB | **PBR**: base colour, roughness, metallic, opacity; H100 ~3 s @512³, ~17 s @1024³, ~60 s @1536³ | "tested only on Linux" |
| Pixal3D (2026-05, SIGGRAPH'26) | MIT | built on TRELLIS.2, same blockers | low-VRAM mode @1024 | PBR | as TRELLIS.2 |
| Hunyuan3D-2.1 (2025-06) | Tencent Community | **Territory excludes EU/UK/South Korea**; Outputs may not be used outside the Territory; >1M MAU needs a licence; outputs may not train other models | 10 GB shape, 21 GB texture, **29 GB total** | PBR (Shape 3.3B + Paint 2B) | claims Win/Mac/Linux |
| Hunyuan3D-2.5 / 3.0 / 3.1 | — | No open weights: HF org lists only -1, -2, -2mini, -2mv, -2.1, -Part, -Omni | — | hosted API only | excluded |
| Step1X-3D (2025-05) | Apache/Apache | requirements pull nvdiffrast; its texture baker reuses Hunyuan code whose header says "TENCENT HUNYUAN NON-COMMERCIAL LICENSE" | 27–29 GB; 152 s / 50 steps | albedo (SDXL multi-view) | UNVERIFIED |
| TripoSG (2025-03) | MIT/MIT | none in requirements | **≥8 GB** | geometry only (1.5B) | UNVERIFIED |
| TripoSF (2025-03) | MIT | only a reconstruction VAE was released | ≥12 GB | — | — |
| SF3D / SPAR3D | Stability Community | free only below **US$1M annual revenue**; registration; "Powered by Stability AI" | ~6 GB / 10.5 GB (7 GB low-VRAM) | UV mesh, delit albedo + material params; low detail | "experimental" |
| SAM 3D Objects (2025-11) | SAM License (commercial OK) | trade controls | **≥32 GB**, linux-64 only | shape + texture + layout from real photos | No |
| Idol (2026-10) | Apache-2.0 | gated DINOv3 + FLUX.2-dev VAE (VAE licence UNVERIFIED) | 7.2 GB, <2 s | **Gaussian splats**, not meshes | UNVERIFIED |

Other 2026 HF image-to-3D entries:
- `facebook/ShapeR` and `UniForge3D`: CC BY-NC.
- `Hi3Dv1.0`: custom licence, geometry refinement, validated only on a 96 GB H20.
- `UltraShape` (Apache): refiner whose base model is Hunyuan3D-2.1.
- `nvidia/asset-harvester`: vehicles and road users.

**Quality for furniture.** Generated assets are single fused meshes, which is bad for the assembly animation. They are weak on thin legs and slats, have no reliable scale, and carry 1–2K textures. Add the VRAM, Linux-only and licence constraints, and **none of them is fit for the v1 runtime.**

## Recommendation

**Primary: a curated, versioned, offline-built library bundled with the installer.**
- No runtime downloads and no runtime API.
- One canonical `.blend` per asset (best fidelity in Cycles) plus a JSON manifest.
- Built in tiers:
  1. **CC0 base:** Poly Haven models, plus Poly Haven and ambientCG materials.
  2. **ABO (CC BY 4.0):** contemporary sofas, beds and headboards, rugs, lamps, mirrors, wall art and pillows. Ship `NOTICE`/`LICENSE-CC-BY-4.0.txt`, an in-app credits line "3D models © Amazon.com (ABO, CC BY 4.0), modified", and a note in the MP4 metadata. Do not encrypt these files and do not show brand names.
  3. **Infinigen Indoors, generated offline:** bedding, towels, office chairs, flat TVs, mirrors, kitchenware.
  4. **In-house procedural generators:** rugs, curtains, throws, flat TV, framed CC0 art.
  5. **Generative 3D:** an R&D track only, offline and with human QA. The only realistic clean path is TRELLIS.2 with BiRefNet (MIT) swapped in and nvdiffrast/nvdiffrec replaced (**UNVERIFIED** effort). It needs ≥24 GB VRAM.

**Fallback, if legal rejects CC BY attribution:** use only CC0 + BSD sources (Poly Haven, ambientCG, Infinigen, procedural), and buy or commission about 20–40 contemporary hero models with an explicit licence to embed them in software.

**Living-room proof of concept: Poly Haven only, all CC0.** 17 assets, measured at **28.8 MB (glTF 1k) / 72.5 MB (glTF 2k)**, with 66 texture maps. Parts that animate well are noted.

| # | ID | Role and animation idea |
|---|---|---|
| 1 | `sofa_02` | main sofa; Base + Seat, so the seat can drop in |
| 2 | `modern_arm_chair_01` | accent chair |
| 3 | `Ottoman_01` | ottoman |
| 4 | `modern_coffee_table_01` | coffee table (`coffee_table_round_01` as alternative) |
| 5 | `side_table_01` | side table |
| 6 | `modern_wooden_cabinet` | 2.44 m TV console; doors swing shut |
| 7 | `wooden_display_shelves_01` | shelving; 3 bins slide in |
| 8 | `book_encyclopedia_set_01` | 20 books land one by one |
| 9 | `potted_plant_01` | floor plant; pot, stem and leaves grow in |
| 10 | `potted_plant_02` | second plant |
| 11 | `ceramic_vase_01` | decor |
| 12 | `ceramic_vase_04` | decor |
| 13 | `hanging_picture_frame_01` | wall art |
| 14 | `throw_pillows_01` | two separate pillows |
| 15 | `wicker_basket_02` | decor |
| 16 | `modern_ceiling_lamp_01` | only if the ceiling is visible |

- **Rug:** procedural 2.0×3.0×0.012 m slab with ambientCG `Carpet016` or Poly Haven `poly_wool_herringbone` / `hessian_230`.
- **Optional modern swap (CC BY):** ABO `B075X4QMW7`, a Rivet mid-century velvet sofa, 2.21 m wide, 42 MB GLB.

## Implementation notes

**Build-time fetch from Poly Haven**
```python
UA={"User-Agent":"StagingStudio-AssetBuilder/0.1"}  # ToS §2.4: unique UA/Referer
info=requests.get(f"https://api.polyhaven.com/info/{aid}",headers=UA).json()
e=requests.get(f"https://api.polyhaven.com/files/{aid}",headers=UA).json()["gltf"]["2k"]["gltf"]
for rel,f in [(Path(e["url"]).name,e),*e["include"].items()]:
    b=requests.get(f["url"],headers=UA).content; assert md5(b).hexdigest()==f["md5"]; save(aid/rel,b)
W,D,H=[d/1000 for d in info["dimensions"]]   # metres, Blender X,Y,Z
```

**Normalisation in Blender** (headless, at build time)
- `bpy.ops.import_scene.gltf(...)`
- Compare the world bounding box with the API dimensions, tolerance about 2%.
- Set the origin at bottom-centre and apply transforms.
- Tag the front axis by hand (**UNVERIFIED** convention).
- Split single-mesh assets for the assembly animation with `bpy.ops.mesh.separate(type='LOOSE')`, then cluster tiny islands.
- `PartCrafter` (MIT) is an option for part splitting (**UNVERIFIED** quality).

**ABO join**
```python
m={r["3dmodel_id"]:r for r in csv.DictReader(gzip.open("3dmodels.csv.gz","rt"))}
# listings/metadata/listings_[0-f].json.gz: o["3dmodel_id"], o["product_type"][0]["value"]
url="https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original/"+m[mid]["path"]
```
After download: decimate anything above about 200k faces and transcode 4K maps to 2K.

**Infinigen** (Linux build box)
```bash
INFINIGEN_MINIMAL_INSTALL=True pip install -e .
python -m infinigen_examples.generate_individual_assets --output_folder out/beds -f BedFactory -n 20 --save_blend
python -m infinigen.tools.export --input_folder out/beds --output_folder out/beds_usd -f usdc -r 2048
```

**Per-asset manifest fields**
- `id`, `source`, `license`, `attribution`, `md5`
- `dims_m`, `origin`, `front`
- `placement` (floor / wall / ceiling / on-surface), `clearances`, `support_surfaces`
- `room_types`, `role`, `style`, `condition`
- `parts[]` with assembly order and pivot
- `tris`, `tex_res`, re-skinnable material slots

**VRAM**
- Cycles keeps full-resolution images in device memory.
- One 2K RGBA8 map is about 16.8 MB, so the proof of concept's 66 maps are about **1.1 GB at 2K**, 0.28 GB at 1K and about 4.4 GB at 4K.
- Cap it with `scene.cycles.texture_limit_render='2048'`. Use 1K for small props and 4K only for hero pieces.

## Risks & limitations
1. **Style mismatch.** Poly Haven furniture is mostly antique or worn, which clashes with contemporary listings. ABO, Infinigen and re-materialing are needed.
2. **ABO legal points:** where attribution goes (app, export, posted video), trademarks, real product designs, and the no-DRM clause. Get a legal review.
3. **Provenance.** CC0 on Sketchfab and Objaverse is self-declared. Poly Haven and ambientCG assets are their own work or donated to them.
4. **Data quality.** Clusters and scattered sets (see Data traps), polygon spikes (up to 654k on Poly Haven, 11.5M on ABO) and a missing glTF all need automated QA.
5. **Assembly animation.** Most meshes are a single node, so part splitting is per-asset work.
6. **Generative 3D.** Non-commercial dependencies, Hunyuan territory and output restrictions, the Stability revenue cap, VRAM needs of 24–32 GB against 8–24 GB consumer cards, and Linux-only CUDA builds. **Keep it out of the .exe.**
7. **Poly Haven API.** No SLA, and runtime use requires a credit. Use snapshots.
8. **Installer size.** Estimated 1–3 GB for 150–300 assets at 2K (**UNVERIFIED**). Ship per-room packs.

## Sources
- Poly Haven API: https://api.polyhaven.com/assets?t=models · https://api.polyhaven.com/info/sofa_03 · https://api.polyhaven.com/files/sofa_03 · https://api.polyhaven.com/assets?t=textures
- Poly Haven API terms: https://github.com/Poly-Haven/Public-API/blob/master/ToS.md
- Poly Haven licence: https://polyhaven.com/license
- ambientCG: https://ambientcg.com/api/v2/full_json?type=Material&category=Carpet · https://docs.ambientcg.com/license/
- ABO: https://amazon-berkeley-objects.s3.amazonaws.com/index.html · …/README.md · …/3dmodels/README.md · …/3dmodels/metadata/3dmodels.csv.gz · …/listings/metadata/listings_0.json.gz
- Infinigen: https://github.com/princeton-vl/infinigen (LICENSE) · branch `indoors-stable`: docs/Installation.md, docs/GeneratingIndividualAssets.md, docs/ExportingToExternalFileFormats.md, tests/assets/list_indoor_meshes.txt
- Sketchfab: https://sketchfab.com/developers/download-api · https://api.sketchfab.com/v3/licenses
- Blendkit: https://www.blendkit.com/docs/licenses/ · https://www.blendkit.com/docs/licenses/licensing-faq/
- Kenney: https://kenney.nl/assets/furniture-kit · Quaternius: https://opengameart.org/content/furniture-low-poly
- Objaverse: https://huggingface.co/datasets/allenai/objaverse · https://github.com/allenai/objaverse-xl
- 3D-FUTURE: https://github.com/3D-FRONT-FUTURE/3D-FUTURE-ToolBox · https://terms.aliyun.com/legal-agreement/terms/suit_bu1_ali_cloud/suit_bu1_ali_cloud202004171628_60052.html
- TRELLIS: https://github.com/microsoft/TRELLIS
- TRELLIS.2: https://github.com/microsoft/TRELLIS.2 · https://huggingface.co/microsoft/TRELLIS.2-4B (README, pipeline.json, ckpts tree)
- Dependency licences: https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt · https://github.com/NVlabs/nvdiffrec/blob/main/LICENSE.txt · https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/LICENSE.md · https://github.com/JeffreyXiang/diffoctreerast · https://github.com/JeffreyXiang/CuMesh · https://github.com/JeffreyXiang/FlexGEMM · https://github.com/NVIDIAGameWorks/kaolin · https://github.com/nv-tlabs/FlexiCubes
- Background removal and encoders: https://huggingface.co/briaai/RMBG-2.0 · https://huggingface.co/ZhengPeng7/BiRefNet · https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md
- Hunyuan3D: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 (LICENSE, README, requirements.txt) · https://github.com/Tencent-Hunyuan/Hunyuan3D-2 (LICENSE) · https://huggingface.co/api/models?author=tencent&search=Hunyuan3D
- Step1X-3D: https://github.com/stepfun-ai/Step1X-3D (LICENSE, README, requirements.txt, step1x3d_texture/differentiable_renderer/setup.py)
- Tripo: https://github.com/VAST-AI-Research/TripoSG · https://github.com/VAST-AI-Research/TripoSF
- Stability: https://github.com/Stability-AI/stable-fast-3d · https://github.com/Stability-AI/stable-point-aware-3d (LICENSE.md)
- SAM 3D Objects: https://github.com/facebookresearch/sam-3d-objects (LICENSE, doc/setup.md)
- Other 2026 models: https://huggingface.co/TencentARC/Pixal3D · https://huggingface.co/speridlabs/idol · https://huggingface.co/hi3d-official/Hi3Dv1.0 · https://huggingface.co/infinith/UltraShape · https://huggingface.co/api/models?pipeline_tag=image-to-3d&sort=likes
- Raw API dumps and analysis scripts: /tmp/claude-0/-home-user/c09d70e1-5a72-5f0b-811a-6fd083338d9a/scratchpad/ph/summary.json and /tmp/claude-0/-home-user/c09d70e1-5a72-5f0b-811a-6fd083338d9a/scratchpad/ph/table.txt

*Firecrawl was unavailable (out of credits). Web pages were read with curl and WebFetch instead.*
