# PLANO V2 — vídeo em até 2 minutos, móveis realistas e designer automático

Data: 16/11/2026. Branch: `claude/sweet-bohr-4uzml2`. **Este plano não reinicia o projeto.** Toda a arquitetura
atual continua: análise → layout → cena → Blender/Cycles em subprocesso → compositor bit a bit → encoder.
Cada pacote de trabalho (PT) abaixo altera funções que já existem.

## 0. Regra absoluta (vale para todos os PTs)

- A foto é uma **placa imutável**. Nenhum pixel do fundo muda, exceto onde aparecem os móveis e objetos
  adicionados e as suas sombras de contato.
- Proibido: re-síntese generativa da foto, movimento de câmera, antes/depois, *upscale* ou reamostragem da placa.
- Tudo o que for mudado para ganhar realismo ou velocidade atua **só nas camadas de CG** (RGBA, sombra, oclusão
  de contato). A placa continua copiada byte a byte fora da máscara "móvel + sombra".
- A verificação por quadro continua obrigatória. Muda só a forma (ver PT5): uma comparação `array_equal`
  independente sobre o complemento da máscara de influência.
- Todo PT tem um teste que falha se algum pixel fora da máscara mudar.
- Licenças: só código e pesos com uso comercial permitido. Ativos e arte só CC0 ou domínio público, com origem
  registrada. Tudo roda localmente, sem API paga.

## 1. Diagnóstico resumido do teste do usuário

| Queixa | Causa encontrada |
|---|---|
| Livros metade na mesa, metade no ar | (a) A mesa de centro era desenhada girada 90° (o arquivo 3D tem o eixo longo em Y e faltava a correção de rotação no catálogo). O layout planejava em um tampo de 1,20 × 0,60 m, e o render mostrava 0,60 m nesse eixo. Correção já pronta na árvore de trabalho, ainda não commitada. (b) Os 20 livros caem um a um de 22 cm, então por cerca de 1 s metade da fileira fica no ar. |
| Móveis "parecem imagem 3D" | Luz do CG cerca de 20% mais escura e mais azul que a foto (o ajuste de cor por canal é calculado e nunca aplicado). A sala proxy não "emite" a foto, então o móvel não recebe a luz real refletida da sala. Quadro de parede é um cartão cinza de modelo ("Ray Homes"). Sofá e poltrona quase pretos numa sala branca. Tapete procedural plano. Sombras de contato fracas. Foto com distorção de barril e CG em lente ideal. Sem reflexo no piso brilhante. |
| 8 min 23 s | Quase todo o tempo é amostragem do path tracing: (pixels do recorte) × (amostras por pixel) × (quadros renderizados). Hoje: 128 amostras, cerca de 75% da imagem por quadro, cerca de 216 quadros com movimento. O vídeo de 0,3 MB indica uma foto de apenas ~0,5–0,6 MP. Numa foto de 1920 × 1440, o pipeline atual levaria ~35 min (estimativa). Custos de primeira execução (compilação de kernels da GPU, carga fria de modelos, downloads, antivírus) podem estar incluídos e precisam ser confirmados. |

Conclusão: a velocidade exige uma **mudança estrutural no render** (camadas estáticas + camada dinâmica), não só
ajuste de parâmetros. O realismo vem de correções quase sem custo de render (luz, cor, ativos, sombras).

## 2. Orçamento de tempo — RTX 5070 Laptop (8 GB), foto 1920 × 1440, segunda execução

Medido do clique em "Gerar" até o MP4 verificado. A primeira execução após instalar (downloads, compilação de
kernels, varredura do antivírus) é medida e exibida à parte e **não conta** nos 120 s, porque é movida para o
instalador (PT2). Os valores de GPU são estimativas (±2×) até o usuário mandar o `report.json` da próxima versão.

| # | Etapa | Orçamento | Como cabe |
|---|---|---|---|
| 1 | Início do motor, detecção de hardware, carga da foto | 3 s | processo já aquecido; sem *imports* repetidos |
| 2 | Análise (geometria métrica, segmentação, sala, sol, oclusores) na GPU | 12 s | modelos sem inicialização aleatória desperdiçada; em paralelo com o início do Blender (PT14) |
| 3 | Layout + decisões de designer | 4 s | já mede 3,6 s em CPU |
| 4 | Ativos (cache pré-baixado) + `scene.json` | 2 s | tudo pré-baixado na instalação |
| 5 | Blender: montar cena, importar, dividir peças | 6 s | servidor Blender já aberto; cena enviada após liberar a VRAM dos modelos |
| 6 | Calibração de luz (4 renders a 240 px) + mapa de ganho/curva de tom | 3 s | |
| 7 | Quadro final (*hold*) a 128 amostras — também mede o desempenho da GPU | 12 s | ~1,4 MP de recorte × 128 amostras |
| 8 | Marcos estáticos (4 × 48–64 amostras) | 20 s | |
| 9 | Quadros dinâmicos (~150 × ~0,35 MP × 14 amostras + custo fixo ≤ 0,08 s/quadro) | 45 s | só a região das peças em movimento |
| 10 | Composição + codificação (em paralelo ao render; só a "cauda") | 5 s | recorte bit a bit, quadros reutilizados em cache |
| 11 | Verificação (byte a byte por quadro já feita no passo 10; PSNR em 3 quadros por *seek*) | 2 s | |
| 12 | Relatório e limpeza | 1 s | |
| 13 | Reserva | 5 s | |
| | **Total** | **120 s** | |

Com a foto do usuário (~0,57 MP), o mesmo pipeline fica em ~55 s. A sobra vai para qualidade do CG
(supersampling 1,5× só das camadas de CG, mais amostras nos marcos), nunca para a placa.

Ordem de corte se a medição prever mais de 120 s (decidida **uma vez**, antes do primeiro quadro animado):
1. amostras dos quadros dinâmicos (24 → 12);
2. escala de render só do CG em movimento (1,0 → 0,8);
3. amostras dos marcos (64 → 40).

Nunca reduzir a qualidade do quadro final. Nunca tocar na placa.

## 3. Pacotes de trabalho

Legenda: **P0** = entra na próxima versão que o usuário vai testar. P1 = versão seguinte. P2 = depois.
Todos os comandos de teste: `cd /home/user/softwareEditor && /home/user/.venvs/staging/bin/python -m pytest -q -m "not slow and not models"`.
Todos os PTs que tocam render ou composição exigem `preservado_exato = True` em todos os quadros.

---

### PT0 — Livros na mesa: integrar a correção e eliminar o "meio no ar" da animação **[P0]**

**Objetivo:** nenhum objeto decorativo fica para fora do tampo, nem parado nem durante a queda.

**Arquivos/funções:**
- Revisar e commitar a correção que está na árvore de trabalho:
  - `mobilia/layout/catalog.py`: `yaw_fix=90` para a mesa de centro moderna 01 e a estante expositora 01; tamanhos certos de almofadas e cesto;
  - `mobilia/layout/surfaces.py`: `ON_TOP_INSET = 0.02`, `rotated_extent()`, altura do tampo da estante 1,542;
  - `mobilia/layout/solver.py` `_decor_spot`;
  - `mobilia/layout/rules.py` `validate_layout`;
  - `mobilia/render/blender_worker.py` `_check_footprint`;
  - `tests/test_on_top.py`, `tests/fixtures/polyhaven_raw_sizes.json`.
  - Se outro PT também mexer em `blender_worker.py` ao mesmo tempo, mesclar com cuidado.
- `blender_worker.py`, depois de `bake_animation`: nova `_check_rest_support(items_rt)`. Para cada item com
  `mount == "on_top"`, faz *ray-cast* para baixo (BVH das malhas do pai) numa grade 5 × 5 sob a base do filho.
  Exige ≥ 90% de acertos a até 8 mm. Se falhar, remove o item, registra um aviso e grava no relatório.
- `mobilia/assets/procedural.py`: novo `book_stack`, uma pilha de 2–3 livros deitados (~30 × 23 × 3 cm), capas
  na paleta, levemente deslocados e girados. A fileira de 20 enciclopédias em pé fica só para estantes
  (`catalog.py`: alvo de decoração por pai).
- `mobilia/render/choreography.py`:
  - `drop_height` com 0,12 m para decoração sobre tampo;
  - em `plan_parts`, para pilhas e conjuntos de livros: no máximo 3 partes, intervalo de 0,10–0,12 s entre elas.

**Testes novos:**
- `test_on_top.py` (já existe, 382 passando).
- `test_choreography.py::test_on_top_parts_airborne_window_short`: soma do tempo em que alguma parte está mais de 1 cm acima do repouso < 0,35 s.
- `test_procedural_assets.py::test_book_stack_footprint`.
- Teste lento: `_check_rest_support` com a mesa em `yaw_fix=0`, que deve remover o item.

**Validação em CPU:** suíte rápida. Re-renderizar só o quadro final da sala de teste a 640 px / 24 amostras
(`scratchpad/realism/exp_render.py`) e inspecionar o recorte da mesa.

**Efeito visível:** livros, vaso e tigela apoiados no tampo. A mesa de centro deixa de aparecer girada. A
decoração parece arrumada por um designer.

---

### PT1 — Telemetria completa e botão "Copiar diagnóstico" **[P0]**

**Objetivo:** a próxima execução do usuário substitui as estimativas por números reais (ver seção 4).

**Arquivos/funções:**
- `mobilia/pipeline.py` `run_job`: cronômetro por subetapa (ver a lista na seção 4) e
  `primeira_execucao: bool`, tirado de `bootstrap.load_state`.
- `blender_worker.py` `main`:
  - registra por quadro, via handler `render_stats` + `time.perf_counter`:
    - tempo de parede;
    - tempo de amostragem;
    - tempo de gravação;
    - área do recorte;
    - amostras;
    - camada (estática/dinâmica);
  - no início, emite o dispositivo efetivo do denoiser e o dispositivo de render.
- `compose_video`: tempos de leitura, composição e codificação por quadro.
- `mobilia/ui/app.py`: botão "Copiar diagnóstico", que gera um zip com `report.json`, logs e `frames.json` (sem a foto, a menos que o usuário marque).

**Testes:**
- `test_pipeline_e2e.py` com `fake_worker.py`: o relatório tem todas as chaves de tempo e a soma das etapas fica
  dentro de 5% do total.
- `test_ui.py`: o zip é criado e não contém a foto por padrão.

**Validação em CPU:** rodar o e2e com o worker falso e uma renderização real mínima a 320 px.

**Efeito visível:** botão de diagnóstico e tempo por etapa na tela de conclusão.

---

### PT2 — Instalação prepara tudo; primeira execução separada **[P0]**

**Objetivo:** nenhum custo único (downloads, compilação de kernels, carga fria) cai nos 2 minutos.

**Arquivos/funções:**
- `mobilia/bootstrap.py` `install`:
  - pré-baixa todos os ativos do catálogo nas texturas 2k (`assets/polyhaven.py`), com progresso;
  - roda um render de aquecimento 64 × 64 com OptiX e outro com CUDA, para preencher o cache de kernels da NVIDIA;
  - define `CUDA_CACHE_MAXSIZE` (~4 GB) para os dois processos;
  - faz uma carga de aquecimento dos modelos de análise.
- `pipeline.py`: `tempos_s.download` separado; aviso na interface "primeira execução — mais lenta".

**Testes:** `test_bootstrap.py` com os passos novos simulados (idempotência, retomada).

**Validação em CPU:** o instalador em modo CPU executa o aquecimento sem erro; o estado é salvo.

**Efeito visível:** a primeira geração deixa de pagar minutos extras.

---

### PT3 — Menos trabalho de render sem perder qualidade visível **[P0]**

**Objetivo:** cortar quadros e amostras desperdiçados.

**Arquivos/funções:**
- `pipeline.py` `_quality`: `final` com 24 fps (cinematográfico; desfoque de movimento já ligado).
- `choreography.py` `PartPlan.pose`: devolve **exatamente** a pose de repouso quando o resíduo projetado for < 0,3 px.
  - Não mexer em `PartPlan.end` nem no 2,2 × duração, porque isso muda a linha do tempo e o início do *hold*.
  - `frame_signature` passa a reutilizar a cauda automaticamente.
- `choreography.py` `schedule_items`: no máximo 2–3 peças em movimento ao mesmo tempo, sem alongar a linha do
  tempo (sobrepor sub-etapas dentro da duração de cada item).
- `blender_worker.py` `configure_render`:
  - `adaptive_min_samples` explícito (8 para dinâmico, 32 para estático);
  - `adaptive_threshold` 0,02;
  - `glossy_bounces` 2 e `transparent_max_bounces` 8; difusos continuam em 4, e a calibração usa os mesmos valores;
  - `use_animated_seed` desligado.
- `_wrap_fade`: o mix transparente fica só durante o *fade-in*; depois o material volta ao original.
- `hardware.py` `choose_profile`: as amostras passam a ser por camada (estática/final/dinâmica; ver PT6), e não um
  valor único de 128.

**Testes:**
- `test_choreography.py::test_pose_snaps_to_rest_subpixel`.
- `::test_max_concurrent_movers`.
- `::test_timeline_length_unchanged`.
- `test_scene_spec.py`: amostras por camada no `scene.json`.

**Validação em CPU:** na `scene.json` da sala de teste, contar os quadros com movimento: esperado ~216 → ~150.

**Efeito visível:** animação igual (o usuário gostou dela), vídeo a 24 fps, ~25–30% menos render.

---

### PT4 — Saída de quadro barata: EXR recortado, só 2 passes **[P0]**

**Objetivo:** a GPU não fica parada esperando o disco.

**Arquivos/funções:**
- `blender_worker.py` `main`/`configure_render`:
  - `use_crop_to_border = True`;
  - codec RLE (melhor que sem compressão por causa do antivírus no Windows);
  - grava só Combined + Shadow Catcher (com um nó File Output se o Blender 5.3 não permitir pular os passes "Noisy"; medir o custo).
- `compositor.py` `load_render_passes` + `mask_unrendered`: lê o recorte e cola num buffer de quadro inteiro usando o mesmo `border_px`.
- Teste de regressão da antiga "linha escura" de 1 px.
- `pipeline.py`: apaga cada EXR depois de compor.

**Testes:**
- `test_compositor.py::test_cropped_exr_equals_full_frame` (byte a byte).
- `::test_border_rounding_no_dark_line`.

**Validação em CPU:** renderizar 3 quadros recortados e 3 inteiros a 640 px; a composição deve ser idêntica.

**Efeito visível:** nenhum; ~40 s a menos em 1920 px.

---

### PT5 — Composição recortada, em paralelo e ainda bit a bit **[P0]**

**Objetivo:** a composição deixa de somar ~170 s e passa a rodar escondida atrás do render.

**Arquivos/funções:**
- `pipeline.py` `compose_video`:
  - (a) cache de `out8` para quadros reutilizados (não chamar `Compositor.composite` de novo);
  - (b) compor só dentro de `recorte = border + raio de influência (0,10 × diagonal) + margem do gaussiano` e colar numa cópia de `plate8`;
  - (c) verificação independente: `np.array_equal(out8[~mask], plate8[~mask])` em cada quadro;
  - (d) uma *thread* consumidora alimentada pelos eventos `render_progress` do worker compõe e codifica em ordem enquanto a GPU renderiza;
  - (e) `estimate_plate`/criação do `Compositor` durante o render.
- `mobilia/video/encoder.py`:
  - preset `medium`, ou NVENC quando `nvenc_available()`;
  - `decode_frames` com *seek* em 3 quadros (início, meio, fim), sem decodificar o arquivo todo.

**Testes:**
- `test_compositor.py::test_crop_composite_bit_identical` (protótipo `scratchpad/speed/crop_bench.py`: diferença máxima 0).
- `::test_verification_catches_crop_paste_bug` (injeta um erro de 1 px fora do recorte; deve falhar).
- `test_encoder.py::test_seek_verify`.
- `test_pipeline_e2e.py` com composição em paralelo.

**Validação em CPU:** `python -m mobilia.cli recompor --work-dir /tmp/claude-0/poc_sala3` antes e depois, com MP4 de
fundo idêntico e tempo de 47 s → < 10 s.

**Efeito visível:** nenhum na imagem; minutos a menos.

---

### PT6 — Render em camadas: marcos estáticos + quadro final + camada dinâmica **[P0, com chave de retorno ao modo atual]**

**Objetivo:** é o único caminho para 1920 px em 120 s, e ainda dá **mais** qualidade aos quadros que o
espectador olha mais tempo.

**Desenho:**
1. **Quadro final (*hold*)** renderizado primeiro, a 128 amostras, com o recorte de todos os móveis. Serve também de
   sonda de desempenho (PT7).
2. **Marcos (3–5)** seguindo as classes da coreografia:
   - móveis grandes;
   - mesas;
   - peças laterais;
   - decoração;
   - quadros de parede.

   Cada marco renderiza todos os itens já assentados, no recorte de todos os móveis, a 48–64 amostras.
   Nada de re-render por item: o modo incremental antigo foi desligado por costuras perto das paredes.
3. **Camada dinâmica:** em cada quadro com movimento, só as peças em movimento, no recorte delas (poses f−1..f+1 +
   margem de contato de 0,3 m + projeção da sombra do sol, como em `_bbox_points`), a 12–16 amostras, com desfoque
   de movimento.
   - As peças já assentadas ficam na cena como `is_shadow_catcher=True`: ocultam corretamente a peça nova e recebem a sombra e a luz rebatida dela.
   - Composição: `out = base_estática · S_din · (1 − A_din) + C_din`, onde `base_estática` é `placa · S' · (1 − A) + C` do último marco.
4. **Transição** de 3–4 quadros (*crossfade*) entre a última camada dinâmica e o marco seguinte, para não "pular" o ruído.
5. **Custo fixo por quadro:** medir primeiro na GPU. Se passar de 0,1 s, renderizar o intervalo animado numa única
   chamada `render(animation=True)` com um handler `frame_change_pre` que define recorte e visibilidade por quadro,
   ou manter a sessão e o denoiser persistentes.

**Arquivos/funções:**
- `blender_worker.py`:
  - `main` (laço de quadros): novo plano de quadros `plan_layers(items_rt)` → lista de (quadro, camada, recorte, amostras);
  - `dynamic_parts`, `incremental_border`;
  - `set_catchers` por item;
  - `frames.json` com `layer`, `base_source`.
- `compositor.py`: `LayerAccumulator` ganha `static_base` e `compose_dynamic`; a máscara de byte a byte é a união dos recortes.
- `choreography.py`: expõe `milestones(schedule)`.
- `pipeline.py` `compose_video`: segue os campos novos.
- Chave `render.camadas = true/false` em `scene.json` para voltar ao modo atual.

**Testes:**
- `test_compositor.py::test_layered_equals_single_layer_when_static` (sem movimento, as camadas têm de reproduzir o quadro único dentro de 1/255).
- `::test_layered_plate_bit_exact`.
- `test_choreography.py::test_milestones_cover_all_items`.
- Teste lento de Blender a 320 px: comparar o último quadro em camadas com o render único (diferença média < 2/255 nas regiões de CG, 0 fora).

**Validação em CPU:** sala de teste a 480 px, 8/32 amostras; inspecionar costuras nos limites dos recortes e nos quadros de transição.

**Efeito visível:** vídeo em ~2 min em 1920 px. Móveis parados com aspecto mais limpo e realista (128/64 amostras em vez de ~24).

---

### PT7 — Controle de orçamento de tempo **[P0 versão simples; P1 completa]**

**Objetivo:** ficar abaixo de 120 s em qualquer tamanho de foto e qualquer GPU.

**Arquivos/funções:**
- `blender_worker.py`, depois do render do quadro final:
  - mede amostras/s e custo fixo por quadro;
  - calcula a previsão do restante com o plano do PT6;
  - escolhe as amostras dinâmicas (12–24) e, só se preciso, a escala do CG em movimento (0,8–0,9; reamostra só RGBA/sombra);
  - grava tudo no relatório.
  - Fotos pequenas usam a sobra em supersampling do CG.
- `mobilia/ui/runner.py`: mostra a previsão.
- `P1`: o mesmo controle também ajusta as amostras dos marcos.

**Testes:** `tests/test_budget.py` (novo): função pura `choose_settings(throughput, overhead, plan, budget)` com
casos de foto pequena, foto grande e GPU lenta; nunca reduz o quadro final.

**Validação em CPU:** a função pura, mais uma execução real a 320 px onde o controle precisa escolher o mínimo.

**Efeito visível:** tempo previsível; previsão "≈ 1 min 40 s" na tela.

---

### PT8 — Luz e cor do CG iguais às da foto **[P0]**

**Objetivo:** acabar com o "mais escuro / mais azul / mais chapado que a sala".

**Arquivos/funções:**
- `blender_worker.py` `proxy_material`: mix por "Is Camera Ray".
  - Raio de câmera: catcher difuso, como hoje (mantém a regra validada "catcher emissivo dilui a própria sombra").
  - Qualquer outro raio: `Emission(foto_linear)`, força 1.
  - **`emission_sampling = 'NONE'`**, senão a malha de oclusores (~690 mil triângulos) entra na árvore de luzes.
  - Aplicar ao piso, às paredes, ao teto e aos oclusores.
- `calibrate`:
  - um caso extra "emissão" como termo fixo no NNLS;
  - **sol ajustado à parte** na máscara das manchas de `sunlight.py` (no teste o NNLS zerou o sol);
  - `AMBIENT_MAX_SHARE` adaptativo (0,5 → 0,7 em salas claras).
- Depois do refit:
  - cor global por canal aplicada nas luzes (hoje `color_ratio` é só diagnóstico);
  - grava a predição `pred` da sala vazia.
- `compositor.py`: mapa de ganho residual de baixa frequência `G = blur(máscara·foto/pred)/blur(máscara)` (σ = 0,1 do lado maior), limitado a [0,85; 1,2], multiplicando **só C**.
  - Se a máscara de calibração tiver poucos pixels (salas brancas acima de 0,97), cai para ganho global.
  - Registrar a fração da máscara.
- Curva de tom monotônica (luminância log, regressão isotônica) da predição linear para o código sRGB da foto,
  aplicada a C/A sem pré-multiplicação e pré-multiplicada de novo.
  - Identidade se o resíduo for alto.
  - Gravar a curva em `frames.json["calibration"]` para que `recompor` funcione.
- `scene_spec.py` `calibration_mask`: pixels quase saturados entram como limite inferior em vez de serem descartados.
- Leve compressão de altas luzes só no CG (joelho em 0,8 linear).
- **Manter** a transformação de vista Standard (AgX/Filmic quebraria a correspondência).

**Testes:**
- `test_compositor.py::test_gain_map_only_touches_cg` (byte a byte fora).
- `::test_gain_map_fallback_bright_room`.
- `::test_tone_curve_identity_on_bad_fit`.
- Teste de Blender lento: `rel_rms` da calibração ≤ 0,13 na sala de teste (medido 0,213 → 0,107).

**Validação em CPU:** `scratchpad/realism/exp_render.py` a 640 px / 24 amostras, com A/B antes/depois, mais um caso
sintético de sala branca clara.

**Efeito visível:** móveis com o mesmo brilho, temperatura de cor e luz rebatida da sala. É o maior ganho de "foto real".

---

### PT9 — Ativos sem cara de CG (custo zero de render) **[P0: a, b, c; P1: d, e]**

- **(a) Quadros de parede reais.** Novo gerador procedural de arte abstrata na paleta (numpy, sem risco de
  licença) e moldura procedural `canvas_art` (moldura fina + passe-partout ou tela de 3,5 cm) com largura e
  altura livres.
  - Tirar os quadros de parede modernos 01/02 (o 01 tem o cartão cinza "Ray Homes", o 02 tem bordas esticadas) do estilo moderno.
  - Usar as molduras "fancy" 01/02 da Poly Haven (pinturas reais escaneadas, CC0) no estilo clássico.
  - Opcional: pacote de 30–40 obras CC0/domínio público com manifesto de licença.
  - Recortar a arte na proporção da UV.
  - Arquivos: `assets/procedural.py`, `layout/catalog.py`, `layout/solver.py` `_place_one_art` (tamanho livre, ~2/3 da largura do móvel abaixo), `blender_worker.import_gltf_item` (troca da imagem do material `*artwork*`).
- **(b) Estofado claro em sala clara.** Regra em `layout/style.py` `asset_preference`/`rank_assets`.
  - Se a luminância média da parede for alta: preferir sofá 01, poltrona 01, cadeira verde 01 e similares.
  - Excluir sofá 02, pufe 01 e poltrona moderna 01 (albedo quase preto).
  - No máximo uma massa escura; acentos ~10%.
- **(c) Tapete com textura real.** Placa chanfrada de 1–2 cm com textura CC0 de tecido/carpete (2k) no lugar do
  padrão vetorial; borda que projeta uma linha fina de sombra. Arquivo: `assets/procedural.py` `_build_rug`, `assets/materials.py`.
- **(d) [P1] Máscaras de estofado feitas offline** por ativo (PNG ao lado do ativo, revisadas à mão), tingimento
  pela paleta preservando a textura, rugosidade ~0,85 e brilho de tecido. A versão automática por texel falhou
  (manchas), por isso é offline.
- **(e) [P1] Bisel de 2 mm + normais ponderadas** em ativos com menos de 1 500 triângulos (mesa de madeira 02, criado-mudo pintado, cadeira pintada).

**Testes:**
- `test_procedural_assets.py::test_canvas_art_sizes`, `::test_art_generator_deterministic`.
- `test_layout.py::test_bright_room_prefers_light_upholstery`, `::test_no_placeholder_art_in_catalog`.
- `test_on_top.py`: o padrão de tamanho real vale para qualquer ativo novo (M7).

**Validação em CPU:** render do quadro final a 640 px, com recorte do quadro e do sofá antes/depois.

**Efeito visível:** some o "retângulo cinza"; o sofá combina com a sala branca; o tapete parece tecido.

---

### PT10 — Sombra de contato analítica **[P0]**

**Objetivo:** recuperar a linha escura de contato que toda foto real tem. A emissão da foto (PT8) suaviza as
sombras, por isso os dois PTs vão juntos.

**Arquivos/funções:**
- `blender_worker.py`, na importação: silhueta ortográfica de cima de cada item, ou o polígono da base, gravada no `scene.json`.
- `compositor.py`: projeta a silhueta no piso pela homografia piso→imagem conhecida e desfoca pela altura atual da
  peça acima do piso (vem da pose da animação). Fica `S' = S · lerp(1, contato, k)` com k ≈ 0,6–0,8, **só dentro da região de influência que já existe**.
- Sem segunda camada de render por quadro (dobraria o custo fixo).
- Opcional P1: AO real uma vez por marco na camada estática.

**Testes:**
- `test_compositor.py::test_contact_shadow_inside_influence_only`.
- `::test_contact_shadow_fades_with_height` (peça subindo/caindo).

**Validação em CPU:** quadro final + 3 quadros da queda do sofá a 640 px.

**Efeito visível:** móveis "pousados" no piso, e não flutuando.

---

### PT11 — Novas entradas cinematográficas: surgir do chão, entrar pelas laterais, montar peça por peça **[P0: a, b; P1: c]**

Requisito novo do usuário. A animação atual (queda + assentamento) continua; as novas se somam por papel do móvel.
Nenhuma alonga a linha do tempo.

- **(a) `emerge` (surgir do chão)** em `choreography.py`:
  - dz de −(altura da peça) até 0, com `ease_out_quint`, pequeno *overshoot* e assentamento.
  - O catcher do piso já esconde tudo abaixo de z = 0, então o corte no chão sai de graça.
  - Faixa de dissolução de 3–5 cm na linha do piso em `_wrap_fade` (limiar em Z no espaço do objeto).
  - Usar em pés/bases, poste de luminária (base → poste que cresce → cúpula que desce), estantes e armários altos.
- **(b) `glide` (entrar pela lateral)**:
  - aproximação lateral de 0,5–1,0 m pelo lado da borda mais próxima da imagem, flutuando 1–2 cm, com `ease_out_cubic`, pouso e assentamento amortecido;
  - o *fade* termina nos primeiros 30%;
  - o caminho varrido é validado em `solver._finalize` contra a grade de ocupação (só FREE/UNKNOWN), as paredes (nunca atravessar a proxy) e os itens anteriores. Senão, volta para `drop`;
  - movimento e direção gravados em `layout.json`;
  - por papel: cadeiras, poltronas, mesas laterais e pufes deslizam; tapete desenrola; plantas crescem; quadros deslizam; decoração pequena cai ou assenta.
- **(c) [P1] Montagem real de ativos de malha única** (`blender_worker.import_gltf_item`, com cache por ativo):
  - separar as ilhas soltas e agrupar em 3–5 grupos (pés, tampo/assento, encosto/braços, portas/gavetas);
  - entregar para `plan_parts` com etiquetas;
  - se os grupos forem ambíguos (ex.: mesa de ripas), voltar para a peça inteira.
  - Sofá, cama, mesas e aparadores fazem: pés surgem → corpo/tampo desce → almofadas assentam → portas fecham.

**Testes:**
- `test_choreography.py::test_emerge_starts_below_floor_ends_at_rest`.
- `::test_glide_path_free`.
- `::test_new_motions_keep_timeline`.
- `test_layout.py::test_glide_fallback_when_blocked`.
- (P1) `::test_assembly_clusters_bounded`.

**Validação em CPU:** prévia a 320 px / 4 amostras da sala de teste (~10 min em CPU), com folha de contato dos quadros.

**Efeito visível:** as três formas de entrada pedidas, suaves, com a peça cortada limpa no piso. Ponto em aberto
para o usuário: corte no piso ou dissolução (ver seção 6).

---

### PT12 — Designer profissional: regras e programas por cômodo **[P1; P0 apenas o item 5]**

**Arquivos:** `layout/rules.py`, `layout/catalog.py`, `layout/solver.py`, `layout/style.py`, `analysis/depth_mesh.py`, `ui/app.py`.

1. **Regras rígidas de tapete:**
   - sala: comprimento ≥ largura do sofá + 0,30 m, com os pés dianteiros de todos os assentos sobre ele;
   - quarto: 45–60 cm além das laterais da cama;
   - jantar: ≥ 60–75 cm além da mesa em todos os lados;
   - novos tamanhos 2,0 × 3,0, 2,5 × 3,5 e 3,0 × 4,0, mais um tapete redondo para mesa redonda.
2. **Programa proporcional à área livre:** acima de 18 m² entram uma segunda poltrona ou banco e um aparador; abaixo de 9 m² saem as poltronas.
3. **Programas por cômodo:**
   - sala: cortinas laterais, manta, trio da mesa de centro (bandeja + pilha de livros + vaso);
   - jantar: pendente apagado, aparador, centro de mesa;
   - cozinha: fruteira, tábua, jogo de chá;
   - banheiro: toalhas, tapete de banho, banquinho;
   - escritório: cadeira de escritório de verdade, luminária de mesa;
   - quarto: banco no pé da cama.
4. **Proporções:** mesa de centro com ~2/3 da largura do sofá e altura até 5 cm da altura do assento (termos suaves).
5. **[P0]** **Narrativa de designer** em `layout.notes`, exibida na interface. Exemplo: "Sala clara, piso porcelanato
   claro, pé-direito 2,6 m, 12 m² livres. Escolhi um sofá de linho de 2,2 m na parede P1, tapete 2,4 × 1,7 m e quadro de 1,4 m centrado a 1,50 m."
6. **Oclusão de lustres:**
   - incluir `ceiling_fixture` em `OCCLUDER_GROUPS`, ou holdout geral por profundidade: passe Z no EXR e, no
     compositor, A = C = 0 onde o CG fica mais de 5% atrás da profundidade de um objeto existente na foto
     (dilatado, concordância 3 × 3);
   - quadros não podem cobrir lustres nem a "desordem de parede" (pixels fora da mediana local: ar-condicionado, interruptores).
7. **[P1] Sofá e poltrona procedurais modernos** com almofadas chanfradas, leve afundamento e pés finos; largura
   ~2/3 da parede livre (1,8–2,6 m); tecidos CC0 2k.

**Testes:** `test_layout.py` com casos por regra (tapete × sofá, programa por área, narrativa presente) e
`test_compositor.py::test_depth_holdout_only_removes_cg`.

**Validação em CPU:** as 30 composições do `test_on_top` mais a sobreposição do layout na foto de teste.

**Efeito visível:** ambientes completos e coerentes, como se um designer tivesse montado; a IA explica as escolhas.

---

### PT13 — Distorção da lente **[P1]**

- Novo `mobilia/analysis/lens.py`:
  - estima k1 (e k2 opcional) pelo método de linhas retas, usando segmentos longos e fronteiras piso/parede/teto da segmentação, no modelo de divisão;
  - aceita só se a retidão melhorar mais de 30% e |k1| passar de um limiar.
- A análise roda numa **cópia** desdistorcida (só entrada da análise, nunca saída).
- O CG é renderizado em lente ideal com 6–8% de sobra de borda (`setup_camera`).
- No compositor, `cv2.remap` bicúbico **só de C, A e S** para a geometria distorcida da foto, **antes** de montar a máscara de byte a byte.
- k1 vai para o relatório.

**Testes:**
- `tests/test_lens.py` com distorção sintética conhecida (recuperar k1 ± 10%) e imagem sem distorção (k1 = 0, desligado).
- `test_compositor.py::test_lens_warp_plate_bit_exact`.

**Validação em CPU:** foto de teste com distorção sintética aplicada.

**Efeito visível:** móveis nas bordas acompanham a curvatura das linhas da sala, sem "flutuar".

---

### PT14 — Sobrepor etapas: análise ao abrir a foto, Blender em modo servidor **[P1]**

- `mobilia/ui/app.py` + `runner.py`: a análise começa **quando a foto é carregada**, antes de o usuário escolher o
  cômodo, porque não depende dele. Esconde 15–40 s do tempo percebido.
- Worker Blender em modo servidor (stdin/JSON com enquadramento binário seguro):
  - abre na hora, configura a GPU e renderiza 16 × 16 para carregar kernels;
  - espera o `scene.json`;
  - só sobe a cena para a GPU depois de liberar os modelos e chamar `torch.cuda.empty_cache()`, mantendo a política de um modelo por vez nos 8 GB.
- `analysis/geometry.py` `estimate_geometry`:
  - construir o modelo sem inicialização aleatória (dispositivo `meta` + `load_state_dict(assign=True)`; conferir os *buffers* não persistentes);
  - checkpoint em safetensors;
  - `np.save` em vez de `savez_compressed`.

**Testes:**
- `test_pipeline_e2e.py::test_analysis_cached_by_photo_hash`.
- Teste do protocolo do servidor com `fake_worker.py`.
- Teste de modelo (marcado `models`): saída idêntica com e sem a construção rápida.

**Validação em CPU:** a carga a quente do modelo de geometria cai de 8,8 s para ~2,5 s (medido: 6,5 s eram inicialização desperdiçada).

**Efeito visível:** o vídeo sai segundos depois da escolha do cômodo.

---

### PT15 — Reflexo dos móveis no piso brilhante **[P2 — depende de decisão do usuário]**

- Detectar piso brilhante (realces e reflexos das janelas no piso; `style.py` já classifica `piso_frio`).
- Só nas camadas estáticas: uma camada de reflexo a meia resolução, 8–16 amostras, desfocada pela rugosidade estimada.
- Composição no piso: `out = placa·S'·(1−A)·(1−f·Ra) + f·R + C`, limitada à região de influência.
- **Questão de regra:** o reflexo altera pixels do piso fora da silhueta e da sombra de contato. Fica desligado por
  padrão até o usuário confirmar que reflexo conta como "efeito do móvel" (seção 6).

### PT16 — Catálogo ampliado **[P2]**

- De ~40 para até 85 móveis + objetos CC0 não usados (pendente, luminária de braço, vasos de latão e cerâmica,
  fruteira, jogo de chá, bancos, relógio).
- Cada novo ativo com tamanho real medido no Blender, `yaw_fix` e superfície de apoio medida (padrão do
  `test_on_top`; M7), para o problema dos livros nunca voltar.

### Descartado por enquanto

Harmonização generativa por quadro, ou mesmo em um quadro só. Custaria 10–20 s do orçamento, mais 2–4 GB de
download e troca de VRAM, tem restrições de uso e ganho não comprovado sobre os PTs 8–10. Nunca colar pixels gerados.

## 4. O que só pode ser validado na GPU do usuário, e a telemetria necessária

**Só valida na RTX 5070 Laptop:**
1. Desempenho real do Cycles/OptiX nesta cena (amostras/s). A estimativa atual é ~22 M/s, ±2×.
2. Custo fixo por chamada de render na GPU (sessão, filme, denoiser, leitura, gravação). Isso decide se o PT6 precisa de `render(animation=True)`.
3. Se o denoiser roda de fato na GPU Blackwell ou cai silenciosamente para a CPU.
4. Qualidade de 12–16 amostras + denoiser nas peças em movimento, sem cintilação, em sala branca com piso brilhante.
5. Costuras entre marcos e camada dinâmica e a transição de 3–4 quadros.
6. Ruído do passe de sombra com a emissão da foto em poucas amostras.
7. Tempo de compilação de kernels na primeira vez e eficácia do aquecimento no instalador.
8. Uso de VRAM: modelos de análise + contexto do Blender ocioso + cena (com a reserva do Windows/tela).
9. CPU do notebook para composição/codificação em paralelo, NVMe e antivírus (cargas frias, EXR).
10. NVENC disponível e qualidade equivalente ao x264.

**Campos novos no `report.json` (PT1):**
```
tempos_s: { inicio_motor, carga_foto, analise: {geometria_carga, geometria_inferencia, segmentacao_carga,
            segmentacao_inferencia, sala, sol, oclusores}, layout, ativos, download, blender_inicio,
            blender_kernels, cena_montagem, calibracao, render_final, render_marcos, render_dinamico,
            composicao, codificacao, verificacao, total, total_percebido }
primeira_execucao: bool
foto: { original, placa_video, megapixels }
gpu: { nome, driver, vram_total_mb, vram_pico_mb: {analise, render} }
render: {
  dispositivo, denoiser_dispositivo, optix: bool,
  desempenho: { amostras_por_s, custo_fixo_quadro_s },      # medidos no quadro final
  orcamento: { previsto_s, spp_dinamico, spp_marcos, escala_cg, motivo },
  camadas: { final: {area_mp, spp, s}, marcos: [{itens, area_mp, spp, s}], dinamico: {quadros, area_media_mp, spp, s} },
  por_quadro: [ {quadro, camada, area_px, spp, amostragem_s, denoise_s, gravacao_s, parede_s} ],
  quadros: { total, renderizados, reutilizados, so_foto }
}
calibracao_luz: { rel_rms, ganhos_canal, fracao_mascara, sol_ajustado, curva_tom_ok }
lente: { k1, melhora_retidao, aplicada }
composicao: { s_por_quadro_medio, recorte_medio_px, paralela: bool }
encode: { codificador, preset, bytes, s }
preservacao: { preservado_exato_todos, quadros_verificados, psnr_fundo_db }
```
Pedir ao usuário, além do novo relatório:
- o `report.json` da execução de 8 min (`foto.original`, `placa_video`, `render.tempos`, `tempos_s`, `encode.bytes`);
- o tempo de uma segunda execução na mesma foto.

## 5. Riscos

| Risco | Mitigação |
|---|---|
| Custo fixo por quadro na GPU (0,2–0,6 s) estoura o orçamento mesmo com poucas amostras | Medir primeiro (PT1). Plano B: `render(animation=True)` + `frame_change_pre`, sessão persistente. |
| Costuras ou "pulo" de ruído entre camadas (PT6) | Marcos com recorte de todos os móveis, não por item. *Crossfade* de 3–4 quadros. Chave para voltar ao modo atual. A/B em GPU. |
| Poucas amostras + denoiser deixam o CG manchado ("cara de CG") | Poucas amostras só no que se move (o desfoque de movimento esconde). Quadro final a 128, marcos a 48–64. |
| Emissão da foto suaviza demais as sombras | PT10 junto com o PT8; sol ajustado à parte. |
| Muitos emissores na árvore de luzes | `emission_sampling = 'NONE'` na proxy. |
| Mapa de ganho ou curva de tom errados em fotos extremas (noite, luz colorida) | Limites estreitos, volta à identidade com resíduo alto, fração mínima de máscara. |
| Estimativa errada de distorção da lente | Exigir melhora de retidão > 30%; senão, desligado. |
| Heurística de montagem agrupa mal | Máximo de 3–5 grupos; volta para a peça inteira. |
| `glide` atravessa parede ou móvel | Validar o caminho varrido no layout; senão, `drop`. |
| VRAM de 8 GB | Um modelo por vez; cena só depois de `empty_cache`; registrar o pico. |
| Duas pessoas editando `blender_worker.py` ao mesmo tempo | Integrar o PT0 primeiro; PTs de render em sequência (PT4 → PT6 → PT7). |
| Licenças | Só CC0/domínio público com manifesto; arte procedural por padrão; sem pesos não comerciais. |
| Reflexo no piso fere a regra | Desligado até decisão explícita do usuário. |
| Fotos pequenas (~0,5 MP) parecem suaves no celular | Avisar abaixo de 1 280 px. Nunca ampliar a placa; supersampling só do CG. |

## 6. Decisões pendentes do usuário

1. Reflexo dos móveis no piso brilhante pode alterar pixels do piso junto aos móveis? (PT15)
2. "Surgir do chão" com corte limpo no piso ou com dissolução? (prévia curta)
3. Pode incluir no instalador ~30 obras de arte de domínio público (~10–15 MB), ou só arte procedural?
4. Fotos reais adicionais (grande angular, lustre, piso brilhante) para validação.

## 7. Ordem de execução sugerida

PT0 → PT1 → PT2 → PT3 → PT4 → PT5 → PT8 → PT9 (a–c) → PT10 → PT6 → PT7 (simples) → PT11 (a, b) → PT12 (5)
→ **versão de teste para o usuário** → PT14 → PT13 → PT11 (c) → PT12 → PT9 (d, e) → PT15/PT16.
