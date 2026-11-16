# Arquitetura do Mobília Studio

> Documento vivo. Decisões tomadas e **validadas com testes reais** neste repositório.
> A pesquisa comparativa de tecnologias está em [`PESQUISA_TECNOLOGIAS.md`](PESQUISA_TECNOLOGIAS.md).

## 1. Objetivo e regra absoluta

Entrada: **uma foto** de um ambiente vazio + **tipo de ambiente** (sala, quarto, cozinha, banheiro,
escritório, sala de jantar). Saída: **MP4** em que os móveis surgem e se montam progressivamente
**dentro da foto original**, com câmera travada.

**Regra absoluta:** a foto é uma placa de fundo imutável. Nenhum pixel fora das regiões cobertas
por móveis (e pelas sombras de contato que eles projetam) pode mudar. O software **não** gera uma
nova foto decorada, **não** usa difusão/inpainting sobre a foto e **não** move a câmera.

Como garantimos isso (e verificamos em todo quadro):

1. Os móveis são **objetos 3D reais** renderizados em camadas independentes (RGBA pré-multiplicado).
2. As sombras vêm do passe **Shadow Catcher** do Cycles: um fator multiplicativo `S` aplicado
   **só** na vizinhança dos móveis.
3. O compositor calcula `saída = foto·S'·(1−A) + C` e, em seguida, **copia os bytes originais da
   foto** em todo pixel com `A = 0` e `S' = 1`. Cada quadro é conferido (`max_diff_outside == 0`);
   se algum pixel "intocado" divergir, o trabalho aborta com erro. O relatório final (`report.json`)
   registra quantos pixels mudaram por quadro.
4. Quadros sem móveis visíveis (o início do vídeo) são cópias exatas da foto.

> O MP4 (H.264/yuv420p) é um formato com perda. A garantia bit a bit vale para os **quadros mestres**
> (compostos antes do encode). O encode usa CRF 14 com conversão RGB→YUV **BT.709 explícita**, e o
> relatório mede o PSNR do vídeo decodificado contra os quadros mestres (fundo intocado).

## 2. Visão geral do pipeline

```
foto.jpg ─┬─► [Preparo]   EXIF, placa imutável na resolução do vídeo (redução única se > perfil)
          ├─► [Análise]   MoGe-2 (profundidade métrica + intrínsecos + normais)
          │               OneFormer ADE20K (piso/parede/teto/janela/porta/obstáculos)
          │               RANSAC: plano do piso, altura da câmera, paredes (retas 2D),
          │               teto, janelas/portas como retângulos nas paredes, grade de ocupação
          ├─► [Layout]    programa de móveis por ambiente + estilo/paleta da foto +
          │               otimização com regras de design de interiores e visibilidade na câmera
          ├─► [Assets]    Poly Haven (CC0, glTF, cache local) + procedurais (tapete, luminária, cama, livros...)
          ├─► [Render]    subprocesso Blender (bpy 5.x, Cycles OptiX/CUDA):
          │               câmera casada, sala-proxy = shadow catcher, luz calibrada pela foto,
          │               montagem animada peça a peça, EXR multicamada por quadro
          ├─► [Composição] saída = foto·S'·(1−A)+C ; cópia bit a bit fora dos móveis/sombras;
          │               grão do sensor e nitidez casados só na camada 3D
          └─► [Vídeo]     FFmpeg (libx264 CRF 14 ou NVENC), BT.709, +faststart; verificação por decode
```

Código: `mobilia/` — `analysis/` (geometria, segmentação, sala), `layout/` (catálogo, solver, estilo),
`assets/` (Poly Haven, procedurais), `render/` (coreografia, especificação da cena, worker do Blender),
`composite/`, `video/`, `pipeline.py`, `cli.py`, `ui/` (PySide6), `hardware.py`.

## 3. Hardware e perfis

A análise do hardware é feita **na máquina do usuário** (este repositório foi desenvolvido em
contêiner Linux sem GPU): `python -m mobilia.cli hardware` ou `scripts/diagnostico_hardware.ps1`
(PowerShell puro, sem instalar nada). O perfil é escolhido automaticamente pela VRAM:

| Perfil | VRAM | Geometria | Render | Resolução máx. | Amostras |
|---|---|---|---|---|---|
| ultra | ≥ 11 GB | MoGe-2 ViT-L fp16 | OptiX | 2560 px | 192 |
| alta | 7,5–11 GB | MoGe-2 ViT-L fp16 | OptiX | 1920 px | 128 |
| média | 5,5–7,5 GB | MoGe-2 ViT-L fp16 (tokens reduzidos) | OptiX | 1920 px | 96 |
| econômica | < 5,5 GB | MoGe-2 ViT-B | OptiX/CUDA | 1600 px | 64 |
| cpu | sem GPU NVIDIA | MoGe-2 ViT-B | CPU | 1280 px | 32 |

**Gestão de VRAM:** nunca há dois modelos na GPU ao mesmo tempo. MoGe-2 → libera → OneFormer → libera →
Blender roda em **outro processo** (a VRAM do PyTorch já foi devolvida). Em GPUs RTX o Cycles usa OptiX;
o denoiser OIDN roda na GPU.

**Render incremental (experimental, desligado):** re-renderizar só a região das peças em movimento
reduziu a área renderizada em ~14% na cena de teste. Mas sombras suaves junto à parede ultrapassam a
região e deixaram emendas sutis, então a qualidade venceu e o modo fica desligado (`render.incremental`).

**Otimizações de render (implementadas):**
- *border* por quadro: só a região da tela ocupada pelos móveis (+ margem de sombra) é renderizada;
- quadros estáticos são reaproveitados (assinatura de pose; o "hold" final custa 1 render);
- quadros sem móveis não são renderizados (são a própria foto);
- semente fixa (`use_animated_seed = False`) → ruído estável, sem "fervura" entre quadros;
- `use_persistent_data` entre quadros; amostragem adaptativa + denoise.

## 4. Análise da cena (validado na foto de teste)

- **MoGe-2 ViT-L** (Microsoft, MIT): nuvem de pontos métrica, intrínsecos (FOV), normais.
  Na foto de teste: FOV horizontal 68,2°, profundidade 2,1–5,1 m; 25 s em CPU (≈1 s em RTX).
- **OneFormer ADE20K Swin-L** (SHI Labs, MIT): 150 classes agrupadas em piso/parede/teto/janela/
  cortina/porta/estrutura/instalações/móveis.
- **Piso:** RANSAC com prior de normal vertical → altura da câmera.
- **Referencial da sala:** Z para cima, piso em Z = 0, eixo Y alinhado à direção de Manhattan das paredes.
- **Paredes:** RANSAC sequencial de retas 2D (vista de cima) com consistência de normais; extensão e
  altura por percentis; fusão de segmentos colineares.
- **Janelas/portas:** componentes conexos da segmentação, atribuídos à parede vizinha e convertidos em
  retângulos 3D por interseção raio-plano.
- **Escala métrica:** pistas (porta ≈ 2,05 m; pé-direito plausível) corrigem a escala do MoGe só quando
  há evidência forte (fator limitado a 0,75–1,33).
- **Resultado na foto de teste:** 4 paredes (resíduo 0,6–3,9 cm), 4 janelas alinhadas à imagem,
  câmera a 1,63 m, pé-direito 2,45 m (ver `tests/fixtures/rooms/stafford/overlay_debug.jpg`).

## 5. Render 3D sobre a foto (validado com testes reais no Blender 5.3)

Comportamentos do Cycles **verificados empiricamente** (`tests/test_render_camera.py` e experimentos
documentados no histórico):

| Fato | Consequência no projeto |
|---|---|
| Shadow catcher recebe sombras de luzes do Blender e do *world* | luzes reais da sala = luzes de área nas janelas + emissão das paredes não fotografadas |
| Catcher **emissivo** dilui a própria sombra | o material da sala-proxy é **difuso, nunca emissivo** |
| Malha emissiva **não-catcher** é tratada como objeto CG (não gera sombra no catcher) | paredes de fechamento são catchers |
| Catcher oculta objetos atrás dele | paredes-proxy fazem a oclusão correta (móvel atrás de quina some) |
| Blender 5: EXR multicamada = multi-part (`media_type = 'MULTI_LAYER_IMAGE'`) | leitor próprio com `OpenEXR.File(..., separate_channels=True)` |
| Câmera: `lens = fx·36/W`, `shift_x = −(cx−W/2)/max(W,H)`, `shift_y = (cy−H/2)/max(W,H)` | erro de projeção < 0,35 px no teste automatizado |

**Sala-proxy:** piso, teto e paredes (com recortes nas janelas) como shadow catchers. O albedo é a
**própria foto projetada pela câmera** (coordenada *Window*), dessaturada e dividida pela cor do
iluminante. Assim a luz rebatida nos móveis tem a cor real do ambiente. As paredes não fotografadas
(atrás da câmera) fecham a sala.

**Luz calibrada pela foto:** dois renders rápidos (240 px) da sala **sem móveis**: um só com as janelas
e outro só com o "ambiente" (paredes não fotografadas). Um ajuste por mínimos quadrados não negativos
casa a luminância renderizada com a da foto em piso/parede/teto, e o ambiente fica limitado a 50% da
luz. A cor da luz é a do iluminante da foto (teto ou paredes claras), então móveis brancos ficam com o
mesmo "branco" da foto. Na foto de teste, o erro relativo do ajuste foi de 20–29%. É um modelo de luz
simples, mas a exposição e o balanço de branco dos móveis ficam coerentes com a foto.

**Sol direto (validado na foto de teste):** manchas de sol no piso são detectadas e a direção do sol é
estimada por busca global (azimute × elevação). O critério é que as partes transparentes das janelas,
projetadas no piso, expliquem as manchas (F0,5 entre projeção e manchas; precisão ≥ 60%). Na foto de
teste, o resultado foi azimute ~139° e elevação ~51°, consistente em 1536 e 1920 px, com 95–99% da
área das manchas explicada. A luz solar entra pelos recortes das janelas e sua intensidade é a 3ª
base da calibração. Sem manchas confiáveis, o sol não é usado.

**Objetos que já existem na foto** (armários, eletrodomésticos, porta aberta, colunas) viram uma malha
de oclusão triangulada a partir da nuvem de pontos do MoGe-2, recuada 1,2% ao longo do raio. Ela também
é *shadow catcher*: oculta os móveis atrás dela e recebe as sombras dos móveis à frente (validado na
foto da cozinha de teste).

**Correções automáticas de assets:** vidro de quadros/vitrines vira "vidro arquitetônico" (senão o
Cycles bloqueia a luz e a arte fica preta); peças com frente fora de −Y recebem `yaw_fix`.

**Decoração sempre apoiada:** depois de gravar a animação, o worker confere na pose final cada item
"on_top" sobre um tampo: uma grade 5 × 5 de raios sob a base do objeto tem de encontrar a malha do
móvel de apoio a até 8 mm em ≥ 90% dos raios (em tampos ripados, um ponto sobre a fresta conta se fica
dentro do fecho convexo dos acertos do tampo ao redor; o que passa da borda externa não). Se não
encontrar, o item sai da cena, com aviso no log e em `report.json` (`render.itens_removidos`). Sobre
mesas de centro, escrivaninhas e racks os livros são uma pilha procedural de 2–3 livros deitados; a
fileira de enciclopédias em pé fica só para estantes.

## 6. Animação (um único estilo, refinado)

`mobilia/render/choreography.py` define tempos e curvas, e o worker grava keyframes **por quadro**:
o vídeo mostra exatamente essas funções.

- Ordem: tapete, depois peças grandes de trás para frente, depois médias, luminárias e plantas,
  quadros e por fim a decoração pequena (almofadas, vaso, bowl, livros).
- Sobreposição: cada peça começa quando a anterior completou ~55% da sua duração.
- Movimentos:
  - **queda** com aceleração e toque com squash sutil (≤ 2%), acomodação por mola amortecida;
  - **crescimento** das plantas (ease-out-back + balanço);
  - **deslizar** dos quadros rente à parede;
  - **desenrolar** do tapete.
- Fade-in: a camada 3D só fica opaca nos primeiros instantes da queda. É feito por material
  (Transparent mix por objeto), então a sombra acompanha a opacidade.
- Montagem peça a peça: base do sofá e depois o assento; vaso da planta, terra, caule e folhas;
  corpo do rack e depois as portas; livros em até 3 levas (0,10 s entre elas).
- Decoração sobre tampo ou assento cai de só 12 cm, em 0,13 s: nenhum objeto fica "meio no ar"
  por mais de ~0,35 s.
- Motion blur do Cycles (obturador 180°), câmera travada, *hold* de 0,8 s no início e 2,5 s no fim.

## 7. Composição e casamento com a câmera

- Grão do sensor: o ruído estimado nas regiões planas da foto é aplicado **só** sobre o CG, com um
  padrão **fixo** entre quadros (a foto é estática; grão variando faria o móvel "chiar").
- Nitidez: um desfoque equivalente ao da foto é estimado e aplicado ao CG.
- Sombra: o desvio `S−1` passa por um encolhimento suave (ruído de render < 0,6% vira 0). Ele é
  ponderado por uma janela que decai suavemente até 10% da diagonal em volta dos móveis e fica
  limitado a +15% de ganho.

## 8. Interface, empacotamento e lote

- A UI (PySide6) segue o fluxo importar foto → escolher ambiente → gerar → exportar. O pipeline roda
  sempre num **processo separado**, com eventos JSON linha a linha (`mobilia/jobs.py`). Isso dá
  isolamento de falhas, liberação de VRAM e cancelamento seguro.
- Lote: a CLI (`python -m mobilia.cli run ...`, `python -m mobilia.cli batch --dir fotos/ --out videos/`
  com CSV opcional `arquivo,ambiente`) e a fila da UI usam o mesmo contrato. Cada trabalho tem sua pasta
  com todos os intermediários e um `report.json`.
- Uso offline: `python -m mobilia.cli prefetch` baixa os pesos dos modelos e todos os assets do
  catálogo; depois disso nada mais é baixado.
- Empacotamento Windows: um `.exe` leve (UI) e um *runtime* pesado (Python + PyTorch CUDA + Blender
  bpy + modelos), instalado no primeiro uso em `%LOCALAPPDATA%\MobiliaStudio`. Detalhes em `packaging/`.
- Telemetria (`mobilia/telemetry.py`): `report.json` traz `tempos_s` com todas as subetapas da seção 4
  do `PLANO_V2.md` (`inicio_motor`, `carga_foto`, `analise.{geometria_carga, geometria_inferencia,
  segmentacao_carga, segmentacao_inferencia, sala, sol, oclusores, gravacao}`, `layout`, `ativos`,
  `download`, `blender_inicio`, `blender_kernels`, `cena_montagem`, `calibracao`, `render_final`,
  `render_marcos`, `render_dinamico`, `blender_fim`, `composicao`, `codificacao`, `verificacao`,
  `relatorio`, `total`, `total_percebido`). As etapas de primeiro nível particionam o total (a soma fica
  dentro de 5%, testado). `total_percebido` conta a partir do clique em "Gerar" (a interface passa o
  instante em `MOBILIA_T_CLIQUE`/`MOBILIA_T_INICIO`). `primeira_execucao` vem do estado da instalação
  (`setup_state.json`, contador `execucoes`), e `primeira_execucao_custos` separa download, kernels e
  carga dos modelos.
  - O worker do Blender emite `worker_ready` com o dispositivo efetivo do Cycles (OptiX/CUDA/CPU, nomes
    das GPUs) e do denoiser (a mesma consulta `has_oidn_gpu_devices` que o Cycles usa para cair na CPU),
    faz um render de aquecimento de ~16 px (kernels de render e do denoiser, medidos à parte) e, por
    quadro, lê as mensagens do handler `render_stats`: sincronização, carga de kernels, amostragem,
    denoise (estimado pelo ritmo das amostras, porque o Cycles só informa "Sample N/N" depois do
    denoiser), gravação ("Saving") e memória. Cada quadro tem uma camada: `dinamico` (peça em
    movimento), `final` (cena completa parada) ou `marco` (parado com parte dos itens).
  - `render.desempenho` (amostras/s nos quadros finais, custo fixo por quadro), `render.camadas`,
    `render.por_quadro`, `render.quadros` (total, renderizados, reutilizados, só foto), cobertura média
    do recorte, `composicao.por_quadro` (leitura, composição, envio ao FFmpeg), `encode` e `gpu`
    (pico de VRAM do dispositivo por fase via NVML, e o do PyTorch na análise).
  - "Copiar diagnóstico" (tela de conclusão e caixa de erro) gera um zip com `report.json`,
    `frames.json`, `scene.json`, `layout.json`, os registros e um manifesto, e copia o arquivo para a área
    de transferência. A foto e tudo o que deriva dela (placa, prévias, sobreposições) só entram se o
    usuário marcar "Incluir a foto". A tela de conclusão mostra o tempo por etapa.

## 9. Confirmação independente (pesquisa no código-fonte do Cycles)

A pesquisa técnica (`PESQUISA_TECNOLOGIAS.md`) leu o código-fonte do Cycles e confirmou, sem
contradições, os comportamentos medidos aqui:
- `Combined` contém só o CG, com alfa pré-multiplicado; o passe Shadow Catcher é a razão
  (completo − matte) / só-catcher, e a composição é `C.rgb + (1 − C.a)·foto·SC`;
- catcher emissivo dilui a sombra;
- luzes criadas via Python contam como luz "real" e o *world* também.

Melhoria futura sugerida: material de catcher com `Light Path › Is Camera Ray` (difuso para a câmera,
emissão da foto para os demais raios), para os móveis refletirem o ambiente real sem diluir sombras.

## 10. Pastas de um trabalho

```
<trabalho>/
  plate.png               placa imutável (referência bit a bit)
  analise/                geometry.npz, segmentation.npz, room.json, grid.npy, analise_overlay.jpg
  layout.json             itens, posições, ordem de montagem
  scene.json              especificação completa para o Blender
  calib_mask.png          pixels usados na calibração de luz
  render/frames/*.exr     camadas 3D (Combined + Shadow Catcher) por quadro renderizado
  render/frames.json      mapa quadro → render/border/reuso + calibração + telemetria do worker
  quadros/*.jpg           prévias de quadros compostos
  report.json             preservação, PSNR do encode, telemetria completa (tempos, GPU, render)
  encode.log, interface.log  registros do FFmpeg e da saída do motor (vão no zip de diagnóstico)
```
