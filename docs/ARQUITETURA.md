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
   foto** em todo pixel com `A = 0` e `S' = 1`. Cada quadro é conferido por uma verificação
   **independente** (`composite/verify.py`): fora da máscara de influência (pixels a até 10% da
   diagonal + 8 px de algum pixel com alfa > 0, montada só a partir do alfa do CG), o quadro tem de
   ser `np.array_equal` à foto. Se algum byte divergir, o trabalho aborta naquele quadro, sem MP4.
   O relatório final (`report.json`) registra quantos pixels mudaram por quadro.
4. Quadros sem móveis visíveis (o início do vídeo) são cópias exatas da foto.

> O MP4 (H.264/yuv420p) é um formato com perda. A garantia bit a bit vale para os **quadros mestres**
> (compostos antes do encode). O encode usa CRF 14 com conversão RGB→YUV **BT.709 explícita**, e o
> relatório mede o PSNR do vídeo decodificado contra os quadros mestres (fundo intocado) no início,
> no meio e no fim do vídeo (cada um por *seek*, sem decodificar o arquivo inteiro).

## 2. Visão geral do pipeline

```
foto.jpg ─┬─► [Preparo]   EXIF, placa imutável na resolução do vídeo (redução única se > perfil)
          ├─► [Análise]   MoGe-2 (profundidade métrica + intrínsecos + normais)
          │               OneFormer ADE20K (piso/parede/teto/janela/porta/obstáculos)
          │               RANSAC: plano do piso, altura da câmera, paredes (retas 2D),
          │               teto, janelas/portas como retângulos nas paredes, grade de ocupação
          ├─► [Layout]    programa de móveis por ambiente + estilo/paleta da foto +
          │               otimização com regras de design de interiores e visibilidade na câmera
          ├─► [Assets]    Poly Haven (CC0, glTF + texturas de tecido, cache local) + procedurais (tapete,
          │               luminária, cama, livros, quadro com pintura gerada na paleta...)
          ├─► [Render]    subprocesso Blender (bpy 5.x, Cycles OptiX/CUDA):
          │               câmera casada, sala-proxy = shadow catcher que emite a foto para os
          │               raios rebatidos, luz calibrada pela foto (janelas, ambiente, sol nas
          │               manchas), montagem animada peça a peça, EXR multicamada por quadro
          ├─► [Composição] saída = foto·S'·(1−A)+C ; cópia bit a bit fora dos móveis/sombras;
          │               ajuste de luz/cor, grão do sensor e nitidez casados só na camada 3D
          └─► [Vídeo]     FFmpeg (NVENC ou libx264 CRF 14 medium), BT.709, +faststart; verificação por seek
                          (composição e codificação correm em paralelo ao render, quadro a quadro)
```

Código: `mobilia/` — `analysis/` (geometria, segmentação, sala), `layout/` (catálogo, solver, estilo),
`assets/` (Poly Haven, procedurais), `render/` (coreografia, especificação da cena, worker do Blender),
`composite/`, `video/`, `pipeline.py`, `cli.py`, `ui/` (PySide6), `hardware.py`.

## 3. Hardware e perfis

A análise do hardware é feita **na máquina do usuário** (este repositório foi desenvolvido em
contêiner Linux sem GPU): `python -m mobilia.cli hardware` ou `scripts/diagnostico_hardware.ps1`
(PowerShell puro, sem instalar nada). O perfil é escolhido automaticamente pela VRAM:

| Perfil | VRAM | Geometria | Render | Resolução máx. | Amostras (final / marco / dinâmico) |
|---|---|---|---|---|---|
| ultra | ≥ 11 GB | MoGe-2 ViT-L fp16 | OptiX | 2560 px | 192 / 96 / 48 |
| alta | 7,5–11 GB | MoGe-2 ViT-L fp16 | OptiX | 1920 px | 128 / 64 / 32 |
| média | 5,5–7,5 GB | MoGe-2 ViT-L fp16 (tokens reduzidos) | OptiX | 1920 px | 96 / 48 / 24 |
| econômica | < 5,5 GB | MoGe-2 ViT-B | OptiX/CUDA | 1600 px | 64 / 32 / 16 |
| cpu | sem GPU NVIDIA | MoGe-2 ViT-B | CPU | 1280 px | 32 / 32 / 16 |

As amostras são **por camada de quadro** (`render/layers.py`, gravadas em `scene.json` como
`render.samples_layers`): `final` é a cena completa em repouso (o *hold*, que nunca perde amostras),
`marco` um quadro parado com parte dos itens e `dinamico` um quadro com alguma peça em movimento (o
desfoque de movimento esconde o ruído residual). A amostragem adaptativa tem limiar 0,02 e mínimo
explícito (8 nos dinâmicos, 32 nos parados). Enquanto cada quadro ainda é a cena inteira (antes do
render em camadas), os 3 quadros dinâmicos logo antes de um quadro parado sobem as amostras em
progressão geométrica, para o ruído não "pular" na entrada do *hold*. Vídeo final a 24 fps.

**Gestão de VRAM:** nunca há dois modelos na GPU ao mesmo tempo. MoGe-2 → libera → OneFormer → libera →
Blender roda em **outro processo** (a VRAM do PyTorch já foi devolvida). Em GPUs RTX o Cycles usa OptiX;
o denoiser (guiado por albedo e normal) roda na GPU: OIDN quando o Cycles acha uma GPU compatível,
senão o do OptiX; só sem GPU de render ele fica na CPU.

**Render incremental (experimental, desligado):** re-renderizar só a região das peças em movimento
reduziu a área renderizada em ~14% na cena de teste. Mas sombras suaves junto à parede ultrapassam a
região e deixaram emendas sutis, então a qualidade venceu e o modo fica desligado (`render.incremental`).

**Otimizações de render (implementadas):**
- *border* por quadro: só a região da tela ocupada pelos móveis (+ margem de sombra) é renderizada;
- quadros estáticos são reaproveitados (assinatura de pose; o "hold" final custa 1 render);
- quadros sem móveis não são renderizados (são a própria foto);
- semente fixa (`use_animated_seed = False`) → ruído estável, sem "fervura" entre quadros;
- `use_persistent_data` entre quadros; amostragem adaptativa + denoise;
- rebatimentos: difusos 4, brilho 2, transparência 8 (a calibração da luz usa os mesmos);
- só os passes usados são gravados: um nó File Output no compositor grava Combined (RGBA) e Shadow
  Catcher; os passes "Noisy" que o denoise acrescenta ao resultado ficam fora do EXR (valores
  idênticos aos do `write_still`, que continua como plano B);
- o EXR de cada quadro é recortado no *border* (`use_crop_to_border`) e comprimido com RLE (mais
  rápido de gravar que ZIP; ver `render/border.py`). O border em pixels é decidido primeiro e o valor
  normalizado entregue ao Blender aponta 1/4 de pixel para dentro de cada aresta, então o retângulo
  anotado no `frames.json` é exatamente o que o Cycles renderiza (antes podia sobrar 1 px sem render,
  a antiga "linha escura"). A composição usa o recorte no lugar dele (RGBA 0 e sombra neutra fora
  dele): o resultado é byte a byte o do EXR de quadro inteiro (`crop_to_border: false`);
- cada EXR é apagado logo depois de o último quadro que o usa ir para o codificador. Para depurar
  ou usar o `recompor`, `MOBILIA_MANTER_EXR=1` mantém os arquivos;
- o mix transparente do fade só fica ligado enquanto a peça está em fade-in: depois o material volta
  ao original (sem BSDF transparente, os raios de sombra não precisam avaliar o material);
- a cauda invisível do assentamento não é renderizada (ver a seção 6).

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
| Shadow catcher recebe sombras de luzes do Blender e do *world* | luzes reais da sala = luzes de área nas janelas + emissão das paredes não fotografadas + sol |
| Catcher **emissivo** dilui a própria sombra | para os raios da **câmera** a sala-proxy é **difusa, nunca emissiva**; a emissão da foto só vale para os outros raios (*Is Camera Ray*) |
| Malha emissiva **não-catcher** é tratada como objeto CG (não gera sombra no catcher) | paredes de fechamento são catchers |
| Catcher oculta objetos atrás dele | paredes-proxy fazem a oclusão correta (móvel atrás de quina some) |
| Blender 5: EXR multicamada = multi-part (`media_type = 'MULTI_LAYER_IMAGE'`) | leitor próprio com `OpenEXR.File(..., separate_channels=True)` |
| Câmera: `lens = fx·36/W`, `shift_x = −(cx−W/2)/max(W,H)`, `shift_y = (cy−H/2)/max(W,H)` | erro de projeção < 0,35 px no teste automatizado |

**Sala-proxy (PT8):** piso, teto, paredes (com recortes nas janelas) e oclusores como shadow
catchers, com um material misturado por *Light Path › Is Camera Ray*:
- **raio da câmera:** difuso, como sempre (a regra validada "catcher emissivo dilui a própria
  sombra" continua respeitada). No passe de sombra o albedo se cancela (razão com e sem móveis no
  mesmo ponto), então ele só pesa nos renders da calibração. Ali ele é **uniforme por superfície**
  (cor mediana do piso/parede/teto × escala, balanceada pelo iluminante): com a própria foto como
  albedo, a luz já "pintada" na foto (manchas de sol, claridade perto da janela) entraria duas
  vezes e a razão foto/render diria só 1/(irradiância do modelo) — era por isso que o ajuste
  antigo zerava o sol;
- **qualquer outro raio** (luz rebatida nos móveis, reflexos): `Emission` da própria foto linear
  (coordenada *Window*), força 1. O móvel recebe a luz real que a sala reflete, com a cor e a
  distribuição da foto. `emission_sampling = 'NONE'`: a malha de oclusores (centenas de milhares
  de triângulos) não entra na árvore de luzes; a sala que emite em volta toda é achada pelos
  raios rebatidos.

Fora do enquadramento (atrás/ao lado da câmera) vale a cor média da superfície. As paredes não
fotografadas fecham a sala. `lighting.photo_emission: false` no `scene.json` volta ao modo antigo
(difuso com a foto como albedo, sem emissão).

**Luz calibrada pela foto (PT8, `render/calibration.py`).** Renders rápidos (240 px) da sala
**sem móveis**, uma fonte por vez: a emissão da foto (termo fixo), as janelas, o ambiente (paredes
não fotografadas) e o sol. O modelo é `predição = E + Σ kᵢ·Rᵢ`, ajustado na luminância dos pixels
de piso/parede/teto:
- a máscara de calibração (`calib_mask.png`) tem 3 níveis: 255 = pixel usado no ajuste, 128 = quase
  saturado (luminância ≥ 0,97 ou um canal ≥ 250), que entra como **limite inferior** (a predição tem
  de ser pelo menos o valor recortado) em vez de ser descartado, e 0 = fora. Os limites nunca
  pesam mais que os pixels normais no total (numa sala branca eles inflariam as luzes);
- janelas + ambiente por mínimos quadrados não negativos fora da área do sol; o ambiente fica
  limitado a uma parcela da luz que sobe de 50% em salas médias até 70% em salas claras;
- o **sol é ajustado à parte**, só nas manchas de sol da foto (`sunlight.py`) que o sol do render
  também ilumina: a força vem do excesso da mancha sobre a própria foto em volta dela (a foto "sem
  sol" interpolada da vizinhança), não do resíduo do modelo. Pixels estourados da mancha exigem
  que o sol estoure ao menos metade do miolo dela (mediana, nunca o pior pixel da borda);
- a cor global por canal das luzes (antes só diagnóstico) é aplicada nas janelas, no sol e no
  ambiente, sem mudar a luminância ajustada (limitada a ±15%);
- a predição final da sala vazia vai para `render/calib_pred.npy` e dela saem o mapa de ganho e a
  curva de tom do CG (seção 7).

`rel_rms` é o erro da luz na escala da sala (resíduo de luminância depois de um desfoque de 4% do
lado maior, sem a textura do albedo uniforme); `rel_rms_ganho` é o mesmo depois do mapa de ganho
(o que o CG de fato recebe); `rel_rms_pixel` é o erro por pixel. Na sala de teste (CPU, 240 px):
emissão da foto ≈ 50% da luz, sol 2,0 nas manchas (antes zerado), `rel_rms` 0,18 e `rel_rms_ganho`
0,12. A transformação de vista continua Standard (AgX/Filmic quebraria a correspondência).

**Sol direto (validado na foto de teste):** manchas de sol no piso são detectadas e a direção do sol é
estimada por busca global (azimute × elevação). O critério é que as partes transparentes das janelas,
projetadas no piso, expliquem as manchas (F0,5 entre projeção e manchas; precisão ≥ 60%). Na foto de
teste, o resultado foi azimute ~139° e elevação ~51°, consistente em 1536 e 1920 px, com 95–99% da
área das manchas explicada. Sem manchas confiáveis, o sol não é usado. No render (PT8):
- um **telhado só de sombra** (catcher invisível para a câmera e para os raios rebatidos) fecha a
  sala por cima: o teto visível não projeta sombra (estendido além das paredes, ele taparia o sol
  que sai pela janela) e, sem nada por cima, o sol entrava por cima das paredes e iluminava quase
  todo o piso. A planta é o retângulo das paredes de fechamento menos as faixas atrás de cada
  parede fotografada (vale para salas não convexas, como bay windows);
- **aberturas de sol**: cada pixel de mancha é levado ao piso pelo raio da câmera e dali, na
  direção do sol, até o plano da parede; os pontos que caem numa janela dão a parte dela por onde
  o sol da foto entra (persianas, caixilhos e telas cortam o resto). O complemento vira "persianas"
  só de sombra, um pouco atrás da parede, passando 5 cm das bordas (sem lâmina de sol rasante) e
  sobrepostas nas emendas. Janelas voltadas para o sol sem mancha nenhuma e com a projeção na foto
  ficam fechadas para o sol; sem evidência (projeção fora da foto), ficam abertas.
  (`docs/poc/v2/pt8_sol_render_vs_foto.jpg`: sol do render antes e depois, sobre as manchas da foto.)

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

**Ativos sem cara de CG (PT9).**
- *Quadros*: os quadros modernos 01/02 da Poly Haven saíram do catálogo (`catalog.PLACEHOLDER_ART`:
  um tem o cartão cinza de modelo "Ray Homes" no lugar da arte, o outro a imagem esticada nas bordas).
  No lugar entra `proc:canvas_art`: a pintura abstrata é gerada localmente por `assets/art.py` (só numpy,
  determinística pela semente; estilos `formas`, `campos` e `arcos`) nas cores da paleta da sala, sobre
  tela de algodão com trama em escala física (~0,9 mm) e relevo de tinta no *bump*. Lado ≥ 0,9 m: tela
  esticada de 3,5 cm sem moldura; menor: moldura fina (metal da paleta ou madeira) com passe-partout.
  A imagem tem exatamente a proporção da área pintada (nunca é esticada). O tamanho segue regras de
  projeto (`solver._art_sizes`): sobre um móvel o conjunto ocupa ~2/3 da largura dele (0,5–1,8 m),
  paisagem 4:3 (3:2 a partir de 1,2 m) ou díptico de retratos 3:4 com 8 cm de vão, base 20–25 cm acima
  do móvel; em parede livre, 60 × 80, 90 × 65 ou 2 × 45 × 60 cm. No estilo clássico entram as molduras
  douradas "fancy" 01/02 da Poly Haven (pinturas holandesas reais escaneadas, CC0). Duas cópias da
  mesma pintura nunca formam um par.
- *Estofado em sala clara*: com a parede a L* ≥ 55 (exposição normalizada; a parede creme da sala de
  teste mede ~58) o estofado claro tem preferência e os assets de albedo quase preto (tag
  `albedo_escuro`: sofá 02, poltrona moderna 01, pufe 01, mediana sRGB da textura < 0,2) ficam fora.
  Em qualquer sala, cada massa escura além da primeira custa pontos (`rules.dark_masses`).
- *Tapete*: placa de quinas vivas (raio ¼ da espessura, a borda projeta a linha fina de sombra) com
  textura fotográfica CC0 de tecido da Poly Haven por estilo (`catalog.RUG_TEXTURES`: lã espinha-de-peixe,
  pelúcia, juta, tricô), baixada na instalação junto com os modelos e cacheada em
  `assets/polyhaven/textures/<id>/<res>/` (o `manifest.json` registra URL de origem e licença). A cor é
  a da paleta vezes o detalhe da foto (luminância/média) com mesclado e *abrash* de tingimento; o desenho
  vira faixas tecidas discretas (`filete`, `borda`, `borda_dupla`). Sem a textura (offline sem cache) o
  tapete volta ao desenho procedural antigo.
- Antes/depois na sala de teste (640 px, 24 amostras): `docs/poc/v2/pt9_antes_depois_640.jpg`.

## 6. Animação (um único estilo, refinado)

`mobilia/render/choreography.py` define tempos e curvas, e o worker grava keyframes **por quadro**:
o vídeo mostra exatamente essas funções.

- Ordem: tapete, depois peças grandes de trás para frente, depois médias, luminárias e plantas,
  quadros e por fim a decoração pequena (almofadas, vaso, bowl, livros).
- Sobreposição: cada peça começa quando a anterior completou ~55% da sua duração, com no máximo
  3 itens mudando na imagem ao mesmo tempo (o item só espera quando o limite seria passado; na sala
  de teste isso nunca acontece e a linha do tempo fica igual).
- Fim do assentamento: quando o deslocamento projetado de qualquer ponto da peça cai abaixo de
  0,3 px (com a ampliação da câmera perto do item, `px_per_m` no `scene.json`) e o fade já terminou,
  a pose vira **exatamente** a de repouso. Os quadros dessa cauda são iguais ao final e reaproveitados;
  o fim nominal (início + 2,2 × duração) e o início do *hold* não mudam. Na sala de teste a 1920 px:
  200 quadros renderizados a 30 fps antes, 153 a 24 fps agora.
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

**Composição em fluxo (PT5, `composite/stream.py`).** O worker do Blender anuncia cada quadro assim
que o EXR fica pronto (evento `frame_ready`), e o motor compõe, verifica e codifica enquanto a GPU
renderiza os quadros seguintes: depois do último render sobra só a "cauda".
- Cada EXR novo é uma tarefa num conjunto de threads (NumPy, SciPy e OpenEXR soltam o GIL): leitura
  do recorte, composição e verificação. Uma thread escritora entrega os quadros ao FFmpeg na ordem
  do vídeo, só depois de a verificação do quadro passar. A memória fica limitada (o anúncio de
  quadros espera quando há quadros demais à espera do codificador).
- Quadros reaproveitados (mesma pose) usam o quadro já composto, em cache; quadros só-foto são a
  própria placa. Nenhum dos dois passa pelo compositor de novo.
- A composição é feita só no **recorte** = caixa dos pixels com móvel + raio de influência + raios
  dos dois gaussianos (sombra e nitidez) + 2 px. Fora dele o resultado de quadro inteiro já seria a
  foto, e dentro dele cada valor intermediário é igual ao do quadro inteiro: a saída é a mesma byte
  a byte (testado; na sala de teste os 251 quadros mestres saíram idênticos aos da versão anterior).
  As camadas vêm do EXR recortado direto para esse retângulo, sem buffers do quadro inteiro.
- A placa (ruído, nitidez e grão da foto) é estimada numa thread logo no início do render. Na
  memória ela é uma cópia só de leitura: o compositor, a verificação e os quadros só-foto usam o
  mesmo buffer, e uma escrita nele (bug) vira erro em vez de mudar os três juntos sem ninguém ver.
- Codificador: NVENC quando a GPU NVIDIA consegue codificar (`MOBILIA_CODIFICADOR` força um),
  senão libx264 CRF 14 preset `medium` (~1,7× mais rápido que `slow`, −0,3 dB de PSNR). O teste do
  NVENC usa as mesmas opções do encode (preset, AQ temporal...) e roda numa thread durante a análise.
- Se o worker não anunciar os quadros (versão antiga), tudo é composto no fim a partir do
  `frames.json`; se os anúncios não baterem com o arquivo, o trabalho para com erro.
- `recompor` usa o mesmo caminho com todos os quadros de uma vez: na sala de teste a 960 × 720 (251
  quadros, 176 EXR), 55 s → 9 s em 4 núcleos de CPU, com os mesmos quadros mestres.
- `report.json`: `composicao` (`paralela`, `trabalhadores`, `compostos_durante_render`, `cauda_s`,
  `quadros_compostos`/`quadros_cache`/`quadros_so_foto`, `recorte_composicao_medio_px`, tempos de
  leitura, composição e verificação por quadro), `preservacao.verificacao` (método, raio da máscara)
  e `encode` (`codificador`, `preset`, `verificacao_metodo: "seek"`). Em `tempos_s`, `composicao` e
  `codificacao` contam só a cauda depois do render (o resto ficou escondido atrás dele).

**Ajuste da luz e da cor do CG (PT8, `composite/grade.py`).** Calculado uma vez na calibração e
gravado em `frames.json["calibration"]["ajuste_cg"]` (o `recompor` usa o mesmo; na composição em
fluxo ele chega no evento `calibration`, sempre antes do primeiro quadro). Multiplica **só C** onde
A > 0, antes do casamento de nitidez; a placa e a sombra não mudam (testado byte a byte):
- **mapa de ganho residual de baixa frequência, por canal**: `G = blur(m·foto/pred)/blur(m)`
  (σ = 0,1 do lado maior), limitado a [0,85; 1,2], tendendo ao ganho global onde a máscara é rala.
  Com pouca máscara (< 5% do quadro, ou < 25% das superfícies não estouradas: sala branca) vira um
  ganho global; com quase nada, identidade. É gravado numa grade 8× menor que o render da
  calibração (cabe no `frames.json`) e reamostrado para o quadro;
- **curva de tom monotônica**: regressão isotônica de log₂(luminância prevista) para log₂(luminância
  da foto) sobre medianas de faixas de quantis (2–98%), inclinação ≥ 0,3, desvio ≤ 0,35 stop,
  voltando à identidade 1 stop fora da faixa dos dados. Aplicada na cor sem pré-multiplicação (C/A)
  e pré-multiplicada de novo; identidade quando a dispersão é alta, o resíduo das medianas é alto
  ou a correção é pequena (`curva_tom_ok` no relatório);
- **joelho nas altas luzes**: compressão suave por canal acima de 0,8 linear (o CG não estoura onde
  a foto "rola" as altas luzes).

A/B na sala de teste a 640 px / 24 amostras em CPU (`docs/poc/v2/pt8_ab_sala_640.jpg`,
`pt8_ab_detalhe.jpg`) e uma sala branca sintética com 78% das superfícies estouradas
(`pt8_sala_clara_640.jpg`). Custo do ajuste: ~0,06–0,12 s por quadro com 1 MP de CG.

**Casamento com a câmera:**

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
- Uso offline: `python -m mobilia.cli prefetch` baixa os pesos dos modelos (com os metadados de classes
  do OneFormer, `shi-labs/oneformer_demo`) e todos os assets do catálogo (`--modelos`/`--ativos`/`--res`
  separam as partes); depois disso nada mais é baixado. Se um peso faltar no cache durante um trabalho,
  o tempo de rede vai para `tempos_s.download` (e `download.modelos_s`), não para `analise.*`.
- Empacotamento Windows: um `.exe` leve (UI) e um *runtime* pesado (Python + PyTorch CUDA + Blender
  bpy + modelos), instalado no primeiro uso em `%LOCALAPPDATA%\MobiliaStudio`. Detalhes em `packaging/`.
- Instalação prepara tudo (PT2, `mobilia/bootstrap.py`): etapas `python`, `pip`, `dependencias`, `app`,
  `modelos` (pesos), `ativos` (móveis do catálogo, texturas 1k+2k com GPU) e `aquecimento`
  (`mobilia.cli aquecer`, `mobilia/warmup.py`): renders de 64 × 64 com OptiX e com CUDA no Blender
  instalado — sem e com desfoque de movimento, shadow catcher, vidro, transparência, texturas e o
  denoiser na GPU — para deixar os kernels compilados nos caches do driver, e uma carga dos dois modelos
  de análise com uma inferência minúscula. `CUDA_CACHE_MAXSIZE`/`OPTIX_CACHE_MAXSIZE` = 4 GiB no worker e
  no Blender (`runtime.GPU_CACHE_ENV`). Cada etapa grava `setup_state.json` ao terminar (retomável) e tem
  uma impressão digital: o código refaz só a cópia do motor, o catálogo refaz `ativos`, os ids dos
  modelos refazem `modelos`, e dependências ou GPU/driver novos refazem o `aquecimento`. O aquecimento
  nunca bloqueia: a falha fica em `setup_state.json` e o primeiro vídeo paga o custo. Ao terminar, ele
  marca a próxima geração como primeira execução (`primeira_execucao_pendente`); a interface mostra o
  aviso "primeira execução" na tela de progresso, e o `report.json` traz
  `primeira_execucao_custos.aquecimento_instalador` (feito/ok, dispositivos, kernels pagos na instalação).
- Telemetria (`mobilia/telemetry.py`): `report.json` traz `tempos_s` com todas as subetapas da seção 4
  do `PLANO_V2.md` (`inicio_motor`, `carga_foto`, `analise.{geometria_carga, geometria_inferencia,
  segmentacao_carga, segmentacao_inferencia, sala, sol, oclusores, gravacao}`, `layout`, `ativos`,
  `download`, `blender_inicio`, `blender_kernels`, `cena_montagem`, `calibracao`, `render_final`,
  `render_marcos`, `render_dinamico`, `blender_fim`, `composicao`, `codificacao`, `verificacao`,
  `relatorio`, `total`, `total_percebido`). As etapas de primeiro nível particionam o total (a soma fica
  dentro de 5%, testado). `total_percebido` conta a partir do clique em "Gerar" (a interface passa o
  instante em `MOBILIA_T_CLIQUE`/`MOBILIA_T_INICIO`). `primeira_execucao` vem do estado da instalação
  (`setup_state.json`, contador `execucoes`), e `primeira_execucao_custos` separa download, kernels e
  carga dos modelos (`kernels_cycles_total_s` soma a carga/compilação de kernels de todas as chamadas
  de render, porque o OptiX pode recriar o pipeline depois do aquecimento quando a cena pede recursos
  novos, como desfoque de movimento).
  - O worker do Blender emite `worker_ready` com o dispositivo efetivo do Cycles (OptiX/CUDA/CPU, nomes
    das GPUs) e do denoiser (a mesma consulta `has_oidn_gpu_devices` que o Cycles usa para cair na CPU),
    faz um render de aquecimento de ~16 px (kernels de render e do denoiser, medidos à parte) e, por
    quadro, lê as mensagens do handler `render_stats`: sincronização, carga de kernels, amostragem,
    denoise (estimado pelo ritmo das amostras, porque o Cycles só informa "Sample N/N" depois do
    denoiser), gravação ("Saving") e memória. Cada quadro tem uma camada: `dinamico` (peça em
    movimento), `final` (cena completa parada) ou `marco` (parado com parte dos itens).
  - `render.desempenho` (amostras/s nos quadros finais, custo fixo por quadro), `render.camadas`,
    `render.por_quadro`, `render.quadros` (total, renderizados, reutilizados, só foto), cobertura média
    do recorte, `composicao.por_quadro` (leitura, composição, verificação, envio ao FFmpeg), `encode` e `gpu`
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

A melhoria sugerida por essa pesquisa — material de catcher com `Light Path › Is Camera Ray`
(difuso para a câmera, emissão da foto para os demais raios) — foi implementada no PT8 (seção 5).

## 10. Pastas de um trabalho

```
<trabalho>/
  plate.png               placa imutável (referência bit a bit)
  analise/                geometry.npz, segmentation.npz, room.json, grid.npy, analise_overlay.jpg
  layout.json             itens, posições, ordem de montagem
  scene.json              especificação completa para o Blender
  calib_mask.png          máscara da calibração de luz (255 = ajuste, 128 = limite inferior, 0 = fora)
  calib_sun_mask.png      manchas de sol da foto (ajuste do sol e aberturas de sol das janelas)
  render/calib_*.exr      renders da sala vazia da calibração (emissão, janelas, ambiente, sol)
  render/calib_pred.npy   predição da sala vazia com a luz final (base do ajuste do CG)
  render/frames/*.exr     camadas 3D (Combined + Shadow Catcher) por quadro renderizado, recortadas
                          no border; apagadas depois da composição (MOBILIA_MANTER_EXR=1 mantém)
  render/frames.json      mapa quadro → render/border/reuso + calibração + telemetria do worker
  quadros/*.jpg           prévias de quadros compostos
  report.json             preservação, PSNR do encode, telemetria completa (tempos, GPU, render)
  encode.log, interface.log  registros do FFmpeg e da saída do motor (vão no zip de diagnóstico)
```
