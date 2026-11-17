# Arquitetura do 7xAi

> Documento vivo. Decisões tomadas e **validadas com testes reais** neste repositório.
> A pesquisa comparativa de tecnologias está em [`PESQUISA_TECNOLOGIAS.md`](PESQUISA_TECNOLOGIAS.md).

## 1. Objetivo e regra absoluta

O 7xAi tem dois modos, escolhidos **automaticamente** pela análise da foto (uma foto vazia vai
sempre para o modo A):

- **Modo A, "sala vazia → mobiliada"** (motor v2 atual): entrada é **uma foto** de um ambiente
  vazio + **tipo de ambiente** (sala, quarto, cozinha, banheiro, escritório, sala de jantar). Saída:
  **MP4** em que os móveis surgem e se montam progressivamente **dentro da foto original**, com
  câmera travada.
- **Modo B, "foto mobiliada"** (V3, ver [`PLANO_V3.md`](PLANO_V3.md) §3): entrada é uma foto de
  um ambiente **mobiliado**. O software tira os móveis por trás (isso não aparece no vídeo). O vídeo
  **começa na sala vazia** (uma placa gerada por inpainting), os **mesmos móveis reais da foto**
  (camadas 2D recortadas dos pixels da própria foto; nunca 3D, nunca PNG de catálogo) entram um a um
  com animação, e o vídeo **termina exatamente na foto original**. A câmera não se move.

**Regra absoluta (vale para os dois modos):** fora da máscara de remoção (modo B) ou de inserção
(modo A), todo quadro mestre é **byte a byte** a foto. Pixels gerados só existem **dentro** dessas
máscaras, e o `report.json` informa a área e a porcentagem deles. O software **não** gera uma nova
foto decorada fora das máscaras e **não** move a câmera.

- **Modo A (inalterado):** o quadro inicial é a foto exata. Não há difusão nem inpainting sobre a
  foto: os pixels novos vêm só dos móveis 3D renderizados e das sombras de contato deles.
- **Modo B:** os quadros com a sala vazia ou parcialmente vazia **contêm pixels gerados** (a placa
  vazia, dentro da máscara de remoção ampliada para sombras e reflexos). Os móveis que entram são
  pixels da própria foto, não gerados. O **quadro final é `np.array_equal` à foto** (na resolução do
  vídeo); se não for, o trabalho aborta sem MP4. Fora da máscara de remoção, a placa vazia também é
  cópia byte a byte da foto. Se a verificação automática de qualidade reprovar a placa vazia (por
  exemplo, sobra de móvel, borrão grande ou linha parede/piso quebrada), o trabalho não entrega um
  vídeo ruim: avisa o usuário e oferece o modo A ou uma nova foto (ou, por escolha explícita dele,
  gerar mesmo assim, com o aviso registrado no `report.json`).

A "exatidão" vale nos quadros mestres e na resolução do vídeo: `load_photo` reduz uma única vez as
fotos acima do limite do perfil (reamostragem, registrada no relatório), e o MP4 tem perda (abaixo).
A interface e o material de venda devem dizer exatamente isso.

Como garantimos isso no **modo A** (e verificamos em todo quadro):

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

No **modo B**, a mesma verificação independente confere: (a) fora da máscara de remoção, todo
quadro é igual à foto; (b) o último quadro é igual à foto inteira; (c) cada camada de móvel só
contém pixels da foto dentro da sua máscara; e o `report.json` registra a área gerada (pixels e %)
da placa vazia e o resultado do controle de qualidade dela.

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
explícito (8 nos dinâmicos, 32 nos parados). No modo de quadro único (chave de retorno do render em
camadas, abaixo), os 3 quadros dinâmicos logo antes de um quadro parado sobem as amostras em
progressão geométrica, para o ruído não "pular" na entrada do *hold*. Vídeo final a 24 fps.

**Gestão de VRAM:** nunca há dois modelos na GPU ao mesmo tempo. MoGe-2 → libera → OneFormer → libera →
Blender roda em **outro processo** (a VRAM do PyTorch já foi devolvida). Em GPUs RTX o Cycles usa OptiX;
o denoiser (guiado por albedo e normal) roda na GPU: OIDN quando o Cycles acha uma GPU compatível,
senão o do OptiX; só sem GPU de render ele fica na CPU.

**Render em camadas (PT6, padrão):** o plano é Python puro (`render/layers.py::plan_layers`), o
worker só o executa (`blender_worker.render_layered`).
- **Marcos estáticos:** um por classe da coreografia (`choreography.milestones`: móveis grandes,
  mesas, peças laterais, decoração, parede; marcos a menos de 0,3 s viram um só). O marco é o
  primeiro quadro em que todas as peças da classe estão em repouso exato e contém **todas** as
  peças já assentadas até ali. O último é o quadro final (*hold*), renderizado **primeiro**, com as
  amostras do final; os outros com as de `marco`. Toda pausa sem movimento cujo conjunto parado não
  é um marco vira um marco implícito. Cada marco usa o border de todas as suas peças (o mesmo do
  quadro único), nada de recorte por item.
- **Camada dinâmica:** num quadro com movimento só são renderizadas as peças fora do marco vigente
  (as que se mexem e as que já pararam depois dele), no recorte delas (poses de f−1, f, f+1 + margem
  de 0,35 m no piso e acima + projeção da sombra do sol), com as amostras de `dinamico` e desfoque de
  movimento. As peças do marco ficam na cena como *shadow catcher* (`set_layer_roles`): escondem a
  peça nova atrás delas, recebem a sombra e a luz rebatida dela e continuam sombreando e
  iluminando a peça nova. As demais somem de todos os raios (visibilidade por raio, que não é
  animada). Renders com a mesma assinatura (marco + poses das peças da camada) do quadro anterior
  são reaproveitados.
- **Transição:** na troca de marco, 3 quadros misturam a composição antiga (marco anterior + a
  camada dele, com as mesmas peças visíveis) com a nova, peso 3/4, 1/2, 1/4, para o ruído das peças
  que entram no marco não "pular". Custa um render da camada antiga por troca (reaproveitado nas
  pausas).
- Validação em CPU (sala de teste, layout PT9, 480 px, 32 amostras nos marcos e 8 na camada
  dinâmica): 5 marcos, 148 renders dinâmicos e 13 da transição. O recorte médio da camada dinâmica
  caiu de 71% (quadro único) para 45% da imagem e o tempo de amostragem total caiu 19% (o resto, na
  CPU, é o custo fixo por quadro do denoiser). Os 251 quadros saíram byte a byte iguais à foto fora
  da união das máscaras; nos pixels de CG a diferença média para o quadro único ficou ≤ 1,5/255 (as
  peças paradas têm 32 amostras em vez de 8), sem emendas nas bordas dos recortes
  (`docs/poc/v2/pt6_camadas_vs_unico_480.jpg`).
- **Aproximações (documentadas):** (1) no pixel de borda em que uma peça do marco está NA FRENTE da
  peça nova, a parte coberta pelo marco vira mistura com a foto (efeito "over" de 1 px); (2) o reflexo
  da peça nova numa superfície brilhante do marco entra só como a razão de luz do catcher (cinza);
  (3) a sombra de contato analítica da camada dinâmica multiplica a do marco (produto, não união) e
  só escurece pixels em que o marco é foto (sobre o tapete vale a sombra do render, como no quadro
  único); (4) peças paradas depois do marco vigente continuam na camada dinâmica, com as amostras
  dela, até o marco seguinte.
- **Chave de retorno:** `render.camadas = false` no `scene.json` (ou `MOBILIA_CAMADAS=0` no motor)
  volta ao quadro único: a cena inteira em todo quadro com movimento. Depuração com `only_frames`
  funciona nos dois modos (no de camadas renderiza só os marcos que esses quadros usam).

**Orçamento de tempo (PT7, versão simples; `render/budget.py`, Python puro):** o quadro final
renderizado primeiro é também a sonda de desempenho. Dele saem a vazão de amostragem (pixel ·
amostras / s de amostragem), o custo fixo por render e o custo por pixel que não depende das
amostras (denoiser + gravação). Com o plano do restante — área de cada marco ainda por renderizar e
de cada render da camada dinâmica (`layers.dynamic_jobs`, mesma regra de reaproveitamento do laço;
o border da camada vem das poses do plano, sem `frame_set`) — `choose_settings(vazão, custo fixo,
plano, orçamento)` decide **uma vez**, antes do primeiro quadro animado, nesta ordem de corte:
1. amostras da camada dinâmica (do perfil até 12);
2. escala de render só do CG em movimento (1,0 → 0,8; `resolution_percentage` inteiro só nesses
   renders, ver abaixo);
3. amostras dos marcos (do perfil até 40).

O orçamento do restante é `total (120 s) − decorrido desde o clique − pós-render (8 s: cauda da
composição, codificação, verificação, relatório) − reserva (5 s)`. A previsão é um teto (todas as
amostras pedidas; a amostragem adaptativa só pode parar antes) e os quadros dinâmicos pagam 1,2× por
amostra pelo desfoque de movimento, que a sonda (parada) não tem. O quadro final nunca perde
amostras e nada sobe acima do perfil; se nem o mínimo cabe, o render segue no mínimo (`cabe =
false`). O `scene.json` leva `render.orcamento` (`total_s`, `t_clique`, mínimos);
`MOBILIA_ORCAMENTO_S` troca o total (`0` desliga). A decisão, a sonda e a previsão vão para
`frames.json["orcamento"]`, para o evento `budget` (o motor o repassa à interface como `forecast`:
"Previsão: ≈ 1 min 40 s no total") e para `report.json` `render.orcamento` (com `real_render_s` e
`real_total_s` para comparar). Só vale no render em camadas; no quadro único fica desligado.
- **Escala do CG em movimento:** o render reduzido usa o mesmo border normalizado na resolução
  `W·pct/100 × H·pct/100` (divisão inteira, como o Blender), arredondado para pixels inteiros nessa
  resolução (`border.scaled_border`). Com `sensor_fit = HORIZONTAL` o campo de visão horizontal é o
  mesmo e o *shift* da lente é medido em larguras da imagem renderizada (conferido no Blender 5.3),
  então o mapa é afim e independe do *shift*: `x = k·x_r`, `y = (H − k·H_r)/2 + k·y_r`, `k = W/W_r`.
  A entrada do `frames.json` (e a `transicao`) leva `escala` (pct, retângulo reduzido, k, origem) e
  `border` = o retângulo do quadro inteiro coberto. O compositor reamostra (bilinear) **só** o
  RGBA pré-multiplicado e a sombra do recorte para esse retângulo (`compositor.resample_crop`) antes
  de compor; a placa nunca é reamostrada e a verificação byte a byte segue igual (a máscara vem do
  alfa já reamostrado).
- Validação em CPU (sala de teste, layout PT9, 320 px, perfil 64/64/32, desfoque de movimento): a
  sonda mediu 534 mil pixel·amostras/s e 0,22 s de custo fixo; sem cortes o restante levaria ~468 s
  e o orçamento era 90,5 s, então o controle foi ao mínimo (dinâmico 12, escala 80%, marcos 40) e
  previu 150 s de render restante; o real foi 142 s (a decisão em si custou 0,18 s para 4 marcos e
  161 renders dinâmicos). Os 251 quadros saíram byte a byte iguais à foto fora da máscara. A camada
  reduzida e reamostrada ficou alinhada com a de escala 1 do mesmo quadro (centroide do alfa a
  ≤ 0,17 px, diferença média de alfa ≤ 0,004).
- Ainda não feito (P1): usar a sobra de fotos pequenas em supersampling do CG, e reajustar a
  decisão durante o render com os tempos reais dos quadros dinâmicos.

**Render incremental (experimental, desligado):** re-renderizar só a região das peças em movimento
colando sobre o quadro anterior reduziu a área renderizada em ~14% na cena de teste. Mas sombras
suaves junto à parede ultrapassam a região e deixaram emendas sutis, então o modo fica desligado
(`render.incremental`; ligado, ele desliga o render em camadas). O render em camadas não acumula
quadros: cada quadro é marco + camada dele, e a sombra da camada é cortada com rampa nas bordas.

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

### 4.1 Sala vazia do modo B (`removal/plate.py`, PT-V3-3)

- **Máscara de efeitos em 2 passadas:** peças dilatadas + candidatos (anel e faixa espelhada abaixo dos pés;
  para planejados também na parede, onde cai a luz do LED). Ficam só os candidatos em que a foto difere de
  uma placa provisória e que tocam a peça; separados em sombra (mais escuro) e luz/reflexo (mais claro).
- **Ray-cast na caixa vazia** (`RoomModel`): piso, teto, parede *i*, "outro" (visto por porta) e
  **protegido** (janela/porta/cortina: nunca recebe textura de parede; inpainting local). Uma parede da caixa
  cujo trecho visível é piso na segmentação é corrigida para piso (erro comum em cozinhas planejadas).
- **Preenchimento por plano** em vistas retificadas com a máscara inteira como buraco: grade periódica
  (azulejo) ou LaMa + grão, casamento de tom (mediana Lab do anel), membrana tipo Poisson e corte de ilhas.
  Parede quase toda escondida: pintura lisa (cor mediana + gradiente de luz ajustado de forma robusta);
  rodapé detectado no perfil por altura e redesenhado reto. Objetos pequenos da parede (tomadas) não servem
  de contexto. Sem material do plano à vista (menos de 1% do buraco na vista retificada) ou peça apoiada em
  algo que fica (copo na bancada mantida: o anel é bancada, não piso), o trecho vai para o espaço da imagem
  com contexto local, em vez de o LaMa inventar uma cor de piso/parede.
- **Nota (`PlateQuality`)** por plano: ΔE de costura local, ΔE da média do preenchimento contra o anel (o
  critério do aceite), razão de grão, distância/continuidade da linha parede/piso, costura do que não tem
  plano (contra o anel inteiro) e OneFormer rodado de novo na placa. Plano preenchido que não pôde ser medido
  (nenhum pixel do próprio material em volta) reprova. **Portão:** reprovada → refaz mantendo os planejados →
  se ainda reprovada, aviso (modo A ou outra foto). `scripts/avaliar_placa.py` mede isso num conjunto.

### 4.2 Camadas 2D do modo B (`layers2d/layers.py`, PT-V3-5)

- **Ordem de pintura** (de trás para frente): borda comum entre peças decidida pela profundidade do MoGe
  (sem ela, quem termina mais embaixo na foto está na frente); tapete e quadro/espelho atrás de tudo o que
  tocam; peça pequena apoiada noutra (almofada no sofá, vaso na mesa) na frente do apoio (`supports`, usado
  pela coreografia para fazer o apoio entrar antes). Peça pequena sobre outra pequena (almofada sobre manta
  ou almofada) herda o apoio grande embaixo dela.
- **RGBA pré-multiplicado** `C = foto − (1−A)·fundo_abaixo` na ordem de pintura, limitado a `0 ≤ C ≤ 255·A`.
- **Sombreamento** = razão `foto/placa` suavizada, ≤ 1, só na sombra perto da peça; acompanha a peça.
- **Delta de contato** (fixo na posição de repouso): o resto até a foto no alpha, no anel de contato
  (dilatação do núcleo da máscara) e na região de efeito da peça (textura da sombra, reflexo claro).
- **Região de efeito:** sombra/luz da placa ao alcance de *alguma* peça (anel ou faixa do `plate.py`), com
  dono = a peça mais próxima cujo alcance a cobre (a luz de LED sob o aéreo é do armário, não do vaso ao lado).
  Em `composite`, o `alpha_mul` (tapete desenrolando) também recorta o delta de contato.
- **Correção na placa:** o que não pertence a nenhuma peça vira o pixel da foto já no quadro 0; pixel
  corrigido que difere > 30 níveis da placa conta como "fantasma" (`placa_corrigida_fantasma_pct`, aviso
  acima de 0,1% da imagem: pedaço da foto mobiliada voltando para a sala vazia). Resultado:
  pilha completa = foto (0 níveis nas fixtures), quadro vazio = placa corrigida byte a byte e 0% de pixels
  fora de alpha ∪ anel ∪ efeito mudando entre o 1º e o último quadro (o protótipo tinha 7,1%).

### 4.3 Coreografia 2D do modo B (`layers2d/choreo.py`, PT-V3-6)

A camada de uma peça só tem o que a foto mostra: o pedaço atrás de outra peça (ou fora do quadro) não existe.
A coreografia nunca deixa esse corte à vista, e `verify` mede isso **em todos os quadros** ("pixel oculto
visível": pontos logo além da borda da peça que pertencem a quem está na frente dela ou ficam fora do quadro;
após o movimento eles têm de cair fora do quadro ou sob outra peça presente).

- **Unidades:** a peça pequena apoiada noutra entra junto com o apoio (almofada + sofá); a bancada assentada
  na base pousa depois dela (a aresta "base na frente da bancada" sai da cópia local de `front`), mas a cuba
  embutida entra junto com a bancada/ilha que cobre a borda dela; ciclos de oclusão viram bloco rígido (seguro em
  qualquer leitura da oclusão) e só quebram na aresta mais fraca quando o bloco faria mais peças surgirem de uma
  vez; peça que só poderia "surgir" tenta entrar em bloco com quem a cobre (manta + poltrona deslizando da borda).
  A medida só enxerga cortes contra quem está "na frente": por isso `occlusion_graph` não deixa sem relação a peça
  pequena encostada numa grande quando as duas descem até a borda da foto (a pequena fica na frente).
- **Ordem:** de frente para trás (a de trás começa só depois de quem a cobre pousar). Entre as livres: tapete
  livre, planejados (base/ilha/eletros/louças → bancada → aéreos/coifa), soltos (os mais perto primeiro),
  quadros e decoração.
- **Movimentos** (cada candidato testado ponto a ponto no caminho inteiro): tapete desenrola no lugar; peça de
  borda desliza de fora do quadro (lateral e de baixo antes do topo, nunca cai); peça coberta sobe por trás
  (amplitude ≤ 40% da peça e ≤ 15% do quadro, ≥ 5% da peça) ou escala 0,94→1 no ponto de contato; peça isolada
  cai ≤ 25% do quadro com quique; planejados sobem do chão / aéreos descem do teto **recortados na linha do
  piso/teto** (nada aparece além dela); último recurso: surge no lugar (opacidade 100%, nunca fade).
- **Contato:** o sombreamento acompanha a peça e cresce conforme ela chega; o delta de contato (parado na
  posição de repouso) só entra com a peça pousada.
- **Tempo:** sala vazia ≤ 0,5 s (0,4 s), total 7–10 s (até 12 s com muitos planejados), dissolução final de 4
  quadros (só o arredondamento da pilha, ≤ 2 níveis) e ≥ 1 s de foto exata. `frame_at`/`render_frames`
  (numpy, CPU) começam na placa corrigida e terminam `array_equal` à foto; a versão em GPU é o PT-V3-7.

### 4.4 Composição, verificação e codificação do modo B (`layers2d/compose.py`, `layers2d/video.py`, PT-V3-7)

- **Compositor** (`FrameCompositor`): a mesma conta de `layers.composite` (placa → sombreamento → RGBA
  pré-multiplicado → delta de contato, float32, `rint`), mas só dentro das caixas das camadas presentes,
  agrupadas em regiões disjuntas; fora delas o quadro é a placa sem conta nenhuma. Backend `numpy` (CPU,
  bit a bit igual a `layers.composite`) ou `torch` (CUDA quando há: as camadas sobem uma vez e só o quadro
  uint8 desce; ≤ 1 nível do numpy). Escala < 1 usa o `warpAffine` da CPU nos dois. `MOBILIA_COMPOSITOR_2D`
  força `numpy`/`torch`/`cuda`; o padrão é CUDA se houver, senão numpy (torch na CPU é ~2× mais lento).
- **Verificação** (`composite/verify.py`, `ModeBVerifier`, em fluxo, 100% dos quadros, regras do PLANO_V3 §3.5):
  quadro 0 = placa; último = foto; fora de (máscara de remoção ∪ caixa das peças em movimento) todo quadro =
  foto; placa = foto fora da máscara; pixel oculto visível = 0 (de `choreo.verify`); anti-morph entre quadros
  consecutivos fora da região dinâmica. A caixa móvel sai só da geometria (caixa + deslocamento + escala), sem
  o compositor.
- **Codificação** (`render_mode_b`): os quadros vão por uma fila para o mesmo `video/encoder.py` do modo A
  (NVENC quando a GPU codifica, x264 senão) e, opcionalmente, para um MKV sem perdas (`libx264rgb -qp 0`, RGB
  sem conversão) de auditoria, que é decodificado inteiro e conferido por hash quadro a quadro. Qualquer falha
  aborta e apaga os arquivos parciais: nenhum MP4 fica no disco. O 1º e o último quadro do MP4 são
  decodificados e medidos (PSNR) contra a placa e a foto. Exige largura e altura pares (a foto é recortada na
  resolução do vídeo antes das camadas, no PT-V3-8).
- **Tempo** (`scripts/medir_composicao_modo_b.py`, sala sintética de 6 peças, 1920×1440, CPU de 4 núcleos do
  ambiente de desenvolvimento, normalizado para 240 quadros): composição ≈ 3 s, verificação ≈ 2 s, total com
  x264 em paralelo ≈ 7,5 s (169 quadros reais, escala linear; 0,7 s fixo de preparo). Na RTX 5070 Laptop é **estimativa, não medido**: ver o relatório do PT-V3-7 no histórico.

### 4.5 Integração do modo B (`pipeline.py`, `pipeline_b.py`, `jobs.py`, UI, PT-V3-8)

- **Detecção do modo:** a análise roda o OneFormer nas tarefas semântica **e** panóptica numa carga só
  (`analise.modo` no relatório); `removal.instances.detect_mode` decide A (vazia) ou B (mobiliada). O pedido
  pode forçar (`JobRequest.mode`, `--mode`), mas a tela inicial só pede a foto e o ambiente: a escolha manual
  só aparece na tela de aviso do portão. O motor anuncia o modo com o evento `mode` (etapas próprias do modo B
  em `jobs.STAGES_B`; a interface troca a lista).
- **Modo B** (`pipeline_b.run_mode_b`), tudo na resolução do vídeo (foto com lados pares, a mesma do último
  quadro): segmentação/panóptico/geometria da análise reamostrados (vizinho mais próximo) → instâncias (+ SAM
  2.1 se os pesos estão no disco) → **portão** → camadas → coreografia → `render_mode_b` (verificação §3.5 em
  100% dos quadros) → MP4. Nada de layout nem Blender.
- **Portão de qualidade** (`plate.build_plate_gated`), parando na 1ª aprovada: sala crua por plano → mantendo
  os planejados (se há) → LaMa direto na imagem → máscara 2,5× mais larga. Fica a melhor (aprovada > menos
  motivos > nota sem teto). Se a área gerada passa do limite (sozinha ou com outros motivos), as alternativas
  não rodam (não diminuem a área). A
  nota usa o OneFormer de novo na placa (móvel que "voltou"); `MOBILIA_MODO_B_REDETECTAR=0` desliga (só
  depuração). Ainda reprovada: **sem MP4**, evento `done` com `output: null` e `warning` em português; a
  interface oferece *gerar mesmo assim* (`accept_weak_plate`, o vídeo sai com o aviso na tela final e no
  relatório), *modo A* ou *outra foto*.
- **`report.json` do modo B:** `modo` (motivo, medidas, forçado), `pixels_gerados.area_pct`, `pecas` (todas, com
  solto/planejado), `pecas_no_video`, `pecas_no_fundo`, `semente`, `placa` (nota, por plano, linha parede/piso,
  `tentativas`), `portao_qualidade`, `camadas`, `coreografia`, `verificacao_modo_b`, `video_modo_b`, `aviso`,
  `video_entregue`, `narrativa` (tela final, em português) e `tempos_s` (`telemetry.ETAPAS_B`: `pecas`,
  `sala_vazia`, `camadas`, `video`).
- **Testes:** `tests/test_pipeline_mode_b.py` roda `run_job` ponta a ponta na CPU com as análises gravadas das
  fixtures (Telea no lugar do LaMa nos rápidos; big-lama real no lento) e confere o MP4, o MKV de auditoria
  (último quadro = foto byte a byte), o relatório, o portão (reprova sem MP4; aceite gera com aviso) e o
  roteamento de foto vazia para o modo A.

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
- Movimentos (entrada por papel, `choreography.entrance_for`; o layout grava `motion`,
  `motion_dir` e `motion_dist` de cada item no `layout.json`, PT11):
  - **queda** com aceleração e toque com squash sutil (≤ 2%), acomodação por mola amortecida:
    sofá, cama, mesas, rack;
  - **surgir do chão** (`emerge`): estante, cômoda e luminária de piso sobem de baixo do piso
    (ease-out-cubic, ultrapassagem de poucos milímetros para cima, assentamento), já opacas. O corte
    é limpo no plano do piso, sem dissolução: o mix de fade de cada material também multiplica o
    fator por `z_mundo > −0,2 mm` (nó Geometry → Position), ligado enquanto a pose está abaixo do
    repouso (`needs_fade_mix`); o catcher do piso, que já esconde o que está atrás dele, é a segunda
    garantia (os dois validados num render real, `tests/test_floor_clip.py`). A luminária monta
    base → haste que cresce de dentro da base → cúpula que desce; malhas com portas/gavetas sobem
    rígidas. A sombra de contato (PT10) pesa um ponto abaixo do piso pela profundidade, então não
    aparece antes da peça;
  - **deslizar pela lateral** (`glide`): poltronas, cadeiras, mesas laterais, criados-mudos, pufes
    e banquetas entram de 0,5–1,0 m pela borda mais próxima da imagem (eixo horizontal da câmera no
    piso, depois o eixo lateral da peça), flutuando 1,5 cm, ease-out-cubic, pouso suave com um
    avanço de milímetros por inércia; fade nos primeiros 30%. Cadeiras de mesa tentam antes entrar
    "empurradas" para a mesa (0,45 m ao longo da frente). O caminho varrido é validado no layout
    (`layout/entrances.py`): base só em FREE/UNKNOWN, sem cruzar paredes-proxy (com as extensões
    das quinas) nem invadir móveis de piso que entram antes; sem caminho livre, a peça cai. Corpo,
    portas e gavetas andam rígidos; almofadas e afins assentam depois do pouso;
  - **crescimento** das plantas (ease-out-back + balanço);
  - **deslizar** dos quadros rente à parede (descem até o prego);
  - **desenrolar** do tapete.
  Nenhuma entrada alonga a linha do tempo (mesmas durações por papel e mesmo fim nominal).
  Validação em CPU (sala de teste, layout PT9, 320 px, 4 amostras, camadas + motion blur, 104 s):
  a mesa lateral desliza rente à parede esquerda a partir da borda da imagem, a luminária surge
  (base, haste, cúpula), a poltrona (sem caminho livre: parede e sofá) cai; os 251 quadros ficam
  idênticos à foto fora da máscara (`docs/poc/v2/pt11_entradas_320.jpg`).
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
- **Camadas (PT6):** o marco é composto uma vez sobre a foto (`LayerAccumulator.static_base`) e
  guardado até o fim; um quadro dinâmico compõe a camada sobre ele com a MESMA conta
  (`compose_dynamic`: `base · S'_din · (1 − A_din) + C_din`, a base no lugar da foto, mesma conversão
  sRGB→linear). A verificação usa a **união** das máscaras de influência do marco e da camada (e das
  duas da transição). A transição mistura em 8 bits (`blend_frames`): onde as duas composições são
  iguais, os bytes não mudam. Os EXR dos marcos são apagados no fim (ou depois do último quadro
  que os usa, no `recompor`).
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

**Sombra de contato analítica (PT10, `composite/contact.py`).** A emissão da foto suaviza a sombra
do shadow catcher e some a linha escura onde o móvel encosta no piso. Ela volta sem segunda camada
de render:
- **No worker**, depois de montar a cena (pose final), cada peça de um item de piso (menos tapete e
  cortina) ganha a "silhueta ortográfica de baixo": numa grade de 1,5 cm sob a peça, a altura do
  primeiro ponto da malha (raio vertical de baixo para cima, BVH), até 0,6 m. Pés e bases ficam a
  0 m, o fundo do sofá a ~0,15 m; tampos altos ficam de fora. Vai para `frames.json["contact"]`
  (alturas em mm, base64), com a câmera, e para o evento `contact` antes do primeiro quadro.
  Custo: ~0,1 s na sala de teste.
- **Em cada quadro renderizado**, `entry["contato"]` traz `[peça, fade, D]` das peças visíveis, com
  `D = M_quadro · M_repouso⁻¹`: a silhueta segue qualquer movimento (queda, giro, deslize, surgir
  do chão; abaixo do piso conta como no piso).
- **Na composição**, cada ponto pesa `fade · 1/(1 + (h/0,10 m)²)` e é desfocado por
  `σ = 1,2 cm + 0,5·h` (faixas de altura), numa grade do piso; o campo é lido em cada pixel pela
  homografia piso→imagem. Entra como `S' = S · (1 − k·c)`, k = 0,65, **antes** do peso da região de
  influência: fora da vizinhança dos móveis o peso é 0 e a placa continua byte a byte a foto
  (verificação independente de cada quadro inalterada; testado com uma silhueta maior que a
  região de influência). Peça subindo: linha mais fraca e mais larga; no alto, nada.
- Limite: a sombra só escurece a **foto**. Sobre um tapete renderizado (CG) a linha não aparece;
  ali vale a sombra do próprio render. `render.contact_shadow: false` no `scene.json` desliga.
- `report.json`: `composicao.sombra_contato` (`aplicada`, `pecas`, `forca`, `pixels_total`).
- Validação em CPU (sala de teste, 640 px, 24 amostras): quadro final + 3 quadros da queda do sofá,
  todos byte a byte iguais à foto fora da máscara de influência; `docs/poc/v2/pt10_contato_detalhe.jpg`
  (sem/com) e `pt10_queda_sofa.jpg`. Custo na composição: ~0,06 s por quadro a 640 px com 9 peças.

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
- Previsão do tempo (PT7): depois do quadro final, o evento `forecast` do motor (`ProgressEmitter.forecast`,
  vindo do evento `budget` do worker) aparece na tela de progresso abaixo do tempo decorrido
  ("Previsão: ≈ 1 min 40 s no total"); o resumo de tempos da tela de resultado repete a previsão.
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
- Narrativa de designer (PT12, `layout/narrative.py`): o solver grava em `layout.notes` frases que
  explicam o projeto só com o que foi medido e decidido — tom da sala, piso, paredes, pé-direito e
  m² de piso livre visível; estilo e paleta; a peça principal (medida, parede, vista); tapete
  (medidas, tecido, pés dianteiros dos assentos sobre ele, verificados na geometria); mesa de
  centro (% da largura do sofá e vão); quadros (medidas, móvel abaixo, altura do centro);
  decoração por apoio; o que não coube e avisos de validação. Determinística, sem custo de render.
  Vai para `report.json` (`narrativa`) e o registro do motor ("Designer: ..."), e a tela de
  conclusão mostra o bloco "Escolhas do designer" (rolável; lê `narrativa` ou o `layout.json`).

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
  layout.json             itens, posições, ordem e entrada de montagem (motion, motion_dir, motion_dist);
                          notes = narrativa de designer (PT12)
  scene.json              especificação completa para o Blender
  calib_mask.png          máscara da calibração de luz (255 = ajuste, 128 = limite inferior, 0 = fora)
  calib_sun_mask.png      manchas de sol da foto (ajuste do sol e aberturas de sol das janelas)
  render/calib_*.exr      renders da sala vazia da calibração (emissão, janelas, ambiente, sol)
  render/calib_pred.npy   predição da sala vazia com a luz final (base do ajuste do CG)
  render/frames/*.exr     camadas 3D (Combined + Shadow Catcher) por render, recortadas no border:
                          m_* marcos, f_* camada dinâmica, t_* camada antiga da transição;
                          apagadas depois da composição (MOBILIA_MANTER_EXR=1 mantém)
  render/frames.json      mapa quadro → render/border/reuso, marco de base e transição (+ poses da
                          sombra de contato) + calibração + silhuetas de contato + plano de camadas
                          (`camadas`) + telemetria do worker
  quadros/*.jpg           prévias de quadros compostos
  report.json             preservação, PSNR do encode, telemetria completa (tempos, GPU, render)
  encode.log, interface.log  registros do FFmpeg e da saída do motor (vão no zip de diagnóstico)
```
