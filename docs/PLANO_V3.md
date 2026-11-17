# 7xAi V3: plano de implementação

Data: 16/11/2026. Este plano junta 2 pesquisas (`r_staging`, `r_removal`), 2 protótipos em CPU (`p_staging`, `p_removal`) e 2 críticas (`critic_realism`, `critic_feasibility`), todos na pasta `scratchpad/v3/`. Nada no repositório foi alterado.

Comparativo visual do melhor quadro de cada abordagem: `scratchpad/v3/comparativo.jpg`.

> **Aviso:** a sandbox não tem GPU. Os tempos na RTX 5070 Laptop 8 GB são **estimativas [E]**. Os tempos de CPU foram **medidos [M]**. As licenças marcadas **[V]** foram conferidas pela API do Hugging Face ou pelo arquivo LICENSE do GitHub.

---

## 1. Decisão e por quê

### 1.1 Decisão

1. **O motor principal da V3 deixa de ser o render 3D e passa a ser composição 2D com camadas feitas de pixels de foto real.** Um único motor de camadas atende os dois modos:
   - **Modo B, "foto mobiliada"** (o pedido atual do usuário, implementado **primeiro**): o software tira os móveis por trás. O vídeo começa na sala vazia, os móveis **da própria foto** voltam um a um com animação, e o vídeo termina **exatamente** na foto original.
   - **Modo A, "sala vazia → mobiliada"** (depois): um modelo de difusão com licença comercial (SDXL/RealVisXL + ControlNet de profundidade) **pinta** os móveis dentro da foto, guiado pelo layout automático atual. O mesmo motor de camadas do modo B separa e anima essas peças. A foto vazia real do cliente serve de fundo, então nesse modo não há inpainting de sala vazia.
2. **PNG de móveis recortados não entra no 7xAi** (testado e reprovado: cara de adesivo, ângulo errado e licenças que não permitem uso comercial).
3. **O motor v2 (Blender Cycles + Poly Haven CC0) fica como fallback determinístico** e como fonte do passe de profundidade (Z) do modo A. Não é apagado.

### 1.2 Evidências

| Abordagem | Nota do crítico (1–10) | Evidência | Por que sim ou não |
|---|---|---|---|
| v2 atual, render 3D | rejeitado pelo usuário | `comparativo.jpg`, painel 1 | "Parece 3D." |
| SDXL sobre o render CG (H1) | **2,5** | `p_staging/final_H1_full.jpg` | Mantém o visual CG. |
| SDXL + depth no lugar (R2) | **4,5** | `p_staging/final_R2_full.jpg`, `critic_realism/r2_grid.jpg` | É o caminho certo para o modo A: escala e perspectiva corretas e aparência de foto em miniatura. Ainda tem móveis moles (upscale de 1,875×), objetos derretidos e, na animação, partes ocultas expostas (luminária sem base, poltrona sem metade). Só 1 de 13 gerações saiu boa. |
| **Modo B: esvaziar e devolver (p_removal)** | **5** (o protótipo se deu 6) | `p_removal/out/antes_vazio_depois.jpg`, `p_removal/out/cmp_all.jpg`, `p_removal/out/video_final/video.mp4` | **Os móveis ficam 9/10**, porque são pixels reais. O último quadro é `array_equal` à foto (verificado também pelo crítico, após decodificar o MKV sem perdas). O ponto fraco é o 1º quadro, a sala vazia, que tem nota 3. |

Conclusão: o modo B **já resolve a demanda 1 (nada de 3D) e a demanda 2** com a arquitetura certa. Todo o esforço de qualidade vai para a **sala vazia** e para a **coreografia**. O modo A reaproveita o mesmo motor. Por isso o modo B vem primeiro.

### 1.3 Mudança de regra do produto

A "regra absoluta" da `docs/ARQUITETURA.md` §1 ("não usa difusão nem inpainting sobre a foto") **precisa ser reescrita**:

> Fora da máscara de remoção ou de inserção, todo quadro mestre é byte a byte a foto. No modo B, o quadro final é a foto exata (na resolução do vídeo). No modo A, o quadro inicial é a foto exata. Pixels gerados só existem dentro das máscaras, e o `report.json` informa a área e a porcentagem deles.

A "exatidão" vale nos quadros mestres e na resolução do vídeo. O `load_photo` reduz fotos acima de 1920 px, e o MP4 H.264 yuv420p tem perda por natureza. A UI e o marketing devem dizer exatamente isso.

---

## 2. Modo A: "sala vazia → mobiliada" com móveis de aparência real

**Entrada:** foto da sala vazia e tipo de ambiente (como hoje). **Saída:** o vídeo começa na foto exata e as peças entram até a sala mobiliada.

### 2.0 Requisito do usuário (17/11/2026) e avaliação dos assets atuais

**Requisito:** no modo A os móveis precisam parecer **100% fotografados dentro do imóvel**: nada de visual de videogame, desenho,
3D simplificado ou material artificial. O 3D pode ser usado **por dentro** (posição, escala, profundidade, sombras de contato,
ordem e tempo da animação), mas **nenhum pixel final pode vir de render CG**. A foto continua intacta fora dos móveis e das sombras,
e os móveis surgem e se montam progressivamente.

**Avaliação dos assets atuais (procedurais + Poly Haven CC0 no Cycles): não atingem esse padrão.**
- Procedurais (`assets/procedural.py`: cama, cortina, tapete, luminárias, pilha de livros, mesa): geometria e materiais simples; leem
  como CG mesmo com a calibração de luz da V2. Servem só como volume para posição e profundidade.
- Poly Haven CC0 (fotoescaneados em parte): os melhores do catálogo, mas o render inteiro ainda foi rejeitado ("parece 3D") e o
  crítico confirmou. Motivo: integração de luz, materiais e microdetalhe de foto, não só a qualidade da malha.
- Assets 3D comerciais "archviz" de alto nível (tipo bibliotecas pagas) melhorariam, mas (a) quase todas proíbem redistribuir os
  modelos dentro de um software e (b) ainda dependem de render CG, que é exatamente o que o usuário não quer ver.
- **Conclusão:** o caminho é o deste capítulo: o 3D (procedural + Poly Haven) vira **só passe de geometria** (Z/normais/máscaras
  por peça) e os pixels finais vêm do modelo de difusão fotográfico guiado por essa geometria. Os assets atuais continuam úteis e
  não precisam ser trocados para isso; a qualidade fotográfica vem da geração, não da malha.

### 2.1 Pipeline

1. **Análise (existente):** MoGe-2, OneFormer e `build_room_model`.
2. **Layout (existente):** `layout/solver.py` escolhe as peças e as posições (pegadas, alturas, rotações).
3. **Passe Z do layout:** profundidade dos modelos CC0 com forma real (Poly Haven), renderizada pelo Blender (só Z, sem Cycles completo: `bpy`/EEVEE ou um rasterizador próprio). **Não** usar caixas: as caixas geraram armários em forma de caixa literal (p_staging, variantes A–C).
4. **Geração:** RealVisXL V5.0 Lightning (ou SDXL base 1.0 + SDXL-Lightning como alternativa de licença limpa), mais ControlNet depth (small ou Union), máscara = pegadas dilatadas e sombras, prompts fixos por tipo de ambiente com embeddings pré-calculados, 6–8 passos.
   - Gerar a 1024×768 e fazer um **2º passe hires em tiles** só nas regiões dos móveis (~1,5–2 MP, força 0,3–0,4). Isso ataca o defeito nº 1 do R2 (móveis moles).
   - **2–3 sementes com nota automática:** área gerada fora das pegadas planejadas, instâncias detectadas pelo OneFormer por classe esperada, e penalidade para objetos pequenos "derretidos". Fica a melhor.
   - Sem acessórios de mesa pequenos no prompt nem na máscara: eles derretem.
5. **Recolocação na foto:** só dentro da máscara final (móveis + sombras). Fora dela, os pixels são byte a byte os da foto (o R2 manteve 69,5% da foto idêntica).
6. **Camadas:** o motor do modo B (§3), com a foto vazia real como placa. As peças são separadas por OneFormer panoptic + SAM 2.1. As sombras são a razão `mobiliada / vazia ≤ 1`.
7. **Partes ocultas:** as partes escondidas de uma peça gerada (pé da luminária atrás do sofá, base da poltrona) **não existem**. Duas saídas:
   - a ordem de entrada e a coreografia do modo B (§3.4);
   - completar a parte oculta com inpainting SDXL por objeto, só para peças que caem de cima.
8. **Animação e codificação:** como no modo B, mas o primeiro quadro é a foto vazia e o último é a composição mobiliada.

### 2.2 Qualidade esperada e o que falta

- Hoje (R2): 4,5/10. Metas da V3: hires pass, seleção de sementes, sem acessórios pequenos e oclusão resolvida. Alvo: ≥ 7/10 numa sala típica.
- Mudar o produto (o sofá sair diferente do planejado) **não é defeito** no modo A: o cliente vende o imóvel, não o sofá. O layout só garante escala e posição.
- Fallback: **nunca** voltar a mostrar render CG (requisito de 17/11). Se nenhuma semente passar na nota mínima: mais sementes,
  depois layout simplificado (menos peças, só as grandes), e se ainda falhar terminar com aviso claro na tela e no relatório.

---

## 3. Modo B: "foto mobiliada: o software esvazia por trás, o vídeo começa vazio e os móveis reais aparecem até ficar igual à foto"

**Entrada:** foto com móveis e tipo de ambiente. O modo é **detectado sozinho** pelas instâncias de móveis do OneFormer (número e área), com chave manual de override. **Saída:** MP4 cujo 1º quadro é a sala vazia (a remoção nunca aparece) e cujo último quadro mestre é a foto original byte a byte.

### 3.1 Segmentação por peça

- OneFormer ADE20K **panoptic** (`label_ids_to_fuse` vazio), unido aos componentes **semânticos** que o panoptic perde. No teste, o panoptic não achou a mesa de jantar e fundiu 2 tapetes.
- SAM 2.1 base+ por caixa para refinar. Se o IoU contra o OneFormer for menor que 0,6, fica a máscara do OneFormer. Manter só o maior componente conexo, cortar alpha < 0,05 fora da máscara dilatada e fundir fragmentos pela regra do fecho convexo (perna separada pela cadeira).
- Classificação: peças **móveis** (sofá, cadeira, mesa, cama, tapete, luminária, planta, quadro) vs **fixas**. Fixas ficam: painéis e armários embutidos, cortina, luminária de parede, eletrodomésticos embutidos. O ADE20K não distingue embutido de solto, então é preciso uma regra geométrica: encostado na parede, do piso ao teto ou largura de parede inteira significa fixo.
- Bordas finas (opcional): BiRefNet (MIT) no recorte de cada peça. **Não usar ViTMatte** (treinado no Composition-1k, risco jurídico) nem RMBG-2.0 (não comercial).
- Objetos pequenos (controle remoto, caixas de som, objetos sobre a mesa) que o OneFormer não detecta **ficam no fundo**. Opcional V3.1: Grounding-DINO tiny (Apache-2.0).

### 3.2 Sala vazia (o gargalo de qualidade)

Ordem que o protótipo validou: C > B > A > D (`p_removal/out/cmp_all.jpg`).

1. **Máscara de remoção = peças + anel de 1,2% da diagonal + sombra e reflexo.** Primeira passada de LaMa. Depois, só ficam na máscara os candidatos (faixa espelhada abaixo dos pés, até 45% da altura) em que a foto difere da placa e que estão ligados à peça. Segunda passada.
2. **LaMa por plano** com o `RoomModel`: cada pixel mascarado é classificado por ray-cast na caixa **vazia** como piso, parede *i* ou "outro" (vista por portas).
   - Piso: retificado para vista de cima, LaMa com contexto só de piso, grão reinjetado.
   - Paredes: retificadas de frente, com o rodapé reto.
   - **A máscara inteira entra em cada vista retificada.** Bug encontrado no protótipo: com só os pixels do próprio plano, o LaMa reconstrói a cadeira a partir dos pedaços dela que caem no plano vizinho.
3. **Harmonização por plano (novo, exigido pela crítica):** casar média, desvio e histograma de cor do piso preenchido com o anel original, por plano retificado, mais Poisson blending na borda da máscara. Ataca as "ilhas" de tom no piso.
4. **Refino SDXL opcional e restrito:**
   - só no **piso** e nos vãos, força 0,3–0,5, ControlNet com a profundidade da **caixa vazia** (é o que mantém as quinas);
   - paredes ficam no LaMa por plano, ou força ≤ 0,3;
   - **janelas, vistas externas e portas são regiões protegidas**: nunca SDXL ali. O protótipo transformou a sacada em ladrilhos e inventou uma porta e uma tomada. Em anúncio imobiliário isso é risco (CDC, art. 37).
5. **2–3 sementes com nota automática:**
   - visibilidade da costura: diferença de estatística de cor e textura entre o interior e o anel;
   - **descarte** de semente em que o OneFormer detecta objeto novo (porta, móvel ou tomada onde antes havia parede);
   - se nenhuma passar, usar a placa só com LaMa por plano (B).
6. **Copiar de volta** a foto byte a byte fora da máscara. Isso garante que a placa só difere da foto dentro dela.
7. **Piso polido:** aceitar que o reflexo é removido, não reconstruído.

### 3.3 Camadas

- Para cada peça: RGBA com **cor pré-multiplicada `C = foto − (1−A)·fundo_abaixo`**, calculada na ordem de pintura. Com isso, a pilha completa reproduz a foto por construção, até o arredondamento. Isso substitui a dissolução longa e evita os "fantasmas" de cor.
- **Efeito (sombra e reflexo) = razão `foto / placa`, limitada a ≤ 1**, só **perto da peça e onde a foto é mais escura que a placa**.
- **O resto da diferença foto − camadas é corrigido na própria placa, não revelado no vídeo.** A crítica mediu que 7,1% da imagem fora de qualquer móvel "se curava" durante o vídeo (carpete virando terrazzo), um morph visível. **Invariante novo:** fora do alpha dilatado da peça e da região de efeito dela, a placa de cada quadro não muda.
- A dissolução final cai para no máximo 4 quadros, só para o arredondamento, seguida de hold com a foto exata.

### 3.4 Coreografia (sem revelar partes que não existem)

Regras validadas no p_removal: nenhum pixel inexistente apareceu.

- **Ordem de pintura:** de trás para frente, pelos pares de oclusão (profundidade do MoGe na borda comum) mais regras de apoio (vaso na mesa, travesseiro na cama, tapete no fundo de tudo). **Toda borda comum ganha uma ordem:** quando a profundidade empata (< 3%, comum entre peças encostadas, como mesa × cadeira), valem as pistas da imagem, nesta ordem: dono do contorno (a borda comum cai dentro da envoltória convexa *local* de quem está atrás: a cadeira "morde" o retângulo da mesa), peça pequena na frente da grande, a que desce mais nas colunas da borda comum, a menor.
- **Ordem de entrada:** o tapete primeiro (**desenrola no lugar**). O tapete coberto por móveis é **completado por baixo deles** (envoltória convexa do tapete visível, só sob peças da frente; preenchido por espelho da própria estampa, sem modelo) e desenrola **antes** de os móveis pousarem nele. Depois, as peças da **frente para trás**, e os apoios antes do que está apoiado. Quando uma peça de trás entra, quem a cobre já está lá.
- **Peça grande que só poderia "surgir":** entra em bloco rígido com parte de quem a cobre (as menores primeiro), para ganhar um movimento de verdade.
- **Quadro de parede cortado só pelo topo:** cresce no lugar preso ao topo (0,85→1) ou desliza rente à parede pelo lado mais perto; descer do topo ("cair do teto") é o último recurso.
- **Peça ocluída:** "sobe por trás" do oclusor, com amplitude limitada para a borda cortada ficar sempre dentro dele. Se a amplitude possível for < 5% da altura, usar **escala 0,94→1 no ponto de contato**, com opacidade 100%.
- **Peça que toca a borda da foto:** desliza a partir dessa borda. Nunca cai.
- **Peça isolada e inteira:** queda curta (≤ 25% da altura do quadro) com quique (curvas de `render/choreography.py`).
- **Nada de fade de opacidade** (a crítica chamou de "fantasma"). Opacidade 100% desde o 1º quadro visível.
- **Sombra de contato cresce** conforme a peça se aproxima do chão (opacidade proporcional à altura). O reflexo acompanha o deslocamento.
- **Duração:** 1º quadro vazio na tela ≤ 0,5 s (o tapete ou a primeira peça já começam a entrar), total de 7–10 s e hold final ≥ 1 s com a foto exata.

### 3.5 Verificações automáticas (`composite/verify.py` reescrito)

1. O quadro 0 é `array_equal` à placa vazia.
2. O último quadro mestre é `array_equal` à foto original (na resolução do vídeo).
3. Em todo quadro, fora de (máscara de remoção ∪ caixa da peça em movimento), o quadro é `array_equal` à foto.
4. A placa é `array_equal` à foto fora da máscara de remoção.
5. Nenhum pixel com "alpha oculto" (parte da peça coberta por outra na foto) fica visível sem o oclusor na frente.
   **5b. A mesma regra, independente do grafo de oclusão** (`choreo.verify_independent`): os cortes vêm só da
   evidência clara gravada nas camadas (tapete/quadro atrás, profundidade com folga de 5%, dono do contorno com
   folga de 4×, nenhuma pista contrária) e da borda da foto; em todo quadro cada ponto de corte de uma peça na tela
   tem de estar coberto por uma peça já pousada ou do mesmo bloco. O `plan` replaneja com a relação que faltava
   (até 2 vezes) e o vídeo é recusado se sobrar algum.
6. Pixels fora de qualquer alpha e de qualquer efeito não mudam entre quadros consecutivos (anti-morph).
7. O `report.json` registra a área gerada (%), o número de peças, as peças deixadas no fundo, a semente escolhida e a nota.

---

### 3.6 Planejados e louças também saem: "cenário cru" (decisão do usuário, 16/11/2026)

O usuário pediu que cozinhas e banheiros também comecem crus. Por isso, no modo B, saem **os móveis soltos e também os fixos**:
armários planejados (de baixo e aéreos), ilha, bancada, coifa, geladeira, fornos e micro-ondas embutidos, louças sanitárias (vaso,
pia, cuba), gabinete, espelho com moldura e box. Ficam: portas, janelas, nichos embutidos na parede, tomadas e interruptores que já
aparecem na foto, teto e luminárias de teto.

**Como a sala crua é montada** (o modelo nunca inventa objetos; ele só continua as superfícies):
- **Por plano.** Cada parede, piso e teto vem da geometria (MoGe-2 + OneFormer + `build_room_model`). O preenchimento é feito na vista
  retificada daquele plano, copiando cor e textura **da parte visível do mesmo plano** ("pega o ambiente em volta e copia").
- **Parede quase toda escondida** (cozinha planejada de ponta a ponta): pintura lisa na cor média das paredes visíveis, com o
  gradiente de luz medido nas áreas visíveis. É o "cru" mais crível; reboco ou pintura lisa não tem detalhe para errar.
- **Revestimento com padrão** (azulejo, porcelanato, 3D): detectar a grade na vista retificada e repetir as placas visíveis
  (síntese periódica), mantendo rejunte alinhado. Sem placas visíveis para copiar, usar a cor lisa.
- **Luz que vem do móvel** (LED sob o aéreo, nicho iluminado, brilho no backsplash, reflexo no piso brilhante): sai junto com o móvel.
  É estimada como camada aditiva/de razão perto da fonte, removida da placa e devolvida na camada de efeito do móvel quando ele entra.
- Nada de pontos hidráulicos, tomadas ou janelas novas: superfície lisa onde não houver informação.

**Coreografia dos fixos (antes dos soltos):** base dos armários e ilha sobem do chão ou deslizam da parede; a bancada desce e
pousa sobre a base; aéreos descem do teto; eletros deslizam para os nichos; louças sobem do piso; coifa desce; depois entram os
móveis soltos e a decoração. As regras de §3.4 (sem pedaço oculto visível, sem fade) valem igual.

**Portão de qualidade:** se a nota da placa crua for ruim (costura, tom, objeto reconhecido pelo OneFormer na placa) ou a área
inventada passar do limite para aquela foto, o software refaz **mantendo os fixos** (só os soltos saem) e registra o motivo no
`report.json`. Nunca entregar uma placa ruim em silêncio.

**Ajuste da rodada de robustez 1 (planejados grandes):** em 10 fotos variadas a sala crua inventou a arquitetura atrás dos
planejados (parede de "pedra derretida", relevo inventado, bloco de carpete flutuando), e numa cozinha a nota ainda aprovou. Por
isso os fixos ficam na foto **por padrão**, sem tentar a sala crua, quando só eles cobrem 10% ou mais da foto ou quando tirar tudo
reconstruiria 35% ou mais dela (`keep_builtins_reason`; o motivo vai para `placa.planejados_por_padrao` no `report.json`). Abaixo
disso a sala crua continua sendo a 1ª tentativa. Painéis fixos na parede que o OneFormer cola na máscara da cama ou da mesa
(cabeceira-painel de hotel, painel ripado) são recortados e ficam na foto, como acabamento de parede; uma mesa presa a um painel
alto (divisória com TV), uma bancada de trabalho de parede a parede e um "armário" que é só um painel no plano da
parede viram planejados. O pedaço do painel que a cama escondia é continuado para baixo (espelhando o próprio painel),
em vez de virar parede lisa. Com os planejados mantidos, o que está guardado neles (livros na estante embutida, vaso
na prateleira) fica junto: tirar só isso abriria buracos cinzentos no móvel que ficou.

**Revisão (rodada 1 de correções do SDXL, 17/11/2026):** a regra acima contrariava a decisão do usuário (os planejados
saem) e escondia o problema. Agora a **sala crua é sempre a 1ª tentativa** (`plate.KEEP_BUILTINS_FIRST = False`); o
`keep_builtins_reason` só vai para o relatório como risco (`placa.risco_planejados`). Os planejados ficam na foto apenas
se o portão reprovar a crua, e o `report.json` diz quais ficaram e por quê (`placa.cobertura`), com aviso na tela e na
narrativa. O que permitiu voltar atrás: o portão ganhou falhas duras que pegam a "parede inventada" (estrutura nova,
material, contorno; ver PT-V3-4). `KEEP_BUILTINS_FIRST = True` volta ao comportamento antigo.

**Teste de aceite:** as fotos de cozinha planejada (paisagem e retrato) e de banheiro do conjunto de teste do usuário: placa crua sem
objetos inventados, linhas parede/piso retas, revestimento contínuo, último quadro idêntico à foto.

## 4. Código atual: o que é reaproveitado, substituído ou mantido como fallback

| Componente (`mobilia/`) | V3 |
|---|---|
| `analysis/geometry.py` (MoGe-2), `analysis/room.py` (`build_room_model`) | **Reaproveitado sem mudança.** Dá as linhas parede/piso para o LaMa por plano e a profundidade da caixa vazia para o ControlNet. |
| `analysis/segmentation.py` (OneFormer) | **Estendido:** modo panoptic + união com a semântica + detecção automática do modo. |
| `analysis/weights.py`, `cli prefetch`, `bootstrap.py`, `warmup.py` | **Estendidos:** novos pesos, SHA-256 e leitura completa dos `.safetensors` no warm-up (para a varredura do Defender não cair no tempo do vídeo). |
| `hardware.py`, `diagnostics.py` | **Estendidos:** VRAM livre (`torch.cuda.mem_get_info`), detecção do *sysmem fallback* e RAM do sistema ≥ 16 GB para SDXL. |
| `layout/` (solver, regras, slots, catálogo) | **Modo A:** reaproveitado inteiro. **Modo B:** não usado. |
| `layout/entrances.py`, `layout/narrative.py` | **Reescritos para camadas 2D** (borda → desliza, oclusão → sobe por trás, ordem por profundidade). As ideias ficam. |
| `render/choreography.py` (easing, mola, marcos) | **Reaproveitado.** |
| `render/blender_worker.py`, `render/scene_spec.py`, `render/layers.py`, `render/calibration.py`, `assets/polyhaven.py` | **Fallback** (motor v2 completo) e, no modo A, **só o passe Z**. Não remover antes do modo A estar pronto. |
| `render/budget.py` | **Reescrita parcial:** nova ordem de corte (sementes → hires → refino SDXL → resolução). A sonda passa a ser a 1ª chamada SDXL. |
| `composite/compositor.py` (`placa·S·(1−A)+C`, cópia bit a bit), `LayerAccumulator`, `blend_frames` | **Reaproveitado.** Placa = sala vazia (B) ou foto vazia (A), `S` = razão foto/placa. |
| `composite/stream.py` | **Portar para torch na GPU, por caixa de camada.** É o maior risco de tempo do modo B: 15–45 s em numpy a 1920 px [E], 33,7 s medidos [M] a 1280×846 na CPU. |
| `composite/verify.py` | **Reescrito** (§3.5). |
| `composite/contact.py`, `composite/grade.py` | `contact.py` vira o gerador da sombra de contato crescente. `grade.py` fica para o modo A (casar a cor da peça gerada). |
| `video/encoder.py` (NVENC/x264, BT.709) | **Reaproveitado.** Opcional: exportar também um MKV sem perdas para auditoria. |
| `pipeline.py`, `jobs.py`, `ui/` | **Ajuste pequeno:** ramo modo A/B, override manual e texto "o fim é idêntico à foto na resolução do vídeo". |
| **Novos módulos** | `mobilia/removal/` (instâncias, máscaras de efeito, placa por plano, harmonização, sementes), `mobilia/gen/` (SDXL + ControlNet + embeddings pré-calculados), `mobilia/layers2d/` (RGBA pré-multiplicado, efeitos, oclusão, coreografia 2D). |
| `docs/ARQUITETURA.md` §1, EULA | **Reescritos** (§1.3 e §5.4). |

---

## 5. Modelos, licenças, VRAM, downloads e tempo

### 5.1 Lista fechada de modelos

Tudo foi conferido [V] e **nada é gated**.

| Função | Modelo | Licença | Download | VRAM pico [E] | Modo |
|---|---|---|---|---|---|
| Geometria (já existe) | `Ruicheng/moge-2-vitl-normal` | MIT | 1,32 GB | ~2 GB | A, B |
| Segmentação (já existe) | `shi-labs/oneformer_ade20k_swin_large` | MIT | 0,88 GB | ~2–3 GB | A, B |
| Máscara por peça | `facebook/sam2.1-hiera-base-plus` | Apache-2.0 | 0,32 GB | < 1 GB | A, B |
| Sala vazia | big-lama (TorchScript, `advimman/lama`) | Apache-2.0 | 0,21 GB | ~1–2 GB | B |
| Gerador SDXL (**um único UNet**) | `SG161222/RealVisXL_V5.0_Lightning` **ou** `stabilityai/stable-diffusion-xl-base-1.0` + `ByteDance/SDXL-Lightning` (licença limpa) | OpenRAIL++ | 5,14 GB (+0,39 da LoRA) | **~6,7–7,2 GB** | A, B (refino) |
| VAE | `madebyollin/sdxl-vae-fp16-fix` | MIT | 0,33 GB | 0,17 GB | A, B |
| ControlNet depth | `diffusers/controlnet-depth-sdxl-1.0-small` (ou `xinsir/controlnet-union-sdxl-1.0`, Apache-2.0, 1,25 GB após conversão fp16 num espelho) | OpenRAIL++ / Apache | 0,32 GB | 0,3 GB | A, B |
| Text encoders | **não vão para o cliente**: embeddings pré-calculados no build para os prompts fixos | OpenRAIL++ | ~MB | 0 | A, B |
| Bordas (opcional) | `ZhengPeng7/BiRefNet` | MIT | 0,44 GB | ~1 GB | A, B |
| Objetos pequenos (V3.1) | `IDEA-Research/grounding-dino-tiny` | Apache-2.0 | 0,69 GB | ~1 GB | B |

**Proibidos (não comerciais) [V]:** tudo derivado do FLUX.1-dev (Fill, Kontext, Redux, OmniEraser, Insert Anything, UNO, In-Context-LoRA, Hyper-FLUX), FLUX.2-klein-9B, SDXL-Turbo, DMD2, pix2gestalt, LBM relighting, IC-Light V2, RMBG-2.0, MAT, MatAnyone, ObjectClear, SV3D, Stable Zero123 e Depth-Anything-V2-Large.

**Excluídos por outro motivo:**
- SD 3.5 e SF3D: comercial só abaixo de US$ 1 M de receita, e são gated.
- FLUX.1-schnell e SAM3: gated, o download anônimo falha no cliente.
- Qwen-Image-Edit-2509: 57,7 GB.
- FLUX.2-klein-4B: ~16 GB em bf16; fica como candidato V3.1 com quantização.
- Hyper-SD, BrushNet e Harmonizer: licença duvidosa.
- ViTMatte: dataset de treino de pesquisa.
- AnyDoor e Zero123++: sem licença.
- TRELLIS: depende de nvdiffrast, não comercial.

**Obrigações:**
- OpenRAIL++: incluir as restrições de uso (Attachment A) no EULA, enviar a cópia da licença com os pesos e marcar os pesos modificados (por exemplo, LoRA fundida).
- MIT e Apache: incluir os textos das licenças.
- O ADE20K e o Places2 (treino do OneFormer e do LaMa) são datasets de pesquisa. O risco residual é baixo e o produto atual já o carrega.

### 5.2 Downloads e pacote

| Pacote | Adicional sobre a v2 |
|---|---|
| Modo B só LaMa (SAM 2.1 + big-lama [+ BiRefNet]) | **+0,5–1,0 GB** |
| + SDXL (refino B e modo A): UNet 5,14 + VAE 0,33 + ControlNet 0,32 + embeddings | **+5,8 GB** |
| Total do motor V3 em `%LOCALAPPDATA%` (com o v2 e o Poly Haven) | **≈ 12–14 GB** |

- O PyTorch continua o do repositório: `torch==2.15.0+cu132` (`requirements/runtime-cuda.txt`), que já suporta Blackwell sm_120. A recomendação de "fixar 2.7/cu128" está **superada**. Não rebaixar.
- Novas dependências: `diffusers` e `peft`, pure-Python, Apache-2.0. Usar **só `from_pretrained`** de pastas diffusers (o `from_single_file` do diffusers 0.35 quebra com transformers 5.x). Fixar as versões depois do teste no Windows. Sem xformers e sem `torch.compile`.
- O `Sam2Model` já existe no transformers instalado. O big-lama roda por `torch.jit`.
- Hospedar um **espelho próprio** dos pesos com SHA-256 (hashes já registrados em `p_staging/models_sha256.txt` e `r_removal/RELATORIO.md`). Download no 1º uso, empacotamento com PyInstaller onedir.

### 5.3 VRAM (8 GB)

- Mantém-se a regra atual "um modelo na GPU por vez": MoGe → OneFormer → SAM2 → LaMa → SDXL, liberando VRAM entre etapas.
- O SDXL é a única etapa apertada (6,7–7,2 GB). Por isso:
  - text encoders fora do runtime;
  - VAE em tiles;
  - medir a VRAM livre antes de carregar e, abaixo de ~7 GB, usar `enable_model_cpu_offload`;
  - abaixo de ~6,5 GB, **pular o refino SDXL** (modo B) ou usar o fallback v2 (modo A). O offload por modelo leva o
    UNet fp16 inteiro (4,9 GiB) para a GPU de uma vez, mais ControlNet e ativações (estimado; revisão de código da
    rodada 2: o limite antigo de 5 GB deixava passar GPU em que o UNet não cabe). No modo B a LaMa e o OneFormer da
    nota saem da GPU enquanto o SDXL gera (`pipeline_b.VramSwap`), e essa VRAM conta como livre.
- **O sysmem fallback do driver NVIDIA no Windows** não dá OOM: deixa o SDXL de 5 a 20 vezes mais lento em silêncio. O diagnóstico precisa detectar isso (tempo por passo da sonda) e registrar no `report.json`.
- RAM do sistema: SDXL com offload precisa de ≥ 16 GB.

### 5.4 Tempo por etapa na RTX 5070 Laptop 8 GB [E], cache quente

**Modo B**

| Etapa | CPU medido [M] | GPU estimado [E] |
|---|---|---|
| Preparo + MoGe-2 + OneFormer (panoptic + semântica) | 18 s + 13,9 s | 3–6 s |
| SAM 2.1 (~18 caixas) | 2,8 s | 0,5–1,5 s |
| LaMa por plano, 2 passadas (~12 chamadas) | 15–33 s por passada | 2–5 s |
| Harmonização + seleção de sementes (numpy/torch) | — | 1–3 s |
| Carga SDXL (quente / frio) | 15 s | 3–6 s / 10–25 s |
| Refino SDXL, 2–3 sementes × 6 passos efetivos, ~0,7 MP | 96 s (1 semente) | 4–12 s |
| Camadas + efeitos | 1,2 s | < 1 s |
| Composição de ~240 quadros + NVENC | 33,7 s (1280×846) | 4–10 s em torch/GPU (15–45 s se ficar em numpy) |
| **Total** | — | **≈ 30–75 s** (sem SDXL: 25–60 s) ✔ ≤ 120 s |

**Modo A**

| Etapa | GPU estimado [E] |
|---|---|
| Análise + layout (já existe) | 10–15 s |
| Passe Z do layout | 2–6 s |
| Carga SDXL | 3–25 s |
| 3 sementes a 0,75 MP + nota | 6–14 s |
| Passe hires nos móveis | 5–15 s |
| OneFormer + SAM 2.1 + camadas | 2–4 s |
| Composição + codificação | 5–10 s (GPU) |
| **Total** | **≈ 40–100 s** ✔ |

Os custos únicos (primeiro download, primeira carga fria) continuam fora da conta, como no PLANO_V2 §1.

---

## 6. Pacotes de trabalho (em ordem de execução)

Cada pacote tem testes de aceite. Os que precisam de GPU estão marcados **[GPU]**. Em CI e na sandbox, rodam com stubs.

### PT-V3-0: Base, licenças e fixtures
- Reescrever a "regra absoluta" (ARQUITETURA §1) e criar `docs/LICENCAS_V3.md` e o rascunho do EULA (OpenRAIL++ Attachment A + MIT/Apache).
- Conseguir **2 fotos mobiliadas com licença comercial** (Commons CC0/PD, conferindo a página de origem): a de Nice (difícil) e **uma típica de anúncio** (piso fosco, sofá + mesa de centro). Registrar a URL e a licença em `tests/fixtures/rooms/*/SOURCE.md`.
- Gerar fixtures cacheadas (seg.npz, room.json, geometry) para os testes rodarem sem modelos.
- **Aceite:** `docs/LICENCAS_V3.md` lista cada modelo com a licença e o link conferidos. A suíte atual continua verde. As fixtures carregam em < 1 s.

### PT-V3-1: Pesos, bootstrap e diagnóstico
- `analysis/weights.py`: SAM 2.1, big-lama, UNet SDXL, VAE, ControlNet e embeddings pré-calculados, com SHA-256 e espelho configurável.
- Script de build que pré-calcula os embeddings dos prompts fixos por tipo de ambiente.
- `hardware.py`: VRAM livre, RAM e sonda de sysmem fallback. Perfis "B sem SDXL", "B com SDXL" e "A".
- Warm-up que lê os `.safetensors` inteiros uma vez.
- **Aceite:** um download corrompido (hash errado) é rejeitado e baixado de novo. Um modelo gated ou ausente gera uma mensagem clara. O teste unitário dos perfis cobre 4, 6, 8 e 12 GB simulados. Nenhum text encoder no pacote do cliente.

### PT-V3-2: Instâncias de móveis (modo B)
- `removal/instances.py`: OneFormer panoptic ∪ semântica, SAM 2.1 por caixa (fallback se IoU < 0,6), maior componente conexo, fusão de fragmentos, regra móvel vs fixo e detecção automática do modo (A ou B).
- **Aceite:** na foto de Nice, ≥ 16 das 18 peças do protótipo detectadas, com a mesa de jantar presente e os 2 tapetes separados. Na foto Stafford vazia, o modo detectado é A. Na foto típica, sofá e mesa de centro detectados com IoU ≥ 0,85 contra uma máscara anotada à mão. Tempo [GPU] ≤ 3 s.

### PT-V3-3: Máscara de efeitos e sala vazia por plano (LaMa)
- `removal/plate.py`: 2 passadas, faixa espelhada de sombra/reflexo, ray-cast na caixa vazia, vistas retificadas com **a máscara inteira**, harmonização por plano (histograma + Poisson), regiões protegidas (janelas e portas: LaMa no espaço da imagem com contexto local) e cópia de volta byte a byte.
- **Aceite:**
  - `array_equal(placa[~mask], foto[~mask])`;
  - a linha parede/piso projetada fica a ≤ 3 px da borda detectada do rodapé na placa (no protótipo, falhava com LaMa puro);
  - diferença de média de cor (Lab) entre o piso preenchido e o anel ≤ 3 ΔE em cada plano;
  - o OneFormer roda na placa e **não encontra nenhuma instância de móvel** com área > 0,5% da imagem;
  - [GPU] ≤ 5 s.

### PT-V3-4: Refino SDXL da sala vazia (opcional, perfil "B com SDXL")
- `gen/sdxl.py`: UNet único + ControlNet depth da caixa vazia + Lightning, força 0,3–0,5 só no piso e nos vãos, janelas e portas protegidas, 2–3 sementes e nota (costura + objetos novos pelo OneFormer). Fallback para a placa LaMa.
- **Aceite:**
  - o mesmo teste do PT-V3-3, mais "nenhuma porta, janela ou tomada nova" (instâncias novas do OneFormer = 0);
  - os pixels de janela são `array_equal` à placa LaMa;
  - com o SDXL desligado o pipeline continua funcionando;
  - [GPU] carga + 3 sementes ≤ 20 s quente, VRAM pico ≤ 7,2 GB e sem sysmem fallback registrado.
- **Estado (17/11/2026, CPU):** implementado (`mobilia/gen/sdxl.py`, `mobilia/removal/plate_score.py`). Pesos:
  RealVisXL V5.0 Lightning (UNet fp16), VAE fp16-fix e ControlNet depth small, revisão e SHA-256 em
  `analysis/weights.py` (`SDXL_WEIGHTS`, ≈ 7,4 GB; `docs/LICENCAS_V3.md` §1.2). Como funciona:
  - estágio **depois da LaMa em cada tentativa do portão** (`build_plate_gated(refine=...)`), ligado pelo perfil
    "B com SDXL" (`pipeline_b.sdxl_choice`: GPU com VRAM, RAM para offload e pesos; 8 GB usa
    `enable_model_cpu_offload` porque o OneFormer da nota divide a VRAM). `MOBILIA_SDXL=0` desliga, `=1` força
    (até na CPU, para testes). Sem pesos/VRAM ou com erro: fica a LaMa e o motivo vai para `report.json`
    (`refino_sdxl`: ligado, motivo, sementes com nota e motivos, semente escolhida, nota da LaMa, tempos);
  - **escopo "piso"** (padrão): o SDXL gera só o piso; paredes e teto lisos e vãos sem plano viram **pintura lisa**
    (membrana com a cor da borda + grão do material visível); parede com textura fica na LaMa por plano. No teste
    com SDXL em toda a máscara ("tudo") o modelo inventou radiador e painéis nas paredes da Nice;
  - ControlNet depth da **caixa vazia** (ray-cast do `RoomModel`, escala casada com o MoGe fora da máscara; dentro
    nunca o MoGe, que tem os móveis); início = tom esperado do plano + grão da LaMa limitado a ±5 níveis (sem o
    contorno do quadro/sofá), força 0,75, 8 passos (6 efetivos), CFG 2,0; prompt "empty unfurnished room" com a
    cor/tipo do piso e a cor das paredes medidos na foto; prompt negativo contra móveis, porta/janela/tomada
    novas, névoa e borrão;
  - composição só dentro da máscara (rampa por dentro), janelas/portas/cortinas visíveis e os pixels mascarados
    rotulados "protegido" ficam `array_equal` à LaMa (com SDXL ali, o balcão visto pela janela virou parede com
    radiador); cor casada por membrana a partir da borda, por plano;
  - 2 (CPU) ou 3 (GPU) sementes, escolhidas pela própria nota do portão; semente com móvel ou porta/janela nova
    (OneFormer) nunca é escolhida; se nenhuma supera a LaMa, fica a LaMa.
- **Nota contínua do portão** (próximo passo 4): 100 menos pontos perdidos por defeito (`plate.SCORE_WEIGHTS`),
  agora com **névoa** (contraste local e véu leitoso contra o material visível do mesmo plano), **fantasma**
  (correlação, por peça, entre a estrutura que sobrou na placa e a peça da foto, só onde a foto mostra o móvel),
  **placas soltas** (manchas que destoam da cor esperada do plano) e **porta/janela nova** (OneFormer). Reprovada:
  0,6 × nota sem teto (contínua, < 60). Medido nas placas LaMa das 2 fixtures a 768 px: típica **48 → 29**
  (antes 90 sem teto) e Nice **23 → 14**. Calibração só nas 2 fixtures e em casos sintéticos; **não** foi
  recalibrada nas 39 fotos do conjunto de teste (os limites de reprovação por névoa/fantasma/placas são
  conservadores: 0,6 / 0,6 / 35%).
- **Teste de fumaça real (CPU, 4 núcleos, bf16, 768 px, 3 sementes; `docs/poc/v3/sdxl_folha.jpg`: foto | LaMa |
  SDXL):** carga 5–33 s (fria/quente), **≈ 21 s por semente** (6 passos a 768×400, 3,4–3,8 s/passo; com outro
  processo usando a CPU, 13–29 s/passo), nota ~1 s por semente sem OneFormer. Nota sem teto: típica **45,8 →
  82,3** (aprovada), Nice **23,3 → 47,0** (reprovada). A olho: na típica o piso vira porcelanato bege coerente
  com o visível, sem a névoa nem o "carpete" inventado; sobram a mesinha fantasma colada à cortina (pixels
  protegidos, ficam na LaMa) e a sombra do quadro **fora** da máscara (máscara de efeitos não pegou a sombra).
  Na Nice as paredes ficam lisas e sem as placas soltas, mas o piso de granilite sai liso demais com manchas e o
  quarto visto pela porta vira um borrão bege-rosado: **melhor que a LaMa, ainda não entregável**. Não medidos:
  GPU (tempo, VRAM, sysmem fallback), as 39 fotos do conjunto de teste e a nota do crítico.
- **Rodada 1 de correções (17/11/2026, CPU, 8 fotos fixas a 768 px, semente 7, 2 sementes SDXL).** A avaliação
  independente deu 2/10 à sala vazia e mostrou que a nota subia com o SDXL enquanto o piso piorava. Mudanças (todas
  gerais, nenhum limiar ajustado a uma foto):
  - **Piso sem material inventado:** o prompt não nomeia mais o tipo do piso ("smooth tiled floor" fazia o SDXL
    desenhar placas com rejunte sobre porcelanato e ardósia sobre madeira); só a cor, e o negativo ganhou "tile grout,
    bricks, pipe". E, por padrão (`RefineConfig.floor_detail="lama"`), **do SDXL no piso só fica a baixa frequência**
    (tom e luz: tira névoa e a mancha da silhueta) e a textura vem da LaMa por plano, que continua o piso visível
    (`floor_texture_from_lama`, corte em 1% da diagonal, textura limitada a ±12 níveis). Testado antes com início pela
    LaMa e força 0,5: o SDXL continuou desenhando placas (descartado).
  - **Falhas duras na nota** (`plate_score.texture_metric` e `contour_metric`, `PlateQuality.hard`): *estrutura nova*
    (densidade de retas longas do LSD dentro do preenchimento contra o anel do mesmo plano; retas que continuam uma
    reta do anel ou saem da máscara, como rodapé e quina, não contam), *material* (orientação dessas retas contra o
    anel), *névoa por alta frequência* (vale também sem plano), *contorno* (degrau de cor ou linha ao longo da borda da
    máscara onde os dois lados são a mesma superfície: o retângulo do quadro, a silhueta do sofá). Acima de 0,6 cada uma
    reprova sozinha; a escolha de tentativa e de semente olha primeiro a **pior** falha dura, depois a nota média. Uma
    semente que piora qualquer falha dura da LaMa (mais de 0,15 e acima de 0,3) nunca troca a LaMa.
  - **Reconferência OneFormer relativa:** objeto que já aparecia na placa LaMa não descarta a semente (antes 14 de 22
    tentativas jogavam fora o SDXL por "toalha"/"janela" que vinham da própria LaMa). Na placa final o motivo continua.
  - **Erro no refino nunca derruba o trabalho:** `refine_plate` e o portão pegam qualquer exceção (OneFormer sem VRAM
    na nota, casamento de cor...), liberam a CUDA e ficam com a LaMa, com o motivo no relatório.
  - **VRAM:** `pipeline_b.sdxl_recheck` reconfere a VRAM livre **depois** de LaMa e OneFormer carregados (liga o
    offload abaixo de 7 GB livres e desliga o SDXL abaixo de 6,5 GB; rodada 2) e o relatório guarda o pico por etapa
    (`refino_sdxl.vram_pico_mb_por_etapa`). **Continua sem medida na GPU real.**
  - **Planejados:** a sala crua é sempre a 1ª tentativa (§3.6, revisão); quadro/espelho/TV pendurados têm a faixa de
    sombra procurada na parede; cor esperada do que não tem plano vem só de piso/parede/teto em volta.
  - **Coreografia:** o relatório traz a cobertura da verificação independente (`oclusao_cobertura_evidencia`: bordas
    do grafo com evidência própria / bordas do grafo). A evidência ainda usa as mesmas pistas do grafo (não foi
    acrescentada uma pista independente nesta rodada).
  - **Resultado medido** (portão; nota sem teto; antes → agora): sala_sofas_brancos reprova → reprova; nice_terrazzo
    reprova (SDXL usado, 71) → reprova (todas as sementes com "sofá" novo, fica a LaMa); foto interna (não versionada) reprova
    (SDXL 87 com piso de placas) → **aprova 85, sala crua**; kitchen_frontal_island reprova → reprova (agora tenta a
    crua: 67–68 com SDXL); hotel_bathroom_accessible aprova 87 → aprova 83 (a crua reprovou, planejados mantidos e
    ditos no relatório); dining_backlit_window, living_bookshelves_cluttered e living_portrait_window reprovam nas duas.
    **A olho:** piso sem placas inventadas na foto interna e sem ardósia no retrato (o problema nº 1 da avaliação); a
    sala de sofás continua com piso de "carpete" manchado e névoa clara à direita; o quarto pela porta da Nice e a sala lotada continuam
    ruins; o banheiro mantém louças. **Falso aceite que continua:** a foto interna aprova com uma mancha escura borrada
    onde estava a cozinha (região sem plano medido, 28% do preenchimento; nenhuma falha dura pega) e a faixa marrom na
    base do vidro. Não medidos: GPU, as 39 fotos, segundo avaliador.

### PT-V3-5: Camadas 2D e anti-morph
- `layers2d/layers.py`: ordem de pintura, RGBA pré-multiplicado `C = foto − (1−A)·fundo`, efeito = razão limitada só perto da peça e mais escuro que a placa, e correção do resíduo restante **na placa**.
- **Aceite:**
  - pilha completa vs foto: máximo ≤ 2 níveis (só arredondamento; era p99 = 10, máximo 40);
  - pixels fora de alpha ∪ efeito que mudam entre o 1º e o último quadro = **0%** (era 7,1%);
  - [GPU] < 1 s.

### PT-V3-6: Coreografia 2D sem partes ocultas
- `layers2d/choreo.py`: tapete desenrola; entrada da frente para trás com os apoios primeiro; ocluída sobe por trás (amplitude limitada) ou escala no lugar; borda → desliza; isolada → queda ≤ 25% + quique; sem fade; sombra de contato crescente; 1º quadro vazio ≤ 0,5 s.
- **Aceite:**
  - teste automático "pixel oculto visível" = 0 em todos os quadros (Nice e típica);
  - nenhum quadro com opacidade de peça entre 0,05 e 0,95;
  - peças tocando a borda nunca têm deslocamento vertical vindo de cima;
  - duração entre 7 e 10 s.

### PT-V3-7: Composição em GPU, verificação e codificação
- Portar `composite/stream.py` para torch (por caixa de camada), reescrever o `verify.py` (§3.5) e o `budget.py` (nova ordem de corte), mais um MKV sem perdas opcional para auditoria.
- **Aceite:**
  - o quadro 0 é `array_equal` à placa e o último é `array_equal` à foto (também após decodificar o MKV);
  - invariante fora da máscara ∪ caixa móvel verificado em 100% dos quadros;
  - mesma saída em CPU e GPU (diferença ≤ 1 nível);
  - [GPU] 240 quadros a 1920×1440 + NVENC ≤ 10 s.

### PT-V3-8: Integração do modo B (pipeline + UI)
- Ramo em `pipeline.py`/`jobs.py`, override manual na UI, textos ("o fim é idêntico à foto na resolução do vídeo"; aviso "imagem ilustrativa" opcional) e `report.json` com a área gerada, as peças, a semente e as notas.
- **Aceite:**
  - e2e com fixtures, na CPU, gera o MP4 e passa todas as verificações;
  - a UI só pede o tipo de ambiente;
  - **[GPU] total ≤ 120 s quente nas 2 fotos, medido na RTX 5070 Laptop do usuário**;
  - avaliação visual: a sala vazia sem invenções e a nota do crítico ≥ 7 na foto típica.
- **Estado (PT-V3-8):** integrado (`pipeline_b.py`, ARQUITETURA §4.5). Validado na CPU: e2e rápido e com big-lama
  real nas 2 fixtures (MP4 + MKV de auditoria, §3.5 em 100% dos quadros) e uma execução completa com todos os
  modelos reais (MoGe-2, OneFormer, SAM 2.1, big-lama, OneFormer da nota). **O portão reprova as placas das 2
  fixtures** (cor/costura de parede, móvel reencontrado, área > 70% na sala de sofás): sem MP4 por padrão, com
  aviso. Não medidos: o tempo na RTX 5070 e a avaliação visual do crítico.
  Revisão: em mais 6 fotos variadas do conjunto de teste (retrato, banheiro só com fixos, panorâmica, cama
  lotada, loft com mesa na borda, cozinha quase vazia) o portão também reprovou todas (0/6), com defeitos reais
  vistos a olho (fantasma da cama, mancha escura no piso, quadro inventado); fotos vazias foram para o modo A.
  O big-lama dá o mesmo resultado na GPU, então **o modo B hoje quase nunca entrega por padrão**: a qualidade da
  placa (PT-V3-4, SDXL) é o próximo gargalo, não o portão.
- **Avaliação final (17/11/2026, `docs/poc/v3/RESULTADO.md`):** vídeos de demonstração a 1440 px nas 2 fixtures
  (`modo_b_tipica.mp4`, `modo_b_nice.mp4`, só com "gerar mesmo assim") e folha `modo_b_folha.jpg`. Notas: sala
  vazia 2, móveis 6–7, animação 5, **geral 3/10 nas duas** (aceite pedia ≥ 7 na típica: **não atingido**). Tempo
  medido na CPU: 156 s (típica) e 163 s (Nice); GPU estimado 20–40 s. Verificações independentes no MKV: fim =
  foto e início = placa byte a byte, sem morph. Refutado: "0 pixel oculto visível" — na Nice a mesa de jantar entra
  antes das cadeiras com o buraco delas (grafo de oclusão sem relação quando a profundidade empata; a verificação não
  vê). Corrigido: o anel das camadas agora segue a dilatação da máscara "ampla" do portão (antes, o contorno dos
  móveis voltava da foto para a sala vazia, 4,9% da imagem na Nice).
- **Correção da coreografia (17/11/2026, depois da avaliação):** os defeitos 2, 3, 4 e 6 da avaliação final
  (`RESULTADO.md`) foram tratados de forma geral (§3.4 e §3.5, regra 5b): (a) pistas da imagem quando a
  profundidade empata, então toda borda comum tem ordem (na Nice, mesa e as duas cadeiras da frente entram como bloco
  rígido em queda; antes a mesa caía sozinha com o buraco das cadeiras); (b) verificação independente do grafo,
  ligada ao `choreo.verify` e ao `ModeBVerifier` (regra 5b), que pega o defeito da Nice quando a relação falta
  (75 300 px de corte à vista na fixture com o grafo antigo; o `plan` repara para 0); (c) tapete completado por baixo
  dos móveis e desenrolando primeiro nas 2 fixtures; (d) sofá principal da típica sobe do chão (sem "surgir"),
  0 "surgir" nas 2 fixtures e em 160 cenas sintéticas aleatórias; quadro no topo cresce no lugar. Medido só nas
  fixtures (placa Telea) e em cenas sintéticas, sem vídeo novo nem nota de crítico. Limite: o preenchimento do tapete
  é por espelho (sem modelo); sob uma mesa de vidro grande ele vira retalhos da estampa, visíveis por ~1 s antes de
  a mesa pousar.

### PT-V3-9: Modo A com o motor novo
- Passe Z do layout (Blender só Z), `gen/sdxl.py` em modo inserção (máscara = pegadas + sombras), 3 sementes + nota, hires pass nos móveis, recolocação por máscara, camadas com a foto vazia como placa, completamento SDXL da parte oculta só para peças que caem e fallback v2.
- **Aceite:**
  - o 1º quadro é `array_equal` à foto vazia;
  - fora da máscara de inserção, todo quadro é `array_equal` à foto;
  - a nota da semente escolhida ≥ limiar e nenhuma classe esperada faltando;
  - "pixel oculto visível" = 0;
  - [GPU] ≤ 120 s;
  - avaliação visual: nota ≥ 7 na Stafford (contra 4,5 do R2).

### PT-V3-10: Empacotamento, EULA e benchmark final
- PyInstaller onedir com as novas dependências fixadas, EULA com as restrições OpenRAIL++ e os textos MIT/Apache, CI com torch cu132 e testes com stubs.
- Benchmark na máquina do usuário (§7.2).
- **Aceite:**
  - instalação limpa no Windows: o primeiro uso baixa os pesos e confere o SHA-256;
  - os 2 modos rodam na RTX 5070 Laptop dentro de 120 s quente;
  - o relatório de benchmark é anexado em `docs/`.

**Ordem e dependências:** 0 → 1 → 2 → 3 → 5 → 6 → 7 → 8 (o **modo B entregável sem SDXL** já sai aqui) → 4 (refino SDXL) → 9 (modo A) → 10.

**Estado (17/11/2026):** PT-V3-0 a PT-V3-8 implementados e testados na CPU. O modo B **funciona de ponta a ponta,
mas não é vendável sem uma sala vazia melhor**: o portão reprova a placa em quase todas as fotos (2 de 2 fixtures,
33 de 39 fotos mobiliadas variadas). PT-V3-4 (refino SDXL do piso + paredes lisas) e a nota contínua do portão
foram implementados e testados na CPU nas 2 fixtures (ver PT-V3-4). Próximos passos, em ordem: (1) medir na RTX
5070 Laptop do usuário (§7.2), agora com o SDXL; (2) rodar as 39 fotos com o SDXL e recalibrar os limites de
névoa/fantasma/placas soltas com a folha do olho; (3) máscara de efeitos que pegue a sombra dos quadros e
embeddings dos prompts pré-calculados (tirar os text encoders do pacote). A relação de
oclusão entre peças encostadas com a mesma profundidade e a verificação de "buraco" independente do grafo foram
feitas (ver "Correção da coreografia" no PT-V3-8). PT-V3-9 e PT-V3-10 não começaram.

---

## 7. Riscos abertos e o que validar na GPU do usuário

### 7.1 Riscos

1. **Qualidade da sala vazia:** é o 1º quadro, o que o espectador vê primeiro. Hoje está em 2/10 nas 2 fixtures (avaliação final do PT-V3-8). Pisos polidos ou estampados e peças grandes encostadas na parede são os piores casos. Mitigações: §3.2, regiões protegidas e seleção de sementes. Pode ser preciso aceitar "imagem ilustrativa".
2. **Invenções do SDXL** (porta, tomada, vista de janela): risco comercial e jurídico (CDC, art. 37). Mitigação: SDXL só no piso, regiões protegidas e descarte automático de sementes com objetos novos.
3. **Partes ocultas:** sem completamento amodal comercial (pix2gestalt é NC). Depende das regras de coreografia. No modo A, o completamento é por SDXL e pode alucinar.
4. **Embutido vs solto:** o ADE20K não distingue. Painéis, prateleiras e armários podem ser removidos ou mantidos errado.
5. **Objetos pequenos** ficam "colados" na sala vazia (controle remoto em cima de um sofá que sumiu). A detecção aberta custa tempo e VRAM.
6. **Modo A:** sensibilidade alta a semente e prompt (1 de 13 bom), objetos derretidos e móveis moles. Depende do hires pass e da nota automática.
7. **Tempo:** a composição de ~240 quadros em numpy estoura o orçamento. Precisa ir para a GPU.
8. **VRAM e driver:** o sysmem fallback no Windows deixa tudo lento em silêncio. Com 8 GB de RAM do sistema não dá para usar SDXL com offload.
9. **Licenças:**
   - RealVisXL tem procedência de treino desconhecida (alternativa: SDXL base + Lightning);
   - OpenRAIL++ exige repassar as restrições;
   - nenhum derivado do FLUX.1-dev pode entrar por bibliotecas ou workflows;
   - os datasets de treino do OneFormer e do LaMa são de pesquisa (risco baixo, já existente).
10. **Exatidão:** "igual à foto" vale nos quadros mestres e na resolução do vídeo. O MP4 tem perda e fotos > 1920 px são reduzidas.
11. **Fotos de teste:** a de Nice (CC0 pelo Commons, não reconferida por causa de HTTP 429), a do rawpixel (CC0 pelo Openverse) e a Stafford (CC BY-SA 3.0) servem **só para teste interno**. Falta uma foto típica de anúncio com licença comercial.
12. **Avaliação:** as notas vêm de um juiz olhando quadros, não de um estudo com usuários.

### 7.2 O que medir na RTX 5070 Laptop 8 GB do usuário (antes de vender)

1. SDXL (UNet único) + ControlNet depth + Lightning a 0,75 MP e 1,5 MP, com e sem `enable_model_cpu_offload`: tempo por passo, VRAM pico e presença de sysmem fallback.
2. Carga fria vs quente do UNet (inclui a varredura do Defender) e o efeito do warm-up.
3. MoGe-2 + OneFormer panoptic + SAM 2.1 + LaMa por plano: tempo real por etapa.
4. Composição torch de 240 quadros a 1920×1440 + NVENC.
5. Tempo total do clique ao MP4, frio e quente, nos modos A e B, em 3 fotos.
6. Comparação visual da sala vazia na foto típica: LaMa por plano vs LaMa + SDXL só no piso vs SDXL-inpainting 0.1 (UNet de 9 canais, ainda não testado de verdade).
7. RAM do sistema usada no pico, com offload.
8. Candidato V3.1: FLUX.2-klein-4B (Apache-2.0) quantizado com offload, só se couber em 120 s.

---

## Anexos (evidência)

- Comparativo: `scratchpad/v3/comparativo.jpg`.
- Modo B: `p_removal/out/antes_vazio_depois.jpg`, `p_removal/out/cmp_all.jpg`, `p_removal/out/video_final/video.mp4`, `p_removal/out/video_final/video_log.json`.
- Crítica: `critic_realism/rem_grid.jpg`, `critic_realism/rem_residuo.png`, `critic_realism/r2_grid.jpg`.
- Modo A: `p_staging/final_R2_full.jpg`, `p_staging/video_staging_R2.mp4`.
- Licenças: `critic_feasibility/RELATORIO.md` §1, `r_staging/RELATORIO.md`, `r_removal/RELATORIO.md`.
- Hashes dos pesos: `p_staging/models_sha256.txt`, `r_removal/RELATORIO.md` (big-lama).
