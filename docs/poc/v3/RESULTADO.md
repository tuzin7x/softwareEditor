# Modo B sem SDXL: resultado da demonstração e avaliação final (PT-V3-8)

Avaliação adversarial do modo B como está no repositório: foto mobiliada → sala vazia por trás → os móveis
reais da foto entram um a um → o vídeo termina igual à foto. Tudo rodou **na CPU** (4 núcleos, sem GPU),
com todos os modelos reais (MoGe-2, OneFormer, SAM 2.1, big-lama, OneFormer na reconferência da placa).

## Arquivos

| Arquivo | O que é |
|---|---|
| `modo_b_tipica.mp4` | foto típica de anúncio (`tests/fixtures/rooms/sala_sofas_brancos`, CC0), 1440×960, 216 quadros, 9,0 s |
| `modo_b_nice.mp4` | foto difícil (`tests/fixtures/rooms/nice_terrazzo`, CC0), 1440×952, 240 quadros, 10,0 s |
| `modo_b_folha.jpg` | linhas 1–2: início (sala vazia), meio e fim de cada vídeo; linha 3: os defeitos em close |

As duas fotos de origem têm 960 px e 1280 px de largura. Para o vídeo de 1440 px pedido, **a foto foi ampliada
uma vez (Lanczos) antes de entrar no pipeline**; o último quadro é essa foto ampliada, byte a byte. Com fotos de
cliente de 2000 px ou mais isso não acontece (a foto só é reduzida). O perfil de CPU limita o lado maior a 1280
px; a demonstração forçou 1440 px.

**Os dois vídeos só existem porque foram gerados com "gerar mesmo assim" (`--aceitar-placa-fraca`).** Por
padrão, o portão de qualidade reprova a sala vazia das duas fotos e o 7xAi não entrega o MP4, só o aviso.

## Notas (1–10, mesmo critério do crítico anterior)

| Critério | Típica | Nice | Por quê |
|---|---|---|---|
| Sala vazia (1º quadro) | **2** | **2** | Típica: névoa branca, retângulo fantasma do quadro na parede, piso de porcelanato virou carpete bege (o tapete cobria tudo), placas brancas soltas perto da janela, triângulo branco no piso. Nice: placas de parede "soltas" na frente da porta e da janela, cama fantasma no quarto, manchas no terrazzo. |
| Realismo dos móveis | 7 | 6 | São os pixels reais da foto (pilha completa = foto, 0 nível de diferença). Em repouso, perfeitos. Em movimento, sem sombra própria além da de contato; na Nice, a mesa de jantar entra com o buraco das cadeiras (abaixo). |
| Animação | 5 | 5 | Sem fade e com movimentos variados (deslizar da borda, queda com quique, subir por trás, tapete desenrolando). Mas: o sofá principal da típica **surge de uma vez** (sem movimento seguro); o tapete desenrola **depois** do sofá e da mesa de centro já pousados em cima dele; as flores caem com quique dentro do vaso; a cama cai do alto. |
| Geral | **3** | **3** | O espectador vê primeiro a sala vazia, e ela é o pior quadro do vídeo. Não é vendável. O portão está certo em reprovar. |

## Verificações (refeitas de forma independente, a partir do MKV sem perdas de auditoria)

| Verificação | Típica | Nice |
|---|---|---|
| Último quadro `array_equal` à foto (na resolução do vídeo) | ✔ | ✔ |
| Quadro 0 `array_equal` à sala vazia | ✔ | ✔ (depois da correção abaixo) |
| Pixels fora da máscara de remoção iguais à foto em 100% dos quadros (verificação do pipeline) | ✔ 0 px | ✔ 0 px |
| Pixel fora da região dinâmica que muda e **fica** mudado (morph) | 0 (a maior mudança dura 20 quadros: peça passando) | 0 (19 quadros) |
| Opacidade parcial de peça | 0 | 0 |
| "Pixel oculto visível" segundo `choreo.verify` | 0 | 0 |
| "Pixel oculto visível" **a olho** | não vi | **sim: a mesa de jantar, ~2,4–3,4 s** |
| PSNR do MP4 (1º quadro / último) | 49,6 / 49,7 dB | 47,8 / 48,7 dB |

### Afirmações dos implementadores que não se sustentam

1. **"0 pixel oculto visível" (PT-V3-6) é falso na Nice.** A mesa de jantar cai às 1,4 s, antes das duas cadeiras
   da frente (2,5 s e 3,0 s). Ela aparece com o recorte das cadeiras: um buraco no formato do encosto e das pernas,
   mostrando a parede inventada atrás. A verificação diz 0 porque só confere recortes contra peças marcadas "na
   frente", e o grafo de oclusão não tem relação mesa × cadeira. Medido na fixture (que reproduz o defeito com
   placa Telea e sem SAM, então ele não depende da placa): o MoGe dá a mesma profundidade às duas na borda comum
   (4,21 × 4,20 m; 3,79 × 3,76 m, dentro da tolerância de 3%) e, pela base, a mesa "desce" mais que as cadeiras
   (815 × 803 e 808 px, a 1280×846). Tentei usar a regra da base quando a profundidade empata:
   não resolve este caso (a base da mesa é a mais baixa) e piora a foto típica (o sofá passa a "surgir"). Revertido;
   fica como defeito aberto.
2. **Contorno dos móveis voltando para a sala vazia (corrigido nesta rodada).** Quando o portão escolhe a tentativa
   `mascara_ampla` (núcleo dilatado 2,5×), as camadas usavam o anel padrão (8 px) em vez do anel da máscara larga
   (19 px a 1440 px). A faixa entre os dois voltava a ser **a foto mobiliada** na sala vazia: 4,9% da imagem na
   Nice, com o contorno escuro dos pés das cadeiras e da base das poltronas desenhado no piso. O portão não via
   (ele avalia a placa antes dessa correção) e o aviso das camadas não chegava ao usuário. Correção:
   `layers.build_layers` usa a dilatação que a placa usou de fato; `pipeline_b` registra os avisos das camadas no log.
   Teste novo: `tests/test_layers2d.py::test_wide_mask_band_is_not_written_back_to_the_empty_plate`. Depois da
   correção: 0% corrigido na placa, quadro 0 = placa byte a byte.
3. **Nota do portão pouco informativa.** As 3 tentativas das 2 fotos saem com 59,0 (teto de reprovação). A nota
   sem teto da tentativa escolhida na típica é **90,2**, para uma placa que eu daria 2/10: só a cor do piso
   (ΔE 8,5) a reprova. A névoa, o fantasma do quadro e as placas soltas não entram na nota.

## Tempos medidos (CPU, 4 núcleos, sem GPU, modelos em cache de disco)

| Etapa | Típica (1440×960) | Nice (1440×952) |
|---|---|---|
| Análise (MoGe-2 + OneFormer, com carga dos modelos) | 24,4 s | 24,3 s |
| Peças (SAM 2.1 + regras) | 5,7 s | 4,9 s |
| Sala vazia (3 tentativas do portão, cada uma com big-lama + OneFormer da nota) | 111,0 s (39,8 + 29,7 + 39,3) | 121,9 s (44,2 + 30,9 + 44,7) |
| Camadas + coreografia | 3,0 s | 1,6 s |
| Composição + verificação de cada quadro + MP4 + MKV de auditoria | 10,5 s | 8,4 s |
| **Total do clique ao MP4** | **156 s** | **163 s** |

A sala vazia é 70–75% do tempo, e quase todo ele é gasto em tentativas que o portão reprova.

## Estimativa na RTX 5070 Laptop 8 GB (não medida)

Premissas: cache quente; um modelo na GPU por vez; fp16; LaMa e OneFormer 8–15× mais rápidos que nesta CPU de
4 núcleos (razão típica para redes convolucionais/transformers dessa escala); composição continua em numpy com
NVENC (o caminho torch existe, mas não foi medido na GPU).

| Etapa | Estimativa |
|---|---|
| Análise | 3–6 s |
| Peças (SAM 2.1) | 0,5–1,5 s |
| Sala vazia, 3 tentativas | 8–18 s |
| Camadas + coreografia (CPU) | 1,5–3 s |
| Composição + MP4 | 4–10 s |
| **Total** | **≈ 20–40 s** (≤ 120 s com folga) |

**Atenção:** a GPU deixa o modo B rápido, mas **não melhora a sala vazia**: o big-lama dá o mesmo resultado na
GPU. Quase todas as fotos continuarão reprovadas pelo portão.

## O que funciona

- Escolha automática do modo: foto vazia → modo A; foto mobiliada → modo B.
- Peças reais da foto, recortadas com alpha (sem 3D, sem catálogo), entrando com movimentos variados e sem fade.
- Regra absoluta: o vídeo termina idêntico à foto; fora da máscara nada muda; sem morph.
- O portão de qualidade reprova as salas vazias ruins e explica o motivo; por padrão nenhum vídeo ruim sai.
- Ordem de grandeza do tempo cabe no orçamento (CPU 2,5–3 min; GPU estimado 20–40 s).

## Defeitos conhecidos (abertos)

1. **Sala vazia ruim na maioria das fotos** (as 2 daqui e 33 de 39 fotos mobiliadas da rodada de robustez): névoa, fantasma de
   móvel, piso inventado com material errado sob tapetes grandes, pedaços de parede "soltos". É o gargalo; o
   próximo passo é o refino SDXL (PT-V3-4), que precisa da GPU para ser avaliado.
2. **Grafo de oclusão sem relação entre peças grandes encostadas com a mesma profundidade** (mesa × cadeira):
   a peça de trás entra antes com o buraco da da frente, e a verificação não acusa (item 1 acima).
3. **Tapete desenrola embaixo de móveis já pousados** quando os móveis cobrem o tapete (ordem de frente para trás).
4. **"Surgir no lugar"** (aparece de uma vez) para o sofá principal da foto típica.
5. **Nota do portão** sem teto chega a 90 em placa ruim; as reprovadas ficam todas em 59.
6. Quadros de parede que tocam o topo deslizam de cima (é proposital, mas parece "cair do teto").

## O que validar na RTX 5070 Laptop do usuário

1. Tempo real de cada etapa (o `report.json` já registra `tempos_s` e o pico de VRAM).
2. Que o MP4 e o MKV de auditoria saem iguais aos da CPU (diferença ≤ 1 nível).
3. NVENC: PSNR do 1º e do último quadro ≥ 30 dB (o pipeline recusa abaixo disso).
4. Sysmem fallback do driver (não deve haver: o modo B sem SDXL usa < 3 GB).
5. A comparação visual da sala vazia com o refino SDXL (PT-V3-4), que é o que decide se o modo B vira produto.
