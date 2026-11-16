# Prova de conceito V2 — sala de teste (CPU)

Rodada completa do pipeline V2 (PLANO_V2, PT0–PT12) na foto de teste da sala vazia, do comando à
MP4 verificada, num contêiner **sem GPU** (CPU de 4 threads). Nada foi editado à mão entre a foto e o
vídeo.

```bash
MOBILIA_MANTER_EXR=1 python -m mobilia.cli run --photo tests/fixtures/photos/sala_vazia_stafford.jpg \
    --room sala --quality previa --seed 7 --work-dir <pasta> --out <pasta>/poc_v2.mp4
```

Arquivos desta pasta:

| Arquivo | Conteúdo |
|---|---|
| `sala_stafford_v2.mp4` | o vídeo (960×720, 24 fps, 251 quadros, 10,5 s, H.264 CRF 14, 1,8 MB) |
| `v2_folha_contato.jpg` | foto + 5 quadros-chave |
| `v2_quadro_042/072/116/148/251.jpg` | queda do sofá, mesa lateral deslizando, luminária surgindo do piso, decoração sobre a mesa, quadro final |
| `sala_stafford_v2_report.json` | `report.json` completo do trabalho (telemetria PT1) |
| `pt*.jpg` | comparações A/B feitas durante cada pacote de trabalho |

## O que mudou em relação à PoC V1

- **Tempo:** render em camadas (marcos estáticos + camada dinâmica só da região que se move), EXR
  recortado só com Combined + Shadow Catcher, composição em paralelo com o render e controle de
  orçamento que decide amostras e escala do CG depois de medir o quadro final.
- **Luz e cor:** a sala 3D emite a própria foto para os raios indiretos, o sol é ajustado nas manchas
  de sol da foto e o CG recebe mapa de ganho, curva de tom e joelho de altas luzes antes de entrar na
  foto.
- **Ativos:** pinturas abstratas geradas localmente (determinísticas) em vez dos quadros de catálogo
  com cara de CG, estofado claro em sala clara, tapete com textura de tecido CC0, pilha de livros
  procedural apoiada na mesa (verificação de apoio no Blender).
- **Contato:** sombra de contato analítica sob cada peça de piso, que acompanha a animação.
- **Entradas:** queda (sofá, mesas), deslize lateral (mesa lateral; a poltrona caiu porque não havia
  caminho livre), surgir do piso com corte exato no plano do piso (luminária), crescer (planta),
  desenrolar (tapete), deslizar na parede (quadros).
- **Designer:** a narrativa das escolhas (`report.json["narrativa"]`) aparece na tela de conclusão.

## Resultado medido (CPU, 4 threads, sem GPU)

**Regra absoluta:** 251/251 quadros bit a bit idênticos à foto fora dos móveis e das sombras de
contato (`preservado_exato_todos = true`, verificação independente por quadro antes do encode). Os 20
quadros iniciais são a própria foto. PSNR do MP4 decodificado: 43,2 dB no fundo, 43,7–44,0 dB no
quadro inteiro (primeiro, meio e último, por *seek*).

**Quadros:** 251 no total. 169 renders (1 final, 4 marcos, 164 da camada dinâmica), 76 quadros
reaproveitados e 20 só de foto. Modo em camadas ativo.

**Orçamento (PT7):** a sonda do quadro final mediu 494 mil pixel·amostras/s. Mesmo no mínimo, a
previsão passava de 120 s (`cabe = false`, como esperado em CPU). Os cortes ficaram em 12 amostras
na camada dinâmica e escala de CG de 80%. Os marcos ficaram nas 24 amostras da prévia, que já estavam
abaixo do mínimo de 40. Render previsto 1 216 s, real 1 110 s. Sem os cortes a previsão era 1 821 s.

| Etapa | Tempo |
|---|---|
| Início do motor + carga da foto | 1,6 s |
| Análise (MoGe-2 + OneFormer + sala + sol), modelos em cache de disco | 20,7 s |
| Layout + narrativa | 2,8 s |
| Blender: início, kernels, cena, sombra de contato | 1,9 s |
| Calibração de luz | 5,1 s |
| Quadro final (24 amostras, 0,60 MP) | 31,3 s |
| 4 marcos (24 amostras, 0,43–0,60 MP) | 101,6 s |
| 164 renders dinâmicos (12 amostras, 80%, média 0,20 MP de região) | 1 008 s (mediana 6,2 s/render) |
| Composição + encode + verificação após o fim do render | 1,1 s |
| **Total** | **1 175 s (19,6 min)** |

Em CPU, cerca de 0,5 s de cada render dinâmico é o denoiser, um custo fixo por render. A composição
correu em paralelo ao render: 171 composições (quadros, marcos e transições) terminaram antes do fim
do render, e a cauda depois dele foi de 0,9 s. Na PoC V1, a mesma foto levou 6 768 s só de render, com 24 amostras em todos os quadros e
quadro inteiro.

## Estimativa para a RTX 5070 Laptop (8 GB), foto 1920×1440

**Premissa:** o Cycles com OptiX nessa GPU rende cerca de **50×** a vazão medida aqui (≈ 25 M
pixel·amostras/s numa cena de sala), e o custo fixo por render (sincronização + denoiser na GPU) fica
perto de 0,08 s. A margem é de ±2× até chegar o primeiro `report.json` da máquina do usuário.

- Quadro final de 128 amostras sobre ~1,4 MP: ~7 s. Quatro marcos de 64 amostras: ~13 s.
- Camada dinâmica, 164 renders de ~0,8 MP de região: com 12 amostras e escala de 80%, ~45 s mais
  ~13 s de custo fixo.
- Somando análise (~12 s), layout (3 s), Blender e calibração (~9 s) e cauda (~5 s): **≈ 105 s**,
  dentro dos 120 s.
- O controle de orçamento mede a GPU no quadro final e decide os cortes **antes** do primeiro quadro
  animado. Se a GPU for mais lenta que a premissa, ele corta amostras e escala na ordem do
  PLANO_V2 §2. Se ainda assim não couber, grava `cabe = false` e a previsão real na tela.

## Defeitos encontrados nesta rodada e corrigidos

1. **Mancha no primeiro quadro de fade-in** (quadro 40, entrada do sofá). Com o mix de fade, toda
   face da peça vira BSDF transparente. Um raio que atravessa o sofá cruza mais de 8 superfícies, e o
   Cycles encerrava o caminho como opaco. Com fade 0, um trecho do braço aparecia opaco no alfa e
   escurecia a parede no passe de sombra. Correção: `transparent_max_bounces` sobe para 128 só nos
   quadros com o mix ligado e volta a 8 em repouso. O teste `tests/test_fade_bounces.py` faz um
   render real do Blender. Rodada refeita inteira: a mancha sumiu.
2. **Concordância na narrativa:** "pés dianteiros do poltrona" passou a "da poltrona", com teste.
3. **Teste da UI saía com código 139:** o processo do pytest caía no encerramento, depois de todos os
   testes passarem, porque "Copiar diagnóstico" deixava um `QMimeData` na área de transferência
   (PySide6, plataforma offscreen). A fixture agora limpa a área de transferência.

## Inspeção visual

- Pilha de livros, vaso e fruteira apoiados no tampo da mesa de centro. Nenhum item foi removido pela
  verificação de apoio.
- Par de pinturas abstratas com moldura fina, sem esticar; sofá claro com almofadas; tapete com filete
  e textura.
- A luminária surge do piso em três etapas (base, haste, cúpula), sem nada visível abaixo do piso. A
  mesa lateral entra deslizando rente à parede, sem atravessá-la.
- Nas regiões já assentadas, a variação entre quadros consecutivos fica em mediana abaixo de 0,002/255. As
  transições de marco (3 quadros) mudam no máximo ~1,5/255 em média, sem *pop* visível. Não há
  linhas nem degraus nas bordas dos recortes.

## Limitações que continuam

- Tudo foi medido em **CPU**. Tempos de GPU, OptiX, OIDN na GPU, NVENC e kernels pré-compilados só
  se confirmam com o `report.json` da máquina do usuário (botão "Copiar diagnóstico").
- Prévia em CPU: 960 px e 12–24 amostras, com ruído visível na cúpula da luminária e no couro da
  poltrona. O perfil da GPU usa 1920 px e 128/64/12–32 amostras.
- O tampo de vidro da mesa de centro sai como um cinza opaco: vidro transparente sobre a foto ainda
  está desligado (`film_transparent_glass`).
- Nos 2–3 quadros de fade-in de uma queda, a peça semitransparente escurece levemente o que está
  atrás dela (sombra parcial do passe de shadow catcher). É breve e faz parte da dissolução.
- O sofá do catálogo é clássico num layout "moderno": o catálogo CC0 tem poucos sofás modernos claros
  (PT16/PT17).
- Só a foto de teste foi usada. O comportamento em outras salas (piso brilhante, luz noturna, fotos
  em retrato) não foi validado.
