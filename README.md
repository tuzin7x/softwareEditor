# Mobília Studio

Software Windows (`.exe`) que transforma **uma foto de um imóvel vazio** em um **vídeo cinematográfico
de decoração virtual**: os móveis surgem e se montam progressivamente **dentro da foto original**, com
a câmera travada. Tudo roda **localmente na sua GPU NVIDIA**, sem APIs pagas.

> **Regra absoluta:** a foto é uma placa de fundo imutável. Só os móveis/decoração (e as sombras de
> contato que eles projetam) são adicionados, em camadas 3D independentes. Paredes, piso, janelas,
> iluminação, cores, perspectiva e enquadramento não mudam. Isso é **verificado quadro a quadro**: fora
> dos móveis e das sombras, os pixels saem idênticos aos da foto.

## Como funciona (resumo)

1. **Análise com IA local:** profundidade métrica e câmera (MoGe-2) e segmentação de piso, paredes,
   janelas e portas (OneFormer). Disso sai a reconstrução 3D da sala.
2. **Layout automático:** a IA escolhe os móveis adequados ao tipo de ambiente e ao estilo/paleta da
   foto, e os posiciona seguindo regras de design de interiores e de enquadramento.
3. **Render 3D (Blender Cycles, OptiX):** móveis 3D reais com a câmera casada à foto, sombras e
   oclusão coerentes e luz calibrada pela própria foto.
4. **Animação de montagem:** tapete, móveis grandes, plantas, quadros e decoração entram peça a peça,
   com física de queda e acomodação e motion blur.
5. **Composição exata + MP4:** as camadas entram sobre a foto original e o vídeo sai em H.264 de alta
   qualidade.

Detalhes: [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md) · pesquisa de tecnologias:
[`docs/PESQUISA_TECNOLOGIAS.md`](docs/PESQUISA_TECNOLOGIAS.md) · limitações conhecidas:
[`docs/LIMITACOES.md`](docs/LIMITACOES.md).

## Uso (linha de comando / desenvolvimento)

```bash
python -m mobilia.cli hardware
```

```bash
python -m mobilia.cli run --photo minha_sala.jpg --room sala --out sala_decorada.mp4
```

Tipos de ambiente: `sala`, `quarto`, `cozinha`, `banheiro`, `escritorio`, `sala_jantar`.
Qualidade: `--quality previa|final|auto` (auto = final na GPU, prévia na CPU).

Para conferir o hardware no Windows sem instalar nada, use `scripts/diagnostico_hardware.ps1`.

## Prova de conceito (resultado real)

Foto de teste de uma sala vazia (960×720), ambiente `sala`, tudo em **CPU** (sem GPU no ambiente de
desenvolvimento): [`docs/poc/sala_stafford_poc.mp4`](docs/poc/sala_stafford_poc.mp4) (10,5 s, 251
quadros), quadros-chave em [`docs/poc/quadros_sala/`](docs/poc/quadros_sala/), folha de contato em
[`docs/poc/sala_stafford_folha_contato.jpg`](docs/poc/sala_stafford_folha_contato.jpg) e relatório
completo em [`docs/poc/sala_stafford_report.json`](docs/poc/sala_stafford_report.json).

- 13 peças escolhidas e posicionadas automaticamente (sofá, poltrona, mesa de centro, mesa lateral,
  tapete, luminária de piso, planta, 2 quadros, almofadas, vaso, livros, tigela de madeira).
- **Preservação:** 251/251 quadros bit a bit idênticos à foto fora dos móveis e das sombras; os 20
  quadros iniciais, antes da primeira peça, são idênticos à foto inteira.
- **Vídeo:** PSNR do MP4 decodificado entre 43,2 e 44,3 dB (H.264 CRF 14, 2 MB).
- **Tempo (CPU, 4 threads):** análise 19 s (modelos já baixados), layout 4 s, render 6 768 s
  (176 de 251 quadros renderizados; o resto é reaproveitado), composição + encode 47 s.

Para refazer só o vídeo (por exemplo, depois de ajustar a composição) sem renderizar de novo:

```bash
python -m mobilia.cli recompor --work-dir <pasta_de_trabalho>
```

## Testes

```bash
python -m pytest -q
```

Os testes cobrem a preservação bit a bit na composição, a coreografia, a reconstrução de sala
sintética, o casamento de câmera no Blender (erro < 0,35 px) e o encode sem desvio de cor.

## Licenças de terceiros

- Modelos de IA: MoGe-2 (MIT) e OneFormer ADE20K (MIT), baixados no primeiro uso.
- Assets 3D: Poly Haven (CC0), mais assets procedurais próprios.
- Blender/bpy (GPL) roda como processo separado; FFmpeg (GPL, build com x264).
- As fotos de teste em `tests/fixtures/photos` são CC BY-SA (ver `ATRIBUICOES.md`) e servem só para
  testes.
