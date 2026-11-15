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
