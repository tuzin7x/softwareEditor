# 7xAi

**7xAi** é um software Windows (`.exe`) que transforma **uma foto de um imóvel** em um **vídeo cinematográfico
de decoração**, com a câmera travada. Tudo roda **localmente na sua GPU NVIDIA**, sem APIs pagas. A interface
pede só a foto e o tipo de ambiente; o modo é **detectado pela foto**:

- **foto vazia (modo A):** móveis 3D surgem e se montam progressivamente **dentro da foto original**;
- **foto mobiliada (modo B):** o vídeo começa com a **sala vazia** (montada por trás, não aparece a remoção) e os
  **mesmos móveis da foto**, recortados dos pixels reais, entram um a um com animação até o vídeo terminar
  **idêntico à foto**. Um **portão de qualidade** dá nota à sala vazia; se ela reprovar em todas as tentativas,
  nenhum MP4 é entregue e a tela mostra o motivo, com as opções *gerar mesmo assim*, *modo A* ou *outra foto*.

> **Regra absoluta:** fora da máscara de remoção ou de inserção, todo quadro mestre é byte a byte a foto.
> **Modo A (foto vazia):** o vídeo começa na foto exata; só os móveis/decoração (e as sombras de contato
> que eles projetam) são adicionados, em camadas 3D independentes. **Modo B (foto mobiliada, V3):** o
> software esvazia a sala por trás (a sala vazia contém pixels gerados por inpainting), os móveis reais
> da própria foto voltam um a um, e o **último quadro é idêntico à foto**. Isso é **verificado quadro a
> quadro**, e o relatório informa a área gerada. Detalhes em `docs/ARQUITETURA.md` §1.

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

## Gerar e instalar no Windows (ícone 7xAi na Área de Trabalho)

1. Instale o [Python 3.13](https://www.python.org/downloads/) (marque *Add python.exe to PATH*).
2. Dê dois cliques em **`GERAR_EXE.bat`**, na pasta do projeto. Leva uns 5 a 10 minutos.
3. No fim aparece *"Pronto! O 7xAi foi instalado e o ícone está na sua Área de Trabalho."*

O script `packaging/build_windows.ps1` gera `dist\7xAi\7xAi.exe` (PyInstaller, `packaging/7xai.spec`),
roda o autoteste, instala o programa em `%LOCALAPPDATA%\Programs\7xAi` e cria o atalho **7xAi** na
Área de Trabalho e no Menu Iniciar. Atalhos antigos do `MobiliaStudio.exe` são removidos. Se o 7xAi
estiver aberto, feche-o antes de rodar de novo.

No primeiro uso, o 7xAi instala o motor de IA e renderização (~7 GB, uma única vez) em
`%LOCALAPPDATA%\MobiliaStudio`. A pasta mantém o nome antigo de propósito: quem já tinha o motor
instalado não baixa nada de novo.

Para instalar em **outros PCs**, instale o [Inno Setup 6](https://jrsoftware.org/isdl.php) e rode o
`GERAR_EXE.bat` de novo. Ele também gera `dist\7xAi-Setup-<versão>.exe`, que instala o programa e cria o
ícone na Área de Trabalho.

O ícone (`packaging/assets/icon.ico`, de 16 a 256 px) e os SVGs da marca são gerados dos vetores com
`python scripts/make_icon.py`.

## Uso (linha de comando / desenvolvimento)

```bash
python -m mobilia.cli hardware
```

```bash
python -m mobilia.cli run --photo minha_sala.jpg --room sala --out sala_decorada.mp4
```

Tipos de ambiente: `sala`, `quarto`, `cozinha`, `banheiro`, `escritorio`, `sala_jantar`.
Qualidade: `--quality previa|final|auto` (auto = final na GPU, prévia na CPU).
Modo: `--mode auto|A|B` (padrão `auto`: foto vazia → A, mobiliada → B). No modo B, `--aceitar-placa-fraca`
gera o vídeo mesmo com a sala vazia reprovada pelo portão (o aviso fica no `report.json`). Com
`MOBILIA_MODO_B_MKV=1` o modo B grava também `auditoria.mkv` (sem perdas) e confere cada quadro byte a byte.

Para conferir o hardware no Windows sem instalar nada, use `scripts/diagnostico_hardware.ps1`.

## Prova de conceito V2 (resultado real, CPU)

Mesma foto, pipeline do `docs/PLANO_V2.md`: render em camadas, controle de orçamento, luz da foto,
sombra de contato, entradas novas e narrativa do designer. Resultado em
[`docs/poc/v2/RESULTADO.md`](docs/poc/v2/RESULTADO.md), vídeo
[`docs/poc/v2/sala_stafford_v2.mp4`](docs/poc/v2/sala_stafford_v2.mp4) e folha de contato
[`docs/poc/v2/v2_folha_contato.jpg`](docs/poc/v2/v2_folha_contato.jpg).

- **Preservação:** 251/251 quadros bit a bit idênticos à foto fora dos móveis e das sombras; PSNR do
  MP4 43,2 dB no fundo.
- **Tempo (CPU, 4 threads, 960×720):** 1 175 s do início ao MP4 verificado, contra 6 768 s só de
  render na V1. 169 renders, 76 quadros reaproveitados, 20 só de foto. Estimativa na RTX 5070 Laptop
  em 1920×1440: ≈ 105 s (premissa no `RESULTADO.md`).

## Prova de conceito V1 (resultado real)

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
