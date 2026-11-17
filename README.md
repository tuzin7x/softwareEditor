# 7xAi

O **7xAi** é um programa para Windows (`.exe`) de **redesenho de ambientes por IA**. A partir de **uma única foto**
de um imóvel, ele gera novas versões do mesmo ambiente, com móveis, materiais e estilo novos, e monta um **vídeo
"antes e depois"** pronto para redes sociais.

Tudo roda **localmente, na placa de vídeo NVIDIA do usuário**, sem APIs pagas. A foto não sai do computador.

> **Estado (17/11/2026):** a reformulação para o redesenho por IA está **em avaliação de qualidade** e ainda não
> foi aprovada para lançamento. O desenvolvimento acontece na branch `claude/sweet-bohr-4uzml2`. A `main` está
> parada de propósito (ver "Política de branches").

## Funções

| Função | O que faz |
|---|---|
| Mudar estilo | mesma sala e mesma arquitetura, com móveis e decoração em outro estilo |
| Reforma completa | troca também pisos, revestimentos e acabamentos |
| Redesign simples | mudança leve, que mantém a sala reconhecível |
| Mobiliar ambiente vazio | mobília completa numa sala vazia |
| Editar área | o usuário pinta uma região e só ela muda; o resto fica idêntico à foto |
| Fachada | redesenho da parte externa |
| Decoração de evento | aniversário, casamento, Natal, festa infantil, chá de bebê, réveillon |
| Texto para ambiente | cria um ambiente a partir de uma descrição, sem foto |

São 14 estilos: moderno, contemporâneo, escandinavo, minimalista, industrial, clássico, luxo, boho, japandi,
rústico, costeiro, tropical brasileiro, mid-century e fazenda moderna. Também há 8 tipos de ambiente.

Cada pedido gera até 4 versões. Janelas e portas originais são recolocadas por cima da imagem gerada, e um controle
automático descarta as versões fracas.

Depois de gerar:

- **Ampliar 4K:** a versão escolhida sai em 3840 px.
- **Vídeo antes e depois:** a foto original se transforma nas versões novas, com movimento de câmera lento, em
  **16:9** (1920×1080) ou **9:16** (1080×1920, Reels e Stories).

**Modos legado e avançado:** pelo botão "Avançado (vídeo de montagem, legado)" continuam disponíveis os vídeos de
montagem antigos:

- **modo A:** móveis 3D entram numa foto vazia;
- **modo B:** a foto mobiliada vira sala vazia, os móveis da própria foto entram um a um e o vídeo termina
  idêntico à foto.

Eles estão documentados em [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md), seções 1 a 10.

**Desempenho medido:** cerca de 8 a 11 s por imagem numa RTX 5070 Laptop de 8 GB.

**Licenças:** só usamos modelos de IA com licença que permite uso comercial. A lista está em
[`docs/LICENCAS_V3.md`](docs/LICENCAS_V3.md).

## Instalar e gerar o `.exe` no Windows

1. Instale o [Python 3.13](https://www.python.org/downloads/) e marque *Add python.exe to PATH*.
2. Dê dois cliques em **`GERAR_EXE.bat`**, na pasta do projeto. Leva de 5 a 10 minutos.
3. No fim aparece *"Pronto! O 7xAi foi instalado e o ícone está na sua Área de Trabalho."*

O `GERAR_EXE.bat` chama `packaging/build_windows.ps1`, que:

- gera `dist\7xAi\7xAi.exe` com o PyInstaller e o arquivo `packaging/7xai.spec`;
- roda o autoteste;
- instala o programa em `%LOCALAPPDATA%\Programs\7xAi`;
- cria o atalho **7xAi** na Área de Trabalho e no Menu Iniciar.

Com o [Inno Setup 6](https://jrsoftware.org/isdl.php) instalado, o script também gera
`dist\7xAi-Setup-<versão>.exe`, o instalador para outros PCs.

No primeiro uso, o 7xAi instala o motor de IA (vários GB, uma única vez) em `%LOCALAPPDATA%\MobiliaStudio`. A
pasta mantém o nome antigo de propósito, para reaproveitar um motor que já esteja instalado.

**Pelo GitHub Actions**, o workflow `.github/workflows/build-windows.yml` gera o `.exe` e o instalador a cada push
em `claude/**`. Ele publica uma *prerelease* quando a mensagem do commit tem `[release]` ou quando é disparado
manualmente com "publicar" marcado.

O ícone (`packaging/assets/icon.ico`) e os SVGs da marca são gerados com `python scripts/make_icon.py`.

## Uso pela linha de comando (desenvolvimento)

```bash
python -m mobilia.cli hardware
python -m mobilia.cli design --photo sala.jpg --funcao mudar_estilo --estilo japandi --ambiente sala --n 4 --out-dir saida/
python -m mobilia.cli ampliar --image saida/<versão>.jpg --out saida/<versão>_4k.jpg
python -m mobilia.cli showcase --before saida/antes.jpg --after saida/<versão1>.jpg saida/<versão2>.jpg --out antes_depois.mp4 --format 9x16
python -m mobilia.cli run --photo sala.jpg --room sala --out montagem.mp4     # modo legado A/B
```

`python -m mobilia.cli <comando> --help` lista as opções de cada comando. A interface usa esses mesmos comandos, num
processo separado, com eventos JSON (`--json`).

Para preparar um ambiente Linux de desenvolvimento, use `bash scripts/preparar_ambiente_dev.sh`. Para conferir o
hardware no Windows, use `scripts/diagnostico_hardware.ps1`.

## Testes

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q -m "not slow and not models and not gpu"   # suíte rápida
QT_QPA_PLATFORM=offscreen python -m pytest -q tests/test_ui.py tests/test_modules.py     # interface
```

Os marcadores ficam em `pyproject.toml`:

| Marcador | O teste precisa de |
|---|---|
| `slow` | render ou modelos de IA em CPU |
| `models` | pesos dos modelos |
| `gpu` | placa NVIDIA |

O CI (`.github/workflows/tests.yml`) roda a suíte rápida a cada push e a cada Pull Request.

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `mobilia/design/` | motor de redesenho por IA: funções, estilos, trava de janelas e portas, escolha automática, ampliar 4K |
| `mobilia/video/` | vídeo antes e depois (`showcase.py`) e codificação H.264 |
| `mobilia/ui/` | interface PySide6: janela principal, modo legado, tema, registro de módulos (`modules.py`) |
| `mobilia/modules/` | módulos opcionais da janela (abas novas). Ainda não há nenhum |
| `mobilia/analysis/`, `layout/`, `render/`, `composite/`, `removal/`, `layers2d/`, `gen/` | modos legados A e B |
| `mobilia/cli.py`, `runtime.py`, `bootstrap.py` | linha de comando, caminhos e worker, instalação no primeiro uso |
| `packaging/`, `GERAR_EXE.bat` | geração do `.exe` e do instalador |
| `docs/` | arquitetura, planos, licenças, limitações, pesquisa e evidências das provas de conceito (`docs/poc/`) |

## Política de branches e desenvolvimento paralelo

- **`claude/sweet-bohr-4uzml2`** é a **branch de integração**, onde está todo o trabalho atual.
- **`main`** fica como está até o usuário aprovar uma versão. Hoje ela tem só o commit inicial.
- **Módulos novos** são feitos em branches `feature/<nome>`, criadas a partir da branch de integração, e voltam
  por Pull Request para ela. O primeiro planejado é o **7xAi Lead Finder**, em `feature/lead-finder`. Ele não deve
  mexer no motor de redesenho nem no de vídeo.
- Não use force-push nem reescreva o histórico.

O guia completo está em [`docs/PARALLEL_DEVELOPMENT.md`](docs/PARALLEL_DEVELOPMENT.md).

## Documentação

- [`RESUME.md`](RESUME.md): estado atual e como retomar o trabalho.
- [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md): arquitetura do redesenho, da interface e dos modos legados.
- [`docs/PARALLEL_DEVELOPMENT.md`](docs/PARALLEL_DEVELOPMENT.md): regras para desenvolver módulos em paralelo.
- [`docs/LICENCAS_V3.md`](docs/LICENCAS_V3.md): licenças dos modelos e dos assets.
- [`docs/LIMITACOES.md`](docs/LIMITACOES.md): limitações conhecidas, sobretudo dos modos legados.
- [`docs/PESQUISA_TECNOLOGIAS.md`](docs/PESQUISA_TECNOLOGIAS.md), [`docs/PLANO_V2.md`](docs/PLANO_V2.md),
  [`docs/PLANO_V3.md`](docs/PLANO_V3.md): pesquisa e planos (histórico).
- `docs/poc/`: vídeos e quadros das provas de conceito dos modos legados. Os resultados estão em
  [`docs/poc/v2/RESULTADO.md`](docs/poc/v2/RESULTADO.md) e [`docs/poc/v3/RESULTADO.md`](docs/poc/v3/RESULTADO.md),
  e a demonstração da v1 em [`docs/poc/sala_stafford_poc.mp4`](docs/poc/sala_stafford_poc.mp4), com quadros em
  [`docs/poc/quadros_sala/`](docs/poc/quadros_sala/).

## Licenças e dados

- Modelos de IA: só modelos com licença comercial ([`docs/LICENCAS_V3.md`](docs/LICENCAS_V3.md)), baixados no
  primeiro uso e nunca guardados no repositório.
- Assets 3D do modo legado: Poly Haven (CC0) e assets procedurais próprios.
- Blender/bpy (GPL) roda num processo separado. FFmpeg: GPL, build com x264.
- Fontes Inter: SIL OFL (`mobilia/ui/fonts/OFL.txt`).
- As fotos de `tests/fixtures/photos` são CC BY-SA (ver `ATRIBUICOES.md`) e servem só para testes.
- **Fotos e dados de clientes nunca entram no repositório.** O `.gitignore` bloqueia `usuario_*`, `.env`, chaves e
  bancos locais.
