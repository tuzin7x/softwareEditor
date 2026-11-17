# RESUME — estado atual (17/11/2026)

## Onde está o trabalho

- **Repositório:** `tuzin7x/softwareEditor` (público). O pacote Python se chama `mobilia` e o produto se chama
  **7xAi**.
- **Branch de integração (desenvolvimento):** `claude/sweet-bohr-4uzml2`. Todo o trabalho está nela; a sessão principal
  envia os commits ao GitHub sem force-push.
- **`main`:** tem só o commit inicial e **não foi atualizada de propósito**. A reformulação ainda está em
  avaliação; a `main` só recebe a integração quando o usuário aprovar uma versão. Detalhes em
  `docs/PARALLEL_DEVELOPMENT.md`, seção 13.
- **`ui-7xai`:** branch local antiga, já contida na branch de integração (worktree local da sessão principal). Não
  mexer.
- **`feature/lead-finder`** (planejada): será feita por outra conta, a partir do commit `435a87b` indicado em
  `docs/PARALLEL_DEVELOPMENT.md`, e volta por Pull Request para `claude/sweet-bohr-4uzml2`.

### Últimos commits relevantes

| Commit | O que fez |
|---|---|
| `d2f929b` | Motor de redesenho de ambientes (mesma sala, móveis e estilo novos) |
| `5b298a6` | Ajuste no "Ampliar 4K" |
| `bfbe546` | Vídeo antes e depois e comando `upscale` |
| `42955e6` | Interface nova, com o redesenho por IA como fluxo principal |
| commits seguintes | Ponto de extensão de módulos (`mobilia/ui/modules.py`, `mobilia/modules/`), `.gitignore` reforçado e documentação do desenvolvimento paralelo |

## Estado do produto

- **Fluxo principal: redesenho por IA da imagem inteira**, com 8 funções:
  - mudar estilo, reforma completa, redesign simples e mobiliar vazio;
  - editar área, fachada, decoração de evento e texto para ambiente.

  Há também **ampliar 4K** e **vídeo antes e depois** em 16:9 e 9:16.
- **Execução:** é 100% local. Na RTX 5070 Laptop de 8 GB leva cerca de 8 a 11 s por imagem, com os codificadores
  de texto na CPU para caber em 8 GB.
- **Qualidade do redesenho:** **em avaliação**, ainda não aprovada pelo usuário.
- **Legado/avançado:** os modos A (montagem 3D) e B (foto mobiliada → sala vazia → móveis entram → foto exata)
  continuam disponíveis pelo botão "Avançado".
- **Interface:** `mobilia/ui/app.py` (janela de redesenho) e `mobilia/ui/legacy.py` (modo legado).
- **Ponto de extensão:** o registro de módulos permite abas novas sem editar `app.py`. Sem módulos, a janela não
  muda.
- **CI:**
  - `tests.yml` roda uma lista fixa de arquivos de teste rápidos em cada push em `claude/**`, `feature/**` e
    `main`, e em PRs para `claude/**` e `main`;
  - `build-windows.yml` gera o `.exe` e o instalador em cada push em `claude/**` e publica uma prerelease com
    `[release]` ou com o disparo manual "publicar".
- **Build local:** `GERAR_EXE.bat`, que chama `packaging/build_windows.ps1`.

## Em andamento (sessão principal)

- Ajustes de qualidade do redesenho por IA: `mobilia/design/*`, `mobilia/video/showcase.py` e a interface
  (`app.py`, `design_widgets.py`, `legacy.py`, `runner.py`), com os testes correspondentes.

## Próximos passos

1. Terminar a avaliação de qualidade do redesenho com o usuário e corrigir o que ele apontar.
2. **"Esvaziar ambiente"** pelo motor de difusão, para servir de sala vazia ao modo B.
3. Revisar e integrar o Pull Request do Lead Finder (`feature/lead-finder` → `claude/sweet-bohr-4uzml2`), quando
   chegar. Depois, atualizar README, RESUME e ARQUITETURA.
4. Com uma versão aprovada pelo usuário: release (`[release]`) e só então atualizar a `main`.

## Como retomar

Numa nova sessão neste repositório, envie:

> Leia RESUME.md e docs/PARALLEL_DEVELOPMENT.md e continue pelos "Próximos passos".

Testes rápidos: `QT_QPA_PLATFORM=offscreen python -m pytest -q -m "not slow and not models and not gpu"` (no
Windows/PowerShell: `$env:QT_QPA_PLATFORM="offscreen"; python -m pytest -q -m "not slow and not models and not gpu"`). Para um
contêiner novo de desenvolvimento, use `bash scripts/preparar_ambiente_dev.sh`.

## Regras permanentes

- Fotos do usuário (`usuario_*`) e dados pessoais **nunca** vão para o repositório.
- Use só modelos com licença comercial (`docs/LICENCAS_V3.md`).
- Não use force-push nem reescreva o histórico.
- Faça commit só dos arquivos alterados, sem `git add -A`.

---

# Histórico (registros anteriores, mantidos como estavam)

# RESUME — atualizado em 2026-11-16 ~06:00 UTC (passos 1–6 concluídos)

## Atualização da retomada (16/11)
- ✅ Passo 1: `surfaces.py` usa `support_surface` dos procedurais. Os 252 testes de layout e procedurais passaram.
- ✅ Passo 2: revisão feita rodando as suítes (em vez de novos agentes, para economizar uso).
- ✅ Passo 3: interface PySide6 (`mobilia/ui/`), instalador de primeiro uso (`mobilia/bootstrap.py`), `packaging/` (spec PyInstaller, Inno Setup, script de build), CI (`.github/workflows/`), `requirements/`, `pyproject.toml`.
  - Testes de UI e de bootstrap passando (8).
  - Build PyInstaller validado em Linux, com autoteste OK.
  - **No GitHub Actions os jobs falham em 3 s sem runner** (runner_id 0, sem logs): problema de conta/cobrança/permissão do Actions, não do código. Verificar em Settings › Actions e em Billing.
- ✅ Passo 4: `docs/PESQUISA_TECNOLOGIAS.md` escrito.
- Passo 5, primeira rodada (`/tmp/claude-0/poc_sala2`): terminou com a foto 100% preservada e PSNR de 44 dB, mas a inspeção visual achou 3 defeitos. Todos foram corrigidos e testados (commit b37481a):
  1. linha escura na parede: linha fora do border do Cycles lida como sombra 0;
  2. degrau de tom no limite do border: agora há rampa suave, e o border ganhou margem vertical;
  3. quadro cortado pela quina: o proxy da parede oblíqua estendia 0,35 m na frente da outra parede; a extensão agora é limitada.
  - Novo comando `python -m mobilia.cli recompor --work-dir ...` refaz o MP4 a partir dos EXRs, sem renderizar de novo.
- ✅ Passo 5, segunda rodada (`/tmp/claude-0/poc_sala3`): 251/251 quadros bit a bit fora dos objetos, PSNR de 43,2–44,3 dB, render de 6 768 s em CPU (176 de 251 quadros). Inspeção visual sem nenhum dos 3 defeitos. MP4, quadros-chave, folha de contato e `report.json` estão em `docs/poc/`.
- ✅ Passo 6: `docs/LIMITACOES.md` (seções 6 e 7) e o README (seção "Prova de conceito") foram atualizados.

### O que falta (fora do escopo já pedido)
- Rodar o `.exe` num Windows com GPU NVIDIA. Depende de liberar o GitHub Actions na conta ou de gerar localmente com `packaging\build_windows.ps1`.
- Melhorias listadas em `docs/LIMITACOES.md` e `docs/PESQUISA_TECNOLOGIAS.md` (distorção de lente, *holdout* por malha de profundidade, IDR único com teto de VBV etc.).

---

# (registro original do checkpoint de 2026-11-15, 23:35 UTC)

Projeto: **Mobília Studio** (decoração virtual em vídeo a partir de uma foto, 100% local).
Repositório: `tuzin7x/softwareEditor`, branch **`claude/sweet-bohr-4uzml2`** (tudo commitado e enviado).

## Como retomar (comando único)

Numa nova sessão do Claude Code neste repositório, envie:

> Leia RESUME.md e continue exatamente do checkpoint, na ordem da seção "Próximos passos".

Para preparar o ambiente Linux de desenvolvimento, se o contêiner for novo:

```bash
bash scripts/preparar_ambiente_dev.sh
```

## Estado por tarefa

| # | Tarefa | Estado |
|---|---|---|
| 1 | Pesquisa de tecnologias (7 relatórios) | **7/7 relatórios concluídos e salvos** em `docs/pesquisa_bruta/`. Faltam o crítico e a síntese em `docs/PESQUISA_TECNOLOGIAS.md` (interrompidos; NÃO refazer as pesquisas, só sintetizar os 7 arquivos). |
| 2 | Ambiente dev (venv py3.13, torch 2.15 CPU, bpy 5.3, MoGe, OneFormer) | Concluído (contêiner efêmero; ver script de preparo). |
| 3 | Fotos de teste | Concluído: `tests/fixtures/photos` (CC BY-SA, só para testes). |
| 4 | Validação empírica do Cycles | Concluído (confirmado pela pesquisa no código-fonte). |
| 5 | Análise da cena (MoGe-2 + OneFormer + RANSAC + sol + oclusores) | Concluído e testado. |
| 6 | Assets 3D + layout automático | **Implementação concluída** (layout: 70 testes; procedurais: 182 testes; relatórios em `docs/relatorios_agentes/`). A **revisão independente do layout** foi interrompida no início, e as revisões dos procedurais ainda não tinham começado. |
| 7 | Render + animação + composição + MP4 | Concluído e testado. O modo incremental é experimental e está desligado. |
| 8 | Interface Windows + .exe + CI | **Interrompido logo no início.** Só existe `mobilia/runtime.py` (parcial, não revisado). Nada em `mobilia/ui/`, `packaging/`, `.github/`, `requirements/`, `pyproject.toml`. |
| 9 | PoC final (pipeline completo, sala, 960×720) | **Interrompido no quadro 43/231 do render.** O estado está salvo em `docs/poc/estado_poc/` (layout.json, scene.json, room.json, overlays, log). Os EXRs parciais ficaram em `/tmp` e não são necessários: basta reexecutar. |

## O que está pronto e validado (não refazer)

- `mobilia/analysis/*`: geometria MoGe-2, segmentação OneFormer, `room.py` (piso, paredes com extensão robusta, janelas e portas, grade), `sunlight.py` (busca global F0,5), `depth_mesh.py` (oclusores), `debug_viz.py`.
- `mobilia/layout/*`: solver com recozimento simulado, regras, estilo/paleta, superfícies medidas, planners para os 6 ambientes, debug.
- `mobilia/assets/*`: Poly Haven (download, cache, MD5, `prefetch_catalog`) e procedurais (tapete com desenrolar por shape keys, luminárias, cama, cortina, escrivaninha, manta).
- `mobilia/render/*`:
  - `choreography.py`: drop, settle, grow, rise, slide, unfold, spread;
  - `scene_spec.py`: adaptador de parâmetros procedurais, iluminante, sol, oclusores;
  - `blender_worker.py`: câmera casada, sala-proxy catcher fechada, calibração NNLS janelas+ambiente+sol, import glTF, fade, vidro, border e reuso.
- `mobilia/composite/compositor.py`: composição bit-exata, grão e nitidez, `LayerAccumulator`.
- `mobilia/video/encoder.py`: x264/NVENC, BT.709 explícito, decode de verificação.
- `mobilia/pipeline.py`, `cli.py` (`run`, `batch`, `analyze`, `hardware`, `prefetch`), `jobs.py` (protocolo JSON para a UI), `hardware.py` (perfis por VRAM).
- Testes:
  - 27 testes rápidos passando;
  - teste ponta a ponta real (`tests/test_pipeline_e2e.py`, slow/models) passou;
  - `tests/test_layout.py` (70) e `tests/test_procedural_assets.py` (182) passaram no relatório dos agentes.
- Docs: `docs/ARQUITETURA.md`, `docs/LIMITACOES.md`, README, `scripts/diagnostico_hardware.ps1`.
- Evidências visuais em `docs/poc/`:
  - teste de animação MP4;
  - sequência da queda;
  - comparação com sol;
  - oclusores da cozinha;
  - layout automático;
  - assets procedurais.

## Próximos passos (em ordem)

1. **Ajustes pendentes pedidos pelos agentes** (não feitos):
   - `mobilia/layout/surfaces.py`: usar `mobilia.assets.support_surface(kind, size, params)` para cama e escrivaninha procedurais. Itens repousam no edredom a 0,602 m, não no colchão a 0,56 m.
   - O papel `curtain` no layout: deixar as tags por peça agirem (já tratado em `choreography.plan_parts`).
2. **Revisão independente** do layout e dos procedurais. Usar um agente revisor como no workflow `mobilia-parallel-build`. O script salvo está em `~/.claude/.../workflows/scripts/`; se não existir, recriar a partir das tarefas descritas aqui.
3. **Interface Windows + empacotamento + CI** (tarefa 8, do zero), conforme `docs/pesquisa_bruta/empacotamento_windows.md`.
   - **UI:** PySide6, tema escuro, em português.
     - Fluxo: importar foto → 6 cartões de ambiente → gerar → player + exportar.
     - Worker em QProcess lendo eventos JSON de `mobilia/jobs.py`.
     - Fila de trabalhos.
     - Painel de GPU (`cli hardware --json`).
   - **Runtime:** `mobilia/runtime.py` (parcial, revisar).
   - **Bootstrap de primeiro uso:** Python embeddable 3.13, torch CUDA (cu128+), bpy, `cli prefetch`.
   - **Empacotamento:** PyInstaller onedir só para a UI; Inno Setup por usuário.
   - **CI:** `.github/workflows/build-windows.yml` (push em `claude/**`) e `tests.yml`, depois verificar a execução no GitHub Actions.
   - **Testes:** `tests/test_ui.py` offscreen com `tests/fake_worker.py` (só teste).
4. **Síntese da pesquisa:** escrever `docs/PESQUISA_TECNOLOGIAS.md` (pt-BR) a partir dos 7 arquivos de `docs/pesquisa_bruta/`. Não refazer as pesquisas.
5. **PoC final:**

   ```bash
   MOBILIA_HOME=$HOME/mobilia_data python -m mobilia.cli run --photo tests/fixtures/photos/sala_vazia_stafford.jpg --room sala --out docs/poc/sala_stafford_poc.mp4 --quality previa --work-dir /tmp/poc_sala --json
   ```

   Leva cerca de 1–2 h em CPU. Depois, conferir `report.json`, preservação e PSNR, e commitar MP4 e quadros.
6. Atualizar `docs/LIMITACOES.md` e o README com o resultado final. Commit e push.

## Decisões técnicas a preservar

- A foto nunca é alterada: `saída = foto·S'·(1−A)+C`, com cópia de bytes fora de A>0 ou S'≠1, verificada em todo quadro.
- Sala-proxy:
  - catcher difuso, nunca emissivo;
  - paredes de fechamento emissivas, invisíveis à câmera e sem bloquear o sol;
  - teto não bloqueia o sol.
- Luz: NNLS em renders de calibração (janelas, ambiente ≤50%, sol). A cor da luz é o iluminante estimado da foto.
- Assets: frente −Y, origem no centro da base. `coffee_table` precisa de `yaw_fix 90` (catálogo/import). O vidro de quadros vira vidro fino.
- Luminárias sempre apagadas. Semente fixa. Render em subprocesso. Um modelo de IA por vez na GPU.
- Licenças: MoGe-2 e OneFormer são MIT; Poly Haven é CC0. **Evitar:** SegFormer (NVIDIA, não comercial), Depth Anything V2-L (não comercial) e Hunyuan3D (restrição territorial).
