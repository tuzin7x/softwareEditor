# Desenvolvimento paralelo no 7xAi

Este guia é para uma **segunda sessão de desenvolvimento** (outra conta do Claude ou outra pessoa) que vai criar um
módulo novo do 7xAi, o **7xAi Lead Finder**, sem atrapalhar o trabalho principal no motor de redesenho e de vídeo.

Leia nesta ordem: este arquivo, depois o [`README.md`](../README.md), o [`RESUME.md`](../RESUME.md) e as seções 11 e
12 de [`docs/ARQUITETURA.md`](ARQUITETURA.md).

---

## 1. Estado atual do projeto

- **O que é o 7xAi:** programa para Windows (`.exe`, interface em PySide6) que faz **redesenho de ambientes por IA**
  a partir de uma única foto. As funções são: mudar estilo, reforma completa, redesign simples, mobiliar ambiente
  vazio, editar área, fachada, decoração de evento e texto para ambiente. Também há "Ampliar 4K" e "Vídeo antes e
  depois", em 16:9 e 9:16.
- **Onde roda:** 100% local, na placa de vídeo do usuário. O alvo é uma RTX 5070 Laptop de 8 GB, que gera cada
  imagem em cerca de 8 a 11 s. Nenhuma foto sai do PC.
- **Modos antigos (legado/avançado):** são vídeos de montagem. No modo A, móveis 3D entram numa foto vazia. No
  modo B, uma foto mobiliada vira sala vazia, os móveis entram um a um e o vídeo termina na foto exata. Esses modos
  continuam no programa pelo botão "Avançado (vídeo de montagem, legado)".
- **Em andamento na sessão principal:** ajuste de qualidade do redesenho, que **ainda está em avaliação e não foi
  aprovado pelo usuário**. A seguir vem "Esvaziar ambiente", feito pelo motor de difusão, para servir de sala vazia
  ao modo B.
- **CI:** o GitHub Actions funciona, porque o repositório é público.
  - `tests.yml` roda os testes rápidos.
  - `build-windows.yml` gera o `.exe` e o instalador.
- **Ponto de extensão para abas novas:** o registro de módulos (`mobilia/ui/modules.py` e o pacote
  `mobilia/modules/`) já existe e está testado (`tests/test_modules.py`). **Nenhum módulo está incluído.** Sem
  módulos, a janela é exatamente a de antes.

## 2. Branch e commit de base

| Item | Valor |
|---|---|
| Branch de integração (desenvolvimento) | `claude/sweet-bohr-4uzml2` |
| Commit de base para o Lead Finder | tag anotada **`lead-finder-base`** (o hash completo é informado pela sessão principal na entrega e aparece em `git rev-parse lead-finder-base^{commit}`) |
| Branch do novo módulo | `feature/lead-finder` (criada a partir do commit de base) |
| `main` | **não usar**: está desatualizada de propósito (ver seção 13) |

Para criar a branch:

```bash
git clone https://github.com/tuzin7x/softwareEditor.git
cd softwareEditor
git fetch origin --tags
git switch -c feature/lead-finder lead-finder-base
git push -u origin feature/lead-finder
```

Se a sessão recusar o push de `feature/lead-finder` (algumas sessões só podem enviar branches `claude/...`), use
`claude/lead-finder` no lugar; o Pull Request e as regras são os mesmos. Observação: push em `claude/**` também
dispara o `build-windows.yml`, que gera o `.exe` mas não publica nada sem `[release]`.

Confira se a base está certa: `git log -1 --oneline` deve mostrar o commit da tag `lead-finder-base` (o mesmo hash
informado na entrega), e `mobilia/ui/modules.py` e
`tests/test_modules.py` devem existir.

## 3. Arquitetura da interface (PySide6), em resumo

```
mobilia/__main__.py, packaging/launcher.py
  └─ mobilia.ui.app.main()        --worker (repassa ao motor) | --selftest | --version | janela
       └─ MainWindow (QMainWindow, sem QTabWidget)
            ├─ cabeçalho: marca 7xAi · [seletor de módulos, só se houver módulo] · botão "Avançado" · selo
            ├─ [QStackedWidget de módulos, só se houver módulo]
            │    ├─ página 0 = fluxo de redesenho (Stepper + palco à esquerda + páginas à direita)
            │    └─ página N = cada módulo, criada na 1ª abertura
            └─ StatusBar (GPU, mensagens, versão)
```

| Arquivo | Papel |
|---|---|
| `mobilia/ui/app.py` | Janela principal do redesenho: páginas, navegação `go()`, pedidos ao motor. **Não editar.** |
| `mobilia/ui/modules.py` | Registro de módulos: `ModuleSpec`, `register()`, `ModuleHost`. **Não editar** (ver seção 9). |
| `mobilia/modules/__init__.py` | Descobre e importa os subpacotes de módulos. Um módulo quebrado é ignorado. |
| `mobilia/ui/runner.py` | `CliTask`: roda o motor (`python -m mobilia.cli <cmd> --json`) num subprocesso e lê eventos JSON. |
| `mobilia/ui/theme.py`, `brand.py` | Cores (tokens), folha de estilo global (QSS), fontes Inter, ícones de linha. |
| `mobilia/ui/widgets.py` | `on_click`, `StepHeader`, `rule`, `tracked_label`, `ImageView`, `DropZone`, `StatusBar`. |
| `mobilia/ui/design_widgets.py` | `Segmented`, `chip`, `CardButton`, comparador antes/depois, editor de máscara. |
| `mobilia/ui/legacy.py` | Janela do modo legado (vídeo de montagem). |
| `mobilia/ui/setup_dialog.py` | Instalação do motor no primeiro uso. |

Regras da interface que valem para qualquer módulo:

- **O `.exe` da interface não leva numpy, PIL, torch, cv2 nem QtWebEngine** (`excludes` em
  `packaging/7xai.spec`). Um módulo não pode importar essas bibliotecas. O teste
  `tests/test_modules.py::test_module_code_does_not_import_engine_libraries` verifica isso.
- Todo `clicked` deve ser ligado com `widgets.on_click(btn, fn, *args)`, que descarta o argumento `checked`.
- Para seguir o visual, use `setObjectName` com os nomes que o QSS já estiliza:
  - estrutura: `card`, `notice`, `section`, `chip`;
  - botões: `primary`, `ghost`, `outline`, `danger`, `link`;
  - texto: `hint`, `muted`, `body`, `caption`, `headline`, `title`, `subtitle`.

  As cores ficam em `theme.ACCENT`, `theme.TEXT`, `theme.MUTED` etc. Não altere `theme.py`. Ícones que não existem
  (busca, telefone, pessoa) o módulo desenha no próprio pacote.
- Trabalho demorado (rede, banco) roda **fora da thread da interface**: `threading.Thread` mais um `Signal` interno
  ligado na thread da interface, o mesmo padrão do `CliTask`. Nunca bloqueie a janela.
- Use a barra de status compartilhada com `window.statusBar().showMessage("...")`.
- Os helpers privados de `app.py` (`_button`, `_section`, `Page`) **não** devem ser importados: copie o padrão
  para o seu pacote.

## 4. Como adicionar um módulo à navegação (passo a passo)

1. Crie o pacote `mobilia/modules/lead_finder/`.
2. Em `mobilia/modules/lead_finder/__init__.py`, registre o módulo. A página é criada só quando o usuário abre a
   aba:

   ```python
   """7xAi Lead Finder: pesquisa de imobiliárias, contatos e CRM local."""
   from mobilia.ui.modules import ModuleSpec, register


   def _page(window):
       from .ui import LeadFinderPage        # import adiado: só carrega ao abrir a aba
       return LeadFinderPage(window)


   register(ModuleSpec(id="lead_finder", title="Lead Finder", factory=_page, order=10,
                       tooltip="Encontrar imobiliárias e organizar contatos comerciais"))
   ```

3. Em `mobilia/modules/lead_finder/ui.py`, crie a página (`QWidget`). Os métodos `on_show()` e `on_hide()` são
   opcionais e são chamados ao entrar e ao sair da aba:

   ```python
   from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

   from mobilia.ui.widgets import on_click  # noqa: F401  (ligar botões sempre com on_click)


   class LeadFinderPage(QWidget):
       def __init__(self, window):
           super().__init__()
           self.main_window = window        # não use self.window: é um método do QWidget
           lay = QVBoxLayout(self)
           title = QLabel("Lead Finder"); title.setObjectName("headline")
           lay.addWidget(title); lay.addStretch(1)

       def on_show(self):
           self.main_window.statusBar().showMessage("Lead Finder")
   ```

4. Pronto. Não é preciso editar `app.py`, `mobilia/modules/__init__.py` nem o spec:
   - `mobilia/modules/__init__.py` importa sozinho todo subpacote que não começa com `_`;
   - `packaging/7xai.spec` já inclui os submódulos com `collect_submodules("mobilia.modules")`.

   Ao abrir o app, o cabeçalho mostra o seletor `Ambientes | Lead Finder`. "Ambientes" é o fluxo de redesenho.
5. Se o seu pacote tiver arquivos que não são `.py` (ícones, JSON), eles precisam de uma linha a mais em `datas`
   no spec. Avise no Pull Request.
6. Para rodar a página isolada durante o desenvolvimento, pode ter um `__main__.py` no pacote com uma
   `QMainWindow` mínima e `theme.apply(app)`:

   ```bash
   python -m mobilia.modules.lead_finder
   ```

Comportamento garantido pelo registro (os três primeiros itens são testados em `tests/test_modules.py`):

- se o import do módulo falhar, a janela abre normalmente e o erro fica em `mobilia.ui.modules.failed`;
- se a `factory` falhar, a aba mostra um aviso em vez de derrubar o app;
- trocar de aba não perde o estado do redesenho;
- um redesenho em andamento continua rodando em segundo plano (por construção: a troca de aba só muda a página
  visível e não toca no processo do motor; ainda não há teste automático para isso).

## 5. Arquivos que a segunda conta PODE criar ou modificar

- `mobilia/modules/lead_finder/**`: todo o código do módulo (interface, busca, CRM, mensagens).
- `tests/lead_finder/**`: testes do módulo, em arquivos `test_lead_finder_*.py` dentro dessa pasta. Use as
  fixtures do padrão de `tests/test_modules.py`, com `QT_QPA_PLATFORM=offscreen` e `MOBILIA_HOME` apontando para
  uma pasta temporária.
- `tests/test_modules.py`: **só** o teste `test_repository_ships_no_module_yet`, e só se for preciso. Ele já aceita
  subpacotes válidos (pasta com `__init__.py`), então um `mobilia/modules/lead_finder/` correto passa sem mudança.
- `docs/LEAD_FINDER*.md`: documentação do módulo.
- **Dependências:** prefira a biblioteca padrão (`urllib`, `json`, `csv`, `sqlite3`) e o PySide6 (inclusive
  `QtNetwork`). Não existe arquivo de requisitos próprio do módulo: o `.exe` e o CI só instalam
  `requirements/ui.txt` e a linha `pip install` do `tests.yml`. Uma dependência nova e **inevitável** precisa
  entrar nesses dois lugares, senão o módulo cai em `failed` no `.exe` sem aviso e o CI falha. **Não edite esses
  arquivos:** peça a inclusão à sessão principal no PR, com a justificativa (pacote pip novo muda o `.exe`).
- Fora disso, **só linhas acrescentadas** e quando não houver alternativa, sempre explicadas no PR. Exemplos:
  - uma entrada em `datas` ou `hiddenimports` de `packaging/7xai.spec`;
  - o caminho `tests/lead_finder` a mais na lista de `.github/workflows/tests.yml`.

## 6. Arquivos e componentes que a segunda conta NÃO DEVE modificar

São o motor de redesenho, de vídeo e de análise, e o esqueleto da interface. Eles estão em edição ativa na sessão
principal:

- `mobilia/design/**`: motor de redesenho por IA.
- `mobilia/video/**`: vídeo antes e depois e codificação.
- `mobilia/removal/**`, `mobilia/layers2d/**`, `mobilia/composite/**`, `mobilia/gen/**`: modos legados A/B.
- `mobilia/analysis/**`, `mobilia/render/**`, `mobilia/layout/**`, `mobilia/assets/**`.
- `mobilia/pipeline.py`, `mobilia/pipeline_b.py`, `mobilia/jobs.py`, `mobilia/runtime.py`, `mobilia/bootstrap.py`,
  `mobilia/hardware.py`, `mobilia/warmup.py`, `mobilia/config.py`.
- `mobilia/cli.py`: um subcomando novo só por combinação prévia com a sessão principal e só por acréscimo.
- `mobilia/ui/app.py`, `legacy.py`, `runner.py`, `theme.py`, `brand.py`, `widgets.py`, `design_widgets.py`,
  `setup_dialog.py` e `modules.py`: a integração acontece **só pelo registro**.
- `packaging/*` (exceto linhas acrescentadas, ver seção 5), `GERAR_EXE.bat`, `.github/workflows/build-windows.yml`.
- `requirements/**` (inclusive `ui.txt` e `dev.txt`; ver "Dependências" na seção 5), `pyproject.toml`.
- `mobilia/__init__.py` (versão), `mobilia/__main__.py`, `mobilia/modules/__init__.py` (descoberta),
  `mobilia/photo.py`, `mobilia/diagnostics.py`, `mobilia/telemetry.py`.
- `scripts/**`, `.gitignore`.
- `tests/test_ui.py`, `tests/test_design.py`, `tests/test_showcase.py`, `tests/conftest.py`, `tests/fake_*.py`,
  `tests/fixtures/**`.
- `README.md`, `RESUME.md`, `docs/ARQUITETURA.md`, este arquivo e os demais `docs/**` (exceto
  `docs/LEAD_FINDER*.md`): a sessão principal atualiza esses arquivos na integração. Descreva o módulo em
  `docs/LEAD_FINDER.md`.

Se precisar de algo dessas áreas (um ícone novo em `theme.py`, um helper movido para `widgets.py`, um subcomando no
motor), **peça no PR ou numa issue**. A sessão principal faz a mudança.

## 7. Como executar os testes

Use Python 3.13, no Linux ou no Windows. O venv precisa ter `numpy`, porque `tests/conftest.py` o importa, além
de `PySide6` e `pytest`.

**Instalação mínima** — basta para `tests/test_modules.py`, `tests/test_ui.py` e `tests/lead_finder/`:

```bash
# Linux / macOS (bash)
python3.13 -m venv .venv && . .venv/bin/activate
pip install numpy pillow PySide6==6.12.0 pytest
```

```powershell
# Windows (PowerShell)
py -3.13 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install numpy pillow PySide6==6.12.0 pytest
```

**Instalação completa** — necessária para a suíte rápida inteira e para `--collect-only`, porque vários arquivos de
teste do motor importam `cv2`, `scipy`, `shapely`, `OpenEXR` e `imageio` direto. É a mesma linha do `tests.yml`:

```bash
pip install numpy scipy pillow shapely opencv-python-headless imageio-ffmpeg OpenEXR PySide6==6.12.0 pytest
```

Comandos no Linux (Qt sem tela: `QT_QPA_PLATFORM=offscreen`):

```bash
# registro de módulos + interface (rápido, ~30 s) — instalação mínima
QT_QPA_PLATFORM=offscreen python -m pytest -q tests/test_modules.py tests/test_ui.py

# só o seu módulo — instalação mínima
QT_QPA_PLATFORM=offscreen python -m pytest -q tests/lead_finder

# a suíte rápida inteira (sem render, sem modelos de IA, sem GPU) — instalação completa
QT_QPA_PLATFORM=offscreen python -m pytest -q -m "not slow and not models and not gpu"

# conferir que tudo é coletado (não roda nada) — instalação completa
python -m pytest --collect-only -q
```

Os mesmos comandos no Windows (PowerShell). No Windows o `offscreen` é opcional: sem ele, as janelas dos testes
aparecem por um instante.

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest -q tests/test_modules.py tests/test_ui.py
python -m pytest -q tests/lead_finder
python -m pytest -q -m "not slow and not models and not gpu"
python -m pytest --collect-only -q
```

Use o caminho da pasta (`tests/lead_finder`), não um curinga como `tests/test_lead_finder*.py`: o PowerShell não
expande curingas para o pytest, e no bash o curinga dá erro enquanto nenhum arquivo existir.

Os marcadores ficam em `pyproject.toml`:

| Marcador | O teste precisa de |
|---|---|
| `slow` | Blender ou modelos de IA em CPU |
| `models` | pesos dos modelos |
| `gpu` | placa NVIDIA |

Os testes do Lead Finder **não podem acessar a internet**: simule as fontes de dados com respostas gravadas e
anônimas.

## 8. Como integrar (Pull Request)

1. Trabalhe só em `feature/lead-finder`. Faça commits pequenos e com mensagens claras, em português.
2. Traga as novidades da base com frequência, **por merge** (sem rebase de branch já publicada e sem force-push):

   ```bash
   git fetch origin
   git merge origin/claude/sweet-bohr-4uzml2
   ```

3. Antes de abrir o PR, rode a suíte rápida completa (seção 7, instalação completa), `tests/test_modules.py` e
   `tests/lead_finder`.
4. Abra o Pull Request de `feature/lead-finder` para `claude/sweet-bohr-4uzml2`, **nunca para `main`**. Na
   descrição, inclua:
   - o que o módulo faz;
   - a lista de arquivos fora de `mobilia/modules/lead_finder/` que foram tocados, com o motivo;
   - as dependências novas, se houver;
   - como testar.
5. O `tests.yml` roda uma lista fixa de arquivos de teste rápidos (inclusive `tests/test_ui.py` e
   `tests/test_modules.py`) a cada push em `claude/**`, `feature/**` e `main`, e em PRs para `claude/**` e `main`.
   Peça no PR a inclusão de `tests/lead_finder` nessa lista (seção 5). Para gerar um `.exe` de teste da
   branch, use **Actions › Build Windows › Run workflow**, escolhendo `feature/lead-finder` e deixando "publicar"
   desmarcado.
6. A sessão principal revisa, integra na branch de desenvolvimento e atualiza README, RESUME e ARQUITETURA.

## 9. Conflitos: como evitar e como resolver

**Evitar:** se você seguir as seções 5 e 6, os únicos arquivos compartilhados são linhas acrescentadas no spec ou
no `tests.yml` e, se preciso, o teste `test_repository_ships_no_module_yet` de `tests/test_modules.py`. Nesses casos, conflito é raro e trivial.

**Resolver**, se acontecer um conflito no `git merge origin/claude/sweet-bohr-4uzml2`:

1. `git status` mostra os arquivos em conflito.
2. Arquivo do motor ou da interface principal (seção 6): **fique com a versão da base**, com
   `git checkout --theirs <arquivo>` durante o merge, e refaça a sua necessidade pelo registro ou peça à sessão
   principal.
3. Linhas acrescentadas em listas (spec, `tests.yml`): mantenha **as duas** versões, a da base e a sua.
4. Arquivos do seu módulo: resolva normalmente.
5. Rode os testes da seção 7. Depois faça `git add <arquivos>`, `git commit` e `git push`, sem `--force`.
6. Se não tiver certeza, aborte com `git merge --abort` e descreva o caso no PR.

Nunca use `git push --force`, `git reset --hard` em branch publicada nem reescrita de histórico.

## 10. Como gerar o `.exe`

- **Localmente, no Windows:** instale o Python 3.13 e dê dois cliques em `GERAR_EXE.bat`. Ele chama
  `packaging/build_windows.ps1`, que usa o PyInstaller com `packaging/7xai.spec`. O resultado é
  `dist\7xAi\7xAi.exe`, já instalado em `%LOCALAPPDATA%\Programs\7xAi`, com atalho na Área de Trabalho. Com o
  Inno Setup 6 instalado, também sai `dist\7xAi-Setup-<versão>.exe`.
- **GitHub Actions:** `build-windows.yml` gera o `.exe` e o instalador a cada push em `claude/**` e em `main`, e
  manualmente por "Run workflow" em qualquer branch.
  - Uma prerelease é publicada só com `[release]` na mensagem do commit ou com "publicar" marcado no disparo
    manual.
  - **A segunda conta não deve usar `[release]` nem "publicar"**: publicar é decisão da sessão principal.
- O `.exe` contém só a interface. O motor de IA (vários GB) é instalado no primeiro uso. Um módulo puramente de
  interface, como o Lead Finder, entra no `.exe` automaticamente pelo `collect_submodules`.

## 11. Dados, privacidade e regras legais do Lead Finder

- **Fontes:** use somente fontes **autorizadas ou públicas** de dados empresariais, como APIs oficiais com chave
  do usuário, bases abertas de CNPJ e diretórios que permitem uso comercial.
  - Respeite os termos de uso de cada site e o `robots.txt`.
  - **Não faça raspagem que viole termos** nem contorne limites, logins ou captchas.
- **LGPD:** são dados de **contato comercial** de empresas.
  - Registre a origem de cada contato (fonte e data).
  - Ofereça como excluir um contato e como exportar os dados.
  - Não colete dados pessoais sensíveis.
  - Não envie dados a terceiros sem ação explícita do usuário.
- **WhatsApp:** só por link `https://wa.me/<número>?text=<mensagem urlencoded>`, aberto com
  `QDesktopServices.openUrl`, sempre por **clique do usuário**. Não há envio automático nem em massa, e não se usa
  API não oficial.
- **Mensagens por IA:** a pessoa sempre revisa o texto antes de abrir o WhatsApp. Se for usada uma API de IA na
  nuvem, isso deve ser opcional e configurado pelo usuário. O motor local de imagens não deve ser usado para isso.
- **CRM:** os dados ficam **no computador do usuário**, num banco `sqlite3` dentro da pasta de dados do app, nunca
  no repositório. O `.gitignore` já ignora `*.db`, `*.sqlite` e `*.sqlite3`.
- **Chaves de API: nunca no repositório**, nem em testes ou exemplos.
  - Elas ficam nas configurações do app, num arquivo na pasta de dados do usuário, ou em variáveis de ambiente.
  - O `.gitignore` já cobre `.env`, `*.key`, `*.pem`, `config.local.json` e `secrets/`.
  - Se uma chave for commitada por engano, avise: ela precisa ser **revogada e trocada**, porque o histórico não
    será reescrito.
- **Fixtures de teste:** só dados fictícios ou anonimizados. Nunca use contatos reais nem fotos do usuário
  (`usuario_*`).

### Onde ficam os dados do app

| Sistema | Pasta |
|---|---|
| Windows | `%LOCALAPPDATA%\MobiliaStudio` (o nome antigo foi mantido de propósito) |
| Linux (desenvolvimento) | `~/.local/share/mobilia-studio` |
| Testes | o valor de `MOBILIA_HOME` |

Use `mobilia.runtime.app_home() / "lead_finder"` para os arquivos do módulo, como o banco do CRM e as
configurações. Essa função respeita `MOBILIA_HOME` nos testes. Importar só `mobilia.runtime` é permitido: ele não
traz bibliotecas do motor.

## 12. Escopo do Lead Finder (planejado, para referência)

O módulo vai incluir:

- pesquisa de imobiliárias por estado e cidade, com várias palavras-chave;
- integração com fontes autorizadas de dados empresariais;
- lista de empresas e contatos comerciais;
- botão de WhatsApp com mensagem preenchida;
- mensagens comerciais escritas por IA, revisadas pelo usuário;
- CRM com histórico de contatos;
- classificação de oportunidades.

A integração futura com o AI Studio (o fluxo de redesenho) será feita pela sessão principal, por exemplo levando
um contato até um redesenho. **Nada disso existe ainda.** O repositório tem só o ponto de extensão.

## 13. Por que a `main` não foi atualizada

A `main` tem só o commit inicial. Ela **não foi atualizada de propósito**:

- a reformulação atual (redesenho por IA como fluxo principal) **ainda está em avaliação** e o usuário não aprovou
  a qualidade;
- um merge agora apresentaria como estável algo que não é;
- o trabalho fica **seguro** na branch de desenvolvimento `claude/sweet-bohr-4uzml2` no GitHub (a sessão principal
  envia os commits sem force-push), e cada push gera `.exe` pelo CI.

Regra: **`claude/sweet-bohr-4uzml2` é a branch de integração.** A `main` só recebe a integração quando o usuário
aprovar uma versão, o que coincide com uma release publicada. Nenhuma das duas sessões faz merge na `main` por
conta própria.
