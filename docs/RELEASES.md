# Como publicar uma versão do 7xAi

Este guia é para o dono do projeto. Cada **versão oficial** vira uma página em
[Releases](https://github.com/tuzin7x/softwareEditor/releases) com:

| Arquivo | O que é |
|---|---|
| `7xAI-Setup-vX.Y.Z.exe` | instalador (recomendado): instala, cria o atalho na Área de Trabalho e atualiza a versão anterior |
| `7xAI-Portable-vX.Y.Z.exe` | portátil: um único `.exe` que roda sem instalar |
| `SHA256SUMS.txt` | a "impressão digital" (SHA-256) de cada arquivo, para conferir que o download não foi alterado |

As notas da Release trazem o número da versão, a data, as novidades, as correções, as melhorias de desempenho, as
notas importantes de atualização e o SHA-256 de cada arquivo. O texto vem do [`CHANGELOG.md`](../CHANGELOG.md).

O link fixo para a versão mais nova é <https://github.com/tuzin7x/softwareEditor/releases/latest>.

## Resumo em 4 passos

1. **Abra um Pull Request para a `main`** com a versão nova e o texto dela no `CHANGELOG.md`.
2. **Faça o merge** do PR, depois que os testes ficarem verdes.
3. Na aba **Actions**, escolha **"Release oficial (instalador + portátil)"** → **Run workflow** → em *Use workflow
   from* deixe **main** → digite a versão (ex.: `0.9.0`) → deixe **prerelease** desmarcado → **Run workflow**.
4. Espere uns 30 a 40 minutos. Se tudo passar, a Release aparece publicada. Se algo falhar, **nada** é publicado.

## Passo a passo detalhado

### 1. Preparar a versão (num PR para a `main`)

No mesmo PR, mude **dois arquivos**:

- **`mobilia/__init__.py`**: troque `__version__ = "..."` pelo número novo, por exemplo `__version__ = "0.9.0"`.
  Essa é a **fonte única** da versão: o executável, o instalador, o `pyproject.toml` e a tag leem dali.
- **`CHANGELOG.md`**: crie a seção da versão, com a data da publicação, e passe para ela os itens que estavam em
  `## [Não publicado]`:

  ```markdown
  ## [Não publicado]

  ## [0.9.0] - 2026-12-01

  ### Novidades
  - ...
  ### Correções
  - ...
  ### Desempenho
  - ...
  ### Notas importantes de atualização
  - ...
  ```

  A seção não pode conter a palavra **RASCUNHO**: enquanto ela estiver lá, o workflow se recusa a publicar. A
  seção `0.9.0` que já existe é um rascunho escrito para você revisar; ajuste o texto, coloque a data e apague
  o aviso. O `__version__` já está em `0.9.0`.

O primeiro PR para a `main` também leva todo o código da branch de desenvolvimento
(`claude/sweet-bohr-4uzml2` → `main`). Os PRs seguintes podem vir da mesma branch ou de outra.

### 2. Merge na `main`

Espere o check **"Testes (CPU)"** do PR ficar verde e faça o merge (de preferência *Create a merge commit* ou
*Squash*; nunca force-push na `main`).

### 3. Disparar a Release

1. Abra **Actions** → **Release oficial (instalador + portátil)**.
2. Clique em **Run workflow**.
3. Em **Use workflow from**, escolha **Branch: main** (com outra branch o workflow para na primeira etapa).
4. Em **versao**, digite exatamente o número do `mobilia/__init__.py`, sem o "v" (ex.: `0.9.0`).
5. **prerelease**: deixe **desmarcado** para uma versão estável (ela vira a "Latest" e o link "Baixar a última
   versão" passa a apontar para ela). Marque **só** para publicar um beta de teste autorizado: ele aparece em
   Releases com a etiqueta *Pre-release*, **não** vira "Latest" e as notas começam com um aviso de beta.
6. Clique no botão verde **Run workflow**.

### 4. O que o pipeline faz (e o que impede a publicação)

| Etapa | Confere | Se falhar |
|---|---|---|
| 1. Validar | a execução é da `main`; a versão é `X.Y.Z`; é igual à do código; o `pyproject.toml` não tem outra versão; o `CHANGELOG.md` tem a seção da versão, com itens e sem RASCUNHO; a tag `vX.Y.Z` e a Release ainda não existem | para aqui |
| 2. Testes | instala as dependências com versões fixas (`packaging/ci/requirements-tests.txt`), roda `pip check` e a suíte rápida (inclui o Lead Finder quando ele estiver na branch) | para aqui |
| 3. Build | dependências fixas do build + `pip check`; gera o programa (PyInstaller), o portátil de arquivo único e o instalador (Inno Setup); abre cada um em modo de autoteste (`--selftest`, sem janela) e confere a versão gravada no `.exe`; instala o instalador em silêncio e abre o programa instalado; calcula o SHA-256 | para aqui |
| 4. Publicar | confere de novo o SHA-256 e que os arquivos vêm do mesmo commit; cria a Release como rascunho, anexa os 3 arquivos, publica (a tag `vX.Y.Z` nasce nesse momento, no commit testado) e confere que a tag aponta para o commit certo | apaga o rascunho; nada fica publicado |

Tudo acontece numa única execução e a partir do mesmo commit: os testes, o build e a tag apontam para ele. Só a
etapa 4 tem permissão de escrever no repositório.

## Versionamento (`MAIOR.MENOR.CORREÇÃO`)

| Mudou | Aumente | Exemplo |
|---|---|---|
| só correções de erros, sem mudar o uso | CORREÇÃO | `1.0.0` → `1.0.1` |
| função nova ou melhoria, compatível com o que já existe | MENOR (e zere a correção) | `1.0.1` → `1.1.0` |
| mudança que quebra algo (ex.: trabalhos antigos não abrem, requisito novo de hardware) | MAIOR (e zere os outros) | `1.4.2` → `2.0.0` |

- A primeira versão oficial recomendada é a **`0.9.0`** (estável, série 0.x): o produto funciona de ponta a
  ponta, mas ainda tem limitações conhecidas de qualidade (ver `CHANGELOG.md`) e o app completo ainda não foi
  validado no Windows do dono. A `1.0.0` fica para quando essas pendências forem resolvidas.
- Versões `0.x` estáveis viram "Latest" normalmente; o número 0 só sinaliza que o produto ainda está amadurecendo.
- Cada número é publicado **uma vez só**. Se precisar corrigir uma versão já publicada, publique a próxima
  correção (`0.9.1`).
- Commits do dia a dia **não** geram Release. Os pushes em `claude/**`, `feature/**` e `main` geram só
  **artefatos** no Actions (instalador de teste e pasta do programa, guardados por 14 dias), úteis para testar
  antes de publicar.

## Desfazer / voltar para a versão anterior

- **Se a Release saiu com problema:** os usuários podem baixar a versão anterior na lista de
  [Releases](https://github.com/tuzin7x/softwareEditor/releases) e instalar por cima (o instalador aceita versão
  mais antiga). Corrija o problema e publique uma versão de correção nova (ex.: `0.9.1`). Esse é o caminho
  recomendado.
- **Para tirar a versão do ar:** na página da Release, **Edit** → marque *Set as a pre-release* (ela deixa de ser a
  "latest", e o link `releases/latest` volta para a anterior) ou **Delete**. Não reutilize o mesmo número: apagar a
  tag e publicar de novo com o mesmo número confunde quem já baixou.
- **Se o workflow parou no meio da etapa 4:** ele apaga o rascunho sozinho. Se sobrar um rascunho ou uma tag
  `vX.Y.Z` sem Release (raro), apague-os em *Releases* / *Tags* e rode de novo.
- **Código:** a tag `vX.Y.Z` marca o código exato daquela versão. Para voltar o código da `main` ao estado de uma
  versão, abra um PR que reverta os commits (botão **Revert** no PR original); não use force-push.

## Proteções recomendadas no GitHub (uma vez só)

Em **Settings**:

1. **Branches → Add branch ruleset** (ou *branch protection rule*) para `main`: exigir Pull Request, exigir o check
   **`tests`** (do workflow "Testes (CPU)"), bloquear force-push e exclusão.
2. **Tags → ruleset** para `v*`: só administradores criam/apagam tags de versão (o workflow usa o
   `GITHUB_TOKEN` do próprio repositório).
3. **Actions → General → Workflow permissions**: *Read repository contents permission* (o workflow de Release pede
   a escrita só para o job de publicar).
4. **Releases → Enable release immutability** (recomendado), para que arquivos e tag de uma Release publicada não
   possam ser trocados depois. O workflow é compatível: ele anexa os arquivos **antes** de publicar o rascunho.

## Atualização automática

O 7xAi instalado se atualiza sozinho a partir destas Releases, sem servidor nem custo (código em
`mobilia/ui/updater.py` e `mobilia/ui/updater_dialog.py`). Por isso **o nome dos arquivos, o `SHA256SUMS.txt` e a
marca "Latest" fazem parte do contrato**: não renomeie os arquivos à mão.

**Como funciona**

1. Alguns segundos depois de abrir, em segundo plano (a interface nunca espera), o app consulta
   `https://api.github.com/repos/tuzin7x/softwareEditor/releases/latest` — sem login, sem dados pessoais (o
   User-Agent é só `7xAi-updater`). No máximo **uma consulta automática por dia**; o botão **Atualizações →
   Procurar atualizações** (barra de status) consulta na hora. Sem internet, tempo esgotado ou limite de consultas
   do GitHub (403/429, 60 por hora por IP): nada aparece; na consulta manual, uma mensagem curta explica.
2. Só conta a Release **estável** marcada como "Latest", com tag `vX.Y.Z` maior que a versão instalada. Rascunhos e
   pré-lançamentos (`prerelease`) **nunca** são oferecidos.
3. Se houver versão nova: janela "Nova versão X.Y.Z disponível" com as notas (sem a tabela de downloads e sem a
   seção de SHA-256; Markdown sem HTML) e os botões **Atualizar agora**, **Depois** e **Pular esta versão** (a
   versão pulada não volta a aparecer sozinha, só numa consulta manual). Há também a caixa "Procurar atualizações
   automaticamente ao abrir" (o mesmo que o item do menu **Atualizações**).
4. **Atualizar agora** (instalado): baixa `SHA256SUMS.txt` e `7xAI-Setup-vX.Y.Z.exe` para
   `%LOCALAPPDATA%\MobiliaStudio\updates`, com barra de progresso e **Cancelar**; confere o SHA-256 (se não bater,
   apaga o arquivo e avisa); roda o instalador com
   `/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS /NORESTARTAPPLICATIONS /DIR=<pasta atual> /AUTOUPDATE`
   e fecha o 7xAi. O instalador espera 3 s o app terminar de fechar, troca os arquivos da pasta do programa e, **só
   com `/AUTOUPDATE`**, reabre o 7xAi (linha extra no `[Run]` do `installer.iss`). Se houver um trabalho em
   andamento, o app pergunta antes.
5. **Portátil** (arquivo único): o botão vira **Baixar nova versão**; o novo `7xAI-Portable-vX.Y.Z.exe` (conferido)
   é salvo ao lado do atual (ou em Downloads, se a pasta não aceitar escrita). Um `.exe` aberto não pode ser
   trocado: o usuário clica em **Abrir a nova versão** (o app abre o novo e fecha) e depois pode apagar o antigo.
6. Ao reabrir, o motor só refaz o que mudou: o `bootstrap` compara as impressões digitais do código e das
   dependências e, em geral, só copia o código do motor (segundos). PyTorch, Blender e os **modelos de IA não são
   baixados de novo**; trabalhos, configurações e o banco do Lead Finder (todos em `%LOCALAPPDATA%\MobiliaStudio`)
   não são tocados — o instalador só escreve na pasta do programa. O código da atualização fica em `mobilia/ui`,
   fora do motor: mudá-lo nem força essa cópia.

**Segurança:** só HTTPS; só os hosts `api.github.com`, `github.com`, `objects.githubusercontent.com` e
`release-assets.githubusercontent.com` (conferidos também em cada redirecionamento); o endereço inicial de cada
arquivo tem de ser `github.com/tuzin7x/softwareEditor/releases/download/...`; o SHA-256 do `SHA256SUMS.txt` é
obrigatório; nada das notas é executado; nenhum token ou segredo. Observação: a soma vem da mesma Release que o
instalador, então ela protege contra download corrompido ou trocado no caminho, não contra alguém com acesso de
escrita ao repositório (por isso as proteções da seção acima, como *release immutability*).

**Preferências:** `%LOCALAPPDATA%\MobiliaStudio\settings\updater.json` (`auto_check`, `last_check`,
`skipped_version`). Apagar o arquivo volta ao padrão (consulta automática ligada).

**Pelo código-fonte** (`python -m mobilia.ui.app`) a atualização fica desligada e o botão não aparece. Para testar
a interface sem gerar o `.exe`, use `MOBILIA_UPDATER_TEST=1`.

### Teste manual no Windows (antes de confiar na primeira atualização)

Os testes automáticos (`tests/test_updater.py`) cobrem a lógica e a janela com dublês; a instalação silenciosa e a
reabertura só dá para conferir num Windows de verdade:

1. Instale a versão N (com este recurso) pelo instalador. Abra o 7xAi, gere algo, crie um lead no Lead Finder.
2. Publique a versão N+1 (estável). Reabra o 7xAi (ou **Atualizações → Procurar atualizações**): a janela "Nova
   versão N+1 disponível" aparece com as notas.
3. **Pular esta versão** → feche e reabra: não aparece. **Procurar atualizações**: aparece de novo.
4. **Atualizar agora** → barra de progresso; teste **Cancelar** uma vez e depois atualize até o fim. O 7xAi fecha e,
   em alguns segundos, abre de novo sozinho; a barra de status mostra a versão N+1.
5. Confira: os trabalhos e os leads continuam lá; os modelos não foram baixados de novo (o assistente do motor, se
   aparecer, só copia o motor); o atalho da Área de Trabalho continua funcionando. O registro do instalador fica em
   `%LOCALAPPDATA%\MobiliaStudio\updates\7xAI-Setup-vN+1.log`; o instalador usado é apagado quando o 7xAi abre de novo.
6. Sem internet: reabra o 7xAi — nada aparece; **Procurar atualizações** avisa "sem internet".
7. Portátil: abra o `7xAI-Portable-vN.exe` → **Baixar nova versão** → o arquivo N+1 aparece na mesma pasta →
   **Abrir a nova versão** abre a N+1.
8. Desligue a consulta automática no menu **Atualizações** e reabra: nenhuma consulta (só a manual).

## Perguntas rápidas

- **Por que o portátil é um único `.exe`?** O programa leva só a interface (cerca de 65 a 75 MB); o motor de IA
  (PyTorch, modelos) é baixado no primeiro uso, então cabe num arquivo só. Ele abre alguns segundos mais devagar
  que o instalado, porque se descompacta numa pasta temporária a cada abertura.
- **Por que não publicar sozinho a cada merge na `main`?** Para que nenhuma versão saia sem você decidir. O clique
  em *Run workflow* é a aprovação. (Se um dia quiser automático, dá para adicionar o gatilho `push` na `main`
  quando `mobilia/__init__.py` mudar; as mesmas verificações continuam valendo.)
- **Onde ficam as chaves e senhas?** Em lugar nenhum do repositório nem do `.exe`. O workflow só usa o
  `GITHUB_TOKEN` temporário que o GitHub cria em cada execução.
