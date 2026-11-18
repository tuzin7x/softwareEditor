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

## Perguntas rápidas

- **Por que o portátil é um único `.exe`?** O programa leva só a interface (cerca de 65 a 75 MB); o motor de IA
  (PyTorch, modelos) é baixado no primeiro uso, então cabe num arquivo só. Ele abre alguns segundos mais devagar
  que o instalado, porque se descompacta numa pasta temporária a cada abertura.
- **Por que não publicar sozinho a cada merge na `main`?** Para que nenhuma versão saia sem você decidir. O clique
  em *Run workflow* é a aprovação. (Se um dia quiser automático, dá para adicionar o gatilho `push` na `main`
  quando `mobilia/__init__.py` mudar; as mesmas verificações continuam valendo.)
- **Onde ficam as chaves e senhas?** Em lugar nenhum do repositório nem do `.exe`. O workflow só usa o
  `GITHUB_TOKEN` temporário que o GitHub cria em cada execução.
