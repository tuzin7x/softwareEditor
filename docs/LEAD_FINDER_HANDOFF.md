# Lead Finder — passagem para a sessão principal

Resumo do que foi entregue na branch `feature/lead-finder`, do que a sessão principal precisa decidir ou fazer
na integração e do que ainda não foi verificado. Descrição completa do módulo em
[`LEAD_FINDER.md`](LEAD_FINDER.md).

## 1. Base e branch

| Item | Valor |
|---|---|
| Commit de base | `54a55abda37b62f7a0c93b06f7b08c8749048a89` (o guia ainda cita `435a87b`; a base pedida foi `54a55ab`) |
| Branch | `feature/lead-finder` |
| Destino do PR | `claude/sweet-bohr-4uzml2` (sem merge automático) |
| Merge da base | `origin/claude/sweet-bohr-4uzml2` (`eaee226`, checkpoint do redesenho) mesclada por merge, sem conflitos; testes rodados depois do merge |

## 2. Arquivos tocados

Todos dentro das áreas permitidas pelo `docs/PARALLEL_DEVELOPMENT.md`:

- `mobilia/modules/lead_finder/**` (código do módulo, só `.py`);
- `tests/lead_finder/**` (testes, `conftest.py`, `__init__.py` e fixtures fictícias em JSON);
- `docs/LEAD_FINDER.md` e `docs/LEAD_FINDER_HANDOFF.md`;
- `.github/workflows/tests.yml`: **só** o caminho `tests/lead_finder` acrescentado ao fim da lista do passo de
  testes rápidos (exceção prevista no §5 do guia; commit `ce0c00c`, separado para ser fácil de revisar).

**Nenhum** outro arquivo foi alterado: nem `app.py`, `modules.py`, `theme.py`, `brand.py`, `widgets.py`,
`design_widgets.py`, motor, vídeo, `packaging/`, `requirements/`, `build-windows.yml`, `pyproject.toml`, README,
RESUME ou ARQUITETURA. `tests/test_modules.py` também não precisou mudar.

## 3. Pedidos à sessão principal

1. **CI:** feito nesta branch (ver §2). Antes disso o `tests.yml` rodava só a lista fixa (123 testes) e **nenhum**
   teste do Lead Finder rodava no GitHub. Ao integrar, manter `tests/lead_finder` na lista.
2. **Empacotamento:** nada a fazer. O módulo só tem `.py` (sem `datas`), e `collect_submodules("mobilia.modules")`
   já inclui os subpacotes `sources` e `ui`. Resultado do build de teste no §6.
3. **Dependências:** nenhuma nova. Só biblioteca padrão (`urllib`, `json`, `csv`, `sqlite3`, `zipfile`,
   `html.parser`, `urllib.robotparser`) e PySide6.
4. **Documentação geral:** na integração, citar o módulo em README (tabela de estrutura: `mobilia/modules/`
   deixa de estar vazio), RESUME e ARQUITETURA §12.
5. **Identidade visual (decisão):** o pedido descrevia o 7xAi como "preto profundo + roxo neon", mas o tema atual
   em `theme.py`/`brand.py` é **grafite + latão**. Para a aba parecer nativa e não alterar o tema, o módulo usa
   **somente os tokens do tema** (cores, fontes, QSS global) e aplica por cima cartões translúcidos, brilho
   suave e animações. Se o 7xAi migrar para roxo neon no `theme.py`, o Lead Finder acompanha sem mudança de
   código. Duas cores de etiqueta de status que não existem no tema (lilás e azul frio) ficam em
   `ui/style.py:STATUS_COLOR`.

## 4. Integração com o AI Studio (opcional, para a sessão principal)

O módulo já lista os projetos do AI Studio (só leitura de `jobs/design_*`) e vincula projeto/vídeo/imagem a uma
empresa. Para o caminho inverso — um botão "Vincular a uma empresa" na tela de resultados ou de vídeo — a sessão
principal pode usar, **numa thread de trabalho**:

```python
from mobilia.modules.lead_finder import api
api.search_leads("alfa")                     # [(id, "Alfa Imóveis — Campinas/SP"), ...]
api.link_output(lead_id, caminho_do_mp4, kind="video")   # "projeto" | "video" | "imagem"
api.links_for(lead_id)
```

Nada do motor de redesenho ou de vídeo foi tocado; `api.py` não importa nada do motor.

## 5. Como testar

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q tests/lead_finder                      # 194 testes
QT_QPA_PLATFORM=offscreen python -m pytest -q tests/test_modules.py tests/test_ui.py
QT_QPA_PLATFORM=offscreen python -m pytest -q -m "not slow and not models and not gpu"
python -m pytest --collect-only -q
python -m mobilia.modules.lead_finder         # janela isolada do módulo
```

Resultado nesta branch (`ce0c00c`, depois do merge com `eaee226` e das correções da revisão):

| Onde | Resultado |
|---|---|
| `tests/lead_finder` (local, `1a03ba9`) | **194 passaram**: 96 originais + 98 de regressão das 5 revisões do Codex |
| `tests/test_modules.py` + `tests/test_ui.py` + `tests/lead_finder` (local) | **237 passaram** |
| Suíte rápida inteira (local, instalação completa do `tests.yml`) | **882 passaram, 257 pulados, 4 falharam** (em `799dc54`): as mesmas 4 falhas da base (`ModuleNotFoundError` de `torch`/`huggingface_hub`), que **falham igualmente em `claude/sweet-bohr-4uzml2` sem o Lead Finder** |
| GitHub Actions, "Testes (CPU)" | **298 passaram** em `07c8371` (123 da lista original + 175 do Lead Finder). Antes de `ce0c00c` o CI rodava só os 123 e nenhum teste do módulo |
| GitHub Actions, "Build Windows" (manual, `publicar=false`, run 44202166745) | **sucesso**: PyInstaller, autoteste do `.exe`, Inno Setup; "Publicar em Releases" **pulado** |
| Conteúdo do `.exe` (artefato `7xAi-onedir` baixado, SHA-256 conferido) | os 39 módulos do Lead Finder estão no `PYZ` do `7xAi.exe`, com o código corrigido; `sqlite3`, `ssl` (`_ssl.pyd`, `_sqlite3.pyd`), `urllib.robotparser`, `html.parser`, `zipfile` e `csv` presentes; nenhuma biblioteca do motor |

## 6. Pendências e testes reais que ainda faltam

O que **não** dá para afirmar sem testar com contas, chaves e um PC Windows de verdade:

| Item | Situação |
|---|---|
| OpenStreetMap (Overpass) ao vivo | `overpass-api.de` recusou conexões do ambiente de nuvem do desenvolvimento. A consulta gerada foi aceita por outro servidor Overpass (200, JSON válido, sem dados por ser regional). **Pendente no PC do usuário.** |
| Google Places ao vivo | Sem chave. Implementado pela documentação oficial (Places API New: Text Search e Place Details, máscara de campos) e testado com respostas gravadas. **Não está verificado com chave real.** |
| WhatsApp de verdade | O link `wa.me` e o fluxo foram testados com clique real na interface (offscreen), mas **não** abrindo o WhatsApp Desktop/Web de verdade. **Pendente no Windows.** |
| Abrir a aba no `.exe` | O `.exe` contém o módulo e o autoteste passa, mas o autoteste não abre a aba. **Pendente no Windows.** |
| IA (API da Anthropic) ao vivo | Sem chave. Opcional e desligada por padrão. |
| Importação da Receita com arquivos reais | Testada com ZIP sintético no formato oficial; não com os arquivos completos (vários GB). |
| BrasilAPI e IBGE ao vivo | **Verificados** (consulta de CNPJ público e 645 municípios de SP). |
| Provedores comerciais de CNPJ | Avaliados e documentados; não implementados (exigem contrato). |

### Roteiro dos testes reais (no Windows, com o `.exe` de teste)

Baixe o artefato `7xAi-onedir` (ou `7xAi-Setup`) do run 44202166745 em Actions › Build Windows. Use uma pasta de
dados limpa, se quiser isolar: defina `MOBILIA_HOME` antes de abrir.

**A. Abrir o módulo no executável**
1. Abrir o `7xAi.exe` › cabeçalho mostra `Ambientes | Lead Finder` › clicar em **Lead Finder**.
2. Esperado: a aba abre sem o aviso "Não foi possível abrir…"; as 5 abas internas trocam com transição suave;
   em 1280×720 nada fica cortado (as abas rolam).

**B. OpenStreetMap (gratuito, sem chave)**
1. Pesquisar › **Municípios** › UF SP › digitar "Campinas" (a lista do IBGE deve completar) › Adicionar.
2. Palavra-chave "Imobiliária" › fonte OpenStreetMap › Pesquisar.
3. Esperado: em até ~1 min, contadores "Novas" > 0 e registros na aba Leads com fonte "OpenStreetMap".
   Repetir a pesquisa: "Novas" = 0 e "Atualizadas" = o mesmo número (nada duplica). Testar **Cancelar** durante
   uma pesquisa por "Estados › SP" (mais lenta): o botão volta em ~1 s e o que chegou fica guardado.
4. Se falhar com "Sem conexão…" ou erro 429/504: trocar o servidor em Configurações (ex.: outro servidor Overpass
   público) e anotar a mensagem exibida.

**C. Google Places (pago — usar limites baixos)**
1. Google Cloud: projeto com faturamento, **Places API (New)** ativada e chave restrita a ela.
2. Configurações › Google: colar a chave › Salvar chave › marcar "Usar o Google Places" › Máximo por pesquisa
   **3**, Máximo por mês **10** › Salvar.
3. Pesquisar › Municípios › Campinas - SP › "Imobiliária" › só a fonte Google › Pesquisar.
4. Esperado: aparece a confirmação com a estimativa ("Até 1 requisição paga…") e o uso do mês; os resultados
   chegam com nome, telefone e site; em Configurações, "Uso neste mês: 1 de 10".
5. Cancelar no meio de uma pesquisa com várias palavras-chave: o uso do mês deve contar as requisições enviadas.
6. Exportar CSV: as linhas que só vieram do Google saem **sem** nome/telefone/endereço, com o `place_id` na
   observação.
7. No painel da empresa: "Atualizar dados do Google" (Place Details) soma 1 ao uso do mês.
8. Conferir no console do Google Cloud que o número de requisições bate com o contador do 7xAi.

**D. WhatsApp (com um número de teste seu)**
1. Leads › Nova empresa › cadastrar uma empresa fictícia com o **seu** celular.
2. Clicar em **OPEN WHATSAPP** na tabela › conferir o texto (nome da empresa, sem marcadores) › Abrir no WhatsApp.
3. Esperado: o WhatsApp Desktop (ou Web no navegador padrão) abre a conversa com o seu número e a mensagem
   preenchida, **sem enviar**. No 7xAi aparece "Você enviou a mensagem?".
4. Responder "Ainda não": o histórico mostra "WhatsApp aberto" e o status continua "Não contatado". Enviar
   manualmente no WhatsApp e registrar "Mensagem enviada" no painel: status vira "Contato iniciado".
5. Empresa com fixo e celular: em "WhatsApp: confirmado" o painel pergunta qual número; o escolhido é o que o
   botão passa a abrir.
6. Marcar "Não contatar": o botão some da tabela e do painel; repetir a pesquisa que a trouxe: ela não volta como
   contatável.
7. Aba Mensagens: escolher uma empresa, depois buscar outro texto com vários resultados: o botão OPEN WHATSAPP
   deve ficar desligado até escolher de novo.

**E. Base de CNPJ (opcional)**
1. Baixar da página oficial (Configurações › Página oficial) `Estabelecimentos*.zip`, `Empresas*.zip`,
   `Municipios.zip`; importar só uma UF pequena (ex.: AC) e pesquisar com "Base aberta de CNPJ".

Registre no PR o resultado de cada letra (passou / falhou + mensagem exibida).

## 7. Decisões que valem revisar

- **Prazos do Google:** interpretação conservadora dos termos do Google Maps Platform (guarda máxima de 30 dias,
  sem exportação, `place_id` permanente). Vale revisão jurídica antes de uso comercial amplo.
- **MEI/empresário individual** fora da busca no CNPJ por padrão (privacidade). A opção para incluir está na aba
  Pesquisar.
- **Frase de não contato** ativada por padrão nas mensagens.
- **Modelo de IA padrão:** `claude-opus-5-5` (configurável em Configurações; esforço baixo para mensagens curtas).

## 8. Revisão automática do Codex (PR #1)

Cada problema tem testes de regressão em `tests/lead_finder/test_lead_finder_review_fixes.py`, conferidos contra o
código anterior à correção (os testes falham lá e passam aqui). Todos os threads foram respondidos com o commit e
os testes, e marcados como resolvidos.

**Rodada 1** (`b5de8d5` → corrigida em `25d83cf`; 66 testes, 31 falham no código anterior):

| # | Problema | Correção |
|---|---|---|
| 1 | Ordenar por prioridade/pontuação gerava SQL inválido (`max(priority DESC, ...)`) | `SORTABLE` com tuplas de expressões explícitas |
| 2 | Cancelar durante uma requisição paga do Google não registrava o uso | uso emitido no `finally`; o Place Details também registra cada requisição e respeita o limite mensal |
| 3 | Mensagens: nova busca mantinha o destinatário anterior ativo (WhatsApp para a empresa errada) | destinatário limpo ao trocar a lista; respostas atrasadas ignoradas |
| 4 | Pontuação não era recalculada quando uma empresa ganhava dados | a `Store` recalcula a pontuação a cada mudança dos dados de entrada |
| 5 | CSV: `=1+1` passava sem proteção | apóstrofo para qualquer valor iniciado por `= + - @ \t \r`; só números puros ficam como estão |
| 6 | Confirmar WhatsApp trocava o celular pelo fixo | o número confirmado é gravado (migração v3) e o painel pergunta qual número |

**Rodada 2** (`ce0c00c` → corrigida em `799dc54`; 13 testes, 10 falham no código anterior):

| # | Problema | Correção |
|---|---|---|
| 7 | Novas tentativas automáticas do HTTP não passavam pelo limite do Google (timeouts → até 4 envios, 1 registrado) | requisições pagas com `retries=0`: cada envio real é contado; a falha aparece para o usuário |
| 8 | Busca por palavra personalizada não achava razão social acentuada (`lower()` do SQLite) | razão social dobrada com `geo.fold()` ao montar o índice |
| 9 | Prazo de guarda do Google não apagava o nome copiado no histórico | o histórico não recebe dados de fontes com prazo; a limpeza apaga cópias antigas |
| 10 | Voltar à aba Mensagens apagava o rascunho | modelos carregados uma vez; recarregar a lista não regenera o texto |
| 11 | Mesma empresa não era consolidada pelo telefone secundário | coluna `phone2_e164` (migração v4) nas chaves de duplicidade |
| 12 | Natureza jurídica desconhecida (sem `Empresas*.zip`) passava pelo filtro de MEI | natureza vazia fica fora da busca padrão; aviso na importação e na fonte |

**Rodada 3** (`799dc54` → corrigida em `505713a`; 10 testes, 7 falham no código anterior):

| # | Problema | Correção |
|---|---|---|
| 13 | CSV ignorava a confirmação manual do WhatsApp e o "Não tem WhatsApp" | `_export_whatsapp` aplica as decisões manuais; número só do Google continua fora |
| 14 | Pesquisa paga e "Atualizar dados do Google" simultâneas podiam estourar o limite mensal | reserva exclusiva de operação paga no `AppContext` |
| 15 | Telefone da lista de não contato voltava a ser contatável via atualização de outro registro | supressão checada em toda ingestão; número bloqueado nunca vira WhatsApp nem pode ser confirmado |
| 16 | Trocar para UF sem cache mantinha a lista de municípios da UF anterior | lista sempre da UF selecionada; sem lista, aceita o nome digitado |

**Rodada 4** (`505713a` → corrigida em `04ac23b`; 4 testes, 2 falham no código anterior):

| # | Problema | Correção |
|---|---|---|
| 17 | Número confirmado manualmente, só vindo do Google, sobrevivia ao prazo de guarda | a confirmação e a cópia no histórico são apagadas se nenhuma fonte restante sustenta o número |
| 18 | Nova análise de site sem WhatsApp mantinha o número da análise anterior | análise bem-sucedida sem WhatsApp remove a evidência anterior do site |

**Rodada 5** (`04ac23b` → corrigida em `1a03ba9`; 5 testes, 4 falham no código anterior):

| # | Problema | Correção |
|---|---|---|
| 19 | Variação por IA gerada para Alfa caía no rascunho de Beta se o destinatário mudasse durante a geração | resposta vinculada ao destinatário e ao rascunho de origem; descartada se mudaram |
| 20 | Bloquear uma empresa não bloqueava o mesmo celular em outra empresa já cadastrada | as empresas que compartilham o número são recalculadas na hora; verificação final antes de abrir o WhatsApp |

Migrações do banco nesta branch: v1 (estrutura), v2 (vínculos AI Studio), v3 (`wa_user_number`), v4
(`phone2_e164`). Todas atômicas; bancos de versões anteriores são atualizados ao abrir.
