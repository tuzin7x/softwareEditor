# 7xAi Lead Finder

Módulo de prospecção do setor imobiliário dentro do 7xAi. Encontra imobiliárias, corretoras, construtoras e
incorporadoras em **fontes autorizadas**, consolida os resultados num **CRM local** (SQLite), qualifica cada
empresa com critérios verificáveis e prepara o primeiro contato pelo **WhatsApp oficial (`wa.me`)**, sempre com
revisão e envio manual.

- Código: `mobilia/modules/lead_finder/` (registro em `__init__.py`, aba "Lead Finder", ordem 10).
- Testes: `tests/lead_finder/` (sem internet; respostas gravadas e fictícias).
- Passagem para a sessão principal: [`LEAD_FINDER_HANDOFF.md`](LEAD_FINDER_HANDOFF.md).

---

## 1. Como usar

| Aba | O que faz |
|---|---|
| **Pesquisar** | Onde (Brasil, região, estados, municípios), palavras-chave (17 padrão + personalizadas salvas), fontes, estimativa de custo, andamento ao vivo e cancelamento. |
| **Leads** | Tabela paginada com busca, filtros (status, UF, fonte, prioridade, WhatsApp, acompanhamento vencido, "não contatar"), ordenação por coluna, botão **OPEN WHATSAPP** por linha, exportação CSV, ações em lote (analisar sites, consultar CNPJ, recalcular pontuação) e o painel de CRM da empresa selecionada. |
| **Mensagens** | Modelos (profissional/direto), personalização com o nome e a cidade reais da empresa, edição, salvar modelo, copiar, abrir no WhatsApp e, opcionalmente, variação por IA. |
| **Painel** | Indicadores calculados do banco: encontradas, qualificadas, contatos iniciados, interessados, demonstrações, negociações, convertidos, funil, estados, cidades, fontes e próximos acompanhamentos. Funciona com zero registros. |
| **Configurações** | Quem envia, chave e limites do Google, servidor Overpass, importação da base de CNPJ, IA opcional e ferramentas de privacidade (LGPD). |

Rodar só o módulo, para desenvolvimento:

```bash
python -m mobilia.modules.lead_finder
```

## 2. Arquitetura

```
mobilia/modules/lead_finder/
├── __init__.py            registro (ModuleSpec id=lead_finder, factory preguiçosa) — não importa a interface
├── __main__.py            janela mínima para desenvolvimento
├── models.py              Lead, GeoScope, SearchQuery, estados do CRM, SourcePolicy
├── phone.py               telefones brasileiros (Anatel), tipo, link wa.me
├── geo.py                 UFs, regiões, municípios (API do IBGE + cache local)
├── keywords.py            palavras-chave → CNAE, etiquetas OSM, tipo Google; classificação
├── net.py                 HTTP (urllib) cancelável, limite por host, novas tentativas, logs sem segredos
├── config.py              settings.json (preferências) e secrets.json/variáveis de ambiente (chaves)
├── sources/               FONTES (pesquisa): osm, google_places, cnpj_open, brasilapi + políticas de licença
├── search.py              orquestração da pesquisa e limite de custo (Budget) — não grava nada
├── dedupe.py              chaves de duplicidade e consolidação campo a campo
├── storage.py             ARMAZENAMENTO: SQLite com migrações, CRM, prazos de guarda, exportação, painel
├── scoring.py             qualificação com justificativa por critério
├── site_check.py          análise da página principal do site oficial (robots.txt respeitado)
├── messages.py            modelos determinísticos + IA opcional
├── export.py              CSV (UTF-8 com BOM, ";", proteção contra fórmulas) + arquivo de atribuições
├── studio.py / api.py     ponte com o AI Studio (só leitura) e interface pública mínima
├── ops.py                 operações compostas executadas na thread do banco
├── tasks.py               Dispatcher (sinal Qt), DbService (thread única do banco), Task (cancelável)
└── ui/                    page, search_view, leads_view, detail, table, messages_view, dashboard_view,
                           settings_view, dialogs, parts, style, icons, context
```

**Regras seguidas:**

- Nada do motor: o módulo não importa numpy, PIL, torch, cv2, diffusers, QtWebEngine nem `mobilia.design`/
  `mobilia.video` (teste por AST). Só biblioteca padrão + PySide6 + `mobilia.runtime`/`mobilia.ui`.
- **Nenhuma rede nem banco na thread da interface.** O banco tem uma thread dedicada (`DbService`); pesquisas,
  importação, análise de sites, IBGE, BrasilAPI, Google e IA rodam em `Task` (thread + cancelamento). Os
  resultados voltam para a interface por um `Signal` (conexão enfileirada), como o `CliTask` do 7xAi.
- Visual: tokens de `mobilia/ui/theme.py` (cores, fontes Inter, QSS global) e `Segmented` de `design_widgets`.
  A folha de estilo do módulo (`ui/style.py`) só complementa o que o QSS global não cobre (tabela, cabeçalho,
  listas, campos de data) e é aplicada apenas à página. Cartões "vidro" com gradiente translúcido e brilho
  suave; transição de opacidade entre abas; números animados no painel.
- Cada aba fica numa área rolável: a janela continua cabendo em 1280×720 (modo compacto do 7xAi).
- Separação **pesquisa × armazenamento**: cada fonte declara uma `SourcePolicy` (prazo de guarda, exportável,
  atribuição). O banco aplica a política; a fonte não sabe como os dados são guardados.

## 3. Fontes de dados (avaliação e o que foi implementado)

| Fonte | Status | Custo | Guarda / exportação | Observações |
|---|---|---|---|---|
| **OpenStreetMap — Overpass API** | Implementada, padrão | Gratuita, sem chave | Sem limite / sim, com atribuição ODbL | Uma consulta por área (UF ou município), intervalo mínimo de 2 s por servidor, User-Agent identificável. Cobertura no Brasil é parcial (depende de voluntários). Servidor configurável. |
| **Receita Federal — dados abertos de CNPJ** | Implementada (índice local) | Gratuita | Sem limite / sim | O usuário baixa os ZIPs oficiais; o importador lê em fluxo e guarda só estabelecimentos **ativos** com CNAE do setor (8 subclasses), das UFs escolhidas. É a fonte mais completa (praticamente todas as empresas formais). |
| **BrasilAPI — CNPJ** | Implementada (enriquecimento) | Gratuita | Sem limite / sim | Consulta um CNPJ já conhecido. Sócios (QSA), e-mail e capital social são descartados. |
| **IBGE — localidades** | Implementada (municípios) | Gratuita | Cache local | Lista oficial de municípios por UF para o autocompletar. |
| **Google Places API (New) — Text Search** | Implementada, desligada por padrão | Paga (chave do usuário) | **30 dias** / **não** | Só `place_id` fica sem prazo. Limites por pesquisa e por mês; estimativa antes de pesquisar; Place Details para atualizar. |
| **Anthropic (mensagens por IA)** | Implementada, opcional | Paga (chave do usuário) | — | Só nome, cidade e segmento da empresa são enviados, por clique. Limite diário. |
| Provedores comerciais de CNPJ (busca por CNAE/cidade, ex.: CNPJá, Casa dos Dados, Speedio) | Avaliados, **não** implementados | Pagos, contrato | Conforme contrato | Exigem contratação e termos próprios; o contrato de `sources/base.py` permite acrescentar um adaptador sem mudar o resto. |
| Raspagem de Google Maps, portais (Zap, VivaReal, OLX), Instagram, WhatsApp Web | **Não** implementados | — | — | Violam termos de uso. Não fazem parte do módulo. |

Palavras-chave → fonte: no Google viram o texto da busca ("Imobiliária em Campinas - SP, Brasil", com
`includedType=real_estate_agency` quando aplicável); no OSM viram etiquetas (`office=estate_agent`...) e um
filtro de nome tolerante a acentos; na base de CNPJ viram códigos CNAE (6821-8/01, 6821-8/02, 6822-6/00,
6810-2/01, 6810-2/02, 6810-2/03, 4110-7/00, 4120-4/00). Palavras personalizadas filtram pelo nome.

**Consolidação:** a mesma empresa vinda de várias fontes vira um registro só. Ordem das chaves: mesmo registro
na fonte → CNPJ → telefone (principal, secundário ou WhatsApp) → domínio de site próprio → nome sem sufixos societários + cidade + UF. Dois CNPJs
diferentes nunca se juntam; domínios de redes sociais e portais não servem de chave. Para cada campo vence a
fonte mais confiável (manual > BrasilAPI > CNPJ aberto > site oficial > Google > OSM). A origem de cada campo e
de cada registro (fonte, data, consulta) aparece no painel da empresa.

## 4. WhatsApp

1. O botão **OPEN WHATSAPP** aparece só para números "adequados": WhatsApp confirmado (informado pela fonte,
   publicado em link oficial no site da empresa ou confirmado pelo usuário) ou celular válido ("provável",
   com contorno tracejado e aviso). Fixos sem confirmação, números de serviço (0800/4004) e empresas em "Não
   contatar" não têm botão.
2. O número é normalizado pelas regras da Anatel (DDD válido, nono dígito, código de operadora, +55).
3. O diálogo mostra o modelo escolhido já personalizado com dados reais e permite editar o texto.
4. `QDesktopServices.openUrl("https://wa.me/<número>?text=<mensagem>")` abre a conversa com o texto preenchido.
   O envio é feito pela pessoa, no WhatsApp.
5. Ao confirmar manualmente que uma empresa tem WhatsApp, o número exato é gravado; se a empresa tem mais de um
   telefone, o painel pergunta qual deles.
6. O CRM registra **"WhatsApp aberto"** e pergunta se a mensagem foi enviada. Só com a confirmação vira
   **"Mensagem enviada"** e o status passa de "Não contatado" para "Contato iniciado".

Não há envio automático, envio em massa nem automação do WhatsApp Web.

## 5. CRM

Estados: Não contatado, Contato iniciado, Aguardando resposta, Interessado, Demonstração enviada, Negociação,
Cliente convertido, Sem interesse, Não contatar. Histórico de ações (com distinção entre abrir o WhatsApp,
copiar, enviar, receber resposta e enviar demonstração), notas, lembretes com data (o próximo vira
"acompanhamento" e aparece no painel), prioridade manual, confirmação manual de WhatsApp, edição manual de
dados e vínculo com projetos e vídeos do AI Studio.

**Não contatar:** bloqueia WhatsApp, mensagens e lembretes, conclui os lembretes abertos e grava telefone, CNPJ
e domínio numa lista de supressão: novas pesquisas não trazem a empresa de volta. Para liberar, é preciso
confirmar que a empresa autorizou. **Excluir dados** apaga a empresa e todo o histórico (opcionalmente
mantendo a supressão).

Banco: `<dados do 7xAi>/lead_finder/leads.sqlite3` (Windows: `%LOCALAPPDATA%\MobiliaStudio\lead_finder`).
Migrações numeradas com `PRAGMA user_version` (atômicas; hoje v4); um banco de versão mais nova é recusado com
mensagem clara.

## 6. Qualificação

Pontuação de 0 a 100, cada critério com pontos, justificativa e evidência: site oficial (20), presença digital
só em rede social (8), telefone comercial (10), WhatsApp confirmado (10) ou provável (4), segmento (até 20:
incorporadora/empreendimentos > construtora > imobiliária > aluguel/administração), CNPJ ativo (5) e, após
"Analisar site", anúncios de imóveis (até 15), menções a imóveis vazios/prontos (até 10), lançamentos (5) e
uso de recursos visuais concorrentes (0, só informativo). Critério sem dado aparece como "não avaliado".
Prioridade: Alta ≥ 70, Média ≥ 40, Baixa. "Qualificada" no painel: pontuação ≥ limite configurável (60).

**Análise de site:** só a página principal do site informado pela fonte, por ação do usuário, se o
`robots.txt` permitir; redes sociais e portais não são lidos; guarda apenas contagens de termos e números de
WhatsApp de links oficiais (`wa.me`, `api.whatsapp.com`) — não guarda o conteúdo da página.

## 7. Custos e limites

| Item | Padrão | Onde mudar |
|---|---|---|
| Preço Google por 1.000 requisições | US$ 35,00 (faixa Text Search Enterprise, com telefone e site) | Configurações |
| Máximo de requisições Google por pesquisa | 20 | Configurações |
| Máximo de requisições Google por mês | 500 | Configurações |
| Páginas por consulta Google (20 resultados cada) | 1 (até 3) | Configurações |
| Gerações por IA por dia | 30 | Configurações |
| Sites analisados por lote | 25 | `settings.json` |

Requisições pagas não têm novas tentativas automáticas (o resultado de um timeout é desconhecido). Toda requisição paga é contada no momento em que é enviada, inclusive se a pesquisa for cancelada ou falhar no
meio; "Atualizar dados do Google" (Place Details) também respeita o limite mensal. Antes de uma pesquisa paga aparece a estimativa (requisições e valor) e o uso do mês. O uso fica em
`api_usage`. Confira sempre a tabela de preços vigente e o crédito gratuito mensal da sua conta Google.

## 8. Configurações e credenciais

- `settings.json`: preferências e limites (sem segredos).
- Chaves: variável de ambiente (`LEADFINDER_GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`) ou `secrets.json` na pasta
  de dados do usuário (permissão 600 onde o sistema permite), digitadas na aba Configurações. Nunca vão para o
  repositório, o executável, os logs (`net.redact`) ou as exportações. A interface mostra só a máscara
  (`AIza…9xQ`).

## 9. LGPD e termos dos provedores

- Dados de contato **comercial** de empresas; origem (fonte e data) registrada por campo e por registro.
- Minimização: sem e-mails, sem sócios, sem capital social; empresários individuais/MEI fora da busca no CNPJ
  por padrão (o nome costuma ser de pessoa física); opção explícita para incluir.
- Exclusão por empresa, exclusão total, exportação, lista de não contato e frase de não contato nas mensagens
  (ativada por padrão).
- Nada é enviado a terceiros sem ação do usuário (pesquisa, consulta, análise, IA: tudo por clique).
- Google: prazo de 30 dias aplicado ao abrir o módulo e sob demanda; dados do Google fora do CSV (fica o
  `place_id` na observação). OSM: atribuição ODbL no arquivo `<nome>_fontes.txt` ao lado do CSV.

## 10. Integração com o AI Studio

- A ponte (`studio.py`) **só lê** `<dados>/jobs/design_*/` (o `design.json`, as versões `.jpg` e os vídeos
  `antes_depois_*.mp4`) e permite vincular um projeto, um vídeo ou uma imagem a uma empresa ("Vincular projeto
  ou vídeo" no painel da empresa). O vínculo entra no histórico; abrir o item abre a pasta/arquivo no sistema.
- Para a sessão principal oferecer o caminho inverso (do vídeo para a empresa), há `lead_finder.api`:
  `search_leads(texto)`, `link_output(lead_id, caminho, kind)` e `links_for(lead_id)`. Nada do motor de mídia
  foi alterado.

## 11. Testes

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q tests/lead_finder
QT_QPA_PLATFORM=offscreen python -m pytest -q tests/test_modules.py tests/test_ui.py
QT_QPA_PLATFORM=offscreen python -m pytest -q -m "not slow and not models and not gpu"
```

185 testes no módulo (96 originais + 89 de regressão das 3 revisões do Codex), rodando também no GitHub Actions, sem internet (transporte HTTP falso com respostas gravadas e fictícias em
`tests/lead_finder/fixtures/`): telefones e `wa.me`, IBGE, palavras-chave, mensagens, IA (formato da
requisição), HTTP (novas tentativas, cancelamento imediato, limite por host, logs sem chaves), OSM, Google
(máscara de campos, limite de custo), BrasilAPI (minimização), importador da Receita com ZIP sintético
(filtros, cancelamento sem corromper o índice), análise de site com `robots.txt`, migrações, consolidação,
CRM, não contatar, prazos de guarda, exportação, painel, AI Studio e a interface (clique real no OPEN
WHATSAPP, pesquisa em thread, cancelamento, painel vazio, mensagens, configurações, janela principal).

## 12. Limitações conhecidas

- **Overpass não verificado ao vivo:** o servidor `overpass-api.de` recusou conexões do ambiente de nuvem onde o
  módulo foi desenvolvido. A consulta gerada foi aceita por outro servidor Overpass (resposta 200 com JSON
  válido, sem dados por ser uma instância regional). Conferir no PC do usuário (Pesquisar › Municípios ›
  Campinas › OpenStreetMap). O servidor pode ser trocado em Configurações.
- **Google Places e IA não verificados ao vivo** por falta de chaves: implementados conforme a documentação
  oficial e testados com respostas gravadas.
- **WhatsApp real não aberto:** o fluxo foi testado com clique na interface e o link `wa.me` validado, mas sem
  abrir o WhatsApp Desktop/Web. Roteiro dos testes reais em `LEAD_FINDER_HANDOFF.md` §6.
- Cobertura do OSM no Brasil é irregular; a base de CNPJ é a fonte mais completa, mas exige baixar os arquivos
  da Receita (vários GB) e importar.
- A base de CNPJ traz o telefone cadastrado na Receita, que pode estar desatualizado; WhatsApp nunca é presumido.
- A análise de sites lê só a página principal; sites feitos só em JavaScript podem render poucas evidências.
- Na base de CNPJ, uma busca "Brasil inteiro" pode trazer dezenas de milhares de empresas: a gravação é feita em
  lotes e a tabela é paginada no banco, mas a primeira consolidação leva alguns minutos.
