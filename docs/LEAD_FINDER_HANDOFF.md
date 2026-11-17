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
- `docs/LEAD_FINDER.md` e `docs/LEAD_FINDER_HANDOFF.md`.

**Nenhum** arquivo fora disso foi alterado: nem `app.py`, `modules.py`, `theme.py`, `brand.py`, `widgets.py`,
`design_widgets.py`, motor, vídeo, `packaging/`, `requirements/`, `.github/`, `pyproject.toml`, README, RESUME
ou ARQUITETURA. `tests/test_modules.py` também não precisou mudar.

## 3. Pedidos à sessão principal

1. **CI:** acrescentar `tests/lead_finder` à lista de arquivos do passo "Testes rápidos" em
   `.github/workflows/tests.yml` (só a linha acrescentada). Os testes rodam em ~10 s e não usam internet.
2. **Empacotamento:** nada a fazer. O módulo só tem `.py` (sem `datas`), e `collect_submodules("mobilia.modules")`
   já inclui os subpacotes `sources` e `ui`. Recomendo gerar um `.exe` de teste pelo **Run workflow** do Build
   Windows na branch, sem "publicar".
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
QT_QPA_PLATFORM=offscreen python -m pytest -q tests/lead_finder                      # 96 testes
QT_QPA_PLATFORM=offscreen python -m pytest -q tests/test_modules.py tests/test_ui.py
QT_QPA_PLATFORM=offscreen python -m pytest -q -m "not slow and not models and not gpu"
python -m pytest --collect-only -q
python -m mobilia.modules.lead_finder         # janela isolada do módulo
```

Resultado nesta branch, depois do merge com `eaee226` (Linux, Python 3.13, instalação completa do `tests.yml`):

- `tests/lead_finder`: **96 passaram**;
- `tests/test_modules.py` + `tests/test_ui.py`: **todos passaram** (com o Lead Finder registrado na janela);
- suíte rápida inteira: **802 passaram, 258 pulados, 4 falharam**. As 4 falhas
  (`test_design_weights_in_prefetch`, `test_vram_swap_moves_models_out_during_generate_and_back`,
  `test_ensure_local_separates_network_time`, `test_prefetch_includes_v3_weights`) são `ModuleNotFoundError` de
  `torch`/`huggingface_hub` e **falham igualmente em `claude/sweet-bohr-4uzml2` sem o Lead Finder** (conferido num
  worktree separado). Não têm relação com o módulo;
- `--collect-only`: coleta completa sem erros.

Teste manual sugerido no Windows:

1. Abrir o 7xAi › **Lead Finder** › Pesquisar › Municípios › SP › Campinas › "Imobiliária" › OpenStreetMap ›
   Pesquisar.
2. Na aba Leads, clicar em **OPEN WHATSAPP** de uma empresa com celular: conferir o texto, abrir, verificar que o
   WhatsApp (app ou web) abre com a mensagem preenchida e **não** envia sozinho; responder "Ainda não" e depois
   registrar "Mensagem enviada" pelo painel.
3. Marcar uma empresa como "Não contatar" e repetir a pesquisa: ela não volta como contatável.
4. (Opcional) Importar a base da Receita para uma UF pequena (ex.: AC) e pesquisar com a fonte "Base aberta de
   CNPJ".

## 6. Pendências e o que NÃO foi verificado

| Item | Situação |
|---|---|
| Overpass (OpenStreetMap) ao vivo | `overpass-api.de` recusou conexões do ambiente de nuvem do desenvolvimento. A consulta gerada foi aceita por outro servidor Overpass (200, JSON válido). **Precisa ser confirmada no PC do usuário.** |
| Google Places ao vivo | Sem chave disponível. Implementado pela documentação oficial (Places API New, Text Search e Place Details, com máscara de campos) e testado com respostas gravadas. **Não afirmo que esteja funcionando com uma chave real até alguém testar.** |
| IA (API da Anthropic) ao vivo | Sem chave. Chamada HTTP direta (o `.exe` não pode ganhar o SDK). Opcional e desligada por padrão. |
| BrasilAPI e IBGE ao vivo | **Verificados** a partir do ambiente de desenvolvimento com o código do módulo (consulta de CNPJ público e lista de 645 municípios de SP). |
| Importação da Receita em arquivos reais | Testada com ZIP sintético no formato oficial (CSV `;`, latin-1, sem cabeçalho, 30 colunas). Não foi rodada nos arquivos completos (vários GB). O endereço de download da Receita muda com frequência; o módulo abre a página do dataset no dados.gov.br e o usuário baixa os arquivos. |
| `.exe` | Não gerado nesta sessão (sem Windows). O módulo só usa PySide6 e biblioteca padrão. |
| Provedores comerciais de CNPJ | Avaliados e documentados; não implementados (exigem contrato). |

## 7. Decisões que valem revisar

- **Prazos do Google:** interpretação conservadora dos termos do Google Maps Platform (guarda máxima de 30 dias,
  sem exportação, `place_id` permanente). Vale revisão jurídica antes de uso comercial amplo.
- **MEI/empresário individual** fora da busca no CNPJ por padrão (privacidade). A opção para incluir está na aba
  Pesquisar.
- **Frase de não contato** ativada por padrão nas mensagens.
- **Modelo de IA padrão:** `claude-opus-5-5` (configurável em Configurações; esforço baixo para mensagens curtas).
