# Changelog

Todas as mudanças importantes do **7xAi** ficam registradas aqui.

O formato segue o [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e as versões seguem o
[Versionamento Semântico](https://semver.org/lang/pt-BR/) (`MAIOR.MENOR.CORREÇÃO`). Como publicar uma versão:
[`docs/RELEASES.md`](docs/RELEASES.md).

Cada seção de versão usa estes títulos (os que não tiverem itens podem ficar de fora):
`### Novidades`, `### Correções`, `### Desempenho`, `### Notas importantes de atualização`,
e, quando houver, `### Removido` e `### Segurança`. O texto da seção vira as notas da Release.

## [Não publicado]

## [0.9.0] - 2026-11-19

### Novidades

- **AI Studio — redesenho de ambientes por IA** a partir de uma única foto, com até 4 versões por pedido:
  - **Mudar estilo:** mesma sala e mesma arquitetura, com móveis e decoração em outro estilo;
  - **Reforma completa:** troca também pisos, revestimentos e acabamentos;
  - **Redesign simples:** mudança leve, que mantém a sala reconhecível;
  - **Mobiliar ambiente vazio:** mobília completa numa sala vazia;
  - **Editar área:** o usuário pinta uma região e só ela muda;
  - **Fachada:** redesenho da parte externa do imóvel;
  - **Decoração de evento:** aniversário, casamento, Natal, festa infantil, chá de bebê e réveillon;
  - **Texto para ambiente:** cria um ambiente a partir de uma descrição, sem foto.
- 14 estilos (moderno, contemporâneo, escandinavo, minimalista, industrial, clássico, luxo, boho, japandi, rústico,
  costeiro, tropical brasileiro, mid-century e fazenda moderna) e 8 tipos de ambiente.
- Janelas e portas originais são recolocadas por cima da imagem gerada, e um controle automático descarta as
  versões fracas.
- **Ampliar 4K:** a versão escolhida sai com 3840 px.
- **Vídeo antes e depois:** a foto original se transforma nas versões novas, com movimento de câmera lento, em
  16:9 (1920×1080) ou 9:16 (1080×1920, para Reels e Stories).
- **Modos legados de vídeo de montagem** (botão "Avançado"): modo A (móveis 3D entram numa foto vazia) e modo B
  (a foto mobiliada vira sala vazia e os móveis da própria foto entram um a um).
- **Lead Finder** (nova aba): pesquisa de imobiliárias em fontes autorizadas (OpenStreetMap/Overpass, base aberta
  de CNPJ da Receita importada localmente, BrasilAPI e, com chave própria do usuário, Google Places API), CRM local
  (SQLite) com histórico, lembretes e lista de "não contatar", pontuação de 0 a 100, painel, exportação CSV e
  mensagem revisável aberta no WhatsApp (`wa.me`), **sem envio automático**.
- Janela principal com módulos (abas de primeiro nível) registrados automaticamente; o selo "Processado no seu
  computador" só aparece nas abas que não usam a internet.
- Instalador para Windows com atalho na Área de Trabalho e versão portátil de arquivo único. O motor de IA é
  baixado e instalado no primeiro uso.
- Releases oficiais com instalador, portátil, `SHA256SUMS.txt` e notas da versão; versão com fonte única
  (`mobilia/__init__.py`).
- **Atualização automática dentro do app** (gratuita, pelo GitHub Releases): ao abrir, o 7xAi procura em segundo
  plano (no máximo uma vez por dia) a última versão estável e mostra "Nova versão X.Y.Z disponível" com as notas,
  e os botões **Atualizar agora**, **Depois** e **Pular esta versão**. "Atualizar agora" baixa o instalador oficial
  com barra de progresso e opção de cancelar, confere o SHA-256 com o `SHA256SUMS.txt`, instala em silêncio e
  reabre o 7xAi. No portátil, baixa o novo `.exe` (conferido) ao lado do atual. Botão **Atualizações** na barra de
  status: procurar agora, ligar/desligar a consulta automática e ver todas as versões.

### Desempenho

- Cerca de 8 a 11 s por imagem numa RTX 5070 Laptop de 8 GB, medidos num teste isolado do motor (codificadores de
  texto na CPU para caber em 8 GB de memória de vídeo). O app completo ainda não foi medido no Windows.
- Executável leve (só a interface); o motor pesado fica fora do instalador e é reaproveitado entre atualizações.
- Com os modelos já baixados, o MoGe e o OneFormer carregam só do disco (sem consultar a internet a cada trabalho):
  numa rede com IPv6 lento cada consulta custava ~42 s (um trabalho caiu de 201 s para 88 s no teste).
- Conexões HTTPS do app (Lead Finder, atualização, downloads) tentam IPv4 primeiro, com tempo curto por endereço.
- Plano de memória calibrado com o pico medido numa placa de 8 GB (5,8 GB na geração a 1024 px): sem offload para
  a RAM; refino leve (até 1280 px, sem CFG) só quando a estimativa cabe.

### Notas importantes de atualização

- Requisitos: Windows 10 ou 11 de 64 bits. Placa NVIDIA com 8 GB de memória de vídeo recomendada (sem GPU NVIDIA
  funciona em modo CPU, muito mais lento).
- No primeiro uso o 7xAi baixa e instala o motor de IA uma única vez (~12 GB: bibliotecas com CUDA e ~8 GB de
  modelos, dos quais ~7,5 GB são do redesenho) em `%LOCALAPPDATA%\MobiliaStudio`. Quem já tinha uma versão de
  teste instalada reaproveita esse motor.
- O AI Studio roda no computador do usuário; as fotos não são enviadas para a internet. O Lead Finder consulta
  serviços on-line (e o Google Places é pago, com chave e limites definidos pelo usuário).
- O instalador ainda não tem assinatura digital paga: o Windows pode mostrar "O Windows protegeu o computador".
  Clique em **Mais informações → Executar assim mesmo**.
- **Limitações conhecidas desta versão:** "Editar área" ainda pode deixar restos do objeto antigo ou uma mancha no piso; a árvore em "Decoração de evento → Natal" nem sempre aparece; "Mobiliar" às vezes deixa um móvel cortado na borda; numa placa de 8 GB a passada de detalhe fica desligada (imagens a 1024 px, use "Ampliar 4K"); a busca no OpenStreetMap depende de servidores públicos e pode demorar ou falhar em parte. Google Places e a variação de mensagem por IA foram testados só com respostas gravadas. Correções chegam pela atualização automática.
- A atualização só usa HTTPS e hosts do GitHub, exige o SHA-256 publicado e nunca toca nos dados do usuário
  (`%LOCALAPPDATA%\MobiliaStudio`: motor, modelos, trabalhos, Lead Finder). Pré-lançamentos nunca são oferecidos.
- O instalador ganhou o parâmetro `/AUTOUPDATE` (usado só pela atualização automática) para reabrir o 7xAi no fim.
  A primeira versão com este recurso ainda precisa ser instalada à mão; as seguintes chegam pelo próprio app.

### Correções

- **Editar área**: só o objeto antigo (segmentado, crescido ~1%) sai, nunca o móvel vizinho que o retângulo pegou
  (o braço do sofá ficava borrado); a geração parte da foto sem o objeto antigo, não de um borrão da área inteira; o
  objeto novo entra opaco (sem a "mesa translúcida") e pedacinhos soltos não são colados.
- **Natal**: a árvore ganhou lugar no prompt (num canto, diante de parede lisa, repetida no fim) e, na GPU, o Natal
  gera 2 candidatos a mais e escolhe os que têm árvore.
- **Mobiliar**: pedaço de móvel cortado pela borda da foto perde pontos na escolha entre candidatos (e entrou no
  negativo).
- **Lead Finder (OpenStreetMap)**: a consulta da cidade vai em pedaços pequenos e troca sozinha entre os servidores
  públicos da Overpass quando um falha (502/504/403); a mensagem não manda mais "conferir a chave" (a Overpass não
  tem chave) e o User-Agent é descritivo.
