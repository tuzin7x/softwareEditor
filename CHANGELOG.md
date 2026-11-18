# Changelog

Todas as mudanças importantes do **7xAi** ficam registradas aqui.

O formato segue o [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e as versões seguem o
[Versionamento Semântico](https://semver.org/lang/pt-BR/) (`MAIOR.MENOR.CORREÇÃO`). Como publicar uma versão:
[`docs/RELEASES.md`](docs/RELEASES.md).

Cada seção de versão usa estes títulos (os que não tiverem itens podem ficar de fora):
`### Novidades`, `### Correções`, `### Desempenho`, `### Notas importantes de atualização`,
e, quando houver, `### Removido` e `### Segurança`. O texto da seção vira as notas da Release.

## [Não publicado]

### Novidades

- Sistema de Releases oficiais: cada versão aprovada ganha uma página em *Releases* com o instalador
  (`7xAI-Setup-vX.Y.Z.exe`), o portátil (`7xAI-Portable-vX.Y.Z.exe`), o `SHA256SUMS.txt` e as notas da versão.
- A versão do programa tem uma fonte única (`mobilia/__init__.py`), usada pelo executável, pelo instalador e pela
  tag da Release.

### Removido

- Publicação de *prereleases* por `[release]` na mensagem do commit ou pelo disparo manual "publicar": builds de
  desenvolvimento agora geram só artefatos no GitHub Actions.

## [1.0.0] - RASCUNHO (aguardando aprovação do dono)

> **RASCUNHO — não publicar ainda.** Este texto descreve o produto como está hoje e precisa ser revisado e aprovado
> pelo dono. Para publicar: ajuste o texto, troque o título por `## [1.0.0] - AAAA-MM-DD` (data da publicação),
> apague este aviso e mude `__version__` para `"1.0.0"` em `mobilia/__init__.py`, tudo no mesmo PR para a `main`.
> O workflow de Release se recusa a publicar enquanto a palavra RASCUNHO estiver nesta seção.

### Novidades

- **Redesenho de ambientes por IA**, a partir de uma única foto, com até 4 versões por pedido:
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
- Instalador para Windows com atalho na Área de Trabalho e versão portátil de arquivo único. O motor de IA é
  baixado e instalado no primeiro uso.
- Ponto de extensão para módulos opcionais na janela principal (abas de primeiro nível).
- **7xAi Lead Finder** (módulo de prospecção de imobiliárias): **pendente de integração**. Só entra nesta lista se
  o Pull Request #1 for integrado antes da publicação; caso contrário, remova este item.

### Desempenho

- Cerca de 8 a 11 s por imagem numa RTX 5070 Laptop de 8 GB, com os codificadores de texto na CPU para caber em
  8 GB de memória de vídeo.
- Executável leve (só a interface); o motor pesado fica fora do instalador e é reaproveitado entre atualizações.

### Notas importantes de atualização

- Requisitos: Windows 10 ou 11 de 64 bits. Placa NVIDIA com 8 GB de memória de vídeo recomendada.
- No primeiro uso o 7xAi baixa o motor de IA (vários GB, uma única vez) para `%LOCALAPPDATA%\MobiliaStudio`.
  Quem já tinha uma versão de teste instalada reaproveita esse motor.
- O instalador ainda não tem assinatura digital paga: o Windows pode mostrar "O Windows protegeu o computador".
  Clique em **Mais informações → Executar assim mesmo**.
- Tudo roda no computador do usuário; as fotos não são enviadas para a internet.
