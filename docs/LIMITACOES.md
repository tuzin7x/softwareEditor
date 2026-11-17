# Limitações conhecidas (estado atual da prova de conceito)

Documento honesto do que **já funciona**, do que foi **validado** e do que **ainda não** está resolvido.
Atualizado conforme os testes reais avançam.

## 1. Ambiente de desenvolvimento x seu computador

- **Seu hardware não pôde ser analisado diretamente.** O desenvolvimento rodou num contêiner Linux na
  nuvem, **sem GPU**, e nenhum computador seu estava conectado à sessão. Por isso:
  - o software detecta GPU, VRAM, driver e capacidade de computação **na sua máquina**, a cada
    execução (`python -m mobilia.cli hardware`), e escolhe o perfil (ver `ARQUITETURA.md`, seção 3);
  - `scripts/diagnostico_hardware.ps1` gera um relatório no Windows sem instalar nada.
- **Todos os renders das provas de conceito foram feitos em CPU**, com resolução e amostras reduzidas
  (960×720; na PoC V2, 24 amostras no quadro final e nos marcos e 12 na camada dinâmica). Numa RTX com
  OptiX, a mesma cena roda em 1920 px com 128/64/12–32 amostras por camada. Os tempos citados para GPU
  são **estimativas**, não medições (premissa em `docs/poc/v2/RESULTADO.md`).
- **O `.exe` não foi executado num Windows real com GPU NVIDIA** por mim.
  - O build do PyInstaller e o autoteste da interface foram validados em Linux.
  - O workflow de build Windows está pronto, mas os jobs do GitHub Actions deste repositório
    **não recebem runner**: falham em 3 s, sem logs. É preciso habilitar ou liberar o Actions na
    conta (Settings › Actions, Billing).
  - Alternativa: gerar localmente com `packaging\build_windows.ps1`.
  - O primeiro teste completo com GPU precisa ser feito na sua máquina.

## 2. Análise da foto

- **Escala métrica:** profundidade monocular tem incerteza típica de 10–20%. Corrigimos com pistas
  (porta ≈ 2,05 m, pé-direito plausível), mas uma foto sem essas pistas pode gerar móveis um pouco
  grandes ou pequenos demais.
- **Distorção de lente:** fotos de imóveis em grande-angular costumam ter distorção de barril. O MoGe-2
  assume câmera *pinhole*, e a foto **não pode** ser retificada (regra absoluta). Em fotos com
  distorção forte, bordas retas de móveis próximos às laterais podem não acompanhar a curvatura das
  paredes. Melhoria planejada: aplicar a mesma distorção à camada 3D.
- **Segmentação:** pisos muito escuros, espelhos, paredes de vidro e cômodos com paredes curvas são
  aproximados por planos e podem falhar. A grade de ocupação marca obstáculos existentes (bancadas,
  colunas, radiadores) como bloqueados, e o layout evita essas áreas.
- **Oclusão por objetos que já existem na foto** (ilha de cozinha, folha de porta aberta, colunas): as
  paredes ocultam corretamente o que fica atrás delas. Objetos não planares ainda **não** recortam os
  móveis 3D. Por isso o layout não posiciona móveis atrás deles. Melhoria planejada: malha de
  profundidade como *holdout*.

## 3. Luz e realismo

- O modelo de luz é **simplificado**: luzes de área nas janelas, luz ambiente da parte da sala que não
  aparece e **sol direto** quando há manchas de sol no piso. As intensidades são calibradas pela foto.
  A sala 3D emite a própria foto para os raios indiretos (PT8). Na foto de teste, o erro relativo do
  ajuste é de 18%, e 12% depois do mapa de ganho do CG (PoC V2). A exposição e o balanço de branco dos
  móveis ficam coerentes, mas não é uma reconstrução física exata da luz.
- **Luminárias acesas na foto**, luz colorida forte e fotos noturnas não são modeladas. As luminárias
  adicionadas ficam **apagadas** de propósito: acendê-las alteraria a iluminação da foto, o que a regra
  absoluta proíbe.
- **Sombras:** sombras suaves além de ~10% da diagonal da imagem em volta dos móveis são cortadas com
  decaimento suave, para preservar a foto. Desvios menores que 0,6% (ruído de render) são ignorados.
- Grão do sensor e nitidez são casados de forma aproximada. Aberração cromática e vinheta não são
  replicadas.

## 4. Móveis (assets 3D)

- O tampo de vidro de mesas sai como um cinza opaco: vidro transparente sobre a foto
  (`film_transparent_glass`) continua desligado.
- Nos 2–3 primeiros quadros de uma queda, a peça em fade-in escurece um pouco o que fica atrás dela
  (sombra parcial no passe de *shadow catcher*). O efeito é breve e faz parte da dissolução.
- Quadros de parede são pinturas abstratas geradas localmente. Num par, as duas usam o mesmo estilo
  e as composições ficam parecidas.

- O catálogo CC0 da Poly Haven tem cerca de 50 itens úteis para interiores, com viés para estilo
  clássico/vintage e poucos sofás modernos. Tapete, luminárias, cama, mesa de escritório, cortina e
  manta são **procedurais** (gerados no Blender).
- Uma biblioteca comercial maior exigiria assets licenciados ou geração 3D por IA local (TRELLIS.2,
  Hunyuan3D etc.), com ressalvas de licença descritas na pesquisa.

## 5. Vídeo

- O MP4 (H.264 yuv420p) **tem perda**. A garantia bit a bit vale para os quadros mestres compostos e
  é verificada antes do encode. O relatório mede o PSNR do vídeo decodificado (fundo intocado).
- Fotos maiores que a resolução do perfil são reduzidas **uma vez** (ex.: 4032×3024 → 1920×1440) antes
  de virar a placa do vídeo. É reamostragem, não edição, e fica registrada no `report.json`.

## 6. Desempenho

- **Medido na PoC V2** (CPU de 4 threads, sem GPU, 960×720, 251 quadros a 24 fps, render em camadas):
  - análise 21 s com os modelos em cache de disco, layout 3 s;
  - render 1 141 s (≈ 19 min): quadro final 31 s, 4 marcos 102 s, 164 renders dinâmicos 1 008 s;
  - composição em paralelo ao render, com cauda de ~1 s depois dele; total 1 175 s.
  - Na PoC V1, a mesma foto levou 6 768 s só de render.
- Na GPU (RTX 5070 Laptop, 1920×1440), a estimativa é de ≈ 105 s do clique ao MP4. O controle de
  orçamento corta amostras e escala do CG em movimento se a GPU medida for mais lenta. Esses tempos
  continuam **estimativas** até chegar o `report.json` da máquina do usuário.
- A primeira execução depois de instalar paga compilação de kernels e carga dos modelos. O instalador
  já faz esse aquecimento (PT2), e o relatório mostra esse custo à parte.
- Ajustes só de composição não exigem novo render: `python -m mobilia.cli recompor --work-dir <pasta>`
  refaz o MP4 a partir dos EXRs. Por padrão os EXRs são apagados; para isso, gere o trabalho com
  `MOBILIA_MANTER_EXR=1`.

## 7. Defeitos encontrados na PoC e corrigidos

A primeira rodada completa passou na verificação automática (foto preservada bit a bit), mas a
inspeção visual dos quadros encontrou três defeitos. Todos foram corrigidos, ganharam teste de
regressão e não aparecem na rodada final:

1. **Linha escura na parede:** uma linha fora da região renderizada pelo Cycles (arredondamento do
   *border*) era lida como sombra total. Agora pixels não renderizados contam como "sem sombra".
2. **Degrau de tom no limite da região renderizada:** a sombra suave era cortada de forma abrupta. Agora
   ela se dissolve numa rampa suave, e a região ganhou margem vertical.
3. **Quadro cortado na quina:** a parede 3D invisível que recebe sombras era estendida 0,35 m além da
   ponta e, numa quina oblíqua, passava na frente da outra parede. A extensão agora para antes de
   cruzar outra parede.

Lição registrada: a verificação bit a bit garante a regra absoluta, mas **não** detecta defeitos
dentro da área tocada pelos móveis. A inspeção visual dos quadros continua necessária.

Na PoC V2, a inspeção encontrou mais um defeito: com o fade-in, o sofá passava do limite de 8
superfícies transparentes do Cycles, e no primeiro quadro da queda aparecia uma mancha opaca. O limite
agora sobe só nos quadros com fade (teste `tests/test_fade_bounces.py`). Detalhes em
`docs/poc/v2/RESULTADO.md`.

## 8. Modo B (foto mobiliada → sala vazia → móveis reais voltando), sem SDXL

Estado em 17/11/2026, depois da avaliação final na CPU (`docs/poc/v3/RESULTADO.md`, vídeos e folha de contato
no mesmo diretório). Notas do crítico: sala vazia 2/10, móveis 6–7/10, animação 5/10, **geral 3/10**.

- **A sala vazia é o gargalo.** O big-lama por plano produz névoa, fantasmas de móveis e quadros, piso com material
  errado (sob tapetes grandes, o piso de verdade não aparece em lugar nenhum) e pedaços de parede "soltos". O
  portão de qualidade reprova a placa das 2 fotos de teste e de 33 das 39 fotos mobiliadas da rodada de robustez.
  **Por padrão, o modo B quase nunca entrega vídeo:** o usuário recebe o aviso e pode gerar mesmo assim. A GPU
  deixa tudo mais rápido, mas não melhora a placa; a esperança é o refino SDXL (PT-V3-4), ainda não feito.
- **Peças encostadas com a mesma profundidade** (cadeira na mesa de jantar) podem ficar sem relação de oclusão. A
  peça de trás entra antes, com o buraco da da frente, e a verificação automática não acusa (ela só confere
  recortes contra peças marcadas como da frente). Visto na foto de Nice; a inspeção visual continua necessária.
- **Tapete coberto por móveis desenrola depois deles**, por baixo (a ordem é de frente para trás para nunca mostrar
  parte escondida do tapete). Fica estranho, mas não mostra nada inventado.
- Quando não há movimento seguro, a peça **surge de uma vez** (sem fade). Na foto típica, isso acontece com o sofá
  principal.
- A **nota do portão** separa mal: todas as reprovadas saem com 59 e uma placa ruim pode ter nota sem teto 90.
- Exatidão: o último quadro é a foto **na resolução do vídeo** (byte a byte nos quadros mestres e no MKV de
  auditoria); o MP4 tem perda (PSNR 47–50 dB medido). Fotos menores que a resolução do perfil não são ampliadas
  pelo produto (a demonstração de 1440 px ampliou de propósito fotos de 960 e 1280 px).
- **Tempo medido na CPU** (4 núcleos, 1440 px): 156–163 s do clique ao MP4, 70–75% na sala vazia (3 tentativas do
  portão). Na RTX 5070 Laptop, a estimativa é de 20–40 s; **não medido**.
- Corrigido nesta rodada: com a tentativa "máscara ampla" do portão, o contorno dos móveis (pés das cadeiras) voltava
  da foto mobiliada para a sala vazia (4,9% da imagem na Nice) sem que o portão visse.
