# Limitações conhecidas (estado atual da prova de conceito)

Documento honesto do que **já funciona**, do que foi **validado** e do que **ainda não** está resolvido.
Atualizado conforme os testes reais avançam.

## 1. Ambiente de desenvolvimento x seu computador

- **Seu hardware não pôde ser analisado diretamente.** O desenvolvimento rodou num contêiner Linux na
  nuvem, **sem GPU**, e nenhum computador seu estava conectado à sessão. Por isso:
  - o software detecta GPU, VRAM, driver e capacidade de computação **na sua máquina**, a cada
    execução (`python -m mobilia.cli hardware`), e escolhe o perfil (ver `ARQUITETURA.md`, seção 3);
  - `scripts/diagnostico_hardware.ps1` gera um relatório no Windows sem instalar nada.
- **Todos os renders desta prova de conceito foram feitos em CPU**, com resolução e amostras reduzidas
  (960×720, 16–48 amostras). Numa RTX com OptiX, a mesma cena roda em 1920 px com 128 amostras. Os
  tempos citados para GPU são **estimativas**, não medições.
- **O `.exe` não foi executado num Windows real com GPU NVIDIA** por mim. Ele é gerado pela CI do
  GitHub (Windows) e tem um autoteste de interface. O primeiro teste completo com GPU precisa ser
  feito na sua máquina.

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
  Na foto de teste, o erro relativo do ajuste fica em 20–29%. A exposição e o balanço de branco dos
  móveis ficam coerentes, mas não é uma reconstrução física exata da luz.
- **Luminárias acesas na foto**, luz colorida forte e fotos noturnas não são modeladas. As luminárias
  adicionadas ficam **apagadas** de propósito: acendê-las alteraria a iluminação da foto, o que a regra
  absoluta proíbe.
- **Sombras:** sombras suaves além de ~10% da diagonal da imagem em volta dos móveis são cortadas com
  decaimento suave, para preservar a foto. Desvios menores que 0,6% (ruído de render) são ignorados.
- Grão do sensor e nitidez são casados de forma aproximada. Aberração cromática e vinheta não são
  replicadas.

## 4. Móveis (assets 3D)

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

- No modo CPU (sem GPU NVIDIA) o render leva de dezenas de segundos a minutos por quadro.
- Na GPU, cerca de 70–85% do tempo é render. As otimizações descritas em `ARQUITETURA.md` (border por
  quadro, reaproveitamento de quadros estáticos, *hold* sem render) reduzem o número de quadros
  renderizados para algo em torno de 60–75% do total.
