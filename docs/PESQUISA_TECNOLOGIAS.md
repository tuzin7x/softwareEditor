# Pesquisa e seleção de tecnologias

Síntese dos 7 relatórios de pesquisa (fontes primárias: código-fonte, LICENSE, cartões do Hugging Face
e documentação oficial). Os relatórios completos estão em [`pesquisa_bruta/`](pesquisa_bruta/), em
inglês, com todas as URLs. Data de referência: 15/11/2026.

> A etapa de "crítico" do processo de pesquisa foi interrompida pela pausa de uso. Por isso esta
> síntese marca o que **foi validado na prática neste repositório** (✅) e o que é **recomendação
> ainda não validada** (🔎).

## 1. Resumo executivo

| Área | Escolha (implementada) | Alternativa | Licença | Motivo |
|---|---|---|---|---|
| Geometria/câmera | **MoGe-2 ViT-L-normal** ✅ (ViT-B em CPU/pouca VRAM) | DepthMaster; DA3-Metric como 2º voto de escala | MIT (código e pesos) | Pontos métricos + FOV + normais em uma passada; ~1–2 GB de VRAM; sem extensões CUDA (roda no Windows) |
| Segmentação | **OneFormer ADE20K Swin-L** ✅ | **EoMT-L** 🔎 (58,4 mIoU, mais rápido); SAM 2.1 para bordas | MIT | Cobre parede/piso/teto/janela/porta/instalações. SegFormer foi descartado (licença NVIDIA não comercial) |
| Estrutura da sala | RANSAC piso + retas 2D das paredes (Manhattan) + janelas/portas 3D ✅ | GeoCalib (gravidade/distorção) 🔎 | código próprio | Determinístico, depurável, não inventa estrutura invisível |
| Luz | Inversa ao estilo Karsch: janelas + ambiente + **sol detectado**, por NNLS sobre renders da sala-proxy ✅ | DiffusionLight-Turbo para reflexos 🔎 | código próprio | Exposição e branco coerentes com a foto, sem difusão sobre a foto |
| Render 3D | **Blender Cycles** (bpy 5.3 em subprocesso), OptiX → CUDA → CPU ✅ | Blender 5.2 LTS portátil 🔎 | GPL (processo separado) | Shadow catcher físico, oclusão, motion blur, OIDN |
| Composição | numpy próprio: `C + (1−A)·foto·S`, cópia bit a bit fora da área tocada ✅ | — | — | É o que garante a regra absoluta (verificada em cada quadro) |
| Assets 3D | **Poly Haven** + procedurais próprios ✅ | ABO (CC BY 4.0), Infinigen Indoors 🔎 | CC0 / próprio | Comercialmente limpo e fotorrealista |
| Layout | Solver próprio: slots + recozimento simulado + regras de design + visibilidade na câmera ✅ | Qwen3-VL-4B (Apache) só para ranquear 🔎 | código próprio | Determinístico, segundos em CPU, otimiza o que aparece *nesta* foto |
| Animação | Um estilo, "posicionar e acomodar" em camadas, sobreposição de 45–60%, motion blur 180° ✅ | — | — | Confirmado pela pesquisa de motion design |
| Vídeo | **libx264** CRF 14, BT.709 explícito ✅ | h264_nvenc (p7, hq, CQ) | GPL (FFmpeg separado) | Mais estável e menor; o encode leva segundos perto do render |
| Interface/pacote | **PySide6 + PyInstaller onedir** (só a UI) + Inno Setup por usuário + motor instalado no 1º uso ✅ | Camada GPU pré-montada e assinada (7z em partes) 🔎 | LGPL (Qt) | A UI abre rápido e não segura VRAM; o motor fica isolado |
| GPU/PyTorch | torch **2.15.0+cu132** (RTX 20/30/40/50, driver ≥ 580) ✅ wheel confirmado | cu130/cu126 (2.14.1) para drivers antigos | BSD | Cobre Blackwell (série 50) |

## 2. Decisões por área (resumo dos relatórios)

**Geometria** ([`geometria.md`](pesquisa_bruta/geometria.md)). MoGe-2 confirmado. Em testes independentes do
pesquisador, portas ficaram a menos de 3% do nominal e o ângulo piso⟂parede deu 89,4–89,9°. Melhorias
recomendadas, ainda não feitas:
- arbitragem de FOV/escala com um segundo modelo e um índice de confiança;
- **distorção de lente aplicada só às camadas 3D**, nunca à foto.

O MoGe-3 existe (MIT), mas depende de Triton/flex_gemm, sem garantia no Windows.

**Segmentação, layout da sala e luz** ([`segmentacao_layout_luz.md`](pesquisa_bruta/segmentacao_layout_luz.md)).
- EoMT-L seria a evolução natural do OneFormer: mesmo contrato e mais rápido.
- SAM 3 tem licença própria e precisa de aprovação jurídica.
- A luz por NNLS com sol detectado é exatamente o que foi implementado.

**Cycles** ([`blender_cycles.md`](pesquisa_bruta/blender_cycles.md)). A leitura do código-fonte confirmou
todos os comportamentos que medimos:
- `Combined` contém só o CG, pré-multiplicado;
- o passe SC é uma razão;
- catcher emissivo dilui a sombra;
- luzes criadas via Python contam como "reais";
- catchers ocultam o CG que fica atrás deles.

Truque recomendado para depois: catcher difuso para a câmera e emissivo (a foto) para os demais raios,
para os móveis refletirem o ambiente real.

**Assets** ([`assets_3d.md`](pesquisa_bruta/assets_3d.md)).
- Base: Poly Haven CC0. Peças contemporâneas: ABO (CC BY 4.0, exige créditos).
- Geração 3D por IA local fica como P&D: TRELLIS.2 exige ≥ 24 GB de VRAM, e Hunyuan3D tem restrição
  territorial de licença.

**Layout automático** ([`layout_automatico.md`](pesquisa_bruta/layout_automatico.md)). O solver
implementado segue a recomendação: slots junto às paredes, recozimento simulado e termos de staging
calculados pela projeção na câmera. Nenhum LLM gera coordenadas.

**Empacotamento** ([`empacotamento_windows.md`](pesquisa_bruta/empacotamento_windows.md)). A recomendação
de produção é uma **camada GPU pré-montada, com hash e assinada**, sem resolver dependências na máquina do
cliente. O protótipo atual usa a alternativa documentada pelo próprio relatório: instalação via pip no
primeiro uso, retomável. É mais simples, mas depende de internet e tem DLLs não assinadas.

**Animação, integração e encode** ([`animacao_realismo_encode.md`](pesquisa_bruta/animacao_realismo_encode.md)).
- Já aplicado: ordem em camadas, ritmo acelerando no fim, grão fixo na tela, desfoque casado,
  BT.709 explícito.
- Pendente de ajuste: um único IDR com teto de VBV (24 Mbps) no x264 (hoje o keyint é 4 s), hold final
  de ~3,6 s a 30 fps, e um master RGB sem perdas para auditoria.

## 3. Tecnologias consideradas e descartadas

| Tecnologia | Por que não |
|---|---|
| **ComfyUI / difusão de imagem (inpainting)** | Regenera pixels da foto: viola a regra absoluta. Também não dá oclusão nem perspectiva 3D consistentes entre quadros. |
| **Vídeo generativo** (Wan, HunyuanVideo, LTX) | Não preserva a foto, a câmera "respira" e o resultado é imprevisível. Pesado em VRAM. |
| **"Antes e depois" / slides** | Proibido pelo pedido. |
| SegFormer, Depth Anything V2-L | Licenças não comerciais. |
| Depth Pro (Apple) | Pesos com termos de pesquisa. |
| ABO/3D-FUTURE no PoC | Licenças com atribuição ou não comerciais. Ficam para avaliação jurídica. |
| PyInstaller com torch embutido | Instalador de vários GB, acima do limite de anexos, e reempacotar a cada atualização. |

## 4. Riscos registrados

- Escala monocular (±10–20%), mitigada por pistas.
- Distorção de lente ainda não modelada.
- Driver NVIDIA abaixo de 580 não roda torch cu132: precisa de aviso e de uma variante legada.
- NVENC do FFmpeg 9 exige driver ≥ 610; o padrão é libx264.
- GPL do Blender e do FFmpeg: ambos rodam como processos separados, e a origem do código precisa ser
  publicada junto.
- Os tempos de render em GPU são estimativas: não houve GPU no desenvolvimento.
