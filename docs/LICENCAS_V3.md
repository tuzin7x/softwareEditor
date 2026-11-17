# 7xAi V3: licenças dos modelos e rascunho do EULA

Data da conferência: 17/11/2026. Complementa o [`PLANO_V3.md`](PLANO_V3.md) §5.

**Como foi conferido [V]:**
- Hugging Face: campo `cardData.license` e `gated` da API pública `https://huggingface.co/api/models/<repo>`;
- GitHub: arquivo `LICENSE` do ramo `main` do repositório oficial (cabeçalho lido).

Todos os modelos abaixo têm **uso comercial permitido** e **nenhum é gated** (o download anônimo funciona
no cliente). Uma licença pode mudar: reconfira antes de cada versão do instalador.

> Este documento não é parecer jurídico. O texto final do EULA deve ser revisado por um advogado antes da venda.

## 1. Modelos permitidos

### 1.1 Modo B sem SDXL (entrega atual) e modo A atual

| Função | Repositório dos pesos | Licença [V] | Link conferido | Código de origem (LICENSE [V]) |
|---|---|---|---|---|
| Geometria (já usado) | `Ruicheng/moge-2-vitl-normal` | MIT | https://huggingface.co/Ruicheng/moge-2-vitl-normal | https://github.com/microsoft/MoGe (MIT, "Copyright (c) Microsoft Corporation") |
| Geometria, perfil leve (já usado) | `Ruicheng/moge-2-vitb-normal` | MIT | https://huggingface.co/Ruicheng/moge-2-vitb-normal | mesmo código do MoGe (MIT) |
| Segmentação semântica e panóptica (já usado) | `shi-labs/oneformer_ade20k_swin_large` | MIT | https://huggingface.co/shi-labs/oneformer_ade20k_swin_large | https://github.com/SHI-Labs/OneFormer (MIT, "Copyright (c) 2022 SHI Labs") |
| Máscara por peça | `facebook/sam2.1-hiera-base-plus` | Apache-2.0 | https://huggingface.co/facebook/sam2.1-hiera-base-plus | https://github.com/facebookresearch/sam2 (Apache-2.0) |
| Sala vazia (inpainting) | big-lama (TorchScript), espelho `smartywu/big-lama` | Apache-2.0 | https://huggingface.co/smartywu/big-lama | https://github.com/advimman/lama (Apache-2.0) |
| Bordas finas (opcional) | `ZhengPeng7/BiRefNet` | MIT | https://huggingface.co/ZhengPeng7/BiRefNet | https://github.com/ZhengPeng7/BiRefNet (MIT, "Copyright (c) 2024 ZhengPeng") |
| Objetos pequenos (V3.1) | `IDEA-Research/grounding-dino-tiny` | Apache-2.0 | https://huggingface.co/IDEA-Research/grounding-dino-tiny | https://github.com/IDEA-Research/GroundingDINO (Apache-2.0) |

Observações:
- O big-lama do protótipo veio de um *release* do GitHub de terceiros (pacote `Sanster/models`). No produto, o
  arquivo vem do **espelho próprio** com SHA-256 (PT-V3-1), e a licença aplicável é a Apache-2.0 do LaMa.
- O OneFormer e o LaMa foram treinados em ADE20K e Places2 (datasets de pesquisa). O risco residual é baixo e
  o produto atual já o carrega (PLANO_V3 §5.1).

### 1.2 Refino SDXL e modo A novo (depois, PT-V3-4 e PT-V3-9)

| Função | Repositório | Licença [V] | Link conferido |
|---|---|---|---|
| Gerador (um único UNet) | `SG161222/RealVisXL_V5.0_Lightning` | OpenRAIL++ | https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning |
| Gerador (alternativa) | `stabilityai/stable-diffusion-xl-base-1.0` | OpenRAIL++ | https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 |
| LoRA de passos | `ByteDance/SDXL-Lightning` | OpenRAIL++ | https://huggingface.co/ByteDance/SDXL-Lightning |
| VAE | `madebyollin/sdxl-vae-fp16-fix` | MIT | https://huggingface.co/madebyollin/sdxl-vae-fp16-fix |
| ControlNet de profundidade | `diffusers/controlnet-depth-sdxl-1.0-small` | OpenRAIL++ | https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0-small |

Nenhum desses pesos entra no pacote enquanto o modo B sem SDXL for o entregável. Os *text encoders* não vão
para o cliente (embeddings pré-calculados no build).

## 2. Proibidos (não comerciais ou sem licença clara)

Não usar, nem como fallback: tudo derivado do FLUX.1-dev (Fill, Kontext, Redux, OmniEraser, Insert Anything,
UNO, In-Context-LoRA, Hyper-FLUX), FLUX.2-klein-9B, SDXL-Turbo, DMD2, pix2gestalt, LBM relighting,
IC-Light V2, RMBG-2.0, MAT, MatAnyone, ObjectClear, SV3D, Stable Zero123, Depth-Anything-V2-Large,
AnyDoor e Zero123++ (sem licença) e TRELLIS (depende de nvdiffrast). Lista completa e motivos: PLANO_V3 §5.1.

O PNG de móveis recortados de catálogo foi **reprovado** pelo usuário e não entra no 7xAi.

## 3. Fotos de teste do repositório

Só entram fotos com licença comercial conferida na página de origem. Cada pasta
`tests/fixtures/rooms/<nome>/` tem um `SOURCE.md` com URL, autor e licença:

| Fixture | Licença | Origem |
|---|---|---|
| `stafford` (análise da foto `photos/sala_vazia_stafford.jpg`) | ver `tests/fixtures/photos/ATRIBUICOES.md` | ver o arquivo |
| `sala_sofas_brancos` | CC0 1.0 | https://stocksnap.io/photo/interior-design-HIH67XC5G0 |
| `nice_terrazzo` | CC0 1.0 | https://commons.wikimedia.org/wiki/File:Apartment_hotel_Nice.jpg |

## 4. Rascunho do EULA (seções de terceiros)

> **RASCUNHO.** Texto de trabalho para o instalador (PT-V3-10). Não usar como está.

### 4.1 Componentes de terceiros

"O 7xAi usa modelos de inteligência artificial e bibliotecas de terceiros, que rodam localmente no seu
computador. Cada componente continua sob a sua própria licença, cujos textos completos acompanham o programa
na pasta `licencas/`. Esses componentes são fornecidos 'no estado em que se encontram', sem garantias dos seus
autores."

### 4.2 Avisos MIT

Para cada componente MIT (MoGe-2, OneFormer, BiRefNet e, quando entrar, o VAE `sdxl-vae-fp16-fix`), o pacote
inclui o texto integral da licença MIT com a linha de copyright do autor, por exemplo:

```
MoGe — Copyright (c) Microsoft Corporation. Licença MIT.
OneFormer — Copyright (c) 2022 SHI Labs. Licença MIT.
BiRefNet — Copyright (c) 2024 ZhengPeng. Licença MIT.
```

O aviso de copyright e o texto da permissão MIT devem acompanhar todas as cópias (inclusive o instalador).

### 4.3 Avisos Apache-2.0

Para cada componente Apache-2.0 (SAM 2.1, LaMa/big-lama, Grounding DINO), o pacote inclui:
- uma cópia do texto da Apache License 2.0;
- o arquivo `NOTICE` do projeto, se existir (conferir no build; hoje não foi encontrado nenhum);
- a indicação de que os pesos foram convertidos ou empacotados pelo 7xAi, quando for o caso (seção 4 da licença).

### 4.4 Restrições de uso OpenRAIL++ (placeholder para o refino SDXL)

> **Placeholder.** Só passa a valer quando um peso OpenRAIL++ (SDXL, RealVisXL, SDXL-Lightning, ControlNet)
> entrar no pacote. Até lá, esta seção fica fora do EULA.

Quando entrar:
1. Copiar **literalmente** o "Attachment A — Use Restrictions" da licença CreativeML OpenRAIL++-M do modelo
   empacotado (arquivo `LICENSE.md` do repositório do peso) para esta seção do EULA, e obrigar o usuário final a
   respeitá-lo. As restrições não podem ser enfraquecidas pelo EULA.
2. Enviar uma cópia da licença OpenRAIL++ junto com os pesos.
3. Marcar os pesos modificados (por exemplo, LoRA fundida ao UNet) como modificados pelo 7xAi.
4. Texto de abertura sugerido: "Partes do 7xAi usam modelos licenciados sob a CreativeML OpenRAIL++-M. Ao usar
   o 7xAi, você concorda em não usar esses modelos, nem as imagens e vídeos gerados com eles, para nenhum dos usos
   proibidos listados abaixo." [seguir com o Attachment A integral]

### 4.5 Fotos do cliente e pixels gerados

"No modo 'foto mobiliada', o 7xAi gera por inteligência artificial a imagem da sala vazia mostrada no início do
vídeo. Essa imagem é uma estimativa e pode não corresponder ao imóvel real. O último quadro do vídeo é a foto
enviada. O relatório de cada trabalho informa a área gerada. Você é responsável por informar ao seu cliente
final, quando exigido, que o vídeo contém imagens geradas."
