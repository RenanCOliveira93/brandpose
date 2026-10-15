# Pesquisa: análise de atenção do olhar em vídeo

Levantamento feito em 14/10/2026.

## O problema

Queremos saber **para onde cada pessoa olha** num vídeo de câmera comum e somar esses olhares em métricas, como a atenção por prateleira. O reel do CactusXR mostra o resultado final, mas não encontrei nenhum projeto open source que faça tudo isso de ponta a ponta. A solução é juntar peças:

1. **Detectar e rastrear pessoas.**
2. **Achar a cabeça** de cada pessoa.
3. **Estimar o alvo do olhar.** Na literatura isso se chama *gaze following* ou *gaze target estimation*.
4. **Ligar o olhar a regiões da cena**, seja com polígonos 2D por câmera, seja com um modelo 3D e a linha do olhar.
5. **Somar** os olhares em mapas de calor e métricas.

## Modelos de alvo do olhar

| Modelo | Ano e evento | Código | Licença | Notas |
|---|---|---|---|---|
| **Gaze-LLE** | CVPR 2025 | `github.com/fkryan/gazelle` | MIT | **Escolhido.** Usa DINOv2 congelado e um decodificador pequeno. Recebe a imagem e as caixas das cabeças normalizadas em [0,1] e devolve um mapa de calor 64×64 por pessoa mais uma nota "olhando dentro ou fora do quadro" (variantes `_inout`). Carrega com `torch.hub.load("fkryan/gazelle", "gazelle_dinov2_vitb14_inout")`. Também tem variantes ViT-L e ChildPlay. |
| **ViTGaze** | Visual Intelligence 2024 | `github.com/hustvl/vitgaze` (pesos no Hugging Face `yhsong/ViTGaze`) | não verificada | Usa um ViT pré-treinado e é baseado em detectron2. |
| **Sharingan** | CVPR 2024 | não encontrado | — | Trata várias pessoas numa única passada, com um token por pessoa. Útil para loja cheia. |
| **Gaze360** | ICCV 2019 | não confirmado | — | Estima a **direção** do olhar à distância, não o alvo. O artigo mostra um exemplo de mapa de calor de clientes diante de uma prateleira. |
| **L2CS-Net** e afins | — | vários | — | Precisam do rosto visível e de perto. Não servem para câmera de loja. |

Uma alternativa mais simples é estimar só a direção da cabeça, por exemplo com 6DRepNet ou com os pontos do rosto do MMPose ou do MediaPipe, e projetar um cone. Em câmera de teto, isso costuma ser o máximo que dá para medir com confiança.

## Detecção, rastreamento e cabeça

- **Ultralytics YOLO11-pose** (AGPL-3.0): detecta a pessoa e 17 pontos do corpo, incluindo nariz, olhos e orelhas, e rastreia com ByteTrack ou BoT-SORT pelo `model.track`. Foi o que usamos.
- **Roboflow `supervision`** (MIT): utilitários de zonas, tempo de permanência e mapa de calor.
- **Ponto fraco encontrado:** em câmera alta, com pessoa pequena ou de costas, o YOLO-pose não acha os pontos do rosto. Vale testar um **detector de cabeças** treinado em multidões vistas de cima.

## Datasets

| Dataset | Conteúdo | Licença e acesso |
|---|---|---|
| **Retail Gaze** | Cerca de 3.900 imagens de clientes olhando produtos, vistos por câmeras de fora, em 12 ângulos. Marca o ponto do olhar, a caixa da cabeça e a máscara do produto. A versão V2 adiciona segmentação das prateleiras. | `github.com/PrimeshShamilka/RetailGazeDataset`, Kaggle e Voxel51. Licença não informada. |
| **GOO (Gaze On Objects)** | Pede para prever a **caixa do produto** olhado, não só um ponto. Tem a parte GOO-Synth (sintética, grande) e GOO-Real. | CVPRW 2021, arXiv 2105.10793 |
| **MERL Shopping** | Cerca de 106 vídeos de 2 minutos com **câmera fixa no teto**. As ações marcadas são pegar da prateleira, devolver, mão na prateleira, inspecionar produto e inspecionar prateleira. | `merl.com/pub/tmarks/MERL_Shopping_Dataset/`, um zip de 1,7 GB. Só pesquisa não comercial. |
| **RetailAction** (Standard AI, ICCVW 2025) | Cerca de 42 mil clipes de câmeras **360° no teto** de 10 lojas reais nos EUA, cerca de 41 h. Foi feito para interação pessoa-produto. | Hugging Face `harpreetsahota/RetailAction`. Licença **proprietária**. De cima, dificilmente dá para estimar o olhar. |
| **Shoplifting (Kaggle e Innovatiana)** | Vídeos reais de câmera de segurança de loja, com rótulo por clipe | 640×480, baixa qualidade |
| **Simuletic CCTV Shoplifting** | Câmera de segurança **renderizada**, com caixas das pessoas e pontos do corpo | Amostra pequena. A licença é conflitante na página (CC BY e CC BY-NC-SA). |
| **MMPTRACK** | Várias câmeras calibradas em cinco ambientes, incluindo varejo, a 15 fps | WACV 2023. Não achei o link de download. |
| **GazeFollow / VideoAttentionTarget / ChildPlay** | Bases gerais em que os modelos acima foram treinados. | — |

## Projetos de varejo

- **Intel `shopper-gaze-monitor-python`** e **`smart-retail-analytics`**: detectam rosto e direção da cabeça e contam quem está "olhando" para a câmera ou vitrine. Usam OpenVINO, trabalham com uma câmera só e estão arquivados.
- **H.E.A.T.**: mapa de calor com câmera de profundidade RealSense. Estava em desenvolvimento.

## CactusXR (Auki)

Não encontrei documentação técnica. O que se sabe vem da legenda do reel e de perfis da empresa:
- Usa as câmeras de segurança que a loja já tem, ligadas a um gêmeo digital 3D, sem equipamento extra.
- A Auki se descreve como construtora do *posemesh*, uma rede de percepção espacial, e o Cactus como uma plataforma de IA espacial para varejo.
- O vídeo mostra uma nuvem de pontos da loja, a pose da pessoa desenhada como esqueleto e um mapa de calor de atenção sobre as prateleiras.

## Fontes

- Gaze-LLE: https://arxiv.org/abs/2412.09586 · https://github.com/fkryan/gazelle · demo https://fffiloni-gaze-lle.hf.space/
- ViTGaze: https://arxiv.org/abs/2403.12778 · https://huggingface.co/yhsong/ViTGaze
- Sharingan: https://openaccess.thecvf.com/content/CVPR2024/html/Tafasca_Sharingan_A_Transformer_Architecture_for_Multi-Person_Gaze_Following_CVPR_2024_paper.html
- Gaze360: https://arxiv.org/abs/1910.10088
- GOO: https://arxiv.org/abs/2105.10793
- Retail Gaze: https://ieee-dataport.org/documents/retail-gaze-gaze-estimation-retail-environment · https://docs.voxel51.com/dataset_zoo/datasets_hf/retail_gaze.html
- MERL Shopping: https://merl.com/research/downloads/MERL_Shopping_Dataset · https://docs.voxel51.com/dataset_zoo/datasets_hf/merl_shopping_dataset.html
- Intel shopper-gaze-monitor: https://github.com/intel-iot-devkit/shopper-gaze-monitor-python
- Auki (perfil): https://www.swedcham.com.hk/member-detail/?id=1648333
