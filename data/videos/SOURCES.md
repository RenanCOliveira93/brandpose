# Vídeos de base

Baixados em 14/10/2026. Todos são de bancos de vídeo gratuitos. Antes de qualquer uso público, confira a licença atual de cada um.

## Mixkit (licença gratuita do Mixkit, sem marca d'água)

720p, câmera na altura dos olhos. URL direta: `https://assets.mixkit.co/videos/<id>/<id>-720.mp4`

| Arquivo | Página | Conteúdo |
|---|---|---|
| `mix_49137.mp4` | https://mixkit.co/free-stock-video/young-woman-looking-for-discounts-at-the-supermarket-49137/ | Mulher com carrinho olhando e pegando produtos (24 s). **Melhor caso.** |
| `mix_49349.mp4` | https://mixkit.co/free-stock-video/girl-chooses-products-from-the-supermarket-refrigerators-49349/ | Mulher escolhendo no freezer |
| `mix_25438.mp4` | https://mixkit.co/free-stock-video/browsing-items-in-a-supermarket-25438/ | Duas mulheres no corredor |
| `mix_49350.mp4` | https://mixkit.co/free-stock-video/man-chooses-products-from-supermarket-refrigerators-49350/ | Homem escolhendo no freezer |
| `mix_49358.mp4` | https://mixkit.co/free-stock-video/girl-looking-for-a-bread-toaster-in-the-supermarket-49358/ | Mulher olhando eletroportáteis |
| `mix_49289.mp4` | https://mixkit.co/free-stock-video/young-dad-choosing-and-buying-diapers-at-the-supermmarket-49289/ | Homem pegando produto no alto da prateleira |
| `mix_49353.mp4` | https://mixkit.co/free-stock-video/girl-chooses-hot-spices-at-the-supermarket-49353/ | Close de mão (pouco útil) |
| `mix_49343.mp4` | https://mixkit.co/free-stock-video/young-man-is-looking-for-ping-pong-equipment-at-the-49343/ | Homem olhando artigos esportivos |

## Pexels (licença do Pexels)

Download: `https://www.pexels.com/download/video/<id>/`

| Arquivo | Página | Conteúdo |
|---|---|---|
| `px_853782.mp4` | https://www.pexels.com/video/blurry-footage-of-people-inside-a-grocery-store-853782/ | Ângulo alto, parecido com câmera de segurança, **desfocado** (15 s) |
| `px_39240106.mp4` | https://www.pexels.com/video/diverse-shopping-experience-in-a-supermarket-39240106/ | Corredores com várias pessoas, câmera baixa, com cortes (91 s, 1080p60) |
| `px_35189918.mp4` | https://www.pexels.com/video/busy-supermarket-aisle-with-shoppers-35189918/ | Vertical, corredor cheio (6 s) |
| `px_37101039.mp4` | https://www.pexels.com/video/busy-supermarket-aisle-with-shoppers-37101039/ | Vertical, corredor (11 s) |
| `px_4081583.mp4` | https://www.pexels.com/video/a-man-walking-inside-the-supermarket-4081583/ | Vertical, homem andando no corredor (25 s) |

**Atenção:** nenhum destes vídeos é de câmera de vigilância. Todos são filmados na altura dos olhos ou com câmera em movimento. O 853782 é o mais parecido, mas está desfocado.

Também baixei e descartei, entre os candidatos a câmera de vigilância (buscas "security camera store" e "high angle supermarket"), os vídeos 10901926, 4135935, 4473193, 39240102, 39238831 e 6258195 do Pexels. A câmera de todos se move ou fica na altura dos olhos.

## Câmera fixa no teto

Fica em `data/datasets/merl_shopping/` (MERL Shopping). Veja `docs/experiments/2026-10-14-merl-shopping.md`. A câmera é fixa e nítida, mas aponta reto para baixo.

## Câmera de segurança real e inclinada (YouTube, só para uso local)

| Arquivo | Fonte | Conteúdo | Direitos |
|---|---|---|---|
| `youtube/yt_-1bRhYjw1qE.mp4` | https://www.youtube.com/watch?v=-1bRhYjw1qE | Demonstração da câmera ACTi E43A numa loja de conveniência nos EUA. 60 s, 1080p, câmera **fixa e inclinada**, várias pessoas. **É o vídeo principal do estudo.** | Copyright ACTi Corporation. Só para estudo local, não redistribuir. |

Baixado com `yt-dlp -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]"`.

## Ainda falta

Mais cenas de câmera inclinada, de lojas diferentes. De preferência com licença aberta, ou então mais vídeos de demonstração de fabricantes, só para uso local.
