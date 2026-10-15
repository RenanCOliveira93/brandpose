# 2026-10-14: câmera fixa no teto (MERL Shopping)

## Por que este dataset

Os vídeos de banco de imagens (`2026-10-14-gaze-lle.md`) não são de câmera de vigilância: a câmera fica na altura dos olhos e se move. O vídeo mais parecido com câmera de vigilância (Pexels 853782) estava desfocado. Também procurei no Pexels e no Pixabay com buscas como "security camera store", "high angle supermarket" e "cctv store", mas nenhum resultado tinha câmera fixa. O MERL Shopping foi a melhor opção real que encontrei.

## Dataset

- **Fonte:** <https://www.merl.com/pub/tmarks/MERL_Shopping_Dataset/>, artigo no CVPR 2016.
- **Licença:** livre para pesquisa e educação, com pedido de citação do artigo.
- **Conteúdo:** 106 vídeos de cerca de 2 minutos, com 41 pessoas e até 3 sessões cada. Formato H.264, 920×680 a 30 fps.
- **Cena:** uma loja montada em laboratório, com prateleiras de produtos reais. A câmera é **fixa, nítida e aponta reto para baixo**, então se vê o topo da cabeça.
- **Rótulos:** cinco ações: pegar da prateleira, retirar da prateleira, mão na prateleira, inspecionar produto e inspecionar prateleira.
- **Onde está:** `data/datasets/merl_shopping/Videos_MERL_Shopping_Dataset.zip`, com 1,8 GB. Só `10_1` e `27_1` foram extraídos para `videos/`. Para extrair um vídeo específico:
  `unzip -j Videos_MERL_Shopping_Dataset.zip "Videos_MERL_Shopping_Dataset/<id>_crop.mp4" -d videos`

## Rodada: `27_1`, primeiros 60 s, 2 quadros por segundo

Saídas em `outputs/2026-10-14-merl/27_1/`.

| Métrica | Valor |
|---|---|
| Olhares (total / dentro do quadro) | 87 / 82 |
| IDs de rastreamento | 6, para **uma pessoa só** |
| Atenção por região (grade 4×3, linha do meio) | 11% · **40%** · **32%** · 0% |

**O que funcionou**
- É a **primeira vez que o mapa acumulado faz sentido**, porque a câmera está parada. A atenção se concentra na frente das prateleiras do centro e da direita, onde a pessoa pega e examina os produtos.
- O Gaze-LLE aponta para a prateleira à frente da pessoa mesmo vendo só o topo da cabeça. A nota "dentro do quadro" ficou alta (cerca de 0,95).

**Limitações**
1. **Vista de cima:** o modelo não vê o rosto e acaba apontando para a frente do corpo. O resultado se parece mais com uma estimativa da direção do corpo do que do olhar.
2. **Não dá para ver a frente da prateleira.** De cima só aparece o topo das gôndolas, então o calor cai na borda da prateleira, não no produto.
3. **O rastreamento se perde:** foram 6 IDs para a mesma pessoa, porque o YOLO-pose foi treinado com pessoas vistas de lado ou de frente.
4. A cena é de laboratório, com uma pessoa só. É menos difícil que uma loja real.

## Conclusão

O MERL serve para validar o **mapa acumulado** e as **métricas por zona** com câmera fixa. Não serve para validar a **precisão do olhar**: para isso é preciso uma câmera **inclinada**, de 30° a 60° em relação ao chão, como as câmeras de segurança comuns e como no reel do CactusXR. Ainda não encontrei vídeo aberto e de boa qualidade com esse ângulo.

## Próximos passos

- [ ] Desenhar zonas por prateleira na cena do MERL, que tem layout fixo, e comparar a atenção por zona com os rótulos "inspecionar produto/prateleira".
- [ ] Rodar mais pessoas e sessões e somar tudo num mapa de calor da loja.
- [ ] Continuar procurando vídeo com câmera **inclinada**. Opções: vídeos de demonstração de fabricantes de câmeras (têm direitos autorais, só para uso local), datasets como MEVA ou VIRAT, ou gravar um vídeo próprio.
