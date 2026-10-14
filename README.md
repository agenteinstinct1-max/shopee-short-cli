# shopee-short-cli

CLI experimental em Python para criar vídeos verticais de produtos a partir de um link Shopee. Extrai o nome do slug sem abrir a página Shopee, procura mídia em fontes alternativas, gera roteiro, narração pt-BR com Piper e legendas.

## Instalação

Requer Python 3.11+, ffmpeg/ffprobe com libx264 e libass, e tesseract com idioma português para a verificação de legendas.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

```sh
shopee-short 'https://shopee.com.br/SLUG-DO-PRODUTO-i.123456789.987654321' -o output/produto
shopee-short --help
```

A disponibilidade de provedores externos pode mudar. O fluxo pode exigir confirmação da mídia e configuração de um provedor de roteiro. Não há promessa de gerar vídeo para qualquer produto. Confira modelo, direitos da mídia e resultado antes de publicar.

## Pronúncia

O texto enviado ao Piper usa um dicionário de pronúncia e tratamento de decimais/códigos. As legendas mantêm o texto original. Para acrescentar ou substituir entradas:

```sh
shopee-short "$PRODUCT_URL" --pronunciation-file examples/pronunciation.json -o output/produto
```

O JSON tem o formato `{"grafia": "pronúncia"}`. Também pode ser indicado pela variável `SHOPEE_SHORT_PRONUNCIATION`. As substituições são sugestões fonéticas, não garantia de pronúncia correta para todas as vozes.

## Rede e dados

O programa não consulta páginas Shopee nem seu CDN. Links `shp.ee` podem ser resolvidos sem visitar o destino Shopee. Bloqueios HTTP, CAPTCHA e desafios devem encerrar o fluxo, sem contorno automatizado. Vídeos e fotos vêm de fontes alternativas confirmadas.

Credenciais opcionais são lidas do ambiente, nunca devem ser incluídas no código ou em arquivos publicados. O diretório de saída pode conter roteiro, mídia, URLs, relatórios e logs. Revise esses arquivos antes de compartilhá-los. Este repositório distribui apenas código e exemplos, sem mídia gerada nem logs de execução.

## Testes

```sh
pip install pytest
pytest
```

Os testes não substituem validação de áudio, vídeo ou um teste completo com provedores reais. Esta versão é experimental.
