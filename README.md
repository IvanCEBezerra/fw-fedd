# fw-fedd
Biblioteca Python para detectar mudanças de conceito em séries temporais univariadas à medida que novas observações chegam. O FW-FEDD extrai 23 características de janelas deslizantes, compara cada janela com uma referência usando distância de Pearson ponderada e sinaliza mudanças com um teste ICI cumulativo.

# FW-FEDD — detecção online de mudanças de conceito

Biblioteca Python para detectar mudanças de conceito em séries temporais univariadas. O detector processa uma observação por vez, extrai 23 características de uma janela deslizante e compara cada janela com a referência do regime atual. A comparação usa distância de Pearson ponderada pelo inverso do desvio padrão das características. O teste ICI cumulativo sinaliza quando as distâncias deixam de ser compatíveis com o regime de referência.

O projeto implementa o componente de detecção inspirado no *Feature Extraction and Weighting for Explicit Concept Drift Detection* (FW-FEDD), da tese de Rodolfo Carneiro Cavalcante. O código é uma reconstrução independente, com escolhas próprias para a extração de características, a estimação do período e a implementação do ICI. [Tese de Cavalcante, capítulos 4 e 5](https://repositorio.ufpe.br/bitstream/123456789/25349/1/TESE%20Rodolfo%20Carneiro%20Cavalcante.pdf)

## Instalação

Requer Python 3.10 ou superior. As dependências do detector são NumPy, SciPy e statsmodels. Na raiz do projeto, instale em modo editável.

```bash
python -m pip install -e .
```

Para instalar a partir de outra aplicação, informe o caminho da pasta que contém o `pyproject.toml`.

```bash
python -m pip install "/caminho/para/fw-fedd-thesis"
```

O pacote de distribuição se chama `fw-fedd` e o módulo importável, `fw_fedd`. Os CSVs e PNGs incluídos no repositório ficam fora do pacote instalado.

## Uso rápido

Crie uma instância por série e forneça os valores em ordem cronológica. `update` recebe um valor escalar finito por chamada e retorna `"normal"` ou `"drift"`. O alarme é retornado na mesma chamada que registra o evento. Durante o aquecimento, `"normal"` indica que nenhum alarme foi emitido. Use `last_test_performed` para saber se houve um teste na amostra atual.

```python
from fw_fedd import FWFEDDDetector

detector = FWFEDDDetector()


def receber_observacao(valor: float) -> str:
    estado = detector.update(valor)
    if estado == "drift":
        indice = detector.n_samples - 1
        distancia = detector.last_distance
        print(f"Alarme no índice {indice}, distância = {distancia:.4f}")
    return estado
```

Chame `receber_observacao(valor)` a cada novo valor do fluxo. O índice é contado a partir de zero e identifica a observação que disparou o alarme. Na amostra seguinte, o detector começa a formar uma nova referência e preserva a contagem global e o histórico de alarmes. Use `reset()` para começar uma série independente na mesma instância, com contadores e histórico zerados.

### Executar sobre um CSV incluído no projeto

Execute o exemplo abaixo a partir da raiz do projeto. A coluna `value` alimenta o detector. A coluna `is_drift` fornece as fronteiras conhecidas do gerador para inspeção dos resultados.

```python
import csv
from pathlib import Path

from fw_fedd import FWFEDDDetector

caminho = Path("fw_fedd/data/dataset/AR/data/series_01.csv")
detector = FWFEDDDetector()
mudancas_rotuladas = []

with caminho.open(encoding="utf-8", newline="") as arquivo:
    for linha in csv.DictReader(arquivo):
        detector.update(float(linha["value"]))
        if linha["is_drift"] == "1":
            mudancas_rotuladas.append(int(linha["index"]))

print("Amostras processadas:", detector.n_samples)
print("Mudanças rotuladas:", mudancas_rotuladas)
print("Alarmes:", [indice for indice, _ in detector.detections])
```

O índice de um alarme pode ocorrer depois da fronteira rotulada. Para medir atraso, falsos alarmes ou mudanças não detectadas, defina a tolerância e a regra de pareamento entre alarmes e fronteiras antes de comparar os resultados.

## Parâmetros e diagnósticos

| Parâmetro | Padrão | Uso |
| --- | --- | --- |
| `window_size` | `150` | Comprimento da janela em observações. Inteiro a partir de 7. |
| `period` | `None` | Período sazonal da STL. Com `None`, é estimado na janela de referência de cada regime. Para fixá-lo, informe um inteiro a partir de 2 e menor que `window_size`. |
| `epsilon` | `1e-6` | Termo positivo e finito somado ao desvio padrão no cálculo dos pesos. |
| `training_size` | `300` | Quantidade de distâncias usadas para calibrar o ICI. Inteiro a partir de 2. |
| `gamma` | `2.5` | Multiplicador positivo e finito da largura do intervalo do ICI. Em condições iguais, valores maiores alargam o intervalo. |

| Atributo | Resultado |
| --- | --- |
| `state` | Estado do detector, inicialmente `"normal"`. |
| `n_samples` e `n_drifts` | Contagens de observações processadas e alarmes desde a criação ou o último `reset()`. |
| `detections` | Cópia da lista de eventos no formato `(índice, "drift")`, em ordem cronológica. |
| `last_phase` | Fase da última atualização. Pode ser `reference_window_collection`, `internal_test_training` ou `monitoring`. |
| `last_active_period` | Período STL da última atualização com uma janela completa. É `None` enquanto a referência está incompleta. |
| `last_distance` | Distância da última atualização, quando calculada. |
| `last_test_performed` | Indica se a última atualização testou a interseção dos intervalos. |
| `last_test_statistics` | Dicionário com o estado do ICI na última atualização com distância, incluindo médias, limites, erro padrão e contadores. Fica vazio na coleta da referência. |

Os diagnósticos da amostra que disparou o alarme continuam acessíveis até a próxima atualização. O caminho de importação `fw_fedd.detection.fw_ici.FWFEDDICIDetector` permanece como alias de `FWFEDDDetector`.

## Como a detecção funciona

1. **Formação da referência.** A primeira janela completa de cada regime define o vetor de referência. Com `period=None`, o detector remove a tendência linear dessa janela para estimar o período. Usa 2 quando a busca não encontra um período válido. O período adotado é mantido durante o regime.
2. **Extração de características.** A decomposição STL separa tendência, sazonalidade e resíduo. A biblioteca calcula 23 características por janela completa.

   | Características | Quantidade | Origem |
   | --- | ---: | --- |
   | Força da tendência e da sazonalidade | 2 | Componentes da STL |
   | Desvio padrão, assimetria, curtose e periodicidade | 4 | Resíduo |
   | Percentual de pontos de virada | 1 | Janela original, pela contagem de máximos e mínimos locais |
   | Autocorrelação (ACF) e autocorrelação parcial (PACF) | 10 | Resíduo, defasagens de 1 a 5 para cada medida |
   | Bicorrelação e informação mútua | 6 | Resíduo, defasagens de 1 a 3 para cada medida |

3. **Ponderação e distância.** O detector atualiza a dispersão de cada característica pelo algoritmo de Welford, incluindo a janela de referência e a janela atual. O peso é proporcional a `1 / (desvio_padrão + epsilon)`, com desvio padrão populacional. A distância é `1 - correlação_de_Pearson_ponderada` e varia de 0 a 2. A referência permanece fixa enquanto os pesos são atualizados.
4. **Calibração e monitoramento.** As primeiras `training_size` distâncias definem a média e o desvio padrão amostral de calibração. A partir da distância seguinte, o ICI de ordem zero calcula um intervalo bilateral centrado na média cumulativa das distâncias, com raio `gamma * desvio_de_calibração / sqrt(n)`. O desvio de calibração permanece fixo, e `n` inclui as distâncias da calibração. O detector sinaliza `drift` quando o novo intervalo torna vazia a interseção acumulada.
5. **Início de um novo regime.** Após o alarme, a janela, os pesos e o ICI são reiniciados. A próxima amostra inicia a coleta da nova referência.

Com os padrões, a referência é formada na amostra de índice **149**, as 300 distâncias de calibração ocupam os índices **150 a 449** e o primeiro teste ocorre no índice **450**, na **451ª amostra**. Em geral, o primeiro teste de um regime ocorre na amostra de número `window_size + training_size + 1`, contando a partir de um.

## Dados sintéticos

A pasta [`fw_fedd/data/dataset/`](fw_fedd/data/dataset/) contém 80 séries do perfil `corrected`, com 20 realizações por família, um CSV e um gráfico por série e um [`metadata.json`](fw_fedd/data/dataset/metadata.json) com sementes individuais, configuração e índices das mudanças.

| Família | Processo gerador |
| --- | --- |
| `AR` | Modelos autorregressivos com diferentes coeficientes e ordens. |
| `LS` | Padrões sazonais periódicos com ruído. |
| `NL` | Modelos autorregressivos não lineares com transição suave. |
| `RW-AR-NL` | Combinação de passeio aleatório, processos autorregressivos e processos não lineares. |

Cada série contém 11.000 observações em 11 segmentos de 1.000 amostras. Os conceitos seguem a sequência recorrente `1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1`. Há 10 fronteiras rotuladas por série, nos índices 1000 a 10000, em passos de 1000.

| Coluna | Conteúdo |
| --- | --- |
| `index` | Índice da observação a partir de zero. |
| `t` | Contagem da observação a partir de um. |
| `value` | Valor observado. |
| `concept_id` | Identificador do regime gerador. |
| `is_drift` | `1` na primeira amostra de um novo segmento e `0` nas demais. |

### Gerar novos dados

Na raiz do projeto, execute o comando abaixo para gerar uma série curta por família, com CSVs e metadados.

```bash
python -m fw_fedd.data.gen_dataset --profile corrected --output ./dataset_demo --n-series 1 --segment-length 60 --seed 42 --no-plots
```

Para gerar também os gráficos, instale o extra `data` e execute o gerador sem `--no-plots`.

```bash
python -m pip install -e ".[data]"
python -m fw_fedd.data.gen_dataset --profile corrected --output ./dataset_com_graficos --seed 42
```

Sem `--output`, o gerador usa `fw_fedd/data/dataset_corrected/` ou `fw_fedd/data/dataset_literal/`, de acordo com o perfil. Se algum arquivo de saída já existir, o gerador exige `--overwrite` para substituí-lo. Consulte as opções com `python -m fw_fedd.data.gen_dataset --help`.

- **`corrected`** usa transição logística, uma alteração de sinal no primeiro conceito AR e reflexão dos polos explosivos na parte autorregressiva de base de alguns modelos não lineares. O manifesto registra os coeficientes impressos e os efetivamente usados quando aplicável.
- **`literal`** usa os coeficientes impressos e, por padrão, a função de transição impressa. A geração pode falhar ao encontrar uma singularidade ou valores não finitos. Para ambos os perfis, a inicialização e a escala do ruído são escolhas da implementação.

A geração usa como referência a seção 5.2.1, a Tabela 2 e as equações 5.1–5.2 da tese. Para repetir uma execução, registre a semente, o perfil, as demais opções e as versões de Python e das dependências. Cada realização inclui sua semente e a versão do NumPy no manifesto.

## Estrutura do projeto

```text
fw-fedd-thesis/
├── README.md
├── pyproject.toml
└── fw_fedd/
    ├── __init__.py              # API pública do detector
    ├── detection/
    │   ├── detector.py          # Fluxo online, calibração e alarmes
    │   ├── pearson.py           # Distância de Pearson ponderada
    │   ├── ici.py               # Teste ICI cumulativo
    │   └── fw_ici.py            # Alias para importações anteriores
    ├── features/                # STL e extração das 23 características
    ├── weighting/
    │   └── fw_module.py         # Pesos pelo inverso do desvio padrão
    └── data/
        ├── gen_dataset.py       # Interface de linha de comando
        ├── synthetic.py         # Geração e exportação das séries
        └── dataset/             # CSVs, gráficos e metadados
```

## Escopo e limitações

- O detector processa valores finitos e rejeita `NaN` e infinitos. O fluxo deve chegar em ordem cronológica, com o tratamento de ausências feito pela aplicação.
- A janela de referência e o período de calibração pressupõem que as primeiras observações de cada regime ofereçam uma referência útil. Mudanças nessa fase podem afetar as decisões posteriores.
- As janelas sobrepostas geram distâncias potencialmente dependentes. `gamma` controla a largura dos intervalos. O valor 2,5 foi selecionado em um piloto de quatro séries. O multiplicador por si só não especifica a taxa de falsos alarmes sequenciais.
- A implementação concentra-se no detector. A tese também descreve um módulo de previsão com um conjunto de modelos OS-ELM. Os experimentos com o sistema completo dependem desse módulo e do protocolo original.

## Referências

- Cavalcante, R. C. (2017). *An Adaptive Learning System for Time Series Forecasting in the Presence of Concept Drift*. Tese de doutorado, UFPE. [Texto integral](https://repositorio.ufpe.br/bitstream/123456789/25349/1/TESE%20Rodolfo%20Carneiro%20Cavalcante.pdf)
- Cavalcante, R. C., Minku, L. L. e Oliveira, A. L. I. (2016). *FEDD: Feature Extraction for Explicit Concept Drift Detection in Time Series*. IJCNN. [Publicação](https://research.birmingham.ac.uk/en/publications/fedd-feature-extraction-for-explicit-concept-drift-detection-in-t/)
- Alippi, C., Boracchi, G. e Roveri, M. (2011). *A Just-in-Time Adaptive Classification System Based on the Intersection of Confidence Intervals Rule*. Neural Networks. [Preprint, seções 3.2 e 3.4](https://boracchi.faculty.polimi.it/docs/2011_NN_Alippi_Boracchi_Roveri_preprint.pdf)
