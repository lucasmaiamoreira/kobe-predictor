# Kobe-Predictor

### Este é um projeto que visa prever a seleção de arremessos de Kobe Bryant usando aprendizado de máquina.

## Descrição
### Este projeto consiste em três etapas principais:

1. Preparação de Dados: Nesta etapa, os dados brutos são pré-processados e filtrados para prepará-los para modelagem.
2. Treinamento do Modelo: Aqui, um modelo de aprendizado de máquina é treinado usando os dados preparados na etapa anterior.
3. Aplicação do Modelo: O modelo treinado é usado para fazer previsões sobre a seleção de arremessos em um ambiente de produção.

## Requisitos

### Para executar este projeto, você precisará ter os seguintes requisitos instalados:

- Python 3.x
- Ambiente virtual (recomendado)
- Pacotes Python listados em requirements.txt

## Instalação

1. Clone este repositório:

```
git clone https://github.com/seu-usuario/kobe-predictor.git
```

2. Navegue até o diretório do projeto:

```
cd kobe-predictor
```

3. Crie um ambiente virtual (opcional, mas recomendado):

```
python -m venv env
```

4. Ative o ambiente virtual:

    - No Windows:

        ```
        env\Scripts\activate
        ```

    - No Linux/macOS:

        ```
        source env/bin/activate
        ```


5. Instale as dependências do projeto:

```
pip install -r requirements.txt
```

## Uso

## Para executar todo o processo desde o servidor do Mlflow, treinamento até o servidor do Streamlit:

Execute o seguinte comando no terminal, dentro do diretório do projeto:

```
python main.py
```

Este comando irá realizar todas as etapas necessárias, desde a preparação dos dados até a execução do servidor do Streamlit para visualização dos resultados.

### Por favor, note que o código está configurado para rodar com Linux. 

### Se estiver executando em um ambiente Windows, você precisará descomentar a linha relevante no código e comentar a linha para Linux no arquivo main.py na função start_mlflow_server.

## MLflow
Para visualizar os resultados do treinamento do modelo usando MLflow, você pode iniciar o servidor MLflow:

```
mlflow ui --backend-store-uri sqlite:///mlruns.db
```
Isso iniciará o servidor MLflow e permitirá que você visualize experimentos, métricas e artefatos do modelo treinado.


1. Preparação de Dados - scripts/preparacao_dados.py:

Este script realizará a preparação dos dados, treinará um modelo de aprendizado de máquina e salvará os artefatos resultantes.


2. Treinamento do Modelo - scripts/treinamento.py:

Este script carregará os dados preparados, treinará o modelo e avaliará sua performance.


3. Aplicação do Modelo - scripts/aplicacao.py:

Este script carregará o modelo treinado e fará previsões sobre novos dados.


4. Visualização de Dados usando Streamlit - scripts/dashboard.py:

Uma interface web onde você poderá explorar os dados e visualizar as previsões feitas pelo modelo.


## Contribuindo
Contribuições são bem-vindas! Sinta-se à vontade para abrir um problema ou enviar uma solicitação de recebimento (pull request) para melhorias.


## Licença
Este projeto está licenciado sob a Licença MIT - consulte o arquivo LICENSE para obter mais detalhes.