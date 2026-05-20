# Sistema de Triagem e Classificação de Obesidade

## Visão Geral
Este projeto apresenta uma solução de Machine Learning para a triagem e classificação de risco de obesidade. Utilizando um conjunto de dados abrangente sobre hábitos de vida, histórico familiar e métricas antropométricas, o sistema permite prever categorias de obesidade com alta precisão, auxiliando na triagem inicial de pacientes de forma ágil e fundamentada. A solução foi encapsulada em uma interface web interativa.

## Metodologia de Desenvolvimento

### 1. Qualidade e Preparação dos Dados
O processo iniciou-se com a organização e padronização das informações coletadas. Foram removidos ruídos e inconsistências da base de dados original para assegurar que a análise fosse fundamentada em informações confiáveis e prontas para o processamento pela inteligência artificial.

### 2. Escopo da Análise
Para construir um perfil abrangente do paciente, cruzamos duas categorias fundamentais de dados:
* **Dados Físicos:** Idade, altura e peso, que compõem a base antropométrica.
* **Dados Comportamentais:** Histórico de saúde familiar, hábitos alimentares, nível de atividade física, hidratação e padrões de deslocamento diário.

### 3. Padronização e Tradução de Dados
Como modelos de inteligência artificial operam a partir de cálculos matemáticos, foi necessário converter informações textuais e categóricas para uma linguagem compatível. Este passo garantiu que dados qualitativos e quantitativos fossem interpretados pelo modelo com o mesmo nível de relevância.

### 4. Garantia de Aprendizado Equilibrado
Para evitar que o modelo apresentasse viés em favor de perfis predominantes na base de dados, foi aplicado técnicas de rebalanceamento. Este procedimento assegurou que o modelo aprendesse a identificar padrões de todas as categorias de obesidade com igual eficiência, aumentando a precisão do diagnóstico em grupos que possuem menos registros no *dataset*.

### 5. Processo de Validação
Estabelecemos um fluxo de trabalho rigoroso para garantir a consistência das predições:
* **Tratamento e Seleção:** Identificação e seleção das variáveis mais relevantes para a predição.
* **Codificação:** Conversão dos dados para o formato matemático adequado.
* **Ajuste de Pesos:** Aplicação de correções para evitar desequilíbrios durante a fase de treinamento.
* **Divisão Estratificada:** Segmentação da base de dados em 80% para o aprendizado e 20% para a validação final, garantindo que o modelo seja capaz de generalizar o conhecimento para dados novos e inéditos.

## Arquitetura e Tecnologias
* **Linguagem:** Python 
* **Processamento de Dados:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Interface Web:** Streamlit
* **Serialização de Modelos:** Joblib

## Resultados
O modelo final alcançou uma acurácia de 97% na base de teste, validando a robustez do pipeline de processamento e a eficácia da seleção de variáveis.

## Estrutura do Repositório
```text
.
├── app.py                      # Código fonte da aplicação web
├── modelo_obesidade.pkl        # Modelo de IA serializado
├── requirements.txt            # Lista de dependências do projeto
└── README.md                   # Documentação do projeto
```
## Instruções de Execução

### Instalação
No terminal, na pasta raiz do projeto, instale as dependências:

```bash
pip install -r requirements.txt
```
## Execução da Aplicação

Para iniciar a interface web, execute:
```bash
streamlit run app.py
```
### 6. Link de Acesso

A aplicação (SteamLit) está disponível aqui [AQUI](https://machine-learning-obesity-jvkfjappxkqsqra5n5vsqwa.streamlit.app/)

