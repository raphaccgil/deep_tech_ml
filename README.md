# deep_tech_ml
Projeto Demo para Apresentação na Deep Tech

![alt text](./img/dt_ml.png)

## Desafio:

### Fonte:
https://www.kaggle.com/sakshigoyal7/credit-card-customers

### Descritivo:
Identificar um perfil de usuário que tenha possibilidade de sair da conta. A idéia é tentar encontrar esse perfil

### Desafio:
Temos 10.000 consumidores com dados desde salário, estado civil, entre outros
 No arquivo possue somente 16,07% que ocorreu saída.

### Estrutura de código

```
-deploy
  - code for microservice
-notebook
  -jupyter notebook
-train
  - code for retrain model
```

#### deploy
```
docker build -t app_sample .
docker run -p 9028:9028 app_sample
```
Aqui apresenta container com os arquivos de configuração e o modelo trreinado

#### train

Aqui encontra onde treinar o modelo

#### notebook 

Aqui encontra o notebook com a exploração dos dados



## Resumoe da Apresentação

A apresentação irá iniciar com uma explicação sobre o que é Machine Learning e
as responsabilidades dos profissionais na áreas de dados. Falaremos dos conhecimentos
e ferramentas mais comuns na área além de ter uma visão inicial dos tipos de modelos e 
treinamentos existente.
Faremos um passo a passo para realizar um treinamento de um modelo e ver problemas comuns 
no mundo de Data Science.
Por fim, iremos visualizar conceitos mais complexos como AutoML, colocar um modelo em produção 
usando o conceito de MLOps e finalizaremos com um hands on. 