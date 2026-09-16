## Rede Neural Artificial com Dataset Iris

Trabalho prático da disciplina de Inteligência Artificial.

O objetivo deste projeto é desenvolver, treinar e avaliar uma Rede Neural Artificial (ANN) utilizando Python e bibliotecas de aprendizado de máquina.

---

##### Professor: Bernardo Alvez Vallarinho Lima
##### Matéria: Inteligencia Artificial

---
##### Alunos:  
Rayana Calumby de Oliveira  
Ricardo Marcarini Sangaletti  
Jean Lucas Biene  
Thiago da Costa  
Eduardo Leandro Brandalise  
Gustavo Lotti Tonieto  

---

## 1. Dataset

Para este projeto foi escolhido o dataset Iris, uma base de dados clássica utilizada em exemplos de aprendizado de máquina.

O dataset possui:

- 150 amostras;
- 4 características para cada amostra;
- 3 classes diferentes.

As características utilizadas são:

- Comprimento da sépala;
- Largura da sépala;
- Comprimento da pétala;
- Largura da pétala.

As três classes representam espécies de flores:

- Setosa;
- Versicolor;
- Virginica.

### Por que o dataset Iris?

O dataset foi escolhido por ser pequeno, simples e adequado para um primeiro projeto de classificação utilizando uma rede neural artificial.

Além disso, suas características são numéricas, facilitando o processo de treinamento e normalização dos dados.

## 2. Stack Tecnológica

Foi utilizada a seguinte stack:

- **Python**: linguagem utilizada para desenvolver o projeto;
- **TensorFlow/Keras**: utilizado para criar e treinar a rede neural;
- **Scikit-Learn**: utilizado para carregar o dataset, dividir os dados, normalizar os valores e realizar a avaliação;
- **NumPy**: utilizado para operações com os dados;
- **Matplotlib**: utilizado para gerar os gráficos dos resultados.

### Por que Python?

Python foi escolhido por possuir uma grande quantidade de bibliotecas voltadas para Inteligência Artificial e Aprendizado de Máquina.

Além disso, sua sintaxe simples facilita o desenvolvimento e o entendimento do código.

## 3. Pré-processamento dos dados

Primeiramente, o dataset Iris foi carregado utilizando o Scikit-Learn.

Os dados foram divididos em duas partes:

- 80% para treinamento;
- 20% para teste.

A divisão foi realizada utilizando a função `train_test_split`.

Depois disso, foi realizada a normalização dos dados utilizando o `StandardScaler`.

A normalização é importante porque coloca os valores das características em uma escala mais adequada para o treinamento da rede neural.

## 4. Arquitetura da Rede Neural

Foi criada uma rede neural do tipo feedforward utilizando o Keras.

A arquitetura utilizada foi:

```text
4 entradas
   ↓
8 neurônios - ReLU
   ↓
8 neurônios - ReLU
   ↓
3 neurônios - Softmax
