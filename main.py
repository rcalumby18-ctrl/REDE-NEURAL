import numpy as np
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt


## CARREGAMENTO DO DATASET

iris = load_iris()

X = iris.data
y = iris.target

print("Quantidade de amostras:", len(X))
print("Quantidade de características:", X.shape[1])
print("Classes:", iris.target_names)


# 2. DIVISÃO DOS DADOS
 

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDados de treinamento:", len(X_train))
print("Dados de teste:", len(X_test))


# 3. NORMALIZAÇÃO DOS DADOS

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. CRIAÇÃO DA REDE NEURAL

model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        8,
        activation="relu",
        input_shape=(4,)
    ),

    tf.keras.layers.Dense(
        8,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        3,
        activation="softmax"
    )
])

# 5. CONFIGURAÇÃO DO TREINAMENTO

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. TREINAMENTO

print("\nIniciando treinamento...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_split=0.2,
    verbose=1
)

# 7. TESTE DO MODELO

print("\nAvaliando o modelo...")

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nResultado do teste:")
print("Loss:", loss)
print("Acurácia:", accuracy)

# 8. PREVISÕES

predictions = model.predict(X_test)

y_pred = np.argmax(predictions, axis=1)

accuracy = accuracy_score(y_test, y_pred)

print("\nAcurácia final:", accuracy)

# 9. MATRIZ DE CONFUSÃO

matrix = confusion_matrix(
    y_test,
    y_pred
)

print("\nMatriz de confusão:")
print(matrix)


# 10. GRÁFICO DA ACURÁCIA

plt.plot(
    history.history["accuracy"],
    label="Treinamento"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validação"
)

plt.title("Acurácia durante o treinamento")
plt.xlabel("Época")
plt.ylabel("Acurácia")

plt.legend()
plt.show()


# 11. GRÁFICO DA MATRIZ DE CONFUSÃO

plt.imshow(matrix)

plt.title("Matriz de Confusão")
plt.xlabel("Classe prevista")
plt.ylabel("Classe real")

plt.colorbar()

plt.xticks(
    [0, 1, 2],
    iris.target_names
)

plt.yticks(
    [0, 1, 2],
    iris.target_names
)

plt.show()