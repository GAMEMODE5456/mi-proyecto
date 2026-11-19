import tensorflow as tf
import numpy as np

celsius = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
fahrenheit = np.array([32, 50, 68, 86, 104, 122, 140, 158, 176, 194, 212], dtype=float)

capa  = tf.keras.layers.Dense(units = 1, input_shape = [1])
modelo  = tf.keras.Sequential([capa])

modelo.compile(
    optimizer = tf.keras.optimizers.Adam(0.1),
    loss = "mean_squared_error"
)

print("Comenzando entrenamiento")
historial = modelo.fit(celsius, fahrenheit, epochs = 700, verbose = False)
print("modelo entrenado")
"""
import matplotlib.pyplot as plt
plt.xlabel("# Epoca")
plt.ylabel("# Magnitud de perdida")
plt.plot(historial.history["loss"])
"""

print("Hagamos una prediccion")
resultado = modelo.predict(x=np.array([100.0]))
print(f"El resultado es {resultado} Fahrenheit")