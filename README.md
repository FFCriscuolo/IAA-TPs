# TP1 - Introducción al Aprendizaje Automático


Este repositorio contiene la resolución del **Trabajo Práctico 1: Árboles de Decisión, Random Forests y k-NN** de la materia **Aprendizaje Automático**.

## Estructura del repositorio

```
IAA-TPs/
│
├── data.py                 # Carga y separación del conjunto de datos en train/test
│
├── knn.py                  # Implementación del algoritmo k-NN (KNNClassif) y Weighted k-NN (WeightedKnnClassif)
├── tree.py                 # Implementación de Árbol de Decisión (CART)
├── random_forest.py        # Implementación de Random Forest (RF)
│
├── 2.ipynb       # Implementación desde cero (CART)
├── 3.ipynb       # Optimización del modelo
├── 4.ipynb       # Extensión a Random Forests
├── 5.ipynb       # k-NN
├── 6.ipynb       # Comparación de resultados
├── 7.ipynb       # Conclusiones finales
│
├── requirements.txt        # Librerías necesarias para ejecutar el TP
│
└── README.md               # Este archivo
```

## Contenido

- **data.py**: carga el dataset, realiza el preprocesamiento necesario y separa los datos en conjuntos de *train* y *test*.
- **knn.py**, **tree.py**, **random_forest.py**: contienen las clases e implementaciones desarrolladas a mano para cada modelo.
- **notebooks**: cada punto del trabajo práctico fue resuelto en un notebook distinto, con visualizaciones, explicaciones y resultados.
- **requirements.txt**: lista de dependencias necesarias para ejecutar los scripts y notebooks.

## Requisitos

Para instalar las dependencias necesarias:

```bash
pip install -r requirements.txt
```

## Ejecución

1. Ejecutar `data.py` para generar los conjuntos de entrenamiento y prueba.
2. Abrir en orden los notebooks para reproducir los resultados de cada punto.
3. Los módulos `knn.py`, `tree.py` y `random_forest.py` son importados y utilizados en los notebooks adicionales.

## Autores

**Mora Maurizio**, **Juan Bautista Gramaglia** y **Facundo F. Criscuolo**
