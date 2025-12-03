# Tarea 3: Clasificación de Imágenes Satelitales con CNN

Este repositorio contiene la implementación de una Red Neuronal Convolucional (CNN) utilizando *PyTorch* para la clasificación de imágenes satelitales en 5 categorías (Lagos, Cultivos, Bosques, etc.), cumpliendo con los requisitos de la evaluación.

## 📋 Estructura del proyecto

* *Imagenes/*: Contiene el dataset organizado en las 5 clases.
* *scripts/*: Contiene el código fuente de la solución.
    * tarea3_p2.py: Script principal que entrena el modelo base y el modelo con Dropout.

## 🚀 Instrucciones de instalación

pasos exactos para configurar el entorno y ejecutar la tarea:

### 1. Clonar el repositorio
bash
git clone https://github.com/Kuttyxo/IA
cd IA


### 2. Crear y activar el entorno virtual

Es importante usar un entorno virtual para aislar las librerías.

*En Windows:*
powershell
python -m venv venv
venv\Scripts\activate


(En Mac/Linux: source venv/bin/activate)

### 3. Instalar Dependencias
comando para instalar torch, matplotlib y otras librerías necesarias.

bash
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org torch torchvision matplotlib scikit-learn pandas numpy


## ▶️ Ejecución de la tarea

Una vez instalado todo, dirígete a la carpeta de scripts y ejecuta el archivo principal:

bash
cd scripts
python tarea3_p2.py


### 📊 Resultados esperados

El script realizará lo siguiente automáticamente:

1. Cargará el dataset y lo dividirá en Entrenamiento (70%), Validación (15%) y Prueba (15%).
2. Entrenará el *Modelo Base* (CNN estándar).
3. Entrenará el *Modelo con Dropout* (CNN con regularización).
4. Generará y guardará los gráficos comparativos de Loss y Accuracy en la carpeta actual (.png).