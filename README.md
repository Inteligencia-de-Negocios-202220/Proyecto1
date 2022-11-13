# Proyecto 1 - BI - Grupo 18
Proyecto 1 realizado por:
- Gabriela Cagua Bolívar - 201812944
- Juan Andrés Méndez Galvis -  20181580138
- Juan Andrés Romero Colmenares - 202013449

Notas:
* Los Joblibs de los modelos resultantes no se pueden cargar a GitHub debido a su peso, por lo tanto, se han subido a un zip en Google Drive. El link se encuentra [aquí](https://drive.google.com/file/d/1iE5CSCAigI9tZsf4mdyk-7I5lV3tHO7A/view?usp=sharing).

Bienvenidos a la wiki del Proyecto 1 del grupo 18 de Inteligencia de Negocios-202220!


# Código
## Entrega 1
En general, se trabajaron 2 archivos .py con el código de los modelos y un notebook .ipynb, el cual sirve para el análisis posterior de los modelos.
1. Un archivo [train.py](https://github.com/Inteligencia-de-Negocios-202220/Proyecto1/blob/main/model_training/train.py), el cual contiene los procesos realizados para encontrar los mejores modelos usando GridSearchCV.
2. Un archivo [best_models.py](https://github.com/Inteligencia-de-Negocios-202220/Proyecto1/blob/main/model_training/best_models.py) que contiene los mejores modelos encontrados incluida su persistencia en archivos .joblib
3. [Notebook de Análisis](https://github.com/Inteligencia-de-Negocios-202220/Proyecto1/blob/main/docs/Proyecto1NLP.ipynb)

Notas: 
* El train.py depende del archivo logger.py, debido a que usó un logger personalizado para registrar la salida del script.
* Los Joblibs de los modelos resultantes no se pueden cargar a GitHub debido a su peso, por lo tanto, se han subido a un zip en Google Drive. El link se encuentra [aquí](https://drive.google.com/file/d/1iE5CSCAigI9tZsf4mdyk-7I5lV3tHO7A/view?usp=sharing).

## Entrega 2
Para la entrega dos solo se trabajó desde el API directamente, pero también incluimos un archivo que contiene la construcción del pipeline utilizado para el desarrollo de la aplicación. En general hay:
1. Un archivo [main.py](https://github.com/Inteligencia-de-Negocios-202220/Proyecto1/blob/main/main.py), el cual es el archivo principal que ejecuta el API. (Para correrlo solo hace falta llamarlo desde python `python main.py`)
2. Un archivo [logistic_regression_pipeline.py](https://github.com/Inteligencia-de-Negocios-202220/Proyecto1/blob/main/logistic_regression_pipeline.py) que contiene la pipeline desarrollada para la entrega.

Notas Adicionales: El proyecto contiene varias dependencias, las cuales necesitan ser instalarlas por medio del comando `pip install -r requirements.txt`
