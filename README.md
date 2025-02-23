# Proyecto final CIP

Dataset: [SIVIGILA](https://medata.gov.co/dataset/1-026-22-000146 "SIVIGILA")

Conjunto de datos que recopila los registros de casos de Intento de Suicidio reportados al SIVIGILA (Sistema Nacional de Vigilancia en Salud Pública).

El siguiente proyecto de final de seminario de ciencias de datos tiene como objetivo desarrollar modelos de machine learning para predecir el condición final (con_fin_) de un intento de sucicidio basado en los diversos factores.

## Contenido

- [Variables del dataset](#variables-del-dataset)
    - [Factores generales](#factores-generales)
    - [Factores desencadenantes](#factores-desencadenantes)
    - [Factores de riesgo](#factores-de-riesgo)
- [Instalación](#instalación)
- [Generar modelos](#generar-modelos)
- [Correr API](#correr-api)
- [Construido con](#construido-con)
- [Autor](#autor)

## Variables del dataset 📋

Las variables de este dataset se dividen en tres categorías: factores generales, factores desencadenantes y factores de riesgo. A continuación se describen cada una de las variables:

### Factores generales 📋

| Nombre       | Descripción                                         | Typo         |      Valor               |
| ------------ | ------------                                        | ------------ | ------------             |
| con_fin_     |  Condición final                                    |  Number      | 1: Vivo 0: Muerto        |
| edad_        |  Edad de la victima                                 |  Number      | De 5 a 91 años           |
| sexo_        |  Sexo de la victima                                 |  Number      | 1: Masculino 0: Femenino |
| inten_prev   |  El paciente ha tenido intentos previos de suicidio |  Number      | 1: Sí 0: No              |

### Factores desencadenantes 📋

| Nombre             | Descripción                                  | Typo         | Valor        |
| ------------       | ------------                                 | ------------ | ------------ |
| prob_parej         | Conflictos con la pareja o expareja          |  Number      | 1: Sí 0: No  |
| enfermedad_cronica | Enfermedad crónica dolorosa o discapacitante |  Number      | 1: Sí 0: No  |
| prob_econo         | Problemas económicos                         |  Number      | 1: Sí 0: No  |
| muerte_fam         | Muerte de un familiar                        |  Number      | 1: Sí 0: No  |
| esco_educ          | Escolar/educativa                            |  Number      | 1: Sí 0: No  |
| prob_legal         | Problemas jurídicos                          |  Number      | 1: Sí 0: No  |
| suici_fm_a         | Suicidio de un familiar o amigo              |  Number      | 1: Sí 0: No  |
| maltr_fps          | Maltrato físico/Psicológico/Sexual           |  Number      | 1: Sí 0: No  |
| prob_labor         | Problemas laborales                          |  Number      | 1: Sí 0: No  |
| prob_famil         | Problemas familiares                         |  Number      | 1: Sí 0: No  |

### Factores de riesgo 📋

| Nombre             | Descripción                                            | Typo         | Valor        |
| ------------       | ------------                                           | ------------ | ------------ |
| prob_consu         | Consumo de SPA(Sustancias Psicoactivas)                |  Number      | 1: Sí 0: No  |
| hist_famil         | Antecedentes familiares de conducta                    |  Number      | 1: Sí 0: No  |
| idea_suici         | Ideación suicida persistente                           |  Number      | 1: Sí 0: No  |
| plan_suici         | Plan organizado de suicidio                            |  Number      | 1: Sí 0: No  |
| antec_tran         | Antecedente de trastorno psiquiátrico                  |  Number      | 1: Sí 0: No  |
| tran_depre         | Trastorno psiquiátrico (solo si antec_tran)            |  Number      | 1: Sí 0: No  |
| trast_personalidad | Trastorno personalidad (solo si antec_tran)            |  Number      | 1: Sí 0: No  |
| trast_bipolaridad  | Trastorno bipolar (solo si antec_tran)                 |  Number      | 1: Sí 0: No  |
| esquizofre         | Esquizofrenia (solo si antec_tran)                     |  Number      | 1: Sí 0: No  |
| antec_v_a          | Antecedentes de violencia o abuso (solo si antec_tran) |  Number      | 1: Sí 0: No  |
| abuso_alco         | Abuso de alcohol (solo si antec_tran)                  |  Number      | 1: Sí 0: No  |

'prob_consu', 'hist_famil', 'idea_suici', 'plan_suici', 'antec_tran', 'tran_depre', 'trast_personalidad', 'trast_bipolaridad', 'esquizofre','antec_v_a', 'abuso_alc
## Instalación 🔧

1. Configurar el entorno virtual

```bash
python3 -m venv env
source env/bin/activate
```
2. Instalar las dependencias

```bash
pip install -r requirements.txt
```

## Generar modelos 🚀

Logistic Regression

```bash
python LogisticRegression.py
```
Random Forest

```bash
python RandomForest.py
```

## Correr API 🚀

```bash
fastapi dev main.py
```

## Construido con 🛠️

* [Python](https://www.python.org/) - El lenguaje de programación usado.
* [Pandas](https://pandas.pydata.org/) - Librería de manipulación y análisis de datos.
* [Numpy](https://numpy.org/) - Librería para cálculos numéricos.
* [Matplotlib](https://matplotlib.org/) - Librería para la generación de gráficos.
* [Seaborn](https://seaborn.pydata.org/) - Librería para la visualización de datos.
* [Scikit-learn](https://scikit-learn.org/stable/) - Librería para machine learning.
* [Joblib](https://joblib.readthedocs.io/en/latest/) - Librería para guardar y cargar modelos.
* [FastAPI](https://fastapi.tiangolo.com/) - Framework para la creación de APIs.

## Autor ✒️

* **Fabian Franco** - *Investigación y Desarrollo* - [Fabian Franco](https://github.com/fabian-franco)