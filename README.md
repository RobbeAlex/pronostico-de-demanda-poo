# Sistema de Pronóstico de Demanda (Ensemble & Clustering)

Este proyecto implementa un sistema robusto de pronóstico de demanda utilizando técnicas avanzadas de Machine Learning y Series Temporales. El sistema está diseñado bajo el paradigma de Programación Orientada a Objetos (POO) para garantizar modularidad y escalabilidad.

## 🚀 Características Principales

El sistema utiliza un enfoque de **Ensamble (Promedio de Modelos)** para mejorar la precisión de las predicciones, combinado con una segmentación previa de series.

1.  **Ingeniería de Datos**: Carga, limpieza y transformación (Melt) de datos históricos.
2.  **Clustering (Segmentación)**:
    * Reducción de dimensionalidad con **PCA**.
    * Agrupamiento de series similares (Producto-Cliente) mediante **K-Means**.
3.  **Modelos de Pronóstico (Ensemble)**:
    * **Prophet**: Para capturar tendencias y estacionalidades complejas.
    * **SARIMA**: Para patrones estacionales y autocorrelación estadística.
    * **Random Forest**: Para capturar relaciones no lineales.
4.  **Visualización**: Generación automática de gráficas para las series más relevantes.

## 📋 Requisitos Previos

* Python 3.8+
* Se recomienda usar un entorno virtual.

## 🛠️ Instalación

1.  Clona el repositorio:
    ```bash
    git clone [https://github.com/TU_USUARIO/forecast-ensemble-system.git](https://github.com/TU_USUARIO/forecast-ensemble-system.git)
    cd forecast-ensemble-system
    ```

2.  Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
## 📊 Diagrama de Flujo
El sistema sigue un flujo lineal estructurado en 4 etapas principales, coordinadas por un orquestador central.

```mermaid
flowchart TD
    Start([Inicio]) --> Config[Cargar Configuración]
    Config --> InitMotor[Inicializar MotorPronostico]

    subgraph "1. Gestión de Datos"
        InitMotor --> LoadData[GestorDatos: Cargar y Procesar CSV]
        LoadData --> MeltData[Transformar a Series Temporales - Melt]
        MeltData --> Filter[Filtrar Top N Productos/Clientes]
    end

    subgraph "2. Clustering (No Supervisado)"
        Filter --> PrepCluster[Preparar Pivot Table]
        PrepCluster --> Scale[Escalar Datos]
        Scale --> PCA[Reducción de Dimensión - PCA]
        PCA --> KMeans[Agrupamiento - KMeans]
        KMeans --> AssignCluster[Asignar Etiquetas de Cluster]
    end

    subgraph "3. Bucle de Pronóstico (Iterativo)"
        AssignCluster --> LoopStart{¿Hay más series?}
        LoopStart -- Sí --> CheckMin[Verificar Min Datos Entrenamiento]
        CheckMin -- Insuficiente --> LoopStart
        CheckMin -- Suficiente --> TrainModels

        subgraph "Ensamble de Modelos"
            TrainModels[Entrenar Modelos] --> M1[Prophet]
            TrainModels --> M2[SARIMA]
            TrainModels --> M3[Random Forest]
            
            M1 --> Pred1[Predicción 1]
            M2 --> Pred2[Predicción 2]
            M3 --> Pred3[Predicción 3]
            
            Pred1 & Pred2 & Pred3 --> Avg[Promedio - Ensemble]
        end
        
        Avg --> Store[Guardar Resultado Serie]
        Store --> LoopStart
    end

    LoopStart -- No --> Consolidate[Consolidar Resultados]
    Consolidate --> MergeClusters[Unir con Etiquetas de Cluster]
    MergeClusters --> ExportCSV[Exportar CSV Final]

    subgraph "4. Visualización"
        ExportCSV --> VizInit[VisualizadorPronosticos]
        VizInit --> LoadRes[Cargar Resultados]
        LoadRes --> GenPlots[Generar Gráficas Top Series]
        GenPlots --> SavePNG[Guardar PNGs]
    end

    SavePNG --> End([Fin])
```
## 🧩 Diagrama de Clases UML

```mermaid
classDiagram
    %% Clase de Configuración
    class Configuracion {
        +STR ARCHIVO_ENTRADA
        +STR ARCHIVO_SALIDA
        +INT TOP_PRODUCTOS
        +INT TOP_CLIENTES
        +TUPLE SARIMA_ORDER
        +INT RF_ESTIMADORES
        +INT K_CLUSTERS
    }

    %% Gestión de Datos
    class GestorDatos {
        +DataFrame df_procesado
        +List top_productos
        +List top_clientes
        +cargar_y_procesar()
        +filtrar_top_series()
        +obtener_datos_para_clustering() DataFrame
    }

    %% Clustering
    class AnalizadorClusters {
        +ejecutar_analisis(df_pivot) DataFrame
    }

    %% Modelos (Polimorfismo)
    class ModeloPronostico {
        <<Abstract>>
        +object modelo
        +str nombre
        +entrenar(df_train)*
        +predecir(fechas_futuras)*
    }

    class ModeloProphet {
        +entrenar(df_train)
        +predecir(fechas_futuras)
    }

    class ModeloSARIMA {
        +entrenar(df_train)
        +predecir(fechas_futuras)
    }

    class ModeloRandomForest {
        +entrenar(df_train)
        +predecir(fechas_futuras)
        -_crear_features(df)
    }

    %% Orquestador
    class MotorPronostico {
        +GestorDatos gestor
        +AnalizadorClusters clusterer
        +List resultados
        +ejecutar()
    }

    %% Visualización
    class VisualizadorPronosticos {
        +str archivo
        +DataFrame df
        +cargar_resultados()
        +graficar_top_series(n)
    }

    %% Relaciones
    ModeloPronostico <|-- ModeloProphet : Herencia
    ModeloPronostico <|-- ModeloSARIMA : Herencia
    ModeloPronostico <|-- ModeloRandomForest : Herencia

    MotorPronostico *-- GestorDatos : Compone
    MotorPronostico *-- AnalizadorClusters : Compone
    MotorPronostico ..> ModeloPronostico : Usa (Instancia Dinámicamente)
    MotorPronostico ..> Configuracion : Lee Parámetros

    VisualizadorPronosticos ..> Configuracion : Lee Rutas
```
## 📂 Estructura de Datos de Entrada

El script espera un archivo CSV en la raíz (o configurado en la clase `Configuracion`) con el nombre:
`Datos Históricos de Pedidos y de Facturación.csv`

El formato esperado debe contener columnas descriptivas (Producto, Cliente) y columnas de fechas en formato ancho (e.g., `23-Jan`, `23-Feb`...) que el sistema transformará automáticamente.

## ▶️ Ejecución

Para ejecutar el pipeline completo (Carga -> Clustering -> Pronóstico -> Exportación):

```bash
python src/main.py
```
## 📂 Estructura del Repositorio
```Plaintext
forecast-ensemble-system/
│
├── data/
│   ├── inputs/     # Datos crudos
│   ├── outputs/    # CSV y gráficas generadas
│   └── docs/       # diagrama de flujo y diagrama UML
│
├── src/
│   ├── __init__.py
│   └── main.py     # Codigo de pronostico de demanda poo
│
├── .gitignore      # Archivo para excluir archivos temporales y datos
├── LICENSE         # Licencia de uso (MIT recomendada)
├── README.md       # Documentación del proyecto
└── requirements.txt # Lista de librerías necesarias
