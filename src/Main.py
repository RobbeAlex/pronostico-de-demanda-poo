import pandas as pd
import numpy as np
import warnings
import logging
from abc import ABC, abstractmethod

# Librerías de Machine Learning y Estadística
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# ==========================================
# 0. CONFIGURACIÓN Y UTILIDADES
# ==========================================

# Silenciar advertencias para mantener la consola limpia
warnings.filterwarnings("ignore")
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

class Configuracion:
    """
    Clase centralizada para parámetros de configuración.
    """
    ARCHIVO_ENTRADA = "Datos Históricos de Pedidos y de Facturación.csv"
    ARCHIVO_SALIDA = "Pronostico_Demanda_Top10x5_ENSEMBLE_POO.csv"

    # Columnas
    COL_METRICA = 'Pedido_Piezas'
    COLS_ID = ['Producto_Descripcion', 'Cliente_Descripcion']

    # Parámetros de Filtrado
    TOP_PRODUCTOS = 10
    TOP_CLIENTES = 5

    # Parámetros de Pronóstico
    FECHA_CORTE = '2025-01-01'
    PERIODOS_FUTUROS = 12
    MIN_DATOS_ENTRENAMIENTO = 12
    PAIS_FESTIVOS = 'MX'

    # Hiperparámetros Modelos
    SARIMA_ORDER = (1, 1, 1)
    SARIMA_SEASONAL = (1, 1, 1, 12)
    RF_ESTIMADORES = 100
    RF_DEPTH = 8

    # Clustering
    PCA_COMPONENTES = 2
    K_CLUSTERS = 4

# ==========================================
# 1. GESTIÓN DE DATOS (ENCAPSULAMIENTO)
# ==========================================

class GestorDatos:
    """
    Clase responsable de Cargar, Limpiar y Preparar los datos.
    """
    def __init__(self):
        self.df_procesado = None
        self.top_productos = []
        self.top_clientes = []

    def cargar_y_procesar(self):
        print("--- [1/4] Cargando y procesando datos ---")
        try:
            # Intentar carga con utf-8, fallback a latin1
            try:
                df_raw = pd.read_csv(Configuracion.ARCHIVO_ENTRADA)
            except UnicodeDecodeError:
                df_raw = pd.read_csv(Configuracion.ARCHIVO_ENTRADA, encoding='latin1')

            # Extraer columnas de fecha (asumiendo estructura fija del notebook original: cols 3 a 84)
            # Nota: Ajusta los índices si el archivo cambia de estructura
            cols_fechas = df_raw.columns[3:84]

            # Melt (Transformar de Ancho a Largo)
            df_melt = df_raw.melt(
                id_vars=Configuracion.COLS_ID,
                value_vars=cols_fechas,
                var_name='Fecha_str',
                value_name=Configuracion.COL_METRICA
            )

            # Conversión de Fechas y Limpieza
            df_melt['Fecha'] = pd.to_datetime(df_melt['Fecha_str'], format='%y-%b', errors='coerce')
            df_melt[Configuracion.COL_METRICA] = df_melt[Configuracion.COL_METRICA].fillna(0)

            # Agregación base
            self.df_procesado = df_melt.groupby(
                Configuracion.COLS_ID + ['Fecha']
            )[Configuracion.COL_METRICA].sum().reset_index()

            print(f"✓ Datos procesados: {len(self.df_procesado)} registros.")

        except Exception as e:
            raise Exception(f"Error crítico cargando datos: {e}")

    def filtrar_top_series(self):
        """Filtra el DataFrame para quedarse solo con los Top Productos y Clientes."""
        if self.df_procesado is None:
            raise ValueError("Datos no cargados.")

        ranking_prods = self.df_procesado.groupby('Producto_Descripcion')[Configuracion.COL_METRICA].sum()
        self.top_productos = ranking_prods.nlargest(Configuracion.TOP_PRODUCTOS).index.tolist()

        ranking_clies = self.df_procesado.groupby('Cliente_Descripcion')[Configuracion.COL_METRICA].sum()
        self.top_clientes = ranking_clies.nlargest(Configuracion.TOP_CLIENTES).index.tolist()

        # Filtrar
        mask = (
                self.df_procesado['Producto_Descripcion'].isin(self.top_productos) &
                self.df_procesado['Cliente_Descripcion'].isin(self.top_clientes)
        )
        self.df_procesado = self.df_procesado[mask].copy()
        print(f"✓ Filtrado: Top {Configuracion.TOP_PRODUCTOS} Productos y Top {Configuracion.TOP_CLIENTES} Clientes.")

    def obtener_datos_para_clustering(self):
        return self.df_procesado.pivot_table(
            index=Configuracion.COLS_ID,
            columns='Fecha',
            values=Configuracion.COL_METRICA,
            fill_value=0
        )

# ==========================================
# 2. CLUSTERING
# ==========================================

class AnalizadorClusters:
    def ejecutar_analisis(self, df_pivot):
        print("--- [2/4] Ejecutando Clustering (PCA + KMeans) ---")
        if df_pivot.empty:
            print("⚠️ No hay datos suficientes para clustering.")
            return None

        # Escalar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_pivot)

        # PCA
        pca = PCA(n_components=Configuracion.PCA_COMPONENTES)
        X_pca = pca.fit_transform(X_scaled)

        # KMeans
        kmeans = KMeans(n_clusters=Configuracion.K_CLUSTERS, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X_pca)

        # Resultados
        df_res = pd.DataFrame(index=df_pivot.index)
        df_res['Cluster_Grupo'] = clusters
        return df_res.reset_index()

# ==========================================
# 3. MODELOS DE PRONÓSTICO (POLIMORFISMO)
# ==========================================

class ModeloPronostico(ABC):
    """
    Superclase Abstracta. Define la interfaz para todos los modelos.
    """
    def __init__(self):
        self.modelo = None
        self.nombre = "Base"

    @abstractmethod
    def entrenar(self, df_train):
        pass

    @abstractmethod
    def predecir(self, fechas_futuras):
        pass

class ModeloProphet(ModeloPronostico):
    def __init__(self):
        super().__init__()
        self.nombre = "Prophet"

    def entrenar(self, df_train):
        # Prophet requiere columnas 'ds' y 'y'
        df_p = df_train.rename(columns={'Fecha': 'ds', Configuracion.COL_METRICA: 'y'})
        self.modelo = Prophet(seasonality_mode='additive', daily_seasonality=False, weekly_seasonality=False)
        if Configuracion.PAIS_FESTIVOS:
            self.modelo.add_country_holidays(country_name=Configuracion.PAIS_FESTIVOS)
        self.modelo.fit(df_p)

    def predecir(self, fechas_futuras):
        future = pd.DataFrame({'ds': fechas_futuras})
        forecast = self.modelo.predict(future)
        return forecast['yhat'].values

class ModeloSARIMA(ModeloPronostico):
    def __init__(self):
        super().__init__()
        self.nombre = "SARIMA"

    def entrenar(self, df_train):
        series = df_train.set_index('Fecha')[Configuracion.COL_METRICA].asfreq('MS').fillna(0)
        self.modelo = SARIMAX(
            series,
            order=Configuracion.SARIMA_ORDER,
            seasonal_order=Configuracion.SARIMA_SEASONAL,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.result_fit = self.modelo.fit(disp=False)

    def predecir(self, fechas_futuras):
        steps = len(fechas_futuras)
        pred = self.result_fit.get_forecast(steps=steps)
        return pred.predicted_mean.values

class ModeloRandomForest(ModeloPronostico):
    def __init__(self):
        super().__init__()
        self.nombre = "RandomForest"
        self.features = ['year', 'month', 'quarter']

    def _crear_features(self, df_in):
        df = df_in.copy()
        # Asegurar que existe columna fecha
        col_fecha = 'Fecha' if 'Fecha' in df.columns else 'ds'

        df['year'] = df[col_fecha].dt.year
        df['month'] = df[col_fecha].dt.month
        df['quarter'] = df[col_fecha].dt.quarter
        return df

    def entrenar(self, df_train):
        df_rf = self._crear_features(df_train)
        X = df_rf[self.features]
        y = df_rf[Configuracion.COL_METRICA]

        self.modelo = RandomForestRegressor(
            n_estimators=Configuracion.RF_ESTIMADORES,
            max_depth=Configuracion.RF_DEPTH,
            random_state=42
        )
        self.modelo.fit(X, y)

    def predecir(self, fechas_futuras):
        df_future = pd.DataFrame({'ds': fechas_futuras})
        df_future = self._crear_features(df_future)
        return self.modelo.predict(df_future[self.features])

# ==========================================
# 4. ORQUESTADOR PRINCIPAL
# ==========================================

class MotorPronostico:
    """
    Clase principal que coordina los datos, el clustering y los modelos.
    """
    def __init__(self):
        self.gestor = GestorDatos()
        self.clusterer = AnalizadorClusters()
        self.resultados = []

    def ejecutar(self):
        # 1. Preparar Datos
        self.gestor.cargar_y_procesar()
        self.gestor.filtrar_top_series()

        # 2. Clustering
        df_pivot = self.gestor.obtener_datos_para_clustering()
        df_clusters = self.clusterer.ejecutar_analisis(df_pivot)

        # 3. Modelado Iterativo
        print(f"--- [3/4] Iniciando entrenamiento de modelos para series temporales ---")
        df_datos = self.gestor.df_procesado
        grupos = df_datos.groupby(Configuracion.COLS_ID)

        count = 0
        total_grupos = len(grupos)

        for (prod, cli), df_serie in grupos:
            count += 1
            if len(df_serie) < Configuracion.MIN_DATOS_ENTRENAMIENTO:
                continue

            print(f"Procesando {count}/{total_grupos}: {prod[:20]}... - {cli[:20]}...")

            # Definir rango futuro
            ultima_fecha = df_serie['Fecha'].max()
            fechas_futuras = pd.date_range(
                start=ultima_fecha + pd.DateOffset(months=1),
                periods=Configuracion.PERIODOS_FUTUROS,
                freq='MS'
            )

            # Lista Polimórfica de Modelos
            modelos = [ModeloProphet(), ModeloSARIMA(), ModeloRandomForest()]
            predicciones_individuales = []

            for modelo in modelos:
                try:
                    modelo.entrenar(df_serie)
                    preds = modelo.predecir(fechas_futuras)
                    # Evitar negativos
                    preds = np.maximum(preds, 0)
                    predicciones_individuales.append(preds)
                except Exception as e:
                    print(f"  Error en modelo {modelo.nombre}: {e}")
                    predicciones_individuales.append(np.zeros(len(fechas_futuras)))

            # Ensemble (Promedio)
            if predicciones_individuales:
                ensemble_preds = np.mean(predicciones_individuales, axis=0)
            else:
                ensemble_preds = np.zeros(len(fechas_futuras))

            # Guardar resultados
            df_res = pd.DataFrame({
                'Fecha': fechas_futuras,
                'Producto_Descripcion': prod,
                'Cliente_Descripcion': cli,
                'Pronostico_Demanda': ensemble_preds
            })
            self.resultados.append(df_res)

        # 4. Consolidación y Exportación
        print("--- [4/4] Consolidando y guardando resultados ---")
        if self.resultados:
            df_final = pd.concat(self.resultados, ignore_index=True)

            # Unir con información de cluster
            if df_clusters is not None:
                df_final = pd.merge(df_final, df_clusters, on=Configuracion.COLS_ID, how='left')

            df_final.to_csv(Configuracion.ARCHIVO_SALIDA, index=False)
            print(f"✅ Archivo generado exitosamente: {Configuracion.ARCHIVO_SALIDA}")
            print(df_final.head())
        else:
            print("⚠️ No se generaron pronósticos.")

# ==========================================
# EJECUCIÓN DEL SCRIPT
# ==========================================
if __name__ == "__main__":
    app = MotorPronostico()
    app.ejecutar()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class VisualizadorPronosticos:
    def __init__(self, archivo_resultados="Pronostico_Demanda_Top10x5_ENSEMBLE_POO.csv"):
        self.archivo = archivo_resultados
        self.df = None

    def cargar_resultados(self):
        if not os.path.exists(self.archivo):
            print(f"Error: No se encuentra el archivo {self.archivo}")
            return

        self.df = pd.read_csv(self.archivo)
        self.df['Fecha'] = pd.to_datetime(self.df['Fecha'])
        print(f"✓ Resultados cargados: {self.df.shape[0]} registros de pronóstico.")

    def graficar_top_series(self, n=3):
        """Grafica las 'n' primeras combinaciones encontradas en el archivo."""
        if self.df is None:
            return

        # Obtener combinaciones únicas
        combinaciones = self.df[['Producto_Descripcion', 'Cliente_Descripcion']].drop_duplicates()

        # Configurar estilo
        sns.set_theme(style="whitegrid")

        # Graficar las primeras 'n'
        for idx, row in combinaciones.head(n).iterrows():
            prod = row['Producto_Descripcion']
            cli = row['Cliente_Descripcion']

            datos_serie = self.df[
                (self.df['Producto_Descripcion'] == prod) &
                (self.df['Cliente_Descripcion'] == cli)
                ]

            plt.figure(figsize=(12, 6))
            sns.lineplot(data=datos_serie, x='Fecha', y='Pronostico_Demanda', marker='o', linewidth=2.5, color='royalblue')

            # Títulos y Etiquetas
            cluster = datos_serie['Cluster_Grupo'].iloc[0] if 'Cluster_Grupo' in datos_serie.columns else 'N/A'
            plt.title(f'Pronóstico de Demanda: {prod}\nCliente: {cli} (Cluster {cluster})', fontsize=14, fontweight='bold')
            plt.xlabel('Fecha Futura')
            plt.ylabel('Cantidad Pronosticada (Piezas)')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Guardar o Mostrar
            nombre_clean = f"Grafica_{idx}_{prod[:10].strip()}.png".replace(" ", "_")
            plt.savefig(nombre_clean)
            print(f"   -> Gráfica guardada: {nombre_clean}")
            plt.show()

if __name__ == "__main__":
    viz = VisualizadorPronosticos()
    viz.cargar_resultados()
    viz.graficar_top_series(n=5) # Cambia este número para ver más o menos gráficas
