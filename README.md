# ISAC Scouting – Player Recommendation (Liga MX / Club América)

Dashboard interactivo en **Streamlit** para analizar rendimiento de equipo y jugadores con datos de **StatsBomb (Events v8)**, estandarizados **per90**, con **normalización por posición (Z-scores)** y un **Score de Eficiencia** (ofensivo + defensivo – penalización por pérdidas). La herramienta también sugiere **reemplazos** en la liga con mejor ajuste posicional y de perfil.

> Proyecto para el Hackatón ISAC – Player Recommendation.

---

## ✨ ¿Qué hace la herramienta?

- Integra **Events v8** de StatsBomb (pases, carries, tiros, duelos, presiones, OBV, etc.).
- Normaliza métricas **por 90 min** para comparabilidad.
- (Opcional) **Estandariza por posición** (Z-scores) para comparar jugadores dentro de su rol.
- Calcula un **Score de Eficiencia** configurable (of + def – turnovers).
- Arma **benchmarks** de liga y **radiales** por jugador.
- Construye la **red de pases (PageRank)** del equipo con grosor de aristas por volumen de conexión.
- Genera **recomendaciones** de jugadores (scouting) como posibles sustitutos por posición + perfil.
- Soporta carga **ligera** (parquet filtrado) o **completa** (todas las columnas/partidos).

---

## 🧮 Métrica de Eficiencia (resumen)

La **Eficiencia** sintetiza el impacto de un jugador combinando contribuciones ofensivas y defensivas, penalizando pérdidas:

\[
\text{Eficiencia}_i \;=\; 
\underbrace{\sum_k w_k^{(of)} \cdot \text{métrica}^{(per90)}_{ik}}_{\text{bloque ofensivo}}
\;+\;
\underbrace{\sum_m w_m^{(def)} \cdot \text{métrica}^{(per90)}_{im}}_{\text{bloque defensivo}}
\;-\;
\lambda \cdot \text{turnovers}_i^{(per90)}
\]

- **per90:** todas las métricas se llevan a base 90 minutos.  
- **Z-score por posición (opcional):** si activas la normalización, primero se z-estandariza cada métrica dentro del “pool” de jugadores **de la misma posición** (evita sesgos por rol).
- **Pesos (`w`) y λ:** configurables (por defecto equilibrados).  
- **Turnovers:** pérdidas no forzadas/acciones negativas (ej. dispossessions, miscontrols, pases fallados en zonas críticas, etc., según tus columnas disponibles).
- **Porteros:** por su rol atípico, la Eficiencia puede adaptarse usando mayor peso a **shot-stopping**, **xG on target faced**, **cross claims**, distribución, etc. (ya dejaste ganchos para un set de pesos específico de GK si lo decides).

> Nota: Las columnas exactas se resuelven con *fallbacks* en `analytics_helpers.py` para tolerar variaciones del feed (nombres alternativos, presencia/ausencia de OBV, etc.).

---

## 🗂️ Tabs de la aplicación

### 1) **Inicio**
- Resumen del objetivo del dashboard y cómo usar los filtros.
- Contexto del **Score de Eficiencia** (qué mide, por qué per90, cuándo activar Z-scores).
- Enlaces rápidos a documentación/credenciales (si aplica).

### 2) **Club / Equipo**
- Rendimiento agregado del equipo (por torneo o rango de partidos).
- KPIs por fase (con/sin balón), tendencias, **per90** del equipo.
- Filtros: competencia, rival, fecha, torneo.

### 3) **Roster**
- Tarjetas de cada jugador con **minutos**, **Eficiencia** y **posición** (etiquetas con iniciales PN/PA).  
- **Al hacer clic** en una tarjeta se muestran **debajo** los **prospectos** que pueden suplirlo (misma/similar posición, Eficiencia alta, buen ajuste de perfil).  
- Scroll horizontal para revisar rápido todo el plantel (optimizado para parquet ligero).

### 4) **Jugadores (Perfil & Radiales)**
- Vista por jugador: **radar** de métricas clave (of/def), heatmaps (si están disponibles), tabla per90.
- Benchmark vs **media de su posición** en la liga (cruza RAW vs Z si activas normalización).
- Detalle por partido y acumulado.

### 5) **Comparativa RAW vs Z (por posición)**
- Comparación lado a lado: métricas **brutas per90** vs **Z-score por posición**.
- Útil para detectar perfiles **inflados por rol** vs **realmente diferenciales** dentro de su posición.

### 6) **Red de Pases (PageRank)**
- **Grafo de pases** del equipo: nodos (jugadores) y aristas (conexiones).
- **Grosor de línea** ∝ volumen de pases entre dos jugadores.
- **PageRank** identifica hubs/puentes de circulación.
- Filtro de **umbral mínimo** de conexiones y **Top-N** a mostrar.

### 7) **Scouting / Recomendaciones**
- **Buscador** de reemplazos por posición con **Eficiencia** alta y perfil estadístico compatible.
- Ranking de **mejor ajuste** (posición, pie, uso de balón, contribución of/def, métricas OBV si disponibles).
- Descarga de shortlist (CSV) para trabajo posterior.

### 8) **Configuración**
- Parámetros de Eficiencia: **pesos of/def** y **λ** (penalización).
- Activar/Desactivar **normalización por posición**.
- Selección de dataset: **parquet ligero** vs **dataset completo**.
- Paths de logos, assets y caché.

---

## 🧱 Estructura (sugerida)
