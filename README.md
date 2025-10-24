# ISAC Scouting – Player Recommendation (Liga MX / Club América)

Dashboard interactivo en **Streamlit** para analizar el rendimiento del equipo y de los jugadores con datos de **StatsBomb (Events v8)**.  
Los datos se transforman en métricas **por 90 minutos (per90)**, con la posibilidad de **normalizar por posición (Z-scores)** para comparar jugadores dentro de su rol, y calcular una **Eficiencia OBV** que resume el impacto neto ofensivo y defensivo de cada jugador.

> Proyecto desarrollado para el **Hackatón ISAC – Player Recommendation**.

---

## ✨ ¿Qué hace la herramienta?

- Integra datos de **StatsBomb Events v8** (pases, carries, tiros, duelos, presiones, OBV, etc.).
- Calcula métricas **per90** para comparabilidad entre jugadores con distintas cargas de minutos.
- Calcula una **Eficiencia basada en OBV (On-Ball Value)**, que mide el impacto neto de un jugador:
  - **OBV ofensivo** (probabilidad de anotar generada).
  - **OBV defensivo** (probabilidad de recibir reducida o aumentada).
- Permite **normalizar la eficiencia por posición** (Z-score) para comparar jugadores solo dentro de su rol (GK, DEF, MID, FWD).
- Genera **radiales**, **redes de pases (PageRank)** y **recomendaciones de reemplazo** en la liga según rendimiento.
- Permite elegir entre **dataset completo** o **parquet filtrado** para optimizar rendimiento.

---

## 🧮 Métrica de Eficiencia (resumen técnico)

La **Eficiencia OBV** se basa en la diferencia entre el valor ofensivo y el valor en contra que un jugador aporta, ambos ajustados a 90 minutos de juego.

$$
\mathrm{efficiency_{raw,i}} = \mathrm{obv_{off,per90,i}} - \mathrm{obv_{def,per90,i}}
$$

Donde:

- **obv_off_per90** = suma de `obv_for_net` del jugador, escalada a 90 minutos.  
- **obv_def_per90** = suma de `obv_against_net` del jugador, escalada a 90 minutos.  
- Si existe la columna `obv_total_net` y se activa el parámetro `use_total_net_direct=True`, entonces:

$$
\mathrm{efficiency_{raw}} = \mathrm{obv_{total,net}} \times \frac{90}{\mathrm{minutos}}
$$

Luego se aplica, si está habilitado, una **normalización posicional** mediante un **Z-score** dentro del grupo de posición (GK, DEF, MID, FWD):

$$
\mathrm{efficiency_{pos,z}} = 
\frac{
\mathrm{efficiency_{raw}} - \mu_{\text{posición}}
}{
\sigma_{\text{posición}}
}
$$

El valor final mostrado depende del parámetro `normalize_by_position`:

- Si `True`:

$$
\mathrm{efficiency} = \mathrm{efficiency_{pos,z}}
$$

*(Z-score dentro del grupo posicional)*

- Si `False`:

$$
\mathrm{efficiency} = \mathrm{efficiency_{raw}}
$$

*(valor absoluto en unidades OBV per90)*

Además, se calcula el **percentil posicional** (`efficiency_pos_pct`) para ubicar al jugador dentro de su grupo de posición.

> No se emplean pesos personalizados ni penalizaciones explícitas por pérdidas; las acciones negativas ya están **implícitamente reflejadas** en el OBV, que mide el cambio esperado en la probabilidad de anotar o conceder en la posesión siguiente.

---
## 🗂️ Tabs de la aplicación

### 1) **Inicio**
- Presentación general de la herramienta.
- Explicación del objetivo: evaluar rendimiento y buscar reemplazos con base en Eficiencia OBV.
- Breve descripción de la métrica y cómo interpretar el puntaje.

### 2) **Club / Equipo**
- Rendimiento agregado del equipo (por torneo o rango de partidos).
- KPIs de ataque, defensa y transición basados en OBV per90.
- Filtros: competencia, rival, temporada o torneo.

### 3) **Roster**
- Muestra tarjetas por jugador con **minutos jugados**, **eficiencia OBV** y **posición**.
- **Click** en un jugador → muestra **recomendaciones** de sustitutos con eficiencia alta y misma posición.
- Disposición **horizontal con scroll** para fácil exploración.

### 4) **Jugadores (Perfil & Radiales)**
- Visualización individual: **radar** de métricas clave ofensivas y defensivas.
- Comparativa vs. **media por posición** (RAW o Z-score).
- Tablas de métricas per90 y rendimiento por partido.

### 5) **Comparativa RAW vs Z (por posición)**
- Compara valores **absolutos (per90)** y **normalizados (Z-score)** por posición.
- Útil para detectar jugadores que destacan dentro de su rol.

### 6) **Red de Pases (PageRank)**
- Grafo de conexiones de pases del equipo.
- **Nodos** = jugadores, **aristas** = relaciones de pase.  
  - Grosor ∝ cantidad de pases.  
  - Tamaño ∝ centralidad de PageRank.
- Ajuste de umbral mínimo de conexiones y top-N a mostrar.

### 7) **Scouting / Recomendaciones**
- Busca jugadores de la liga con **eficiencia alta** y **perfil posicional compatible**.
- Ranking ordenado por eficiencia (Z o RAW).
- Descarga de shortlist en CSV.

### 8) **Configuración**
- Parámetros principales:
  - Activar/desactivar **normalización por posición**.
  - Fijar **minutos mínimos** (default 270).
  - Seleccionar **dataset (parquet filtrado o completo)**.
  - Cambiar **logos o rutas de datos**.

---

## 🧱 Estructura sugerida del proyecto
