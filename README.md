# Statsbomb_ligamx
Repositorio para el hackaton del ISAC ITAM con análisis de rendimiento de club américa


## ⚽️ Métrica de Eficiencia Compuesta (Composite Efficiency v1)

La **Eficiencia Compuesta v1** es un indicador unificado del rendimiento de cada jugador, calculado a partir de los eventos individuales (StatsBomb Events v8).  
A diferencia de métricas que dependen de modelos de Expected Goals (xG) o de *On-Ball Value (OBV)*, esta métrica se basa en **acciones observables** y **contribuciones directas** dentro del juego, normalizadas por 90 minutos y ponderadas según la **posición del jugador**.

---

### 🔍 1. Enfoque general

Para cada jugador, se calculan una serie de **features por 90 minutos** (`per90`):

| Categoría | Variables principales |
|------------|------------------------|
| **Ofensivas** | `xg`, `shots`, `key_passes`, `prog_passes`, `carries_prog`, `box_entries` |
| **Defensivas** | `pressures`, `recoveries`, `tib` (tackles + interceptions + blocks), `clearances`, `duels_won` |
| **Negativas (pérdidas)** | `turnovers_raw` (miscontrols, dispossessed, dribbles fallidos y pases incompletos) |
| **Distribución / Control** | `pass_attempts`, `pass_completed`, `pass_completion` |

Cada métrica se normaliza en unidades *por 90 minutos* para garantizar comparabilidad entre jugadores con diferentes cargas de juego:

\[
\text{valor\_per90} = \frac{\text{conteo}}{\text{minutos}} \times 90
\]

---

### ⚙️ 2. Pesos diferenciales por grupo posicional

No todas las acciones tienen el mismo impacto según la posición.  
Por ejemplo, los delanteros son evaluados principalmente por **productividad ofensiva**, mientras que los defensas lo son por **acciones de recuperación y contención**.  
Por ello, la fórmula de eficiencia utiliza **ponderaciones distintas** para cada grupo:

#### 🧤 Porteros (GK)
- +0.50 × `recoveries` (participación tipo sweeper)
- +0.30 × `clearances`
- +0.80 × (`pass_completion` − 0.75)  _(por encima del promedio 75%)_
- −0.20 × `turnovers_raw`

#### 🧱 Defensas (DEF)
- +0.90 × `tib`  (tackles + interceptions + blocks)
- +0.70 × `clearances`
- +0.60 × `duels_won`
- +0.40 × `recoveries`
- +0.20 × `prog_passes`
- −0.35 × `turnovers_raw`

#### ⚙️ Mediocampistas (MID)
- +0.75 × `prog_passes`
- +0.65 × `carries_prog`
- +0.60 × `key_passes`
- +0.45 × `recoveries`
- +0.35 × `pressures`
- +0.50 × `tib`
- −0.45 × `turnovers_raw`

#### 🎯 Delanteros (FWD)
- +1.20 × `xg`
- +0.70 × `box_entries`
- +0.60 × `key_passes`
- +0.45 × `carries_prog`
- +0.40 × `prog_passes`
- −0.60 × `turnovers_raw`

---

### 🧮 3. Fórmula general

\[
\text{Eficiencia}_i = \sum_j \text{peso}_j \cdot \text{métrica}_{ij}^{(per90)} - \lambda \cdot \text{turnovers}_{i}^{(per90)}
\]

donde:
- \( i \) = jugador  
- \( j \) = acción relevante según la posición  
- \( \lambda \) = penalización proporcional al tipo de acción negativa

---

### 🎯 4. Interpretación

- **Valor alto de eficiencia** → jugador que contribuye consistentemente a la fase dominante de su rol (ataque o defensa) con bajo costo en pérdidas.  
- **Valor medio (≈0)** → participación equilibrada o bajo volumen de acciones.  
- **Valor negativo** → jugador con participación frecuente pero de baja efectividad o con alto número de pérdidas.

La escala no representa goles esperados ni goles anotados, sino **impacto relativo por 90 minutos dentro de su grupo posicional**.

---

### 📈 5. Ventajas del modelo

- **Posición-aware:** adapta automáticamente las ponderaciones según el rol del jugador (`GK`, `DEF`, `MID`, `FWD`).
- **Independiente de modelos predictivos:** se basa en conteos reales y no requiere un modelo de probabilidad como xG o VAEP.
- **Interpretable:** cada componente de la fórmula es tangible y puede visualizarse o auditarse.
- **Comparable entre jugadores:** normalización por 90 minutos permite comparar rendimientos independientemente del tiempo jugado.
- **Flexible:** las ponderaciones pueden ajustarse según la liga, el club o la filosofía de juego.

---

### 📊 6. Uso en el dashboard

En la pestaña **Roster**, la métrica `efficiency` se utiliza para:
- Ordenar el **roster del equipo base** (ej. Club América) por rendimiento.
- Identificar **jugadores candidatos** de otros equipos dentro del **mismo grupo posicional**.
- Priorizar reemplazos con **posición fina similar** (`position_role`, ej. *Right Center Midfield*).

Esta métrica permite comparar objetivamente jugadores en contextos similares, equilibrando producción ofensiva, aportes defensivos y control del balón.
