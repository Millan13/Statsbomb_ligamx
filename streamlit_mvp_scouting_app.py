"""
Streamlit MVP – ISAC Hackatón (Player Recommendation)
Autor: Tú (Miguel Millán) — scaffold regenerado por ChatGPT

🔧 Requisitos (requirements.txt)
streamlit>=1.36
pandas>=2.2
numpy>=1.26
plotly>=5.24
mplsoccer>=1.4.0
statsbombpy>=1.13
python-dateutil>=2.9

▶️ Cómo ejecutar
1) Asegura el módulo de utilidades `analytics_helpers.py` (ya está en el proyecto).
2) Ajusta las rutas de logos e input de datos en el sidebar o en las variables por defecto.
3) Ejecuta:  
   `streamlit run streamlit_mvp_scouting_app.py`

🧩 Qué hace este MVP (con datos REALES)
- Lee **varios Parquet** (p. ej., 2023_2024 y 2024_2025) y concatena.
- Selector de **torneo(s)** (Apertura/Clausura). Puedes elegir uno o varios (p. ej., *últimos 4 torneos*).
- KPIs del/los torneos seleccionados: **PJ, G, E, P**, máximos (**goleador, asistidor, GK con más atajadas**).
- **Resumen de estilo** (heurístico v0) y **heatmap** de zonas de gol.
- Pestañas: Inicio (explicativa), Club América (radar + insights), Roster (placeholder para iterar después).

"""

from __future__ import annotations
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# === Helpers propios ===
from analytics_helpers import (
    load_events,
    filter_last_tournament,
    filter_team_tournament,
    compute_kpis_from_matches,
    top_performers,
    infer_style_summary,
    goals_xy_for_heatmap,
    build_pass_network,
    plot_pass_network

)

# Opcional: mplsoccer para dibujar cancha (usamos Plotly como fallback)
try:
    from mplsoccer import Pitch
    _HAS_MPLSOCCER = True
except Exception:
    _HAS_MPLSOCCER = False

# -------------------------------
# Configuración general
# -------------------------------
st.set_page_config(
    page_title="ISAC Scouting – Club América",
    page_icon="⚽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Auth GCS (lee el JSON desde Secrets) ----------
sa_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT"])

def join_gs(prefix: str, *parts: str) -> str:
    """Une rutas para gs:// de forma segura sin romper el esquema."""
    return "/".join([prefix.rstrip("/"), *[p.strip("/") for p in parts]])

def first_col(df: pd.DataFrame, candidates: List[str], default: Optional[str] = None) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return default

# --- LOGOS EN LA CABECERA (locales) ---
# Ruta del archivo actual
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "images")

# Carga de imágenes
logo_ame = os.path.join(IMAGE_PATH, "logo_ame.png")
logo_isac = os.path.join(IMAGE_PATH, "logo_isac.png")

col_logo_left, col_title, col_logo_right = st.columns([0.15, 0.7, 0.15])
with col_logo_left:
    if os.path.exists(logo_ame):
        st.image(logo_ame, width=100)
with col_title:
    st.markdown("""<h1 style='text-align:center;'>ISAC Scouting – Club América</h1>""", unsafe_allow_html=True)
    st.caption("MVP visual con datos reales. Enfocado en objetivos y rúbrica del hackatón.")
with col_logo_right:
    if os.path.exists(logo_isac):
        st.image(logo_isac, width=100)

# -------------------------------
# UI helper (igual que tenías)
# -------------------------------
def donut_kpi(label: str, value: float, total: float, color: str = "#00529F") -> go.Figure:
    remaining = max(total - value, 0)
    fig = go.Figure(go.Pie(
        values=[value, remaining],
        labels=[label, ""],
        hole=0.7,
        textinfo="none",
        sort=False,
        direction="clockwise",
        marker=dict(colors=[color, "#E9ECEF"]),
        showlegend=False,
    ))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=210)
    fig.add_annotation(text=f"<b>{int(value)}</b><br><span style='font-size:12px'>{label}</span>",
                       showarrow=False, font=dict(size=18))
    return fig

# -------------------------------
# Lectura desde GCS (cacheada)
# -------------------------------
def _read_gcs(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path, storage_options={"token": sa_info})
    elif path.lower().endswith(".csv"):
        return pd.read_csv(path, storage_options={"token": sa_info})
    else:
        raise ValueError(f"Extensión no soportada en {path}. Usa .parquet o .csv")

@st.cache_data(show_spinner="Cargando eventos desde GCS…")
def load_events_multi(paths: List[str]) -> pd.DataFrame:
    """Carga (desde GCS) y concatena múltiples archivos Parquet/CSV en uno solo.
    Cachea por sesión a menos que cambien 'paths'."""
    dfs = []
    for p in paths:
        if not p:
            continue
        try:
            df = _read_gcs(p)
            dfs.append(df)
        except Exception as e:
            st.warning(f"No se pudo cargar {p}: {e}")
    if not dfs:
        return pd.DataFrame()
    ev = pd.concat(dfs, ignore_index=True)

    # Normaliza y filtra torneos a Apertura/Clausura
    if "torneo" in ev.columns:
        ev["torneo"] = ev["torneo"].astype(str).str.strip().str.title()
        ev = ev[ev["torneo"].isin(["Apertura", "Clausura"])]

    return ev

# -------------------------------
# Sidebar – configuración de datos
# -------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Datos (GCS)")
    gcs_prefix = st.text_input(
        "Ruta base en GCS",
        value="gs://statsbomb_itam/eventos/",
        help="Prefijo donde están tus archivos .parquet/.csv (termina en /).",
    )

    path_23_24 = join_gs(gcs_prefix, "events_merged_LigaMX_2023_2024.parquet")
    path_24_25 = join_gs(gcs_prefix, "events_merged_LigaMX_2024_2025.parquet")

    use_23_24 = st.checkbox("Usar 2023–2024", value=True)
    use_24_25 = st.checkbox("Usar 2024–2025", value=True)

    team_name = st.text_input("Equipo", value="América")
    st.caption("Cargamos 1 sola vez (cache). Luego todo son filtros en memoria.")

# Cargar eventos (solo una lectura; se revalida si cambian 'paths')
paths = []
if use_23_24:
    paths.append(path_23_24)
if use_24_25:
    paths.append(path_24_25)

ev_all = load_events_multi(paths)
if ev_all.empty:
    st.error("No se cargaron eventos. Revisa prefijo de GCS, nombres de archivo o permisos IAM.")
    st.stop()

# -------------------------------
# Selector de torneos (solo Apertura/Clausura)
# -------------------------------
with st.sidebar:
    st.markdown("### 🏆 Torneos a analizar")
    torneos_disp = sorted(ev_all["torneo"].dropna().unique().tolist()) if "torneo" in ev_all else []
    # Por defecto selecciona lo disponible (Apertura/Clausura)
    selected_torneos = st.multiselect("Elige torneo(s)", torneos_disp, default=torneos_disp)
    st.divider()
    st.caption("ISAC Hackatón – Player Recommendation | KPIs + Estilo + Heatmap")

# -------------------------------
# Filtros en memoria (sin re-lectura)
# -------------------------------
df_sel = ev_all.copy()
if selected_torneos and "torneo" in df_sel:
    df_sel = df_sel[df_sel["torneo"].isin(selected_torneos)]

# Filtro de equipo: detecta columna disponible
team_col = first_col(df_sel, ["team", "team_name", "team.name", "possession_team_name"])
if team_col and team_name:
    df_team = df_sel[df_sel[team_col].astype(str).str.contains(team_name, case=False, na=False)].copy()
else:
    df_team = df_sel.copy()
    if not team_col:
        st.warning("No se encontró columna de equipo ('team', 'team_name', 'team.name', 'possession_team_name').")

# --- Ya puedes usar df_team para el resto de la app ---
st.success(f"Eventos cargados: {len(df_team):,}  |  Torneos: {', '.join(selected_torneos) if selected_torneos else '—'}")
st.dataframe(df_team.head())

# -------------------------------
# Pestañas
# -------------------------------
TAB_INICIO, TAB_AMERICA, TAB_ROSTER = st.tabs(["🏁 Inicio", "🦅 Club América", "👥 Roster"])

# -------------------------------
# Tab: Inicio (explicación + selección)
# -------------------------------
with TAB_INICIO:
    st.subheader("¿Qué puedes observar en esta herramienta?")
    st.markdown(
        """
        - **Comportamiento del equipo**: KPIs del/los torneos seleccionados.
        - **Estilo de juego**: resumen heurístico (open play, balón parado, contraataque, media distancia, juego aéreo).
        - **Zonas de impacto**: Heatmap de **goles** (coordenadas StatsBomb 120×80).
        - **Roster**: vista por jugador (iteraremos con métricas reales a continuación).
        """
    )

# -------------------------------
# Tab: Club América (análisis real)
# -------------------------------
with TAB_AMERICA:
    st.subheader(f"Resumen Torneo: {', '.join(selected_torneos)}")

    # KPIs y máximos
    #kpis, matches_agg = compute_kpis_from_matches(df_team, team_name=team_name)

    # df_sel = eventos de los torneos seleccionados (ambos equipos)
    kpis, matches_agg = compute_kpis_from_matches(df_sel, team_name=team_name)
    #team_col = "team" if "team" in df_sel.columns else ("team_name" if "team_name" in df_sel.columns else "possession_team_name")
    #st.write("Teams in df_sel:", df_sel[team_col].value_counts().to_frame("rows"))


    # ========================
    # ESTILO: cards y tipografías
    # ========================
    st.markdown("""
    <style>
    .kpi-section {
    padding: 16px 18px; border-radius: 16px; margin: 10px 0 18px 0;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    }
    .kpi-title {
    font-size: 1.1rem; font-weight: 700; letter-spacing: .3px; margin-bottom: 12px;
    }
    .kpi-subtle { color: #9aa0a6; font-size: .9rem; }
    .kpi-num { font-size: 1.8rem; font-weight: 800; line-height: 1; }
    .kpi-label { font-size: .86rem; color: #c9cdd3; margin-top: 4px; }
    .kpi-split { display:flex; gap:16px; margin-top: 8px; }
    .kpi-pill {
    display:inline-block; padding: 4px 10px; border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.12); font-size:.78rem; color:#cbd5e1;
    }
    </style>
    """, unsafe_allow_html=True)

    def fmt_pct(x):
        import math
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "–"
        return f"{x*100:.1f}%"

    def fmt_num(x):
        return "–" if x is None else f"{x:,}"

    def safe_div(a, b):
        return (a / b) if b and b != 0 else None

    # Derivados útiles
    PJ = kpis.get("PJ", 0)
    GF = kpis.get("GoalsFor", 0)
    GA = kpis.get("GoalsAgainst", 0)
    DG = kpis.get("GoalDiff", 0)
    shots_for = kpis.get("ShotsFor", None)
    shots_ag  = kpis.get("ShotsAgainst", None)
    conv = kpis.get("ConversionRate", None)
    xg_for = kpis.get("xG_for", None)
    xg_ag  = kpis.get("xG_against", None)
    xg_bal = (xg_for - xg_ag) if (xg_for is not None and xg_ag is not None) else None
    gf_pm  = safe_div(GF, PJ)
    ga_pm  = safe_div(GA, PJ)

    # ========================
    # BLOQUE 1 — Rendimiento global
    # ========================
    st.markdown("<div class='kpi-section'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>📊 Rendimiento global</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.plotly_chart(donut_kpi("PJ", kpis["PJ"], max(kpis["PJ"],1)), use_container_width=True)
    with c2: st.plotly_chart(donut_kpi("G", kpis["G"], kpis["PJ"], "#28a745"), use_container_width=True)
    with c3: st.plotly_chart(donut_kpi("E", kpis["E"], kpis["PJ"], "#ffc107"), use_container_width=True)
    with c4: st.plotly_chart(donut_kpi("P", kpis["P"], kpis["PJ"], "#dc3545"), use_container_width=True)

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown(f"<div class='kpi-num'>{kpis['Points']}</div><div class='kpi-label'>Puntos</div>", unsafe_allow_html=True)
    with g2:
        st.markdown(f"<div class='kpi-num'>{fmt_pct(kpis['WinRate'])}</div><div class='kpi-label'>Win%</div>", unsafe_allow_html=True)
    with g3:
        st.markdown(f"<span class='kpi-pill'>Torneos: {', '.join(selected_torneos)}</span>", unsafe_allow_html=True)
    with g4:
        st.empty()
    st.markdown("</div>", unsafe_allow_html=True)

    # ========================
    # BLOQUE 2 — Ataque
    # ========================
    st.markdown("<div class='kpi-section'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>⚽️ Ataque</div>", unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.markdown(f"<div class='kpi-num'>{GF}</div><div class='kpi-label'>Goles a favor</div>", unsafe_allow_html=True)
    with a2:
        st.markdown(f"<div class='kpi-num'>{fmt_num(shots_for)}</div><div class='kpi-label'>Tiros For</div>", unsafe_allow_html=True)
    with a3:
        st.markdown(f"<div class='kpi-num'>{fmt_pct(conv)}</div><div class='kpi-label'>Conv%</div>", unsafe_allow_html=True)
    with a4:
        xg_for_txt = "–" if xg_for is None else f"{xg_for:.2f}"
        st.markdown(f"<div class='kpi-num'>{xg_for_txt}</div><div class='kpi-label'>xG For</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ========================
    # BLOQUE 3 — Defensa
    # ========================
    st.markdown("<div class='kpi-section'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>🛡️ Defensa</div>", unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(f"<div class='kpi-num'>{GA}</div><div class='kpi-label'>Goles en contra</div>", unsafe_allow_html=True)
    with d2:
        st.markdown(f"<div class='kpi-num'>{kpis['CleanSheets']}</div><div class='kpi-label'>Porterías en cero</div>", unsafe_allow_html=True)
    with d3:
        st.markdown(f"<div class='kpi-num'>{fmt_num(shots_ag)}</div><div class='kpi-label'>Tiros Ag.</div>", unsafe_allow_html=True)
    with d4:
        xg_ag_txt = "–" if xg_ag is None else f"{xg_ag:.2f}"
        st.markdown(f"<div class='kpi-num'>{xg_ag_txt}</div><div class='kpi-label'>xG Ag.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ========================
    # BLOQUE 4 — Eficiencia total
    # ========================
    st.markdown("<div class='kpi-section'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi-title'>🚀 Eficiencia total</div>", unsafe_allow_html=True)
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.markdown(f"<div class='kpi-num'>{DG}</div><div class='kpi-label'>Diferencia de goles</div>", unsafe_allow_html=True)
    with e2:
        gfpm_txt = "–" if gf_pm is None else f"{gf_pm:.2f}"
        st.markdown(f"<div class='kpi-num'>{gfpm_txt}</div><div class='kpi-label'>GF por partido</div>", unsafe_allow_html=True)
    with e3:
        gapm_txt = "–" if ga_pm is None else f"{ga_pm:.2f}"
        st.markdown(f"<div class='kpi-num'>{gapm_txt}</div><div class='kpi-label'>GA por partido</div>", unsafe_allow_html=True)
    with e4:
        xgb_txt = "–" if xg_bal is None else f"{xg_bal:+.2f}"
        st.markdown(f"<div class='kpi-num'>{xgb_txt}</div><div class='kpi-label'>Balance xG (For−Ag)</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    
    # ========================
    # Jugadores destacados
    # ========================
    st.markdown("### 🧩 Jugadores destacados (Top performers)")

    tops = top_performers(df_team, team_name=team_name)

    # Primera fila: goleador, asistidor y portero
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Máximo anotador",
            tops.get("max_scorer", "-"),
            f'{tops.get("max_scorer_goals", 0)} goles'
        )
    with c2:
        st.metric(
            "Máximo asistidor",
            tops.get("max_assist", "-"),
            f'{tops.get("max_assists", 0)} asist.'
        )
    with c3:
        st.metric(
            "Mejor portero",
            tops.get("top_shotstopper", "-"),
            f'{tops.get("top_saves", 0)} atajadas'
        )

    # Segunda fila: jugador más central (PageRank)
    c4, _ = st.columns([1, 1])
    with c4:
        st.metric(
            "Jugador más influyente (red de pases)",
            tops.get("most_central_player", "-"),
            f'Centralidad: {tops.get("pagerank_score", 0):.4f}'
        )


    st.markdown("### 🕸️ Red de pases del equipo (PageRank)")
    min_passes = st.slider("Umbral mínimo de conexiones (nº de pases entre dos jugadores)", 1, 10, 3, 1)

    nodes_df, edges_df, pr_map, pr_top = build_pass_network(
        df_team, team_name=team_name, min_passes=min_passes
    )

    if nodes_df.empty or edges_df.empty:
        st.info("No hay suficientes pases para construir la red en esta selección.")
    else:
        st.caption(f"Jugador más influyente por PageRank: **{pr_top}**")
        fig_net = plot_pass_network(nodes_df, edges_df, highlight=pr_top, title="Red de pases (conexiones ≥ umbral)")
        st.plotly_chart(fig_net, use_container_width=True)

        # (opcional) tabla top-10 por PageRank
        st.markdown("#### Top-10 por centralidad (PageRank)")
        st.dataframe(nodes_df[["player","pr","indeg","outdeg","total"]].head(5), use_container_width=True)

    st.markdown("### Estilo de juego (v0 real)")
    st.info(infer_style_summary(df_team, team_name=team_name))

    st.markdown("### Zonas donde se anotan más goles")
    xy = goals_xy_for_heatmap(df_team, team_name=team_name)
    if xy.empty:
        st.info("No hay goles registrados para el heatmap en la selección actual.")
    else:
        if _HAS_MPLSOCCER:
            import matplotlib.pyplot as plt
            pitch = Pitch(pitch_type='statsbomb', line_zorder=2)
            fig, ax = pitch.draw(figsize=(10, 6))
            hb = ax.hexbin(xy["x"], xy["y"], gridsize=20, extent=(0, 120, 0, 80), mincnt=1)
            cbar = fig.colorbar(hb, ax=ax)
            cbar.set_label('Frecuencia de goles')
            ax.set_title("Mapa de calor – Zonas de gol")
            st.pyplot(fig, use_container_width=True)
        else:
            fig = px.density_heatmap(xy, x="x", y="y", nbinsx=24, nbinsy=16, title="Mapa de calor – Zonas de gol")
            fig.update_yaxes(autorange="reversed", range=[0,80])
            fig.update_xaxes(range=[0,120])
            st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Tab: Roster (placeholder)
# -------------------------------
with TAB_ROSTER:
    st.subheader("Roster de jugadores (v0)")
    st.info("En la siguiente iteración, poblaremos el roster real y métricas por jugador desde los eventos.")

# -------------------------------
# Footer / Roadmap
# -------------------------------
st.sidebar.divider()
st.sidebar.markdown(
    """
    **Roadmap inmediato**
    1) Métricas por jugador (xG/xA, tiros, presiones, duelos, intercepciones).
    2) Radar real de estilo (porcentajes exactos por dimensión).
    3) Shortlist de reemplazos basados en debilidades detectadas.
    4) Exportables (CSV/Excel) y bookmarks de filtros.
    """
)
