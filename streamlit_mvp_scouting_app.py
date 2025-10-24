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
import json
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import dask.dataframe as dd
import gcsfs
from pathlib import PurePosixPath

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
    plot_pass_network,
    display_play_style_section,
    display_goal_zones_section,
    #ah_build_players_df_from_events,
    ah_compute_obv_like,
    ah_build_players_df_from_events_v2

)

@st.cache_data(show_spinner=True)
def _build_players_from_filtered_events(
    events_df: pd.DataFrame,
    normalize_by_position: bool = True,
    min_minutes: float = 270.0
):
    """
    Wrapper cacheado que genera métricas OBV por jugador a partir de eventos filtrados.
    Basado en ah_build_players_df_from_events_v2.
    """
    return ah_build_players_df_from_events_v2(
        events=events_df,
        roster_positions=None,             # o tu mapeo por posición si lo tienes
        normalize_by_position=normalize_by_position,
        add_position_relative_cols=True,
        min_minutes=min_minutes,
        use_total_net_direct=False         # o True si prefieres usar obv_total_net per90 directo
    )

def _get_filtered_events_from_session() -> pd.DataFrame | None:
    # Lista de claves candidatas típicas para el DF de eventos filtrado
    candidate_keys = [
        "events_df_filtered", "events_filtered", "df_events_filtered",
        "events_df", "df_events"  # fallback si no guardas una versión filtrada separada
    ]
    for k in candidate_keys:
        df = st.session_state.get(k)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return None



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
#  Correcto con formato TOML
sa_info = dict(st.secrets["gcp_service_account"])

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



@st.cache_data(show_spinner="Cargando eventos desde GCS…",ttl=3600)
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



# ============================================================
# FUNCIÓN PRINCIPAL DE CARGA (DASK)
# ============================================================
@st.cache_data(show_spinner="Leyendo en streaming desde GCS…", ttl=3600)
def load_events_multi_dask(
    paths: list[str],
    sa_info: dict,
    keep_only: list[str] | None = None,
    filename_filter: str | None = None,   # opcional: filtra por nombre o patrón
    case_insensitive: bool = True
) -> pd.DataFrame:
    """
    Lee varios Parquet/CSV desde GCS con Dask.
    - Si filename_filter no es None, sólo procesa archivos cuyo nombre coincida (fnmatch, soporta '*').
    - Si keep_only no es None, selecciona esas columnas (si existen).
    Devuelve un pandas.DataFrame materializado y cacheado.
    """
    import fnmatch

    def _match(fname: str, patt: str, ci: bool) -> bool:
        if ci:
            return fnmatch.fnmatch(fname.lower(), patt.lower())
        return fnmatch.fnmatch(fname, patt)

    ddfs = []
    for p in paths or []:
        if not p:
            continue

        # Filtro opcional por nombre
        if filename_filter:
            fname = PurePosixPath(p).name
            if not _match(fname, filename_filter, case_insensitive):
                continue

        try:
            if p.lower().endswith(".parquet"):
                ddf = dd.read_parquet(p, storage_options={"token": sa_info})
            elif p.lower().endswith(".csv"):
                ddf = dd.read_csv(p, storage_options={"token": sa_info})
            else:
                raise ValueError(f"Extensión no soportada en {p}")
            ddfs.append(ddf)
        except Exception as e:
            st.error(f"❌ No se pudo cargar {p}\n\n**Error:** {e}")

    if not ddfs:
        if filename_filter:
            st.warning(f"No se encontró ningún archivo que coincida con '{filename_filter}'.")
        return pd.DataFrame()

    # Concatena particiones (streaming)
    ddf_all = dd.concat(ddfs, interleave_partitions=True)

    # Selección de columnas (modo ligero)
    if keep_only:
        cols = [c for c in keep_only if c in ddf_all.columns]
        if cols:
            ddf_all = ddf_all[cols]

    # Materializa
    df = ddf_all.compute()

    # Normaliza 'torneo'
    if "torneo" in df.columns:
        df["torneo"] = df["torneo"].astype(str).str.strip().str.title()
        df = df[df["torneo"].isin(["Apertura", "Clausura"])]

    return df



# ================================
# Utilidad para unir rutas GCS
# ================================
def join_gs(prefix: str, name: str) -> str:
    return prefix.rstrip("/") + "/" + name.lstrip("/")

# ================================
# Carga directa de UN parquet GCS
# ================================
@st.cache_data(show_spinner="Leyendo parquet único desde GCS…", ttl=3600)
def load_single_parquet_gcs(path: str) -> pd.DataFrame:
    """
    Lee ÚNICAMENTE el parquet indicado en `path` (gs://... .parquet)
    usando credenciales del Service Account guardadas en st.secrets["gcp_service_account"].
    No concatena nada. Retorna un pandas.DataFrame.
    """
    try:
        #sa_info = json.loads(st.secrets["gcp_service_account"])  # <- credenciales desde secrets
        sa_info = dict(st.secrets["gcp_service_account"])
    except Exception:
        st.error("No encontré `st.secrets['gcp_service_account']`. Agrega tu JSON de Service Account a los secrets.")
        return pd.DataFrame()

    storage_opts = {"token": sa_info}  # fuerza gcsfs a usar este SA (evita metadata server)
    try:
        df = pd.read_parquet(path, storage_options=storage_opts, engine="pyarrow")
    except Exception as e:
        st.error(f"❌ Error leyendo {path}\n\n**Detalle:** {e}")
        return pd.DataFrame()

    # Normaliza/filtra 'torneo' si existe
    if "torneo" in df.columns:
        df["torneo"] = df["torneo"].astype(str).str.strip().str.title()
        df = df[df["torneo"].isin(["Apertura", "Clausura"])]

    return df

# ================================
# Sidebar – SOLO un archivo
# ================================
with st.sidebar:
    st.markdown("### ⚙️ Datos (GCS)")
    gcs_prefix = st.text_input(
        "Ruta base en GCS",
        value="gs://statsbomb_itam/eventos/",
        help="Prefijo donde está tu parquet único (termina en /).",
    )
    # Archivo ÚNICO optimizado para la app
    clausura_parquet = "events_merged_LigaMX_2024_2025_clausura_streamlit_app.parquet"
    path_clausura = join_gs(gcs_prefix, clausura_parquet)

    team_name = st.text_input("Equipo", value="América")
    st.caption("Leemos 1 sola vez (cache). Todo lo demás son filtros en memoria.")

# ================================
# Lectura ÚNICA
# ================================
needed_cols = [
    # IDs y tiempo
    "id","match_id","period","minute","second","timestamp",
    # equipos/jugadores
    "team","team_name","team.name","possession_team_name","player","player_name","player.name","pass_recipient",
    # marcador y metadata
    "home_team","away_team","home_score","away_score",
    # tipo de evento y derivados
    "type","type.name","shot_outcome","play_pattern","shot_type","shot_body_part","body_part",
    # xG y ubicaciones
    "shot_statsbomb_xg","shot_xg","xg","location","shot_end_location",
    # portero
    "goalkeeper_type","goalkeeper_outcome",
    # linking de asistencias
    "pass_goal_assist","shot_key_pass_id",
    # torneo
    "torneo"
]

ev_all = load_single_parquet_gcs(path_clausura)
if ev_all.empty:
    st.error("No se cargaron eventos. Revisa el prefijo GCS, el nombre del archivo o tus secrets.")
    st.stop()

# Subselección de columnas si existen (modo ligero)
keep = [c for c in needed_cols if c in ev_all.columns]
if keep:
    ev_all = ev_all[keep].copy()

# ================================
# Selector de torneos (opcional)
# ================================
with st.sidebar:
    st.markdown("### 🏆 Torneos a analizar")
    torneos_disp = sorted(ev_all["torneo"].dropna().unique().tolist()) if "torneo" in ev_all else []
    selected_torneos = st.multiselect("Elige torneo(s)", torneos_disp, default=torneos_disp)
    st.divider()
    st.caption("ISAC Hackatón – Player Recommendation | KPIs + Estilo + Heatmap")

# ================================
# Filtros en memoria
# ================================
df_sel = ev_all
if selected_torneos and "torneo" in df_sel.columns:
    df_sel = df_sel[df_sel["torneo"].isin(selected_torneos)]

def first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

team_col = first_col(df_sel, ["team", "team_name", "team.name", "possession_team_name"])
if team_col and team_name:
    df_team = df_sel[df_sel[team_col].astype(str).str.contains(team_name, case=False, na=False)].copy()
else:
    df_team = df_sel.copy()
    if not team_col:
        st.warning("No se encontró columna de equipo ('team', 'team_name', 'team.name', 'possession_team_name').")
# Después de construir df_sel (filtrado por torneos)
st.session_state["events_df_filtered_allteams"] = df_sel.copy()
# Después de construir df_team (filtrado por torneos + equipo)
st.session_state["events_df_filtered_team"] = df_team.copy()
# ================================
# Resumen / vista rápida
# ================================
st.success(f"Eventos cargados: {len(df_team):,}  |  Torneos: {', '.join(selected_torneos) if selected_torneos else '—'}")
st.dataframe(df_team.head())


    #         st.plotly_chart(fig, use_container_width=True)


# -------------------------------
# Tab: Roster (completa)
# -------------------------------
# ============================== TAB: ROSTER (SIN RECARGA) ==============================
import streamlit as st
import pandas as pd
import numpy as np
import math, html as py_html, textwrap, base64, json

from streamlit.components.v1 import html as st_html

with TAB_ROSTER:

    st.subheader("Roster y recomendaciones por posición")

    # 0) DataFrames de tu pipeline
    ev_team = st.session_state.get("events_df_filtered_team")        # equipo base (ej. América)
    ev_all  = st.session_state.get("events_df_filtered_allteams")    # mismos torneos, todos los equipos
    if ev_all is None or ev_all.empty:
        st.info("Aplica filtros de torneos en el sidebar; no hay eventos disponibles.")
        st.stop()
    if ev_team is None or ev_team.empty:
        st.info("Selecciona un equipo en el sidebar para ver su roster.")
        st.stop()

    # 1) Builder v2 (tu helper con OBV nativo)
    @st.cache_data(show_spinner=False)
    def _build_players_v2(
        df: pd.DataFrame,
        normalize_by_position: bool = True,
        min_minutes: float = 270.0,
        use_total_net_direct: bool = False
    ):
        return ah_build_players_df_from_events_v2(
            events=df,
            roster_positions=None,
            normalize_by_position=normalize_by_position,
            add_position_relative_cols=True,
            min_minutes=min_minutes,
            use_total_net_direct=use_total_net_direct
        )

    players_df_team = _build_players_v2(ev_team, normalize_by_position=True)
    players_df_all  = _build_players_v2(ev_all,  normalize_by_position=True)

    # 2) Filtros UI
    POS_LABELS = {"Porteros":"GK","Defensas":"DEF","Medios":"MID","Delanteros":"FWD"}
    pos_label = st.radio("Posición", list(POS_LABELS.keys()), horizontal=True, index=2)
    pos_code  = POS_LABELS[pos_label]

    default_team = st.session_state.get("selected_team_name", None)
    if default_team is None and "team" in ev_team.columns and not ev_team["team"].dropna().empty:
        default_team = ev_team["team"].dropna().astype(str).value_counts().idxmax()
    team_name   = st.text_input("Equipo base", value=default_team or "Club América")
    minutes_min = st.slider("Minutos mínimos", 0, 4000, 300, 50)
    top_n_reco  = st.slider("Top-N candidatos (otros equipos)", 5, 30, 12, 1)

    # 3) Helpers de datos/UI
    def _fmt_eff(x):
        return "—" if (x is None or (isinstance(x, float) and (math.isnan(x) or np.isinf(x)))) else f"{x:.3f}"

    def _initials_from_name(fullname: str) -> str:
        parts = [p for p in (fullname or "").strip().split() if p]
        if not parts: return "??"
        if len(parts) == 1: return parts[0][:2].upper()
        return (parts[0][0] + parts[-1][0]).upper()

    def _avatar_svg_data_uri(initials: str, bg="#1f2937", fg="#ffffff") -> str:
        svg = f'''
                <svg xmlns="http://www.w3.org/2000/svg" width="240" height="240">
                <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0" stop-color="{bg}"/><stop offset="1" stop-color="#111827"/>
                </linearGradient></defs>
                <rect width="240" height="240" rx="28" fill="url(#g)"/>
                <text x="50%" y="52%" text-anchor="middle" dominant-baseline="middle"
                        font-family="Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial"
                        font-size="92" font-weight="800" fill="{fg}">{py_html.escape(initials)}</text>
                </svg>'''
        return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")

    def _photo_or_initials(name: str, photo_url: str | None) -> str:
        if isinstance(photo_url, str) and photo_url.strip():
            return photo_url
        return _avatar_svg_data_uri(_initials_from_name(name))

    # 4) Construir ROSTER (servidor)
    mask_team = (players_df_team["position_group"] == pos_code) & (players_df_team["minutes"] >= minutes_min)
    roster_df = (
        players_df_team.loc[
            mask_team,
            ["player_id","player_name","minutes","efficiency","position_role","team"] +
            (["photo_url"] if "photo_url" in players_df_team.columns else [])
        ]
        .drop_duplicates("player_id")
        .sort_values("efficiency", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
    if roster_df.empty:
        st.info(f"No hay jugadores de {team_name} que cumplan los criterios para {pos_label}.")
        st.stop()

    # 5) Construir CANDIDATOS por jugador (servidor) ⇒ lo mandamos ya listo al componente
    base_cands = players_df_all.loc[
        (players_df_all["position_group"] == pos_code) &
        (players_df_all["team"] != team_name) &
        (players_df_all["minutes"] >= minutes_min),
        ["player_id","player_name","team","minutes","efficiency","position_role"] +
        (["photo_url"] if "photo_url" in players_df_all.columns else [])
    ].drop_duplicates("player_id").copy()

    # Mapa: jugador_roster -> lista de candidatos (ya ordenados y truncados)
    cand_map: dict[str, list[dict]] = {}
    for _, r in roster_df.iterrows():
        sel_role = r.get("position_role", None)
        cands = base_cands.copy()
        if sel_role is not None and "position_role" in cands.columns:
            cands["priority"] = (cands["position_role"] == sel_role).astype(int)
        else:
            cands["priority"] = 0
        cands = cands.sort_values(["priority","efficiency","minutes"], ascending=[False, False, False]).head(top_n_reco)

        lst = []
        for _, c in cands.iterrows():
            cname = str(c.get("player_name","") or "")
            cteam = str(c.get("team","") or "")
            cmins = int(c.get("minutes",0) or 0)
            ceff  = c.get("efficiency", None)
            ceff_txt = _fmt_eff(ceff)
            crole = str(c.get("position_role","") or "—")
            cphoto = _photo_or_initials(cname, c.get("photo_url", None))
            lst.append({
                "name": cname,
                "team": cteam,
                "mins": cmins,
                "eff_txt": ceff_txt,
                "role": crole,
                "photo": cphoto,
            })
        cand_map[str(r["player_name"])] = lst

    # 6) Datos para el componente (roster cards)
    roster_payload = []
    for _, r in roster_df.iterrows():
        name = str(r.get("player_name","") or "")
        role = str(r.get("position_role","") or "—")
        mins = int(r.get("minutes",0) or 0)
        eff  = r.get("efficiency", None)
        eff_txt = _fmt_eff(eff)
        photo = _photo_or_initials(name, r.get("photo_url", None))
        roster_payload.append({
            "name": name,
            "role": role,
            "mins": mins,
            "eff_txt": eff_txt,
            "photo": photo
        })

    # 7) Render en UN SOLO componente: roster + candidatos (sin recargar)
    html_prefix = textwrap.dedent(f"""
        <!doctype html>
        <html lang="es">
        <head>
        <meta charset="utf-8">
        <style>
            .hsnap {{ display:flex; gap:12px; overflow-x:auto; padding:8px 2px 14px; scroll-snap-type:x mandatory; }}
            .hsnap::-webkit-scrollbar {{ height:10px; }}
            .hsnap::-webkit-scrollbar-thumb {{ background:rgba(0,0,0,.25); border-radius:8px; }}
            .card, .c-card {{
            flex:0 0 auto; width:210px; scroll-snap-align:start; border:1px solid rgba(0,0,0,.08);
            border-radius:14px; background:rgba(255,255,255,.92); box-shadow:0 1px 6px rgba(0,0,0,.06);
            transition:transform .16s, box-shadow .16s, border-color .16s; text-decoration:none; color:inherit; padding:10px;
            }}
            .card:hover {{ transform:translateY(-2px); box-shadow:0 6px 22px rgba(0,0,0,.10); border-color:rgba(0,0,0,.16); }}
            .card.selected {{ outline:2px solid rgba(99,102,241,.9); box-shadow:0 6px 22px rgba(99,102,241,.25); }}
            .imgwrap {{ width:100%; height:130px; border-radius:10px; overflow:hidden; background:#e5e7eb; }}
            .imgwrap img {{ width:100%; height:100%; object-fit:cover; display:block; }}
            .name {{ font-weight:700; margin-top:8px; line-height:1.2; font-size:15px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
            .meta {{ margin-top:4px; font-size:12px; opacity:.8; display:flex; gap:6px; align-items:center; flex-wrap:wrap; }}
            .role {{ padding:2px 6px; border-radius:999px; border:1px solid rgba(0,0,0,.12); }}
            .sep {{ opacity:.5; }}
            .eff {{ margin-top:8px; display:flex; align-items:center; gap:8px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Courier New"; }}
            .badge {{ font-size:11px; padding:2px 6px; border-radius:6px; border:1px solid rgba(0,0,0,.12); background:rgba(0,0,0,.04); font-weight:600; }}
            .val {{ font-weight:700; }}
            .teamchip {{
            margin-top:6px; font-size:12px; padding:3px 8px; border-radius:999px;
            border:1px solid rgba(0,0,0,.12); background:rgba(0,0,0,.035); display:inline-block;
            max-width:100%; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
            }}
            .section-title {{
                            margin: 10px 0 4px;
                            font-weight: 800;
                            font-size: 16px;
                            color: #ffffff;              /* ← texto blanco */
                            }}

                            .section-sub {{
                            margin: -2px 0 8px;
                            font-size: 12px;
                            opacity: .9;
                            color: #f1f1f1;              /* ← texto blanco/gris suave */
                            }}
            @media (max-width:420px) {{ .card, .c-card {{ width:72vw; }} }}
        </style>
        </head>
        <body>
        <div style="opacity:.8; font-size:13px; margin-bottom:4px;">
            Roster de {py_html.escape(team_name)} — {py_html.escape(pos_label)} (ordenado por eficiencia)
        </div>
        <div id="roster" class="hsnap"></div>
        <div id="selected_info" style="margin:6px 0 6px; font-size:14px;"></div>

        <div id="cands_section" style="display:none;">
            <div id="cands_title" class="section-title"></div>
            <div class="section-sub">Mismo grupo posicional; prioridad a la misma posición fina.</div>
            <div id="cands" class="hsnap"></div>
        </div>
        """)

    # 👇 Bloque JS puro (NO es f-string)
    html_script = """
    <script>
    // Decodificador Base64 → UTF-8 seguro
    function b64json(b64) {
        const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const text = new TextDecoder('utf-8').decode(bytes);
        return JSON.parse(text);
    }

    const ROSTER  = b64json('%s');
    const CANDMAP = b64json('%s');

    const rosterEl   = document.getElementById('roster');
    const candsEl    = document.getElementById('cands');
    const candsBox   = document.getElementById('cands_section');
    const infoEl     = document.getElementById('selected_info');
    const candsTitle = document.getElementById('cands_title');

    function cardHTML(p, selectable=true) {
        const cls = selectable ? 'card' : 'c-card';
        const teamLine = (!selectable && p.team)
        ? `<div class="teamchip" title="${p.team}">${p.team}</div>` : '';
        return `
        <div class="${cls}" data-name="${p.name}">
            <div class="imgwrap"><img src="${p.photo}" alt="player"/></div>
            <div class="name" title="${p.name}">${p.name}</div>
            <div class="meta">
            <span class="role">${p.role || '—'}</span><span class="sep">•</span><span class="mins">${p.mins}′</span>
            </div>
            <div class="eff"><span class="badge">EFF</span><span class="val">${p.eff_txt}</span></div>
            ${teamLine}
        </div>`;
    }

    // Render roster
    rosterEl.innerHTML = ROSTER.map(p => cardHTML(p, true)).join('');

    // Click sin recarga
    rosterEl.addEventListener('click', (ev) => {
        const card = ev.target.closest('.card');
        if (!card) return;
        const name = card.getAttribute('data-name');

        rosterEl.querySelectorAll('.card').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');

        const player = ROSTER.find(x => x.name === name);
        // Info breve arriba
        infoEl.innerHTML = player
        ? `<b>Jugador seleccionado:</b> ${player.name} • Posición: <i>${player.role || '—'}</i> • Eficiencia: <b>${player.eff_txt}</b>`
        : '';

        // 🔹 Subheader dinámico (visible siempre que hay selección)
        candsTitle.textContent = `Mejores prospectos para sustituir a ${player ? player.name : name}`;

        // Render candidatos
        const lst = CANDMAP[name] || [];
        if (!lst.length) {
        candsEl.innerHTML = '<div style="opacity:.7;">No hay candidatos que cumplan con los filtros actuales.</div>';
        } else {
        candsEl.innerHTML = lst.map(p => cardHTML(p, false)).join('');
        }

        candsBox.style.display = 'block';
        candsBox.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    </script>
    </body></html>
    """ % (
        base64.b64encode(json.dumps(roster_payload, ensure_ascii=False).encode('utf-8')).decode('ascii'),
        base64.b64encode(json.dumps(cand_map,      ensure_ascii=False).encode('utf-8')).decode('ascii'),
    )

    html_block = html_prefix + html_script
    st_html(html_block, height=520, scrolling=True)
    #.markdown('<div style="padding:8px;border:1px solid #ddd;border-radius:8px;">HOLA HTML</div>', unsafe_allow_html=True)
    #st.write(ev_all[["obv_for_net","obv_against_net","obv_total_net"]].describe())
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
