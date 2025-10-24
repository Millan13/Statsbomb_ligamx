"""
analytics_helpers.py — funciones de apoyo para el dashboard ISAC Scouting.

Diseñado para trabajar con eventos de StatsBomb (Events v8) ya integrados en un
DataFrame (CSV o Parquet). Incluye utilidades robustas que usan "fallbacks" de
columnas para adaptarse a variaciones frecuentes del feed.

Autor: Miguel Millán (scaffold por ChatGPT)
"""
from __future__ import annotations
import ast
from typing import List, Tuple, Dict, Optional, Any,Iterable, Union
import numpy as np
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.colors as pc




# ----------------------------
# Utilidades generales
# ----------------------------

def first_col(df: pd.DataFrame, candidates: List[str], default: Optional[str] = None) -> Optional[str]:
    """Devuelve la primera columna existente en df de entre candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return default


def parse_xy_series(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Convierte una serie de ubicaciones (listas [x,y] o strings "[x, y]") en columnas x,y numéricas."""
    xs, ys = [], []
    for v in series:
        x = y = np.nan
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            x, y = v[0], v[1]
        elif isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                try:
                    # Usa ast.literal_eval para listas como texto
                    arr = ast.literal_eval(v)
                    if isinstance(arr, (list, tuple)) and len(arr) >= 2:
                        x, y = float(arr[0]), float(arr[1])
                except Exception:
                    pass
        xs.append(x)
        ys.append(y)
    xs = pd.to_numeric(pd.Series(xs), errors="coerce")
    ys = pd.to_numeric(pd.Series(ys), errors="coerce")
    return xs, ys

def real_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Devuelve el nombre REAL de la columna en df buscando de forma
    case-insensitive y con strip de espacios.
    """
    norm = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in norm:
            return norm[key]
    return None

def to_numeric_safe(s: pd.Series) -> pd.Series:
    """Convierte a numérico con coerce para evitar strings '0.12', etc."""
    return pd.to_numeric(s, errors="coerce")
# ----------------------------
# Carga / filtros
# ----------------------------

def load_events(path: str) -> pd.DataFrame:
    """Carga eventos desde Parquet o CSV, según la extensión."""
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.lower().endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError("Extensión no soportada. Usa .parquet o .csv")


def filter_team_tournament(df: pd.DataFrame, torneo: str, team_name: str,
                           torneo_col: str = "torneo", team_cols: List[str] = ["team", "team_name", "possession_team_name"]) -> pd.DataFrame:
    """Filtra por torneo y equipo.
    - torneo: etiqueta del torneo (e.g., 'Apertura', 'Clausura')
    - team_name: nombre exacto del equipo en el feed (e.g., 'América')
    """
    df2 = df.copy()
    if torneo_col in df2.columns:
        df2 = df2[df2[torneo_col] == torneo]
    # Hallar columna de equipo en evento
    team_col = first_col(df2, team_cols)
    if team_col:
        df2 = df2[df2[team_col] == team_name]
    return df2


def filter_last_tournament(df: pd.DataFrame, torneo_col: str = "torneo") -> Tuple[pd.DataFrame, str]:
    """Devuelve DF del último torneo (por orden lexicográfico) y su nombre."""
    if torneo_col in df.columns and df[torneo_col].notna().any():
        last = df[torneo_col].astype(str).sort_values().iloc[-1]
        return df[df[torneo_col] == last].copy(), str(last)
    return df.copy(), "(sin torneo)"

# ----------------------------
# KPIs de partidos
# ----------------------------

from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np

def compute_kpis_from_matches(df_t: pd.DataFrame, team_name: str = "Club América") -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    KPIs de equipo desde la perspectiva de `team_name`.

    Devuelve:
      - kpis: dict con
          PJ, G, E, P, Points, WinRate,
          GoalsFor (GF), GoalsAgainst (GA), GoalDiff (GD),
          CleanSheets, ConversionRate (GF / ShotsFor),
          xG_for, xG_against (si hay columna xG)
      - matches_agg: tabla por partido (match_id, marcador, resultado)
    """
    out = {
        "PJ": 0, "G": 0, "E": 0, "P": 0, "Points": 0, "WinRate": np.nan,
        "GoalsFor": 0, "GoalsAgainst": 0, "GoalDiff": 0,
        "CleanSheets": 0, "ConversionRate": np.nan,
        "ShotsFor": None, "ShotsAgainst": None,
        "xG_for": np.nan, "xG_against": np.nan,
    }

    # --------------------------
    # 1) Agregado por partido
    # --------------------------
    needed = {"match_id", "home_team", "away_team", "home_score", "away_score"}
    if not needed.issubset(df_t.columns):
        # Si vienes de events, suele haber duplicados por match_id; agregamos por first
        cols = [c for c in ["match_id", "home_team", "away_team", "home_score", "away_score"] if c in df_t.columns]
        if "match_id" not in cols:
            # sin match_id no podemos agrupar por partido; devolvemos con mínimos
            return out, pd.DataFrame()
        matches_agg = df_t.groupby("match_id").agg(
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            home_score=("home_score", "first"),
            away_score=("away_score", "first"),
        ).reset_index()
    else:
        matches_agg = df_t[list(needed)].drop_duplicates()

    if matches_agg.empty:
        return out, matches_agg

    # Resultado desde la perspectiva del equipo
    def _res(r):
        if r["home_team"] == team_name:
            gf, ga = r["home_score"], r["away_score"]
        elif r["away_team"] == team_name:
            gf, ga = r["away_score"], r["home_score"]
        else:
            # Partido que no involucra al equipo (por seguridad)
            return None
        if gf > ga: return "W"
        if gf == ga: return "D"
        return "L"

    matches_agg["result"] = matches_agg.apply(_res, axis=1)
    matches_agg = matches_agg.dropna(subset=["result"])

    # KPIs básicos
    out["PJ"] = int(len(matches_agg))
    out["G"]  = int((matches_agg["result"] == "W").sum())
    out["E"]  = int((matches_agg["result"] == "D").sum())
    out["P"]  = int((matches_agg["result"] == "L").sum())
    out["Points"] = 3 * out["G"] + out["E"]
    out["WinRate"] = (out["G"] / out["PJ"]) if out["PJ"] > 0 else np.nan

    # GF / GA / GD desde el marcador
    def _gf(r):
        if r["home_team"] == team_name: return r["home_score"]
        if r["away_team"] == team_name: return r["away_score"]
        return 0
    def _ga(r):
        if r["home_team"] == team_name: return r["away_score"]
        if r["away_team"] == team_name: return r["home_score"]
        return 0

    out["GoalsFor"]     = int(matches_agg.apply(_gf, axis=1).sum())
    out["GoalsAgainst"] = int(matches_agg.apply(_ga, axis=1).sum())
    out["GoalDiff"]     = int(out["GoalsFor"] - out["GoalsAgainst"])

    # Clean sheets
    def _cs(r):
        if r["home_team"] == team_name: return r["away_score"] == 0
        if r["away_team"] == team_name: return r["home_score"] == 0
        return False
    out["CleanSheets"] = int(matches_agg.apply(_cs, axis=1).sum())

    # --------------------------
    # 2) Eventos de tiro (Shots)
    #    para Conversion y xG
    # --------------------------
    # --------------------------
    # 2) Eventos de tiro (Shots)
    #    para Conversion y xG
    # --------------------------
    team_col = None
    for c in ["team", "team_name", "possession_team_name"]:
        if c in df_t.columns:
            team_col = c
            break

    if "type" in df_t.columns:
        shots = df_t[df_t["type"] == "Shot"].copy()

        # Tiros del equipo
        shots_for = shots if team_col is None else shots[shots[team_col] == team_name]
        out["ShotsFor"] = int(len(shots_for)) if shots_for is not None else None

        # Tiros del rival (solo si el DF incluye ambos equipos)
        shots_against = None
        if team_col is not None and (df_t[team_col] != team_name).any():
            shots_against = shots[shots[team_col] != team_name]
            out["ShotsAgainst"] = int(len(shots_against))
        else:
            out["ShotsAgainst"] = None  # puede venir None si df_t ya estaba filtrado a un solo equipo

        # Conversión del equipo
        goals_for_ev = 0
        if "shot_outcome" in shots_for.columns:
            goals_for_ev = int((shots_for["shot_outcome"] == "Goal").sum())
        if out["ShotsFor"] and out["ShotsFor"] > 0:
            out["ConversionRate"] = goals_for_ev / out["ShotsFor"]

        # === Detección robusta de columna xG ===
        xg_col = real_col(shots, ["shot_statsbomb_xg", "shot_xg", "xg"])
        if xg_col is not None:
            shots_for_xg = to_numeric_safe(shots_for[xg_col]) if not shots_for.empty else pd.Series(dtype=float)
            out["xG_for"] = float(shots_for_xg.sum()) if shots_for_xg.size else np.nan

            if shots_against is not None and not shots_against.empty:
                shots_ag_xg = to_numeric_safe(shots_against[xg_col])
                out["xG_against"] = float(shots_ag_xg.sum()) if shots_ag_xg.size else np.nan
        else:
            # no se encontró columna de xG en el DF de tiros
            out["xG_for"] = np.nan
            out["xG_against"] = np.nan

    return out, matches_agg








# ----------------------------
# Top performers (goleador, asistidor, portero)
# ----------------------------

def get_assist_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in ["pass_assist", "assist_player", "pass_recipient", "pass_assisted_shot_id"] if c in df.columns]




def top_performers(df_t: pd.DataFrame, team_name: str = "Club América") -> Dict[str, Any]:
    """
    Calcula:
      - Máximo goleador (nombre y #goles)
      - Máximo asistidor (nombre y #asistencias)
      - Portero con más atajadas (nombre y #atajadas)
      - Jugador con mayor centralidad (PageRank) en la red de pases

    Basado en StatsBomb Events v8:
      * Goles: type == 'Shot' & shot_outcome == 'Goal'
      * Asistencias: type == 'Pass' & pass_goal_assist == True
      * Atajadas: type in {'Goal Keeper','Goalkeeper'} & goalkeeper_type/outcome contiene 'Saved'
      * Red de pases: edges entre player -> pass_recipient
    """
    out = {
        "max_scorer": "-",
        "max_scorer_goals": 0,
        "max_assist": "-",
        "max_assists": 0,
        "top_shotstopper": "-",
        "top_saves": 0,
        "most_central_player": "-",
        "pagerank_score": np.nan,
    }

    # Detección columna de equipo
    team_col = None
    for c in ["team", "team_name", "possession_team_name"]:
        if c in df_t.columns:
            team_col = c
            break

    # ========================
    # 1) Máximo goleador
    # ========================
    if all(c in df_t.columns for c in ["type", "shot_outcome", "player"]):
        shots = df_t[df_t["type"] == "Shot"].copy()
        if team_col:
            shots = shots[shots[team_col] == team_name]
        goals = shots[shots["shot_outcome"] == "Goal"].copy()
        if not goals.empty:
            scorer_counts = goals["player"].value_counts()
            out["max_scorer"] = scorer_counts.idxmax()
            out["max_scorer_goals"] = int(scorer_counts.max())

    # ========================
    # 2) Máximo asistidor
    # ========================
    if "type" in df_t.columns:
        passes = df_t[df_t["type"] == "Pass"].copy()
        if team_col:
            passes = passes[passes[team_col] == team_name]

        if "pass_goal_assist" in passes.columns:
            assists = passes[passes["pass_goal_assist"] == True]
            if not assists.empty and "player" in assists.columns:
                counts = assists["player"].value_counts()
                out["max_assist"] = counts.idxmax()
                out["max_assists"] = int(counts.max())
        else:
            # fallback: vincular via shot_key_pass_id
            if "shot_key_pass_id" in df_t.columns and "id" in df_t.columns:
                goals = df_t[(df_t["type"] == "Shot") & (df_t["shot_outcome"] == "Goal")]
                if team_col:
                    goals = goals[goals[team_col] == team_name]
                id2player = passes.set_index("id")["player"].to_dict()
                passers = goals["shot_key_pass_id"].map(id2player).dropna()
                if not passers.empty:
                    counts = passers.value_counts()
                    out["max_assist"] = counts.idxmax()
                    out["max_assists"] = int(counts.max())

    # ========================
    # 3) Portero con más atajadas
    # ========================
    if "type" in df_t.columns:
        gk = df_t[df_t["type"].isin(["Goal Keeper", "Goalkeeper"])].copy()
        if team_col:
            gk = gk[gk[team_col] == team_name]
        if not gk.empty:
            save_types = {
                "Shot Saved", "Penalty Saved", "Shot Saved Off Target",
                "Shot Saved To Post", "Saved To Post", "Save"
            }
            gk["goalkeeper_type"] = gk["goalkeeper_type"].astype(str)
            saves = gk[gk["goalkeeper_type"].isin(save_types)]
            if saves.empty and "goalkeeper_outcome" in gk.columns:
                saves = gk[gk["goalkeeper_outcome"].astype(str).str.contains("Saved", case=False, na=False)]
            if not saves.empty and "player" in saves.columns:
                cnt = saves["player"].value_counts()
                out["top_shotstopper"] = cnt.idxmax()
                out["top_saves"] = int(cnt.max())

    # ========================
    # 4) Jugador con mayor centralidad (PageRank)
    # ========================
    if all(c in df_t.columns for c in ["type", "player", "pass_recipient"]):
        passes = df_t[df_t["type"] == "Pass"].copy()
        if team_col:
            passes = passes[passes[team_col] == team_name]

        # colapsa a aristas con peso (conteo de pases) y aplica el mismo umbral que en build_pass_network
        edges_df = (
            passes.dropna(subset=["player", "pass_recipient"])
                  .groupby(["player", "pass_recipient"])
                  .size().reset_index(name="weight")
        )
        # usa el mismo min_passes que en tu app (ajústalo o pásalo como parámetro si lo tienes global)
        MIN_PASSES_FOR_PAGERANK = 3
        edges_df = edges_df[edges_df["weight"] >= max(1, int(MIN_PASSES_FOR_PAGERANK))]

        # arma el grafo y calcula PageRank ponderado
        G = nx.DiGraph()
        for _, r in edges_df.iterrows():
            G.add_edge(r["player"], r["pass_recipient"], weight=int(r["weight"]))

        if len(G) > 0:
            pr = nx.pagerank(G, alpha=0.85, weight="weight")
            top_p = max(pr.items(), key=lambda x: x[1])
            out["most_central_player"] = top_p[0]
            out["pagerank_score"] = round(top_p[1], 4)

    return out




def build_pass_network(
    df_t: pd.DataFrame,
    team_name: str = "Club América",
    min_passes: int = 3,
    top_n: int = 10,   # <<--- NUEVO
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], str]:
    # detectar columna de equipo
    team_col = next((c for c in ["team", "team_name", "possession_team_name"] if c in df_t.columns), None)

    # filtrar pases del equipo
    passes = df_t[df_t.get("type") == "Pass"].copy()
    if team_col:
        passes = passes[passes[team_col] == team_name]

    if passes.empty or "player" not in passes.columns or "pass_recipient" not in passes.columns:
        return (pd.DataFrame(columns=["player","pr","indeg","outdeg","total","size"]),
                pd.DataFrame(columns=["player","pass_recipient","weight"]),
                {}, "-")

    # aristas: player -> pass_recipient con peso
    edges_df = (
        passes.dropna(subset=["player","pass_recipient"])
              .groupby(["player","pass_recipient"])
              .size().reset_index(name="weight")
    )
    edges_df = edges_df[edges_df["weight"] >= max(1, int(min_passes))]

    # grafo completo para calcular centralidad
    G = nx.DiGraph()
    for _, r in edges_df.iterrows():
        G.add_edge(r["player"], r["pass_recipient"], weight=int(r["weight"]))

    if len(G) == 0:
        return (pd.DataFrame(columns=["player","pr","indeg","outdeg","total","size"]),
                edges_df, {}, "-")

    pr = nx.pagerank(G, alpha=0.85, weight="weight")
    top_player = max(pr.items(), key=lambda x: x[1])[0] if pr else "-"

    indeg  = dict(G.in_degree(weight="weight"))
    outdeg = dict(G.out_degree(weight="weight"))
    total  = {n: indeg.get(n,0)+outdeg.get(n,0) for n in G.nodes}

    nodes_df_full = pd.DataFrame({
        "player": list(G.nodes),
        "pr": [pr.get(n,0.0) for n in G.nodes],
        "indeg": [indeg.get(n,0) for n in G.nodes],
        "outdeg": [outdeg.get(n,0) for n in G.nodes],
        "total": [total.get(n,0) for n in G.nodes],
    }).sort_values("pr", ascending=False)

    # --- mantener solo TOP-N por PageRank ---
    top_players = nodes_df_full.head(top_n)["player"].tolist()
    nodes_df = nodes_df_full[nodes_df_full["player"].isin(top_players)].copy()

    # filtrar aristas entre los TOP-N
    edges_df_top = edges_df[
        edges_df["player"].isin(top_players) & edges_df["pass_recipient"].isin(top_players)
    ].reset_index(drop=True)

    # tamaño de marker normalizado dentro del subconjunto
    if nodes_df["pr"].max() > 0:
        nodes_df["size"] = 10 + 40 * (nodes_df["pr"] - nodes_df["pr"].min()) / (nodes_df["pr"].max() - nodes_df["pr"].min() + 1e-9)
    else:
        nodes_df["size"] = 15

    return nodes_df.reset_index(drop=True), edges_df_top, pr, top_player



import plotly.colors as pc


def plot_pass_network(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    highlight: str = "",
    title: str = "Red de pases (PageRank)",
    seed: int = 7,
    min_line_width: float = 1.0,
    max_line_width: float = 6.0,
    edge_alpha: float = 0.65,
    colorscale_name: str = "Turbo",  # "Viridis", "Cividis", etc.
) -> go.Figure:
    """
    Dibuja la red de pases con:
      - Nodos escalados por PageRank (columna 'size' en nodes_df)
      - Aristas con grosor y color proporcionales a 'weight'
      - Resalta 'highlight' (jugador más influyente)
    Espera:
      nodes_df: columns ['player','pr','indeg','outdeg','total','size']
      edges_df: columns ['player','pass_recipient','weight']
    """
    # --- Construir grafo desde edges_df ---
    G = nx.DiGraph()
    for _, r in edges_df.iterrows():
        G.add_edge(r["player"], r["pass_recipient"], weight=int(r["weight"]))
    for _, r in nodes_df.iterrows():
        if r["player"] not in G:
            G.add_node(r["player"])

    if len(G) == 0:
        return go.Figure()

    # --- Layout estable por pesos ---
    pos = nx.spring_layout(G, seed=seed, weight="weight", k=1.2)

    # --- Preparar escalas para width/color por peso ---
    weights = np.array([data.get("weight", 1) for _, _, data in G.edges(data=True)], dtype=float)
    if weights.size > 0:
        w_min, w_max = weights.min(), weights.max()
        # evita división por cero cuando todos los pesos son iguales
        denom = (w_max - w_min) if (w_max - w_min) > 1e-9 else 1.0
        width_scaled = min_line_width + (max_line_width - min_line_width) * ((weights - w_min) / denom)

        # colores continuos según peso (0..1) → escala
        norm_vals = (weights - w_min) / denom
        colorscale = pc.get_colorscale(colorscale_name)
        edge_colors = [pc.sample_colorscale(colorscale, float(v))[0] for v in norm_vals]
        # añadir alpha a cada color (edge_alpha)
        def with_alpha(rgb_str: str, alpha: float) -> str:
            # rgb_str puede venir como 'rgb(r,g,b)' o '#RRGGBB'
            if rgb_str.startswith("rgb"):
                nums = rgb_str[rgb_str.find("(")+1:rgb_str.find(")")].split(",")
                r, g, b = [int(float(x)) for x in nums]
            else:
                # hex → rgb
                rgb = pc.hex_to_rgb(rgb_str)
                r, g, b = rgb
            return f"rgba({r},{g},{b},{alpha})"
        edge_rgba = [with_alpha(c, edge_alpha) for c in edge_colors]
    else:
        width_scaled = np.array([])
        edge_rgba = []

    # --- Trazos de aristas (uno por arista para widths/colors distintos) ---
    edge_traces = []
    edges_list = list(G.edges(data=True))
    for i, (u, v, data) in enumerate(edges_list):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        w = int(data.get("weight", 1))
        lw = float(width_scaled[i]) if width_scaled.size else 1.5
        lc = edge_rgba[i] if edge_rgba else "rgba(180,180,180,0.45)"

        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=lw, color=lc),
                hoverinfo="text",
                hovertext=f"{u} → {v}<br>{w} pases",
                showlegend=False,
            )
        )

    # --- Nodos ---
    xs, ys, sizes, texts, colors, htexts = [], [], [], [], [], []
    for _, r in nodes_df.iterrows():
        p = r["player"]
        x, y = pos[p]
        xs.append(x); ys.append(y)
        sizes.append(r.get("size", 15))
        colors.append("#F9B233" if p == highlight else "#1f77b4")  # dorado para highlight
        texts.append(p.split()[0])  # etiqueta corta (primer nombre)
        htexts.append(f"{p}<br>PR={r['pr']:.4f}<br>Total pases={int(r['total'])}")

    node_trace = go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        text=texts,
        textposition="bottom center",
        hovertext=htexts,
        hoverinfo="text",
        marker=dict(
            size=sizes,
            line=dict(width=1, color="#ffffff"),
            opacity=0.95,
            color=colors
        ),
        showlegend=False,
    )

    # --- Figura final ---
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ----------------------------
# Estilo de juego (heurístico v0)
# ----------------------------

def get_xg_col(df: pd.DataFrame) -> Optional[str]:
    return first_col(df, ["shot_statsbomb_xg", "shot_xg", "xg"])


def infer_style_summary(df_t: pd.DataFrame, team_name: str = "Club América") -> str:
    """Heurísticas explicativas sobre goles/tiros del equipo."""
    if "type" not in df_t.columns:
        return "No hay eventos para inferir estilo."
    team_col = first_col(df_t, ["team", "team_name", "possession_team_name"])  
    shots = df_t[df_t["type"] == "Shot"].copy()
    if team_col:
        shots = shots[shots[team_col] == team_name]
    if shots.empty:
        return "No hay tiros registrados para inferir estilo."

    goals = shots.copy()
    if "shot_outcome" in shots.columns:
        goals = shots[shots["shot_outcome"] == "Goal"].copy()
    if goals.empty:
        goals = shots

    # shares
    play_pattern = goals["play_pattern"] if "play_pattern" in goals.columns else pd.Series(dtype=object)
    shot_type = goals["shot_type"] if "shot_type" in goals.columns else pd.Series(dtype=object)
    body_part = (goals["shot_body_part"] if "shot_body_part" in goals.columns else
                 goals["body_part"] if "body_part" in goals.columns else pd.Series(dtype=object))

    def share_bool(s: pd.Series) -> float:
        denom = len(goals) if len(goals) else 1
        return float(s.sum())/denom

    share_open_play = share_bool(shot_type.astype(str).eq("Open Play")) if not shot_type.empty else np.nan
    share_counter   = share_bool(play_pattern.astype(str).eq("From Counter")) if not play_pattern.empty else np.nan
    share_set_piece = share_bool(play_pattern.astype(str).isin(["From Corner","From Free Kick","From Kick Off"])) if not play_pattern.empty else np.nan
    share_headers   = share_bool(body_part.astype(str).eq("Head")) if not body_part.empty else np.nan

    # tiros lejanos ~ x < 102 sobre 'location'
    far_out = np.nan
    if "location" in goals.columns:
        xs, _ = parse_xy_series(goals["location"])
        if xs.notna().any():
            far_out = float((xs < 102).mean())

    tags = []
    def tag(cond, text):
        if pd.notna(cond) and cond:
            tags.append(text)

    tag(share_open_play > 0.65 if pd.notna(share_open_play) else False, "predominantemente de juego abierto")
    tag(share_set_piece > 0.22 if pd.notna(share_set_piece) else False, "fuerte a balón parado")
    tag(share_counter   > 0.18 if pd.notna(share_counter)   else False, "peligroso en contraataques")
    tag(share_headers   > 0.22 if pd.notna(share_headers)   else False, "con buen juego aéreo en área")
    tag(far_out         > 0.18 if pd.notna(far_out)         else False, "con remate desde media distancia")

    if not tags:
        tags.append("equilibrado entre fases ofensivas")

    return f"Con base en {len(goals)} tiros/goles, el equipo luce " + ", ".join(tags) + "."

# ----------------------------
# Datos para heatmap (goles)
# ----------------------------

def goals_xy_for_heatmap(df_t: pd.DataFrame, team_name: str = "Club América") -> pd.DataFrame:
    if "type" not in df_t.columns:
        return pd.DataFrame(columns=["x","y"])
    team_col = first_col(df_t, ["team", "team_name", "possession_team_name"])  
    shots = df_t[df_t["type"] == "Shot"].copy()
    if team_col:
        shots = shots[shots[team_col] == team_name]
    if "shot_outcome" in shots.columns:
        shots = shots[shots["shot_outcome"] == "Goal"]
    if shots.empty:
        return pd.DataFrame(columns=["x","y"])

    # Prefer shot_end_location, fallback a location
    if "shot_end_location" in shots.columns:
        xs, ys = parse_xy_series(shots["shot_end_location"])
    elif "location" in shots.columns:
        xs, ys = parse_xy_series(shots["location"])
    else:
        return pd.DataFrame(columns=["x","y"])

    out = pd.DataFrame({"x": xs, "y": ys}).dropna()
    # Limitar al rango de StatsBomb (120x80)
    out = out[(out["x"].between(0, 120)) & (out["y"].between(0, 80))]
    return out


# ===== Estilo de juego (adaptado a apertura_america.csv) =====



# ----------------------------
# Helpers para posiciones (vienen como "[x y]" en CSV)
# ----------------------------
def _to_xy(val: Union[str, Iterable, float, int]):
    """Devuelve (x, y) como floats a partir de columnas tipo '[x y]' o listas."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return (np.nan, np.nan)
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 2:
        return float(val[0]), float(val[1])
    if isinstance(val, str):
        s = val.strip().replace(",", " ").replace("  ", " ")
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        parts = [p for p in s.split() if p]
        if len(parts) >= 2:
            try:
                return float(parts[0]), float(parts[1])
            except:
                return (np.nan, np.nan)
    return (np.nan, np.nan)

def _series_x(series: pd.Series) -> pd.Series:
    return series.map(lambda v: _to_xy(v)[0])

def _series_y(series: pd.Series) -> pd.Series:
    return series.map(lambda v: _to_xy(v)[1])

# ----------------------------
# Cálculo de las 5 dimensiones
# ----------------------------
def compute_style_metrics_sb8(df_team: pd.DataFrame) -> dict:
    """
    Calcula índices 0–1 por dimensión usando columnas reales del CSV:
      - xG: shot_statsbomb_xg
      - tipo de evento: type
      - jugada: play_pattern
      - presión: type == 'Pressure' y 'under_pressure'
      - duelos (tackles): type == 'Duel' + duel_type == 'Tackle'
      - intercepciones: type == 'Interception'
      - ubicaciones: location, shot_end_location, pass_end_location
    """
    metrics = {}

    # ------- 1) Intensidad ofensiva -------
    xg = df_team["shot_statsbomb_xg"].fillna(0).sum()
    shots = (df_team["type"] == "Shot").sum()
    progressive_flag = (
        # aproximación: pase que avanza significativamente hacia portería (delta x>10)
        (_series_x(df_team["pass_end_location"]) - _series_x(df_team["location"])) > 10
    )
    progressive_passes = (df_team["type"].eq("Pass") & progressive_flag).sum()

    # toques en el área (x > ~102 en la escala StatsBomb 120x80)
    touches_box = (_series_x(df_team["location"]) > 102).sum()

    # Escala suave con tanh para 0–1
    metrics["Intensidad ofensiva"] = float(
        np.tanh((xg + 0.08 * shots + 0.04 * progressive_passes + 0.02 * touches_box) / 10)
    )

    # ------- 2) Estilo de ataque (transición vs posesión) -------
    # From Counter / duración promedio de posesiones
    possessions = df_team["possession"].nunique()
    avg_pos_dur = (
        df_team.groupby("possession")["duration"].sum().fillna(0).mean()
        if possessions > 0 else 0
    )
    share_counters = (
        (df_team["play_pattern"] == "From Counter").sum() / max(1, possessions)
    )

    # normalizamos: ataques directos altos -> más vertical
    # posesión larga también aporta, pero en sentido opuesto (mezclamos 50/50)
    direct_score = min(1.0, share_counters * 3)    # resalta si > ~33% de posesiones
    long_pos_score = min(1.0, avg_pos_dur / 20.0)  # 20s como "larga"
    metrics["Estilo de ataque"] = float(np.tanh((0.6 * direct_score + 0.4 * long_pos_score)))

    # ------- 3) Juego a balón parado -------
    set_piece_events = df_team["play_pattern"].isin(["From Corner", "From Free Kick", "Penalty"]).sum()
    total_shots = (df_team["type"] == "Shot").sum()
    metrics["Balón parado"] = float(min(1.0, set_piece_events / max(1, total_shots)))

    # ------- 4) Juego defensivo -------
    pressures = (df_team["type"] == "Pressure").sum()
    tackles = (df_team["type"].eq("Duel") & df_team["duel_type"].eq("Tackle")).sum()
    interceptions = (df_team["type"] == "Interception").sum()
    # también consideramos eventos realizados "under_pressure" (agresividad defensiva)
    recv_under_pressure = df_team["under_pressure"].fillna(False).sum()
    metrics["Juego defensivo"] = float(
        np.tanh((pressures + 0.8 * tackles + 0.8 * interceptions + 0.5 * recv_under_pressure) / 250)
    )

    # ------- 5) Distribución espacial -------
    # Medimos cuán dispersos son los destinos de tiro en el eje X e Y (spread -> distribución)
    sx = _series_x(df_team["shot_end_location"])
    sy = _series_y(df_team["shot_end_location"])
    # si no hay tiros, caemos a dispersión de pases ofensivos
    if sx.notna().sum() < 5:
        sx = _series_x(df_team.loc[df_team["type"]=="Pass", "pass_end_location"])
        sy = _series_y(df_team.loc[df_team["type"]=="Pass", "pass_end_location"])
    # std conjunta -> cuanto mayor, más distribuido. Cap a 1 usando tanh sobre escala típica (30/20)
    spread = np.sqrt(np.nanvar(sx) + np.nanvar(sy))
    metrics["Distribución espacial"] = float(np.tanh(spread / 35.0))

    return metrics

# ----------------------------
# Resumen textual automático
# ----------------------------
def summarize_play_style(metrics: dict) -> str:
    parts = []
    m = metrics  # alias

    parts.append(
        "Muy ofensivo y generador de peligro."
        if m["Intensidad ofensiva"] > 0.7 else
        "Equilibrado en la generación de peligro."
        if m["Intensidad ofensiva"] > 0.4 else
        "Producción ofensiva limitada."
    )

    parts.append(
        "Ataca con transiciones rápidas/verticales."
        if m["Estilo de ataque"] > 0.6 else
        "Prefiere construir con posesiones más largas."
    )

    parts.append(
        "Amenaza real en balón parado."
        if m["Balón parado"] > 0.28 else
        "Baja dependencia del balón parado."
    )

    parts.append(
        "Presión e intervenciones defensivas frecuentes."
        if m["Juego defensivo"] > 0.6 else
        "Bloque medio/bajo y defensa más posicional."
    )

    parts.append(
        "Ataque distribuido por varias zonas."
        if m["Distribución espacial"] > 0.5 else
        "Ataque concentrado en carriles específicos."
    )

    return " ".join(parts)

# ----------------------------
# Radar Plotly
# ----------------------------
def plot_playstyle_radar(metrics: dict):
    cats = list(metrics.keys())
    vals = [float(metrics[c]) for c in cats]
    # cerrar polígono
    cats += cats[:1]
    vals += vals[:1]

    fig = go.Figure(data=[
        go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            name="Estilo de juego", line_color="#3BA3FF"
        )
    ])
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False, template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        height=420
    )
    return fig

# ----------------------------
# Componente de Streamlit
# ----------------------------
def display_play_style_section(df_team: pd.DataFrame, team_name: str):
    st.markdown("## 🎯 Estilo de juego")
    metrics = compute_style_metrics_sb8(df_team)
    fig = plot_playstyle_radar(metrics)
    summary = summarize_play_style(metrics)

    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.plotly_chart(fig, use_container_width=True)
        # tabla rápida opcional:
        #st.dataframe(pd.DataFrame({"Dimensión": list(metrics.keys()),
        #                           "Índice (0–1)": [round(v, 3) for v in metrics.values()]}))
    with c2:
        st.markdown(f"### {team_name}")
        st.markdown(summary)
        #st.caption("Los índices se normalizan a [0,1] con escalas robustas (tanh). Ajusta umbrales según tu liga/muestra.")




# ================== Heatmap avanzado de goles (origen del tiro) ==================

# —— parsers que ya usas ——
def _to_xy(val):
    if val is None or (isinstance(val, float) and np.isnan(val)): return (np.nan, np.nan)
    if isinstance(val, (list, tuple, np.ndarray)) and len(val) >= 2: return float(val[0]), float(val[1])
    if isinstance(val, str):
        s = val.strip().replace(",", " ").replace("  ", " ")
        if s.startswith("[") and s.endswith("]"): s = s[1:-1]
        parts = [p for p in s.split() if p]
        if len(parts) >= 2:
            try: return float(parts[0]), float(parts[1])
            except: return (np.nan, np.nan)
    return (np.nan, np.nan)

def _series_x(s: pd.Series) -> pd.Series: return s.map(lambda v: _to_xy(v)[0])
def _series_y(s: pd.Series) -> pd.Series: return s.map(lambda v: _to_xy(v)[1])

# —— cancha —— (igual que antes)
def _add_pitch_shapes(fig: go.Figure):
    fig.add_shape(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color="#CCCCCC", width=2))
    fig.add_shape(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color="#BBBBBB", width=1))
    fig.add_shape(type="circle", x0=50, y0=30, x1=70, y1=50, line=dict(color="#BBBBBB", width=1))
    # Áreas
    fig.add_shape(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color="#BBBBBB", width=1))
    fig.add_shape(type="rect", x0=0, y0=30.34, x1=6, y1=49.66, line=dict(color="#BBBBBB", width=1))
    fig.add_shape(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color="#BBBBBB", width=1))
    fig.add_shape(type="rect", x0=114, y0=30.34, x1=120, y1=49.66, line=dict(color="#BBBBBB", width=1))
    # Penales y arcos del área (aprox)
    fig.add_trace(go.Scatter(x=[12,108], y=[40,40], mode="markers",
                             marker=dict(size=4, color="#BBBBBB"), hoverinfo="skip", showlegend=False))
    for cx, sign in [(12, +1), (108, -1)]:
        theta = np.linspace(-np.pi/2, np.pi/2, 100) if sign>0 else np.linspace(np.pi/2, 3*np.pi/2, 100)
        r=10; x = cx + r*np.cos(theta); y = 40 + r*np.sin(theta)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="#BBBBBB", width=1),
                                 hoverinfo="skip", showlegend=False))

# —— util: hist2d con NaN en celdas vacías ——
def _hist2d_nan(xs, ys, nbinsx, nbinsy):
    xb = np.linspace(0, 120, nbinsx + 1)
    yb = np.linspace(0, 80,  nbinsy + 1)
    H, xedges, yedges = np.histogram2d(xs, ys, bins=[xb, yb])
    xcent = (xedges[:-1] + xedges[1:]) / 2.0
    ycent = (yedges[:-1] + yedges[1:]) / 2.0
    Z = H.T  # (ny, nx)
    Z = np.where(Z == 0, np.nan, Z)
    return xcent, ycent, Z

# —— suavizado gaussiano (usa scipy si está, si no, fallback simple) ——
def _gaussian_blur(Z, sigma):
    try:
        from scipy.ndimage import gaussian_filter  # si está disponible
        return gaussian_filter(np.nan_to_num(Z, nan=0.0), sigma=sigma)
    except Exception:
        # Fallback: pequeño kernel gaussiano manual
        k = int(max(1, round(3*sigma)))
        ax = np.arange(-k, k+1)
        kernel = np.exp(-(ax**2)/(2*sigma**2))
        kernel = kernel / kernel.sum()
        A = np.nan_to_num(Z, nan=0.0)
        # conv separable 1D en x e y
        A = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, A)
        A = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, A)
        return A

def plot_goal_heatmap_advanced(df_team: pd.DataFrame,
                               nbinsx: int = 36,
                               nbinsy: int = 24,
                               style: str = "Suave",
                               sigma: float = 1.2,
                               show_grid: bool = False):
    """
    style: "Suave" (blur tipo KDE) o "Cuadrícula" (celdas marcadas).
    """
    m = (df_team["type"] == "Shot") & (df_team["shot_outcome"] == "Goal")
    goals = df_team.loc[m].copy()
    if goals.empty:
        return go.Figure().update_layout(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            annotations=[dict(text="No hay goles para mostrar", x=0.5, y=0.5, showarrow=False)]
        )

    xs = _series_x(goals["location"])
    ys = _series_y(goals["location"])
    mask = xs.notna() & ys.notna()
    xs = xs[mask].clip(0, 120).values
    ys = ys[mask].clip(0, 80).values

    xcent, ycent, Z = _hist2d_nan(xs, ys, nbinsx, nbinsy)

    if style == "Suave":
        Zs = _gaussian_blur(Z, sigma=sigma)
        # normaliza 0–1 para un look tipo “heatmap continuo”
        maxz = Zs.max() if np.isfinite(Zs).any() else 1.0
        Zplot = np.where(Zs<=0, np.nan, Zs / (maxz if maxz>0 else 1))
        hm = go.Heatmap(
            x=xcent, y=ycent, z=Zplot,
            colorscale="YlOrRd", showscale=True, opacity=0.95,
            hovertemplate="x %{x:.1f}, y %{y:.1f}<br>intensidad %{z:.2f}<extra></extra>",
            zmin=0, zmax=1, zsmooth="best"
        )
    else:  # "Cuadrícula"
        hm = go.Heatmap(
            x=xcent, y=ycent, z=Z,
            colorscale="YlOrRd", showscale=True, opacity=0.9,
            hovertemplate="x %{x:.1f}, y %{y:.1f}<br>goles %{z:.0f}<extra></extra>",
            zauto=True, zsmooth=False
        )

    fig = go.Figure(data=[hm])
    _add_pitch_shapes(fig)

    # opcional: dibujar líneas de la cuadrícula
    if style == "Cuadrícula" and show_grid:
        for xc in np.linspace(0, 120, nbinsx+1):
            fig.add_shape(type="line", x0=xc, y0=0, x1=xc, y1=80,
                          line=dict(color="rgba(255,255,255,0.15)", width=1))
        for yc in np.linspace(0, 80, nbinsy+1):
            fig.add_shape(type="line", x0=0, y0=yc, x1=120, y1=yc,
                          line=dict(color="rgba(255,255,255,0.15)", width=1))

    fig.update_layout(
        xaxis=dict(range=[0,120], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0,80],  showgrid=False, zeroline=False, visible=False, autorange="reversed"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10), height=520
    )
    return fig

# —— sección Streamlit ——
def display_goal_zones_section(df_team: pd.DataFrame, team_name: str):
    import streamlit as st
    st.markdown("## 🗺️ Zonas de origen de goles")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        style = st.radio("Estilo de mapa", ["Suave", "Cuadrícula"], horizontal=True)
        nbx = st.slider("Resolución horizontal (bins)", 18, 72, 36, 2)
        nby = st.slider("Resolución vertical (bins)", 12, 48, 24, 2)
        sigma = st.slider("Suavizado (σ)", 0.6, 3.0, 1.2, 0.1) if style=="Suave" else 1.2
        show_grid = st.toggle("Mostrar líneas de cuadrícula", value=False) if style=="Cuadrícula" else False

        fig = plot_goal_heatmap_advanced(df_team, nbinsx=nbx, nbinsy=nby,
                                         style=style, sigma=sigma, show_grid=show_grid)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        total_goals = int(((df_team["type"]=="Shot") & (df_team["shot_outcome"]=="Goal")).sum())
        st.metric("Goles en el torneo", total_goals)
        st.caption("Se usa **location** (origen del tiro que terminó en gol). Ajusta bins y σ para acercarte al look de tus ejemplos.")

# === helpers de roster / OBV-like (append al final de analytics_helpers.py) ===


# ==== ROSTER: construir players_df desde un DF de eventos ya filtrado ====
#from __future__ import annotations

from typing import Optional, Dict, Tuple

def ah_first_col(df: pd.DataFrame, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default

def ah_to_xy(row: pd.Series) -> Tuple[float, float]:
    loc = row.get("location", None)
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    if "x" in row and "y" in row:
        return float(row["x"]), float(row["y"])
    return np.nan, np.nan

def ah_pitch_value(x: float, y: float, attacking_left_to_right: bool=True) -> float:
    if not np.isfinite(x) or not np.isfinite(y):
        return 0.0
    x_norm = np.clip(x / 120.0, 0, 1)
    y_norm = np.clip(y / 80.0, 0, 1)
    if not attacking_left_to_right:
        x_norm = 1.0 - x_norm
    centrality = 1.0 - (2.0 * abs(y_norm - 0.5))
    base = 0.7 * (x_norm ** 1.6) + 0.3 * (centrality ** 2)
    back_penalty = 0.10 * (1.0 - x_norm)
    return float(base - back_penalty)

def ah_action_delta_value(curr_row: pd.Series, next_row: Optional[pd.Series]) -> float:
    bx, by = ah_to_xy(curr_row)
    ax, ay = np.nan, np.nan
    pel = curr_row.get("pass_end_location", None)
    if isinstance(pel, (list, tuple)) and len(pel) >= 2:
        ax, ay = float(pel[0]), float(pel[1])
    elif next_row is not None:
        ax, ay = ah_to_xy(next_row)
    if not np.isfinite(ax) or not np.isfinite(ay):
        ax, ay = bx, by
    return ah_pitch_value(ax, ay, True) - ah_pitch_value(bx, by, True)

# def ah_compute_obv_like(events: pd.DataFrame) -> pd.DataFrame:
#     df = events.copy()
#     c_team      = ah_first_col(df, ["team","team.name","possession_team","possession_team.name"], "team")
#     c_player_id = ah_first_col(df, ["player_id","player.id"], "player_id")
#     c_event_id  = ah_first_col(df, ["id","event_id"], "id")
#     c_type      = ah_first_col(df, ["type","type.name"], "type")
#     c_poss      = ah_first_col(df, ["possession"], "possession")
#     for c in (c_team, c_player_id, c_event_id, c_type, c_poss):
#         if c not in df.columns:
#             df[c] = np.nan

#     order_cols = [c for c in ["match_id","period","minute","second","index"] if c in df.columns]
#     if order_cols:
#         df = df.sort_values(order_cols).reset_index(drop=True)

#     df["_next_idx_same_poss"] = (
#         df.groupby(c_poss, dropna=False).apply(lambda g: g.index.to_series().shift(-1)).reset_index(level=0, drop=True)
#     )

#     defensive_keywords = {"Interception","Tackle","Block","Clearance","Ball Recovery","Pressure","Foul Won","Foul Committed"}
#     pass_keywords = {"Pass","Carry","Dribble","Shot","Cross"}

#     rows = []
#     for _, row in df.iterrows():
#         ev_type = str(row.get(c_type))
#         next_row = df.loc[row["_next_idx_same_poss"]] if pd.notna(row["_next_idx_same_poss"]) else None
#         delta = ah_action_delta_value(row, next_row)
#         obv_off = 0.0; obv_def = 0.0
#         if any(k in ev_type for k in pass_keywords):
#             obv_off = max(delta, 0.0)
#         elif any(k in ev_type for k in defensive_keywords):
#             obv_def = max(-delta, 0.0)
#         else:
#             if delta > 0:
#                 obv_off = 0.5 * delta
#         rows.append({
#             "event_id": row.get(c_event_id),
#             "player_id": row.get(c_player_id),
#             "team": row.get(c_team),
#             "possession": row.get(c_poss),
#             "obv_like_off": float(obv_off),
#             "obv_like_def": float(obv_def),
#         })
#     return pd.DataFrame(rows)

def ah_compute_obv_like(events: pd.DataFrame) -> pd.DataFrame:
    """
    Usa los campos OBV nativos del DF:
      - obv_total_net  (si no existe, se deriva como obv_for_net - obv_against_net)
      - obv_for_net    (si no existe, se deriva como obv_for_after - obv_for_before)
      - obv_against_net(si no existe, se deriva como obv_against_after - obv_against_before)

    Regresa por evento:
      ['event_id','player_id','team','possession','obv_like_off','obv_like_def']
    donde:
      - obv_like_off = max(obv_total_net, 0)
      - obv_like_def = max(-obv_total_net, 0)
    """
    df = events.copy()

    # Columnas clave
    c_team      = ah_first_col(df, ["team","team.name","possession_team","possession_team.name"], "team")
    c_player_id = ah_first_col(df, ["player_id","player.id"], "player_id")
    c_event_id  = ah_first_col(df, ["id","event_id"], "id")
    c_poss      = ah_first_col(df, ["possession"], "possession")

    for c in (c_team, c_player_id, c_event_id, c_poss):
        if c not in df.columns:
            df[c] = np.nan

    # Asegura/deriva obv_for_net y obv_against_net
    if "obv_for_net" not in df.columns:
        df["obv_for_net"] = (
            pd.to_numeric(df.get("obv_for_after"), errors="coerce")
            - pd.to_numeric(df.get("obv_for_before"), errors="coerce")
        )
    if "obv_against_net" not in df.columns:
        df["obv_against_net"] = (
            pd.to_numeric(df.get("obv_against_after"), errors="coerce")
            - pd.to_numeric(df.get("obv_against_before"), errors="coerce")
        )

    # Asegura/deriva obv_total_net
    if "obv_total_net" not in df.columns:
        df["obv_total_net"] = pd.to_numeric(df["obv_for_net"], errors="coerce") - pd.to_numeric(df["obv_against_net"], errors="coerce")

    # Limpieza numérica
    total = pd.to_numeric(df["obv_total_net"], errors="coerce").fillna(0.0)

    obv_like_off = np.where(total > 0, total, 0.0)
    obv_like_def = np.where(total < 0, -total, 0.0)

    out = pd.DataFrame({
        "event_id":   df.get(c_event_id),
        "player_id":  df.get(c_player_id),
        "team":       df.get(c_team),
        "possession": df.get(c_poss),
        "obv_like_off": obv_like_off.astype(float),
        "obv_like_def": obv_like_def.astype(float),
    })

    return out

def ah_estimate_minutes(events: pd.DataFrame) -> pd.DataFrame:
    c_player_id = ah_first_col(events, ["player_id","player.id"], "player_id")
    c_minute    = ah_first_col(events, ["minute"], "minute")
    c_second    = ah_first_col(events, ["second"], "second")
    df = events.copy()
    for c in (c_player_id, c_minute, c_second):
        if c not in df.columns:
            df[c] = np.nan
    if "match_id" not in df.columns:
        df["match_id"] = 0
    df["_t"] = df[c_minute].fillna(0) * 60 + df[c_second].fillna(0)
    agg = (
        df.groupby(["match_id", c_player_id], dropna=True)["_t"]
          .agg(lambda s: max(s) - min(s) if len(s) > 1 else max(s))
          .reset_index().rename(columns={c_player_id: "player_id", "_t": "seconds"})
    )
    agg["minutes"] = agg["seconds"] / 60.0
    return agg.groupby("player_id", dropna=True)["minutes"].sum().reset_index()

def ah_map_position_group(events: pd.DataFrame, roster_positions: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Determina el grupo posicional del jugador.
    Prioridad:
      1) Si existe 'position' en eventos → usa mapeo _AH_POSITION_TO_GROUP y guarda 'position_role'.
      2) Si no hay 'position' o es nula → fallback por x_mean (+ GK por tipo de evento).
      3) Si pasas 'roster_positions' (dict {player_id: 'GK'|'DEF'|'MID'|'FWD'}) prevalece sobre lo anterior.
    Devuelve: ['player_id','position_group','position_role']
    """
    df = events.copy()
    c_player_id = ah_first_col(df, ["player_id","player.id"], "player_id")
    if c_player_id not in df.columns:
        df[c_player_id] = np.nan

    # 1) Usar columna 'position' si existe
    pos_df = _ah_coalesce_position_group_from_position_col(df, player_id_col=c_player_id)
    if pos_df is None or pos_df.empty or pos_df["position_group"].isna().all():
        # 2) Fallback por x_mean/GK
        fallback = _fallback_position_group_by_xy(df, c_player_id=c_player_id)
        fallback["position_role"] = None
        pos_df = fallback[["player_id","position_group","position_role"]]
    else:
        # Completar los que quedaron sin grupo con fallback
        unknown_ids = pos_df.loc[pos_df["position_group"].isna(), "player_id"].astype(str).tolist()
        if unknown_ids:
            fallback = _fallback_position_group_by_xy(df[df[c_player_id].astype(str).isin(unknown_ids)], c_player_id=c_player_id)
            pos_df = pos_df.merge(fallback, on="player_id", how="left", suffixes=("","_fb"))
            pos_df["position_group"] = pos_df["position_group"].fillna(pos_df["position_group_fb"])
            pos_df = pos_df.drop(columns=[c for c in pos_df.columns if c.endswith("_fb")])

    # 3) Sobrescribir con mapping explícito si lo pasas
    if roster_positions:
        pos_df["position_group"] = pos_df["player_id"].astype(str).map(roster_positions).fillna(pos_df["position_group"])

    return pos_df[["player_id","position_group","position_role"]]


def _fallback_position_group_by_xy(df: pd.DataFrame, c_player_id: str) -> pd.DataFrame:
    """Heurística cuando no hay columna 'position': GK por evento; DEF/MID/FWD por x_mean; DEF por tipos defensivos si x faltante."""
    c_type = ah_first_col(df, ["type","type.name"], "type")
    if c_type not in df.columns:
        df[c_type] = np.nan

    # GK por tipo de evento
    gk_mask = df[c_type].astype(str).str.contains("Goal Keeper|Goalkeeper|Keeper|Save|GK", case=False, na=False)
    gk_ids = set(df.loc[gk_mask, c_player_id].dropna().astype(str).unique())

    xy = df.apply(ah_to_xy, axis=1, result_type="expand")
    df["_x"] = xy[0]

    pos_xy = df.groupby(c_player_id, dropna=True)["_x"].mean().reset_index()
    pos_xy = pos_xy.rename(columns={c_player_id:"player_id","_x":"x_mean"})

    def _group(pid, x):
        if str(pid) in gk_ids:
            return "GK"
        if not np.isfinite(x):
            # Fallback defensivo si hay eventos típicos de defensa
            evs = df.loc[df[c_player_id] == pid, c_type].astype(str).tolist()
            if any(any(k in e for k in ["Clearance","Interception","Tackle","Blocked"]) for e in evs):
                return "DEF"
            return "MID"
        if x <= 25:  return "DEF"
        if x <= 70:  return "MID"
        return "FWD"

    pos_xy["position_group"] = pos_xy.apply(lambda r: _group(r["player_id"], r["x_mean"]), axis=1)
    return pos_xy[["player_id","position_group"]]

import numpy as np
import pandas as pd
from typing import Optional, Dict

def ah_build_players_df_from_events_v2(
    events: pd.DataFrame,
    roster_positions: Optional[Dict[str, str]] = None,  # opcional: {player_id -> position_group}
    *,
    normalize_by_position: bool = True,
    add_position_relative_cols: bool = True,
    min_minutes: float = 270.0,          # filtra muestras muy pequeñas
    use_total_net_direct: bool = False,  # si True usa obv_total_net per90 directamente (si existe)
) -> pd.DataFrame:
    """
    Construye un DataFrame de jugadores calculando eficiencia OBV como:
        efficiency_raw = obv_for_per90 - obv_against_per90
    (equivalente a obv_total_net per90 si esa columna es coherente).

    Entradas:
      - events: dataframe de eventos StatsBomb (mismo que usas en tu app).
      - roster_positions: dict opcional {player_id -> 'FWD'|'MID'|'DEF'|'GK'} para
        forzar grupo posicional si no viene en los eventos.
      - normalize_by_position: si True, 'efficiency' = z-score dentro del grupo posicional.
      - add_position_relative_cols: añade efficiency_raw, z y percentil.
      - min_minutes: umbral de minutos para considerar muestra "válida".
      - use_total_net_direct: si True y existe 'obv_total_net', usa directamente
        (obv_total_net * 90/minutos) como eficiencia_raw.

    Salida (columnas mínimas):
      ['player_id','player_name','team','position_group','position_role',
       'minutes','efficiency','obv_off_per90','obv_def_per90',
       (opcional) 'efficiency_raw','efficiency_pos_z','efficiency_pos_pct']
    """

    df = events.copy()

    # --------- Detección flexible de columnas clave
    def _first_col(cols, candidates, default=None):
        for c in candidates:
            if c in cols: return c
        return default

    c_player    = _first_col(df.columns, ["player","player.name"], "player")
    c_player_id = _first_col(df.columns, ["player_id","player.id"], "player_id")
    c_team      = _first_col(df.columns, ["team","team.name","possession_team","possession_team.name"], "team")
    c_position  = _first_col(df.columns, ["position","position.name"], None)  # puede no existir
    c_match_id  = _first_col(df.columns, ["match_id","game_id","fixture_id"], "match_id")

    # Crear columnas faltantes sin romper el flujo
    for c in (c_player, c_player_id, c_team, c_match_id):
        if c not in df.columns:
            df[c] = np.nan

    # --------- Asegurar OBV en float y presentes
    for col in ["obv_for_net", "obv_against_net", "obv_total_net"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # --------- Base única de jugadores
    base = (
        df[[c_player_id, c_player, c_team]]
        .dropna(subset=[c_player_id])
        .drop_duplicates()
        .rename(columns={c_player_id:"player_id", c_player:"player_name", c_team:"team"})
    )

    # --------- Minutos por jugador (aprox robusta por partido)
    # minutos_jugador_partido ≈ max(minute + duration/60) cap 100
    if "minute" not in df.columns:
        df["minute"] = 0.0
    if "duration" not in df.columns:
        df["duration"] = 0.0

    def _minutes_by_player(dfe: pd.DataFrame) -> pd.Series:
        g = dfe.groupby(["player_id", c_match_id], dropna=True)
        per_match = g.apply(lambda x: float((pd.to_numeric(x["minute"], errors="coerce").fillna(0)
                                             + pd.to_numeric(x["duration"], errors="coerce").fillna(0)/60.0).max()))
        per_match = per_match.clip(lower=0, upper=100)
        return per_match.groupby("player_id").sum().rename("minutes")

    minutes = _minutes_by_player(df)

    # --------- Agregados OBV por jugador
    agg_cols = []
    if "obv_for_net" in df.columns:     agg_cols.append("obv_for_net")
    if "obv_against_net" in df.columns: agg_cols.append("obv_against_net")
    if "obv_total_net" in df.columns:   agg_cols.append("obv_total_net")

    if len(agg_cols) == 0:
        raise ValueError("No se encontraron columnas OBV ('obv_for_net', 'obv_against_net' o 'obv_total_net').")

    obv_sum = df.groupby("player_id")[agg_cols].sum().reset_index()

    # --------- Posición (grupo y rol)
    # 1) si viene 'position' en eventos —> úsala como 'position_role' y mapea a grupo simple
    # 2) si roster_positions está dado —> sobrescribe 'position_group' con ese dict
    pos = pd.DataFrame({"player_id": base["player_id"].values})
    if c_position and c_position in df.columns:
        pos_role = (df[["player_id", c_position]]
                    .dropna(subset=["player_id"])
                    .drop_duplicates(subset=["player_id"])
                    .rename(columns={c_position: "position_role"}))
        pos = pos.merge(pos_role, on="player_id", how="left")
    else:
        pos["position_role"] = np.nan

    # Mapeo naive role->grupo si no hay dict externo (puedes ajustar)
    def _map_group(role: Optional[str]) -> Optional[str]:
        if not isinstance(role, str): return None
        r = role.lower()
        if any(k in r for k in ["keeper","goalkeeper","gk"]): return "GK"
        if any(k in r for k in ["back","center back","full back","wing back","cb","rb","lb","lcb","rcb"]): return "DEF"
        if any(k in r for k in ["mid","pivot","winger","interior","am","dm","cm","rm","lm"]): return "MID"
        if any(k in r for k in ["forward","striker","wing","cf","lw","rw","st"]): return "FWD"
        return None

    pos["position_group"] = pos["position_role"].apply(_map_group)

    # Sobrescribe con roster_positions si viene
    if roster_positions:
        # roster_positions: {player_id(str/int): 'FWD'|'MID'|'DEF'|'GK'}
        rp = pd.DataFrame({"player_id": list(roster_positions.keys()),
                           "position_group": list(roster_positions.values())})
        pos = pos.drop(columns=["position_group"]).merge(rp, on="player_id", how="left")

    # --------- Arma tabla jugadores
    players = (
        base.merge(obv_sum, on="player_id", how="left")
            .merge(minutes, on="player_id", how="left")
            .merge(pos, on="player_id", how="left")
    )
    players["minutes"] = pd.to_numeric(players["minutes"], errors="coerce").fillna(0.0)

    # --------- Per90
    denom = players["minutes"].where(players["minutes"] > 0, np.nan)

    # ofensa / contra
    if "obv_for_net" in players.columns:
        players["obv_off_per90"] = players["obv_for_net"] * (90.0 / denom)
    else:
        players["obv_off_per90"] = 0.0

    if "obv_against_net" in players.columns:
        players["obv_def_per90"] = players["obv_against_net"] * (90.0 / denom)
    else:
        # si no tenemos 'against', lo compensamos con total si existe (no ideal, pero evita NaN)
        if "obv_total_net" in players.columns and "obv_for_net" in players.columns:
            players["obv_def_per90"] = (players["obv_for_net"] - players["obv_total_net"]) * (90.0 / denom)
        else:
            players["obv_def_per90"] = 0.0

    players[["obv_off_per90","obv_def_per90"]] = players[["obv_off_per90","obv_def_per90"]].replace([np.inf,-np.inf], np.nan).fillna(0.0)

    # --------- Eficiencia cruda
    if use_total_net_direct and "obv_total_net" in players.columns:
        # vía directa: (obv_total_net * 90/min)
        players["efficiency_raw"] = players["obv_total_net"] * (90.0 / denom)
    else:
        # recomendada: diferencia per90 (coherente con tu explicación)
        players["efficiency_raw"] = players["obv_off_per90"] - players["obv_def_per90"]

    players["efficiency_raw"] = players["efficiency_raw"].replace([np.inf,-np.inf], np.nan).fillna(0.0)

    # --------- Filtra muestras pequeñas
    valid = players["minutes"] >= float(min_minutes)
    players.loc[~valid, ["obv_off_per90","obv_def_per90","efficiency_raw"]] = np.nan

    # --------- Normalización por grupo posicional
    if "position_group" not in players.columns:
        players["position_group"] = np.nan
    players["position_group"] = players["position_group"].astype(object).fillna("UNK")

    if normalize_by_position or add_position_relative_cols:
        grp = players.groupby("position_group", dropna=False, group_keys=False)

        mu = grp["efficiency_raw"].transform("mean")
        sd = grp["efficiency_raw"].transform("std").fillna(0.0)
        sd_safe = sd.mask(sd.abs() < 1e-6, 1.0)

        players["efficiency_pos_z"] = (players["efficiency_raw"] - mu) / sd_safe

        def _rank_pct(s: pd.Series) -> pd.Series:
            n = s.size
            if n <= 1: return pd.Series(np.full(n, 0.5), index=s.index)
            r = s.rank(method="average", na_option="keep")
            return (r - 1.0) / (n - 1.0)

        players["efficiency_pos_pct"] = grp["efficiency_raw"].transform(_rank_pct)

        players["efficiency"] = players["efficiency_pos_z"] if normalize_by_position else players["efficiency_raw"]
    else:
        players["efficiency"] = players["efficiency_raw"]

    # --------- Arreglo final
    out_cols = [
        "player_id","player_name","team","position_group","position_role",
        "minutes","efficiency","obv_off_per90","obv_def_per90"
    ]
    if add_position_relative_cols:
        out_cols += ["efficiency_raw","efficiency_pos_z","efficiency_pos_pct"]

    out = players[out_cols].copy()
    out["position_group"] = out["position_group"].replace("UNK", np.nan)
    out = out.sort_values("efficiency", ascending=False, na_position="last").reset_index(drop=True)
    return out

# === Posiciones: mapeo desde columna 'position' a {GK, DEF, MID, FWD} ===

# Valores únicos que dijiste que existen en tu DF:
# [None, 'Center Forward', 'Right Defensive Midfield', 'Left Defensive Midfield',
#  'Left Back', 'Left Center Back','Goalkeeper', 'Right Center Back',
#  'Center Attacking Midfield','Right Back', 'Left Center Midfield',
#  'Center Defensive Midfield', 'Left Midfield', 'Right Wing', 'Left Wing',
#  'Right Midfield','Right Center Midfield', 'Left Center Forward',
#  'Right Center Forward', 'Center Back', 'Left Wing Back', 'Right Wing Back',
#  'Left Attacking Midfield', 'Right Attacking Midfield', 'Substitute']

# 1) Mapeo a grupo posicional
_AH_POSITION_TO_GROUP = {
    # GK
    "Goalkeeper": "GK",

    # DEFENSAS
    "Left Back": "DEF",
    "Right Back": "DEF",
    "Center Back": "DEF",
    "Left Center Back": "DEF",
    "Right Center Back": "DEF",
    "Left Wing Back": "DEF",
    "Right Wing Back": "DEF",

    # MEDIOS
    "Center Defensive Midfield": "MID",
    "Left Defensive Midfield": "MID",
    "Right Defensive Midfield": "MID",
    "Left Midfield": "MID",
    "Right Midfield": "MID",
    "Left Center Midfield": "MID",
    "Right Center Midfield": "MID",
    "Center Attacking Midfield": "MID",
    "Left Attacking Midfield": "MID",
    "Right Attacking Midfield": "MID",

    # DELANTEROS
    "Center Forward": "FWD",
    "Left Center Forward": "FWD",
    "Right Center Forward": "FWD",
    "Left Wing": "FWD",
    "Right Wing": "FWD",

    # Otros / sin rol
    "Substitute": None,
    None: None,
}

def _ah_coalesce_position_group_from_position_col(df: pd.DataFrame, player_id_col: str = "player_id"):
    """
    Si existe la columna 'position', retorna DataFrame con:
      ['player_id','position_role','position_group']
    position_role = valor 'position' tal cual (limpio)
    position_group = mapeo a {GK,DEF,MID,FWD} (o None si no aplica)
    """
    if "position" not in df.columns:
        return None

    tmp = (
        df[[player_id_col, "position"]]
        .dropna(subset=[player_id_col])
        .drop_duplicates()
        .rename(columns={player_id_col: "player_id"})
    )

    # Usa el valor de 'position' como 'position_role' (texto original)
    tmp["position_role"] = tmp["position"].astype(object)

    # Mapea a grupo
    tmp["position_group"] = tmp["position"].map(_AH_POSITION_TO_GROUP)

    # Si algún jugador aparece con múltiples 'position', prioriza la más frecuente
    tmp = (
        tmp.groupby(["player_id","position_role","position_group"])
           .size().reset_index(name="_n")
           .sort_values(["player_id","_n"], ascending=[True, False])
    )
    tmp = tmp.drop_duplicates(subset=["player_id"], keep="first").drop(columns=["_n"])

    return tmp[["player_id","position_role","position_group"]]

    # =======================
# Eficiencia compuesta v1
# =======================
import numpy as np
import pandas as pd

def _ah_col(df, candidates, default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default

def _ah_str_contains(series: pd.Series, keywords: list[str]) -> pd.Series:
    s = series.astype(str)
    mask = False
    for k in keywords:
        mask = mask | s.str.contains(k, case=False, na=False)
    return mask

def ah_compute_player_features_per90(events: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un set de features por jugador normalizados por 90:
    - xg_per90, shots_per90
    - key_passes_per90 (aprox.)
    - pass_attempts_per90, pass_completed_per90, pass_completion
    - prog_passes_per90 (aprox. si hay location; si no, se cae a 0)
    - carries_per90, carries_prog_per90 (aprox.)
    - box_entries_per90 (pases/conducciones a área si hay coords)
    - pressures_per90, recoveries_per90
    - tackles_interceptions_blocks_per90 (TIB)
    - clearances_per90, duels_won_per90
    - turnovers_per90 (miscontrol + dispossessed + dribble fail + pass incompleto)
    """
    if events is None or events.empty:
        return pd.DataFrame(columns=["player_id"])

    df = events.copy()
    c_player_id = _ah_col(df, ["player_id","player.id"], "player_id")
    c_type      = _ah_col(df, ["type","type.name"], "type")

    for c in (c_player_id, c_type):
        if c not in df.columns:
            df[c] = np.nan

    # Minutos por jugador (usa tu estimador actual)
    minutes_df = ah_estimate_minutes(df)  # ['player_id','minutes']
    m = minutes_df.set_index("player_id")["minutes"]

    # ---- Señales por tipo de evento ----
    t = df[c_type].astype(str)

    is_shot      = _ah_str_contains(t, ["Shot"])
    is_pass      = _ah_str_contains(t, ["Pass", "Cross"])
    is_carry     = _ah_str_contains(t, ["Carry"])
    is_dribble   = _ah_str_contains(t, ["Dribble"])
    is_pressure  = _ah_str_contains(t, ["Pressure"])
    is_recovery  = _ah_str_contains(t, ["Ball Recovery"])
    is_tackle    = _ah_str_contains(t, ["Tackle"])
    is_interc    = _ah_str_contains(t, ["Interception"])
    is_block     = _ah_str_contains(t, ["Block"])
    is_clear     = _ah_str_contains(t, ["Clearance"])
    is_duel      = _ah_str_contains(t, ["Duel"])
    is_foul_comm = _ah_str_contains(t, ["Foul Committed"])
    is_misctrl   = _ah_str_contains(t, ["Miscontrol"])
    is_disposs   = _ah_str_contains(t, ["Dispossessed"])

    # ---- xG (si existe) y key passes (aprox) ----
    # StatsBomb suele traer 'shot.statsbomb_xg' o 'shot.xg' (según flatten)
    c_xg = _ah_col(df, ["shot.statsbomb_xg","shot.xg","xg","expected_goals"], None)
    df["_xg"] = pd.to_numeric(df.get(c_xg, 0.0), errors="coerce").fillna(0.0) if c_xg else 0.0

    # Key pass aprox: pases que tienen flag de asistir tiro o 'pass.shot_assist'
    # Si no existe, usa pases hacia último tercio como proxy
    c_assist_flag = _ah_col(df, ["pass.shot_assist","key_pass","is_key_pass"], None)
    key_pass_mask = (is_pass & (df.get(c_assist_flag, False).astype(bool) if c_assist_flag else False))
    # Proxy adicional: si hay 'pass.goal_assist' etc.
    c_goal_ast = _ah_col(df, ["pass.goal_assist","goal_assist"], None)
    if c_goal_ast:
        key_pass_mask = key_pass_mask | (is_pass & df[c_goal_ast].astype(bool))

    # ---- Pass outcomes (completion) ----
    c_pass_outcome = _ah_col(df, ["pass.outcome.name","pass.outcome","outcome.name","outcome"], None)
    pass_complete_mask = is_pass & (
        (df[c_pass_outcome].isna()) if c_pass_outcome else True  # En StatsBomb "NaN" outcome = pase completado
    )

    # ---- Aprox progresivos / box entries usando coords si existen ----
    def _get_xy_pair(row, col="location"):
        v = row.get(col, None)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return float(v[0]), float(v[1])
        return np.nan, np.nan

    have_coords = ("location" in df.columns) or ({"x","y"}.issubset(df.columns))
    if "location" not in df.columns and {"x","y"}.issubset(df.columns):
        df["location"] = df.apply(lambda r: [r["x"], r["y"]], axis=1)

    if have_coords:
        # end location para pases
        if "pass_end_location" not in df.columns and {"pass.end_location_x","pass.end_location_y"}.issubset(df.columns):
            df["pass_end_location"] = df.apply(lambda r: [r["pass.end_location_x"], r["pass.end_location_y"]], axis=1)

        # progresivo si aumenta x ≥15m o entra a último tercio (x>=80)
        def _is_prog_pass(r):
            if not is_pass.loc[r.name]:
                return False
            x1, y1 = _get_xy_pair(r, "location")
            x2, y2 = _get_xy_pair(r, "pass_end_location")
            if not np.isfinite(x1) or not np.isfinite(x2):
                return False
            dx = x2 - x1
            return (dx >= 15) or (x2 >= 80)

        prog_pass_mask = df.apply(_is_prog_pass, axis=1) if have_coords else pd.Series(False, index=df.index)

        # carries progresivos (aumenta x ≥10)
        def _is_prog_carry(r):
            if not is_carry.loc[r.name]:
                return False
            x1, y1 = _get_xy_pair(r, "location")
            # Para carry no siempre hay end_location; si no hay, usa siguiente evento de misma posesión sería ideal (omitimos por simplicidad)
            x2, y2 = _get_xy_pair(r, "carry.end_location") if "carry.end_location" in df.columns else (np.nan, np.nan)
            if not np.isfinite(x1) or not np.isfinite(x2):
                return False
            return (x2 - x1) >= 10

        carries_prog_mask = df.apply(_is_prog_carry, axis=1) if have_coords else pd.Series(False, index=df.index)

        # Entradas al área (x>102 aprox en campo 120)
        def _is_box_entry_pass(r):
            if not is_pass.loc[r.name]:
                return False
            x2, y2 = _get_xy_pair(r, "pass_end_location")
            return np.isfinite(x2) and (x2 >= 102)

        def _is_box_entry_carry(r):
            if not is_carry.loc[r.name]:
                return False
            x2, y2 = _get_xy_pair(r, "carry.end_location") if "carry.end_location" in df.columns else (np.nan, np.nan)
            return np.isfinite(x2) and (x2 >= 102)

        box_entry_mask = df.apply(_is_box_entry_pass, axis=1) | df.apply(_is_box_entry_carry, axis=1)
    else:
        prog_pass_mask   = pd.Series(False, index=df.index)
        carries_prog_mask= pd.Series(False, index=df.index)
        box_entry_mask   = pd.Series(False, index=df.index)

    # ---- Dribbles & duels outcomes (aprox ganados/perdidos) ----
    c_dribble_outcome = _ah_col(df, ["dribble.outcome.name","dribble.outcome"], None)
    dribble_success = is_dribble & (df[c_dribble_outcome].astype(str).str.contains("Complete|Completed|Success", case=False, na=False) if c_dribble_outcome else False)
    dribble_failed  = is_dribble & (df[c_dribble_outcome].astype(str).str.contains("Incomplete|Lost|Fail", case=False, na=False) if c_dribble_outcome else False)

    c_duel_outcome = _ah_col(df, ["duel.outcome.name","duel.outcome"], None)
    duels_won = is_duel & (df[c_duel_outcome].astype(str).str.contains("Won", case=False, na=False) if c_duel_outcome else False)

    # ---- Agregación por jugador ----
    grp = df.groupby(c_player_id, dropna=True)

    def _per90(count_series: pd.Series, minutes_map: pd.Series) -> pd.Series:
        cnt = count_series.astype(float)
        mins = minutes_map.reindex(cnt.index).fillna(0.0)
        per90 = np.where(mins > 0, cnt * (90.0 / mins), 0.0)
        return pd.Series(per90, index=cnt.index)

    feats = pd.DataFrame(index=grp.size().index)  # index = player_id

    # Ofensivos
    feats["xg"]                = grp["_xg"].sum()
    feats["shots"]             = grp.apply(lambda g: is_shot.loc[g.index].sum())
    feats["key_passes"]        = grp.apply(lambda g: key_pass_mask.loc[g.index].sum())
    feats["pass_attempts"]     = grp.apply(lambda g: is_pass.loc[g.index].sum())
    feats["pass_completed"]    = grp.apply(lambda g: pass_complete_mask.loc[g.index].sum())
    feats["prog_passes"]       = grp.apply(lambda g: prog_pass_mask.loc[g.index].sum())
    feats["carries"]           = grp.apply(lambda g: is_carry.loc[g.index].sum())
    feats["carries_prog"]      = grp.apply(lambda g: carries_prog_mask.loc[g.index].sum())
    feats["box_entries"]       = grp.apply(lambda g: box_entry_mask.loc[g.index].sum())

    # Defensivos
    feats["pressures"]         = grp.apply(lambda g: is_pressure.loc[g.index].sum())
    feats["recoveries"]        = grp.apply(lambda g: is_recovery.loc[g.index].sum())
    feats["tib"]               = grp.apply(lambda g: (is_tackle|is_interc|is_block).loc[g.index].sum())  # tackles+intec+blocks
    feats["clearances"]        = grp.apply(lambda g: is_clear.loc[g.index].sum())
    feats["duels_won"]         = grp.apply(lambda g: duels_won.loc[g.index].sum())

    # Pérdidas de posesión
    pass_incomplete = feats["pass_attempts"] - feats["pass_completed"]
    feats["turnovers_raw"]     = grp.apply(lambda g: (is_misctrl|is_disposs|dribble_failed).loc[g.index].sum()) + pass_incomplete

    # per90
    feats_per90 = feats.copy()
    minutes_map = m.reindex(feats.index).fillna(0.0)
    for col in feats.columns:
        feats_per90[col] = np.where(minutes_map > 0, feats[col] * (90.0 / minutes_map), 0.0)

    # tasas
    feats_per90["pass_completion"] = np.where(feats["pass_attempts"] > 0, feats["pass_completed"] / feats["pass_attempts"], np.nan)

    feats_per90 = feats_per90.reset_index().rename(columns={"index":"player_id"})
    feats_per90["player_id"] = feats_per90["player_id"].astype(feats_per90["player_id"].dtype)
    return feats_per90

def ah_efficiency_composite_v1(features_per90: pd.DataFrame, positions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula 'efficiency' por jugador con pesos por grupo posicional:
    - FWD: xg, box_entries, key_passes, carries_prog, prog_passes  penaliza turnovers
    - MID: prog_passes, carries_prog, key_passes, recoveries, pressures, tib  penaliza turnovers
    - DEF: tib, clearances, duels_won, recoveries  + pequeño crédito a prog_passes  penaliza turnovers
    - GK : recoveries (sweeper), pass_completion, clearances  penaliza turnovers
    """
    f = features_per90.copy()
    pos = positions_df[["player_id","position_group"]].copy()
    df = pos.merge(f, on="player_id", how="left")

    # Default NaN→0 para métricas de conteo per90; tasas mantenemos NaN donde aplique
    count_cols = [c for c in df.columns if c not in ["player_id","position_group","pass_completion"]]
    df[count_cols] = df[count_cols].fillna(0.0)

    # Ponderaciones (ajústalas si lo deseas)
    # Nota: 'turnovers_raw' está ya per90.
    eff = np.zeros(len(df), dtype=float)

    is_fwd = df["position_group"].eq("FWD")
    eff[is_fwd] = (
        1.20 * df.loc[is_fwd, "xg"] +
        0.70 * df.loc[is_fwd, "box_entries"] +
        0.60 * df.loc[is_fwd, "key_passes"] +
        0.45 * df.loc[is_fwd, "carries_prog"] +
        0.40 * df.loc[is_fwd, "prog_passes"] -
        0.60 * df.loc[is_fwd, "turnovers_raw"]
    )

    is_mid = df["position_group"].eq("MID")
    eff[is_mid] = (
        0.75 * df.loc[is_mid, "prog_passes"] +
        0.65 * df.loc[is_mid, "carries_prog"] +
        0.60 * df.loc[is_mid, "key_passes"] +
        0.45 * df.loc[is_mid, "recoveries"] +
        0.35 * df.loc[is_mid, "pressures"] +
        0.50 * df.loc[is_mid, "tib"] -
        0.45 * df.loc[is_mid, "turnovers_raw"]
    )

    is_def = df["position_group"].eq("DEF")
    eff[is_def] = (
        0.90 * df.loc[is_def, "tib"] +
        0.70 * df.loc[is_def, "clearances"] +
        0.60 * df.loc[is_def, "duels_won"] +
        0.40 * df.loc[is_def, "recoveries"] +
        0.20 * df.loc[is_def, "prog_passes"] -
        0.35 * df.loc[is_def, "turnovers_raw"]
    )

    is_gk = df["position_group"].eq("GK")
    # Para GK usamos recovery (sweeper), completion y clearances como aproximación
    pc = df.loc[is_gk, "pass_completion"].fillna(0.85)  # si no está, asume 85%
    eff[is_gk] = (
        0.50 * df.loc[is_gk, "recoveries"] +
        0.30 * df.loc[is_gk, "clearances"] +
        0.80 * (pc - 0.75)  # normaliza completion en torno a 75%
        -
        0.20 * df.loc[is_gk, "turnovers_raw"]
    )

    out = df[["player_id"]].copy()
    out["efficiency"] = eff
    return out
