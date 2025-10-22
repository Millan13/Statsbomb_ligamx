"""
analytics_helpers.py — funciones de apoyo para el dashboard ISAC Scouting.

Diseñado para trabajar con eventos de StatsBomb (Events v8) ya integrados en un
DataFrame (CSV o Parquet). Incluye utilidades robustas que usan "fallbacks" de
columnas para adaptarse a variaciones frecuentes del feed.

Autor: Miguel Millán (scaffold por ChatGPT)
"""
from __future__ import annotations
import ast
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go



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
        G = nx.DiGraph()
        for _, r in passes.iterrows():
            p_from, p_to = r.get("player"), r.get("pass_recipient")
            if isinstance(p_from, str) and isinstance(p_to, str):
                G.add_edge(p_from, p_to)
        if len(G.nodes) > 0:
            pr = nx.pagerank(G, alpha=0.85)
            top_p = max(pr.items(), key=lambda x: x[1])
            out["most_central_player"] = top_p[0]
            out["pagerank_score"] = round(top_p[1], 4)

    return out




def build_pass_network(
    df_t: pd.DataFrame,
    team_name: str = "Club América",
    min_passes: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], str]:
    """
    Construye la red de pases del equipo:
      - nodes_df: player, pr (pagerank), indeg, outdeg, total, size
      - edges_df: source, target, weight (número de pases)
      - pagerank: dict {player: score}
      - top_player: jugador con mayor pagerank
    Filtra por type=='Pass' y por el equipo.
    """
    # detectar columna de equipo
    team_col = None
    for c in ["team", "team_name", "possession_team_name"]:
        if c in df_t.columns:
            team_col = c
            break

    # filtrar pases del equipo
    passes = df_t[df_t.get("type") == "Pass"].copy()
    if team_col:
        passes = passes[passes[team_col] == team_name]

    # necesitamos player (pasador) y pass_recipient (receptor)
    if passes.empty or "player" not in passes.columns or "pass_recipient" not in passes.columns:
        return (pd.DataFrame(columns=["player","pr","indeg","outdeg","total","size"]),
                pd.DataFrame(columns=["source","target","weight"]),
                {}, "-")

    # aristas: player -> pass_recipient con peso = conteo
    edges_df = (
        passes.dropna(subset=["player","pass_recipient"])
              .groupby(["player","pass_recipient"])
              .size()
              .reset_index(name="weight")
    )

    # filtra conexiones muy poco frecuentes para limpiar el grafo
    edges_df = edges_df[edges_df["weight"] >= max(1, int(min_passes))]

    # construir grafo dirigido
    G = nx.DiGraph()
    for _, r in edges_df.iterrows():
        G.add_edge(r["player"], r["pass_recipient"], weight=int(r["weight"]))

    if len(G) == 0:
        return (pd.DataFrame(columns=["player","pr","indeg","outdeg","total","size"]),
                edges_df, {}, "-")

    # PageRank ponderado por peso de pase
    pr = nx.pagerank(G, alpha=0.85, weight="weight")
    top_player = max(pr.items(), key=lambda x: x[1])[0] if pr else "-"

    # métricas de nodos
    indeg  = dict(G.in_degree(weight="weight"))
    outdeg = dict(G.out_degree(weight="weight"))
    total  = {n: indeg.get(n,0)+outdeg.get(n,0) for n in G.nodes}

    nodes_df = pd.DataFrame({
        "player": list(G.nodes),
        "pr": [pr.get(n,0.0) for n in G.nodes],
        "indeg": [indeg.get(n,0) for n in G.nodes],
        "outdeg": [outdeg.get(n,0) for n in G.nodes],
        "total": [total.get(n,0) for n in G.nodes],
    }).sort_values("pr", ascending=False)

    # tamaño para marker (normalizado)
    if nodes_df["pr"].max() > 0:
        nodes_df["size"] = 10 + 40 * (nodes_df["pr"] - nodes_df["pr"].min()) / (nodes_df["pr"].max() - nodes_df["pr"].min() + 1e-9)
    else:
        nodes_df["size"] = 15

    return nodes_df.reset_index(drop=True), edges_df.reset_index(drop=True), pr, top_player


def plot_pass_network(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    highlight: str = "",
    title: str = "Red de pases (PageRank)",
    seed: int = 7,
) -> go.Figure:
    """
    Dibuja la red de pases con Plotly:
      - nodos escalados por PageRank (size)
      - aristas con grosor por weight
      - resalta 'highlight' (jugador más influyente)
    """
    G = nx.DiGraph()
    for _, r in edges_df.iterrows():
        G.add_edge(r["player"], r["pass_recipient"], weight=int(r["weight"]))
    for _, r in nodes_df.iterrows():
        if r["player"] not in G:
            G.add_node(r["player"])

    if len(G) == 0:
        return go.Figure()

    pos = nx.spring_layout(G, seed=seed, weight="weight", k=1.2)  # layout estable

    # aristas (como segmentos)
    edge_x, edge_y, edge_w = [], [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_w.append(data.get("weight", 1))

    # normaliza ancho de arista
    if edge_w:
        w = np.array(edge_w, dtype=float)
        w = 1 + 4*(w - w.min())/(w.max()-w.min()+1e-9)
    else:
        w = np.array([])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="rgba(120,120,120,0.5)"),
        hoverinfo="none",
        showlegend=False,
    )

    # nodos
    xs, ys, sizes, texts, colors = [], [], [], [], []
    for _, r in nodes_df.iterrows():
        p = r["player"]
        x, y = pos[p]
        xs.append(x); ys.append(y)
        sizes.append(r["size"])
        pr = r["pr"]
        txt = f"{p}<br>PR={pr:.4f}<br>Total pases (in+out)={int(r['total'])}"
        texts.append(txt)
        colors.append("#F9B233" if p == highlight else "#1f77b4")

    node_trace = go.Scatter(
        x=xs, y=ys, mode="markers+text",
        text=[p.split()[0] for p in nodes_df["player"]],  # etiqueta corta
        textposition="bottom center",
        hovertext=texts, hoverinfo="text",
        marker=dict(size=sizes, line=dict(width=1, color="#fff"), opacity=0.95, color=colors),
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="white",
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
