#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive 3D vector explorer using Oracle DB + matplotlib.

- Builds a PCA (3D) projection view over a random subset of vectors.
- Optionally (re)creates a KMeans clustering model and maps cluster IDs to colors.
- Plots points; clicking a point shows the chunk text + nearest neighbors.
- A radio control switches the distance metric (COSINE/EUCLIDEAN) for nearest-neighbor lookup.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import oracledb
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import RadioButtons
import argparse


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("vector_explorer")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Your application')
        parser.add_argument('--dsn', default=os.environ.get("DSN", "localhost:1521/FREEPDB1"), help='DSN')
        parser.add_argument('--user', default=os.environ.get("USERID", "vector"), help='User ID')
        parser.add_argument('--password', default=os.environ.get("PASSWORD", "vector"), help='Password')
        parser.add_argument('--table', default=os.environ.get("TABLE", "LANGCHAIN_VECTOR_STORE"), help='Table name')
        parser.add_argument('--distance-metric-default', default="COSINE", help='Distance metric default')
        parser.add_argument('--topk', type=int, default=int(os.environ.get("TOPK", "4")), help='Top K')
        parser.add_argument('--subset-dim', type=int, default=int(os.environ.get("SUBSET_DIM", "100")), help='Subset table')
        parser.add_argument('--subset-dim-plot', type=int, default=int(os.environ.get("SUBSET_DIM_PLOT", "50")), help='Subset plot table')
       
        args = parser.parse_args()

        self.dsn = args.dsn
        self.user = args.user
        self.password = args.password
        self.table = args.table
        self.distance_metric_default = args.distance_metric_default
        self.topk = args.topk + 1  # +1 to exclude the query point
        self.pca_fit_rows = args.subset_dim #subset size to be used to create model
        self.plot_rows = args.subset_dim_plot #how many vectors to be plotted (less than "subset_dim" ). Around 100 to avoid python crash
        self.lib_dir: str = os.environ.get("LIB_DIR", os.path.expanduser("~/instantclient_23_3"))
        self.seed = "42" #to plot/sample always the same random distributed sequence to compare analysis
        
CFG = Config()

# One-time Oracle client init (safe to call multiple times; we guard anyway).
_oracle_client_inited = False
def _init_oracle():
    global _oracle_client_inited
    if not _oracle_client_inited:
        oracledb.init_oracle_client(lib_dir=CFG.lib_dir)
        _oracle_client_inited = True

# -----------------------------------------------------------------------------
# SQL helpers
# -----------------------------------------------------------------------------
def sql_drop_if_exists(table_or_view: str, kind: str = "TABLE") -> str:
    """
    Returns an anonymous block to drop a TABLE/VIEW without error if missing.
    """
    if kind not in {"TABLE", "VIEW"}:
        raise ValueError("kind must be 'TABLE' or 'VIEW'")
    code = f"""
    DECLARE
      v_stmt VARCHAR2(32767);
    BEGIN
      v_stmt := 'DROP {kind} {table_or_view} CASCADE CONSTRAINTS';
      EXECUTE IMMEDIATE v_stmt;
    EXCEPTION
      WHEN OTHERS THEN
        IF SQLCODE != -942 THEN -- ORA-00942: table or view does not exist
          RAISE;
        END IF;
    END;
    """
    return code

def sql_drop_model_if_exists(model_name: str) -> str:
    return f"""
    BEGIN
        DBMS_DATA_MINING.DROP_MODEL('{model_name}');
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLCODE != -40284 THEN -- ORA-04043: object does not exist
                RAISE;
            END IF;
    END;
    """

# -----------------------------------------------------------------------------
# DB operations
# -----------------------------------------------------------------------------
def create_pca_view(base_table: str) -> str:
    """
    Create a random sample table, train a PCA (SVD) model, and expose a 3D view.
    Returns the name of the 3D view.
    """
    _init_oracle()
    table_vect = f"{base_table}_VECT"
    view_reduced = f"{base_table}_3D"
    pca_model = f"PCA_MODEL_1_{base_table}"

    query_create_sample = f"""
        CREATE TABLE {table_vect} AS
        SELECT 
            RAWTOHEX(ID) AS ID,
            EMBEDDING,
            TEXT
        FROM (
            SELECT *
            FROM {base_table}
            ORDER BY STANDARD_HASH(ID || '{CFG.seed}', 'SHA1')
        )
        FETCH FIRST {CFG.pca_fit_rows} ROWS ONLY
    """

    query_create_model = f"""
    DECLARE
        v_setlst DBMS_DATA_MINING.SETTING_LIST;
    BEGIN
        v_setlst('ALGO_NAME')         := 'ALGO_SINGULAR_VALUE_DECOMP';
        v_setlst('SVDS_SCORING_MODE') := 'SVDS_SCORING_PCA';
        v_setlst('FEAT_NUM_FEATURES') := '3';

        DBMS_DATA_MINING.CREATE_MODEL2(
            MODEL_NAME          => '{pca_model}',
            MINING_FUNCTION     => 'FEATURE_EXTRACTION',
            DATA_QUERY          => 'SELECT EMBEDDING FROM {table_vect}',
            CASE_ID_COLUMN_NAME => NULL,
            SET_LIST            => v_setlst
        );
    END;
    """

    query_create_view = f"""
        CREATE OR REPLACE VIEW {view_reduced} AS
        SELECT
            ID,
            VECTOR_EMBEDDING({pca_model} USING *) AS EMBEDDING3D,
            TEXT,
            EMBEDDING
        FROM {table_vect}
    """

    with oracledb.connect(dsn=CFG.dsn, user=CFG.user, password=CFG.password) as conn:
        with conn.cursor() as cur:
            logger.info("Preparing 3D PCA view...")
            cur.execute(sql_drop_if_exists(table_vect, "TABLE"))
            conn.commit()

            cur.execute(query_create_sample)
            conn.commit()
            logger.info("Sample table created: %s", table_vect)

            cur.execute(sql_drop_model_if_exists(pca_model))
            conn.commit()

            cur.execute(query_create_model)
            conn.commit()
            logger.info("PCA model created: %s", pca_model)

            cur.execute(query_create_view)
            conn.commit()
            logger.info("3D view created: %s", view_reduced)

    return view_reduced

def get_cluster_map(table: str, model: str = "KM_SH_CLUS1") -> Dict[str, int]:
    """
    Returns mapping: ID -> cluster_id from an existing clustering model.
    """
    _init_oracle()
    query = f"""
        SELECT CLUSTER_ID({model} USING *) AS CLUS, ID
        FROM {table}
        ORDER BY CLUS DESC
    """
    cluster: Dict[str, int] = {}
    with oracledb.connect(dsn=CFG.dsn, user=CFG.user, password=CFG.password) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for clus, id_ in cur:
                cluster[str(id_)] = int(clus)
    return cluster

def get_random_vectors(view_name: str, n: int) -> List[Dict[str, object]]:
    """
    Returns a list of n dict rows from the given view, sampled randomly.
    Each row contains ID, EMBEDDING3D (np.array), TEXT, EMBEDDING (np.array).
    """
    _init_oracle()
    sampled_view = f"SAMPLED_{view_name}"

    create_sampled_view = f"""
        CREATE TABLE {sampled_view} AS
        SELECT *
        FROM (
            SELECT *
            FROM {view_name}
            ORDER BY STANDARD_HASH(ID || '{CFG.seed}', 'SHA1')
        )
        FETCH FIRST {n} ROWS ONLY
        """

    with oracledb.connect(dsn=CFG.dsn, user=CFG.user, password=CFG.password) as conn:
        with conn.cursor() as cur:
            cur.execute(sql_drop_if_exists(f"SAMPLED_{view_name}", "TABLE"))
            conn.commit()

            cur.execute(create_sampled_view)
            conn.commit()
            cur.execute(f"SELECT * FROM {sampled_view}")

            cols = [d[0] for d in cur.description]
            rows = []
            for rec in cur:
                row = dict(zip(cols, rec))
                # Convert Oracle vector types to numpy arrays
                row["EMBEDDING3D"] = np.array(row["EMBEDDING3D"])
                row["EMBEDDING"] = np.array(row["EMBEDDING"])
                # LOB -> string
                txt = row.get("TEXT", "")
                if isinstance(txt, oracledb.LOB):
                    row["TEXT"] = txt.read()
                rows.append(row)
            return rows

def nearest_ids(table_vect: str, origin_id: str, metric: str, n: int) -> List[str]:
    """
    Returns the n nearest IDs (including the query itself) by metric from table_vect.
    """
    _init_oracle()
    metric = metric.upper()
    query = f"""
        SELECT ID
        FROM {table_vect}
        ORDER BY VECTOR_DISTANCE(
            EMBEDDING,
            (SELECT EMBEDDING FROM {table_vect} WHERE ID = :1),
            {metric}
        )
        FETCH EXACT FIRST {n} ROWS ONLY
    """
    with oracledb.connect(dsn=CFG.dsn, user=CFG.user, password=CFG.password) as conn:
        with conn.cursor() as cur:
            cur.execute(query, [origin_id])
            return [str(r[0]) for r in cur]

# -----------------------------------------------------------------------------
# Plot utilities
# -----------------------------------------------------------------------------
def limit_lines(s: str, max_lines: int) -> str:
    return "\n".join(s.splitlines()[:max_lines])

def project_3d_to_2d(ax: Axes, x: float, y: float, z: float) -> Tuple[float, float]:
    """
    Projects 3D (x,y,z) into 2D display coords for hit-testing.
    """
    world = np.array([x, y, z, 1.0])
    ndc = ax.get_proj() @ world
    ndc /= ndc[3]
    x_scr, y_scr = ax.transData.transform((ndc[0], ndc[1]))
    return float(x_scr), float(y_scr)

# -----------------------------------------------------------------------------
# Interactive app
# -----------------------------------------------------------------------------
class VectorExplorer:
    def __init__(self, base_table: str, subset: int, plot_subset:int, fig: Figure, ax: Axes):
        self.base_table = base_table
        self.table_vect = f"{base_table}_VECT"

        #Create a subset of original vectors, the PCA model, apply on a new view
        self.view_reduced = create_pca_view(base_table)

        # Load cluster map for coloring
        cluster_map = get_cluster_map(self.view_reduced, "KM_SH_CLUS1")
        unique_clusters = sorted(set(cluster_map.values()))
        palette = ["blue", "green", "brown", "yellow", "purple", "cyan", "magenta", "orange", "grey"]
        self.color_by_cluster = {
            c: palette[i % len(palette)] for i, c in enumerate(unique_clusters)
        }

        # Sample points for plottinf
        rows = get_random_vectors(self.view_reduced, plot_subset)
        self.points: List[Tuple[float, float, float]] = []
        self.id_for_point: Dict[Tuple[float, float, float], str] = {}
        self.text_for_point: Dict[Tuple[float, float, float], str] = {}
        self.colors: List[str] = []

        for r in rows:
            p = (float(r["EMBEDDING3D"][0]), float(r["EMBEDDING3D"][1]), float(r["EMBEDDING3D"][2]))
            self.points.append(p)
            id_str = str(r["ID"])
            self.id_for_point[p] = id_str
            self.text_for_point[p] = str(r.get("TEXT", ""))
            clus = cluster_map.get(id_str)
            self.colors.append(self.color_by_cluster.get(clus, "black"))

        # Matplotlib artifacts
        self.fig = fig
        self.ax = ax
        self.scatter = self.ax.scatter(
            [p[0] for p in self.points],
            [p[1] for p in self.points],
            [p[2] for p in self.points],
            marker="o",
            c=self.colors,
        )
        self.highlight = self.ax.scatter([], [], [], color="red", s=70)
        self.annotations: List[plt.Text] = []

        self.distance_metric = CFG.distance_metric_default.upper()
        self._setup_axes()
        self._setup_controls()

        # Connect events
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    # ---- setup ----
    def _setup_axes(self) -> None:
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

    def _setup_controls(self) -> None:
        radio_ax = plt.axes([0.80, 0.70, 0.15, 0.12])
        self.radio = RadioButtons(radio_ax, ["EUCLIDEAN", "COSINE"],
                                  active=1 if self.distance_metric == "COSINE" else 0)
        self.radio.on_clicked(self.on_metric_change)

    # ---- interactions ----
    def on_metric_change(self, label: str) -> None:
        self.distance_metric = label.upper()
        logger.info("Distance metric -> %s", self.distance_metric)

    def on_click(self, event) -> None:
        if event.inaxes is None:
            return

        x_click, y_click = event.x, event.y
        if x_click is None or y_click is None:
            return

        # Find nearest plotted point in screen coords
        closest = None
        min_d = float("inf")
        for (x, y, z) in self.points:
            sx, sy = project_3d_to_2d(self.ax, x, y, z)
            d = (sx - x_click) ** 2 + (sy - y_click) ** 2
            if d < min_d:
                min_d = d
                closest = (x, y, z)

        if closest is None:
            return

        # Update annotation for the selected point
        self._clear_annotations()
        self._annotate_point(closest)

        # Selected chunk (UI shows 8 lines, logs show FULL text)
        self.fig.texts.clear()
        sel_id = self.id_for_point[closest]
        sel_text = self.text_for_point[closest]
        logger.info("Selected CHUNK ID: %s\n----- FULL TEXT BEGIN -----\n%s\n----- FULL TEXT END -----", sel_id, sel_text)

        header = f"CHUNK: {sel_id}\n" + "-" * 38 + "\n"
        self.fig.text(0.02, 0.98, header + limit_lines(sel_text, 8),
                    fontsize=8, color="black", ha="left", va="top")

        # Nearest neighbors (exclude the selected itself)
        nn = nearest_ids(f"SAMPLED_{self.view_reduced}", sel_id, self.distance_metric, CFG.topk)
        nn = [n for n in nn if n != sel_id]

        n_x, n_y, n_z = [], [], []
        y_base, step = 0.70, 0.10
        for i, nid in enumerate(nn):
            p = self._point_by_id(nid)
            if p is None:
                continue

            # FULL neighbor text in logs
            full_neighbor_text = self.text_for_point.get(p, "")
            logger.info("Neighbor #%d CHUNK ID: %s\n----- FULL TEXT BEGIN -----\n%s\n----- FULL TEXT END -----",
                        i, nid, full_neighbor_text)

            n_x.append(p[0]); n_y.append(p[1]); n_z.append(p[2])
            snippet = limit_lines(full_neighbor_text, 4)
            self.fig.text(0.02, y_base - i * step, f"#{i}. CHUNK: {nid}\n{snippet}",
                        fontsize=8, color="green", ha="left", va="top")

        self._update_highlight(n_x, n_y, n_z)
        self.fig.canvas.draw_idle()

    # ---- helpers ----
    def _clear_annotations(self) -> None:
        while self.annotations:
            t = self.annotations.pop()
            t.remove()

    def _annotate_point(self, p: Tuple[float, float, float]) -> None:
        t = self.ax.text(
            p[0], p[1], p[2],
            f"({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})",
            color="red", fontsize=10, bbox=dict(facecolor="white", alpha=0.6),
        )
        self.annotations.append(t)

    def _update_highlight(self, xs: Sequence[float], ys: Sequence[float], zs: Sequence[float]) -> None:
        self.highlight.remove()
        self.highlight = self.ax.scatter(xs, ys, zs, color="red", s=50)

    def _point_by_id(self, id_: str) -> Tuple[float, float, float] | None:
        # Reverse lookup (build a tiny index once for convenience)
        # In practice, points are small; for very large sets, keep a dict id->point.
        for p, pid in self.id_for_point.items():
            if pid == id_:
                return p
        return None

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    np.random.seed(19680801)  # reproducible colors/markers if any randomness creeps in

    fig = plt.figure(figsize=(12, 9))
    try:
        # Not all backends/managers support resize; guard it
        manager = plt.get_current_fig_manager()
        manager.set_window_title("Vector 3D Explorer")  # Set the window title
        manager.resize(int(CFG.screen_width * 0.8), int(CFG.screen_height * 0.8))
    except Exception:
        pass

    ax = fig.add_subplot(projection="3d")

    logger.info("Building explorer for table: %s", CFG.table)
    explorer = VectorExplorer(CFG.table, CFG.pca_fit_rows, CFG.plot_rows, fig, ax)

    plt.show()

if __name__ == "__main__":
    main()
