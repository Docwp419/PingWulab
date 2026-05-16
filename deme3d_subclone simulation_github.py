# -*- coding: utf-8 -*-
import logging
import math
import os
import random
from collections import defaultdict, deque

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
import hashlib  # Offset for stable color sampling

# ---------- Optional Numba acceleration ----------
try:
    from numba import njit
except Exception:
    def njit(*njit_args, **njit_kwargs):
        if len(njit_args) == 1 and callable(njit_args[0]) and not njit_kwargs:
            return njit_args[0]
        def _decorator(func):
            return func
        return _decorator

# ---------- Logging and fonts ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Keep INFO-level logs, but suppress fontTools INFO messages for PDF font subsetting.
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
# Use editable/vector-friendly fonts in PDF/PS outputs.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# ==================== Parameter settings ====================
base_dir = r"C:\subclone_simulation_outputs"
os.makedirs(base_dir, exist_ok=True)

# --- Run control ---
RUNS       = 1
SEED_BASE  = None

# --- 3D spherical lattice domain ---
initial_radius_deme = 6
neighbors = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]  # Six-neighbor lattice connectivity

# --- Deme capacity and demographic parameters ---
C_deme          = int(10**3)
m               = 5e-3
dt              = 0.5
t_max           = 250.0
gamma           = 1.0
gamma_env       = 1.0
d_rate          = 0.1
INIT_FILL_FRACT = 0.2

# --- Global environmental carrying capacity ---
K_env_total_cells = int(1e7)

# --- Genetic events ---
p_d                 = 1e-5
PASSENGER_LAMBDA   = 1.2        # Passenger mutations per division ~ Poisson(lambda)
DRIVER_LAMBDA      = 1.0        # Driver mutations per driver event ~ zero-truncated Poisson(lambda)

# --- Fitness: s = s0 + s_max*(1 - exp(-alpha*n)) ---
s_max           = 0.1
alpha_advantage = 0.15
S0_RANGE        = (0.05, 0.06)

# --- Noise-induced branching ---
NOISE_RHO_BASE  = 2e-5
BDRY_BETA       = 1.0
BDRY_WEIGHT     = 2.0
BETA_A          = 2.0
BETA_B          = 8.0
NOISE_SCALE     = 0.2
KAPPA_COOP      = 0.5
NOISE_SEED_FRAC = 0.01

# --- CIN-like mutational burst ---
CIN_ENABLED      = True
A                = 10.0
mu_b             = 1.0
CIN_R_INNER_FRAC = 0.3
CIN_R_OUTER_FRAC = 0.8
CIN_T_START      = 50.0
CIN_T_END        = 150.0

# --- Subclone seeding at the deme level ---
FOUNDER_INIT_DEMES = 10
DRIVER_SEED_DEMES  = 10
SEED_CELLS_FRAC    = 0.01

# --- Filtering thresholds and VAF ---
# Dynamic F_cells threshold: 1% of the total cell count at each snapshot time.
F_CELLS_FRAC = 0.01           # Filtering threshold fraction; default is 1%.
F_CELLS_MIN  = 1              # Minimum threshold to avoid zero values at very early stages.
F_v          = 0.01           # Gene VAF threshold (0-0.5) used for final visualization.

# --- Competition and noise switches ---
ENABLE_BIRTH_SELECTION = True
COMP_STRENGTH          = 2.0

ENABLE_SOFTMAX_CAP     = True
SOFTMAX_BETA           = 5.0

NOISE_BIRTH_SIGMA      = 0.40
NOISE_ENV_SIGMA        = 0.30

# --- Figure output and slice rendering ---
SAVE_PDF          = False
SLICE_NUM         = 9
PIXELS_PER_DEME   = 6
BACKGROUND_RGBA   = (1.0, 1.0, 1.0, 0.0)
TREE_BASE_DIAM_PT  = 40.0
TREE_SCALE_DIAM_PT = 260.0

# Clone filtering mode for Z-plane display.
ZPLANE_FILTER_MODE = "all"  # "filtered" or "all"

# Intermediate exports.
ENABLE_INTERMEDIATE_EXPORTS = True
EXPORT_TIMES = [100.0, 150.0, 200.0]
EXPORT_FINAL_ALWAYS = True

# ==================== Global naming counters ====================
global_N_counter = 1
global_D_counter = 1

# ==================== Numba-friendly numerical kernels ====================
@njit
def _fitness_from_drivers_count(n, s0, s_max, alpha):
    return s0 + s_max * (1.0 - math.exp(-alpha * n))

@njit
def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)

# ==================== Non-array utility functions ====================
def in_sphere_origin(x, y, z, radius):
    return (x*x + y*y + z*z) <= radius*radius

def deme_neighbors(p, radius):
    x,y,z = p
    out = []
    for dx,dy,dz in neighbors:
        q = (x+dx, y+dy, z+dz)
        if in_sphere_origin(q[0], q[1], q[2], radius):
            out.append(q)
    return out

def empty_neighbor_fraction_deme(p, counts, radius):
    neigh = deme_neighbors(p, radius)
    if not neigh:
        return 0.0
    empty = 0
    for q in neigh:
        n_q = sum(counts.get(q, {}).values())
        if n_q < C_deme:
            empty += 1
    return empty / float(len(neigh))

def poisson_trunc_pos(lmbda):
    for _ in range(16):
        k = int(np.random.poisson(lmbda))
        if k > 0:
            return k
    return 1

def make_new_N(k):
    global global_N_counter
    out = [f"N{global_N_counter+i}" for i in range(k)]
    global_N_counter += k
    return out

def make_new_D(k):
    global global_D_counter
    out = [f"D{global_D_counter+i}" for i in range(k)]
    global_D_counter += k
    return out

def fitness_from_drivers(s0, drivers):
    n = sum(1 for g in drivers if not g.startswith("D_o"))
    return _fitness_from_drivers_count(n, s0, s_max, alpha_advantage)

def CIN_factor(p, t, radius):
    if not CIN_ENABLED:
        return 1.0
    x,y,z = p
    r = math.sqrt(x*x + y*y + z*z)
    rin  = CIN_R_INNER_FRAC * radius
    rout = CIN_R_OUTER_FRAC * radius
    M = 1.0 if (rin <= r <= rout) else 0.0
    P = 1.0 if (CIN_T_START <= t <= CIN_T_END) else 0.0
    return 1.0 + A * M * P

def simple_hierarchy_layout(G, root='G0', x_gap=1.6, y_gap=1.8):
    levels = defaultdict(list)
    depth = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        d = depth[u]
        levels[d].append(u)
        for v in G.successors(u):
            if v not in depth:
                depth[v] = d+1
                q.append(v)
    pos = {}
    for d, nodes in levels.items():
        n = len(nodes)
        start_x = -(n-1)*x_gap/2.0
        for i, node in enumerate(nodes):
            pos[node] = (start_x + i*x_gap, -d*y_gap)
    return pos

# ==================== Stable color mapping across snapshots ====================
class StableColorManager:
    """
    Stable hash-based mapping from Clone ID to colormap(t). This ensures that:
    - The same clone ID uses the same color across different snapshots.
    - Newly emerging clone IDs also receive deterministic colors.
    """
    def __init__(self, founder_id="G0", cmap_name="turbo", founder_color="skyblue",
                 avoid_founder_dist=0.20):
        self.founder_id = str(founder_id)
        self.cmap = plt.get_cmap(cmap_name)
        self.founder_rgb = np.array(mcolors.to_rgb(founder_color), dtype=np.float64)
        self.avoid_founder_dist = float(avoid_founder_dist)
        self.color_map = {self.founder_id: tuple(self.founder_rgb.tolist())}

    @staticmethod
    def _hash01(s: str) -> float:
        # sha1 -> [0,1)
        h = hashlib.sha1(s.encode("utf-8")).hexdigest()
        v = int(h[:12], 16)  # 48-bit precision is sufficient for stable hashing.
        return (v % (10**12)) / float(10**12)

    def _sample_rgb(self, t: float) -> tuple:
        r, g, b, _ = self.cmap(float(t) % 1.0)
        return (float(r), float(g), float(b))

    def get_color(self, cid: str) -> tuple:
        cid = str(cid)
        if cid in self.color_map:
            return self.color_map[cid]
        if cid == self.founder_id:
            return self.color_map[self.founder_id]

        t = self._hash01(cid)
        rgb = np.array(self._sample_rgb(t), dtype=np.float64)

        # Avoid colors that are visually too close to the founder color.
        if np.linalg.norm(rgb - self.founder_rgb) < self.avoid_founder_dist:
            rgb = np.array(self._sample_rgb((t + 0.5) % 1.0), dtype=np.float64)

        self.color_map[cid] = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
        return self.color_map[cid]

    def ensure_ids(self, ids_iterable):
        for cid in ids_iterable:
            self.get_color(cid)

    def as_dict(self):
        return dict(self.color_map)

# ==================== Clone class ====================
class Clone:
    def __init__(self, cid, parent=None, generation=0,
                 s0=None, drivers=None, passengers=None, p_d_factor=1.0):
        self.id         = cid
        self.parent     = parent
        self.children   = []
        self.generation = generation
        self.drivers    = list(drivers or [])
        self.passengers = list(passengers or [])
        self.s0 = s0 if s0 is not None else random.uniform(*S0_RANGE)
        self.s  = fitness_from_drivers(self.s0, self.drivers)
        self.b  = max(0.0, self.s + d_rate)
        self.p_d_factor = float(p_d_factor)

# ==================== Safe figure saving to avoid tight-layout clipping ====================
def savefig_safe(fig, path, pad=0.03):
    fig.savefig(path, bbox_inches='tight', pad_inches=pad)
    plt.close(fig)

# ==================== Dynamic filtering threshold ====================
def compute_dynamic_F_cells(total_cells_snapshot, frac=F_CELLS_FRAC, min_cells=F_CELLS_MIN):
    v = int(round(float(frac) * float(max(0, total_cells_snapshot))))
    return int(max(int(min_cells), v))

# ==================== Output module: snapshot export ====================
def compute_totals_from_counts(counts_snapshot):
    totals_by_clone = defaultdict(int)
    for dct in counts_snapshot.values():
        for cid, v in dct.items():
            totals_by_clone[cid] += int(v)
    return totals_by_clone

def compute_filtered_ids_from_totals(totals_by_clone, root_id="G0", F_cells_dynamic=0):
    filtered = [cid for cid, cells in totals_by_clone.items() if int(cells) >= int(F_cells_dynamic)]
    if root_id in totals_by_clone and root_id not in filtered:
        filtered = [root_id] + filtered
    return filtered

def export_filtered_clone_counts(folder, t_tag, clones_dict, totals_snapshot, filtered_ids,
                                 F_cells_dynamic, root_id="G0"):
    filtered_total_cells = sum(totals_snapshot.get(cid, 0) for cid in filtered_ids) if filtered_ids else 0

    gene_display_map = {}
    driver_ctr = [1]
    passenger_ctr = [1]

    def map_gene_display(g_internal):
        if g_internal in gene_display_map:
            return gene_display_map[g_internal]
        if g_internal.startswith('D'):
            disp = f"Δ{driver_ctr[0]}"; driver_ctr[0] += 1
        else:
            disp = f"π{passenger_ctr[0]}"; passenger_ctr[0] += 1
        gene_display_map[g_internal] = disp
        return disp

    rows = []
    for cid in filtered_ids:
        cobj = clones_dict.get(cid, None)
        drivers_str = ""
        if cobj is not None:
            drivers_str = ", ".join(sorted(map(map_gene_display, cobj.drivers)))
        cells = int(totals_snapshot.get(cid, 0))
        prop = (cells/float(filtered_total_cells)) if filtered_total_cells > 0 else 0.0
        rows.append((cid, cells, prop, drivers_str))
    df_counts = pd.DataFrame(rows, columns=['Clone ID','Final Cells','Proportion','Driver Genes'])

    # Diversity indices computed on the filtered clone set.
    if filtered_ids:
        U = nx.Graph()
        U.add_nodes_from(filtered_ids)
        for cid in filtered_ids:
            if cid == root_id:
                continue
            parent = clones_dict[cid].parent if cid in clones_dict else None
            while parent and parent.id not in filtered_ids:
                parent = parent.parent
            if parent:
                U.add_edge(parent.id, cid)
            else:
                U.add_edge(root_id, cid)

        dist = dict(nx.all_pairs_shortest_path_length(U))
        p_vals = dict(zip(df_counts['Clone ID'], df_counts['Proportion']))

        phi = 0.0
        for i in filtered_ids:
            for j in filtered_ids:
                dij = dist[i][j] if (i in dist and j in dist[i]) else 0
                phi += p_vals.get(i, 0.0) * p_vals.get(j, 0.0) * dij

        shannon = -sum(p_ * math.log(p_) for p_ in p_vals.values() if p_ > 0)
        simpson = 1.0 - sum(p_*p_ for p_ in p_vals.values())
    else:
        phi = 0.0; shannon = 0.0; simpson = 0.0

    df_metrics = pd.DataFrame(
        [
            ('F_cells_dynamic', int(F_cells_dynamic), '', ''),
            ('TotalCells_snapshot', int(sum(totals_snapshot.values())), '', ''),
            ('PHI', phi, '', ''),
            ('Shannon', shannon, '', ''),
            ('Simpson', simpson, '', '')
        ],
        columns=['Clone ID','Final Cells','Proportion','Driver Genes']
    )

    df_out = pd.concat([df_counts, df_metrics], ignore_index=True)

    out_xlsx = os.path.join(folder, f"filtered_clone_counts_t{t_tag}.xlsx")
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        df_out.to_excel(writer, sheet_name='FilteredCounts', index=False)
    logger.info("[EXPORT t=%s] filtered_clone_counts -> %s (F_cells=%d)", t_tag, out_xlsx, int(F_cells_dynamic))

    return df_counts

def export_clone_growth_curves(folder, t_tag, clone_history, filtered_ids, color_map):
    fig = plt.figure(figsize=(7.2, 5.2), dpi=300)
    plotted_ids = sorted(filtered_ids)
    for cid in plotted_ids:
        dat = clone_history.get(cid)
        if not dat:
            continue
        color = color_map.get(cid, (0.6, 0.6, 0.6))
        plt.plot(dat['time'], dat['N'], lw=1.0, color=color, label=cid)
    plt.xlabel('Time'); plt.ylabel('Cells')
    plt.title(f'Filtered Clone Growth (up to t={t_tag})')

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ncols = max(2, min(8, int(math.ceil(len(labels) / 8.0)))) if labels else 2
    if labels:
        plt.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, -0.18), ncol=ncols,
                   frameon=False, fontsize=6)
        plt.subplots_adjust(bottom=0.28)

    out_pdf = os.path.join(folder, f"clone_growth_curves_filtered_t{t_tag}.pdf")
    savefig_safe(fig, out_pdf, pad=0.05)
    logger.info("[EXPORT t=%s] clone growth curves -> %s", t_tag, out_pdf)

def export_phylogenetic_tree(folder, t_tag, clones_dict, df_counts, filtered_ids, color_map, root_id="G0"):
    if not filtered_ids:
        return

    G = nx.DiGraph()
    for cid in filtered_ids:
        G.add_node(cid)

    for cid in filtered_ids:
        if cid == root_id:
            continue
        parent = clones_dict[cid].parent if cid in clones_dict else None
        while parent and parent.id not in filtered_ids:
            parent = parent.parent
        if parent:
            G.add_edge(parent.id, cid)
        else:
            G.add_edge(root_id, cid)

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB')
    except Exception:
        try:
            pos = nx.nx_pydot.graphviz_layout(G, prog='dot')
        except Exception:
            pos = simple_hierarchy_layout(G, root=root_id)

    props = dict(zip(df_counts['Clone ID'], df_counts['Proportion']))
    node_list = list(G.nodes())
    diam_pt = {n: (TREE_BASE_DIAM_PT + TREE_SCALE_DIAM_PT * max(0.0, props.get(n,0.0))) for n in node_list}
    node_sizes = [ (diam_pt[n] ** 2) for n in node_list ]

    fig = plt.figure(figsize=(10, 7), dpi=300)
    nx.draw(G, pos,
            with_labels=True,
            labels={n: n for n in node_list},
            node_size=node_sizes,
            node_color=[color_map.get(n, (0.6,0.6,0.6)) for n in node_list],
            font_size=7, arrows=True)
    plt.title(f'Phylogenetic Tree (Filtered clones; t={t_tag})')
    plt.axis('off')

    out_pdf = os.path.join(folder, f"phylogenetic_tree_filtered_t{t_tag}.pdf")
    savefig_safe(fig, out_pdf, pad=0.05)
    logger.info("[EXPORT t=%s] phylogenetic tree -> %s", t_tag, out_pdf)

def render_z_planes_filtered(counts_snapshot, filtered_ids, color_map, out_prefix,
                             title_suffix="", mode="filtered"):
    if not counts_snapshot:
        return
    if mode not in ("filtered", "all"):
        mode = "filtered"

    filtered_set = set(filtered_ids)

    zs = [z for (x, y, z) in counts_snapshot.keys()]
    zmin, zmax = min(zs), max(zs)

    planes_raw = np.linspace(zmin, zmax, SLICE_NUM, dtype=int)
    planes = []
    seen = set()
    for z0 in planes_raw.tolist():
        if int(z0) not in seen:
            planes.append(int(z0))
            seen.add(int(z0))
    if len(planes) < SLICE_NUM:
        for z0 in range(int(zmin), int(zmax) + 1):
            if z0 not in seen:
                planes.append(int(z0))
                seen.add(int(z0))
            if len(planes) >= SLICE_NUM:
                break
    planes = planes[:SLICE_NUM]

    bg = BACKGROUND_RGBA

    for i, z0 in enumerate(planes, start=1):
        layer_sites = [(x, y) for (x, y, z) in counts_snapshot.keys() if z == z0]
        if not layer_sites:
            continue

        xs = [p[0] for p in layer_sites]
        ys = [p[1] for p in layer_sites]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        NX = x_max - x_min + 1
        NY = y_max - y_min + 1
        K  = PIXELS_PER_DEME

        img = np.ones((NY * K, NX * K, 4), dtype=np.float32)
        img[..., 0] = bg[0]; img[..., 1] = bg[1]; img[..., 2] = bg[2]; img[..., 3] = bg[3]

        for (x, y) in layer_sites:
            dct_all = counts_snapshot.get((x, y, z0), {})
            if not dct_all:
                continue

            if mode == "filtered":
                dct = {cid: v for cid, v in dct_all.items() if (cid in filtered_set and v > 0)}
            else:
                dct = {cid: v for cid, v in dct_all.items() if (v > 0)}

            tot = sum(dct.values())
            if tot <= 0:
                continue

            col0 = (x - x_min) * K
            row0 = (y - y_min) * K

            clone_ids = list(dct.keys())
            probs = np.array([dct[cid] / float(tot) for cid in clone_ids], dtype=np.float64)
            probs /= probs.sum()

            idx_flat = np.random.choice(len(clone_ids), size=K*K, p=probs)
            tile = np.zeros((K*K, 4), dtype=np.float32)
            for j, ci in enumerate(idx_flat):
                rgb = color_map.get(clone_ids[ci], (0.7, 0.7, 0.7))
                tile[j, 0] = rgb[0]
                tile[j, 1] = rgb[1]
                tile[j, 2] = rgb[2]
                tile[j, 3] = 1.0
            tile = tile.reshape((K, K, 4))
            img[row0:row0+K, col0:col0+K, :] = tile

        fig = plt.figure(dpi=300)
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        extent = [x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5]
        plt.imshow(img[::-1, :, :], origin='lower', interpolation='nearest', extent=extent)
        plt.axis('off')

        mode_tag = "filtered" if mode == "filtered" else "all"
        plt.title(f'Z-plane {i}/{len(planes)} (z={z0}) | {mode_tag}{title_suffix}')

        savefig_safe(fig, f"{out_prefix}_{i:02d}.pdf", pad=0.03)

def export_snapshot_outputs(folder, t_target, counts_snapshot, clone_history, clones_dict,
                            color_manager: StableColorManager, root_id="G0"):
    t_tag = str(int(round(t_target))) if abs(t_target - round(t_target)) < 1e-6 else f"{t_target:g}"

    totals_snapshot = compute_totals_from_counts(counts_snapshot)
    N_total_snapshot = int(sum(totals_snapshot.values()))
    F_cells_dynamic = compute_dynamic_F_cells(N_total_snapshot, frac=F_CELLS_FRAC, min_cells=F_CELLS_MIN)

    filtered_ids = compute_filtered_ids_from_totals(
        totals_snapshot, root_id=root_id, F_cells_dynamic=F_cells_dynamic
    )

    # Register all filtered clone IDs and all clone IDs currently present in counts into the global color map.
    ids_for_colors = set()
    for dct in counts_snapshot.values():
        ids_for_colors.update(dct.keys())
    ids_for_colors.update(filtered_ids)
    color_manager.ensure_ids(ids_for_colors)
    color_map = color_manager.as_dict()

    # 1) filtered_clone_counts
    df_counts = export_filtered_clone_counts(
        folder, t_tag, clones_dict, totals_snapshot, filtered_ids,
        F_cells_dynamic=F_cells_dynamic, root_id=root_id
    )

    # 2) Growth curves for filtered clones.
    export_clone_growth_curves(folder, t_tag, clone_history, filtered_ids, color_map)

    # 3) Phylogenetic tree for filtered clones.
    export_phylogenetic_tree(folder, t_tag, clones_dict, df_counts, filtered_ids, color_map, root_id=root_id)

    # 4) Z-plane slices.
    z_prefix = os.path.join(folder, f"z_plane_t{t_tag}")
    render_z_planes_filtered(
        counts_snapshot, filtered_ids, color_map, z_prefix,
        title_suffix=f" | t={t_target:g} | F_cells={F_cells_dynamic} (={F_CELLS_FRAC*100:.1f}% of N={N_total_snapshot})",
        mode=ZPLANE_FILTER_MODE
    )

    logger.info(
        "[EXPORT t=%s] Snapshot total cells=%d, dynamic F_cells=%d (%.1f%%)",
        t_tag, N_total_snapshot, F_cells_dynamic, F_CELLS_FRAC*100.0
    )

# ==================== Core simulation for one run ====================
def simulate_once(run_idx: int):
    global global_N_counter, global_D_counter
    global_N_counter = 1
    global_D_counter = 1

    folder_root = os.path.join(base_dir, "deme3d_run_005")
    folder = os.path.join(folder_root, f"run_{run_idx+1:03d}")
    os.makedirs(folder, exist_ok=True)
    logger.info("[Run %03d] Output directory: %s", run_idx+1, folder)

    counts = {}
    radius = initial_radius_deme

    # --- Founder clone ---
    s0_founder = random.uniform(*S0_RANGE)
    founder_drivers    = ["D_o1", "D_o2"]
    founder_passengers = [f"N_o{i}" for i in range(1, 11)]
    root_id = "G0"
    clone0  = Clone(root_id, None, 0, s0=s0_founder,
                    drivers=founder_drivers, passengers=founder_passengers)

    clones = [clone0]
    clones_dict = {root_id: clone0}

    # Global color manager remains fixed throughout each run to ensure cross-snapshot color consistency.
    color_manager = StableColorManager(founder_id=root_id, cmap_name="turbo", founder_color="skyblue")
    color_manager.ensure_ids([root_id])

    # Initial occupancy.
    all_sites = []
    R = radius
    for x in range(-R, R+1):
        for y in range(-R, R+1):
            for z in range(-R, R+1):
                if in_sphere_origin(x, y, z, R):
                    all_sites.append((x, y, z))
    all_sites.sort(key=lambda p: p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
    init_sites = all_sites[:FOUNDER_INIT_DEMES]
    init_cells = int(round(C_deme * INIT_FILL_FRACT))
    for p in init_sites:
        counts[p] = {root_id: init_cells}

    # Records for clone growth history.
    clone_history = {root_id: {'time': [0.0], 'N': [FOUNDER_INIT_DEMES * init_cells]}}
    time_points = [0.0]
    total_cells = [FOUNDER_INIT_DEMES * init_cells]

    # ===== Bulk passenger-event records; original logic is retained. =====
    bulk_passenger_events_step = defaultdict(int)
    bulk_passenger_records = []
    prev_totals_by_clone = {root_id: FOUNDER_INIT_DEMES * init_cells}

    # Export schedule.
    export_schedule = []
    if ENABLE_INTERMEDIATE_EXPORTS and EXPORT_TIMES:
        export_schedule.extend([float(t) for t in EXPORT_TIMES])
    if EXPORT_FINAL_ALWAYS:
        export_schedule.append(float(t_max))
    export_schedule = sorted(set(export_schedule))
    export_idx = 0

    # ============ Time loop ============
    t = 0.0
    while t < t_max - 1e-12:
        t = t + dt

        # Incremental radius expansion.
        if counts:
            max_r = max(math.sqrt(x*x+y*y+z*z) for (x,y,z) in counts.keys())
            if max_r >= radius - 1:
                radius += 2
                logger.info("[Run %03d] Expanded allowed radius to %d", run_idx+1, radius)

        # Global crowding suppression.
        N_total_now = sum(sum(dct.values()) for dct in counts.values())
        G_env = (1.0 - N_total_now/float(K_env_total_cells))
        if G_env < 0.0: G_env = 0.0
        G_env = G_env ** gamma_env
        if NOISE_ENV_SIGMA and NOISE_ENV_SIGMA > 0:
            G_env *= float(np.random.lognormal(mean=0.0, sigma=NOISE_ENV_SIGMA))
            G_env = max(0.0, min(G_env, 2.0))

        # 1) Birth and death.
        births_cache = defaultdict(lambda: defaultdict(int))
        deaths_cache = defaultdict(lambda: defaultdict(int))

        for p, dct in list(counts.items()):
            if not dct:
                continue
            N_i = sum(dct.values())
            crowd = (1.0 - N_i/float(C_deme))
            if crowd < 0.0: crowd = 0.0
            crowd = crowd ** gamma
            crowd_eff = crowd * G_env

            if ENABLE_BIRTH_SELECTION and N_i > 0:
                s_bar = sum(clones_dict[_cid].s * _n for _cid, _n in dct.items()) / float(N_i)
            else:
                s_bar = None

            for cid, n_ic in list(dct.items()):
                if n_ic <= 0:
                    continue
                c = clones_dict[cid]
                b_c = max(0.0, c.s + d_rate)
                lam_birth = n_ic * b_c * crowd_eff * dt
                if ENABLE_BIRTH_SELECTION and s_bar is not None:
                    lam_birth *= max(0.0, 1.0 + COMP_STRENGTH * (c.s - s_bar))
                if NOISE_BIRTH_SIGMA and NOISE_BIRTH_SIGMA > 0:
                    lam_birth *= float(np.random.lognormal(mean=0.0, sigma=NOISE_BIRTH_SIGMA))

                lam_death = n_ic * d_rate * dt
                nb = int(np.random.poisson(lam_birth)) if lam_birth > 0 else 0
                nd = int(np.random.poisson(lam_death)) if lam_death > 0 else 0
                if nd > n_ic + nb:
                    nd = n_ic + nb
                births_cache[p][cid] = nb
                deaths_cache[p][cid] = nd

        # 2) Apply births and deaths.
        for p, dct in births_cache.items():
            counts.setdefault(p, {})
            for cid, nb in dct.items():
                if nb > 0:
                    counts[p][cid] = counts[p].get(cid, 0) + nb

        for p, dct in deaths_cache.items():
            if p not in counts:
                continue
            for cid, nd in dct.items():
                after = counts[p].get(cid, 0) - nd
                if after <= 0:
                    counts[p].pop(cid, None)
            if not counts[p]:
                counts.pop(p, None)

        # 3) Branching, seeding, and passenger mutations; original logic is retained.
        for p, dct in list(counts.items()):
            if not dct:
                continue
            f_e = empty_neighbor_fraction_deme(p, counts, radius)
            p_noise = NOISE_RHO_BASE * (1.0 + BDRY_WEIGHT * (f_e ** BDRY_BETA))
            if p_noise > 1.0: p_noise = 1.0
            Fcin = CIN_factor(p, t, radius)

            for cid, n_ic in list(dct.items()):
                if n_ic <= 0:
                    continue
                c = clones_dict[cid]
                nb = births_cache[p].get(cid, 0)
                if nb <= 0:
                    continue

                # Aggregated passenger-event records.
                lambda_bulk = PASSENGER_LAMBDA * Fcin
                if lambda_bulk > 0:
                    kN_bulk = int(np.random.poisson(nb * lambda_bulk))
                    if kN_bulk > 0:
                        bulk_passenger_events_step[cid] += kN_bulk

                # Driver events.
                p_d_eff = p_d * c.p_d_factor * Fcin
                if p_d_eff > 1.0: p_d_eff = 1.0
                k_driver_events = int(np.random.binomial(nb, p_d_eff)) if p_d_eff > 0 else 0

                # Noise events.
                k_noise_events = int(np.random.binomial(nb, p_noise)) if p_noise > 0 else 0

                seed_cells_per_deme = max(1, int(round(SEED_CELLS_FRAC * C_deme)))
                noise_seed_cells    = max(1, int(round(NOISE_SEED_FRAC * C_deme)))

                # Candidate seeding demes.
                cand = [p] + deme_neighbors(p, radius)
                for q in list(cand):
                    cand += deme_neighbors(q, radius)
                seen = set(); cand_unique = []
                for q in cand:
                    if q not in seen:
                        seen.add(q); cand_unique.append(q)
                cand = cand_unique

                # Driver-derived subclones.
                for _ in range(k_driver_events):
                    kD = poisson_trunc_pos(DRIVER_LAMBDA * Fcin)
                    new_D = make_new_D(kD)
                    kN = int(np.random.poisson(PASSENGER_LAMBDA * Fcin))
                    addN = make_new_N(kN) if kN > 0 else []

                    cid_child = f"{c.id}.{len(c.children)+1}"
                    child = Clone(cid_child, parent=c, generation=c.generation+1,
                                  s0=c.s0, drivers=c.drivers + new_D,
                                  passengers=c.passengers + addN,
                                  p_d_factor=c.p_d_factor)
                    child.s = fitness_from_drivers(child.s0, child.drivers)
                    child.b = max(0.0, child.s + d_rate)
                    clones.append(child); clones_dict[cid_child] = child
                    c.children.append(child)

                    # Register each new clone color immediately to preserve cross-snapshot consistency.
                    color_manager.ensure_ids([cid_child])

                    # Select seeding demes.
                    targets = []
                    for q in cand:
                        if not in_sphere_origin(q[0], q[1], q[2], radius):
                            continue
                        cap_q = C_deme - sum(counts.get(q, {}).values())
                        if cap_q > 0:
                            targets.append(q)
                        if len(targets) >= DRIVER_SEED_DEMES:
                            break
                    if not targets:
                        targets = [p]

                    total_needed = seed_cells_per_deme * len(targets)
                    avail = counts[p].get(cid, 0)
                    move_total = min(avail, total_needed)
                    if move_total <= 0:
                        continue

                    per_target = seed_cells_per_deme if len(targets) > 0 else 0
                    for q in targets:
                        if move_total <= 0:
                            break
                        if not in_sphere_origin(q[0], q[1], q[2], radius):
                            continue
                        room = C_deme - sum(counts.get(q, {}).values())
                        put = min(per_target, room, move_total)
                        if put <= 0:
                            continue
                        counts.setdefault(q, {})
                        counts[q][cid_child] = counts[q].get(cid_child, 0) + put
                        counts[p][cid] = max(0, counts[p].get(cid, 0) - put)
                        move_total -= put
                    if counts[p].get(cid, 0) <= 0:
                        counts[p].pop(cid, None)
                        if not counts[p]:
                            counts.pop(p, None)

                # Noise-induced branches.
                for _ in range(k_noise_events):
                    kN2 = int(np.random.poisson(PASSENGER_LAMBDA * Fcin))
                    addN2 = make_new_N(kN2) if kN2 > 0 else []

                    noise_child_D = list(c.drivers)
                    noise_child_N = list(c.passengers) + addN2

                    s_driver = fitness_from_drivers(c.s0, noise_child_D)
                    beta_val = np.random.beta(BETA_A, BETA_B)
                    beta_mu  = BETA_A/(BETA_A+BETA_B)
                    delta_s  = s_max * NOISE_SCALE * (beta_val - beta_mu)
                    s_beta   = _clamp(c.s + delta_s, c.s0, c.s0 + s_max)
                    s_child  = max(s_driver, s_beta)

                    cid_noise = f"{c.id}.{len(c.children)+1}"
                    child2 = Clone(cid_noise, parent=c, generation=c.generation+1,
                                   s0=c.s0, drivers=noise_child_D,
                                   passengers=noise_child_N, p_d_factor=1.0)
                    child2.s = s_child
                    child2.b = max(0.0, child2.s + d_rate)
                    clones.append(child2); clones_dict[cid_noise] = child2
                    c.children.append(child2)

                    # Register color.
                    color_manager.ensure_ids([cid_noise])

                    avail = counts[p].get(cid, 0)
                    put = min(noise_seed_cells, avail, C_deme - sum(counts.get(p, {}).values()))
                    if put <= 0:
                        put = 1 if avail > 0 else 0
                    if put > 0:
                        counts[p][cid_noise] = counts[p].get(cid_noise, 0) + put
                        counts[p][cid] = max(0, counts[p][cid] - put)
                        if counts[p].get(cid, 0) <= 0:
                            counts[p].pop(cid, None)
                            if not counts[p]:
                                counts.pop(p, None)

        # 4) Migration.
        outflow_cache = defaultdict(lambda: defaultdict(int))
        inflow_cache  = defaultdict(lambda: defaultdict(int))
        for p, dct in list(counts.items()):
            if not dct:
                continue
            neigh = deme_neighbors(p, radius)
            if not neigh:
                continue
            per_deme_prob = min(1.0, m*dt)
            for cid, n_ic in list(dct.items()):
                if n_ic <= 0:
                    continue
                move_total = int(np.random.binomial(n_ic, per_deme_prob))
                if move_total <= 0:
                    continue
                for _ in range(move_total):
                    q = random.choice(neigh)
                    if not in_sphere_origin(q[0], q[1], q[2], radius):
                        continue
                    if sum(counts.get(q, {}).values()) + sum(inflow_cache[q].values()) < C_deme:
                        outflow_cache[p][cid] += 1
                        inflow_cache[q][cid]  += 1

        for p, dct in outflow_cache.items():
            for cid, v in dct.items():
                after = counts[p].get(cid, 0) - v
                if after <= 0:
                    counts[p].pop(cid, None)
            if not counts[p]:
                counts.pop(p, None)
        for q, dct in inflow_cache.items():
            counts.setdefault(q, {})
            for cid, v in dct.items():
                counts[q][cid] = counts[q].get(cid, 0) + v

        # 4.5) softmax cap
        if ENABLE_SOFTMAX_CAP:
            for p, dct in list(counts.items()):
                if not dct:
                    continue
                total = sum(dct.values())
                if total <= C_deme:
                    continue
                weights = {}
                for cid, nval in dct.items():
                    s_val = clones_dict[cid].s if cid in clones_dict else 0.0
                    w = max(0.0, float(nval)) * math.exp(SOFTMAX_BETA * s_val)
                    weights[cid] = w
                sum_w = sum(weights.values())
                if sum_w <= 0:
                    scale = C_deme / float(total)
                    for cid in list(dct.keys()):
                        dct[cid] = int(math.floor(dct[cid] * scale))
                    missing = C_deme - sum(dct.values())
                    if missing > 0 and dct:
                        for cid, _ in sorted(dct.items(), key=lambda kv: -kv[1]):
                            if missing <= 0:
                                break
                            dct[cid] += 1
                            missing -= 1
                    continue

                target = {}
                fracs  = []
                assigned = 0
                for cid, w in weights.items():
                    share = (w / sum_w) * C_deme
                    q_ = int(math.floor(share))
                    target[cid] = q_
                    assigned += q_
                    fracs.append((share - q_, cid))
                remain = C_deme - assigned
                fracs.sort(reverse=True)
                idx = 0
                while remain > 0 and idx < len(fracs):
                    target[fracs[idx][1]] += 1
                    remain -= 1
                    idx += 1

                for cid in list(dct.keys()):
                    dct[cid] = target.get(cid, 0)
                for cid in [k for k,v in dct.items() if v <= 0]:
                    dct.pop(cid, None)

        # 5) Record clone growth history.
        totals_by_clone = defaultdict(int)
        for dct in counts.values():
            for cid, v in dct.items():
                totals_by_clone[cid] += v

        for cid, tot in totals_by_clone.items():
            rec = clone_history.setdefault(cid, {'time': [], 'N': []})
            rec['time'].append(t)
            rec['N'].append(tot)

        total_cells.append(sum(totals_by_clone.values()))
        time_points.append(t)

        # Passenger-event records; original logic is retained.
        if bulk_passenger_events_step:
            for cid, ev in list(bulk_passenger_events_step.items()):
                if ev <= 0:
                    continue
                N_now  = max(1, totals_by_clone.get(cid, 0))
                N_prev = max(1, prev_totals_by_clone.get(cid, 1))
                bulk_passenger_records.append((cid, t, ev, N_prev, N_now))
            bulk_passenger_events_step.clear()
        prev_totals_by_clone = dict(totals_by_clone)

        # Export when a scheduled time is crossed, using the same color manager.
        while export_idx < len(export_schedule) and t >= export_schedule[export_idx] - 1e-12:
            t_target = export_schedule[export_idx]
            export_snapshot_outputs(
                folder=folder,
                t_target=t_target,
                counts_snapshot=counts,
                clone_history=clone_history,
                clones_dict=clones_dict,
                color_manager=color_manager,
                root_id=root_id
            )
            export_idx += 1

    # Terminal logs for the dynamic threshold.
    final_totals = compute_totals_from_counts(counts)
    N_final = int(sum(final_totals.values()))
    F_cells_final = compute_dynamic_F_cells(N_final, frac=F_CELLS_FRAC, min_cells=F_CELLS_MIN)
    filtered_ids_final = compute_filtered_ids_from_totals(final_totals, root_id=root_id, F_cells_dynamic=F_cells_final)

    logger.info("[Run %03d] Total cells at endpoint: %d", run_idx+1, N_final)
    logger.info("[Run %03d] Dynamic filtering threshold at endpoint: F_cells=%d (=%.1f%% of total)", run_idx+1, F_cells_final, F_CELLS_FRAC*100.0)
    logger.info("[Run %03d] Number of filtered clones: %d", run_idx+1, len(filtered_ids_final))
    logger.info("[Run %03d] Output completed.", run_idx+1)

# ==================== Main entry point ====================
def main():
    if SEED_BASE is not None:
        random.seed(SEED_BASE)
        np.random.seed(SEED_BASE)
    for run_idx in range(RUNS):
        if SEED_BASE is not None:
            random.seed(SEED_BASE + run_idx)
            np.random.seed(SEED_BASE + run_idx)
        simulate_once(run_idx)

if __name__ == "__main__":
    main()
