import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize prepared PrimeKG++ bundle using NetworkX k-hop neighborhoods around target drugs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/primekgpp_grace_redaf"),
        help="Directory produced by preprocessing/prepare_biomedkg_primekgpp.py",
    )
    parser.add_argument(
        "--target-drug-map",
        type=Path,
        default=Path("tahoe_to_primekg_map.csv"),
        help="CSV with target drugs (uses x_name if available, else drug).",
    )
    parser.add_argument(
        "--selected-drugs",
        type=str,
        default="",
        help="Optional comma-separated list of seed drugs. If empty, uses first N from target-drug-map.",
    )
    parser.add_argument(
        "--num-seed-drugs",
        type=int,
        default=1,
        help="Number of seed drugs from target-drug-map to visualize when --selected-drugs is not provided.",
    )
    parser.add_argument(
        "--hops",
        type=int,
        default=5,
        help="Neighborhood hop distance from each seed drug.",
    )
    parser.add_argument(
        "--max-subgraph-nodes",
        type=int,
        default=300,
        help="Maximum nodes to render in the neighborhood plot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling nodes.",
    )
    parser.add_argument(
        "--layout-k",
        type=float,
        default=0.2,
        help="Spring layout k parameter for NetworkX visualization.",
    )
    parser.add_argument(
        "--no-pca",
        action="store_true",
        help="Disable PCA-based node positioning and use NetworkX spring layout.",
    )
    parser.add_argument(
        "--max-edge-labels",
        type=int,
        default=80,
        help="Maximum number of edges to annotate with relation labels.",
    )
    parser.add_argument(
        "--max-edges-per-relation",
        type=int,
        default=3,
        help="At each hop, keep up to this many edges per relation type for expansion.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="primekgpp",
        help="Prefix for saved figure files.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("figs"),
        help="Directory to save generated figures and summary CSV.",
    )
    return parser.parse_args()


def load_inputs(input_dir: Path):
    node_embed_path = input_dir / "node_embeddings.npy"
    node_index_path = input_dir / "node_index.csv"
    rel_index_path = input_dir / "relation_index.csv"
    edges_path = input_dir / "edges_detailed.csv"

    for p in [node_embed_path, node_index_path, rel_index_path, edges_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    node_embeddings = np.load(node_embed_path)
    node_df = pd.read_csv(node_index_path)
    rel_df = pd.read_csv(rel_index_path)
    edges_df = pd.read_csv(edges_path)
    return node_embeddings, node_df, rel_df, edges_df


def load_seed_drugs(target_drug_map: Path, selected_drugs: str, num_seed_drugs: int):
    if selected_drugs.strip():
        return [s.strip() for s in selected_drugs.split(",") if s.strip()]

    if not target_drug_map.exists():
        raise FileNotFoundError(f"Target drug map not found: {target_drug_map}")

    df = pd.read_csv(target_drug_map)
    lower_to_col = {c.lower(): c for c in df.columns}
    name_col = lower_to_col.get("x_name") or lower_to_col.get("drug")
    if name_col is None:
        raise ValueError("Target drug map must contain x_name or drug column.")

    names = [str(v).strip() for v in df[name_col].dropna().tolist() if str(v).strip()]
    unique_names = sorted(set(names))
    return unique_names[:num_seed_drugs]


def build_graph(edges_df: pd.DataFrame):
    g = nx.Graph()

    relation_col = "relation_name" if "relation_name" in edges_df.columns else "relation_id"
    subset = edges_df[["head_id", "tail_id", relation_col]].dropna().copy()
    subset["head_id"] = subset["head_id"].astype(int)
    subset["tail_id"] = subset["tail_id"].astype(int)

    for row in subset.itertuples(index=False):
        src = int(row.head_id)
        dst = int(row.tail_id)
        rel = str(getattr(row, relation_col))

        if g.has_edge(src, dst):
            g[src][dst]["relations"].add(rel)
        else:
            g.add_edge(src, dst, relations={rel})

    for src, dst in g.edges():
        rels = sorted(g[src][dst]["relations"])
        if len(rels) <= 2:
            g[src][dst]["relation_label"] = "|".join(rels)
        else:
            g[src][dst]["relation_label"] = f"{rels[0]}|{rels[1]}|+{len(rels)-2}"

    return g


def seed_node_ids(node_df: pd.DataFrame, seed_drugs: list[str]):
    name_to_id = {
        str(row.node_name).strip().casefold(): int(row.node_id)
        for row in node_df.itertuples(index=False)
    }
    ids = []
    missing = []
    for drug in seed_drugs:
        node_id = name_to_id.get(drug.strip().casefold())
        if node_id is None:
            missing.append(drug)
        else:
            ids.append(node_id)
    return ids, missing


def get_khop_nodes(g: nx.Graph, seed_ids: list[int], hops: int):
    keep = set(seed_ids)
    for seed_id in seed_ids:
        if seed_id not in g:
            continue
        distances = nx.single_source_shortest_path_length(g, seed_id, cutoff=hops)
        keep.update(distances.keys())
    return keep


def get_min_hop_map(g: nx.Graph, seed_ids: list[int], hops: int):
    hop_map = {}
    for seed_id in seed_ids:
        if seed_id not in g:
            continue
        distances = nx.single_source_shortest_path_length(g, seed_id, cutoff=hops)
        for nid, d in distances.items():
            hop_map[nid] = min(hop_map.get(nid, d), d)
    return hop_map


def pca_2d(x: np.ndarray):
    x = np.asarray(x, dtype=np.float32)
    centered = x - x.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    return (u[:, :2] * s[:2]).astype(np.float32)


def sample_subgraph_edges_by_relation(sg: nx.Graph, max_edges_per_relation: int, seed: int):
    rng = np.random.default_rng(seed)

    rel_to_edges = {}
    for src, dst, attrs in sg.edges(data=True):
        rel_key = attrs.get("relation_label", "unknown")
        rel_to_edges.setdefault(rel_key, []).append((src, dst))

    keep_edges = set()
    for rel_key, rel_edges in rel_to_edges.items():
        if len(rel_edges) <= max_edges_per_relation:
            picked = rel_edges
        else:
            picked_idx = rng.choice(len(rel_edges), size=max_edges_per_relation, replace=False)
            picked = [rel_edges[int(i)] for i in picked_idx]
        for edge in picked:
            keep_edges.add(tuple(sorted(edge)))

    sampled = nx.Graph()
    for src, dst in sg.edges():
        key = tuple(sorted((src, dst)))
        if key not in keep_edges:
            continue
        sampled.add_node(src, **sg.nodes[src])
        sampled.add_node(dst, **sg.nodes[dst])
        sampled.add_edge(src, dst, **sg[src][dst])

    sampled.remove_nodes_from(list(nx.isolates(sampled)))
    return sampled


def build_positions(sg: nx.Graph, node_embeddings: np.ndarray, layout_k: float, seed: int, use_pca: bool):
    ordered_nodes = list(sg.nodes())
    if not ordered_nodes:
        return {}

    if not use_pca:
        return nx.spring_layout(sg, seed=seed, k=layout_k)

    emb = np.vstack([node_embeddings[int(nid)] for nid in ordered_nodes]).astype(np.float32)
    pca_xy = pca_2d(emb)
    if np.allclose(pca_xy.std(axis=0), 0.0):
        return nx.spring_layout(sg, seed=seed, k=layout_k)
    return {nid: pca_xy[i] for i, nid in enumerate(ordered_nodes)}


def draw_graph_on_axis(
    ax,
    sg: nx.Graph,
    pos,
    id_to_name,
    seed_set,
    max_edge_labels: int,
    seed: int,
    title: str,
):
    ordered_nodes = list(sg.nodes())
    node_colors = ["#d62728" if nid in seed_set else "#1f77b4" for nid in ordered_nodes]
    node_sizes = [160 if nid in seed_set else 30 for nid in ordered_nodes]
    node_labels = {nid: id_to_name.get(nid, str(nid)) for nid in ordered_nodes}

    nx.draw_networkx_edges(sg, pos, alpha=0.25, width=0.8, ax=ax)
    nx.draw_networkx_nodes(sg, pos, node_color=node_colors, node_size=node_sizes, alpha=0.95, ax=ax)
    nx.draw_networkx_labels(sg, pos, labels=node_labels, font_size=7, ax=ax)

    edge_items = list(sg.edges())
    if len(edge_items) <= max_edge_labels:
        label_edges = edge_items
    else:
        rng = np.random.default_rng(seed)
        picked = rng.choice(len(edge_items), size=max_edge_labels, replace=False)
        label_edges = [edge_items[int(i)] for i in picked]

    edge_labels = {(u, v): sg[u][v].get("relation_label", "") for u, v in label_edges}
    nx.draw_networkx_edge_labels(sg, pos, edge_labels=edge_labels, font_size=6, alpha=0.7, ax=ax)

    ax.set_title(title)
    ax.axis("off")


def trim_nodes_by_distance(g: nx.Graph, nodes: set[int], seed_ids: list[int], max_nodes: int):
    if len(nodes) <= max_nodes:
        return nodes

    if not seed_ids:
        return set(list(nodes)[:max_nodes])

    dist_map = {}
    for seed_id in seed_ids:
        if seed_id not in g:
            continue
        d = nx.single_source_shortest_path_length(g, seed_id)
        for nid in nodes:
            if nid in d:
                dist_map[nid] = min(dist_map.get(nid, d[nid]), d[nid])

    scored = sorted(nodes, key=lambda n: (dist_map.get(n, 10**9), n))
    return set(scored[:max_nodes])


def expand_spiderweb_subgraph(
    g: nx.Graph,
    center_id: int,
    hops: int,
    max_edges_per_relation: int,
    max_nodes: int,
    seed: int,
):
    rng = np.random.default_rng(seed)

    sub = nx.Graph()
    sub.add_node(center_id, hop=0)

    visited = {center_id}
    frontier = {center_id}

    for hop in range(1, hops + 1):
        if not frontier or sub.number_of_nodes() >= max_nodes:
            break

        rel_to_candidates = {}
        for src in sorted(frontier):
            if src not in g:
                continue
            for nbr, attrs in g[src].items():
                if nbr in visited:
                    continue
                rel = attrs.get("relation_label", "unknown")
                rel_to_candidates.setdefault(rel, []).append((src, nbr, attrs))

        next_frontier = set()
        for rel, candidates in rel_to_candidates.items():
            if len(candidates) > max_edges_per_relation:
                picked_idx = rng.choice(len(candidates), size=max_edges_per_relation, replace=False)
                picked = [candidates[int(i)] for i in picked_idx]
            else:
                picked = candidates

            for src, nbr, attrs in picked:
                if nbr in visited:
                    continue
                sub.add_node(nbr, hop=hop)
                sub.add_edge(src, nbr, **attrs)
                visited.add(nbr)
                next_frontier.add(nbr)

                if sub.number_of_nodes() >= max_nodes:
                    break
            if sub.number_of_nodes() >= max_nodes:
                break

        frontier = next_frontier

    return sub


def radial_positions_by_hop(sg: nx.Graph, seed: int):
    rng = np.random.default_rng(seed)
    hop_to_nodes = {}
    for nid, attrs in sg.nodes(data=True):
        hop = int(attrs.get("hop", 0))
        hop_to_nodes.setdefault(hop, []).append(nid)

    pos = {}
    for hop, nodes in sorted(hop_to_nodes.items(), key=lambda x: x[0]):
        if hop == 0:
            for nid in nodes:
                pos[nid] = np.array([0.0, 0.0], dtype=np.float32)
            continue

        nodes_sorted = sorted(nodes)
        n = len(nodes_sorted)
        if n == 0:
            continue

        start_angle = float(rng.uniform(0.0, 2.0 * np.pi))
        radius = float(hop)
        for i, nid in enumerate(nodes_sorted):
            angle = start_angle + (2.0 * np.pi * i / n)
            pos[nid] = np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=np.float32)

    return pos


def visualize_khop_graph(
    fig_dir: Path,
    g: nx.Graph,
    node_embeddings: np.ndarray,
    node_df: pd.DataFrame,
    seed_ids: list[int],
    hops: int,
    max_subgraph_nodes: int,
    layout_k: float,
    seed: int,
    max_edge_labels: int,
    no_pca: bool,
    max_edges_per_relation: int,
    out_prefix: str,
):
    if len(seed_ids) == 0:
        raise ValueError("No valid center drug node found in graph.")

    center_id = int(seed_ids[0])
    sg = expand_spiderweb_subgraph(
        g=g,
        center_id=center_id,
        hops=hops,
        max_edges_per_relation=max_edges_per_relation,
        max_nodes=max_subgraph_nodes,
        seed=seed,
    )

    if sg.number_of_nodes() == 0:
        raise ValueError("No nodes available after spiderweb expansion.")

    id_to_name = {
        int(row.node_id): str(row.node_name)
        for row in node_df.itertuples(index=False)
    }
    seed_set = {center_id}
    pos_spider = radial_positions_by_hop(sg, seed=seed)

    fig1_path = fig_dir / f"{out_prefix}_spiderweb_{hops}hop.png"
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 11))
    draw_graph_on_axis(
        ax1,
        sg,
        pos_spider,
        id_to_name=id_to_name,
        seed_set=seed_set,
        max_edge_labels=max_edge_labels,
        seed=seed,
        title=(
            f"PrimeKG++ Spiderweb Expansion from Center Drug ({hops} hops) "
            f"nodes={sg.number_of_nodes():,}, edges={sg.number_of_edges():,}"
        ),
    )
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label="center drug", markerfacecolor="#d62728", markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label="expanded entity", markerfacecolor="#1f77b4", markersize=8),
    ]
    fig1.legend(handles=handles, loc="upper center", ncol=2, frameon=True)
    fig1.tight_layout()
    fig1.savefig(fig1_path, dpi=200)
    plt.close(fig1)

    return fig1_path, sg.number_of_nodes(), sg.number_of_edges()


def main():
    args = parse_args()
    node_embeddings, node_df, _, edges_df = load_inputs(args.input_dir)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    seeds = load_seed_drugs(
        target_drug_map=args.target_drug_map,
        selected_drugs=args.selected_drugs,
        num_seed_drugs=args.num_seed_drugs,
    )
    g = build_graph(edges_df)
    seed_ids, missing_seeds = seed_node_ids(node_df=node_df, seed_drugs=seeds)

    fig1, n_nodes, n_edges = visualize_khop_graph(
        fig_dir=args.fig_dir,
        g=g,
        node_embeddings=node_embeddings,
        node_df=node_df,
        seed_ids=seed_ids,
        hops=args.hops,
        max_subgraph_nodes=args.max_subgraph_nodes,
        layout_k=args.layout_k,
        seed=args.seed,
        max_edge_labels=args.max_edge_labels,
        max_edges_per_relation=args.max_edges_per_relation,
        no_pca=args.no_pca,
        out_prefix=args.out_prefix,
    )

    print("PrimeKG++ visualization complete")
    print(f"Spiderweb figure      : {fig1}")
    print(f"Subgraph size         : nodes={n_nodes}, edges={n_edges}")
    print(f"Requested seeds       : {len(seeds)}")
    print(f"Matched seeds         : {len(seed_ids)}")
    if missing_seeds:
        print(f"Missing seeds preview : {missing_seeds[:20]}")


if __name__ == "__main__":
    main()
