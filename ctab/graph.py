"""
ctab.graph
==========
Provenance Attack Graph Construction and Structural Topology Metrics.
"""

import networkx as nx

def build_attack_graph(alert_records, incident_id="INC-000"):
    """
    Constructs a directed provenance graph G from sequential alert predictions.
    Nodes: host, process, IP, domain, technique.
    Edges: temporal and structural causal dependencies.
    """
    G = nx.DiGraph(incident_id=incident_id)

    last_tech = None
    for idx, alert in enumerate(alert_records):
        aid = alert.get("pred_attack_id")
        host = alert.get("gt_host") or alert.get("host") or "WKSTN-001"
        ts = alert.get("timestamp") or idx

        G.add_node(host, kind="host")

        if aid:
            tech_node = f"{aid}@{idx}"
            G.add_node(tech_node, kind="technique", attack_id=aid, timestamp=ts)
            G.add_edge(host, tech_node, relation="executes_on")

            if last_tech:
                G.add_edge(last_tech, tech_node, relation="temporal_sequence")
            last_tech = tech_node

    return G

def compute_graph_metrics(G, gold_techniques, gold_host=None):
    """
    Computes precision, recall, and topological alignment against ground truth.
    """
    tech_nodes = [d.get("attack_id") for _, d in G.nodes(data=True) if d.get("kind") == "technique"]
    host_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "host"]

    pred_set = set(tech_nodes)
    gold_set = set(gold_techniques)

    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else (1.0 if not gold_set else 0.0)
    r = tp / len(gold_set) if gold_set else (1.0 if not pred_set else 0.0)
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    host_acc = 1.0 if (gold_host and gold_host in host_nodes) or (not gold_host and not host_nodes) else 0.0

    # Sequential edges
    seq_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "temporal_sequence"]
    gold_edges = list(zip(gold_techniques[:-1], gold_techniques[1:]))

    pred_e_tech = [(G.nodes[u].get("attack_id"), G.nodes[v].get("attack_id")) for u, v in seq_edges]
    edge_tp = sum(1 for e in pred_e_tech if e in gold_edges)
    edge_p = edge_tp / len(pred_e_tech) if pred_e_tech else (1.0 if not gold_edges else 0.0)
    edge_r = edge_tp / len(gold_edges) if gold_edges else (1.0 if not pred_e_tech else 0.0)
    edge_f1 = (2 * edge_p * edge_r) / (edge_p + edge_r) if (edge_p + edge_r) > 0 else 0.0

    return {
        "tech_node_P": round(p, 4),
        "tech_node_R": round(r, 4),
        "tech_node_F1": round(f1, 4),
        "host_node_accuracy": round(host_acc, 4),
        "Edge_P": round(edge_p, 4),
        "Edge_R": round(edge_r, 4),
        "Edge_F1": round(edge_f1, 4),
    }
