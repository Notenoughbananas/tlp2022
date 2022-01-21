import networkx as nx


def build_heuristic(heuristic_name):
    # Implementations inspired by networkx https://github.com/networkx/networkx/blob/93c99da588bf5b31c42cbad7de09f96f1754dbf7/networkx/algorithms/link_prediction.py

    def _apply_prediction(G, func, ebunch=None):
        if ebunch is None:
            ebunch = nx.non_edges(G)
        return ((u, v, func(u, v)) for u, v in ebunch)

    if heuristic_name == "cn":

        def predict_cn(G, ebunch=None):
            def cn(u, v):
                return sum(1 for _ in nx.common_neighbors(G, u, v))

            return _apply_prediction(G, cn, ebunch)

        return predict_cn

    elif heuristic_name == "aa":
        return nx.adamic_adar_index
    elif heuristic_name == "jaccard":
        return nx.jaccard_coefficient
    elif heuristic_name == "newton":

        def predict_newton(G, ebunch=None):

            shortest_path = nx.shortest_path(G)

            def newton(u, v):
                try:
                    path_len = len(shortest_path[u][v])
                except KeyError:  # Path does not exist
                    return 0

                return (G.degree[u] * G.degree[v]) / path_len ** 2

            return _apply_prediction(G, newton, ebunch)

        return predict_newton

    elif heuristic_name == "ccpa":

        def predict_ccpa(G, ebunch=None, alpha=0.8):
            shortest_path = nx.shortest_path(G)

            def ccpa(u, v):
                try:
                    path_len = len(shortest_path[u][v])
                except KeyError:  # Path does not exist
                    return 0

                return alpha * len(list(nx.common_neighbors(G, u, v))) + (1 - alpha) * (
                    G.number_of_nodes() / (path_len - 1)
                )

            return _apply_prediction(G, ccpa, ebunch)

    # 1/Shortest path
    elif heuristic_name == "invspl":
        # Returns 1/shortest path.
        # Thus the higher the value the stronger the connection (nodes are closer)
        # This is also to allow the "no shortest path" case to be represented as 0.

        def predict_spl(G, ebunch=None):
            spl = dict(nx.shortest_path_length(G))
            inf = float("inf")

            def pred_spl(u, v):
                if u == v:
                    raise nx.NetworkXAlgorithmError("Self links are not supported")
                return 1 / spl[u].get(v, inf)

            return _apply_prediction(G, pred_spl, ebunch)

        return predict_spl
