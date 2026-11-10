class RRFRetriever:
    """
    Classe permettant de combiner les résultats de plusieurs systèmes de recherche
    en utilisant la méthode Reciprocal Rank Fusion (RRF).
    """

    def __init__(self):
        pass

    def fuse(self, results_a, results_b, k=60, top_k=100):
        """
        Fusionne deux ensembles de résultats selon la méthode RRF.
        
        Args:
            results_a (dict): Résultats du système A {qid: [(pid, score), ...]}.
            results_b (dict): Résultats du système B {qid: [(pid, score), ...]}.
            k (int): Constante de lissage pour pénaliser les rangs inférieurs (défaut: 60).
            top_k (int): Nombre de documents à conserver par requête dans la liste finale.
            
        Returns:
            dict: Résultats fusionnés et triés {qid: [(pid, rrf_score), ...]}.
        """
        fused_results = {}
        
        # On récupère toutes les requêtes uniques (union des clés des deux systèmes)
        all_queries = set(results_a.keys()).union(set(results_b.keys()))
        
        for qid in all_queries:
            doc_scores = {}
            
            # Traitement des résultats de la liste A (ex: BM25)
            if qid in results_a:
                # enumerate commence à 0, donc le rang réel est rank + 1
                for rank, (pid, _) in enumerate(results_a[qid]):
                    doc_scores[pid] = doc_scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
                    
            # Traitement des résultats de la liste B (ex: Dense)
            if qid in results_b:
                for rank, (pid, _) in enumerate(results_b[qid]):
                    doc_scores[pid] = doc_scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
            
            # Tri des documents par score RRF décroissant
            sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
            
            # On conserve uniquement les top_k documents
            fused_results[qid] = sorted_docs[:top_k]
            
        return fused_results