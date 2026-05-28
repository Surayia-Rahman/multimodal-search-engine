# src/search_engine.py
import numpy as np

class SemanticSearchEngine:
    """
    An enterprise-grade mathematical search utility engineered to perform 
    vector similarity metrics over structured embeddings.
    """
    def __init__(self, dataframe):
        self.df = dataframe
        # Extract the list of product vectors and convert into a fast numpy matrix
        self.vector_matrix = np.array(dataframe["vectors"].tolist())

    def _cosine_similarity(self, query_vector, matrix_vectors):
        """
        Calculates the cosine angle distance between a query vector 
        and a matrix of candidate vectors.
        """
        # Vectorized dot product calculation
        dot_product = np.dot(matrix_vectors, query_vector)
        
        # Calculate matrix norms for normalization
        matrix_norms = np.linalg.norm(matrix_vectors, axis=1)
        query_norm = np.linalg.norm(query_vector)
        
        # Avoid division by zero warnings by adding a tiny epsilon value
        similarity_scores = dot_product / (matrix_norms * query_norm + 1e-9)
        return similarity_scores

    def execute_query(self, query_vector, top_k: int = 3):
        """
        Scores all products against a given query vector and returns 
        the top K highest ranking results.
        """
        scores = self._cosine_similarity(query_vector, self.vector_matrix)
        
        # Assign scores back to a temporary view of our data layer
        results_df = self.df.copy()
        results_df["similarity_score"] = scores
        
        # Sort values descending and pull out the best matches
        top_matches = results_df.sort_values(by="similarity_score", ascending=False).head(top_k)
        return top_matches[["product_id", "title", "category", "similarity_score"]]
