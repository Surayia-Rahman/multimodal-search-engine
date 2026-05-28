# src/embedder.py
import torch
from sentence_transformers import SentenceTransformer

class ProductEmbedder:
    """
    A high-performance embedding generation engine engineered to leverage 
    GPU hardware for mapping text descriptions into dense vector spaces.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Detect and assign the absolute fastest hardware backend available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing embedding model on device backend: {self.device.upper()}")
        
        # Load the pre-trained transformer model onto the selected device
        self.model = SentenceTransformer(model_name, device=self.device)

    def generate_embeddings(self, text_list: list) -> list:
        """
        Converts a list of raw text strings into high-dimensional vector embeddings.
        Utilizes batched parallel processing natively via the GPU.
        """
        print(f"Extracting features for {len(text_list)} items...")
        
        # Calculate embeddings; show progress bar and convert the output to standard lists
        embeddings = self.model.encode(
            text_list, 
            batch_size=32, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )
        return embeddings.tolist()
