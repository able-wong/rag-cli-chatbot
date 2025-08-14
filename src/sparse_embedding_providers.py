from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging
from config import SparseEmbeddingConfig


class SparseEmbeddingProvider(ABC):
    """Abstract base class for sparse embedding providers."""

    @abstractmethod
    def generate_sparse_embedding(self, text: str) -> Dict[str, List[int]]:
        """Generate sparse embedding for the given text."""
        pass

    def generate_sparse_embeddings(
        self, texts: List[str]
    ) -> List[Dict[str, List[int]]]:
        """
        Generate sparse embeddings for multiple texts efficiently.

        Default implementation falls back to single-text processing.
        Providers should override this for better batch performance.
        """
        return [self.generate_sparse_embedding(text) for text in texts]

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the provider is available and working."""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get information about the provider."""
        pass


class SpladeProvider(SparseEmbeddingProvider):
    """SPLADE sparse embedding provider using neural sparse retrieval."""

    def __init__(self, config: SparseEmbeddingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._tokenizer = None
        self._model = None

        # Initialize model and tokenizer
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the SPLADE model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForMaskedLM
            import torch

            if not self.config.splade:
                raise ValueError("SPLADE configuration is required")

            model_name = self.config.splade.model
            device = self.config.splade.device

            self.logger.info(f"Loading SPLADE model: {model_name}")

            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForMaskedLM.from_pretrained(model_name)

            # Set device
            if device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
                self.device = "cuda"
            elif device == "mps" and torch.backends.mps.is_available():
                self._model = self._model.to("mps")
                self.device = "mps"
            else:
                self.device = "cpu"

            # Set to evaluation mode
            self._model.eval()

            self.logger.info(f"SPLADE model loaded successfully on {self.device}")

        except ImportError as e:
            raise ImportError(
                f"Required libraries not found: {e}. "
                "Install with: pip install transformers torch"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SPLADE model: {e}")

    def generate_sparse_embedding(self, text: str) -> Dict[str, List[int]]:
        """Generate sparse embedding from text using SPLADE."""
        if not text or not text.strip():
            return {"indices": [], "values": []}

        try:
            import torch

            # Tokenize input with truncation and padding
            inputs = self._tokenizer(
                text, return_tensors="pt", truncation=True, padding=True, max_length=512
            )

            # Move to appropriate device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate sparse representation
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits

                # Apply ReLU and log(1 + x) transformation as per SPLADE
                sparse_repr = torch.log(1 + torch.relu(logits))

                # Take max over sequence length for each token in vocabulary
                sparse_vector = torch.max(sparse_repr, dim=1)[0].squeeze()

                # Move back to CPU for processing
                if self.device != "cpu":
                    sparse_vector = sparse_vector.cpu()

                # Filter to non-zero values with threshold
                threshold = 0.01
                non_zero_mask = sparse_vector > threshold
                non_zero_indices = torch.nonzero(non_zero_mask).squeeze().tolist()

                # Handle single index case
                if isinstance(non_zero_indices, int):
                    non_zero_indices = [non_zero_indices]
                elif len(non_zero_indices) == 0:
                    return {"indices": [], "values": []}

                non_zero_values = sparse_vector[non_zero_indices].tolist()

                self.logger.debug(
                    f"Generated sparse vector: {len(non_zero_indices)} non-zero dimensions"
                )

                return {"indices": non_zero_indices, "values": non_zero_values}

        except Exception as e:
            self.logger.error(f"Error generating sparse embedding: {e}")
            return {"indices": [], "values": []}

    def generate_sparse_embeddings(
        self, texts: List[str]
    ) -> List[Dict[str, List[int]]]:
        """Generate sparse embeddings for multiple texts efficiently using batch processing."""
        if not texts:
            return []

        # Filter out empty texts and keep track of original positions
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)

        if not valid_texts:
            return [{"indices": [], "values": []} for _ in texts]

        try:
            import torch

            # Tokenize all texts in batch
            inputs = self._tokenizer(
                valid_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )

            # Move to appropriate device
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate sparse representations for all texts at once
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits

                # Apply ReLU and log(1 + x) transformation as per SPLADE
                sparse_repr = torch.log(1 + torch.relu(logits))

                # Take max over sequence length for each text
                sparse_vectors = torch.max(sparse_repr, dim=1)[0]

                # Move back to CPU for processing
                if self.device != "cpu":
                    sparse_vectors = sparse_vectors.cpu()

                # Process each sparse vector
                threshold = 0.01
                results = []

                for sparse_vector in sparse_vectors:
                    # Filter to non-zero values with threshold
                    non_zero_mask = sparse_vector > threshold
                    non_zero_indices = torch.nonzero(non_zero_mask).squeeze().tolist()

                    # Handle single index case
                    if isinstance(non_zero_indices, int):
                        non_zero_indices = [non_zero_indices]
                    elif len(non_zero_indices) == 0:
                        results.append({"indices": [], "values": []})
                        continue

                    non_zero_values = sparse_vector[non_zero_indices].tolist()
                    results.append(
                        {"indices": non_zero_indices, "values": non_zero_values}
                    )

                # Create final results array with empty embeddings for invalid texts
                final_results = [{"indices": [], "values": []} for _ in texts]
                for i, result in enumerate(results):
                    final_results[valid_indices[i]] = result

                self.logger.debug(
                    f"Generated {len(valid_texts)} sparse vectors in batch (out of {len(texts)} total)"
                )

                return final_results

        except Exception as e:
            self.logger.error(f"Error generating batch sparse embeddings: {e}")
            # Fallback to individual processing
            return super().generate_sparse_embeddings(texts)

    def test_connection(self) -> bool:
        """Test if SPLADE model is loaded and working."""
        try:
            # Test with simple text
            result = self.generate_sparse_embedding("test")
            return len(result["indices"]) > 0
        except Exception as e:
            self.logger.error(f"SPLADE connection test failed: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        """Get information about the SPLADE provider."""
        if not self.config.splade:
            return {"error": "SPLADE configuration not found"}

        return {
            "provider": "splade",
            "model": self.config.splade.model,
            "device": self.device,
            "status": "ready" if self._model is not None else "not_initialized",
        }


def create_sparse_embedding_provider(
    config: SparseEmbeddingConfig,
) -> SparseEmbeddingProvider:
    """Factory function to create sparse embedding provider based on config."""
    if config.provider.lower() == "splade":
        return SpladeProvider(config)
    else:
        raise ValueError(f"Unknown sparse embedding provider: {config.provider}")
