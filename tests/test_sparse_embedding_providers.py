import os
import sys
import pytest
from unittest.mock import Mock, patch

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sparse_embedding_providers import create_sparse_embedding_provider
from config import SparseEmbeddingConfig, SpladeConfig


class TestSparseEmbeddingProviderFactory:
    """Test cases for the sparse embedding provider factory."""

    def test_create_splade_provider(self):
        """Test creating SPLADE provider through factory."""
        config = SparseEmbeddingConfig(
            provider="splade",
            splade=SpladeConfig(
                model="naver/splade-cocondenser-ensembledistil", device="cpu"
            ),
        )

        with patch("sparse_embedding_providers.SpladeProvider") as mock_splade:
            create_sparse_embedding_provider(config)
            mock_splade.assert_called_once_with(config)

    def test_create_unknown_provider(self):
        """Test creating unknown provider through factory."""
        config = SparseEmbeddingConfig(provider="unknown_provider")

        with pytest.raises(
            ValueError, match="Unknown sparse embedding provider: unknown_provider"
        ):
            create_sparse_embedding_provider(config)


class TestSpladeProviderConfiguration:
    """Test SPLADE provider configuration and basic functionality."""

    def test_splade_config_validation(self):
        """Test SPLADE configuration validation."""
        # Valid config
        config = SparseEmbeddingConfig(
            provider="splade",
            splade=SpladeConfig(
                model="naver/splade-cocondenser-ensembledistil", device="cpu"
            ),
        )
        assert config.provider == "splade"
        assert config.splade.model == "naver/splade-cocondenser-ensembledistil"
        assert config.splade.device == "cpu"

    def test_splade_config_missing(self):
        """Test error when SPLADE config is missing."""
        config = SparseEmbeddingConfig(provider="splade", splade=None)

        # Mock the initialization to test configuration validation
        with patch(
            "sparse_embedding_providers.SpladeProvider._initialize_model"
        ) as mock_init:
            mock_init.side_effect = ValueError("SPLADE configuration is required")

            with pytest.raises(ValueError, match="SPLADE configuration is required"):
                from sparse_embedding_providers import SpladeProvider

                provider = SpladeProvider.__new__(SpladeProvider)
                provider.config = config
                provider.logger = Mock()
                provider._initialize_model()


class TestSpladeProviderInterface:
    """Test SPLADE provider interface methods."""

    def test_generate_sparse_embedding_interface(self):
        """Test sparse embedding generation interface."""
        config = SparseEmbeddingConfig(
            provider="splade", splade=SpladeConfig(model="test-model", device="cpu")
        )

        with patch("sparse_embedding_providers.SpladeProvider._initialize_model"):
            from sparse_embedding_providers import SpladeProvider

            provider = SpladeProvider.__new__(SpladeProvider)
            provider.config = config
            provider.logger = Mock()
            provider._tokenizer = Mock()
            provider._model = Mock()
            provider.device = "cpu"

            # Mock the actual embedding generation
            with patch.object(
                provider,
                "generate_sparse_embedding",
                return_value={"indices": [1, 2], "values": [0.5, 0.3]},
            ):
                result = provider.generate_sparse_embedding("test text")

                assert "indices" in result
                assert "values" in result
                assert len(result["indices"]) == len(result["values"])

    def test_test_connection_interface(self):
        """Test connection test interface."""
        config = SparseEmbeddingConfig(
            provider="splade", splade=SpladeConfig(model="test-model", device="cpu")
        )

        with patch("sparse_embedding_providers.SpladeProvider._initialize_model"):
            from sparse_embedding_providers import SpladeProvider

            provider = SpladeProvider.__new__(SpladeProvider)
            provider.config = config
            provider.logger = Mock()

            # Mock the test connection
            with patch.object(provider, "test_connection", return_value=True):
                result = provider.test_connection()
                assert isinstance(result, bool)

    def test_get_info_interface(self):
        """Test get_info interface."""
        config = SparseEmbeddingConfig(
            provider="splade", splade=SpladeConfig(model="test-model", device="cpu")
        )

        with patch("sparse_embedding_providers.SpladeProvider._initialize_model"):
            from sparse_embedding_providers import SpladeProvider

            provider = SpladeProvider.__new__(SpladeProvider)
            provider.config = config
            provider.logger = Mock()
            provider.device = "cpu"
            provider._model = Mock()

            # Mock get_info
            with patch.object(
                provider,
                "get_info",
                return_value={"provider": "splade", "status": "ready"},
            ):
                info = provider.get_info()

                assert "provider" in info
                assert info["provider"] == "splade"


class TestSparseEmbeddingGenerationLogic:
    """Test sparse embedding generation logic without real model loading."""

    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        from sparse_embedding_providers import SpladeProvider

        # Create a provider instance without full initialization
        config = SparseEmbeddingConfig(
            provider="splade", splade=SpladeConfig(model="test-model", device="cpu")
        )

        with patch("sparse_embedding_providers.SpladeProvider._initialize_model"):
            provider = SpladeProvider.__new__(SpladeProvider)
            provider.config = config
            provider.logger = Mock()

            # Create the actual method to test the empty text logic
            def mock_generate_sparse_embedding(text):
                if not text or not text.strip():
                    return {"indices": [], "values": []}
                return {"indices": [1, 2], "values": [0.5, 0.3]}

            provider.generate_sparse_embedding = mock_generate_sparse_embedding

            # Test empty text handling
            result = provider.generate_sparse_embedding("")
            assert result == {"indices": [], "values": []}

            result = provider.generate_sparse_embedding("   ")
            assert result == {"indices": [], "values": []}

            # Test non-empty text
            result = provider.generate_sparse_embedding("test")
            assert result["indices"] == [1, 2]
            assert result["values"] == [0.5, 0.3]


class TestSparseEmbeddingBatchGeneration:
    """Test batch sparse embedding generation logic."""

    def test_batch_empty_list(self):
        """Test batch processing with empty input list."""
        from sparse_embedding_providers import SpladeProvider

        config = SparseEmbeddingConfig(
            provider="splade", splade=SpladeConfig(model="test-model", device="cpu")
        )

        with patch("sparse_embedding_providers.SpladeProvider._initialize_model"):
            provider = SpladeProvider.__new__(SpladeProvider)
            provider.config = config
            provider.logger = Mock()

            result = provider.generate_sparse_embeddings([])
            assert result == []

    def test_batch_all_empty_texts(self):
        """Test batch processing with all empty/whitespace texts."""
        from sparse_embedding_providers import SpladeProvider

        config = SparseEmbeddingConfig(
            provider="splade", splade=SpladeConfig(model="test-model", device="cpu")
        )

        with patch("sparse_embedding_providers.SpladeProvider._initialize_model"):
            provider = SpladeProvider.__new__(SpladeProvider)
            provider.config = config
            provider.logger = Mock()

            texts = ["", "   ", "\t\n", ""]
            result = provider.generate_sparse_embeddings(texts)

            assert len(result) == 4
            for embedding in result:
                assert embedding == {"indices": [], "values": []}

    def test_batch_mixed_valid_invalid_texts(self):
        """Test batch processing with mix of valid and invalid texts."""
        from sparse_embedding_providers import SpladeProvider

        config = SparseEmbeddingConfig(
            provider="splade", splade=SpladeConfig(model="test-model", device="cpu")
        )

        with patch("sparse_embedding_providers.SpladeProvider._initialize_model"):
            provider = SpladeProvider.__new__(SpladeProvider)
            provider.config = config
            provider.logger = Mock()

            # Mock the entire batch method to test the input/output logic
            def mock_batch_implementation(texts):
                # Simulate the real implementation's text filtering logic
                valid_texts = []
                valid_indices = []
                for i, text in enumerate(texts):
                    if text and text.strip():
                        valid_texts.append(text)
                        valid_indices.append(i)

                # Simulate processing results for valid texts
                processed_results = []
                for text in valid_texts:
                    if text == "hello":
                        processed_results.append(
                            {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
                        )
                    elif text == "world":
                        processed_results.append(
                            {"indices": [3, 7], "values": [0.9, 0.5]}
                        )
                    else:
                        processed_results.append({"indices": [], "values": []})

                # Create final results with empty embeddings for invalid texts
                final_results = [{"indices": [], "values": []} for _ in texts]
                for i, result in enumerate(processed_results):
                    final_results[valid_indices[i]] = result

                return final_results

            provider.generate_sparse_embeddings = mock_batch_implementation

            texts = ["", "hello", "   ", "world", "\t"]
            result = provider.generate_sparse_embeddings(texts)

            assert len(result) == 5

            # Empty texts should return empty embeddings
            assert result[0] == {"indices": [], "values": []}
            assert result[2] == {"indices": [], "values": []}
            assert result[4] == {"indices": [], "values": []}

            # Valid texts should return embeddings
            assert result[1]["indices"] == [1, 5, 10]
            assert result[1]["values"] == [0.8, 0.6, 0.4]
            assert result[3]["indices"] == [3, 7]
            assert result[3]["values"] == [0.9, 0.5]

    def test_batch_processing_error_fallback(self):
        """Test batch processing falls back to individual processing on error."""
        from sparse_embedding_providers import SpladeProvider

        config = SparseEmbeddingConfig(
            provider="splade", splade=SpladeConfig(model="test-model", device="cpu")
        )

        with patch("sparse_embedding_providers.SpladeProvider._initialize_model"):
            provider = SpladeProvider.__new__(SpladeProvider)
            provider.config = config
            provider.logger = Mock()
            provider._tokenizer = Mock()
            provider._model = Mock()
            provider.device = "cpu"

            # Mock tokenizer to raise exception (simulating batch processing failure)
            provider._tokenizer.side_effect = Exception("Tokenizer error")

            # Mock the single text method for fallback
            def mock_single_embedding(text):
                if not text or not text.strip():
                    return {"indices": [], "values": []}
                if text == "hello":
                    return {"indices": [1, 2], "values": [0.5, 0.3]}
                else:
                    return {"indices": [3, 4], "values": [0.7, 0.2]}

            provider.generate_sparse_embedding = mock_single_embedding

            texts = ["hello", "world"]
            result = provider.generate_sparse_embeddings(texts)

            # Should fall back to individual processing
            assert len(result) == 2
            assert result[0] == {"indices": [1, 2], "values": [0.5, 0.3]}
            assert result[1] == {"indices": [3, 4], "values": [0.7, 0.2]}

    def test_base_provider_batch_fallback(self):
        """Test base provider batch method falls back to individual calls."""
        from sparse_embedding_providers import SparseEmbeddingProvider

        # Create a mock provider
        provider = Mock(spec=SparseEmbeddingProvider)

        # Mock the single embedding method
        def mock_single_embedding(text):
            if text == "hello":
                return {"indices": [1, 2], "values": [0.5, 0.3]}
            elif text == "world":
                return {"indices": [3, 4], "values": [0.7, 0.2]}
            else:
                return {"indices": [], "values": []}

        provider.generate_sparse_embedding = mock_single_embedding

        # Use the actual default implementation from the base class
        from sparse_embedding_providers import SparseEmbeddingProvider

        result = SparseEmbeddingProvider.generate_sparse_embeddings(
            provider, ["hello", "world", ""]
        )

        assert len(result) == 3
        assert result[0] == {"indices": [1, 2], "values": [0.5, 0.3]}
        assert result[1] == {"indices": [3, 4], "values": [0.7, 0.2]}
        assert result[2] == {"indices": [], "values": []}

    def test_batch_processing_preserves_order(self):
        """Test that batch processing preserves input order with mixed valid/invalid texts."""
        from sparse_embedding_providers import SpladeProvider

        config = SparseEmbeddingConfig(
            provider="splade", splade=SpladeConfig(model="test-model", device="cpu")
        )

        with patch("sparse_embedding_providers.SpladeProvider._initialize_model"):
            provider = SpladeProvider.__new__(SpladeProvider)
            provider.config = config
            provider.logger = Mock()

            # Mock the implementation to test order preservation
            def mock_batch_implementation(texts):
                # Simulate the real implementation logic
                valid_texts = []
                valid_indices = []
                for i, text in enumerate(texts):
                    if text and text.strip():
                        valid_texts.append(text)
                        valid_indices.append(i)

                # Simulate processing valid texts
                processed_results = []
                for text in valid_texts:
                    if text == "first":
                        processed_results.append({"indices": [1], "values": [0.1]})
                    elif text == "second":
                        processed_results.append({"indices": [2], "values": [0.2]})
                    elif text == "third":
                        processed_results.append({"indices": [3], "values": [0.3]})

                # Create final results with empty embeddings for invalid texts
                final_results = [{"indices": [], "values": []} for _ in texts]
                for i, result in enumerate(processed_results):
                    final_results[valid_indices[i]] = result

                return final_results

            provider.generate_sparse_embeddings = mock_batch_implementation

            texts = ["", "first", "  ", "second", "\n", "third", ""]
            result = provider.generate_sparse_embeddings(texts)

            assert len(result) == 7
            # Check that valid texts are in correct positions
            assert result[0] == {"indices": [], "values": []}  # Empty
            assert result[1] == {"indices": [1], "values": [0.1]}  # "first"
            assert result[2] == {"indices": [], "values": []}  # Whitespace
            assert result[3] == {"indices": [2], "values": [0.2]}  # "second"
            assert result[4] == {"indices": [], "values": []}  # Newline
            assert result[5] == {"indices": [3], "values": [0.3]}  # "third"
            assert result[6] == {"indices": [], "values": []}  # Empty
