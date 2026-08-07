"""Unit tests for custom scikit-learn transformers in features.py."""

import numpy as np
import pandas as pd
import pytest
from abcam_task.features import (
    OneHotLetterVectorTransformer,
    LetterCompositionTransformer,
    AMINOACIDS,
    CHAR_TO_IDX,
)


@pytest.fixture
def sample_sequences():
    """Returns a pandas Series of sample amino acid sequences."""
    return pd.Series([
        "ACD",    
        "ACDEFGH", 
        "Y"    
    ])



# Tests for OneHotLetterVectorTransformer

class TestOneHotLetterVectorTransformer:
    """Test suite for OneHotLetterVectorTransformer."""

    def test_init(self):
        """Tests initial transformer state."""
        transformer = OneHotLetterVectorTransformer()
        assert transformer.max_len is None

    def test_fit(self, sample_sequences):
        """Tests if fit correctly finds and stores the maximum sequence length."""
        transformer = OneHotLetterVectorTransformer()
        fitted_transformer = transformer.fit(sample_sequences)

        # Longest sequence is "ACDEFGH" (length 7)
        assert transformer.max_len == 7
        assert isinstance(transformer.max_len, int)
        assert fitted_transformer is transformer  # Returns self 

    def test_transform_before_fit_raises_error(self, sample_sequences):
        """Tests that invoking transform before fit raises a RuntimeError."""
        transformer = OneHotLetterVectorTransformer()
        
        with pytest.raises(RuntimeError, match="Please fit 'OneHotLetterVectorTransformer' first before invoking transform"):
            transformer.transform(sample_sequences)

    def test_transform_shape_and_dtype(self, sample_sequences):
        """Tests output matrix shape and data type."""
        transformer = OneHotLetterVectorTransformer()
        transformer.fit(sample_sequences)
        
        matrix = transformer.transform(sample_sequences)
        
        # 3 samples, max_len=7, 21 amino acids in alphabet -> 7 * 21 = 147
        expected_shape = (len(sample_sequences), 21 * 7)
        
        assert isinstance(matrix, np.ndarray)
        assert matrix.dtype == np.float32
        assert matrix.shape == expected_shape

    def test_transform_padding_character(self):
        """Tests that shorter sequences are correctly padded with 'X' (index 20)."""
        seqs = pd.Series(["A"])  # Length 1, max_len=3
        transformer = OneHotLetterVectorTransformer()
        transformer.max_len = 3  # Simulate fit with max_len 3

        matrix = transformer.transform(seqs)
        # Sequence "A" padded to "AXX"
        # Position 0 -> 'A' (index 0)
        # Position 1 -> 'X' (index 20)
        # Position 2 -> 'X' (index 20)

        row = matrix[0]
        pos0 = row[0:21]   # First 21 columns
        pos1 = row[21:42]  # Second 21 columns
        pos2 = row[42:63]  # Third 21 columns

        # Position 0 ('A')
        assert pos0[CHAR_TO_IDX['A']] == 1.0
        assert np.sum(pos0) == 1.0

        # Position 1 ('X')
        assert pos1[CHAR_TO_IDX['X']] == 1.0
        assert np.sum(pos1) == 1.0

        # Position 2 ('X')
        assert pos2[CHAR_TO_IDX['X']] == 1.0
        assert np.sum(pos2) == 1.0

    def test_empty_string_handling(self):
        """Tests transformer execution on empty string sequences."""
        seqs = pd.Series(["", "ACD"])
        transformer = OneHotLetterVectorTransformer()
        transformer.fit(seqs)  # max_len = 3 ("ACD")

        matrix = transformer.transform(seqs)
        
        # Empty string "" gets padded to "XXX"
        first_row = matrix[0]
        assert len(first_row) == 3 * 21
        
        # Check that all 3 positions in the first row are encoded as 'X'
        for i in range(3):
            chunk = first_row[i * 21 : (i + 1) * 21]
            assert chunk[CHAR_TO_IDX['X']] == 1.0


# Tests for LetterCompositionTransformer

class TestLetterCompositionTransformer:
    """Test suite for LetterCompositionTransformer."""

    def test_fit(self, sample_sequences):
        """Tests that fit returns self without modifying state."""
        transformer = LetterCompositionTransformer()
        fitted_transformer = transformer.fit(sample_sequences)
        assert fitted_transformer is transformer

    def test_transform_shape_and_dtype(self, sample_sequences):
        """Tests output matrix shape and data type."""
        transformer = LetterCompositionTransformer()
        matrix = transformer.fit_transform(sample_sequences)

        # 3 samples, 21 amino acids in alphabet
        assert isinstance(matrix, np.ndarray)
        assert matrix.dtype == np.float32
        assert matrix.shape == (len(sample_sequences), len(AMINOACIDS))

    def test_frequency_calculation_single_letter(self):
        """Tests composition frequency calculation for a homopolymer sequence."""
        seqs = pd.Series(["AAAA"])
        transformer = LetterCompositionTransformer()
        matrix = transformer.fit_transform(seqs)

        row = matrix[0]
        # 'A' is index 0 -> 4/4 = 1.0
        assert row[CHAR_TO_IDX['A']] == 1.0
        # Sum of frequencies should equal 1.0
        assert np.isclose(np.sum(row), 1.0)

    def test_frequency_calculation_mixed_letters(self):
        """Tests composition frequency calculation for a multi-letter sequence."""
        seqs = pd.Series(["ACD"])  # Length 3 -> 'A': 1/3, 'C': 1/3, 'D': 1/3
        transformer = LetterCompositionTransformer()
        matrix = transformer.fit_transform(seqs)

        row = matrix[0]
        assert np.isclose(row[CHAR_TO_IDX['A']], 1/3)
        assert np.isclose(row[CHAR_TO_IDX['C']], 1/3)
        assert np.isclose(row[CHAR_TO_IDX['D']], 1/3)
        assert np.isclose(np.sum(row), 1.0)

    def test_empty_string_frequency(self):
        """Tests that empty strings produce zero frequencies instead of crash/NaN."""
        seqs = pd.Series([""])
        transformer = LetterCompositionTransformer()
        matrix = transformer.fit_transform(seqs)

        row = matrix[0]
        # Should be a vector of all zeros (no NaNs)
        assert not np.isnan(row).any()
        assert np.sum(row) == 0.0