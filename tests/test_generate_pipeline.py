
import pandas as pd
import numpy as np
import pytest
from abcam_task.generate_pipeline import FeatureGeneratePipeline


@pytest.fixture
def sample_uniprot_df(tmp_path):
    """Creates a temporary CSV file like UniProt data and returns its path."""
    df = pd.DataFrame({
        "ID": ["1", "2", "3"],
        "Sequence": ["ACD", "ACDEFGH", "Y"]
    })
    
    file_path = tmp_path / "test_uniprot.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


def test_pipeline_missing_columns(tmp_path):
    """Tests if the pipeline raises ValueError on missing required columns."""
    df_missing = pd.DataFrame({"BadColumn": ["A"], "AnotherBad": ["B"]})
    file_path = tmp_path / "bad_data.csv"
    df_missing.to_csv(file_path, index=False)
    
    pipeline = FeatureGeneratePipeline()
    
    with pytest.raises(ValueError, match="Expected an amino acid 'Sequence' column"):
        pipeline.run_pipeline(str(file_path))


def test_pipeline_execution(sample_uniprot_df):
    """Tests if the end-to-end pipeline produces the correct DataFrame structure."""
    pipeline = FeatureGeneratePipeline()
    result_df = pipeline.run_pipeline(sample_uniprot_df)
    
    # Validate final DataFrame columns
    expected_columns = ["ID", "one_hot", "letter_comp"]
    assert list(result_df.columns) == expected_columns
    
    # Validate that feature columns contain vector lists
    assert isinstance(result_df.loc[0, "one_hot"], np.ndarray)
    assert isinstance(result_df.loc[0, "letter_comp"], np.ndarray)
    
    # Validate row count matches expected 3 rows
    assert len(result_df) == 3