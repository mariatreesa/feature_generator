import pandas as pd 
from abcam_task.features import OneHotLetterVectorTransformer, LetterCompositionTransformer

class FeatureGeneratePipeline:
    """Pipeline  that generates new feature vectors from CSV datasets.

    Attributes:
        one_hot_transformer (OneHotLetterVectorTransformer): One-hot vector transformer instance.
        comp_transformer (LetterCompositionTransformer): letter composition transformer instance.
    """

    def __init__(self):


        self.transformes = [
            ("one_hot", OneHotLetterVectorTransformer()),
            ("letter_comp", LetterCompositionTransformer())
        ]


    def run_pipeline(self, input_path: str):
        """Reads  data from input CSV and generates new feature vectors.

        Args:
            input_path (str): File path to input CSV.

        Returns:
            pd.DataFrame: DataFrame containing 'ID', 'one_hot', and 'letter_comp'
                columns where feature values are stored as 1D NumPy arrays per row.

        Raises:
            ValueError: If 'Sequence' or 'ID' columns are missing from input dataset.
        """

        uniprot_df = pd.read_csv(input_path)

        if "Sequence" not in uniprot_df.columns:
            raise ValueError("Expected an amino acid 'Sequence' column in the input csv")
        
        if "ID" not in uniprot_df.columns:
            raise ValueError("Expected an ID column in the dataset")

        # data cleaning
        uniprot_df["Sequence"] = uniprot_df["Sequence"].fillna("")
        uniprot_df["Sequence"] = uniprot_df["Sequence"].astype(str)
        uniprot_df["Sequence"] = uniprot_df["Sequence"].str.strip()
        uniprot_df["Sequence"] = uniprot_df["Sequence"].str.upper()

        sequences = uniprot_df["Sequence"]
        feature_dict = {"ID": uniprot_df["ID"].values}

        for name, transformer in self.transformes:
            transformed_feature = transformer.fit_transform(sequences)
            feature_dict[name] = list(transformed_feature) # convert to 1D array

        # output dataframe with 'ID' as first column and the new feature vectors in next  columns
        new_feature_df = pd.DataFrame(feature_dict)

        return new_feature_df
