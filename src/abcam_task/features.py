import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin

AMINOACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']  
CHAR_TO_IDX = {aa: idx for idx, aa in enumerate(AMINOACIDS)}


class OneHotLetterVectorTransformer(TransformerMixin):
    """
    Transformer to convert given input sequences into one-hot encoded vector matrices.
    Attributes:
        max_len (int) : Lenth of the longet sequence in the given dataset
    """

    def __init__(self):
        self.max_len = None

    def fit(self, X: pd.Series, y=None) -> 'OneHotLetterVectorTransformer':
        """
        Calculates the length of the longest sequence in the given dataset

        Args:
            X (pd.Series) : Series of amino acid sequences
            y(None, Optional): Not used
        """
        
        self.max_len = int(X.str.len().max()) # why is this float64?
        return self

    def transform(self, X: pd.Series) -> np.ndarray:
        """
        Each letter in the sequence is encoded as a 21-length one-hot encoded
        vector resulting in a vector of length max_len * 21,
        where max_len is the length of the longest sequence in the input series 

        Args:
             X (pd.Series) : Series of input sequences

        Returns:
            np.ndarray: A 2D NumPy array of shape (len(X), 21 * max_len) containing
                float32 one-hot representations.

        Raises:
            RuntimeError: If transform is called prior to fitting the transformer.

        """
        if self.max_len is None:
            raise RuntimeError("Please fit 'OneHotLetterVectorTransformer' first before invoking transform")

        # Padding sequences with 'X' to make all the sequences to similar length sequences
        padded_seqs = X.str.ljust(self.max_len, fillchar="X")

        # converting the charcter sequences to corresponding index/number sequences
        index_list = []
        for seq in padded_seqs:
            row_numbers = [CHAR_TO_IDX[aa] for aa in seq]
            index_list.append(row_numbers)

        # convert the nested list to 2D matrix
        index_matrix = np.array(index_list)

        onehot = np.eye(21, dtype=np.float32)[index_matrix]

        return onehot.reshape(len(X), 21*self.max_len)


class LetterCompositionTransformer(TransformerMixin):
    """Transformer to compute the normalized letter frequencies"""

    def fit(self, X: pd.Series, y=None) -> 'LetterCompositionTransformer':
        return self

    def transform(self, X: pd.Series) -> np.ndarray:
        """Transforms sequences into a frequency of each input letter in the sequence.

        Args:
            X (pd.Series): Series of input sequence strings.

        Returns:
            np.ndarray: A 2D NumPy array of shape (n_samples, 21) containing
                float32 letter frequencies for each sequence.
        """

        seq_lengths = X.str.len().replace(0,1) # length of sequence for each row 
        # replace(0,1) to avoid division by 0
   
        letter_count = {aa: X.str.count(aa) for aa in AMINOACIDS} 
        letter_count_df = pd.DataFrame(letter_count)

        freq_df = letter_count_df.div(seq_lengths, axis=0) # element wise division in each row

        return freq_df.to_numpy(dtype=np.float32)
