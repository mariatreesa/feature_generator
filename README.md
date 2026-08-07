# Sequence Feature Extraction Pipeline

A  Python library with tools that extracts feature vectors from amino acid sequence datasets. It cleans input sequences and generates two feature representations:
* **One-Hot Encoded Vectors** (padded to maximum sequence length)
* **Amino Acid Composition Frequencies** (normalized 21-letter distribution)

---

## How to Run

### 1. Installation

Clone the repository and install dependencies using `uv`:

```bash
git clone <repository-url>
cd <repository-folder>
uv sync
```

Run the following from cli to generate features and save then as parquet.

'''bash
uv run python main.py --input data/uniprot_sequences.csv --output new_features.parquet
```
or

```
uv run generate-features --input ../uniport_sequences.csv --output ../test_output.parquet   
```

To run tests

```
uv run pytest
```