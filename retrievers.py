import pandas as pd

# Login using e.g. `huggingface-cli login` to access this dataset
df = pd.read_parquet("hf://datasets/agents-course/unit3-invitees/data/train-00000-of-00001.parquet")