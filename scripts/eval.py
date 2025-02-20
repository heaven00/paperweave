import pandas as pd

from paperweave.evaluation.eval import create_results
from pathlib import Path

data_folder = Path(__file__).parent.parent / "data"
results_folder = data_folder / "pipeline_output"
annotation_file = data_folder / "annotation" / "annotation.csv"

if annotation_file.exists():
    previous_annotation = pd.read_csv(annotation_file, index_col=0)
else:
    previous_annotation = pd.DataFrame()
df = create_results(results_folder, previous_annotation)
df.to_csv(annotation_file)

print(df)
