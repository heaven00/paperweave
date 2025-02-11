from paperweave.evaluation.eval import create_results
from pathlib import Path

results_folder = Path(__file__).parent.parent / "data" / "pipeline_output"

df = create_results(results_folder)

print(df)