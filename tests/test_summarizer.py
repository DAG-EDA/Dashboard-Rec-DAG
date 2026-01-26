# test_summarizer.py
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.llm_client import GeminiClient
from src.data_summarizer import DataSummarizer

load_dotenv()

df = pd.read_json('datasets/movies.json')

# Initialize and run
llm = GeminiClient()
summarizer = DataSummarizer(llm)

summary = summarizer.summarize_dataset(
    df=df,
    dataset_name='test_movies',
    domain_hint='movies',
    output_path=Path('output/test_summary.json')
)

print("\n" + "="*50)
print("Attributes found:")
for attr in summary['attributes']:
    print(f"  - {attr['name']}: {attr['description'][:60]}...")

print("\nClusters found:")
for cluster in summary['attribute_clusters']:
    print(f"  - {cluster['cluster_name']}: {cluster['attributes']}")

print("\nConcepts suggested:")
for concept in summary['suggested_concepts']:
    print(f"  - {concept['name']}: {concept['definition'][:60]}...")