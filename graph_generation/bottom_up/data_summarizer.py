import pandas as pd
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from llm_client import GeminiClient

class DataSummarizer:
    def __init__(self, llm_client: GeminiClient):
        self.llm = llm_client
        self.prompts_dir = Path(__file__).parent.parent / 'bottom_up/prompts'
    
    def analyze_attributes(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze each attribute in the dataset."""
        attributes = []
        
        for col in df.columns:
            attr = {
                'name': col,
                'type': self._infer_type(df[col]),
                'role': self._infer_role(col, df[col]),
                'stats': self._compute_stats(df[col]),
                'sample_values': df[col].dropna().head(5).tolist()
            }
            attributes.append(attr)
        
        return attributes
    
    def _infer_type(self, series: pd.Series) -> str:
        """Infer data type."""
        dtype = series.dtype
        if pd.api.types.is_numeric_dtype(dtype):
            if pd.api.types.is_integer_dtype(dtype):
                return 'integer'
            return 'numerical'
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return 'temporal'
        elif pd.api.types.is_bool_dtype(dtype):
            return 'boolean'
        else:
            # Check if categorical (few unique values)
            n_unique = series.nunique()
            if n_unique < 20 or n_unique / len(series) < 0.05:
                return 'categorical'
            return 'text'
    
    def _infer_role(self, col_name: str, series: pd.Series) -> str:
        """Infer semantic role of attribute."""
        col_lower = col_name.lower()
        
        # Check for identifiers
        if 'id' in col_lower or col_name.endswith('_id'):
            return 'identifier'
        
        # Check for temporal
        if 'date' in col_lower or 'time' in col_lower or 'year' in col_lower:
            return 'temporal'
        
        # Check for measures (numerical)
        if pd.api.types.is_numeric_dtype(series.dtype):
            # Common measure keywords
            if any(word in col_lower for word in ['amount', 'price', 'cost', 'revenue', 'sales', 'count', 'score', 'rating']):
                return 'measure'
            return 'measure'  # default for numerical
        
        # Otherwise dimension/descriptor
        return 'dimension'
    
    def _compute_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Compute basic statistics."""
        stats = {
            'count': int(series.count()),
            'missing': int(series.isna().sum()),
            'missing_pct': float(series.isna().sum() / len(series))
        }
        
        if pd.api.types.is_numeric_dtype(series.dtype):
            stats.update({
                'min': float(series.min()) if not series.isna().all() else None,
                'max': float(series.max()) if not series.isna().all() else None,
                'mean': float(series.mean()) if not series.isna().all() else None,
                'median': float(series.median()) if not series.isna().all() else None
            })
        else:
            stats['n_unique'] = int(series.nunique())
            stats['top_values'] = series.value_counts().head(5).to_dict()
        
        return stats
    
    def generate_descriptions(self, attributes: List[Dict[str, Any]], 
                            domain_hint: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate human-readable descriptions for attributes using LLM."""
        
        # Load prompt template
        prompt_template = self._load_prompt('attribute_description.txt')
        
        # Format attributes for prompt
        attrs_text = json.dumps(attributes, indent=2)
        
        prompt = prompt_template.format(
            domain_hint=domain_hint or "unknown domain",
            attributes=attrs_text
        )
        
        try:
            response = self.llm.call_gemini(prompt)
            
            # Merge descriptions back into attributes
            descriptions = {item['name']: item['description'] 
                          for item in response.get('attributes', [])}
            
            for attr in attributes:
                attr['description'] = descriptions.get(attr['name'], 
                                                      f"The {attr['name']} attribute")
            
            return attributes
        
        except Exception as e:
            print(f"Warning: Failed to generate descriptions: {e}")
            # Fallback to simple descriptions
            for attr in attributes:
                attr['description'] = f"The {attr['name']} attribute"
            return attributes
    
    def cluster_attributes(self, attributes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Cluster related attributes using LLM."""
        
        prompt_template = self._load_prompt('attribute_clustering.txt')
        
        # Prepare attribute summary for clustering
        attr_summary = [
            {
                'name': attr['name'],
                'type': attr['type'],
                'role': attr['role'],
                'description': attr.get('description', '')
            }
            for attr in attributes
        ]
        
        prompt = prompt_template.format(
            attributes=json.dumps(attr_summary, indent=2)
        )
        
        try:
            response = self.llm.call_gemini(prompt)
            return response.get('clusters', [])
        
        except Exception as e:
            print(f"Warning: Failed to cluster attributes: {e}")
            # Fallback: group by role
            clusters = {}
            for attr in attributes:
                role = attr['role']
                if role not in clusters:
                    clusters[role] = []
                clusters[role].append(attr['name'])
            
            return [
                {
                    'cluster_name': role,
                    'attributes': attrs,
                    'rationale': f"Attributes with role: {role}"
                }
                for role, attrs in clusters.items()
            ]
    
    def suggest_concepts(self, attributes: List[Dict[str, Any]], 
                        clusters: List[Dict[str, Any]],
                        domain_hint: Optional[str] = None) -> List[Dict[str, Any]]:
        """Suggest initial analytical concepts using LLM."""
        
        prompt_template = self._load_prompt('concept_suggestion.txt')
        
        prompt = prompt_template.format(
            domain_hint=domain_hint or "unknown domain",
            attributes=json.dumps([
                {
                    'name': attr['name'],
                    'type': attr['type'],
                    'role': attr['role'],
                    'description': attr.get('description', '')
                }
                for attr in attributes
            ], indent=2),
            clusters=json.dumps(clusters, indent=2)
        )
        
        try:
            response = self.llm.call_gemini(prompt)
            return response.get('concepts', [])
        
        except Exception as e:
            print(f"Warning: Failed to suggest concepts: {e}")
            return []
    
    def summarize_dataset(self, 
                         df: pd.DataFrame,
                         dataset_name: str,
                         domain_hint: Optional[str] = None,
                         output_path: Optional[Path] = None) -> Dict[str, Any]:
        """Main function: summarize dataset and return structured output."""
        
        print(f"Analyzing dataset: {dataset_name}")
        print(f"Shape: {df.shape}")
        
        # Step 1: Analyze attributes
        print("\n1. Analyzing attributes...")
        attributes = self.analyze_attributes(df)
        print(f"   Found {len(attributes)} attributes")
        
        # Step 2: Generate descriptions
        print("\n2. Generating descriptions...")
        attributes = self.generate_descriptions(attributes, domain_hint)
        
        # Step 3: Cluster attributes
        print("\n3. Clustering attributes...")
        clusters = self.cluster_attributes(attributes)
        print(f"   Found {len(clusters)} clusters")
        
        # Step 4: Suggest concepts
        print("\n4. Suggesting concepts...")
        concepts = self.suggest_concepts(attributes, clusters, domain_hint)
        print(f"   Suggested {len(concepts)} concepts")
        
        # Build summary
        summary = {
            'dataset_info': {
                'name': dataset_name,
                'n_rows': len(df),
                'n_columns': len(df.columns),
                'domain': domain_hint
            },
            'attributes': attributes,
            'attribute_clusters': clusters,
            'suggested_concepts': concepts
        }
        
        # Save if output path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n✓ Summary saved to: {output_path}")
        
        return summary
    
    def _load_prompt(self, filename: str) -> str:
        """Load prompt template from file."""
        path = self.prompts_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text()


# Example usage
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize
    llm = GeminiClient()
    summarizer = DataSummarizer(llm)
    
    # Load dataset
    df = pd.read_csv('datasets/movies.csv')
    
    # Summarize
    summary = summarizer.summarize_dataset(
        df=df,
        dataset_name='movie_data',
        domain_hint='movies and entertainment',
        output_path=Path('graph_generation/bottom_up/output/movie_data_summary.json')
    )
    
    print("\n" + "="*50)
    print("SUMMARY COMPLETE")
    print("="*50)
