# Knowledge Graph Builder (kg_builder.py)

## Overview

The KG Builder constructs knowledge graphs from data summaries produced by the Data Summarizer. It implements multiple edge inference strategies and integrates with the canonical edge type system.

## Key Features

### 1. Multi-Layer Graph Construction
- **Data Layer**: Nodes for each dataset attribute with automatic node ID generation
- **Concept Layer**: Nodes from suggested concepts with semantic relationships
- **Bridge Layer**: Links between concepts and data variables (operationalized_by edges)

### 2. Multiple Edge Inference Strategies

#### Oneshot Strategy (Default)
- Single LLM call with all nodes
- Fast and cost-effective
- Good for initial exploration
- May miss some relationships

```python
kg, stats = builder.infer_causal_relationships(
    kg, node_mapping, summary,
    strategy='oneshot',
    confidence_threshold=0.6
)
```

#### Pairwise Strategy
- Checks each node pair individually
- Batches pairs for efficiency (10 pairs per call)
- More thorough but slower
- Better recall

```python
kg, stats = builder.infer_causal_relationships(
    kg, node_mapping, summary,
    strategy='pairwise',
    confidence_threshold=0.6
)
```

### 3. Edge Post-Processing

Automatic post-processing includes:
1. **Confidence filtering**: Removes edges below threshold
2. **Deduplication**: Removes exact duplicates
3. **Symmetric normalization**: Keeps only one direction for symmetric edges
4. **Inverse duplicate removal**: Prevents both A→B (part_of) and B→A (has_part)

```python
processed_edges = builder.post_process_edges(
    raw_edges,
    layer='data',
    kg=kg,
    confidence_threshold=0.6
)
```

## API Reference

### KGBuilder Class

#### Core Methods

**`build_data_layer(summary) -> (KG, Dict)`**
- Creates data nodes from summary attributes
- Returns graph and node mapping

**`infer_causal_relationships(kg, node_mapping, summary, domain_hint, strategy, confidence_threshold) -> (KG, Dict)`**
- Infers causal edges between data variables
- Supports 'oneshot' and 'pairwise' strategies
- Returns updated graph and inference stats

**`build_concept_layer(kg, summary) -> (KG, Dict)`**
- Creates concept nodes and semantic edges
- Returns updated graph and concept mapping

**`create_bridge_edges(kg, data_mapping, concept_mapping, summary) -> KG`**
- Links concepts to their operationalizing data variables
- Returns updated graph

**`build_initial_kg(summary, domain_hint) -> KG`**
- Main orchestration function
- Builds complete KG with all layers
- Performs validation and reports statistics

#### Inference Strategy Methods

**`infer_edges_oneshot(nodes, layer, domain_hint) -> List[Edge]`**
- Single LLM call for all relationships
- Returns raw edge list

**`infer_edges_pairwise_all(nodes, layer, domain_hint, max_api_calls, batch_size) -> List[Edge]`**
- Checks all node pairs
- Batches for efficiency
- Returns raw edge list

**`post_process_edges(edges, layer, kg, confidence_threshold) -> List[Edge]`**
- Filters, deduplicates, and normalizes edges
- Returns processed edge list

#### Utility Methods

**`save_kg(kg, output_path)`**
- Saves graph to JSON file

**`load_kg(input_path) -> KG`**
- Loads graph from JSON file

## Usage Examples

### Basic Usage

```python
from src.kg_builder import KGBuilder
from src.llm_client import GeminiClient
import json

# Initialize
llm = GeminiClient()
builder = KGBuilder(llm)

# Load summary
with open('output/test_summary.json', 'r') as f:
    summary = json.load(f)

# Build knowledge graph
kg = builder.build_initial_kg(summary, domain_hint='movies')

# Save
builder.save_kg(kg, Path('output/initial_kg.json'))
```

### Step-by-Step Construction

```python
# Step 1: Build data layer
kg, data_mapping = builder.build_data_layer(summary)

# Step 2: Infer causal relationships (with strategy selection)
kg, stats = builder.infer_causal_relationships(
    kg, data_mapping, summary,
    domain_hint='movies',
    strategy='oneshot',  # or 'pairwise'
    confidence_threshold=0.7
)

print(f"Added {stats['added_edges']} edges in {stats['elapsed_time']:.2f}s")

# Step 3: Build concept layer
kg, concept_mapping = builder.build_concept_layer(kg, summary)

# Step 4: Create bridge edges
kg = builder.create_bridge_edges(kg, data_mapping, concept_mapping, summary)

# Validate
validation = kg.validate()
if validation['valid']:
    print("✓ Graph is valid")
```

### Comparing Strategies

```python
# Test oneshot
kg1, data_mapping = builder.build_data_layer(summary)
kg1, oneshot_stats = builder.infer_causal_relationships(
    kg1, data_mapping, summary,
    strategy='oneshot',
    confidence_threshold=0.6
)

# Test pairwise
kg2, data_mapping = builder.build_data_layer(summary)
kg2, pairwise_stats = builder.infer_causal_relationships(
    kg2, data_mapping, summary,
    strategy='pairwise',
    confidence_threshold=0.6
)

# Compare
print(f"Oneshot: {oneshot_stats['added_edges']} edges in {oneshot_stats['elapsed_time']:.2f}s")
print(f"Pairwise: {pairwise_stats['added_edges']} edges in {pairwise_stats['elapsed_time']:.2f}s")
```

## Inference Stats

The `infer_causal_relationships` method returns stats:

```python
{
    'strategy': 'oneshot',
    'raw_edges': 12,           # Before post-processing
    'processed_edges': 10,     # After post-processing
    'added_edges': 9,          # Successfully added to graph
    'elapsed_time': 3.45,      # Seconds
    'confidence_threshold': 0.6
}
```

## Edge Type Support

### Data Layer
- `causal_influence`: Direct causal relationships
- `computed_from`: Mathematical derivations
- `correlates_with`: Statistical associations

### Concept Layer
- `part_of` / `has_part`: Component relationships
- `similar_to`: Semantic similarity
- `contrasts_with`: Opposing concepts
- `prerequisite_for` / `requires`: Learning dependencies
- `measured_by_multiple`: Multi-component operationalization

### Bridge Layer
- `operationalized_by`: Concept-to-data measurement links

## Testing

### Basic Tests (Fast, No LLM)
```bash
# Test structure only
pytest tests/test_kg_builder.py::TestKGBuilder -v -k "not integration"
```

### Integration Tests (Slow, Requires API Key)
```bash
# With LLM inference
pytest tests/test_kg_builder.py::TestKGBuilderIntegration -v -s
```

### Strategy Comparison Test
```bash
# Compare oneshot vs pairwise
pytest tests/test_inference_strategies.py -v -s
```

## Strategy Selection Guidelines

### Use Oneshot When:
- Initial exploration of dataset
- Time/cost constraints
- Dataset has < 20 attributes
- Need quick insights

### Use Pairwise When:
- Thorough analysis required
- Dataset has complex relationships
- High recall is important
- Time/cost less critical

### Comparison Metrics
- **Edge Count**: Pairwise typically finds more edges
- **Execution Time**: Oneshot is ~3-5x faster
- **Confidence**: Both produce high-confidence edges
- **Overlap**: Common edges are high-confidence relationships

## Prompt Engineering

### Causal Inference Prompts
- Emphasize domain knowledge
- Request specific relationship types
- Specify confidence thresholds
- Provide example patterns

### Concept Relationship Prompts
- Focus on semantic relationships
- Prevent inverse duplicates
- Consider abstraction levels
- Request clear reasoning

## Validation

The built graph is automatically validated:
- No dangling edges
- All edge types are canonical
- Bridge edges connect correct layers
- No simple cycles with directional edges

## Output Format

Graphs are saved as JSON with structure:

```json
{
  "layers": {
    "data": {
      "nodes": {...},
      "edges": [...]
    },
    "concept": {
      "nodes": {...},
      "edges": [...]
    },
    "question": {
      "nodes": {...},
      "edges": []
    }
  },
  "bridge_edges": [...],
  "metadata": {}
}
```

## Performance Considerations

### Oneshot Strategy
- API Calls: 1
- Time: ~2-5 seconds
- Cost: Low
- Edges Found: Moderate

### Pairwise Strategy
- API Calls: ceil(n_pairs / batch_size)
- Time: ~30-120 seconds (depends on n)
- Cost: Higher
- Edges Found: More comprehensive

### Optimization Tips
1. Filter attributes before inference (remove IDs, low-info columns)
2. Use higher confidence threshold to reduce noise
3. Start with oneshot, use pairwise for specific subgraphs
4. Cache LLM responses for repeated calls

## Error Handling

The builder handles various error conditions:
- **Missing nodes**: Warnings, skips edge
- **Invalid edge types**: Converts to default or skips
- **Cycle detection**: Prevents simple cycles
- **Inverse duplicates**: Automatically removed
- **LLM failures**: Graceful fallback, continues

## Integration with Data Summarizer

```python
# Complete pipeline
from src.data_summarizer import DataSummarizer
from src.kg_builder import KGBuilder
from src.llm_client import GeminiClient
import pandas as df

# Initialize
llm = GeminiClient()
summarizer = DataSummarizer(llm)
builder = KGBuilder(llm)

# Load data
df = pd.read_csv('data/movies.csv')

# Summarize
summary = summarizer.summarize_dataset(
    df, 'movies', domain_hint='movies and entertainment'
)

# Build KG
kg = builder.build_initial_kg(summary, domain_hint='movies')

# Use the graph
stats = kg.get_layer_stats()
print(f"Data nodes: {stats['data']['node_count']}")
print(f"Concept nodes: {stats['concept']['node_count']}")
print(f"Bridge edges: {stats['bridge']['edge_count']}")
```

## Future Enhancements

Potential improvements:
1. **Hybrid strategy**: Combine oneshot + targeted pairwise
2. **Active learning**: User feedback to refine edges
3. **Confidence calibration**: Learn optimal thresholds
4. **Relationship extraction**: From text descriptions
5. **Graph refinement**: Iterative improvement cycles
6. **Domain-specific templates**: Pre-configured for common domains

---

**Status**: ✅ Fully implemented with multiple strategies
**Version**: 1.0
**Last Updated**: 2025-12-05
