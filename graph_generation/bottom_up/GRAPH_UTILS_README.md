# Canonical Edge Type System

## Overview

The canonical edge type system provides a robust framework for managing three-layer knowledge graphs with enforced edge type constraints and validation rules.

## Implementation Summary

### Files Created

1. **src/graph_utils.py** (780+ lines)
   - `CANONICAL_EDGE_TYPES`: Complete edge type definitions for all layers
   - `KnowledgeGraph`: Main class with full node/edge operations

2. **tests/test_graph_utils.py** (850+ lines)
   - 39 comprehensive tests covering all functionality
   - All tests passing ✓

3. **demo_graph_utils.py**
   - 6 demonstrations showing all features in action

## Key Features

### 1. Canonical Edge Types

#### Data Layer
- `causal_influence`: Direct causal relationships (directional, no inverse)
- `correlates_with`: Statistical associations (symmetric, self-inverse)
- `computed_from`: Mathematical derivations (directional, no inverse)

#### Concept Layer
- `part_of` / `has_part`: Component relationships (inverse pair)
- `contrasts_with`: Opposing concepts (symmetric, self-inverse)
- `similar_to`: Related concepts (symmetric, self-inverse)
- `prerequisite_for` / `requires`: Learning dependencies (inverse pair)
- `measured_by_multiple`: Multi-component operationalization (directional)

#### Bridge Layer
- `operationalized_by`: Concept-to-data measurement (directional)

### 2. Core Operations

#### Node Operations
```python
kg = KnowledgeGraph()

# Add nodes
kg.add_node('data', 'data_revenue', {'name': 'revenue', 'type': 'float'})

# Get nodes
node = kg.get_node('data_revenue')
node = kg.get_node('data_revenue', layer='data')

# Update nodes
kg.update_node('data_revenue', {'activated': True})

# Remove nodes (and connected edges)
kg.remove_node('data_revenue')

# Find nodes with filters
nodes = kg.find_nodes(layer='concept', activated=True)
```

#### Edge Operations
```python
# Add edges with validation
kg.add_edge('data_A', 'data_B', 'causal_influence', 'data',
           metadata={'confidence': 0.95})

# Add bridge edges
kg.add_bridge_edge('concept_Revenue', 'data_revenue', 'operationalized_by')

# Query edges
edges = kg.get_edges(node_id='data_A')
edges = kg.get_edges(edge_type='causal_influence', layer='data')

# Get neighbors
neighbors = kg.get_neighbors('data_A', direction='out')
neighbors = kg.get_neighbors('data_A', edge_type='causal_influence', direction='in')
```

### 3. Validation Rules

#### ✓ Prevents Inverse Duplicates
If `A --part_of--> B` exists, prevents `B --has_part--> A`

#### ✓ Prevents Simple Cycles
If `A --causal_influence--> B` exists, prevents `B --causal_influence--> A`

#### ✓ Allows Longer Cycles
`A → B → C → A` is permitted (only direct cycles blocked)

#### ✓ Canonical Type Enforcement
Only allows edge types defined in `CANONICAL_EDGE_TYPES`

### 4. Path Finding

```python
# Find first path
path = kg.find_path('data_A', 'data_C')
# Returns: ['data_A', 'data_B', 'data_C']

# Find all paths
paths = kg.find_all_paths('data_A', 'data_C', max_depth=5, max_paths=100)
# Returns: [['data_A', 'data_B', 'data_C'], ['data_A', 'data_D', 'data_C']]

# Filter by edge type
path = kg.find_path('data_A', 'data_C', edge_type='causal_influence')
```

### 5. Utilities

```python
# Get statistics
stats = kg.get_layer_stats()
# Returns: {'data': {'node_count': 5, 'edge_count': 3, 'edge_types': {...}}, ...}

# Get edge type info
info = kg.get_edge_type_info('causal_influence', 'data')
# Returns: {'definition': '...', 'directional': True, ...}

# Validate graph
result = kg.validate()
# Returns: {'valid': True, 'errors': [], 'warnings': [], ...}

# Serialize/deserialize
json_data = kg.to_json()
kg2 = KnowledgeGraph.from_json(json_data)
```

## Usage Examples

### Example 1: Building a Marketing Knowledge Graph

```python
from src.graph_utils import KnowledgeGraph

kg = KnowledgeGraph()

# Add data nodes
kg.add_node('data', 'data_ad_spend', {'name': 'ad_spend', 'type': 'float'})
kg.add_node('data', 'data_clicks', {'name': 'clicks', 'type': 'int'})
kg.add_node('data', 'data_revenue', {'name': 'revenue', 'type': 'float'})
kg.add_node('data', 'data_ctr', {'name': 'click_through_rate', 'type': 'float'})

# Add concept nodes
kg.add_node('concept', 'concept_Marketing_ROI', {'name': 'Marketing ROI'})
kg.add_node('concept', 'concept_Customer_Engagement', {'name': 'Customer Engagement'})

# Create causal relationships
kg.add_edge('data_ad_spend', 'data_clicks', 'causal_influence', 'data')
kg.add_edge('data_clicks', 'data_revenue', 'causal_influence', 'data')

# Create derived variable
kg.add_edge('data_clicks', 'data_ctr', 'computed_from', 'data',
           metadata={'formula': 'clicks / impressions'})

# Link concepts to data
kg.add_bridge_edge('concept_Marketing_ROI', 'data_revenue', 'operationalized_by')
kg.add_bridge_edge('concept_Customer_Engagement', 'data_clicks', 'operationalized_by')

# Query the graph
path = kg.find_path('data_ad_spend', 'data_revenue')
print(f"Causal chain: {path}")
# Output: ['data_ad_spend', 'data_clicks', 'data_revenue']
```

### Example 2: Concept Hierarchy

```python
kg = KnowledgeGraph()

# Build hierarchy
kg.add_node('concept', 'concept_Satisfaction', {'name': 'Customer Satisfaction'})
kg.add_node('concept', 'concept_Loyalty', {'name': 'Brand Loyalty'})
kg.add_node('concept', 'concept_Trust', {'name': 'Trust'})
kg.add_node('concept', 'concept_Quality', {'name': 'Service Quality'})

# Part-of relationships
kg.add_edge('concept_Loyalty', 'concept_Satisfaction', 'part_of', 'concept')
kg.add_edge('concept_Trust', 'concept_Satisfaction', 'part_of', 'concept')

# Prerequisites
kg.add_edge('concept_Quality', 'concept_Trust', 'prerequisite_for', 'concept')
kg.add_edge('concept_Trust', 'concept_Loyalty', 'prerequisite_for', 'concept')

# Query
components = kg.get_neighbors('concept_Satisfaction', direction='in')
print(f"Components of Satisfaction: {components}")
```

## Testing

Run the complete test suite:

```bash
python -m pytest tests/test_graph_utils.py -v
```

All 39 tests pass:
- 4 tests for canonical edge type structure
- 8 tests for node operations
- 10 tests for edge operations and validation
- 3 tests for bridge edges
- 5 tests for query operations
- 3 tests for validation
- 3 tests for serialization
- 3 tests for utility methods

## Demo

Run the interactive demo:

```bash
python demo_graph_utils.py
```

Shows 6 demonstrations:
1. Basic node and edge operations
2. Validation features (inverse duplicates, simple cycles)
3. Longer cycles (allowed)
4. Path finding (single and multiple paths)
5. Canonical edge type information
6. Serialization and deserialization

## API Reference

### KnowledgeGraph Class

#### Constructor
- `__init__()`: Initialize empty graph

#### Node Methods
- `add_node(layer, node_id, node_data)`: Add node to layer
- `get_node(node_id, layer=None)`: Get node by ID
- `update_node(node_id, updates)`: Update node attributes
- `remove_node(node_id)`: Remove node and connected edges
- `find_nodes(**filters)`: Find nodes matching filters

#### Edge Methods
- `add_edge(source, target, edge_type, layer, metadata=None)`: Add validated edge
- `add_bridge_edge(source, target, edge_type, metadata=None)`: Add bridge edge
- `get_edges(node_id=None, edge_type=None, layer=None)`: Query edges
- `get_neighbors(node_id, edge_type=None, direction='out')`: Get neighbor nodes

#### Query Methods
- `find_path(source, target, edge_type=None, max_depth=5)`: Find first path (BFS)
- `find_all_paths(source, target, edge_type=None, max_depth=5, max_paths=100)`: Find all paths (DFS)

#### Validation Methods
- `validate()`: Check graph integrity
- `_has_inverse_edge(source, target, edge_type, layer)`: Check for inverse duplicate (private)
- `_would_create_simple_cycle(source, target, edge_type, layer)`: Check for simple cycle (private)

#### Serialization Methods
- `to_json()`: Export to dictionary
- `from_json(json_data)`: Import from dictionary (class method)

#### Utility Methods
- `get_layer_stats()`: Get node/edge counts by layer
- `get_edge_type_info(edge_type, layer)`: Get edge type definition

## Design Decisions

### 1. Cycle Policy
**Only simple cycles (A→B→A) are prevented**, not longer cycles. This allows:
- Complex causal chains: `A → B → C → A`
- Feedback loops in systems modeling
- More flexible graph structures

### 2. Inverse Handling
**Self-inverse edges** (correlates_with, similar_to) prevent duplicates:
- `A correlates_with B` blocks `B correlates_with A`
- Maintains semantic correctness

**Inverse pairs** (part_of/has_part) also prevent duplicates:
- `A part_of B` blocks `B has_part A`
- Choose the semantically correct direction

### 3. Validation Strategy
- **At insertion time**: Validates edge types, node existence, cycles
- **On demand**: `validate()` checks graph integrity
- **Efficient**: Uses adjacency indices for O(1) neighbor lookup

### 4. Extensibility
- Easy to add new edge types to `CANONICAL_EDGE_TYPES`
- Layer structure supports future expansion
- Metadata fields allow custom attributes

## Performance

- **Node lookup**: O(1) with hash maps
- **Neighbor queries**: O(1) with adjacency indices
- **Path finding**: O(V + E) BFS/DFS
- **Validation**: O(E) edge checks

Adjacency indices are rebuilt on:
- Deserialization
- Node removal

## Next Steps

### Recommended Enhancements
1. **Add question layer edge types** (currently empty)
2. **Implement graph visualization** (NetworkX, Graphviz)
3. **Add batch operations** (bulk add/remove)
4. **Implement undo/redo** (command pattern)
5. **Add graph merging** (combine multiple graphs)
6. **Optimize for large graphs** (lazy loading, pagination)

### Integration Points
- Use with `data_summarizer.py` for variable extraction
- Use with LLM for automated edge suggestion
- Export to visualization tools
- Store in database (serialize to JSON)

## License

Part of the DAG knowledge graph project.

---

**Status**: ✅ Fully implemented and tested
**Version**: 1.0
**Last Updated**: 2025-12-05
