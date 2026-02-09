"""
Comprehensive test suite for graph_utils.py

Tests the canonical edge type system and KnowledgeGraph class.
"""

import pytest
import json
from src.graph_utils import KnowledgeGraph, CANONICAL_EDGE_TYPES


class TestCanonicalEdgeTypes:
    """Test suite for CANONICAL_EDGE_TYPES structure."""

    def test_canonical_edge_types_structure(self):
        """Verify CANONICAL_EDGE_TYPES is well-formed."""
        # Check all layers exist
        assert 'data' in CANONICAL_EDGE_TYPES
        assert 'concept' in CANONICAL_EDGE_TYPES
        assert 'bridge' in CANONICAL_EDGE_TYPES
        assert 'question' in CANONICAL_EDGE_TYPES

        # Check each layer has proper edge types
        for layer_name, edge_types in CANONICAL_EDGE_TYPES.items():
            for edge_type, properties in edge_types.items():
                # Each edge type must have these fields
                assert 'definition' in properties, f"{layer_name}.{edge_type} missing 'definition'"
                assert 'directional' in properties, f"{layer_name}.{edge_type} missing 'directional'"
                assert 'inverse' in properties, f"{layer_name}.{edge_type} missing 'inverse'"
                assert 'examples' in properties, f"{layer_name}.{edge_type} missing 'examples'"

                # Validate types
                assert isinstance(properties['definition'], str)
                assert isinstance(properties['directional'], bool)
                assert properties['inverse'] is None or isinstance(properties['inverse'], str)
                assert isinstance(properties['examples'], list)

    def test_data_layer_edge_types(self):
        """Test data layer has required edge types."""
        data_types = CANONICAL_EDGE_TYPES['data']
        assert 'causal_influence' in data_types
        assert 'correlates_with' in data_types
        assert 'computed_from' in data_types

        # Test properties
        assert data_types['causal_influence']['directional'] is True
        assert data_types['causal_influence']['inverse'] is None

        assert data_types['correlates_with']['directional'] is False
        assert data_types['correlates_with']['inverse'] == 'correlates_with'  # Self-inverse

        assert data_types['computed_from']['directional'] is True
        assert data_types['computed_from']['inverse'] is None

    def test_concept_layer_edge_types(self):
        """Test concept layer has required edge types."""
        concept_types = CANONICAL_EDGE_TYPES['concept']
        assert 'part_of' in concept_types
        assert 'has_part' in concept_types
        assert 'contrasts_with' in concept_types
        assert 'similar_to' in concept_types
        assert 'prerequisite_for' in concept_types
        assert 'requires' in concept_types
        assert 'measured_by_multiple' in concept_types

        # Test inverse relationships
        assert concept_types['part_of']['inverse'] == 'has_part'
        assert concept_types['has_part']['inverse'] == 'part_of'
        assert concept_types['prerequisite_for']['inverse'] == 'requires'
        assert concept_types['requires']['inverse'] == 'prerequisite_for'

    def test_bridge_layer_edge_types(self):
        """Test bridge layer has required edge types."""
        bridge_types = CANONICAL_EDGE_TYPES['bridge']
        assert 'operationalized_by' in bridge_types
        assert bridge_types['operationalized_by']['directional'] is True
        assert bridge_types['operationalized_by']['inverse'] is None


class TestNodeOperations:
    """Test suite for node operations."""

    def test_add_nodes(self):
        """Test adding nodes to different layers."""
        kg = KnowledgeGraph()

        # Add data node
        kg.add_node('data', 'data_revenue', {
            'node_type': 'raw_variable',
            'name': 'revenue',
            'data_type': 'float'
        })

        # Add concept node
        kg.add_node('concept', 'concept_Revenue', {
            'node_type': 'data_concept',
            'name': 'Revenue',
            'activated': False
        })

        # Verify nodes exist
        assert kg.get_node('data_revenue')
        assert kg.get_node('concept_Revenue')

        # Check layer assignment
        assert kg.get_node('data_revenue')['layer'] == 'data'
        assert kg.get_node('concept_Revenue')['layer'] == 'concept'

    def test_add_node_duplicate(self):
        """Test that adding duplicate node raises error."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_revenue', {'name': 'revenue'})

        with pytest.raises(ValueError, match="already exists"):
            kg.add_node('data', 'data_revenue', {'name': 'revenue2'})

    def test_add_node_invalid_layer(self):
        """Test that invalid layer raises error."""
        kg = KnowledgeGraph()

        with pytest.raises(ValueError, match="Invalid layer"):
            kg.add_node('invalid_layer', 'node1', {})

    def test_get_node_by_layer(self):
        """Test getting node with layer specified."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_revenue', {'name': 'revenue'})

        # Should find in correct layer
        assert kg.get_node('data_revenue', layer='data') is not None

        # Should not find in wrong layer
        assert kg.get_node('data_revenue', layer='concept') is None

    def test_update_node(self):
        """Test updating node attributes."""
        kg = KnowledgeGraph()
        kg.add_node('concept', 'concept_Revenue', {'activated': False})

        kg.update_node('concept_Revenue', {'activated': True, 'new_field': 'value'})

        node = kg.get_node('concept_Revenue')
        assert node['activated'] is True
        assert node['new_field'] == 'value'

    def test_update_node_not_found(self):
        """Test updating non-existent node raises error."""
        kg = KnowledgeGraph()

        with pytest.raises(KeyError, match="not found"):
            kg.update_node('nonexistent', {'field': 'value'})

    def test_remove_node(self):
        """Test removing node and its edges."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_budget', {'name': 'budget'})
        kg.add_node('data', 'data_revenue', {'name': 'revenue'})
        kg.add_edge('data_budget', 'data_revenue', 'causal_influence', 'data')

        kg.remove_node('data_budget')

        # Node should be gone
        assert kg.get_node('data_budget') is None

        # Edge should be removed
        edges = kg.get_edges(layer='data')
        assert len(edges) == 0

    def test_find_nodes(self):
        """Test finding nodes with filters."""
        kg = KnowledgeGraph()
        kg.add_node('concept', 'concept_Revenue', {'activated': True, 'category': 'financial'})
        kg.add_node('concept', 'concept_Profit', {'activated': False, 'category': 'financial'})
        kg.add_node('concept', 'concept_Loyalty', {'activated': True, 'category': 'behavioral'})

        # Find by activation status
        active_nodes = kg.find_nodes(activated=True)
        assert len(active_nodes) == 2

        # Find by category
        financial_nodes = kg.find_nodes(category='financial')
        assert len(financial_nodes) == 2

        # Find by multiple filters
        active_financial = kg.find_nodes(activated=True, category='financial')
        assert len(active_financial) == 1
        assert active_financial[0]['node_id'] == 'concept_Revenue'


class TestEdgeOperations:
    """Test suite for edge operations with validation."""

    def test_add_edges_valid(self):
        """Test adding valid edges."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_budget', {'name': 'budget'})
        kg.add_node('data', 'data_revenue', {'name': 'revenue'})

        # Add valid edge
        kg.add_edge('data_budget', 'data_revenue', 'causal_influence', 'data')

        edges = kg.get_edges(layer='data')
        assert len(edges) == 1
        assert edges[0]['source'] == 'data_budget'
        assert edges[0]['target'] == 'data_revenue'
        assert edges[0]['edge_type'] == 'causal_influence'

    def test_add_edge_invalid_type(self):
        """Test that non-canonical edge type raises error."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_budget', {'name': 'budget'})
        kg.add_node('data', 'data_revenue', {'name': 'revenue'})

        with pytest.raises(ValueError, match="Invalid edge type"):
            kg.add_edge('data_budget', 'data_revenue', 'invalid_type', 'data')

    def test_add_edge_nodes_not_found(self):
        """Test that missing nodes raise errors."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_budget', {'name': 'budget'})

        with pytest.raises(KeyError, match="not found"):
            kg.add_edge('data_budget', 'nonexistent', 'causal_influence', 'data')

    def test_add_edge_with_metadata(self):
        """Test adding edge with metadata."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_budget', {'name': 'budget'})
        kg.add_node('data', 'data_revenue', {'name': 'revenue'})

        kg.add_edge('data_budget', 'data_revenue', 'causal_influence', 'data',
                   metadata={'confidence': 0.95, 'source': 'expert'})

        edges = kg.get_edges(layer='data')
        assert edges[0]['metadata']['confidence'] == 0.95
        assert edges[0]['metadata']['source'] == 'expert'

    def test_inverse_duplicate_prevention(self):
        """Test prevention of inverse duplicate edges."""
        kg = KnowledgeGraph()
        kg.add_node('concept', 'concept_Loyalty', {'name': 'Loyalty'})
        kg.add_node('concept', 'concept_Satisfaction', {'name': 'Satisfaction'})

        # Add part_of edge
        kg.add_edge('concept_Loyalty', 'concept_Satisfaction', 'part_of', 'concept')

        # Try to add inverse (should fail)
        with pytest.raises(ValueError, match="Inverse edge already exists"):
            kg.add_edge('concept_Satisfaction', 'concept_Loyalty', 'has_part', 'concept')

    def test_simple_cycle_prevention(self):
        """Test prevention of simple cycles (A→B→A with same edge type)."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_A', {'name': 'A'})
        kg.add_node('data', 'data_B', {'name': 'B'})

        # Add A → B
        kg.add_edge('data_A', 'data_B', 'causal_influence', 'data')

        # Try to add B → A with same edge type (should fail)
        with pytest.raises(ValueError, match="simple cycle"):
            kg.add_edge('data_B', 'data_A', 'causal_influence', 'data')

    def test_longer_cycles_allowed(self):
        """Test that longer cycles (A→B→C→A) are allowed."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_A', {'name': 'A'})
        kg.add_node('data', 'data_B', {'name': 'B'})
        kg.add_node('data', 'data_C', {'name': 'C'})

        # Create A → B → C → A (should succeed)
        kg.add_edge('data_A', 'data_B', 'causal_influence', 'data')
        kg.add_edge('data_B', 'data_C', 'causal_influence', 'data')
        kg.add_edge('data_C', 'data_A', 'causal_influence', 'data')

        # Verify all edges exist
        edges = kg.get_edges(layer='data')
        assert len(edges) == 3

    def test_symmetric_edges(self):
        """Test symmetric edges (correlates_with, similar_to)."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_ice_cream', {'name': 'ice_cream_sales'})
        kg.add_node('data', 'data_drowning', {'name': 'drowning_incidents'})

        # Add symmetric edge
        kg.add_edge('data_ice_cream', 'data_drowning', 'correlates_with', 'data')

        # Try to add reverse (should fail as it's the inverse)
        with pytest.raises(ValueError, match="Inverse edge already exists"):
            kg.add_edge('data_drowning', 'data_ice_cream', 'correlates_with', 'data')

    def test_get_edges_with_filters(self):
        """Test getting edges with various filters."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_A', {'name': 'A'})
        kg.add_node('data', 'data_B', {'name': 'B'})
        kg.add_node('data', 'data_C', {'name': 'C'})

        kg.add_edge('data_A', 'data_B', 'causal_influence', 'data')
        kg.add_edge('data_B', 'data_C', 'computed_from', 'data')
        kg.add_edge('data_A', 'data_C', 'correlates_with', 'data')

        # Filter by node
        edges_with_A = kg.get_edges(node_id='data_A')
        assert len(edges_with_A) == 2

        # Filter by edge type
        causal_edges = kg.get_edges(edge_type='causal_influence')
        assert len(causal_edges) == 1

        # Filter by layer
        data_edges = kg.get_edges(layer='data')
        assert len(data_edges) == 3

    def test_get_neighbors(self):
        """Test getting neighbors in different directions."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_A', {'name': 'A'})
        kg.add_node('data', 'data_B', {'name': 'B'})
        kg.add_node('data', 'data_C', {'name': 'C'})

        kg.add_edge('data_A', 'data_B', 'causal_influence', 'data')
        kg.add_edge('data_A', 'data_C', 'causal_influence', 'data')
        kg.add_edge('data_C', 'data_A', 'computed_from', 'data')

        # Outgoing neighbors
        out_neighbors = kg.get_neighbors('data_A', direction='out')
        assert set(out_neighbors) == {'data_B', 'data_C'}

        # Incoming neighbors
        in_neighbors = kg.get_neighbors('data_A', direction='in')
        assert in_neighbors == ['data_C']

        # Both directions
        all_neighbors = kg.get_neighbors('data_A', direction='both')
        assert len(all_neighbors) == 3

        # Filter by edge type
        causal_neighbors = kg.get_neighbors('data_A', edge_type='causal_influence', direction='out')
        assert set(causal_neighbors) == {'data_B', 'data_C'}


class TestBridgeEdges:
    """Test suite for bridge edges between layers."""

    def test_add_bridge_edge(self):
        """Test adding bridge edges."""
        kg = KnowledgeGraph()
        kg.add_node('concept', 'concept_Revenue', {'name': 'Revenue'})
        kg.add_node('data', 'data_revenue', {'name': 'revenue'})

        kg.add_bridge_edge('concept_Revenue', 'data_revenue', 'operationalized_by')

        assert len(kg.bridge_edges) == 1
        assert kg.bridge_edges[0]['source'] == 'concept_Revenue'
        assert kg.bridge_edges[0]['target'] == 'data_revenue'

    def test_add_bridge_edge_invalid_type(self):
        """Test that invalid bridge edge type raises error."""
        kg = KnowledgeGraph()
        kg.add_node('concept', 'concept_Revenue', {'name': 'Revenue'})
        kg.add_node('data', 'data_revenue', {'name': 'revenue'})

        with pytest.raises(ValueError, match="Invalid bridge edge type"):
            kg.add_bridge_edge('concept_Revenue', 'data_revenue', 'invalid_type')

    def test_query_bridge_edges(self):
        """Test querying bridge edges."""
        kg = KnowledgeGraph()
        kg.add_node('concept', 'concept_Revenue', {'name': 'Revenue'})
        kg.add_node('data', 'data_revenue', {'name': 'revenue'})
        kg.add_bridge_edge('concept_Revenue', 'data_revenue', 'operationalized_by')

        # Get all edges including bridge
        all_edges = kg.get_edges()
        assert len(all_edges) == 1
        assert all_edges[0]['layer'] == 'bridge'

        # Get only bridge edges
        bridge_edges = kg.get_edges(layer='bridge')
        assert len(bridge_edges) == 1


class TestQueryOperations:
    """Test suite for path finding and query operations."""

    def test_find_path(self):
        """Test finding a path between nodes."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_A', {'name': 'A'})
        kg.add_node('data', 'data_B', {'name': 'B'})
        kg.add_node('data', 'data_C', {'name': 'C'})

        kg.add_edge('data_A', 'data_B', 'causal_influence', 'data')
        kg.add_edge('data_B', 'data_C', 'causal_influence', 'data')

        path = kg.find_path('data_A', 'data_C')
        assert path == ['data_A', 'data_B', 'data_C']

    def test_find_path_no_path(self):
        """Test when no path exists."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_A', {'name': 'A'})
        kg.add_node('data', 'data_B', {'name': 'B'})

        path = kg.find_path('data_A', 'data_B')
        assert path is None

    def test_find_path_with_edge_type_filter(self):
        """Test path finding with edge type filter."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_A', {'name': 'A'})
        kg.add_node('data', 'data_B', {'name': 'B'})
        kg.add_node('data', 'data_C', {'name': 'C'})

        kg.add_edge('data_A', 'data_B', 'causal_influence', 'data')
        kg.add_edge('data_B', 'data_C', 'computed_from', 'data')

        # Path exists with causal_influence
        path1 = kg.find_path('data_A', 'data_B', edge_type='causal_influence')
        assert path1 == ['data_A', 'data_B']

        # No path exists using only causal_influence to C
        path2 = kg.find_path('data_A', 'data_C', edge_type='causal_influence')
        assert path2 is None

    def test_find_all_paths(self):
        """Test finding all paths between nodes."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_A', {'name': 'A'})
        kg.add_node('data', 'data_B', {'name': 'B'})
        kg.add_node('data', 'data_C', {'name': 'C'})
        kg.add_node('data', 'data_D', {'name': 'D'})

        # Create multiple paths from A to D
        # Path 1: A → B → D
        # Path 2: A → C → D
        kg.add_edge('data_A', 'data_B', 'causal_influence', 'data')
        kg.add_edge('data_A', 'data_C', 'causal_influence', 'data')
        kg.add_edge('data_B', 'data_D', 'causal_influence', 'data')
        kg.add_edge('data_C', 'data_D', 'causal_influence', 'data')

        paths = kg.find_all_paths('data_A', 'data_D')
        assert len(paths) == 2
        assert ['data_A', 'data_B', 'data_D'] in paths
        assert ['data_A', 'data_C', 'data_D'] in paths

    def test_find_all_paths_max_depth(self):
        """Test that max_depth limits path length."""
        kg = KnowledgeGraph()
        for i in range(5):
            kg.add_node('data', f'data_{i}', {'name': f'{i}'})

        # Create chain: 0 → 1 → 2 → 3 → 4
        for i in range(4):
            kg.add_edge(f'data_{i}', f'data_{i+1}', 'causal_influence', 'data')

        # With max_depth=2, should not find path to node 4
        paths = kg.find_all_paths('data_0', 'data_4', max_depth=2)
        assert len(paths) == 0

        # With max_depth=5, should find path
        paths = kg.find_all_paths('data_0', 'data_4', max_depth=5)
        assert len(paths) == 1


class TestValidation:
    """Test suite for graph validation."""

    def test_validate_valid_graph(self):
        """Test validation of a valid graph."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_revenue', {'name': 'revenue'})
        kg.add_node('data', 'data_profit', {'name': 'profit'})
        kg.add_edge('data_revenue', 'data_profit', 'causal_influence', 'data')

        result = kg.validate()
        assert result['valid'] is True
        assert len(result['errors']) == 0

    def test_validate_dangling_edge(self):
        """Test validation catches dangling edges."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_revenue', {'name': 'revenue'})

        # Manually add invalid edge
        kg.layers['data']['edges'].append({
            'source': 'data_revenue',
            'target': 'nonexistent',
            'edge_type': 'causal_influence',
            'metadata': {}
        })

        result = kg.validate()
        assert result['valid'] is False
        assert len(result['errors']) > 0
        assert 'not found' in result['errors'][0]

    def test_validate_non_canonical_edge(self):
        """Test validation catches non-canonical edge types."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_revenue', {'name': 'revenue'})
        kg.add_node('data', 'data_profit', {'name': 'profit'})

        # Manually add invalid edge type
        kg.layers['data']['edges'].append({
            'source': 'data_revenue',
            'target': 'data_profit',
            'edge_type': 'invalid_type',
            'metadata': {}
        })

        result = kg.validate()
        assert result['valid'] is False
        assert any('Non-canonical' in error for error in result['errors'])


class TestSerialization:
    """Test suite for JSON serialization."""

    def test_to_json(self):
        """Test exporting graph to JSON."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_revenue', {'name': 'revenue'})
        kg.add_node('data', 'data_profit', {'name': 'profit'})
        kg.add_edge('data_revenue', 'data_profit', 'causal_influence', 'data')

        json_data = kg.to_json()

        assert 'layers' in json_data
        assert 'bridge_edges' in json_data
        assert 'metadata' in json_data
        assert 'data_revenue' in json_data['layers']['data']['nodes']

    def test_from_json(self):
        """Test importing graph from JSON."""
        kg1 = KnowledgeGraph()
        kg1.add_node('data', 'data_revenue', {'name': 'revenue'})
        kg1.add_node('data', 'data_profit', {'name': 'profit'})
        kg1.add_edge('data_revenue', 'data_profit', 'causal_influence', 'data')

        # Export and reimport
        json_data = kg1.to_json()
        kg2 = KnowledgeGraph.from_json(json_data)

        # Verify structure preserved
        assert kg2.get_node('data_revenue') is not None
        assert kg2.get_node('data_profit') is not None
        edges = kg2.get_edges(layer='data')
        assert len(edges) == 1

    def test_serialization_roundtrip(self):
        """Test that serialization roundtrip preserves data."""
        kg1 = KnowledgeGraph()
        kg1.add_node('data', 'data_revenue', {'name': 'revenue', 'value': 100})
        kg1.add_node('concept', 'concept_Revenue', {'name': 'Revenue'})
        kg1.add_edge('data_revenue', 'data_revenue', 'correlates_with', 'data')
        kg1.add_bridge_edge('concept_Revenue', 'data_revenue', 'operationalized_by')
        kg1.metadata['version'] = '1.0'

        # Roundtrip
        json_str = json.dumps(kg1.to_json())
        json_data = json.loads(json_str)
        kg2 = KnowledgeGraph.from_json(json_data)

        # Verify everything preserved
        assert kg2.get_node('data_revenue')['value'] == 100
        assert kg2.metadata['version'] == '1.0'
        assert len(kg2.bridge_edges) == 1


class TestUtilityMethods:
    """Test suite for utility methods."""

    def test_get_layer_stats(self):
        """Test getting layer statistics."""
        kg = KnowledgeGraph()
        kg.add_node('data', 'data_A', {'name': 'A'})
        kg.add_node('data', 'data_B', {'name': 'B'})
        kg.add_node('concept', 'concept_C', {'name': 'C'})

        kg.add_edge('data_A', 'data_B', 'causal_influence', 'data')
        kg.add_bridge_edge('concept_C', 'data_A', 'operationalized_by')

        stats = kg.get_layer_stats()

        assert stats['data']['node_count'] == 2
        assert stats['data']['edge_count'] == 1
        assert stats['concept']['node_count'] == 1
        assert stats['bridge']['edge_count'] == 1

    def test_get_edge_type_info(self):
        """Test getting edge type information."""
        kg = KnowledgeGraph()

        info = kg.get_edge_type_info('causal_influence', 'data')

        assert 'definition' in info
        assert 'directional' in info
        assert 'inverse' in info
        assert 'examples' in info
        assert info['directional'] is True

    def test_get_edge_type_info_invalid(self):
        """Test that invalid edge type raises error."""
        kg = KnowledgeGraph()

        with pytest.raises(KeyError):
            kg.get_edge_type_info('invalid_type', 'data')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
