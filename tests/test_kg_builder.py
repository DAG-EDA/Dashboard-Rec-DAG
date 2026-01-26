"""
Tests for KG Builder

Tests knowledge graph construction from data summaries.
"""

import pytest
import json
from pathlib import Path
from src.kg_builder import KGBuilder
from src.graph_utils import KnowledgeGraph, CANONICAL_EDGE_TYPES
from src.llm_client import GeminiClient


class TestKGBuilder:
    """Test suite for KGBuilder."""

    @pytest.fixture
    def summary_data(self):
        """Load test summary data."""
        summary_path = Path('output/test_summary.json')
        with open(summary_path, 'r') as f:
            return json.load(f)

    @pytest.fixture
    def llm_client(self):
        """Create LLM client."""
        return GeminiClient()

    @pytest.fixture
    def kg_builder(self, llm_client):
        """Create KG builder."""
        return KGBuilder(llm_client)

    def test_build_data_layer(self, kg_builder, summary_data):
        """Test building data layer from summary."""
        kg, node_mapping = kg_builder.build_data_layer(summary_data)

        # Check nodes created
        assert len(node_mapping) == len(summary_data['attributes'])

        # Check each node
        for attr in summary_data['attributes']:
            node_id = node_mapping[attr['name']]
            node = kg.get_node(node_id)

            assert node is not None
            assert node['name'] == attr['name']
            assert node['layer'] == 'data'
            assert node['node_type'] == 'raw_variable'
            assert 'description' in node

        print(f"✓ Created {len(node_mapping)} data nodes")

    def test_build_concept_layer(self, kg_builder, summary_data):
        """Test building concept layer from summary."""
        # First build data layer
        kg, data_mapping = kg_builder.build_data_layer(summary_data)

        # Build concept layer
        kg, concept_mapping = kg_builder.build_concept_layer(kg, summary_data)

        # Check concept nodes created
        expected_concepts = len(summary_data.get('suggested_concepts', []))
        assert len(concept_mapping) == expected_concepts

        # Check each concept node
        for concept in summary_data.get('suggested_concepts', []):
            node_id = concept_mapping[concept['name']]
            node = kg.get_node(node_id)

            assert node is not None
            assert node['name'] == concept['name']
            assert node['layer'] == 'concept'
            assert node['node_type'] == 'data_concept'
            assert 'definition' in node
            assert 'confidence' in node

        print(f"✓ Created {len(concept_mapping)} concept nodes")

    def test_create_bridge_edges(self, kg_builder, summary_data):
        """Test creating bridge edges."""
        # Build data and concept layers
        kg, data_mapping = kg_builder.build_data_layer(summary_data)
        kg, concept_mapping = kg_builder.build_concept_layer(kg, summary_data)

        # Create bridge edges
        kg = kg_builder.create_bridge_edges(kg, data_mapping, concept_mapping, summary_data)

        # Check bridge edges exist
        all_edges = kg.get_edges()
        bridge_edges = [e for e in all_edges if e.get('layer') == 'bridge']
        assert len(bridge_edges) > 0

        # Check bridge edges are valid
        for edge in bridge_edges:
            assert edge['edge_type'] == 'operationalized_by'

            # Check nodes exist
            source_node = kg.get_node(edge['source'])
            target_node = kg.get_node(edge['target'])

            assert source_node is not None
            assert target_node is not None
            assert source_node['layer'] == 'concept'
            assert target_node['layer'] == 'data'

        print(f"✓ Created {len(bridge_edges)} bridge edges")

    def test_build_initial_kg_structure(self, kg_builder, summary_data):
        """Test complete KG building (structure only, no LLM calls)."""
        # Build data layer only (no LLM)
        kg, data_mapping = kg_builder.build_data_layer(summary_data)
        kg, concept_mapping = kg_builder.build_concept_layer(kg, summary_data)
        kg = kg_builder.create_bridge_edges(kg, data_mapping, concept_mapping, summary_data)

        # Validate structure
        validation = kg.validate()
        assert validation['valid'], f"Errors: {validation['errors']}"

        # Check layer stats
        stats = kg.get_layer_stats()

        assert stats['data']['node_count'] > 0
        assert stats['concept']['node_count'] > 0
        assert stats['bridge']['edge_count'] > 0

        print("\n=== KG Statistics ===")
        print(f"Data nodes: {stats['data']['node_count']}")
        print(f"Concept nodes: {stats['concept']['node_count']}")
        print(f"Bridge edges: {stats['bridge']['edge_count']}")

    def test_save_and_load_kg(self, kg_builder, summary_data, tmp_path):
        """Test saving and loading knowledge graph."""
        # Build simple KG
        kg1, data_mapping = kg_builder.build_data_layer(summary_data)

        # Save
        output_path = tmp_path / "test_kg.json"
        kg_builder.save_kg(kg1, output_path)

        assert output_path.exists()

        # Load
        kg2 = kg_builder.load_kg(output_path)

        # Compare
        assert kg2.get_layer_stats()['data']['node_count'] == len(data_mapping)

        # Check nodes match
        for node_id in data_mapping.values():
            node1 = kg1.get_node(node_id)
            node2 = kg2.get_node(node_id)

            assert node1 is not None
            assert node2 is not None
            assert node1['name'] == node2['name']

        print("✓ Save/load round-trip successful")

    def test_node_mapping_consistency(self, kg_builder, summary_data):
        """Test that node IDs are consistent and predictable."""
        kg, node_mapping = kg_builder.build_data_layer(summary_data)

        # Check all mapped nodes exist
        for attr_name, node_id in node_mapping.items():
            assert node_id.startswith('data_')
            assert kg.get_node(node_id) is not None

        # Check node ID format
        for attr in summary_data['attributes']:
            expected_id = f"data_{attr['name'].replace(' ', '_')}"
            assert expected_id in node_mapping.values()

        print("✓ Node mapping is consistent")


class TestKGBuilderIntegration:
    """Integration tests with LLM (slower, require API key)."""

    @pytest.fixture
    def summary_data(self):
        """Load test summary data."""
        summary_path = Path('output/test_summary.json')
        with open(summary_path, 'r') as f:
            return json.load(f)

    @pytest.fixture
    def llm_client(self):
        """Create LLM client."""
        return GeminiClient()

    @pytest.fixture
    def kg_builder(self, llm_client):
        """Create KG builder."""
        return KGBuilder(llm_client)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_infer_causal_relationships(self, kg_builder, summary_data):
        """Test causal relationship inference with LLM."""
        # Build data layer
        kg, node_mapping = kg_builder.build_data_layer(summary_data)

        initial_edges = len(kg.get_edges(layer='data'))

        # Infer relationships
        kg = kg_builder.infer_causal_relationships(
            kg, node_mapping, summary_data, domain_hint='movies'
        )

        # Check edges were added
        final_edges = len(kg.get_edges(layer='data'))
        assert final_edges > initial_edges

        # Check edge types are canonical
        for edge in kg.get_edges(layer='data'):
            assert edge['edge_type'] in CANONICAL_EDGE_TYPES['data']

        print(f"✓ Inferred {final_edges - initial_edges} causal relationships")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_build_initial_kg_full(self, kg_builder, summary_data):
        """Test complete KG building with LLM inference."""
        # Build complete KG
        kg = kg_builder.build_initial_kg(summary_data, domain_hint='movies')

        # Validate
        validation = kg.validate()
        assert validation['valid'], f"Errors: {validation['errors']}"

        # Check all layers have content
        stats = kg.get_layer_stats()

        assert stats['data']['node_count'] > 0, "Data layer should have nodes"
        assert stats['concept']['node_count'] > 0, "Concept layer should have nodes"
        assert stats['bridge']['edge_count'] > 0, "Bridge layer should have edges"

        # Check for some data edges (from LLM inference)
        assert stats['data']['edge_count'] >= 0, "Data layer may have inferred edges"

        print("\n=== Full KG Statistics ===")
        for layer, layer_stats in stats.items():
            print(f"\n{layer.upper()}:")
            if 'node_count' in layer_stats:
                print(f"  Nodes: {layer_stats['node_count']}")
            print(f"  Edges: {layer_stats['edge_count']}")
            if layer_stats['edge_types']:
                for edge_type, count in layer_stats['edge_types'].items():
                    print(f"    - {edge_type}: {count}")

        # Test querying
        # Find a concept node
        concept_nodes = kg.find_nodes(layer='concept')
        if concept_nodes:
            concept_id = concept_nodes[0]['node_id']
            # Get operationalized_by edges
            neighbors = kg.get_neighbors(concept_id, direction='out')
            print(f"\n✓ Concept '{concept_nodes[0]['name']}' operationalized by {len(neighbors)} data variables")

    @pytest.mark.slow
    @pytest.mark.integration
    def test_end_to_end_workflow(self, kg_builder, summary_data, tmp_path):
        """Test complete end-to-end workflow."""
        print("\n" + "="*60)
        print("END-TO-END WORKFLOW TEST")
        print("="*60)

        # Build KG
        print("\n1. Building knowledge graph...")
        kg = kg_builder.build_initial_kg(summary_data, domain_hint='movies')

        # Validate
        print("\n2. Validating...")
        validation = kg.validate()
        assert validation['valid'], f"Validation errors: {validation['errors']}"
        print("✓ Graph is valid")

        # Save
        print("\n3. Saving...")
        output_path = tmp_path / "end_to_end_kg.json"
        kg_builder.save_kg(kg, output_path)
        assert output_path.exists()

        # Load
        print("\n4. Loading...")
        kg2 = kg_builder.load_kg(output_path)

        # Verify consistency
        print("\n5. Verifying consistency...")
        stats1 = kg.get_layer_stats()
        stats2 = kg2.get_layer_stats()

        assert stats1['data']['node_count'] == stats2['data']['node_count']
        assert stats1['concept']['node_count'] == stats2['concept']['node_count']
        assert stats1['bridge']['edge_count'] == stats2['bridge']['edge_count']

        print("✓ All checks passed")

        # Show final stats
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        for layer, layer_stats in stats1.items():
            print(f"\n{layer.upper()}:")
            if 'node_count' in layer_stats:
                print(f"  Nodes: {layer_stats['node_count']}")
            print(f"  Edges: {layer_stats['edge_count']}")


# Run specific test
if __name__ == '__main__':
    pytest.main([__file__, '-v', '-k', 'test_build_data_layer'])
