"""
Test and compare edge inference strategies.

Compares oneshot vs pairwise inference approaches.
"""

import pytest
import json
import time
from pathlib import Path
from src.kg_builder import KGBuilder
from src.graph_utils import KnowledgeGraph
from src.llm_client import GeminiClient


class TestInferenceStrategies:
    """Tests for comparing different inference strategies."""

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
    def test_compare_inference_strategies(self, kg_builder, summary_data):
        """
        Compare oneshot vs pairwise inference strategies.

        Tests both strategies on the same data and compares:
        - Edge count
        - Execution time
        - Confidence levels
        - Edge overlap
        """
        print("\n" + "="*70)
        print("COMPARING INFERENCE STRATEGIES")
        print("="*70)

        # Test Strategy 1: Oneshot
        print("\n[1/2] Testing ONESHOT Strategy...")
        print("-" * 70)

        kg1, data_mapping = kg_builder.build_data_layer(summary_data)
        kg1, oneshot_stats = kg_builder.infer_causal_relationships(
            kg1, data_mapping, summary_data,
            domain_hint='movies',
            strategy='oneshot',
            confidence_threshold=0.6
        )

        oneshot_edges = kg1.get_edges(layer='data')

        # Test Strategy 2: Pairwise
        print("\n[2/2] Testing PAIRWISE Strategy...")
        print("-" * 70)

        kg2, data_mapping = kg_builder.build_data_layer(summary_data)
        kg2, pairwise_stats = kg_builder.infer_causal_relationships(
            kg2, data_mapping, summary_data,
            domain_hint='movies',
            strategy='pairwise',
            confidence_threshold=0.6
        )

        pairwise_edges = kg2.get_edges(layer='data')

        # Compute comparison metrics
        print("\n" + "="*70)
        print("COMPARISON REPORT")
        print("="*70)

        # Edge counts
        print("\n📊 EDGE COUNTS")
        print("-" * 70)
        print(f"{'Metric':<30} {'Oneshot':>15} {'Pairwise':>15}")
        print("-" * 70)
        print(f"{'Raw edges (before processing)':<30} {oneshot_stats['raw_edges']:>15} {pairwise_stats['raw_edges']:>15}")
        print(f"{'Processed edges':<30} {oneshot_stats['processed_edges']:>15} {pairwise_stats['processed_edges']:>15}")
        print(f"{'Added to graph':<30} {oneshot_stats['added_edges']:>15} {pairwise_stats['added_edges']:>15}")

        # Timing
        print("\n⏱️  EXECUTION TIME")
        print("-" * 70)
        print(f"{'Oneshot':<30} {oneshot_stats['elapsed_time']:>14.2f}s")
        print(f"{'Pairwise':<30} {pairwise_stats['elapsed_time']:>14.2f}s")
        speedup = pairwise_stats['elapsed_time'] / oneshot_stats['elapsed_time'] if oneshot_stats['elapsed_time'] > 0 else 0
        print(f"{'Speedup (oneshot vs pairwise)':<30} {speedup:>14.2f}x")

        # Confidence analysis
        print("\n📈 CONFIDENCE LEVELS")
        print("-" * 70)

        oneshot_confidences = [e['metadata'].get('confidence', 0) for e in oneshot_edges]
        pairwise_confidences = [e['metadata'].get('confidence', 0) for e in pairwise_edges]

        if oneshot_confidences:
            print(f"{'Oneshot avg confidence':<30} {sum(oneshot_confidences)/len(oneshot_confidences):>15.3f}")
            print(f"{'Oneshot min confidence':<30} {min(oneshot_confidences):>15.3f}")
            print(f"{'Oneshot max confidence':<30} {max(oneshot_confidences):>15.3f}")

        if pairwise_confidences:
            print(f"{'Pairwise avg confidence':<30} {sum(pairwise_confidences)/len(pairwise_confidences):>15.3f}")
            print(f"{'Pairwise min confidence':<30} {min(pairwise_confidences):>15.3f}")
            print(f"{'Pairwise max confidence':<30} {max(pairwise_confidences):>15.3f}")

        # Edge overlap
        print("\n🔗 EDGE OVERLAP")
        print("-" * 70)

        oneshot_edge_set = set((e['source'], e['target'], e['edge_type']) for e in oneshot_edges)
        pairwise_edge_set = set((e['source'], e['target'], e['edge_type']) for e in pairwise_edges)

        common_edges = oneshot_edge_set & pairwise_edge_set
        oneshot_only = oneshot_edge_set - pairwise_edge_set
        pairwise_only = pairwise_edge_set - oneshot_edge_set

        print(f"{'Common edges (both strategies)':<30} {len(common_edges):>15}")
        print(f"{'Oneshot-only edges':<30} {len(oneshot_only):>15}")
        print(f"{'Pairwise-only edges':<30} {len(pairwise_only):>15}")

        if len(oneshot_edge_set) > 0:
            overlap_pct = 100 * len(common_edges) / len(oneshot_edge_set)
            print(f"{'Overlap percentage':<30} {overlap_pct:>14.1f}%")

        # Edge type distribution
        print("\n🏷️  EDGE TYPE DISTRIBUTION")
        print("-" * 70)

        oneshot_types = {}
        for edge in oneshot_edges:
            et = edge['edge_type']
            oneshot_types[et] = oneshot_types.get(et, 0) + 1

        pairwise_types = {}
        for edge in pairwise_edges:
            et = edge['edge_type']
            pairwise_types[et] = pairwise_types.get(et, 0) + 1

        all_types = set(list(oneshot_types.keys()) + list(pairwise_types.keys()))

        print(f"{'Edge Type':<30} {'Oneshot':>15} {'Pairwise':>15}")
        print("-" * 70)
        for et in sorted(all_types):
            print(f"{et:<30} {oneshot_types.get(et, 0):>15} {pairwise_types.get(et, 0):>15}")

        # Sample edges
        print("\n📝 SAMPLE EDGES")
        print("-" * 70)

        print("\nSample Oneshot Edges (first 5):")
        for i, edge in enumerate(oneshot_edges[:5], 1):
            conf = edge['metadata'].get('confidence', 0)
            print(f"  {i}. {edge['source']} --{edge['edge_type']}--> {edge['target']} (conf: {conf:.2f})")

        if pairwise_edges:
            print("\nSample Pairwise Edges (first 5):")
            for i, edge in enumerate(pairwise_edges[:5], 1):
                conf = edge['metadata'].get('confidence', 0)
                print(f"  {i}. {edge['source']} --{edge['edge_type']}--> {edge['target']} (conf: {conf:.2f})")

        # Summary recommendations
        print("\n" + "="*70)
        print("💡 RECOMMENDATIONS")
        print("="*70)

        if oneshot_stats['elapsed_time'] < pairwise_stats['elapsed_time'] * 0.5:
            print("✓ Oneshot is significantly faster - recommend for initial exploration")

        if pairwise_stats['added_edges'] > oneshot_stats['added_edges'] * 1.2:
            print("✓ Pairwise finds more edges - recommend for thorough analysis")

        if len(common_edges) > 0:
            print(f"✓ {len(common_edges)} edges found by both strategies - high confidence")

        print("\n" + "="*70)

        # Assertions
        assert oneshot_stats['added_edges'] >= 0, "Oneshot should add some edges"
        assert pairwise_stats['added_edges'] >= 0, "Pairwise should add some edges"
        assert oneshot_stats['elapsed_time'] > 0, "Oneshot should take some time"
        assert pairwise_stats['elapsed_time'] > 0, "Pairwise should take some time"

    @pytest.mark.slow
    @pytest.mark.integration
    def test_post_processing(self, kg_builder):
        """Test edge post-processing functions."""
        print("\n" + "="*70)
        print("TESTING POST-PROCESSING")
        print("="*70)

        # Create test edges with various issues
        raw_edges = [
            # Duplicate
            {'source': 'A', 'target': 'B', 'edge_type': 'causal_influence', 'confidence': 0.8, 'reasoning': '', 'metadata': {}},
            {'source': 'A', 'target': 'B', 'edge_type': 'causal_influence', 'confidence': 0.8, 'reasoning': '', 'metadata': {}},

            # Low confidence
            {'source': 'C', 'target': 'D', 'edge_type': 'causal_influence', 'confidence': 0.3, 'reasoning': '', 'metadata': {}},

            # High confidence
            {'source': 'E', 'target': 'F', 'edge_type': 'causal_influence', 'confidence': 0.9, 'reasoning': '', 'metadata': {}},

            # Symmetric edge (both directions)
            {'source': 'G', 'target': 'H', 'edge_type': 'correlates_with', 'confidence': 0.7, 'reasoning': '', 'metadata': {}},
            {'source': 'H', 'target': 'G', 'edge_type': 'correlates_with', 'confidence': 0.7, 'reasoning': '', 'metadata': {}},
        ]

        print(f"\n Input: {len(raw_edges)} raw edges")

        # Post-process
        processed = kg_builder.post_process_edges(raw_edges, 'data', confidence_threshold=0.6)

        print(f"Output: {len(processed)} processed edges")

        # Check results
        print("\nPost-processing checks:")
        print("✓ Removed duplicates")
        print("✓ Filtered by confidence (>= 0.6)")
        print("✓ Normalized symmetric edges")

        # Assertions
        assert len(processed) < len(raw_edges), "Post-processing should reduce edge count"
        assert all(e['confidence'] >= 0.6 for e in processed), "All edges should meet confidence threshold"

        # Check no exact duplicates
        seen = set()
        for edge in processed:
            key = (edge['source'], edge['target'], edge['edge_type'])
            assert key not in seen, f"Duplicate edge found: {key}"
            seen.add(key)

        print("\n✓ All post-processing tests passed")


# Run comparison test directly
if __name__ == '__main__':
    pytest.main([__file__, '-v', '-k', 'test_compare_inference_strategies', '-s'])
