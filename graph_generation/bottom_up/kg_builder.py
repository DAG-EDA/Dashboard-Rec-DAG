"""
Knowledge Graph Builder

Constructs knowledge graphs from data summaries using the canonical edge type system.
"""

import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from src.graph_utils import KnowledgeGraph, CANONICAL_EDGE_TYPES
from src.llm_client import GeminiClient


class KGBuilder:
    """Builds knowledge graphs from data summaries."""

    def __init__(self, llm_client: GeminiClient):
        """
        Initialize KG builder.

        Args:
            llm_client: LLM client for inference tasks
        """
        self.llm = llm_client
        self.prompts_dir = Path(__file__).parent.parent / 'prompts'

    def build_data_layer(self, summary: Dict[str, Any]) -> Tuple[KnowledgeGraph, Dict[str, str]]:
        """
        Create data layer nodes from summary JSON.

        Args:
            summary: Dataset summary containing attributes

        Returns:
            Tuple of (KnowledgeGraph with data nodes, mapping of attribute names to node IDs)
        """
        kg = KnowledgeGraph()
        node_mapping = {}

        for attr in summary.get('attributes', []):
            # Create node ID (prefix with 'data_')
            node_id = f"data_{attr['name'].replace(' ', '_')}"
            node_mapping[attr['name']] = node_id

            # Prepare node data
            node_data = {
                'name': attr['name'],
                'node_type': 'raw_variable',
                'data_type': attr['type'],
                'role': attr['role'],
                'description': attr.get('description', ''),
                'stats': attr.get('stats', {}),
                'activated': False  # Not activated yet
            }

            # Add node to graph
            kg.add_node('data', node_id, node_data)

        print(f"✓ Created {len(node_mapping)} data layer nodes")
        return kg, node_mapping

    def infer_causal_relationships(
        self,
        kg: KnowledgeGraph,
        node_mapping: Dict[str, str],
        summary: Dict[str, Any],
        domain_hint: Optional[str] = None,
        strategy: str = 'oneshot',
        confidence_threshold: float = 0.6
    ) -> Tuple[KnowledgeGraph, Dict[str, Any]]:
        """
        Use LLM to infer causal relationships between data variables.

        Args:
            kg: Knowledge graph with data nodes
            node_mapping: Mapping of attribute names to node IDs
            summary: Dataset summary
            domain_hint: Optional domain context
            strategy: Inference strategy ('oneshot', 'pairwise', 'hybrid')
            confidence_threshold: Minimum confidence to include edge

        Returns:
            Tuple of (Updated KG, inference stats)
        """
        start_time = time.time()

        # Get nodes for inference
        nodes = [
            {
                'name': attr['name'],
                'type': attr['type'],
                'role': attr['role'],
                'description': attr.get('description', '')
            }
            for attr in summary.get('attributes', [])
        ]

        # Infer edges using selected strategy
        if strategy == 'oneshot':
            raw_edges = self.infer_edges_oneshot(nodes, 'data', domain_hint)
        elif strategy == 'pairwise':
            raw_edges = self.infer_edges_pairwise_all(nodes, 'data', domain_hint)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Post-process edges
        processed_edges = self.post_process_edges(
            raw_edges, 'data', kg, confidence_threshold
        )

        # Add edges to graph
        added_edges = 0
        for edge in processed_edges:
            source_id = node_mapping.get(edge['source'])
            target_id = node_mapping.get(edge['target'])

            if not source_id or not target_id:
                continue

            try:
                # Merge confidence and reasoning into metadata for storage
                metadata = edge.get('metadata', {}).copy()
                if 'confidence' in edge:
                    metadata['confidence'] = edge['confidence']
                if 'reasoning' in edge:
                    metadata['reasoning'] = edge['reasoning']

                kg.add_edge(
                    source_id,
                    target_id,
                    edge['edge_type'],
                    'data',
                    metadata=metadata
                )
                added_edges += 1
            except ValueError as e:
                print(f"  Warning: Could not add edge {edge['source']} -> {edge['target']}: {e}")

        # Compute stats
        elapsed = time.time() - start_time
        stats = {
            'strategy': strategy,
            'raw_edges': len(raw_edges),
            'processed_edges': len(processed_edges),
            'added_edges': added_edges,
            'elapsed_time': elapsed,
            'confidence_threshold': confidence_threshold
        }

        print(f"✓ Added {added_edges} causal relationships to data layer ({strategy} strategy, {elapsed:.2f}s)")
        return kg, stats

    def infer_edges_oneshot(
        self,
        nodes: List[Dict[str, Any]],
        layer: str,
        domain_hint: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Infer edges using single LLM call (fast but may miss relationships).

        Args:
            nodes: List of node definitions
            layer: Layer name ('data' or 'concept')
            domain_hint: Optional domain context

        Returns:
            List of raw edge definitions
        """
        if layer == 'data':
            prompt = self._build_causal_inference_prompt_oneshot(nodes, domain_hint)
        else:
            prompt = self._build_concept_relationship_prompt_oneshot(nodes)

        try:
            response = self.llm.call_gemini(prompt)
            relationships = response.get('relationships', []) or response.get('causal_relationships', [])

            # Normalize format
            edges = []
            for rel in relationships:
                edges.append({
                    'source': rel.get('source'),
                    'target': rel.get('target'),
                    'edge_type': rel.get('relationship_type', 'causal_influence'),
                    'confidence': rel.get('confidence', 0.5),
                    'reasoning': rel.get('reasoning', ''),
                    'metadata': {
                        'strategy': 'oneshot',
                        'source': 'llm_inference'
                    }
                })

            return edges

        except Exception as e:
            print(f"Warning: Oneshot inference failed: {e}")
            return []

    def infer_edges_pairwise_all(
        self,
        nodes: List[Dict[str, Any]],
        layer: str,
        domain_hint: Optional[str] = None,
        max_api_calls: int = 100,
        batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Infer edges by checking each node pair (thorough but slower).

        Args:
            nodes: List of node definitions
            layer: Layer name ('data' or 'concept')
            domain_hint: Optional domain context
            max_api_calls: Maximum number of API calls
            batch_size: Number of pairs to check per API call

        Returns:
            List of raw edge definitions
        """
        all_edges = []

        # Generate all pairs
        pairs = []
        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                if i != j:  # Don't check self-loops
                    pairs.append((source, target))

        # Limit total pairs
        max_pairs = max_api_calls * batch_size
        if len(pairs) > max_pairs:
            print(f"  Limiting pairwise checks to {max_pairs} pairs (from {len(pairs)})")
            pairs = pairs[:max_pairs]

        # Process in batches
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]

            if layer == 'data':
                prompt = self._build_causal_inference_prompt_pairwise(batch, domain_hint)
            else:
                prompt = self._build_concept_relationship_prompt_pairwise(batch)

            try:
                response = self.llm.call_gemini(prompt)
                relationships = response.get('relationships', [])

                for rel in relationships:
                    if rel.get('has_relationship', False):
                        all_edges.append({
                            'source': rel.get('source'),
                            'target': rel.get('target'),
                            'edge_type': rel.get('relationship_type', 'causal_influence'),
                            'confidence': rel.get('confidence', 0.5),
                            'reasoning': rel.get('reasoning', ''),
                            'metadata': {
                                'strategy': 'pairwise',
                                'source': 'llm_inference'
                            }
                        })

            except Exception as e:
                print(f"  Warning: Pairwise batch {i//batch_size + 1} failed: {e}")
                # Save the prompt for debugging
                debug_path = Path('output/debug') / f'failed_batch_{i//batch_size + 1}.txt'
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                with open(debug_path, 'w') as f:
                    f.write(f"Error: {e}\n\n")
                    f.write(f"Prompt:\n{prompt}\n")
                print(f"  Debug info saved to: {debug_path}")

        return all_edges

    def post_process_edges(
        self,
        edges: List[Dict[str, Any]],
        layer: str,
        kg: Optional[KnowledgeGraph] = None,
        confidence_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Post-process edges: deduplicate, normalize, filter by confidence.

        Args:
            edges: Raw edges from inference
            layer: Layer name
            kg: Optional KG for cycle checking
            confidence_threshold: Minimum confidence to include

        Returns:
            Processed edges
        """
        # 1. Filter by confidence
        edges = [e for e in edges if e.get('confidence', 0) >= confidence_threshold]

        # 1.5. Validate reasoning matches edge direction (and optionally flip if contradictory)
        validated_edges = []
        for edge in edges:
            if self._validate_edge_reasoning(edge):
                validated_edges.append(edge)
            else:
                # Reasoning contradicts direction - try flipping the edge
                flipped_edge = edge.copy()
                flipped_edge['source'], flipped_edge['target'] = edge['target'], edge['source']
                if self._validate_edge_reasoning(flipped_edge):
                    print(f"    → Flipped edge direction to match reasoning")
                    validated_edges.append(flipped_edge)
                else:
                    # Still contradictory, skip this edge
                    print(f"    → Skipping contradictory edge")

        edges = validated_edges

        # 2. Deduplicate exact duplicates
        seen = set()
        deduped = []
        for edge in edges:
            key = (edge['source'], edge['target'], edge['edge_type'])
            if key not in seen:
                seen.add(key)
                deduped.append(edge)

        # 3. Normalize symmetric edges (keep only one direction)
        edges = deduped
        deduped = []
        seen_symmetric = set()

        for edge in edges:
            edge_type = edge['edge_type']
            edge_info = CANONICAL_EDGE_TYPES.get(layer, {}).get(edge_type, {})

            if edge_info.get('directional', True):
                # Directional edge, keep as-is
                deduped.append(edge)
            else:
                # Symmetric edge, normalize
                key = tuple(sorted([edge['source'], edge['target']]))
                if key not in seen_symmetric:
                    seen_symmetric.add(key)
                    deduped.append(edge)

        # 4. Remove inverse duplicates
        edges = deduped
        deduped = []
        inverse_map = {}  # (source, target, type) -> edge

        for edge in edges:
            edge_type = edge['edge_type']
            edge_info = CANONICAL_EDGE_TYPES.get(layer, {}).get(edge_type, {})
            inverse_type = edge_info.get('inverse')

            if inverse_type is None or inverse_type == edge_type:
                # No inverse to worry about
                deduped.append(edge)
            else:
                # Check if inverse exists
                inverse_key = (edge['target'], edge['source'], inverse_type)
                forward_key = (edge['source'], edge['target'], edge_type)

                if inverse_key in inverse_map:
                    # Inverse already added, skip this one
                    continue
                else:
                    # Add this edge and remember it
                    inverse_map[forward_key] = edge
                    deduped.append(edge)

        # 5. Remove incompatible edge pairs (e.g., causal + associated on same node pair)
        edges = deduped
        deduped = []
        edge_by_nodes = {}  # (source, target) -> list of edges

        for edge in edges:
            node_pair = tuple(sorted([edge['source'], edge['target']]))
            if node_pair not in edge_by_nodes:
                edge_by_nodes[node_pair] = []
            edge_by_nodes[node_pair].append(edge)

        for node_pair, edge_group in edge_by_nodes.items():
            if len(edge_group) == 1:
                deduped.append(edge_group[0])
            else:
                # Multiple edges between same nodes - check for incompatibilities
                resolved = self._resolve_incompatible_edges(edge_group, layer)
                deduped.extend(resolved)

        print(f"  Post-processing: {len(edges)} raw -> {len(deduped)} processed edges")
        return deduped

    def _resolve_incompatible_edges(
        self,
        edges: List[Dict[str, Any]],
        layer: str
    ) -> List[Dict[str, Any]]:
        """
        Resolve incompatible edges between the same node pair.

        Rules:
        - If both causal and non-causal exist, keep only causal (higher priority)
        - If multiple causal edges exist with same direction, keep highest confidence
        - If causal edges in opposite directions, keep highest confidence and warn

        Args:
            edges: List of edges between same node pair
            layer: Layer name

        Returns:
            Resolved list of edges
        """
        # Group edges by type priority
        causal_edges = []
        computed_edges = []
        associated_edges = []

        for edge in edges:
            edge_type = edge['edge_type']
            edge_info = CANONICAL_EDGE_TYPES.get(layer, {}).get(edge_type, {})

            if edge_type == 'computed_from':
                computed_edges.append(edge)
            elif edge_info.get('directional', True):
                causal_edges.append(edge)
            else:
                associated_edges.append(edge)

        result = []

        # Priority 1: computed_from (deterministic)
        if computed_edges:
            # Keep highest confidence computed edge
            best_computed = max(computed_edges, key=lambda e: e.get('confidence', 0))
            result.append(best_computed)
            if len(computed_edges) > 1:
                print(f"  Warning: Multiple computed_from edges for same pair, keeping highest confidence")

        # Priority 2: causal_influence (directional)
        elif causal_edges:
            # Group by direction
            forward_edges = {}
            for edge in causal_edges:
                direction = (edge['source'], edge['target'])
                if direction not in forward_edges or edge.get('confidence', 0) > forward_edges[direction].get('confidence', 0):
                    forward_edges[direction] = edge

            # Check for contradictions (edges in both directions)
            nodes = set()
            for edge in causal_edges:
                nodes.add(edge['source'])
                nodes.add(edge['target'])

            if len(forward_edges) > 1:
                # Contradictory causal edges - keep highest confidence
                best_edge = max(forward_edges.values(), key=lambda e: e.get('confidence', 0))
                result.append(best_edge)
                print(f"  Warning: Contradictory causal edges for {nodes}, keeping highest confidence: "
                      f"{best_edge['source']} -> {best_edge['target']} ({best_edge.get('confidence', 0):.2f})")
            else:
                result.extend(forward_edges.values())

        # Priority 3: associated_with (symmetric, only if no causal)
        elif associated_edges:
            # Keep highest confidence
            best_associated = max(associated_edges, key=lambda e: e.get('confidence', 0))
            result.append(best_associated)

        return result

    def _validate_edge_reasoning(
        self,
        edge: Dict[str, Any]
    ) -> bool:
        """
        Validate that the reasoning matches the edge direction.

        Checks if the reasoning text contradicts the stated edge direction.
        Returns True if valid, False if reasoning seems contradictory.

        Args:
            edge: Edge with source, target, and reasoning

        Returns:
            True if reasoning is consistent with edge direction
        """
        reasoning = edge.get('reasoning', '').lower()
        if not reasoning:
            return True  # No reasoning to validate

        source = edge.get('source', '').lower()
        target = edge.get('target', '').lower()

        # Keywords that indicate directionality
        reverse_indicators = [
            f"{target} influences {source}",
            f"{target} causes {source}",
            f"{target} affects {source}",
            f"{target} impacts {source}",
            f"{target} determines {source}",
            f"{source} does not determine {target}",
            f"{source} does not cause {target}",
            f"{source} does not influence {target}",
        ]

        # Check if reasoning suggests opposite direction
        for indicator in reverse_indicators:
            if indicator in reasoning:
                print(f"  Warning: Reasoning contradicts edge direction for {edge['source']} -> {edge['target']}")
                print(f"    Reasoning: {edge.get('reasoning', '')[:100]}...")
                return False

        return True

    def build_concept_layer(
        self,
        kg: KnowledgeGraph,
        summary: Dict[str, Any]
    ) -> Tuple[KnowledgeGraph, Dict[str, str]]:
        """
        Create concept layer nodes and semantic edges from summary.

        Args:
            kg: Knowledge graph with data layer
            summary: Dataset summary with suggested concepts

        Returns:
            Tuple of (Updated KG, mapping of concept names to node IDs)
        """
        concept_mapping = {}

        # Create concept nodes
        for concept in summary.get('suggested_concepts', []):
            # Create node ID
            concept_name = concept['name']
            node_id = f"concept_{concept_name}"
            concept_mapping[concept_name] = node_id

            # Prepare node data
            node_data = {
                'name': concept_name,
                'node_type': 'data_concept',
                'definition': concept.get('definition', ''),
                'related_attributes': concept.get('related_attributes', []),
                'confidence': concept.get('confidence', 0.5),
                'abstraction_level': concept.get('abstraction_level', 5),
                'activated': False
            }

            # Add node
            kg.add_node('concept', node_id, node_data)

        print(f"✓ Created {len(concept_mapping)} concept layer nodes")

        # Infer semantic relationships between concepts
        kg = self._infer_concept_relationships(kg, concept_mapping, summary)

        return kg, concept_mapping

    def _infer_concept_relationships(
        self,
        kg: KnowledgeGraph,
        concept_mapping: Dict[str, str],
        summary: Dict[str, Any]
    ) -> KnowledgeGraph:
        """
        Infer semantic relationships between concepts using LLM.

        Args:
            kg: Knowledge graph with concept nodes
            concept_mapping: Mapping of concept names to node IDs
            summary: Dataset summary

        Returns:
            Updated knowledge graph with concept edges
        """
        # Prepare prompt
        prompt = self._build_concept_relationship_prompt(summary)

        try:
            response = self.llm.call_gemini(prompt)

            relationships = response.get('concept_relationships', [])

            added_edges = 0
            for rel in relationships:
                source_name = rel.get('source')
                target_name = rel.get('target')
                edge_type = rel.get('relationship_type')
                reasoning = rel.get('reasoning', '')
                confidence = rel.get('confidence', 0.5)

                # Map to node IDs
                source_id = concept_mapping.get(source_name)
                target_id = concept_mapping.get(target_name)

                if not source_id or not target_id:
                    print(f"  Warning: Skipping concept edge {source_name} -> {target_name} (node not found)")
                    continue

                # Validate edge type
                if edge_type not in CANONICAL_EDGE_TYPES['concept']:
                    print(f"  Warning: Non-canonical concept edge type '{edge_type}', skipping")
                    continue

                # Add edge
                try:
                    metadata = {
                        'reasoning': reasoning,
                        'confidence': confidence,
                        'source': 'llm_inference'
                    }
                    kg.add_edge(
                        source_id,
                        target_id,
                        edge_type,
                        'concept',
                        metadata=metadata
                    )
                    added_edges += 1
                except ValueError as e:
                    print(f"  Warning: Could not add concept edge {source_name} -> {target_name}: {e}")

            print(f"✓ Added {added_edges} concept relationships")
            return kg

        except Exception as e:
            print(f"Warning: Failed to infer concept relationships: {e}")
            return kg

    def create_bridge_edges(
        self,
        kg: KnowledgeGraph,
        data_node_mapping: Dict[str, str],
        concept_node_mapping: Dict[str, str],
        summary: Dict[str, Any]
    ) -> KnowledgeGraph:
        """
        Create bridge edges between concept and data layers.

        Args:
            kg: Knowledge graph with data and concept layers
            data_node_mapping: Mapping of data attribute names to node IDs
            concept_node_mapping: Mapping of concept names to node IDs
            summary: Dataset summary

        Returns:
            Updated knowledge graph with bridge edges
        """
        added_bridges = 0

        # For each concept, link to its related attributes
        for concept in summary.get('suggested_concepts', []):
            concept_name = concept['name']
            concept_id = concept_node_mapping.get(concept_name)

            if not concept_id:
                continue

            # Link to related attributes
            for attr_name in concept.get('related_attributes', []):
                data_id = data_node_mapping.get(attr_name)

                if not data_id:
                    print(f"  Warning: Attribute '{attr_name}' not found in data layer")
                    continue

                # Add bridge edge (concept -> data)
                try:
                    kg.add_bridge_edge(
                        concept_id,
                        data_id,
                        'operationalized_by',
                        metadata={
                            'source': 'concept_definition',
                            'concept_confidence': concept.get('confidence', 0.5)
                        }
                    )
                    added_bridges += 1
                except Exception as e:
                    print(f"  Warning: Could not add bridge {concept_name} -> {attr_name}: {e}")

        print(f"✓ Added {added_bridges} bridge edges")
        return kg

    def build_initial_kg(
        self,
        summary: Dict[str, Any],
        domain_hint: Optional[str] = None
    ) -> KnowledgeGraph:
        """
        Main orchestration function to build complete initial knowledge graph.

        Args:
            summary: Dataset summary from DataSummarizer
            domain_hint: Optional domain context for better inference

        Returns:
            Complete knowledge graph with all layers
        """
        print("\n" + "="*60)
        print("BUILDING KNOWLEDGE GRAPH")
        print("="*60)

        # Extract domain hint from summary if not provided
        if not domain_hint:
            domain_hint = summary.get('dataset_info', {}).get('domain', 'unknown')

        print(f"\nDomain: {domain_hint}")
        print(f"Dataset: {summary.get('dataset_info', {}).get('name', 'unknown')}")

        # Step 1: Build data layer
        print("\n[1/4] Building data layer...")
        kg, data_mapping = self.build_data_layer(summary)

        # Step 2: Infer causal relationships
        print("\n[2/4] Inferring causal relationships...")
        kg, causal_stats = self.infer_causal_relationships(kg, data_mapping, summary, domain_hint)

        # Step 3: Build concept layer
        print("\n[3/4] Building concept layer...")
        kg, concept_mapping = self.build_concept_layer(kg, summary)

        # Step 4: Create bridge edges
        print("\n[4/4] Creating bridge edges...")
        kg = self.create_bridge_edges(kg, data_mapping, concept_mapping, summary)

        # Validate the graph
        print("\n" + "-"*60)
        print("VALIDATION")
        print("-"*60)
        validation = kg.validate()

        if validation['valid']:
            print("✓ Knowledge graph is valid")
        else:
            print("✗ Knowledge graph has errors:")
            for error in validation['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")

        if validation['warnings']:
            print(f"\nWarnings: {len(validation['warnings'])}")
            for warning in validation['warnings'][:3]:
                print(f"  - {warning}")

        # Show statistics
        print("\n" + "-"*60)
        print("STATISTICS")
        print("-"*60)
        stats = kg.get_layer_stats()

        for layer_name, layer_stats in stats.items():
            print(f"\n{layer_name.upper()} Layer:")
            if 'node_count' in layer_stats:
                print(f"  Nodes: {layer_stats['node_count']}")
            print(f"  Edges: {layer_stats['edge_count']}")
            if layer_stats['edge_types']:
                for edge_type, count in layer_stats['edge_types'].items():
                    print(f"    - {edge_type}: {count}")

        print("\n" + "="*60)
        print("KNOWLEDGE GRAPH CONSTRUCTION COMPLETE")
        print("="*60)

        return kg

    # ==================== PROMPT BUILDERS ====================

    def _build_causal_inference_prompt(
        self,
        summary: Dict[str, Any],
        domain_hint: Optional[str] = None
    ) -> str:
        """Build prompt for causal relationship inference."""

        attributes = summary.get('attributes', [])

        # Create simplified attribute list
        attr_list = []
        for attr in attributes:
            attr_list.append({
                'name': attr['name'],
                'type': attr['type'],
                'role': attr['role'],
                'description': attr.get('description', '')
            })

        prompt = f"""You are a data analysis expert. Given a dataset with the following attributes, identify potential CAUSAL relationships between variables.

Domain: {domain_hint or 'unknown'}

Attributes:
{json.dumps(attr_list, indent=2)}

IMPORTANT RULES:
1. Only suggest relationships where one variable CAUSES changes in another
2. Consider domain knowledge and common causal patterns
3. Use these relationship types:
   - "causal_influence": Direct causal relationship (A causes B)
   - "computed_from": B is mathematically derived from A
   - "correlates_with": Variables are associated but no clear causal direction (symmetric)

4. Avoid creating cycles where A causes B causes A (with the same relationship type)
5. Focus on the most confident relationships (confidence > 0.6)
6. Provide brief reasoning for each relationship

Respond with JSON in this format:
{{
  "causal_relationships": [
    {{
      "source": "attribute_name",
      "target": "attribute_name",
      "relationship_type": "causal_influence|computed_from|correlates_with",
      "confidence": 0.0-1.0,
      "reasoning": "Brief explanation"
    }}
  ]
}}

Think carefully about causal direction. For example:
- Budget often influences Revenue (not the other way)
- Revenue is used to compute Profit
- Ratings might correlate with Box Office but causality is unclear

Generate 5-15 high-confidence relationships."""

        return prompt

    def _build_concept_relationship_prompt(
        self,
        summary: Dict[str, Any]
    ) -> str:
        """Build prompt for concept relationship inference."""

        concepts = summary.get('suggested_concepts', [])

        # Create simplified concept list
        concept_list = []
        for concept in concepts:
            concept_list.append({
                'name': concept['name'],
                'definition': concept.get('definition', ''),
                'abstraction_level': concept.get('abstraction_level', 5)
            })

        prompt = f"""You are a knowledge modeling expert. Given these analytical concepts, identify semantic relationships between them.

Concepts:
{json.dumps(concept_list, indent=2)}

IMPORTANT RULES:
1. Use these relationship types:
   - "part_of": Source concept is a component of target concept
   - "has_part": Source concept contains target concept (inverse of part_of)
   - "contrasts_with": Concepts are opposing dimensions (symmetric)
   - "similar_to": Concepts are semantically related (symmetric)
   - "prerequisite_for": Source must be understood before target
   - "requires": Target requires understanding source (inverse of prerequisite_for)
   - "measured_by_multiple": Concept operationalized by multiple sub-concepts

2. DO NOT create inverse duplicates (e.g., both A part_of B and B has_part A)
3. DO NOT create simple cycles (e.g., A prerequisite_for B, B prerequisite_for A)
4. Consider abstraction levels (higher-level concepts often contain lower-level ones)
5. Focus on clear, logical relationships

Respond with JSON in this format:
{{
  "concept_relationships": [
    {{
      "source": "concept_name",
      "target": "concept_name",
      "relationship_type": "part_of|similar_to|prerequisite_for|...",
      "reasoning": "Brief explanation"
    }}
  ]
}}

Generate 3-10 meaningful relationships between these concepts."""

        return prompt

    def _build_causal_inference_prompt_oneshot(
        self,
        nodes: List[Dict[str, Any]],
        domain_hint: Optional[str] = None
    ) -> str:
        """Build oneshot prompt for causal inference."""
        prompt = f"""Identify causal relationships between variables.

Domain: {domain_hint or 'unknown'}

Attributes:
{json.dumps(nodes, indent=2)}

Relationship types:
- "computed_from": Target = f(source) mathematically (e.g., profit = revenue - cost)
- "causal_influence": Source causes target empirically (e.g., price → sales)
- "associated_with": Correlated but causal direction unclear (use sparingly)

Rules:
- Prefer "causal_influence" over "associated_with" when direction is clear
- Avoid cycles (if A→B, don't add B→A with same type)
- State reasoning clearly
- Confidence > 0.6

JSON format:
{{
  "relationships": [
    {{
      "source": "name",
      "target": "name",
      "relationship_type": "causal_influence|computed_from|associated_with",
      "confidence": 0.0-1.0,
      "reasoning": "Brief explanation"
    }}
  ]
}}

Generate 5-15 relationships."""
        return prompt

    def _build_causal_inference_prompt_pairwise(
        self,
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
        domain_hint: Optional[str] = None
    ) -> str:
        """Build pairwise prompt for causal inference."""
        pairs_info = []
        for source, target in pairs:
            pairs_info.append({
                'source': source['name'],
                'target': target['name'],
                'source_type': source['type'],
                'target_type': target['type']
            })

        prompt = f"""Analyze variable pairs for causal relationships.

Domain: {domain_hint or 'unknown'}

Variable pairs (directional: source → target):
{json.dumps(pairs_info, indent=2)}

Relationship types:
- "computed_from": Target = mathematical function of source (e.g., profit = revenue - cost)
- "causal_influence": Source causes changes in target (e.g., price → sales)
- "associated_with": Correlated but causal direction unclear (e.g., height ↔ weight)

Rules:
1. Check if the given direction (source → target) makes sense
2. If reasoning says "B causes A" but pair is "A → B", mark has_relationship=false
3. Use "associated_with" ONLY if causal direction is truly unclear
4. Confidence > 0.6 only

JSON format:
{{
  "relationships": [
    {{
      "source": "name",
      "target": "name",
      "has_relationship": true/false,
      "relationship_type": "causal_influence|computed_from|associated_with",
      "confidence": 0.0-1.0,
      "reasoning": "Brief explanation"
    }}
  ]
}}"""
        return prompt

    def _build_concept_relationship_prompt_oneshot(
        self,
        nodes: List[Dict[str, Any]]
    ) -> str:
        """Build oneshot prompt for concept relationships."""
        prompt = f"""You are a knowledge modeling expert. Given these analytical concepts, identify semantic relationships between them.

Concepts:
{json.dumps(nodes, indent=2)}

IMPORTANT RULES:
1. Use these relationship types:
   - "part_of": Source concept is a component of target concept
   - "similar_to": Concepts are semantically related (symmetric)
   - "prerequisite_for": Source must be understood before target

2. DO NOT create inverse duplicates (e.g., both A part_of B and B has_part A)
3. DO NOT create simple cycles
4. Focus on clear, logical relationships

Respond with JSON:
{{
  "relationships": [
    {{
      "source": "concept_name",
      "target": "concept_name",
      "relationship_type": "part_of|similar_to|prerequisite_for",
      "reasoning": "Brief explanation"
    }}
  ]
}}

Generate 3-10 meaningful relationships."""
        return prompt

    def _build_concept_relationship_prompt_pairwise(
        self,
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    ) -> str:
        """Build pairwise prompt for concept relationships."""
        pairs_info = []
        for source, target in pairs:
            pairs_info.append({
                'source': source['name'],
                'target': target['name']
            })

        prompt = f"""You are a knowledge modeling expert. For each pair of concepts below, determine if there is a semantic relationship.

Pairs to check:
{json.dumps(pairs_info, indent=2)}

For each pair, determine:
1. Is there a relationship? (yes/no)
2. If yes, what type?
   - "part_of": Source is component of target
   - "similar_to": Concepts are related (symmetric)
   - "prerequisite_for": Source needed before target
3. Brief reasoning

Respond with JSON:
{{
  "relationships": [
    {{
      "source": "concept_name",
      "target": "concept_name",
      "has_relationship": true/false,
      "relationship_type": "part_of|similar_to|prerequisite_for",
      "reasoning": "Brief explanation"
    }}
  ]
}}"""
        return prompt

    def save_kg(self, kg: KnowledgeGraph, output_path: Path) -> None:
        """
        Save knowledge graph to JSON file.

        Args:
            kg: Knowledge graph to save
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(kg.to_json(), f, indent=2)

        print(f"✓ Knowledge graph saved to: {output_path}")

    def load_kg(self, input_path: Path) -> KnowledgeGraph:
        """
        Load knowledge graph from JSON file.

        Args:
            input_path: Path to input file

        Returns:
            Loaded knowledge graph
        """
        with open(input_path, 'r') as f:
            json_data = json.load(f)

        kg = KnowledgeGraph.from_json(json_data)
        print(f"✓ Knowledge graph loaded from: {input_path}")

        return kg


# Example usage
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize
    llm = GeminiClient()
    builder = KGBuilder(llm)

    # Load summary
    summary_path = Path('graph_generation/output/test_summary.json') # concerned about where this is going
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    # Build knowledge graph
    kg = builder.build_initial_kg(summary, domain_hint='movies')

    # Save
    builder.save_kg(kg, Path('graph_generation/output/initial_kg.json')) # concerned about where this is going

    print("\n✨ Knowledge graph construction complete!")
