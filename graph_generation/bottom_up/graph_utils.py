"""
Knowledge Graph utilities for managing three-layer knowledge graphs with canonical edge types.

This module provides:
- CANONICAL_EDGE_TYPES: Definitions of allowed edge types for each layer
- KnowledgeGraph: Class for managing nodes, edges, and validation
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from collections import defaultdict, deque
import json


# Part 1: Canonical Edge Type Definitions
CANONICAL_EDGE_TYPES = {
    'data': {
        # Causal/Directional relationships (mutually exclusive with associated_with)
        'causal_influence': {
            'definition': 'Source variable directly causes changes in target variable (empirical causation)',
            'directional': True,
            'inverse': None,
            'incompatible_with': ['associated_with'],  # Cannot co-exist with symmetric association
            'examples': [
                'marketing_budget → sales_revenue',
                'price → demand',
                'training_hours → employee_productivity'
            ],
            'notes': 'Use when causality is clear from domain knowledge or experimental evidence. Implies correlation but is stronger.'
        },
        'computed_from': {
            'definition': 'Target variable is mathematically derived or calculated from source variable (definitional relationship)',
            'directional': True,
            'inverse': None,
            'incompatible_with': ['associated_with', 'causal_influence'],  # Mathematical, not empirical
            'examples': [
                'revenue, costs → profit',
                'US_gross, international_gross → worldwide_gross',
                'clicks, impressions → click_through_rate'
            ],
            'notes': 'Use for mathematical/definitional relationships where target is computed. This is deterministic, not probabilistic.'
        },

        # Non-causal/Symmetric relationships (use when causality unclear)
        'associated_with': {
            'definition': 'Variables show statistical association but causal direction is unclear, bidirectional, or spurious',
            'directional': False,
            'inverse': 'associated_with',  # Self-inverse
            'incompatible_with': ['causal_influence', 'computed_from'],  # Cannot have both causal and non-causal
            'examples': [
                'ice_cream_sales ↔ drowning_incidents (confounded by temperature)',
                'height ↔ weight (bidirectional/correlational)',
                'movie_genre ↔ runtime (associated but no clear causation)'
            ],
            'notes': 'Use ONLY when you cannot determine causal direction. If causality exists, use causal_influence instead.'
        }
    },
    'concept': {
        'part_of': {
            'definition': 'Source concept is a component, aspect, or subset of target concept',
            'directional': True,
            'inverse': 'has_part',
            'examples': [
                'Brand Loyalty → Customer Satisfaction',
                'Short-term Memory → Cognitive Ability',
                'Agreeableness → Personality'
            ]
        },
        'has_part': {
            'definition': 'Source concept contains or is composed of target concept',
            'directional': True,
            'inverse': 'part_of',
            'examples': [
                'Customer Satisfaction → Brand Loyalty',
                'Cognitive Ability → Short-term Memory',
                'Personality → Agreeableness'
            ]
        },
        'contrasts_with': {
            'definition': 'Concepts represent opposing or contrasting dimensions',
            'directional': False,
            'inverse': 'contrasts_with',  # Self-inverse
            'examples': [
                'Individualism ↔ Collectivism',
                'Risk Aversion ↔ Risk Seeking',
                'Intrinsic Motivation ↔ Extrinsic Motivation'
            ]
        },
        'similar_to': {
            'definition': 'Concepts are semantically related or share characteristics',
            'directional': False,
            'inverse': 'similar_to',  # Self-inverse
            'examples': [
                'Customer Loyalty ↔ Brand Attachment',
                'Anxiety ↔ Stress',
                'Leadership ↔ Influence'
            ]
        },
        'prerequisite_for': {
            'definition': 'Source concept must be understood before target concept',
            'directional': True,
            'inverse': 'requires',
            'examples': [
                'Basic Statistics → Regression Analysis',
                'Trust → Commitment',
                'Awareness → Consideration'
            ]
        },
        'requires': {
            'definition': 'Target concept requires understanding of source concept',
            'directional': True,
            'inverse': 'prerequisite_for',
            'examples': [
                'Regression Analysis → Basic Statistics',
                'Commitment → Trust',
                'Consideration → Awareness'
            ]
        },
        'measured_by_multiple': {
            'definition': 'Concept is operationalized by combining multiple sub-concepts',
            'directional': True,
            'inverse': None,
            'examples': [
                'Socioeconomic Status → (Income, Education, Occupation)',
                'Job Satisfaction → (Pay Satisfaction, Work Environment, Career Growth)',
                'Academic Performance → (GPA, Test Scores, Project Quality)'
            ]
        }
    },
    'bridge': {
        'operationalized_by': {
            'definition': 'Concept is measured or operationalized using data variable',
            'directional': True,
            'inverse': None,
            'examples': [
                'Customer Satisfaction → satisfaction_score',
                'Revenue → revenue_usd',
                'Brand Loyalty → repeat_purchase_rate'
            ]
        }
    },
    'question': {
        # Question layer edge types can be added in future
    }
}


class KnowledgeGraph:
    """
    Knowledge Graph with three layers (data, concept, question) and canonical edge types.

    This class manages nodes and edges across multiple layers while enforcing:
    - Canonical edge types
    - Prevention of inverse duplicates
    - Prevention of simple cycles (A→B→A with same edge type)
    """

    def __init__(self):
        """Initialize an empty knowledge graph."""
        self.layers = {
            'data': {'nodes': {}, 'edges': []},
            'concept': {'nodes': {}, 'edges': []},
            'question': {'nodes': {}, 'edges': []}
        }
        self.bridge_edges = []
        self.metadata = {}

        # Build adjacency indices for efficient lookups
        self._rebuild_indices()

    def _rebuild_indices(self):
        """Rebuild adjacency indices for efficient edge lookups."""
        self._adjacency_out = defaultdict(list)  # node_id -> [(target, edge_type, layer)]
        self._adjacency_in = defaultdict(list)   # node_id -> [(source, edge_type, layer)]

        # Index layer edges
        for layer_name, layer in self.layers.items():
            for edge in layer['edges']:
                source = edge['source']
                target = edge['target']
                edge_type = edge['edge_type']

                self._adjacency_out[source].append((target, edge_type, layer_name))
                self._adjacency_in[target].append((source, edge_type, layer_name))

        # Index bridge edges
        for edge in self.bridge_edges:
            source = edge['source']
            target = edge['target']
            edge_type = edge['edge_type']

            self._adjacency_out[source].append((target, edge_type, 'bridge'))
            self._adjacency_in[target].append((source, edge_type, 'bridge'))

    # ==================== NODE OPERATIONS ====================

    def add_node(self, layer: str, node_id: str, node_data: Dict[str, Any]) -> None:
        """
        Add a node to the specified layer.

        Args:
            layer: Layer name ('data', 'concept', or 'question')
            node_id: Unique identifier for the node
            node_data: Dictionary containing node attributes

        Raises:
            ValueError: If layer is invalid or node_id already exists
        """
        if layer not in self.layers:
            raise ValueError(f"Invalid layer '{layer}'. Must be one of {list(self.layers.keys())}")

        if node_id in self.layers[layer]['nodes']:
            raise ValueError(f"Node '{node_id}' already exists in layer '{layer}'")

        # Store node with its layer information
        node_data['layer'] = layer
        node_data['node_id'] = node_id
        self.layers[layer]['nodes'][node_id] = node_data

    def get_node(self, node_id: str, layer: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a node by ID.

        Args:
            node_id: Node identifier
            layer: Optional layer to search in. If None, searches all layers.

        Returns:
            Node data dictionary or None if not found
        """
        if layer:
            return self.layers[layer]['nodes'].get(node_id)

        # Search all layers
        for layer_name, layer_data in self.layers.items():
            if node_id in layer_data['nodes']:
                return layer_data['nodes'][node_id]

        return None

    def update_node(self, node_id: str, updates: Dict[str, Any]) -> None:
        """
        Update node attributes.

        Args:
            node_id: Node identifier
            updates: Dictionary of attributes to update

        Raises:
            KeyError: If node not found
        """
        node = self.get_node(node_id)
        if not node:
            raise KeyError(f"Node '{node_id}' not found")

        # Update the node in its layer
        layer = node['layer']
        self.layers[layer]['nodes'][node_id].update(updates)

    def remove_node(self, node_id: str) -> None:
        """
        Remove a node and all connected edges.

        Args:
            node_id: Node identifier

        Raises:
            KeyError: If node not found
        """
        node = self.get_node(node_id)
        if not node:
            raise KeyError(f"Node '{node_id}' not found")

        layer = node['layer']

        # Remove node
        del self.layers[layer]['nodes'][node_id]

        # Remove all edges connected to this node
        self.layers[layer]['edges'] = [
            e for e in self.layers[layer]['edges']
            if e['source'] != node_id and e['target'] != node_id
        ]

        # Remove bridge edges
        self.bridge_edges = [
            e for e in self.bridge_edges
            if e['source'] != node_id and e['target'] != node_id
        ]

        # Rebuild indices
        self._rebuild_indices()

    def find_nodes(self, **filters) -> List[Dict[str, Any]]:
        """
        Find nodes matching the given filters.

        Args:
            **filters: Key-value pairs to filter nodes by

        Returns:
            List of nodes matching all filter criteria

        Examples:
            find_nodes(layer='concept', activated=True)
            find_nodes(node_type='raw_variable')
        """
        results = []

        for layer_name, layer in self.layers.items():
            for node_id, node_data in layer['nodes'].items():
                # Check if all filters match
                if all(node_data.get(key) == value for key, value in filters.items()):
                    results.append(node_data.copy())

        return results

    # ==================== EDGE OPERATIONS ====================

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        layer: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an edge with validation.

        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of edge (must be canonical)
            layer: Layer name ('data', 'concept', or 'question')
            metadata: Optional metadata for the edge

        Raises:
            ValueError: If validation fails
            KeyError: If nodes don't exist
        """
        # Validate layer
        if layer not in self.layers:
            raise ValueError(f"Invalid layer '{layer}'. Must be one of {list(self.layers.keys())}")

        # Validate edge type
        if edge_type not in CANONICAL_EDGE_TYPES.get(layer, {}):
            available = list(CANONICAL_EDGE_TYPES.get(layer, {}).keys())
            raise ValueError(
                f"Invalid edge type '{edge_type}' for layer '{layer}'. "
                f"Available types: {available}"
            )

        # Check nodes exist
        source_node = self.get_node(source, layer)
        target_node = self.get_node(target, layer)

        if not source_node:
            raise KeyError(f"Source node '{source}' not found in layer '{layer}'")
        if not target_node:
            raise KeyError(f"Target node '{target}' not found in layer '{layer}'")

        # Check for inverse duplicate
        if self._has_inverse_edge(source, target, edge_type, layer):
            edge_info = CANONICAL_EDGE_TYPES[layer][edge_type]
            inverse = edge_info['inverse']
            raise ValueError(
                f"Cannot add edge {source} --{edge_type}--> {target}. "
                f"Inverse edge already exists: {target} --{inverse}--> {source}"
            )

        # Check for simple cycles (A→B→A with same edge type)
        if self._would_create_simple_cycle(source, target, edge_type, layer):
            raise ValueError(
                f"Cannot add edge {source} --{edge_type}--> {target}. "
                f"This would create a simple cycle: reverse edge already exists."
            )

        # Create edge
        edge = {
            'source': source,
            'target': target,
            'edge_type': edge_type,
            'metadata': metadata or {}
        }

        self.layers[layer]['edges'].append(edge)

        # Update indices
        self._adjacency_out[source].append((target, edge_type, layer))
        self._adjacency_in[target].append((source, edge_type, layer))

    def add_bridge_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a bridge edge between layers.

        Args:
            source: Source node ID (typically in concept layer)
            target: Target node ID (typically in data layer)
            edge_type: Type of bridge edge (must be canonical)
            metadata: Optional metadata for the edge

        Raises:
            ValueError: If validation fails
            KeyError: If nodes don't exist
        """
        # Validate edge type
        if edge_type not in CANONICAL_EDGE_TYPES.get('bridge', {}):
            available = list(CANONICAL_EDGE_TYPES.get('bridge', {}).keys())
            raise ValueError(
                f"Invalid bridge edge type '{edge_type}'. "
                f"Available types: {available}"
            )

        # Check nodes exist
        source_node = self.get_node(source)
        target_node = self.get_node(target)

        if not source_node:
            raise KeyError(f"Source node '{source}' not found")
        if not target_node:
            raise KeyError(f"Target node '{target}' not found")

        # Create edge
        edge = {
            'source': source,
            'target': target,
            'edge_type': edge_type,
            'metadata': metadata or {}
        }

        self.bridge_edges.append(edge)

        # Update indices
        self._adjacency_out[source].append((target, edge_type, 'bridge'))
        self._adjacency_in[target].append((source, edge_type, 'bridge'))

    def get_edges(
        self,
        node_id: Optional[str] = None,
        edge_type: Optional[str] = None,
        layer: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get edges with optional filters.

        Args:
            node_id: Filter by source or target node
            edge_type: Filter by edge type
            layer: Filter by layer (or 'bridge')

        Returns:
            List of edges matching filters
        """
        edges = []

        # Collect edges from specified layers
        layers_to_search = [layer] if layer and layer != 'bridge' else list(self.layers.keys())

        for layer_name in layers_to_search:
            if layer_name in self.layers:
                for edge in self.layers[layer_name]['edges']:
                    # Apply filters
                    if node_id and edge['source'] != node_id and edge['target'] != node_id:
                        continue
                    if edge_type and edge['edge_type'] != edge_type:
                        continue

                    edges.append({**edge, 'layer': layer_name})

        # Include bridge edges if requested
        if not layer or layer == 'bridge':
            for edge in self.bridge_edges:
                # Apply filters
                if node_id and edge['source'] != node_id and edge['target'] != node_id:
                    continue
                if edge_type and edge['edge_type'] != edge_type:
                    continue

                edges.append({**edge, 'layer': 'bridge'})

        return edges

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None,
        direction: str = 'out'
    ) -> List[str]:
        """
        Get neighboring nodes.

        Args:
            node_id: Node to get neighbors for
            edge_type: Optional filter by edge type
            direction: 'out' (outgoing), 'in' (incoming), or 'both'

        Returns:
            List of neighbor node IDs

        Raises:
            ValueError: If direction is invalid
        """
        if direction not in ['out', 'in', 'both']:
            raise ValueError(f"Invalid direction '{direction}'. Must be 'out', 'in', or 'both'")

        neighbors = []

        if direction in ['out', 'both']:
            for target, et, layer in self._adjacency_out[node_id]:
                if edge_type is None or et == edge_type:
                    neighbors.append(target)

        if direction in ['in', 'both']:
            for source, et, layer in self._adjacency_in[node_id]:
                if edge_type is None or et == edge_type:
                    neighbors.append(source)

        return neighbors

    # ==================== VALIDATION HELPERS ====================

    def _has_inverse_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        layer: str
    ) -> bool:
        """
        Check if inverse relationship already exists.

        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Edge type being added
            layer: Layer name

        Returns:
            True if inverse edge exists
        """
        edge_info = CANONICAL_EDGE_TYPES[layer][edge_type]
        inverse_type = edge_info['inverse']

        # If no inverse or self-inverse, no duplicate possible
        if inverse_type is None:
            return False

        # Check if inverse edge exists
        for edge in self.layers[layer]['edges']:
            if edge['source'] == target and edge['target'] == source:
                if edge['edge_type'] == inverse_type:
                    return True
                # For self-inverse edges (symmetric), also check same edge type
                if inverse_type == edge_type and edge['edge_type'] == edge_type:
                    return True

        return False

    def _would_create_simple_cycle(
        self,
        source: str,
        target: str,
        edge_type: str,
        layer: str
    ) -> bool:
        """
        Check if adding edge would create a simple cycle (A→B→A with same edge type).

        Only checks for direct reverse edges with the same edge type.
        Does not prevent longer cycles like A→B→C→A.

        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Edge type being added
            layer: Layer name

        Returns:
            True if reverse edge with same type already exists
        """
        edge_info = CANONICAL_EDGE_TYPES[layer][edge_type]

        # Non-directional edges don't create problematic cycles
        if not edge_info['directional']:
            return False

        # Check if direct reverse edge exists with same edge type
        for edge in self.layers[layer]['edges']:
            if (edge['source'] == target and
                edge['target'] == source and
                edge['edge_type'] == edge_type):
                return True

        return False

    # ==================== QUERY OPERATIONS ====================

    def find_path(
        self,
        source: str,
        target: str,
        edge_type: Optional[str] = None,
        max_depth: int = 5
    ) -> Optional[List[str]]:
        """
        Find a path between two nodes using BFS.

        Args:
            source: Starting node ID
            target: Target node ID
            edge_type: Optional filter by edge type
            max_depth: Maximum path length to search

        Returns:
            List of node IDs representing path, or None if no path found
        """
        if source == target:
            return [source]

        # BFS
        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            current, path = queue.popleft()

            if len(path) > max_depth:
                continue

            # Get outgoing neighbors
            for next_node in self.get_neighbors(current, edge_type=edge_type, direction='out'):
                if next_node == target:
                    return path + [next_node]

                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, path + [next_node]))

        return None

    def find_all_paths(
        self,
        source: str,
        target: str,
        edge_type: Optional[str] = None,
        max_depth: int = 5,
        max_paths: int = 100
    ) -> List[List[str]]:
        """
        Find all paths between two nodes using DFS.

        Args:
            source: Starting node ID
            target: Target node ID
            edge_type: Optional filter by edge type
            max_depth: Maximum path length to search
            max_paths: Maximum number of paths to return (prevents infinite search)

        Returns:
            List of paths, where each path is a list of node IDs
        """
        if source == target:
            return [[source]]

        all_paths = []

        def dfs(current: str, path: List[str], visited: Set[str]):
            """Recursive DFS to find all paths."""
            if len(all_paths) >= max_paths:
                return

            if len(path) > max_depth:
                return

            # Check if we reached the target
            if current == target:
                all_paths.append(path.copy())
                return

            # Explore neighbors
            for next_node in self.get_neighbors(current, edge_type=edge_type, direction='out'):
                if next_node not in visited:
                    visited.add(next_node)
                    path.append(next_node)
                    dfs(next_node, path, visited)
                    path.pop()
                    visited.remove(next_node)

        # Start DFS
        dfs(source, [source], {source})

        return all_paths

    # ==================== VALIDATION ====================

    def validate(self) -> Dict[str, Any]:
        """
        Validate graph integrity.

        Checks:
        - No dangling edges (all sources/targets exist)
        - All edge types are canonical
        - Bridge edges connect correct layers

        Returns:
            Dictionary with validation results and any errors found
        """
        errors = []
        warnings = []

        # Check layer edges
        for layer_name, layer in self.layers.items():
            for i, edge in enumerate(layer['edges']):
                source = edge['source']
                target = edge['target']
                edge_type = edge['edge_type']

                # Check nodes exist
                if source not in layer['nodes']:
                    errors.append(
                        f"Layer '{layer_name}', edge {i}: "
                        f"Source node '{source}' not found"
                    )

                if target not in layer['nodes']:
                    errors.append(
                        f"Layer '{layer_name}', edge {i}: "
                        f"Target node '{target}' not found"
                    )

                # Check edge type is canonical
                if edge_type not in CANONICAL_EDGE_TYPES.get(layer_name, {}):
                    errors.append(
                        f"Layer '{layer_name}', edge {i}: "
                        f"Non-canonical edge type '{edge_type}'"
                    )

        # Check bridge edges
        for i, edge in enumerate(self.bridge_edges):
            source = edge['source']
            target = edge['target']
            edge_type = edge['edge_type']

            # Check nodes exist
            source_node = self.get_node(source)
            target_node = self.get_node(target)

            if not source_node:
                errors.append(
                    f"Bridge edge {i}: Source node '{source}' not found"
                )

            if not target_node:
                errors.append(
                    f"Bridge edge {i}: Target node '{target}' not found"
                )

            # Check edge type is canonical
            if edge_type not in CANONICAL_EDGE_TYPES.get('bridge', {}):
                errors.append(
                    f"Bridge edge {i}: Non-canonical edge type '{edge_type}'"
                )

            # Warn if bridge doesn't connect expected layers
            if source_node and target_node:
                if source_node['layer'] == 'concept' and target_node['layer'] != 'data':
                    warnings.append(
                        f"Bridge edge {i}: Expected concept→data, "
                        f"but got {source_node['layer']}→{target_node['layer']}"
                    )

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'node_count': sum(len(layer['nodes']) for layer in self.layers.values()),
            'edge_count': sum(len(layer['edges']) for layer in self.layers.values()),
            'bridge_edge_count': len(self.bridge_edges)
        }

    # ==================== SERIALIZATION ====================

    def to_json(self) -> Dict[str, Any]:
        """
        Export entire graph to JSON-serializable dictionary.

        Returns:
            Dictionary containing all graph data
        """
        return {
            'layers': self.layers,
            'bridge_edges': self.bridge_edges,
            'metadata': self.metadata
        }

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'KnowledgeGraph':
        """
        Import graph from JSON data.

        Args:
            json_data: Dictionary containing graph data

        Returns:
            New KnowledgeGraph instance
        """
        kg = cls()
        kg.layers = json_data.get('layers', kg.layers)
        kg.bridge_edges = json_data.get('bridge_edges', [])
        kg.metadata = json_data.get('metadata', {})
        kg._rebuild_indices()
        return kg

    # ==================== UTILITY ====================

    def get_layer_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics about each layer.

        Returns:
            Dictionary with node counts and edge counts by type for each layer
        """
        stats = {}

        for layer_name, layer in self.layers.items():
            edge_type_counts = defaultdict(int)
            for edge in layer['edges']:
                edge_type_counts[edge['edge_type']] += 1

            stats[layer_name] = {
                'node_count': len(layer['nodes']),
                'edge_count': len(layer['edges']),
                'edge_types': dict(edge_type_counts)
            }

        # Bridge stats
        bridge_type_counts = defaultdict(int)
        for edge in self.bridge_edges:
            bridge_type_counts[edge['edge_type']] += 1

        stats['bridge'] = {
            'edge_count': len(self.bridge_edges),
            'edge_types': dict(bridge_type_counts)
        }

        return stats

    def get_edge_type_info(self, edge_type: str, layer: str) -> Dict[str, Any]:
        """
        Get definition and properties of an edge type.

        Args:
            edge_type: Edge type name
            layer: Layer name

        Returns:
            Dictionary with edge type information

        Raises:
            KeyError: If edge type not found
        """
        if layer not in CANONICAL_EDGE_TYPES:
            raise KeyError(f"Layer '{layer}' not found in canonical edge types")

        if edge_type not in CANONICAL_EDGE_TYPES[layer]:
            raise KeyError(
                f"Edge type '{edge_type}' not found in layer '{layer}'. "
                f"Available: {list(CANONICAL_EDGE_TYPES[layer].keys())}"
            )

        return CANONICAL_EDGE_TYPES[layer][edge_type].copy()
