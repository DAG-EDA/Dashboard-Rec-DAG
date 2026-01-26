# graph_builder.py
import json
from datetime import datetime
import uuid
from models import GraphNode, GraphEdge, GraphOutput, GraphMetadata

class VariableGraph:
    def __init__(self, analytical_question):
        self.analytical_question = analytical_question
        self.nodes: list[GraphNode] = []
        self.edges: list[GraphEdge] = []
        self.node_id_map = {}  # concept_name -> node_id mapping
    
    def add_node(self, concept_name: str, node_type: str, **attributes) -> str:
        """Add a node to the graph"""
        node_id = str(uuid.uuid4())
        
        node = GraphNode(
            id=node_id,
            type=node_type,
            concept_name=concept_name,
            **attributes
        )
        
        self.nodes.append(node)
        self.node_id_map[concept_name] = node_id
        return node_id
    
    def add_edge(self, source_name: str, target_name: str, edge_type: str, **metadata) -> GraphEdge:
        """Add an edge between two nodes"""
        source_id = self.node_id_map.get(source_name)
        target_id = self.node_id_map.get(target_name)
        
        if not source_id or not target_id:
            print(f"Warning: Could not find node for {source_name} or {target_name}")
            return None
        
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            source_name=source_name,
            target_name=target_name,
            edge_type=edge_type,
            metadata=metadata if metadata else None
        )
        
        self.edges.append(edge)
        return edge
    
    def to_pydantic(self) -> GraphOutput:
        """Export graph as Pydantic model"""
        return GraphOutput(
            analytical_question=self.analytical_question,
            metadata=GraphMetadata(
                generated_at=datetime.now(),
                num_nodes=len(self.nodes),
                num_edges=len(self.edges)
            ),
            nodes=self.nodes,
            edges=self.edges
        )
    
    def save_json(self, filepath: str):
        """Save graph to JSON file"""
        graph_output = self.to_pydantic()
        
        with open(filepath, 'w') as f:
            # Use Pydantic's JSON export
            f.write(graph_output.model_dump_json(indent=2))
        
        print(f"Graph saved to {filepath}")