# models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

# Stage 1 Models
class ConceptualVariable(BaseModel):
    concept_name: str
    definition: str
    relevance: str
    domain: str
    variable_type: Literal["predictor", "outcome", "control", "mediator", "moderator"]

class Stage1Response(BaseModel):
    analytical_question: str
    conceptual_variables: List[ConceptualVariable]

# Stage 2 Models
class OperationalizedVariable(BaseModel):
    concept_name: str
    definition: str
    relevance: str
    variable_type: Literal["predictor", "outcome", "control", "mediator", "moderator"]
    operationalized_as: str  # "None" if split, otherwise the operationalization
    stem_from: str  # "None" if top-level, otherwise parent concept name

class Stage2Response(BaseModel):
    analytical_question: str
    conceptual_variables: List[OperationalizedVariable]

# Graph Models
class GraphNode(BaseModel):
    id: str
    type: Literal["conceptual", "operationalized"]
    concept_name: str
    definition: str
    relevance: str
    variable_type: str
    operationalized_as: Optional[str] = None

class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    source_name: str
    target_name: str
    edge_type: Literal["stem_from", "operationalized_as", "causes", "mediates", "moderates"]
    metadata: Optional[dict] = None

class GraphMetadata(BaseModel):
    generated_at: datetime
    num_nodes: int
    num_edges: int

class GraphOutput(BaseModel):
    analytical_question: str
    metadata: GraphMetadata
    nodes: List[GraphNode]
    edges: List[GraphEdge]