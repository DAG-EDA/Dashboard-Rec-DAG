from enum import Enum
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, model_validator

# enums

class IntentType(str, Enum):
    '''
    intent types supported by the grammar
    '''

    EXPLORE = "EXPLORE"
    EXPLAIN = "EXPLAIN"
    COMPARE = "COMPARE"
    COMPARE_OUTCOMES = "COMPARE_OUTCOMES"
    DESCRIBE = "DESCRIBE"
    PREDICT = "PREDICT"

class ParameterState(str, Enum):
    '''
    parameter states supported by the grammar
    '''
        
    UNDERSPECIFIED = "UNDERSPECIFIED"
    AMBIGUOUS = "AMBIGUOUS"
    PARTIAL = "PARTIAL"
    GROUNDED = "GROUNDED"

class DataType(str, Enum):
    '''
    data types supported by the grammar
    '''
        
    NUMERICAL = "NUMERICAL"
    CATEGORICAL = "CATEGORICAL"

class RelationshipType(str, Enum):
    '''
    types of relationships between two intents
    '''

    DECOMPOSES_TO = "DECOMPOSES_TO"
    REFINES_TO = "REFINES_TO"
    COMPOSES_TO = "COMPOSES_TO"
    CONTEXTUALIZES_WITH = "CONTEXTUALIZES_WITH"

class IntentState(str, Enum):
    '''
    runtime state of an intent node
    '''

    ACTIVE = "ACTIVE"
    TERMINAL = "TERMINAL"
    DECOMPOSED = "DECOMPOSED"
    SUGGESTED = "SUGGESTED"
    REJECTED = "REJECTED"
    DORMANT = "DORMANT"

class ParameterSpec(BaseModel):
    '''
    specifies a parameter in an intent signature
    example parameters: OUTCOME, PREDICTOR, GROUPING_FACTOR, etc
    '''

    name: str
    required: bool = True

    # constraints
    
    # allowed data types
    allowed_types: List[DataType]

    # min/max cardinality
    min_cardinality: Optional[int] = None
    max_cardinality: Optional[int] = None

    # dependency on another parameter
    depends_on: Optional[str] = None  # parameter name

    # semantic role specified by discovery rules
    role: Optional[str] = None  # OUTCOME, PREDICTOR, GROUPING_FACTOR

class IntentSignature(BaseModel):
    '''
    full spec of an intent type

    intent_type: EXPLAIN, COMPARE, etc
    parameters: parameters (as specified by ParameterSpec)
    terminal_condition: boolean expr to evaluate if intent is terminal
    allow_auto_discovery: whether auto-discovery of parameters is allowed
    '''
    
    intent_type: IntentType
    parameters: Dict[str, ParameterSpec]
    terminal_condition: str
    allow_auto_discovery: bool = False

    @model_validator(mode="after")
    def validate_dependencies(self):
        for p in self.parameters.values():
            if p.depends_on and p.depends_on not in self.parameters:
                raise ValueError(
                    f"unknown dependency: '{p.name}' depends_on '{p.depends_on}'"
                )
        return self

class PropagationRule(BaseModel):
    '''
    defines a rule for propagating changes within an intent
    '''
    intent_type: IntentType
    trigger: str  # boolean expression
    action: str   # action (ex. "DISCOVER_PREDICTORS")

class RelationshipRule(BaseModel):
    '''
    defines a rule for relationships btwn two intents

    relationship_type: DECOMPOSES_TO, REFINES_TO, etc
    source_intent: intent type of source intent
    target_intent: intent type of target intent
    condition: boolean expression to evaluate if relationship holds
    mandatory: whether relationship is mandatory
    description: human-readable description of relationship
    '''
    relationship_type: RelationshipType
    source_intent: IntentType
    target_intent: IntentType

    condition: str  # boolean expression
    mandatory: bool = False
    description: Optional[str]

# grammar container
class GrammarSpec(BaseModel):
    '''
    container for full grammar spec including intents, propagation rules, relationship rules
    '''
    intents: Dict[IntentType, IntentSignature]
    propagation_rules: List[PropagationRule]
    relationship_rules: List[RelationshipRule]

# runtime states
class CandidateBinding(BaseModel):
    '''
    candidate mapping between a parameter and a concept/data variable, and confidence score

    concept_id: concept being considered -> from KG
    data_variables: list of attributes from dataset associated with this candidate
    score: confidence score for this candidate binding
    '''
    concept_id: str
    data_variables: List[str]
    score: float

class ParameterInstance(BaseModel):
    '''
    runtime state of a parameter in an intent node

    name: parameter name
    state: UNDERSPECIFIED, GROUNDED, etc.
    concept_binding: KG concepts bound to this parameter
    data_binding: data variables bound to this parameter
    candidates: list of candidate bindings under consideration
    '''
    name: str
    state: ParameterState

    concept_binding: Optional[Union[str, List[str]]] = None
    data_binding: Optional[List[str]] = None

    candidates: List[CandidateBinding] = Field(default_factory=list)

class IntentNode(BaseModel):
    '''
    single intent node in intent tree in the system -> tracks all of the above info

    id: unique id for intent
    intent_type: EXPLAIN, COMPARE, etc
    state: current IntentState
    parameters: ParameterInstance objects related to the intent
    parent: parent of this intent in intent tree

    activated_concepts, activated_data, activated_edges: together form activated subgraph
    '''
    id: str
    intent_type: IntentType
    state: IntentState

    parameters: Dict[str, ParameterInstance]

    parent: Optional[str] = None

    activated_concepts: List[str] = Field(default_factory=list)
    activated_data: List[str] = Field(default_factory=list)
    activated_edges: List[str] = Field(default_factory=list)


