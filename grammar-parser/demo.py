import json
import time
# from experta import Fact

from models import (
    GrammarSpec,
    IntentNode,
    IntentType,
    IntentState,
    ParameterInstance,
    ParameterState
)
from engine import ExplainEngine, IntentFact

with open("example_explain.json") as f:
    grammar = GrammarSpec(**json.load(f))

explain_spec = grammar.intents[IntentType.EXPLAIN]


intent = IntentNode(
    id="intent-1",
    intent_type=IntentType.EXPLAIN,
    state=IntentState.ACTIVE,
    parameters={
        "outcome": ParameterInstance(
            name="outcome",
            state=ParameterState.UNDERSPECIFIED
        ),
        "predictors": ParameterInstance(
            name="predictors",
            state=ParameterState.UNDERSPECIFIED
        )
    },
    created_at=time.time(),
    user_messages=[]
)

print("initial intent state:")
print(intent)

engine = ExplainEngine(intent, explain_spec)
engine.reset()

# initial fact
engine.declare(
    IntentFact(
        intent_type=intent.intent_type,
        outcome_state=intent.parameters["outcome"].state,
        predictors_state=intent.parameters["predictors"].state
    )
)

# simulate user grounding outcome
print("\n-- user grounds outcome")
intent.parameters["outcome"].state = ParameterState.GROUNDED

engine.declare(
    IntentFact(
        intent_type=intent.intent_type,
        outcome_state=intent.parameters["outcome"].state,
        predictors_state=intent.parameters["predictors"].state
    )
)

engine.run()
    
print("\nfinal intent state:")
print(intent)
