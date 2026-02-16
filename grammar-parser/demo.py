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
from engine import ExplainEngine, IntentFact, build_engine

with open("example_explain.json") as f:
    grammar = GrammarSpec(**json.load(f))

explain_spec = grammar.intents[IntentType.EXPLAIN]
compare_spec = grammar.intents[IntentType.COMPARE]
describe_spec = grammar.intents[IntentType.DESCRIBE]
explore_spec = grammar.intents[IntentType.EXPLORE]

intent_explain = IntentNode(
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

intent_compare = IntentNode(
    id="intent-2",
    intent_type=IntentType.COMPARE,
    state=IntentState.ACTIVE,
    parameters={
        "variable": ParameterInstance(
            name="variable",
            state=ParameterState.UNDERSPECIFIED
        ),
        "grouping_factor": ParameterInstance(
            name="grouping_factor",
            state=ParameterState.UNDERSPECIFIED
        )
    },
    created_at=time.time(),
    user_messages=[]
)

intent_describe = IntentNode(
    id="intent-3",
    intent_type=IntentType.DESCRIBE,
    state=IntentState.ACTIVE,
    parameters={
        "variable": ParameterInstance(
            name="variable",
            state=ParameterState.UNDERSPECIFIED
        )
    },
    created_at=time.time(),
    user_messages=[]
)

intent_explore = IntentNode(
    id="intent-4",
    intent_type=IntentType.EXPLORE,
    state=IntentState.ACTIVE,
    parameters={
        "topic": ParameterInstance(
            name="name",
            state=ParameterState.UNDERSPECIFIED
        )
    },
    created_at=time.time(),
    user_messages=[]
)



print("\n-----EXPLAIN INTENT-----:")
print("initial intent states:")
print(intent_explain)

engine = build_engine(intent_explain, grammar)
engine.reset()

# initial fact
engine.declare(
    IntentFact(
        intent_type=intent_explain.intent_type,
        outcome_state=intent_explain.parameters["outcome"].state,
        predictors_state=intent_explain.parameters["predictors"].state
    )
)

# simulate user grounding outcome
print("\n-- user grounds outcome")
intent_explain.parameters["outcome"].state = ParameterState.GROUNDED

engine.declare(
    IntentFact(
        intent_type=intent_explain.intent_type,
        outcome_state=intent_explain.parameters["outcome"].state,
        predictors_state=intent_explain.parameters["predictors"].state
    )
)

engine.run()

print("\nfinal intent state:")
print(intent_explain)

print("\n\n-----COMPARE INTENT-----:")
print("initial intent states:")
print(intent_compare)

engine = build_engine(intent_compare, grammar)
engine.reset()

# initial fact
engine.declare(
    IntentFact(
        intent_type=intent_compare.intent_type,
        variable_state=intent_compare.parameters["variable"].state,
        grouping_factor_state=intent_compare.parameters["grouping_factor"].state,
    )
)

# simulate user grounding outcome
print("\n-- user grounds outcome")
intent_compare.parameters["variable"].state = ParameterState.GROUNDED
intent_compare.parameters["grouping_factor"].state = ParameterState.GROUNDED

engine.declare(
    IntentFact(
        intent_type=intent_compare.intent_type,
        variable_state=intent_compare.parameters["variable"].state,
        grouping_factor_state=intent_compare.parameters["grouping_factor"].state,
    )
)

engine.run()

print("\nfinal intent state:")
print(intent_compare)

print("\n\n-----DESCRIBE INTENT-----:")
print("initial intent states:")
print(intent_describe)

engine = build_engine(intent_describe, grammar)
engine.reset()

# initial fact
engine.declare(
    IntentFact(
        intent_type=intent_describe.intent_type,
        variable_state=intent_describe.parameters["variable"].state,
    )
)

# simulate user grounding outcome
print("\n-- user grounds outcome")
intent_describe.parameters["variable"].state = ParameterState.GROUNDED

engine.declare(
    IntentFact(
        intent_type=intent_describe.intent_type,
        variable_state=intent_describe.parameters["variable"].state,    )
)

engine.run()

print("\nfinal intent state:")
print(intent_describe)

print("\n\n-----EXPLORE INTENT-----:")
print("initial intent states:")
print(intent_explore)

engine = build_engine(intent_explore, grammar)
engine.reset()

# this should not work as explore intent cannot be made terminal
# engine.declare(
#     IntentFact(
#         intent_type=intent_explore.intent_type,
#         topic_state=intent_explore.parameters["topic"].state,
#     )
# )

engine.run()

print("\nfinal intent state:")
print(intent_explore)
