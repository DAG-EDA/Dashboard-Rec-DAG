from simpleeval import simple_eval
from experta import Fact, KnowledgeEngine, Rule
from typing import List
from models import IntentNode, IntentState, ParameterState, IntentType, GrammarSpec


class GrammarProcessor:
    '''
    processor for validating and reasoning over JSON intents according to
    a grammar spec defined in models.py
    '''
    def __init__(self, grammar: GrammarSpec):
        self.grammar = grammar
    
    def validate_intent(self, intent: IntentNode) -> List[str]:
        '''
        validate intent against its grammar definition
        '''
        errors = []
        signature = self.grammar.intents[intent.intent_type]

        # required parameters check
        for name, spec in signature.parameters.items():
            if spec.required and name not in intent.parameters:
                errors.append(f"missing required parameter: {name}")

        # dependencies check
        for name, spec in signature.parameters.items():
            if spec.depends_on:
                param = intent.parameters.get(name)
                dep = intent.parameters.get(spec.depends_on)
                if param and param.state != ParameterState.UNDERSPECIFIED:
                    if dep.state != ParameterState.GROUNDED:
                        errors.append(
                            f"{name} cannot be bound before {spec.depends_on}"
                        )

        return errors

    def _terminal_context(self, intent: IntentNode) -> dict:
        '''
        build a context dictionary for evaluating terminal conditions
        '''
        ctx = {}
        for name, param in intent.parameters.items():
            ctx[name] = param
        ctx["auto_discoverable"] = lambda p: len(p.candidates) > 0
        return ctx

    def is_terminal(self, intent: IntentNode) -> bool:
        '''
        determine whether an intent has reached terminal state
        '''
        signature = self.grammar.intents[intent.intent_type]
        ctx = self._terminal_context(intent)
        return simple_eval(signature.terminal_condition, ctx)

class IntentFact(Fact):
    '''
    experta fact representing state of intent
    '''
    intent_id: str
    intent_type: IntentType

class ParameterFact(Fact):
    '''
    experta fact representing state of a parameter
    '''
    intent_id: str
    name: str
    state: ParameterState

class GrammarRuleEngine(KnowledgeEngine):
    '''
    defines the system based on experta for reasoning over intents
    '''
    def __init__(self, grammar: GrammarSpec):
        super().__init__()
        self.grammar = grammar
        self.triggered_actions = []


class ExplainEngine(KnowledgeEngine):
    '''
    reasoning engine for EXPLAIN intent built on top of experta's KnowledgeEngine
    - manages the life cycle of a single EXPLAIN intent
    - todo: extend to other intent types

    rules:
    - auto-discovery of predictors when outcome is grounded
        - if allow_auto_discovery
            - automatically discovers candidate predictors
            - grounds predictors when candidates are found
        
    - terminal condition check
        - evaluates whether EXPLAIN intent has reached terminal state
        - if so, transition intent state to TERMINAL and halt engine

    - relationship suggestions
        - suggests cross-intent relationships
    '''

    def __init__(self, intent: IntentNode, intent_spec):
        super().__init__()
        self.intent = intent
        self.spec = intent_spec

    # auto discovery
    @Rule(
        IntentFact(
            intent_type=IntentType.EXPLAIN,
            outcome_state=ParameterState.GROUNDED
        )
    )
    def auto_discover_predictors(self):
        predictors = self.intent.parameters["predictors"]

        if predictors.state != ParameterState.GROUNDED:
            print("-- auto-discovering predictors...")
            predictors.state = ParameterState.GROUNDED

    # terminal check
    @Rule(IntentFact(intent_type=IntentType.EXPLAIN))
    def check_terminal(self):
        params = self.intent.parameters

        ctx = {
            "outcome": params["outcome"],
            "predictors": params["predictors"],
            "auto_discover": self.spec.allow_auto_discovery
        }

        if simple_eval(self.spec.terminal_condition, names=ctx):
            print("-- EXPLAIN intent reached TERMINAL")
            self.intent.state = IntentState.TERMINAL
            self.halt()


def relationship_suggestions(self, intents: List[IntentNode]) -> List[dict]:
    suggestions = []

    for rule in self.grammar.relationship_rules:
        sources = [i for i in intents if i.intent_type == rule.source_intent]
        targets = [i for i in intents if i.intent_type == rule.target_intent]

        for src in sources:
            for tgt in targets:
                ctx = {
                    "source": src,
                    "target": tgt
                }
                if simple_eval(rule.condition, ctx):
                    suggestions.append({
                        "type": rule.relationship_type,
                        "source": src.id,
                        "target": tgt.id,
                        "description": rule.description
                    })

    return suggestions

    
