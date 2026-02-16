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

class BaseIntentEngine(KnowledgeEngine):
    '''
    reasoning engine for generic intent built on top of experta's KnowledgeEngine
    - manages the life cycle of a single intent
    - todo: subclass for specific intent types

    rules:
    - auto-discovery of predictors when outcome is grounded
        - if allow_auto_discovery
            - automatically discovers candidate predictors
            - grounds predictors when candidates are found

    - terminal condition check
        - evaluates whether intent has reached terminal state
        - if so, transition intent state to TERMINAL and halt engine

    - relationship suggestions
        - suggests cross-intent relationships
    '''

    def __init__(self, intent: IntentNode, intent_spec):
        super().__init__()
        self.intent = intent
        self.spec = intent_spec

    def _build_ctx(self):
        return {
            name: param
            for name, param in self.intent.parameters.items()
        }

    def check_terminal_common(self):
        ctx = self._build_ctx()
        if simple_eval(self.spec.terminal_condition, names=ctx):
            print(f"-- {self.intent.intent_type} intent reached TERMINAL")
            self.intent.state = IntentState.TERMINAL
            self.halt()


class ExplainEngine(BaseIntentEngine):
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
        self.check_terminal_common()

class ExploreEngine(BaseIntentEngine):
    @Rule(IntentFact(intent_type=IntentType.EXPLORE))
    def terminal_check(self):
        self.check_terminal_common()

class DescribeEngine(BaseIntentEngine):
    @Rule(IntentFact(intent_type=IntentType.DESCRIBE))
    def terminal_check(self):
        self.check_terminal_common()

class PredictEngine(BaseIntentEngine):
    @Rule(
        IntentFact(intent_type=IntentType.PREDICT)
    )
    def terminal_check(self):
        self.check_terminal_common()

class CompareEngine(BaseIntentEngine):
    @Rule(IntentFact(intent_type=IntentType.COMPARE))
    def terminal_check(self):
        self.check_terminal_common()

class CompareOutcomesEngine(BaseIntentEngine):
    @Rule(IntentFact(intent_type=IntentType.COMPARE_OUTCOMES))
    def terminal_check(self):
        self.check_terminal_common()

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

ENGINE_REGISTRY = {
    IntentType.EXPLORE: ExploreEngine,
    IntentType.EXPLAIN: ExplainEngine,
    IntentType.DESCRIBE: DescribeEngine,
    IntentType.PREDICT: PredictEngine,
    IntentType.COMPARE: CompareEngine,
    IntentType.COMPARE_OUTCOMES: CompareOutcomesEngine
}


def build_engine(intent: IntentNode, grammar: GrammarSpec):
    spec = grammar.intents[intent.intent_type]
    engine_cls = ENGINE_REGISTRY[intent.intent_type]
    return engine_cls(intent, spec)


