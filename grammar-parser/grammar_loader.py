'''
loads grammar from json spec

json file format:

{
  "intents": {
    ...  // as defined in intent_grammar.IntentSignature
  },

  "propagation_rules": [
    ... // as defined in intent_grammar.PropagationRule
  ],

  "relationship_rules": [
    ... // as defined in intent_grammar.RelationshipRule
  ]
}

see grammar.json for more info
'''

import json
from models import GrammarSpec
from pathlib import Path

with open(Path("grammar.json")) as f:
    grammar = GrammarSpec(**json.load(f))