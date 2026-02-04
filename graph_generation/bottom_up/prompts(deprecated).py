# prompts.py
from models import Stage1Response, Stage2Response
import json

def get_stage1_prompt(analytical_question):
    """Generate Stage 1 prompt with JSON schema"""
    
    # Get JSON schema from Pydantic model
    schema = Stage1Response.model_json_schema()
    
    return f"""You are an experienced data expert tasked with identifying conceptual variables that may be relevant to a given analytical question.

A conceptual variable is an abstract idea or concept that experts are interested in studying. It represents a broad construct that cannot be directly observed or measured. Instead, they create operational definitions or measurable indicators that represent the conceptual variable.

Please identify all conceptual variables needed to comprehensively answer this analytical question. Think about: 
* The abstract nature of what you're trying to capture 
* How this concept relates to the analytical question 
* What practical domain or area of business this concept belongs to 

CRITICAL: You must respond with ONLY valid JSON, no additional text before or after. Use this exact structure:

{{
  "analytical_question": "<restate the analytical question>",
  "conceptual_variables": [
    {{
      "concept_name": "<name of the conceptual variable>",
      "definition": "<what this abstract concept represents>",
      "relevance": "<specific explanation of how this helps answer the analytical question>",
      "domain": "<practical business domain this concept belongs to>",
      "variable_type": "<predictor|outcome|control|mediator|moderator>"
    }}
  ]
}}

DEFINITIONS:
- Predictor: Variables that directly influence the outcome
- Outcome: What you're trying to understand or predict (may have multiple)
- Mediator: Variables that explain HOW predictors affect outcomes
- Moderator: Variables that affect WHEN or FOR WHOM relationships are stronger/weaker
- Control: Variables that need to be accounted for to isolate true relationships

Ensure conceptual completeness while maintaining parsimony - include all essential concepts but avoid redundancy.

ANALYTICAL QUESTION: {analytical_question}

Remember: Respond with ONLY the JSON object, nothing else."""


def get_stage2_prompt(stage1_response: Stage1Response):
    """Generate Stage 2 prompt with JSON schema"""
    
    # Convert Pydantic model to dict for JSON serialization
    stage1_dict = stage1_response.model_dump()
    
    return f"""For each conceptual variable, suggest reasonable ways to operationalize it. If there are multiple ways that are not redundant, consider whether the conceptual variable can be further split.

CRITICAL: You must respond with ONLY valid JSON, no additional text before or after. Use this exact structure:

{{
  "analytical_question": "<restate the analytical question>",
  "conceptual_variables": [
    {{
      "concept_name": "<name of the conceptual variable>",
      "definition": "<what this abstract concept represents>",
      "relevance": "<specific explanation of how this helps answer the analytical question>",
      "variable_type": "<predictor|outcome|control|mediator|moderator>",
      "operationalized_as": "<None|operational definitions or measurable indicators that represent the conceptual variable>",
      "stem_from": "<None|the conceptual_variable it split from>"
    }}
  ]
}}

IMPORTANT RULES:
1. For conceptual variables that can be operationalized in multiple distinct ways, split them into separate variables
2. For split variables, set operationalized_as to "None" for the parent, and create child variables with specific operationalizations
3. Child variables must set stem_from to the parent's concept_name
4. Top-level variables (not split from anything) should have stem_from set to "None"

Here is the input from Stage 1:

{json.dumps(stage1_dict, indent=2)}

Remember: Respond with ONLY the JSON object, nothing else."""