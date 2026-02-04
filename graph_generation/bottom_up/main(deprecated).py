# main.py
import json
from llm_client import GeminiClient
from prompts import get_stage1_prompt, get_stage2_prompt
from graph_builder import VariableGraph
from models import Stage1Response, Stage2Response

class VariableGraphPipeline:
    def __init__(self):
        self.client = GeminiClient()
        self.graph = None
    
    def run(self, analytical_question: str, output_file: str = "output_graph.json"):
        """Run the full pipeline"""
        print(f"Starting pipeline for question: {analytical_question}\n")
        
        # Initialize graph
        self.graph = VariableGraph(analytical_question)
        
        # Stage 1: Generate conceptual variables
        print("Stage 1: Generating conceptual variables...")
        stage1_result = self._stage1_conceptual_variables(analytical_question)
        print(f"✓ Generated {len(stage1_result.conceptual_variables)} conceptual variables\n")
        
        # Stage 2: Operationalize variables
        print("Stage 2: Operationalizing variables...")
        stage2_result = self._stage2_operationalize(stage1_result)
        print(f"✓ Generated {len(stage2_result.conceptual_variables)} operationalized variables\n")
        
        # Build graph
        print("Building graph structure...")
        self._build_graph(stage2_result)
        print(f"✓ Graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges\n")
        
        # Save
        self.graph.save_json(output_file)
        print(f"✓ Complete! Graph saved to {output_file}")
        
        return self.graph
    
    def _stage1_conceptual_variables(self, analytical_question: str) -> Stage1Response:
        """Stage 1: Generate high-level conceptual variables"""
        prompt = get_stage1_prompt(analytical_question)
        
        # Use Pydantic-aware parsing
        result = self.client.call_and_parse(
            prompt=prompt,
            response_model=Stage1Response,
            system_instruction="You are an expert data analyst. Respond only with valid JSON."
        )
        
        # Save result using Pydantic's JSON export
        with open("stage1_result.json", "w") as f:
            f.write(result.model_dump_json(indent=2))
        
        return result
    
    def _stage2_operationalize(self, stage1_result: Stage1Response) -> Stage2Response:
        """Stage 2: Operationalize and split variables"""
        prompt = get_stage2_prompt(stage1_result)
        
        # Use Pydantic-aware parsing
        result = self.client.call_and_parse(
            prompt=prompt,
            response_model=Stage2Response,
            system_instruction="You are an expert data analyst. Respond only with valid JSON."
        )
        
        # Save result
        with open("stage2_result.json", "w") as f:
            f.write(result.model_dump_json(indent=2))
        
        return result
    
    def _build_graph(self, stage2_result: Stage2Response):
        """Build graph from stage 2 results"""
        # Add all variables as nodes
        for var in stage2_result.conceptual_variables:
            node_type = "operationalized" if var.operationalized_as != "None" else "conceptual"
            
            self.graph.add_node(
                concept_name=var.concept_name,
                node_type=node_type,
                definition=var.definition,
                relevance=var.relevance,
                variable_type=var.variable_type,
                operationalized_as=var.operationalized_as if var.operationalized_as != "None" else None
            )
        
        # Add stem_from edges
        for var in stage2_result.conceptual_variables:
            if var.stem_from and var.stem_from != "None":
                self.graph.add_edge(
                    source_name=var.stem_from,
                    target_name=var.concept_name,
                    edge_type="stem_from"
                )


def main():
    # Example usage
    pipeline = VariableGraphPipeline()
    
    analytical_question = "What are the main characteristics of successful movies?"
    
    try:
        graph = pipeline.run(analytical_question, output_file="movie_success_graph.json")
        
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Nodes: {len(graph.nodes)}")
        print(f"Edges: {len(graph.edges)}")
        print("\nNode types:")
        conceptual = sum(1 for n in graph.nodes if n.type == 'conceptual')
        operationalized = sum(1 for n in graph.nodes if n.type == 'operationalized')
        print(f"  - Conceptual: {conceptual}")
        print(f"  - Operationalized: {operationalized}")
    
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        print("Check the error_response_*.txt files for debugging")
        raise


if __name__ == "__main__":
    main()