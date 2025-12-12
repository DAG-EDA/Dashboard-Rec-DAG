from graph_builder import VariableGraph
import json

class VisualizationSpec:
	def __init__(self, analytical_question, knowledge_graph=None, causal_graph=None, intent_tree=None):
		self.analytical_question = analytical_question
		self.intent_tree = intent_tree
		self.domain_knowledge_graph = knowledge_graph
		self.causal_graph = causal_graph

	def addKG(self, knowledge_graph):
		# Use Graph Builder to create a valid knowledge graph to save
		with open(knowledge_graph, 'r') as file:
			domain_kg = json.load(file)

		refined_kg = VariableGraph(self.analytical_question)

		for elem in domain_kg:
			if isinstance(elem,str):
				print(elem)
				refined_kg.add_node(elem)


	def addCG(self, causal_graph):
		# use graph builder to create valid causal graph to save
		pass

if __name__== '__main__':
	viz = VisualizationSpec("Movies?")
	viz.addKG("knowledge-graph.json")
