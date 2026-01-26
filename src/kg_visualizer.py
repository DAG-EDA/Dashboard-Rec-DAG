"""
Knowledge Graph Visualizer

Interactive visualization with layer toggling using Pyvis and Plotly.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from src.graph_utils import KnowledgeGraph, CANONICAL_EDGE_TYPES


class KGVisualizer:
    """Visualizes knowledge graphs with interactive layer toggling."""

    def __init__(self, kg: KnowledgeGraph):
        """
        Initialize visualizer.

        Args:
            kg: Knowledge graph to visualize
        """
        self.kg = kg

        # Color scheme for layers
        self.layer_colors = {
            'data': '#3498db',      # Blue
            'concept': '#e74c3c',   # Red
            'question': '#2ecc71',  # Green
            'bridge': '#95a5a6'     # Gray
        }

        # Edge colors by type
        self.edge_colors = {
            # Data layer
            'causal_influence': '#e67e22',    # Orange
            'computed_from': '#9b59b6',       # Purple
            'associated_with': '#1abc9c',     # Teal (renamed from correlates_with)
            'correlates_with': '#1abc9c',     # Teal (legacy, for backward compatibility)

            # Concept layer
            'part_of': '#f39c12',             # Yellow
            'has_part': '#f39c12',
            'similar_to': '#16a085',
            'contrasts_with': '#c0392b',
            'prerequisite_for': '#8e44ad',
            'requires': '#8e44ad',

            # Bridge layer
            'operationalized_by': '#34495e'   # Dark gray
        }

    def generate_pyvis_html(
        self,
        output_path: Path,
        layers: Optional[List[str]] = None,
        show_bridge: bool = True,
        physics_enabled: bool = True,
        title: str = "Knowledge Graph"
    ) -> None:
        """
        Generate interactive HTML visualization using Pyvis.

        Args:
            output_path: Path to save HTML file
            layers: List of layers to show (None = all)
            show_bridge: Whether to show bridge edges
            physics_enabled: Enable physics simulation
            title: Graph title
        """
        try:
            from pyvis.network import Network
        except ImportError:
            print("⚠️  Pyvis not installed. Install with: pip install pyvis")
            return

        # Create network
        net = Network(
            height='800px',
            width='100%',
            bgcolor='#ffffff',
            font_color='#000000',
            directed=True
        )

        # Configure physics
        if physics_enabled:
            net.set_options("""
            {
              "physics": {
                "forceAtlas2Based": {
                  "gravitationalConstant": -50,
                  "centralGravity": 0.01,
                  "springLength": 200,
                  "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {"iterations": 150}
              }
            }
            """)

        # Determine which layers to show
        if layers is None:
            layers = ['data', 'concept', 'question']

        # Add nodes
        node_count = 0
        for layer in layers:
            layer_nodes = self.kg.layers.get(layer, {}).get('nodes', {})

            for node_id, node_data in layer_nodes.items():
                # Node properties
                label = node_data.get('name', node_id)
                color = self.layer_colors.get(layer, '#95a5a6')

                # Build hover text
                hover_text = f"<b>{label}</b><br>"
                hover_text += f"Layer: {layer}<br>"
                if 'description' in node_data:
                    hover_text += f"<br>{node_data['description'][:100]}"

                # Add node
                net.add_node(
                    node_id,
                    label=label,
                    title=hover_text,
                    color=color,
                    size=25,
                    shape='dot'
                )
                node_count += 1

        # Add layer edges
        edge_count = 0
        for layer in layers:
            layer_edges = self.kg.layers.get(layer, {}).get('edges', [])

            for edge in layer_edges:
                source = edge['source']
                target = edge['target']
                edge_type = edge['edge_type']

                # Skip if nodes not in selected layers
                if source not in [n for n in net.nodes]:
                    continue
                if target not in [n for n in net.nodes]:
                    continue

                # Edge properties
                color = self.edge_colors.get(edge_type, '#95a5a6')
                edge_info = CANONICAL_EDGE_TYPES.get(layer, {}).get(edge_type, {})

                # Build hover text
                hover_text = f"{edge_type}<br>"
                if 'definition' in edge_info:
                    hover_text += f"{edge_info['definition'][:100]}"

                # Add edge with label
                net.add_edge(
                    source,
                    target,
                    title=hover_text,
                    label=edge_type,
                    color=color,
                    arrows='to',
                    width=2,
                    font={'size': 10, 'color': '#666666'}
                )
                edge_count += 1

        # Add bridge edges
        if show_bridge:
            for edge in self.kg.bridge_edges:
                source = edge['source']
                target = edge['target']
                edge_type = edge['edge_type']

                # Skip if nodes not visible
                if source not in [n for n in net.nodes]:
                    continue
                if target not in [n for n in net.nodes]:
                    continue

                # Add edge with label
                net.add_edge(
                    source,
                    target,
                    title=f"Bridge: {edge_type}",
                    label=edge_type,
                    color=self.edge_colors.get(edge_type, '#34495e'),
                    arrows='to',
                    width=1,
                    dashes=True,
                    font={'size': 10, 'color': '#666666'}
                )
                edge_count += 1

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        net.show(str(output_path))

        print(f"✓ Visualization saved: {output_path}")
        print(f"  Nodes: {node_count}, Edges: {edge_count}")

    def generate_html_with_toggles(
        self,
        output_path: Path,
        title: str = "Knowledge Graph Explorer"
    ) -> None:
        """
        Generate HTML with layer toggle controls.

        Args:
            output_path: Path to save HTML file
            title: Page title
        """
        # Prepare graph data
        graph_data = self._prepare_graph_data()

        # Generate HTML
        html = self._generate_toggle_html(graph_data, title)

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html)

        print(f"✓ Interactive visualization saved: {output_path}")

    def _prepare_graph_data(self) -> Dict[str, Any]:
        """Prepare graph data for visualization."""
        nodes = []
        edges = []
        edge_types = set()

        # Add nodes
        for layer_name, layer in self.kg.layers.items():
            for node_id, node_data in layer['nodes'].items():
                nodes.append({
                    'id': node_id,
                    'label': node_data.get('name', node_id),
                    'layer': layer_name,
                    'color': self.layer_colors.get(layer_name, '#95a5a6'),
                    'description': node_data.get('description', ''),
                    'data': node_data
                })

        # Add layer edges
        for layer_name, layer in self.kg.layers.items():
            for edge in layer['edges']:
                edge_type = edge['edge_type']
                edge_types.add(edge_type)
                edges.append({
                    'source': edge['source'],
                    'target': edge['target'],
                    'type': edge_type,
                    'layer': layer_name,
                    'color': self.edge_colors.get(edge_type, '#95a5a6'),
                    'metadata': edge.get('metadata', {})
                })

        # Add bridge edges
        for edge in self.kg.bridge_edges:
            edge_type = edge['edge_type']
            edge_types.add(edge_type)
            edges.append({
                'source': edge['source'],
                'target': edge['target'],
                'type': edge_type,
                'layer': 'bridge',
                'color': self.edge_colors.get(edge_type, '#34495e'),
                'metadata': edge.get('metadata', {}),
                'dashed': True
            })

        return {'nodes': nodes, 'edges': edges, 'edge_types': sorted(edge_types)}

    def _generate_toggle_html(self, graph_data: Dict[str, Any], title: str) -> str:
        """Generate HTML with layer toggle controls."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        h1 {{
            margin-top: 0;
            color: #333;
        }}

        .controls {{
            margin-bottom: 20px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 5px;
        }}

        .control-group {{
            margin-bottom: 10px;
        }}

        .control-group label {{
            font-weight: bold;
            margin-right: 10px;
        }}

        .checkbox-group {{
            display: inline-block;
            margin-right: 20px;
        }}

        .checkbox-group input {{
            margin-right: 5px;
        }}

        #graph {{
            border: 1px solid #ddd;
            border-radius: 5px;
            background: white;
        }}

        .legend {{
            margin-top: 15px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 5px;
        }}

        .legend-item {{
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 5px;
        }}

        .legend-color {{
            display: inline-block;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            margin-right: 5px;
            vertical-align: middle;
        }}

        .stats {{
            margin-top: 10px;
            padding: 10px;
            background: #e8f4f8;
            border-radius: 5px;
            font-size: 14px;
        }}

        .tooltip {{
            position: absolute;
            padding: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 5px;
            pointer-events: none;
            font-size: 12px;
            max-width: 300px;
            z-index: 1000;
        }}

        .modal {{
            display: none;
            position: fixed;
            z-index: 2000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
        }}

        .modal-content {{
            background-color: #fefefe;
            margin: 10% auto;
            padding: 20px;
            border: 1px solid #888;
            border-radius: 8px;
            width: 60%;
            max-width: 600px;
        }}

        .close {{
            color: #aaa;
            float: right;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }}

        .close:hover,
        .close:focus {{
            color: #000;
        }}

        .edge-label {{
            font-size: 8px;
            fill: #666;
            pointer-events: none;
            user-select: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>

        <div class="controls">
            <div class="control-group">
                <label>Layers:</label>
                <div class="checkbox-group">
                    <input type="checkbox" id="layer-data" checked>
                    <label for="layer-data">Data Layer</label>
                </div>
                <div class="checkbox-group">
                    <input type="checkbox" id="layer-concept" checked>
                    <label for="layer-concept">Concept Layer</label>
                </div>
                <div class="checkbox-group">
                    <input type="checkbox" id="layer-question" checked>
                    <label for="layer-question">Question Layer</label>
                </div>
                <div class="checkbox-group">
                    <input type="checkbox" id="layer-bridge" checked>
                    <label for="layer-bridge">Bridge Edges</label>
                </div>
            </div>

            <div class="control-group">
                <label>Edge Types:</label>
                <button onclick="toggleAllEdgeTypes(true)" style="margin-left: 10px;">Select All</button>
                <button onclick="toggleAllEdgeTypes(false)">Deselect All</button>
                <div id="edge-type-filters" style="margin-top: 10px;">
                    <!-- Edge type checkboxes will be generated here -->
                </div>
            </div>

            <div class="stats" id="stats"></div>
        </div>

        <svg id="graph" width="1360" height="700"></svg>

        <div class="legend">
            <div style="margin-bottom: 15px;">
                <strong>Node Layers:</strong><br>
                <div class="legend-item">
                    <span class="legend-color" style="background-color: {self.layer_colors['data']}"></span>
                    Data Layer
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background-color: {self.layer_colors['concept']}"></span>
                    Concept Layer
                </div>
                <div class="legend-item">
                    <span class="legend-color" style="background-color: {self.layer_colors['question']}"></span>
                    Question Layer
                </div>
            </div>
            <div>
                <strong>Edge Types:</strong> <span style="font-size: 12px; color: #666;">(Click edge for details)</span><br>
                <div id="edge-legend">
                    <!-- Edge type legend will be generated here -->
                </div>
            </div>
        </div>
    </div>

    <div class="tooltip" id="tooltip" style="display: none;"></div>

    <!-- Modal for edge details -->
    <div id="edge-modal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeEdgeModal()">&times;</span>
            <h2 id="modal-title">Edge Details</h2>
            <div id="modal-body"></div>
        </div>
    </div>

    <script>
        // Graph data
        const graphData = {json.dumps(graph_data)};

        // SVG setup
        const svg = d3.select("#graph");
        const width = +svg.attr("width");
        const height = +svg.attr("height");

        const g = svg.append("g");

        // Zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});

        svg.call(zoom);

        // Simulation
        const simulation = d3.forceSimulation()
            .force("link", d3.forceLink().id(d => d.id).distance(150))
            .force("charge", d3.forceManyBody().strength(-400))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(30));

        // State
        let visibleLayers = {{
            'data': true,
            'concept': true,
            'question': true,
            'bridge': true
        }};

        // Initialize edge type filters
        let visibleEdgeTypes = {{}};
        graphData.edge_types.forEach(type => {{
            visibleEdgeTypes[type] = true;
        }});

        // Generate edge type filters and legend
        function initializeEdgeTypeControls() {{
            const filterContainer = document.getElementById('edge-type-filters');
            const legendContainer = document.getElementById('edge-legend');

            graphData.edge_types.forEach(type => {{
                // Create checkbox
                const checkboxGroup = document.createElement('div');
                checkboxGroup.className = 'checkbox-group';
                checkboxGroup.innerHTML = `
                    <input type="checkbox" id="edge-type-${{type}}" checked>
                    <label for="edge-type-${{type}}">${{type.replace(/_/g, ' ')}}</label>
                `;
                filterContainer.appendChild(checkboxGroup);

                // Add event listener
                const checkbox = checkboxGroup.querySelector('input');
                checkbox.addEventListener('change', (e) => {{
                    visibleEdgeTypes[type] = e.target.checked;
                    updateGraph();
                }});

                // Create legend item
                const edgeColor = graphData.edges.find(e => e.type === type)?.color || '#999';
                const legendItem = document.createElement('div');
                legendItem.className = 'legend-item';
                legendItem.innerHTML = `
                    <span class="legend-color" style="background-color: ${{edgeColor}}"></span>
                    ${{type.replace(/_/g, ' ')}}
                `;
                legendContainer.appendChild(legendItem);
            }});
        }}

        function toggleAllEdgeTypes(checked) {{
            graphData.edge_types.forEach(type => {{
                visibleEdgeTypes[type] = checked;
                document.getElementById(`edge-type-${{type}}`).checked = checked;
            }});
            updateGraph();
        }}

        // Initialize
        function updateGraph() {{
            // Filter nodes and edges
            const visibleNodes = graphData.nodes.filter(n => visibleLayers[n.layer]);
            const visibleNodeIds = new Set(visibleNodes.map(n => n.id));

            const visibleEdges = graphData.edges.filter(e =>
                visibleNodeIds.has(e.source.id || e.source) &&
                visibleNodeIds.has(e.target.id || e.target) &&
                (e.layer !== 'bridge' || visibleLayers.bridge) &&
                visibleEdgeTypes[e.type]
            );

            // Update stats
            updateStats(visibleNodes, visibleEdges);

            // Clear
            g.selectAll("*").remove();

            // Add arrow markers
            const defs = g.append("defs");
            defs.selectAll("marker")
                .data(["arrow"])
                .enter().append("marker")
                .attr("id", d => d)
                .attr("viewBox", "0 -5 10 10")
                .attr("refX", 20)
                .attr("refY", 0)
                .attr("markerWidth", 6)
                .attr("markerHeight", 6)
                .attr("orient", "auto")
                .append("path")
                .attr("d", "M0,-5L10,0L0,5")
                .attr("fill", "#999");

            // Create links
            const link = g.append("g")
                .selectAll("line")
                .data(visibleEdges)
                .enter().append("line")
                .attr("stroke", d => d.color)
                .attr("stroke-width", 2)
                .attr("stroke-dasharray", d => d.dashed ? "5,5" : "0")
                .attr("marker-end", "url(#arrow)")
                .style("cursor", "pointer")
                .on("click", showEdgeDetails)
                .on("mouseover", function(event, d) {{
                    d3.select(this).attr("stroke-width", 4);
                }})
                .on("mouseout", function(event, d) {{
                    d3.select(this).attr("stroke-width", 2);
                }});

            // Create nodes
            const node = g.append("g")
                .selectAll("circle")
                .data(visibleNodes)
                .enter().append("circle")
                .attr("r", 10)
                .attr("fill", d => d.color)
                .attr("stroke", "#fff")
                .attr("stroke-width", 2)
                .call(drag(simulation))
                .on("mouseover", showTooltip)
                .on("mouseout", hideTooltip);

            // Create node labels
            const label = g.append("g")
                .selectAll("text")
                .data(visibleNodes)
                .enter().append("text")
                .text(d => d.label)
                .attr("font-size", "10px")
                .attr("dx", 12)
                .attr("dy", 4);

            // Create edge labels (with improved positioning to reduce overlap)
            const edgeLabel = g.append("g")
                .selectAll("text")
                .data(visibleEdges)
                .enter().append("text")
                .text(d => d.type.replace(/_/g, ' '))
                .attr("class", "edge-label")
                .attr("text-anchor", "middle")
                .style("cursor", "pointer")
                .on("click", showEdgeDetails);

            // Update simulation
            simulation.nodes(visibleNodes);
            simulation.force("link").links(visibleEdges);
            simulation.alpha(1).restart();

            simulation.on("tick", () => {{
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);

                label
                    .attr("x", d => d.x)
                    .attr("y", d => d.y);

                edgeLabel
                    .attr("x", d => (d.source.x + d.target.x) / 2)
                    .attr("y", d => (d.source.y + d.target.y) / 2);
            }});
        }}

        function updateStats(nodes, edges) {{
            const stats = document.getElementById('stats');
            const byLayer = {{}};

            nodes.forEach(n => {{
                byLayer[n.layer] = (byLayer[n.layer] || 0) + 1;
            }});

            stats.innerHTML = `<strong>Visible:</strong> ${{nodes.length}} nodes, ${{edges.length}} edges | ` +
                `Data: ${{byLayer.data || 0}}, Concept: ${{byLayer.concept || 0}}, Question: ${{byLayer.question || 0}}`;
        }}

        function showTooltip(event, d) {{
            const tooltip = document.getElementById('tooltip');
            tooltip.style.display = 'block';
            tooltip.style.left = (event.pageX + 10) + 'px';
            tooltip.style.top = (event.pageY - 10) + 'px';
            tooltip.innerHTML = `<strong>${{d.label}}</strong><br>Layer: ${{d.layer}}<br>${{d.description.substring(0, 100)}}`;
        }}

        function hideTooltip() {{
            document.getElementById('tooltip').style.display = 'none';
        }}

        function showEdgeDetails(event, d) {{
            event.stopPropagation();
            const modal = document.getElementById('edge-modal');
            const modalTitle = document.getElementById('modal-title');
            const modalBody = document.getElementById('modal-body');

            // Get source and target node names
            const sourceNode = graphData.nodes.find(n => n.id === (d.source.id || d.source));
            const targetNode = graphData.nodes.find(n => n.id === (d.target.id || d.target));

            modalTitle.textContent = `Edge: ${{d.type.replace(/_/g, ' ')}}`;

            let bodyHTML = `
                <p><strong>From:</strong> ${{sourceNode?.label || 'Unknown'}}</p>
                <p><strong>To:</strong> ${{targetNode?.label || 'Unknown'}}</p>
                <p><strong>Type:</strong> ${{d.type.replace(/_/g, ' ')}}</p>
                <p><strong>Layer:</strong> ${{d.layer}}</p>
            `;

            // Add metadata
            if (d.metadata) {{
                if (d.metadata.confidence !== undefined) {{
                    bodyHTML += `<p><strong>Confidence:</strong> ${{d.metadata.confidence}}</p>`;
                }}
                if (d.metadata.reasoning) {{
                    bodyHTML += `<p><strong>Reasoning:</strong></p><p style="background: #f5f5f5; padding: 10px; border-radius: 5px;">${{d.metadata.reasoning}}</p>`;
                }}
                if (d.metadata.strategy) {{
                    bodyHTML += `<p><strong>Strategy:</strong> ${{d.metadata.strategy}}</p>`;
                }}
            }}

            modalBody.innerHTML = bodyHTML;
            modal.style.display = 'block';
        }}

        function closeEdgeModal() {{
            document.getElementById('edge-modal').style.display = 'none';
        }}

        // Close modal when clicking outside
        window.onclick = function(event) {{
            const modal = document.getElementById('edge-modal');
            if (event.target === modal) {{
                modal.style.display = 'none';
            }}
        }}

        function drag(simulation) {{
            function dragstarted(event, d) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }}

            function dragged(event, d) {{
                d.fx = event.x;
                d.fy = event.y;
            }}

            function dragended(event, d) {{
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }}

            return d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended);
        }}

        // Event listeners
        document.getElementById('layer-data').addEventListener('change', (e) => {{
            visibleLayers.data = e.target.checked;
            updateGraph();
        }});

        document.getElementById('layer-concept').addEventListener('change', (e) => {{
            visibleLayers.concept = e.target.checked;
            updateGraph();
        }});

        document.getElementById('layer-question').addEventListener('change', (e) => {{
            visibleLayers.question = e.target.checked;
            updateGraph();
        }});

        document.getElementById('layer-bridge').addEventListener('change', (e) => {{
            visibleLayers.bridge = e.target.checked;
            updateGraph();
        }});

        // Initialize edge type controls and legend
        initializeEdgeTypeControls();

        // Initial render
        updateGraph();
    </script>
</body>
</html>
"""
        return html

    def export_to_graphml(self, output_path: Path) -> None:
        """
        Export graph to GraphML format (for Gephi, Cytoscape, etc.).

        Args:
            output_path: Path to save GraphML file
        """
        graphml = ['<?xml version="1.0" encoding="UTF-8"?>']
        graphml.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">')
        graphml.append('  <graph id="KG" edgedefault="directed">')

        # Add nodes
        for layer_name, layer in self.kg.layers.items():
            for node_id, node_data in layer['nodes'].items():
                graphml.append(f'    <node id="{node_id}">')
                graphml.append(f'      <data key="layer">{layer_name}</data>')
                graphml.append(f'      <data key="name">{node_data.get("name", "")}</data>')
                graphml.append('    </node>')

        # Add edges
        edge_id = 0
        for layer_name, layer in self.kg.layers.items():
            for edge in layer['edges']:
                graphml.append(f'    <edge id="e{edge_id}" source="{edge["source"]}" target="{edge["target"]}">')
                graphml.append(f'      <data key="type">{edge["edge_type"]}</data>')
                graphml.append(f'      <data key="layer">{layer_name}</data>')
                graphml.append('    </edge>')
                edge_id += 1

        # Bridge edges
        for edge in self.kg.bridge_edges:
            graphml.append(f'    <edge id="e{edge_id}" source="{edge["source"]}" target="{edge["target"]}">')
            graphml.append(f'      <data key="type">{edge["edge_type"]}</data>')
            graphml.append(f'      <data key="layer">bridge</data>')
            graphml.append('    </edge>')
            edge_id += 1

        graphml.append('  </graph>')
        graphml.append('</graphml>')

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(graphml))

        print(f"✓ GraphML exported: {output_path}")


# Example usage
if __name__ == '__main__':
    from src.kg_builder import KGBuilder
    from src.llm_client import GeminiClient
    import json

    # Load existing KG
    kg_path = Path('output/initial_kg.json')
    if kg_path.exists():
        with open(kg_path, 'r') as f:
            kg_data = json.load(f)
        kg = KnowledgeGraph.from_json(kg_data)

        # Create visualizer
        viz = KGVisualizer(kg)

        # Generate visualizations
        viz.generate_html_with_toggles(
            Path('output/kg_interactive.html'),
            title="Movie Knowledge Graph"
        )

        print("\n✨ Open output/kg_interactive.html in your browser!")
