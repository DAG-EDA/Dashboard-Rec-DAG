"""
Movie Knowledge Graph Construction Test

Usage (run from project root):
  PYTHONPATH=. python tests/test_kg_movie.py              # Auto-load if saved files exist, else rebuild
  PYTHONPATH=. python tests/test_kg_movie.py --use-saved  # Load from saved JSON files (faster)
  PYTHONPATH=. python tests/test_kg_movie.py --rebuild    # Force rebuild even if files exist

Note: The script auto-detects saved files by default. No need to specify --use-saved if files exist.
"""

from src.kg_builder import KGBuilder
from src.kg_visualizer import KGVisualizer
from src.llm_client import GeminiClient
from pathlib import Path
import json
import sys
import argparse
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Parse arguments
parser = argparse.ArgumentParser(
    description='Movie Knowledge Graph Construction',
    epilog='By default, saved graphs are loaded if they exist. Use --rebuild to force rebuilding.'
)
parser.add_argument('--use-saved', action='store_true',
                    help='Load graphs from saved JSON files instead of rebuilding')
parser.add_argument('--rebuild', action='store_true',
                    help='Force rebuild graphs even if saved files exist')
args = parser.parse_args()

print("="*70)
print("MOVIE KNOWLEDGE GRAPH CONSTRUCTION")
print("="*70)

# Define paths
oneshot_path = Path('output/movie_kg_oneshot.json')
pairwise_path = Path('output/movie_kg_pairwise.json')

# Determine whether to load or build graphs
should_load = args.use_saved or (not args.rebuild and oneshot_path.exists() and pairwise_path.exists())

# Show mode
if args.rebuild:
    print("Mode: REBUILD (forcing rebuild)")
elif args.use_saved:
    print("Mode: LOAD SAVED (--use-saved)")
elif should_load:
    print("Mode: AUTO-LOAD (saved files exist)")
else:
    print("Mode: BUILD (no saved files found)")

# Initialize
llm = GeminiClient()
builder = KGBuilder(llm)

# Load summary
print("\n[1/3] Loading summary..." if should_load else "\n[1/4] Loading summary...")
summary = json.load(open('output/test_summary.json'))
print(f"✓ Loaded summary for {summary['dataset_info']['name']}")

if should_load and oneshot_path.exists() and pairwise_path.exists():
    print("\n[2/3] Loading saved knowledge graphs...")
    print(f"  Loading from {oneshot_path}")
    kg_oneshot = builder.load_kg(oneshot_path)
    print(f"✓ Loaded oneshot KG")

    print(f"  Loading from {pairwise_path}")
    kg_pairwise = builder.load_kg(pairwise_path)
    print(f"✓ Loaded pairwise KG")
else:
    if args.use_saved:
        print("\n⚠️  Saved files not found, building graphs instead...")

    # Build KG with oneshot (fast)
    print("\n[2/4] Building knowledge graph (oneshot strategy)...")
    kg_oneshot = builder.build_initial_kg(summary, domain_hint='movies')

    # Save oneshot KG
    builder.save_kg(kg_oneshot, oneshot_path)

    # Build with pairwise strategy (more thorough)
    print("\n[3/4] Building knowledge graph (pairwise strategy)...")
    kg_pairwise, data_mapping = builder.build_data_layer(summary)
    kg_pairwise, concept_mapping = builder.build_concept_layer(kg_pairwise, summary)
    kg_pairwise, stats = builder.infer_causal_relationships(
        kg_pairwise, data_mapping, summary,
        strategy='pairwise',
        confidence_threshold=0.6
    )
    kg_pairwise = builder.create_bridge_edges(kg_pairwise, data_mapping, concept_mapping, summary)

    # Save pairwise KG
    builder.save_kg(kg_pairwise, pairwise_path)

# Generate visualizations
step_num = 3 if should_load else 4
print(f"\n[{step_num}/{step_num}] Generating visualizations...")

# Oneshot visualization
viz_oneshot = KGVisualizer(kg_oneshot)
viz_oneshot.generate_html_with_toggles(
    Path('output/movie_kg_oneshot.html'),
    title="Movie Knowledge Graph (Oneshot Strategy)"
)

# Pairwise visualization
viz_pairwise = KGVisualizer(kg_pairwise)
viz_pairwise.generate_html_with_toggles(
    Path('output/movie_kg_pairwise.html'),
    title="Movie Knowledge Graph (Pairwise Strategy)"
)

# Export to GraphML (for external tools)
viz_pairwise.export_to_graphml(Path('output/movie_kg_pairwise.graphml'))

# Show statistics
print("\n" + "="*70)
print("RESULTS")
print("="*70)

print("\n📊 ONESHOT STRATEGY:")
stats_oneshot = kg_oneshot.get_layer_stats()
for layer, layer_stats in stats_oneshot.items():
    if 'node_count' in layer_stats:
        print(f"  {layer.capitalize()}: {layer_stats['node_count']} nodes, {layer_stats['edge_count']} edges")
    else:
        print(f"  {layer.capitalize()}: {layer_stats['edge_count']} edges")

print("\n📊 PAIRWISE STRATEGY:")
stats_pairwise = kg_pairwise.get_layer_stats()
for layer, layer_stats in stats_pairwise.items():
    if 'node_count' in layer_stats:
        print(f"  {layer.capitalize()}: {layer_stats['node_count']} nodes, {layer_stats['edge_count']} edges")
    else:
        print(f"  {layer.capitalize()}: {layer_stats['edge_count']} edges")

print("\n" + "="*70)
print("✨ VISUALIZATION FILES CREATED")
print("="*70)
print("\n1. Interactive HTML (with layer toggles):")
print("   - output/movie_kg_oneshot.html")
print("   - output/movie_kg_pairwise.html")
print("\n2. JSON files (for loading/processing):")
print("   - output/movie_kg_oneshot.json")
print("   - output/movie_kg_pairwise.json")
print("\n3. GraphML (for Gephi/Cytoscape):")
print("   - output/movie_kg_pairwise.graphml")
print("\n💡 Open the HTML files in your browser to explore the graphs!")
print("   You can toggle layers on/off to focus on specific parts.")
print("="*70)