#!/usr/bin/env python3
"""
Render all ML Pipeline GraphViz diagrams to PNG
"""
import subprocess
import sys
from pathlib import Path

def render_diagram(dot_file, output_file):
    """Render a single diagram"""
    if not dot_file.exists():
        print(f"❌ {dot_file.name} not found!")
        return False

    try:
        cmd = ["dot", "-Tpng", str(dot_file), "-o", str(output_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ {output_file.name} rendered successfully")
            return True
        else:
            print(f"❌ Failed to render {dot_file.name}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error rendering {dot_file.name}: {e}")
        return False

def render_all_diagrams():
    """Render all pipeline diagrams"""
    script_dir = Path(__file__).parent

    diagrams = [
        ("data_prep_graph.dot", "data_prep_graph.png"),
        ("feature_eng_graph.dot", "feature_eng_graph.png"),
        ("model_dev_graph.dot", "model_dev_graph.png"),
        ("app_deployment_graph.dot", "app_deployment_graph.png"),
    ]

    success_count = 0

    for dot_name, png_name in diagrams:
        dot_file = script_dir / dot_name
        png_file = script_dir / png_name

        if render_diagram(dot_file, png_file):
            success_count += 1

    return success_count == len(diagrams)

if __name__ == "__main__":
    print("🎨 Rendering ML Pipeline Diagrams...")
    print("=" * 40)

    try:
        success = render_all_diagrams()
        if success:
            print("\n✅ All diagrams rendered successfully!")
            print("📊 Available diagrams:")
            print("  • data_prep_graph.png - Data preparation pipeline")
            print("  • feature_eng_graph.png - Feature engineering pipeline")
            print("  • model_dev_graph.png - Model development pipeline")
            print("  • app_deployment_graph.png - Application deployment")
        else:
            print("\n❌ Some diagrams failed to render.")
            sys.exit(1)
    except FileNotFoundError:
        print("❌ GraphViz (dot) not installed. Install with: brew install graphviz")
        sys.exit(1)