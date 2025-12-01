#!/usr/bin/env python3
"""
Render the ML Pipeline GraphViz diagram to PNG
"""
import subprocess
import sys
from pathlib import Path

def render_pipeline_graph():
    """Render the pipeline graph to PNG using GraphViz"""
    script_dir = Path(__file__).parent
    dot_file = script_dir / "pipeline_graph.dot"
    output_file = script_dir / "pipeline_graph.png"

    if not dot_file.exists():
        print(f"Error: {dot_file} not found!")
        return False

    try:
        # Render to PNG
        cmd = ["dot", "-Tpng", str(dot_file), "-o", str(output_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✅ Pipeline graph rendered successfully: {output_file}")
            return True
        else:
            print(f"❌ GraphViz rendering failed: {result.stderr}")
            return False

    except FileNotFoundError:
        print("❌ GraphViz (dot) not installed. Install with: brew install graphviz")
        return False
    except Exception as e:
        print(f"❌ Error rendering graph: {e}")
        return False

if __name__ == "__main__":
    success = render_pipeline_graph()
    if success:
        print("\n📊 ML Pipeline diagram created successfully!")
        print("You can now include pipeline_graph.png in your presentation.")
    else:
        print("\n❌ Failed to create pipeline diagram.")
        sys.exit(1)