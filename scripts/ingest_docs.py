"""
ingest_docs.py — CLI tool to ingest documents from a directory.

Usage:
    cd research_copilot
    python scripts/ingest_docs.py --dir data/raw
    python scripts/ingest_docs.py --dir data/raw --parent-child
"""

import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ingestion.indexer import ingest_directory


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the Research Copilot knowledge base")
    parser.add_argument("--dir", required=True, help="Directory containing documents to ingest")
    parser.add_argument("--parent-child", action="store_true", help="Use parent-child chunking")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Error: {args.dir} is not a directory")
        sys.exit(1)

    print(f"Ingesting documents from {args.dir}...")
    n = ingest_directory(args.dir, use_parent_child=args.parent_child)
    print(f"✅ Indexed {n} chunks")


if __name__ == "__main__":
    main()
