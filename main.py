#!/usr/bin/env python3
"""
Example usage of the circelizer library.
"""

import sys
from pathlib import Path
from circelizer import process_images


def main():
    """Example usage of circelizer."""
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_path> <output_path>")
        print("Example: python main.py ./images ./processed")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    print(f"Processing images from {input_path} to {output_path}")
    
    try:
        stats = process_images(input_path, output_path)
        print(f"Processing complete!")
        print(f"Total images: {stats['total']}")
        print(f"Processed: {stats['processed']}")
        print(f"Failed: {stats['failed']}")
    except Exception as e:
        print(f"Error: {repr(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
