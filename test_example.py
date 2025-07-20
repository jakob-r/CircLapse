#!/usr/bin/env python3
"""
Test script to demonstrate circelizer functionality.
"""

from pathlib import Path

from circelizer import process_images


def test_circelizer():
    """Test the circelizer with a simple example."""
    print("🧪 Testing Circelizer...")

    input_dir = Path("./examples/example_in")
    output_dir = Path("./examples/example_out")

    try:
        stats = process_images(str(input_dir), str(output_dir), "gif")
        print("\n✅ Test completed successfully!")
        print(f"   Stats: {stats}")
    except Exception as e:
        print(f"\n❌ Test failed: {repr(e)}")


if __name__ == "__main__":
    test_circelizer()
