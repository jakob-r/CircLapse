#!/usr/bin/env python3
"""
Test script to demonstrate circelizer functionality.
"""

import tempfile
import os
from pathlib import Path
from circelizer import process_images


def test_circelizer():
    """Test the circelizer with a simple example."""
    print("🧪 Testing Circelizer...")
    
    # Create temporary directories for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = Path('./example_in')
        output_dir = Path('./example_out')
        
        # Create input directory
        input_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Created test directories:")
        print(f"   Input: {input_dir}")
        print(f"   Output: {output_dir}")
        
        # Note: In a real test, you would add some test images here
        print("\n📝 Note: This is a demonstration script.")
        print("   To test with real images, add JPG files to the input directory.")
        print("   Then run: uv run python test_example.py")
        
        # Test the function (will return stats for 0 images)
        try:
            stats = process_images(str(input_dir), str(output_dir))
            print(f"\n✅ Test completed successfully!")
            print(f"   Stats: {stats}")
        except Exception as e:
            print(f"\n❌ Test failed: {repr(e)}")


if __name__ == "__main__":
    test_circelizer() 