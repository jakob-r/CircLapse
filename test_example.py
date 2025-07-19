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
        input_dir = Path('./examples/example_in')
        output_dir = Path('./examples/example_out')
        
        try:
            stats = process_images(str(input_dir), str(output_dir), 'gif')
            print(f"\n✅ Test completed successfully!")
            print(f"   Stats: {stats}")
        except Exception as e:
            print(f"\n❌ Test failed: {repr(e)}")


if __name__ == "__main__":
    test_circelizer() 