"""
Command-line interface for circelizer.
"""

import argparse
import logging
import sys
from pathlib import Path

from .processor import process_images


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Detect circles in images, center them, and crop to square"
    )
    parser.add_argument(
        "input_path",
        help="Path to directory containing images to process"
    )
    parser.add_argument(
        "output_path", 
        help="Path to directory where processed images will be saved"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--detection-method",
        choices=['hough', 'contour', 'ransac', 'gradient'],
        default='hough',
        help="Circle detection method to use (default: hough)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (saves debug images)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Configure detection method and debug mode
    from . import settings
    settings.DETECTION_METHOD = args.detection_method
    settings.DEBUG = args.debug
    
    try:
        # Process images
        stats = process_images(args.input_path, args.output_path)
        
        # Print summary
        print(f"\nProcessing Summary:")
        print(f"Total images found: {stats['total']}")
        print(f"Successfully processed: {stats['processed']}")
        print(f"Failed: {stats['failed']}")
        
        if stats['failed'] > 0:
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {repr(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 