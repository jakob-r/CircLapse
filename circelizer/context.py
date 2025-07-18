"""
Context variables for the circelizer package.
"""

from contextvars import ContextVar
from pathlib import Path
from typing import Optional

# Context variable to store the current output directory
output_dir: ContextVar[Path] = ContextVar('output_dir', default=Path("./")) 