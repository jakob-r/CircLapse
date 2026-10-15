"""
Context variables for the circlapse package.
"""

from contextvars import ContextVar
from pathlib import Path

# Context variable to store the current output directory
output_dir: ContextVar[Path] = ContextVar("output_dir", default=Path("./"))
