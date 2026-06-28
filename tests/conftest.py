"""
Shared pytest fixtures and configuration for the RAG test suite.
"""

import os
import sys
import pytest

# Add project root to path so tests can import from backend/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Set test environment before any imports
os.environ.setdefault("APP_ENV", "test")

# Suppress ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"
