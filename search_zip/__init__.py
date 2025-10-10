"""
Search and Zip Trajectory Package

This package provides functionality for:
1. Searching similar cases using AI models
2. Compressing and analyzing agent trajectories
3. Managing workflow between different modules
"""

from .flow import FlowManager, TrajectoryProcessor, SimilarCaseFinder
from .search import search
from .zip_traj import zip_traj

__all__ = [
    'FlowManager',
    'TrajectoryProcessor',
    'SimilarCaseFinder',
    'search',
    'zip_traj'
]

__version__ = "2.0"

__author__ = "JoyCode Team"
