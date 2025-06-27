#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARDIO-LR Library Verification Script
------------------------------------
This script verifies that all required libraries for the CARDIO-LR system
are correctly installed and importable.
"""

import sys
import importlib
from packaging import version

# Define required libraries with their minimum versions
REQUIRED_LIBRARIES = {
    'torch': '2.0.0',              # PyTorch
    'transformers': '4.30.0',      # Hugging Face Transformers
    'faiss': '1.7.4',              # FAISS for vector search (from faiss-cpu or faiss-gpu)
    'torch_geometric': '2.3.0',    # PyTorch Geometric (PyG)
    'neo4j': '5.8.0',              # Neo4j for graph database
    'numpy': '1.24.0',             # NumPy
    'pandas': '2.0.0',             # Pandas
    'sklearn': '1.2.0',            # Scikit-learn (package name is sklearn)
    'matplotlib': '3.7.0',         # Matplotlib
    'networkx': '3.1',             # NetworkX
    'tqdm': '4.65.0',              # TQDM for progress bars
}

def check_library(lib_name, min_version):
    """Check if a library is installed and meets the minimum version requirement."""
    try:
        # Handle special cases
        if lib_name == 'sklearn':
            module = importlib.import_module('sklearn')
            lib_name = 'scikit-learn'
        else:
            module = importlib.import_module(lib_name)
        
        # Get the version
        if hasattr(module, '__version__'):
            lib_version = module.__version__
        else:
            lib_version = 'Unknown'
        
        # Check version if available
        if lib_version != 'Unknown':
            try:
                meets_version = version.parse(lib_version) >= version.parse(min_version)
            except:
                meets_version = False
        else:
            meets_version = "Unknown"
        
        return {
            'installed': True,
            'version': lib_version,
            'min_version': min_version,
            'meets_version': meets_version
        }
    except ImportError:
        return {
            'installed': False,
            'version': None,
            'min_version': min_version,
            'meets_version': False
        }

def main():
    """Main function to verify all libraries."""
    print(f"\n{'=' * 60}")
    print(f"CARDIO-LR Library Verification")
    print(f"{'=' * 60}")
    
    print(f"\nPython version: {sys.version}")
    
    all_installed = True
    all_versions_ok = True
    
    print("\nChecking required libraries:")
    print(f"{'Library':<20} {'Installed':<10} {'Version':<15} {'Minimum':<10} {'Status'}")
    print(f"{'-' * 70}")
    
    for lib_name, min_version in REQUIRED_LIBRARIES.items():
        result = check_library(lib_name, min_version)
        
        display_name = lib_name
        if lib_name == 'sklearn':
            display_name = 'scikit-learn'
        
        installed_str = '✓' if result['installed'] else '✗'
        
        if not result['installed']:
            status = 'MISSING'
            all_installed = False
            all_versions_ok = False
        elif result['meets_version'] is True:
            status = 'OK'
        elif result['meets_version'] == 'Unknown':
            status = 'UNKNOWN'
        else:
            status = 'OUTDATED'
            all_versions_ok = False
        
        version_str = result['version'] if result['version'] else 'N/A'
        
        print(f"{display_name:<20} {installed_str:<10} {version_str:<15} {min_version:<10} {status}")
    
    print(f"\n{'-' * 70}")
    
    if all_installed and all_versions_ok:
        print("\n✓ SUCCESS: All required libraries are installed and meet version requirements.")
        print("\nYou're ready to run CARDIO-LR!")
    elif all_installed and not all_versions_ok:
        print("\n⚠ WARNING: All libraries are installed but some version requirements are not met.")
        print("The system may still work, but you might encounter unexpected issues.")
    else:
        print("\n✗ ERROR: Some required libraries are missing.")
        print("Please install the missing libraries before running CARDIO-LR.")
    
    print(f"\n{'=' * 60}\n")

if __name__ == "__main__":
    main()