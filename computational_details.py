#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computational Details for CARDIO-LR System
-----------------------------------------
This module provides utilities to check hardware compatibility, assess computational
resources, and initialize the necessary software environment for the CARDIO-LR system.
"""

import os
import sys
import platform
import subprocess
import importlib
import psutil
import torch
import numpy as np
import json
from datetime import datetime

class ComputationalDetails:
    """Class to assess and report computational resources for CARDIO-LR"""
    
    def __init__(self):
        """Initialize the computational details checker"""
        self.hardware_requirements = {
            'preferred_gpu': 'NVIDIA Tesla V100 or A100',
            'min_gpu_vram_gb': 16,
            'recommended_gpu_vram_gb': 32,
            'min_cpu_cores': 8,
            'min_ram_gb': 64,
            'recommended_ram_gb': 128
        }
        
        self.required_libraries = {
            'pytorch': '2.0.0',
            'transformers': '4.30.0',
            'faiss-cpu': '1.7.4',
            'torch_geometric': '2.3.0',
            'neo4j': '5.8.0',
            'numpy': '1.24.0',
            'pandas': '2.0.0',
            'scikit-learn': '1.2.0',
            'matplotlib': '3.7.0',
            'networkx': '3.1',
            'tqdm': '4.65.0'
        }
        
        # Platform information
        self.platform_info = {
            'os': platform.system(),
            'os_version': platform.version(),
            'os_release': platform.release(),
            'python_version': platform.python_version(),
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Storage info
        self.storage_requirements = {
            'raw_data_size_gb': 10, 
            'processed_data_size_gb': 5,
            'models_size_gb': 15,
            'total_required_gb': 30
        }
        
    def check_hardware(self):
        """Check available hardware resources and compare against requirements"""
        hardware_report = {}
        
        # CPU information
        cpu_cores = psutil.cpu_count(logical=False)
        cpu_threads = psutil.cpu_count(logical=True)
        hardware_report['cpu'] = {
            'physical_cores': cpu_cores,
            'logical_cores': cpu_threads,
            'meets_requirement': cpu_cores >= self.hardware_requirements['min_cpu_cores']
        }
        
        # RAM information
        ram_gb = round(psutil.virtual_memory().total / (1024**3))
        hardware_report['ram'] = {
            'total_gb': ram_gb,
            'meets_requirement': ram_gb >= self.hardware_requirements['min_ram_gb'],
            'meets_recommended': ram_gb >= self.hardware_requirements['recommended_ram_gb']
        }
        
        # GPU information
        hardware_report['gpu'] = {'available': False}
        
        if torch.cuda.is_available():
            hardware_report['gpu']['available'] = True
            hardware_report['gpu']['count'] = torch.cuda.device_count()
            hardware_report['gpu']['devices'] = []
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                # Approximate VRAM detection - this is not always reliable
                try:
                    # This only works for NVIDIA GPUs with CUDA
                    free_mem, total_mem = torch.cuda.mem_get_info(i)
                    vram_gb = round(total_mem / (1024**3), 1)
                except:
                    vram_gb = "Unknown"
                
                hardware_report['gpu']['devices'].append({
                    'index': i,
                    'name': gpu_name,
                    'vram_gb': vram_gb,
                    'meets_requirement': vram_gb != "Unknown" and vram_gb >= self.hardware_requirements['min_gpu_vram_gb'],
                    'meets_recommended': vram_gb != "Unknown" and vram_gb >= self.hardware_requirements['recommended_gpu_vram_gb']
                })
                
        # Storage information
        disk = psutil.disk_usage('/')
        hardware_report['storage'] = {
            'total_gb': round(disk.total / (1024**3)),
            'free_gb': round(disk.free / (1024**3)),
            'meets_requirement': disk.free / (1024**3) > self.storage_requirements['total_required_gb']
        }
            
        return hardware_report
    
    def check_libraries(self):
        """Check if required libraries are installed with correct versions"""
        library_report = {}
        
        for lib, required_version in self.required_libraries.items():
            try:
                installed = importlib.import_module(lib.split('-')[0])  # Handle cases like 'faiss-cpu'
                try:
                    version = installed.__version__
                except AttributeError:
                    version = "Unknown"
                
                # Simple version checking - in a real implementation you'd want more sophisticated version comparison
                meets_requirement = version == required_version or version.startswith(required_version.split('.')[0])
                
                library_report[lib] = {
                    'installed': True,
                    'required_version': required_version,
                    'installed_version': version,
                    'meets_requirement': meets_requirement
                }
            except ImportError:
                library_report[lib] = {
                    'installed': False,
                    'required_version': required_version,
                    'installed_version': None,
                    'meets_requirement': False
                }
        
        return library_report
    
    def check_platform(self):
        """Check platform compatibility"""
        platform_report = self.platform_info.copy()
        
        # Check if running in a Google Colab environment
        platform_report['is_colab'] = 'google.colab' in sys.modules
        
        # Check for GPU acceleration in Colab
        if platform_report['is_colab']:
            try:
                gpu_info = subprocess.check_output(['nvidia-smi'], universal_newlines=True)
                platform_report['colab_has_gpu'] = True
            except:
                platform_report['colab_has_gpu'] = False
        
        # Check for common cloud environments
        platform_report['is_cloud'] = any([
            os.environ.get('KUBERNETES_SERVICE_HOST'),  # Kubernetes
            os.environ.get('AWS_EXECUTION_ENV'),        # AWS Lambda
            os.environ.get('GOOGLE_CLOUD_PROJECT'),     # GCP
            os.environ.get('AZURE_FUNCTIONS_ENVIRONMENT') # Azure
        ]) is not None
        
        return platform_report
    
    def check_neo4j_connection(self, uri='bolt://localhost:7687', user='neo4j', password='password'):
        """Test connection to Neo4j database"""
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(uri, auth=(user, password))
            with driver.session() as session:
                result = session.run("MATCH (n) RETURN count(n) AS count")
                node_count = result.single()["count"]
            driver.close()
            return {
                'connected': True,
                'node_count': node_count,
                'uri': uri
            }
        except Exception as e:
            return {
                'connected': False,
                'error': str(e),
                'uri': uri
            }
    
    def suggest_optimization(self, hardware_report, library_report):
        """Suggest optimizations based on hardware and library configuration"""
        suggestions = []
        
        # CPU suggestions
        if not hardware_report['cpu']['meets_requirement']:
            suggestions.append("Upgrade to a system with at least 8 CPU cores for optimal performance.")
        
        # RAM suggestions
        if not hardware_report['ram']['meets_requirement']:
            suggestions.append(f"Increase RAM to at least {self.hardware_requirements['min_ram_gb']}GB for full dataset processing.")
        elif not hardware_report['ram']['meets_recommended']:
            suggestions.append(f"Consider increasing RAM to {self.hardware_requirements['recommended_ram_gb']}GB for optimal performance with large datasets.")
        
        # GPU suggestions
        if not hardware_report['gpu']['available']:
            suggestions.append("Add a GPU with at least 16GB VRAM for significantly faster training and inference.")
        else:
            for gpu in hardware_report['gpu']['devices']:
                if not gpu['meets_requirement']:
                    suggestions.append(f"GPU {gpu['name']} has insufficient VRAM. Consider upgrading to {self.hardware_requirements['preferred_gpu']}.")
        
        # Library suggestions
        missing_libs = [lib for lib, info in library_report.items() if not info['installed']]
        outdated_libs = [lib for lib, info in library_report.items() if info['installed'] and not info['meets_requirement']]
        
        if missing_libs:
            suggestions.append(f"Install missing libraries: {', '.join(missing_libs)}")
        
        if outdated_libs:
            suggestions.append(f"Update outdated libraries: {', '.join(outdated_libs)}")
            
        return suggestions
    
    def generate_full_report(self):
        """Generate a comprehensive report of computational resources"""
        hardware_report = self.check_hardware()
        library_report = self.check_libraries()
        platform_report = self.check_platform()
        
        # Only attempt Neo4j connection if the library is available
        neo4j_report = {"connected": False, "note": "Neo4j library not installed"}
        if library_report.get('neo4j', {}).get('installed', False):
            neo4j_report = self.check_neo4j_connection()
        
        suggestions = self.suggest_optimization(hardware_report, library_report)
        
        full_report = {
            'hardware': hardware_report,
            'libraries': library_report,
            'platform': platform_report,
            'neo4j': neo4j_report,
            'storage_requirements': self.storage_requirements,
            'optimization_suggestions': suggestions
        }
        
        return full_report
    
    def print_report_summary(self, report=None):
        """Print a summary of the computational resources report"""
        if report is None:
            report = self.generate_full_report()
        
        print("\n" + "="*80)
        print("CARDIO-LR COMPUTATIONAL REQUIREMENTS REPORT")
        print("="*80)
        
        print("\nHARDWARE SUMMARY:")
        print(f"CPU: {report['hardware']['cpu']['physical_cores']} cores "
              f"({'✓' if report['hardware']['cpu']['meets_requirement'] else '✗'})")
        
        print(f"RAM: {report['hardware']['ram']['total_gb']}GB "
              f"({'✓' if report['hardware']['ram']['meets_requirement'] else '✗'})")
        
        if report['hardware']['gpu']['available']:
            print("GPU(s):")
            for gpu in report['hardware']['gpu']['devices']:
                print(f"  - {gpu['name']} with {gpu['vram_gb']}GB VRAM "
                     f"({'✓' if gpu['meets_requirement'] else '✗'})")
        else:
            print("GPU: Not available ✗")
        
        print(f"\nPLATFORM: {report['platform']['os']} {report['platform']['os_release']}")
        print(f"Python: {report['platform']['python_version']}")
        
        if report['platform']['is_colab']:
            gpu_status = "with GPU" if report['platform']['colab_has_gpu'] else "without GPU"
            print(f"Running in Google Colab {gpu_status}")
            
        print("\nSOFTWARE LIBRARIES:")
        installed_count = sum(1 for lib, info in report['libraries'].items() if info['installed'])
        print(f"{installed_count}/{len(report['libraries'])} required libraries installed")
        
        missing_libs = [lib for lib, info in report['libraries'].items() if not info['installed']]
        if missing_libs:
            print(f"Missing: {', '.join(missing_libs)}")
            
        print("\nSTORAGE REQUIREMENTS:")
        print(f"Total space needed: {report['storage_requirements']['total_required_gb']}GB")
        print(f"Available space: {report['hardware']['storage']['free_gb']}GB "
              f"({'✓' if report['hardware']['storage']['meets_requirement'] else '✗'})")
        
        if report['neo4j']['connected']:
            print(f"\nNeo4j: Connected ✓ ({report['neo4j']['node_count']} nodes in database)")
        else:
            print("\nNeo4j: Not connected ✗")
            
        if report['optimization_suggestions']:
            print("\nSUGGESTIONS FOR OPTIMIZATION:")
            for i, suggestion in enumerate(report['optimization_suggestions'], 1):
                print(f"{i}. {suggestion}")
        
        print("\n" + "="*80)

def save_report_to_file(report, filename="computational_report.json"):
    """Save the computational report to a JSON file"""
    # Convert any non-serializable objects to strings
    def serialize(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        return str(obj)
    
    # Apply serialization to the entire report
    serialized_report = json.loads(
        json.dumps(report, default=serialize)
    )
    
    with open(filename, 'w') as f:
        json.dump(serialized_report, f, indent=2)
    
    print(f"Report saved to {filename}")

def main():
    """Run the computational details assessment"""
    details = ComputationalDetails()
    report = details.generate_full_report()
    details.print_report_summary(report)
    
    # Save report to file
    save_report_to_file(report)
    
    # Return overall assessment
    meets_min_requirements = (
        report['hardware']['cpu']['meets_requirement'] and
        report['hardware']['ram']['meets_requirement'] and
        report['hardware']['storage']['meets_requirement']
    )
    
    meets_recommended = meets_min_requirements and (
        report['hardware']['ram']['meets_recommended'] and
        (report['hardware']['gpu']['available'] and 
         any(gpu['meets_recommended'] for gpu in report['hardware']['gpu']['devices']))
    )
    
    if meets_recommended:
        print("\nSUMMARY: System meets all recommended requirements for CARDIO-LR! ✓")
    elif meets_min_requirements:
        print("\nSUMMARY: System meets minimum requirements but some optimizations are recommended.")
    else:
        print("\nSUMMARY: System does not meet minimum requirements for CARDIO-LR. ✗")
        print("Please address the suggestions above before proceeding.")

if __name__ == "__main__":
    main()