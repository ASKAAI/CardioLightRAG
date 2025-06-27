#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARDIO-LR Benchmark Tool
------------------------
This script benchmarks the CARDIO-LR system components to validate 
computational requirements and assess performance across different hardware.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import mock system components for benchmarking
try:
    from mock_pipeline import MockCardiologyLightRAG
    from gnn.rgcn_model import MockRGCNModel
    from retrieval.hybrid_retriever import MockHybridRetriever
    from generation.biomed_generator import MockBiomedGenerator
except ImportError:
    print("Warning: Using placeholder mock components for benchmarking")
    
    class MockRGCNModel:
        def __init__(self, hidden_dim=128, num_relations=5):
            self.hidden_dim = hidden_dim
            self.num_relations = num_relations
            
        def train_step(self, batch_size=32):
            # Simulate training step with PyTorch operations
            # to benchmark GPU performance
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
                
            x = torch.randn(batch_size, 1000, self.hidden_dim).to(device)
            edge_index = torch.randint(0, batch_size, (2, 5000)).to(device)
            edge_type = torch.randint(0, self.num_relations, (5000,)).to(device)
            
            # Simulate GNN operations
            start = time.time()
            for i in range(5):  # 5 layers
                x = torch.matmul(x, torch.randn(self.hidden_dim, self.hidden_dim).to(device))
                x = torch.relu(x)
            end = time.time()
            
            return end - start
    
    class MockHybridRetriever:
        def benchmark_retrieval(self, query_count=100, top_k=10):
            start = time.time()
            # Simulate retrieval operations
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            
            # Create random embeddings
            db_size = 100000
            dim = 768
            db_embeddings = torch.randn(db_size, dim).to(device)
            queries = torch.randn(query_count, dim).to(device)
            
            # Simulate vector search
            scores = torch.matmul(queries, db_embeddings.t())
            _, indices = torch.topk(scores, k=top_k, dim=1)
            
            end = time.time()
            return end - start
    
    class MockBiomedGenerator:
        def benchmark_generation(self, prompt_count=10, max_length=512):
            start = time.time()
            # Simulate text generation with a transformer model
            if torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
            
            batch_size = prompt_count
            seq_len = 128
            vocab_size = 50000
            d_model = 768
            
            # Simplified transformer operations
            x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
            embed = torch.nn.Embedding(vocab_size, d_model).to(device)
            embedded = embed(x)
            
            # Simulate attention mechanism
            for i in range(3):  # 3 layers of attention
                q = torch.matmul(embedded, torch.randn(d_model, d_model).to(device))
                k = torch.matmul(embedded, torch.randn(d_model, d_model).to(device))
                v = torch.matmul(embedded, torch.randn(d_model, d_model).to(device))
                
                scores = torch.matmul(q, k.transpose(-2, -1))
                attention = torch.softmax(scores, dim=-1)
                context = torch.matmul(attention, v)
                
                embedded = embedded + context
            
            # Output projection
            logits = torch.matmul(embedded, embed.weight.transpose(0, 1))
            
            end = time.time()
            return end - start
    
    class MockCardiologyLightRAG:
        def __init__(self):
            self.gnn = MockRGCNModel()
            self.retriever = MockHybridRetriever()
            self.generator = MockBiomedGenerator()
        
        def process_query(self, query, patient_context=None):
            # Simple mock implementation to allow benchmarking
            return "This is a mock answer for benchmarking purposes.", "Mock explanation"


class BenchmarkResults:
    """Class to store, analyze and visualize benchmark results"""
    
    def __init__(self):
        self.results = []
        
    def add_result(self, component, operation, batch_size, time_taken, device_info):
        self.results.append({
            'component': component,
            'operation': operation,
            'batch_size': batch_size,
            'time_seconds': time_taken,
            'device': device_info
        })
    
    def to_dataframe(self):
        return pd.DataFrame(self.results)
    
    def plot_comparison(self, save_path=None):
        """Plot comparison of component performance across different batch sizes"""
        df = self.to_dataframe()
        
        # Group by component, operation, batch size and device
        grouped = df.groupby(['component', 'operation', 'batch_size', 'device'])['time_seconds'].mean().reset_index()
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        components = grouped['component'].unique()
        devices = grouped['device'].unique()
        
        for i, component in enumerate(components):
            component_data = grouped[grouped['component'] == component]
            
            for device in devices:
                device_data = component_data[component_data['device'] == device]
                if len(device_data) > 0:
                    plt.subplot(len(components), 1, i+1)
                    plt.plot(device_data['batch_size'], device_data['time_seconds'], 
                             marker='o', label=f"{device}")
                    plt.title(f"{component} Performance")
                    plt.xlabel("Batch Size")
                    plt.ylabel("Time (seconds)")
                    plt.legend()
                    plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def save_to_csv(self, filename="benchmark_results.csv"):
        """Save benchmark results to CSV file"""
        df = self.to_dataframe()
        df.to_csv(filename, index=False)
        print(f"Benchmark results saved to {filename}")
    
    def summary(self):
        """Generate a summary of benchmark results"""
        df = self.to_dataframe()
        
        summary = {
            'components': df['component'].unique().tolist(),
            'devices_tested': df['device'].unique().tolist(),
            'total_benchmarks': len(df),
            'avg_times_by_component': df.groupby('component')['time_seconds'].mean().to_dict(),
            'min_times_by_component': df.groupby('component')['time_seconds'].min().to_dict(),
            'max_times_by_component': df.groupby('component')['time_seconds'].max().to_dict()
        }
        
        # Check if we have data for both CPU and GPU
        if 'CPU' in df['device'].values and 'CUDA' in df['device'].values:
            # Calculate speedup
            cpu_df = df[df['device'] == 'CPU']
            gpu_df = df[df['device'] == 'CUDA']
            
            # Group by component and operation
            cpu_times = cpu_df.groupby(['component', 'operation'])['time_seconds'].mean()
            gpu_times = gpu_df.groupby(['component', 'operation'])['time_seconds'].mean()
            
            # Only keep components that have both CPU and GPU measurements
            common_ops = set(cpu_times.index).intersection(set(gpu_times.index))
            
            speedups = {}
            for comp_op in common_ops:
                speedups[f"{comp_op[0]}_{comp_op[1]}"] = float(cpu_times[comp_op] / gpu_times[comp_op])
            
            summary['gpu_speedup'] = speedups
        
        return summary


def benchmark_rgcn_training(batch_sizes=[32, 64, 128, 256], runs=3):
    """Benchmark R-GCN model training"""
    results = BenchmarkResults()
    model = MockRGCNModel(hidden_dim=128, num_relations=5)
    
    device_info = "CUDA" if torch.cuda.is_available() else "CPU"
    
    for batch_size in batch_sizes:
        for i in range(runs):
            time_taken = model.train_step(batch_size=batch_size)
            results.add_result(
                component="R-GCN",
                operation="Training",
                batch_size=batch_size,
                time_taken=time_taken,
                device_info=device_info
            )
            
    return results


def benchmark_vector_retrieval(query_counts=[10, 50, 100, 500], runs=3):
    """Benchmark hybrid vector retrieval"""
    results = BenchmarkResults()
    retriever = MockHybridRetriever()
    
    device_info = "CUDA" if torch.cuda.is_available() else "CPU"
    
    for query_count in query_counts:
        for i in range(runs):
            time_taken = retriever.benchmark_retrieval(query_count=query_count, top_k=10)
            results.add_result(
                component="Hybrid Retriever", 
                operation="Vector Search",
                batch_size=query_count,
                time_taken=time_taken,
                device_info=device_info
            )
            
    return results


def benchmark_text_generation(prompt_counts=[1, 5, 10, 20], runs=3):
    """Benchmark biomedical text generation"""
    results = BenchmarkResults()
    generator = MockBiomedGenerator()
    
    device_info = "CUDA" if torch.cuda.is_available() else "CPU"
    
    for prompt_count in prompt_counts:
        for i in range(runs):
            time_taken = generator.benchmark_generation(prompt_count=prompt_count, max_length=512)
            results.add_result(
                component="Biomed Generator",
                operation="Text Generation",
                batch_size=prompt_count,
                time_taken=time_taken,
                device_info=device_info
            )
    
    return results


def benchmark_end_to_end(query_counts=[1, 5, 10], runs=3):
    """Benchmark end-to-end pipeline"""
    results = BenchmarkResults()
    system = MockCardiologyLightRAG()
    
    device_info = "CUDA" if torch.cuda.is_available() else "CPU"
    
    example_queries = [
        "What are the symptoms of heart failure?",
        "How do beta blockers work for angina?",
        "What are the side effects of statins?",
        "What is the recommended treatment for atrial fibrillation?",
        "How does coronary artery disease develop?",
        "What are risk factors for myocardial infarction?",
        "How do ACE inhibitors affect blood pressure?",
        "What is the mechanism of action for nitroglycerin?",
        "How does hypertension affect the heart?",
        "What are common complications of heart valve disease?"
    ]
    
    patient_contexts = [
        "Patient has diabetes and hypertension",
        "Patient has a history of stroke",
        "Patient has chronic kidney disease",
        "Patient is pregnant",
        "Patient has liver disease"
    ]
    
    for count in query_counts:
        for i in range(runs):
            start_time = time.time()
            
            # Process multiple queries
            for j in range(min(count, len(example_queries))):
                query = example_queries[j % len(example_queries)]
                context = patient_contexts[j % len(patient_contexts)]
                system.process_query(query, context)
            
            time_taken = time.time() - start_time
            
            # Record average time per query
            results.add_result(
                component="Full Pipeline",
                operation="End-to-End QA",
                batch_size=count,
                time_taken=time_taken / count,  # Average time per query
                device_info=device_info
            )
    
    return results


def check_system_compatibility():
    """Run system compatibility check"""
    from computational_details import ComputationalDetails
    
    details = ComputationalDetails()
    return details.generate_full_report()


def run_all_benchmarks(args):
    """Run all benchmark tests and combine results"""
    all_results = BenchmarkResults()
    
    print("Running CARDIO-LR benchmarks...")
    print("\nStep 1: Checking system compatibility...")
    compatibility_report = check_system_compatibility()
    
    print("\nStep 2: Benchmarking R-GCN training...")
    rgcn_results = benchmark_rgcn_training(
        batch_sizes=args.batch_sizes, 
        runs=args.runs
    )
    
    print("\nStep 3: Benchmarking vector retrieval...")
    retrieval_results = benchmark_vector_retrieval(
        query_counts=args.query_counts,
        runs=args.runs
    )
    
    print("\nStep 4: Benchmarking text generation...")
    generation_results = benchmark_text_generation(
        prompt_counts=args.prompt_counts,
        runs=args.runs
    )
    
    print("\nStep 5: Benchmarking end-to-end pipeline...")
    pipeline_results = benchmark_end_to_end(
        query_counts=args.pipeline_queries,
        runs=args.runs
    )
    
    # Combine all results
    all_results.results.extend(rgcn_results.results)
    all_results.results.extend(retrieval_results.results)
    all_results.results.extend(generation_results.results)
    all_results.results.extend(pipeline_results.results)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Save results
    all_results.save_to_csv(os.path.join(args.output_dir, "benchmark_results.csv"))
    all_results.plot_comparison(save_path=os.path.join(args.output_dir, "benchmark_plot.png"))
    
    # Print summary
    summary = all_results.summary()
    print("\nBenchmark Summary:")
    print(f"Total benchmarks: {summary['total_benchmarks']}")
    print(f"Components tested: {', '.join(summary['components'])}")
    print(f"Devices tested: {', '.join(summary['devices_tested'])}")
    
    print("\nAverage times by component (seconds):")
    for component, avg_time in summary['avg_times_by_component'].items():
        print(f"  {component}: {avg_time:.4f}s")
    
    if 'gpu_speedup' in summary:
        print("\nGPU speedup factors:")
        for op, speedup in summary['gpu_speedup'].items():
            print(f"  {op}: {speedup:.2f}x faster with GPU")
    
    # Final recommendation
    device = "GPU" if torch.cuda.is_available() else "CPU"
    if device == "CPU" and any(t > 5 for t in summary['avg_times_by_component'].values()):
        print("\nRECOMMENDATION: Consider using a GPU for significantly faster performance.")
    
    return all_results, compatibility_report


def main():
    parser = argparse.ArgumentParser(description="CARDIO-LR Benchmark Tool")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128, 256],
                        help="Batch sizes for RGCN training benchmarks")
    parser.add_argument("--query-counts", type=int, nargs="+", default=[10, 50, 100, 500],
                        help="Query counts for retrieval benchmarks")
    parser.add_argument("--prompt-counts", type=int, nargs="+", default=[1, 5, 10, 20],
                        help="Prompt counts for generation benchmarks")
    parser.add_argument("--pipeline-queries", type=int, nargs="+", default=[1, 5, 10],
                        help="Query counts for end-to-end pipeline benchmarks")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs for each benchmark configuration")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to save benchmark results")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CARDIO-LR Benchmark Tool")
    print("=" * 80)
    
    # Run all benchmarks
    results, compatibility = run_all_benchmarks(args)
    
    print("\nBenchmarks complete! Results saved to", args.output_dir)
    
    # Compute estimated requirements based on results
    device = "GPU" if torch.cuda.is_available() else "CPU"
    df = results.to_dataframe()
    
    print("\nEstimated Resource Requirements:")
    
    # Estimate memory requirements
    rgcn_mem = {32: 4, 64: 6, 128: 10, 256: 18}  # Approximate GB of VRAM needed
    max_batch = max(args.batch_sizes)
    if max_batch in rgcn_mem:
        print(f"- Estimated VRAM for training: {rgcn_mem[max_batch]}+ GB")
    else:
        print(f"- Estimated VRAM for training: 16+ GB")
    
    # Estimate processing time
    if 'Full Pipeline' in df['component'].values:
        avg_time = df[df['component'] == 'Full Pipeline']['time_seconds'].mean()
        print(f"- Average processing time per query: {avg_time:.2f} seconds on {device}")
        
        # Estimate throughput
        print(f"- Estimated throughput: {int(60/avg_time)} queries/minute on {device}")
    
    print("\nThank you for using the CARDIO-LR Benchmark Tool!")

if __name__ == "__main__":
    main()