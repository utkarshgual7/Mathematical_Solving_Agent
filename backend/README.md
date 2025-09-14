# Backend

This directory contains the backend implementation of the Mathematical Routing Agent with Agno integration.

## JEE Benchmark Evaluation

The system includes a benchmark evaluation suite for testing the agent's performance against JEE (Joint Entrance Examination) problems.

### Running the Benchmark

To run the JEE benchmark evaluation:

```bash
cd backend
python run_benchmark.py
```

This will:
1. Initialize the mathematical routing agent
2. Load the JEE Main 2025 Math dataset
3. Solve each problem using the agent
4. Evaluate the accuracy of the solutions
5. Generate detailed results and summary statistics

### Benchmark Components

- **Evaluator**: [app/utils/evaluation.py](file:///C:/Users/utkar/Downloads/ai_assign/backend/app/utils/evaluation.py) - Contains the `JEEBenchmarkEvaluator` class
- **Execution Script**: [run_benchmark.py](file:///C:/Users/utkar/Downloads/ai_assign/backend/run_benchmark.py) - Script to execute the benchmark
- **Dataset**: Currently uses a sample dataset, but can be extended to use the full HuggingFace dataset

### Output Files

After running the benchmark, two files will be generated:
1. `jee_benchmark_results_detailed.json` - Detailed results for each problem
2. `jee_benchmark_summary.json` - Summary statistics

### Extending the Benchmark

To use the full PhysicsWallahAI/JEE-Main-2025-Math dataset from HuggingFace, you would need to:
1. Install the datasets library: `pip install datasets`
2. Update the [_load_jee_dataset](file:///C:/Users/utkar/Downloads/ai_assign/backend/app/utils/evaluation.py#L74-L91) method in [app/utils/evaluation.py](file:///C:/Users/utkar/Downloads/ai_assign/backend/app/utils/evaluation.py) to load from HuggingFace