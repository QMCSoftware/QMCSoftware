""" Unit tests for long-running demo notebooks execution using testbook with Parsl parallelization """
import parsl
import multiprocessing
import time
import unittest
import sys
import os
from parsl.app.app import python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider


# Add the QMCPy path to ensure imports work
current_dir = os.getcwd()
if 'test/ntbktests' in current_dir:
    qmcpy_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
else:
    qmcpy_path = os.path.abspath('../../..')

if qmcpy_path not in sys.path:
    sys.path.insert(0, qmcpy_path)

# Configure Parsl with HighThroughputExecutor
config = Config(
    executors=[
        HighThroughputExecutor(
            label='htex_local',
            worker_debug=False,  # Turn off debug for better performance
            cores_per_worker=10,  # Use more cores per worker for CPU-intensive notebooks
            max_workers_per_node=multiprocessing.cpu_count()//2,  # Better resource utilization
            provider=LocalProvider(
                init_blocks=1,
                max_blocks=2,  # Allow scaling up if needed
                parallelism=0.5,  # More conservative scaling
            ),
            heartbeat_period=30,  # Reduce heartbeat overhead
            heartbeat_threshold=120,  # Give workers more time before declaring them dead
        )
    ],
    retries=1,  # Retry failed tasks once
)
parsl.load(config)

@python_app
def execute_notebook_parsl(notebook_path):
    """Execute a notebook using Parsl app decorator"""
    from testbook import testbook
    
    try:
        with testbook(notebook_path, execute=True) as tb:
            return {"status": "success", "notebook": notebook_path}
    except Exception as e:
        return {"status": "failed", "notebook": notebook_path, "error": str(e)}


class LongNotebookTestsParsl(unittest.TestCase):
    """Test class for long-running notebook execution using Parsl"""
    
    def setUp(self):
        """Set up test notebooks list"""
        self.notebooks = [
            "../../demos/control_variates.ipynb",
            "../../demos/elliptic-pde.ipynb",
            "../../demos/nei_demo.ipynb",
            "../../demos/qei-demo-for-blog.ipynb",
            "../../demos/ray_tracing.ipynb"
        ]
    
    def test_all_notebooks_parallel(self):
        """Execute all notebooks in parallel using Parsl"""
        
        print("Starting parallel notebook execution with Parsl HighThroughputExecutor...")
        start_time = time.time()
        
        # Submit all notebook executions as Parsl futures
        futures = [execute_notebook_parsl(notebook) for notebook in self.notebooks]
        
        # Collect results
        results = []
        for future in futures:
            result = future.result()
            results.append(result)
            if result["status"] == "success":
                print(f"✓ {result['notebook']} executed successfully")
            else:
                print(f"✗ {result['notebook']} failed: {result.get('error', 'Unknown error')}")
        
        end_time = time.time()
        print(f"Parsl execution completed in {end_time - start_time:.2f} seconds")
        
        # Clean up Parsl
        parsl.clear()
        
        # Assert that all notebooks executed successfully
        failed_notebooks = [r for r in results if r["status"] != "success"]
        if failed_notebooks:
            self.fail(f"Failed notebooks: {[r['notebook'] for r in failed_notebooks]}")


if __name__ == '__main__':
        unittest.main()
        # python long_notebook_tests_parsl.py
        #================= 1 passed, 1 warning in 97.41s (0:01:37) =================
        # speedup 1.2