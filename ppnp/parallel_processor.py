import multiprocessing as mp
import numpy as np
from typing import List, Callable, Any, Tuple, Dict, Optional, Union
import time
from dataclasses import dataclass
import logging
from concurrent.futures import ProcessPoolExecutor
import os
from abc import ABC, abstractmethod
import torch

# Set the start method to 'spawn' for CUDA compatibility
if torch.cuda.is_available():
    mp.set_start_method('spawn', force=True)

@dataclass
class TaskResult:
    task_id: int
    result: Any
    processing_time: float
    core_id: int
    metadata: Dict[str, Any] = None

class DataChunker(ABC):
    """Abstract base class for data chunking strategies"""
    @abstractmethod
    def create_chunks(self, data: Any, chunk_size: int) -> List[Any]:
        """Create chunks from input data"""
        pass

class WorkloadEstimator(ABC):
    """Abstract base class for workload estimation strategies"""
    @abstractmethod
    def estimate(self, data: Any) -> float:
        """Estimate workload for given data"""
        pass

class DefaultChunker(DataChunker):
    """Default chunking strategy for list-like data"""
    def create_chunks(self, data: List[Any], chunk_size: int) -> List[List[Any]]:
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

class DefaultWorkloadEstimator(WorkloadEstimator):
    """Default workload estimation strategy"""
    def estimate(self, data: Any) -> float:
        if isinstance(data, (list, tuple)):
            return len(data)
        return 1.0

class ParallelProcessor:
    def __init__(
        self,
        num_cores: Optional[int] = None,
        chunk_size: int = 1000,
        max_tasks_per_core: int = 10,
        logging_level: int = logging.INFO,
        chunker: Optional[DataChunker] = None,
        workload_estimator: Optional[WorkloadEstimator] = None,
        result_aggregator: Optional[Callable[[List[Any]], Any]] = None
    ):
        """
        Initialize the parallel processor.
        
        Args:
            num_cores: Number of CPU cores to use. If None, uses all available cores.
            chunk_size: Size of data chunks for processing
            max_tasks_per_core: Maximum number of tasks to assign per core
            logging_level: Logging level for the processor
            chunker: Custom data chunking strategy
            workload_estimator: Custom workload estimation strategy
            result_aggregator: Custom function to aggregate results
        """
        self.num_cores = num_cores or mp.cpu_count()
        self.chunk_size = chunk_size
        self.max_tasks_per_core = max_tasks_per_core
        self.logger = self._setup_logger(logging_level)
        self.chunker = chunker or DefaultChunker()
        self.workload_estimator = workload_estimator or DefaultWorkloadEstimator()
        self.result_aggregator = result_aggregator or (lambda x: x)
        
    def _setup_logger(self, level: int) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('ParallelProcessor')
        logger.setLevel(level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def _process_chunk(
        self, 
        chunk: Any, 
        task_id: int,
        process_func: Callable,
        *args,
        **kwargs
    ) -> TaskResult:
        """Process a single chunk of data"""
        start_time = time.time()
        core_id = os.getpid()
        
        try:
            result = process_func(chunk, *args, **kwargs)
            processing_time = time.time() - start_time
            
            self.logger.debug(
                f"Task {task_id} completed on core {core_id} "
                f"in {processing_time:.2f} seconds"
            )
            
            return TaskResult(
                task_id=task_id,
                result=result,
                processing_time=processing_time,
                core_id=core_id,
                metadata=kwargs.get('metadata', {})
            )
        except Exception as e:
            self.logger.error(f"Error processing task {task_id}: {str(e)}")
            raise

    def process_parallel(
        self,
        data: Any,
        process_func: Callable,
        *args,
        **kwargs
    ) -> Union[List[TaskResult], Any]:
        """
        Process data in parallel using multiple cores.
        
        Args:
            data: Data to process
            process_func: Function to process each chunk of data
            *args, **kwargs: Additional arguments for process_func
            
        Returns:
            Aggregated results if result_aggregator is provided, else List[TaskResult]
        """
        # Create chunks of data
        chunks = self.chunker.create_chunks(data, self.chunk_size)
        self.logger.info(f"Created {len(chunks)} chunks for parallel processing")
        
        # Process chunks in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            # Submit tasks to the executor
            future_to_chunk = {
                executor.submit(
                    self._process_chunk,
                    chunk,
                    i,
                    process_func,
                    *args,
                    **kwargs
                ): i for i, chunk in enumerate(chunks)
            }
            
            # Collect results as they complete
            for future in future_to_chunk:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Task failed: {str(e)}")
                    raise
        
        # Aggregate results if aggregator is provided
        return self.result_aggregator(results)

    def process_with_workload_balancing(
        self,
        data: Any,
        process_func: Callable,
        *args,
        **kwargs
    ) -> Union[List[TaskResult], Any]:
        """
        Process data in parallel with workload balancing.
        
        Args:
            data: Data to process
            process_func: Function to process each chunk of data
            *args, **kwargs: Additional arguments for process_func
            
        Returns:
            Aggregated results if result_aggregator is provided, else List[TaskResult]
        """
        # Estimate workload for each item
        workloads = [self.workload_estimator.estimate(item) for item in data]
        
        # Sort data by workload
        sorted_indices = np.argsort(workloads)[::-1]  # Descending order
        sorted_data = [data[i] for i in sorted_indices]
        
        # Create balanced chunks
        chunks = []
        current_chunk = []
        current_workload = 0
        
        for item in sorted_data:
            item_workload = self.workload_estimator.estimate(item)
            if (len(current_chunk) >= self.chunk_size or 
                current_workload + item_workload > self.max_tasks_per_core):
                chunks.append(current_chunk)
                current_chunk = [item]
                current_workload = item_workload
            else:
                current_chunk.append(item)
                current_workload += item_workload
        
        if current_chunk:
            chunks.append(current_chunk)
        
        self.logger.info(
            f"Created {len(chunks)} balanced chunks for parallel processing"
        )
        
        # Process chunks in parallel
        return self.process_parallel(chunks, process_func, *args, **kwargs)

    def get_performance_metrics(self, results: List[TaskResult]) -> dict:
        """Calculate performance metrics from processing results"""
        processing_times = [r.processing_time for r in results]
        core_usage = {}
        
        for result in results:
            core_id = result.core_id
            if core_id not in core_usage:
                core_usage[core_id] = 0
            core_usage[core_id] += 1
        
        return {
            'total_tasks': len(results),
            'avg_processing_time': np.mean(processing_times),
            'max_processing_time': max(processing_times),
            'min_processing_time': min(processing_times),
            'total_processing_time': sum(processing_times),
            'core_usage': core_usage,
            'metadata': [r.metadata for r in results if r.metadata]
        }

    def set_chunker(self, chunker: DataChunker):
        """Set custom chunking strategy"""
        self.chunker = chunker

    def set_workload_estimator(self, estimator: WorkloadEstimator):
        """Set custom workload estimation strategy"""
        self.workload_estimator = estimator

    def set_result_aggregator(self, aggregator: Callable[[List[Any]], Any]):
        """Set custom result aggregation function"""
        self.result_aggregator = aggregator 