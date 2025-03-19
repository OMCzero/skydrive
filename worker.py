"""
File processing worker script.

This script runs Celery workers for processing files in the background.
"""

if __name__ == "__main__":
    import os
    from task_queue import celery_app
    import multiprocessing
    
    # Determine the number of worker processes
    # Use 1/2 of available cores, but at least 1 and at most 4
    num_cores = multiprocessing.cpu_count()
    num_workers = max(1, min(4, num_cores // 2))
    
    print(f"Starting {num_workers} file processing workers...")
    
    # Use the Celery worker API directly for more control
    celery_app.worker_main(
        argv=[
            'worker',
            '--loglevel=INFO',
            f'--concurrency={num_workers}',
            '--without-gossip',  # Disable gossip for better performance in single machine setup
            '--without-mingle',  # Disable mingle when not needed
            '--pool=prefork',    # Use process pool for CPU-bound tasks
            '--queues=celery',   # Default queue
        ]
    )