# FileSystem Stress Testing Guide

This document outlines approaches for stress-testing the file management system's queue implementation and asynchronous processing capabilities.

## Concurrent Upload Testing

Test how the system handles multiple simultaneous uploads:

```bash
# Using curl to upload multiple files simultaneously
for i in {1..20}; do
  curl -X POST -F "file=@large_file.mp4" http://localhost:8000/upload &
done
```

## Mixed Media Type Batch

Test how the system handles different file types that activate different processing paths:

```python
import requests
import time

# Test with diverse file types to hit different processing paths
files = [
  "large_video.mp4",       # Tests Whisper transcription
  "large_document.pdf",    # Tests PDF extraction
  "complex_image.jpg",     # Tests vision model
  "audio_file.mp3",        # Tests audio transcription
  "spreadsheet.xlsx"       # Tests structured data extraction
]

# Upload all files quickly
for file in files:
  with open(file, 'rb') as f:
    requests.post('http://localhost:8000/upload', files={'file': f})
```

## Large File Testing

Generate and test with large files of different types:

```bash
# Create very large text file
dd if=/dev/urandom bs=1M count=500 | base64 > large_file.txt

# Create large image file
convert -size 10000x10000 plasma:fractal large_image.jpg

# Create large synthetic audio file
ffmpeg -f lavfi -i "sine=frequency=1000:duration=3600" large_audio.mp3
```

## System Monitoring During Tests

Monitor system resources during testing:

```bash
# Install monitoring tools
pip install glances psutil

# Monitor system resources in one terminal
glances

# In another terminal, check Redis queue status
watch -n 1 "redis-cli -n 0 INFO | grep used_memory"
watch -n 1 "redis-cli -n 0 LLEN celery"
```

## Memory Profiling

Add memory profiling to identify potential memory leaks:

```python
# Add to your task_queue.py for profiling specific tasks
import tracemalloc

# Add to process_file function
tracemalloc.start()
# ...process file...
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
print("[ Top 10 memory usage ]")
for stat in top_stats[:10]:
    print(stat)
```

## Reliability Testing

Test system resilience when components fail:

```bash
# Upload files while stopping/starting services
curl -X POST -F "file=@test.jpg" http://localhost:8000/upload
# Stop Redis
docker stop redis-container
# Wait 10 seconds
sleep 10
# Restart Redis
docker start redis-container
```

## Worker Scaling Test

Test how the system performs with different worker configurations:

```bash
# Run multiple workers with different concurrency settings
python worker.py --concurrency=1 &
python worker.py --concurrency=2 &

# Then upload many files and monitor queue length
```

## Task Timeout Testing

Test the system's handling of long-running tasks:

```python
# Create a special test file that triggers long processing
with open('timeout_test.txt', 'w') as f:
    # Write pattern that causes LLM to generate maximum tokens
    f.write("Please write an extremely detailed essay about " * 1000)
```

## Performance Benchmarking

Measure and compare performance metrics:

```bash
# Create a benchmark script
cat > benchmark.py << 'EOF'
import requests
import time
import os
import sys
import statistics

def upload_file(filepath):
    start_time = time.time()
    with open(filepath, 'rb') as f:
        response = requests.post('http://localhost:8000/upload', files={'file': f})
    end_time = time.time()
    
    if response.status_code == 200:
        task_id = response.json().get('task_id')
        return {
            'status': 'success',
            'time': end_time - start_time,
            'task_id': task_id
        }
    else:
        return {
            'status': 'error',
            'time': end_time - start_time,
            'error': response.text
        }

def check_task_completion(task_id, timeout=3600):
    start_time = time.time()
    completed = False
    
    while not completed and (time.time() - start_time) < timeout:
        response = requests.get(f'http://localhost:8000/task/{task_id}')
        if response.status_code == 200:
            data = response.json()
            if data.get('status') in ['SUCCESS', 'ERROR']:
                end_time = time.time()
                return {
                    'status': data.get('status'),
                    'time': end_time - start_time,
                    'data': data
                }
        time.sleep(5)  # Check every 5 seconds
    
    return {
        'status': 'timeout',
        'time': timeout
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <file_path> [iterations]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    upload_times = []
    processing_times = []
    
    for i in range(iterations):
        print(f"Iteration {i+1}/{iterations}")
        
        # Upload file
        result = upload_file(file_path)
        upload_times.append(result['time'])
        
        if result['status'] == 'success':
            # Check task completion
            task_result = check_task_completion(result['task_id'])
            processing_times.append(task_result['time'])
            print(f"  Upload: {result['time']:.2f}s, Processing: {task_result['time']:.2f}s")
        else:
            print(f"  Upload failed: {result.get('error')}")
    
    # Print statistics
    if upload_times:
        print("\nUpload time statistics (seconds):")
        print(f"  Min: {min(upload_times):.2f}")
        print(f"  Max: {max(upload_times):.2f}")
        print(f"  Avg: {statistics.mean(upload_times):.2f}")
        print(f"  Med: {statistics.median(upload_times):.2f}")
    
    if processing_times:
        print("\nProcessing time statistics (seconds):")
        print(f"  Min: {min(processing_times):.2f}")
        print(f"  Max: {max(processing_times):.2f}")
        print(f"  Avg: {statistics.mean(processing_times):.2f}")
        print(f"  Med: {statistics.median(processing_times):.2f}")
EOF

# Run benchmark
python benchmark.py test.mp4 3
```

## Load Testing with Different File Sizes

Test system performance across file size spectrum:

```bash
# Create test files of various sizes
for size in 1 10 50 100 500; do
  dd if=/dev/urandom bs=1M count=$size | base64 > ${size}MB_file.txt
done

# Create test script
cat > size_test.py << 'EOF'
import requests
import time
import os
import json

results = {}

for file in [f for f in os.listdir(".") if f.endswith("MB_file.txt")]:
    size = file.split("_")[0]
    print(f"Testing with {size}MB file...")
    
    start_time = time.time()
    with open(file, 'rb') as f:
        response = requests.post('http://localhost:8000/upload', files={'file': f})
    
    if response.status_code == 200:
        task_id = response.json().get('task_id')
        
        # Poll for completion
        completed = False
        start_processing = time.time()
        
        while not completed:
            task_response = requests.get(f'http://localhost:8000/task/{task_id}')
            if task_response.status_code == 200:
                data = task_response.json()
                if data.get('status') in ['SUCCESS', 'ERROR']:
                    completed = True
                    end_time = time.time()
                    results[size] = {
                        'upload_time': start_processing - start_time,
                        'processing_time': end_time - start_processing,
                        'total_time': end_time - start_time,
                        'status': data.get('status')
                    }
            time.sleep(5)
    else:
        results[size] = {
            'error': response.text,
            'status': 'failed'
        }

# Save results
with open('size_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved to size_test_results.json")
EOF

# Run the test
python size_test.py
```

## Testing Redis Persistence

Test if tasks survive Redis restarts:

```bash
# Set up persistent Redis
docker run -d --name redis-persistent -p 6379:6379 -v $(pwd)/redis-data:/data redis redis-server --appendonly yes

# Create a test
cat > persistence_test.py << 'EOF'
import requests
import time
import os
import sys
import subprocess

def upload_file(filepath):
    with open(filepath, 'rb') as f:
        response = requests.post('http://localhost:8000/upload', files={'file': f})
    
    if response.status_code == 200:
        return response.json().get('task_id')
    return None

def restart_redis():
    subprocess.run(["docker", "restart", "redis-persistent"])
    time.sleep(5)  # Give Redis time to restart

def main():
    if len(sys.argv) < 2:
        print("Usage: python persistence_test.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Upload file
    task_id = upload_file(file_path)
    if not task_id:
        print("Failed to upload file")
        sys.exit(1)
    
    print(f"Uploaded file, task ID: {task_id}")
    time.sleep(2)  # Give time for task to be registered
    
    # Restart Redis
    print("Restarting Redis...")
    restart_redis()
    
    # Check if task survived
    print("Checking task status after Redis restart...")
    time.sleep(5)  # Wait a bit for system to reconnect
    
    retries = 10
    for i in range(retries):
        response = requests.get(f'http://localhost:8000/task/{task_id}')
        if response.status_code == 200:
            data = response.json()
            print(f"Task status: {data.get('status')}")
            print(f"Task message: {data.get('message')}")
            
            if data.get('status') in ['SUCCESS', 'ERROR']:
                print("Task completed after Redis restart!")
                return
            else:
                print(f"Task still processing... ({i+1}/{retries})")
        else:
            print(f"Failed to get task status: {response.text}")
        
        time.sleep(10)
    
    print("Task did not complete within the expected time after Redis restart")

if __name__ == "__main__":
    main()
EOF

# Run the persistence test
python persistence_test.py test.pdf
```

## Profile CPU and Memory Usage

Monitor detailed resource usage:

```bash
# Install profiling tools
pip install memory_profiler psutil

# Create a custom worker with profiling
cat > profiled_worker.py << 'EOF'
import os
from task_queue import celery_app
import psutil
import time
import multiprocessing
from memory_profiler import profile

process = psutil.Process(os.getpid())

@profile
def run_worker():
    # Determine the number of worker processes
    num_cores = multiprocessing.cpu_count()
    num_workers = max(1, min(4, num_cores // 2))
    
    print(f"Starting {num_workers} file processing workers with profiling...")
    
    # Start monitoring
    start_time = time.time()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Start worker
    celery_app.worker_main(
        argv=[
            'worker',
            '--loglevel=INFO',
            f'--concurrency={num_workers}',
            '--without-gossip',
            '--without-mingle',
            '--pool=prefork',
            '--queues=celery',
        ]
    )

if __name__ == "__main__":
    run_worker()
EOF

# Run with profiling
python -m memory_profiler profiled_worker.py
```

These test scripts will help you thoroughly validate the robustness, performance, and reliability of your queue-based file processing system under various load conditions.