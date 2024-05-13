from datetime import datetime
import psutil
import GPUtil
import time
import json
import os


def ensure_dir(directory):
    """Ensure directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def log_hardware(interval=10, log_interval=3, log_file_name='hardware_usage_report',
                 log_dir='./hardware_usage_reports'):

    # Ensure the directory exists
    ensure_dir(log_dir)

    start_time = datetime.now().strftime("%Y%m%d%H%M%S")
    log_file = log_file_name + f'_{start_time}.json'
    log_file = os.path.join(log_dir, log_file)

    logs_data = []
    cycle_count = 0

    while True:
        # Gather data
        cpu_usage = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        timestamp = datetime.now().isoformat()

        # Dictionary of CPU and Log Data
        log_data = {
            'Log': cycle_count,
            'timestamp': timestamp,
            'cpu_usage_percent': cpu_usage,
            'memory_usage_percent': memory.percent,
            'memory_total_GB': memory.total / (1024**3),
            'gpus': []
        }


        gpus = GPUtil.getGPUs()
        gpu = gpus[0]
        gpu_data = {
            'id': gpu.id,
            'name': gpu.name,
            'load_percent': gpu.load * 100,
            'memory_used_MB': gpu.memoryUsed,
            'memory_total_MB': gpu.memoryTotal,
            'temperature_C': gpu.temperature
        }
        log_data['gpus'].append(gpu_data)

        print(log_data, "\n")

        # Increment the cycle count
        cycle_count += 1

        # Append log data to the list
        if cycle_count % log_interval == 0:
            logs_data.append(log_data)
            # write to the json file
            with open(log_file, 'w') as file:
                json.dump(logs_data, file, indent=4)

        # Wait before logging the next entry
        time.sleep(interval)

if __name__ == "__main__":
    log_hardware(1, 30)  # Logs every 1 second, records every 30 log
