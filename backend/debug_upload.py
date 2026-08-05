import requests
import pandas as pd
import io
import time

# Create a dummy CSV
csv_content = """Machine failure,Sensor1,Sensor2
0,10.5,20.1
1,11.0,22.3
0,10.2,20.0
"""
dummy_file = io.BytesIO(csv_content.encode('utf-8'))

url = "http://localhost:8000/upload"
files = {"file": ("test.csv", dummy_file, "text/csv")}
data = {"machine_name": "Test Bot"}

print("Sending request...")
start_time = time.time()
try:
    response = requests.post(url, files=files, data=data, timeout=10)
    elapsed = time.time() - start_time
    print(f"Upload completed in {elapsed:.4f} seconds.")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Upload failed: {e}")
    exit(1)

print("\nStarting polling simulation...")
profile_url = "http://localhost:8000/domain_profile"
max_polls = 40
poll_interval = 5

for i in range(max_polls):
    try:
        res = requests.get(profile_url)
        data = res.json()
        status = data.get("status", "unknown")
        print(f"Poll #{i+1}: Status = {status}")
        if status == "completed":
            print("Successfully loaded domain profile!")
            print(f"Profile: {data}")
            break
        elif status == "failed":
            print("Profiling failed on backend!")
            break
    except Exception as e:
        print(f"Polling request failed: {e}")
    
    time.sleep(poll_interval)
else:
    print("Polling timed out.")
