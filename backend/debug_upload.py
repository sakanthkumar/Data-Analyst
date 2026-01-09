import requests
import pandas as pd
import io

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
try:
    response = requests.post(url, files=files, data=data, timeout=5)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
