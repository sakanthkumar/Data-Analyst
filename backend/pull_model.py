import requests
import json
import sys

def pull_model(model_name="nomic-embed-text"):
    url = "http://localhost:11434/api/pull"
    payload = {"name": model_name}
    
    print(f"Attempting to pull model '{model_name}' via API...")
    
    try:
        # Stream=True is important to prevent timeouts on large downloads
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    status = data.get('status', '')
                    
                    # Print progress if available
                    if 'completed' in data and 'total' in data:
                        percent = (data['completed'] / data['total']) * 100
                        sys.stdout.write(f"\r{status}: {percent:.1f}%")
                        sys.stdout.flush()
                    else:
                        print(f"{status}")
                        
            print(f"\nSuccessfully pulled {model_name}!")
            return True
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama. Make sure 'ollama serve' is running.")
        return False
    except Exception as e:
        print(f"\nError pulling model: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        model = "nomic-embed-text"
        
    success = pull_model(model)
    if not success:
        sys.exit(1)
