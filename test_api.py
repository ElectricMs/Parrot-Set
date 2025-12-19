import requests

try:
    response = requests.get("http://localhost:8000/stats/species")
    print(f"Status Code: {response.status_code}")
    print(f"Content: {response.text}")
except Exception as e:
    print(f"Error: {e}")
