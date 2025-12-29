import requests

url = "http://127.0.0.1:5000/predictdata"
response = requests.post(url)
print("Browser would see:", response.text)
