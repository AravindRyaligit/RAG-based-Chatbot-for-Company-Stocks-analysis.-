import requests

response = requests.post(
    "http://127.0.0.1:8000/chat/",
    json={"question": "What is the 52 week low for Apollo Tyres Ltd.?"}
)

print(response.json())
