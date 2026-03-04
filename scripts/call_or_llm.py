import requests
import json

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer <token>",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "openai/gpt-5-mini",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is the capital of sudan?"
          }
        ]
      }
    ]
  })
)

print(response)

"""
curl https://openrouter.ai/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
  "model": "openai/gpt-5-mini",
  "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What is the capital of England?"
          },
        ]
      }
    ]
  
}'
"""