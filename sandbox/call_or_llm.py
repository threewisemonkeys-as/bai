import requests
import json

response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer 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",
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
  -H "Authorization: Bearer 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" \
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