#!/bin/bash
set -u

export ACCTOKEN=$(curl -s http://localhost:8899/access-token)

JSON_PAYLOAD='{
  "messages": [
    {
      "role": "user",
      "content": "I need a recipe for caramel flan"
    }
  ],
  "max_tokens": 5000,
  "temperature": 0.7
}'

echo "token acquired"
curl -s "$1/v1/chat/completions" \
  -H "Authorization: Bearer $ACCTOKEN" \
  -H "Content-Type: application/json" \
  -d "$JSON_PAYLOAD" | jq '.'
