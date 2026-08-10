import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

try:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": ""
            }
        ]
    )

    print("✅ API Key Working")
    print(response.choices[0].message.content)

except Exception as e:
    print("❌ API Key Error")
    print(e)