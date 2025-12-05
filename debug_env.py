from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("GEMINI_API_KEY")

if key:
    print(f"SUCCESS: Found API Key ending in ...{key[-4:]}")
else:
    print("ERROR: GEMINI_API_KEY is missing or empty.")
    # Debug: Print all keys to see what is actually there
    print("Keys found in environment:", [k for k in os.environ.keys() if "GEMINI" in k])