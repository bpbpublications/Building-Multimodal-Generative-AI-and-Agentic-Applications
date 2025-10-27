# rag/utils.py
import json
import re

def safe_json_parse(text):
    try:
        match = re.search(r'{.*?}', text.strip(), re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            raise ValueError("No JSON found in LLM output")
    except Exception as e:
        raise ValueError(f"JSON parsing failed: {e}\nRaw Text:\n{text}")
