"""Sanity sample: two short chats and one ~6k-token recall question against a local server."""
import json, urllib.request

URL = "http://127.0.0.1:30000/v1/chat/completions"


def ask(content, max_tokens=200):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"]


print("Q1:", ask("What is the capital of France? Answer in one sentence."))
print("Q2:", ask("Write a Python function that returns the n-th Fibonacci number."))
# Past index_topk (2048 tokens): the needle sits early, so only DSA's selection can reach it.
filler = " ".join(f"Line {i}: the weather report says nothing important today." for i in range(600))
doc = "The secret code is 7341-PLUM. " + filler
print("Q3:", ask(doc + "\n\nWhat is the secret code mentioned at the very beginning? Reply with the code only.", 30))
