import os
from sales_rag_agent import SalesRAGAgent
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

def main():
    pdf_path = '/home/ubuntu/Omniagent/Emaar_FAQ.pdf' # Update as needed
    agent = SalesRAGAgent(pdf_path)
    print("Welcome to the Agentic Sales Assistant! Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            print("Goodbye! Thank you for your interest.")
            break
        if not user_input:
            print("Please enter a message.")
            continue
        result = agent.process(user_input)
        print("\nBot:", result['response'])
        # Removed lead info and state printout for cleaner output

# --- FastAPI endpoint ---
app = FastAPI()
agent_instance = None

@app.on_event("startup")
def startup_event():
    global agent_instance
    agent_instance = SalesRAGAgent(pdf_path='/home/ubuntu/Omniagent/Emaar_FAQ.pdf')

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    message = data.get("message", "")
    if not message:
        return JSONResponse({"error": "No message provided."}, status_code=400)
    result = agent_instance.process(message)
    return JSONResponse(result)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        main()
