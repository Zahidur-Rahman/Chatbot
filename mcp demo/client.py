from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

async def main():
    # Debug: Verify API key
    print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))

    client = MultiServerMCPClient({
        "math": {
            "command": "python",
            "args": [os.path.abspath("mathserver.py")],
            "transport": "stdio",
        }
    })

    try:
        tools = await client.get_tools()
        print("Available tools:", tools)
    except Exception as e:
        print(f"Error retrieving tools: {e}")
        return

    model = ChatGroq(model_name="llama3-8b-8192")  # Use valid model
    agent = create_react_agent(model, tools)

    try:
        math_response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "what's (12*4)+3"}]}
        )
        print("Math Response:", math_response['messages'][-1].content)
    except Exception as e:
        print(f"Error invoking agent: {e}")

if __name__ == "__main__":
    asyncio.run(main())