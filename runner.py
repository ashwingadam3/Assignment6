#!/usr/bin/env python3
"""
Runner script for agent.py
"""
import asyncio
from agent import run

async def main():
    query = "What is 2 + 2?"
    print(f"[runner] Starting agent with query: {query}")
    result = await run(query)
    print(f"\n[runner] Final result:\n{result}")

if __name__ == "__main__":
    asyncio.run(main())

# Made with Bob
