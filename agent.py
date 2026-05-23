"""
Agent orchestrator for Session 6.
Wires the four cognitive roles (Memory, Perception, Decision, Action) together.
"""
import uuid
import asyncio
import subprocess
import sys
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Import the four cognitive roles
import perception
import decision
import action
from memory import get_memory
from artifacts import get_artifact_store
from schemas import Goal, DecisionOutput

# Constants
MAX_ITERATIONS = 20
GATEWAY_URL = "http://localhost:8101"

# Global instances
memory = get_memory()
artifacts = get_artifact_store()


def ensure_gateway():
    """Check that the LLM gateway is running at localhost:8101."""
    import httpx

    try:
        response = httpx.get(f"{GATEWAY_URL}/v1/status", timeout=5.0)
        if response.status_code == 200:
            print(f"[agent] Gateway is running at {GATEWAY_URL}")
            return
        raise RuntimeError(
            f"Gateway health check failed with status {response.status_code} at {GATEWAY_URL}/v1/status"
        )
    except Exception as exc:
        raise RuntimeError(
            f"Gateway not reachable at {GATEWAY_URL}. "
            "Start it with: cd llm_gatewayV3 && ./run.sh"
        ) from exc


@asynccontextmanager
async def mcp_session():
    """Create an MCP client session connected to mcp_server.py via stdio."""
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def load_tools(session: ClientSession) -> list:
    """Load available tools from the MCP session."""
    result = await session.list_tools()
    # FastMCP returns a ListToolsResult with a tools attribute
    if hasattr(result, 'tools'):
        return list(result.tools)
    return []


def mcp_tools_for_decision(mcp_tools: list) -> list[dict]:
    """
    Convert MCP tool list to the format expected by the LLM Gateway V3.
    Gateway expects tools as ToolDef objects with: name, description, input_schema.
    
    FastMCP tools have: name, description, inputSchema attributes.
    """
    formatted_tools = []
    for tool in mcp_tools:
        # FastMCP tools are objects with attributes
        name = getattr(tool, 'name', '')
        description = getattr(tool, 'description', '')
        input_schema = getattr(tool, 'inputSchema', {})
        
        # Convert inputSchema to dict if it's a Pydantic model or similar
        if not isinstance(input_schema, dict):
            if hasattr(input_schema, 'model_dump'):
                input_schema = input_schema.model_dump()
            elif hasattr(input_schema, 'dict'):
                input_schema = input_schema.dict()
            else:
                input_schema = {}
        
        formatted_tools.append({
            "name": name,
            "description": description,
            "input_schema": input_schema
        })
    
    return formatted_tools


def final_answer_from(history: list[dict]) -> str:
    """
    Extract the final answer from history.
    Returns the last answer event's text, or a summary if no answer exists.
    """
    # Find the last answer event
    for event in reversed(history):
        if event.get('kind') == 'answer':
            return event.get('text', 'No answer provided.')
    
    # Fallback: summarize the last few actions
    recent_actions = [e for e in history if e.get('kind') == 'action'][-3:]
    if recent_actions:
        summary_parts = ["Completed actions:"]
        for action in recent_actions:
            tool = action.get('tool', 'unknown')
            result = action.get('result_descriptor', '')[:100]
            summary_parts.append(f"- {tool}: {result}")
        return "\n".join(summary_parts)
    
    return "No answer or actions recorded."


async def run(query: str) -> str:
    """
    Main agent loop. Orchestrates the four cognitive roles.
    
    Five key invariants:
    1. memory.remember(query) runs BEFORE the loop (durable memory contract)
    2. Memory is read at the TOP of every iteration
    3. Perception runs every iteration and maintains goal state
    4. Decision works on one goal at a time
    5. Action dispatches tools and creates artifacts when needed
    
    Args:
        query: User's natural language query
    
    Returns:
        Final answer as a string
    """
    ensure_gateway()
    run_id = uuid.uuid4().hex[:8]
    history: list[dict] = []
    prior_goals: list[Goal] = []

    # Durable memory: classify the user's query so facts/preferences
    # in it survive into future runs.
    memory.remember(query, source="user_query", run_id=run_id)

    async with mcp_session() as session:
        mcp_tools = await load_tools(session)
        tools = mcp_tools_for_decision(mcp_tools)

        for it in range(1, MAX_ITERATIONS + 1):
            print(f"\n[agent] === ITERATION {it} ===")
            hits = memory.read(query, history)
            print(f"[agent] Memory hits: {len(hits)}")
            for hit in hits:
                print(f"  - [{hit.kind}] {hit.descriptor}")
                
            obs = await perception.observe(query, hits, history, prior_goals, run_id)
            prior_goals = obs.goals
            print(f"[agent] Perception Goals:")
            for g in obs.goals:
                status = "DONE" if g.done else "OPEN"
                attach_note = f" (attached: {g.attach_artifact_id})" if g.attach_artifact_id else ""
                print(f"  - [{status}] {g.text}{attach_note}")
                
            if obs.all_done:
                print(f"[agent] All goals completed. Breaking loop.")
                break

            goal = obs.next_unfinished()
            if goal is None:
                break
            
            print(f"[agent] Current Goal: {goal.text}")
            attached = []
            if goal.attach_artifact_id and artifacts.exists(goal.attach_artifact_id):
                attached_bytes = artifacts.get_bytes(goal.attach_artifact_id)
                attached.append((goal.attach_artifact_id, attached_bytes))
                print(f"[agent] Attached artifact {goal.attach_artifact_id} ({len(attached_bytes)} bytes)")

            out = await decision.next_step(goal, hits, attached, history, tools)

            if out.is_answer:
                print(f"[agent] Decision: Provide Answer")
                print(f"  Answer preview: {out.answer[:150]}...")
                history.append({"iter": it, "kind": "answer",
                                "goal_id": goal.id, "text": out.answer})
                continue

            if out.tool_call is None:
                history.append({"iter": it, "kind": "error",
                                "goal_id": goal.id, "text": "Decision returned neither answer nor tool_call"})
                continue
            
            print(f"[agent] Decision: Call Tool '{out.tool_call.name}'")
            print(f"  Arguments: {out.tool_call.arguments}")
            
            result_text, art_id = await action.execute(session, out.tool_call)
            print(f"[agent] Action: Result preview (first 150 chars): {result_text[:150]}...")
            if art_id:
                print(f"  Created Artifact: {art_id}")
            
            tool_call_dict = {
                "name": out.tool_call.name,
                "arguments": out.tool_call.arguments
            }
            
            memory.record_outcome(
                tool_call=tool_call_dict,
                result_text=result_text,
                artifact_id=art_id,
                run_id=run_id,
                goal_id=goal.id,
            )
            history.append({"iter": it, "kind": "action",
                            "goal_id": goal.id, "tool": out.tool_call.name,
                            "arguments": out.tool_call.arguments,
                            "result_descriptor": result_text[:300],
                            "artifact_id": art_id})

    return final_answer_from(history)


if __name__ == "__main__":
    import asyncio
    import sys

    query = " ".join(sys.argv[1:])

    query_list=["Find 3 family-friendly things to do in Tokyo this weekend.Check Saturday's weather forecast there and tell me which one is most appropriate.",
    "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory.",
    "My mom's birthday is 15 May 2026. Remember that and give me a calendar reminder for two weeks before and on the day.",
    "Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on."]

    for q in  query_list:
        result = asyncio.run(run(q))

        print("\nFINAL ANSWER:\n")
        print(result)