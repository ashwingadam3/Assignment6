"""
Decision role for Session 6 agent.
One LLM call, two possible outputs: a final answer OR a single tool call.

Picks the next action for one bounded goal. Never both.
Does not narrate, does not pick more than one tool, does not declare goals done.
"""
from schemas import Goal, MemoryItem, DecisionOutput, ToolCall
from llm_gatewayV3.client import LLM


async def next_step(
    goal: Goal,
    hits: list[MemoryItem],
    attached: list[tuple[str, bytes]],
    history: list[dict],
    mcp_tools: list[dict],
) -> DecisionOutput:
    """
    Pick the next action for one bounded goal.
    
    Returns either:
    - A final answer in plain text (DecisionOutput with answer set)
    - A single tool call (DecisionOutput with tool_call set)
    
    Never returns both. Never picks more than one tool.
    
    Args:
        goal: The current goal to work on
        hits: Relevant memory items from keyword search
        attached: List of (artifact_id, bytes) tuples attached by Perception
        history: Run history accumulated so far
        mcp_tools: Available MCP tools in JSON Schema format
    
    Returns:
        DecisionOutput with exactly one field populated
    """
    llm = LLM()
    
    # Build system prompt with exactly three rules
    system_prompt = """You are a decision-making agent. Your job is to pick the next action for the current goal.

RULES (follow exactly):

1. Respond with EXACTLY ONE of two outputs:
   - A final answer in plain text (when you can answer the goal directly), OR
   - A single tool call (when you need to use a tool to make progress)
   Never provide both. Never narrate or explain your choice.

2. Artifact handles (strings starting with "art:") are internal references to the artifact store.
   - Do NOT pass artifact handles as path or url arguments to tools
   - Tools accept real file paths and URLs only
   - If you need artifact content, it appears under ATTACHED ARTIFACTS in your context
   - Passing an artifact handle to a tool will cause an error

3. For extraction, list, comparison, or selection goals, your answer must be substantive:
   - At least 3 sentences, OR
   - A list of items with details
   - No meta-answers like "I have fetched the page" or "How would you like to proceed?"
   - Do the actual work the goal requires

4. DO NOT CHEAT WITH SEARCH SNIPPETS: If the goal requires you to read, fetch, analyze, or summarize the content of search results, web pages, articles, links, or files, you MUST call `fetch_url` or `read_file` to get their full content. Search snippets or previews in memory hits/artifacts are ONLY index references; they do NOT contain the full content. Do not attempt to answer or summarize directly using search results or snippets from memory. You must call `fetch_url` on the actual URL(s) to fetch them first.
   - For multiple URLs, call `fetch_url` on one URL at a time.
   - You are FORBIDDEN from providing a final answer until you have fetched the full content of all target URLs.

When you call a tool, use the tool calling mechanism. When you answer, provide the complete answer as plain text."""

    # Build user prompt with context
    prompt_parts = []
    
    # Current goal
    prompt_parts.append(f"CURRENT GOAL:\n{goal.text}\n")
    
    # Inject dynamic guidance for reading/fetching/analyzing goals to prevent cheating from search snippets
    goal_text_lower = goal.text.lower()
    if any(kw in goal_text_lower for kw in ["read", "fetch", "analyze", "summarize", "extract", "retrieve"]):
        prompt_parts.append(
            "CRITICAL WARNING: This goal requires reading, fetching, analyzing, or summarizing actual page/file contents. "
            "You are STRICTLY FORBIDDEN from providing a final answer using only search result snippets or memory hit previews. "
            "If you have not called `fetch_url` or `read_file` on the target pages/files yet to read their full text, you MUST call "
            "the tool now. You must fetch each URL/file one by one. Output a tool call to `fetch_url` or `read_file` for the first "
            "un-fetched URL/file that you need to read.\n"
        )

    
    # Memory hits
    if hits:
        prompt_parts.append("MEMORY HITS (relevant facts and prior outcomes):")
        for i, hit in enumerate(hits, 1):
            artifact_note = f" [artifact: {hit.artifact_id}]" if hit.artifact_id else ""
            prompt_parts.append(f"{i}. [{hit.kind}] {hit.descriptor}{artifact_note}")
        prompt_parts.append("")
    
    # Attached artifacts
    if attached:
        prompt_parts.append("ATTACHED ARTIFACTS (raw content for this goal):")
        for artifact_id, content_bytes in attached:
            try:
                content_text = content_bytes.decode('utf-8', errors='replace')
                # Truncate very large content
                if len(content_text) > 100000:
                    content_text = content_text[:100000] + "\n\n[... content truncated ...]"
                prompt_parts.append(f"\n--- {artifact_id} ---")
                prompt_parts.append(content_text)
                prompt_parts.append(f"--- end {artifact_id} ---\n")
            except Exception as e:
                prompt_parts.append(f"\n[Error decoding {artifact_id}: {e}]\n")
        prompt_parts.append("")
    
    # Recent history (last 5 events for context)
    if history:
        prompt_parts.append("RECENT HISTORY:")
        for event in history[-5:]:
            kind = event.get('kind', 'unknown')
            if kind == 'action':
                tool = event.get('tool', 'unknown')
                result = event.get('result_descriptor', '')[:150]
                prompt_parts.append(f"  - Action: {tool} → {result}")
            elif kind == 'answer':
                text = event.get('text', '')[:150]
                prompt_parts.append(f"  - Answer: {text}")
        prompt_parts.append("")
    
    prompt_parts.append("Decide: If you have already fetched and have the full contents of the target pages/files in your context (under ATTACHED ARTIFACTS or MEMORY HITS), provide a final answer. Otherwise, you MUST call a tool (like fetch_url or read_file) on one of the target URLs/paths to retrieve the full content first. Do not summarize from snippets.")
    
    user_prompt = "\n".join(prompt_parts)
    
    # Make the gateway call with auto_route="decision"
    # Retry with backoff to handle provider cooldown (502/503 errors)
    response = None
    last_err = None
    for attempt in range(4):
        try:
            response = llm.chat(
                prompt=user_prompt,
                system=system_prompt,
                auto_route="decision",
                provider="g",  # Pin to Gemini for reliable tool-calling and reasoning
                tools=mcp_tools,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=4000
            )
            break  # Success
        except Exception as retry_err:
            last_err = retry_err
            if attempt < 3 and ("503" in str(retry_err) or "502" in str(retry_err) or "429" in str(retry_err)):
                import asyncio
                wait_time = 5 * (2 ** attempt)  # 5, 10, 20 seconds
                print(f"[decision.next_step] Retry {attempt+1}/3 after {wait_time}s: {retry_err}")
                await asyncio.sleep(wait_time)
            else:
                raise
    
    if response is None:
        raise last_err
    
    # Check if response has tool_calls
    tool_calls = response.get("tool_calls", [])
    
    if tool_calls and len(tool_calls) > 0:
        # Return the first tool call wrapped in ToolCall
        first_call = tool_calls[0]
        tool_call = ToolCall(
            name=first_call.get("name", first_call.get("function", {}).get("name", "")),
            arguments=first_call.get("arguments", first_call.get("function", {}).get("arguments", {}))
        )
        return DecisionOutput(answer=None, tool_call=tool_call)
    else:
        # Return the text as answer
        answer_text = response.get("text", "").strip()
        if not answer_text:
            answer_text = "No response generated. Please try rephrasing the goal."
        return DecisionOutput(answer=answer_text, tool_call=None)


# Made with Bob