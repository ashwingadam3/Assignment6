"""
Perception role for Session 6 agent.
The orchestrator. Maintains state across iterations.

Four obligations (encoded in system prompt):
1. If prior_goals is empty, decompose query into bounded, imperative goals
2. For each prior goal, examine history; set done: true once satisfied (sticky-done)
3. May set attach_artifact_id on next unfinished goal to one of the artifact handles
4. Preserve goal order; never reorder, insert in middle, or drop

Two structural anti-hallucination choices:
- Positional identity: no goal id field in schema sent to LLM
- Indexed artifact references: present artifacts with integer index
"""
import uuid
from schemas import Goal, MemoryItem, Observation
from llm_gatewayV3.client import LLM


# Synthesis keywords that trigger force-attach safety net
SYNTHESIS_KEYWORDS = {
    "synthesise", "synthesize", "extract", "list", "compare", "decide",
    "summarize", "summarise", "analyze", "analyse", "combine", "merge"
}


async def observe(
    query: str,
    hits: list[MemoryItem],
    history: list[dict],
    prior_goals: list[Goal],
    run_id: str
) -> Observation:
    """
    Orchestrate the goal list with done flags and artifact attachments.
    
    Four obligations:
    1. Decompose query into goals on first call (when prior_goals is empty)
    2. Re-evaluate done flags based on history (sticky-done: never flip back)
    3. Optionally attach artifact_id to next unfinished goal
    4. Preserve goal order (no reorder, no insert, no drop)
    
    Args:
        query: Original user query
        hits: Memory items from keyword search
        history: Run history accumulated so far
        prior_goals: Goal list from previous iteration (empty on first call)
        run_id: Current run identifier
    
    Returns:
        Observation with updated goal list
    """
    llm = LLM()
    
    # Build system prompt encoding all four obligations
    system_prompt = """You are the Perception orchestrator. Your job is to maintain the goal list across iterations.

FOUR OBLIGATIONS (follow exactly):

1. DECOMPOSITION (first call only):
   If the prior goal list is empty, decompose the user query into one or more bounded goals.
   Each goal must be:
   - A short imperative statement (5-15 words)
   - Bounded and actionable
   - Independent enough to work on separately
   
2. DONE FLAGS (every call):
   For each prior goal, examine the run history.
   Mark done: true the moment history contains an action that satisfies the goal.
   STICKY-DONE: Once a goal is marked done, it stays done forever. Never flip back to false.
   
3. ARTIFACT ATTACHMENT (optional):
   For the FIRST unfinished goal only, decide if it needs raw bytes from a fetched artifact.
   If yes, set attach_artifact_id to the artifact_index of one artifact from MEMORY HITS.
   Use the integer index (0, 1, 2, ...), not the artifact handle string.
   Only attach if the goal requires reading/analyzing the artifact content.
   
4. GOAL ORDER (always):
   Preserve the exact order of goals from the prior list.
   Never reorder, never insert in the middle, never drop a goal.
   Only add new goals at the end if decomposing on first call.

OUTPUT FORMAT:
Return a JSON object with a "goals" array. Each goal has:
- text: short imperative description
- done: boolean (true if satisfied by history)
- artifact_index: integer or null (index from MEMORY HITS, only for next unfinished goal)

Do NOT include an "id" field. Goals are identified by position."""

    # Build user prompt with context
    prompt_parts = []
    
    # Original query
    prompt_parts.append(f"USER QUERY:\n{query}\n")
    
    # Memory hits with indexed artifacts
    if hits:
        prompt_parts.append("MEMORY HITS (relevant facts and prior outcomes):")
        artifact_index_map = {}  # Maps index -> artifact_id
        current_index = 0
        
        for i, hit in enumerate(hits, 1):
            if hit.artifact_id:
                artifact_index_map[current_index] = hit.artifact_id
                prompt_parts.append(
                    f"{i}. [{hit.kind}] {hit.descriptor} "
                    f"[artifact_index={current_index}, handle={hit.artifact_id}]"
                )
                current_index += 1
            else:
                prompt_parts.append(f"{i}. [{hit.kind}] {hit.descriptor}")
        prompt_parts.append("")
    else:
        artifact_index_map = {}
    
    # Run history
    if history:
        prompt_parts.append("RUN HISTORY (actions and answers so far):")
        for event in history:
            iter_num = event.get('iter', '?')
            kind = event.get('kind', 'unknown')
            goal_id = event.get('goal_id', '')
            
            if kind == 'action':
                tool = event.get('tool', 'unknown')
                args = event.get('arguments', {})
                result = event.get('result_descriptor', '')[:150]
                artifact_note = f" [created artifact: {event.get('artifact_id')}]" if event.get('artifact_id') else ""
                prompt_parts.append(
                    f"  iter {iter_num}: Action for goal {goal_id}: "
                    f"{tool}({args}) → {result}{artifact_note}"
                )
            elif kind == 'answer':
                text = event.get('text', '')[:150]
                prompt_parts.append(
                    f"  iter {iter_num}: Answer for goal {goal_id}: {text}"
                )
        prompt_parts.append("")
    
    # Prior goals
    if prior_goals:
        prompt_parts.append("PRIOR GOALS (from last iteration):")
        for i, goal in enumerate(prior_goals):
            status = "DONE" if goal.done else "OPEN"
            attachment = f" [attached: {goal.attach_artifact_id}]" if goal.attach_artifact_id else ""
            prompt_parts.append(f"{i+1}. [{status}] {goal.text}{attachment}")
        prompt_parts.append("")
        prompt_parts.append(
            "Re-evaluate each goal based on history. "
            "Mark done: true if history shows the goal is satisfied. "
            "Preserve order. Never drop goals."
        )
    else:
        prompt_parts.append("PRIOR GOALS: (empty - this is the first call)")
        prompt_parts.append("")
        prompt_parts.append(
            "Decompose the user query into one or more bounded, imperative goals. "
            "Each goal should be a short actionable statement."
        )
    
    prompt_parts.append(
        "\nReturn the updated goal list as JSON with structure: "
        '{"goals": [{"text": "...", "done": true/false, "artifact_index": null or integer}]}'
    )
    
    user_prompt = "\n".join(prompt_parts)
    
    # Define the schema for structured output (no id field - positional identity)
    # This is the schema sent to the LLM (Gemini-compatible: no union types, no additionalProperties)
    goal_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "done": {"type": "boolean"},
            "artifact_index": {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "null"}
                ]
            }
        },
        "required": ["text", "done"]
    }
    
    observation_schema = {
        "type": "object",
        "properties": {
            "goals": {
                "type": "array",
                "items": goal_schema
            }
        },
        "required": ["goals"]
    }
    
    # Gateway call with provider="g" (Gemini), temperature=1.0, response_format
    # Retry with backoff to handle Gemini cooldown (503 errors)
    try:
        response = None
        last_err = None
        for attempt in range(4):
            try:
                response = llm.chat(
                    prompt=user_prompt,
                    system=system_prompt,
                    auto_route="perception",
                    provider="g",  # Pin to Gemini for reliability
                    temperature=1.0,  # Avoid Gemini 3.1 flash-lite low-temp looping
                    response_format={
                        "type": "json_schema",
                        "schema": observation_schema,
                        "name": "observation",
                        "strict": True
                    },
                    max_tokens=2000
                )
                break  # Success
            except Exception as retry_err:
                last_err = retry_err
                if attempt < 3 and ("503" in str(retry_err) or "502" in str(retry_err) or "429" in str(retry_err)):
                    import asyncio
                    wait_time = 5 * (2 ** attempt)  # 5, 10, 20 seconds
                    print(f"[perception.observe] Retry {attempt+1}/3 after {wait_time}s: {retry_err}")
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        if response is None:
            raise last_err
        
        # Parse the structured response
        parsed = response.get("parsed", {})
        if not parsed or "goals" not in parsed:
            # Fallback: try to parse from text
            import json
            text = response.get("text", "{}")
            parsed = json.loads(text)
        
        goals_data = parsed.get("goals", [])
        
    except Exception as e:
        print(f"[perception.observe] LLM call failed: {e}")
        # Fallback: preserve prior goals or create a single goal from query
        if prior_goals:
            goals_data = [
                {
                    "text": g.text,
                    "done": g.done,
                    "artifact_index": None
                }
                for g in prior_goals
            ]
        else:
            goals_data = [{"text": query[:100], "done": False, "artifact_index": None}]
    
    # Map positional goals back to Goal objects with stable IDs
    new_goals = []
    for i, goal_data in enumerate(goals_data):
        # Preserve ID from prior_goals if position matches, otherwise create new
        if i < len(prior_goals):
            goal_id = prior_goals[i].id
        else:
            goal_id = f"{run_id}-g{i+1}"
        
        # Map artifact_index back to actual artifact_id
        artifact_index = goal_data.get("artifact_index")
        if artifact_index is not None and artifact_index in artifact_index_map:
            attach_artifact_id = artifact_index_map[artifact_index]
        else:
            attach_artifact_id = None
        
        # Enforce sticky-done: if prior goal was done, keep it done
        done = goal_data.get("done", False)
        if i < len(prior_goals):
            if prior_goals[i].done:
                done = True
        else:
            if not prior_goals:
                done = False
        
        new_goals.append(Goal(
            id=goal_id,
            text=goal_data.get("text", ""),
            done=done,
            attach_artifact_id=attach_artifact_id
        ))
    
    # Force-attach safety net for synthesis goals
    # If next unfinished goal contains synthesis keyword and artifacts exist, attach most recent
    unfinished_idx = next((i for i, g in enumerate(new_goals) if not g.done), None)
    if unfinished_idx is not None:
        goal = new_goals[unfinished_idx]
        goal_text_lower = goal.text.lower()
        
        # Check if goal contains synthesis keywords
        has_synthesis_keyword = any(kw in goal_text_lower for kw in SYNTHESIS_KEYWORDS)
        
        # Check if any artifacts exist in memory hits
        artifacts_available = any(hit.artifact_id for hit in hits)
        
        # If synthesis goal and artifacts exist but no attachment, force-attach most recent
        if has_synthesis_keyword and artifacts_available and not goal.attach_artifact_id:
            # Find most recent artifact in hits
            for hit in reversed(hits):
                if hit.artifact_id:
                    goal.attach_artifact_id = hit.artifact_id
                    print(f"[perception] Force-attached {hit.artifact_id} to synthesis goal: {goal.text}")
                    break
    
    return Observation(goals=new_goals)


# Made with Bob