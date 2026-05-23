"""
Action role for Session 6 agent.
Simplest role. No LLM. Three behaviours.

Dispatches MCP tool calls and manages artifact threshold.
"""
from mcp import ClientSession
from schemas import ToolCall
from artifacts import get_artifact_store

# Threshold for creating artifacts (4 KB)
ARTIFACT_THRESHOLD_BYTES = 4096


async def execute(session: ClientSession, tool_call: ToolCall) -> tuple[str, str | None]:
    """
    Execute a tool call through MCP session.
    
    Three behaviors:
    1. Artifact-handle guard: Refuse dispatch if any argument value starts with "art:"
    2. MCP dispatch: Call the tool and collapse content blocks into text
    3. Threshold check: Create artifact if result exceeds 4KB, otherwise return text directly
    
    Args:
        session: Active MCP client session
        tool_call: Typed tool call with name and arguments
    
    Returns:
        tuple[str, str | None]: (descriptor_text, artifact_id_or_None)
    """
    artifacts = get_artifact_store()
    
    # Behavior 1: Artifact-handle guard
    # Check if any argument value starts with "art:"
    for key, value in tool_call.arguments.items():
        if isinstance(value, str) and value.startswith("art:"):
            error_msg = (
                f"Error: Cannot dispatch tool with artifact handle in arguments. "
                f"Argument '{key}' has value '{value}' which is an internal artifact handle, "
                f"not a valid path or URL. Artifact handles reference the artifact store and "
                f"cannot be passed to MCP tools. If you need the content, it should be "
                f"attached to your context by Perception."
            )
            return error_msg, None
    
    # Behavior 2: MCP dispatch
    try:
        result = await session.call_tool(tool_call.name, arguments=tool_call.arguments)
        
        # Collapse content blocks into single text string
        if hasattr(result, 'content') and isinstance(result.content, list):
            # MCP returns content as list of content blocks
            text_parts = []
            for block in result.content:
                if hasattr(block, 'text'):
                    text_parts.append(block.text)
                elif isinstance(block, dict) and 'text' in block:
                    text_parts.append(block['text'])
                elif isinstance(block, str):
                    text_parts.append(block)
            result_text = "\n".join(text_parts)
        elif isinstance(result, dict) and 'content' in result:
            # Handle dict response format
            content = result['content']
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and 'text' in block:
                        text_parts.append(block['text'])
                    elif isinstance(block, str):
                        text_parts.append(block)
                result_text = "\n".join(text_parts)
            else:
                result_text = str(content)
        else:
            result_text = str(result)
        
    except Exception as e:
        error_msg = f"Tool execution failed: {tool_call.name} - {str(e)}"
        return error_msg, None
    
    # Behavior 3: Threshold check
    result_bytes = result_text.encode('utf-8')
    size_bytes = len(result_bytes)
    
    if size_bytes > ARTIFACT_THRESHOLD_BYTES:
        # Create artifact for large results
        artifact_id = artifacts.put(
            result_bytes,
            content_type="text/plain",
            source=f"tool:{tool_call.name}",
            descriptor=f"Result from {tool_call.name}({list(tool_call.arguments.keys())})"
        )
        
        # Return short descriptor with preview
        preview_chars = 200
        preview = result_text[:preview_chars]
        if len(result_text) > preview_chars:
            preview += "..."
        
        descriptor = f"[artifact {artifact_id}, {size_bytes} bytes] preview: {preview}"
        return descriptor, artifact_id
    else:
        # Return text directly for small results
        return result_text, None


# Made with Bob