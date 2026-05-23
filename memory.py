"""
Memory service for Session 6 agent.
Implements typed storage with keyword search, LLM classification, and persistence.
"""
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal
from schemas import MemoryItem
from llm_gatewayV3.client import LLM

# Storage configuration
STATE_DIR = Path(__file__).parent / "state"
MEMORY_FILE = STATE_DIR / "memory.json"

# Stopwords for keyword search
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "should",
    "could", "may", "might", "must", "can", "of", "to", "in", "on", "at",
    "for", "with", "from", "by", "about", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "over", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "and", "but", "or", "if", "because", "while", "this", "that", "these", "those"
}


class Memory:
    """Typed memory service with keyword search and LLM classification."""
    
    def __init__(self):
        self.items: list[MemoryItem] = []
        self.llm = LLM()
        self._load()
    
    def _load(self):
        """Load memory from persistent storage."""
        if MEMORY_FILE.exists():
            with open(MEMORY_FILE, 'r') as f:
                data = json.load(f)
                self.items = [MemoryItem(**item) for item in data]
    
    def _save(self):
        """Save memory to persistent storage."""
        STATE_DIR.mkdir(exist_ok=True)
        with open(MEMORY_FILE, 'w') as f:
            json.dump([item.model_dump(mode='json') for item in self.items], f, indent=2, default=str)
    
    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for keyword search, removing stopwords."""
        tokens = text.lower().split()
        return [t.strip('.,!?;:()[]{}"\'-') for t in tokens if t.lower() not in STOPWORDS and len(t) > 2]
    
    def _keyword_score(self, query_tokens: list[str], item: MemoryItem) -> float:
        """Calculate keyword overlap score between query and memory item."""
        item_tokens = set(item.keywords + self._tokenize(item.descriptor))
        query_set = set(query_tokens)
        
        if not query_set or not item_tokens:
            return 0.0
        
        intersection = query_set & item_tokens
        return len(intersection) / len(query_set)
    
    def read(self, query: str, history: list[dict], kinds: Optional[list[str]] = None, top_k: int = 8) -> list[MemoryItem]:
        """
        Keyword-based retrieval. No LLM cost.
        Returns top-k items ranked by keyword overlap.
        """
        query_tokens = self._tokenize(query)
        
        # Add tokens from recent history
        for event in history[-3:]:
            if 'text' in event:
                query_tokens.extend(self._tokenize(event['text']))
        
        # Filter by kinds if specified
        candidates = self.items
        if kinds:
            candidates = [item for item in candidates if item.kind in kinds]
        
        # Score and rank
        scored = [(item, self._keyword_score(query_tokens, item)) for item in candidates]
        scored = [(item, score) for item, score in scored if score > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [item for item, _ in scored[:top_k]]
    
    def filter(self, kinds: Optional[list[str]] = None, goal_id: Optional[str] = None, recent: Optional[int] = None) -> list[MemoryItem]:
        """
        Structured filter by kind, goal_id, or recency. No LLM cost.
        """
        results = self.items
        
        if kinds:
            results = [item for item in results if item.kind in kinds]
        
        if goal_id:
            results = [item for item in results if item.goal_id == goal_id]
        
        if recent:
            results = sorted(results, key=lambda x: x.created_at, reverse=True)[:recent]
        
        return results
    
    def relevant(self, query: str, kinds: Optional[list[str]] = None, top_k: int = 5) -> list[MemoryItem]:
        """
        LLM-scored relevance. One gateway call with auto_route="memory".
        Used when keyword search is insufficient.
        """
        candidates = self.items
        if kinds:
            candidates = [item for item in candidates if item.kind in kinds]
        
        if not candidates:
            return []
        
        # Build prompt for LLM scoring
        items_text = "\n".join([
            f"{i}. [{item.kind}] {item.descriptor}"
            for i, item in enumerate(candidates)
        ])
        
        prompt = f"""Given this query: "{query}"

Rate the relevance of each memory item (0-10 scale):
{items_text}

Return a JSON array of indices for the top {top_k} most relevant items, ordered by relevance.
Example: [3, 1, 7, 2, 5]"""
        
        try:
            response = self.llm.chat(
                prompt=prompt,
                auto_route="memory",
                provider="g",
                temperature=0.3,
                max_tokens=200
            )
            
            # Parse indices from response
            text = response.get("text", "[]")
            import re
            indices = re.findall(r'\d+', text)
            indices = [int(i) for i in indices if int(i) < len(candidates)][:top_k]
            
            return [candidates[i] for i in indices]
        except Exception as e:
            print(f"[memory.relevant] LLM scoring failed: {e}, falling back to keyword search")
            return self.read(query, [], kinds=kinds, top_k=top_k)
    
    def remember(self, raw_text: str, source: str, run_id: str, goal_id: Optional[str] = None) -> MemoryItem:
        """
        Classify and store ambiguous content. One LLM call with auto_route="memory".
        Extracts kind, keywords, descriptor, and structured value.
        """
        prompt = f"""Classify this text into a structured memory item:
"{raw_text}"

Extract:
1. kind: one of ["fact", "preference", "scratchpad"]
   - fact: durable observed truth (dates, locations, relationships)
   - preference: user preference or habit
   - scratchpad: temporary working note for current task
2. keywords: list of 3-8 important tokens for search
3. descriptor: one short human-readable sentence
4. value: structured dict with canonical representation

Return valid JSON matching this schema:
{{
  "kind": "fact|preference|scratchpad",
  "keywords": ["word1", "word2", ...],
  "descriptor": "short summary",
  "value": {{"key": "canonical_value"}}
}}

Examples:
Input: "John's birthday is 15 May 2026"
Output: {{"kind": "fact", "keywords": ["John", "birthday", "May", "2026"], "descriptor": "John's birthday is on 15 May 2026", "value": {{"entity": "John", "attribute": "birthday", "date": "2026-05-15"}}}}

Input: "I prefer morning meetings"
Output: {{"kind": "preference", "keywords": ["prefer", "morning", "meetings"], "descriptor": "User prefers morning meetings", "value": {{"preference_type": "scheduling", "time": "morning", "activity": "meetings"}}}}"""
        
        try:
            import time
            response = None
            last_err = None
            for attempt in range(4):
                try:
                    response = self.llm.chat(
                        prompt=prompt,
                        auto_route="memory",
                        provider="g",
                        temperature=0.3,
                        max_tokens=500
                    )
                    break  # Success
                except Exception as retry_err:
                    last_err = retry_err
                    if attempt < 3 and ("503" in str(retry_err) or "502" in str(retry_err) or "429" in str(retry_err)):
                        wait_time = 5 * (2 ** attempt)
                        print(f"[memory.remember] Retry {attempt+1}/3 after {wait_time}s: {retry_err}")
                        time.sleep(wait_time)
                    else:
                        raise
            
            if response is None:
                raise last_err
            
            # Parse JSON from response
            text = response.get("text", "{}")
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(text)
            
            # Create memory item
            item = MemoryItem(
                id=uuid.uuid4().hex[:12],
                kind=data.get("kind", "scratchpad"),
                keywords=data.get("keywords", self._tokenize(raw_text)[:8]),
                descriptor=data.get("descriptor", raw_text[:100]),
                value=data.get("value", {"raw": raw_text}),
                artifact_id=None,
                source=source,
                run_id=run_id,
                goal_id=goal_id,
                confidence=0.8,
                created_at=datetime.now()
            )
            
            self.items.append(item)
            self._save()
            return item
            
        except Exception as e:
            print(f"[memory.remember] Classification failed: {e}, storing as scratchpad")
            # Fallback: store as scratchpad with basic extraction
            item = MemoryItem(
                id=uuid.uuid4().hex[:12],
                kind="scratchpad",
                keywords=self._tokenize(raw_text)[:8],
                descriptor=raw_text[:100],
                value={"raw": raw_text},
                artifact_id=None,
                source=source,
                run_id=run_id,
                goal_id=goal_id,
                confidence=0.5,
                created_at=datetime.now()
            )
            self.items.append(item)
            self._save()
            return item
    
    def record_outcome(self, tool_call: dict, result_text: str, artifact_id: Optional[str], run_id: str, goal_id: str) -> MemoryItem:
        """
        Record MCP tool dispatch outcome. No LLM cost.
        Kind is always "tool_outcome", keywords from tool name and arguments.
        """
        # Extract keywords from tool name and arguments
        keywords = [tool_call.get("name", "")]
        for key, val in tool_call.get("arguments", {}).items():
            keywords.append(key)
            if isinstance(val, str):
                keywords.extend(self._tokenize(val)[:3])
        
        keywords = [k for k in keywords if k][:10]
        
        # Create descriptor
        args_str = ", ".join([f"{k}={v}" for k, v in tool_call.get("arguments", {}).items()])
        descriptor = f"{tool_call.get('name')}({args_str[:50]}) → {result_text[:80]}"
        
        item = MemoryItem(
            id=uuid.uuid4().hex[:12],
            kind="tool_outcome",
            keywords=keywords,
            descriptor=descriptor,
            value={
                "tool": tool_call.get("name"),
                "arguments": tool_call.get("arguments", {}),
                "result_preview": result_text[:200]
            },
            artifact_id=artifact_id,
            source="tool_dispatch",
            run_id=run_id,
            goal_id=goal_id,
            confidence=1.0,
            created_at=datetime.now()
        )
        
        self.items.append(item)
        self._save()
        return item


# Global instance
_memory = None

def get_memory() -> Memory:
    """Get or create global memory instance."""
    global _memory
    if _memory is None:
        _memory = Memory()
    return _memory

# Made with Bob
