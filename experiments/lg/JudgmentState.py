import operator
from typing import TypedDict, Optional, List, Annotated

from langchain_core.messages import AnyMessage


class JudgmentState(TypedDict):
    chunks_processed: List[str]  # Processed summaries of each chunk
    full_text_summary: str  # Running summary of the entire text
    judgment_prediction: Optional[str]  # Final prediction (allow/dismiss)
    chunks: List[str]  # List of chunks to process
    current_chunk_idx: int  # Index of the current chunk being processed