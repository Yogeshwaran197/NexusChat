import os
import psycopg
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import START, END , StateGraph
from langgraph.types import Command, interrupt
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import connect
from psycopg.rows import dict_row
from typing import TypedDict, List, Sequence, Annotated, Literal
from pydantic import BaseModel, Field

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("TAVILY API, Invalid API key, give right one")


llm =  ChatGroq(
    model = "openai/gpt-oss-120b",
    api_key=GROQ_API_KEY
)


class AgentState(TypedDict):

    original_query : str
    query : str
    messages : Annotated[Sequence[str], add_messages]
    websearch : bool
    validate_query : bool
    grounded_answer : bool


class llm_validate_query(BaseModel):

    needs_rewrite_query: Literal["yes", "no"] = Field(
    ...,
    description="'yes' if the query is vague or incomplete and must be rewritten before answering, 'no' if it is clear as-is."
)

class llm_ground_answer(BaseModel):

    grounded : Literal["yes", "no"] = Field(
        ...,
        description="'yes' if every claim in the answer is supported by the provided search results, 'no' if any part is unsupported or made up."
        
    )

