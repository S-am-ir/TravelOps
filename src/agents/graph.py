from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.config.settings import settings
from src.agents.nodes.Orchestrator import (
    classify_intent_node,
    route_to_agent,
    unknown_handler_node,
)
from src.agents.nodes.Travel import travel_agent_node
from src.agents.nodes.Reminder import reminder_agent_node
from src.agents.nodes.Creative import creative_agent_node

# Graph builder
def build_graph(checkpointer=None):
    """
    Compile the full agent graph.

    Args:
        checkpointer: A langGraph checkpointer instance (Postgres, SQLite 
        MemorySaver). If None, uses MemorySaver (no persistance).

    """
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("travel_agent", travel_agent_node)
    graph.add_node("reminder_agent", reminder_agent_node)
    graph.add_node("creative_agent", creative_agent_node)
    graph.add_node("unknown_handler", unknown_handler_node)

    # Entry
    graph.set_entry_point("classify_intent")

    # Routing
    graph.add_conditional_edges(
        "classify_intent",
        route_to_agent,
        {
            "travel_agent": "travel_agent",
            "reminder_agent": "reminder_agent",
            "creative_agent": "creative_agent",
            "unknown_handler": "unknown_handler",
        },
    )

    # Terminal edges
    for node in ("travel_agent", "reminder_agent", "creative_agent", "unknown_handler"):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=checkpointer)

# Checkpointer factories
async def create_postgres_checkpointer():
    """
    Async Postgres checkpointer backed by Supabase.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    

    conn_string = settings.supabase_url.get_secret_value()

    # The saver handles connection(s) internally
    checkpointer = AsyncPostgresSaver.from_conn_string(conn_string)

    # Creates tables if missing 
    await checkpointer.setup()

    print("[Graph] Postgres checkpointer ready ")
    return checkpointer

async def create_memory_checkpointer():
    """In-memory checkpointer - dev/testing only. State lost on restart"""
    from langgraph.checkpoint.memory import MemorySaver
    print("[Graph] Using MemorySaver checkpointer (no persistance)")
    return MemorySaver()

async def create_agent():

    if settings.supabase_url:
        try:
            checkpointer = await create_postgres_checkpointer()
        except Exception as e:
            print(f"[Graph] Postgres checkpointer failed {str(e)} failling back to memory")
            checkpointer = create_memory_checkpointer()
    else:
        checkpointer = create_memory_checkpointer()

    return build_graph(checkpointer=checkpointer)



    
