from langgraph.graph import StateGraph, END
from langchain_core.messages import ToolMessage
from src.agents.state import AgentState
from src.config.settings import settings
from src.agents.nodes.Orchestrator import (
    classify_intent_node,
    route_to_agent,
    unknown_handler_node,
)
from src.agents.nodes.Travel import travel_agent_node
from src.agents.nodes.Reminder import reminder_agent_node


def _clean_node(node_fn):
    """Wrap a node function to filter ToolMessages from its return value.
    This prevents tool results from accumulating in the checkpointer across turns.
    ToolMessages from search/hotel/weather APIs are large and only needed within
    a single turn's ReAct loop — not for future turns' short-term memory.
    If the node is interrupted (HITL), messages are NOT filtered so the
    interrupt state is preserved."""

    async def wrapper(state):
        result = await node_fn(state)
        # Don't filter if interrupt happened (HITL needs full state)
        if not isinstance(result, dict):
            return result
        msgs = result.get("messages", [])
        if msgs:
            # Check if graph is paused (interrupt happened)
            from langgraph.types import Interrupt

            has_interrupt = False
            try:
                for m in msgs:
                    if isinstance(m, Interrupt):
                        has_interrupt = True
                        break
            except Exception:
                pass
            if not has_interrupt:
                result["messages"] = [m for m in msgs if not isinstance(m, ToolMessage)]
        return result

    return wrapper


def build_graph(checkpointer=None):
    """Compile the full agent graph with ToolMessage filtering on all nodes."""
    graph = StateGraph(AgentState)

    # Nodes — wrapped to filter ToolMessages from checkpointer storage
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("travel_agent", _clean_node(travel_agent_node))
    graph.add_node("reminder_agent", _clean_node(reminder_agent_node))
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
            "unknown_handler": "unknown_handler",
        },
    )

    # Terminal edges
    for node in ("travel_agent", "reminder_agent", "unknown_handler"):
        graph.add_edge(node, END)

    return graph.compile(checkpointer=checkpointer)


# Checkpointer factories
_pg_pool = None  # Keep connection pool alive for app lifetime


async def create_postgres_checkpointer():
    """
    Async Postgres checkpointer backed by Neon/Supabase.
    Manages connection pool lifetime so it stays alive across requests.

    Key settings to handle Neon/Supabase SSL timeouts:
    - check: ping the DB before lending a connection to detect dead ones
    - max_idle=60: close connections idle for >60s (before Neon/Supabase kills them)
    - max_lifetime=300: never keep a connection longer than 5 minutes
    - TCP keepalives: periodically ping the OS-level connection
    - reconnect_timeout=5: reconnect quickly if a connection dies
    """
    global _pg_pool
    from psycopg_pool import AsyncConnectionPool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    conn_string = settings.supabase_url.get_secret_value()

    async def _check_connection(conn):
        """Verify the connection is still alive before lending it from the pool."""
        await conn.execute("SELECT 1")

    _pg_pool = AsyncConnectionPool(
        conn_string,
        min_size=1,
        max_size=10,
        open=False,
        max_idle=60,        # Close connections idle for >60 seconds
        max_lifetime=300,   # Recycle every connection after 5 minutes
        reconnect_timeout=5,
        check=_check_connection,
        kwargs={
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        },
    )
    await _pg_pool.open()

    # Create checkpointer from the pool
    checkpointer = AsyncPostgresSaver(_pg_pool)

    # Create tables: setup() uses a transaction internally, but migrations 6-8
    # contain CREATE INDEX CONCURRENTLY which cannot run inside a transaction.
    # Run each migration individually in autocommit mode to work around this.
    try:
        async with _pg_pool.connection() as conn:
            await conn.set_autocommit(True)
            for i, migration_sql in enumerate(checkpointer.MIGRATIONS):
                # Replace CONCURRENTLY with regular INDEX for initial setup
                safe_sql = migration_sql.replace("CONCURRENTLY ", "")
                try:
                    await conn.execute(safe_sql)
                except Exception as mig_err:
                    if "already exists" not in str(mig_err):
                        print(f"[Graph] Migration {i} warning: {mig_err}")
    except Exception as setup_err:
        print(f"[Graph] Postgres setup warning (non-fatal): {setup_err}")

    print("[Graph] Postgres checkpointer ready (pool optimized for Neon/Supabase)")
    return checkpointer



async def create_memory_checkpointer():
    """In-memory checkpointer - dev/testing only. State lost on restart"""
    from langgraph.checkpoint.memory import MemorySaver

    print("[Graph] Using MemorySaver checkpointer (no persistance)")
    return MemorySaver()


async def create_agent():
    """Create the agent graph and return (compiled_graph, checkpointer)."""

    if settings.supabase_url:
        try:
            checkpointer = await create_postgres_checkpointer()
        except Exception as e:
            print(
                f"[Graph] Postgres checkpointer failed {str(e)} failling back to memory"
            )
            checkpointer = await create_memory_checkpointer()
    else:
        checkpointer = await create_memory_checkpointer()

    return build_graph(checkpointer=checkpointer), checkpointer
