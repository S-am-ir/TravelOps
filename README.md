# TravelOps &nbsp;·&nbsp;

Personal AI travel assistant for Nepal — built on LangGraph, MCP, and Groq.

Plans trips, searches live flights and hotels, checks weather, schedules Telegram reminders, and generates moodboards. Every outbound message goes through a human confirmation step before anything is sent.

---

## Architecture

![Architecture](architecture.svg)

Three independent MCP servers handle tool access (travel APIs, Telegram, image generation). LangGraph manages state, routing, and the interrupt/resume flow for human-in-the-loop confirmations. The Postgres checkpointer (Supabase) persists conversation threads across restarts.

---

## What it does

**Travel planning** — given a natural language query, the travel agent fires parallel tool calls for flights, hotels, and weather in a single pass, then returns a structured response with NPR price conversions.

**Reminders** — extracts the reminder content and time from the message (including referring back to a prior trip plan in the same thread), pauses for confirmation, then sends via Telegram or schedules it with APScheduler.

**Moodboard generation** — expands a short creative prompt into a detailed visual description and generates images via fal.ai FLUX.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Agent framework | LangGraph | Stateful graph, native interrupt/resume for HITL |
| LLM | qwen3-32b on Groq | Fast, reliable tool calling, good structured output |
| LLM fallback | llama-3.3-70b on Groq | Text-only fallback if primary fails |
| Tool protocol | MCP (FastMCP) | Each server runs independently, clean separation |
| Flight search | Sky-Scrapper (RapidAPI) | Skyscanner data with official API access |
| Hotel search | Booking.com15 (RapidAPI) | Booking.com wrapper via same RapidAPI key |
| Weather | WeatherAPI | Free tier covers Nepal forecasts cleanly |
| Notifications | Telegram Bot API | Simple outbound delivery |
| Image gen | fal.ai FLUX schnell | Fast and cheap for moodboard generation |
| Persistence | Supabase (Postgres) | Thread history across sessions |
| API layer | FastAPI | Thin layer — handles thread routing and interrupt resume |

---

## Setup

**Requirements:** Python 3.11+, a `.env` file (see `.env.example`)

```bash
# Install
pip install -e .

# Start MCP servers (each in its own terminal)
python src/mcp_internals/servers/trave_l.py
python src/mcp_internals/servers/comms.py
python src/mcp_internals/servers/moodboard.py

# Start the API
python src/main.py

# Open index.html in a browser (point it at localhost:8000)
```

---

## Project layout

```
src/
├── main.py               # FastAPI app
├── graph.py              # LangGraph graph definition
├── model_api.py          # Shared LLM factory (qwen3 + llama fallback)
├── config/
│   └── settings.py
├── agents/
│   ├── state.py          # AgentState + Pydantic extraction schemas
│   └── nodes/
│       ├── Orchestrator.py
│       ├── Travel.py
│       ├── Reminder.py
│       └── Creative.py
└── mcp_internals/
    ├── client.py
    └── servers/
        ├── trave_l.py    # Travel tools
        ├── comms.py      # Telegram
        └── moodboard.py  # Image generation
```

---

## Demo

See [DEMO.md](DEMO.md) for annotated screenshots of the main flows.
