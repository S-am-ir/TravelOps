# Multi-Agent AI Travel Planner

An intelligent travel planning assistant powered by multi-agent architecture. Combines real-time flight/hotel search, weather forecasts, web search, and email reminders through a conversational interface.

## Features

- **Travel Planning** — Flights, hotels, weather, bus routes, budget planning with real-time data
- **Email Reminders** — Schedule reminders sent via Gmail SMTP to any email address
- **Conversational Generalist** — Intelligent handling of greetings, date/time, and general casual chat
- **Real-time Web Search** — Tavily-powered search for places, transport, tips, currency rates
- **Streaming Responses** — Token-by-token streaming via Server-Sent Events
- **Persistent Conversations** — Postgres-backed memory with auth (JWT)
- **Multi-model Fallback** — Groq primary with OpenRouter fallback (6 models total)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Frontend (SSE)                              │
│  Auth · Chat · Settings · Email Config                              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ HTTP / SSE
┌──────────────────────────▼──────────────────────────────────────────┐
│                    FastAPI (src/main.py)                            │
│  /auth/* · /chat · /chat/stream · /settings/* · /health            │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                   LangGraph Agent (src/graph.py)                    │
│                                                                     │
│  classify_intent ──► travel_agent ──► END                           │
│        │                reminder_agent ──► END                      │
│        └──────────► general_agent ──► END                           │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │     Tool Library (Native - src/mcp/servers/)                  │  │
│  ├─────────────┬────────────────┬───────────────┬────────────────┤  │
│  │   travel    │    search      │     comms     │    general     │  │
│  │ (RapidAPI)  │   (Tavily)     │    (SMTP)     │   (No tools)   │  │
│  └─────────────┴────────────────┴───────────────┴────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
│  └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
  │ tool: travel | │ tool: search | │ tool : comms │
  │              │ │              │ │              │
  │ • get_weather│ │ • web_search │ │ • send_email │
  │ • search_    │ │ • web_search │ │              │
  │   flights    │ │   _multi     │ │              │
  │ • search_    │ │   (Tavily)   │ │              │
  │   hotels     │ │              │ │              │
  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
         │                │                │
    WeatherAPI      Tavily API        Gmail SMTP
    RapidAPI        (clean, cited     (per-user
    (flights,       real-time         app password)
    hotels)         search)
```

### Model Fallback Chain

```
Groq openai/gpt-oss-120b (primary)
  └─ fail ──► Groq qwen/qwen3-32b (fallback)
                └─ fail ──► OpenRouter qwen3.6-plus-preview:free (last resort, with retry)
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11, FastAPI, LangGraph, LangChain |
| LLM | Groq (primary), OpenRouter (fallback) |
| Search | Tavily API |
| Flights/Hotels | RapidAPI (Sky-Scrapper, Booking.com15) |
| Weather | WeatherAPI |
| Email | Gmail SMTP (per-user app password) |
| Auth | JWT (python-jose) + bcrypt |
| Database | PostgreSQL (checkpointer + user profiles) |
| Container | Docker Compose (5 services) |
| Frontend | Vanilla JS, SSE streaming |

## Quick Start

### Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/S-am-ir/Multi-Agent-AI-Travel-Planner.git
cd Multi-Agent-AI-Travel-Planner

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys (Groq, Tavily, RapidAPI, WeatherAPI)

# Start all services
docker compose up -d

# Open http://localhost:8000
```

### Manual Setup

```bash
# Install dependencies
pip install -e .

# Start the API (Tools are automatically imported natively)
uvicorn src.main:app --reload
```

## API Keys Required

| Service | Purpose | Free Tier |
|---------|---------|-----------|
| [Groq](https://console.groq.com/keys) | Primary LLM | Yes (paid for higher limits) |
| [OpenRouter](https://openrouter.ai/settings/keys) | Fallback LLM | 50 req/day free |
| [Tavily](https://tavily.com) | Web search | 1000 req/month free |
| [RapidAPI](https://rapidapi.com) | Flights + Hotels | Free tier available |
| [WeatherAPI](https://www.weatherapi.com/signup.aspx) | Weather forecasts | Free tier |

## Usage

### Travel Planning
```
"Plan a weekend trip to Pokhara under 15k NPR"
"What's the weather in Kathmandu?"
"Find cheap flights from KTM to PKR"
"Best bus routes from Kathmandu to Pokhara"
```

### Email Reminders
```
"Email me a reminder to book the flight tomorrow at 10am"
"Send a reminder to friend@gmail.com about the trip"
"Email me right now with the trip plan"
```

### Email Setup
1. Open Settings (gear icon)
2. Enter your Gmail address
3. Generate an App Password at [Google App Passwords](https://myaccount.google.com/apppasswords)
4. Paste the 16-character password
5. Click Save

## Docker Services

| Service | Port | Description |
|---------|------|-------------|
| `api` | 8000 | Main agent node, tool integration, and frontend |
| `postgres` | 5432 | PostgreSQL database (Thread memory & Auth) |

## Project Structure

```
src/
├── main.py               # FastAPI app + endpoints
├── graph.py              # LangGraph agent with ToolMessage filtering
├── model_api.py          # 6-model fallback chain
├── email_service.py      # Gmail SMTP sending
├── auth/
│   ├── service.py        # User auth, JWT, SMTP config
│   └── middleware.py     # Auth dependencies
├── config/
│   └── settings.py       # Pydantic settings
├── agents/
│   ├── state.py          # AgentState, IntentClassification, ReminderExtraction
│   └── nodes/
│       ├── Orchestrator.py  # Intent classification & General chat
│       ├── Travel.py        # Travel agent with ReAct loop
│       └── Reminder.py      # Email reminder scheduling
├── mcp/
│   └── servers/          # Native Tool Modules
│       ├── travel.py     # Weather, flights, hotels
│       ├── comms.py      # Email
│       └── search.py     # Tavily web search (optimized asyncio)
frontend/
├── index.html            # Auth, chat, settings UI
├── js/app.js             # SSE streaming, auth flow
└── css/styles.css        # Dark theme, responsive
```

