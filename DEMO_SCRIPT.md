# NMAFC Demo Video Script

Complete walkthrough for hackathon submission video. Two agents, different
conversations, full memory lifecycle.

---

## Setup (before recording)

```bash
# Terminal 1 -- Ollama
ollama serve &
ollama pull nomic-embed-text

# Terminal 2 -- NMAFC (backend + frontend)
nmafc start

# Terminal 3 -- CLI chat (Agent 1)
nmafc chat

# Browser -- web UI at http://localhost:3000
```

Keep terminals visible. The CLI chat is the star; the web UI shows live state.

---

## Part 1: Agent 1 -- Personal Assistant (13 turns)

### Turn 1 -- Introduction

```
You: My name is Sarah Chen and I work as a senior software engineer at Stripe.
```

Expected extraction:
- CoreAnchor: user_name = "Sarah Chen"
- CoreAnchor: user_job = "Senior software engineer at Stripe"

### Turn 2 -- Location

```
You: I just moved to Brooklyn last weekend from San Francisco.
```

Expected extraction:
- ActiveContext: user_location = "Brooklyn, New York"
- EphemeralState: relocation_event = "Moved from San Francisco to Brooklyn last weekend"

### Turn 3 -- Preferences

```
You: I'm vegetarian and I have a cat named Miso.
```

Expected extraction:
- CoreAnchor: user_diet = "Vegetarian"
- CoreAnchor: user_pet = "Cat named Miso"

### Turn 4 -- Work project

```
You: At work I'm leading the migration from our legacy payment processing system to a new microservices architecture. It's been going on for three months.
```

Expected extraction:
- ActiveContext: work_project = "Leading migration from legacy payment processing to microservices"
- EphemeralState: project_timeline = "Migration has been going on for three months"

### Turn 5 -- Check state

```
/stats
```

Show the Hot RAM record count, types breakdown, and avg weight.
Then run:

```
/memory
```

Point out CoreAnchor records at weight 1.0, ActiveContext decaying.

### Turn 6 -- Override detection (contradiction)

```
You: Actually I moved back to San Francisco. Brooklyn was a mistake.
```

Narrate: "Watch -- the system detects that San Francisco contradicts
Brooklyn. The old record is suppressed."

Expected:
- Old Brooklyn record weight *= 0.1 (suppression event)
- New ActiveContext: user_location = "San Francisco"

### Turn 7 -- Second override

```
You: I got promoted last week. I'm now a staff engineer.
```

Expected:
- Old "senior software engineer" record suppressed
- New CoreAnchor: user_job = "Staff engineer at Stripe"

### Turn 8 -- Verify overrides

```
/memory
```

Narrate: "The old records now have weight 0.1. They're near the prune
threshold and will be evicted on the next prune cycle."

### Turn 9 -- Event log

```
/events
```

Narrate: "Every cognitive event is logged -- weight updates, overrides,
suppressions. This is the full audit trail."

### Turn 10 -- More facts

```
You: My favorite programming language is Rust. I've been using it for personal projects for two years.
```

Expected:
- CoreAnchor: user_language = "Rust"
- ActiveContext: user_language_experience = "Using Rust for personal projects for two years"

### Turn 11 -- Lifestyle

```
You: I'm training for a marathon in October. I run every morning at 6am.
```

Expected:
- ActiveContext: user_fitness_goal = "Training for a marathon in October"
- EphemeralState: user_routine = "Runs every morning at 6am"

### Turn 12 -- Check stats again

```
/stats
```

Point out: "We're at turn 12 now. The avg weight has dropped because
EphemeralState records decay fast -- that's the Ebbinghaus forgetting
curve in action."

### Turn 13 -- Rollback

```
/rollback 5
```

Narrate: "This reconstructs memory state from turn 5 by replaying
the Cold ROM event log. Everything after turn 5 is undone."

```
/memory
```

Show that the marathon, Rust preference, and staff engineer promotion
are gone. We're back to the Brooklyn era.

---

## Part 2: Agent 2 -- Customer Support (10 turns)

### Switch agent

```
/quit
```

```bash
NMAFC_AGENT_ID=support-bot NMAFC_CONVERSATION_ID=ticket-4521 nmafc chat
```

Narrate: "Now we're a completely different agent with a different
conversation. The memory is isolated -- Sarah Chen's data is invisible here."

### Turn 1 -- Ticket context

```
You: The customer is reporting that their order #8834 was charged twice on their credit card. They placed the order three days ago.
```

Expected:
- ActiveContext: ticket_issue = "Customer order #8834 charged twice"
- ActiveContext: ticket_order = "Order placed three days ago"

### Turn 2 -- Customer info

```
You: Customer name is Marcus Webb, email is marcus@example.com. He's on the premium plan.
```

Expected:
- CoreAnchor: customer_name = "Marcus Webb"
- CoreAnchor: customer_email = "marcus@example.com"
- CoreAnchor: customer_plan = "Premium"

### Turn 3 -- Resolution attempt

```
You: I've initiated a refund for the duplicate charge. It should appear in 3-5 business days.
```

Expected:
- ActiveContext: ticket_status = "Refund initiated for duplicate charge"
- EphemeralState: ticket_resolution = "Refund appears in 3-5 business days"

### Turn 4 -- Escalation

```
You: The customer says the refund is not enough. They want compensation for the inconvenience. I'm escalating to a supervisor.
```

Expected:
- Override on ticket_status: updated to "Escalated to supervisor"
- ActiveContext: customer_sentiment = "Dissatisfied, requesting compensation"

### Turn 5 -- Check state

```
/stats
/memory
```

Narrate: "Completely different memory from Agent 1. This is the customer
support agent's isolated context."

### Turn 6 -- Supervisor response

```
You: Supervisor approved a $50 credit to the customer's account. Marcus was satisfied with this resolution.
```

Expected:
- ActiveContext: ticket_resolution = "Supervisor approved $50 credit"
- CoreAnchor: customer_sentiment = "Satisfied after compensation"

### Turn 7 -- Ticket closure

```
You: Ticket resolved. Marcus thanked us and said he'll continue using the premium plan.
```

Expected:
- ActiveContext: ticket_status = "Resolved"
- EphemeralState: customer_feedback = "Thanked us, will continue premium plan"

### Turn 8 -- Events

```
/events
```

Show the support agent's event log -- completely separate from Agent 1.

### Turn 9 -- Stats

```
/stats
```

Narrate: "Different agent, different conversation, fully isolated memory.
No cross-contamination with Sarah Chen's personal assistant data."

### Turn 10 -- Graph

```
/memory
```

Show the entity graph for this conversation -- Marcus Webb, ticket entities,
all scoped to this agent+conversation pair.

---

## Part 3: Web UI Tenant Switch (2 minutes)

### Narrate over browser

1. Open http://localhost:3000
2. Show the Dashboard -- it starts empty or with Agent 1's data (default)
3. In the sidebar, type "support-bot" in the Agent field and
   "ticket-4521" in the Conversation field
4. Click Switch
5. The entire UI refreshes -- Dashboard shows the support agent's stats
6. Navigate to Memory Explorer -- show Marcus Webb's records
7. Navigate to Entity Graph -- show the support agent's entity graph
8. Navigate to Event Timeline -- show the support agent's events
9. Navigate to Decay Curves -- show the active context decay projections
10. Switch back to Agent=default, Conversation=default
11. The UI refreshes again -- now Sarah Chen's data is shown
12. Navigate to Memory Explorer -- show Sarah's records
13. Point out: "Same UI, same backend, completely isolated memory spaces"

---

## Part 4: Documentation Page (30 seconds)

Click Documentation in the sidebar. Scroll through the in-app docs:
- Overview
- Quick Start
- CLI Reference
- API Endpoints
- Configuration
- Library Usage
- Architecture
- Providers
- Decay System

Narrate: "The full documentation lives inside the app itself."

---

## Narration Cheat Sheet

### Opening (15 seconds)
"NMAFC is a neuromorphic memory architecture for conversational AI. It
mimics how biological memory works -- facts decay over time, important ones
get reinforced through retrieval, and contradictions are immediately
suppressed. Let me show you."

### During Part 1 (2-3 minutes)
- Turn 1-4: "Watch as I talk to the assistant. Every fact is classified
  into three tiers: permanent identity facts, current context, and
  transient state. Each tier has a different decay rate."
- Turn 5: "At turn 5, here's what the system remembers."
- Turn 6: "Now I contradict myself. The system detects the override and
  suppresses the old record instantly."
- Turn 8: "The old facts are nearly gone. They'll be pruned on the next
  cycle."
- Turn 9: "Every cognitive event is logged. This isn't just logging --
  it's the system's audit trail that enables rollback."
- Turn 13: "And we can roll back. The Cold ROM replays every event and
  reconstructs the exact state from turn 5."

### During Part 2 (1-2 minutes)
"Now let me show multi-tenancy. I'm launching a second agent -- a customer
support bot handling a billing ticket. Completely different conversation,
completely isolated memory. The two agents never see each other's data."

### During Part 3 (1 minute)
"The web UI supports live tenant switching. I change the agent ID and
conversation ID in the sidebar, and the entire dashboard refreshes with
the correct memory scope."

### Closing (15 seconds)
"NMAFC gives conversational AI real memory -- forgetful, prioritized,
recoverable. The memory horizon outlasts the conversation. The source is
open, the architecture is pluggable, and it works today."
