---
title: "LiveKit: A Complete Guide to Real-Time Communication and AI Voice Agents"
publishDate: "2026-04-17"
category: "software-development"
subcategory: "System Design"
tags:
  [
    "livekit",
    "webrtc",
    "real-time-communication",
    "voice-agents",
    "system-design",
    "sfu",
    "streaming",
    "ai-agents",
    "sip",
  ]
date: "2026-04-17"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "LiveKit is the open-source platform powering real-time audio, video, and AI voice agents. This guide covers SFU architecture, room/participant/track model, the Agents framework, scaling, egress/ingress, SIP telephony, and interview-ready depth on real-time system design."
---

## What Is LiveKit?

LiveKit is an open-source, real-time communication platform built on **WebRTC**. Written in Go (using the Pion WebRTC library), it provides the infrastructure for multi-user video conferencing, live streaming, and — increasingly — **AI-powered voice and video agents**.

Think of LiveKit as the plumbing layer that handles everything needed for real-time media: network transport, codec negotiation, bandwidth adaptation, encryption, recording, telephony bridging, and participant management. You build your application logic on top.

```
┌─────────────────────────────────────────────────────────┐
│                    Your Application                      │
│  (Video call UI, AI agent logic, live stream controls)   │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                      LiveKit                             │
│                                                          │
│  ┌──────────┐ ┌─────────────┐ ┌──────────────────────┐  │
│  │   SFU    │ │   Agents    │ │  Egress / Ingress    │  │
│  │ (media   │ │ (AI voice   │ │  (record, stream,    │  │
│  │  routing)│ │  pipeline)  │ │   ingest from RTMP)  │  │
│  └──────────┘ └─────────────┘ └──────────────────────┘  │
│  ┌──────────┐ ┌─────────────┐ ┌──────────────────────┐  │
│  │   SIP    │ │  Webhooks   │ │  Client SDKs         │  │
│  │ (phone   │ │  (events)   │ │  (JS, Swift, Kotlin, │  │
│  │  bridge) │ │             │ │   Flutter, Rust, ...) │  │
│  └──────────┘ └─────────────┘ └──────────────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
                           │
                    WebRTC / WHIP / SIP
                           │
┌──────────────────────────▼──────────────────────────────┐
│              Clients (browsers, mobile, IoT)              │
└─────────────────────────────────────────────────────────┘
```

## Core Concepts: Rooms, Participants, Tracks

LiveKit's data model has three fundamental abstractions:

### Room

A Room is a communication session — the container where participants meet. Each room has a unique name, lives on a single server node, and has a lifecycle (created when the first participant joins, destroyed when the last leaves or via API).

```
Room: "team-standup-2026-04-17"
├── Max participants: 50
├── Empty timeout: 300s
├── Metadata: {"org": "engineering", "recurring": true}
└── Created: 2026-04-17T09:00:00Z
```

### Participant

A Participant is a user (or agent) that has joined a room. Each participant is authenticated via a **JWT token** that encodes their identity and permissions (can publish video? can subscribe to others? admin privileges?).

```
Participant: "alice-uuid-1234"
├── Identity: "alice"
├── Name: "Alice Chen"
├── Permissions: {canPublish: true, canSubscribe: true}
├── Tracks published: [camera_video, microphone_audio]
└── Tracks subscribed: [bob_camera, bob_microphone, screen_share]
```

### Track

A Track is a single media stream — an audio track from a microphone, a video track from a camera, a screen share, or a data channel. Tracks are **published** by participants and **subscribed** to by other participants.

```
Track: "TR_camera_alice"
├── Kind: VIDEO
├── Source: CAMERA
├── Codec: VP8 (or H.264, VP9, AV1)
├── Simulcast layers:
│   ├── HIGH:   1280×720, 30fps, 2.5 Mbps
│   ├── MEDIUM: 640×360,  20fps, 500 Kbps
│   └── LOW:    320×180,  15fps, 150 Kbps
└── Subscribers: [bob, charlie, ai-agent-1]
```

**Selective subscription**: Clients don't have to receive every track. A participant can subscribe only to the tracks they need — e.g., subscribe to the active speaker's video but not the other 20 participants' cameras. This saves bandwidth dramatically in large rooms.

## SFU Architecture: How Media Flows

### What Is an SFU?

A **Selective Forwarding Unit (SFU)** is a server that receives media streams from each participant and selectively forwards them to other participants — **without transcoding**. This is in contrast to:

- **Peer-to-Peer (P2P)**: Each participant sends their stream directly to every other participant. Works for 2-3 people, collapses at scale ($N \times (N-1)$ streams).
- **MCU (Multipoint Control Unit)**: Server receives all streams, decodes them, composites into a single mixed stream, re-encodes, and sends to each participant. High server CPU cost, high latency from re-encoding.

```
P2P (N=4):                    MCU:                      SFU (LiveKit):
A ←→ B                        A → [decode]              A → [forward] → B
A ←→ C                        B → [decode]              A → [forward] → C
A ←→ D                        C → [decode] → [mix]      B → [forward] → A
B ←→ C                        D → [decode]    ↓         B → [forward] → C
B ←→ D                            [encode] → A,B,C,D    (no transcoding)
C ←→ D

Upload: (N-1) streams          Upload: 1 stream         Upload: 1 stream
Download: (N-1) streams         Download: 1 stream       Download: (N-1) streams
Total connections: N(N-1)/2     Server CPU: very high    Server CPU: low
Scales to: ~3 participants      Scales to: ~50+          Scales to: ~100-200+
```

**Why SFU wins**: It requires only 1 upload per participant (low client bandwidth), no transcoding (low server CPU), and supports per-subscriber quality selection via simulcast. The trade-off is that each subscriber downloads individual streams, but simulcast and adaptive bitrate keep this manageable.

### Simulcast and Dynacast

**Simulcast** is critical to SFU performance. Each publisher encodes their video at **multiple quality levels** simultaneously (e.g., 720p, 360p, 180p). The SFU selects which quality layer to forward to each subscriber based on:

- Subscriber's available bandwidth
- Size of the video element in the subscriber's UI (why send 720p for a thumbnail?)
- Network congestion detected via RTCP feedback

```
Publisher (Alice):
  Encodes 3 layers:
    HIGH:   720p @ 2.5 Mbps
    MEDIUM: 360p @ 500 Kbps
    LOW:    180p @ 150 Kbps

SFU (LiveKit):
  → Bob (on WiFi, large screen):  forwards HIGH layer
  → Charlie (on 4G, small phone): forwards LOW layer
  → Dana (gallery view):          forwards MEDIUM layer
```

**Dynacast** optimizes simulcast further: if no subscriber is currently receiving the HIGH layer (everyone is on mobile), the publisher **stops encoding** that layer entirely, saving CPU and bandwidth. When a subscriber requests HIGH again, encoding resumes.

### Adaptive Bitrate

LiveKit continuously monitors network conditions via:
- **RTCP receiver reports**: packet loss and jitter from subscribers
- **Transport-CC (Transport-Wide Congestion Control)**: per-packet delay measurements
- **Bandwidth estimation**: REMB/GCC algorithms

When congestion is detected, the SFU:
1. Switches subscribers to a lower simulcast layer
2. Signals publishers to reduce their encoding bitrate
3. Prioritizes audio over video (audio is always transmitted first)

## Distributed Architecture and Scaling

### Single-Node Capacity

A single LiveKit node can handle:
- **~100-200 participants** in a single room (limited by CPU for SRTP encryption and bandwidth)
- **Many concurrent rooms** (each room is independent)
- Approximately **10-50 Gbps of media throughput** on modern hardware

### Multi-Node Clustering

For larger deployments, LiveKit runs as a cluster with **Redis** as the coordination layer:

```
┌─────────────────────────────────────────────────────┐
│                    Redis Cluster                     │
│  (room registry, node health, message bus)           │
└────────┬──────────────┬──────────────┬──────────────┘
         │              │              │
    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
    │ Node 1  │   │ Node 2  │   │ Node 3  │
    │ (US-E)  │   │ (US-W)  │   │ (EU)    │
    │         │   │         │   │         │
    │ Room A  │   │ Room B  │   │ Room C  │
    │ Room D  │   │ Room E  │   │ Room F  │
    └─────────┘   └─────────┘   └─────────┘
```

**Room-to-node assignment**: When a new room is created, the receiving node selects a cluster member to host it based on:
1. System load (`sysload_limit` — nodes above the threshold are excluded)
2. Geographic proximity (latitude/longitude configuration for multi-region)
3. Current room count

**Critical constraint**: A single room must fit on a single node — rooms cannot span multiple servers. This means the maximum participants per room is limited by single-node capacity (~100-200). For larger events, use LiveKit's Egress to stream to platforms like YouTube Live.

**Multi-region**: Nodes report their geographic coordinates to Redis. When a room is created, the geo-aware load selector routes it to the nearest data center. DNS-level geo-routing (Route53, Cloudflare) directs clients to the nearest node cluster.

### Graceful Shutdown and Deployments

On SIGTERM/SIGINT, a node enters **drain mode**:
1. Existing rooms continue running
2. New participants can join existing rooms on this node
3. New rooms are NOT created on this node (routed to other nodes)
4. Shutdown completes only when all participants disconnect

This enables zero-downtime rolling deployments in Kubernetes.

## The Agents Framework: Building AI Voice Agents

This is LiveKit's fastest-growing feature and the primary reason many teams adopt it today. The Agents framework lets backend programs **join LiveKit rooms as participants** to build AI-powered voice and video agents.

### Architecture

```
┌─────────────┐     WebRTC      ┌──────────────┐     HTTP/WS     ┌──────────────┐
│   User       │ ←────────────→ │  LiveKit SFU  │ ←────────────→ │  Agent Server │
│  (browser/   │                │               │                │  (Python/     │
│   mobile)    │                │  Room: "call" │                │   Node.js)    │
└─────────────┘                └──────────────┘                └──────┬───────┘
                                                                      │
                                                               ┌──────▼───────┐
                                                               │  AI Pipeline  │
                                                               │  STT → LLM   │
                                                               │    → TTS      │
                                                               └──────────────┘
```

### The STT → LLM → TTS Pipeline

The standard voice agent pipeline:

1. **STT (Speech-to-Text)**: Transcribes the user's speech in real-time (streaming). Integrations: Deepgram, OpenAI Whisper, Google Cloud Speech, Azure, AssemblyAI.

2. **LLM (Large Language Model)**: Processes the transcribed text, generates a response (with streaming token output). Integrations: OpenAI, Anthropic, Google Gemini, Groq, Cerebras, any OpenAI-compatible API.

3. **TTS (Text-to-Speech)**: Converts the LLM's text response back to speech (streaming). Integrations: ElevenLabs, OpenAI TTS, Google Cloud TTS, Cartesia, PlayHT.

```python
# LiveKit Agent — minimal voice agent example
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # Wait for a human participant to join
    participant = await ctx.wait_for_participant()
    
    # Create the voice agent pipeline
    agent = VoicePipelineAgent(
        vad=silero.VAD.load(),            # Voice Activity Detection
        stt=deepgram.STT(),                # Speech-to-Text
        llm=openai.LLM(model="gpt-4o"),   # Language Model
        tts=openai.TTS(),                  # Text-to-Speech
    )
    
    # Start the agent — it joins the room as a participant
    agent.start(ctx.room, participant)
    
    # Initial greeting
    await agent.say("Hello! How can I help you today?")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### Key Agent Capabilities

**Turn detection**: LiveKit's custom turn-detection model determines when the user has finished speaking and it's the agent's turn to respond. This is more sophisticated than simple silence detection — it handles pauses mid-sentence, filler words ("um", "uh"), and interruptions.

**Interruption handling**: If the user speaks while the agent is talking, the agent stops its current speech and processes the new input. The TTS output is cancelled mid-stream.

**Tool use**: Agents can call external tools/APIs (check weather, query databases, make reservations) using the LLM's function calling capability:

```python
from livekit.agents import llm

class AssistantFunctions(llm.FunctionContext):
    @llm.ai_callable(description="Look up current weather for a city")
    async def get_weather(self, city: str) -> str:
        # Call weather API
        response = await weather_api.get(city)
        return f"The weather in {city} is {response.temp}°F and {response.condition}"

agent = VoicePipelineAgent(
    vad=silero.VAD.load(),
    stt=deepgram.STT(),
    llm=openai.LLM(model="gpt-4o"),
    tts=openai.TTS(),
    fnc_ctx=AssistantFunctions(),  # register tools
)
```

**Multi-agent handoff**: An agent can transfer the conversation to another agent (e.g., a general receptionist agent transfers to a billing specialist agent).

**Session management**: Agents can persist conversation state across sessions using external storage.

## Egress: Recording and Streaming Out

LiveKit Egress captures media from rooms and outputs it as recordings or live streams.

### Egress Types

| Type | What It Captures | Use Case |
|------|-----------------|----------|
| **RoomComposite** | Entire room via headless Chrome with a web layout | Meeting recordings, custom layouts |
| **Web** | Arbitrary web page content | Dashboard recordings, presentations |
| **Participant** | Single participant's audio + video | Testimonial recording, AI agent output |
| **TrackComposite** | Specific synced audio + video tracks | Podcast recording (selected guests) |
| **Track** | Individual track (no transcoding) | Raw media archival |
| **Auto Egress** | Automatically starts when room is created | Always-record policy |

### Output Formats

- **File**: MP4, OGG, WebM → uploaded to S3, GCS, Azure Blob, or local disk
- **Segmented**: HLS (`.m3u8` + `.ts` segments) for on-demand playback
- **Stream**: RTMP/RTMPS to YouTube Live, Twitch, Facebook Live, or custom RTMP servers

```python
# Start recording a room
from livekit import api

lk = api.LiveKitAPI()
egress = await lk.egress.start_room_composite_egress(
    room_name="team-standup",
    layout="speaker",  # or "grid", custom URL
    file_outputs=[api.EncodedFileOutput(
        file_type=api.EncodedFileType.MP4,
        filepath="recordings/standup-2026-04-17.mp4",
        s3=api.S3Upload(
            bucket="my-recordings",
            region="us-east-1",
        ),
    )],
)
```

## Ingress: Streaming In

LiveKit Ingress accepts media from external sources and publishes them as tracks in a room.

### Supported Protocols

| Protocol | Mode | Use Case |
|----------|------|----------|
| **RTMP/RTMPS** | Push (publisher sends to LiveKit) | OBS Studio, streaming software |
| **WHIP** | Push (WebRTC-based) | Ultra-low-latency ingress from browsers |
| **SRT** | Pull (LiveKit fetches from source) | Professional broadcast equipment |
| **HTTP (HLS, MP4)** | Pull (LiveKit fetches from URL) | Re-streaming existing content |

```python
# Create an RTMP ingress point
ingress = await lk.ingress.create_ingress(
    input_type=api.IngressInput.RTMP_INPUT,
    room_name="live-event",
    participant_identity="streamer",
    participant_name="Main Camera",
)
# Returns an RTMP URL: rtmp://ingest.livekit.cloud/x/stream-key
# Point OBS Studio at this URL to stream into the LiveKit room
```

## SIP Integration: Bridging Phone Calls

LiveKit bridges PSTN (phone network) calls into rooms via **SIP (Session Initiation Protocol)**. This is critical for AI voice agent use cases — users can call a phone number and talk to an AI agent.

```
Phone User                    LiveKit                    AI Agent
    │                            │                           │
    │  Dials +1-555-0100         │                           │
    │ ──────────────────────→    │                           │
    │                     SIP INVITE                         │
    │                   (trunk + dispatch)                    │
    │                            │    Room: "call-123"       │
    │                            │ ←─── Agent joins room     │
    │                            │                           │
    │ ← RTP audio (via SIP) ──→ │ ← WebRTC audio ────────→ │
    │                            │                           │
    │  "I'd like to check        │  STT → "check balance"   │
    │   my balance"              │  LLM → "Your balance..." │
    │                            │  TTS → audio response     │
    │ ← "Your balance is $500"  │ ←──────────────────────── │
```

### Components

- **SIP Trunk**: Configuration linking a third-party SIP provider (Twilio, Vonage, Telnyx) to LiveKit
- **Dispatch Rules**: Route inbound calls to rooms based on caller ID, dialed number, or custom rules
- **SIP Participant**: Phone callers appear as standard LiveKit participants — agents interact with them via the same APIs as WebRTC participants

```python
# Create a SIP trunk for inbound calls
trunk = await lk.sip.create_sip_inbound_trunk(
    name="Customer Support Line",
    numbers=["+15550100"],
    allowed_addresses=["sip.twilio.com"],
)

# Create a dispatch rule: route all calls to a room and trigger an agent
rule = await lk.sip.create_sip_dispatch_rule(
    trunk_ids=[trunk.sip_trunk_id],
    rule=api.SIPDispatchRuleIndividual(
        room_prefix="call-",     # each call gets its own room
        pin="",                   # no PIN required
    ),
)
```

## Authentication and Security

### JWT-Based Authentication

Every participant receives a **JWT (JSON Web Token)** generated by your server, containing:

```python
from livekit import api

token = api.AccessToken(api_key="your-api-key", api_secret="your-api-secret")
token.with_identity("alice")
token.with_name("Alice Chen")
token.with_grants(api.VideoGrants(
    room_join=True,
    room="team-standup",
    can_publish=True,
    can_subscribe=True,
    can_publish_data=True,
))
jwt_string = token.to_jwt()  # send this to the client
```

The client connects to LiveKit with this JWT. The server validates it before allowing the participant to join. JWTs have expiration times and can be scoped to specific rooms with specific permissions.

### End-to-End Encryption (E2EE)

LiveKit supports **E2EE** where media is encrypted client-to-client. The SFU forwards encrypted packets without being able to decrypt them — even LiveKit's servers can't access the media content. This is critical for healthcare (HIPAA), finance, and privacy-sensitive applications.

**Trade-off**: E2EE disables server-side features that require media access — recording (egress), speech-to-text, and server-side composition.

## Client SDKs

| Platform | SDK | Features |
|----------|-----|----------|
| Web (JavaScript/TypeScript) | `livekit-client` + React components | Full featured, most mature |
| iOS/macOS | Swift SDK + SwiftUI components | Native performance, camera/mic management |
| Android | Kotlin SDK + Compose components | Native performance, background operation |
| Flutter | `livekit_client` | Cross-platform (iOS, Android, web, desktop) |
| React Native | `@livekit/react-native` | Cross-platform mobile |
| Rust | `livekit-rust-sdk` | High-performance, embedded, server-side |
| Unity | WebGL support | Gaming/XR applications |
| C++ | Native SDK | Embedded systems, custom platforms |
| ESP32 | Embedded SDK | IoT devices, hardware agents |

## LiveKit Cloud vs Self-Hosted

| Aspect | Self-Hosted | LiveKit Cloud |
|--------|-------------|---------------|
| Cost | Infrastructure costs only | Per-minute pricing |
| Scaling | Manual (add nodes, configure Redis) | Automatic |
| Multi-region | Manual (DNS routing, node geo-config) | Automatic |
| SIP/Telephony | Self-managed SIP provider | Managed phone numbers |
| Monitoring | DIY (Prometheus, Grafana) | Built-in observability dashboard |
| Compliance | Your responsibility | HIPAA-eligible (Enterprise) |
| Egress | Requires GStreamer and headless Chrome | Managed |
| Maintenance | Upgrades, security patches | Managed |

**When to self-host**: Strict data sovereignty requirements, cost optimization at very high scale (millions of minutes/month), or need for deep infrastructure customization.

**When to use Cloud**: Most teams — especially when starting. The operational overhead of running WebRTC infrastructure reliably (TURN servers, STUN, codec updates, network edge optimization) is significant.

## Case Studies

### Case Study 1: AI Customer Support Agent (Telecom)

**Problem**: A telecom company handled 50,000 customer calls/day. Wait times averaged 12 minutes. Human agent costs: $15/hour per agent, 500 agents.

**Solution**: Deployed LiveKit-based AI voice agents to handle tier-1 support (billing inquiries, plan changes, troubleshooting). LiveKit SIP integration connected to their existing Twilio trunk. Agents used GPT-4o for conversation and Deepgram for STT.

**Architecture**:
```
Customer phone → Twilio SIP → LiveKit SIP Bridge → Room
                                                    ↕
                                              AI Agent (LiveKit Agents)
                                              STT (Deepgram) → LLM (GPT-4o)
                                              → TTS (ElevenLabs)
                                                    ↕
                                              Internal APIs (billing, CRM)
```

**Results**: AI agents handled 60% of calls without human escalation. Average handle time dropped from 8 minutes to 3 minutes. Wait times dropped to <30 seconds. Cost per call: $0.15 (AI) vs $2.00 (human).

### Case Study 2: Telehealth Platform

**Problem**: A telehealth startup needed HIPAA-compliant video calls between doctors and patients, with recording for medical records and phone dial-in for patients without internet.

**Solution**: LiveKit Cloud (Enterprise, HIPAA-eligible) with:
- WebRTC for browser/mobile video calls
- SIP integration for phone dial-in
- Egress for encrypted recording to S3
- E2EE for patient privacy
- AI agent for pre-consultation triage (asks symptoms before doctor joins)

**Key decisions**:
- E2EE disabled for recorded sessions (can't record E2EE streams server-side) — instead used encrypted S3 storage + strict access controls
- SIP dial-in for elderly patients without smartphones
- AI triage agent reduces doctor face-time by 30% by collecting symptoms beforehand

### Case Study 3: Live Coding Interview Platform

**Problem**: A hiring platform needed real-time video interviews with screen sharing, collaborative coding, and recording for review.

**Solution**: LiveKit with RoomComposite egress:
- Interviewer + candidate join a room with video, audio, and screen sharing
- Collaborative code editor synced via LiveKit data channels (arbitrary data pub/sub)
- RoomComposite egress records the entire session with a custom web layout showing video, code editor, and chat side-by-side
- Recordings uploaded to S3 for hiring committee review

**Architecture insight**: Data channels are key — LiveKit supports arbitrary data pub/sub alongside audio/video, perfect for syncing application state (cursor position, code changes, whiteboard strokes) in real-time.

### Case Study 4: Connection Scaling Issues

**Problem**: A gaming company used LiveKit for voice chat in 500-player game lobbies. Single rooms with 500 participants exceeded node capacity.

**Solution**: Architectural redesign:
1. Split the 500-player lobby into "voice zones" of 25 players each (based on in-game proximity)
2. Each zone is a separate LiveKit room
3. Players automatically switch rooms as they move between zones
4. A coordination service manages room assignment and cross-zone audio (can hear adjacent zones at reduced volume)

**Lesson**: LiveKit rooms are limited to single-node capacity (~100-200 participants). For very large groups, design your application to partition participants into smaller rooms, with application-level logic for cross-room communication.

## Interview Questions and Answers

### Q: What is an SFU and how does it compare to P2P and MCU?

**P2P (Peer-to-Peer)**: Each participant sends their media directly to every other participant. Upload bandwidth scales as $O(N-1)$, and total connections as $O(N^2)$. Works for 2-3 people, unusable beyond that.

**MCU (Multipoint Control Unit)**: Server receives all streams, decodes them, mixes into a single composite stream, re-encodes, and sends to each participant. Low client bandwidth (1 upload + 1 download) but extremely high server CPU from transcoding. Adds latency (decode → mix → encode).

**SFU (Selective Forwarding Unit)**: Server receives each participant's stream and forwards it to other participants without transcoding. 1 upload per participant, $N-1$ downloads, but the server can select which quality layer (simulcast) to forward based on each subscriber's bandwidth. Low server CPU (no transcoding), low latency (no re-encoding), and with simulcast the download bandwidth is manageable.

LiveKit is an SFU. The SFU architecture dominates modern video conferencing (Zoom, Google Meet, Teams all use SFU or SFU-like architectures) because it provides the best balance of quality, latency, and scalability.

### Q: Explain simulcast and dynacast. Why are they important?

**Simulcast**: The publisher encodes their video at multiple quality levels simultaneously (e.g., 720p, 360p, 180p). The SFU selects which layer to forward to each subscriber based on their bandwidth, display size, and network conditions. Without simulcast, all subscribers receive the same quality — if one subscriber has poor bandwidth, either they get a degraded experience or the publisher must lower quality for everyone.

**Dynacast**: An optimization on top of simulcast. If no subscriber is currently receiving a particular quality layer (e.g., no one needs the 720p layer), the publisher **stops encoding** that layer entirely — saving CPU and bandwidth. When a subscriber requests that layer again, encoding resumes. This is significant because encoding multiple simulcast layers is expensive (3x CPU on the publisher).

Together, they enable a room where one participant on a desktop with fast internet sees 720p video while another on a phone with 4G sees 180p — both from the same publisher, with the publisher only encoding the layers actually being consumed.

### Q: How does LiveKit's distributed architecture work? What are the limitations?

LiveKit runs as a cluster of SFU nodes coordinated via **Redis**. Each node periodically reports its health (CPU, bandwidth, room count) to Redis. When a new room is created, a node is selected based on system load and geographic proximity (configurable lat/long per node).

**Key limitation**: A single room must fit on a single node — rooms cannot span multiple servers. This caps room size at ~100-200 participants (single-node capacity). The cluster can handle thousands of concurrent rooms, but each room is hosted entirely on one node.

**For larger events**: Use LiveKit Egress to stream to YouTube Live/Twitch (which handle millions of viewers), or partition participants into smaller rooms with application-level coordination (e.g., voice zones in a game).

**Multi-region**: Nodes in different regions report their coordinates. The room assignment algorithm prefers the geographically nearest node with available capacity. DNS-level geo-routing directs clients to the nearest cluster entry point.

### Q: Describe the LiveKit Agents pipeline for AI voice agents.

The Agents framework lets backend processes join LiveKit rooms as participants. The standard voice agent pipeline chains three components:

1. **VAD (Voice Activity Detection)**: Detects when the user is speaking (Silero VAD). Determines when the user has finished their turn.
2. **STT (Speech-to-Text)**: Transcribes the user's speech in real-time with streaming (Deepgram, Whisper, Google).
3. **LLM (Large Language Model)**: Processes the transcript, generates a response with streaming tokens (GPT-4o, Claude, Gemini).
4. **TTS (Text-to-Speech)**: Converts the LLM's text response to speech in real-time with streaming (ElevenLabs, OpenAI TTS).

The agent receives the user's audio track via WebRTC, runs it through the pipeline, and publishes the response audio track back to the room. The user hears the AI response with end-to-end latency typically 500ms-1.5s depending on model choices.

Key features: turn detection (knows when the user has finished speaking vs pausing), interruption handling (stops speaking if the user interrupts), tool use (agents can call APIs during the conversation), and multi-agent handoff.

### Q: How does LiveKit handle authentication and security?

**Authentication**: JWT-based. Your backend generates a JWT containing the participant's identity, the room they can join, and their permissions (can publish, can subscribe, admin). The JWT is signed with your API secret. LiveKit validates the JWT before allowing the participant to join. JWTs have configurable expiration times.

**Permissions**: Per-participant control over publishing (audio, video, data), subscribing, and admin actions (mute others, remove participants). Permissions are encoded in the JWT and can be updated server-side at runtime.

**Encryption in transit**: All WebRTC media is encrypted via SRTP (Secure Real-time Transport Protocol) and DTLS (Datagram Transport Layer Security). Signaling is over WSS (WebSocket Secure).

**End-to-End Encryption (E2EE)**: Optional. Media is encrypted at the sender and decrypted at the receiver — the SFU forwards encrypted packets it cannot read. Protects against server-side eavesdropping. Trade-off: disables server-side features requiring media access (recording, STT, composition).

### Q: What are LiveKit's egress capabilities? How does recording work?

Egress captures media from rooms as recordings or live streams. Five modes: **RoomComposite** (records entire room via headless Chrome rendering a web layout — most flexible), **Participant** (records a single participant), **TrackComposite** (records specific synced tracks), **Track** (raw individual track without transcoding), **Web** (records arbitrary web content).

Outputs: **Files** (MP4, WebM, OGG) uploaded to S3/GCS/Azure, **segmented HLS** for on-demand playback, **RTMP streams** to YouTube Live/Twitch/Facebook.

RoomComposite uses a headless Chrome instance that loads a configurable web page showing the room's participants. This means you can design any recording layout (speaker view, grid, custom branded templates) as a web page. GStreamer handles the actual encoding.

**Auto Egress** can be configured to automatically start recording when any room is created — useful for compliance (record all meetings) or AI agent conversations (record all calls for quality assurance).

### Q: How does SIP integration work for phone calls to AI agents?

LiveKit bridges PSTN phone calls via SIP. Three components:

1. **SIP Trunk**: Links a third-party SIP provider (Twilio, Telnyx, Vonage) to LiveKit. Inbound trunks receive calls; outbound trunks make calls.

2. **Dispatch Rules**: Determine what happens when a call arrives — route to a specific room, create a new room per call, or route based on the dialed number.

3. **SIP Participant**: The phone caller appears as a standard LiveKit participant in the room. Agents interact with them via the same APIs as WebRTC participants — no special handling needed.

Flow: Phone call → SIP provider → LiveKit SIP bridge → creates room → dispatches AI agent → agent joins room → conversation over WebRTC (agent side) / RTP (phone side).

This enables AI voice agents to answer phone calls — the same agent code works for both WebRTC (browser) and PSTN (phone) callers.

### Q: Design a system for a telehealth platform using LiveKit.

**Requirements**: HIPAA-compliant video calls, recording, phone dial-in, AI triage.

**Architecture**:

```
Patient (browser/phone) → LiveKit Cloud (Enterprise, HIPAA)
                              ├── Video/Audio (WebRTC)
                              ├── SIP dial-in (phone patients)
                              ├── Recording (Egress → encrypted S3)
                              └── AI Triage Agent (pre-consultation)
                                    ├── STT (medical vocabulary)
                                    ├── LLM (symptom collection)
                                    └── TTS (patient communication)
```

**Key decisions**:
1. **LiveKit Cloud Enterprise** for HIPAA eligibility, managed SIP, and SLA
2. **No E2EE** for recorded sessions (can't record E2EE server-side). Instead: SRTP encryption in transit + encrypted S3 at rest + strict IAM policies
3. **SIP integration** with Twilio for elderly patients who only have phones
4. **AI triage agent** joins the room before the doctor. Collects symptoms, medical history, and chief complaint. Summarizes for the doctor, reducing face-time by 30%
5. **Auto Egress** records all consultations to encrypted S3 for medical records
6. **JWT permissions**: patients can publish audio/video but cannot record. Doctors have admin permissions
7. **Webhooks** notify the scheduling system when consultations start/end

### Q: What are the main challenges when building on LiveKit and how do you address them?

**1. Room size limits**: Single room caps at ~100-200 participants (single-node constraint). Solution: partition large groups into smaller rooms with application-level coordination.

**2. AI agent latency**: The STT→LLM→TTS pipeline adds 500ms-1.5s end-to-end latency. Solutions: use streaming at every stage (don't wait for full transcription), choose low-latency providers (Deepgram for STT, Groq/Cerebras for fast LLM inference, Cartesia for streaming TTS), use LiveKit's turn detection model instead of simple silence detection.

**3. Network quality variation**: Users on poor networks experience degraded quality. Solutions: simulcast + dynacast (automatic quality adaptation), TURN servers for users behind strict NATs/firewalls, TCP fallback when UDP is blocked.

**4. Recording quality**: RoomComposite egress uses headless Chrome, which can be resource-intensive and sometimes produces layout glitches. Solutions: test your recording layout thoroughly, use Track/Participant egress for simpler recordings, provision sufficient CPU for egress nodes.

**5. Cost at scale**: LiveKit Cloud pricing is per-minute. AI agent minutes + STT + LLM + TTS costs compound quickly. Solutions: use cheaper models for simple tasks (Whisper for STT, smaller LLMs for FAQ), implement call routing to human agents for complex cases, monitor per-call costs.

## References

1. [LiveKit Documentation](https://docs.livekit.io/)
2. [LiveKit GitHub Repository](https://github.com/livekit/livekit)
3. [LiveKit Agents Framework](https://docs.livekit.io/agents/)
4. [LiveKit Blog — Pipeline vs Realtime Voice Agent Architecture](https://livekit.io/blog)
5. [WebRTC Standard — W3C](https://www.w3.org/TR/webrtc/)
6. [Pion WebRTC (Go implementation)](https://github.com/pion/webrtc)
7. [WHIP — WebRTC-HTTP Ingestion Protocol (RFC Draft)](https://datatracker.ietf.org/doc/draft-ietf-wish-whip/)
8. [SIP — Session Initiation Protocol (RFC 3261)](https://datatracker.ietf.org/doc/html/rfc3261)
