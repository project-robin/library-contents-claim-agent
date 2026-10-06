# Reference app: what was kept, changed and thrown away

Reference: [Insurance Claim Live Agent Team](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/voice_ai_agents/insurance_claim_live_agent_team)

> TODO (Phase B2): run the reference locally and add observations from the run (latency, how often it
> described rather than measured, what broke). Notes below are from reading its code.

## Kept (pattern, code rewritten)
- **Backend relay.** The browser talks to a FastAPI websocket, which holds the Gemini Live session
  (`client.aio.live.connect`, `send_realtime_input` for PCM16 16 kHz audio and JPEG frames, `send_tool_response`).
- **A claim file that builds itself from tool calls during the conversation.**
- **A downloadable packet at the end.**

## Changed
- **Vision.** The reference describes damage in a frame. Here every keyframe goes to a schema-constrained detector
  that returns spine boxes and verbatim text. Counting, measuring and pricing happen in code, in separate stages.
- **The server keeps evidence.** Every full-resolution frame is saved and referenced by `frame_ref`. Only a
  downscaled 1 fps copy goes to the Live model.
- **The agent directs the capture.** Deterministic blur, glare, motion and unreadable-spine checks feed
  `[MONITOR]` messages into the Live session, so the agent reacts while the user is still at the shelf.
- **Session robustness.** Context-window compression and session resumption are on, because audio+video sessions
  end at about 2 minutes without them.
- **Background agents.** The reference has policy and intake rules. Here they are identification, measurement
  (GPU reconstruction + marker scale), pricing (sourced, never LLM), review flagging and an independent checker.

## Thrown away
- Avatar video, incident sketch generation, the mock policy directory and policy lookup (none are needed here).
- Narrative notebook output, replaced by an itemised, priced inventory with sources and confidence.
