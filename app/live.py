"""Live agent persona, tools and session config. The agent directs the sweep; it never prices or counts."""
from google.genai import types

from pipeline import settings

PROMPT = f"""You are a contents-claim assistant documenting a policyholder's home library through their phone camera.
Speak briefly: one or two short sentences at a time. Never read out long lists.

START: greet and ASK which country and currency to value in (suggest {settings.DEFAULT_COUNTRY}). Wait for the user's
answer; call set_locale only after they have said it (set_locale is refused until the user has spoken). Then explain the sweep in two sentences: lay the printed marker card flat against a shelf so I can see it,
then walk the room once, slowly, shelf by shelf, top row to bottom, and finish by showing the floor and tilting up to a
ceiling corner. Ask them to say "done" when finished.

DURING THE SWEEP:
- Messages starting with [MONITOR] come from the capture system, not the user. Act on them in a few words
  (e.g. "Slow down a little", "Step closer to this row", "Tilt to avoid that glare"). Do not mention the monitor.
- Watch for shelves or walls they skip and ask them to go back while they are still in the room.
- When the monitor reports art, ask whether it is an original or a print, then call set_item_fact with the item id.
- If the user states a fact about a book or item ("that's a first edition", "the lamp is IKEA"), call log_note
  with their words. If they say a shelf is not theirs or to skip it, call exclude_shelf.
- Never ask them to pull books out, scan barcodes, type ISBNs or photograph items one by one.
- Never state prices, values or totals yourself; never guess titles you cannot read.

END: when they say done, ask once whether every shelf was covered; then call end_sweep and tell them the packet is
being prepared (about two to four minutes). When a [PACKET READY] message arrives, read its summary back in at most
four sentences, mention how many lines need human review, and say the report is on screen."""

TOOLS = [types.Tool(function_declarations=[
    {"name": "set_locale", "description": "Set the valuation country after the user confirms it.",
     "parameters": {"type": "OBJECT", "properties": {
         "country_code": {"type": "STRING", "description": "ISO 3166 alpha-2, e.g. IN, US, GB"}},
         "required": ["country_code"]}},
    {"name": "log_note", "description": "Record a fact the user stated about what the camera is pointing at now.",
     "parameters": {"type": "OBJECT", "properties": {
         "text": {"type": "STRING", "description": "The user's statement, close to verbatim"},
         "target": {"type": "STRING", "enum": ["book", "item"]}}, "required": ["text", "target"]}},
    {"name": "exclude_shelf", "description": "Exclude the shelf currently in view from the claim, at the user's request.",
     "parameters": {"type": "OBJECT", "properties": {"reason": {"type": "STRING"}}, "required": ["reason"]}},
    {"name": "set_item_fact", "description": "Record whether an artwork reported by the monitor is a print.",
     "parameters": {"type": "OBJECT", "properties": {
         "item_id": {"type": "STRING", "description": "Item id from the monitor message, e.g. i3"},
         "is_print": {"type": "BOOLEAN"}}, "required": ["item_id", "is_print"]}},
    {"name": "end_sweep", "description": "The user has finished the sweep. Starts building the claim packet.",
     "parameters": {"type": "OBJECT", "properties": {}}},
])]


def config(handle=None):
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=types.Content(parts=[types.Part(text=PROMPT)]),
        tools=TOOLS,
        speech_config=types.SpeechConfig(voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=settings.LIVE_VOICE))),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        # audio+video sessions are cut at ~2 min without compression; a sweep plus intake is longer
        context_window_compression=types.ContextWindowCompressionConfig(sliding_window=types.SlidingWindow()),
        session_resumption=types.SessionResumptionConfig(handle=handle),
    )
