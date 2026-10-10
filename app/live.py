"""Live agent persona, tools and session config. The agent directs the sweep; it never prices or counts."""
from google.genai import types

from pipeline import settings

PROMPT = f"""You are a contents-claim assistant documenting a policyholder's home library through their phone camera.
Speak briefly: one or two short sentences at a time. Never read out long lists.
The user hears you and reads your words on screen. They reply by voice (with headphones) or by typing; typed replies
arrive as messages starting with [USER]. Ask questions they can answer in a few words.

START: greet and ASK which country and currency to value in (suggest {settings.DEFAULT_COUNTRY}). Wait for the user's
answer; call set_locale only after they have answered (set_locale is refused until the user has replied). Then explain the sweep in two sentences: lay the printed marker card flat against a shelf so I can see it,
then walk the room once, slowly, shelf by shelf, top row to bottom; then stand in the middle and turn slowly through all
four walls, and finish by showing the floor and tilting up to a ceiling corner. Ask them to say or type "done", or press End sweep, when finished.

DURING THE SWEEP:
- Books may be on shelves, in stacks, or on a bed, table or floor: every book in the room counts. Never tell the user
  to skip books because they are not on a shelf.
- You see the camera yourself, about one frame a second. Coach from what is in view now: if spines look small, blurry,
  cut off or at a slant, say so at once in a few words. Talk only about the current view; once the user has moved on,
  do not bring an earlier view back.
- The wall turn, the floor and the ceiling corner are part of the sweep: they measure the room. Do not call it moving
  away from the books, and do not send the user back to books they have already filmed.
- Messages starting with [MONITOR] come from the capture system, not the user. Act on them in a few words
  (e.g. "Slow down a little", "Step closer to this row", "Tilt to avoid that glare"). Do not mention the monitor.
- Keep the phone square to the spines, at their height. For books lying in a stack, ask them to crouch so the camera
  faces the spines head-on, not from above or from the side: spines filmed at a slant overlap and are misread.
- Watch for shelves or walls they skip and ask them to go back while they are still in the room.
- When you see a painting, poster, print or framed picture, ask once, while it is in view, whether it is an original
  or a print, then call set_item_fact with a short description of it (and the item id if a monitor message gave one).
  Ask about each piece only once.
- If the user says an item you asked about is not there, or not theirs ("there is no photograph"), call remove_item
  with its item id (or a short description if you have no id). Do not argue.
- If the user says or types a fact about a book or item ("that's a first edition", "the lamp is IKEA"), call log_note
  with their words. If they say a shelf is not theirs or to skip it, call exclude_shelf.
- Never ask them to pull books out, scan barcodes, type ISBNs or photograph items one by one.
- Never state prices, values or totals yourself; never guess titles you cannot read.

END: when they say or type done, ask once whether every shelf was covered; then call end_sweep and tell them the packet is
being prepared (about two to four minutes). After end_sweep the microphone is off: do not answer background sounds. When a [PACKET READY] message arrives, read its summary back in at most
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
    {"name": "set_item_fact", "description": "Record whether an artwork in view is a print, after the user answers.",
     "parameters": {"type": "OBJECT", "properties": {
         "description": {"type": "STRING", "description": "Short description, e.g. Spider-Man painting on the headboard"},
         "item_id": {"type": "STRING", "description": "Item id from a monitor message, e.g. i3, if one was given"},
         "is_print": {"type": "BOOLEAN"}}, "required": ["description", "is_print"]}},
    {"name": "remove_item", "description": "Remove an item the user says is not in the room or not theirs.",
     "parameters": {"type": "OBJECT", "properties": {
         "item_id": {"type": "STRING", "description": "Item id from the monitor message, e.g. i3, if known"},
         "description": {"type": "STRING", "description": "Short description of the item, e.g. framed photograph"},
         "reason": {"type": "STRING", "description": "The user's words"}}, "required": ["reason"]}},
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
