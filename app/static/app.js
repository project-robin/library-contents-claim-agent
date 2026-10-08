// Phone client: mic 16 kHz PCM16 + camera JPEG ~3 fps up; agent audio 24 kHz PCM16 + JSON events down.
// Binary frames carry a 1-byte tag: 1 = audio, 2 = jpeg.
const $ = (s) => document.querySelector(s);
const FPS = 3;
let ws, micCtx, outCtx, playHead = 0, playing = [], camTimer, lastRole = null, lastLine = null;

const MIC_WORKLET = `class Mic extends AudioWorkletProcessor {
  constructor() { super(); this.buf = []; }
  process(inputs) {
    const ch = inputs[0][0];
    if (ch) { for (const v of ch) this.buf.push(v);
      if (this.buf.length >= 1600) { this.port.postMessage(new Float32Array(this.buf)); this.buf = []; } }
    return true;
  }
}
registerProcessor('mic', Mic);`;

function tagged(tag, payload) {
  const out = new Uint8Array(payload.byteLength + 1);
  out[0] = tag;
  out.set(new Uint8Array(payload), 1);
  return out;
}

async function startMic(stream) {
  micCtx = new AudioContext({ sampleRate: 16000 });
  await micCtx.audioWorklet.addModule(URL.createObjectURL(new Blob([MIC_WORKLET], { type: 'text/javascript' })));
  const node = new AudioWorkletNode(micCtx, 'mic');
  node.port.onmessage = (e) => {
    // ponytail: half-duplex. No echo cancellation (it forces Android's quiet call mode), so the mic is not sent
    // while the agent speaks or it hears itself; the user can speak once it stops (replies are 1-2 sentences).
    if (ws.readyState !== 1 || outCtx.currentTime < playHead + 0.3) return;
    const f = e.data, pcm = new Int16Array(f.length);
    for (let i = 0; i < f.length; i++) pcm[i] = Math.max(-1, Math.min(1, f[i])) * 0x7fff;
    ws.send(tagged(1, pcm.buffer));
  };
  micCtx.createMediaStreamSource(stream).connect(node);
}

function playPcm(buf) {
  const pcm = new Int16Array(buf), f = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) f[i] = pcm[i] / 0x8000;
  const ab = outCtx.createBuffer(1, f.length, 24000);
  ab.copyToChannel(f, 0);
  const src = outCtx.createBufferSource();
  src.buffer = ab;
  src.connect(outCtx.destination);
  playHead = Math.max(playHead, outCtx.currentTime);
  src.start(playHead);
  playHead += ab.duration;
  playing.push(src);
  src.onended = () => (playing = playing.filter((s) => s !== src));
}

function stopPlayback() {  // user barged in
  playing.forEach((s) => { try { s.stop(); } catch {} });
  playing = [];
  playHead = 0;
}

function startCamera() {
  const video = $('#video'), canvas = document.createElement('canvas');
  camTimer = setInterval(() => {
    if (video.readyState < 2 || ws.readyState !== 1 || ws.bufferedAmount > 3e6) return;  // drop frames on slow links
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob(async (b) => b && ws.readyState === 1 && ws.send(tagged(2, await b.arrayBuffer())), 'image/jpeg', 0.85);
  }, 1000 / FPS);
}

function say(role, text) {
  if (role === lastRole && lastLine && role !== 'mon') { lastLine.textContent += text; }
  else {
    lastLine = document.createElement('div');
    lastLine.className = role;
    lastLine.textContent = (role === 'mon' ? '⚠ ' : '') + text;
    $('#talk').appendChild(lastLine);
  }
  lastRole = role;
  $('#talk').parentElement.scrollTop = 1e9;
}

function onMessage(e) {
  if (typeof e.data !== 'string') {
    const b = new Uint8Array(e.data);
    if (b[0] === 1) playPcm(b.slice(1).buffer);
    return;
  }
  const m = JSON.parse(e.data);
  if (m.type === 'transcript') say(m.role, m.text);
  else if (m.type === 'interrupted') stopPlayback();
  else if (m.type === 'monitor') say('mon', m.text);
  else if (m.type === 'quality') {
    $('#quality').textContent = m.issue ? m.issue : 'camera ok';
    $('#quality').className = m.issue ? 'bad' : '';
  } else if (m.type === 'inventory') {
    $('#counts').textContent = `${m.books.length} titles read · ${m.items.length} items · ${m.unreadable_now} unreadable in view`;
    $('#inv').innerHTML = '';
    for (const it of m.items) $('#inv').insertAdjacentHTML('beforeend', `<div>▣ ${it.id} ${esc(it.category)}: ${esc(it.description)}</div>`);
    for (const b of m.books.slice().reverse()) $('#inv').insertAdjacentHTML('beforeend', `<div>📕 ${esc(b.title)} <small>${esc(b.author)}</small></div>`);
  } else if (m.type === 'status' || m.type === 'sweep') $('#status').textContent = m.text || `Sweep ${m.id}`;
  else if (m.type === 'tool' && m.name === 'end_sweep') endLocal();
  else if (m.type === 'tool') say('mon', `logged: ${m.name} ${JSON.stringify(m.args)}`);
  else if (m.type === 'packet') {
    $('#status').textContent = m.summary;
    $('#packet').innerHTML = `<a href="${m.report}" target="_blank">Open claim report</a><a href="${m.json}" target="_blank">claim_packet.json</a>`;
  }
}

const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function endLocal() {  // stop sending video; keep audio so the agent can read the summary back
  clearInterval(camTimer);
  $('#end').disabled = true;
}

$('#start').onclick = async () => {
  $('#start').disabled = true;
  outCtx = new AudioContext({ sampleRate: 24000 });
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },  // loudspeaker
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } },
  });
  $('#video').srcObject = stream;
  try { await navigator.wakeLock?.request('screen'); } catch {}
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = onMessage;
  ws.onclose = () => { $('#status').textContent = 'Disconnected.'; endLocal(); };
  ws.onopen = async () => {
    ws.send(JSON.stringify({ type: 'start', device: navigator.userAgent }));
    await startMic(stream);
    startCamera();
    $('#end').disabled = false;
  };
};

$('#end').onclick = () => { ws.send(JSON.stringify({ type: 'end' })); endLocal(); };
