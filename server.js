import express from 'express'
import { createServer } from 'http'
import { WebSocketServer } from 'ws'
import twilio from 'twilio'
import Anthropic from '@anthropic-ai/sdk'
import { createClient } from '@supabase/supabase-js'
import fetch from 'node-fetch'
import audioop from 'audioop'

const app = express()
app.use(express.json())
app.use(express.urlencoded({ extended: false }))

const httpServer = createServer(app)
const wss = new WebSocketServer({ server: httpServer })

const twilioClient = twilio(process.env.TWILIO_ACCOUNT_SID, process.env.TWILIO_AUTH_TOKEN)
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY })
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_KEY)

// In-memory call state (keyed by Twilio StreamSid)
const activeCalls = new Map()

// ─── STEP 1: Twilio dials out, hits this endpoint for TwiML ───────────────
app.post('/twiml', (req, res) => {
  const { contactId, agentId, campaignStep, callDbId } = req.query
  const twiml = `<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="wss://${req.headers.host}/media-stream?contactId=${contactId}&agentId=${agentId}&campaignStep=${campaignStep || 0}&callDbId=${callDbId}" />
  </Connect>
</Response>`
  res.type('text/xml').send(twiml)
})

// ─── STEP 2: WebSocket handles live audio ─────────────────────────────────
wss.on('connection', async (ws, req) => {
  const url = new URL(req.url, `http://localhost`)
  const contactId = url.searchParams.get('contactId')
  const agentId = url.searchParams.get('agentId')
  const campaignStep = parseInt(url.searchParams.get('campaignStep') || '0')
  const callDbId = url.searchParams.get('callDbId')

  // Load contact and agent from Supabase
  const [{ data: contact }, { data: agent }] = await Promise.all([
    supabase.from('contacts').select('*').eq('id', contactId).single(),
    supabase.from('agents').select('*').eq('id', agentId).single()
  ])

  if (!contact || !agent) { ws.close(); return }

  // Build initial conversation history
  const systemPrompt = buildSystemPrompt(agent, contact)
  const conversationHistory = []
  let streamSid = null
  let audioBuffer = Buffer.alloc(0)
  let silenceTimer = null
  let isAgentSpeaking = false
  let currentTtsAbort = null

  // Open Deepgram STT stream
  const deepgramWs = await openDeepgramStream()

  // Store call state
  activeCalls.set(streamSid, {
    ws, deepgramWs, conversationHistory, contact, agent,
    systemPrompt, callDbId, campaignStep, streamSid: null
  })

  // ── Greet the contact immediately when call connects ──
  let greeted = false

  ws.on('message', async (data) => {
    const msg = JSON.parse(data)

    if (msg.event === 'start') {
      streamSid = msg.start.streamSid
      const callState = activeCalls.get(null) || {}
      callState.streamSid = streamSid
      activeCalls.set(streamSid, { ...callState, streamSid })
      activeCalls.delete(null)

      // Update DB with streamSid
      await supabase.from('calls').update({ status: 'in-progress' }).eq('id', callDbId)

      // Send greeting after 800ms (gives Twilio time to settle)
      if (!greeted) {
        greeted = true
        setTimeout(async () => {
          const greeting = buildGreeting(agent, contact)
          await speakText(greeting, ws, streamSid, agent.voice_id)
          conversationHistory.push({ role: 'assistant', content: greeting })
        }, 800)
      }
    }

    if (msg.event === 'media') {
      const chunk = Buffer.from(msg.media.payload, 'base64')
      // Forward raw mulaw to Deepgram
      if (deepgramWs.readyState === 1) deepgramWs.send(chunk)
    }

    if (msg.event === 'stop') {
      cleanup(streamSid)
    }
  })

  ws.on('close', () => cleanup(streamSid))

  // ── Deepgram transcript handler ──
  deepgramWs.on('message', async (raw) => {
    const result = JSON.parse(raw)
    const alt = result?.channel?.alternatives?.[0]
    if (!alt || !result.is_final) return

    const transcript = alt.transcript?.trim()
    if (!transcript || transcript.length < 2) return

    // If agent is speaking, interrupt it
    if (isAgentSpeaking) {
      isAgentSpeaking = false
      if (currentTtsAbort) currentTtsAbort()
      // Send clear event to stop Twilio audio
      ws.send(JSON.stringify({ event: 'clear', streamSid }))
    }

    // Clear silence timer
    if (silenceTimer) clearTimeout(silenceTimer)

    conversationHistory.push({ role: 'user', content: transcript })

    // Get LLM response
    const response = await getLLMResponse(systemPrompt, conversationHistory, anthropic)
    conversationHistory.push({ role: 'assistant', content: response })

    // Check for outcome signals in response
    const outcome = detectOutcome(response)
    if (outcome) {
      await handleOutcome(outcome, contact, callDbId, conversationHistory, supabase)
    }

    // Speak the response
    isAgentSpeaking = true
    await speakText(response, ws, streamSid, agent.voice_id, () => {
      isAgentSpeaking = false
    }, (abort) => { currentTtsAbort = abort })
    isAgentSpeaking = false
  })
})

// ─── LLM ──────────────────────────────────────────────────────────────────
async function getLLMResponse(systemPrompt, history, client) {
  const response = await client.messages.create({
    model: 'claude-sonnet-4-20250514',
    max_tokens: 150,
    system: systemPrompt,
    messages: history,
  })
  return response.content[0].text
}

function buildSystemPrompt(agent, contact) {
  const isRE = agent.type === 'real_estate'
  return `${agent.system_prompt}

CONTACT INFO:
- Name: ${contact.name}
- Company: ${contact.company || 'unknown'}
${isRE ? `- Listing Address: ${contact.listing_address || 'not provided'}` : ''}
- Notes: ${contact.notes || 'none'}

RULES:
- Keep responses to 1-2 sentences max. This is a phone call.
- Be natural and conversational, not robotic.
- If they want to book a demo/appointment, say "BOOKING_CONFIRMED: [day and time they said]"
- If they're clearly interested but not ready to book, say "OUTCOME: interested"
- If they're not interested, say "OUTCOME: not_interested" and politely end the call.
- If you reached voicemail, leave a brief message and say "OUTCOME: voicemail"`
}

function buildGreeting(agent, contact) {
  const firstName = contact.name.split(' ')[0]
  if (agent.type === 'real_estate') {
    return `Hey ${firstName}, this is ${agent.name} calling about the property at ${contact.listing_address || 'your listing'}. Do you have a quick minute?`
  }
  return `Hey ${firstName}, this is ${agent.name} from our team. I'm reaching out about ${contact.company || 'your business'} — do you have a quick minute?`
}

// ─── TTS (ElevenLabs) ─────────────────────────────────────────────────────
async function speakText(text, ws, streamSid, voiceId, onDone, onAbort) {
  let aborted = false
  if (onAbort) onAbort(() => { aborted = true })

  try {
    const res = await fetch(
      `https://api.elevenlabs.io/v1/text-to-speech/${voiceId || process.env.DEFAULT_VOICE_ID}/stream`,
      {
        method: 'POST',
        headers: {
          'xi-api-key': process.env.ELEVENLABS_API_KEY,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text,
          model_id: 'eleven_turbo_v2',
          output_format: 'ulaw_8000', // ElevenLabs can output mulaw directly
        }),
      }
    )

    if (!res.ok || aborted) return

    const chunks = []
    for await (const chunk of res.body) {
      if (aborted) break
      chunks.push(chunk)
    }

    if (!aborted && ws.readyState === 1) {
      const audio = Buffer.concat(chunks)
      // Send in 20ms frames (160 bytes at 8kHz mulaw)
      for (let i = 0; i < audio.length; i += 160) {
        if (aborted) break
        const frame = audio.slice(i, i + 160)
        ws.send(JSON.stringify({
          event: 'media',
          streamSid,
          media: { payload: frame.toString('base64') }
        }))
      }
    }
  } catch (e) {
    console.error('TTS error:', e.message)
  }

  if (onDone) onDone()
}

// ─── DEEPGRAM ─────────────────────────────────────────────────────────────
async function openDeepgramStream() {
  const WebSocket = (await import('ws')).default
  const ws = new WebSocket(
    'wss://api.deepgram.com/v1/listen?' +
    'encoding=mulaw&sample_rate=8000&model=nova-3&' +
    'interim_results=false&utterance_end_ms=1000&vad_events=true',
    { headers: { Authorization: `Token ${process.env.DEEPGRAM_API_KEY}` } }
  )
  return ws
}

// ─── OUTCOME DETECTION ────────────────────────────────────────────────────
function detectOutcome(text) {
  if (text.includes('BOOKING_CONFIRMED:')) return 'booked'
  if (text.includes('OUTCOME: interested')) return 'interested'
  if (text.includes('OUTCOME: not_interested')) return 'not_interested'
  if (text.includes('OUTCOME: voicemail')) return 'voicemail'
  return null
}

async function handleOutcome(outcome, contact, callDbId, history, supabase) {
  const callStatusMap = {
    booked: 'booked', interested: 'warm',
    not_interested: 'dead', voicemail: 'voicemail'
  }

  await supabase.from('calls').update({
    outcome,
    transcript: history,
    status: 'completed',
    ended_at: new Date().toISOString()
  }).eq('id', callDbId)

  await supabase.from('contacts').update({
    call_status: callStatusMap[outcome] || 'no_answer',
    last_call_at: new Date().toISOString()
  }).eq('id', contact.id)
}

// ─── CLEANUP ──────────────────────────────────────────────────────────────
function cleanup(streamSid) {
  const state = activeCalls.get(streamSid)
  if (!state) return
  if (state.deepgramWs?.readyState === 1) state.deepgramWs.close()
  activeCalls.delete(streamSid)
}

// ─── OUTBOUND CALL INITIATOR ──────────────────────────────────────────────
// Called by your Supabase campaign runner instead of fireVapiCall()
app.post('/initiate-call', async (req, res) => {
  const { contactId, agentId, campaignStep } = req.body

  try {
    const { data: contact } = await supabase.from('contacts').select('*').eq('id', contactId).single()
    if (!contact?.phone) return res.status(400).json({ error: 'No phone number' })

    // Create call record in DB first to get an ID
    const { data: callRecord } = await supabase.from('calls').insert({
      contact_id: contactId,
      agent_id: agentId,
      campaign_step: campaignStep,
      status: 'initiated'
    }).select().single()

    const call = await twilioClient.calls.create({
      to: contact.phone,
      from: process.env.TWILIO_PHONE_NUMBER,
      url: `${process.env.SERVER_URL}/twiml?contactId=${contactId}&agentId=${agentId}&campaignStep=${campaignStep}&callDbId=${callRecord.id}`,
      statusCallback: `${process.env.SERVER_URL}/call-status`,
      statusCallbackMethod: 'POST',
      machineDetection: 'Enable',
      asyncAmd: true,
      asyncAmdStatusCallback: `${process.env.SERVER_URL}/amd-status`,
    })

    await supabase.from('calls').update({ twilio_call_sid: call.sid }).eq('id', callRecord.id)

    res.json({ success: true, callSid: call.sid, callId: callRecord.id })
  } catch (e) {
    res.status(500).json({ error: e.message })
  }
})

// ─── TWILIO STATUS CALLBACKS ──────────────────────────────────────────────
app.post('/call-status', async (req, res) => {
  const { CallSid, CallStatus } = req.body
  if (CallStatus === 'no-answer' || CallStatus === 'busy' || CallStatus === 'failed') {
    const { data: call } = await supabase.from('calls')
      .select('contact_id').eq('twilio_call_sid', CallSid).single()
    if (call) {
      await supabase.from('contacts').update({
        call_status: CallStatus === 'no-answer' ? 'no_answer' : CallStatus,
        last_call_at: new Date().toISOString()
      }).eq('id', call.contact_id)
    }
  }
  res.sendStatus(200)
})

// AMD: answering machine detection
app.post('/amd-status', async (req, res) => {
  const { CallSid, AnsweredBy } = req.body
  if (AnsweredBy === 'machine_start') {
    // Optionally hang up on voicemail or leave a message
    await twilioClient.calls(CallSid).update({ status: 'completed' })
    const { data: call } = await supabase.from('calls')
      .select('contact_id').eq('twilio_call_sid', CallSid).single()
    if (call) {
      await supabase.from('contacts').update({ call_status: 'voicemail' }).eq('id', call.contact_id)
    }
  }
  res.sendStatus(200)
})

const PORT = process.env.PORT || 3000
httpServer.listen(PORT, () => console.log(`Calling server running on port ${PORT}`))
