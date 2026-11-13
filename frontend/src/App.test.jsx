import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

// A fake backend: `routes` maps "METHOD /path" to a list of replies,
// used in order (the last one repeats). Every request is recorded.
const USER = {
  email: 'friend@gmail.com',
  name: 'Friend',
  picture: '',
  login_enabled: true,
  questions_left_today: 20,
  daily_question_limit: 20,
}

function fakeBackend(routes) {
  // signed in, unless a test says otherwise
  routes = {
    'GET /api/me': [[200, USER]],
    'POST /api/auth/logout': [[204, null]],
    'GET /api/chats': [[200, []]],
    ...routes,
  }
  const calls = []
  const fetchMock = vi.fn(async (url, options = {}) => {
    const method = options.method || 'GET'
    const key = `${method} ${url}`
    calls.push({ key, headers: options.headers || {}, body: options.body })
    const replies = routes[key]
    if (!replies) throw new Error(`No fake reply for ${key}`)
    const [status, body] = replies.length > 1 ? replies.shift() : replies[0]
    return { ok: status < 400, status, json: async () => body }
  })
  vi.stubGlobal('fetch', fetchMock)
  return calls
}

const READY_DOC = { name: 'manual.pdf', chunks: 12, status: 'ready' }

const COVERED_ANSWER = {
  answer: 'Insert the filament into the **PTFE tube**.',
  covered: true,
  best_score: 0.812,
  sources: [{ source: 'manual.pdf', page: 19, score: 0.812, text: 'Loading filament: cut the tip…' }],
  trace_id: null,
  trace_url: null,
}

const TRACE_ID = '549d3b0af7e2aae54c27028ee260eb9a'

const METRICS = {
  days: 7,
  environment: 'production',
  questions: 10,
  answered: 6,
  refused: 3,
  failed: 1,
  avg_seconds: 2.4,
  p95_seconds: 6.1,
  feedback_votes: 4,
  helpful_rate: 0.75,
}

async function ask(user, question) {
  await user.type(screen.getByPlaceholderText('Ask a question…'), question)
  await user.click(screen.getByRole('button', { name: 'Ask' }))
}

describe('App', () => {
  beforeEach(() => {
    vi.unstubAllGlobals()
  })

  it('shows Google sign-in when not signed in, then the app after signing in', async () => {
    const calls = fakeBackend({
      'GET /api/me': [[401, { detail: 'Please sign in.' }], [200, USER]],
      'GET /api/auth/config': [[200, { google_client_id: 'test-client' }]],
      'POST /api/auth/google': [[200, USER]],
      'GET /api/documents': [[200, []]],
    })
    // a stand-in for Google's library: its "button" signs in straight away
    const google = { initialize: vi.fn(), renderButton: vi.fn(), disableAutoSelect: vi.fn() }
    google.renderButton.mockImplementation((slot) => {
      const button = document.createElement('button')
      button.textContent = 'Sign in with Google'
      button.onclick = () => google.initialize.mock.calls[0][0].callback({ credential: 'google-token' })
      slot.appendChild(button)
    })
    vi.stubGlobal('google', { accounts: { id: google } })
    const user = userEvent.setup()
    render(<App />)

    await user.click(await screen.findByRole('button', { name: 'Sign in with Google' }))

    expect(await screen.findByText('friend@gmail.com')).toBeInTheDocument()
    expect(screen.getByText('20 of 20 questions left today')).toBeInTheDocument()
    expect(google.initialize.mock.calls[0][0].client_id).toBe('test-client')
    const login = calls.find((call) => call.key === 'POST /api/auth/google')
    expect(JSON.parse(login.body)).toEqual({ credential: 'google-token' })
  })

  it('explains when an email is not on the access list', async () => {
    fakeBackend({
      'GET /api/me': [[403, { detail: 'stranger@gmail.com no longer has access.' }]],
      'GET /api/auth/config': [[200, { google_client_id: 'test-client' }]],
    })
    vi.stubGlobal('google', { accounts: { id: { initialize: vi.fn(), renderButton: vi.fn() } } })
    render(<App />)

    expect(await screen.findByText('stranger@gmail.com no longer has access.')).toBeInTheDocument()
  })

  it('signs out', async () => {
    const calls = fakeBackend({ 'GET /api/documents': [[200, []]], 'GET /api/auth/config': [[200, { google_client_id: 'x' }]] })
    vi.stubGlobal('google', { accounts: { id: { initialize: vi.fn(), renderButton: vi.fn(), disableAutoSelect: vi.fn() } } })
    const user = userEvent.setup()
    render(<App />)

    await user.click(await screen.findByRole('button', { name: 'Sign out' }))

    expect(await screen.findByText(/Sign in with the Google account/)).toBeInTheDocument()
    expect(calls.some((call) => call.key === 'POST /api/auth/logout')).toBe(true)
  })

  it('lists documents with their status and failure reason', async () => {
    fakeBackend({
      'GET /api/documents': [[200, [
        READY_DOC,
        { name: 'scan.pdf', chunks: 0, status: 'failed: No extractable text found' },
      ]]],
    })
    render(<App />)

    expect(await screen.findByText('manual.pdf')).toBeInTheDocument()
    expect(screen.getByText('12 chunks')).toBeInTheDocument()
    expect(screen.getByText('No extractable text found')).toBeInTheDocument()
  })

  it('shows an answer with its sources', async () => {
    fakeBackend({
      'GET /api/documents': [[200, [READY_DOC]]],
      'POST /api/ask': [[200, COVERED_ANSWER]],
    })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('manual.pdf')

    await ask(user, 'How do I load filament?')

    expect(await screen.findByText('Answered from your documents')).toBeInTheDocument()
    expect(screen.getByText('PTFE tube').tagName).toBe('STRONG') // markdown rendered
    expect(screen.getByText('1 passage used as sources')).toBeInTheDocument()
    expect(screen.getByText(/manual.pdf, page 19/)).toBeInTheDocument()
  })

  it('marks answers the documents do not cover', async () => {
    fakeBackend({
      'GET /api/documents': [[200, [READY_DOC]]],
      'POST /api/ask': [[200, { ...COVERED_ANSWER, covered: false, best_score: 0.41, answer: 'Sorry, I won’t guess.' }]],
    })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('manual.pdf')

    await ask(user, 'What should I feed a cat?')

    expect(await screen.findByText('Not covered by your documents')).toBeInTheDocument()
    expect(screen.getByText('1 passage checked, none close enough')).toBeInTheDocument()
  })

  it('shows the server message when the model is busy', async () => {
    fakeBackend({
      'GET /api/documents': [[200, [READY_DOC]]],
      'POST /api/ask': [[503, { detail: 'The AI model is busy right now. Please try again in a minute.' }]],
    })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('manual.pdf')

    await ask(user, 'How do I load filament?')

    expect(await screen.findByText(/model is busy/)).toBeInTheDocument()
  })

  it('uploads a PDF and checks its status until it is ready', async () => {
    const calls = fakeBackend({
      'GET /api/documents': [
        [200, []],
        [200, [{ name: 'new.pdf', chunks: 0, status: 'processing' }]],
        [200, [{ name: 'new.pdf', chunks: 5, status: 'ready' }]],
      ],
      'POST /api/documents': [[202, { name: 'new.pdf', chunks: 0, status: 'processing' }]],
    })
    const user = userEvent.setup()
    const { container } = render(<App />)
    await screen.findByText(/No documents yet/)

    const file = new File(['%PDF-1.4'], 'new.pdf', { type: 'application/pdf' })
    await user.upload(container.querySelector('input[type="file"]'), file)

    expect(await screen.findByText('processing…')).toBeInTheDocument()
    // the app polls every 3 seconds while something is processing
    await waitFor(() => expect(screen.getByText('5 chunks')).toBeInTheDocument(), { timeout: 5000 })
    expect(calls.some((call) => call.key === 'POST /api/documents')).toBe(true)
  })

  it('sends 👍/👎 feedback for an answer and lets you change it', async () => {
    const calls = fakeBackend({
      'GET /api/documents': [[200, [READY_DOC]]],
      'POST /api/ask': [[200, { ...COVERED_ANSWER, trace_id: TRACE_ID }]],
      'POST /api/feedback': [[204, null]],
    })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('manual.pdf')
    await ask(user, 'How do I load filament?')

    await user.click(await screen.findByRole('button', { name: 'Helpful' }))
    expect(await screen.findByText('Thanks for the feedback')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Helpful' })).toHaveAttribute('aria-pressed', 'true')

    await user.click(screen.getByRole('button', { name: 'Not helpful' }))
    const votes = calls.filter((call) => call.key === 'POST /api/feedback').map((call) => JSON.parse(call.body))
    expect(votes).toEqual([
      { trace_id: TRACE_ID, helpful: true },
      { trace_id: TRACE_ID, helpful: false },
    ])
  })

  it('hides the feedback buttons when tracing is off', async () => {
    fakeBackend({
      'GET /api/documents': [[200, [READY_DOC]]],
      'POST /api/ask': [[200, COVERED_ANSWER]], // trace_id: null
    })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('manual.pdf')
    await ask(user, 'How do I load filament?')

    await screen.findByText('Answered from your documents')
    expect(screen.queryByRole('button', { name: 'Helpful' })).not.toBeInTheDocument()
  })

  it('shows the metrics page and switches the time range', async () => {
    const calls = fakeBackend({
      'GET /api/documents': [[200, [READY_DOC]]],
      'GET /api/metrics?days=7': [[200, METRICS]],
      'GET /api/metrics?days=30': [[200, { ...METRICS, days: 30, questions: 40 }]],
    })
    const user = userEvent.setup()
    render(<App />)
    await screen.findByText('manual.pdf')

    await user.click(screen.getByRole('button', { name: 'Metrics' }))
    expect(await screen.findByText('Questions asked')).toBeInTheDocument()
    expect(screen.getByText('60% of questions')).toBeInTheDocument() // 6 of 10 answered
    expect(screen.getByText('75%')).toBeInTheDocument() // helpful rate
    expect(screen.getByText('4 votes')).toBeInTheDocument()
    expect(screen.getByText('2.4 s')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '30 days' }))
    expect(await screen.findByText('40')).toBeInTheDocument()
    expect(calls.at(-1).key).toBe('GET /api/metrics?days=30')
  })

  it('saves the first question as a new chat and reopens it after a refresh', async () => {
    const chat = { id: 'chat-1', title: 'How do I load filament?', updated_at: new Date().toISOString() }
    const calls = fakeBackend({
      'GET /api/documents': [[200, [READY_DOC]]],
      'POST /api/ask': [[200, { ...COVERED_ANSWER, chat_id: 'chat-1' }]],
      'GET /api/chats': [[200, []], [200, [chat]]],
      'GET /api/chats/chat-1': [[200, [
        { role: 'user', content: 'How do I load filament?', details: null },
        { role: 'assistant', content: COVERED_ANSWER.answer, details: { ...COVERED_ANSWER, answer: undefined } },
      ]]],
    })
    const user = userEvent.setup()
    const { unmount } = render(<App />)
    await screen.findByText('manual.pdf')
    await ask(user, 'How do I load filament?')

    // the new chat shows up in the list
    expect(await screen.findByRole('button', { name: /^How do I load filament\?/ })).toBeInTheDocument()
    expect(JSON.parse(calls.find((c) => c.key === 'POST /api/ask').body)).toEqual({
      question: 'How do I load filament?',
      chat_id: null,
    })

    // "refresh": the page starts again and reopens the same chat
    unmount()
    render(<App />)
    expect(await screen.findByText('PTFE tube')).toBeInTheDocument()
    expect(calls.at(-1).key).toBe('GET /api/chats/chat-1')
  })

  it('sends follow-up questions to the open chat', async () => {
    const chat = { id: 'chat-7', title: 'Old chat', updated_at: new Date().toISOString() }
    const calls = fakeBackend({
      'GET /api/documents': [[200, [READY_DOC]]],
      'GET /api/chats': [[200, [chat]]],
      'GET /api/chats/chat-7': [[200, [{ role: 'user', content: 'Earlier question', details: null }]]],
      'POST /api/ask': [[200, { ...COVERED_ANSWER, chat_id: 'chat-7' }]],
    })
    const user = userEvent.setup()
    render(<App />)

    await user.click(await screen.findByRole('button', { name: /^Old chat/ }))
    expect(await screen.findByText('Earlier question')).toBeInTheDocument()
    await ask(user, 'And then?')

    const sent = JSON.parse(calls.find((c) => c.key === 'POST /api/ask').body)
    expect(sent).toEqual({ question: 'And then?', chat_id: 'chat-7' })
  })

  it('deletes a chat and starts a new one', async () => {
    const chat = { id: 'chat-9', title: 'Remove me', updated_at: new Date().toISOString() }
    const calls = fakeBackend({
      'GET /api/documents': [[200, [READY_DOC]]],
      'GET /api/chats': [[200, [chat]], [200, [chat]], [200, []]],
      'GET /api/chats/chat-9': [[200, [{ role: 'user', content: 'Old question', details: null }]]],
      'DELETE /api/chats/chat-9': [[204, null]],
    })
    vi.spyOn(window, 'confirm').mockReturnValue(true)
    const user = userEvent.setup()
    render(<App />)

    await user.click(await screen.findByRole('button', { name: /^Remove me/ }))
    await screen.findByText('Old question')
    await user.click(screen.getByRole('button', { name: 'Delete chat: Remove me' }))

    await waitFor(() => expect(screen.queryByText('Old question')).not.toBeInTheDocument())
    expect(calls.some((c) => c.key === 'DELETE /api/chats/chat-9')).toBe(true)
  })
})
