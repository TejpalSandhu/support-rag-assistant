import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'

// A fake backend: `routes` maps "METHOD /path" to a list of replies,
// used in order (the last one repeats). Every request is recorded.
function fakeBackend(routes) {
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

async function ask(user, question) {
  await user.type(screen.getByPlaceholderText('Ask a question…'), question)
  await user.click(screen.getByRole('button', { name: 'Ask' }))
}

describe('App', () => {
  beforeEach(() => {
    vi.unstubAllGlobals()
  })

  it('asks for the access key on 401 and sends it afterwards', async () => {
    const calls = fakeBackend({
      'GET /api/documents': [[401, { detail: 'Missing or wrong access key.' }], [200, []]],
    })
    const user = userEvent.setup()
    render(<App />)

    await user.type(await screen.findByLabelText('Access key'), 'my-key')
    await user.click(screen.getByRole('button', { name: 'Continue' }))

    expect(await screen.findByText(/No documents yet/)).toBeInTheDocument()
    expect(calls.at(-1).headers['X-API-Key']).toBe('my-key')
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
})
