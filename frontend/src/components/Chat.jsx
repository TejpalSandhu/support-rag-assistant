import { useEffect, useRef, useState } from 'react'
import Markdown from 'react-markdown'

const SNIPPET_CHARS = 300

function Sources({ sources, covered }) {
  if (sources.length === 0) return null
  const count = `${sources.length} ${sources.length === 1 ? 'passage' : 'passages'}`
  return (
    <details className="sources">
      {/* when not covered, these matches were too weak to answer from */}
      <summary>{covered ? `${count} used as sources` : `${count} checked, none close enough`}</summary>
      <ol>
        {sources.map((source, index) => (
          <li key={index}>
            <div className="source-title">
              {source.source}, page {source.page}
              <span className="muted"> · score {source.score.toFixed(3)}</span>
            </div>
            <p className="source-text">
              {source.text.length > SNIPPET_CHARS
                ? `${source.text.slice(0, SNIPPET_CHARS)}…`
                : source.text}
            </p>
          </li>
        ))}
      </ol>
    </details>
  )
}

function Message({ message }) {
  if (message.role === 'user') {
    return <div className="message user">{message.text}</div>
  }
  if (message.role === 'error') {
    return <div className="message error">{message.text}</div>
  }

  const { result } = message
  return (
    <div className={`message assistant ${result.covered ? '' : 'not-covered'}`}>
      <div className="message-tag">
        {result.covered ? 'Answered from your documents' : 'Not covered by your documents'}
        <span className="muted"> · best match {result.best_score.toFixed(3)}</span>
      </div>
      <div className="answer">
        <Markdown>{result.answer}</Markdown>
      </div>
      <Sources sources={result.sources} covered={result.covered} />
      {result.trace_url && (
        <a className="trace-link" href={result.trace_url} target="_blank" rel="noreferrer">
          View trace in Langfuse
        </a>
      )}
    </div>
  )
}

export default function Chat({ messages, asking, onAsk, hasDocuments }) {
  const [question, setQuestion] = useState('')
  const bottom = useRef(null)

  // keep the newest message in view
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, asking])

  function submit() {
    const text = question.trim()
    if (!text || asking) return
    setQuestion('')
    onAsk(text)
  }

  function handleKeyDown(event) {
    // Enter sends, Shift+Enter makes a new line
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <main className="panel chat">
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty-chat">
            <h2>Ask about your documents</h2>
            <p className="muted">
              {hasDocuments
                ? 'Answers come only from the uploaded PDFs, with page references. If the documents don’t cover a question, the assistant says so instead of guessing.'
                : 'Upload a PDF on the left, then ask a question about it.'}
            </p>
          </div>
        )}
        {messages.map((message, index) => (
          <Message key={index} message={message} />
        ))}
        {asking && <div className="message assistant thinking">Searching the documents…</div>}
        <div ref={bottom} />
      </div>

      <form
        className="ask-form"
        onSubmit={(event) => {
          event.preventDefault()
          submit()
        }}
      >
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question…"
          rows={2}
          maxLength={1000}
        />
        <button type="submit" disabled={asking || !question.trim()}>
          Ask
        </button>
      </form>
    </main>
  )
}
