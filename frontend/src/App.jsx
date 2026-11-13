import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import Chat from './components/Chat'
import DocumentPanel from './components/DocumentPanel'
import MetricsPage from './components/MetricsPage'
import SignIn from './components/SignIn'

const POLL_MS = 3000

export default function App() {
  // undefined = still checking, null = signed out, object = signed in
  const [user, setUser] = useState(undefined)
  const [signInMessage, setSignInMessage] = useState('')
  const [documents, setDocuments] = useState([])
  const [documentsError, setDocumentsError] = useState('')
  const [messages, setMessages] = useState([])
  const [asking, setAsking] = useState(false)
  const [page, setPage] = useState('chat') // 'chat' or 'metrics'
  const signedIn = Boolean(user)

  // 401 = not signed in (or the session ran out), 403 = removed from the list
  const handleApiError = useCallback((error) => {
    if (error.status === 401 || error.status === 403) {
      setSignInMessage(error.status === 403 ? error.message : '')
      setUser(null)
      return true
    }
    return false
  }, [])

  // who is signed in? (the session cookie is sent automatically)
  useEffect(() => {
    api
      .me()
      .then(setUser)
      .catch((error) => {
        if (!handleApiError(error)) {
          setSignInMessage(error.message)
          setUser(null)
        }
      })
  }, [handleApiError])

  const refreshDocuments = useCallback(async () => {
    try {
      setDocuments(await api.listDocuments())
      setDocumentsError('')
    } catch (error) {
      if (!handleApiError(error)) setDocumentsError(error.message)
    }
  }, [handleApiError])

  // load the list after signing in
  useEffect(() => {
    if (signedIn) refreshDocuments()
  }, [signedIn, refreshDocuments])

  // while a PDF is being processed, check its status every few seconds
  const processing = documents.some((doc) => doc.status === 'processing')
  useEffect(() => {
    if (!processing) return undefined
    const timer = setInterval(refreshDocuments, POLL_MS)
    return () => clearInterval(timer)
  }, [processing, refreshDocuments])

  async function handleAsk(question) {
    setMessages((previous) => [...previous, { role: 'user', text: question }])
    setAsking(true)
    try {
      const result = await api.ask(question)
      setMessages((previous) => [...previous, { role: 'assistant', result }])
    } catch (error) {
      if (!handleApiError(error)) {
        setMessages((previous) => [...previous, { role: 'error', text: error.message }])
      }
    } finally {
      setAsking(false)
      api.me().then(setUser).catch(() => {}) // update "questions left today"
    }
  }

  const handleSignedIn = useCallback(async () => {
    setSignInMessage('')
    setUser(await api.me())
  }, [])

  async function handleSignOut() {
    await api.signOut().catch(() => {})
    window.google?.accounts?.id?.disableAutoSelect() // don't sign straight back in
    setMessages([])
    setDocuments([])
    setPage('chat')
    setUser(null)
  }

  if (user === undefined) return <div className="key-screen muted">Loading…</div>
  if (user === null) return <SignIn onSignedIn={handleSignedIn} message={signInMessage} />

  return (
    <div className="layout">
      <header className="top-bar">
        <span className="logo">Support RAG Assistant</span>
        <span className="muted tagline">Answers only from your PDFs</span>
        <nav className="nav" aria-label="Pages">
          {['chat', 'metrics'].map((name) => (
            <button
              key={name}
              type="button"
              className={`tab ${page === name ? 'active' : ''}`}
              aria-current={page === name ? 'page' : undefined}
              onClick={() => setPage(name)}
            >
              {name === 'chat' ? 'Chat' : 'Metrics'}
            </button>
          ))}
        </nav>
        <div className="account">
          <span className="account-email" title={user.email}>
            {user.email}
          </span>
          <span className="muted small">
            {user.questions_left_today} of {user.daily_question_limit} questions left today
          </span>
          {user.login_enabled && (
            <button type="button" className="tab" onClick={handleSignOut}>
              Sign out
            </button>
          )}
        </div>
      </header>
      {page === 'metrics' ? (
        <MetricsPage onApiError={handleApiError} />
      ) : (
        <>
          <DocumentPanel
            documents={documents}
            error={documentsError}
            onChange={refreshDocuments}
            onApiError={handleApiError}
          />
          {/* the chat stays in memory while you look at the metrics */}
          <Chat
            messages={messages}
            asking={asking}
            onAsk={handleAsk}
            hasDocuments={documents.some((doc) => doc.status === 'ready')}
            onApiError={handleApiError}
          />
        </>
      )}
    </div>
  )
}
