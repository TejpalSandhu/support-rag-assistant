import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import Chat from './components/Chat'
import ChatList from './components/ChatList'
import DocumentPanel from './components/DocumentPanel'
import MetricsPage from './components/MetricsPage'
import SignIn from './components/SignIn'

const POLL_MS = 3000

// the chat you had open, so a page refresh brings it back (per account)
const lastChatKey = (email) => `support-rag-last-chat:${email}`
function rememberChat(email, chatId) {
  try {
    if (chatId) localStorage.setItem(lastChatKey(email), chatId)
    else localStorage.removeItem(lastChatKey(email))
  } catch {
    // storage blocked: the chat just won't reopen after a refresh
  }
}
function lastChat(email) {
  try {
    return localStorage.getItem(lastChatKey(email))
  } catch {
    return null
  }
}

// saved messages -> the shape the chat view shows
function toMessages(saved) {
  return saved.map((message) =>
    message.role === 'user'
      ? { role: 'user', text: message.content }
      : { role: 'assistant', result: { ...message.details, answer: message.content } },
  )
}

export default function App() {
  // undefined = still checking, null = signed out, object = signed in
  const [user, setUser] = useState(undefined)
  const [signInMessage, setSignInMessage] = useState('')
  const [documents, setDocuments] = useState([])
  const [documentsError, setDocumentsError] = useState('')
  const [messages, setMessages] = useState([])
  const [asking, setAsking] = useState(false)
  const [page, setPage] = useState('chat') // 'chat' or 'metrics'
  const [chats, setChats] = useState([])
  const [chatsError, setChatsError] = useState('')
  const [currentChatId, setCurrentChatId] = useState(null) // null = a new chat
  const email = user?.email

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

  const refreshChats = useCallback(async () => {
    try {
      setChats(await api.listChats())
      setChatsError('')
    } catch (error) {
      if (!handleApiError(error)) setChatsError(error.message)
    }
  }, [handleApiError])

  const openChat = useCallback(
    async (chatId) => {
      try {
        setMessages(toMessages(await api.getChat(chatId)))
        setCurrentChatId(chatId)
        rememberChat(email, chatId)
        setPage('chat')
      } catch (error) {
        if (handleApiError(error)) return
        if (error.status === 404) rememberChat(email, null) // deleted meanwhile
        else setChatsError(error.message)
      }
    },
    [email, handleApiError],
  )

  // after signing in: documents, chats, and the chat you had open
  useEffect(() => {
    if (!email) return
    refreshDocuments()
    refreshChats()
    const previous = lastChat(email)
    if (previous) openChat(previous)
  }, [email, refreshDocuments, refreshChats, openChat])

  function newChat() {
    setCurrentChatId(null)
    setMessages([])
    rememberChat(email, null)
  }

  async function handleDeleteChat(chat) {
    if (!window.confirm(`Delete the chat "${chat.title}"?`)) return
    try {
      await api.deleteChat(chat.id)
      if (chat.id === currentChatId) newChat()
    } catch (error) {
      if (!handleApiError(error)) setChatsError(error.message)
    }
    refreshChats()
  }

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
      const result = await api.ask(question, currentChatId)
      setMessages((previous) => [...previous, { role: 'assistant', result }])
      if (result.chat_id !== currentChatId) {
        // the first question started a new chat
        setCurrentChatId(result.chat_id)
        rememberChat(email, result.chat_id)
      }
      refreshChats()
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
    setChats([])
    setCurrentChatId(null)
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
          <div className="sidebar">
            <DocumentPanel
              documents={documents}
              error={documentsError}
              onChange={refreshDocuments}
              onApiError={handleApiError}
            />
            {/* switching chats waits until an answer has arrived */}
            <fieldset className="chats-fieldset" disabled={asking}>
              <ChatList
                chats={chats}
                currentChatId={currentChatId}
                onOpen={openChat}
                onNew={newChat}
                onDelete={handleDeleteChat}
                error={chatsError}
              />
            </fieldset>
          </div>
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
