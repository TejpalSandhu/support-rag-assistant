import { useCallback, useEffect, useState } from 'react'
import { api, setAccessKey } from './api'
import AccessKeyForm from './components/AccessKeyForm'
import Chat from './components/Chat'
import DocumentPanel from './components/DocumentPanel'

const POLL_MS = 3000

export default function App() {
  const [needsKey, setNeedsKey] = useState(false)
  const [documents, setDocuments] = useState([])
  const [documentsError, setDocumentsError] = useState('')
  const [messages, setMessages] = useState([])
  const [asking, setAsking] = useState(false)

  // a 401 anywhere means the key is missing or wrong: show the key form
  const handleApiError = useCallback((error) => {
    if (error.status === 401) {
      setNeedsKey(true)
      return true
    }
    return false
  }, [])

  const refreshDocuments = useCallback(async () => {
    try {
      setDocuments(await api.listDocuments())
      setDocumentsError('')
    } catch (error) {
      if (!handleApiError(error)) setDocumentsError(error.message)
    }
  }, [handleApiError])

  // load the list once at start (and again after a new key is entered)
  useEffect(() => {
    if (!needsKey) refreshDocuments()
  }, [needsKey, refreshDocuments])

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
    }
  }

  function handleKey(key) {
    setAccessKey(key)
    setNeedsKey(false)
  }

  if (needsKey) return <AccessKeyForm onSubmit={handleKey} />

  return (
    <div className="layout">
      <header className="top-bar">
        <span className="logo">Support RAG Assistant</span>
        <span className="muted">Answers only from your PDFs</span>
      </header>
      <DocumentPanel
        documents={documents}
        error={documentsError}
        onChange={refreshDocuments}
        onApiError={handleApiError}
      />
      <Chat
        messages={messages}
        asking={asking}
        onAsk={handleAsk}
        hasDocuments={documents.some((doc) => doc.status === 'ready')}
      />
    </div>
  )
}
