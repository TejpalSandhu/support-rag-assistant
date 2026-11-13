import { useState } from 'react'

export default function AccessKeyForm({ onSubmit }) {
  const [key, setKey] = useState('')

  function handleSubmit(event) {
    event.preventDefault()
    if (key.trim()) onSubmit(key.trim())
  }

  return (
    <div className="key-screen">
      <form className="card key-card" onSubmit={handleSubmit}>
        <h1>Support RAG Assistant</h1>
        <p className="muted">This assistant is private. Enter the access key to continue.</p>
        <label htmlFor="access-key">Access key</label>
        <input
          id="access-key"
          type="password"
          value={key}
          onChange={(event) => setKey(event.target.value)}
          autoFocus
          autoComplete="current-password"
        />
        <button type="submit" disabled={!key.trim()}>
          Continue
        </button>
      </form>
    </div>
  )
}
