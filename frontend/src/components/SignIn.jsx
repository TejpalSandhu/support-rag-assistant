import { useEffect, useRef, useState } from 'react'
import { api } from '../api'

const GOOGLE_SCRIPT = 'https://accounts.google.com/gsi/client'

// Google's sign-in library; loaded only on this screen
function loadGoogleScript() {
  if (window.google?.accounts?.id) return Promise.resolve()
  return new Promise((resolve, reject) => {
    const script = document.createElement('script')
    script.src = GOOGLE_SCRIPT
    script.async = true
    script.onload = resolve
    script.onerror = () =>
      reject(new Error('Could not load Google sign-in. Check your connection or ad blocker.'))
    document.head.appendChild(script)
  })
}

export default function SignIn({ onSignedIn, message }) {
  const buttonSlot = useRef(null)
  const [error, setError] = useState('')
  const [signingIn, setSigningIn] = useState(false)

  useEffect(() => {
    let active = true

    // Google calls this with an ID token after the user picks an account
    async function handleCredential({ credential }) {
      setSigningIn(true)
      setError('')
      try {
        onSignedIn(await api.signInWithGoogle(credential))
      } catch (err) {
        setError(err.message)
      } finally {
        setSigningIn(false)
      }
    }

    async function showButton() {
      try {
        const { google_client_id: clientId } = await api.getAuthConfig()
        if (!clientId) throw new Error('Google sign-in isn’t set up on this server.')
        await loadGoogleScript()
        if (!active) return
        const dark = window.matchMedia?.('(prefers-color-scheme: dark)').matches
        window.google.accounts.id.initialize({ client_id: clientId, callback: handleCredential })
        window.google.accounts.id.renderButton(buttonSlot.current, {
          theme: dark ? 'filled_black' : 'outline',
          size: 'large',
          shape: 'pill',
          text: 'signin_with',
        })
      } catch (err) {
        if (active) setError(err.message)
      }
    }

    showButton()
    return () => {
      active = false
    }
  }, [onSignedIn])

  return (
    <div className="key-screen">
      <div className="card key-card">
        <h1>Support RAG Assistant</h1>
        <p className="muted">
          Upload PDFs and ask questions about them. Sign in with the Google account the owner
          invited.
        </p>
        <div ref={buttonSlot} className="google-button" />
        {signingIn && <p className="muted small">Signing in…</p>}
        {(error || message) && <p className="error-text">{error || message}</p>}
      </div>
    </div>
  )
}
