// Everything that talks to the backend lives here.

export class ApiError extends Error {
  constructor(status, message) {
    super(message)
    this.status = status
  }
}

function errorMessage(body, status) {
  const detail = body?.detail
  if (typeof detail === 'string') return detail
  // FastAPI validation errors come as a list
  if (Array.isArray(detail)) return detail.map((item) => item.msg).join('; ')
  return `Request failed (${status})`
}

// the session cookie from signing in goes along automatically (same site)
async function request(path, options = {}) {
  let response
  try {
    response = await fetch(`/api${path}`, options)
  } catch {
    throw new ApiError(0, 'Cannot reach the server. Is the backend running?')
  }

  if (response.status === 204) return null
  const body = await response.json().catch(() => null)
  if (!response.ok) throw new ApiError(response.status, errorMessage(body, response.status))
  return body
}

function postJson(path, body) {
  return request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export const api = {
  getAuthConfig: () => request('/auth/config'),
  signInWithGoogle: (credential) => postJson('/auth/google', { credential }),
  signOut: () => request('/auth/logout', { method: 'POST' }),
  me: () => request('/me'),

  listDocuments: () => request('/documents'),

  uploadDocument: (file) => {
    const form = new FormData()
    form.append('file', file)
    return request('/documents', { method: 'POST', body: form })
  },

  deleteDocument: (name) =>
    request(`/documents/${encodeURIComponent(name)}`, { method: 'DELETE' }),

  // chatId: continue that chat; null starts a new one
  ask: (question, chatId) => postJson('/ask', { question, chat_id: chatId || null }),

  listChats: () => request('/chats'),
  getChat: (chatId) => request(`/chats/${encodeURIComponent(chatId)}`),
  deleteChat: (chatId) => request(`/chats/${encodeURIComponent(chatId)}`, { method: 'DELETE' }),

  sendFeedback: (traceId, helpful) => postJson('/feedback', { trace_id: traceId, helpful }),

  getMetrics: (days) => request(`/metrics?days=${days}`),
}
