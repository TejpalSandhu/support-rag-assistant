// Everything that talks to the backend lives here.

const KEY_STORAGE = 'support-rag-access-key'

export function getAccessKey() {
  try {
    return localStorage.getItem(KEY_STORAGE) || ''
  } catch {
    return '' // storage blocked (e.g. private browsing)
  }
}

export function setAccessKey(key) {
  try {
    localStorage.setItem(KEY_STORAGE, key)
  } catch {
    // storage blocked: the key just won't be remembered
  }
}

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

async function request(path, options = {}) {
  const headers = { ...options.headers }
  const key = getAccessKey()
  if (key) headers['X-API-Key'] = key

  let response
  try {
    response = await fetch(`/api${path}`, { ...options, headers })
  } catch {
    throw new ApiError(0, 'Cannot reach the server. Is the backend running?')
  }

  if (response.status === 204) return null
  const body = await response.json().catch(() => null)
  if (!response.ok) throw new ApiError(response.status, errorMessage(body, response.status))
  return body
}

export const api = {
  listDocuments: () => request('/documents'),

  uploadDocument: (file) => {
    const form = new FormData()
    form.append('file', file)
    return request('/documents', { method: 'POST', body: form })
  },

  deleteDocument: (name) =>
    request(`/documents/${encodeURIComponent(name)}`, { method: 'DELETE' }),

  ask: (question) =>
    request('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    }),
}
