import { useRef, useState } from 'react'
import { api } from '../api'

function StatusBadge({ status }) {
  if (status === 'ready') return <span className="badge badge-ok">ready</span>
  if (status === 'processing') return <span className="badge badge-busy">processing…</span>
  return (
    <span className="badge badge-error" title={status}>
      failed
    </span>
  )
}

export default function DocumentPanel({ documents, error, onChange, onApiError }) {
  const fileInput = useRef(null)
  const [uploadError, setUploadError] = useState('')
  const [uploading, setUploading] = useState(false)

  async function handleFiles(event) {
    const files = [...event.target.files]
    event.target.value = '' // lets you pick the same file again later
    setUploadError('')
    setUploading(true)
    for (const file of files) {
      try {
        await api.uploadDocument(file)
      } catch (err) {
        if (!onApiError(err)) setUploadError(`${file.name}: ${err.message}`)
      }
    }
    setUploading(false)
    onChange()
  }

  async function handleDelete(name) {
    if (!window.confirm(`Delete ${name} and everything learned from it?`)) return
    try {
      await api.deleteDocument(name)
    } catch (err) {
      if (!onApiError(err)) setUploadError(err.message)
    }
    onChange()
  }

  return (
    <aside className="panel documents">
      <div className="panel-header">
        <h2>Documents</h2>
        <button type="button" onClick={() => fileInput.current.click()} disabled={uploading}>
          {uploading ? 'Uploading…' : 'Upload PDF'}
        </button>
        <input
          ref={fileInput}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          hidden
          onChange={handleFiles}
        />
      </div>

      {error && <p className="error-text">{error}</p>}
      {uploadError && <p className="error-text">{uploadError}</p>}

      {documents.length === 0 ? (
        <p className="muted">No documents yet. Upload a PDF to start asking questions.</p>
      ) : (
        <ul className="document-list">
          {documents.map((doc) => (
            <li key={doc.name}>
              <div className="document-name" title={doc.name}>
                {doc.name}
              </div>
              <div className="document-meta">
                <StatusBadge status={doc.status} />
                {doc.status === 'ready' && <span className="muted">{doc.chunks} chunks</span>}
                <button
                  type="button"
                  className="link-button"
                  onClick={() => handleDelete(doc.name)}
                  disabled={doc.status === 'processing'}
                >
                  Delete
                </button>
              </div>
              {doc.status.startsWith('failed') && (
                <p className="error-text small">{doc.status.slice('failed: '.length)}</p>
              )}
            </li>
          ))}
        </ul>
      )}
    </aside>
  )
}
