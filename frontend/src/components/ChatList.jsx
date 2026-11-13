// Your saved chats: open one, start a new one, or delete one.

function when(isoDate) {
  const date = new Date(isoDate)
  const today = new Date()
  return date.toDateString() === today.toDateString()
    ? date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : date.toLocaleDateString([], { day: 'numeric', month: 'short' })
}

export default function ChatList({ chats, currentChatId, onOpen, onNew, onDelete, error }) {
  return (
    <section className="panel chats" aria-label="Chats">
      <div className="panel-header">
        <h2>Chats</h2>
        <button type="button" className="tab" onClick={onNew} disabled={!currentChatId}>
          New chat
        </button>
      </div>

      {error && <p className="error-text small">{error}</p>}
      {chats.length === 0 && !error && (
        <p className="muted small">Your questions and answers are saved here.</p>
      )}

      <ul className="chat-list">
        {chats.map((chat) => (
          <li key={chat.id} className={chat.id === currentChatId ? 'current' : ''}>
            <button
              type="button"
              className="chat-open"
              aria-current={chat.id === currentChatId ? 'true' : undefined}
              onClick={() => onOpen(chat.id)}
            >
              <span className="chat-title">{chat.title}</span>
              <span className="muted small">{when(chat.updated_at)}</span>
            </button>
            <button
              type="button"
              className="link-button chat-delete"
              aria-label={`Delete chat: ${chat.title}`}
              onClick={() => onDelete(chat)}
            >
              ✕
            </button>
          </li>
        ))}
      </ul>
    </section>
  )
}
