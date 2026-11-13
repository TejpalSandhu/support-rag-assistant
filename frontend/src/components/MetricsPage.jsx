import { useEffect, useState } from 'react'
import { api } from '../api'

const RANGES = [
  { days: 1, label: '24 hours' },
  { days: 7, label: '7 days' },
  { days: 30, label: '30 days' },
]

function percent(part, whole) {
  return whole ? `${Math.round((part / whole) * 100)}% of questions` : '—'
}

function PeopleTable({ people }) {
  if (people.length === 0) return <p className="muted small">Nobody has asked anything in this period.</p>
  return (
    <table className="people-table">
      <caption>Questions per person (from the database)</caption>
      <thead>
        <tr>
          <th scope="col">Person</th>
          <th scope="col">Questions</th>
          <th scope="col">Last active</th>
        </tr>
      </thead>
      <tbody>
        {people.map((person) => (
          <tr key={person.email}>
            <td>{person.email}</td>
            <td className="number">{person.questions}</td>
            <td>{person.last_active}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function Stat({ label, value, detail }) {
  return (
    <div className="stat card">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {detail && <div className="stat-detail muted small">{detail}</div>}
    </div>
  )
}

export default function MetricsPage({ onApiError, isOwner }) {
  const [days, setDays] = useState(7)
  const [scope, setScope] = useState('me') // 'me' or 'all' (owner only)
  const [reloads, setReloads] = useState(0)
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  // fetch whenever the range changes or Refresh is clicked
  useEffect(() => {
    let current = true // ignore a slow reply after the range changed again
    api
      .getMetrics(days, scope)
      .then((result) => {
        if (current) {
          setData(result)
          setError('')
        }
      })
      .catch((err) => {
        if (current && !onApiError(err)) setError(err.message)
      })
      .finally(() => {
        if (current) setLoading(false)
      })
    return () => {
      current = false
    }
  }, [days, scope, reloads, onApiError])

  function reload(newDays = days, newScope = scope) {
    setLoading(true)
    setDays(newDays)
    setScope(newScope)
    setReloads((count) => count + 1)
  }

  return (
    <main className="panel metrics-page">
      <div className="metrics-header">
        <div>
          <h2>Metrics</h2>
          <p className="muted small">
            {scope === 'all' ? 'Everyone’s questions' : 'Your questions'}, from the Langfuse traces
            of this site{data && ` (environment "${data.environment}")`}. New questions show up
            within a minute.
          </p>
        </div>
        <div className="metrics-controls">
          {isOwner && (
            <div className="range" role="group" aria-label="Whose numbers">
              {[
                ['me', 'Just me'],
                ['all', 'Everyone'],
              ].map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  className={`tab ${scope === value ? 'active' : ''}`}
                  aria-pressed={scope === value}
                  onClick={() => reload(days, value)}
                >
                  {label}
                </button>
              ))}
            </div>
          )}
          <div className="range" role="group" aria-label="Time range">
            {RANGES.map((range) => (
              <button
                key={range.days}
                type="button"
                className={`tab ${days === range.days ? 'active' : ''}`}
                aria-pressed={days === range.days}
                onClick={() => reload(range.days)}
              >
                {range.label}
              </button>
            ))}
          </div>
          <button type="button" className="tab" onClick={() => reload()} disabled={loading}>
            {loading ? 'Loading…' : 'Refresh'}
          </button>
        </div>
      </div>

      {error && <p className="error-text">{error}</p>}

      {data && (
        <div className={`metrics-grid ${loading ? 'stale' : ''}`}>
          <Stat label="Questions asked" value={data.questions} />
          <Stat
            label="Answered from the documents"
            value={data.answered}
            detail={percent(data.answered, data.questions)}
          />
          <Stat
            label="Not covered (no model call)"
            value={data.refused}
            detail={percent(data.refused, data.questions)}
          />
          <Stat
            label="Failed"
            value={data.failed}
            detail={data.failed ? 'model busy or usage limit' : 'no errors'}
          />
          <Stat
            label="Average response time"
            value={data.avg_seconds === null ? '—' : `${data.avg_seconds} s`}
            detail={data.p95_seconds === null ? null : `95% within ${data.p95_seconds} s`}
          />
          <Stat
            label="Helpful (👍)"
            value={data.helpful_rate === null ? '—' : `${Math.round(data.helpful_rate * 100)}%`}
            detail={
              data.feedback_votes
                ? `${data.feedback_votes} ${data.feedback_votes === 1 ? 'vote' : 'votes'}`
                : 'no votes yet'
            }
          />
        </div>
      )}

      {data?.people && (
        <div className={`people ${loading ? 'stale' : ''}`}>
          <PeopleTable people={data.people} />
        </div>
      )}
    </main>
  )
}
