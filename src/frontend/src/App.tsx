import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
}

const SUGGESTED = [
  { icon: '🎓', text: 'What are the admissions requirements?' },
  { icon: '📚', text: 'Tell me about the curriculum' },
  { icon: '⚽', text: 'What clubs and activities are available?' },
  { icon: '📅', text: 'When is the next open day?' },
  { icon: '💷', text: 'What are the school fees?' },
  { icon: '🏫', text: 'Tell me about the school facilities' },
];

const API_BASE = process.env.REACT_APP_API_URL || '';

async function streamChat(
  message: string,
  history: Message[],
  onToken: (token: string) => void
): Promise<string[]> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      history: history.map(({ role, content }) => ({ role, content })),
    }),
  });

  if (!response.ok) throw new Error('Chat request failed');

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let full = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    full += chunk;
    if (!chunk.includes('__sources__:')) {
      onToken(chunk);
    }
  }

  const sourceMatch = full.match(/__sources__:(.+)$/);
  return sourceMatch ? sourceMatch[1].split(',').map(s => s.trim()).filter(Boolean) : [];
}

// ── Icons ──────────────────────────────────────────────────────────
const SendIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
);

const SchoolIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 3L1 9l4 2.18V16c0 1.1.9 2 2 2h10c1.1 0 2-.9 2-2v-4.82L21 9l-9-6zm5 13H7v-3.99l5 2.72 5-2.72V16zm-5-4.28L3.53 9 12 4.28 20.47 9 12 11.72z" />
  </svg>
);

// ── App ────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height =
        Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [input]);

  const sendMessage = async (override?: string) => {
    const text = (override ?? input).trim();
    if (!text || isLoading) return;

    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setInput('');
    setIsLoading(true);
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

    try {
      const sources = await streamChat(text, messages, token => {
        setMessages(prev => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last.role === 'assistant') {
            updated[updated.length - 1] = { ...last, content: last.content + token };
          }
          return updated;
        });
      });

      if (sources.length > 0) {
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { ...updated[updated.length - 1], sources };
          return updated;
        });
      }
    } catch {
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          content: 'Sorry, something went wrong. Please try again.',
        };
        return updated;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="app">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="sidebar__brand">
          <div className="sidebar__logo"><SchoolIcon /></div>
          <div>
            <div className="sidebar__school-name">Warwick Prep</div>
            <div className="sidebar__tagline">School Assistant</div>
          </div>
        </div>

        <div className="sidebar__divider" />
        <div className="sidebar__section-title">Quick questions</div>

        <div className="sidebar__suggestions">
          {SUGGESTED.map((s, i) => (
            <button key={i} className="sidebar__suggestion" onClick={() => sendMessage(s.text)}>
              <span className="sidebar__suggestion-icon">{s.icon}</span>
              <span>{s.text}</span>
            </button>
          ))}
        </div>

        <div className="sidebar__footer">
          <a href="https://www.warwickprep.com" target="_blank" rel="noopener noreferrer" className="sidebar__link">
            warwickprep.com ↗
          </a>
        </div>
      </aside>

      {/* ── Main ── */}
      <div className="main">
        {/* Mobile header */}
        <header className="mobile-header">
          <div className="mobile-header__brand">
            <SchoolIcon />
            <span>Warwick Prep Assistant</span>
          </div>
        </header>

        {/* Chat area */}
        <div className="chat-area">
          {messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome__icon"><SchoolIcon /></div>
              <h1 className="welcome__title">How can I help you today?</h1>
              <p className="welcome__subtitle">
                Ask me anything about Warwick Prep School — admissions, curriculum, events, and more.
              </p>
              <div className="welcome__chips">
                {SUGGESTED.map((s, i) => (
                  <button key={i} className="welcome__chip" onClick={() => sendMessage(s.text)}>
                    <span>{s.icon}</span>
                    <span>{s.text}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages">
              {messages.map((msg, i) => (
                <div key={i} className={`message message--${msg.role}`}>
                  {msg.role === 'assistant' && (
                    <div className="message__avatar message__avatar--assistant">
                      <SchoolIcon />
                    </div>
                  )}
                  <div className="message__body">
                    <div className="message__bubble">
                      {msg.role === 'assistant' ? (
                        msg.content === '' && isLoading && i === messages.length - 1 ? (
                          <div className="typing"><span /><span /><span /></div>
                        ) : (
                          <ReactMarkdown>{msg.content}</ReactMarkdown>
                        )
                      ) : (
                        msg.content
                      )}
                    </div>
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="message__sources">
                        {msg.sources.map((s, j) => (
                          <a
                            key={j}
                            href={s.startsWith('http') ? s : `https://${s}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="source-chip"
                          >
                            🔗 {s}
                          </a>
                        ))}
                      </div>
                    )}
                  </div>
                  {msg.role === 'user' && (
                    <div className="message__avatar message__avatar--user">U</div>
                  )}
                </div>
              ))}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        {/* Input */}
        <div className="input-wrapper">
          <div className="input-bar">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about Warwick Prep School…"
              disabled={isLoading}
              rows={1}
              className="input-bar__textarea"
            />
            <button
              onClick={() => sendMessage()}
              disabled={isLoading || !input.trim()}
              className="input-bar__send"
              aria-label="Send message"
            >
              <SendIcon />
            </button>
          </div>
          <p className="input-bar__hint">Enter to send · Shift+Enter for new line</p>
        </div>
      </div>
    </div>
  );
}


interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
}

const API_BASE = process.env.REACT_APP_API_URL || '';

async function streamChat(
  message: string,
  history: Message[],
  onToken: (token: string) => void
): Promise<string[]> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      history: history.map(({ role, content }) => ({ role, content })),
    }),
  });

  if (!response.ok) throw new Error('Chat request failed');

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let full = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    full += chunk;
    onToken(chunk);
  }

  // Extract sources if present
  const sourceMatch = full.match(/__sources__:(.+)$/);
  return sourceMatch ? sourceMatch[1].split(',') : [];
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || isLoading) return;

    const userMsg: Message = { role: 'user', content: text };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    const assistantMsg: Message = { role: 'assistant', content: '' };
    setMessages(prev => [...prev, assistantMsg]);

    try {
      const sources = await streamChat(text, messages, (token) => {
        if (!token.startsWith('__sources__:')) {
          setMessages(prev => {
            const updated = [...prev];
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              content: updated[updated.length - 1].content + token,
            };
            return updated;
          });
        }
      });

      if (sources.length > 0) {
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { ...updated[updated.length - 1], sources };
          return updated;
        });
      }
    } catch {
      setMessages(prev => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          content: 'Sorry, something went wrong. Please try again.',
        };
        return updated;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="app">
      <header className="header">
        <img src="/logo.png" alt="Warwick Prep School" className="logo" />
        <h1>Warwick Prep School Assistant</h1>
      </header>

      <main className="chat-window">
        {messages.length === 0 && (
          <div className="empty-state">
            <p>Ask me anything about Warwick Prep School — admissions, curriculum, events, and more.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message message--${msg.role}`}>
            <div className="message__bubble">{msg.content}</div>
            {msg.sources && msg.sources.length > 0 && (
              <div className="message__sources">
                Sources: {msg.sources.map((s, j) => <span key={j} className="source-tag">{s}</span>)}
              </div>
            )}
          </div>
        ))}
        {isLoading && messages[messages.length - 1]?.content === '' && (
          <div className="message message--assistant">
            <div className="message__bubble typing">
              <span /><span /><span />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      <footer className="input-area">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about Warwick Prep School…"
          disabled={isLoading}
          rows={2}
        />
        <button onClick={sendMessage} disabled={isLoading || !input.trim()}>
          Send
        </button>
      </footer>
    </div>
  );
}
