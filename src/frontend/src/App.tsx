import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: { url: string; title: string }[];
}

interface PromptDef {
  icon: string;
  text: string;
  category: string;
}

const ALL_PROMPTS: PromptDef[] = [
  { icon: '🎓', text: 'What are the admissions requirements?', category: 'Admissions' },
  { icon: '📅', text: 'When is the next open day?', category: 'Events' },
  { icon: '💷', text: 'What are the school fees?', category: 'Admissions' },
  { icon: '📚', text: 'Tell me about the curriculum', category: 'Academic' },
  { icon: '⭐', text: 'What clubs and activities are available?', category: 'Activities' },
  { icon: '🍽️', text: "What's on the lunch menu this week?", category: 'Daily Life' },
  { icon: '🏫', text: 'Tell me about the school facilities', category: 'School' },
  { icon: '🗓️', text: 'What are the term dates?', category: 'Events' },
  { icon: '🚌', text: 'Is there a school bus service?', category: 'Transport' },
  { icon: '📞', text: 'How do I contact the admissions team?', category: 'Admissions' },
  { icon: '🎽', text: 'What sports are offered?', category: 'Activities' },
  { icon: '🎭', text: 'Tell me about the arts programme', category: 'Activities' },
];

// ── Prompt usage tracking via localStorage ────────────────────────────────
const COUNTS_KEY = 'tpet_prompt_counts';

function getCounts(): Record<string, number> {
  try { return JSON.parse(localStorage.getItem(COUNTS_KEY) || '{}'); }
  catch { return {}; }
}

function recordClick(text: string): Record<string, number> {
  const counts = getCounts();
  counts[text] = (counts[text] || 0) + 1;
  try { localStorage.setItem(COUNTS_KEY, JSON.stringify(counts)); } catch {}
  return counts;
}

function sortedPrompts(counts: Record<string, number>): PromptDef[] {
  return [...ALL_PROMPTS].sort((a, b) => (counts[b.text] || 0) - (counts[a.text] || 0));
}

const API_BASE = import.meta.env.VITE_API_URL || '';

/** Derive a human-readable label from a URL path as a fallback when no title is stored. */
function urlToLabel(url: string): string {
  try {
    const parsed = new URL(url.startsWith('http') ? url : `https://${url}`);
    const { pathname } = parsed;
    const segments = pathname.split('/').filter(Boolean);
    if (segments.length === 0) return 'Ask Warwick';
    const last = segments[segments.length - 1];
    // Hex hash filenames (e.g. ED13E431B4F0ABE4E536FE4074E3ECD2.pdf)
    if (/^[0-9a-f]{16,}\.(pdf|html)$/i.test(last)) return 'School Document';
    return last
      .replace(/\.[^.]+$/, '')   // strip file extension
      .replace(/[-_]/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase());
  } catch {
    return 'Source';
  }
}

async function streamChat(
  message: string,
  history: Message[],
  onToken: (token: string) => void
): Promise<{ url: string; title: string }[]> {
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
  if (!sourceMatch) return [];
  return sourceMatch[1]
    .split(',')
    .map(s => s.trim())
    .filter(Boolean)
    .map(part => {
      const sepIdx = part.indexOf(':::');
      const url = sepIdx >= 0 ? part.slice(0, sepIdx).trim() : part;
      const rawTitle = sepIdx >= 0 ? part.slice(sepIdx + 3).trim() : '';
      return { url, title: rawTitle || urlToLabel(url) };
    });
}

// ── Icons ────────────────────────────────────────────────────────────────────
const SendIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
);

const SchoolIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 3L1 9l4 2.18V16c0 1.1.9 2 2 2h10c1.1 0 2-.9 2-2v-4.82L21 9l-9-6zm5 13H7v-3.99l5 2.72 5-2.72V16zm-5-4.28L3.53 9 12 4.28 20.47 9 12 11.72z" />
  </svg>
);

const SunIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="5"/>
    <line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/>
    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
    <line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/>
    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
  </svg>
);

const MoonIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
  </svg>
);

const TrashIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6"/>
    <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
    <path d="M10 11v6"/><path d="M14 11v6"/>
    <path d="M9 6V4h6v2"/>
  </svg>
);

const SparkleIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
    <path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17l-6.2 4.3 2.4-7.4L2 9.4h7.6z"/>
  </svg>
);

const CloseIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
  </svg>
);

// ── Responsive table ──────────────────────────────────────────────────────────
function ResponsiveTable({ children }: { children?: React.ReactNode; [k: string]: unknown }) {
  const headers: string[] = [];
  React.Children.forEach(children, (section) => {
    if (!React.isValidElement(section)) return;
    const el = section as React.ReactElement<any>;
    if (el.type !== 'thead') return;
    React.Children.forEach(el.props.children, (row) => {
      if (!React.isValidElement(row)) return;
      const tr = row as React.ReactElement<any>;
      React.Children.forEach(tr.props.children, (cell) => {
        if (!React.isValidElement(cell)) return;
        const th = cell as React.ReactElement<any>;
        headers.push(React.Children.toArray(th.props.children).map(c => typeof c === 'string' ? c : '').join(''));
      });
    });
  });
  const processedChildren = React.Children.map(children, (section) => {
    if (!React.isValidElement(section)) return section;
    const el = section as React.ReactElement<any>;
    if (el.type !== 'tbody') return el;
    const rows = React.Children.map(el.props.children, (row) => {
      if (!React.isValidElement(row)) return row;
      const tr = row as React.ReactElement<any>;
      const cells = React.Children.map(tr.props.children, (cell, i) => {
        if (!React.isValidElement(cell)) return cell;
        return React.cloneElement(cell as React.ReactElement<any>, { 'data-label': headers[i] ?? '' });
      });
      return React.cloneElement(tr, {}, cells);
    });
    return React.cloneElement(el, {}, rows);
  });
  return <div className="responsive-table-wrapper"><table>{processedChildren}</table></div>;
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showDrawer, setShowDrawer] = useState(false);
  const [counts, setCounts] = useState<Record<string, number>>(getCounts);
  const [darkMode, setDarkMode] = useState<boolean>(() => {
    try {
      const s = localStorage.getItem('theme');
      return s ? s === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
    } catch { return false; }
  });
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light');
    try { localStorage.setItem('theme', darkMode ? 'dark' : 'light'); } catch {}
  }, [darkMode]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 100) + 'px';
    }
  }, [input]);

  const sendMessage = async (override?: string) => {
    const text = (override ?? input).trim();
    if (!text || isLoading) return;
    setInput('');
    setShowDrawer(false);
    setMessages(prev => [...prev, { role: 'user', content: text }, { role: 'assistant', content: '' }]);
    setIsLoading(true);
    try {
      const sources = await streamChat(text, messages, token => {
        setMessages(prev => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          if (last?.role === 'assistant') updated[updated.length - 1] = { ...last, content: last.content + token };
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
        updated[updated.length - 1] = { ...updated[updated.length - 1], content: 'Sorry, something went wrong. Please try again.' };
        return updated;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handlePromptClick = (text: string) => {
    const newCounts = recordClick(text);
    setCounts(newCounts);
    sendMessage(text);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  const sorted = sortedPrompts(counts);
  // Prompts clicked at least once, sorted by frequency — shown in "Popular" section
  const popular = sorted.filter(p => counts[p.text] > 0).slice(0, 4);
  // Remaining prompts for the "More" list
  const remaining = sorted.filter(p => !popular.includes(p));
  // Chip strip: top 6 sorted prompts shown inline when in conversation
  const chipPrompts = sorted.slice(0, 6);

  return (
    <div className="app">

      {/* ── Header ── */}
      <header className="header">
        <div className="header__brand">
          <div className="header__avatar"><SchoolIcon /></div>
          <div className="header__info">
            <span className="header__name">Ask Warwick</span>
            <span className="header__status">
              <span className="header__status-dot" />
              Warwick Prep School
            </span>
          </div>
        </div>
        <div className="header__actions">
          {messages.length > 0 && (
            <button className="header__btn" onClick={() => setMessages([])} aria-label="Clear chat">
              <TrashIcon />
            </button>
          )}
          <button className="header__btn" onClick={() => setDarkMode(d => !d)} aria-label="Toggle dark mode">
            {darkMode ? <SunIcon /> : <MoonIcon />}
          </button>
        </div>
      </header>

      {/* ── Chat / Welcome ── */}
      <div className="chat-area">
        {messages.length === 0 ? (
          <div className="welcome">
            <div className="welcome__logo"><SchoolIcon /></div>
            <h1 className="welcome__title">How can I help?</h1>
            <p className="welcome__sub">
              Ask me anything about Warwick Prep School — admissions, curriculum, events and more.
            </p>

            {popular.length > 0 && (
              <>
                <span className="welcome__section-label">🔥 Popular with you</span>
                <div className="welcome__popular">
                  {popular.map((p, i) => (
                    <button key={i} className="welcome__popular-card welcome__popular-card--hot" onClick={() => handlePromptClick(p.text)}>
                      <span className="welcome__card-badge">Popular</span>
                      <span className="welcome__card-icon">{p.icon}</span>
                      <span className="welcome__card-text">{p.text}</span>
                    </button>
                  ))}
                </div>
              </>
            )}

            <span className="welcome__section-label">
              {popular.length > 0 ? 'More questions' : 'Try asking'}
            </span>

            {popular.length === 0 ? (
              <div className="welcome__popular">
                {sorted.slice(0, 6).map((p, i) => (
                  <button key={i} className="welcome__popular-card" onClick={() => handlePromptClick(p.text)}>
                    <span className="welcome__card-icon">{p.icon}</span>
                    <span className="welcome__card-text">{p.text}</span>
                  </button>
                ))}
              </div>
            ) : (
              <div className="welcome__more">
                {remaining.slice(0, 5).map((p, i) => (
                  <button key={i} className="welcome__more-item" onClick={() => handlePromptClick(p.text)}>
                    <span>{p.icon}</span>
                    <span>{p.text}</span>
                  </button>
                ))}
                <button className="welcome__more-item welcome__more-item--all" onClick={() => setShowDrawer(true)}>
                  <SparkleIcon />
                  <span>View all questions…</span>
                </button>
              </div>
            )}
          </div>
        ) : (
          <div className="messages">
            {messages.map((msg, i) => (
              <div key={i} className={`message message--${msg.role}`}>
                {msg.role === 'assistant' && (
                  <div className="message__avatar message__avatar--assistant"><SchoolIcon /></div>
                )}
                <div className="message__body">
                  <div className="message__bubble">
                    {msg.role === 'assistant' ? (
                      msg.content === '' && isLoading && i === messages.length - 1 ? (
                        <div className="typing"><span /><span /><span /></div>
                      ) : (
                        <ReactMarkdown remarkPlugins={[remarkGfm]} components={{ table: ResponsiveTable as any }}>
                          {msg.content}
                        </ReactMarkdown>
                      )
                    ) : msg.content}
                  </div>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="message__sources">
                      {msg.sources.map((s, j) => (
                        <a key={j} href={s.url.startsWith('http') ? s.url : `https://${s.url}`}
                          target="_blank" rel="noopener noreferrer" className="source-chip">
                          🔗 {s.title}
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

      {/* ── Horizontal chip strip — visible during conversation ── */}
      {messages.length > 0 && (
        <div className="chips-strip">
          <button className="chip chip--expand" onClick={() => setShowDrawer(true)}>
            <SparkleIcon /> All
          </button>
          {chipPrompts.map((p, i) => (
            <button key={i}
              className={`chip${counts[p.text] > 0 ? ' chip--used' : ''}`}
              onClick={() => handlePromptClick(p.text)}>
              {p.icon} {p.text.length > 26 ? p.text.slice(0, 24) + '…' : p.text}
            </button>
          ))}
        </div>
      )}

      {/* ── Input ── */}
      <div className="input-area">
        <div className="input-bar">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask Warwick anything…"
            disabled={isLoading}
            rows={1}
            className="input-bar__textarea"
          />
          <button onClick={() => sendMessage()} disabled={isLoading || !input.trim()}
            className="input-bar__send" aria-label="Send">
            <SendIcon />
          </button>
        </div>
        <p className="input-area__hint">Enter to send · Shift+Enter for new line</p>
      </div>

      {/* ── Prompts drawer (bottom sheet) ── */}
      {showDrawer && (
        <>
          <div className="drawer-overlay" onClick={() => setShowDrawer(false)} />
          <div className="drawer" role="dialog" aria-modal="true" aria-label="Quick questions">
            <div className="drawer__handle" />
            <div className="drawer__header">
              <span className="drawer__title">Quick questions</span>
              <button className="drawer__close" onClick={() => setShowDrawer(false)} aria-label="Close">
                <CloseIcon />
              </button>
            </div>
            <div className="drawer__body">
              {popular.length > 0 && (
                <>
                  <p className="drawer__section-label">🔥 Your most used</p>
                  <div className="drawer__popular-grid">
                    {popular.map((p, i) => (
                      <button key={i} className="drawer__popular-card" onClick={() => handlePromptClick(p.text)}>
                        <span className="drawer__popular-icon">{p.icon}</span>
                        <span className="drawer__popular-text">{p.text}</span>
                        <span className="drawer__popular-count">Used {counts[p.text]}×</span>
                      </button>
                    ))}
                  </div>
                </>
              )}
              <p className="drawer__section-label">All questions</p>
              <div className="drawer__list">
                {sorted.map((p, i) => (
                  <button key={i} className="drawer__item" onClick={() => handlePromptClick(p.text)}>
                    <span className="drawer__item-icon">{p.icon}</span>
                    <span className="drawer__item-text">{p.text}</span>
                    {counts[p.text] > 0 && (
                      <span className="drawer__item-count">{counts[p.text]}×</span>
                    )}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </>
      )}

    </div>
  );
}

