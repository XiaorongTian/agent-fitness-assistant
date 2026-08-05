import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { FormEvent, KeyboardEvent } from 'react';
import {
  Activity,
  Ban,
  Bot,
  Check,
  ChevronLeft,
  ChevronRight,
  CirclePlus,
  Dumbbell,
  ImagePlus,
  Leaf,
  MessageCircle,
  Play,
  SendHorizontal,
  Sparkles,
  ThumbsUp,
  Utensils,
  X,
} from 'lucide-react';

type Role = 'user' | 'assistant';

type Message = {
  id: string;
  role: Role;
  content: string;
  toolNames?: string[];
  imageUrl?: string;
};

type ImageAttachment = {
  file: File;
  previewUrl: string;
};

type ChatSession = {
  id: string;
  title: string;
  messages: Message[];
};

type ChatApiResponse = {
  session_id: string;
  result: {
    reply: string;
  };
  tool_calls?: Array<{ name: string; output?: unknown }>;
};

type FoodDraft = {
  draft_id: string;
  meal_type: string;
  summary: string;
  foods: Array<{ name: string; amount_description: string }>;
  assumptions: string[];
};

type ExerciseTask = {
  task_id: string;
  title: string;
  plan: string;
  activity: string;
  planned_minutes: number;
  intensity: 'low' | 'moderate';
  status: 'pending' | 'in_progress';
};

type ExerciseStage = 'ready' | 'in_progress' | 'skipped_reason' | 'disliked_reason';

const USER_ID_STORAGE_KEY = 'fitness-assistant-user-id';
const SESSIONS_STORAGE_KEY = 'fitness-assistant-sessions';
const ACTIVE_SESSION_STORAGE_KEY = 'fitness-assistant-active-session';
const MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024;
const FOOD_IMAGE_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);

const WELCOME_MESSAGE: Message = {
  id: 'welcome',
  role: 'assistant',
  content: '你好，我是你的健康助手。告诉我今天的饮食情况或可运动时间，我会帮你做一个更容易坚持的安排。',
};

const STARTER_QUESTIONS = [
  '帮我安排今天 30 分钟的运动任务',
  '我久坐了一整天，下班后适合做什么？',
  '我今天蛋白质吃得够吗？',
];

function createId() {
  return crypto.randomUUID();
}

function createSession(): ChatSession {
  return {
    id: createId(),
    title: '新的健康对话',
    messages: [WELCOME_MESSAGE],
  };
}

function loadSessions(): ChatSession[] {
  try {
    const value = localStorage.getItem(SESSIONS_STORAGE_KEY);
    const sessions = value ? (JSON.parse(value) as ChatSession[]) : [];
    return sessions.length ? sessions : [createSession()];
  } catch {
    return [createSession()];
  }
}

function sessionTitle(message: string) {
  return message.length > 16 ? `${message.slice(0, 16)}…` : message;
}

type TypewriterTextProps = {
  content: string;
  isTyping: boolean;
  onComplete: () => void;
  onProgress: () => void;
};

function TypewriterText({ content, isTyping, onComplete, onProgress }: TypewriterTextProps) {
  const characters = useMemo(() => Array.from(content), [content]);
  const [visibleLength, setVisibleLength] = useState(() => (isTyping ? 0 : characters.length));

  useEffect(() => {
    if (!isTyping) {
      setVisibleLength(characters.length);
      return;
    }

    setVisibleLength(0);
    let nextLength = 0;
    const step = characters.length > 500 ? 3 : characters.length > 250 ? 2 : 1;
    const timer = window.setInterval(() => {
      nextLength = Math.min(nextLength + step, characters.length);
      setVisibleLength(nextLength);
      onProgress();
      if (nextLength === characters.length) {
        window.clearInterval(timer);
        onComplete();
      }
    }, 18);

    return () => window.clearInterval(timer);
  }, [characters.length, isTyping, onComplete, onProgress]);

  return (
    <p aria-live={isTyping ? 'polite' : undefined}>
      {characters.slice(0, visibleLength).join('')}
      {isTyping && <span aria-hidden="true" className="typing-cursor" />}
    </p>
  );
}

function App() {
  const [allSessions, setAllSessions] = useState<ChatSession[]>(() => loadSessions());
  const [activeSessionId, setActiveSessionId] = useState(() => localStorage.getItem(ACTIVE_SESSION_STORAGE_KEY) || '');
  const [input, setInput] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [foodDraft, setFoodDraft] = useState<FoodDraft | null>(null);
  const [isConfirmingDraft, setIsConfirmingDraft] = useState(false);
  const [draftConfirmationMessage, setDraftConfirmationMessage] = useState<string | null>(null);
  const [exerciseTask, setExerciseTask] = useState<ExerciseTask | null>(null);
  const [exerciseStage, setExerciseStage] = useState<ExerciseStage>('ready');
  const [exerciseReason, setExerciseReason] = useState('');
  const [isUpdatingExercise, setIsUpdatingExercise] = useState(false);
  const [exerciseError, setExerciseError] = useState<string | null>(null);
  const [typingMessageId, setTypingMessageId] = useState<string | null>(null);
  const [imageAttachment, setImageAttachment] = useState<ImageAttachment | null>(null);
  const [isUploadingImage, setIsUploadingImage] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, []);

  const activeSession = useMemo(() => {
    return allSessions.find((session) => session.id === activeSessionId) || allSessions[0];
  }, [activeSessionId, allSessions]);

  useEffect(() => {
    if (activeSession && activeSession.id !== activeSessionId) {
      setActiveSessionId(activeSession.id);
    }
  }, [activeSession, activeSessionId]);

  useEffect(() => {
    localStorage.setItem(SESSIONS_STORAGE_KEY, JSON.stringify(allSessions));
  }, [allSessions]);

  useEffect(() => {
    if (activeSessionId) localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, activeSessionId);
  }, [activeSessionId]);

  useEffect(() => {
    scrollToBottom();
  }, [activeSession?.messages, isSending, scrollToBottom]);

  const updateSession = (sessionId: string, updater: (session: ChatSession) => ChatSession) => {
    setAllSessions((current) => current.map((session) => (session.id === sessionId ? updater(session) : session)));
  };

  const clearImageAttachment = () => {
    if (imageAttachment) URL.revokeObjectURL(imageAttachment.previewUrl);
    setImageAttachment(null);
    if (imageInputRef.current) imageInputRef.current.value = '';
  };

  const selectImage = (file: File | undefined) => {
    if (!file) return;
    if (!FOOD_IMAGE_TYPES.has(file.type)) {
      setError('请选择 JPG、PNG 或 WebP 格式的食物图片。');
      return;
    }
    if (file.size > MAX_IMAGE_SIZE_BYTES) {
      setError('图片不能超过 10MB。');
      return;
    }
    if (imageAttachment) URL.revokeObjectURL(imageAttachment.previewUrl);
    setImageAttachment({ file, previewUrl: URL.createObjectURL(file) });
    setError(null);
  };

  const uploadFoodImage = async (file: File) => {
    const extension = file.name.split('.').pop()?.toLowerCase() || 'jpg';
    const objectKey = `food-images/${new Date().toISOString().slice(0, 10)}/${createId()}.${extension}`;
    const presignResponse = await fetch(`/api/oss/presign?filename=${encodeURIComponent(objectKey)}`);
    const presign = (await presignResponse.json()) as {
      uploadUrl?: string;
      accessUrl?: string;
      contentType?: string;
      detail?: string;
    };
    if (!presignResponse.ok || !presign.uploadUrl || !presign.accessUrl || !presign.contentType) {
      throw new Error(presign.detail || '无法获取图片上传凭证。');
    }
    const uploadResponse = await fetch(presign.uploadUrl, {
      method: 'PUT',
      headers: { 'Content-Type': presign.contentType },
      body: file,
    });
    if (!uploadResponse.ok) throw new Error('图片上传到 OSS 失败，请稍后重试。');
    return presign.accessUrl;
  };

  const startNewChat = () => {
    const next = createSession();
    setAllSessions((current) => [next, ...current]);
    setActiveSessionId(next.id);
    setError(null);
  };

  const sendMessage = async (message = input) => {
    const content = message.trim();
    const attachment = imageAttachment;
    if ((!content && !attachment) || !activeSession || isSending) return;

    const currentSessionId = activeSession.id;
    setInput('');
    setError(null);
    setIsSending(true);

    try {
      const userId = localStorage.getItem(USER_ID_STORAGE_KEY) || 'local-health-user';
      localStorage.setItem(USER_ID_STORAGE_KEY, userId);
      let imageUrl: string | undefined;
      if (attachment) {
        setIsUploadingImage(true);
        imageUrl = await uploadFoodImage(attachment.file);
        clearImageAttachment();
      }
      const displayContent = content || '已上传食物图片，请帮我识别并记录。';
      const userMessage: Message = { id: createId(), role: 'user', content: displayContent, imageUrl };
      updateSession(currentSessionId, (session) => ({
        ...session,
        title: session.messages.length <= 1 ? sessionTitle(displayContent) : session.title,
        messages: [...session.messages, userMessage],
      }));
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          session_id: currentSessionId,
          message: content,
          image_url: imageUrl,
        }),
      });
      const payload = (await response.json()) as ChatApiResponse | { detail?: string };
      if (!response.ok || !('result' in payload)) {
        throw new Error('detail' in payload ? payload.detail || '健康助手暂时不可用。' : '健康助手暂时不可用。');
      }
      const toolNames = [...new Set((payload.tool_calls || []).map((tool) => tool.name))];
      const draftOutput = payload.tool_calls?.find((tool) => tool.name === 'create_food_record_draft')?.output;
      if (
        draftOutput
        && typeof draftOutput === 'object'
        && 'draft_id' in draftOutput
        && 'foods' in draftOutput
      ) {
        setFoodDraft(draftOutput as FoodDraft);
        setDraftConfirmationMessage(null);
      }
      const taskOutput = payload.tool_calls?.find((tool) => tool.name === 'create_user_requested_exercise_task')?.output;
      if (
        taskOutput
        && typeof taskOutput === 'object'
        && 'task_id' in taskOutput
        && 'plan' in taskOutput
      ) {
        setExerciseTask(taskOutput as ExerciseTask);
        setExerciseStage('ready');
        setExerciseReason('');
        setExerciseError(null);
      }
      const assistantMessageId = createId();
      updateSession(currentSessionId, (session) => ({
        ...session,
        messages: [
          ...session.messages,
          { id: assistantMessageId, role: 'assistant', content: payload.result.reply, toolNames },
        ],
      }));
      setTypingMessageId(assistantMessageId);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : '发送失败，请稍后再试。');
    } finally {
      setIsUploadingImage(false);
      setIsSending(false);
    }
  };

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    void sendMessage();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  };

  const closeFoodDraft = () => {
    setFoodDraft(null);
    setDraftConfirmationMessage(null);
  };

  const confirmFoodDraft = async () => {
    if (!foodDraft || isConfirmingDraft) return;
    setIsConfirmingDraft(true);
    setDraftConfirmationMessage(null);
    try {
      const userId = localStorage.getItem(USER_ID_STORAGE_KEY) || 'local-health-user';
      const response = await fetch(`/api/diet/drafts/${foodDraft.draft_id}/confirm`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, confirmed: true }),
      });
      const payload = (await response.json()) as { detail?: string };
      if (!response.ok) throw new Error(payload.detail || '饮食记录确认失败。');
      closeFoodDraft();
    } catch (requestError) {
      setDraftConfirmationMessage(requestError instanceof Error ? requestError.message : '饮食记录确认失败。');
    } finally {
      setIsConfirmingDraft(false);
    }
  };

  const closeExerciseTask = () => {
    setExerciseTask(null);
    setExerciseReason('');
    setExerciseError(null);
    setExerciseStage('ready');
  };

  const updateExerciseTask = async (path: string, body: Record<string, unknown>) => {
    if (!exerciseTask || isUpdatingExercise) return;
    setIsUpdatingExercise(true);
    setExerciseError(null);
    try {
      const userId = localStorage.getItem(USER_ID_STORAGE_KEY) || 'local-health-user';
      const response = await fetch(`/api/exercise/tasks/${exerciseTask.task_id}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, ...body }),
      });
      const payload = (await response.json()) as ExerciseTask | { detail?: string };
      if (!response.ok || !('task_id' in payload)) {
        throw new Error('detail' in payload ? payload.detail || '运动任务更新失败。' : '运动任务更新失败。');
      }
      if (payload.status === 'in_progress') {
        setExerciseTask(payload);
        setExerciseStage('in_progress');
        return;
      }
      closeExerciseTask();
    } catch (requestError) {
      setExerciseError(requestError instanceof Error ? requestError.message : '运动任务更新失败。');
    } finally {
      setIsUpdatingExercise(false);
    }
  };

  const submitExerciseReason = (status: 'skipped' | 'disliked') => {
    if (!exerciseReason.trim()) {
      setExerciseError('请填写原因后再提交。');
      return;
    }
    void updateExerciseTask('/feedback', { status, feedback_note: exerciseReason.trim() });
  };

  if (!activeSession) return null;

  return (
    <main className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? '' : 'sidebar--collapsed'}`}>
        <div className="brand">
          <div className="brand__mark"><Leaf size={22} /></div>
          <div className="brand__copy"><strong>轻盈计划</strong><span>个人健康助手</span></div>
        </div>

        <button className="new-chat" onClick={startNewChat} type="button">
          <CirclePlus size={18} /> 新建对话
        </button>

        <nav className="primary-nav" aria-label="健康助手功能">
          <span className="nav-item nav-item--active"><MessageCircle size={18} /> 健康对话</span>
          <span className="nav-item"><Utensils size={18} /> 饮食记录</span>
          <span className="nav-item"><Dumbbell size={18} /> 运动任务</span>
        </nav>

        <div className="conversation-list">
          <p>最近对话</p>
          {allSessions.map((session) => (
            <button
              className={`conversation-item ${session.id === activeSession.id ? 'conversation-item--active' : ''}`}
              key={session.id}
              onClick={() => setActiveSessionId(session.id)}
              type="button"
            >
              {session.title}
            </button>
          ))}
        </div>

        <div className="sidebar-tip">
          <Activity size={17} />
          <span>从一顿饭、一段运动开始。</span>
        </div>
      </aside>

      <section className="chat-area">
        <header className="topbar">
          <button aria-label="切换侧边栏" className="icon-button" onClick={() => setSidebarOpen((open) => !open)} type="button">
            {sidebarOpen ? <ChevronLeft size={20} /> : <ChevronRight size={20} />}
          </button>
          <div>
            <h1>健康对话</h1>
            <p><span className="online-dot" /> 健康助手在线</p>
          </div>
        </header>

        <div className="chat-scroll">
          <div className="chat-content">
            {activeSession.messages.map((message) => (
              <article className={`message message--${message.role}`} key={message.id}>
                {message.role === 'assistant' && <div className="avatar avatar--assistant"><Bot size={18} /></div>}
                <div className="message__body">
                  {message.role === 'assistant' ? (
                    <TypewriterText
                      content={message.content}
                      isTyping={message.id === typingMessageId}
                      onComplete={() => setTypingMessageId((current) => (current === message.id ? null : current))}
                      onProgress={scrollToBottom}
                    />
                  ) : <p>{message.content}</p>}
                  {message.role === 'user' && message.imageUrl && (
                    <img alt="用户上传的食物图片" className="message-image" src={message.imageUrl} />
                  )}
                  {message.toolNames && message.toolNames.length > 0 && message.id !== typingMessageId && (
                    <div className="tool-tags" aria-label="本轮已调用能力">
                      {message.toolNames.map((name) => <span key={name}>已使用：{name}</span>)}
                    </div>
                  )}
                </div>
              </article>
            ))}

            {activeSession.messages.length === 1 && (
              <div className="starter-grid">
                {STARTER_QUESTIONS.map((question) => (
                  <button key={question} onClick={() => void sendMessage(question)} type="button">{question}</button>
                ))}
              </div>
            )}

            {isSending && (
              <article className="message message--assistant">
                <div className="avatar avatar--assistant"><Bot size={18} /></div>
                <div className="message__body message__body--loading"><i /><i /><i /></div>
              </article>
            )}
            <div ref={bottomRef} />
          </div>
        </div>

        <div className="composer-wrap">
          {error && <div className="error-banner">{error}</div>}
          <form className="composer" onSubmit={handleSubmit}>
            <input
              accept="image/jpeg,image/png,image/webp"
              aria-label="上传食物图片"
              className="image-input"
              disabled={isSending}
              onChange={(event) => selectImage(event.target.files?.[0])}
              ref={imageInputRef}
              type="file"
            />
            <button
              aria-label="上传食物图片"
              className="image-upload-button"
              disabled={isSending}
              onClick={() => imageInputRef.current?.click()}
              type="button"
            >
              <ImagePlus size={19} />
            </button>
            <textarea
              aria-label="输入健康问题"
              disabled={isSending}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="描述饮食或上传食物图片…"
              rows={1}
              value={input}
            />
            <button aria-label="发送" className="send-button" disabled={(!input.trim() && !imageAttachment) || isSending} type="submit">
              <SendHorizontal size={19} />
            </button>
          </form>
          {imageAttachment && (
            <div className="image-preview">
              <img alt="待上传的食物图片" src={imageAttachment.previewUrl} />
              <span>{imageAttachment.file.name}</span>
              <button aria-label="移除图片" onClick={clearImageAttachment} type="button"><X size={15} /></button>
            </div>
          )}
          {isUploadingImage && <p className="image-uploading">正在上传图片…</p>}
          <p className="composer-hint"><Sparkles size={14} /> 健康建议仅供日常参考；不适或疼痛请及时咨询专业人士。</p>
        </div>
      </section>

      {foodDraft && (
        <div className="modal-backdrop" role="presentation">
          <section aria-labelledby="food-draft-title" aria-modal="true" className="food-draft-modal" role="dialog">
            <button aria-label="稍后确认" className="modal-close" onClick={closeFoodDraft} type="button"><X size={18} /></button>
            <div className="modal-icon"><Utensils size={21} /></div>
            <p className="modal-eyebrow">待确认饮食记录</p>
            <h2 id="food-draft-title">这是你刚才吃的吗？</h2>
            <p className="modal-summary">{foodDraft.summary}</p>
            <ul className="food-list">
              {foodDraft.foods.map((food) => <li key={`${food.name}-${food.amount_description}`}><strong>{food.name}</strong><span>{food.amount_description}</span></li>)}
            </ul>
            {foodDraft.assumptions.length > 0 && <p className="food-assumption">估算说明：{foodDraft.assumptions.join('；')}</p>}
            {draftConfirmationMessage && <p className="draft-feedback">{draftConfirmationMessage}</p>}
            <div className="modal-actions">
              <button className="secondary-button" onClick={closeFoodDraft} type="button">稍后确认</button>
              <button className="confirm-button" disabled={isConfirmingDraft} onClick={() => void confirmFoodDraft()} type="button">
                <Check size={17} /> {isConfirmingDraft ? '确认中…' : '确认并入账'}
              </button>
            </div>
          </section>
        </div>
      )}

      {exerciseTask && (
        <div className="modal-backdrop" role="presentation">
          <section aria-labelledby="exercise-task-title" aria-modal="true" className="food-draft-modal exercise-task-modal" role="dialog">
            <button aria-label="稍后处理运动任务" className="modal-close" onClick={closeExerciseTask} type="button"><X size={18} /></button>
            <div className="modal-icon"><Dumbbell size={21} /></div>
            <p className="modal-eyebrow">今日运动任务</p>
            <h2 id="exercise-task-title">{exerciseTask.title}</h2>
            <p className="exercise-meta">{exerciseTask.planned_minutes} 分钟 · {exerciseTask.intensity === 'low' ? '低强度' : '适中强度'} · {exerciseTask.activity}</p>
            <p className="exercise-plan">{exerciseTask.plan}</p>
            {exerciseError && <p className="exercise-error">{exerciseError}</p>}

            {exerciseStage === 'ready' && (
              <div className="modal-actions">
                <button className="secondary-button" disabled={isUpdatingExercise} onClick={() => { setExerciseStage('skipped_reason'); setExerciseError(null); }} type="button"><Ban size={16} /> 暂不执行</button>
                <button className="confirm-button" disabled={isUpdatingExercise} onClick={() => void updateExerciseTask('/start', {})} type="button"><Play size={16} /> {isUpdatingExercise ? '处理中…' : '开始执行'}</button>
              </div>
            )}

            {exerciseStage === 'in_progress' && (
              <div className="modal-actions">
                <button className="secondary-button" disabled={isUpdatingExercise} onClick={() => { setExerciseStage('disliked_reason'); setExerciseError(null); }} type="button">不满意</button>
                <button className="confirm-button" disabled={isUpdatingExercise} onClick={() => void updateExerciseTask('/feedback', { status: 'completed' })} type="button"><ThumbsUp size={16} /> {isUpdatingExercise ? '提交中…' : '我已完成'}</button>
              </div>
            )}

            {(exerciseStage === 'skipped_reason' || exerciseStage === 'disliked_reason') && (
              <div className="reason-panel">
                <label htmlFor="exercise-reason">{exerciseStage === 'skipped_reason' ? '请告诉我这次为什么不执行' : '请告诉我哪里不满意'}</label>
                <textarea
                  id="exercise-reason"
                  maxLength={500}
                  onChange={(event) => setExerciseReason(event.target.value)}
                  placeholder={exerciseStage === 'skipped_reason' ? '例如：今天加班太晚，时间不够' : '例如：动作强度太高，不喜欢原地踏步'}
                  rows={3}
                  value={exerciseReason}
                />
                <div className="modal-actions">
                  <button className="secondary-button" disabled={isUpdatingExercise} onClick={() => { setExerciseStage(exerciseTask.status === 'in_progress' ? 'in_progress' : 'ready'); setExerciseReason(''); setExerciseError(null); }} type="button">返回</button>
                  <button className="confirm-button" disabled={isUpdatingExercise || !exerciseReason.trim()} onClick={() => submitExerciseReason(exerciseStage === 'skipped_reason' ? 'skipped' : 'disliked')} type="button">
                    {isUpdatingExercise ? '提交中…' : '提交原因'}
                  </button>
                </div>
              </div>
            )}
          </section>
        </div>
      )}
    </main>
  );
}

export default App;
