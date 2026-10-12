
import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const GENERATE_API_URL = "http://127.0.0.1:8000/api/generate";
const CLASSIFY_API_URL = "http://127.0.0.1:8000/api/classify";
const PAPER_API_URL = "http://127.0.0.1:8000/api/generate-paper";
const ANALYZE_API_URL = "http://127.0.0.1:8000/api/analyze-paper";
const FEEDBACK_API_URL = "http://127.0.0.1:8000/api/feedback";
const FEEDBACK_SUMMARY_API_URL =
  "http://127.0.0.1:8000/api/feedback/summary";

const STORAGE_KEY = "qgen-ai-chat-history";

const BLOOM_LEVELS = [
  "Remember",
  "Understand",
  "Apply",
  "Analyze",
  "Evaluate",
  "Create",
];

const DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"];

const TASKS = [
  {
    id: "paper",
    icon: "P",
    title: "Generate Question Paper",
    description:
      "Create a complete question paper using available reference materials.",
  },
  {
    id: "questions",
    icon: "Q",
    title: "Generate Questions",
    description:
      "Generate unique questions grounded in your reference materials.",
  },
  {
    id: "validate",
    icon: "V",
    title: "Validate Questions",
    description:
      "Check question classification, difficulty, and reference support.",
  },
  {
    id: "analyze",
    icon: "A",
    title: "Analyze Question Paper",
    description:
      "View overall Bloom's levels and difficulty distribution.",
  },
];

function createChat() {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    title: "New Chat",
    messages: [],
    createdAt: new Date().toISOString(),
  };
}

function loadChats() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return [];

    const parsed = JSON.parse(stored);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function makeTitle(text) {
  const cleaned = String(text || "").replace(/\s+/g, " ").trim();
  if (!cleaned) return "New Chat";

  return cleaned.length > 42 ? `${cleaned.slice(0, 42)}...` : cleaned;
}

function normalizeBloom(value) {
  if (typeof value !== "string" || !value.trim()) return "N/A";

  const normalized = value
    .trim()
    .replace(
      /^bloom(?:'s)?\s*(?:cognitive\s*)?(?:level)?\s*[:=-]\s*/i,
      ""
    )
    .toLowerCase();

  const aliases = {
    remember: "Remember",
    remembering: "Remember",
    recall: "Remember",
    knowledge: "Remember",
    understand: "Understand",
    understanding: "Understand",
    comprehension: "Understand",
    apply: "Apply",
    applying: "Apply",
    application: "Apply",
    analyze: "Analyze",
    analysing: "Analyze",
    analyzing: "Analyze",
    analysis: "Analyze",
    evaluate: "Evaluate",
    evaluating: "Evaluate",
    evaluation: "Evaluate",
    create: "Create",
    creating: "Create",
    synthesis: "Create",
    synthesise: "Create",
    synthesize: "Create",
  };

  return aliases[normalized] || "N/A";
}

function getBloom(data = {}) {
  const candidates = [
    data.bloom_level,
    data.bloom,
    data.predicted_bloom,
    data.predicted_bloom_level,
    data.bloom_prediction,
    data["Bloom's Level"],
    data["Bloom Level"],
    data.classification?.bloom_level,
    data.classification?.bloom,
    data.prediction?.bloom_level,
    data.prediction?.bloom,
    data.qdiff?.bloom_level,
    data.qdiff?.bloom,
    data.llm?.bloom_level,
    data.llm?.bloom,
  ];

  for (const value of candidates) {
    const result = normalizeBloom(value);
    if (result !== "N/A") return result;
  }

  return "N/A";
}

function normalizeDifficulty(value) {
  if (typeof value !== "string") return "N/A";

  const normalized = value.trim().toLowerCase();

  if (normalized === "easy") return "Easy";
  if (normalized === "medium" || normalized === "moderate") return "Medium";
  if (normalized === "hard" || normalized === "difficult") return "Hard";

  return "N/A";
}

function getDifficulty(data = {}) {
  const candidates = [
    data.difficulty,
    data.predicted_difficulty,
    data.difficulty_level,
    data["Difficulty Level"],
    data.classification?.difficulty,
    data.prediction?.difficulty,
    data.qdiff?.difficulty,
    data.llm?.difficulty,
  ];

  for (const value of candidates) {
    const result = normalizeDifficulty(value);
    if (result !== "N/A") return result;
  }

  return "N/A";
}

function getConfidence(value) {
  if (value === undefined || value === null || value === "") return null;

  let numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;
  if (numeric >= 0 && numeric <= 1) numeric *= 100;

  return Math.max(0, Math.min(100, Math.round(numeric)));
}

function confidenceLabel(value) {
  const confidence = getConfidence(value);

  if (confidence === null) return "Not available";
  if (confidence >= 80) return "High confidence";
  if (confidence >= 60) return "Moderate confidence";
  if (confidence >= 40) return "Low confidence";

  return "Very low confidence";
}

function getQuestionText(item) {
  if (typeof item === "string") return item;

  return (
    item?.question ||
    item?.original_question ||
    item?.text ||
    item?.content ||
    "Question unavailable."
  );
}

function isVerified(item = {}) {
  return (
    item.verified === true ||
    item.accepted === true ||
    item.status === "verified"
  );
}

function isGrounded(item = {}) {
  return item.grounded === true || item.reference_grounded === true;
}

function parseGenerationPrompt(prompt) {
  const text = String(prompt || "").trim();
  const lower = text.toLowerCase();
  let numberOfQuestions = 5;

  const numberMatch =
    lower.match(/\b(\d+)\s*(?:questions?|qs?)\b/i) ||
    lower.match(/\b(?:generate|create|make|give|provide|write)\s+(\d+)\b/i);

  if (numberMatch) {
    numberOfQuestions = Number(numberMatch[1]) || 5;
  } else {
    const wordNumbers = {
      one: 1,
      two: 2,
      three: 3,
      four: 4,
      five: 5,
      six: 6,
      seven: 7,
      eight: 8,
      nine: 9,
      ten: 10,
    };

    const wordMatch = lower.match(
      /\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:questions?|qs?)\b/i
    );

    if (wordMatch) {
      numberOfQuestions = wordNumbers[wordMatch[1]] || 5;
    }
  }

  numberOfQuestions = Math.max(1, Math.min(30, numberOfQuestions));

  let bloom = "";

  for (const level of BLOOM_LEVELS) {
    if (new RegExp(`\\b${level}\\b`, "i").test(text)) {
      bloom = level;
      break;
    }
  }

  const difficultyMatch = lower.match(/\b(easy|medium|hard|difficult)\b/i);
  let difficulty = "";

  if (difficultyMatch) {
    difficulty = normalizeDifficulty(difficultyMatch[1]);
  }

  let topic = "";

  const topicMatch = text.match(
    /\b(?:on|about|regarding|from)\s+(.+?)(?:,\s*(?:with|at|difficulty|bloom)|$)/i
  );

  if (topicMatch) topic = topicMatch[1].trim();

  topic = topic
    .replace(/[?.!]+$/, "")
    .replace(/\b(?:at|with)\s+(?:easy|medium|hard|difficult)\b.*$/i, "")
    .trim();

  if (!topic) {
    const knownTopics = [
      "ReAct agents",
      "ReAct",
      "Decision Trees",
      "Artificial Intelligence",
      "Machine Learning",
      "Deep Learning",
      "Natural Language Processing",
      "Computer Networks",
      "Operating Systems",
      "Database Management Systems",
      "DBMS",
      "Data Structures",
      "Algorithms",
      "Neural Networks",
      "Fuzzy Logic",
      "Data Mining",
    ];

    topic =
      knownTopics.find((candidate) =>
        lower.includes(candidate.toLowerCase())
      ) || text;
  }

  return {
    topic: topic || "General Topic",
    subtopic: topic || "General Topic",
    bloom,
    difficulty,
    numberOfQuestions,
  };
}

function isQuestionForClassification(text) {
  const trimmed = String(text || "").trim();
  if (!trimmed) return false;

  if (/^(?:generate|create|make|provide|write|produce)\b/i.test(trimmed)) {
    return false;
  }

  if (
    /\b(?:generate|create|make)\s+\d+\s+(?:questions?|qs?)\b/i.test(trimmed)
  ) {
    return false;
  }

  if (
    /\b(?:classify|classification|bloom(?:'s)? level|difficulty level)\b/i.test(
      trimmed
    ) &&
    !trimmed.endsWith("?")
  ) {
    return false;
  }

  return (
    trimmed.endsWith("?") ||
    /^(?:what|why|how|which|when|where|define|describe|explain|calculate|compare|discuss|analyze|analyse|evaluate|derive|find|solve|justify|illustrate)\b/i.test(
      trimmed
    )
  );
}

function ConfidenceBar({ value }) {
  const confidence = getConfidence(value);

  if (confidence === null) {
    return (
      <div className="chat-confidence">
        <div className="chat-confidence-header">
          <span>QDiff confidence</span>
          <strong>N/A</strong>
        </div>
        <small>Confidence was not returned by the backend.</small>
      </div>
    );
  }

  return (
    <div className="chat-confidence">
      <div className="chat-confidence-header">
        <span>QDiff confidence</span>
        <strong>{confidence}%</strong>
      </div>

      <div className="chat-confidence-bar">
        <div
          className="chat-confidence-fill"
          style={{ width: `${confidence}%` }}
        />
      </div>

      <small>{confidenceLabel(confidence)}</small>
    </div>
  );
}

function FeedbackWidget({ feature, data = {}, resultId }) {
  const [rating, setRating] = useState(0);
  const [reason, setReason] = useState("");
  const [comment, setComment] = useState("");
  const [correctedBloom, setCorrectedBloom] = useState("");
  const [correctedDifficulty, setCorrectedDifficulty] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState("");

  const [showSummary, setShowSummary] = useState(false);
  const [summary, setSummary] = useState(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [summaryError, setSummaryError] = useState("");

  // Feedback is collapsed by default.
  const [showFeedback, setShowFeedback] = useState(false);

  const question =
    typeof data.question === "string" ? data.question : getQuestionText(data);

  const predictedBloom = getBloom(data);
  const predictedDifficulty = getDifficulty(data);

  const feedbackFeature = {
    "question-paper": "question_paper",
    generation: "question_generation",
    classification: "validation",
    analysis: "paper_analysis",
  }[feature] || feature || "general";

  async function submitFeedback(event) {
    event.preventDefault();

    if (!rating || submitting || submitted) return;

    const ratingValue =
      rating >= 4
        ? "helpful"
        : rating === 3
          ? "partly_helpful"
          : "not_helpful";

    setSubmitting(true);
    setError("");

    try {
      const response = await fetch(FEEDBACK_API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          feature: feedbackFeature,
          rating: ratingValue,
          reason: reason || null,
          comment: comment.trim() || null,
          question: question || null,
          predicted_bloom: predictedBloom !== "N/A" ? predictedBloom : null,
          corrected_bloom: correctedBloom || null,
          predicted_difficulty:
            predictedDifficulty !== "N/A" ? predictedDifficulty : null,
          corrected_difficulty: correctedDifficulty || null,
          result_id: resultId || null,
        }),
      });

      let payload = {};

      try {
        payload = await response.json();
      } catch {
        payload = {};
      }

      if (!response.ok || payload.success === false) {
        throw new Error(
          payload.detail ||
            payload.message ||
            payload.error ||
            "Could not save feedback. Check that the feedback API is running."
        );
      }

      setSubmitted(true);
    } catch (submitError) {
      setError(submitError.message || "Could not save feedback.");
    } finally {
      setSubmitting(false);
    }
  }

  async function loadSummary() {
    if (showSummary && summary) {
      setShowSummary(false);
      return;
    }

    setShowSummary(true);
    setSummaryLoading(true);
    setSummaryError("");

    try {
      const response = await fetch(FEEDBACK_SUMMARY_API_URL);

      let payload = {};

      try {
        payload = await response.json();
      } catch {
        payload = {};
      }

      if (!response.ok || payload.success === false) {
        throw new Error(
          payload.detail ||
            payload.message ||
            "Could not load feedback summary."
        );
      }

      setSummary(payload);
    } catch (summaryLoadError) {
      setSummaryError(
        summaryLoadError.message || "Could not load feedback summary."
      );
    } finally {
      setSummaryLoading(false);
    }
  }

  const ratings = summary?.ratings || {};

  const issues = Array.isArray(summary?.common_issues)
    ? summary.common_issues
    : [];

  // Collapsed view: only show the Show Feedback control.
  if (!showFeedback) {
    return (
      <div className="feedback-collapsed">
        <button
          type="button"
          className="feedback-show-button"
          onClick={() => setShowFeedback(true)}
          aria-expanded="false"
        >
          <span aria-hidden="true">＋</span>
          Show Feedback
        </button>
      </div>
    );
  }

  return (
    <section className="feedback-panel">
      <div className="feedback-heading">
        <div>
          <strong>Was this result useful?</strong>
          <p>Your feedback helps improve QGen AI.</p>
        </div>

        <div className="feedback-heading-actions">
          {submitted && (
            <span className="feedback-success">Feedback saved</span>
          )}

          <button
            type="button"
            className="feedback-close-button"
            aria-label="Close feedback"
            title="Close feedback"
            onClick={() => setShowFeedback(false)}
          >
            ×
          </button>
        </div>
      </div>

      {submitted ? (
        <p className="feedback-thanks">
          Thank you for helping us improve question quality.
        </p>
      ) : (
        <form className="feedback-form" onSubmit={submitFeedback}>
          <div
            className="feedback-rating-row"
            aria-label="Rating from one to five"
          >
            {[1, 2, 3, 4, 5].map((value) => (
              <button
                className={`feedback-star ${
                  value <= rating ? "selected" : ""
                }`}
                key={value}
                type="button"
                onClick={() => setRating(value)}
                aria-label={`Rate ${value} out of 5`}
                aria-pressed={rating === value}
              >
                ★
              </button>
            ))}

            <span className="feedback-rating-caption">
              {rating ? `${rating}/5` : "Select a rating"}
            </span>
          </div>

          <div className="feedback-fields">
            <label>
              Main feedback reason
              <select
                value={reason}
                onChange={(event) => setReason(event.target.value)}
              >
                <option value="">Choose a reason (optional)</option>
                <option value="Quality of questions">
                  Quality of questions
                </option>
                <option value="Difficulty level">Difficulty level</option>
                <option value="Bloom's taxonomy distribution">
                  Bloom's taxonomy distribution
                </option>
                <option value="Reference relevance">Reference relevance</option>
                <option value="Duplicate questions">Duplicate questions</option>
                <option value="Incorrect classification">
                  Incorrect classification
                </option>
                <option value="Other">Other</option>
              </select>
            </label>

            <label>
              Correct Bloom's level (optional)
              <select
                value={correctedBloom}
                onChange={(event) => setCorrectedBloom(event.target.value)}
              >
                <option value="">No correction</option>
                {BLOOM_LEVELS.map((level) => (
                  <option key={level} value={level}>
                    {level}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Correct difficulty (optional)
              <select
                value={correctedDifficulty}
                onChange={(event) =>
                  setCorrectedDifficulty(event.target.value)
                }
              >
                <option value="">No correction</option>
                {DIFFICULTY_LEVELS.map((level) => (
                  <option key={level} value={level}>
                    {level}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <label className="feedback-comment-label">
            Additional comments (optional)
            <textarea
              value={comment}
              onChange={(event) => setComment(event.target.value)}
              rows={2}
              maxLength={2000}
              placeholder="What should be improved?"
            />
          </label>

          {error && (
            <p className="feedback-error" role="alert">
              {error}
            </p>
          )}

          <button
            className="feedback-submit"
            type="submit"
            disabled={!rating || submitting}
          >
            {submitting ? "Saving feedback..." : "Submit feedback"}
          </button>
        </form>
      )}

      <div className="feedback-summary-toggle">
        <button
          type="button"
          onClick={loadSummary}
          disabled={summaryLoading}
        >
          {summaryLoading
            ? "Loading summary..."
            : showSummary
              ? "Hide feedback insights"
              : "View feedback insights"}
        </button>
      </div>

      {showSummary && (
        <div className="feedback-summary">
          {summaryLoading && <p>Loading feedback summary…</p>}

          {summaryError && (
            <p className="feedback-error">{summaryError}</p>
          )}

          {summary && !summaryLoading && (
            <>
              <div className="feedback-summary-stats">
                <div>
                  <span>Total feedback</span>
                  <strong>{summary.total_feedback ?? 0}</strong>
                </div>

                <div>
                  <span>Helpful</span>
                  <strong>{ratings.helpful ?? 0}</strong>
                </div>

                <div>
                  <span>Partly helpful</span>
                  <strong>{ratings.partly_helpful ?? 0}</strong>
                </div>

                <div>
                  <span>Not helpful</span>
                  <strong>{ratings.not_helpful ?? 0}</strong>
                </div>

                <div>
                  <span>Corrected examples</span>
                  <strong>
                    {Array.isArray(summary.corrected_examples)
                      ? summary.corrected_examples.length
                      : summary.corrected_examples ?? 0}
                  </strong>
                </div>
              </div>

              <h4>Common improvement areas</h4>

              {issues.length ? (
                <ul className="feedback-issues">
                  {issues.map((item, index) => (
                    <li key={`${item.reason}-${index}`}>
                      <span>{item.reason}</span>
                      <strong>{item.count}</strong>
                    </li>
                  ))}
                </ul>
              ) : (
                <p>No feedback reasons have been recorded yet.</p>
              )}
            </>
          )}
        </div>
      )}
    </section>
  );
}

export default function App() {
  const [chats, setChats] = useState(() => loadChats());
  const [activeChatId, setActiveChatId] = useState(null);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [currentView, setCurrentView] = useState("dashboard");
  const [selectedTask, setSelectedTask] = useState(null);

  // Stores hidden analysis result IDs without affecting other messages.
  const [hiddenAnalysis, setHiddenAnalysis] = useState({});

  const [paperSettings, setPaperSettings] = useState({
    topic: "",
    numberOfQuestions: 8,
    difficulty: "Mixed",
    questionType: "Mixed",
    useAllReferences: true,
    avoidDuplicates: true,
    diversifyConcepts: true,
  });

  const [validationInput, setValidationInput] = useState("");
  const [analysisInput, setAnalysisInput] = useState("");
  const [selectedFileName, setSelectedFileName] = useState("");

  const textareaRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (chats.length === 0) {
      const initialChat = createChat();
      setChats([initialChat]);
      setActiveChatId(initialChat.id);
      return;
    }

    if (!activeChatId || !chats.some((chat) => chat.id === activeChatId)) {
      setActiveChatId(chats[0].id);
    }
  }, [chats, activeChatId]);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
    } catch {
      // Local history is optional; storage errors should not break the app.
    }
  }, [chats]);

  const activeChat = useMemo(
    () => chats.find((chat) => chat.id === activeChatId),
    [chats, activeChatId]
  );

  const messages = activeChat?.messages || [];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    if (!textareaRef.current) return;

    textareaRef.current.style.height = "auto";
    textareaRef.current.style.height =
      `${Math.min(textareaRef.current.scrollHeight, 140)}px`;
  }, [input]);

  function updateChat(chatId, updater) {
    setChats((previous) =>
      previous.map((chat) =>
        chat.id === chatId ? updater(chat) : chat
      )
    );
  }

  function addMessage(chatId, message) {
    updateChat(chatId, (chat) => ({
      ...chat,
      messages: [
        ...chat.messages,
        {
          id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
          ...message,
        },
      ],
    }));
  }

  function createNewChat() {
    if (loading) return;

    const chat = createChat();

    setChats((previous) => [chat, ...previous]);
    setActiveChatId(chat.id);
    setInput("");
    setSelectedTask(null);
    setCurrentView("dashboard");
    setValidationInput("");
    setAnalysisInput("");
    setSelectedFileName("");
    setHiddenAnalysis({});
  }

  function deleteChat(chatId, event) {
    event?.stopPropagation();

    if (loading) return;

    const remaining = chats.filter((chat) => chat.id !== chatId);

    if (remaining.length === 0) {
      const replacement = createChat();

      setChats([replacement]);
      setActiveChatId(replacement.id);
      setCurrentView("dashboard");
      setSelectedTask(null);
      setHiddenAnalysis({});
      return;
    }

    setChats(remaining);

    if (activeChatId === chatId) {
      setActiveChatId(remaining[0].id);
      setCurrentView("conversation");
      setSelectedTask(null);
      setHiddenAnalysis({});
    }
  }

  function selectChat(chatId) {
    if (loading) return;

    setActiveChatId(chatId);
    setCurrentView("conversation");
    setSelectedTask(null);
    setInput("");
    setValidationInput("");
    setAnalysisInput("");
    setSelectedFileName("");
    setHiddenAnalysis({});
  }

  function goToDashboard() {
    if (loading) return;

    setCurrentView("dashboard");
    setSelectedTask(null);
    setInput("");
    setValidationInput("");
    setAnalysisInput("");
    setSelectedFileName("");
  }

  function openTask(taskId) {
    if (loading) return;

    setSelectedTask(taskId);
    setCurrentView("task");
    setInput("");
    setValidationInput("");
    setAnalysisInput("");
    setSelectedFileName("");
  }

  function closeTask() {
    if (loading) return;

    setSelectedTask(null);
    setCurrentView("dashboard");
    setValidationInput("");
    setAnalysisInput("");
    setSelectedFileName("");
  }

  async function postJson(url, body, fallbackMessage) {
    let response;

    try {
      response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch {
      throw new Error(
        "Could not connect to the backend. Make sure FastAPI is running at http://127.0.0.1:8000."
      );
    }

    let data;

    try {
      data = await response.json();
    } catch {
      throw new Error("The backend returned an invalid response.");
    }

    if (!response.ok || data?.success === false) {
      throw new Error(
        data?.detail ||
          data?.message ||
          data?.error ||
          fallbackMessage
      );
    }

    return data;
  }

  async function generateQuestions(parsed) {
    return postJson(
      GENERATE_API_URL,
      {
        topic: parsed.topic,
        subtopic: parsed.subtopic,
        bloom: parsed.bloom || "",
        difficulty: parsed.difficulty || "",
        number_of_questions: parsed.numberOfQuestions,
      },
      "Question generation failed."
    );
  }

  async function classifyQuestion(question) {
    return postJson(
      CLASSIFY_API_URL,
      { question },
      "Question classification failed."
    );
  }

  async function generateQuestionPaper() {
    const subject = paperSettings.topic.trim();
    const numberOfQuestions =
      Number(paperSettings.numberOfQuestions) || 8;

    const patternMap = {
      8: 30,
      10: 30,
      15: 50,
      20: 60,
    };

    const pattern = patternMap[numberOfQuestions] || 30;

    const difficulty =
      paperSettings.difficulty.toLowerCase() === "mixed"
        ? "balanced"
        : paperSettings.difficulty.toLowerCase();

    return postJson(
      PAPER_API_URL,
      {
        subject: subject || "General Subject",
        scope: paperSettings.useAllReferences ? "all" : "selected",
        topics:
          !paperSettings.useAllReferences && subject ? [subject] : [],
        pattern,
        difficulty,
        number_of_questions: numberOfQuestions,
        question_type: paperSettings.questionType,
        avoid_duplicates: paperSettings.avoidDuplicates,
        diversify_concepts: paperSettings.diversifyConcepts,
      },
      "Question paper generation failed."
    );
  }

  async function analyzeQuestionPaper(questions) {
    const data = await postJson(
      ANALYZE_API_URL,
      { questions },
      "Question paper analysis failed."
    );

    // Preserve original questions when the backend returns labels only.
    const backendResults = Array.isArray(data.results)
      ? data.results
      : [];

    const results = questions.map((submittedQuestion, index) => {
      const backendItem = backendResults[index];

      if (typeof backendItem === "string") {
        return {
          question: submittedQuestion,
          classification: backendItem,
        };
      }

      return {
        ...(backendItem || {}),
        question:
          backendItem?.question ||
          backendItem?.original_question ||
          backendItem?.text ||
          backendItem?.content ||
          submittedQuestion,
      };
    });

    return {
      ...data,
      questions,
      results,
      total_questions: data.total_questions ?? questions.length,
    };
  }

  async function runAndStore({ prompt, type, request, title }) {
    if (loading || !activeChatId) return;

    const chatId = activeChatId;
    const isFirstMessage = !(activeChat?.messages?.length);

    setCurrentView("conversation");
    setSelectedTask(null);
    setInput("");
    setLoading(true);

    addMessage(chatId, { role: "user", content: prompt });

    if (isFirstMessage) {
      updateChat(chatId, (chat) => ({
        ...chat,
        title: makeTitle(title || prompt),
      }));
    }

    try {
      const data = await request();

      addMessage(chatId, {
        role: "assistant",
        type,
        data,
      });
    } catch (error) {
      console.error(error);

      addMessage(chatId, {
        role: "assistant",
        type: "error",
        error:
          error?.message ||
          "Something went wrong while processing the request.",
      });
    } finally {
      setLoading(false);
      setTimeout(() => textareaRef.current?.focus(), 50);
    }
  }

  async function handleSubmit(event) {
    event?.preventDefault();

    const prompt = input.trim();

    if (!prompt || loading || !activeChatId) return;

    if (isQuestionForClassification(prompt)) {
      await runAndStore({
        prompt,
        type: "classification",
        title: prompt,
        request: async () => {
          const data = await classifyQuestion(prompt);
          return {
            ...data,
            question: data.question || prompt,
          };
        },
      });

      return;
    }

    const parsed = parseGenerationPrompt(prompt);

    await runAndStore({
      prompt,
      type: "generation",
      title: prompt,
      request: () => generateQuestions(parsed),
    });
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSubmit(event);
    }
  }

  async function submitQuestionPaperSettings(event) {
    event.preventDefault();

    const subject = paperSettings.topic.trim();

    const prompt = subject
      ? `Generate a ${paperSettings.numberOfQuestions}-question paper for ${subject}.`
      : `Generate a ${paperSettings.numberOfQuestions}-question paper using all relevant reference materials.`;

    await runAndStore({
      prompt,
      type: "question-paper",
      title: prompt,
      request: generateQuestionPaper,
    });
  }

  async function submitValidationWorkflow(event) {
    event.preventDefault();

    const question = validationInput.trim();

    if (question) {
      await runAndStore({
        prompt: `Validate question: ${question}`,
        type: "classification",
        title: `Validate: ${question}`,
        request: async () => {
          const data = await classifyQuestion(question);

          return {
            ...data,
            question: data.question || question,
          };
        },
      });

      return;
    }

    if (selectedFileName && activeChatId) {
      setCurrentView("conversation");
      setSelectedTask(null);

      addMessage(activeChatId, {
        role: "assistant",
        type: "error",
        error: `The file "${selectedFileName}" is selected, but document extraction is not connected. Paste the question text to validate it.`,
      });
    }
  }

  async function submitAnalysisWorkflow(event) {
    event.preventDefault();

    const questions = analysisInput
      .split(/\r?\n/)
      .map((question) => question.trim())
      .filter(Boolean);

    if (questions.length > 0) {
      await runAndStore({
        // Display the actual entered questions in the user message.
        prompt: questions.join("\n"),
        type: "analysis",
        title: "Question Paper Analysis",
        request: () => analyzeQuestionPaper(questions),
      });

      return;
    }

    if (selectedFileName && activeChatId) {
      setCurrentView("conversation");
      setSelectedTask(null);

      addMessage(activeChatId, {
        role: "assistant",
        type: "error",
        error: `The file "${selectedFileName}" is selected, but document extraction is not connected. Paste the questions into the analysis form.`,
      });
    }
  }

  function renderQuestionResults(data = {}) {
    const questions = Array.isArray(data.questions) ? data.questions : [];

    const referenceChunks =
      data.reference_chunks ?? data.rag?.results_found ?? 0;

    const requested =
      data.requested_number ??
      data.number_requested ??
      data.number_of_questions ??
      questions.length;

    const verifiedCount =
      data.verified_count ?? questions.filter(isVerified).length;

    return (
      <div className="assistant-result">
        <div className="assistant-intro">
          <div className="ai-avatar small-avatar">Q</div>
          <div>
            <div className="assistant-name">Question Generator</div>
            <div className="assistant-subtitle">
              Reference-grounded generation with QDiff
            </div>
          </div>
        </div>

        <div className="generated-results">
          <div className="result-summary">
            <div>
              <strong>
                {verifiedCount} verified · {questions.length} generated
              </strong>
              <span>{referenceChunks} reference chunks</span>
            </div>

            <div className="request-tags">
              {data.requested_bloom && <span>{data.requested_bloom}</span>}
              {data.requested_difficulty && (
                <span>{data.requested_difficulty}</span>
              )}
              <span>{requested} requested</span>
            </div>
          </div>

          {questions.length === 0 ? (
            <div className="no-results">
              <strong>No questions generated</strong>
              <p>
                {data.message ||
                  "The generator could not produce questions satisfying the request and verification requirements."}
              </p>
            </div>
          ) : (
            <div className="chat-question-list">
              {questions.map((item, index) => {
                const question = getQuestionText(item);

                const confidence =
                  item.qdiff_confidence ??
                  item.qdiff?.confidence ??
                  item.confidence ??
                  item.score;

                return (
                  <div
                    className="chat-question-card"
                    key={item.id || `${index}-${question}`}
                  >
                    <div className="chat-question-top">
                      <span className="question-index">Q{index + 1}</span>

                      {isVerified(item) && (
                        <span className="accepted-badge">✓ Verified</span>
                      )}
                    </div>

                    <div className="chat-question-text">{question}</div>

                    <div className="chat-question-meta">
                      <div className="chat-meta">
                        <span>Bloom's Level</span>
                        <strong>{getBloom(item)}</strong>
                      </div>

                      <div className="chat-meta">
                        <span>Difficulty</span>
                        <strong>{getDifficulty(item)}</strong>
                      </div>

                      <ConfidenceBar value={confidence} />
                    </div>

                    {isGrounded(item) && (
                      <div className="grounded-line">
                        <span>✓</span> Generated from reference material
                      </div>
                    )}

                    {!isVerified(item) && (
                      <div className="generation-warning">
                        <span>!</span>
                        <div>
                          The backend has not marked this question as verified.
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {data.message && questions.length > 0 && (
            <div className="generation-verification">
              <strong>Generation status</strong>
              <p>{data.message}</p>
            </div>
          )}
        </div>
      </div>
    );
  }

  function renderQuestionPaperResults(data = {}) {
    const questions = Array.isArray(data.questions) ? data.questions : [];

    const totalRequested =
      data.total_requested ?? data.requested_number ?? questions.length;

    const totalGenerated = data.total_generated ?? questions.length;

    const topics = Array.isArray(data.topics) ? data.topics : [];

    return (
      <div className="assistant-result">
        <div className="assistant-intro">
          <div className="ai-avatar small-avatar">Q</div>
          <div>
            <div className="assistant-name">Question Paper Generator</div>
            <div className="assistant-subtitle">
              Reference-grounded paper generation
            </div>
          </div>
        </div>

        <div className="generated-results">
          <div className="result-summary">
            <div>
              <strong>{totalGenerated} questions generated</strong>
              <span>{data.subject || "Reference-based question paper"}</span>
            </div>

            <div className="request-tags">
              <span>{data.difficulty || paperSettings.difficulty}</span>
              <span>{totalRequested} requested</span>

              {data.grounded === true && <span>RAG Grounded</span>}

              {data.grounded === false && (
                <span>Grounding not verified</span>
              )}
            </div>
          </div>

          {topics.length > 0 && (
            <div className="generation-verification">
              <strong>Reference-backed topics</strong>
              <p>{topics.join(" · ")}</p>
            </div>
          )}

          {questions.length === 0 ? (
            <div className="no-results">
              <strong>No questions generated</strong>
              <p>
                {data.message ||
                  "The system could not generate enough reference-supported questions."}
              </p>
            </div>
          ) : (
            <div className="chat-question-list">
              {questions.map((item, index) => {
                const question = getQuestionText(item);

                const confidence =
                  item.qdiff_confidence ??
                  item.qdiff?.confidence ??
                  item.confidence ??
                  item.score;

                return (
                  <div
                    className="chat-question-card"
                    key={item.id || `${index}-${question}`}
                  >
                    <div className="chat-question-top">
                      <span className="question-index">Q{index + 1}</span>

                      {isVerified(item) && (
                        <span className="accepted-badge">✓ Verified</span>
                      )}
                    </div>

                    <div className="chat-question-text">{question}</div>

                    {item.marks != null && (
                      <div className="paper-question-topic">
                        Marks: {item.marks}
                      </div>
                    )}

                    {item.topic && (
                      <div className="paper-question-topic">
                        Topic: {item.topic}
                      </div>
                    )}

                    <div className="chat-question-meta">
                      <div className="chat-meta">
                        <span>Bloom's Level</span>
                        <strong>{getBloom(item)}</strong>
                      </div>

                      <div className="chat-meta">
                        <span>Difficulty</span>
                        <strong>{getDifficulty(item)}</strong>
                      </div>

                      <ConfidenceBar value={confidence} />
                    </div>

                    {isGrounded(item) && (
                      <div className="grounded-line">
                        <span>✓</span> Generated from reference material
                      </div>
                    )}

                    {!isVerified(item) && (
                      <div className="generation-warning">
                        <span>!</span>
                        <div>
                          The backend has not marked this question as verified.
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {data.message && (
            <div className="generation-verification">
              <strong>Generation status</strong>
              <p>{data.message}</p>
            </div>
          )}
        </div>
      </div>
    );
  }

  function renderClassificationResults(data = {}) {
    const bloom = getBloom(data);
    const difficulty = getDifficulty(data);

    const confidence =
      data.qdiff_confidence ??
      data.qdiff?.confidence ??
      data.confidence;

    return (
      <div className="assistant-result">
        <div className="assistant-intro">
          <div className="ai-avatar small-avatar">Q</div>
          <div>
            <div className="assistant-name">Question Intelligence</div>
            <div className="assistant-subtitle">
              Bloom's level and difficulty classification
            </div>
          </div>
        </div>

        <div className="generated-results">
          <div className="classification-card">
            <div className="classification-question-label">Question</div>
            <div className="classification-question">
              {data.question || "Question unavailable."}
            </div>
          </div>

          <div className="classification-card">
            <div className="classification-meta">
              <div className="classification-item">
                <span>Bloom's Level</span>
                <strong>{bloom}</strong>
              </div>

              <div className="classification-item">
                <span>Difficulty</span>
                <strong>{difficulty}</strong>
              </div>
            </div>

            <ConfidenceBar value={confidence} />

            {data.classification_source && (
              <p className="classification-source">
                Classification source: {data.classification_source}
              </p>
            )}

            {bloom === "N/A" && (
              <div className="generation-warning">
                <span>!</span>
                <div>
                  No recognized Bloom's level was returned by the backend.
                  Check that /api/classify returns a field such as bloom_level
                  or bloom.
                </div>
              </div>
            )}

            {difficulty === "N/A" && (
              <div className="generation-warning">
                <span>!</span>
                <div>
                  No recognized difficulty was returned by the backend.
                  Expected values are Easy, Medium, or Hard.
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  function renderAnalysisResults(data = {}, messageId) {
    const results = Array.isArray(data.results) ? data.results : [];

    const bloomDistribution = data.bloom_distribution || {};
    const difficultyDistribution = data.difficulty_distribution || {};

    const bloomCounts = Object.fromEntries(
      BLOOM_LEVELS.map((level) => [
        level,
        Number(bloomDistribution[level] ?? 0),
      ])
    );

    const difficultyCounts = Object.fromEntries(
      DIFFICULTY_LEVELS.map((level) => [
        level,
        Number(difficultyDistribution[level] ?? 0),
      ])
    );

    // Show only Bloom's levels and difficulty categories with questions.
    const visibleBloom = Object.entries(bloomCounts).filter(
      ([, count]) => count > 0
    );

    const visibleDifficulty = Object.entries(difficultyCounts).filter(
      ([, count]) => count > 0
    );

    const unclassified = Number(bloomDistribution.Unclassified ?? 0);

    const total = Number(
      data.total_questions ?? data.analyzed_questions ?? results.length
    );

    if (hiddenAnalysis[messageId]) {
      return (
        <div className="analysis-hidden-bar">
          <span>Question paper analysis is hidden.</span>

          <button
            type="button"
            className="analysis-show-button"
            onClick={() =>
              setHiddenAnalysis((previous) => ({
                ...previous,
                [messageId]: false,
              }))
            }
          >
            Show results
          </button>
        </div>
      );
    }

    return (
      <div className="assistant-result">
        <div className="assistant-intro">
          <div className="ai-avatar small-avatar">Q</div>

          <div className="analysis-title-block">
            <div className="assistant-name">Question Paper Analysis</div>
            <div className="assistant-subtitle">
              Overall Bloom's levels and difficulty distribution
            </div>
          </div>

          <button
            type="button"
            className="analysis-hide-button"
            aria-label="Hide analysis results"
            title="Hide analysis results"
            onClick={() =>
              setHiddenAnalysis((previous) => ({
                ...previous,
                [messageId]: true,
              }))
            }
          >
            ×
          </button>
        </div>

        <div className="generated-results">
          <div className="analysis-total-card">
            <span>Total Questions</span>
            <strong>{total}</strong>
          </div>

          <div className="analysis-distribution-grid">
            <section className="classification-card analysis-distribution-card">
              <h3>Bloom's Distribution</h3>

              {visibleBloom.length > 0 ? (
                visibleBloom.map(([level, count]) => (
                  <div className="analysis-distribution-row" key={level}>
                    <span>{level}</span>
                    <strong>{count}</strong>
                  </div>
                ))
              ) : (
                <p>No classified Bloom levels available.</p>
              )}

              {unclassified > 0 && (
                <div className="analysis-distribution-row">
                  <span>Unclassified</span>
                  <strong>{unclassified}</strong>
                </div>
              )}
            </section>

            <section className="classification-card analysis-distribution-card">
              <h3>Difficulty Distribution</h3>

              {visibleDifficulty.length > 0 ? (
                visibleDifficulty.map(([level, count]) => (
                  <div className="analysis-distribution-row" key={level}>
                    <span>{level}</span>
                    <strong>{count}</strong>
                  </div>
                ))
              ) : (
                <p>No classified difficulty levels available.</p>
              )}
            </section>
          </div>

          {data.overall_bloom_level && (
            <div className="generation-verification">
              <strong>Overall Bloom's level</strong>
              <p>{data.overall_bloom_level}</p>
            </div>
          )}

          {data.overall_difficulty && (
            <div className="generation-verification">
              <strong>Overall difficulty</strong>
              <p>{data.overall_difficulty}</p>
            </div>
          )}

          {data.feedback && (
            <div className="generation-verification">
              <strong>Overall feedback</strong>
              <p>{data.feedback}</p>
            </div>
          )}

          {data.message && (
            <div className="generation-verification">
              <strong>Analysis notes</strong>
              <p>{data.message}</p>
            </div>
          )}
        </div>
      </div>
    );
  }

  function renderDashboard() {
    return (
      <div className="welcome-screen">
        <div className="welcome-icon">Q</div>

        <h1>
          AI Question <span>Generator</span>
        </h1>

        <p>
          Generate reference-grounded questions, validate question
          classification, and analyze question papers.
        </p>

        <div className="example-grid">
          {TASKS.map((task) => (
            <button
              type="button"
              key={task.id}
              onClick={() => openTask(task.id)}
              disabled={loading}
            >
              <div className="example-icon">{task.icon}</div>

              <div>
                <strong>{task.title}</strong>
                <small>{task.description}</small>
              </div>
            </button>
          ))}
        </div>

        {messages.length > 0 && (
          <button
            type="button"
            className="task-example-link"
            onClick={() => setCurrentView("conversation")}
          >
            Return to the current conversation
          </button>
        )}
      </div>
    );
  }

  function renderTaskPanel() {
    if (!selectedTask) return null;

    const task = TASKS.find((item) => item.id === selectedTask);

    const panelStyle = {
      width: "100%",
      maxWidth: "900px",
      margin: "0 auto",
      textAlign: "left",
      background: "rgba(255,255,255,0.96)",
      border: "1px solid rgba(0,0,0,0.08)",
      borderRadius: "18px",
      padding: "24px",
      boxSizing: "border-box",
    };

    const labelStyle = {
      display: "block",
      fontSize: "13px",
      fontWeight: 700,
      marginBottom: "7px",
    };

    const inputStyle = {
      width: "100%",
      boxSizing: "border-box",
      padding: "11px 13px",
      borderRadius: "10px",
      border: "1px solid rgba(0,0,0,0.14)",
      background: "#fff",
      fontSize: "14px",
      outline: "none",
    };

    const primaryButtonStyle = {
      border: "none",
      borderRadius: "10px",
      padding: "11px 17px",
      fontWeight: 700,
      cursor: "pointer",
    };

    const secondaryButtonStyle = {
      border: "1px solid rgba(0,0,0,0.12)",
      borderRadius: "10px",
      padding: "11px 17px",
      background: "transparent",
      fontWeight: 600,
      cursor: "pointer",
    };

    const buttonRowStyle = {
      display: "flex",
      justifyContent: "flex-end",
      gap: "10px",
      flexWrap: "wrap",
      marginTop: "18px",
    };

    return (
      <div className="task-workspace">
        <div className="task-workspace-heading">
          <span className="task-workspace-eyebrow">WORKSPACE</span>
          <h1>{task?.title || "Task Workspace"}</h1>
          <p>{task?.description}</p>
        </div>

        <div style={panelStyle}>
          {selectedTask === "paper" && (
            <>
              <h2>Paper Settings</h2>

              <form onSubmit={submitQuestionPaperSettings}>
                <div style={{ marginBottom: "15px" }}>
                  <label style={labelStyle}>Subject or topic (optional)</label>

                  <input
                    style={inputStyle}
                    value={paperSettings.topic}
                    onChange={(event) =>
                      setPaperSettings((previous) => ({
                        ...previous,
                        topic: event.target.value,
                      }))
                    }
                    placeholder="Leave empty to use all relevant references"
                  />
                </div>

                <div className="task-form-grid">
                  <div>
                    <label style={labelStyle}>Number of Questions</label>

                    <select
                      style={inputStyle}
                      value={paperSettings.numberOfQuestions}
                      onChange={(event) =>
                        setPaperSettings((previous) => ({
                          ...previous,
                          numberOfQuestions: Number(event.target.value),
                        }))
                      }
                    >
                      {[8, 10, 15, 20].map((number) => (
                        <option key={number} value={number}>
                          {number}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label style={labelStyle}>Difficulty</label>

                    <select
                      style={inputStyle}
                      value={paperSettings.difficulty}
                      onChange={(event) =>
                        setPaperSettings((previous) => ({
                          ...previous,
                          difficulty: event.target.value,
                        }))
                      }
                    >
                      <option>Mixed</option>
                      <option>Easy</option>
                      <option>Medium</option>
                      <option>Hard</option>
                    </select>
                  </div>

                  <div>
                    <label style={labelStyle}>Question Type</label>

                    <select
                      style={inputStyle}
                      value={paperSettings.questionType}
                      onChange={(event) =>
                        setPaperSettings((previous) => ({
                          ...previous,
                          questionType: event.target.value,
                        }))
                      }
                    >
                      <option>Mixed</option>
                      <option>Short Answer</option>
                      <option>Descriptive</option>
                      <option>Application</option>
                    </select>
                  </div>
                </div>

                <div className="task-checkbox-list">
                  {[
                    [
                      "useAllReferences",
                      "Use all relevant available references",
                    ],
                    ["avoidDuplicates", "Avoid duplicate questions"],
                    ["diversifyConcepts", "Diversify question concepts"],
                  ].map(([key, label]) => (
                    <label key={key}>
                      <input
                        type="checkbox"
                        checked={paperSettings[key]}
                        onChange={(event) =>
                          setPaperSettings((previous) => ({
                            ...previous,
                            [key]: event.target.checked,
                          }))
                        }
                      />
                      {label}
                    </label>
                  ))}
                </div>

                <div style={buttonRowStyle}>
                  <button
                    type="button"
                    style={secondaryButtonStyle}
                    onClick={closeTask}
                    disabled={loading}
                  >
                    Back to Dashboard
                  </button>

                  <button
                    type="submit"
                    style={primaryButtonStyle}
                    disabled={loading}
                  >
                    {loading ? "Generating..." : "Generate Question Paper"}
                  </button>
                </div>
              </form>
            </>
          )}

          {selectedTask === "questions" && (
            <>
              <h2>Question Generation Request</h2>

              <p>
                Specify the topic, requested Bloom's level, difficulty, and
                number of questions.
              </p>

              <form
                onSubmit={(event) => {
                  event.preventDefault();
                  if (input.trim()) handleSubmit(event);
                }}
              >
                <label style={labelStyle}>Your request</label>

                <textarea
                  style={{
                    ...inputStyle,
                    minHeight: "110px",
                    resize: "vertical",
                  }}
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  placeholder="Example: Generate 5 Apply-level questions on Decision Trees with Medium difficulty"
                />

                <div style={buttonRowStyle}>
                  <button
                    type="button"
                    style={secondaryButtonStyle}
                    onClick={closeTask}
                    disabled={loading}
                  >
                    Back to Dashboard
                  </button>

                  <button
                    type="submit"
                    style={primaryButtonStyle}
                    disabled={loading || !input.trim()}
                  >
                    {loading ? "Generating..." : "Generate Questions"}
                  </button>
                </div>
              </form>
            </>
          )}

          {selectedTask === "validate" && (
            <>
              <h2>Validate Questions</h2>

              <p>
                Paste a question to classify its Bloom's level and difficulty.
                File extraction is not connected in this frontend yet.
              </p>

              <form onSubmit={submitValidationWorkflow}>
                <label style={labelStyle}>Question text</label>

                <textarea
                  style={{
                    ...inputStyle,
                    minHeight: "120px",
                    resize: "vertical",
                  }}
                  value={validationInput}
                  onChange={(event) => setValidationInput(event.target.value)}
                  placeholder="Paste the question you want to validate..."
                />

                <div style={{ marginTop: "18px" }}>
                  <label style={labelStyle}>Or select a file</label>

                  <input
                    type="file"
                    accept=".pdf,.doc,.docx,.png,.jpg,.jpeg"
                    onChange={(event) =>
                      setSelectedFileName(
                        event.target.files?.[0]?.name || ""
                      )
                    }
                  />

                  {selectedFileName && (
                    <p className="selected-file-label">
                      Selected: {selectedFileName}
                    </p>
                  )}
                </div>

                <div style={buttonRowStyle}>
                  <button
                    type="button"
                    style={secondaryButtonStyle}
                    onClick={closeTask}
                    disabled={loading}
                  >
                    Back to Dashboard
                  </button>

                  <button
                    type="submit"
                    style={primaryButtonStyle}
                    disabled={
                      loading ||
                      (!validationInput.trim() && !selectedFileName)
                    }
                  >
                    {loading ? "Validating..." : "Validate Question"}
                  </button>
                </div>
              </form>
            </>
          )}

          {selectedTask === "analyze" && (
            <>
              <h2>Analyze Question Paper</h2>

              <p>
                Paste one question per line to calculate the overall Bloom's
                level and difficulty distribution.
              </p>

              <form onSubmit={submitAnalysisWorkflow}>
                <label style={labelStyle}>Questions (one per line)</label>

                <textarea
                  style={{
                    ...inputStyle,
                    minHeight: "180px",
                    resize: "vertical",
                  }}
                  value={analysisInput}
                  onChange={(event) => setAnalysisInput(event.target.value)}
                  placeholder={
                    "1. Explain the concept of entropy.\n2. Apply the decision tree algorithm to a dataset.\n3. Compare two classification methods."
                  }
                />

                <div style={{ marginTop: "18px" }}>
                  <label style={labelStyle}>Or select a file</label>

                  <input
                    type="file"
                    accept=".pdf,.doc,.docx,.png,.jpg,.jpeg"
                    onChange={(event) =>
                      setSelectedFileName(
                        event.target.files?.[0]?.name || ""
                      )
                    }
                  />

                  {selectedFileName && (
                    <p className="selected-file-label">
                      Selected: {selectedFileName}
                    </p>
                  )}
                </div>

                <div style={buttonRowStyle}>
                  <button
                    type="button"
                    style={secondaryButtonStyle}
                    onClick={closeTask}
                    disabled={loading}
                  >
                    Back to Dashboard
                  </button>

                  <button
                    type="submit"
                    style={primaryButtonStyle}
                    disabled={
                      loading ||
                      (!analysisInput.trim() && !selectedFileName)
                    }
                  >
                    {loading ? "Analyzing..." : "Analyze Question Paper"}
                  </button>
                </div>
              </form>
            </>
          )}
        </div>
      </div>
    );
  }

  function renderConversation() {
    return (
      <div className="conversation">
        {messages.map((message) => {
          if (message.role === "user") {
            return (
              <div className="message-row user-row" key={message.id}>
                <div className="message-content">
                  <div className="user-bubble">{message.content}</div>
                </div>
              </div>
            );
          }

          return (
            <div className="message-row assistant-row" key={message.id}>
              <div className="ai-avatar">Q</div>

              <div className="message-content">
                {message.type === "generation" &&
                  renderQuestionResults(message.data)}

                {message.type === "question-paper" &&
                  renderQuestionPaperResults(message.data)}

                {message.type === "classification" &&
                  renderClassificationResults(message.data)}

                {message.type === "analysis" &&
                  renderAnalysisResults(message.data, message.id)}

                {[
                  "generation",
                  "question-paper",
                  "classification",
                  "analysis",
                ].includes(message.type) && (
                  <FeedbackWidget
                    feature={message.type}
                    data={message.data || {}}
                    resultId={message.id}
                  />
                )}

                {message.type === "error" && (
                  <div className="error-message">
                    <strong>Request failed</strong>
                    <p>{message.error}</p>
                  </div>
                )}
              </div>
            </div>
          );
        })}

        {loading && (
          <div className="message-row assistant-row">
            <div className="ai-avatar">Q</div>

            <div className="typing-message">
              <div className="typing-header">
                <strong>Question Intelligence</strong>
                <span>Processing</span>
              </div>

              <div className="typing-dots">
                <span />
                <span />
                <span />
              </div>

              <p>
                Retrieving context and processing your request...
              </p>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>
    );
  }

  return (
    <div className="chat-app">
      <aside className="sidebar">
        <div className="sidebar-top">
          <div className="sidebar-brand">
            <div className="sidebar-logo">Q</div>

            <div>
              <div className="sidebar-title">AI Question Generator</div>
              <div className="sidebar-caption">RAG + QDiff</div>
            </div>
          </div>

          <button
            type="button"
            className="new-chat-button"
            onClick={createNewChat}
            disabled={loading}
          >
            <span className="plus-icon">+</span> New Chat
          </button>

          <button
            type="button"
            className="sidebar-dashboard-button"
            onClick={goToDashboard}
            disabled={loading}
          >
            <span>⌂</span> Dashboard
          </button>
        </div>

        <div className="sidebar-history">
          <div className="sidebar-history-title">Previous Chats</div>

          <div className="sidebar-history-list">
            {chats.length === 0 ? (
              <div className="sidebar-item">No previous chats</div>
            ) : (
              chats.map((chat) => (
                <div
                  key={chat.id}
                  className={`history-chat ${
                    chat.id === activeChatId ? "history-chat-active" : ""
                  }`}
                  onClick={() => selectChat(chat.id)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") selectChat(chat.id);
                  }}
                >
                  <div className="history-chat-icon">Q</div>

                  <div className="history-chat-title">{chat.title}</div>

                  <button
                    type="button"
                    className="history-delete-button"
                    title="Delete chat"
                    onClick={(event) => deleteChat(chat.id, event)}
                    disabled={loading}
                  >
                    ×
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="online-dot" />

            <div>
              <strong>System ready</strong>
              <small>RAG · Qwen · QDiff</small>
            </div>
          </div>
        </div>
      </aside>

      <main className="chat-main">
        <header className="chat-header">
          <div className="chat-header-title">
            <div className="header-ai-icon">Q</div>

            <div>
              <strong>Question Intelligence</strong>
              <span>AI Question Generation &amp; Classification</span>
            </div>
          </div>

          <div className="chat-header-actions">
            {currentView !== "dashboard" && (
              <button
                type="button"
                className="header-dashboard-button"
                onClick={goToDashboard}
                disabled={loading}
              >
                ← Back to Dashboard
              </button>
            )}

            <div className="header-ready">
              <span />
              Ready
            </div>
          </div>
        </header>

        <div className="messages-container">
          {currentView === "dashboard" && renderDashboard()}
          {currentView === "task" && renderTaskPanel()}
          {currentView === "conversation" && renderConversation()}
        </div>

        <div className="composer-area">
          <form className="composer" onSubmit={handleSubmit}>
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                currentView === "task"
                  ? "Use the task workspace above..."
                  : "Ask to generate questions or classify an existing question..."
              }
              disabled={loading || currentView === "task"}
              rows={1}
            />

            <button
              type="submit"
              className="send-button"
              disabled={
                loading || !input.trim() || currentView === "task"
              }
              aria-label="Send"
            >
              ↑
            </button>
          </form>

          <div className="composer-hint">
            Enter to send · Shift + Enter for a new line
          </div>
        </div>
      </main>
    </div>
  );
}
