import { useEffect, useMemo, useRef, useState } from "react";

import "./App.css";

const GENERATE_API_URL = "http://127.0.0.1:8000/api/generate";

const CLASSIFY_API_URL = "http://127.0.0.1:8000/api/classify";

const STORAGE_KEY = "qgen-ai-chat-history";

/* =========================================================
   CHAT HELPERS
========================================================= */

function createChat() {
  return {
    id: Date.now().toString(),
    title: "New Chat",
    messages: [],
    createdAt: new Date().toISOString(),
  };
}

function loadChats() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);

    if (!saved) {
      return [];
    }

    const parsed = JSON.parse(saved);

    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function createChatTitle(text) {
  const cleaned = String(text || "")
    .replace(/\s+/g, " ")
    .trim();

  if (!cleaned) {
    return "New Chat";
  }

  return cleaned.length > 42
    ? `${cleaned.slice(0, 42)}...`
    : cleaned;
}

/* =========================================================
   NUMBER HELPERS
========================================================= */

const WORD_NUMBERS = {
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

/* =========================================================
   PROMPT PARSER
========================================================= */

function parsePrompt(text) {
  const original = String(text || "").trim();

  const lower = original
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();

  let numberOfQuestions = 1;

  const generationNumberMatch = lower.match(
    /\b(?:generate|geerate|create|make|give(?:\s+me)?|provide|write|produce)\s+(\d+)\b/i
  );

  const questionNumberMatch = lower.match(
    /\b(\d+)\s*(?:questions?|qs?|q)\b/i
  );

  const generationWordNumberMatch = lower.match(
    /\b(?:generate|geerate|create|make|give(?:\s+me)?|provide|write|produce)\s+(one|two|three|four|five|six|seven|eight|nine|ten)\b/i
  );

  const wordQuestionMatch = lower.match(
    /\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:questions?|qs?|q)\b/i
  );

  if (generationNumberMatch) {
    numberOfQuestions = Number(generationNumberMatch[1]);
  } else if (questionNumberMatch) {
    numberOfQuestions = Number(questionNumberMatch[1]);
  } else if (generationWordNumberMatch) {
    numberOfQuestions =
      WORD_NUMBERS[generationWordNumberMatch[1]] || 1;
  } else if (wordQuestionMatch) {
    numberOfQuestions =
      WORD_NUMBERS[wordQuestionMatch[1]] || 1;
  }

  numberOfQuestions = Math.min(
    Math.max(Number(numberOfQuestions) || 1, 1),
    20
  );

  /* -------------------------------------------------------
     BLOOM LEVEL
  ------------------------------------------------------- */

  const bloomLevels = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
  ];

  let bloom = "";

  for (const level of bloomLevels) {
    const regex = new RegExp(`\\b${level}\\b`, "i");

    if (regex.test(original)) {
      bloom = level;
      break;
    }
  }

  /* -------------------------------------------------------
     DIFFICULTY
  ------------------------------------------------------- */

  let difficulty = "";

  const difficultyMatch = lower.match(
    /\b(easy|medium|hard|difficult)\b/i
  );

  if (difficultyMatch) {
    difficulty = difficultyMatch[1];

    if (difficulty.toLowerCase() === "difficult") {
      difficulty = "Hard";
    } else {
      difficulty =
        difficulty.charAt(0).toUpperCase() +
        difficulty.slice(1).toLowerCase();
    }
  }

  /* -------------------------------------------------------
     TOPIC
  ------------------------------------------------------- */

  let topic = "";
  let subtopic = "";

  const onMatch = original.match(
    /\bon\s+(.+?)(?:\s*,|\s+specifically\b|\s+about\b|\s+using\b|\s+at\b|\s+with\b|\s*$)/i
  );

  const aboutMatch = original.match(
    /\babout\s+(.+?)(?:\s*,|\s+specifically\b|\s+with\b|\s+at\b|\s*$)/i
  );

  const specificallyMatch = original.match(
    /\bspecifically\s+(.+?)(?:\s*,|\s+with\b|\s+at\b|\s*$)/i
  );

  if (onMatch) {
    topic = onMatch[1].trim();
  } else if (aboutMatch) {
    topic = aboutMatch[1].trim();
  }

  if (specificallyMatch) {
    subtopic = specificallyMatch[1].trim();
  }

  /* -------------------------------------------------------
     REMOVE GENERATION PREFIXES FROM TOPIC
  ------------------------------------------------------- */

  topic = topic
    .replace(
      /^(?:generate|geerate|create|make|give(?:\s+me)?|provide|write|produce)\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)?\s*(?:questions?|qs?|q)?\s*/i,
      ""
    )
    .trim();

  /* -------------------------------------------------------
     FALLBACK TOPIC DETECTION
  ------------------------------------------------------- */

  const knownTopics = [
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "natural language processing",
    "computer networks",
    "operating systems",
    "database management systems",
    "dbms",
    "data structures",
    "algorithms",
    "decision trees",
    "random forest",
    "support vector machine",
    "svm",
    "linear regression",
    "logistic regression",
    "knn",
    "k-nearest neighbors",
    "neural networks",
    "fuzzy logic",
    "blockchain",
    "data mining",
    "data science",
  ];

  if (!topic) {
    const foundTopic = knownTopics.find((candidate) =>
      lower.includes(candidate)
    );

    if (foundTopic) {
      topic = foundTopic;
    }
  }

  if (!topic) {
    const fallback = original
      .replace(
        /^(?:generate|geerate|create|make|give(?:\s+me)?|provide|write|produce)\b/i,
        ""
      )
      .replace(
        /\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:questions?|qs?|q)\b/i,
        ""
      )
      .replace(
        /\b(?:remember|understand|apply|analyze|evaluate|create)\b/i,
        ""
      )
      .replace(
        /\b(?:easy|medium|hard|difficult)\b/i,
        ""
      )
      .replace(
        /\b(?:with|having)\s+(?:easy|medium|hard|difficult)\s+difficulty\b/i,
        ""
      )
      .trim();

    if (fallback) {
      topic = fallback;
    }
  }

  if (!subtopic && topic) {
    subtopic = topic;
  }

  return {
    topic: topic || "General Topic",
    subtopic: subtopic || topic || "General Topic",
    bloom,
    difficulty,
    numberOfQuestions,
  };
}

/* =========================================================
   GENERATION / CLASSIFICATION ROUTING
========================================================= */

function isExistingQuestion(text) {
  const cleanText = String(text || "")
    .replace(/\s+/g, " ")
    .trim();

  if (!cleanText) {
    return false;
  }

  const generationPatterns = [
    /^\s*generate\b/i,
    /^\s*geerate\b/i,
    /^\s*create\b/i,
    /^\s*make\b/i,
    /^\s*give(?:\s+me)?\b/i,
    /^\s*provide\b/i,
    /^\s*write\b/i,
    /^\s*produce\b/i,
  ];

  if (
    generationPatterns.some((pattern) =>
      pattern.test(cleanText)
    )
  ) {
    return false;
  }

  /*
     Example:
     "2 questions on Machine Learning"
     should still be generation.
  */

  const numericGeneration =
    /\b\d+\s*(?:questions?|qs?|q)\b/i.test(cleanText) &&
    /\b(?:on|about|regarding|from)\b/i.test(cleanText);

  if (numericGeneration) {
    return false;
  }

  const wordGeneration =
    /\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:questions?|qs?|q)\b/i.test(
      cleanText
    ) &&
    /\b(?:on|about|regarding|from)\b/i.test(cleanText);

  if (wordGeneration) {
    return false;
  }

  /*
     Explicit classification requests.
  */

  const classificationPatterns = [
    /\bclassify\b/i,
    /\bclassification\b/i,
    /\bwhat\s+is\s+the\s+bloom/i,
    /\bbloom(?:'s)?\s+level\b/i,
    /\bdifficulty\s+(?:level|is)\b/i,
    /\bidentify\s+(?:the\s+)?(?:bloom|difficulty)\b/i,
  ];

  if (
    classificationPatterns.some((pattern) =>
      pattern.test(cleanText)
    )
  ) {
    return true;
  }

  /*
     Strong signs of an existing question.
  */

  const questionIndicators = [
    /\?$/,
    /^what\b/i,
    /^why\b/i,
    /^how\b/i,
    /^which\b/i,
    /^when\b/i,
    /^where\b/i,
    /^define\b/i,
    /^describe\b/i,
    /^explain\b/i,
    /^calculate\b/i,
    /^compare\b/i,
    /^discuss\b/i,
    /^analyze\b/i,
    /^evaluate\b/i,
    /^derive\b/i,
    /^find\b/i,
    /^solve\b/i,
  ];

  if (
    questionIndicators.some((pattern) =>
      pattern.test(cleanText)
    )
  ) {
    return true;
  }

  /*
     If the user gives a long natural-language sentence,
     treat it as an existing question rather than generation.
  */

  const wordCount = cleanText.split(/\s+/).length;

  if (wordCount >= 15) {
    return true;
  }

  return false;
}

/* =========================================================
   CONFIDENCE HELPERS
========================================================= */

function getConfidence(value) {
  if (value === undefined || value === null) {
    return 0;
  }

  let numeric = Number(value);

  if (Number.isNaN(numeric)) {
    return 0;
  }

  if (numeric >= 0 && numeric <= 1) {
    numeric *= 100;
  }

  numeric = Math.round(numeric);

  return Math.min(Math.max(numeric, 0), 100);
}

function getConfidenceLabel(value) {
  const confidence = getConfidence(value);

  if (confidence >= 80) {
    return "High confidence";
  }

  if (confidence >= 60) {
    return "Moderate confidence";
  }

  if (confidence >= 40) {
    return "Low confidence";
  }

  return "Very low confidence";
}

/* =========================================================
   MAIN APP
========================================================= */

export default function App() {
  const [chats, setChats] = useState(() => loadChats());

  const [activeChatId, setActiveChatId] = useState(null);

  const [input, setInput] = useState("");

  const [loading, setLoading] = useState(false);

  const textareaRef = useRef(null);

  const messagesEndRef = useRef(null);

  /* -------------------------------------------------------
     INITIAL CHAT
  ------------------------------------------------------- */

  useEffect(() => {
    if (chats.length === 0) {
      const initialChat = createChat();

      setChats([initialChat]);

      setActiveChatId(initialChat.id);

      return;
    }

    if (!activeChatId) {
      setActiveChatId(chats[0].id);
    }
  }, []);

  /* -------------------------------------------------------
     SAVE CHAT HISTORY
  ------------------------------------------------------- */

  useEffect(() => {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(chats)
      );
    } catch {
      // Ignore storage errors.
    }
  }, [chats]);

  /* -------------------------------------------------------
     ACTIVE CHAT
  ------------------------------------------------------- */

  const activeChat = useMemo(() => {
    return chats.find(
      (chat) => chat.id === activeChatId
    );
  }, [chats, activeChatId]);

  const messages = activeChat?.messages || [];

  /* -------------------------------------------------------
     AUTO SCROLL
  ------------------------------------------------------- */

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({
        behavior: "smooth",
      });
    }
  }, [messages, loading]);

  /* -------------------------------------------------------
     TEXTAREA HEIGHT
  ------------------------------------------------------- */

  useEffect(() => {
    const textarea = textareaRef.current;

    if (!textarea) {
      return;
    }

    textarea.style.height = "auto";

    textarea.style.height = `${Math.min(
      textarea.scrollHeight,
      140
    )}px`;
  }, [input]);

  /* =======================================================
     CHAT MANAGEMENT
  ======================================================= */

  function updateChat(chatId, updater) {
    setChats((previousChats) =>
      previousChats.map((chat) =>
        chat.id === chatId
          ? updater(chat)
          : chat
      )
    );
  }

  function createNewChat() {
    if (loading) {
      return;
    }

    const newChat = createChat();

    setChats((previousChats) => [
      newChat,
      ...previousChats,
    ]);

    setActiveChatId(newChat.id);

    setInput("");

    setTimeout(() => {
      textareaRef.current?.focus();
    }, 50);
  }

  function deleteChat(chatId, event) {
    event?.stopPropagation();

    if (loading) {
      return;
    }

    setChats((previousChats) => {
      const remaining = previousChats.filter(
        (chat) => chat.id !== chatId
      );

      if (remaining.length === 0) {
        const replacement = createChat();

        setTimeout(() => {
          setActiveChatId(replacement.id);
        }, 0);

        return [replacement];
      }

      if (chatId === activeChatId) {
        setTimeout(() => {
          setActiveChatId(remaining[0].id);
        }, 0);
      }

      return remaining;
    });
  }

  function selectChat(chatId) {
    if (loading) {
      return;
    }

    setActiveChatId(chatId);

    setInput("");
  }

  /* =======================================================
     ADD MESSAGE
  ======================================================= */

  function addMessage(chatId, message) {
    updateChat(chatId, (chat) => ({
      ...chat,
      messages: [
        ...chat.messages,
        {
          id:
            Date.now().toString() +
            Math.random().toString(36).slice(2),
          ...message,
        },
      ],
    }));
  }

  /* =======================================================
     CLASSIFICATION API
  ======================================================= */

  async function classifyQuestion(question) {
    const response = await fetch(
      CLASSIFY_API_URL,
      {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          question,
        }),
      }
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data?.detail ||
          data?.message ||
          "Classification request failed."
      );
    }

    return data;
  }

  /* =======================================================
     GENERATION API
  ======================================================= */

  async function generateQuestions(parsed) {
    const response = await fetch(
      GENERATE_API_URL,
      {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          topic: parsed.topic,
          subtopic: parsed.subtopic,
          bloom: parsed.bloom || "",
          difficulty: parsed.difficulty || "",
          number_of_questions:
            parsed.numberOfQuestions,
        }),
      }
    );

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data?.detail ||
          data?.message ||
          "Question generation failed."
      );
    }

    return data;
  }

  /* =======================================================
     GENERATION RESULT RENDERER
  ======================================================= */

  function renderQuestionResults(data) {
    const questions = Array.isArray(data.questions)
      ? data.questions
      : [];

    /*
      Backend currently returns:

      reference_chunks: 5

      Older frontend expected:

      data.rag.results_found

      Use the new backend field first while keeping
      the old format as a fallback.
    */
    const referenceChunks =
      data.reference_chunks ??
      data.rag?.results_found ??
      0;

    const requestedNumber =
      data.requested_number ||
      data.number_requested ||
      questions.length ||
      0;

    const verifiedCount =
      data.verified_count ??
      questions.filter(
        (question) =>
          question.verified === true ||
          question.accepted === true
      ).length;

    return (
      <div className="assistant-result">
        <div className="assistant-intro">
          <div className="ai-avatar small-avatar">
            Q
          </div>

          <div>
            <div className="assistant-name">
              Question Generator
            </div>

            <div className="assistant-subtitle">
              RAG-grounded generation with QDiff
            </div>
          </div>
        </div>

        <div className="generated-results">
          <div className="result-summary">
            <div>
              <strong>
                {verifiedCount} verified ·{" "}
                {questions.length} generated
              </strong>

              <span>
                {referenceChunks} reference chunks
              </span>
            </div>

            <div className="request-tags">
              {data.requested_bloom && (
                <span>
                  {data.requested_bloom}
                </span>
              )}

              {data.requested_difficulty && (
                <span>
                  {data.requested_difficulty}
                </span>
              )}

              <span>
                {requestedNumber} question
                {requestedNumber !== 1
                  ? "s"
                  : ""}
              </span>
            </div>
          </div>

          {questions.length > 0 ? (
            <div className="chat-question-list">
              {questions.map(
                (question, index) => {
                  const questionText =
                    question.question ||
                    question.text ||
                    "Question unavailable.";

                  const bloom =
                    question.bloom ||
                    question.bloom_level ||
                    question.predicted_bloom ||
                    data.requested_bloom ||
                    "N/A";

                  const difficulty =
                    question.difficulty ||
                    question.predicted_difficulty ||
                    data.requested_difficulty ||
                    "N/A";

                  const confidence = getConfidence(
                    question.confidence ??
                      question.qdiff_confidence ??
                      question.qdiff?.confidence ??
                      question.score ??
                      0
                  );

                  const verified =
                    question.verified === true ||
                    question.accepted === true ||
                    question.status === "verified";

                  return (
                    <div
                      className="chat-question-card"
                      key={
                        question.id ||
                        `${index}-${questionText}`
                      }
                    >
                      <div className="chat-question-top">
                        <span className="question-index">
                          Q{index + 1}
                        </span>

                        {verified && (
                          <span className="accepted-badge">
                            ✓ Verified
                          </span>
                        )}
                      </div>

                      <div className="chat-question-text">
                        {questionText}
                      </div>

                      <div className="chat-question-meta">
                        <div className="chat-meta">
                          <span>
                            Bloom
                          </span>

                          <strong>
                            {bloom}
                          </strong>
                        </div>

                        <div className="chat-meta">
                          <span>
                            Difficulty
                          </span>

                          <strong>
                            {difficulty}
                          </strong>
                        </div>

                        <div className="chat-confidence">
                          <div className="chat-confidence-header">
                            <span>
                              QDiff confidence
                            </span>

                            <strong>
                              {confidence}%
                            </strong>
                          </div>

                          <div className="chat-confidence-bar">
                            <div
                              className="chat-confidence-fill"
                              style={{
                                width: `${confidence}%`,
                              }}
                            />
                          </div>

                          <small>
                            {getConfidenceLabel(
                              confidence
                            )}
                          </small>
                        </div>
                      </div>

                      {(question.grounded ||
                        question.from_rag ||
                        question.reference_grounded ||
                        question.verified) && (
                        <div className="grounded-line">
                          <span>✓</span>

                          Generated from reference
                          material
                        </div>
                      )}

                      {!verified &&
                        data.verified_count !==
                          undefined && (
                          <div className="generation-warning">
                            <span>!</span>

                            <div>
                              This question did not
                              pass the final verification
                              criteria.
                            </div>
                          </div>
                        )}
                    </div>
                  );
                }
              )}
            </div>
          ) : (
            <div className="no-results">
              <div className="generation-status-title">
                No verified questions generated
              </div>

              <div className="generation-status-message">
                The generator could not produce a
                question satisfying the requested
                Bloom level and difficulty after
                verification.
              </div>

              <div className="generation-status-grid">
                <div>
                  <span>
                    Requested
                  </span>

                  <strong>
                    {requestedNumber}
                  </strong>
                </div>

                <div>
                  <span>
                    Attempts
                  </span>

                  <strong>
                    {data.attempts ??
                      data.generation_attempts ??
                      "N/A"}
                  </strong>
                </div>

                <div>
                  <span>
                    References
                  </span>

                  <strong>
                    {referenceChunks}
                  </strong>
                </div>
              </div>

              {data.message && (
                <div className="generation-verification">
                  <div className="generation-verification-title">
                    Generation status
                  </div>

                  <p>
                    {data.message}
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    );
  }

  /* =======================================================
     CLASSIFICATION RESULT RENDERER
  ======================================================= */

  function renderClassificationResults(data) {
    const confidence = getConfidence(
      data.confidence
    );

    return (
      <div className="assistant-result">
        <div className="assistant-intro">
          <div className="ai-avatar small-avatar">
            Q
          </div>

          <div>
            <div className="assistant-name">
              Question Intelligence
            </div>

            <div className="assistant-subtitle">
              Question classification with QDiff
            </div>
          </div>
        </div>

        <div className="generated-results">
          <div className="result-summary">
            <div>
              <strong>
                Question classification
              </strong>

              <span>
                Bloom's level and difficulty prediction
              </span>
            </div>

            <div className="request-tags">
              <span>
                QDiff
              </span>
            </div>
          </div>

          {/* QUESTION */}

          <div className="classification-card">
            <div className="classification-question-label">
              Question
            </div>

            <div className="classification-question">
              {data.question}
            </div>
          </div>

          {/* CLASSIFICATION */}

          <div className="classification-card">
            <div className="classification-meta">
              <div className="classification-item">
                <span>
                  Bloom's Level
                </span>

                <strong>
                  {data.bloom_level || "N/A"}
                </strong>

                {data.bloom_confidence !==
                  undefined && (
                  <small>
                    {getConfidence(
                      data.bloom_confidence
                    )}
                    % confidence
                  </small>
                )}
              </div>

              <div className="classification-item">
                <span>
                  Difficulty
                </span>

                <strong>
                  {data.difficulty || "N/A"}
                </strong>

                {data.difficulty_confidence !==
                  undefined && (
                  <small>
                    {getConfidence(
                      data.difficulty_confidence
                    )}
                    % confidence
                  </small>
                )}
              </div>
            </div>

            {/* OVERALL CONFIDENCE */}

            <div className="classification-confidence">
              <div className="chat-confidence-header">
                <span>
                  QDiff confidence
                </span>

                <strong>
                  {confidence}%
                </strong>
              </div>

              <div className="chat-confidence-bar">
                <div
                  className="chat-confidence-fill"
                  style={{
                    width: `${confidence}%`,
                  }}
                />
              </div>

              <small>
                {getConfidenceLabel(
                  confidence
                )}
              </small>
            </div>
          </div>
        </div>
      </div>
    );
  }

  /* =======================================================
     SUBMIT
  ======================================================= */

  async function handleSubmit(event) {
    event?.preventDefault();

    const prompt = input.trim();

    if (!prompt || loading || !activeChatId) {
      return;
    }

    setInput("");

    setLoading(true);

    const chatId = activeChatId;

    const firstMessage =
      !activeChat?.messages?.length;

    addMessage(chatId, {
      role: "user",
      content: prompt,
    });

    if (firstMessage) {
      updateChat(chatId, (chat) => ({
        ...chat,
        title: createChatTitle(prompt),
      }));
    }

    try {
      const classificationMode =
        isExistingQuestion(prompt);

      console.log("Prompt mode:", {
        prompt,
        mode: classificationMode
          ? "classification"
          : "generation",
      });

      if (classificationMode) {
        const data =
          await classifyQuestion(prompt);

        console.log(
          "Classification response:",
          data
        );

        addMessage(chatId, {
          role: "assistant",
          type: "classification",
          data: {
            ...data,
            question:
              data.question || prompt,
          },
        });
      } else {
        const parsed =
          parsePrompt(prompt);

        console.log(
          "Parsed generation request:",
          parsed
        );

        const data =
          await generateQuestions(parsed);

        console.log(
          "Generation response:",
          data
        );

        addMessage(chatId, {
          role: "assistant",
          type: "generation",
          data,
        });
      }
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

      setTimeout(() => {
        textareaRef.current?.focus();
      }, 50);
    }
  }

  /* =======================================================
     ENTER KEY
  ======================================================= */

  function handleKeyDown(event) {
    if (
      event.key === "Enter" &&
      !event.shiftKey
    ) {
      event.preventDefault();

      handleSubmit(event);
    }
  }

  /* =======================================================
     EXAMPLE PROMPTS
  ======================================================= */

  function useExample(prompt) {
    if (loading) {
      return;
    }

    setInput(prompt);

    setTimeout(() => {
      textareaRef.current?.focus();
    }, 50);
  }

  const examplePrompts = [
    {
      icon: "Q",
      title: "Generate questions",
      text:
        "Generate 3 Understand questions on Decision Trees with Easy difficulty.",
    },
    {
      icon: "A",
      title: "Apply level",
      text:
        "Generate 2 Apply questions on Machine Learning, specifically K-Nearest Neighbors, with Medium difficulty.",
    },
    {
      icon: "B",
      title: "Classify a question",
      text:
        "What is the definition of entropy in a decision tree, and what does it measure?",
    },
    {
      icon: "R",
      title: "RAG grounded",
      text:
        "Generate 2 Analyze questions on Artificial Intelligence with Hard difficulty.",
    },
  ];

  /* =======================================================
     RENDER
  ======================================================= */

  return (
    <div className="chat-app">
      {/* ===================================================
          SIDEBAR
      =================================================== */}

      <aside className="sidebar">
        <div className="sidebar-top">
          <div className="sidebar-brand">
            <div className="sidebar-logo">
              Q
            </div>

            <div>
              <div className="sidebar-title">
                AI Question Generator
              </div>

              <div className="sidebar-caption">
                RAG + QDiff
              </div>
            </div>
          </div>

          <button
            type="button"
            className="new-chat-button"
            onClick={createNewChat}
            disabled={loading}
          >
            <span className="plus-icon">
              +
            </span>

            New Chat
          </button>
        </div>

        {/* =================================================
            CHAT HISTORY
        ================================================= */}

        <div className="sidebar-history">
          <div className="sidebar-history-title">
            Previous Chats
          </div>

          <div className="sidebar-history-list">
            {chats.length === 0 ? (
              <div className="sidebar-item">
                <span>—</span>
                No previous chats
              </div>
            ) : (
              chats.map((chat) => (
                <div
                  key={chat.id}
                  className={`history-chat ${
                    chat.id === activeChatId
                      ? "history-chat-active"
                      : ""
                  }`}
                  onClick={() =>
                    selectChat(chat.id)
                  }
                >
                  <div className="history-chat-icon">
                    Q
                  </div>

                  <div className="history-chat-title">
                    {chat.title}
                  </div>

                  <button
                    type="button"
                    className="history-delete-button"
                    title="Delete chat"
                    onClick={(event) =>
                      deleteChat(
                        chat.id,
                        event
                      )
                    }
                  >
                    ×
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        {/* =================================================
            SIDEBAR STATUS
        ================================================= */}

        <div className="sidebar-bottom">
          <div className="system-status">
            <span className="online-dot" />

            <div>
              <strong>
                System ready
              </strong>

              <small>
                RAG · Qwen · QDiff
              </small>
            </div>
          </div>
        </div>
      </aside>

      {/* ===================================================
          MAIN
      =================================================== */}

      <main className="chat-main">
        {/* =================================================
            HEADER
        ================================================= */}

        <header className="chat-header">
          <div className="chat-header-title">
            <div className="header-ai-icon">
              Q
            </div>

            <div>
              <strong>
                Question Intelligence
              </strong>

              <span>
                AI Question Generation & Classification
              </span>
            </div>
          </div>

          <div className="header-ready">
            <span />

            Ready
          </div>
        </header>

        {/* =================================================
            MESSAGES
        ================================================= */}

        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <div className="welcome-icon">
                Q
              </div>

              <h1>
                AI Question
                <span> Generator</span>
              </h1>

              <p>
                Generate RAG-grounded questions with
                Bloom's level and difficulty verification,
                or classify an existing question using
                QDiff.
              </p>

              <div className="example-grid">
                {examplePrompts.map(
                  (example, index) => (
                    <button
                      type="button"
                      key={index}
                      onClick={() =>
                        useExample(
                          example.text
                        )
                      }
                    >
                      <div className="example-icon">
                        {example.icon}
                      </div>

                      <div>
                        <strong>
                          {example.title}
                        </strong>

                        <small>
                          {example.text}
                        </small>
                      </div>
                    </button>
                  )
                )}
              </div>
            </div>
          ) : (
            <div className="conversation">
              {messages.map((message) => {
                if (message.role === "user") {
                  return (
                    <div
                      className="message-row user-row"
                      key={message.id}
                    >
                      <div className="message-content">
                        <div className="user-bubble">
                          {message.content}
                        </div>
                      </div>
                    </div>
                  );
                }

                return (
                  <div
                    className="message-row assistant-row"
                    key={message.id}
                  >
                    <div className="ai-avatar">
                      Q
                    </div>

                    <div className="message-content">
                      {message.type ===
                        "generation" &&
                        renderQuestionResults(
                          message.data
                        )}

                      {message.type ===
                        "classification" &&
                        renderClassificationResults(
                          message.data
                        )}

                      {message.type ===
                        "error" && (
                        <div className="error-message">
                          <strong>
                            Request failed
                          </strong>

                          <p>
                            {message.error}
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}

              {loading && (
                <div className="message-row assistant-row">
                  <div className="ai-avatar">
                    Q
                  </div>

                  <div className="typing-message">
                    <div className="typing-header">
                      <strong>
                        Question Intelligence
                      </strong>

                      <span>
                        processing
                      </span>
                    </div>

                    <div className="typing-dots">
                      <span />
                      <span />
                      <span />
                    </div>

                    <p>
                      Retrieving context and analyzing
                      the question...
                    </p>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* =================================================
            COMPOSER
        ================================================= */}

        <div className="composer-area">
          <form
            className="composer"
            onSubmit={handleSubmit}
          >
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(event) =>
                setInput(event.target.value)
              }
              onKeyDown={handleKeyDown}
              placeholder="Ask to generate questions or enter an existing question to classify..."
              disabled={loading}
              rows={1}
            />

            <button
              type="submit"
              className="send-button"
              disabled={
                loading ||
                !input.trim()
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