import { useState } from "react";
import "./App.css";

function App() {
  const [topic, setTopic] = useState("");
  const [subtopic, setSubtopic] = useState("");
  const [numberOfQuestions, setNumberOfQuestions] = useState(5);

  const [questions, setQuestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const generateQuestions = async () => {
    if (!topic.trim() || !subtopic.trim()) {
      setError("Please enter both a topic and a subtopic.");
      return;
    }

    setLoading(true);
    setError("");
    setQuestions([]);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/api/generate",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            topic: topic.trim(),
            subtopic: subtopic.trim(),
            number_of_questions: Number(numberOfQuestions),
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Unable to generate questions."
        );
      }

      setQuestions(data.questions || []);
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setTopic("");
    setSubtopic("");
    setQuestions([]);
    setError("");
  };

  const getConfidence = (confidence) => {
    if (confidence === undefined || confidence === null) {
      return 0;
    }

    // Handles both 0.63 and 63
    return confidence <= 1
      ? Math.round(confidence * 100)
      : Math.round(confidence);
  };

  const getConfidenceLabel = (confidence) => {
    const value = getConfidence(confidence);

    if (value >= 75) return "High";
    if (value >= 50) return "Moderate";
    return "Low";
  };

  return (
    <div className="app">

      {/* ================= HEADER ================= */}
      <header className="header">
        <div className="header-inner">

          <div className="brand">
            <div className="brand-icon">Q</div>

            <div>
              <h1>AI Question Generator</h1>
              <p>RAG-powered intelligent question generation</p>
            </div>
          </div>

          <div className="header-status">
            <span className="status-dot"></span>
            AI System Ready
          </div>

        </div>
      </header>


      {/* ================= MAIN ================= */}
      <main className="main-container">

        {/* Intro */}
        <section className="intro">

  <div className="intro-content">

    <span className="eyebrow">
      AI-POWERED ASSESSMENT SYSTEM
    </span>

    <h2>
      Intelligent Questions.
      <br />
      <span>Grounded in Knowledge.</span>
    </h2>

    <p>
      Transform your reference material into
      high-quality, context-grounded questions
      using Retrieval-Augmented Generation,
      Large Language Models, and QDiff analysis.
    </p>

    <div className="hero-features">
      <span>✓ Reference Grounded</span>
      <span>✓ AI Generated</span>
      <span>✓ Bloom & Difficulty Analysis</span>
    </div>

  </div>

</section>


        {/* ================= GENERATOR ================= */}
        <section className="generator-card">

          <div className="card-heading">

            <div>
              <h3>Generate Questions</h3>
              <p>
                Configure the parameters below to create
                questions from your knowledge base.
              </p>
            </div>

            <div className="pipeline">
              <span>RAG</span>
              <span className="arrow">→</span>
              <span>LLM</span>
              <span className="arrow">→</span>
              <span>QDiff</span>
            </div>

          </div>


          <div className="form-grid">

            {/* Topic */}
            <div className="input-group">

              <label>
                Topic
              </label>

              <div className="input-wrapper">
                <span className="input-icon">◈</span>

                <input
                  type="text"
                  placeholder="e.g. Machine Learning"
                  value={topic}
                  onChange={(e) => setTopic(e.target.value)}
                />
              </div>

            </div>


            {/* Subtopic */}
            <div className="input-group">

              <label>
                Subtopic
              </label>

              <div className="input-wrapper">
                <span className="input-icon">◇</span>

                <input
                  type="text"
                  placeholder="e.g. Decision Trees"
                  value={subtopic}
                  onChange={(e) => setSubtopic(e.target.value)}
                />
              </div>

            </div>

          </div>


          {/* Question count */}
          <div className="bottom-form">

            <div className="input-group question-count">

              <label>
                Number of Questions
              </label>

              <select
                value={numberOfQuestions}
                onChange={(e) =>
                  setNumberOfQuestions(e.target.value)
                }
              >
                <option value="1">1 Question</option>
                <option value="2">2 Questions</option>
                <option value="3">3 Questions</option>
                <option value="5">5 Questions</option>
                <option value="10">10 Questions</option>
              </select>

            </div>


            <div className="button-group">

              <button
                className="clear-button"
                onClick={clearAll}
                disabled={loading}
              >
                Clear
              </button>

              <button
                className="generate-button"
                onClick={generateQuestions}
                disabled={loading}
              >

                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Generating...
                  </>
                ) : (
                  <>
                    Generate Questions
                    <span className="button-arrow">→</span>
                  </>
                )}

              </button>

            </div>

          </div>

        </section>


        {/* ================= ERROR ================= */}
        {error && (
          <div className="error-box">
            <div className="error-icon">!</div>

            <div>
              <strong>Generation failed</strong>
              <p>{error}</p>
            </div>
          </div>
        )}


        {/* ================= LOADING ================= */}
        {loading && (
          <div className="loading-card">

            <div className="loading-animation">
              <div></div>
              <div></div>
              <div></div>
            </div>

            <h3>Generating your questions...</h3>

            <p>
              Searching reference material and
              analyzing the generated questions.
            </p>

            <div className="loading-pipeline">
              <span>Retrieving context</span>
              <span>→</span>
              <span>Generating</span>
              <span>→</span>
              <span>Analyzing</span>
            </div>

          </div>
        )}


        {/* ================= RESULTS ================= */}
        {questions.length > 0 && !loading && (

          <section className="results-section">

            <div className="results-header">

              <div>
                <span className="eyebrow">GENERATED OUTPUT</span>

                <h2>
                  Your Questions
                </h2>

                <p>
                  {questions.length} question
                  {questions.length !== 1 ? "s" : ""} generated
                  for <strong>{subtopic}</strong>
                </p>
              </div>

              <div className="result-count">
                {questions.length}
                <span>Questions</span>
              </div>

            </div>


            <div className="question-list">

              {questions.map((question, index) => {

                const confidence =
                  getConfidence(question.confidence);

                return (
                  <article
                    className="question-card"
                    key={index}
                  >

                    <div className="question-number">
                      Q{String(index + 1).padStart(2, "0")}
                    </div>


                    <div className="question-content">

                      <p className="question-text">
                        {question.question}
                      </p>


                      <div className="question-meta">

                        <div className="meta-item">

                          <span className="meta-label">
                            Bloom's Level
                          </span>

                          <span className="badge bloom">
                            {question.bloom_level ||
                              "Not available"}
                          </span>

                        </div>


                        <div className="meta-item">

                          <span className="meta-label">
                            Difficulty
                          </span>

                          <span className="badge difficulty">
                            {question.difficulty ||
                              "Not available"}
                          </span>

                        </div>


                        <div className="meta-item confidence-item">

                          <span className="meta-label">
                            QDiff Confidence
                          </span>

                          <div className="confidence-wrapper">

                            <div className="confidence-bar">

                              <div
                                className="confidence-fill"
                                style={{
                                  width: `${confidence}%`,
                                }}
                              ></div>

                            </div>

                            <span className="confidence-value">
                              {confidence}%
                            </span>

                          </div>

                          <span className="confidence-label">
                            {getConfidenceLabel(
                              question.confidence
                            )}
                          </span>

                        </div>

                      </div>


                      <div className="grounded-badge">
                        <span className="check">✓</span>
                        Generated from reference context
                      </div>

                    </div>

                  </article>
                );
              })}

            </div>

          </section>
        )}


        {/* ================= EMPTY STATE ================= */}
        {questions.length === 0 && !loading && !error && (

          <div className="empty-state">

            <div className="empty-icon">
              ✦
            </div>

            <h3>
              Ready to generate
            </h3>

            <p>
              Enter your topic and subtopic above to
              create reference-grounded questions.
            </p>

          </div>

        )}

      </main>


      {/* ================= FOOTER ================= */}
      <footer>

        <div>
          AI Question Generator
        </div>

        <div className="footer-tech">
          RAG&nbsp;&nbsp;•&nbsp;&nbsp;LLM&nbsp;&nbsp;•&nbsp;&nbsp;QDiff
        </div>

      </footer>

    </div>
  );
}

export default App;