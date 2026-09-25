import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {

  const [topic, setTopic] = useState("");
  const [subtopic, setSubtopic] = useState("");
  const [numberOfQuestions, setNumberOfQuestions] = useState(5);

  const [questions, setQuestions] = useState([]);
  const [loading, setLoading] = useState(false);

  const generateQuestions = async () => {

    if (!topic.trim()) {
      alert("Please enter a topic");
      return;
    }

    try {

      setLoading(true);

      const response = await axios.post(
        "http://127.0.0.1:8000/api/generate",
        {
          topic: topic,
          subtopic: subtopic,
          number_of_questions: numberOfQuestions
        }
      );

      console.log("Backend response:", response.data);

      setQuestions(response.data.questions);

    } catch (error) {

      console.error("Error:", error);

      alert("Could not connect to backend.");

    } finally {

      setLoading(false);

    }
  };


  return (

    <div className="app">

      {/* HEADER */}

      <header className="header">

        <h1>
          AI Question Generator
        </h1>

        <p>
          Generate questions using RAG, LLM & QDiff
        </p>

      </header>


      {/* MAIN */}

      <main className="container">

        {/* INPUT SECTION */}

        <section className="input-card">

          <h2>
            Generate Questions
          </h2>


          {/* TOPIC */}

          <label>
            Topic
          </label>

          <input
            type="text"
            placeholder="e.g. Machine Learning"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
          />


          {/* SUBTOPIC */}

          <label>
            Subtopic
          </label>

          <input
            type="text"
            placeholder="e.g. Decision Trees"
            value={subtopic}
            onChange={(e) => setSubtopic(e.target.value)}
          />


          {/* NUMBER */}

          <label>
            Number of Questions
          </label>

          <select
            value={numberOfQuestions}
            onChange={(e) =>
              setNumberOfQuestions(Number(e.target.value))
            }
          >

            <option value={3}>3</option>
            <option value={5}>5</option>
            <option value={10}>10</option>

          </select>


          {/* BUTTON */}

          <button
            onClick={generateQuestions}
            disabled={loading}
          >

            {loading
              ? "Generating..."
              : "Generate Questions"
            }

          </button>

        </section>


        {/* RESULTS */}

        {questions.length > 0 && (

          <section className="results">

            <h2>
              Generated Questions
            </h2>


            {questions.map((item) => (

              <div
                className="question-card"
                key={item.id}
              >

                <h3>
                  Q{item.id}. {item.question}
                </h3>


                <div className="tags">

                  <span>
                    Bloom: {item.bloom_level}
                  </span>

                  <span>
                    Difficulty: {item.difficulty}
                  </span>

                  <span>
                    Confidence:{" "}
                    {(item.confidence * 100).toFixed(0)}%
                  </span>

                </div>

              </div>

            ))}

          </section>

        )}

      </main>

    </div>

  );
}

export default App;