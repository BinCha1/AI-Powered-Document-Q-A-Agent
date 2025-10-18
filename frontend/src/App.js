import { useState } from "react";

function App() {
  // ----------- State Variables -----------

  const [file, setFile] = useState(null); // Store the selected file for upload
  const [uploadMessage, setUploadMessage] = useState(""); // Status message after upload
  const [query, setQuery] = useState(""); // User's chat question
  const [response, setResponse] = useState(""); // Response from RAG or web search

  const backendUrl = "http://127.0.0.1:8000"; // FastAPI backend URL

  // ----------- Handle File Upload -----------
  const handleFileUpload = async (e) => {
    e.preventDefault(); // Prevent page reload on form submit

    if (!file) return alert("Select a file first");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${backendUrl}/upload`, {
        method: "POST",
        body: formData,
      });

      // If response is not OK, throw error to catch
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Upload failed");
      }

      const data = await res.json();
      setUploadMessage(
        `File uploaded successfully! Added ${data.total_chunks_added} chunks.`
      );
      setFile(null);
    } catch (err) {
      console.error(err);
      // Show the exact backend error in UI
      setUploadMessage(`File upload failed: ${err.message}`);
    }
  };

  // ----------- Handle Chat Submit -----------
  const handleChatSubmit = async (e) => {
    e.preventDefault(); // Prevent page reload

    if (!query) return; // Validate query

    setResponse("Loading..."); // Show temporary loading message

    try {
      // POST request to FastAPI /chat endpoint
      const res = await fetch(`${backendUrl}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }), // Send query as JSON
      });

      const data = await res.json();
      // Display the answer and source
      setResponse(`Answer: ${data.answer}\nSource: ${data.source}`);
      setQuery(""); // Clear chat input
    } catch (err) {
      console.error(err);
      setResponse("Error connecting to server"); // Show error if fetch fails
    }
  };

  // ----------- JSX / UI -----------

  return (
    <div
      style={{ maxWidth: "600px", margin: "50px auto", fontFamily: "Arial" }}
    >
      <h2>AgenticRAG - Upload & Chat</h2>

      {/* ----------- Step 1: File Upload ----------- */}
      <h3>Step 1: Upload Document</h3>
      <form onSubmit={handleFileUpload}>
        {/* File input */}
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
        {/* Upload button */}
        <button type="submit" style={{ marginLeft: "10px" }}>
          Upload
        </button>
      </form>
      {/* Display upload status */}
      {uploadMessage && <p>{uploadMessage}</p>}

      <hr />

      {/* ----------- Step 2: Chat ----------- */}
      <h3>Step 2: Ask a Question</h3>
      <form onSubmit={handleChatSubmit}>
        {/* Chat input */}
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question based on uploaded documents..."
          rows={4}
          style={{ width: "100%" }}
        />
        {/* Submit button */}
        <button type="submit" style={{ marginTop: "10px" }}>
          Ask
        </button>
      </form>

      {/* Display RAG / Web Search response */}
      {response && (
        <div
          style={{
            marginTop: "20px",
            padding: "10px",
            border: "1px solid #ccc",
            whiteSpace: "pre-wrap", // Preserve newlines in answer
          }}
        >
          {response}
        </div>
      )}
    </div>
  );
}

export default App;
