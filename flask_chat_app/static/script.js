// Endpoint of the Flask backend's /chatbot route.
// Local/standard setup: http://127.0.0.1:5000/chatbot
const CHATBOT_ENDPOINT = "http://127.0.0.1:5000/chatbot";

const chatLog = document.getElementById("chat-log");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

function appendMessage(sender, text) {
  const el = document.createElement("div");
  el.className = "msg " + sender;
  el.textContent = (sender === "user" ? "You: " : "Bot: ") + text;
  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
}

async function sendMessage() {
  const prompt = userInput.value.trim();
  if (!prompt) return;

  appendMessage("user", prompt);
  userInput.value = "";

  try {
    const res = await fetch(CHATBOT_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt }),
    });
    const text = await res.text();
    appendMessage("bot", text);
  } catch (err) {
    appendMessage("bot", "Error reaching the server: " + err);
  }
}

sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});
