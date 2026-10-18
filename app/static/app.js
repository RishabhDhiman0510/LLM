const state = {
  chats: [],
  activeId: null,
  busy: false,
};

const elements = {
  chatList: document.getElementById("chat-list"),
  messages: document.getElementById("messages"),
  composer: document.getElementById("composer"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  newChat: document.getElementById("new-chat"),
};

async function request(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return response.status === 204 ? null : response.json();
}

function renderChatList() {
  const items = state.chats.map((chat) => {
    const item = document.createElement("li");
    item.className = chat.id === state.activeId ? "chat-item active" : "chat-item";

    const title = document.createElement("button");
    title.type = "button";
    title.className = "chat-title";
    title.textContent = chat.title || "New chat";
    title.addEventListener("click", () => openChat(chat.id));

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "chat-delete";
    remove.textContent = "Delete";
    remove.addEventListener("click", () => deleteChat(chat.id));

    item.append(title, remove);
    return item;
  });
  elements.chatList.replaceChildren(...items);
}

function renderMessages(messages) {
  elements.messages.replaceChildren();
  if (messages.length === 0) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "Start a new conversation.";
    elements.messages.append(empty);
    return;
  }
  messages.forEach((message) => appendMessage(message.role, message.content));
}

function appendMessage(kind, text) {
  elements.messages.querySelector(".empty")?.remove();
  const row = document.createElement("div");
  row.className = text ? `message message-${kind}` : `message message-${kind} message-blank`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text || "No reply generated.";
  row.append(bubble);
  elements.messages.append(row);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return row;
}

async function loadChats() {
  state.chats = await request("/chats");
  renderChatList();
}

async function openChat(id) {
  if (state.busy) return;
  const chat = await request(`/chats/${id}`);
  state.activeId = id;
  renderChatList();
  renderMessages(chat.messages);
  elements.input.focus();
}

function startNewChat() {
  if (state.busy) return;
  state.activeId = null;
  renderChatList();
  renderMessages([]);
  elements.input.focus();
}

async function deleteChat(id) {
  if (state.busy || !window.confirm("Delete this chat?")) return;
  await request(`/chats/${id}`, { method: "DELETE" });
  if (id === state.activeId) startNewChat();
  await loadChats();
}

function setBusy(busy) {
  state.busy = busy;
  elements.send.disabled = busy;
}

async function sendMessage(text) {
  setBusy(true);
  appendMessage("user", text);
  const pending = appendMessage("pending", "Generating...");
  try {
    if (state.activeId === null) {
      const chat = await request("/chats", { method: "POST" });
      state.activeId = chat.id;
    }
    const reply = await request(`/chats/${state.activeId}/messages`, {
      method: "POST",
      body: JSON.stringify({ content: text }),
    });
    pending.remove();
    appendMessage("assistant", reply.content);
  } catch (error) {
    pending.remove();
    appendMessage("error", "Could not get a reply. Check the terminal where the app is running.");
    console.error(error);
  } finally {
    setBusy(false);
    await loadChats().catch(console.error);
  }
}

function resizeInput() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${elements.input.scrollHeight}px`;
}

elements.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = elements.input.value.trim();
  if (!text || state.busy) return;
  elements.input.value = "";
  resizeInput();
  sendMessage(text);
});

elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    elements.composer.requestSubmit();
  }
});

elements.input.addEventListener("input", resizeInput);
elements.newChat.addEventListener("click", startNewChat);

renderMessages([]);
loadChats();
