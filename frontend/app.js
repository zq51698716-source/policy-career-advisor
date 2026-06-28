/**
 * 政策通 & 职业顾问 - 前端交互逻辑
 * 功能：SSE 流式接收、Markdown 渲染、聊天管理
 */

// ===== DOM 元素 =====
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');
const menuToggle = document.getElementById('menuToggle');
const sidebar = document.getElementById('sidebar');
const statusIndicator = document.getElementById('statusIndicator');
const welcomeMessage = document.querySelector('.welcome-message');
// RAG 元素
const modeSearch = document.getElementById('modeSearch');
const modeRAG = document.getElementById('modeRAG');
const quickActions = document.getElementById('quickActions');
const ragPanel = document.getElementById('ragPanel');
const pdfUpload = document.getElementById('pdfUpload');
const uploadStatus = document.getElementById('uploadStatus');
const docList = document.getElementById('docList');

// ===== 状态管理 =====
let isStreaming = false;
let chatHistory = [];
let currentAssistantBubble = null;
let currentContent = '';
let currentMode = 'search'; // 'search' | 'rag'

// ===== 初始化 =====
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    autoResizeTextarea();
    checkServerStatus();
});

function setupEventListeners() {
    // 发送按钮
    sendBtn.addEventListener('click', handleSend);

    // 输入框键盘事件
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    // 自动调整输入框高度
    chatInput.addEventListener('input', autoResizeTextarea);

    // 新对话按钮
    newChatBtn.addEventListener('click', startNewChat);

    // 模式切换
    modeSearch.addEventListener('click', () => switchMode('search'));
    modeRAG.addEventListener('click', () => switchMode('rag'));

    // PDF 上传
    pdfUpload.addEventListener('click', () => {}); // trigger file dialog
    pdfUpload.addEventListener('change', handleFileUpload);

    // 快捷入口按钮
    document.querySelectorAll('.quick-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const query = btn.dataset.query;
            chatInput.value = query;
            handleSend();
            // 移动端关闭侧边栏
            if (window.innerWidth <= 768) {
                sidebar.classList.remove('open');
            }
        });
    });

    // 移动端菜单切换
    menuToggle.addEventListener('click', () => {
        sidebar.classList.toggle('open');
    });

    // 点击侧边栏外部关闭
    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 &&
            !sidebar.contains(e.target) &&
            e.target !== menuToggle) {
            sidebar.classList.remove('open');
        }
    });
}

// ===== 核心功能 =====

async function handleSend() {
    if (isStreaming) return;

    const query = chatInput.value.trim();
    if (!query) return;

    // RAG 模式使用 RAG 接口
    if (currentMode === 'rag') {
        await handleRAGChat(query);
        return;
    }

    // 隐藏欢迎消息
    if (welcomeMessage) {
        welcomeMessage.style.display = 'none';
    }

    // 清空输入框
    chatInput.value = '';
    autoResizeTextarea();

    // 添加用户消息
    addMessage('user', query);
    chatHistory.push({ role: 'user', content: query });

    // 开始流式请求
    isStreaming = true;
    updateUIState('streaming');
    currentContent = '';

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                history: chatHistory.slice(0, -1) // 不包含刚才添加的 query
            })
        });

        if (!response.ok) {
            throw new Error(`请求失败 (${response.status})`);
        }

        // 读取 SSE 流
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // 解析 SSE 事件
            const lines = buffer.split('\n');
            buffer = lines.pop() || ''; // 保留未完成的行

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const event = JSON.parse(line.substring(6));
                        handleSSEEvent(event);
                    } catch (e) {
                        console.warn('SSE 解析失败:', line);
                    }
                }
            }
        }
    } catch (error) {
        handleError(error.message);
    } finally {
        isStreaming = false;
        updateUIState('idle');
    }
}

function handleSSEEvent(event) {
    const { type, data } = event;

    switch (type) {
        case 'thinking':
            showThinkingIndicator(data);
            break;

        case 'tool_call':
            try {
                const tc = typeof data === 'string' ? JSON.parse(data) : data;
                showToolCallIndicator(tc.name);
            } catch (e) {
                showToolCallIndicator('搜索中');
            }
            break;

        case 'tool_result':
            try {
                const tr = typeof data === 'string' ? JSON.parse(data) : data;
                showToolCallIndicator(`${tr.tool_name}: ${tr.summary || '完成'}`);
            } catch (e) {
                // ignore
            }
            break;

        case 'content':
            appendToAssistantBubble(data);
            break;

        case 'done':
            finalizeAssistantBubble();
            removeThinkingIndicators();
            break;

        case 'error':
            handleError(data);
            break;
    }
}

// ===== UI 更新 =====

function updateUIState(state) {
    switch (state) {
        case 'streaming':
            sendBtn.disabled = true;
            chatInput.disabled = true;
            statusIndicator.innerHTML = '🟡 思考中...';
            break;
        case 'idle':
            sendBtn.disabled = false;
            chatInput.disabled = false;
            chatInput.focus();
            statusIndicator.innerHTML = '🟢 在线';
            break;
        case 'error':
            sendBtn.disabled = false;
            chatInput.disabled = false;
            statusIndicator.innerHTML = '🔴 异常';
            setTimeout(() => {
                statusIndicator.innerHTML = '🟢 在线';
            }, 3000);
            break;
    }
}

function showThinkingIndicator(text) {
    removeThinkingIndicators();

    const indicator = document.createElement('div');
    indicator.className = 'thinking-indicator';
    indicator.innerHTML = `
        <div class="thinking-dots">
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
        </div>
        <span>${escapeHtml(text)}</span>
    `;
    chatMessages.appendChild(indicator);
    scrollToBottom();
}

function showToolCallIndicator(toolName) {
    removeThinkingIndicators();

    // 移除上一个工具调用指示器（保留最新的）
    const existing = document.querySelectorAll('.tool-call-indicator');
    if (existing.length > 0) {
        existing[existing.length - 1].remove();
    }

    const indicator = document.createElement('div');
    indicator.className = 'tool-call-indicator';
    indicator.textContent = `🔍 ${escapeHtml(toolName)}`;
    chatMessages.appendChild(indicator);
    scrollToBottom();
}

function removeThinkingIndicators() {
    document.querySelectorAll('.thinking-indicator').forEach(el => el.remove());
    document.querySelectorAll('.tool-call-indicator').forEach(el => el.remove());
}

function appendToAssistantBubble(content) {
    if (!currentAssistantBubble) {
        // 创建新的 assistant 气泡
        currentAssistantBubble = createMessageBubble('assistant', '');
        currentContent = '';
    }

    currentContent += content;
    currentAssistantBubble.querySelector('.message-content').innerHTML =
        renderMarkdown(currentContent);
    scrollToBottom();
}

function finalizeAssistantBubble() {
    if (currentAssistantBubble && currentContent) {
        chatHistory.push({ role: 'assistant', content: currentContent });
    }
    currentAssistantBubble = null;
    currentContent = '';
}

function addMessage(role, content) {
    const bubble = createMessageBubble(role, content);
    chatMessages.appendChild(bubble);
    scrollToBottom();
}

function createMessageBubble(role, content) {
    const div = document.createElement('div');
    div.className = `message ${role}`;

    const avatar = role === 'user' ? '👤' : '🤖';
    const avatarName = role === 'user' ? '你' : 'AI';

    div.innerHTML = `
        <div class="message-avatar" title="${avatarName}">${avatar}</div>
        <div class="message-content">${renderMarkdown(content)}</div>
    `;

    chatMessages.appendChild(div);
    return div;
}

function renderMarkdown(text) {
    if (!text) return '';
    try {
        // 配置 marked
        if (typeof marked !== 'undefined') {
            marked.setOptions({
                breaks: true,
                gfm: true,
            });
            return marked.parse(text);
        }
    } catch (e) {
        console.warn('Markdown 渲染失败:', e);
    }
    // 降级：简单换行处理
    return escapeHtml(text).replace(/\n/g, '<br>');
}

function handleError(message) {
    removeThinkingIndicators();
    currentAssistantBubble = null;
    currentContent = '';
    updateUIState('error');

    const errorDiv = document.createElement('div');
    errorDiv.className = 'message assistant';
    errorDiv.innerHTML = `
        <div class="message-avatar">⚠️</div>
        <div class="message-content" style="color: var(--error);">
            出错了：${escapeHtml(message)}<br>
            <small>请检查 API 配置或稍后重试</small>
        </div>
    `;
    chatMessages.appendChild(errorDiv);
    scrollToBottom();
}

// ===== 辅助功能 =====

function startNewChat() {
    chatHistory = [];
    currentAssistantBubble = null;
    currentContent = '';

    // 清空消息区
    chatMessages.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">🤖</div>
            <h2>你好！我是你的 AI 政策 &amp; 职业顾问</h2>
            <p>我可以帮你：</p>
            <ul>
                <li>🔍 <strong>搜索最新政策</strong> — 就业、人才、补贴、社保，自动查最新</li>
                <li>📖 <strong>解读官方文件</strong> — 复杂政策说人话，告诉你该怎么申请</li>
                <li>💼 <strong>分析行业趋势</strong> — 薪资水平、岗位需求、技能要求</li>
                <li>📝 <strong>优化简历</strong> — 针对岗位给出具体修改建议</li>
                <li>🎯 <strong>面试辅导</strong> — 常见问题、回答思路、模拟练习</li>
            </ul>
            <p class="welcome-hint">👇 直接提问，或点击左侧快捷入口开始</p>
        </div>
    `;

    chatInput.focus();
}

function autoResizeTextarea() {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 150) + 'px';
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function checkServerStatus() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();

        if (!data.claude_configured || !data.tavily_configured) {
            const missing = [];
            if (!data.claude_configured) missing.push('ANTHROPIC_API_KEY');
            if (!data.tavily_configured) missing.push('TAVILY_API_KEY');

            showConfigWarning(missing);
        }
    } catch (e) {
        // 服务器不可用
        statusIndicator.innerHTML = '🔴 离线';
    }
}

function showConfigWarning(missingKeys) {
    const warning = document.createElement('div');
    warning.style.cssText = `
        position: fixed;
        top: 16px;
        left: 50%;
        transform: translateX(-50%);
        background: var(--error);
        color: white;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 14px;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        max-width: 500px;
        text-align: center;
    `;
    warning.innerHTML = `
        ⚠️ 缺少配置：<strong>${missingKeys.join(', ')}</strong><br>
        <small>请复制 .env.example 为 .env 并填入你的 API Keys</small>
    `;

    document.body.appendChild(warning);
    setTimeout(() => warning.remove(), 10000);
}

// ===== RAG 功能 =====

function switchMode(mode) {
    currentMode = mode;
    if (mode === 'search') {
        modeSearch.classList.add('active');
        modeRAG.classList.remove('active');
        quickActions.style.display = '';
        ragPanel.style.display = 'none';
        chatInput.placeholder = '输入你的问题，例如：深圳2026年人才补贴政策...';
        startNewChat();
    } else {
        modeRAG.classList.add('active');
        modeSearch.classList.remove('active');
        quickActions.style.display = 'none';
        ragPanel.style.display = '';
        chatInput.placeholder = '基于已上传的政策文件提问...';
        loadDocuments();
        startNewChat();
    }
}

async function handleFileUpload(e) {
    const files = e.target.files;
    if (!files.length) return;

    uploadStatus.textContent = '正在处理...';
    uploadStatus.className = 'upload-status';

    for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const resp = await fetch('/api/rag/upload', { method: 'POST', body: formData });
            const data = await resp.json();
            if (data.success) {
                uploadStatus.textContent = `✅ ${file.name} — ${data.chunk_count} 个文本块`;
                uploadStatus.className = 'upload-status success';
            } else {
                uploadStatus.textContent = `❌ ${file.name}: ${data.error}`;
                uploadStatus.className = 'upload-status error';
            }
        } catch (err) {
            uploadStatus.textContent = `❌ 上传失败: ${err.message}`;
            uploadStatus.className = 'upload-status error';
        }
    }

    pdfUpload.value = '';
    loadDocuments();
}

async function loadDocuments() {
    try {
        const resp = await fetch('/api/rag/documents');
        const data = await resp.json();
        if (data.documents.length === 0) {
            docList.innerHTML = '<p class="empty-hint">暂无文档，上传 PDF 开始</p>';
        } else {
            docList.innerHTML = data.documents.map(d => `
                <div class="doc-item">
                    <span class="doc-item-name" title="${escapeHtml(d.filename)}">📄 ${escapeHtml(d.filename)}</span>
                    <span class="doc-item-meta">${d.chunk_count} 块</span>
                    <button class="doc-item-del" onclick="deleteDoc('${d.doc_id}')">✕</button>
                </div>
            `).join('');
        }
    } catch (e) {
        docList.innerHTML = '<p class="empty-hint">加载失败</p>';
    }
}

async function deleteDoc(docId) {
    await fetch(`/api/rag/documents/${docId}`, { method: 'DELETE' });
    loadDocuments();
}

async function handleRAGChat(query) {
    if (isStreaming) return;
    if (welcomeMessage) welcomeMessage.style.display = 'none';

    chatInput.value = '';
    autoResizeTextarea();
    addMessage('user', query);
    chatHistory.push({ role: 'user', content: query });

    isStreaming = true;
    updateUIState('streaming');
    currentContent = '';

    try {
        const response = await fetch('/api/rag/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, top_k: 5 })
        });

        if (!response.ok) throw new Error(`请求失败 (${response.status})`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const event = JSON.parse(line.substring(6));
                        handleSSEEvent(event);
                    } catch (e) {}
                }
            }
        }
    } catch (error) {
        handleError(error.message);
    } finally {
        isStreaming = false;
        updateUIState('idle');
    }
}
