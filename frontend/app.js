/**
 * Parrot Set 前端应用
 * 
 * 功能：
 * 1. 批量上传图片
 * 2. 调用后端 API 进行识别
 * 3. 显示识别结果
 * 4. 自动保存分类结果到指定文件夹
 * 5. Agent 聊天交互
 * 6. 分类树展示
 */

// ========== 全局变量 ==========
const API_BASE_URL = 'http://localhost:8000';
let fileList = []; // 存储文件列表
let stats = {
    total: 0,
    success: 0,
    failed: 0,
    saved: 0
};

// ========== DOM 元素 ==========
const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('upload-area');
const fileListContainer = document.getElementById('file-list');
const startBtn = document.getElementById('start-btn');
const clearBtn = document.getElementById('clear-btn');
const outputPathInput = document.getElementById('output-path');
const browseBtn = document.getElementById('browse-btn');
const autoSaveCheckbox = document.getElementById('auto-save');

// 聊天元素
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const clearChatBtn = document.getElementById('clear-chat-btn');

const progressSection = document.getElementById('progress-section');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const progressPercent = document.getElementById('progress-percent');

const resultsContainer = document.getElementById('results-section');

// 统计元素
const totalCountEl = document.getElementById('total-count');
const successCountEl = document.getElementById('success-count');
const failedCountEl = document.getElementById('failed-count');

// 状态提示元素
const statusAlert = document.getElementById('status-alert');
const statusIcon = document.getElementById('status-icon');
const statusMessage = document.getElementById('status-message');

// 分类树元素
const speciesTreeContainer = document.getElementById('species-tree');
const collectedCountEl = document.getElementById('collected-count');
const totalSpeciesCountEl = document.getElementById('total-species-count');

// 知识库元素
const kbPathInput = document.getElementById('kb-path-input');
const changeKbPathBtn = document.getElementById('change-kb-path-btn');
const refreshKbBtn = document.getElementById('refresh-kb-btn');
const kbUploadBtn = document.getElementById('kb-upload-btn');
const kbFileInput = document.getElementById('kb-file-input');
const kbDocumentsList = document.getElementById('kb-documents-list');
const clearKbBtn = document.getElementById('clear-kb-btn');
const kbDocCountEl = document.getElementById('kb-doc-count');
const kbChunkCountEl = document.getElementById('kb-chunk-count');
const kbSizeEl = document.getElementById('kb-size');

// ========== 初始化 ==========
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    loadSavedConfig();
    updateStatsUI(); // 初始化统计显示
    loadSpeciesStats(); // 加载分类树
    checkBackendHealth(); // 检查服务状态
    loadKnowledgeBase(); // 加载知识库信息
});

/**
 * 检查后端服务健康状态
 */
async function checkBackendHealth() {
    try {
        // 设置3秒超时，快速失败
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);
        
        const response = await fetch(`${API_BASE_URL}/health`, { 
            signal: controller.signal 
        });
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            throw new Error('Backend error');
        }
        
        const data = await response.json();
        
        if (!data.ollama_available) {
            showStatusAlert('warning', 'Ollama 服务未启动，模型识别功能不可用');
        } else {
            hideStatusAlert();
        }
    } catch (error) {
        showStatusAlert('error', '后端服务器未连接，请运行 "python app.py"');
    }
}

function showStatusAlert(type, message) {
    if (!statusAlert) return;
    statusAlert.classList.remove('hidden', 'error', 'warning');
    statusAlert.classList.add(type);
    statusMessage.textContent = message;
    
    // Update icon
    if (type === 'error') statusIcon.textContent = '❌';
    if (type === 'warning') statusIcon.textContent = '⚠️';
}

function hideStatusAlert() {
    if (statusAlert) statusAlert.classList.add('hidden');
}

/**
 * 初始化事件监听器
 */
function initEventListeners() {
    // 文件选择
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    
    // 拖拽上传
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // 按钮事件
    startBtn.addEventListener('click', startProcessing);
    clearBtn.addEventListener('click', clearFileList);
    browseBtn.addEventListener('click', handleBrowsePath);
    
    // 聊天事件
    sendBtn.addEventListener('click', handleSendMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    });
    clearChatBtn.addEventListener('click', clearChat);
    
    // 配置保存
    outputPathInput.addEventListener('change', saveConfig);
    outputPathInput.addEventListener('blur', saveConfig);
    autoSaveCheckbox.addEventListener('change', saveConfig);
    
    // 知识库事件
    refreshKbBtn.addEventListener('click', loadKnowledgeBase);
    kbUploadBtn.addEventListener('click', () => kbFileInput.click());
    kbFileInput.addEventListener('change', handleKbFileUpload);
    changeKbPathBtn.addEventListener('click', handleChangeKbPath);
    clearKbBtn.addEventListener('click', handleClearKnowledgeBase);
}

/**
 * 加载保存的配置
 */
function loadSavedConfig() {
    const savedPath = localStorage.getItem('outputPath');
    const savedAutoSave = localStorage.getItem('autoSave');
    
    if (savedPath) {
        outputPathInput.value = savedPath;
    }
    if (savedAutoSave !== null) {
        autoSaveCheckbox.checked = savedAutoSave === 'true';
    }
}

/**
 * 保存配置到本地存储
 */
function saveConfig() {
    localStorage.setItem('outputPath', outputPathInput.value);
    localStorage.setItem('autoSave', autoSaveCheckbox.checked);
}

// 存储选择的文件夹句柄（用于直接保存文件）
let selectedDirectoryHandle = null;

/**
 * 处理浏览路径按钮点击
 */
async function handleBrowsePath() {
    // 检查是否支持 File System Access API
    if ('showDirectoryPicker' in window) {
        try {
            const directoryHandle = await window.showDirectoryPicker({
                mode: 'readwrite',
                startIn: 'documents'
            });
            
            selectedDirectoryHandle = directoryHandle;
            const folderName = directoryHandle.name;
            
            // 提示用户输入完整路径或确认相对路径
            const message = `✅ 已选择文件夹：${folderName}\n\n` +
                `由于浏览器安全限制，无法直接获取完整路径。\n\n` +
                `请确认保存路径（如果是项目内文件夹可使用相对路径）：\n` +
                `当前路径：${outputPathInput.value || './dataset'}`;
            
            const userPath = prompt(
                message,
                outputPathInput.value || './dataset'
            );
            
            if (userPath !== null && userPath.trim()) {
                updatePath(userPath.trim());
            }
            
        } catch (error) {
            if (error.name === 'AbortError') return;
            
            if (error.name === 'NotAllowedError') {
                showNotification('需要文件夹访问权限', 'warning');
                return;
            }
            
            console.error('选择文件夹失败:', error);
            fallbackToPrompt();
        }
    } else {
        showNotification('您的浏览器不支持文件夹选择，请手动输入', 'info');
        fallbackToPrompt();
    }
}

/**
 * 回退方案：使用提示框输入路径
 */
function fallbackToPrompt() {
    const currentPath = outputPathInput.value || './dataset';
    const newPath = prompt('请输入分类保存路径：', currentPath);
    
    if (newPath !== null && newPath.trim()) {
        updatePath(newPath.trim());
    }
}

function updatePath(path) {
    if (isValidPath(path)) {
        outputPathInput.value = path;
        saveConfig();
        showNotification(`路径已更新：${path}`, 'success');
        loadSpeciesStats(); // 路径更新后刷新分类树
    } else {
        if (confirm(`路径格式可能不正确：${path}\n是否仍要使用？`)) {
            outputPathInput.value = path;
            saveConfig();
            showNotification('路径已更新（请确保路径正确）', 'warning');
            loadSpeciesStats();
        }
    }
}

/**
 * 验证路径格式（基本验证）
 */
function isValidPath(path) {
    if (!path) return false;
    if (path.startsWith('./') || path.startsWith('../')) return true;
    if (path.match(/^[A-Za-z]:[\\/]/) || path.startsWith('/') || path.startsWith('~')) return true;
    if (!path.includes('..') && !path.includes('//') && !path.includes('\\\\')) return true;
    return false;
}

/**
 * 显示通知消息
 */
function showNotification(message, type = 'info') {
    const container = document.getElementById('notification-container');
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    
    let icon = 'ℹ️';
    if (type === 'success') icon = '✅';
    if (type === 'error') icon = '❌';
    if (type === 'warning') icon = '⚠️';
    
    notification.innerHTML = `
        <span class="notification-icon">${icon}</span>
        <span class="notification-content">${message}</span>
    `;
    
    container.appendChild(notification);
    
    // 3秒后自动移除
    setTimeout(() => {
        notification.style.animation = 'slideIn 0.3s ease reverse forwards';
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
}

// ========== 文件处理 ==========

function handleFileSelect(e) {
    addFiles(Array.from(e.target.files));
    // 清空 input，允许重复选择相同文件
    e.target.value = '';
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files).filter(file => file.type.startsWith('image/'));
    addFiles(files);
}

function addFiles(files) {
    let addedCount = 0;
    files.forEach(file => {
        // 检查重复
        if (fileList.find(f => f.name === file.name && f.size === file.size)) return;
        
        fileList.push({
            id: Date.now() + Math.random(),
            file: file,
            status: 'pending', // pending, processing, success, error
            result: null,
            saved: false
        });
        addedCount++;
    });
    
    if (addedCount > 0) {
        updateFileList();
        updateStartButton();
        fileListContainer.classList.remove('hidden');
    }
}

function updateFileList() {
    fileListContainer.innerHTML = '';
    
    if (fileList.length === 0) {
        fileListContainer.classList.add('hidden');
        return;
    }
    
    fileList.forEach(item => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item'; // 可以根据状态添加更多 class
        if (item.status === 'processing') fileItem.classList.add('processing');
        if (item.status === 'success') fileItem.classList.add('success');
        if (item.status === 'error') fileItem.classList.add('error');
        
        const thumbnail = URL.createObjectURL(item.file);
        
        fileItem.innerHTML = `
            <img src="${thumbnail}" alt="${item.file.name}" class="file-thumbnail">
            <div class="file-name" title="${item.file.name}">${truncateFileName(item.file.name)}</div>
            <div class="file-status">${getStatusText(item)}</div>
            <button class="file-remove" onclick="removeFile('${item.id}')">×</button>
        `;
        
        fileListContainer.appendChild(fileItem);
    });
}

function getStatusText(item) {
    switch (item.status) {
        case 'pending': return '等待处理';
        case 'processing': return '⏳ 识别中...';
        case 'success':
            const prob = item.result?.top_candidates?.[0]?.probability || 0;
            return `✅ ${prob}%`;
        case 'error': return '❌ 失败';
        default: return '-';
    }
}

function truncateFileName(name, maxLength = 18) {
    if (name.length <= maxLength) return name;
    return name.substring(0, maxLength - 3) + '...';
}

// 暴露给全局以便 HTML onclick 调用
window.removeFile = function(id) {
    // 找到要删除的项并释放 URL 对象
    const item = fileList.find(i => i.id == id);
    if (item) {
        // 如果我们保存了 blob URL 可以在这里释放，但这里是在 updateFileList 动态生成的
    }
    
    fileList = fileList.filter(item => item.id != id);
    updateFileList();
    updateStartButton();
};

function clearFileList() {
    if (fileList.length === 0) return;
    if (confirm('确定要清空列表吗？')) {
        fileList = [];
        updateFileList();
        updateStartButton();
        resultsContainer.innerHTML = '';
        hideProgress();
        resetStats();
    }
}

function updateStartButton() {
    const hasPending = fileList.some(item => item.status === 'pending');
    startBtn.disabled = !hasPending;
}

// ========== 处理流程 ==========

async function startProcessing() {
    const pendingFiles = fileList.filter(item => item.status === 'pending');
    if (pendingFiles.length === 0) return;
    
    resetStats();
    stats.total = pendingFiles.length; // 仅统计本次批次
    updateStatsUI();
    
    showProgress();
    resultsContainer.innerHTML = ''; // 清空之前结果? 或者保留? 用户可能想保留。这里先清空。
    
    for (let i = 0; i < pendingFiles.length; i++) {
        const item = pendingFiles[i];
        await processFile(item, i + 1, pendingFiles.length);
    }
    
    hideProgress();
    showNotification(`处理完成！成功: ${stats.success}, 失败: ${stats.failed}`, 'success');
}

async function processFile(item, current, total) {
    item.status = 'processing';
    updateFileList();
    updateProgress(current, total, `正在识别: ${item.file.name}`);
    
    try {
        const result = await classifyImage(item.file);
        item.result = result;
        item.status = 'success';
        stats.success++;
        
        displayResult(item);
        
        // 自动保存
        if (autoSaveCheckbox.checked && item.result.top_candidates?.length > 0) {
            try {
                await saveClassifiedImage(item);
                item.saved = true;
                stats.saved++;
                
                // 保存成功后刷新分类树
                loadSpeciesStats();
                
            } catch (err) {
                console.error('保存失败', err);
                item.saved = false;
            }
            // 更新结果卡片显示保存状态
            updateResultCardSaveStatus(item);
        }
        
    } catch (error) {
        console.error(error);
        item.status = 'error';
        item.error = error.message;
        stats.failed++;
        displayErrorResult(item, error);
    }
    
    updateStatsUI();
    updateFileList();
}

// ========== 分类树逻辑 ==========

async function loadSpeciesStats() {
    const outputPath = outputPathInput.value.trim() || './dataset';
    try {
        const response = await fetch(`${API_BASE_URL}/stats/species?output_path=${encodeURIComponent(outputPath)}`);
        if (!response.ok) throw new Error('Failed to fetch stats');
        
        const data = await response.json();
        renderSpeciesTree(data);
    } catch (error) {
        console.error('加载分类统计失败:', error);
        // showNotification('无法加载分类统计', 'error');
    }
}

function renderSpeciesTree(data) {
    if (!speciesTreeContainer) return;
    
    // 更新统计数字
    if (collectedCountEl) collectedCountEl.textContent = data.collected_species;
    if (totalSpeciesCountEl) totalSpeciesCountEl.textContent = data.total_species;
    
    speciesTreeContainer.innerHTML = '';
    
    data.species_list.forEach(species => {
        const card = document.createElement('div');
        card.className = `species-card ${species.collected ? 'collected' : 'uncollected'}`;
        
        card.innerHTML = `
            <div class="species-icon">🦜</div>
            <div class="species-name" title="${species.name}">${species.name}</div>
            <div class="species-count">${species.count} 张</div>
        `;
        
        speciesTreeContainer.appendChild(card);
    });
}

// ========== API 调用 ==========

async function classifyImage(file) {
    const formData = new FormData();
    formData.append('image', file);
    
    const response = await fetch(`${API_BASE_URL}/classify`, {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        let msg = `HTTP ${response.status}`;
        try {
            const err = await response.json();
            msg = err.detail || msg;
        } catch(e) {}
        throw new Error(msg);
    }
    
    const result = await response.json();
    return result;
}

async function saveClassifiedImage(item) {
    const species = item.result.top_candidates[0].name;
    const outputPath = outputPathInput.value.trim() || './dataset';
    
    const formData = new FormData();
    formData.append('image', item.file);
    formData.append('species', species);
    formData.append('output_path', outputPath);
    
    const response = await fetch(`${API_BASE_URL}/save_classified`, {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        throw new Error('保存失败');
    }
    return await response.json();
}

// ========== UI 更新 ==========

function displayResult(item) {
    const result = item.result;
    const top1 = result.top_candidates?.[0];
    const thumbnail = URL.createObjectURL(item.file);
    
    const card = document.createElement('div');
    card.className = 'result-card';
    card.id = `result-${item.id}`; // 给卡片加 ID 方便后续更新
    
    card.innerHTML = `
        <img src="${thumbnail}" alt="${item.file.name}" class="result-image">
        <div class="result-top">
            <div class="result-species">${top1?.name || '未知'}</div>
            <div class="result-probability">${top1?.probability || 0}%</div>
        </div>
        
        <div class="result-candidates">
            ${result.top_candidates?.slice(0, 3).map((cand, idx) => `
                <div class="candidate-item">
                    <span class="candidate-name">${idx + 1}. ${cand.name}</span>
                    <span class="candidate-score">${cand.probability}%</span>
                </div>
            `).join('') || '<div>无候选结果</div>'}
        </div>
        
        ${result.visual_features_description ? `
            <div class="result-features">
                <div class="result-features-title">视觉特征</div>
                <div class="result-features-text">${result.visual_features_description}</div>
            </div>
        ` : ''}
        
        <div class="result-save-status hidden" id="save-status-${item.id}">
            <!-- 动态更新 -->
        </div>
    `;
    
    resultsContainer.appendChild(card);
}

function displayErrorResult(item, error) {
    const thumbnail = URL.createObjectURL(item.file);
    
    const card = document.createElement('div');
    card.className = 'result-card';
    card.style.borderColor = 'var(--error-color)';
    
    card.innerHTML = `
        <img src="${thumbnail}" class="result-image" style="opacity: 0.5">
        <div class="result-top">
            <div class="result-species" style="color: var(--error-color)">识别失败</div>
        </div>
        <div class="result-features">
            <div class="result-features-text" style="color: var(--error-color)">
                ${error.message || '未知错误'}
            </div>
        </div>
    `;
    
    resultsContainer.appendChild(card);
}

function updateResultCardSaveStatus(item) {
    const statusEl = document.getElementById(`save-status-${item.id}`);
    if (statusEl) {
        statusEl.classList.remove('hidden');
        statusEl.className = `result-save-status ${item.saved ? 'saved' : 'failed'}`;
        statusEl.textContent = item.saved ? '✓ 已归档' : '✗ 归档失败';
    }
}

function updateProgress(current, total, text) {
    const percentage = Math.round((current / total) * 100);
    progressFill.style.width = `${percentage}%`;
    progressPercent.textContent = `${percentage}%`;
    progressText.textContent = text;
}

function showProgress() {
    progressSection.classList.remove('hidden');
    progressFill.style.width = '0%';
    progressPercent.textContent = '0%';
}

function hideProgress() {
    progressSection.classList.add('hidden');
}

function resetStats() {
    stats = { total: 0, success: 0, failed: 0, saved: 0 };
    updateStatsUI();
}

function updateStatsUI() {
    totalCountEl.textContent = stats.total;
    successCountEl.textContent = stats.success;
    failedCountEl.textContent = stats.failed;
}

// ========== 聊天功能 ==========

function handleSendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    
    // 添加用户消息
    addMessage(text, 'user');
    chatInput.value = '';
    
    // 模拟 Agent 回复 (待接入后端)
    setTimeout(() => {
        let response = "抱歉，Agent 服务暂未接入后端。";
        
        if (text.includes('你好') || text.includes('hello')) {
            response = "你好！我是鹦鹉集助手。有什么我可以帮你的吗？";
        } else if (text.includes('识别') || text.includes('分类')) {
            response = "请上传图片，我会自动识别鹦鹉品种。";
        } else if (text.includes('保存') || text.includes('路径')) {
            response = "你可以在上方设置栏修改保存路径，支持自动归档功能。";
        } else {
            response = "我还在学习中，暂时无法回答这个问题。建议你尝试上传鹦鹉图片进行识别。";
        }
        
        addMessage(response, 'agent');
    }, 1000);
}

function addMessage(text, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = text;
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // 滚动到底部
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function clearChat() {
    if (confirm('确定要清空聊天记录吗？')) {
        // 保留系统欢迎消息
        const systemMsg = chatMessages.querySelector('.message.system');
        chatMessages.innerHTML = '';
        if (systemMsg) chatMessages.appendChild(systemMsg);
    }
}

// ========== 知识库管理功能 ==========

/**
 * 加载知识库信息和文档列表
 */
async function loadKnowledgeBase() {
    try {
        // 加载配置
        const configResponse = await fetch(`${API_BASE_URL}/kb/config`);
        if (configResponse.ok) {
            const config = await configResponse.json();
            if (kbPathInput) {
                kbPathInput.value = config.kb_dir || 'knowledge_base';
            }
        }
        
        // 加载统计信息
        const statsResponse = await fetch(`${API_BASE_URL}/kb/stats`);
        if (statsResponse.ok) {
            const stats = await statsResponse.json();
            if (kbDocCountEl) kbDocCountEl.textContent = stats.total_documents || 0;
            if (kbChunkCountEl) kbChunkCountEl.textContent = stats.total_chunks || 0;
            if (kbSizeEl) kbSizeEl.textContent = (stats.total_size_mb || 0).toFixed(2) + ' MB';
        }
        
        // 加载文档列表
        await loadKbDocuments();
    } catch (error) {
        console.error('加载知识库信息失败:', error);
        showNotification('加载知识库信息失败', 'error');
    }
}

/**
 * 加载知识库文档列表
 */
async function loadKbDocuments() {
    if (!kbDocumentsList) return;
    
    try {
        kbDocumentsList.innerHTML = '<div class="kb-loading">加载中...</div>';
        
        const response = await fetch(`${API_BASE_URL}/kb/documents`);
        if (!response.ok) throw new Error('获取文档列表失败');
        
        const data = await response.json();
        const documents = data.documents || [];
        
        if (documents.length === 0) {
            kbDocumentsList.innerHTML = '<div class="kb-empty">暂无文档，请上传文档到知识库</div>';
            return;
        }
        
        kbDocumentsList.innerHTML = '';
        documents.forEach(doc => {
            const docItem = document.createElement('div');
            docItem.className = 'kb-document-item';
            
            const fileIcon = getFileIcon(doc.filename);
            const fileSize = formatFileSize(doc.file_size || 0);
            const chunks = doc.chunks_count || 0;
            
            docItem.innerHTML = `
                <div class="doc-icon">${fileIcon}</div>
                <div class="doc-info">
                    <div class="doc-name" title="${doc.filename}">${doc.filename}</div>
                    <div class="doc-meta">
                        <span>${chunks} 片段</span>
                        <span>•</span>
                        <span>${fileSize}</span>
                    </div>
                </div>
                <button class="doc-delete-btn" onclick="deleteKbDocument('${doc.filename}')" title="删除">
                    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                </button>
            `;
            
            kbDocumentsList.appendChild(docItem);
        });
    } catch (error) {
        console.error('加载文档列表失败:', error);
        kbDocumentsList.innerHTML = '<div class="kb-error">加载失败: ' + error.message + '</div>';
    }
}

/**
 * 处理知识库文件上传
 */
async function handleKbFileUpload(e) {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;
    
    // 验证文件格式
    const allowedExts = ['.pdf', '.txt', '.md', '.docx', '.csv', '.xlsx', '.xls'];
    const invalidFiles = files.filter(f => {
        const ext = '.' + f.name.split('.').pop().toLowerCase();
        return !allowedExts.includes(ext);
    });
    
    if (invalidFiles.length > 0) {
        showNotification(`不支持的文件格式: ${invalidFiles.map(f => f.name).join(', ')}`, 'error');
        e.target.value = '';
        return;
    }
    
    // 保存原始按钮内容
    const originalButtonHTML = kbUploadBtn.innerHTML;
    
    // 检查按钮元素是否存在
    if (!kbUploadBtn) {
        console.error('上传按钮元素不存在');
        showNotification('上传按钮未找到', 'error');
        return;
    }
    
    try {
        kbUploadBtn.disabled = true;
        kbUploadBtn.innerHTML = '<span>上传中...</span>';
        
        console.log(`开始上传 ${files.length} 个文件`);
        const formData = new FormData();
        files.forEach((file, index) => {
            console.log(`添加文件 ${index + 1}: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`);
            formData.append('files', file);
        });
        
        // 创建超时控制器（根据文件大小动态调整超时时间）
        // 基础超时15分钟，大文件（>5MB）增加到30分钟
        const totalSizeMB = files.reduce((sum, f) => sum + f.size, 0) / 1024 / 1024;
        const timeoutMinutes = totalSizeMB > 5 ? 30 : 15;  // 增加超时时间
        const timeoutMs = timeoutMinutes * 60 * 1000;
        
        console.log(`文件总大小: ${totalSizeMB.toFixed(2)} MB, 超时时间: ${timeoutMinutes} 分钟`);
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            console.warn(`上传超时（${timeoutMinutes}分钟）`);
            controller.abort();
        }, timeoutMs);
        
        try {
            console.log(`发送请求到: ${API_BASE_URL}/kb/upload_batch`);
            const response = await fetch(`${API_BASE_URL}/kb/upload_batch`, {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            console.log(`收到响应: ${response.status} ${response.statusText}`);
            
            if (!response.ok) {
                let errorMsg = '上传失败';
                try {
                    const error = await response.json();
                    errorMsg = error.detail || errorMsg;
                    console.error('服务器错误:', error);
                } catch (e) {
                    const text = await response.text();
                    errorMsg = `HTTP ${response.status}: ${response.statusText}`;
                    console.error('响应文本:', text);
                }
                throw new Error(errorMsg);
            }
            
            const result = await response.json();
            console.log('上传结果:', result);
            
            const successCount = result.success_count || 0;
            const failedCount = result.failed_count || 0;
            
            if (successCount > 0) {
                showNotification(`成功上传 ${successCount} 个文档`, 'success');
                await loadKnowledgeBase(); // 刷新列表和统计
            }
            
            if (failedCount > 0) {
                const failedFiles = result.failed || [];
                const failedNames = failedFiles.map(f => f.filename).join(', ');
                const failedErrors = failedFiles.map(f => `${f.filename}: ${f.error}`).join('; ');
                console.warn('失败的文件:', failedErrors);
                showNotification(`${failedCount} 个文档上传失败: ${failedNames}`, 'warning');
            }
            
            if (successCount === 0 && failedCount === 0) {
                showNotification('没有文件被处理', 'warning');
            }
            
        } catch (fetchError) {
            clearTimeout(timeoutId);
            console.error('Fetch错误:', fetchError);
            
            if (fetchError.name === 'AbortError') {
                throw new Error(`上传超时（超过${timeoutMinutes}分钟），文件可能过大或处理时间较长，请稍后重试或联系管理员`);
            } else if (fetchError.name === 'TypeError' && fetchError.message.includes('Failed to fetch')) {
                throw new Error('无法连接到服务器，请检查后端服务是否运行');
            }
            throw fetchError;
        }
        
    } catch (error) {
        console.error('上传失败:', error);
        showNotification('上传失败: ' + (error.message || '未知错误'), 'error');
    } finally {
        // 确保按钮状态恢复
        if (kbUploadBtn) {
            kbUploadBtn.disabled = false;
            kbUploadBtn.innerHTML = originalButtonHTML;
        }
        if (e && e.target) {
            e.target.value = '';
        }
    }
}

/**
 * 删除知识库文档
 */
window.deleteKbDocument = async function(filename) {
    if (!confirm(`确定要删除文档 "${filename}" 吗？`)) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/kb/documents/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '删除失败');
        }
        
        showNotification(`文档 "${filename}" 已删除`, 'success');
        await loadKnowledgeBase(); // 刷新列表和统计
    } catch (error) {
        console.error('删除失败:', error);
        showNotification('删除失败: ' + error.message, 'error');
    }
}

/**
 * 处理修改知识库路径
 */
async function handleChangeKbPath() {
    const currentPath = kbPathInput.value || 'knowledge_base';
    const newPath = prompt('请输入新的知识库路径（相对路径或绝对路径）：', currentPath);
    
    if (newPath === null || !newPath.trim()) return;
    
    // 注意：由于后端知识库路径是硬编码的，这里只是保存到本地存储
    // 实际修改路径需要重启服务或修改后端配置
    localStorage.setItem('kbPath', newPath.trim());
    kbPathInput.value = newPath.trim();
    showNotification('路径已保存（需要重启服务才能生效）', 'warning');
}

/**
 * 处理清空知识库
 */
async function handleClearKnowledgeBase() {
    if (!confirm('确定要清空知识库吗？此操作不可恢复！')) return;
    
    const clearFiles = confirm('是否同时删除知识库中的文件？');
    
    try {
        const response = await fetch(`${API_BASE_URL}/kb/clear?clear_files=${clearFiles}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '清空失败');
        }
        
        showNotification('知识库已清空', 'success');
        await loadKnowledgeBase(); // 刷新列表和统计
    } catch (error) {
        console.error('清空失败:', error);
        showNotification('清空失败: ' + error.message, 'error');
    }
}

/**
 * 获取文件图标
 */
function getFileIcon(filename) {
    const ext = '.' + filename.split('.').pop().toLowerCase();
    const icons = {
        '.pdf': '📄',
        '.txt': '📝',
        '.md': '📋',
        '.docx': '📘',
        '.csv': '📊',
        '.xlsx': '📊',
        '.xls': '📊'
    };
    return icons[ext] || '📄';
}

/**
 * 格式化文件大小
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return (bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i];
}