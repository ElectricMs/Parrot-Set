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
 * 7. 图集展示 (Gallery)
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
let currentGalleryPath = null; // 当前查看的图集路径

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
const chatUploadBtn = document.getElementById('chat-upload-btn');
const chatFileInput = document.getElementById('chat-file-input');
const chatAttachmentBar = document.getElementById('chat-attachment-bar');
const chatAttachmentName = document.getElementById('chat-attachment-name');
const chatAttachmentClear = document.getElementById('chat-attachment-clear');
const chatStatus = document.getElementById('chat-status');

let chatSelectedImageFile = null; // Agent 对话中选择的图片
let lastAnalyzeResult = null; // 最近一次 analyze 的结果（用于“刚才那只...”类问题）
let agentSessionId = localStorage.getItem('agentSessionId') || null;

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

// Modal 元素
const galleryModal = document.getElementById('galleryModal');
const closeModalBtn = document.querySelector('.close-modal');
const galleryTitle = document.getElementById('galleryTitle');
const galleryGrid = document.getElementById('galleryGrid');
const galleryEmpty = document.getElementById('galleryEmpty');
const openFolderBtn = document.getElementById('openFolderBtn');

// Knowledge Base 元素
const kbUploadArea = document.getElementById('kb-upload-area');
const kbFileInput = document.getElementById('kb-file-input');
const kbListBody = document.getElementById('kb-list-body');
const kbEmpty = document.getElementById('kb-empty');
const refreshKbBtn = document.getElementById('refresh-kb-btn');
const reindexKbBtn = document.getElementById('reindex-kb-btn');
const kbStatus = document.getElementById('kb-status');

// ========== 初始化 ==========
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    loadSavedConfig();
    updateStatsUI(); // 初始化统计显示
    loadSpeciesStats(); // 加载分类树
    checkBackendHealth(); // 检查服务状态
    loadKnowledgeBase(); // 加载知识库列表
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

function setKbStatus(text) {
    if (!kbStatus) return;
    if (!text) {
        kbStatus.textContent = '';
        kbStatus.classList.add('hidden');
        return;
    }
    kbStatus.textContent = text;
    kbStatus.classList.remove('hidden');
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

    // Agent 图片上传（聊天区）
    if (chatUploadBtn && chatFileInput) {
        chatUploadBtn.addEventListener('click', () => chatFileInput.click());
        chatFileInput.addEventListener('change', handleChatFileSelect);
    }
    if (chatAttachmentClear) {
        chatAttachmentClear.addEventListener('click', clearChatAttachment);
    }
    
    // 配置保存
    outputPathInput.addEventListener('change', saveConfig);
    outputPathInput.addEventListener('blur', saveConfig);
    autoSaveCheckbox.addEventListener('change', saveConfig);

    // Modal 事件
    closeModalBtn.addEventListener('click', closeGallery);
    window.addEventListener('click', (e) => {
        if (e.target === galleryModal) {
            closeGallery();
        }
    });
    openFolderBtn.addEventListener('click', openCurrentGalleryFolder);

    // Knowledge Base 事件
    kbUploadArea.addEventListener('click', () => kbFileInput.click());
    kbFileInput.addEventListener('change', handleKbUpload);
    refreshKbBtn.addEventListener('click', loadKnowledgeBase);
    reindexKbBtn.addEventListener('click', handleKbReindex);
}

function handleChatFileSelect(e) {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    const file = files[0];
    if (!file.type.startsWith('image/')) {
        showNotification('请选择图片文件', 'warning');
        return;
    }
    chatSelectedImageFile = file;
    if (chatAttachmentName) chatAttachmentName.textContent = file.name;
    if (chatAttachmentBar) chatAttachmentBar.classList.remove('hidden');
    e.target.value = '';
}

function clearChatAttachment() {
    chatSelectedImageFile = null;
    if (chatAttachmentName) chatAttachmentName.textContent = '';
    if (chatAttachmentBar) chatAttachmentBar.classList.add('hidden');
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
    const item = fileList.find(i => i.id == id);
    if (item) {
        // 如果我们保存了 blob URL 可以在这里释放
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
    resultsContainer.innerHTML = ''; // 清空之前结果
    
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
        
        // 添加点击事件，仅对已收集的物种有效
        if (species.collected) {
            card.onclick = () => openGallery(species.name);
            card.title = "点击查看收集的照片";
        }
        
        card.innerHTML = `
            <div class="species-icon">🦜</div>
            <div class="species-name">${species.name}</div>
            <div class="species-count">${species.count} 张</div>
        `;
        
        speciesTreeContainer.appendChild(card);
    });
}

// ========== Gallery Modal 逻辑 ==========

async function openGallery(speciesName) {
    galleryTitle.textContent = speciesName;
    galleryGrid.innerHTML = '<div class="loading">加载中...</div>';
    galleryEmpty.style.display = 'none';
    galleryModal.style.display = 'block';
    
    // 获取当前配置的保存路径
    const outputPath = outputPathInput.value.trim() || './dataset';
    
    // 构建当前图集的完整路径 (这里只是简单的路径拼接，如果需要更精确的处理，可以让后端返回)
    // 为了兼容 Windows 和 Unix，我们暂时用简单的拼接，因为后端接口会处理 resolve
    currentGalleryPath = outputPath + (outputPath.endsWith('/') || outputPath.endsWith('\\') ? '' : '/') + speciesName;
    
    try {
        const response = await fetch(`${API_BASE_URL}/collection/${encodeURIComponent(speciesName)}?output_path=${encodeURIComponent(outputPath)}`);
        if (!response.ok) throw new Error('无法加载图片');
        
        const data = await response.json();
        const images = data.images || [];
        
        galleryGrid.innerHTML = '';
        
        if (images.length === 0) {
            galleryEmpty.style.display = 'block';
            return;
        }
        
        images.forEach(imgUrl => {
            const fullUrl = `${API_BASE_URL}${imgUrl}`;
            // Grid shows thumbnail
            const thumbUrl = `${fullUrl}&thumbnail=true&width=300`;
            
            const img = document.createElement('img');
            img.src = thumbUrl;
            img.className = 'gallery-item';
            img.loading = 'lazy'; // Native lazy loading
            img.onclick = () => window.open(fullUrl, '_blank'); // Click to view full image
            galleryGrid.appendChild(img);
        });
        
    } catch (error) {
        console.error('加载图集失败:', error);
        galleryGrid.innerHTML = '<div class="error">加载失败</div>';
    }
}

async function openCurrentGalleryFolder() {
    if (!currentGalleryPath) return;
    
    try {
        const formData = new FormData();
        formData.append('path', currentGalleryPath);
        
        const response = await fetch(`${API_BASE_URL}/open_folder`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('打开文件夹失败');
        }
        
        showNotification('已在资源管理器中打开', 'success');
    } catch (error) {
        console.error(error);
        showNotification('无法打开文件夹: ' + error.message, 'error');
    }
}

function closeGallery() {
    galleryModal.style.display = 'none';
    currentGalleryPath = null;
}

// ========== Knowledge Base 逻辑 ==========

async function loadKnowledgeBase() {
    try {
        const response = await fetch(`${API_BASE_URL}/knowledge/list`);
        if (!response.ok) throw new Error('Failed to list knowledge base');
        
        const data = await response.json();
        renderKbList(data.documents);
    } catch (error) {
        console.error('加载知识库失败:', error);
        showNotification('加载知识库列表失败', 'error');
    }
}

function renderKbList(documents) {
    if (!kbListBody) return;
    kbListBody.innerHTML = '';
    
    if (!documents || documents.length === 0) {
        kbEmpty.classList.remove('hidden');
        return;
    }
    
    kbEmpty.classList.add('hidden');
    
    documents.forEach(doc => {
        const tr = document.createElement('tr');
        const sizeStr = (doc.size / 1024).toFixed(1) + ' KB';
        const dateStr = new Date(doc.mtime * 1000).toLocaleString();
        
        tr.innerHTML = `
            <td>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span>📄</span>
                    <span title="${doc.name}">${doc.name}</span>
                </div>
            </td>
            <td>${sizeStr}</td>
            <td>${dateStr}</td>
            <td>
                <button class="btn-icon-small" onclick="deleteKnowledge('${doc.name}')" title="删除">
                    <svg width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                </button>
            </td>
        `;
        kbListBody.appendChild(tr);
    });
}

async function handleKbUpload(e) {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    
    // 显示处理中状态（这里简单用通知）
    showNotification(`正在上传 ${files.length} 个文件...`, 'info');
    
    let successCount = 0;
    
    for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch(`${API_BASE_URL}/knowledge/upload`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Upload failed');
            }
            
            successCount++;
        } catch (error) {
            console.error(`上传 ${file.name} 失败:`, error);
            showNotification(`上传 ${file.name} 失败: ${error.message}`, 'error');
        }
    }
    
    if (successCount > 0) {
        showNotification(`成功上传 ${successCount} 个文件`, 'success');
        loadKnowledgeBase(); // 刷新列表
    }
    
    e.target.value = ''; // 重置 input
}

// 暴露给全局以便 HTML 调用
window.deleteKnowledge = async function(filename) {
    if (!confirm(`确定要删除 "${filename}" 吗？`)) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/knowledge/${encodeURIComponent(filename)}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) throw new Error('Delete failed');
        
        showNotification(`已删除 ${filename}`, 'success');
        loadKnowledgeBase(); // 刷新列表
    } catch (error) {
        console.error('删除失败:', error);
        showNotification(`删除失败: ${error.message}`, 'error');
    }
};

async function handleKbReindex() {
    if (!confirm('重建索引可能需要一些时间，确定要继续吗？')) return;
    
    showNotification('正在向量化/同步索引（支持实时进度）…', 'info');
    setKbStatus('正在连接后端进度流…');
    reindexKbBtn.disabled = true;
    
    try {
        // SSE 流式获取进度
        const response = await fetch(`${API_BASE_URL}/knowledge/reindex/stream`, { method: 'POST' });
        if (!response.ok || !response.body) throw new Error('Reindex failed');
        
        showProgress();
        updateProgress(0, 100, '正在向量化/同步索引…');

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        let lastTotalFiles = 0;
        let lastFileIndex = 0;
        let lastChunkDone = 0;
        let lastChunkTotal = 0;
        let doneResult = null;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const parsed = parseSseChunk(buffer);
            buffer = parsed.rest;

            for (const block of parsed.complete) {
                const ev = parseSseEvent(block);
                if (ev.event === 'status') {
                    const text = ev.data && ev.data.text ? ev.data.text : String(ev.data || '');
                    setKbStatus(text);
                    continue;
                }
                if (ev.event === 'progress') {
                    const d = ev.data || {};
                    if (typeof d.total_files === 'number') lastTotalFiles = d.total_files;
                    if (typeof d.file_index === 'number') lastFileIndex = d.file_index;
                    if (typeof d.chunk_done === 'number') lastChunkDone = d.chunk_done;
                    if (typeof d.chunk_total === 'number') lastChunkTotal = d.chunk_total;

                    // 进度估算：按文件推进 + 文件内 chunk 推进
                    let percent = 0;
                    if (lastTotalFiles > 0) {
                        const within = (lastChunkTotal > 0) ? (lastChunkDone / lastChunkTotal) : 0;
                        const completedFiles = Math.max(0, (lastFileIndex - 1));
                        percent = Math.floor(((completedFiles + within) / lastTotalFiles) * 100);
                    } else if (typeof d.chunk_total_total === 'number' && typeof d.chunk_done_total === 'number' && d.chunk_total_total > 0) {
                        // full rebuild: total chunks progress
                        percent = Math.floor((d.chunk_done_total / d.chunk_total_total) * 100);
                    }
                    percent = Math.max(0, Math.min(100, percent));
                    updateProgress(percent, 100, `正在向量化… ${percent}%`);

                    if (d.file) {
                        const fileText = (d.action === 'update') ? '更新' : (d.action === 'add') ? '新增' : '处理';
                        const chunkText = (lastChunkTotal > 0) ? `（${lastChunkDone}/${lastChunkTotal}）` : '';
                        setKbStatus(`${fileText}：${d.file} ${chunkText}`.trim());
                    }
                    continue;
                }
                if (ev.event === 'done') {
                    doneResult = ev.data && ev.data.result ? ev.data.result : null;
                }
                if (ev.event === 'error') {
                    const msg = ev.data && ev.data.detail ? ev.data.detail : '未知错误';
                    throw new Error(msg);
                }
            }
        }

        hideProgress();
        setKbStatus('');

        if (doneResult && doneResult.mode === 'incremental') {
            const added = (doneResult.added || []).length;
            const modified = (doneResult.modified || []).length;
            const removed = (doneResult.removed || []).length;
            showNotification(`索引同步完成：新增${added}、更新${modified}、移除${removed}`, 'success');
        } else {
            showNotification('索引同步完成', 'success');
        }

        loadKnowledgeBase(); // 刷新列表
    } catch (error) {
        console.error('重建索引失败:', error);
        showNotification(`重建索引失败: ${error.message}`, 'error');
    } finally {
        hideProgress();
        setKbStatus('');
        reindexKbBtn.disabled = false;
    }
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

async function sendAgentMessage({ text = '', imageFile = null } = {}) {
    const formData = new FormData();
    if (agentSessionId) formData.append('session_id', agentSessionId);
    if (text) formData.append('message', text);
    if (imageFile) formData.append('image', imageFile);
    
    const response = await fetch(`${API_BASE_URL}/agent/message`, {
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
    if (result.session_id) {
        agentSessionId = result.session_id;
        localStorage.setItem('agentSessionId', agentSessionId);
    }
    return result;
}

function setChatStatus(text) {
    if (!chatStatus) return;
    if (!text) {
        chatStatus.textContent = '';
        chatStatus.classList.add('hidden');
        return;
    }
    chatStatus.textContent = text;
    chatStatus.classList.remove('hidden');
}

async function sendAgentMessageStream({ text = '', imageFile = null } = {}) {
    const formData = new FormData();
    if (agentSessionId) formData.append('session_id', agentSessionId);
    if (text) formData.append('message', text);
    if (imageFile) formData.append('image', imageFile);

    const response = await fetch(`${API_BASE_URL}/agent/message/stream`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok || !response.body) {
        let msg = `HTTP ${response.status}`;
        try {
            const err = await response.json();
            msg = err.detail || msg;
        } catch(e) {}
        throw new Error(msg);
    }

    return response.body.getReader();
}

function parseSseChunk(buffer) {
    // SSE events are separated by blank line
    const parts = buffer.split('\n\n');
    const complete = parts.slice(0, -1);
    const rest = parts[parts.length - 1];
    return { complete, rest };
}

function parseSseEvent(block) {
    // Minimal SSE parse: event + data (JSON)
    const lines = block.split('\n').filter(Boolean);
    let eventName = 'message';
    let dataLines = [];
    for (const line of lines) {
        if (line.startsWith('event:')) {
            eventName = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
            dataLines.push(line.slice(5).trim());
        }
    }
    const dataStr = dataLines.join('\n');
    let data = dataStr;
    try {
        data = JSON.parse(dataStr);
    } catch(e) {}
    return { event: eventName, data };
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
        
        ${result.confidence_level ? `
            <div class="result-features">
                 <div class="result-features-title">置信度</div>
                 <div class="result-features-text">${result.confidence_level}</div>
            </div>
        ` : ''}

        ${result.explanation ? `
             <div class="result-features">
                 <div class="result-features-title">判定依据</div>
                 <div class="result-features-text">${result.explanation}</div>
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

async function handleSendMessage() {
    const text = chatInput.value.trim();
    const hasImage = !!chatSelectedImageFile;
    if (!text && !hasImage) return;
    
    // 用户消息
    if (hasImage) {
        addImageMessage(chatSelectedImageFile, 'user');
    }
    if (text) {
    addMessage(text, 'user');
    }
    chatInput.value = '';
    
    // 统一走 /agent/message，让后端路由决定 analyze/ask/prompt
    const placeholder = addMessage(hasImage ? '正在处理…' : '正在思考…', 'agent');
    try {
        // 优先使用 SSE 流式；失败则回退非流式
        setChatStatus('思考中…');
        placeholder.textContent = '';

        let donePayload = null;
        try {
            const reader = await sendAgentMessageStream({ text, imageFile: hasImage ? chatSelectedImageFile : null });
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                const parsed = parseSseChunk(buffer);
                buffer = parsed.rest;

                for (const block of parsed.complete) {
                    const { event, data } = parseSseEvent(block);

                    if (event === 'status' && data?.text) {
                        setChatStatus(data.text);
                    } else if (event === 'tool_start') {
                        const name = data?.tool_name || 'tool';
                        setChatStatus(`正在调用 ${name}…`);
                    } else if (event === 'tool_end') {
                        const name = data?.tool_name || 'tool';
                        setChatStatus(`已完成 ${name}`);
                    } else if (event === 'token') {
                        const delta = data?.delta ?? '';
                        // internal channel tokens are for debugging (e.g. final_decision JSON); don't show to user
                        if (data?.channel === 'internal') continue;
                        if (delta) placeholder.textContent += delta;
                    } else if (event === 'done') {
                        donePayload = data;
                        // Image analyze path does not stream visible tokens (final_decision tokens are internal),
                        // so we must finalize the bubble with done.reply.
                        if ((!placeholder.textContent || !placeholder.textContent.trim()) && data?.reply) {
                            placeholder.textContent = data.reply;
                        }
                    }
                }
            }
        } catch (streamErr) {
            // Stream failed -> fallback
            console.warn('SSE stream failed, fallback to /agent/message:', streamErr);
            setChatStatus('（流式不可用，已回退普通模式）');
            const resp = await sendAgentMessage({ text, imageFile: hasImage ? chatSelectedImageFile : null });
            placeholder.textContent = resp.reply || '（无回复）';
            donePayload = resp;
        }
        
        // 处理 done：更新 session_id、清除状态、缓存 analyze artifacts
        if (donePayload?.session_id) {
            agentSessionId = donePayload.session_id;
            localStorage.setItem('agentSessionId', agentSessionId);
        }
        setChatStatus('');

        const mode = donePayload?.mode;
        const artifacts = donePayload?.artifacts;
        if (mode === 'analyze' && artifacts) {
            lastAnalyzeResult = artifacts;
        }
    } catch (err) {
        placeholder.textContent = `请求失败：${err.message || err}`;
        setChatStatus('');
    } finally {
        if (hasImage) clearChatAttachment();
    }
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

    return contentDiv;
}

function addImageMessage(file, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.alt = file.name;
    img.style.maxWidth = '240px';
    img.style.borderRadius = '12px';
    img.style.display = 'block';
    
    const caption = document.createElement('div');
    caption.textContent = file.name;
    caption.style.marginTop = '0.5rem';
    caption.style.fontSize = '0.8rem';
    caption.style.opacity = '0.8';
    
    contentDiv.appendChild(img);
    contentDiv.appendChild(caption);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
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
