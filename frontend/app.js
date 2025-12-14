/**
 * Parrot Set 前端应用
 * 
 * 功能：
 * 1. 批量上传图片
 * 2. 调用后端 API 进行识别
 * 3. 显示识别结果
 * 4. 自动保存分类结果到指定文件夹
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

const progressSection = document.getElementById('progress-section');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const progressPercent = document.getElementById('progress-percent');

const resultsContainer = document.getElementById('results-section');

// 统计元素
const totalCountEl = document.getElementById('total-count');
const successCountEl = document.getElementById('success-count');
const failedCountEl = document.getElementById('failed-count');

// ========== 初始化 ==========
document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    loadSavedConfig();
    updateStatsUI(); // 初始化统计显示
});

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
    
    // 配置保存
    outputPathInput.addEventListener('change', saveConfig);
    outputPathInput.addEventListener('blur', saveConfig);
    autoSaveCheckbox.addEventListener('change', saveConfig);
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
    } else {
        if (confirm(`路径格式可能不正确：${path}\n是否仍要使用？`)) {
            outputPathInput.value = path;
            saveConfig();
            showNotification('路径已更新（请确保路径正确）', 'warning');
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
