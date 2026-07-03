// ===================== 公共工具函数 =====================
const api = (url, opts = {}) => fetch(url, {
    ...opts,
    headers: {
        ...opts.headers,
        'Authorization': `Bearer ${localStorage.getItem('access_token') || ''}`,
    }
}).then(r => { if (!r.ok) return r.json().then(d => { throw new Error(d.detail || '请求失败') }); return r.json(); });

// ===================== 认证 =====================
async function checkAuth() {
    try {
        const user = await api('/api/auth/me');
        document.getElementById('sidebarUser').textContent = user.username;
    } catch {
        window.location.href = '/login';
    }
}

function logout() {
    localStorage.removeItem('access_token');
    api('/api/auth/logout', { method: 'POST' }).catch(() => {});
    window.location.href = '/login';
}

// ===================== 侧边栏导航 =====================
document.querySelectorAll('.nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
        e.preventDefault();
        document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
        item.classList.add('active');

        const tab = item.dataset.tab;
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        document.getElementById(`tab-${tab}`).classList.add('active');
        document.getElementById('topbarTitle').textContent = item.textContent.trim().replace(/^\S+\s*/, '');

        if (tab === 'documents') loadDocStatus();
        if (tab === 'settings') loadSystemInfo();
    });
});

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('open');
}

// ===================== 问答功能 =====================
async function sendQuestion() {
    const input = document.getElementById('questionInput');
    const btn = document.getElementById('sendBtn');
    const question = input.value.trim();
    if (!question) return;

    const stream = document.getElementById('streamToggle').checked;

    addMessage(question, 'user');
    input.value = '';
    input.style.height = 'auto';
    btn.disabled = true;
    document.getElementById('topbarStatus').textContent = '思考中...';

    if (stream) {
        await askStream(question);
    } else {
        await askNormal(question);
    }

    btn.disabled = false;
    input.focus();
}

function addMessage(text, type, meta) {
    const container = document.getElementById('chatMessages');
    document.querySelector('.chat-welcome')?.remove();

    const div = document.createElement('div');
    div.className = `message ${type}`;

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.textContent = text;
    div.appendChild(bubble);

    if (meta) {
        const metaDiv = document.createElement('div');
        metaDiv.className = 'msg-meta';
        metaDiv.textContent = meta.text || '';

        if (meta.contexts && meta.contexts.length) {
            const toggle = document.createElement('span');
            toggle.className = 'context-toggle';
            toggle.textContent = ' [查看参考来源]';
            toggle.onclick = () => {
                const ctx = div.querySelector('.msg-context');
                ctx.classList.toggle('show');
            };
            metaDiv.appendChild(toggle);

            const ctxDiv = document.createElement('div');
            ctxDiv.className = 'msg-context';
            ctxDiv.textContent = meta.contexts.map((c, i) => `[${i + 1}] ${c}`).join('\n\n');
            div.appendChild(ctxDiv);
        }

        div.appendChild(metaDiv);
    }

    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
}

async function askNormal(question) {
    try {
        const t0 = performance.now();
        const data = await api('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question, stream: false })
        });
        const elapsed = ((performance.now() - t0) / 1000).toFixed(1);

        addMessage(data.answer, 'assistant', {
            text: `检索 ${data.retrieval_time_ms.toFixed(0)}ms · 生成 ${data.generation_time_ms.toFixed(0)}ms · 总计 ${elapsed}s`,
            contexts: data.contexts
        });
        document.getElementById('topbarStatus').textContent = '就绪';
    } catch (err) {
        addMessage('错误: ' + err.message, 'user');
        document.getElementById('topbarStatus').textContent = '错误';
    }
}

async function askStream(question) {
    const msgDiv = addMessage('', 'assistant');
    const bubble = msgDiv.querySelector('.msg-bubble');
    let fullText = '';
    bubble.innerHTML = '<span class="typing-dots"><span></span><span></span><span></span></span>';

    try {
        const token = localStorage.getItem('access_token') || '';
        const resp = await fetch('/api/ask/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
            body: JSON.stringify({ question, stream: true })
        });

        if (!resp.ok) throw new Error('请求失败');

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let contexts = [];

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split('\n');
            buffer = lines.pop();
            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const data = line.slice(6);
                if (data === '[DONE]') continue;
                try {
                    const parsed = JSON.parse(data);
                    if (parsed.error) throw new Error(parsed.error);
                    if (parsed.content) {
                        if (fullText === '') bubble.textContent = '';
                        fullText += parsed.content;
                        bubble.textContent = fullText;
                        msgDiv.parentElement.scrollTop = msgDiv.parentElement.scrollHeight;
                    }
                    if (parsed.contexts) contexts = parsed.contexts;
                } catch {}
            }
        }

        if (fullText) {
            const metaDiv = msgDiv.querySelector('.msg-meta') || document.createElement('div');
            metaDiv.className = 'msg-meta';
            metaDiv.textContent = '';
            if (contexts.length) {
                const toggle = document.createElement('span');
                toggle.className = 'context-toggle';
                toggle.textContent = '[查看参考来源]';
                toggle.onclick = () => msgDiv.querySelector('.msg-context')?.classList.toggle('show');
                metaDiv.appendChild(toggle);

                const ctxDiv = document.createElement('div');
                ctxDiv.className = 'msg-context';
                ctxDiv.textContent = contexts.map((c, i) => `[${i + 1}] ${c}`).join('\n\n');
                msgDiv.appendChild(ctxDiv);
            }
            msgDiv.appendChild(metaDiv);
        }
        document.getElementById('topbarStatus').textContent = '就绪';
    } catch (err) {
        bubble.textContent = '错误: ' + err.message;
        document.getElementById('topbarStatus').textContent = '错误';
    }
}

// 自动调整 textarea 高度
document.getElementById('questionInput').addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 150) + 'px';
});

// ===================== 文档管理 =====================
async function loadDocStatus() {
    try {
        const data = await api('/api/documents/status');
        document.getElementById('statDocs').textContent = data.manifest?.total_documents ?? '-';
        document.getElementById('statChunks').textContent = data.manifest?.total_chunks ?? '-';
        document.getElementById('statVectors').textContent = data.manifest?.vector_count ?? '-';

        const tbody = document.getElementById('docTableBody');
        if (!data.documents?.length) {
            tbody.innerHTML = '<tr><td colspan="7" class="empty-hint">暂无文档，请上传</td></tr>';
            return;
        }

        tbody.innerHTML = data.documents.map(d => `
            <tr>
                <td title="${d.path || ''}">${escapeHtml(d.file)}</td>
                <td><span class="badge badge-success">${(d.type || '').replace('.','')}</span></td>
                <td>${d.chunks}</td>
                <td>${d.indexed_at ? d.indexed_at.slice(0,16).replace('T',' ') : '-'}</td>
                <td>${d.on_disk ? '<span class="badge badge-success">存在</span>' : '<span class="badge badge-danger">丢失</span>'}</td>
                <td>${d.hash_match === true ? '<span class="badge badge-success">一致</span>' : d.hash_match === false ? '<span class="badge badge-warning">变更</span>' : '<span class="badge badge-warning">未知</span>'}</td>
                <td>
                    <button class="btn btn-sm btn-outline" onclick="deleteDocument('${escapeHtml(d.file)}')" title="删除">&#128465;</button>
                </td>
            </tr>
        `).join('');
    } catch (err) {
        document.getElementById('docTableBody').innerHTML = `<tr><td colspan="7" class="empty-hint">加载失败: ${escapeHtml(err.message)}</td></tr>`;
    }
}

async function uploadFiles(input) {
    const files = Array.from(input.files);
    if (!files.length) return;

    const bar = document.getElementById('uploadProgress');
    const fill = document.getElementById('progressFill');
    bar.style.display = 'block';

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        fill.style.width = `${(i / files.length) * 100}%`;

        const formData = new FormData();
        formData.append('file', file);

        try {
            await api('/api/documents/upload', { method: 'POST', body: formData });
        } catch (err) {
            console.error('Upload failed:', file.name, err);
        }
    }

    fill.style.width = '100%';
    setTimeout(() => { bar.style.display = 'none'; fill.style.width = '0%'; }, 1000);
    input.value = '';

    setTimeout(loadDocStatus, 2000);
}

async function deleteDocument(filename) {
    if (!confirm(`确定要删除 "${filename}" 吗？此操作将从向量库中移除该文档的索引。`)) return;
    try {
        await api(`/api/documents/${encodeURIComponent(filename)}`, { method: 'DELETE' });
        await loadDocStatus();
    } catch (err) {
        alert('删除失败: ' + err.message);
    }
}

async function syncDocuments() {
    const btn = event.target;
    btn.disabled = true;
    btn.textContent = '同步中...';
    try {
        const stats = await api('/api/documents/sync', { method: 'POST' });
        alert(`同步完成: 新增 ${stats.added}, 移除 ${stats.removed}, 更新 ${stats.updated}, 跳过 ${stats.skipped}`);
        await loadDocStatus();
    } catch (err) {
        alert('同步失败: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '🔄 同步目录';
    }
}

// ===================== 设置 =====================
async function loadSystemInfo() {
    try {
        const health = await api('/api/health');
        document.getElementById('infoChain').textContent = health.rag_chain_ready ? '就绪' : '未初始化';
        document.getElementById('infoVector').textContent = health.vector_store_ready ? '就绪' : '未初始化';
        document.getElementById('infoRerank').textContent = health.rerank_available ? '可用' : '不可用';
    } catch {
        document.getElementById('infoChain').textContent = '未知';
        document.getElementById('infoVector').textContent = '未知';
        document.getElementById('infoRerank').textContent = '未知';
    }
}

document.getElementById('passwordForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const oldPw = document.getElementById('oldPassword').value;
    const newPw = document.getElementById('newPassword').value;
    const confirmPw = document.getElementById('confirmPassword').value;

    document.getElementById('pwError').style.display = 'none';
    document.getElementById('pwSuccess').style.display = 'none';

    if (newPw !== confirmPw) {
        document.getElementById('pwError').textContent = '两次输入的新密码不一致';
        document.getElementById('pwError').style.display = 'block';
        return;
    }
    if (newPw.length < 4) {
        document.getElementById('pwError').textContent = '新密码长度至少4位';
        document.getElementById('pwError').style.display = 'block';
        return;
    }

    const formData = new FormData();
    formData.append('old_password', oldPw);
    formData.append('new_password', newPw);

    try {
        await api('/api/auth/change-password', { method: 'POST', body: formData });
        document.getElementById('pwSuccess').textContent = '密码修改成功';
        document.getElementById('pwSuccess').style.display = 'block';
        document.getElementById('passwordForm').reset();
    } catch (err) {
        document.getElementById('pwError').textContent = err.message;
        document.getElementById('pwError').style.display = 'block';
    }
});

function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[m]);
}

// 初始化
checkAuth();
