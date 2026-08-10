(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const esc = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = n => Number(n).toFixed(4);

  async function api(path, body) {
    const res = await fetch(path, {
      method: body ? 'POST' : 'GET',
      headers: body ? {'Content-Type':'application/json'} : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail ? JSON.stringify(data.detail) : `HTTP ${res.status}`);
    return data;
  }

  async function initStatus() {
    const host = $('#backendStatus');
    if (!host) return;
    try {
      const data = await api('/api/health');
      host.innerHTML = `<span class="status-dot ok"></span><span>Backend Python conectado · FastAPI ${esc(data.version)}</span><span class="lab-badge"><strong>/docs</strong> OpenAPI</span>`;
    } catch (e) {
      host.innerHTML = `<span class="status-dot"></span><span>Modo frontend: el backend Python no está disponible.</span>`;
    }
  }

  const configs = {
    neuron: {
      title: 'Neurona + paso de gradiente',
      fields: [
        ['x1','x₁',1,-2,2,.1], ['x2','x₂',.5,-2,2,.1], ['w1','w₁',.8,-2,2,.1],
        ['w2','w₂',-.4,-2,2,.1], ['bias','bias',.1,-2,2,.1], ['target','target',1,0,1,.1],
        ['learning_rate','learning rate',.3,.05,1,.05]
      ],
      endpoint: '/api/labs/neuron',
      render: d => `z = ${fmt(d.z)}\npredicción = ${fmt(d.prediction)}\nloss = ${fmt(d.loss)}\n\ngradiente\n  dw1 = ${fmt(d.gradient.w1)}\n  dw2 = ${fmt(d.gradient.w2)}\n  db  = ${fmt(d.gradient.bias)}\n\nparámetros después de 1 paso\n  w1 = ${fmt(d.updated.w1)}\n  w2 = ${fmt(d.updated.w2)}\n  b  = ${fmt(d.updated.bias)}`
    },
    attention: {
      title: 'Softmax y temperatura',
      fields: [['temperature','temperatura',1,.1,2,.1]],
      endpoint: '/api/labs/attention',
      body: root => ({logits:[2.4,1.1,.2,-.6], temperature:+$('#temperature',root).value}),
      renderHtml: d => {
        const labels = ['token A','token B','token C','token D'];
        return d.probabilities.map((p,i)=>`<div class="compare-row"><span>${labels[i]} ${(p*100).toFixed(1)}%</span><span class="compare-track"><i style="width:${p*100}%"></i></span></div>`).join('') + `<p class="api-note">Suma: ${d.sum.toFixed(6)}</p>`;
      }
    },
    rag: {
      title: 'Mini-RAG en Python',
      custom: true,
      endpoint: '/api/labs/rag',
      render: d => d.results.map((x,i)=>`${i+1}. ${x.title} · score ${x.score.toFixed(3)}${i < 2 ? '  ← recuperado' : ''}`).join('\n') + `\n\n${d.answer}`
    },
    quantization: {
      title: 'Memoria de pesos cuantizados',
      fields: [['params_b','parámetros (B)',7,1,70,1], ['bits','bits',4,2,32,1], ['context_k','contexto (K)',8,2,128,2], ['layers','capas',32,8,96,4]],
      endpoint: '/api/labs/quantization',
      render: d => `Pesos teóricos: ${fmt(d.weights_theoretical_gb)} GB\nPesos + overhead: ${fmt(d.weights_with_overhead_gb)} GB\nKV cache relativa: ${fmt(d.kv_cache_relative_gb)} GB\n\n${d.didactic_note}`
    }
  };

  let active = 'neuron';
  const root = $('#backendLabApp');
  if (!root) { initStatus(); return; }

  function fieldHtml(f) {
    const [id,label,value,min,max,step] = f;
    return `<label class="control"><span>${esc(label)}: <b data-v="${id}">${value}</b></span><input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${value}"></label>`;
  }

  function renderLab() {
    const cfg = configs[active];
    const controls = cfg.custom
      ? `<label class="control"><span>Consulta</span><input id="ragQueryPy" class="input" value="¿Qué diferencia hay entre RAG y un agente?"></label><label class="control"><span>Top-k</span><select id="ragTopKPy" class="select"><option>1</option><option selected>2</option><option>3</option><option>4</option></select></label>`
      : `<div class="backend-fields">${cfg.fields.map(fieldHtml).join('')}</div>`;

    root.innerHTML = `
      <div class="backend-lab-grid">
        <div class="backend-panel">
          <h3>${cfg.title}</h3>
          <p class="mini">Los controles están en JavaScript; el cálculo se envía a FastAPI y vuelve como JSON.</p>
          ${controls}
          <div class="backend-actions">
            <button id="runPythonLab" class="btn primary" type="button">Ejecutar en Python</button>
            <button id="showCode" class="btn" type="button">Ver Python vs JavaScript</button>
          </div>
          <p class="api-note">Endpoint: <code>${cfg.endpoint}</code></p>
        </div>
        <div class="backend-panel">
          <h3>Resultado del backend</h3>
          <div id="pythonResult" class="backend-output" aria-live="polite">Pulsa “Ejecutar en Python”.</div>
        </div>
      </div>
      <div id="codeCompare"></div>`;

    root.querySelectorAll('input[type="range"]').forEach(input => {
      const target = root.querySelector(`[data-v="${input.id}"]`);
      input.addEventListener('input', () => { if (target) target.textContent = input.value; });
    });
    $('#runPythonLab',root).addEventListener('click', runLab);
    $('#showCode',root).addEventListener('click', showCode);
  }

  function bodyFromFields(cfg) {
    if (cfg.body) return cfg.body(root);
    if (cfg.custom) return {query:$('#ragQueryPy',root).value, top_k:+$('#ragTopKPy',root).value};
    const body = {};
    cfg.fields.forEach(([id]) => body[id] = +$('#'+id,root).value);
    return body;
  }

  async function runLab() {
    const cfg = configs[active];
    const out = $('#pythonResult',root);
    out.textContent = 'Ejecutando…';
    try {
      const data = await api(cfg.endpoint, bodyFromFields(cfg));
      if (cfg.renderHtml) out.innerHTML = cfg.renderHtml(data);
      else out.textContent = cfg.render(data);
    } catch (e) {
      out.textContent = `Error: ${e.message}`;
    }
  }

  async function showCode() {
    const host = $('#codeCompare',root);
    try {
      const data = await api('/api/examples/'+active);
      host.innerHTML = `<div class="code-compare"><div class="code-box"><h4>Python</h4><pre><code>${esc(data.python)}</code></pre></div><div class="code-box"><h4>JavaScript</h4><pre><code>${esc(data.javascript)}</code></pre></div></div>`;
    } catch (e) {
      host.innerHTML = `<p class="api-note">No se pudo cargar el código: ${esc(e.message)}</p>`;
    }
  }

  document.querySelectorAll('[data-backend-tab]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-backend-tab]').forEach(b => b.classList.toggle('active', b === btn));
      active = btn.dataset.backendTab;
      renderLab();
    });
  });

  setTimeout(() => {
    const nav = $('#nav');
    if (nav && !nav.querySelector('[data-python-nav]')) {
      const label = document.createElement('div');
      label.className = 'nav-part';
      label.textContent = 'App';
      const link = document.createElement('button');
      link.className = 'nav-link';
      link.dataset.pythonNav = '1';
      link.textContent = 'Laboratorios Python';
      link.addEventListener('click', () => document.querySelector('#python-backend-labs')?.scrollIntoView({behavior:'smooth'}));
      nav.append(label, link);
    }
  }, 150);

  initStatus();
  renderLab();
})();
