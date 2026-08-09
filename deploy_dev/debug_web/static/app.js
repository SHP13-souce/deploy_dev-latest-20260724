const $ = selector => document.querySelector(selector);
const connection = $('#connection');
const raw = $('#raw');
const history = [];
let lastFrame = null;
let lastReceivedAt = 0;
const parameterDefinitions = [
  ['conf_threshold', '检测置信度', 0.01], ['binary_threshold', '二值阈值', 1],
  ['bullet_speed', '弹速 m/s', 0.1], ['yaw_offset', 'Yaw 偏置 °', 0.05],
  ['pitch_offset', 'Pitch 偏置 °', 0.05], ['comming_angle', '进入角 °', 0.5],
  ['leaving_angle', '离开角 °', 0.5], ['decision_speed', '转速阈值 rad/s', 0.1],
  ['high_speed_delay_time', '高速延时 s', 0.001], ['low_speed_delay_time', '低速延时 s', 0.001],
];

const valueAt = (data, path) => path.split('.').reduce((value, key) => value?.[key], data);
const fixed = (value, digits = 2) => Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '-';
const yesNo = value => value ? '是' : '否';
const degrees = value => Number.isFinite(Number(value)) ? `${(Number(value) * 180 / Math.PI).toFixed(2)}°` : '-';

function metric(label, value, tone = '') {
  return `<div class="metric ${tone}"><small>${label}</small><strong>${value}</strong></div>`;
}

function renderCards(data) {
  const tracker = data.tracker ?? {};
  const aim = data.aim ?? {};
  const gate = data.fire_gate ?? {};
  const latency = data.latency_ms ?? {};
  const cards = [
    ['跟踪', tracker.state_name ?? '-', tracker.target_valid ? 'good' : ''],
    ['检测', data.detection?.count ?? 0, data.detection?.count ? 'good' : ''],
    ['目标', yesNo(tracker.target_valid), tracker.target_valid ? 'good' : ''],
    ['瞄准', yesNo(aim.valid), aim.valid ? 'good' : ''],
    ['开火门', gate.allowed ? '允许' : '拒绝', gate.allowed ? 'danger' : 'safe'],
    ['总延迟', `${fixed(latency.total)} ms`, Number(latency.total) > 10 ? 'warn' : 'good'],
  ];
  $('#core-cards').innerHTML = cards.map(([label, value, tone]) => metric(label, value, tone)).join('');
}

function renderSections(data) {
  const detection = data.detection ?? {};
  const pnp = data.pnp ?? {};
  const tracker = data.tracker ?? {};
  const state = data.state ?? {};
  const aim = data.aim ?? {};
  const planner = data.planner ?? {};
  const gimbal = data.gimbal ?? {};
  const gate = data.fire_gate ?? {};
  $('#vision-grid').innerHTML = [
    metric('最高置信度', fixed(detection.confidence_max, 3)),
    metric('PnP 有效 / 拒绝', `${pnp.valid_count ?? 0} / ${pnp.rejected_count ?? 0}`),
    metric('平均重投影', `${fixed(pnp.reprojection_mean_px, 3)} px`),
    metric('最大重投影', `${fixed(pnp.reprojection_max_px, 3)} px`),
    metric('NIS', fixed(tracker.nis, 3), Number(tracker.nis) > 13.277 ? 'danger' : ''),
    metric('NIS 失败率', `${fixed(Number(tracker.nis_failure_ratio) * 100, 1)}%`),
  ].join('');
  $('#state-grid').innerHTML = [
    ['X', state.x, 'm'], ['Vx', state.vx, 'm/s'], ['Y', state.y, 'm'], ['Vy', state.vy, 'm/s'],
    ['Z', state.z, 'm'], ['Vz', state.vz, 'm/s'], ['Yaw', state.yaw, 'rad'], ['Yaw rate', state.yaw_rate, 'rad/s'],
    ['R1', state.radius_1, 'm'], ['ΔR', state.radius_delta, 'm'], ['ΔH', state.height_delta, 'm'],
  ].map(([label, value, unit]) => metric(label, `${fixed(value, 3)} ${unit}`)).join('');
  $('#aim-grid').innerHTML = [
    metric('瞄准 Yaw', degrees(aim.yaw_rad)), metric('瞄准 Pitch', degrees(aim.pitch_rad)),
    metric('飞行时间', `${fixed(Number(aim.fly_time_s) * 1000, 1)} ms`),
    metric('拒火原因', gate.reason ?? '-', gate.allowed ? 'good' : 'safe'),
    metric('反馈 Yaw', gimbal.feedback_valid ? `${fixed(gimbal.yaw_deg)}°` : '无反馈'),
    metric('反馈 Pitch', gimbal.feedback_valid ? `${fixed(gimbal.pitch_deg)}°` : '无反馈'),
    metric('MPC', planner.valid ? '有效' : '关闭 / 无效', planner.valid ? 'good' : ''),
    metric('姿态', data.frame?.pose_valid ? '有效' : '静态 / 缺失', data.frame?.pose_valid ? 'good' : 'warn'),
  ].join('');
}

function renderLatency(data) {
  const values = data.latency_ms ?? {};
  const rows = [['检测', values.detection], ['PnP', values.pnp], ['跟踪', values.tracker], ['瞄准', values.aim]];
  const max = Math.max(1, ...rows.map(([, value]) => Number(value) || 0));
  $('#latency-bars').innerHTML = rows.map(([label, value]) =>
    `<div class="bar-row"><span>${label}</span><i><b style="width:${Math.min(100, (Number(value) || 0) / max * 100)}%"></b></i><em>${fixed(value, 3)} ms</em></div>`
  ).join('');
}

function drawChart() {
  const canvas = $('#chart');
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const context = canvas.getContext('2d');
  context.scale(ratio, ratio);
  context.clearRect(0, 0, width, height);
  context.strokeStyle = '#26343b';
  context.lineWidth = 1;
  for (let line = 1; line < 4; ++line) {
    const y = line * height / 4;
    context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke();
  }
  if (history.length < 2) return;
  const max = Math.max(1, ...history.flatMap(point => [point.total, point.detect, point.nis]));
  [['total', '#65d9bb'], ['detect', '#f2b84b'], ['nis', '#d678ff']].forEach(([key, color]) => {
    context.strokeStyle = color; context.lineWidth = 2; context.beginPath();
    history.forEach((point, index) => {
      const x = index * width / (history.length - 1);
      const y = height - Math.min(height, point[key] / max * (height - 8)) - 4;
      index ? context.lineTo(x, y) : context.moveTo(x, y);
    });
    context.stroke();
  });
}

function setViews() {
  const container = $('#views');
  const enabled = [...document.querySelectorAll('[data-view]:checked')].map(input => input.dataset.view);
  container.replaceChildren(...enabled.map(view => {
    const wrapper = document.createElement('div');
    wrapper.className = 'video-view'; wrapper.dataset.view = view;
    const title = document.createElement('b');
    title.textContent = {raw: '原图', annotated: '检测标注', gray: '灰度', binary: '二值'}[view];
    const image = document.createElement('img');
    image.src = `/video/${view}?t=${Date.now()}`; image.alt = title.textContent;
    if (view === 'raw') image.id = 'stream';
    wrapper.append(title, image); return wrapper;
  }));
  container.className = `views count-${Math.max(1, enabled.length)}`;
}

async function loadParameters() {
  const status = $('#parameter-status');
  try {
    const response = await fetch('/parameters', {cache: 'no-store'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || response.status);
    const values = data.parameters;
    $('#parameter-form').innerHTML = parameterDefinitions.map(([key, label, step]) =>
      `<label><span>${label}</span><input type="number" data-parameter="${key}" step="${step}" value="${values[key]}"></label>`
    ).join('');
    status.textContent = data.saved ? '已保存' : '运行态';
    status.className = data.ok ? 'ok' : 'error';
  } catch (error) { status.textContent = `读取失败: ${error.message}`; status.className = 'error'; }
}

async function submitParameters(save) {
  const status = $('#parameter-status');
  const parameters = Object.fromEntries([...document.querySelectorAll('[data-parameter]')].map(input => [input.dataset.parameter, Number(input.value)]));
  status.textContent = save ? '正在保存...' : '正在应用...'; status.className = '';
  try {
    const response = await fetch('/parameters', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({save, parameters})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || response.status);
    await new Promise(resolve => setTimeout(resolve, 500));
    await loadParameters();
  } catch (error) { status.textContent = `失败: ${error.message}`; status.className = 'error'; }
}

async function refresh() {
  try {
    const response = await fetch('/data', {cache: 'no-store'});
    if (!response.ok) throw new Error(response.status);
    const data = await response.json();
    if (!Object.keys(data).length) throw new Error('empty');
    const now = performance.now();
    const frame = Number(valueAt(data, 'frame.index'));
    const elapsed = now - lastReceivedAt;
    const frameDelta = frame - lastFrame;
    const fps = lastFrame !== null && elapsed > 0 ? frameDelta * 1000 / elapsed : 0;
    lastFrame = frame; lastReceivedAt = now;
    connection.textContent = '数据在线';
    connection.className = 'pill online';
    $('#frame-rate').textContent = fps > 0 ? `${fps.toFixed(1)} FPS` : '-- FPS';
    const gate = data.fire_gate ?? {};
    const feedback = data.gimbal?.feedback_valid;
    $('#safety').textContent = gate.allowed ? 'FIRE GATE OPEN' : `SAFE / NO COMMAND · ${gate.reason ?? 'disabled'} · ${feedback ? '反馈在线' : '无云台反馈'}`;
    $('#safety').className = `safety ${gate.allowed ? 'armed' : ''}`;
    $('#video-meta').textContent = `Frame ${frame} · ${fixed(data.latency_ms?.total)} ms · ${data.tracker?.state_name ?? '-'} · ${data.detection?.count ?? 0} detections`;
    renderCards(data);
    renderSections(data);
    renderLatency(data);
    history.push({total: Number(data.latency_ms?.total) || 0, detect: Number(data.latency_ms?.detection) || 0, nis: Number(data.tracker?.nis) || 0});
    if (history.length > 40) history.shift();
    drawChart();
    raw.textContent = JSON.stringify(data, null, 2);
  } catch (_) {
    connection.textContent = '连接断开，重试中';
    connection.className = 'pill offline';
  }
}

$('#fullscreen').addEventListener('click', () => $('#views').requestFullscreen?.());
document.querySelectorAll('[data-view]').forEach(input => input.addEventListener('change', setViews));
$('#reload-parameters').addEventListener('click', loadParameters);
$('#apply-parameters').addEventListener('click', () => submitParameters(false));
$('#save-parameters').addEventListener('click', () => { if (confirm('确认将当前参数写入 NUC YAML？')) submitParameters(true); });
window.addEventListener('resize', drawChart);
loadParameters();
refresh();
setInterval(refresh, 500);
