const summary = JSON.parse(document.querySelector('#heldout-data').textContent);
const rowsElement = document.querySelector('#leaderboard-rows');
const statusElement = document.querySelector('#board-status');
const modelFilter = document.querySelector('#model-filter');
const buttons = [...document.querySelectorAll('[data-distance]')];
let distance = 120;
let visibleRows = [];
function cell(tag,text,className=''){const node=document.createElement(tag);node.textContent=text;node.className=className;return node;}
function renderBoard(){
 const i=summary.distance_ms.indexOf(distance);
 const ranked=summary.models.map(model=>({...model,score:model.strict_both_win_pct[i],margin:model.mean_min_margin[i]})).sort((a,b)=>b.score-a.score);
 visibleRows=ranked.map((model,index)=>({...model,rank:index+1})).filter(model=>modelFilter.value==='all'||model.family===modelFilter.value);
 rowsElement.replaceChildren();
 for(const row of visibleRows){const tr=document.createElement('tr');if(row.family==='ours')tr.className='our-row';const th=cell('th',row.name);th.scope='row';const delta=row.score-summary.models[0].strict_both_win_pct[i];tr.append(cell('td',String(row.rank)),th,cell('td',row.score.toFixed(2)+'%','numeric'),cell('td',row.id==='stablesyncnet'?'—':(delta>0?'+':'')+delta.toFixed(2)+' pp','numeric'),cell('td',row.margin.toFixed(4),'numeric'));rowsElement.append(tr);}
 document.querySelector('#primary-column').textContent='Both-win'+distance+' ↑';
 document.querySelector('#board-caption').textContent='44 records / 4 speakers / 96 B-windows · '+distance+'ms strict both-win';
 statusElement.textContent='완료된 paired common-support 비교 · 표시 '+visibleRows.length+'개 모델';
 for(const button of buttons)button.setAttribute('aria-pressed',String(Number(button.dataset.distance)===distance));
}
function download(text,type,name){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
buttons.forEach(button=>button.addEventListener('click',()=>{distance=Number(button.dataset.distance);renderBoard();}));
modelFilter.addEventListener('change',renderBoard);
document.querySelector('#download-results').addEventListener('click',()=>{const rows=[['distance_ms','rank','model','strict_both_win_percent','mean_min_margin','records','speakers','b_windows'],...visibleRows.map(r=>[distance,r.rank,r.name,r.score,r.margin,44,4,96])];download('\ufeff'+rows.map(r=>r.map(v=>'"'+String(v).replaceAll('"','""')+'"').join(',')).join('\r\n'),'text/csv;charset=utf-8','konet-'+distance+'ms-heldout.csv');statusElement.textContent='현재 표의 집계 CSV를 저장했습니다.';});
document.querySelector('#download-metrics').addEventListener('click',()=>{download(JSON.stringify(summary,null,2),'application/json;charset=utf-8','konet-heldout-summary-20261029.json');statusElement.textContent='전체 거리의 집계 JSON을 저장했습니다.';});
renderBoard();

const range = document.querySelector('#audio-offset');
function updateOffset() {
  const offset = Number(range.value);
  document.querySelector('#offset-output').textContent = `${offset > 0 ? '+' : ''}${offset} ms`;
  document.querySelector('#audio-track').style.transform = `translateX(${offset / 2}px)`;
  document.querySelector('#demo-status').textContent = offset === 0 ? '0ms · 두 타임라인이 같은 기준 시각에 놓여 있습니다.'
    : `${offset > 0 ? '+' : ''}${offset}ms · 파형을 ${offset > 0 ? '오른쪽' : '왼쪽'}으로 옮긴 개념도입니다.`;
}
range.addEventListener('input', updateOffset);
document.querySelector('#reset-offset').addEventListener('click', () => { range.value = '0'; updateOffset(); });

const navigation = [...document.querySelectorAll('.nav-links a')];
const observer = new IntersectionObserver(entries => {
  for (const entry of entries) if (entry.isIntersecting) {
    for (const link of navigation) {
      if (link.hash === `#${entry.target.id}`) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    }
  }
}, { rootMargin: '-15% 0px -55% 0px' });
for (const link of navigation) { const section = document.querySelector(link.hash); if (section) observer.observe(section); }

const publicMedia=[...document.querySelectorAll('video,audio')];publicMedia.forEach(player=>player.addEventListener('play',()=>{publicMedia.forEach(other=>{if(other!==player)other.pause();});}));

﻿(() => {
  const root = document.querySelector('[data-c293-local-root]');
  if (!root) return;
  const players = [root.querySelector('#c293-aligned'), root.querySelector('#c293-shifted')];
  const expected = ['fc0de5ce8b98fb9b5f44ac50c4ff6dc8b1b5dd1128036511f2799fea4081744e', '148bf3e7d00bb991762e0333e4974945f020f88deebb9e5718fa83bba8118387'];
  const scores = ['0.2321', '0.6257'];
  const labels = ['정합 입력 · 0ms', '영상 시간 이동 · −640ms'];
  const status = root.querySelector('#c293-status');
  const sequenceButton = root.querySelector('#c293-play');
  const objectUrls = [null, null];
  const revisions = [0, 0];
  let sequence = null;
  let generation = 0;
  function refresh() {
    sequenceButton.disabled = players.some(player => !player.currentSrc || player.readyState < 1);
    players.forEach((player, index) => {
      root.querySelector(`[data-c293-empty="${index}"]`).hidden = Boolean(player.getAttribute('src'));
      player.hidden = !player.getAttribute('src');
      root.querySelector(`[data-c293-badge="${index}"]`).textContent = player.getAttribute('src') && player.dataset.verified !== 'true' ? '선택 영상 · 미측정' : `기록 cosine ${scores[index]}`;
      root.querySelector(`[data-c293-score="${index}"]`).textContent = player.dataset.verified === 'true'
        ? `기록 cosine ${scores[index]} · 원본 표시 파일 확인됨`
        : player.getAttribute('src') ? '선택한 파일 · 점수 미측정 / 기록 점수 연결 미확인' : '기록 점수 적용 파일 · 선택 대기';
    });
  }
  function stop(reset = false) {
    generation += 1;
    sequence = null;
    players.forEach(player => { player.pause(); if (reset && player.readyState > 0) player.currentTime = 0; });
    root.querySelectorAll('[data-c293-stage]').forEach(stage => stage.classList.remove('playing'));
  }
  async function playStage(index, token) {
    if (token !== generation) return;
    sequence = index;
    players[index].currentTime = 0;
    try { await players[index].play(); }
    catch { if (token === generation) { stop(); status.textContent = '재생 버튼을 눌러 영상을 시작해 주세요.'; } }
  }
  players.forEach((player, index) => {
    player.addEventListener('loadedmetadata', refresh);
    player.addEventListener('play', () => {
      if (sequence !== index) { generation += 1; sequence = null; }
      document.querySelectorAll('video,audio').forEach(other => { if (other !== player) other.pause(); });
      root.querySelectorAll('[data-c293-stage]').forEach((stage, i) => stage.classList.toggle('playing', i === index));
      status.textContent = `${labels[index]} 재생 중 · ${player.dataset.verified === 'true' ? '원본 표시 파일 확인됨' : '선택한 파일은 점수 미측정'}`;
    });
    player.addEventListener('timeupdate', () => {
      const inWindow = player.currentTime >= 1.28 && player.currentTime < 1.92;
      root.querySelector('#c293-window').textContent = (inWindow && player.dataset.verified === 'true' ? '지금은 기록 점수의 측정 구간입니다. ' : '') + '기록 점수 구간: 재생 1.28–1.92초 · 16프레임 / 0.64초. ' + (players.every(item => item.dataset.verified === 'true') ? '두 원본 표시 파일은 같은 한국어 음성을 포함합니다.' : '현재 파일의 점수 연결 여부는 각 플레이어에서 확인할 수 있습니다.');
    });
    player.addEventListener('pause', () => {
      root.querySelector(`[data-c293-stage="${index}"]`).classList.remove('playing');
      if (!player.ended && player.currentTime > 0 && players.every(item => item.paused)) status.textContent = `${labels[index]} 일시정지`;
    });
    player.addEventListener('ended', () => {
      root.querySelector(`[data-c293-stage="${index}"]`).classList.remove('playing');
      if (sequence !== index) return;
      if (index === 0) void playStage(1, generation);
      else { sequence = null; status.textContent = '정합 → 시간 이동 비교 재생을 마쳤습니다.'; }
    });
    player.addEventListener('error', () => { stop(); status.textContent = '영상을 읽지 못했습니다. 해당 MP4 파일을 다시 선택해 주세요.'; });
  });
  document.querySelectorAll('video,audio').forEach(player => {
    if (!players.includes(player)) player.addEventListener('play', () => stop());
  });
  root.querySelectorAll('[data-c293-pick]').forEach(input => input.addEventListener('change', async () => {
    const index = Number(input.dataset.c293Pick);
    const file = input.files?.[0];
    if (!file) return;
    stop(true);
    const revision = ++revisions[index];
    if (objectUrls[index]) URL.revokeObjectURL(objectUrls[index]);
    objectUrls[index] = URL.createObjectURL(file);
    players[index].dataset.verified = 'false';
    players[index].src = objectUrls[index];
    refresh();
    status.textContent = `${labels[index]} 파일 연결 · 원본 표시 파일 확인 중`;
    try {
      const digest = await crypto.subtle.digest('SHA-256', await file.arrayBuffer());
      const hash = [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('');
      if (revision !== revisions[index]) return;
      players[index].dataset.verified = String(hash === expected[index]);
      refresh();
      status.textContent = hash === expected[index] ? `${labels[index]} · 원본 표시 파일이 확인됐습니다.` : `${labels[index]} · 다른 파일이므로 기록 점수와 연결하지 않습니다.`;
    } catch { if (revision === revisions[index]) status.textContent = '로컬 재생 준비 완료 · 원본 파일 확인은 이 브라우저에서 지원되지 않습니다.'; }
  }));
  sequenceButton.addEventListener('click', () => { stop(true); void playStage(0, generation); });
  root.querySelector('#c293-stop').addEventListener('click', () => { stop(true); status.textContent = '재생을 정지하고 처음으로 돌아왔습니다.'; });
  window.addEventListener('pagehide', () => { stop(); objectUrls.forEach(url => { if (url) URL.revokeObjectURL(url); }); });
  refresh();
})();
