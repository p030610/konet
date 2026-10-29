const summary = JSON.parse(document.querySelector('#heldout-data').textContent);
const rowsElement = document.querySelector('#leaderboard-rows');
const statusElement = document.querySelector('#board-status');
const modelFilter = document.querySelector('#model-filter');
const buttons = [...document.querySelectorAll('[data-distance]')];
let distance = 80;
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
