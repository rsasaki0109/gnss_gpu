const $=id=>document.getElementById(id),labels=['開けた場所','森林','濡れた森林'],colors=['#91f8da','#ffcf68','#b4a4ff'];
let scene,source,result,worker,activeSettings;
const settings=()=>({seed:Number($('seed').value),wetLoss:Number($('wet-loss').value),threshold:Number($('threshold').value),location:$('location').value});
function svg(id,title){const root=$(id);root.replaceChildren();root.setAttribute('viewBox','0 0 480 260');const add=(tag,attrs,text)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,String(v));if(text!==undefined)e.textContent=text;root.append(e);return e;};add('text',{x:20,y:24,fill:'#dfefea','font-size':14},title);return add;}
function linePlot(id,title,value){
  const add=svg(id,title),maximum=Math.max(1,...result.epochs.flatMap(e=>e.cases.map(value)).filter(Number.isFinite))*1.1;
  for(let j=0;j<=4;j++){const y=220-j*45;add('line',{x1:48,x2:460,y1:y,y2:y,stroke:'#34544c'});add('text',{x:42,y:y+4,fill:'#abc6bc','text-anchor':'end','font-size':11},(maximum*j/4).toFixed(1));}
  for(let i=0;i<3;i++){let d='',gap=true;for(const e of result.epochs){const v=value(e.cases[i]);if(v===null){gap=true;continue;}const x=48+e.seconds/1200*412,y=220-v/maximum*180;d+=`${gap?'M':'L'}${x},${y} `;gap=false;add('circle',{cx:x,cy:y,r:2,fill:colors[i]});}add('path',{d,fill:'none',stroke:colors[i],'stroke-width':1.6});}
  for(const m of [0,5,10,15,20])add('text',{x:48+m/20*412,y:245,fill:'#abc6bc','text-anchor':'middle','font-size':11},`${m}分`);
}
function row(root,values){const tr=document.createElement('tr');for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}root.append(tr);}
function epochView(){
  if(!result)return;const epoch=result.epochs[Number($('epoch').value)];$('time').textContent=`+${epoch.seconds}秒`;
  $('epoch-status').textContent=epoch.cases.map((c,i)=>`${labels[i]}: ${c.used}衛星 / ${c.horizontalErrorM===null?'測位不能':c.horizontalErrorM.toFixed(2)+' m'}`).join(' · ');
  const sky=svg('sky','天頂図 · GPS · 北が上'),cx=240,cy=145,r=85;
  for(const radius of [r,r*2/3,r/3])sky('circle',{cx,cy,r:radius,fill:'none',stroke:'#45645a'});
  sky('text',{x:cx,y:48,fill:'#abc6bc','text-anchor':'middle'},'N');sky('text',{x:cx+r+12,y:cy,fill:'#abc6bc'},'E');
  const add=svg('strength','C/N₀ · dB-Hz（遮蔽は点なし）'),channels=epoch.cases[0].channels,n=channels.length;
  add('line',{x1:40,x2:460,y1:220-result.settings.threshold/50*170,y2:220-result.settings.threshold/50*170,stroke:'#abc6bc','stroke-dasharray':'4 4'});
  for(const v of [0,20,40,50])add('text',{x:30,y:224-v/50*170,fill:'#abc6bc','font-size':11,'text-anchor':'end'},v);
  $('channels').replaceChildren();
  channels.forEach((c,j)=>{
    const az=c.azimuth*Math.PI/180,rr=r*(90-c.elevation)/90,x=cx+rr*Math.sin(az),y=cy-rr*Math.cos(az);
    sky('circle',{cx:x,cy:y,r:4,fill:'#dfefea'});sky('text',{x:x+6,y:y-5,fill:'#dfefea','font-size':11},c.id);
    const values=epoch.cases.map(v=>v.channels[j]);row($('channels'),[c.id,...values.map(v=>v.cn0===null?'遮蔽':`${v.cn0.toFixed(1)}${v.used?'':'（下限未満）'}`)]);
    const xx=52+j*395/Math.max(1,n-1);add('text',{x:xx,y:242,fill:'#abc6bc','font-size':10,'text-anchor':'middle'},c.id);
    values.forEach((v,i)=>{if(v.cn0!==null)add('circle',{cx:xx+(i-1)*5,cy:220-Math.max(0,v.cn0)/50*170,r:3,fill:colors[i]});});
  });
}
function render(){
  $('results').hidden=false;$('summary').replaceChildren();
  result.summary.forEach((s,i)=>row($('summary'),[labels[i],`${s.available}/${s.total} (${(s.availability*100).toFixed(1)}%)`,s.meanUsed.toFixed(1),s.horizontalP95M===null?'測位不能':`${s.horizontalP95M.toFixed(2)} m`]));
  const s=result.settings;$('capture').textContent=`実行条件: ${s.location==='canopy'?'樹冠下':'林道'} · 濡れ ${s.wetLoss} dB/m · 下限 ${s.threshold} dB-Hz · シード ${s.seed} · GPST週 ${result.week} / TOW ${result.startTow}`;
  linePlot('errors','水平測位誤差 · m',c=>c.horizontalErrorM);linePlot('counts','使用衛星数',c=>c.used);epochView();
}
function stop(){worker?.terminate();worker=null;$('run').disabled=!scene;$('cancel').disabled=true;}
$('cancel').onclick=()=>{stop();$('comparison').dataset.state='cancelled';$('status').textContent='計算を中止しました。';};
$('settings').onchange=()=>{$('stale').hidden=!result||JSON.stringify(result.settings)===JSON.stringify(settings());};
$('settings').onsubmit=event=>{
  event.preventDefault();stop();activeSettings=settings();$('run').disabled=true;$('save').disabled=true;$('cancel').disabled=false;$('stale').hidden=true;$('results').hidden=true;result=null;$('comparison').dataset.state='running';
  const active=worker=new Worker(new URL('./environment-worker.mjs',import.meta.url),{type:'module'});
  const fail=message=>{stop();$('comparison').dataset.state='error';$('status').textContent=message;};
  worker.onerror=()=>{if(worker===active)fail('計算に失敗しました。再実行してください。');};
  worker.onmessage=({data})=>{
    if(worker!==active)return;
    if(data.error){fail(data.error);return;}
    if(data.progress){$('status').textContent=`3条件を計算中… ${data.progress.completed}/${data.progress.total}時刻`;return;}
    result={...data.result,source};stop();render();$('save').disabled=false;$('comparison').dataset.state='ready';$('status').textContent='20分間・41時刻の比較が完了しました。';$('settings').onchange();
  };
  worker.postMessage({scene,settings:activeSettings});
};
$('epoch').oninput=epochView;
$('save').onclick=()=>{if(!result)return;const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)+'\n'],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='forest-environment-comparison.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
async function init(){
  const base=new URL('../plateau/',import.meta.url),response=await fetch(new URL('manifest.json',base));if(!response.ok)throw Error('軌道の目録を取得できません');
  const {scene:asset}=await response.json();if(!/^[a-z0-9._-]+$/.test(asset.file)||!Number.isInteger(asset.bytes)||asset.bytes>2e6)throw Error('軌道の目録が不正です');
  const data=await fetch(new URL(`${asset.file}?sha256=${asset.sha256}`,base));if(!data.ok)throw Error('軌道データを取得できません');
  const bytes=await data.arrayBuffer(),hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes)),b=>b.toString(16).padStart(2,'0')).join('');
  if(bytes.byteLength!==asset.bytes||hash!==asset.sha256)throw Error('軌道データの検証に失敗しました');
  scene=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).json();source={file:asset.file,sha256:hash};$('run').disabled=false;$('comparison').dataset.state='idle';$('status').textContent='準備完了。3条件を比較してください。';
}
init().catch(error=>{$('comparison').dataset.state='error';$('status').textContent=error.message;});
