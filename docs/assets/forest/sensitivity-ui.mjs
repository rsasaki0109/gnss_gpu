export function setupSensitivity(capture,apply){
  const $=id=>document.getElementById(id);let worker,result;
  const same=()=>{const s=capture().settings;return result&&s.location===result.settings.location&&s.seed===result.settings.seed;};
  const changed=()=>{$('sensitivity-stale').hidden=!result||same();};
  const stop=()=>{worker?.terminate();worker=null;$('sensitivity-run').disabled=!capture().scene;$('sensitivity-cancel').disabled=true;};
  const format=v=>v===null?'測位不能':`${v.toFixed(2)} m`;
  function render(){
    $('sensitivity-results').hidden=false;$('sensitivity-grid').replaceChildren();
    const s=result.settings;$('sensitivity-context').textContent=`${result.locationLabel} · シード ${s.seeds.join(', ')} · 20分 / 41時刻 · 各セルは濡れた森林の結果`;
    for(const threshold of s.thresholds){
      const tr=document.createElement('tr'),th=document.createElement('th');th.scope='row';th.textContent=`${threshold} dB-Hz`;tr.append(th);
      for(const wetLoss of s.wetLosses){
        const cell=result.cells.find(c=>c.threshold===threshold&&c.wetLoss===wetLoss),v=cell.summary[2],td=document.createElement('td'),button=document.createElement('button');
        button.type='button';button.dataset.threshold=threshold;button.dataset.wetLoss=wetLoss;
        const available=v.minAvailable===v.maxAvailable?`${v.minAvailable}/${v.total}`:`${v.minAvailable}–${v.maxAvailable}/${v.total}`;
        button.textContent=`${available} · ${format(v.medianSeedP95M)}`;
        button.title=`濡れ ${wetLoss} dB/m・下限 ${threshold} dB-Hz。成功時刻のp95のシード範囲: ${format(v.minSeedP95M)}〜${format(v.maxSeedP95M)}`;
        button.style.background=v.minAvailable===0?'#572f36':v.minAvailable<v.total?'#534225':'#193c35';
        button.onclick=()=>{
          apply({location:s.location,seed:s.seed,wetLoss,threshold});
          $('sensitivity-detail').textContent=`選択: ${wetLoss} dB/m / ${threshold} dB-Hz。測位可能 ${available}時刻、p95のシード範囲 ${format(v.minSeedP95M)}〜${format(v.maxSeedP95M)}（解あり ${v.seedsWithSolution}/8シード）。上の「3条件を比較する」で先頭シードの時系列を確認できます。`;
        };
        td.append(button);tr.append(td);
      }
      $('sensitivity-grid').append(tr);
    }
    changed();
  }
  $('settings').addEventListener('change',changed);
  $('sensitivity-run').onclick=()=>{
    if(!$('seed').reportValidity())return;
    const input=capture();if(!input.scene)return;
    stop();result=null;$('sensitivity-save').disabled=true;$('sensitivity-results').hidden=true;$('sensitivity-stale').hidden=true;$('sensitivity-detail').textContent='';
    $('sensitivity-run').disabled=true;$('sensitivity-cancel').disabled=false;$('sensitivity').dataset.state='running';
    const active=worker=new Worker(new URL('./sensitivity-worker.mjs?v=1',import.meta.url),{type:'module'});
    const fail=message=>{stop();$('sensitivity').dataset.state='error';$('sensitivity-status').textContent=message;};
    worker.onerror=()=>{if(worker===active)fail('感度マップの計算に失敗しました。');};
    worker.onmessage=({data})=>{
      if(worker!==active)return;
      if(data.error){fail(data.error);return;}
      if(data.progress){const p=data.progress;$('sensitivity-status').textContent=`${p.phase==='geometry'?'衛星軌道・遮蔽を計算':'8シードで条件を比較'}… ${p.completed}/${p.total}`;return;}
      result={...data.result,source:input.source};stop();render();$('sensitivity-save').disabled=false;$('sensitivity').dataset.state='ready';$('sensitivity-status').textContent='16条件 × 8シードの比較が完了しました。';
    };
    worker.postMessage({scene:input.scene,settings:input.settings});
  };
  $('sensitivity-cancel').onclick=()=>{stop();$('sensitivity').dataset.state='cancelled';$('sensitivity-status').textContent='感度マップの計算を中止しました。';};
  $('sensitivity-save').onclick=()=>{if(!result)return;const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)+'\n'],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='forest-sensitivity.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  return {ready:()=>{$('sensitivity-run').disabled=false;},changed};
}
