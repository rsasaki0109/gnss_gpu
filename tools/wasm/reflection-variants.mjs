// Expand control flow only. Floating-point expressions remain in the WAT.
import assert from "node:assert/strict";

function replaceLine(source,key,replacement,expected) {
  const lines=source.split("\n").filter(line=>line.includes(key));
  assert.equal(lines.length,1,"Expected one WAT instruction: "+key);
  assert.equal(lines[0].trim(),expected,"WAT operation changed: "+key);
  return source.replace(lines[0],replacement);
}

function specializeProjection(source) {
  const axisStart=source.indexOf("          (local.set $axis ");
  const bodyStart=source.indexOf("          (local.set $ep ");
  const bodyEnd=source.indexOf("\n        )\n        (local.set $t",bodyStart);
  assert(axisStart>=0&&bodyStart>axisStart&&bodyEnd>bodyStart,"Unexpected triangle loop");
  for(const [name,step] of [["i",1],["j",2]])assert(source.includes(`(local.set $${name} (i32.rem_u (i32.add (local.get $axis) (i32.const ${step})) (i32.const 3)))`),"Projection setup changed");
  const body=source.slice(bodyStart,bodyEnd),variants=[];
  // MeshBVH always supplies projection 0, 1 or 2. Select once per triangle.
  for(const axis of [0,1,2]) {
    const i=(axis+1)%3,j=(axis+2)%3;let variant=body;
    for(const [name,offset,component] of [["ai",0,i],["aj",0,j],["bi",24,i],["bj",24,j],["ci",48,i],["cj",48,j]]) {
      const coordinate=name.endsWith("i")?"i":"j";
      const expected=`(local.set $${name} (f64.load offset=${offset} (i32.add (local.get $ep) (i32.mul (local.get $${coordinate}) (i32.const 8)))))`;
      variant=replaceLine(variant,"(local.set $"+name+" ",`          (local.set $${name} (f64.load offset=${offset+component*8} (local.get $ep)))`,expected);
    }
    for(const [name,component,base] of [["xv",i,"aiv"],["yv",j,"ajv"]]) {
      const coordinate=name==="xv"?"i":"j";
      const selected=`(if (result v128) (i32.eq (local.get $${coordinate}) (i32.const 0)) (then (local.get $qxv)) (else (if (result v128) (i32.eq (local.get $${coordinate}) (i32.const 1)) (then (local.get $qyv)) (else (local.get $qzv)))))`;
      const expected=`(local.set $${name} (f64x2.sub ${selected} (local.get $${base})))`;
      variant=replaceLine(variant,"(local.set $"+name+" ",`                (local.set $${name} (f64x2.sub (local.get $q${"xyz"[component]}v) (local.get $${base})))`,expected);
    }
    variants.push(variant);
  }
  const prefix=source.slice(0,axisStart)+source.slice(axisStart,source.indexOf("\n",axisStart)+1);
  let result=prefix+'          (if (i32.eq (local.get $axis) (i32.const 0)) (then\n'+variants[0]+'\n          ) (else\n            (if (i32.eq (local.get $axis) (i32.const 1)) (then\n'+variants[1]+'\n            ) (else\n'+variants[2]+'\n            ))\n          ))'+source.slice(bodyEnd);
  for(const name of ["i","j"])result=result.replace("    (local $"+name+" i32)\n","");
  return result;
}

export function expandReflectionSIMD(source) {
  source=specializeProjection(source);
  const start=source.indexOf("  (func "),end=source.lastIndexOf("\n))");
  assert(start>=0&&end>start,"Unexpected WAT module");
  const generic=source.slice(start,end)+"\n  )\n",signature=generic.split("\n")[0];
  const parameters=[...signature.matchAll(/\(param (\$\w+) (?:i32|f64)\)/g)].map(match=>match[1]);
  assert.equal(parameters.length,11,"Unexpected scan signature");
  const args=parameters.map(name=>"(local.get "+name+")").join(" ");
  const functions=[generic.replace('(func (export "scan")','(func $generic')];
  const laneCheck='(if (i32.and (i32.ne (i32.and (local.get $bits) (i32.const 2)) (i32.const 0)) (i32.lt_u (i32.add (local.get $s) (i32.const 1)) (local.get $sourceCount))) (then';
  for(let packets=1;packets<=8;packets++) {
    let fixed=generic.replace('(func (export "scan")','(func $fixed'+packets),search=0,loops=0;
    while(true) {
      const loopStart=fixed.indexOf("          (local.set $s (i32.const 0))",search);
      if(loopStart<0)break;
      const begin=fixed.indexOf("              (block $nextSource",loopStart);
      const finish=fixed.indexOf("\n              )\n              (local.set $s ",begin)+"\n              )".length;
      const terminator="              (br $sources)))",loopEnd=fixed.indexOf(terminator,finish)+terminator.length;
      assert(begin>loopStart&&finish>begin&&loopEnd>finish,"Unexpected source loop");
      // Reject source edits that would otherwise be skipped by expansion.
      assert.equal(fixed.slice(loopStart,begin).trim(),"(local.set $s (i32.const 0))\n          (block $endSources\n            (loop $sources\n              (br_if $endSources (i32.ge_u (local.get $s) (local.get $sourceCount)))");
      assert.equal(fixed.slice(finish,loopEnd).trim(),"(local.set $s (i32.add (local.get $s) (i32.const 2)))\n              (br $sources)))");
      const block=fixed.slice(begin,finish),blocks=[];
      assert(block.includes(laneCheck),"Missing odd-lane guard");
      for(let packet=0;packet<packets;packet++) {
        // Dispatch guarantees ceil(sourceCount/2)==packets. Only the final
        // packet can have an unused lane; preserve its original guard.
        const part=packet<packets-1?block.replace(laneCheck,'(if (i32.and (local.get $bits) (i32.const 2)) (then'):block;
        blocks.push(part.replaceAll("(local.get $s)","(i32.const "+packet*2+")"));
      }
      const unrolled=blocks.join("\n");
      fixed=fixed.slice(0,loopStart)+unrolled+fixed.slice(loopEnd);
      search=loopStart+unrolled.length;loops++;
    }
    assert.equal(loops,3,"Expected all projection loops");
    functions.push(fixed);
  }
  const dispatch=signature+"\n    (local $packets i32)\n    (local.set $packets (i32.shr_u (i32.add (local.get $sourceCount) (i32.const 1)) (i32.const 1)))\n"+
    Array.from({length:8},(_,i)=>`    (if (i32.eq (local.get $packets) (i32.const ${i+1})) (then (return (call $fixed${i+1} ${args}))))\n`).join("")+`    (call $generic ${args})\n  )\n`;
  return source.slice(0,start)+functions.join("")+dispatch+")\n";
}
