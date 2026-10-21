const test = require("node:test");
const assert = require("node:assert/strict");
const geometry = import("../../docs/assets/simulation/view-geometry.mjs");
const close = (a, b, epsilon = 1e-9) => assert.ok(Math.abs(a - b) <= epsilon, `${a} != ${b}`);

test("view axes form an orthonormal frame matching the rendering convention", async () => {
  const { viewAxes } = await geometry;
  for (const yaw of [-Math.PI, -.63, 0, 1.4]) for (const pitch of [-1.48,-.82,-.15,.15,.82,1.48]) {
    const {right, up, depth} = viewAxes(yaw, pitch);
    for (const axis of [right, up, depth]) close(Math.hypot(...axis), 1);
    const dot = (a,b) => a.reduce((sum,v,i)=>sum+v*b[i],0);
    close(dot(right,up),0); close(dot(up,depth),0); close(dot(depth,right),0);
    assert.deepEqual(right,[Math.cos(yaw),-Math.sin(yaw),0]);
  }
});

test("surface orientation faces the receiver side of a wall regardless of winding and handles roofs", async () => {
  const { surfaceOrientation } = await geometry;
  const wall = {point:[0,0,1],vertices:[[0,-2,0],[0,2,0],[0,2,3]]};
  const original = structuredClone(wall);
  const front = surfaceOrientation([10,0,0],wall), back = surfaceOrientation([-10,0,0],wall);
  close(front.yaw,Math.PI/2); close(back.yaw,-Math.PI/2); close(front.pitch,.4);
  assert.deepEqual(surfaceOrientation([10,0,0],{...wall,vertices:[...wall.vertices].reverse()}),front);
  const roof = surfaceOrientation([2,3,10],{point:[0,0,0],vertices:[[0,0,0],[1,0,0],[0,1,0]]});
  assert.ok(Number.isFinite(roof.yaw)); close(roof.pitch,1.3);
  const underside = surfaceOrientation([2,3,-10],{point:[0,0,0],vertices:[[0,0,0],[1,0,0],[0,1,0]]});
  close(underside.pitch,-1.3);
  assert.deepEqual(wall,original);
});

test("camera framing fits receiver and every surface vertex at mobile and desktop sizes under orbit", async () => {
  const { framePoints } = await geometry;
  const points = [[-1014.82,-1252.84,.10],[-924.11,-1321.45,29.84],[-904.23,-1350.90,43.49],[-925.52,-1319.36,43.47],[-925.52,-1319.38,-.88]];
  const original=structuredClone(points);
  for(const [width,height] of [[220,385],[365,385],[940,565]]) for(const yaw of [-2.5,-.63,0,1.2]) for(const pitch of [-1.48,-.82,-.15,.15,.82,1.48]) {
    const frame=framePoints(points,yaw,pitch,width,height);
    assert.ok(frame.span>=48 && Number.isFinite(frame.scale));
    const projected=points.map(point=>{
      const [x,y,z]=point.map((v,i)=>v-frame.target[i]);
      // Independent screen equations, rather than the production axes helper.
      return [width/2+(Math.cos(yaw)*x-Math.sin(yaw)*y)*frame.scale,height*.56-(-Math.sin(yaw)*Math.sin(pitch)*x-Math.cos(yaw)*Math.sin(pitch)*y+Math.cos(pitch)*z)*frame.scale];
    });
    for(const [x,y] of projected) {assert.ok(x>=50-1e-8 && x<=width-50+1e-8);assert.ok(y>=150-1e-8 && y<=height-100+1e-8);}
    close((Math.min(...projected.map(p=>p[0]))+Math.max(...projected.map(p=>p[0])))/2,width/2);
    close((Math.min(...projected.map(p=>p[1]))+Math.max(...projected.map(p=>p[1])))/2,height*.56);
  }
  assert.deepEqual(points,original);
});

test("focused orbit can cross between upper and lower views without reaching the ground-plane singularity", async () => {
  const {clampViewPitch}=await geometry;
  close(clampViewPitch(-1),.15); close(clampViewPitch(2),1.48);
  close(clampViewPitch(-1,true),-1); close(clampViewPitch(-2,true),-1.48);
  close(clampViewPitch(-.05,true,.1),.15); close(clampViewPitch(.05,true,-.1),-.15);
  for(let i=-300;i<=300;i++)assert.ok(Math.abs(clampViewPitch(i/100,true))>=.15);
});

test("framing is translation invariant and bounds magnification for coincident points", async () => {
  const { framePoints } = await geometry;
  const points=[[1,2,3],[50,-80,20],[-30,7,40]],shift=[-3963426,3350882,3694865];
  const a=framePoints(points,-.63,.82,940,565),b=framePoints(points.map(p=>p.map((v,i)=>v+shift[i])),-.63,.82,940,565);
  close(a.span,b.span,1e-8); a.target.forEach((v,i)=>close(v+shift[i],b.target[i],1e-8));
  const small=framePoints([[4,5,6],[4,5,6]],0,.82,365,385);
  assert.equal(small.span,48); assert.deepEqual(small.target,[4,5,6]);
});
