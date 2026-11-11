"""3D reachable-workspace volume: samples tool-down poses on several z slices and writes one self-contained HTML
page (inline canvas JS, no libraries, no network) with drag-to-rotate, z-range sliders and colour modes."""
import json

import numpy as np

from reachlens.mapping import map_slice

_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>reachlens __ARM__ volume</title><style>
:root{--bg:#f6f7f9;--fg:#1d2230;--mut:#5b6377;--card:#fff;--line:#d9dde6}
@media (prefers-color-scheme:dark){:root{--bg:#14171f;--fg:#e8ebf2;--mut:#98a0b3;--card:#1d222d;--line:#323949}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}h1{font-size:18px;margin:0 0 4px}p{margin:0 0 12px;color:var(--mut)}
canvas{width:100%;height:auto;aspect-ratio:4/3;background:var(--card);border:1px solid var(--line);border-radius:8px;touch-action:none;cursor:grab}
.row{display:flex;flex-wrap:wrap;gap:14px 22px;margin-top:12px;align-items:center}label{display:flex;gap:8px;align-items:center}
input[type=range]{width:160px}select{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:3px}
.leg{display:flex;align-items:center;gap:8px;color:var(--mut)}.leg canvas{width:160px;height:12px;aspect-ratio:auto;border-radius:3px;cursor:default}
</style></head><body><main>
<h1>__ARM__ reachable workspace</h1>
<p>__SUMMARY__ Drag to rotate, scroll or pinch to zoom. Each point is a tool-down pose that IK solved.</p>
<canvas id="c" width="960" height="720"></canvas>
<div class="row">
<label>z from <input type="range" id="zlo"><span id="zlov"></span></label>
<label>z to <input type="range" id="zhi"><span id="zhiv"></span></label>
<label>colour <select id="mode"><option value="m">manipulability</option><option value="g">joint-limit margin</option><option value="z">height</option></select></label>
<label><input type="checkbox" id="un"> show unreachable</label>
<label><input type="checkbox" id="spin" checked> auto-rotate</label>
<span class="leg"><span id="lo"></span><canvas id="bar" width="160" height="12"></canvas><span id="hi"></span></span>
</div></main><script>
const D=__DATA__;const cv=document.getElementById('c'),cx=cv.getContext('2d');
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const P=[[68,1,84],[71,44,122],[59,82,139],[44,114,142],[33,145,140],[39,173,129],[92,200,99],[170,220,50],[253,231,37]];
function ramp(t){t=Math.max(0,Math.min(1,t))*(P.length-1);const i=Math.min(P.length-2,Math.floor(t)),f=t-i;
return P[i].map((v,k)=>Math.round(v+(P[i+1][k]-v)*f));}
const zs=[...new Set(D.z)].sort((a,b)=>a-b);
const zlo=document.getElementById('zlo'),zhi=document.getElementById('zhi');
for(const s of [zlo,zhi]){s.min=0;s.max=zs.length-1;s.step=1}zlo.value=0;zhi.value=zs.length-1;
const un=document.getElementById('un'),mode=document.getElementById('mode'),spin=document.getElementById('spin');
let yaw=-0.7,pitch=0.55,zoom=1,drag=null;
const ex={x:[Math.min(...D.x),Math.max(...D.x)],y:[Math.min(...D.y),Math.max(...D.y)]};
const cxm=(ex.x[0]+ex.x[1])/2,cym=(ex.y[0]+ex.y[1])/2,czm=(zs[0]+zs[zs.length-1])/2;
const R=Math.max(ex.x[1]-ex.x[0],ex.y[1]-ex.y[0],zs[zs.length-1]-zs[0])/2||1;
const mmax=Math.max(...D.m.filter(v=>v>0))||1;
function val(i){const m=mode.value;return m=='m'?D.m[i]/mmax:m=='g'?D.g[i]:(D.z[i]-zs[0])/((zs[zs.length-1]-zs[0])||1);}
function proj(x,y,z){x-=cxm;y-=cym;z-=czm;const cyw=Math.cos(yaw),syw=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
const a=x*cyw-y*syw,b=x*syw+y*cyw,u=a,v=z*cp-b*sp,d=z*sp+b*cp;const s=zoom*cv.width*0.34/R;return[cv.width/2+u*s,cv.height/2-v*s,d];}
function draw(){const lo=+zlo.value,hi=+zhi.value,zl=zs[Math.min(lo,hi)],zh=zs[Math.max(lo,hi)];
document.getElementById('zlov').textContent=zl.toFixed(2)+' m';document.getElementById('zhiv').textContent=zh.toFixed(2)+' m';
cx.clearRect(0,0,cv.width,cv.height);const pts=[];
for(let i=0;i<D.x.length;i++){if(D.z[i]<zl-1e-9||D.z[i]>zh+1e-9)continue;if(!D.r[i]&&!un.checked)continue;pts.push(i);}
const pr=pts.map(i=>({i,p:proj(D.x[i],D.y[i],D.z[i])})).sort((a,b)=>b.p[2]-a.p[2]);
const sp=(ex.x[1]-ex.x[0])/Math.max(D.n-1,1),rad=Math.max(1.6,0.3*sp*zoom*cv.width*0.34/R);
for(const q of pr){const i=q.i;if(D.r[i]){const c=ramp(val(i));cx.fillStyle=`rgba(${c[0]},${c[1]},${c[2]},.85)`;}
else cx.fillStyle='rgba(150,150,160,.25)';cx.beginPath();cx.arc(q.p[0],q.p[1],D.r[i]?rad:rad*.6,0,6.3);cx.fill();}
const o=proj(0,0,zs[0]-(zs[zs.length-1]-zs[0])*.02),ax=[['x',[.12,0,0],'#d9534f'],['y',[0,.12,0],'#3cb371'],['z',[0,0,.12],'#4a7bd0']];
cx.lineWidth=2;cx.font='14px system-ui';for(const[n,v,c]of ax){const e=proj(v[0],v[1],zs[0]+v[2]-(zs[zs.length-1]-zs[0])*.02);cx.strokeStyle=c;cx.fillStyle=c;cx.beginPath();cx.moveTo(o[0],o[1]);cx.lineTo(e[0],e[1]);cx.stroke();cx.fillText(n,e[0]+4,e[1]-4);}
const b=proj(0,0,D.base);cx.fillStyle=css('--fg');cx.beginPath();cx.arc(b[0],b[1],5,0,6.3);cx.fill();cx.fillText('base',b[0]+8,b[1]+4);
cx.fillStyle=css('--mut');cx.fillText(`${pts.filter(i=>D.r[i]).length} reachable points shown`,12,cv.height-12);}
function loop(){if(spin.checked&&!drag)yaw+=0.004;draw();requestAnimationFrame(loop);}
cv.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];cv.setPointerCapture(e.pointerId);cv.style.cursor='grabbing'});
cv.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*0.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-drag[1])*0.008));drag=[e.clientX,e.clientY]});
cv.addEventListener('pointerup',()=>{drag=null;cv.style.cursor='grab'});
cv.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.4,Math.min(4,zoom*(e.deltaY<0?1.1:.91)))},{passive:false});
function bar(){const b=document.getElementById('bar').getContext('2d');for(let x=0;x<160;x++){const c=ramp(x/159);b.fillStyle=`rgb(${c})`;b.fillRect(x,0,1,12);}
const m=mode.value;document.getElementById('lo').textContent=m=='z'?zs[0].toFixed(2)+' m':'0';
document.getElementById('hi').textContent=m=='m'?mmax.toFixed(3):m=='g'?'1 (mid-range)':zs[zs.length-1].toFixed(2)+' m';}
mode.addEventListener('change',bar);bar();loop();
</script></body></html>"""


def sample_volume(model, data, site, q_home, zs, half, n, tool_length=0.0, yaw=0.0):
    """Map each z slice and flatten to column lists. Returns dict of lists (x, y, z, r, m, g)."""
    cols = {k: [] for k in "xyzrmg"}
    for z in zs:
        res = map_slice(model, data, site, float(z), half=half, n=n, q_home=q_home, yaw=yaw, tool_length=tool_length)
        for row in res["rows"]:
            cols["x"].append(round(row["x"], 4))
            cols["y"].append(round(row["y"], 4))
            cols["z"].append(round(row["z"], 4))
            ok = bool(row["reachable"])
            cols["r"].append(int(ok))
            cols["m"].append(round(row["manipulability"], 5) if ok else 0)
            cols["g"].append(round(row["limit_margin"], 3) if ok else 0)
    return cols


def write_html(cols, arm, out, n, base_z=0.0):
    """Self-contained page from sampled columns. base_z is where the arm base sits (drawn as a dot)."""
    reach = sum(cols["r"])
    total = len(cols["r"])
    summ = f"{reach} of {total} sampled poses reachable ({reach / max(total, 1):.0%})."
    data = dict(cols, n=n, base=base_z)
    html = (_PAGE.replace("__ARM__", arm).replace("__SUMMARY__", summ)
            .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return dict(points=total, reachable=reach, bytes=len(html))
