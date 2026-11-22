const fs=require('node:fs');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const html=fs.readFileSync('2048-replay.html','utf-8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
assert.equal([...html.matchAll(/<script(?:\s|>)/g)].length,1);
assert.equal([...html.matchAll(/<(script|img|link|iframe)\s[^>]*(href|src)=/ig)].length,0);
class Node{
 constructor(tag='DIV',content=''){this.tagName=tag;this.textContent=content;this.attrs={};this.children=[];this.dataset={};this.events={};this.q={};this.style={setProperty:(k,v)=>this.style[k]=v};this.classList={toggle:(k,val)=>this.classList[k]=val}}
 append(n){this.children.push(n)} replaceChildren(...a){this.children=a}
 querySelector(k){assert(this.q[k],'missing selector '+k);return this.q[k]}
 addEventListener(k,cb){this.events[k]=cb}
 setAttribute(k,v){this.attrs[k]=v} removeAttribute(k){delete this.attrs[k];if(k==='title')this.title=undefined}
 fire(k,x={}){if(k==='click'&&this.onclick)this.onclick(x);else this.events[k]?.(x)}
}
const doc=new Node('DOCUMENT');
const expectedIds={
 seed:'SELECT',seek:'INPUT',play:'BUTTON',prev:'BUTTON',next:'BUTTON',speed:'SELECT',marker:'BUTTON','turn-output':'OUTPUT',
 ticks:'DIV','jump-divergence':'BUTTON','jump-start':'BUTTON','jump-end':'BUTTON'};
for(const [id,tag] of Object.entries(expectedIds)){assert(html.includes('id="'+id+'"'));doc.q['#'+id]=new Node(tag)}
const panels=['identity','d4'].map((k,i)=>{
 assert(html.includes('data-policy="'+k+'"'));const p=new Node('ARTICLE');
 p.dataset.policy=k;p.q.h2=new Node('H2',i?'八方向投票':'原版 V2');
 for(const key of ['board','status','score-label','score','move-label','move-name','arrow','best','final']){
  assert(html.match(new RegExp('class="[^"]*'+key.replace('-','\\-')+'[^"]*"')));p.q['.'+key]=new Node()
 }
 return p
});
doc.querySelectorAll=k=>{assert.equal(k,'.panel');return panels};doc.createElement=k=>new Node(k.toUpperCase());
let counter=1,tasks=new Map;
const ctx={document:doc,HTMLElement:Node,Intl,setTimeout:(cb,ms)=>{const id=counter++;tasks.set(id,{cb,ms});return id},clearTimeout:id=>tasks.delete(id)};
doc.q['#speed'].value='1';
vm.runInNewContext(script,ctx,{filename:'embedded-replay.js'});
const get=id=>doc.q['#'+id], click=id=>get(id).fire('click'), seek=n=>{get('seek').value=n;get('seek').fire('input')};
const [l,r]=panels, val=(p,k)=>p.q['.'+k].textContent;
const frames=p=>p.q['.board'].children.map(n=>n.dataset.value);
const marker=p=>p.q['.board'].children.findIndex(n=>n.classList.spawn===true);
const turn=()=>get('turn-output').textContent;
const keyboard=k=>doc.fire('keydown',{target:new Node('MAIN'),key:k,preventDefault(){}});
const tick=()=>{const [entry]=tasks;if(!entry)return false;tasks.delete(entry[0]);entry[1].cb();return true};
assert.equal(turn(),'第 53 / 178 步');
assert.equal(val(l,'score'),'420');assert.equal(val(r,'score'),'280');assert.equal(val(r,'status'),'已结束');
assert.deepEqual(frames(l),[0,0,2,0,0,4,8,2,0,0,4,16,2,16,2,64]);
assert.deepEqual(frames(r),[8,2,4,2,4,32,8,4,16,8,4,8,2,16,2,4]);
assert.equal(marker(l),2);assert.equal(marker(r),3);
click('next');assert.equal(turn(),'第 54 / 178 步');assert.equal(marker(l),9);assert.equal(marker(r),-1);
assert.equal(val(r,'move-name'),'无新动作');assert.equal(val(r,'score'),'280');
click('marker');assert.equal(turn(),'第 53 / 178 步');
click('jump-start');assert.equal(turn(),'第 0 / 178 步');assert.deepEqual(frames(l),frames(r));assert.equal(marker(l),-1);
click('jump-divergence');assert.equal(turn(),'第 2 / 178 步');assert.equal(val(l,'move-name'),'向下');assert.equal(val(r,'move-name'),'向右');
keyboard('ArrowLeft');assert.equal(turn(),'第 1 / 178 步');assert.deepEqual(frames(l),frames(r));
keyboard(' ');assert.equal(get('play').attrs['aria-label'],'暂停');assert.equal(tasks.size,1);
get('speed').value='4';get('speed').fire('change');assert.equal(tasks.size,1);assert.equal([...tasks.values()][0].ms,162.5);
assert(tick());assert.equal(turn(),'第 2 / 178 步');click('play');assert(!tick());assert.equal(turn(),'第 2 / 178 步');
seek(176);click('play');assert(tick());assert(tick());assert(!tick());assert.equal(turn(),'第 178 / 178 步');
assert.equal(get('play').attrs['aria-label'],'播放');assert.equal(val(l,'score'),'1,752');
for(const [seed,end,ls,rs,status] of [['96626002',198,'2,296','820','已结束'],['96626003',240,'3,004','168','到达上限'],['96626004',133,'1,264','696','已结束']]){
 get('seed').value=seed;get('seed').fire('change');assert.equal(turn(),'第 0 / '+end+' 步');assert.deepEqual(frames(l),frames(r));
 click('jump-end');assert.equal(turn(),'第 '+end+' / '+end+' 步');assert.equal(val(l,'score'),ls);assert.equal(val(r,'score'),rs);assert.equal(val(l,'status'),status);assert.equal(val(r,'status'),'已结束');assert.equal(marker(r),-1)
}
console.log('PASS: embedded application JS init/render, step 0/1/2/53/54/178 reference, board/arrows/spawn, seek/jumps/keys, playback speed/pause/autostop, seed changes and true cap; no external scripts/assets.');
