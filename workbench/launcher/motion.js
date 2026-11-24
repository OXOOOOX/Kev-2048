/* Deterministic one-slide visual layout, independent of the logged random spawn. */
function slidePlan(board, direction) {
 if (!['up','down','left','right'].includes(direction)) return null;
 const vertical=direction==='up'||direction==='down';
 const reverse=direction==='down'||direction==='right';
 const slid=Array(16).fill(0), moves=[], merges=[];
 for(let lane=0;lane<4;lane++){
  const cells=Array.from({length:4},(_,i)=>vertical?4*i+lane:4*lane+i);
  if(reverse) cells.reverse();
  const used=cells.filter(i=>board[i]!==0);
  for(let i=0,k=0;i<used.length;i++,k++){
   const from=used[i], to=cells[k], value=board[from];
   moves.push({from,to,value});
   if(i+1<used.length && board[used[i+1]]===value){
    moves.push({from:used[++i],to,value});
    slid[to]=value*2;
    merges.push(to);
   }else slid[to]=value;
  }
 }
 return {slid,moves,merges};
}
const reducedMotion=typeof matchMedia==='function'?matchMedia('(prefers-reduced-motion: reduce)'):null;
function boardTransition(board,tiles,before,after,action,spawn){
 const plan=slidePlan(before,action);
 if(!plan)return null;
 // Live frames carry both boards. The spawn must be their sole difference after sliding.
 if(spawn===undefined)spawn=after.findIndex((tile,i)=>tile!==plan.slid[i]);
 if(!Number.isInteger(spawn) || spawn<0 || spawn>15 || plan.slid[spawn]!==0 || ![2,4].includes(after[spawn])
         || !after.every((tile,i)=>i===spawn||tile===plan.slid[i])) return null;
 const bounds=board.getBoundingClientRect();
 if(!bounds.width || !bounds.height)return null;
 const cells=tiles.map(tile=>{
  const r=tile.getBoundingClientRect();
  return {x:r.left-bounds.left,y:r.top-bounds.top,w:r.width,h:r.height}
 });
 return {board,plan,cells,after,spawn};
}
function overlayTile(layer,index,value,spawn=false){
 const tile=document.createElement('div');
 const {x,y,w,h}=layer.cells[index];
 tile.className='tile fx-tile'+(spawn?' spawn':'');
 tile.dataset.value=value;tile.textContent=value;
 Object.assign(tile.style,{left:x+'px',top:y+'px',width:w+'px',height:h+'px'});
 layer.node.append(tile);
 return tile;
}
function createBoardMotion(){
 let effects=null;
 function finish(){
  if(!effects)return;
  const {timers,animations,layers}=effects;
  effects=null;
  for(const id of timers)clearTimeout(id);
  for(const animation of animations)animation.cancel();
  for(const layer of layers){layer.node.remove();layer.board.classList.toggle('is-sliding',false)}
 }
 function animateBoards(inputs,speedValue){
  if(!inputs.length||reducedMotion?.matches||typeof Element==='undefined'||typeof Element.prototype.animate!=='function')return;
  const slideLength=Math.min(255,Math.round(216/speedValue));
  const mergeLength=Math.min(132,Math.round(108/speedValue));
  const popLength=Math.min(139,Math.round(122/speedValue));
  const fx={layers:[],timers:[],animations:[]};effects=fx;
  const defer=(ms, work)=>fx.timers.push(setTimeout(()=>{if(effects===fx)work()},ms));
  const animate=(el,keys,options)=>fx.animations.push(el.animate(keys,options));
  for(const info of inputs){
   const node=document.createElement('div');
   node.className='fx-layer';node.setAttribute('aria-hidden','true');
   const layer={...info,node};fx.layers.push(layer);
   layer.board.classList.toggle('is-sliding',true);
   layer.board.append(node);
   for(const {from,to,value} of layer.plan.moves){
    const el=overlayTile(layer,from,value);
    if(from!==to){
     const dx=layer.cells[to].x-layer.cells[from].x;
     const dy=layer.cells[to].y-layer.cells[from].y;
     animate(el,[{transform:'translate(0,0)'},{transform:'translate('+dx+'px,'+dy+'px)'}],
         {duration:slideLength,easing:'cubic-bezier(.23,.74,.35,1)',fill:'forwards'});
    }
   }
  }
  defer(slideLength,()=>{
   for(const layer of fx.layers){
    layer.node.replaceChildren();
    for(const [i,value] of layer.plan.slid.entries()){
     if(!value)continue;
     const el=overlayTile(layer,i,value);
     if(layer.plan.merges.includes(i)){
      animate(el,[{transform:'scale(.92)'},{transform:'scale(1.14)',offset:.42},{transform:'scale(1)'}],
          {duration:mergeLength,easing:'ease-out'});
     }
    }
   }
  });
  defer(slideLength+mergeLength,()=>{
   for(const layer of fx.layers){
    const el=overlayTile(layer,layer.spawn,layer.after[layer.spawn],true);
    animate(el,[{opacity:0,transform:'scale(.48)'},{opacity:1,transform:'scale(1.12)',offset:.65},{opacity:1,transform:'scale(1)'}],
        {duration:popLength,easing:'ease-out',fill:'both'});
   }
  });
  defer(slideLength+mergeLength+popLength,finish);
 }
 return {finish,animate:animateBoards};
}
