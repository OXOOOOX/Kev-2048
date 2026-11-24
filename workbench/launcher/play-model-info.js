function playModelOption(model, checkbox, make) {
 const card = make('div', 'play-option');
 const label = make('label', 'play-model-label');
 const info = make('span', 'play-model-info');
 info.id = 'play-info-' + model.id;
 checkbox.setAttribute('aria-label', model.name);
 checkbox.setAttribute('aria-describedby', info.id);
 info.append(make('strong', 'play-model-name', model.name));
 const latest = model.evaluation?.latest;
 const overall = model.evaluation?.overall;
 const valid = batch => Number.isInteger(batch?.games) && batch.games > 0 && Number.isFinite(batch.mean_score);
 const score = value => value.toLocaleString('zh-CN', {maximumFractionDigits: 1});
 if (valid(latest)) {
  const metric = make('span', 'play-model-score');
  metric.append(make('span', '', '整局平均分'), make('strong', '', score(latest.mean_score)),
   make('span', '', latest.games + ' 局'));
  info.append(metric);
  const context = [latest.label];
  if (latest.reused) context.push('复用既有对局');
  if (valid(overall) && overall.games > latest.games) {
   context.push('累计 ' + overall.games + ' 个开局均分 ' + score(overall.mean_score));
  }
  info.append(make('span', 'play-model-source', context.filter(Boolean).join(' · ')));
 } else {
  info.append(make('span', 'play-model-source', '暂无整局评测'));
 }
 if (model.training_summary) {
  const training = make('span', 'play-model-training', model.training_summary);
  training.setAttribute('aria-label', '训练简述');
  info.append(training);
 }
 label.append(checkbox, info);
 card.append(label);
 const batches = (model.evaluation?.batches || []).filter(valid);
 if (batches.length > 1) {
  const details = make('details', 'play-model-batches');
  details.append(make('summary', '', '各轮均分（' + batches.length + '）'));
  const list = make('ul');
  for (const batch of batches) {
   const row = make('li');
   const reuse = batch.reused ? '（复用）' : '';
   row.append(make('span', '', batch.label + ' · ' + batch.games + ' 局' + reuse),
    make('strong', '', score(batch.mean_score) + ' 分'));
   list.append(row);
  }
  details.append(list);
  card.append(details);
 }
 return card;
}

function showPlayModelOptions(container, choices, chosen, make) {
 const entries = [...choices.values()];
 if (entries.length <= 6) {
  container.replaceChildren(...entries.map(choice => choice.option));
  return;
 }
 const previous = container.querySelector('.play-model-others');
 const more = make('details', 'play-model-others');
 const remaining = entries.slice(3);
 more.open = previous ? previous.open : remaining.some(choice => chosen.has(choice.checkbox.value));
 const summary = make('summary', '', '其他模型（' + remaining.length + '）');
 const list = make('div', 'play-model-extra-list');
 list.append(...remaining.map(choice => choice.option));
 more.append(summary, list);
 container.replaceChildren(...entries.slice(0, 3).map(choice => choice.option), more);
}
