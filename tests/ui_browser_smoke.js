// Run against a fresh tests.ui_smoke_server with playwright-cli run-code --filename.
async (page) => {
  const posts = [];
  page.on('request', r => { if (r.method() === 'POST') posts.push(r.url()); });
  await page.getByRole('textbox', {name: '手动代码'}).fill('AAPL');
  await page.getByRole('button', {name: '启动分析', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('.run-message').textContent.includes('正在形成建议'));
  await page.reload();
  await page.waitForFunction(() => document.querySelector('.run-message').textContent.includes('正在形成建议'));
  if (posts.filter(u => u.endsWith('/api/runs')).length !== 1) throw Error('Refresh started extra run');
  await page.getByRole('button', {name: '取消并保存', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('.run-message').textContent.includes('已取消并保存'));
  if (await page.locator('.stage-succeeded').filter({hasText: '形成建议'}).count()) throw Error('Cancelled stage displayed as complete');
  await page.getByText('历史记录与可恢复任务', {exact: true}).click();
  await page.getByRole('button', {name: '手动恢复 AAPL', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('.run-message').textContent.includes('分析完成'));
  await page.getByRole('textbox', {name: '手动代码'}).fill('AAPL');
  await page.getByRole('button', {name: '启动分析', exact: true}).click();
  await page.waitForFunction(() => document.querySelector('.run-message').textContent.includes('正在形成建议'));
  await page.getByRole('button', {name: /^成功 run_UI_/}).click();
  const other = await page.context().newPage();
  try {
    await other.goto(page.url());
    await other.getByRole('button', {name: '取消并保存', exact: true}).click();
    await other.waitForFunction(() => document.querySelector('.run-message').textContent.includes('已取消并保存'));
    await page.waitForFunction(() => document.querySelector('#runState').textContent === '空闲');
  } finally { await other.close(); }
  console.log(JSON.stringify({refreshDidNotStartRun: true, cancelledStage: true,
    manualResume: true, historicalViewTracksRuntime: true, posts}));
}
