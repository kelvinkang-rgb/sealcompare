const { test, expect } = require('@playwright/test')
const path = require('path')

test('PDF：圖像1 可跳頁預覽並手動框選；圖像2 可逐頁編輯多印鑑框', async ({ page }) => {
  const pdfPath = path.resolve(__dirname, '..', '..', 'test_images', '案例一-印章有壓到線上.pdf')

  await page.goto('/multi-seal-test')

  // 圖像1：上傳 PDF → 會自動跳到建議頁並打開印鑑框選對話框
  await page.setInputFiles('#image1-upload', pdfPath)

  const image1Dialog = page.getByRole('dialog').filter({ hasText: '調整圖像1印鑑位置' })
  await expect(image1Dialog).toBeVisible()
  // 調整 bbox（用數值輸入避免拖曳不穩定），確認後主畫面預覽應反映（以「再次打開」的初始值驗證）
  const xInput = image1Dialog.getByLabel('X 位置')
  const oldX = Number(await xInput.inputValue())
  // 不能假設往右一定能加：若 bbox 已貼近右邊界，UI 會 clamp 回最大值
  // 這裡用「往左移 1px」來驗證「保存後重開仍維持」
  const newX = String(oldX > 0 ? oldX - 1 : 1)
  await xInput.fill(newX)
  await Promise.all([
    page.waitForResponse((r) => r.url().includes('/seal-location') && r.request().method() === 'PUT' && r.status() === 200),
    image1Dialog.getByRole('button', { name: '確認' }).click(),
  ])
  await expect(image1Dialog).toBeHidden()

  // 再次開啟編輯，初始 bbox 應為剛才保存的值（代表 state/DB 同步，主畫面預覽也會同步）
  // 頁面上同時有 IconButton 與 fullWidth Button 兩個「編輯印鑑位置」，用 hasText 避免 strict mode 衝突
  await page.getByRole('button', { name: '編輯印鑑位置' }).filter({ hasText: '編輯印鑑位置' }).click()
  await expect(image1Dialog).toBeVisible()
  await expect(image1Dialog.getByLabel('X 位置')).toHaveValue(newX)
  await image1Dialog.getByRole('button', { name: '確認' }).click()
  await expect(image1Dialog).toBeHidden()

  // 應該出現模板頁選擇器
  await expect(page.getByLabel('模板頁')).toBeVisible()

  // 圖像2：上傳 PDF → 顯示逐頁偵測摘要 + 可選頁預覽/編輯
  await page.setInputFiles('#image2-upload', pdfPath)
  // 等待 PDF 多印鑑逐頁偵測完成（避免按下「編輯此頁多印鑑框」時 handler 因狀態尚未就緒而早退）
  await page.waitForResponse((r) => r.url().includes('/detect-multiple-seals') && r.request().method() === 'POST' && r.status() === 200, { timeout: 120_000 })
  await expect(page.getByText(/圖像2為 PDF/)).toBeVisible()
  // MUI TextField(select) 的 label 關聯在不同 render 時機可能讓 getByLabel 不穩，
  // 用 role=combobox + name 包含「預覽/編輯頁」更可靠
  await expect(page.getByRole('combobox', { name: /預覽\/編輯頁/ })).toBeVisible()

  // 打開多印鑑編輯，至少可新增一個框並保存
  const editPdfPageBtn = page.getByRole('button', { name: '編輯此頁多印鑑框' })
  await expect(editPdfPageBtn).toBeEnabled({ timeout: 60_000 })
  await editPdfPageBtn.click()
  const image2Dialog = page.getByRole('dialog').filter({ hasText: '調整圖像2多印鑑位置' })
  await expect(image2Dialog).toBeVisible()
  await image2Dialog.getByRole('button', { name: '添加' }).click()
  await image2Dialog.getByRole('button', { name: /確認/ }).click()
  await expect(image2Dialog).toBeHidden()
})

test('PNG/JPG：既有多印鑑流程不回歸（基本 smoke）', async ({ page }) => {
  const image1Path = path.resolve(__dirname, '..', '..', 'test_images', '1-1.png')
  const image2Path = path.resolve(__dirname, '..', '..', 'test_images', '1-2.png')

  await page.goto('/multi-seal-test')

  // 圖像1（PNG）：上傳後可能自動偵測並自動保存，或開啟手動對話框
  await page.setInputFiles('#image1-upload', image1Path)
  const image1Dialog = page.getByRole('dialog').filter({ hasText: '調整圖像1印鑑位置' })
  // 先等背景「自動檢測」流程結束，避免競態：以 /detect-seal 的 network response 作為同步事實
  await page.waitForResponse((r) => r.url().includes('/detect-seal') && r.request().method() === 'POST', { timeout: 120_000 })

  // 若自動檢測失敗而已經開了 dialog，就直接確認；否則再手動打開確認一次，確保印鑑已標記
  if (await image1Dialog.isVisible().catch(() => false)) {
    await Promise.all([
      page.waitForResponse((r) => r.url().includes('/seal-location') && r.request().method() === 'PUT' && r.status() === 200),
      image1Dialog.getByRole('button', { name: '確認' }).click(),
    ])
    await expect(image1Dialog).toBeHidden({ timeout: 60_000 })
  } else {
    const editImage1Btn = page.getByRole('button', { name: '編輯印鑑位置' }).filter({ hasText: '編輯印鑑位置' })
    await expect(editImage1Btn).toBeVisible({ timeout: 60_000 })
    await editImage1Btn.click()
    await expect(image1Dialog).toBeVisible({ timeout: 60_000 })
    await Promise.all([
      page.waitForResponse((r) => r.url().includes('/seal-location') && r.request().method() === 'PUT' && r.status() === 200),
      image1Dialog.getByRole('button', { name: '確認' }).click(),
    ])
    await expect(image1Dialog).toBeHidden({ timeout: 60_000 })
  }
  // 確認 UI 進入「已標記」狀態，確保後續「開始比對多印鑑」不會因圖像1未標記而 disabled
  await expect(page.getByText('已標記')).toBeVisible()

  // 圖像2（PNG）：上傳後可編輯多印鑑框並開始比對
  await page.setInputFiles('#image2-upload', image2Path)
  // 若未自動跳出多印鑑 dialog，主動打開並至少新增 1 個框再確認，確保「開始比對多印鑑」可點
  // 同頁會出現 icon 與 fullWidth 兩個「編輯印鑑位置」，用 hasText 避免 strict mode
  const editImage2Btn = page.getByRole('button', { name: '編輯印鑑位置' }).filter({ hasText: '編輯印鑑位置' })
  await expect(editImage2Btn).toBeVisible({ timeout: 60_000 })
  await editImage2Btn.click()
  const image2Dialog = page.getByRole('dialog').filter({ hasText: '調整圖像2多印鑑位置' })
  await expect(image2Dialog).toBeVisible()
  // 盡量走最穩定路徑：新增一個框再確認
  if (await image2Dialog.getByRole('button', { name: '添加' }).isVisible().catch(() => false)) {
    await image2Dialog.getByRole('button', { name: '添加' }).click()
  }
  await image2Dialog.getByRole('button', { name: /確認/ }).click()
  await expect(image2Dialog).toBeHidden()

  // 防護：圖像1 的背景流程可能在稍後把 dialog 又打開（例如自動確認失敗）
  // 若 dialog 開著，背景區塊會被 aria-hidden，導致後續按鈕用 role 找不到。
  if (await image1Dialog.isVisible().catch(() => false)) {
    await Promise.all([
      page.waitForResponse((r) => r.url().includes('/seal-location') && r.request().method() === 'PUT' && r.status() === 200),
      image1Dialog.getByRole('button', { name: '確認' }).click(),
    ])
    await expect(image1Dialog).toBeHidden({ timeout: 60_000 })
  }

  // 觸發比對（只要能順利進到結果區塊即可）
  // 按鈕會在保存/裁切/比對階段顯示不同文字，先等待回到「開始比對多印鑑」再點擊
  const compareBtn = page.getByRole('button', { name: /開始比對多印鑑|保存中\\.\\.\\.|裁切中\\.\\.\\.|比對中\\.\\.\\./ })
  await expect(compareBtn).toBeVisible({ timeout: 60_000 })
  await expect(compareBtn).toHaveText('開始比對多印鑑', { timeout: 60_000 })
  await compareBtn.click()
  await expect(page.getByText(/比對結果/)).toBeVisible({ timeout: 120_000 })
})


