'use strict';

/**
 * metalsmarket.net login modülü
 * Kullanıcı adı ve şifre ile oturum açar.
 */

const LOGIN_URL = 'https://www.metalsmarket.net/login/';

// Olası login form selector kombinasyonları
const SELECTORS = {
  username: [
    'input[name="username"]',
    'input[name="email"]',
    'input[type="email"]',
    'input[id="username"]',
    'input[id="email"]',
    'input[placeholder*="user" i]',
    'input[placeholder*="email" i]',
    'input[placeholder*="kullanıcı" i]',
  ],
  password: [
    'input[name="password"]',
    'input[type="password"]',
    'input[id="password"]',
    'input[placeholder*="password" i]',
    'input[placeholder*="şifre" i]',
  ],
  submit: [
    'button[type="submit"]',
    'input[type="submit"]',
    'button[class*="login" i]',
    'button[class*="submit" i]',
    '.login-btn',
    '#login-btn',
  ],
};

async function findSelector(page, selectorList) {
  for (const sel of selectorList) {
    try {
      const el = await page.$(sel);
      if (el) return sel;
    } catch (_) {}
  }
  return null;
}

/**
 * Sayfada login formu varsa oturum açar.
 * @param {import('puppeteer-core').Page} page
 * @param {string} username
 * @param {string} password
 * @returns {Promise<boolean>} - true: başarılı, false: başarısız
 */
async function login(page, username, password) {
  console.log(`🔑 Login sayfasına yönlendiriliyor: ${LOGIN_URL}`);
  await page.goto(LOGIN_URL, { waitUntil: 'networkidle2', timeout: 30000 });

  // Form elementlerini bul
  const userSel = await findSelector(page, SELECTORS.username);
  const passSel = await findSelector(page, SELECTORS.password);

  if (!userSel || !passSel) {
    // Belki zaten login olmuşuz veya farklı bir URL'de
    const currentUrl = page.url();
    console.log(`⚠️  Login form bulunamadı. Mevcut URL: ${currentUrl}`);

    // Prices sayfasına gitmeyi dene, erişim varsa login gerekmiyordur
    return false;
  }

  console.log(`📝 Kullanıcı adı giriliyor (${userSel})`);
  await page.type(userSel, username, { delay: 60 });

  console.log(`🔒 Şifre giriliyor (${passSel})`);
  await page.type(passSel, password, { delay: 60 });

  const submitSel = await findSelector(page, SELECTORS.submit);
  if (submitSel) {
    console.log('🚪 Giriş yapılıyor...');
    await Promise.all([
      page.waitForNavigation({ waitUntil: 'networkidle2', timeout: 30000 }),
      page.click(submitSel),
    ]);
  } else {
    // Submit butonu yoksa Enter'a bas
    await Promise.all([
      page.waitForNavigation({ waitUntil: 'networkidle2', timeout: 30000 }),
      page.keyboard.press('Enter'),
    ]);
  }

  const afterUrl = page.url();
  console.log(`📍 Login sonrası URL: ${afterUrl}`);

  // Login başarı kontrolü: hâlâ login sayfasındaysak başarısız
  if (afterUrl.includes('/login') || afterUrl.includes('/signin')) {
    const errMsg = await page.$eval(
      '.error, .alert, [class*="error"], [class*="alert"]',
      el => el.innerText
    ).catch(() => 'Bilinmeyen hata');
    console.error(`❌ Login başarısız: ${errMsg}`);
    return false;
  }

  console.log('✅ Login başarılı!');
  return true;
}

module.exports = { login };
