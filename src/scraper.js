'use strict';

const puppeteer = require('puppeteer-core');

const CHROMIUM_PATH = process.env.CHROMIUM_PATH ||
  '/root/.cache/ms-playwright/chromium-1194/chrome-linux/chrome';

const TARGET_URL = 'https://www.metalsmarket.net/lme-prices/';

const METALS = {
  copper: { tr: 'Bakır', keywords: ['copper', 'bakır', 'cu'] },
  zinc:   { tr: 'Çinko', keywords: ['zinc', 'çinko', 'zn'] },
  lead:   { tr: 'Kurşun', keywords: ['lead', 'kurşun', 'pb'] },
};

async function launchBrowser() {
  return puppeteer.launch({
    executablePath: CHROMIUM_PATH,
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
      '--no-first-run',
      '--no-zygote',
      '--single-process',
    ],
  });
}

async function setStealthHeaders(page) {
  await page.setUserAgent(
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ' +
    'AppleWebKit/537.36 (KHTML, like Gecko) ' +
    'Chrome/122.0.0.0 Safari/537.36'
  );
  await page.setExtraHTTPHeaders({
    'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache',
  });
}

/**
 * Sayfadan fiyat verilerini çeker.
 * metalsmarket.net'in HTML yapısı değişirse bu fonksiyon güncellenmeli.
 */
async function extractPrices(page) {
  return page.evaluate((metals) => {
    const results = {};

    // ---- Strateji 1: <table> içindeki satırları tara ----
    const rows = Array.from(document.querySelectorAll('table tr, .price-row, .metal-row, [class*="metal"], [class*="price"]'));
    for (const row of rows) {
      const text = row.innerText || row.textContent || '';
      const lower = text.toLowerCase();

      for (const [key, meta] of Object.entries(metals)) {
        if (results[key]) continue;
        const hit = meta.keywords.some(kw => lower.includes(kw));
        if (!hit) continue;

        // Sayısal değerleri topla
        const numbers = text.match(/[\d,.']+/g) || [];
        const parsed = numbers
          .map(n => parseFloat(n.replace(/,/g, '').replace(/'/g, '')))
          .filter(n => !isNaN(n) && n > 100); // LME fiyatları USD/ton, >100

        if (parsed.length > 0) {
          const cells = Array.from(row.querySelectorAll('td, th'));
          results[key] = {
            name: meta.tr,
            spot: parsed[0] || null,
            change: parsed[1] !== undefined ? parsed[1] : null,
            changePct: parsed[2] !== undefined ? parsed[2] : null,
            rawText: text.trim().substring(0, 200),
            cells: cells.map(c => c.innerText.trim()),
          };
        }
      }
    }

    // ---- Strateji 2: Tüm sayfada LME fiyat bölümlerini ara ----
    if (Object.keys(results).length < 3) {
      const allText = document.body.innerText;
      const lines = allText.split('\n').map(l => l.trim()).filter(Boolean);

      for (let i = 0; i < lines.length; i++) {
        const lower = lines[i].toLowerCase();
        for (const [key, meta] of Object.entries(metals)) {
          if (results[key]) continue;
          const hit = meta.keywords.some(kw => lower.includes(kw));
          if (!hit) continue;

          // Sonraki 5 satırda sayı ara
          const nearby = lines.slice(i, i + 6).join(' ');
          const numbers = nearby.match(/[\d,.]+/g) || [];
          const parsed = numbers
            .map(n => parseFloat(n.replace(/,/g, '')))
            .filter(n => !isNaN(n) && n > 100);

          if (parsed.length > 0) {
            results[key] = {
              name: meta.tr,
              spot: parsed[0] || null,
              change: null,
              changePct: null,
              rawText: nearby.substring(0, 200),
              cells: [],
            };
          }
        }
      }
    }

    // ---- Strateji 3: JSON-LD veya data attribute ----
    const scripts = Array.from(document.querySelectorAll('script[type="application/ld+json"], script[type="application/json"]'));
    for (const s of scripts) {
      try {
        const data = JSON.parse(s.textContent);
        // Yapıyı genel olarak tara
        const str = JSON.stringify(data).toLowerCase();
        for (const [key, meta] of Object.entries(metals)) {
          if (results[key]) continue;
          if (meta.keywords.some(kw => str.includes(kw))) {
            results[key] = results[key] || { name: meta.tr, spot: null, jsonData: data };
          }
        }
      } catch (_) {}
    }

    return results;
  }, metals);
}

async function scrape() {
  let browser;
  try {
    console.log('🚀 Tarayıcı başlatılıyor...');
    browser = await launchBrowser();
    const page = await browser.newPage();
    await setStealthHeaders(page);

    // İstekleri filtrele (resim, font, media'yı atla — hız için)
    await page.setRequestInterception(true);
    page.on('request', req => {
      const type = req.resourceType();
      if (['image', 'font', 'media', 'stylesheet'].includes(type)) {
        req.abort();
      } else {
        req.continue();
      }
    });

    console.log(`📡 ${TARGET_URL} adresine bağlanılıyor...`);
    await page.goto(TARGET_URL, {
      waitUntil: 'networkidle2',
      timeout: 30000,
    });

    // Sayfa başlığını doğrula
    const title = await page.title();
    console.log(`📄 Sayfa: ${title}`);

    // Dinamik içerik yüklenene kadar bekle
    await page.waitForTimeout(2000);

    // Fiyatları çıkar
    const prices = await extractPrices(page);

    // Sayfa kaynağını da kaydet (debug amaçlı isteğe bağlı)
    const pageUrl = page.url();

    return {
      url: pageUrl,
      timestamp: new Date().toISOString(),
      pageTitle: title,
      prices,
    };
  } finally {
    if (browser) await browser.close();
  }
}

module.exports = { scrape };
