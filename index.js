'use strict';

/**
 * metalsmarket.net — LME Bakır, Çinko ve Kurşun Spot Fiyat Scraper & Analiz
 *
 * Kullanım:
 *   node index.js --user <email> --pass <şifre>
 *   METALS_USER=email METALS_PASS=sifre node index.js
 *
 * Seçenekler:
 *   --user, -u   : Kullanıcı adı / e-posta
 *   --pass, -p   : Şifre
 *   --no-login   : Login olmadan çalıştır (açık erişim)
 *   --json       : Sonucu JSON olarak çıktıla
 *   --out <file> : Sonucu dosyaya kaydet
 *   --interval <dakika> : Tekrarlı çalıştır (örn: --interval 30)
 */

const puppeteer = require('puppeteer-core');
const path      = require('path');
const fs        = require('fs');
const { login }       = require('./src/login');
const { buildReport, printReport } = require('./src/analyzer');

// ─── Sabitler ────────────────────────────────────────────────────────────────
const CHROMIUM_PATH = process.env.CHROMIUM_PATH ||
  '/root/.cache/ms-playwright/chromium-1194/chrome-linux/chrome';

const TARGET_URL = 'https://www.metalsmarket.net/lme-prices/';

const METALS = {
  copper: { tr: 'Bakır',  keywords: ['copper', 'bakır', 'cu', 'lme copper'] },
  zinc:   { tr: 'Çinko',  keywords: ['zinc',   'çinko', 'zn', 'lme zinc']  },
  lead:   { tr: 'Kurşun', keywords: ['lead',   'kurşun','pb', 'lme lead']  },
};

// ─── Argüman Ayrıştırma ───────────────────────────────────────────────────────
function parseArgs() {
  const args = process.argv.slice(2);
  const opts = {
    username: process.env.METALS_USER || '',
    password: process.env.METALS_PASS || '',
    noLogin:  false,
    json:     false,
    outFile:  null,
    interval: null,
  };
  for (let i = 0; i < args.length; i++) {
    if ((args[i] === '--user' || args[i] === '-u') && args[i + 1]) opts.username = args[++i];
    else if ((args[i] === '--pass' || args[i] === '-p') && args[i + 1]) opts.password = args[++i];
    else if (args[i] === '--no-login') opts.noLogin = true;
    else if (args[i] === '--json')     opts.json = true;
    else if (args[i] === '--out' && args[i + 1]) opts.outFile = args[++i];
    else if (args[i] === '--interval' && args[i + 1]) opts.interval = parseInt(args[++i], 10);
  }
  return opts;
}

// ─── Tarayıcı Başlatma ────────────────────────────────────────────────────────
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

// ─── Stealth Header ───────────────────────────────────────────────────────────
async function setStealthHeaders(page) {
  await page.setUserAgent(
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ' +
    'AppleWebKit/537.36 (KHTML, like Gecko) ' +
    'Chrome/122.0.0.0 Safari/537.36'
  );
  await page.setExtraHTTPHeaders({
    'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
  });
}

// ─── Kaynak Filtreleme ────────────────────────────────────────────────────────
async function enableResourceFilter(page) {
  await page.setRequestInterception(true);
  page.on('request', req => {
    if (['image', 'font', 'media'].includes(req.resourceType())) {
      req.abort();
    } else {
      req.continue();
    }
  });
}

// ─── Fiyat Çekme ──────────────────────────────────────────────────────────────
async function extractPrices(page, metals) {
  return page.evaluate((metals) => {
    const results = {};

    // Strateji 1: Tablo satırları
    const rows = Array.from(document.querySelectorAll(
      'table tr, .price-row, .metal-row, [class*="metal"], [class*="price"], [class*="commodity"]'
    ));

    for (const row of rows) {
      const text  = (row.innerText || row.textContent || '').trim();
      const lower = text.toLowerCase();

      for (const [key, meta] of Object.entries(metals)) {
        if (results[key]) continue;
        if (!meta.keywords.some(kw => lower.includes(kw))) continue;

        const cells = Array.from(row.querySelectorAll('td, th'))
          .map(c => c.innerText.trim());

        const numbers = text.match(/[\d,.']+/g) || [];
        const parsed  = numbers
          .map(n => parseFloat(n.replace(/[,']/g, '')))
          .filter(n => !isNaN(n) && n > 100);

        if (parsed.length === 0) continue;

        results[key] = {
          name:      meta.tr,
          spot:      parsed[0]                                   ?? null,
          change:    parsed.length > 1 ? parsed[1]              : null,
          changePct: parsed.length > 2 ? parsed[2]              : null,
          cells,
          source:    'table-row',
        };
      }
    }

    // Strateji 2: Metin satırları tarama
    if (Object.keys(results).length < Object.keys(metals).length) {
      const lines = document.body.innerText.split('\n').map(l => l.trim()).filter(Boolean);
      for (let i = 0; i < lines.length; i++) {
        const lower = lines[i].toLowerCase();
        for (const [key, meta] of Object.entries(metals)) {
          if (results[key]) continue;
          if (!meta.keywords.some(kw => lower.includes(kw))) continue;

          const block   = lines.slice(i, i + 8).join(' ');
          const numbers = block.match(/[\d,.]+/g) || [];
          const parsed  = numbers
            .map(n => parseFloat(n.replace(/,/g, '')))
            .filter(n => !isNaN(n) && n > 100);

          if (parsed.length > 0) {
            results[key] = {
              name:      meta.tr,
              spot:      parsed[0],
              change:    parsed[1] ?? null,
              changePct: parsed[2] ?? null,
              cells:     [],
              source:    'text-scan',
            };
          }
        }
      }
    }

    // Strateji 3: JSON-LD / Script tag
    for (const script of document.querySelectorAll('script')) {
      if (!script.textContent.includes('{')) continue;
      try {
        const raw  = script.textContent.trim();
        const json = JSON.parse(raw.startsWith('{') || raw.startsWith('[') ? raw : '{}');
        const str  = JSON.stringify(json).toLowerCase();
        for (const [key, meta] of Object.entries(metals)) {
          if (results[key]) continue;
          if (meta.keywords.some(kw => str.includes(kw))) {
            // JSON'dan fiyat çekmeyi dene
            const nums = str.match(/["':]\s*([\d.]+)\s*["',}/]/g) || [];
            const parsed = nums
              .map(m => parseFloat(m.replace(/[^0-9.]/g, '')))
              .filter(n => !isNaN(n) && n > 100);
            if (parsed.length > 0) {
              results[key] = {
                name: meta.tr,
                spot: parsed[0],
                change: null,
                changePct: null,
                cells: [],
                source: 'json-ld',
              };
            }
          }
        }
      } catch (_) {}
    }

    return results;
  }, metals);
}

// ─── Ana Scrape Fonksiyonu ────────────────────────────────────────────────────
async function runScrape(opts) {
  let browser;
  try {
    console.log('\n🚀 Tarayıcı başlatılıyor...');
    browser = await launchBrowser();
    const page = await browser.newPage();
    await setStealthHeaders(page);
    await enableResourceFilter(page);

    const hasCredentials = !opts.noLogin && opts.username && opts.password;

    // HTTP Basic Auth (bazı siteler bunu kullanır)
    if (hasCredentials) {
      await page.authenticate({ username: opts.username, password: opts.password });
    } else if (!opts.noLogin) {
      console.warn('ℹ️  Kullanıcı bilgisi yok. (--user, --pass veya METALS_USER/METALS_PASS env kullanın)');
    }

    // LME fiyat sayfasına git
    console.log(`📡 Fiyat sayfasına bağlanılıyor: ${TARGET_URL}`);
    let gotoResult;
    try {
      gotoResult = await page.goto(TARGET_URL, { waitUntil: 'networkidle2', timeout: 30000 });
    } catch (err) {
      if (err.message.includes('ERR_INVALID_AUTH_CREDENTIALS') && hasCredentials) {
        // Basic Auth olmayabilir — form login dene
        console.log('🔄 Basic Auth çalışmadı, form login deneniyor...');
        await page.authenticate(null); // Basic auth'u devre dışı bırak
        const ok = await login(page, opts.username, opts.password);
        if (!ok) console.warn('⚠️  Form login de başarısız.');
        await page.goto(TARGET_URL, { waitUntil: 'networkidle2', timeout: 30000 });
      } else {
        throw err;
      }
    }

    // Basic Auth başarılı ama form login de gerekiyorsa
    if (hasCredentials && gotoResult && gotoResult.status() === 401) {
      console.log('🔄 Basic Auth yetersiz, form login deneniyor...');
      await page.authenticate(null);
      await login(page, opts.username, opts.password);
      await page.goto(TARGET_URL, { waitUntil: 'networkidle2', timeout: 30000 });
    }

    const title = await page.title();
    console.log(`📄 Sayfa: ${title}`);

    // Dinamik içerik için bekle
    await new Promise(r => setTimeout(r, 2000));

    // Fiyatları çıkar
    const prices = await extractPrices(page, METALS);

    const foundCount = Object.keys(prices).length;
    if (foundCount === 0) {
      console.error('❌ Hiç fiyat verisi bulunamadı!');
      console.log('🔍 Sayfa HTML\'i inceleniyor...');
      const html = await page.content();
      const snippet = html.substring(0, 2000);
      console.log(snippet);
    } else {
      console.log(`✅ ${foundCount}/3 metal verisi bulundu.`);
    }

    return {
      url:       page.url(),
      pageTitle: title,
      timestamp: new Date().toISOString(),
      prices,
    };
  } finally {
    if (browser) await browser.close();
  }
}

// ─── Çıktı ────────────────────────────────────────────────────────────────────
function outputResult(scraped, opts) {
  const report = buildReport(scraped);

  if (opts.json) {
    const jsonOut = JSON.stringify({ ...scraped, analyzed: report.analyzed }, null, 2);
    console.log(jsonOut);
    if (opts.outFile) fs.writeFileSync(opts.outFile, jsonOut, 'utf8');
  } else {
    printReport(report);
    if (opts.outFile) {
      // Dosyaya da kaydet
      const jsonOut = JSON.stringify({ ...scraped, analyzed: report.analyzed }, null, 2);
      fs.writeFileSync(opts.outFile, jsonOut, 'utf8');
      console.log(`💾 Sonuç kaydedildi: ${opts.outFile}`);
    }
  }
}

// ─── Giriş Noktası ───────────────────────────────────────────────────────────
async function main() {
  const opts = parseArgs();

  if (opts.interval && opts.interval > 0) {
    console.log(`⏱️  ${opts.interval} dakikada bir çalıştırılacak. Durdurmak için Ctrl+C.`);
    const run = async () => {
      try {
        const scraped = await runScrape(opts);
        outputResult(scraped, opts);
      } catch (err) {
        console.error('❌ Hata:', err.message);
      }
    };
    await run();
    setInterval(run, opts.interval * 60 * 1000);
  } else {
    try {
      const scraped = await runScrape(opts);
      outputResult(scraped, opts);
    } catch (err) {
      console.error('❌ Kritik hata:', err.message);
      if (process.env.DEBUG) console.error(err.stack);
      process.exit(1);
    }
  }
}

main();
