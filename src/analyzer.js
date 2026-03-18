'use strict';

/**
 * LME Metal Fiyat Analiz Modülü
 * Spot fiyat, değişim, yüzde değişim ve trend analizi yapar.
 */

const METAL_LABELS = {
  copper: 'Bakır  (Cu)',
  zinc:   'Çinko  (Zn)',
  lead:   'Kurşun (Pb)',
};

function fmt(num, decimals = 2) {
  if (num === null || num === undefined || isNaN(num)) return 'N/A';
  return num.toLocaleString('tr-TR', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function sign(num) {
  if (num === null || num === undefined || isNaN(num)) return '';
  return num >= 0 ? '+' : '';
}

function trend(changePct) {
  if (changePct === null || changePct === undefined || isNaN(changePct)) return '━  Veri Yok';
  if (changePct > 1.5)  return '▲▲ Güçlü Yükseliş';
  if (changePct > 0)    return '▲  Yükseliş';
  if (changePct < -1.5) return '▼▼ Güçlü Düşüş';
  if (changePct < 0)    return '▼  Düşüş';
  return '━  Yatay';
}

function signal(changePct) {
  if (changePct === null || isNaN(changePct)) return '⚪ Veri Yok';
  if (changePct > 2)    return '🟢 AL';
  if (changePct > 0.5)  return '🟡 İZLE (Yükseliş)';
  if (changePct < -2)   return '🔴 SAT';
  if (changePct < -0.5) return '🟡 İZLE (Düşüş)';
  return '⚪ BEKLE';
}

/**
 * Fiyat nesnesinden değişim yüzdesini türet (eğer ham veri eksikse)
 */
function deriveChangePct(price) {
  if (price.changePct !== null && !isNaN(price.changePct)) return price.changePct;
  if (price.change !== null && price.spot !== null && price.spot !== 0) {
    const base = price.spot - price.change;
    return base !== 0 ? (price.change / base) * 100 : null;
  }
  return null;
}

function analyzePrice(key, price) {
  const changePct = deriveChangePct(price);
  return {
    key,
    label: METAL_LABELS[key] || price.name || key,
    spot: price.spot,
    change: price.change,
    changePct,
    trend: trend(changePct),
    signal: signal(changePct),
  };
}

function comparePrices(analyzed) {
  const valid = analyzed.filter(a => a.spot !== null);
  if (valid.length < 2) return null;

  const best  = valid.reduce((a, b) => (a.changePct > b.changePct ? a : b));
  const worst = valid.reduce((a, b) => (a.changePct < b.changePct ? a : b));

  return { best, worst };
}

function buildReport(scraped) {
  const { url, timestamp, pageTitle, prices } = scraped;
  const date = new Date(timestamp);
  const dateStr = date.toLocaleString('tr-TR', { timeZone: 'Europe/Istanbul' });

  const analyzed = Object.entries(prices).map(([key, p]) => analyzePrice(key, p));
  const comparison = comparePrices(analyzed);

  return { url, pageTitle, dateStr, analyzed, comparison };
}

function printReport(report) {
  const { url, pageTitle, dateStr, analyzed, comparison } = report;

  const line = '─'.repeat(60);
  const dline = '═'.repeat(60);

  console.log('\n' + dline);
  console.log('  🏭  LME METAL FİYAT ANALİZİ');
  console.log('  📅  ' + dateStr);
  console.log('  🌐  ' + url);
  if (pageTitle) console.log('  📄  ' + pageTitle);
  console.log(dline);

  for (const a of analyzed) {
    console.log('\n' + line);
    console.log(`  🔩 ${a.label}`);
    console.log(line);
    console.log(`  💵 Spot Fiyat  : ${fmt(a.spot)} USD/ton`);
    console.log(`  📊 Değişim     : ${sign(a.change)}${fmt(a.change)} USD`);
    console.log(`  📈 Değişim %   : ${sign(a.changePct)}${fmt(a.changePct)}%`);
    console.log(`  📉 Trend       : ${a.trend}`);
    console.log(`  🎯 Sinyal      : ${a.signal}`);
  }

  if (comparison) {
    console.log('\n' + dline);
    console.log('  📊 KARŞILAŞTIRMALI ANALİZ');
    console.log(dline);
    console.log(`  🏆 En İyi Performans : ${comparison.best.label}  (${sign(comparison.best.changePct)}${fmt(comparison.best.changePct)}%)`);
    console.log(`  📉 En Kötü Performans: ${comparison.worst.label} (${sign(comparison.worst.changePct)}${fmt(comparison.worst.changePct)}%)`);
  }

  // Özet tablo
  console.log('\n' + dline);
  console.log('  📋 ÖZET TABLO');
  console.log(dline);
  console.log(`  ${'Metal'.padEnd(15)} ${'Spot (USD/t)'.padStart(14)} ${'Δ USD'.padStart(10)} ${'Δ %'.padStart(8)}  Sinyal`);
  console.log('  ' + '─'.repeat(58));
  for (const a of analyzed) {
    const spotStr  = a.spot  !== null ? fmt(a.spot)             : 'N/A';
    const chStr    = a.change !== null ? sign(a.change) + fmt(a.change) : 'N/A';
    const pctStr   = a.changePct !== null ? sign(a.changePct) + fmt(a.changePct) + '%' : 'N/A';
    console.log(
      `  ${a.label.padEnd(15)} ${spotStr.padStart(14)} ${chStr.padStart(10)} ${pctStr.padStart(8)}  ${a.signal}`
    );
  }

  console.log('\n' + dline + '\n');
}

module.exports = { buildReport, printReport, analyzePrice };
