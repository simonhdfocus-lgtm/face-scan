# -*- coding: utf-8 -*-
"""
阿里巴巴国际站店铺适配器

国际站店铺是纯 JS 渲染，且商品详情页有 WAF 反爬，普通 HTTP 请求只能拿到跳转页。
方案：
  1. 商品列表 —— 走店铺公开的 pc_ajax/module.json 接口（无需登录），可枚举全部商品
  2. 详情长图 —— 需借助已登录浏览器；由 helper 生成 JS，在浏览器内并发 fetch 提取
  3. 图片下载 —— alicdn 图片可匿名下载，仍由 Python 完成
"""
import re
import json
import time
import requests
from urllib.parse import urlparse, parse_qs

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/124.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}

ALIBABA_SHOP_RE = re.compile(r'^https?://([\w\-]+)\.en\.alibaba\.com', re.I)


def is_alibaba_shop(url):
    """判断是否为国际站店铺链接"""
    if not url.startswith('http'):
        url = 'https://' + url
    return bool(ALIBABA_SHOP_RE.match(url))


def shop_root(url):
    if not url.startswith('http'):
        url = 'https://' + url
    m = ALIBABA_SHOP_RE.match(url)
    return f'https://{m.group(1)}.en.alibaba.com' if m else url.rstrip('/')


def discover_params(root, log=print):
    """
    从店铺商品列表页解析出调用 module.json 所需的 companyId / moduleId / pageId。
    这些 ID 每个店铺固定不变。
    """
    url = root + '/productlist.html'
    html = ''
    for attempt in range(4):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            html = r.text
            if 'module-id=' in html or 'companyId' in html:
                break
            log(f'店铺页返回异常（第 {attempt + 1} 次），稍后重试')
        except Exception as e:
            log(f'店铺页请求失败（第 {attempt + 1} 次）：{e}')
        time.sleep(3 + attempt * 3)

    def pick(*patterns):
        for p in patterns:
            m = re.search(p, html)
            if m:
                return m.group(1)
        return None

    company_id = pick(r'"companyId"\s*:\s*"?(\d+)', r'companyId=(\d+)', r'company_id["\']?\s*[:=]\s*["\']?(\d+)')
    page_id = pick(r'"pageId"\s*:\s*"?(\d+)', r'pageId=(\d+)', r'data-page-id="(\d+)"')
    # moduleId 挂在商品列表组件的 div 上：<div module-id="xxx" module-name="icbu-pc-productListPc">
    module_id = pick(
        r'module-id="(\d+)"[^>]*module-name="icbu-pc-productList',
        r'module-name="icbu-pc-productList[^"]*"[^>]*module-id="(\d+)"',
        r'"widgetId"%3A%22(\d+)%22',
        r'"moduleId"\s*:\s*"?(\d+)', r'moduleId=(\d+)',
    )

    log(f'店铺参数: companyId={company_id} pageId={page_id} moduleId={module_id}')
    return company_id, module_id, page_id


def fetch_product_list(root, company_id, module_id, page_id, log=print, max_products=2000):
    """
    调用 module.json 翻页枚举全部商品。
    返回 [{'id':..., 'url':..., 'subject':..., 'main_images':[...]}, ...]
    """
    api = root + '/pc_ajax/module.json'
    products, cur, total = [], 1, None
    failed_pages = []

    while True:
        params = {
            'companyId': company_id, 'moduleId': module_id, 'pageId': page_id,
            'path': 'products.htm', 'subPage': 'all', 'filter': 'all',
            'sortType': 'modified-desc', 'isGallery': 'Y', 'curPage': cur,
        }
        data = None
        for attempt in range(6):
            try:
                r = requests.get(api, params=params, headers=HEADERS, timeout=30)
                data = r.json()
                break
            except Exception:
                # 触发限流后需要较长冷却
                time.sleep(3 + attempt * 5)
        if data is None:
            log(f'商品列表第 {cur} 页多次获取失败，跳过该页')
            failed_pages.append(cur)
            cur += 1
            if total and len(products) + len(failed_pages) * 16 >= total:
                break
            if cur > (200 if total is None else (total // 16 + 3)):
                break
            continue

        try:
            block = data['moduleData']['data']
            lst = block.get('productList') or []
            nav = block.get('pageNavView') or {}
        except (KeyError, TypeError):
            log(f'第 {cur} 页返回结构异常，停止翻页')
            break

        if total is None:
            total = nav.get('totalLines')
            per = nav.get('pageLines') or 16
            log(f'店铺共 {total} 个商品，每页 {per} 个')

        if not lst:
            break

        products.extend(_parse_products(lst))

        log(f'已获取商品列表 {len(products)}/{total or "?"}')
        if len(products) >= (total or 0) or len(products) >= max_products:
            break
        cur += 1
        if cur > 200:
            break

    # 补抓此前失败的页，避免漏商品
    if failed_pages:
        log(f'补抓失败页：{failed_pages}')
        still_bad = []
        for pg in failed_pages:
            params = {
                'companyId': company_id, 'moduleId': module_id, 'pageId': page_id,
                'path': 'products.htm', 'subPage': 'all', 'filter': 'all',
                'sortType': 'modified-desc', 'isGallery': 'Y', 'curPage': pg,
            }
            ok = False
            for attempt in range(6):
                try:
                    time.sleep(2 + attempt * 4)
                    r = requests.get(api, params=params, headers=HEADERS, timeout=30)
                    lst = r.json()['moduleData']['data'].get('productList') or []
                    products.extend(_parse_products(lst))
                    ok = True
                    break
                except Exception:
                    continue
            if not ok:
                still_bad.append(pg)
        if still_bad:
            log(f'仍有 {len(still_bad)} 页未取到：{still_bad}')
        else:
            log(f'补抓完成，商品总数 {len(products)}')

    # 按商品 ID 去重
    seen, uniq = set(), []
    for p in products:
        if p['id'] and p['id'] not in seen:
            seen.add(p['id'])
            uniq.append(p)
    return uniq


def _parse_products(lst):
    """把接口返回的商品数组转成统一结构"""
    out = []
    for p in lst:
        u = p.get('url') or ''
        if u.startswith('//'):
            u = 'https:' + u
        imgs = []
        iu = p.get('imageUrls') or {}
        # 取最大尺寸并还原为原图
        for key in ('x960', 'x600', 'x350', 'x120'):
            if iu.get(key):
                im = iu[key]
                if im.startswith('//'):
                    im = 'https:' + im
                imgs.append(re.sub(r'\.(jpg|png|jpeg|webp)_\d+x\d+\.(jpg|png|jpeg|webp)$',
                                   r'.\1', im, flags=re.I))
                break
        for sk in (p.get('skuImg') or []):
            if isinstance(sk, str):
                s = 'https:' + sk if sk.startswith('//') else sk
                imgs.append(re.sub(r'\.(jpg|png|jpeg|webp)_\d+x\d+\.(jpg|png|jpeg|webp)$',
                                   r'.\1', s, flags=re.I))
        out.append({
            'id': str(p.get('id') or ''),
            'url': u,
            'subject': p.get('subject') or '',
            'main_images': imgs,
        })
    return out


def build_browser_script(product_ids, out_var='__scanResult'):
    """
    生成在已登录浏览器 Console 中执行的 JS：
    并发抓取商品详情页，提取全尺寸 alicdn 图片，结果写入 window[out_var]
    """
    ids = json.dumps(product_ids)
    return f"""
(async () => {{
  const ids = {ids};
  const CONC = 12, out = {{}};
  let i = 0;
  async function worker() {{
    while (i < ids.length) {{
      const id = ids[i++];
      try {{
        const r = await fetch(`https://www.alibaba.com/product-detail/x_${{id}}.html`,
                              {{credentials:'include'}});
        const t = await r.text();
        if (t.length < 20000) {{ out[id] = {{err:'blocked', len:t.length}}; continue; }}
        const imgs = [...new Set((t.match(
          /(?:https?:)?\\/\\/[a-z0-9\\-]*\\.alicdn\\.com\\/kf\\/[A-Za-z0-9]+\\.(?:jpg|jpeg|png|webp)/gi
        )||[]))].map(u => u.startsWith('//') ? 'https:'+u : u);
        out[id] = {{imgs}};
      }} catch(e) {{ out[id] = {{err:String(e).slice(0,80)}}; }}
    }}
  }}
  await Promise.all(Array.from({{length:CONC}}, worker));
  window.{out_var} = out;
  return JSON.stringify({{done:Object.keys(out).length,
                         imgs:Object.values(out).reduce((s,v)=>s+((v.imgs||[]).length),0)}});
}})()
"""


def collect_shop(site_url, log=print, max_products=2000):
    """
    入口：返回 (products, meta)
    products: 商品列表（含主图）
    meta: 店铺参数，供后续详情图抓取使用
    """
    root = shop_root(site_url)
    log(f'识别为阿里国际站店铺：{root}')
    company_id, module_id, page_id = discover_params(root, log)
    if not (company_id and page_id):
        raise RuntimeError('未能解析店铺参数，可能店铺页面结构有变化')
    products = fetch_product_list(root, company_id, module_id, page_id, log, max_products)
    meta = {'root': root, 'companyId': company_id, 'moduleId': module_id, 'pageId': page_id}
    return products, meta
