# -*- coding: utf-8 -*-
"""临时诊断接口：查看云端请求阿里店铺页拿到什么"""
import requests

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/124.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
}


def probe():
    out = {}
    try:
        r = requests.get('https://hd-focus.en.alibaba.com/productlist.html',
                         headers=HEADERS, timeout=30)
        html = r.text
        out['status'] = r.status_code
        out['length'] = len(html)
        out['final_url'] = r.url
        out['has_module_id'] = 'module-id=' in html
        out['has_company'] = 'companyId' in html
        out['head'] = html[:600]
        out['resp_headers'] = dict(list(r.headers.items())[:12])
    except Exception as e:
        out['error'] = f'{type(e).__name__}: {e}'
    # 顺带看看出口 IP 归属
    try:
        ip = requests.get('https://api.ipify.org?format=json', timeout=15).json()
        out['egress_ip'] = ip.get('ip')
    except Exception as e:
        out['egress_ip'] = f'查询失败 {e}'
    return out
