# -*- coding: utf-8 -*-
"""
离职员工素材排查引擎
sitemap/BFS 收集全站页面 -> 提取图片 -> 人脸检测与比对 -> 命中清单
"""
import os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in [os.path.join(_HERE, 'libs'), _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import re
import gc
import hashlib
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse, urldefrag

import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup

import alibaba_adapter
import image_matcher

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(BASE_DIR, 'models'))
MODEL_DET = os.path.join(MODEL_DIR, 'face_detection_yunet_2023mar.onnx')
MODEL_REC = os.path.join(MODEL_DIR, 'face_recognition_sface_2021dec.onnx')

# 模型不入代码库，首次启动时自动下载
MODEL_URLS = {
    MODEL_DET: 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/'
               'models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
    MODEL_REC: 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/'
               'models/face_recognition_sface/face_recognition_sface_2021dec.onnx',
}


def ensure_models():
    """确保人脸模型就位，缺失则下载"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    for path, url in MODEL_URLS.items():
        if os.path.exists(path) and os.path.getsize(path) > 100_000:
            continue
        print(f'正在下载模型 {os.path.basename(path)} ...', flush=True)
        import urllib.request
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=180) as r, open(path, 'wb') as f:
            f.write(r.read())
        print(f'  完成 {os.path.getsize(path) // 1024} KB', flush=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/124.0 Safari/537.36'
}
IMG_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
SKIP_EXT = ('.pdf', '.zip', '.rar', '.doc', '.docx', '.xls', '.xlsx', '.mp4', '.avi',
            '.mp3', '.exe', '.svg', '.ico', '.css', '.js')

_model_lock = threading.Lock()
_engine_lock = threading.Lock()
_shared_engine = None


def get_engine():
    """
    全局共享一个人脸模型实例。
    模型约 37MB，每个任务各加载一份会在低内存环境直接撑爆内存。
    实际比对时已有 _model_lock 保证线程安全，共享不会出错。
    """
    global _shared_engine
    with _engine_lock:
        if _shared_engine is None:
            _shared_engine = FaceEngine()
        return _shared_engine

# 低内存环境（如云端免费层 512MB）需要收紧并发与图片尺寸
LOW_MEM = os.environ.get('LOW_MEM', '0') == '1'
MAX_SIDE = int(os.environ.get('MAX_SIDE', '900' if LOW_MEM else '1600'))
MAX_WORKERS = int(os.environ.get('MAX_WORKERS', '3' if LOW_MEM else '12'))
MAX_IMG_BYTES = int(os.environ.get('MAX_IMG_BYTES', '6000000'))
# 单批处理的图片数。超过此数量会自动分批依次处理，每批结束释放内存，
# 使内存占用与站点规模无关——再大的站也能完整扫完。
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '600' if LOW_MEM else '5000'))

# 同时执行的扫描任务数上限。低内存环境必须串行，否则多人同时扫会撑爆内存；
# 超出的任务不会被拒绝，而是排队等待。
MAX_CONCURRENT = int(os.environ.get('MAX_CONCURRENT', '1' if LOW_MEM else '3'))
_task_slot = threading.Semaphore(MAX_CONCURRENT)
_queue_lock = threading.Lock()
_waiting = 0


# ---------------- 人脸模型 ----------------
class FaceEngine:
    def __init__(self):
        ensure_models()
        self.det = cv2.FaceDetectorYN.create(MODEL_DET, '', (320, 320), 0.7, 0.3, 5000)
        self.rec = cv2.FaceRecognizerSF.create(MODEL_REC, '')

    def features(self, img, max_side=None):
        """返回图中所有人脸的特征向量列表"""
        if img is None:
            return []
        max_side = max_side or MAX_SIDE
        h, w = img.shape[:2]
        if max(h, w) > max_side:  # 限制尺寸，兼顾速度与精度
            s = max_side / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)))
            h, w = img.shape[:2]
        if min(h, w) < 40:
            return []
        with _model_lock:
            self.det.setInputSize((w, h))
            try:
                _, faces = self.det.detect(img)
            except cv2.error:
                return []
            if faces is None:
                return []
            out = []
            for f in faces:
                try:
                    aligned = self.rec.alignCrop(img, f)
                    out.append(self.rec.feature(aligned).copy())
                except cv2.error:
                    continue
            return out

    def match(self, a, b):
        with _model_lock:
            return float(self.rec.match(a, b, cv2.FaceRecognizerSF_FR_COSINE))


# ---------------- 工具函数 ----------------
def norm_url(u):
    u, _ = urldefrag(u)
    return u.rstrip('/') if u.endswith('/') and urlparse(u).path != '/' else u


def same_site(url, root_host):
    try:
        h = urlparse(url).netloc.lower()
        return h == root_host or h.endswith('.' + root_host.replace('www.', '')) \
               or root_host.replace('www.', '') == h.replace('www.', '')
    except Exception:
        return False


def is_page(url):
    p = urlparse(url).path.lower()
    return not p.endswith(SKIP_EXT + IMG_EXT)


def fetch(url, timeout=20, is_binary=False):
    r = requests.get(url, headers=HEADERS, timeout=timeout, stream=is_binary)
    r.raise_for_status()
    return r


# ---------------- 链接收集 ----------------
def urls_from_sitemap(root, log, limit=50000):
    """递归解析 sitemap，返回页面 URL 集合"""
    host = urlparse(root).netloc
    found, seen_maps = set(), set()
    queue = [urljoin(root, '/sitemap.xml'), urljoin(root, '/sitemap-image.xml')]

    # robots.txt 里声明的 sitemap
    try:
        rb = fetch(urljoin(root, '/robots.txt'), timeout=10).text
        for m in re.findall(r'(?i)sitemap:\s*(\S+)', rb):
            queue.append(m.strip())
    except Exception:
        pass

    while queue and len(found) < limit:
        sm = queue.pop(0)
        if sm in seen_maps:
            continue
        seen_maps.add(sm)
        try:
            txt = fetch(sm, timeout=20).text
        except Exception:
            continue
        soup = BeautifulSoup(txt, 'xml')
        for node in soup.find_all('sitemap'):
            loc = node.find('loc')
            if loc and loc.text.strip() not in seen_maps:
                queue.append(loc.text.strip())
        for node in soup.find_all('url'):
            loc = node.find('loc')
            if not loc:
                continue
            u = norm_url(loc.text.strip())
            if same_site(u, host) and is_page(u):
                found.add(u)
        log(f'解析 sitemap: {sm} → 累计 {len(found)} 个页面')
    return found


def crawl_bfs(root, log, limit, stop_flag):
    """BFS 兜底抓站内链接"""
    host = urlparse(root).netloc
    seen, pages = {norm_url(root)}, []
    q = deque([norm_url(root)])
    while q and len(pages) < limit and not stop_flag():
        u = q.popleft()
        try:
            html = fetch(u).text
        except Exception:
            continue
        pages.append(u)
        soup = BeautifulSoup(html, 'lxml')
        for a in soup.find_all('a', href=True):
            nu = norm_url(urljoin(u, a['href']))
            if nu not in seen and same_site(nu, host) and is_page(nu) and nu.startswith('http'):
                seen.add(nu)
                q.append(nu)
        if len(pages) % 10 == 0:
            log(f'BFS 已抓取 {len(pages)} 个页面，队列 {len(q)}')
    return pages


# ---------------- 图片提取 ----------------
LAZY_ATTRS = ['src', 'data-src', 'data-original', 'data-lazy-src', 'data-echo',
              'data-url', 'data-image', 'data-lazyload']


def extract_images(page_url, html):
    """从页面 HTML 提取所有图片 URL（含懒加载与 CSS 背景图）"""
    soup = BeautifulSoup(html, 'lxml')
    urls = set()

    def add(raw):
        if not raw:
            return
        raw = raw.strip()
        if raw.startswith('data:') or not raw:
            return
        full = urljoin(page_url, raw)
        if full.startswith('http'):
            path = urlparse(full).path.lower()
            if path.endswith(IMG_EXT) or '/uploads/' in path or 'img' in path:
                urls.add(full)

    for tag in soup.find_all(['img', 'source']):
        for attr in LAZY_ATTRS:
            add(tag.get(attr))
        for sattr in ('srcset', 'data-srcset'):
            if tag.get(sattr):
                for part in tag[sattr].split(','):
                    add(part.strip().split(' ')[0])

    for tag in soup.find_all(style=True):
        for m in re.findall(r'url\((.*?)\)', tag['style']):
            add(m.strip('\'"'))

    for m in re.findall(r'background(?:-image)?\s*:\s*url\((.*?)\)', html, re.I):
        add(m.strip('\'"'))

    return urls


def canonical_img(u):
    """去掉 CDN 尺寸参数，避免同一素材重复下载"""
    p = urlparse(u)
    return f'{p.scheme}://{p.netloc}{p.path}'


# ---------------- 主扫描流程 ----------------
def run_scan(ref_image_path, site_url, job, threshold=0.363,
             max_pages=3000, max_images=30000, workers=8, mode='face'):
    """对外入口：先排队拿到执行名额，再真正开扫"""
    global _waiting
    with _queue_lock:
        _waiting += 1
        ahead = _waiting - 1
    if not _task_slot.acquire(blocking=False):
        # 前面有任务在跑，本任务进入等待
        job['stage'] = f'排队中（前面还有 {ahead} 个任务）'
        job['queued'] = True
        job['logs'].append(
            f'[{time.strftime("%H:%M:%S")}] 当前有其他扫描正在进行，已排队等待')
        _task_slot.acquire()
    job['queued'] = False
    try:
        return _run_scan_inner(ref_image_path, site_url, job, threshold,
                               max_pages, max_images, workers, mode)
    finally:
        _task_slot.release()
        with _queue_lock:
            _waiting -= 1
        gc.collect()


def _run_scan_inner(ref_image_path, site_url, job, threshold=0.363,
                    max_pages=3000, max_images=30000, workers=8, mode='face'):
    """
    job: dict，用于向前端汇报进度；含 log/progress/status/results 等字段
    threshold: 相似度阈值。人脸模式 0.363（SFace 官方推荐），图片模式建议 0.80
    mode: face=人脸比对 / image=图片相似 / both=两者都查
    """
    def log(msg):
        job['logs'].append(f'[{time.strftime("%H:%M:%S")}] {msg}')
        # 低内存环境少留日志；切片会复制列表，用 del 原地裁剪更省
        keep = 80 if LOW_MEM else 300
        if len(job['logs']) > keep * 2:
            del job['logs'][:-keep]

    def stopped():
        return job.get('cancel', False)

    job['mode'] = mode
    need_face = mode in ('face', 'both')
    need_image = mode in ('image', 'both')

    # 1. 读取参考图并提取特征
    job['stage'] = '读取参考图片'
    ref_img = cv2.imdecode(np.fromfile(ref_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if ref_img is None:
        job['status'] = 'error'
        job['error'] = '参考图片无法读取，请换一张 jpg/png 格式的图片'
        return

    engine = get_engine() if need_face else None
    ref_feat = None
    if need_face:
        ref_feats = engine.features(ref_img)
        if ref_feats:
            ref_feat = ref_feats[0]
            log(f'参考图人脸提取成功（检测到 {len(ref_feats)} 张脸，取最主要一张）')
        elif mode == 'face':
            job['status'] = 'error'
            job['error'] = ('参考图中未检测到人脸。若要查找非人像素材，'
                            '请把识别模式改为「图片相似」')
            return
        else:
            log('参考图中未检测到人脸，本次仅按图片相似比对')
            need_face = False

    ref_sig = None
    if need_image:
        ref_sig = image_matcher.ImageSignature(ref_img)
        log('参考图图像特征提取完成（哈希 / 色调 / 特征点）')

    if not site_url.startswith('http'):
        site_url = 'https://' + site_url
    root = site_url.rstrip('/')

    img_map = {}   # 规范化图片URL -> {url, pages[]}
    lock = threading.Lock()

    # ===== 分支：阿里国际站店铺走专用适配器 =====
    if alibaba_adapter.is_alibaba_shop(root):
        job['site_type'] = 'alibaba'
        job['stage'] = '枚举店铺商品'
        try:
            products, meta = alibaba_adapter.collect_shop(root, log=log,
                                                          max_products=max_pages * 10)
        except Exception as e:
            job['status'] = 'error'
            job['error'] = f'国际站店铺解析失败：{e}'
            return
        job['total_pages'] = len(products)
        job['done_pages'] = len(products)
        job['shop_meta'] = meta
        job['product_ids'] = [p['id'] for p in products if p['id']]
        job['progress'] = 25
        log(f'商品枚举完成，共 {len(products)} 个商品')

        # 商品主图直接入库
        for p in products:
            for iu in p['main_images']:
                key = canonical_img(iu)
                rec = img_map.setdefault(key, {'url': iu, 'pages': []})
                if p['url'] and p['url'] not in rec['pages']:
                    rec['pages'].append(p['url'])
        log(f'已收录商品主图 {len(img_map)} 张')

        # 详情长图：由浏览器侧回填（若已提供）
        detail_map = job.get('detail_images') or {}
        if detail_map:
            id2url = {p['id']: p['url'] for p in products}
            cnt = 0
            for pid, urls in detail_map.items():
                purl = id2url.get(str(pid), '')
                for iu in urls:
                    key = canonical_img(iu)
                    rec = img_map.setdefault(key, {'url': iu, 'pages': []})
                    if purl and purl not in rec['pages']:
                        rec['pages'].append(purl)
                    cnt += 1
            log(f'已合并详情页长图 {cnt} 条，去重后图片总数 {len(img_map)}')
        else:
            log('提示：未提供详情长图数据，本次仅扫描商品主图')

        pages = [p['url'] for p in products]
        job['progress'] = 40
    else:
        # ===== 通用网站流程 =====
        job['site_type'] = 'generic'
        job['stage'] = '收集全站链接'
        pages = set()
        try:
            pages = urls_from_sitemap(root, log)
        except Exception as e:
            log(f'sitemap 解析失败：{e}')
        if len(pages) < 5:
            log('sitemap 结果偏少，改用站内 BFS 抓取')
            pages = set(crawl_bfs(root, log, max_pages, stopped))
        pages.add(norm_url(root))
        pages = list(pages)[:max_pages]
        job['total_pages'] = len(pages)
        log(f'共收集到 {len(pages)} 个页面')

        # 3. 抓页面提图片
        job['stage'] = '抓取页面图片'
        _collect_generic_images(pages, img_map, lock, job, log, stopped, workers)

    _scan_images(img_map, max_images, engine, ref_feat, ref_sig,
                 need_face, need_image, threshold,
                 job, log, stopped, workers, lock)


def _collect_generic_images(pages, img_map, lock, job, log, stopped, workers):
    """通用网站：并发抓取页面并提取图片"""
    done_pages = 0

    def handle_page(pu):
        nonlocal done_pages
        if stopped():
            return
        try:
            html = fetch(pu).text
        except Exception:
            html = ''
        imgs = extract_images(pu, html) if html else set()
        with lock:
            for iu in imgs:
                key = canonical_img(iu)
                rec = img_map.setdefault(key, {'url': iu, 'pages': []})
                if pu not in rec['pages']:
                    rec['pages'].append(pu)
            done_pages += 1
            job['done_pages'] = done_pages
            job['progress'] = round(done_pages / max(len(pages), 1) * 40, 1)
            if done_pages % 10 == 0 or done_pages == len(pages):
                log(f'已解析 {done_pages}/{len(pages)} 个页面，累计发现 {len(img_map)} 张图片')
            if LOW_MEM and done_pages % 50 == 0:
                gc.collect()

    workers = min(workers, MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(handle_page, pages))


def _scan_images(img_map, max_images, engine, ref_feat, ref_sig,
                 need_face, need_image, threshold,
                 job, log, stopped, workers, lock):
    """
    下载图片并按所选模式做比对。

    图片数量超过单批上限时自动分批：每批处理完释放该批占用的内存再进入下一批，
    命中结果跨批累积。这样无论站点多大都能全部扫完，而不是丢弃超出的部分。
    """
    items = list(img_map.values())[:max_images]
    img_map.clear()          # 索引已转成列表，及时释放
    gc.collect()
    total = len(items)
    job['total_images'] = total

    batch_size = max(1, BATCH_SIZE if total > BATCH_SIZE else total)
    n_batch = max(1, (total + batch_size - 1) // batch_size)
    job['total_batches'] = n_batch
    if n_batch > 1:
        log(f'去重后待检测图片 {total} 张，将分 {n_batch} 批依次处理（每批 {batch_size} 张）')
    else:
        log(f'去重后待检测图片 {total} 张')

    stage_name = {(True, False): '人脸检测与比对',
                  (False, True): '图片相似比对',
                  (True, True): '人脸 + 图片比对'}.get((need_face, need_image), '比对中')
    job['stage'] = stage_name
    hits, checked, faces_total = [], 0, 0
    seen_md5 = set()

    def handle_img(rec):
        nonlocal checked, faces_total
        if stopped():
            with lock:
                checked += 1
                job['done_images'] = checked
            return
        iu = rec['url']
        new_hit = False
        try:
            r = fetch(iu, timeout=25, is_binary=True)
            content = r.content
            if len(content) < 2000:      # 太小的图（图标等）跳过
                raise ValueError('too small')
            if len(content) > MAX_IMG_BYTES:   # 超大图跳过，防止内存溢出
                raise ValueError('too large')
            md5 = hashlib.md5(content).hexdigest()
            with lock:
                if md5 in seen_md5:
                    return          # 内容重复，计数由 finally 统一处理
                seen_md5.add(md5)
            arr = np.frombuffer(content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            del content, arr, r
            if img is None:
                raise ValueError('decode failed')
            # 低内存环境先缩图再处理，避免大图占满内存
            ih, iw = img.shape[:2]
            if max(ih, iw) > MAX_SIDE:
                sc = MAX_SIDE / max(ih, iw)
                img = cv2.resize(img, (int(iw * sc), int(ih * sc)),
                                 interpolation=cv2.INTER_AREA)

            best, kind, detail, nface = 0.0, '', None, 0

            # 人脸比对
            if need_face and ref_feat is not None:
                feats = engine.features(img)
                nface = len(feats)
                if feats:
                    with lock:
                        faces_total += 1
                    fscore = max(engine.match(ref_feat, f) for f in feats)
                    if fscore >= threshold:
                        best, kind = fscore, '人脸匹配'

            # 图片相似比对
            if need_image and ref_sig is not None:
                iscore, ikind, idetail = image_matcher.compare(
                    ref_sig, image_matcher.ImageSignature(img))
                if iscore >= threshold and iscore > best:
                    best, kind, detail = iscore, ikind, idetail

            if best >= threshold and kind:
                with lock:
                    hits.append({
                        'image_url': iu,
                        'pages': rec['pages'],
                        'score': round(best, 4),
                        'faces': nface,
                        'kind': kind,
                        'detail': detail,
                    })
                    new_hit = True
                    log(f'★ 命中[{kind}] {best:.3f} → {iu.split("/")[-1][:48]}')
            del img
        except Exception:
            pass
        finally:
            with lock:
                checked += 1
                job['done_images'] = checked
                job['faces_found'] = faces_total
                # 命中列表每 10 张才重排一次，避免每张图都产生一份排序副本
                if new_hit or checked % 10 == 0:
                    job['hits'] = sorted(hits, key=lambda x: -x['score'])
                job['progress'] = round(40 + checked / max(total, 1) * 60, 1)
                if checked % 25 == 0 or checked == total:
                    extra = f'，含人脸 {faces_total} 张' if need_face else ''
                    log(f'已检测 {checked}/{total} 张图片{extra}，命中 {len(hits)} 张')
                # 低内存环境定期回收，避免容器被 OOM 杀掉
                if LOW_MEM and checked % 10 == 0:
                    gc.collect()

    workers = min(workers, MAX_WORKERS)
    for bi in range(n_batch):
        if stopped():
            log('已手动停止')
            break
        batch = items[bi * batch_size:(bi + 1) * batch_size]
        if not batch:
            break
        job['cur_batch'] = bi + 1
        if n_batch > 1:
            job['stage'] = f'{stage_name}（第 {bi + 1}/{n_batch} 批）'
            log(f'--- 开始第 {bi + 1}/{n_batch} 批，本批 {len(batch)} 张 ---')
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(handle_img, batch))
        # 本批结束：释放该批数据与去重缓存，保证内存不跨批累积
        items[bi * batch_size:(bi + 1) * batch_size] = [None] * len(batch)
        del batch
        if LOW_MEM and len(seen_md5) > 4000:
            seen_md5.clear()
        gc.collect()
        if n_batch > 1:
            log(f'--- 第 {bi + 1}/{n_batch} 批完成，累计命中 {len(hits)} 张 ---')

    job['hits'] = sorted(hits, key=lambda x: -x['score'])
    job['progress'] = 100
    job['stage'] = '已完成'
    job['status'] = 'done'
    log(f'扫描完成：{job.get("total_pages", 0)} 个页面 / {checked} 张图片 / 命中 {len(hits)} 张')
