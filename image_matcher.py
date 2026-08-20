# -*- coding: utf-8 -*-
"""
非人像图片相似识别

三种互补的比对方式，覆盖不同的"相似"含义：

1. pHash 感知哈希  —— 找「同一张图」：改尺寸、压缩、加水印、轻微裁剪后仍能认出
2. 颜色直方图      —— 找「同色调场景」：同一批次拍摄、同一背景的图
3. ORB 特征点匹配  —— 找「同一物体」：同一产品不同角度、局部出现在拼图里

综合三者给出最终相似度，避免单一方法的盲区。
"""
import cv2
import numpy as np


# ---------------- 感知哈希 ----------------
def phash(img, size=32, hash_size=8):
    """DCT 感知哈希，对缩放/压缩/轻微调色稳定"""
    if img is None:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(g))
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:, 1:])       # 跳过直流分量
    return (low > med).flatten()


def dhash(img, hash_size=8):
    """差值哈希，对整体亮度变化稳定，与 pHash 互补"""
    if img is None:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    g = cv2.resize(g, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def hash_similarity(h1, h2):
    """汉明距离转相似度 0~1"""
    if h1 is None or h2 is None:
        return 0.0
    return float(np.count_nonzero(h1 == h2) / len(h1))


# ---------------- 颜色直方图 ----------------
def color_hist(img, bins=(8, 8, 8)):
    """HSV 空间颜色分布，识别同色调、同场景"""
    if img is None:
        return None
    im = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(h, h)
    return h.flatten()


def hist_similarity(a, b):
    if a is None or b is None:
        return 0.0
    return float(max(0.0, cv2.compareHist(np.float32(a), np.float32(b),
                                          cv2.HISTCMP_CORREL)))


# ---------------- ORB 特征点 ----------------
_orb = cv2.ORB_create(nfeatures=500)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def orb_features(img, max_side=640):
    """提取 ORB 关键点描述子，用于同一物体识别"""
    if img is None:
        return None
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = g.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        g = cv2.resize(g, (int(w * s), int(h * s)))
    _, des = _orb.detectAndCompute(g, None)
    return des


def orb_similarity(d1, d2):
    """良好匹配点占比"""
    if d1 is None or d2 is None or len(d1) < 10 or len(d2) < 10:
        return 0.0
    try:
        matches = _bf.match(d1, d2)
    except cv2.error:
        return 0.0
    if not matches:
        return 0.0
    good = [m for m in matches if m.distance < 50]
    return float(len(good) / min(len(d1), len(d2)))


# ---------------- 综合特征 ----------------
def crop_hashes(img, ratios=(0.9, 0.8, 0.7, 0.6, 0.5)):
    """
    预先计算若干中心裁剪版本的哈希。
    用途：网站上常把同一张图裁掉边缘后再用，此时整图哈希会失配，
    用裁剪版本互相比对可以把这类"局部使用"找出来。
    """
    if img is None:
        return []
    h, w = img.shape[:2]
    out = []
    for r in ratios:
        dh, dw = int(h * (1 - r) / 2), int(w * (1 - r) / 2)
        sub = img[dh:h - dh, dw:w - dw]
        if sub.size and min(sub.shape[:2]) >= 32:
            out.append(phash(sub))
    return out


class ImageSignature:
    """一张图的完整特征集合"""

    __slots__ = ('phash', 'dhash', 'hist', 'orb', 'crops')

    def __init__(self, img):
        self.phash = phash(img)
        self.dhash = dhash(img)
        self.hist = color_hist(img)
        self.orb = orb_features(img)
        self.crops = crop_hashes(img)


def compare(sig_a, sig_b):
    """
    返回 (综合相似度, 匹配类型, 各项明细)
    匹配类型说明使用者该如何理解这次命中
    """
    ph = hash_similarity(sig_a.phash, sig_b.phash)
    dh = hash_similarity(sig_a.dhash, sig_b.dhash)
    hs = hist_similarity(sig_a.hist, sig_b.hist)
    ob = orb_similarity(sig_a.orb, sig_b.orb)

    # 交叉比对各自的裁剪版本：应对「同一素材裁掉边缘后再用」
    cross = 0.0
    for ca in [sig_a.phash] + list(sig_a.crops):
        for cb in [sig_b.phash] + list(sig_b.crops):
            v = hash_similarity(ca, cb)
            if v > cross:
                cross = v

    detail = {'phash': round(ph, 3), 'dhash': round(dh, 3),
              'hist': round(hs, 3), 'orb': round(ob, 3),
              'crop': round(cross, 3)}

    hash_avg = (ph + dh) / 2

    # 裁剪版本高度吻合，且色调一致 —— 判定为同素材的裁剪使用
    if cross >= 0.86 and cross > ph + 0.06 and hs >= 0.60:
        score = cross * 0.6 + hs * 0.25 + ob * 0.15
        return round(min(1.0, score), 4), '裁剪使用', detail

    # 同一张图：哈希高度一致
    if ph >= 0.90 and dh >= 0.85:
        return round(min(1.0, hash_avg), 4), '同一张图', detail

    # 同一张图的改动版：哈希较高，或哈希中等但特征点大量吻合
    if hash_avg >= 0.80 or (hash_avg >= 0.72 and ob >= 0.15):
        score = hash_avg * 0.7 + ob * 0.2 + hs * 0.1
        return round(min(1.0, score), 4), '改动版本', detail

    # 同一物体/场景：特征点吻合度高
    if ob >= 0.20:
        score = ob * 0.6 + hash_avg * 0.25 + hs * 0.15
        return round(min(1.0, score), 4), '同一物体', detail

    # 相似场景：色调与结构都接近
    if hs >= 0.75 and hash_avg >= 0.62:
        score = hs * 0.5 + hash_avg * 0.4 + ob * 0.1
        return round(min(1.0, score), 4), '相似场景', detail

    score = hash_avg * 0.5 + hs * 0.3 + ob * 0.2
    return round(min(1.0, score), 4), '低相关', detail
