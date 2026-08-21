# -*- coding: utf-8 -*-
"""素材排查网页服务：上传照片 + 网站链接 -> 自动检索命中图片"""
import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import io
import json
import uuid
import threading
import traceback
from datetime import datetime

from flask import Flask, request, jsonify, render_template, send_file, abort
from werkzeug.utils import secure_filename
from openpyxl import Workbook

import scanner

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 云端容器通常只有特定目录可写，用环境变量指定
UPLOAD_DIR = os.environ.get('UPLOAD_DIR', os.path.join(BASE_DIR, 'uploads'))
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024   # 20MB

JOBS = {}


def start_job(job_id, ref_path, site, threshold, max_pages, max_images, mode='face'):
    job = JOBS[job_id]
    try:
        scanner.run_scan(ref_path, site, job, threshold=threshold,
                         max_pages=max_pages, max_images=max_images, mode=mode)
    except Exception as e:
        job['status'] = 'error'
        job['error'] = f'{e}'
        job['logs'].append('错误：' + traceback.format_exc()[-500:])
    finally:
        # 参考图用完即删，不长期留存
        try:
            os.remove(ref_path)
        except OSError:
            pass


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/scan', methods=['POST'])
def api_scan():
    site = (request.form.get('site') or '').strip()
    if not site:
        return jsonify({'error': '请填写要检索的网站链接'}), 400
    f = request.files.get('photo')
    if not f or not f.filename:
        return jsonify({'error': '请上传参考照片'}), 400

    mode = (request.form.get('mode') or 'face').strip()
    if mode not in ('face', 'image', 'both'):
        mode = 'face'
    # 未指定阈值时按模式取默认值
    default_th = 0.363 if mode == 'face' else 0.80
    threshold = float(request.form.get('threshold') or default_th)
    max_pages = int(request.form.get('max_pages') or 3000)
    max_images = int(request.form.get('max_images') or 30000)

    job_id = uuid.uuid4().hex[:12]
    ext = os.path.splitext(secure_filename(f.filename))[1] or '.jpg'
    ref_path = os.path.join(UPLOAD_DIR, job_id + ext)
    f.save(ref_path)

    # 国际站店铺：若本地已缓存详情长图数据，自动带上，扫描更全
    detail_images = {}
    cache = os.path.join(BASE_DIR, '_detail_images.json')
    if 'en.alibaba.com' in site and os.path.exists(cache):
        try:
            detail_images = json.load(open(cache, encoding='utf-8'))
        except Exception:
            detail_images = {}

    JOBS[job_id] = {
        'id': job_id, 'site': site, 'status': 'running', 'stage': '排队中',
        'progress': 0, 'logs': [], 'hits': [], 'error': '',
        'total_pages': 0, 'done_pages': 0, 'total_images': 0, 'done_images': 0,
        'faces_found': 0, 'threshold': threshold, 'site_type': '',
        'mode': mode,
        'started': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ref_file': os.path.basename(ref_path),
        'detail_images': detail_images,
    }
    # 只保留最近若干个任务，避免历史结果长期占用内存；
    # 运行中和排队中的任务永不清理
    if len(JOBS) > 20:
        for old in sorted(JOBS, key=lambda k: JOBS[k].get('started', ''))[:-20]:
            if JOBS[old].get('status') not in ('running',):
                JOBS.pop(old, None)

    t = threading.Thread(target=start_job,
                         args=(job_id, ref_path, site, threshold,
                               max_pages, max_images, mode),
                         daemon=True)
    t.start()
    return jsonify({'job_id': job_id})


@app.route('/api/status/<job_id>')
def api_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': '任务不存在'}), 404
    skip = ('cancel', 'detail_images', 'product_ids')
    return jsonify({k: v for k, v in job.items() if k not in skip})


@app.route('/api/debug-ali')
def api_debug_ali():
    """临时诊断：查看云端访问阿里店铺页的真实返回"""
    import debug_ali
    return jsonify(debug_ali.probe())


@app.route('/api/jobs')
def api_jobs():
    """最近的扫描任务列表，供前端恢复与回看"""
    out = []
    for jid, j in JOBS.items():
        out.append({
            'id': jid,
            'site': j.get('site', ''),
            'mode': j.get('mode', 'face'),
            'status': j.get('status', ''),
            'stage': j.get('stage', ''),
            'progress': j.get('progress', 0),
            'hits': len(j.get('hits', [])),
            'started': j.get('started', ''),
            'queued': j.get('queued', False),
        })
    out.sort(key=lambda x: x['started'], reverse=True)
    return jsonify({'jobs': out[:12]})


@app.route('/api/cancel/<job_id>', methods=['POST'])
def api_cancel(job_id):
    job = JOBS.get(job_id)
    if job:
        job['cancel'] = True
        job['status'] = 'canceled'
        job['stage'] = '已取消'
    return jsonify({'ok': True})


@app.route('/api/export/<job_id>')
def api_export(job_id):
    job = JOBS.get(job_id)
    if not job:
        abort(404)
    wb = Workbook()
    ws = wb.active
    ws.title = '整改清单'
    ws.append(['序号', '相似度', '命中类型', '图片链接', '出现页面数', '所在页面链接'])
    for i, h in enumerate(job['hits'], 1):
        ws.append([i, h['score'], h.get('kind', '人脸匹配'), h['image_url'],
                   len(h['pages']), '\n'.join(h['pages'])])
    widths = [6, 10, 12, 70, 12, 80]
    for idx, w in enumerate(widths, 1):
        ws.column_dimensions[chr(64 + idx)].width = w
    bio = io.BytesIO()
    wb.save(bio)
    bio.seek(0)
    host = job['site'].replace('https://', '').replace('http://', '').replace('/', '_')
    name = f'素材排查_{host}_{datetime.now().strftime("%Y%m%d_%H%M")}.xlsx'
    return send_file(bio, as_attachment=True, download_name=name,
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f'素材排查工具已启动： http://127.0.0.1:{port}')
    app.run(host='0.0.0.0', port=port, threaded=True)
