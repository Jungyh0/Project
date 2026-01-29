# /usr/bin/env python

import os
import datetime
import json
import random
import shutil
from PIL import Image
import numpy as np
import cv2
import ssl

# Google GenAI Imports
import google.genai as genai
from google.genai.errors import APIError

# Flask Imports
# 🌟 (수정) jsonify 추가
from flask import Flask, request, render_template, session, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename

# Local Module
import selfy_swapper as swapper

# ==========================================================================================
# Configuration

# 🌟 config.json 파일이 존재한다고 가정합니다.
with open('./config.json', 'rb') as f:
    config_file = json.loads(f.read().decode())

SERVER_IP = config_file.get('server_ip')
SERVER_PORT = config_file.get('port')

# 디렉토리 설정
upload_dir = './static/upload/'
os.makedirs(upload_dir, exist_ok=True)
out_dir = './static/generated/'
os.makedirs(out_dir, exist_ok=True)
LOG_DIR = './logs/'
os.makedirs(LOG_DIR, exist_ok=True)

# 🌟🌟🌟 (수정) target -> targets (오타 수정)
TARGETS_BASE_DIR = './static/target/'

# 허용되는 파일 확장자
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# ==========================================================================================
# Flask App Setup

app = Flask(__name__)
app.secret_key = 'your_strong_secret_key_here'
app.config['UPLOAD_FOLDER'] = upload_dir

# ==========================================================================================
# Gemini AI Client Initialization

try:
    gemini_client = genai.Client()
    print("Gemini API Client initialized successfully.")
except Exception as e:
    print(f"Error initializing Gemini client: {e}")


def get_target_face_index(image_path: str, prompt_criterion: str) -> int:
    print("WARNING: get_target_face_index 함수가 호출되었으나, page_6에서는 사용하지 않습니다.")
    return 0


# ==========================================================================================
# Helper Functions

def allowed_file(filename):
    """허용된 확장자인지 확인합니다."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_url_to_local_path(url_or_path):
    """
    웹 URL 또는 웹 경로에서 로컬 파일 시스템 경로를 추출합니다.
    """
    if '://' in url_or_path:
        if '/static/' in url_or_path:
            path_part = url_or_path.split('/static/', 1)[-1]
            return os.path.join('static', path_part).replace('/', os.sep)
        else:
            return url_or_path

    elif url_or_path.startswith('/static/'):
        path_part = url_or_path[len('/static/'):]
        return os.path.join('static', path_part).replace('/', os.sep)

    else:
        return url_or_path.replace('/', os.sep)

    # ==========================================================================================


# Routes

@app.route('/', methods=['GET', 'POST'])
def index():
    print('index()')
    session.clear()
    msg = request.args.get('message', '')
    return render_template('index.html', message=msg)


@app.route('/page_2', methods=['GET', 'POST'])
def page_2():
    print('page_2()')
    session.clear()
    msg = request.args.get('message', '')
    return render_template('page_2.html', message=msg)


# ------------------------------------------------------------------------------------------
# 캡처 라우트 (page_3, page_3_2, ..., page_3_8)

@app.route('/page_3', methods=['GET', 'POST'])
def page_3():
    return handle_capture_page('page_3', next_page_url='page_3_2', max_captures=8)


@app.route('/page_3_2', methods=['GET', 'POST'])
def page_3_2():
    return handle_capture_page('page_3_2', next_page_url='page_3_3', max_captures=8)


@app.route('/page_3_3', methods=['GET', 'POST'])
def page_3_3():
    return handle_capture_page('page_3_3', next_page_url='page_3_4', max_captures=8)


@app.route('/page_3_4', methods=['GET', 'POST'])
def page_3_4():
    return handle_capture_page('page_3_4', next_page_url='page_3_5', max_captures=8)


@app.route('/page_3_5', methods=['GET', 'POST'])
def page_3_5():
    return handle_capture_page('page_3_5', next_page_url='page_3_6', max_captures=8)


@app.route('/page_3_6', methods=['GET', 'POST'])
def page_3_6():
    return handle_capture_page('page_3_6', next_page_url='page_3_7', max_captures=8)


@app.route('/page_3_7', methods=['GET', 'POST'])
def page_3_7():
    return handle_capture_page('page_3_7', next_page_url='page_3_8', max_captures=8)


@app.route('/page_3_8', methods=['GET', 'POST'])
def page_3_8():
    return handle_capture_page('page_3_8', next_page_url='page_4', max_captures=8)


def handle_capture_page(current_page_name, next_page_url, max_captures):
    """캡처 페이지 (page_3 ~ page_3_8)의 공통 로직을 처리합니다."""
    print(f'{current_page_name}()')

    if request.method == 'POST':
        try:
            file = request.files.get('up_file')

            if file and allowed_file(file.filename):

                if 'dt_id' not in session:
                    session['dt_id'] = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                dt_id = session['dt_id']

                save_dir = os.path.join(out_dir, dt_id)
                os.makedirs(save_dir, exist_ok=True)

                capture_list = session.get('capture_list', [])
                current_count = len(capture_list)

                if current_count >= max_captures:
                    return redirect(url_for(next_page_url))

                filename = f"capture_{current_count + 1:02d}.png"
                filepath = os.path.join(save_dir, filename)
                file.save(filepath)

                # 🌟 (수정) 'generated/'가 포함된 올바른 경로
                web_path = os.path.join(os.path.basename(out_dir), dt_id, filename).replace(os.sep, '/')
                capture_list.append(web_path)
                session['capture_list'] = capture_list

                print(f'Captured {len(capture_list)}/{max_captures} to {web_path}.')

                return redirect(url_for(next_page_url))
            else:
                return redirect(url_for(current_page_name, message='오류: 파일을 찾거나 형식을 저장할 수 없습니다.'))

        except Exception as ex:
            print(f"Capture Error on {current_page_name}: {str(ex)}")
            return redirect(url_for(current_page_name, message=f"오류: 서버 처리 중 문제가 발생했습니다. ({ex.__class__.__name__})"))

    # GET 요청 처리 (페이지 렌더링)
    dt_id = session.get('dt_id', '')
    capture_list = session.get('capture_list', [])
    msg = request.args.get('message', '')

    return render_template(f'{current_page_name}.html', message=msg, dt_id=dt_id, capture_list=capture_list)


# ------------------------------------------------------------------------------------------
# 선택 및 인화/합성 라우트

@app.route('/page_4', methods=['GET', 'POST'])
def page_4():
    print('page_4()')
    dt_id = session.get('dt_id', '')
    capture_list = session.get('capture_list', [])
    msg = request.args.get('message', '')

    if not dt_id or not capture_list:
        return redirect(url_for('page_2', message="오류: 캡처된 이미지가 없습니다. 다시 시작해주세요."))

    return render_template('page_4.html', message=msg, dt_id=dt_id, capture_list=capture_list)


@app.route('/page_4_2', methods=['GET', 'POST'])
def page_4_2():
    """인화 직전 최종 확인 페이지 (AI 합성 없이 인화)"""
    print('page_4_2()')

    if request.method == 'POST':
        dt_id = request.form.get('dt_id')
        select1 = request.form.get('select1')
        select2 = request.form.get('select2')
        select3 = request.form.get('select3')
        select4 = request.form.get('select4')

        if not all([select1, select2, select3, select4]):
            return redirect(url_for('page_4', message="4장의 사진을 모두 선택해야 인화할 수 있습니다."))

        user_face_urls = [select1, select2, select3, select4]
        session['source_image_paths'] = user_face_urls
        session['target_paths'] = []
        session['source_id'] = dt_id  # 🌟 (수정) source_dt_id -> source_id
        session['ai_mode'] = 'print_only'

        return render_template('page_4_2.html', dt_id=dt_id, select1=select1, select2=select2, select3=select3,
                               select4=select4)

    return redirect(url_for('page_4'))


# ------------------------------------------------------------------------------------------
# 🌟 (수정) /page_5 함수 (로딩 페이지만 렌더링)
# ------------------------------------------------------------------------------------------
@app.route('/page_5', methods=['GET', 'POST'])
def page_5():
    """
    (수정) AI 합성 준비만 하고 'page_5.html' (로딩 화면)을 띄웁니다.
    """
    print('page_5() - Setting up AI mode and rendering loading screen.')

    msg = request.args.get('message', '')

    if request.method == 'POST':
        try:
            # (이전과 동일) 폼 데이터와 AI 모드를 세션에 저장합니다.
            dt_id = request.form.get('dt_id')
            select1 = request.form.get('select1')
            select2 = request.form.get('select2')
            select3 = request.form.get('select3')
            select4 = request.form.get('select4')
            ai_mode = request.form.get('ai_mode')

            if not ai_mode or not select1:
                return redirect(url_for('page_4', message="오류: AI 모드 또는 사용자 얼굴 선택 데이터가 누락되었습니다."))

            user_face_urls = [select1, select2, select3, select4]

            if len(user_face_urls) != 4:
                return redirect(url_for('page_4', message="오류: 4장의 사용자 얼굴 선택 데이터가 필요합니다."))

            mode_dir = os.path.join(TARGETS_BASE_DIR, ai_mode)
            if not os.path.isdir(mode_dir):
                return redirect(url_for('page_4', message=f"오류: '{ai_mode}' 모드의 대상 이미지 폴더를 찾을 수 없습니다."))

            all_target_files = [f for f in os.listdir(mode_dir) if allowed_file(f)]

            if len(all_target_files) < 4:
                return redirect(url_for('page_4', message=f"오류: '{ai_mode}' 모드에 사용 가능한 대상 이미지가 4장 미만입니다."))

            random.seed(datetime.datetime.now().timestamp())

            num_to_select = 4
            selected_filenames = random.sample(all_target_files, num_to_select)

            selected_target_paths = [os.path.join(TARGETS_BASE_DIR.strip('./'), ai_mode, filename).replace(os.sep, '/')
                                     for filename in selected_filenames]

            session['source_image_paths'] = user_face_urls
            session['target_paths'] = selected_target_paths
            session['source_id'] = dt_id  # 🌟 (수정) source_dt_id -> source_id
            session['ai_mode'] = ai_mode

            # 🌟 (수정) /page_6으로 리다이렉트하는 대신, 로딩 페이지(page_5.html)를 렌더링합니다.
            return render_template('page_5.html', message="AI 합성 준비 완료. 잠시 후 시작합니다...")

        except Exception as ex:
            print(f"Error in page_5: {str(ex)}")
            msg = f'알 수 없는 에러가 발생하였습니다: {ex.__class__.__name__}'
            return redirect(url_for('page_4', message=msg))

    # GET 요청으로 page_5에 접근하면 page_4로 돌려보냄
    return redirect(url_for('page_4'))


# ------------------------------------------------------------------------------------------
# 🌟 (추가) /perform_swap 함수 (실제 합성 수행)
# ------------------------------------------------------------------------------------------
@app.route('/perform_swap', methods=['POST'])
def perform_swap():
    """
    (새로운 함수)
    page_5.html의 JavaScript(fetch) 호출을 받아 실제 합성을 수행하고,
    결과를 세션에 저장한 뒤, page_6으로 가라는 JSON 응답을 보냅니다.
    """
    print('perform_swap() - Starting background Face Swap process.')

    try:
        # 1. 세션에서 데이터 가져오기
        dt_id = session.get('source_id', '')
        source_urls = session.get('source_image_paths', [])
        target_local_paths = session.get('target_paths', [])
        ai_mode = session.get('ai_mode', 'default')

        if not source_urls or not target_local_paths:
            raise Exception('세션에서 소스 또는 타겟 이미지 정보를 찾을 수 없습니다.')

        # 2. 사용자 얼굴 로드 (1:1 매칭 확인)
        source_local_paths = [convert_url_to_local_path(url) for url in source_urls]
        all_source_faces = swapper.load_source_faces(source_local_paths)

        if not all_source_faces:
            raise Exception('사용자 얼굴 인식에 실패했습니다.')
        if len(all_source_faces) != len(target_local_paths):
            raise Exception('소스 얼굴 개수와 타겟 이미지 개수가 일치하지 않습니다.')

        # 3. 저장 디렉토리 생성
        result_dt_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        save_dir = os.path.join(out_dir, dt_id, result_dt_id)
        os.makedirs(save_dir, exist_ok=True)

        swap_list = []  # 결과 리스트

        # 4. 합성 실행 (1:1 매칭)
        for i, target_path_to_swap in enumerate(target_local_paths):
            source_face_to_use = all_source_faces[i]
            target_local_path = convert_url_to_local_path(target_path_to_swap)
            output_filename = f"{ai_mode}_{i + 1:02d}.png"
            output_path_local = os.path.join(save_dir, output_filename)

            swapper.swap_and_save(
                source_face_to_use=source_face_to_use,
                target_path=target_local_path,
                output_path=output_path_local
            )

            # 🌟 5. (버그 수정) 'os.path.basename(out_dir)' -> 'generated'로 직접 수정
            output_web_path = os.path.join(
                'generated',  # 👈👈👈 이 부분이 수정되었습니다!
                dt_id,
                result_dt_id,
                output_filename
            ).replace(os.sep, '/')

            # 🌟 6. 'static' 별명 사용 (이건 이미 올바르게 되어 있었습니다)
            swap_list.append(url_for('static', filename=output_web_path))

        # 7. 프레임 합성
        if swap_list:
            frame_pil = Image.open('static/assets/Seizmo_frame.jpg')
            frame_np = np.array(frame_pil)
            for dst_url in swap_list:
                dst_path = convert_url_to_local_path(dst_url)
                if os.path.exists(dst_path):
                    img_pil = Image.open(dst_path).convert('RGB')
                    img_np = np.array(img_pil)
                    img_np = cv2.resize(img_np, (400, 600))
                    frame_np_copy = frame_np.copy()
                    frame_np_copy[21: 621, 24: 424] = img_np
                    Image.fromarray(frame_np_copy).save(dst_path)
                    print(f"Frame applied to {dst_path}")
                else:
                    print(f"Skipping frame for {dst_path}, file not found.")

        # 8. 결과 리스트를 세션에 저장
        session['swap_list'] = swap_list
        session['swap_dt_id'] = dt_id

        # 9. 세션 정리
        session.pop('source_image_paths', None)
        session.pop('target_paths', None)
        session.pop('source_id', None)
        session.pop('ai_mode', None)

        # 10. 성공 JSON 응답 반환
        return jsonify({'status': 'success', 'redirect_url': url_for('page_6')})

    except Exception as ex:
        print(f"Fatal Error in perform_swap: {str(ex)}")
        msg = f'치명적 오류: 합성 실패 ({ex.__class__.__name__}). 다시 시도해 주세요.'
        # 11. 에러 JSON 응답 반환
        return jsonify({'status': 'error', 'message': msg, 'redirect_url': url_for('page_4', message=msg)})


# ------------------------------------------------------------------------------------------
# 🌟 (수정) /page_6 함수 (결과 페이지만 렌더링)
# ------------------------------------------------------------------------------------------
@app.route('/page_6', methods=['GET', 'POST'])
def page_6():
    """
    (수정) 합성 완료된 결과'만' 세션에서 가져와서 표시합니다.
    """
    print('page_6() - Displaying results.')

    # 1. 세션에서 합성 결과('swap_list')를 가져옵니다.
    swap_list = session.get('swap_list', [])
    dt_id = session.get('swap_dt_id', '')
    msg = request.args.get('message', '')

    # 2. 세션에서 결과 리스트를 삭제 (새로고침 시 중복 방지)
    session.pop('swap_list', None)
    session.pop('swap_dt_id', None)

    # 3. (수정) 모든 AI 합성 로직이 사라지고 렌더링만 남습니다.
    return render_template('page_6.html', swap_list=swap_list, dt_id=dt_id, message=msg)


# ------------------------------------------------------------------------------------------

@app.route('/page_7', methods=['GET', 'POST'])
def page_7():
    print('page_7()')
    msg = request.args.get('message', '')
    return render_template('page_7.html', message=msg)


# ------------------------------------------------------------------------------------------
# 🚫 사용하지 않는 라우트 (제거 또는 에러 처리)

@app.route('/upload_targets', methods=['GET', 'POST'])
def upload_targets():
    return redirect(url_for('page_4', message="오류: 잘못된 경로입니다. AI 모드를 선택해 주세요."))


# ==========================================================================================

if __name__ == '__main__':
    print('Starting Flask server with HTTPS...')

    ssl_context = ('cert.pem', 'key.pem')

    app.run(
        host='0.0.0.0',
        port=8000,
        debug=True,
        ssl_context=ssl_context
    )