# /usr/bin/env python

import os
import time
import datetime
from datetime import timedelta
import shutil

import copy
import json
import gzip, pickle

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torchvision import transforms
import torch.nn as nn
from torch.autograd import Variable

import warnings

warnings.filterwarnings('ignore')

import insightface
import onnxruntime

# ============================================================================================
# Models

PROVIDERS = onnxruntime.get_available_providers()

det_size = (320, 320)
# FACE_ANALYZER 및 FACE_SWAPPER는 프로젝트 내 'weights/' 폴더에 모델 파일이 존재해야 합니다.
FACE_ANALYZER = insightface.app.FaceAnalysis(name='buffalo_l', root='weights/', providers=PROVIDERS)
FACE_ANALYZER.prepare(ctx_id=0, det_size=det_size)

FACE_SWAPPER = insightface.model_zoo.get_model('weights/inswapper_128.onnx')


# ============================================================================================
# Functions

def swap_face(swapper, source_faces, target_faces, source_index, target_index, temp_frame):
    """실제 얼굴 합성 코어 로직."""
    source_face = source_faces[source_index]
    target_face = target_faces[target_index]
    temp_frame = swapper.get(temp_frame, target_face, source_face, paste_back=True)
    return temp_frame


# ============================================================================================
# 🌟🌟🌟 웹서버 page_6에서 사용하는 새로운 핵심 함수 🌟🌟🌟

def load_source_faces(source_paths: list):
    """
    여러 장의 사용자 얼굴 이미지 경로에서 각각 가장 큰 얼굴을 추출하여 얼굴 객체 리스트로 반환합니다.
    (page_6에서 사용자 얼굴 로드용으로 사용)
    """
    all_source_faces = []

    for src_path in source_paths:
        try:
            # 전달되는 경로는 이미 selfy_webserver.py에서 로컬 경로로 변환된 상태입니다.
            source_pil = Image.open(src_path).convert('RGB')
            source_np = np.array(source_pil)
            source_img = cv2.cvtColor(source_np, cv2.COLOR_RGB2BGR)

            current_faces = FACE_ANALYZER.get(source_img)

            if current_faces:
                current_faces = sorted(
                    current_faces,
                    key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                    reverse=True
                )
                all_source_faces.append(current_faces[0])  # 가장 큰 얼굴 1개만 사용

        except Exception as e:
            print(f"WARNING: Face analysis failed for source image {src_path}: {e}")
            continue

    return all_source_faces


def swap_and_save(source_face_to_use, target_path, output_path, target_face_index=0):
    """
    단일 소스 얼굴 객체를 대상 이미지의 특정 얼굴에 합성하고 저장합니다.
    (page_6에서 실제 합성 작업용으로 사용)
    """
    target_pil = Image.open(target_path).convert('RGB')
    target_np = np.array(target_pil)
    target_img = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)

    target_faces = []
    try:
        target_faces = FACE_ANALYZER.get(target_img)
        target_faces = sorted(
            target_faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
            reverse=True
        )
    except:
        raise Exception('대상 이미지에 얼굴이 인식되지 않았습니다.target_path')

    if not target_faces:
        raise Exception('대상 이미지에 합성할 유효한 얼굴이 인식되지 않았습니다.target_path')
        print("target_path")

    if target_face_index >= len(target_faces):
        print(f"WARNING: Target index {target_face_index} out of bounds. Using index 0.")
        target_face_index = 0

    temp_frame = copy.deepcopy(target_img)

    # swap_face 함수 호출
    temp_frame = swap_face(
        FACE_SWAPPER, [source_face_to_use], [target_faces[target_face_index]], 0, 0, temp_frame)

    result_img = cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_img)
    result_pil.save(output_path)

    print(f'Swap success: {output_path}')
    return output_path


# ============================================================================================
# 기존 함수 (유지)
# 이 함수들은 이전 버전 호환성을 위해 유지되지만, 현재 플로우에서는 load_source_faces와 swap_and_save가 주로 사용됩니다.
def faceswap(src_path, target_path, save_path, target_face_index=None):
    # 이 함수는 기존의 단일 소스 이미지 기반 합성 함수입니다.
    temp = shutil.copy(src_path, save_path)

    try:
        source_pil = Image.open(src_path).convert('RGB')
        source_img = np.array(source_pil)
        source_faces = []
        try:
            source_faces = FACE_ANALYZER.get(source_img)
            source_faces = sorted(
                source_faces,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True
            )
        except:
            raise Exception('원본 이미지에 얼굴이 인식되지 않았습니다.')

        target_pil = Image.open(target_path).convert('RGB')
        target_np = np.array(target_pil)

        target_img = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)
        target_faces = []
        try:
            target_faces = FACE_ANALYZER.get(target_img)
            target_faces = sorted(
                target_faces,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True
            )
        except:
            raise Exception('영화포스터 이미지에 얼굴이 인식되지 않았습니다.')

        num_target_faces = len(target_faces)
        num_source_faces = len(source_faces)

        if target_faces is not None and len(target_faces) > 0 and num_source_faces > 0:
            temp_frame = copy.deepcopy(target_img)

            if target_face_index is not None and 0 <= target_face_index < num_target_faces:
                source_index = 0
                target_index = target_face_index

                temp_frame = swap_face(
                    FACE_SWAPPER, source_faces, target_faces, source_index, target_index, temp_frame)

            else:
                num_swap_faces = np.min([num_source_faces, num_target_faces])
                for i in range(num_swap_faces):
                    source_index = i
                    target_index = i

                    temp_frame = swap_face(
                        FACE_SWAPPER, source_faces, target_faces, source_index, target_index, temp_frame)

            result_img = temp_frame
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            result_pil = Image.fromarray(result_img)

            result_pil.save(save_path)

        else:
            raise Exception('합성을 위한 유효한 얼굴을 찾을 수 없습니다.')

    except Exception as ex:
        print(str(ex))
    return save_path


def faceswap_multi(source_paths: list, target_path, dst_path, target_face_index=0):
    """
    기존 faceswap_multi 함수는 이제 swap_and_save 로직으로 대체되었지만, 코드를 유지합니다.
    """

    # 1. 모든 사용자 얼굴 이미지에서 얼굴을 추출합니다. (가장 큰 얼굴 1개씩)
    all_source_faces = load_source_faces(source_paths)  # 새로 만든 함수 사용

    num_source_faces = len(all_source_faces)

    if num_source_faces == 0:
        raise Exception('사용자 얼굴 사진에서 유효한 얼굴을 찾을 수 없습니다.')

    # 2. 대상 이미지에서 얼굴을 추출합니다.
    target_pil = Image.open(target_path).convert('RGB')
    target_np = np.array(target_pil)
    target_img = cv2.cvtColor(target_np, cv2.COLOR_BGR2RGB)

    target_faces = []
    try:
        target_faces = FACE_ANALYZER.get(target_img)
        target_faces = sorted(
            target_faces,
            key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
            reverse=True
        )
    except:
        raise Exception('대상 이미지에 얼굴이 인식되지 않았습니다.')

    num_target_faces = len(target_faces)

    # 3. Swap: 대상 이미지의 얼굴을 사용자 얼굴 목록 순서대로 바꿉니다.
    if target_faces is not None and len(target_faces) > 0:
        temp_frame = copy.deepcopy(target_img)

        source_face_to_swap = all_source_faces[0]
        target_face_to_swap = target_faces[target_face_index]  # AI가 선택한 인덱스 사용

        temp_frame = swap_face(
            FACE_SWAPPER, [source_face_to_swap], [target_face_to_swap], 0, 0, temp_frame)

        result_img = temp_frame
    else:
        raise Exception('대상 이미지에 합성할 얼굴이 인식되지 않았습니다.')

    # 4. 결과 저장
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(result_img)
    result_pil.save(dst_path)

    print(f'Swap success: {dst_path}')
    return dst_path