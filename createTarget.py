import base64
from io import BytesIO

import cv2
import face_recognition
import numpy as np
from PIL import Image


def extract_and_identify_faces_from_video(video_path):
    face_encodings = []  # 얼굴별 인코딩 저장
    face_images = []  # 얼굴별 이미지 저장
    identified_faces = []  # 식별된 얼굴별 (객체) 저장

    video_capture = cv2.VideoCapture(video_path)

    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

        face_locations = face_recognition.face_locations(frame)  # 현재 프레임에서 얼굴 위치 탐지
        current_encodings = face_recognition.face_encodings(frame, face_locations)  # 얼굴 위치에 대한 인코딩

        for (top, right, bottom, left), encoding in zip(face_locations, current_encodings):
            # 얼굴 이미지 추출
            face_image = frame[top:bottom, left:right]
            face_images.append(face_image)

            face_encodings.append(encoding)

    # 인식된 얼굴 분류
    for idx, encoding in enumerate(face_encodings):
        if not identified_faces:
            identified_faces.append([(face_images[idx], encoding)])
        else:
            matched = False
            for face_group in identified_faces:
                group_encodings = [enc for _, enc in face_group]
                avg_encoding = np.mean(group_encodings, axis=0)
                dist = np.linalg.norm(avg_encoding - encoding)
                if dist < 0.555:  # 같은 사람으로 판단하는 임계값
                    face_group.append((face_images[idx], encoding))
                    matched = True
                    break
            if not matched:
                identified_faces.append([(face_images[idx], encoding)])

        # 얼굴 수가 많은 순서로 그룹 정렬
        identified_faces.sort(key=lambda x: len(x), reverse=True)

    video_capture.release()
    print('end1')

    # 인식된 얼굴 이미지를 Base64로 인코딩하여 반환
    return save_faces(identified_faces)


def save_faces(identified_faces):
    face_base64_arrays = []

    for face_group in identified_faces:
        encoded_faces = []
        count = 0  # 각 그룹별로 이미지 개수를 세는 카운터
        for face_image, _ in face_group:
            # OpenCV는 BGR 형식으로 이미지를 읽기 때문에 RGB로 변환
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            # 이미지를 PIL 이미지 객체로 변환
            pil_img = Image.fromarray(face_image)
            # 메모리 내에서 이미지를 저장하기 위한 버퍼 생성
            buf = BytesIO()
            # 이미지를 JPEG 포맷으로 저장
            pil_img.save(buf, format="JPEG")
            # 버퍼의 바이트 데이터를 Base64 인코딩 문자열로 변환
            base64_string = base64.b64encode(buf.getvalue()).decode('utf-8')
            # 해당 인물의 인코딩된 이미지를 추가
            encoded_faces.append(base64_string)
            count += 1
            if count == 3:  # 각 인물 그룹에서 최대 3개의 이미지만 저장
                break
        # 모든 인물의 인코딩된 이미지를 배열에 추가
        face_base64_arrays.append(encoded_faces)

    return face_base64_arrays