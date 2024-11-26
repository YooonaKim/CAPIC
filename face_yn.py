import cv2
import face_recognition
import os
import numpy as np

def extract_and_identify_faces_from_video(video_path):
    face_encodings = []  # 얼굴별 인코딩 저장
    face_images = []  # 얼굴별 이미지 저장
    identified_faces = []  # 식별된 얼굴별 (객체) 저장

    video_capture = cv2.VideoCapture(video_path)

    while video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break

      # face_locations = face_recognition.face_locations(frame)	# 현재 프레임에서 얼굴 위치 탐지
        face_locations = face_recognition.face_locations(frame, model='cnn')
        current_encodings = face_recognition.face_encodings(frame, face_locations) 	# 얼굴 위치에 대한 인코딩

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
                if dist < 0.56:  # 같은 사람으로 판단하는 임계값
                    face_group.append((face_images[idx], encoding))
                    matched = True
                    break
            if not matched:
                identified_faces.append([(face_images[idx], encoding)])

    video_capture.release()
    return identified_faces 	# 객체별 리턴

def save_faces(identified_faces):
    if not os.path.exists('saved_faces'):	# 얼굴별 디렉토리 생성
        os.makedirs('saved_faces')

    for i, group in enumerate(identified_faces):
        group_dir = f'saved_faces/group_{i}'
        if not os.path.exists(group_dir):
            os.makedirs(group_dir)
        for j, (face, _) in enumerate(group[:3]):		 # 인물별 최대 3개까지 저장
            cv2.imwrite(f'{group_dir}/face_{j}.jpg', face)

video_path = './video4.mp4' # 비디오 파일 삽입
identified_faces = extract_and_identify_faces_from_video(video_path)	# identified_faces에 저장
save_faces(identified_faces)	# 식별된 얼굴 저장

if identified_faces:
    print(f"'saved_faces'에 저장 완료.")
else:
    print("얼굴이미지없음.")


