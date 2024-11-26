from deepface import DeepFace
import cv2
import os
import numpy as np

# 이미지가 있는 디렉토리
directory_path = r'save/train/Gongyoo'

# 저장할 디렉토리 경로
save_directory = r'save/train/yoo'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# 디렉토리 내의 모든 이미지 파일을 순회
for filename in os.listdir(directory_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일 형식 확인
        img_path = os.path.join(directory_path, filename).replace('\\', '/')
        img = cv2.imread(img_path)

        # RetinaFace로 얼굴 감지
        try:
            face = DeepFace.detectFace(img_path, detector_backend='retinaface', enforce_detection=False)

            # 감지된 얼굴 이미지 저장
            face_filename = os.path.join(save_directory, filename).replace('\\', '/')
            if face is not None:
                cv2.imshow("Original Face", face)
                cv2.waitKey(0)
                if face.dtype != np.uint8:
                    # 데이터 스케일링 및 타입 변환
                    face = np.clip(face * 255.0, 0, 255).astype(np.uint8)
                # face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)

                success = cv2.imwrite(face_filename, face)
                if success:
                    print(f"{face_filename} 저장됨")
                else:
                    print(f"{face_filename} 저장 실패")
            else:
                print(f"{filename}: 얼굴 감지 실패 또는 이미지 데이터 없음")

        except Exception as e:
            print(f"{filename} 처리 중 오류 발생: {str(e)}")
