import cv2
import torch
import os
import time
from deepface import DeepFace
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
import gc

def mosaic(video_path, image_paths):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # print("deepdepp")
    # print(device_lib.list_local_devices())
    gc.collect()
    torch.cuda.empty_cache()
    print(tf.test.is_gpu_available())

    start_time = time.time()  # 시작 시간 기록
    # YOLOv5 모델 로드
    model = torch.hub.load('pytorch/yolov5', 'custom', path='./best.pt', source='local')  # 모델 로드
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 디바이스 설정
    # torch.cuda.empty_cache()
    # model.to(device)

    output_video_path = os.path.join('pytorch/tmp', video_path)

    # 모자이크 처리할 사이즈 정의
    block_size = 10

    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)

    # 결과 동영상 파일 생성
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, (frame_width, frame_height))

    embedding_list = []
    for image_path in image_paths:
        embedding_result = DeepFace.create_verification_result(
            img1_path=image_path,
            detector_backend='retinaface',
            model_name='Facenet',
            enforce_detection=False
        )
        embedding_list.append(embedding_result["embeddings"][0])

    # 동영상 프레임마다 처리
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv5를 사용하여 객체 감지
        results = model(frame)

        threshold = 0.465

        # 감지된 얼굴에 모자이크 처리
        for result in results.xyxy[0]:
            if result[5] == 0:  # 클래스 인덱스가 0일 때(사람 얼굴을 의미하는 클래스)
                x1, y1, x2, y2 = result[:4].int().tolist()

                # 얼굴 영역 추출
                face_roi = frame[y1:y2, x1:x2]

                # 얼굴 이미지 크기 조정
                face_roi_resized = cv2.resize(face_roi, (224, 224))
                cv2.imwrite("pytorch/output_image.jpeg", face_roi_resized)

                # 특정 사람과의 얼굴 일치 여부 확인
                match = False
                for ref_face in embedding_list:
                    image = cv2.imread(image_path)
                    similarity = DeepFace.verify(img1_path= face_roi_resized, img2_path= ref_face, model_name ='Facenet512', enforce_detection=False)
                    if similarity["verified"] and similarity['distance'] < threshold:  # 유사성이 임계값보다 크면 얼굴이 일치한다고 판단
                        match = True
                        break
                # os.remove("output_image.jpg")

                if match:  # 특정 사람과 일치하지 않는 경우에만 모자이크 처리
                    # 얼굴 영역에 모자이크 처리
                    continue

                blurred_face = cv2.resize(face_roi, (block_size, block_size))
                blurred_face = cv2.resize(blurred_face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
                frame[y1:y2, x1:x2] = blurred_face

                # 모자이크 처리된 프레임 결과 동영상에 추가
        out.write(frame)

            # 작업 완료 후 파일 닫기
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    end_time = time.time()  # 종료 시간 기록
    elapsed_time = end_time - start_time  # 소요된 시간 계산
    print(f"걸린 시간: {elapsed_time} 초")

    return output_video_path

if __name__ == "__main__":
    import sys
    video_path = sys.argv[1]
    image_paths = ["save/train/Gongyoo/1.jpeg","save/train/Gongyoo/2.jpeg","save/train/Gongyoo/3.jpeg","save/train/Gongyoo/4.jpeg","save/train/Gongyoo/5.jpeg","save/train/Gongyoo/6.jpeg"]
    mosaic(video_path, image_paths)
#     import os
#
#     os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#     print("GPU 사용 가능 여부:", tf.test.is_gpu_available())