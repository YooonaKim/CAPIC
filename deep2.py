import cv2
import os
from deepface import DeepFace
import torch


def mosaic(video_path, image_paths):
    model_name = "Facenet512"

    output_video_path = os.path.join('tmp', 'output.mp4')
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, (frame_width, frame_height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 동영상의 전체 프레임 수를 얻음

    print(f"Total number of frames in the video: {total_frames}")
    block_size = 10

    threshold = 0.465
    not_threshold = 0.55

    faces_dir = os.path.join('tmp', 'faces')
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)

    face_count = 0
    current_frame_count = 0

    # model = torch.hub.load('./yolov5', 'custom', path='macWideface.pt', force_reload=True, source='local')
    model = torch.hub.load('./yolov5', 'custom', path='best.pt', force_reload=True, source='local')

    embedding_list = []
    for image_path in image_paths:
        embedding_result = DeepFace.create_verification_result(
            img1_path=image_path,
            detector_backend='retinaface',
            model_name=model_name,
            enforce_detection=False
        )
        embedding_list.append(embedding_result["embeddings"][0])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # detections = RetinaFace.detect_faces(img_path=frame)
        detections = model(frame)

        print(f"{current_frame_count}감지 시작")

        for result in detections.xyxy[0]:
            if result[5] == 0:  # 클래스 인덱스가 0일 때(사람 얼굴을 의미하는 클래스)
                x1, y1, x2, y2 = result[:4].int().tolist()

                # 얼굴 영역 추출
                face_roi = frame[y1:y2, x1:x2]

                # 특정 사람과의 얼굴 일치 여부 확인
                match = False
                for ref_face in embedding_list:
                    similarity = DeepFace.verify(img1_path=face_roi, img2_path=ref_face,
                                                 model_name=model_name, enforce_detection=False)

                    if similarity["verified"] and not_threshold >= similarity['distance'] >= threshold:
                        face_filename = f"face_{face_count}.jpg"
                        verified_str = 'Different'
                        distance_str = '(%.4f <= %.4f)' % (similarity['distance'], threshold)
                        print(face_filename,verified_str, distance_str, similarity["verified"])
                        face_filepath = os.path.join(faces_dir, face_filename)
                        cv2.imwrite(face_filepath, face_roi)
                        face_count += 1

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
        current_frame_count += 1
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_video_path

if __name__ == "__main__":
    import sys
    video_path = sys.argv[1]
    image_paths = ["save/train/bbo/bbo.png","save/train/bbo/bbo2.png","save/train/bbo/bbo3.png","save/train/bbo/bbo4.png"]
    mosaic(video_path, image_paths)