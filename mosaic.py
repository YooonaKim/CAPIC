import cv2
import os
from deepface import DeepFace
import torch


def mosaic(video_path, image_paths):
    model_name = "Facenet"

    output_video_path = os.path.join('tmp', 'output2.mp4')
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, (frame_width, frame_height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 동영상의 전체 프레임 수를 얻음

    print(f"Total number of frames in the video: {total_frames}")

    faces_dir = os.path.join('tmp', 'faces')
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)

    face_count = 0
    current_frame_count = 0

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

    threshold = 0.3
    not_threshold = 0.47

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # detections = RetinaFace.detect_faces(img_path=frame)
        detections = model(frame)

        print(f"{current_frame_count}감지 시작")

        for face_id in detections.xyxy[0]:
            x1, y1, x2, y2 = face_id[:4].int().tolist()
            if y2 - y1 > 50 and x2 - x1 > 50:
                face_image = frame[y1:y2, x1:x2]
                for ref_face in embedding_list:
                    result = DeepFace.verify(img1_path=face_image, img2_path=ref_face, model_name=model_name,
                                             detector_backend='retinaface', enforce_detection=False)
                    distance = result['distance']

                    if not_threshold >= distance >= threshold:
                        face_filename = f"face_{face_count}.jpg"
                        verified_str = 'Different'
                        distance_str = '(%.4f >= %.4f)' % (distance, threshold)
                        print(face_filename,verified_str, distance_str)
                        face = cv2.resize(face_image, (10, 10))
                        face = cv2.resize(face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
                        frame[y1:y2, x1:x2] = face
                        face_filepath = os.path.join(faces_dir, face_filename)
                        cv2.imwrite(face_filepath, face_image)
                        break

                    if distance < threshold:
                        face_filename = f"face_{face_count}.jpg"
                        verified_str = 'Same'
                        distance_str = '(%.4f >= %.4f)' % (distance, threshold)
                        print(face_filename, verified_str, distance_str)
                        face_filepath = os.path.join(faces_dir, face_filename)
                        cv2.imwrite(face_filepath, face_image)
                        break

                    if distance > not_threshold:
                        face_filename = f"D/face_{face_count}.jpg"
                        verified_str = 'Different'
                        distance_str = '(%.4f >= %.4f)' % (distance, threshold)
                        print(face_filename, verified_str, distance_str)
                        face = cv2.resize(face_image, (10, 10))
                        face = cv2.resize(face, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
                        frame[y1:y2, x1:x2] = face
                        face_filepath = os.path.join(faces_dir, face_filename)
                        cv2.imwrite(face_filepath, face_image)
                        break

                face_count += 1
        current_frame_count += 1
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return output_video_path


if __name__ == "__main__":
    import sys
    video_path = sys.argv[1]
    image_paths = ["save/train/Gongyoo/1.jpeg","save/train/Gongyoo/2.jpeg","save/train/Gongyoo/3.jpeg","save/train/Gongyoo/4.jpeg","save/train/Gongyoo/5.jpeg","save/train/Gongyoo/6.jpeg"]
    mosaic(video_path, image_paths)
