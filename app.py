import os
from createTarget import extract_and_identify_faces_from_video
from yoona_target import arcface_recognition, group_and_save_faces
# import mosaic
import mosaic_jiyeon
# import deep2


from flask import (Flask, request, send_file, jsonify)

app = Flask(__name__)

@app.route('/target', methods=['POST'])
def process_video():
    video_file = request.files['video']

    video_filename = video_file.filename
    save_path = os.path.join('./', video_filename)
    video_file.save(save_path)


    identified_faces = extract_and_identify_faces_from_video(save_path)
    # face_base64_arrays = save_faces(identified_faces)  # 이미지를 Base64 인코딩된 문자열로 반환
    return jsonify({"images": identified_faces})  # JSON 객체로 변환

@app.route('/target2', methods=['POST'])
def yoona():
    video_path='./cutVideo.mp4'
    identified_faces = arcface_recognition(video_path)
    base64_faces = group_and_save_faces(identified_faces)
    # face_base64_arrays = save_faces(identified_faces)  # 이미지를 Base64 인코딩된 문자열로 반환
    return jsonify({"images": identified_faces})  # JSON 객체로 변환


@app.route('/video', methods=['POST'])
def handle_video():
    video_file = request.files['video']
    image_count = int(request.form['imageSize'])
    print(image_count)

    video_file.save(video_file.filename)

    image_paths = []
    for i in range(1, image_count + 1):
        image_file = request.files[f'image{i}']
        if image_file:
            filename, extension = os.path.splitext(image_file.filename)
            image_filename = os.path.join('save/train', f'image{i}{extension}')
            image_file.save(image_filename)
            image_paths.append(image_filename)
            print(f'Image {i} saved successfully.')

    # output_video_path = mosaic.mosaic(video_file.filename, image_paths)
    output_video_path = mosaic_jiyeon.mosaic(video_file.filename, image_paths)
    # output_video_path = deep2.mosaic(video_file.filename, image_paths)
    print(output_video_path)
    return send_file(output_video_path, mimetype='video/mp4', as_attachment=True, download_name='output_video.mp4')

@app.route('/image', methods=['POST'])
def image_test():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # 파일 저장 경로 지정, 여기서는 임시로 파일 이름으로 저장
    filename = os.path.join('/tmp', file.filename)
    file.save(filename)

    return send_file(filename, mimetype='image.jpeg', as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')