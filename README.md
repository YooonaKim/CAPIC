![header](https://capsule-render.vercel.app/api?type=waving&color=BF92FB&height=300&section=header&text=capic&fontSize=50&fontColor=FFF&fontAlignY=40&desc=영상%20자동%20모자이크%20서비스&descAlign=80)
<br>

## 작품 개요
capic은 영상에 등장하는 얼굴을 추출하여 선별하고, 자동으로 모자이크 처리된 동영상을 만들어주는 서비스입니다.<br>
사용자들은 더욱 자유롭게 자신들의 영상을 공유할 수 있는 환경이 마련되어 다채로운 콘텐츠를 만드는 데에 기여합니다.<br><br>
<strong><a href="https://drive.google.com/file/d/13O_nF0qPKMlHoY94VeWhbGRIJdn8Af98/view?usp=drive_link">시연영상 보러가기</a></strong><br>
<strong>영상 용량에 의해 미리보기가 제공되지 않습니다. 다운로드 후 시청 부탁드립니다.</strong>

<br><br>

## 기대 효과
* <strong>자유로운 비디오 공유</strong>: 개인이 자신의 비디오를 더 자유롭게 공유할 수 있게 되어, <br>
공공장소에서의 촬영 부담이 줄어든다. 이로 인해 사용자는 더 창의적이고 다양한 콘텐츠를 만들 수 있는 환경이 조성된다.<br><br>

* <strong>프라이버시 보호와 사회적 이익 증진</strong>: 광고, 교육, 의료, 보안 등 다양한 분야에서 개인의 프라이버시를 보호하고, <br>
이를 통해 사회적 이익을 증진시킬 수 있다.<br><br>

* <strong>효율적인 비디오 관리</strong>: 저희 서비스는 '블러미'와 같은 유사 서비스와 비교하여 더욱 발전된 기능을 제공한다. <br>
영상에서 자주 노출되는 얼굴을 자동으로 식별하고, 사용자가 원하지 않는 얼굴만 선택적으로 블러 처리할 수 있어 <br>
 사용자의 필요에 따라 안전하고 맞춤화된 경험을 제공할 수 있다.
<br><br>

## 팀원
<table>
  <tr> 
    <td><a href="https://github.com/finenana"><img src="https://github.com/Capic2024/capic-react/blob/main/capic/src/profile/yn.jpeg" style="width:150px;"></a></td>
    <td><a href="https://github.com/sycuuui"><img src="https://github.com/Capic2024/capic-react/blob/main/capic/src/profile/sy.jpeg" style="width:150px;"></a></td>
    <td><a href="https://github.com/sengooooo"><img src="https://github.com/MutsaMarket/MutsaMarket-apk/assets/77336664/67df1c19-c0e0-490a-afa8-b8cf97e9bc40" style="width:150px;"></a></td>
    <td><a href="https://github.com/ljy6712"><img src="https://github.com/Capic2024/capic-react/blob/main/capic/src/profile/jy.png" style="width:150px;"></a></td>
    
  </tr>
  <tr> 
    <td align='center'><strong>김유나</strong></td> 
    <td align='center'><strong>이서연</strong></td> 
    <td align='center'><strong>이세은</strong></td> 
    <td align='center'><strong>이지연</strong></td> 
  </tr>
</table>
<br><br>

## 작품 설명
### 1)시작페이지
<p float="center">
  <img src="https://github.com/Capic2024/capic-react/blob/main/capic/src/profile/start.png" width="500" />
  <img src="https://github.com/Capic2024/capic-react/blob/main/capic/src/profile/file.png" width="500" />
</p>

* 사용자가 모자이크 처리를 원하는 동영상을 선택합니다.<br>

### 2)얼굴추출
<p float="center">
  <img src="https://github.com/Capic2024/capic-react/blob/main/capic/src/profile/before.png" width="500" />
  <img src="https://github.com/Capic2024/capic-react/blob/main/capic/src/profile/mosaicloading.png" width="500" />
</p>

* 영상에 나오는 인물들 얼굴을 추출하여 보여줍니다.
* 사용자가 모자이크 처리 제외할 얼굴을 선택합니다.
* 영상에 나오는 빈도에 따라 많이 나온 순서대로 추천합니다.
* 선택이 완료되면, 모자이크 처리되는 동안 로딩 페이지를 보여줍니다.<br><br>
* 구현 방법<br>
  동영상을 프레임 단위로 사람들의 얼굴을 추출하였습니다.<br> <strong>OpenCV</strong>로 각 프레임을 읽고, <strong>insightface</strong> 라이브러리를 이용해 얼굴들 위치를 탐지하여 특징에 따라 각 얼굴 인코딩을 생성합니다.<br> <strong>Numpy</strong>를 이용해 인코딩 간의 거리 계산법을 사용하여 유사도에 따라 인물 별로 식별합니다.<br> 최종적으로 슬라이더에 추출된 얼굴의 빈도수에 따라 순서대로 보여집니다.<br>
  
### 3)모자이크
<p float="center">
  <img src="https://github.com/Capic2024/capic-react/blob/main/capic/src/profile/after.png" width="500" />
</p>

* 변환된 영상 미리보기를 제공합니다.
* 변환된 영상을 사용자가 다운로드 받을 수 있습니다.<br><br>
* 구현 방법<br>
  객체 탐지 모델인 <strong>yolov5</strong> 모델을 사람의 얼굴만 탐지를 할 수 있도록 <strong>wider-face dataset</strong>을 이용해서 학습시켰습니다.<br>
  모자이크 처리 시에 특정 사람의 얼굴만 제외하기 위해 <strong>deepface</strong> 라이브러리의 <strong>facenet</strong> 모델을 사용하였고<br> 소요 시간을 줄이기 위해서 <strong>라이브러리를 튜닝</strong>하였습니다.

<br><br>

## 아키텍처
<p float="left">
  <img src="https://github.com/Capic2024/capic-react/blob/main/capic/src/profile/architecture.jpeg" width="800" />
</p>
<br><br>

## 서비스 구조
<p float="left">
  <img src="https://github.com/Capic2024/capic-react/blob/main/capic/src/profile/image.png" width="800" />
</p>
<br><br>

## 주요기술
* 개발 언어<br>
![HTML5](https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E)
![Python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)
![Java17](https://img.shields.io/badge/java-%23ED8B00.svg?style=for-the-badge&logo=JAVA&logoColor=white)

* 사용 기술<br>
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8.svg?style=for-the-badge&logo=OpenCV&logoColor=white)
![Insightface](https://img.shields.io/badge/Insightface-FF6F61.svg?style=for-the-badge)
![YOLOv5](https://img.shields.io/badge/YOLOv5-00FFFF.svg?style=for-the-badge)
![DeepFace](https://img.shields.io/badge/DeepFace-FFD700.svg?style=for-the-badge)
![Spring Boot](https://img.shields.io/badge/Spring%20Boot-6DB33F.svg?style=for-the-badge&logo=Spring-Boot&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000.svg?style=for-the-badge&logo=Flask&logoColor=white)
![Amazon S3](https://img.shields.io/badge/Amazon%20S3-569A31.svg?style=for-the-badge&logo=Amazon-S3&logoColor=white)

![footer](https://capsule-render.vercel.app/api?section=footer&type=waving&color=BF92FB&height=300)
