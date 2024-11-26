FROM python:latest

# 가상 환경 설치
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 앱 코드 복사
COPY app.py /myapp/app.py
COPY requirements.txt /myapp/requirements.txt

# 가상 환경 활성화 및 의존성 설치
WORKDIR /myapp/
RUN /opt/venv/bin/pip install -r requirements.txt

# 포트 노출 및 앱 실행
EXPOSE 5000
CMD ["/opt/venv/bin/python", "app.py"]

