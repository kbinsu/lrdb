# 원수손해율 Panel Dashboard 배포용

## 로컬 실행

```bash
pip install -r requirements.txt
panel serve app.py --show
```

## Render 배포

1. 이 폴더 전체를 GitHub 저장소에 업로드
2. Render 가입 및 GitHub 연결
3. New > Web Service 선택
4. 저장소 선택
5. 아래 설정 사용

Build Command

```bash
pip install -r requirements.txt
```

Start Command

```bash
panel serve app.py --address 0.0.0.0 --port $PORT --allow-websocket-origin="*"
```

6. 배포 완료 후 `https://앱이름.onrender.com` 주소로 접속

## 포함 파일

- app.py
- loss_ratio.xlsx
- year2.xlsx
- health.png
- requirements.txt
- render.yaml