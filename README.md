# Bitcoin AI Trading System

비트코인 선물 거래를 위한 AI 기반 자동 트레이딩 시스템과 실시간 모니터링 대시보드입니다.

## 주요 기능

- 🤖 AI 기반 자동 트레이딩
- 📊 실시간 대시보드 모니터링
- 📈 멀티 타임프레임 분석
- 📰 뉴스 감성 분석
- 💰 동적 레버리지 및 포지션 사이징
- 🛡️ 자동 리스크 관리
- 📝 거래 기록 및 성과 분석

## 설치 방법

1. Python 가상환경 생성 및 활성화:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

3. 환경 변수 설정:
`.env` 파일을 생성하고 다음 변수들을 설정합니다:
```
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
GOOGLE_API_KEY=your_google_api_key
SERP_API_KEY=your_serp_api_key
```

## 실행 방법

1. 트레이딩 봇 실행:
```bash
python erus_GM.py
```

2. 대시보드 실행:
```bash
streamlit run streamlit_app.py
```

## 시스템 요구사항

- Python 3.8 이상
- SQLite3
- 인터넷 연결
- 바이낸스 계정 및 API 키
- Google API 키 (Gemini AI 사용)
- SERP API 키 (뉴스 데이터 수집)

## 프로젝트 구조

```
├── erus_GM.py              # 메인 트레이딩 봇
├── streamlit_app.py        # 대시보드 애플리케이션
├── bitcoin_trading.db      # SQLite 데이터베이스
├── requirements.txt        # 의존성 패키지 목록
├── .env                    # 환경 변수 설정
└── README.md              # 프로젝트 문서
```

## 주요 기능 설명

### AI 트레이딩 시스템
- 멀티 타임프레임 분석 (5분, 15분, 1시간, 4시간)
- 동적 레버리지 최적화
- 자동 스탑로스/테이크프로핏 설정
- 뉴스 감성 분석 통합
- 실시간 포지션 모니터링 및 재평가

### 대시보드 기능
- 실시간 포지션 모니터링
- 거래 성과 분석
- 수익/손실 차트
- AI 분석 기록 추적
- 승률 및 리스크 메트릭스

## 주의사항

- 이 시스템은 실제 자금을 거래하므로 신중하게 사용하세요.
- 테스트넷에서 충분한 테스트 후 실제 거래에 사용하세요.
- API 키는 절대 공개하지 마세요.
- 거래 전에 리스크 관리 정책을 검토하세요.

## 라이선스

MIT License

## 기여

버그 리포트나 기능 제안은 Issues 섹션을 이용해 주세요.
Pull Request도 환영합니다.

## 연락처

문의사항이 있으시면 Issues 섹션을 이용해 주세요. 