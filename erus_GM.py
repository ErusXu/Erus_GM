# ===== 필요한 라이브러리 임포트 =====
import ccxt  # 암호화폐 거래소 API 라이브러리
import os  # 환경 변수 및 파일 시스템 접근
import math  # 수학 연산
import time  # 시간 지연 및 타임스탬프
import pandas as pd  # 데이터 분석 및 조작
import requests  # HTTP 요청
import json  # JSON 데이터 처리
import sqlite3  # 로컬 데이터베이스
from dotenv import load_dotenv  # 환경 변수 로드
load_dotenv()  # .env 파일에서 환경 변수 로드
from openai import OpenAI  # OpenAI API 접근
from datetime import datetime, timedelta  # 날짜 및 시간 처리
import logging  # 로깅 추가

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== 설정 및 초기화 =====
class TradingState:
    def __init__(self):
        self.last_ai_re_evaluation_time = datetime.now() - timedelta(hours=1)

trading_state = TradingState()

# 바이낸스 API 설정
exchange = ccxt.binance({
    'apiKey': os.getenv('BINANCE_API_KEY'),
    'secret': os.getenv('BINANCE_SECRET_KEY'),  # 올바른 환경 변수 이름으로 수정
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True
    }
})
symbol = "BTC/USDT"  # 거래 페어 설정

# Google Gemini API 클라이언트 초기화
# Gemini API는 일반적으로 Google Cloud SDK 또는 'google-generative-ai' 라이브러리를 사용하지만,
# 'openai' 라이브러리의 호환성을 활용하여 호출할 수도 있습니다.
# 이 경우에는 Google AI Studio 또는 Vertex AI 엔드포인트를 사용합니다.
# 여기서는 Google AI Studio의 기본 엔드포인트를 가정합니다.
# API 키는 Google Cloud Project에서 생성한 Gemini API Key여야 합니다.
client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"), # .env 파일에 GOOGLE_API_KEY로 저장된 키를 사용
    base_url="https://generativelanguage.googleapis.com/v1beta/" # Gemini API의 기본 URL
)

# SERP API 설정 (뉴스 데이터 수집용)
serp_api_key = os.getenv("SERP_API_KEY")  # 서프 API 키

# SQLite 데이터베이스 설정
DB_FILE = "bitcoin_trading.db"  # 데이터베이스 파일명

# ===== 데이터베이스 관련 함수 =====
def setup_database():
    """
    데이터베이스 및 필요한 테이블 생성
    
    거래 기록과 AI 분석 결과를 저장하기 위한 테이블을 생성합니다.
    - trades: 모든 거래 정보 (진입가, 청산가, 손익 등)
    - ai_analysis: AI의 분석 결과 및 추천 사항
    - data_collection_log: 데이터 수집 로그
    - news_cache: 최신 뉴스 데이터
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 거래 기록 테이블
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,           -- 거래 시작 시간
        entry_timestamp TEXT NOT NULL,     -- 진입 시간 (추가된 컬럼)
        action TEXT NOT NULL,              -- long 또는 short
        entry_price REAL NOT NULL,         -- 진입 가격
        amount REAL NOT NULL,              -- 거래량 (BTC)
        leverage INTEGER NOT NULL,         -- 레버리지 배수
        sl_price REAL NOT NULL,            -- 스탑로스 가격
        tp_price REAL NOT NULL,            -- 테이크프로핏 가격
        sl_percentage REAL NOT NULL,       -- 스탑로스 백분율
        tp_percentage REAL NOT NULL,       -- 테이크프로핏 백분율
        position_size_percentage REAL NOT NULL,  -- 자본 대비 포지션 크기
        investment_amount REAL NOT NULL,   -- 투자 금액 (USDT)
        status TEXT DEFAULT 'OPEN',        -- 거래 상태 (OPEN/CLOSED)
        exit_price REAL,                   -- 청산 가격
        exit_timestamp TEXT,               -- 청산 시간
        profit_loss REAL,                  -- 손익 (USDT)
        profit_loss_percentage REAL,       -- 손익 백분율
        sl_order_id TEXT,                  -- 스탑로스 주문 ID
        tp_order_id TEXT                   -- 테이크프로핏 주문 ID
    )
    ''')
    
    # 기존 trades 테이블에 entry_timestamp 컬럼이 없으면 추가
    cursor.execute("PRAGMA table_info(trades)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'entry_timestamp' not in columns:
        cursor.execute('ALTER TABLE trades ADD COLUMN entry_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP')
    
    # AI 분석 결과 테이블
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ai_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,               -- 분석 시간
        current_price REAL NOT NULL,           -- 분석 시점 가격
        direction TEXT NOT NULL,               -- 방향 추천 (LONG/SHORT/NO_POSITION)
        recommended_position_size REAL NOT NULL,  -- 추천 포지션 크기
        recommended_leverage INTEGER NOT NULL,    -- 추천 레버리지
        stop_loss_percentage REAL NOT NULL,       -- 추천 스탑로스 비율
        take_profit_percentage REAL NOT NULL,     -- 추천 테이크프로핏 비율
        reasoning TEXT NOT NULL,                  -- 분석 근거 설명
        trade_id INTEGER,                         -- 연결된 거래 ID
        FOREIGN KEY (trade_id) REFERENCES trades (id)  -- 외래 키 설정
    )
    ''')
    
    # 데이터 수집 로그 테이블
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS data_collection_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,           -- 로깅 시간
        data_type TEXT NOT NULL,           -- 데이터 유형
        success BOOLEAN NOT NULL,          -- 성공 여부
        error_message TEXT                 -- 오류 메시지
    )
    ''')
    
    # 뉴스 데이터 캐시 테이블
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS news_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,           -- 뉴스 수집 시간
        news_data TEXT NOT NULL,           -- 뉴스 데이터 (JSON)
        source TEXT NOT NULL               -- 뉴스 소스
    )
    ''')
    
    conn.commit()
    conn.close()
    print("데이터베이스 설정 완료")

def log_data_collection(data_type, success, error_message=None):
    """
    데이터 수집 결과를 데이터베이스에 로깅
    
    매개변수:
        data_type (str): 수집한 데이터 유형
        success (bool): 성공 여부
        error_message (str, optional): 오류 메시지
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO data_collection_log (timestamp, data_type, success, error_message) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(), data_type, success, error_message)
    )
    
    conn.commit()
    conn.close()

def save_ai_analysis(analysis_data, trade_id=None):
    """
    AI 분석 결과를 데이터베이스에 저장
    
    매개변수:
        analysis_data (dict): AI 분석 결과 데이터
        trade_id (int, optional): 연결된 거래 ID
        
    반환값:
        int: 생성된 분석 기록의 ID
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO ai_analysis (
        timestamp, 
        current_price, 
        direction, 
        recommended_position_size, 
        recommended_leverage, 
        stop_loss_percentage, 
        take_profit_percentage, 
        reasoning,
        trade_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),  # 현재 시간
        analysis_data.get('current_price', 0),  # 현재 가격
        analysis_data.get('direction', 'NO_POSITION'),  # 추천 방향
        analysis_data.get('recommended_position_size', 0),  # 추천 포지션 크기
        analysis_data.get('recommended_leverage', 0),  # 추천 레버리지
        analysis_data.get('stop_loss_percentage', 0),  # 스탑로스 비율
        analysis_data.get('take_profit_percentage', 0),  # 테이크프로핏 비율
        analysis_data.get('reasoning', ''),  # 분석 근거
        trade_id  # 연결된 거래 ID
    ))
    
    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return analysis_id

def save_trade(trade_data):
    """
    거래 정보를 데이터베이스에 저장
    
    매개변수:
        trade_data (dict): 거래 정보 데이터
        
    반환값:
        int: 생성된 거래 기록의 ID
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO trades (
        timestamp,
        entry_timestamp,
        action,
        entry_price,
        amount,
        leverage,
        sl_price,
        tp_price,
        sl_percentage,
        tp_percentage,
        position_size_percentage,
        investment_amount
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),  # 거래 시작 시간
        datetime.now().isoformat(),  # 진입 시간
        trade_data.get('action', ''),  # 포지션 방향
        trade_data.get('entry_price', 0),  # 진입 가격
        trade_data.get('amount', 0),  # 거래량
        trade_data.get('leverage', 0),  # 레버리지
        trade_data.get('sl_price', 0),  # 스탑로스 가격
        trade_data.get('tp_price', 0),  # 테이크프로핏 가격
        trade_data.get('sl_percentage', 0),  # 스탑로스 비율
        trade_data.get('tp_percentage', 0),  # 테이크프로핏 비율
        trade_data.get('position_size_percentage', 0),  # 자본 대비 포지션 크기
        trade_data.get('investment_amount', 0)  # 투자 금액
    ))
    
    trade_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return trade_id

def update_trade_status(trade_id, status, exit_price, exit_timestamp, profit_loss, profit_loss_percentage):
    """거래 상태 업데이트"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute('''
        UPDATE trades 
        SET status = ?,
            exit_price = ?,
            exit_timestamp = ?,
            profit_loss = ?,
            profit_loss_percentage = ?
        WHERE id = ?
        ''', (status, exit_price, exit_timestamp, profit_loss, profit_loss_percentage, trade_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"거래 상태 업데이트 완료: ID {trade_id}, 상태: {status}, 손익: {profit_loss:.2f} USDT ({profit_loss_percentage:.2f}%)")
    except Exception as e:
        logger.error(f"거래 상태 업데이트 중 오류 발생 (ID {trade_id}): {e}")

def get_latest_open_trade():
    """
    가장 최근의 열린 거래 정보를 가져옵니다
    
    반환값:
        dict: 거래 정보 또는 None (열린 거래가 없는 경우)
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, action, entry_price, amount, leverage, sl_price, tp_price
    FROM trades
    WHERE status = 'OPEN'
    ORDER BY timestamp DESC  -- 가장 최근 거래 먼저
    LIMIT 1
    ''')
    
    result = cursor.fetchone()
    conn.close()
    
    # 결과가 있을 경우 사전 형태로 변환하여 반환
    if result:
        return {
            'id': result[0],
            'action': result[1],
            'entry_price': result[2],
            'amount': result[3],
            'leverage': result[4],
            'sl_price': result[5],
            'tp_price': result[6]
        }
    return None  # 열린 거래가 없음

def get_trade_summary(days=7):
    """
    지정된 일수 동안의 거래 요약 정보를 가져옵니다
    
    매개변수:
        days (int): 요약할 기간(일)
        
    반환값:
        dict: 거래 요약 정보 또는 None
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT 
        COUNT(*) as total_trades,                            -- 총 거래 수
        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,  -- 이익 거래 수
        SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,   -- 손실 거래 수
        SUM(profit_loss) as total_profit_loss,               -- 총 손익
        AVG(profit_loss_percentage) as avg_profit_loss_percentage  -- 평균 손익률
    FROM trades
    WHERE exit_timestamp IS NOT NULL  -- 청산된 거래만
    AND timestamp >= datetime('now', ?)  -- 지정된 일수 내 거래만
    ''', (f'-{days} days',))
    
    result = cursor.fetchone()
    conn.close()
    
    # 결과가 있을 경우 사전 형태로 변환하여 반환
    if result:
        return {
            'total_trades': result[0] or 0,
            'winning_trades': result[1] or 0,
            'losing_trades': result[2] or 0,
            'total_profit_loss': result[3] or 0,
            'avg_profit_loss_percentage': result[4] or 0
        }
    return None

def get_historical_trading_data(limit=10):
    """
    과거 거래 내역과 관련 AI 분석 결과를 가져옵니다
    
    매개변수:
        limit (int): 가져올 최대 거래 기록 수
        
    반환값:
        list: 거래 및 분석 데이터 사전 목록
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # 컬럼명으로 결과에 접근 가능하게 설정
    cursor = conn.cursor()
    
    # 완료된 거래 내역과 관련 AI 분석 함께 조회 (LEFT JOIN 사용)
    cursor.execute('''
    SELECT 
        t.id as trade_id,
        t.timestamp as trade_timestamp,
        t.action,
        t.entry_price,
        t.exit_price,
        t.amount,
        t.leverage,
        t.sl_price,
        t.tp_price,
        t.sl_percentage,
        t.tp_percentage,
        t.position_size_percentage,
        t.status,
        t.profit_loss,
        t.profit_loss_percentage,
        a.id as analysis_id,
        a.reasoning,
        a.direction,
        a.recommended_leverage,
        a.recommended_position_size,
        a.stop_loss_percentage,
        a.take_profit_percentage
    FROM 
        trades t
    LEFT JOIN 
        ai_analysis a ON t.id = a.trade_id
    WHERE 
        t.status = 'CLOSED'  -- 완료된 거래만
    ORDER BY 
        t.timestamp DESC  -- 최신 거래 먼저
    LIMIT ?
    ''', (limit,))
    
    results = cursor.fetchall()
    
    # 결과를 사전 목록으로 변환
    historical_data = []
    for row in results:
        historical_data.append({k: row[k] for k in row.keys()})
    
    conn.close()
    return historical_data

def get_performance_metrics():
    """
    거래 성과 메트릭스를 계산합니다
    
    이 함수는 다음을 포함한 전체 및 방향별(롱/숏) 성과 지표를 계산합니다:
    - 총 거래 수
    - 승률
    - 평균 수익률
    - 최대 이익/손실
    - 방향별 성과
    
    반환값:
        dict: 성과 메트릭스 데이터
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 전체 거래 성과 쿼리
    cursor.execute('''
    SELECT 
        COUNT(*) as total_trades,
        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
        SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
        SUM(profit_loss) as total_profit_loss,
        AVG(profit_loss_percentage) as avg_profit_loss_percentage,
        MAX(profit_loss_percentage) as max_profit_percentage,
        MIN(profit_loss_percentage) as max_loss_percentage,
        AVG(CASE WHEN profit_loss > 0 THEN profit_loss_percentage ELSE NULL END) as avg_win_percentage,
        AVG(CASE WHEN profit_loss < 0 THEN profit_loss_percentage ELSE NULL END) as avg_loss_percentage
    FROM trades
    WHERE status = 'CLOSED'
    ''')
    
    overall_metrics = cursor.fetchone()
    
    # 방향별(롱/숏) 성과 쿼리
    cursor.execute('''
    SELECT 
        action,
        COUNT(*) as total_trades,
        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
        SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_trades,
        SUM(profit_loss) as total_profit_loss,
        AVG(profit_loss_percentage) as avg_profit_loss_percentage
    FROM trades
    WHERE status = 'CLOSED'
    GROUP BY action
    ''')
    
    directional_metrics = cursor.fetchall()
    
    conn.close()
    
    # 결과 구성
    metrics = {
        "overall": {
            "total_trades": overall_metrics[0] or 0,
            "winning_trades": overall_metrics[1] or 0,
            "losing_trades": overall_metrics[2] or 0,
            "total_profit_loss": overall_metrics[3] or 0,
            "avg_profit_loss_percentage": overall_metrics[4] or 0,
            "max_profit_percentage": overall_metrics[5] or 0,
            "max_loss_percentage": overall_metrics[6] or 0,
            "avg_win_percentage": overall_metrics[7] or 0,
            "avg_loss_percentage": overall_metrics[8] or 0
        },
        "directional": {}
    }
    
    # 승률 계산
    if metrics["overall"]["total_trades"] > 0:
        metrics["overall"]["win_rate"] = (metrics["overall"]["winning_trades"] / metrics["overall"]["total_trades"]) * 100
    else:
        metrics["overall"]["win_rate"] = 0
    
    # 방향별 메트릭스 추가
    for row in directional_metrics:
        action = row[0]  # 'long' 또는 'short'
        total = row[1] or 0
        winning = row[2] or 0
        
        direction_metrics = {
            "total_trades": total,
            "winning_trades": winning,
            "losing_trades": row[3] or 0,
            "total_profit_loss": row[4] or 0,
            "avg_profit_loss_percentage": row[5] or 0,
            "win_rate": (winning / total * 100) if total > 0 else 0
        }
        
        metrics["directional"][action] = direction_metrics
    
    return metrics

# ===== 데이터 수집 함수 =====
def fetch_multi_timeframe_data():
    """
    여러 타임프레임의 가격 데이터를 수집합니다
    
    각 타임프레임(5분, 15분, 1시간, 4시간)에 대해 다음 데이터를 가져옵니다:
    - 날짜/시간
    - 시가
    - 고가
    - 저가
    - 종가
    - 거래량
    
    반환값:
        dict: 타임프레임별 DataFrame 데이터
    """
    # 타임프레임별 데이터 수집 설정
    timeframes = {
        "5m": {"timeframe": "5m", "limit": 300},   # 1500분
        "15m": {"timeframe": "15m", "limit": 96},  # 24시간 (15분 * 96)
        "1h": {"timeframe": "1h", "limit": 48},    # 48시간 (1시간 * 48)
        "4h": {"timeframe": "4h", "limit": 30}     # 5일 (4시간 * 30)
    }
    
    multi_tf_data = {}
    
    # 각 타임프레임별로 데이터 수집
    for tf_name, tf_params in timeframes.items():
        try:
            # OHLCV 데이터 가져오기 (시가, 고가, 저가, 종가, 거래량)
            ohlcv = exchange.fetch_ohlcv(
                symbol, 
                timeframe=tf_params["timeframe"], 
                limit=tf_params["limit"]
            )
            
            # 데이터프레임으로 변환
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 타임스탬프를 날짜/시간 형식으로 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            # JSON 직렬화를 위해 datetime 객체를 ISO 8601 문자열로 변환
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')
            
            # 결과 딕셔너리에 저장
            multi_tf_data[tf_name] = df
            print(f"Collected {tf_name} data: {len(df)} candles")
        except Exception as e:
            print(f"Error fetching {tf_name} data: {e}")
    
    return multi_tf_data

def fetch_order_book(limit=50):
    """
    호가창 데이터 수집
    
    매개변수:
        limit (int): 가져올 호가 레벨 수
        
    반환값:
        dict: 호가창 데이터
    """
    try:
        order_book = exchange.fetch_order_book(symbol, limit=limit)
        
        # 데이터프레임으로 변환
        bids_df = pd.DataFrame(order_book['bids'], columns=['price', 'amount'])
        asks_df = pd.DataFrame(order_book['asks'], columns=['price', 'amount'])
        
        # 총 매수/매도 금액 계산
        bids_df['total'] = bids_df['price'] * bids_df['amount']
        asks_df['total'] = asks_df['price'] * asks_df['amount']
        
        result = {
            'timestamp': order_book['timestamp'],
            'datetime': order_book['datetime'],
            'bids': bids_df.to_dict('records'),
            'asks': asks_df.to_dict('records'),
            'bid_total': bids_df['total'].sum(),
            'ask_total': asks_df['total'].sum(),
            'bid_ask_ratio': bids_df['total'].sum() / asks_df['total'].sum() if asks_df['total'].sum() > 0 else 0
        }
        
        log_data_collection("order_book", True)
        print(f"Successfully collected order book data: {limit} levels")
        return result
    except Exception as e:
        error_message = str(e)
        log_data_collection("order_book", False, error_message)
        print(f"Error fetching order book data: {error_message}")
        return None

def fetch_large_trades(min_btc_amount=10):
    """
    대량 체결 데이터 수집
    
    매개변수:
        min_btc_amount (float): 최소 BTC 거래량
        
    반환값:
        list: 대량 체결 데이터
    """
    try:
        # 최근 거래 내역 가져오기
        trades = exchange.fetch_trades(symbol, limit=1000)
        
        # 대량 거래 필터링 (10 BTC 이상)
        large_trades = []
        current_price = exchange.fetch_ticker(symbol)['last']
        
        for trade in trades:
            # USDT 금액을 BTC로 변환
            btc_amount = trade['amount']
            
            if btc_amount >= min_btc_amount:
                large_trades.append({
                    'timestamp': trade['timestamp'],
                    'datetime': trade['datetime'],
                    'side': trade['side'],
                    'price': trade['price'],
                    'amount': btc_amount,
                    'cost': trade['cost']
                })
        
        log_data_collection("large_trades", True)
        print(f"Successfully collected large trades: {len(large_trades)} trades >= {min_btc_amount} BTC")
        return large_trades
    except Exception as e:
        error_message = str(e)
        log_data_collection("large_trades", False, error_message)
        print(f"Error fetching large trades data: {error_message}")
        return None

def fetch_bitcoin_news():   
    """
    비트코인 관련 최신 뉴스를 가져옵니다
    
    SERP API를 사용해 Google 뉴스에서 비트코인 관련 최신 뉴스 10개를 가져옵니다.
    하루 최대 3회만 API를 호출하고, 연속 호출 사이에 최소 8시간 간격을 유지합니다.
    이 조건을 충족하지 못하는 경우 캐시된 결과를 반환합니다.
    
    반환값:
        list: 최신 뉴스 기사 제목 목록
    """
    
    # 현재 날짜와 시간
    now = datetime.now()
    today = now.strftime('%Y-%m-%d')
    
    # 데이터베이스 연결
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # API 사용 추적 및 뉴스 캐싱 테이블 생성
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS api_usage (
        api_name TEXT PRIMARY KEY,
        daily_count INTEGER,
        last_update_date TEXT,
        last_call_timestamp TEXT
    )
    ''')
    conn.commit()
    
    # SERP API 사용 횟수 및 마지막 호출 시간 확인
    cursor.execute("SELECT daily_count, last_update_date, last_call_timestamp FROM api_usage WHERE api_name = 'serp'")
    usage_data = cursor.fetchone()
    
    # 새로운 날짜인 경우 사용량 초기화
    if not usage_data or usage_data[1] != today:
        cursor.execute(
            "INSERT OR REPLACE INTO api_usage (api_name, daily_count, last_update_date, last_call_timestamp) VALUES (?, ?, ?, ?)",
            ('serp', 0, today, None)
        )
        conn.commit()
        daily_count = 0
        last_call_timestamp = None
    else:
        daily_count = usage_data[0]
        last_call_timestamp = usage_data[2]
    
    # 마지막 API 호출 이후 경과 시간 계산
    hours_since_last_call = float('inf')
    if last_call_timestamp:
        last_call_time = datetime.fromisoformat(last_call_timestamp)
        time_diff = now - last_call_time
        hours_since_last_call = time_diff.total_seconds() / 3600
    
    # API 호출 조건 확인
    can_call_api = daily_count < 3 and hours_since_last_call >= 8
    
    # API 호출 조건을 충족하지 못하면 캐시된 뉴스 사용
    if not can_call_api:
        if daily_count >= 3:
            print(f"SERP API 일일 사용 한도({daily_count}/3)에 도달했습니다. 캐시된 뉴스를 사용합니다.")
        elif hours_since_last_call < 8:
            print(f"마지막 API 호출 후 {hours_since_last_call:.1f}시간 경과했습니다. 최소 8시간을 기다려야 합니다. 캐시된 뉴스를 사용합니다.")
        
        # 가장 최근 캐시된 뉴스 가져오기
        cursor.execute(
            "SELECT news_data, timestamp FROM news_cache ORDER BY timestamp DESC LIMIT 1"
        )
        cache_result = cursor.fetchone()
        conn.close()
        
        if cache_result:
            cached_news = json.loads(cache_result[0])
            cache_time = datetime.fromisoformat(cache_result[1])
            time_diff = now - cache_time
            hours_ago = time_diff.total_seconds() / 3600
            
            print(f"캐시된 뉴스 사용 (약 {int(hours_ago)}시간 전 데이터)")
            return [news['title'] for news in cached_news]  # 제목만 반환
        else:
            print("캐시된 뉴스가 없습니다.")
            return []
    
    # API 호출 조건을 충족하면 실제 API 호출 수행
    try:
        # SERP API 요청 설정
        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google_news",
            "q": "bitcoin",
            "gl": "us",
            "hl": "en",
            "api_key": serp_api_key
        }
        
        # API 요청 보내기
        response = requests.get(url, params=params)
        
        # 응답 확인 및 처리
        if response.status_code == 200:
            data = response.json()
            news_results = data.get("news_results", [])
            
            # 최신 뉴스 10개만 추출
            recent_news = []
            for i, news in enumerate(news_results[:10]):
                news_item = {
                    "title": news.get("title", ""),
                    "date": news.get("date", "")
                }
                recent_news.append(news_item)
            
            # API 사용 횟수 및 마지막 호출 시간 업데이트
            cursor.execute(
                "UPDATE api_usage SET daily_count = daily_count + 1, last_call_timestamp = ? WHERE api_name = 'serp'",
                (now.isoformat(),)
            )
            
            # 결과를 캐시에 저장
            cursor.execute(
                "INSERT INTO news_cache (timestamp, news_data, source) VALUES (?, ?, ?)",
                (now.isoformat(), json.dumps(recent_news), "serp")
            )
            
            # 오래된 캐시 항목 정리 (최근 10개만 유지)
            cursor.execute(
                "DELETE FROM news_cache WHERE id NOT IN (SELECT id FROM news_cache ORDER BY timestamp DESC LIMIT 10)"
            )
            
            conn.commit()
            conn.close()
            
            # 다음 가능한 API 호출 시간 계산
            next_call_time = now + timedelta(hours=8)
            print(f"Collected {len(recent_news)} recent news articles (SERP API 호출: {daily_count + 1}/3)")
            print(f"다음 API 호출 가능 시간: {next_call_time.strftime('%Y-%m-%d %H:%M:%S')}")
            return [news['title'] for news in recent_news]  # 제목만 반환
        else:
            print(f"Error fetching news: Status code {response.status_code}")
            conn.close()
            return []
    except Exception as e:
        print(f"Error fetching news: {e}")
        conn.close()
        return []

# ===== 포지션 관리 함수 =====
def handle_position_closure(current_price, side, amount, current_trade_id=None):
    """
    포지션 종료 시 데이터베이스를 업데이트하고 결과를 표시합니다
    
    매개변수:
        current_price (float): 현재 가격(청산 가격)
        side (str): 포지션 방향 ('long' 또는 'short')
        amount (float): 포지션 수량
        current_trade_id (int, optional): 현재 거래 ID
    """
    try:
        # 실제 포지션 확인
        positions = exchange.fetch_positions([symbol])
        actual_side = None
        actual_amount = 0
        
        for position in positions:
            if position['symbol'] == 'BTC/USDT:USDT':
                amt = float(position['info']['positionAmt'])
                if abs(amt) > 0.0001:  # 매우 작은 수량은 무시
                    if amt > 0:
                        actual_side = 'long'
                        actual_amount = amt
                    elif amt < 0:
                        actual_side = 'short'
                        actual_amount = abs(amt)
                    break
        
        # 실제 포지션과 입력된 포지션이 일치하는지 확인
        if actual_side != side or abs(actual_amount - amount) > 0.0001:
            print(f"경고: 실제 포지션({actual_side}, {actual_amount})과 입력된 포지션({side}, {amount})이 일치하지 않습니다.")
            return False
        
        # 거래 ID가 제공되지 않은 경우 최신 열린 거래 정보 조회
        if current_trade_id is None:
            latest_trade = get_latest_open_trade()
            if latest_trade:
                current_trade_id = latest_trade['id']
        
        if current_trade_id:
            # 가장 최근의 열린 거래 가져오기
            latest_trade = get_latest_open_trade()
            if latest_trade:
                entry_price = latest_trade['entry_price']
                action = latest_trade['action']
                
                # 미체결 주문 취소
                open_orders = exchange.fetch_open_orders(symbol)
                for order in open_orders:
                    exchange.cancel_order(order['id'], symbol)
                
                # 포지션 청산
                if actual_side == 'long':
                    close_order = exchange.create_market_sell_order(symbol, actual_amount)
                else:
                    close_order = exchange.create_market_buy_order(symbol, actual_amount)
                
                # 실제 거래 내역 조회 (최근 1분 이내)
                since = int((datetime.now() - timedelta(minutes=1)).timestamp() * 1000)
                trades = exchange.fetch_my_trades(symbol, since=since)
                
                # 해당 거래와 일치하는 체결 내역 찾기
                actual_trade = None
                for trade in trades:
                    if (trade['side'] == 'sell' and actual_side == 'long') or (trade['side'] == 'buy' and actual_side == 'short'):
                        if abs(float(trade['amount']) - actual_amount) < 0.001:  # 수량이 거의 같은 경우
                            actual_trade = trade
                            break
                
                if actual_trade:
                    # 실제 체결 가격과 수수료 정보 사용
                    actual_exit_price = float(actual_trade['price'])
                    fee = float(actual_trade['fee']['cost'])
                    
                    # 손익 계산 (실제 체결 가격과 수수료 포함)
                    if actual_side == 'long':
                        profit_loss = (actual_exit_price - entry_price) * actual_amount - fee
                        profit_loss_percentage = ((actual_exit_price - entry_price) / entry_price) * 100
                    else:  # 'short'
                        profit_loss = (entry_price - actual_exit_price) * actual_amount - fee
                        profit_loss_percentage = ((entry_price - actual_exit_price) / entry_price) * 100
                    
                    # 거래 상태 업데이트
                    update_trade_status(
                        current_trade_id,
                        'CLOSED',
                        exit_price=actual_exit_price,
                        exit_timestamp=actual_trade['datetime'],
                        profit_loss=profit_loss,
                        profit_loss_percentage=profit_loss_percentage
                    )
                    
                    # 결과 출력
                    print(f"\n=== Position Closed ===")
                    print(f"Entry: ${entry_price:,.2f}")
                    print(f"Exit: ${actual_exit_price:,.2f}")
                    print(f"Fee: ${fee:,.2f}")
                    print(f"P/L: ${profit_loss:,.2f} ({profit_loss_percentage:.2f}%)")
                    print("=======================")
                    return True
                else:
                    print("경고: 실제 거래 내역을 찾을 수 없습니다.")
                    return False
        return False
    except Exception as e:
        print(f"포지션 종료 중 오류 발생: {e}")
        return False

def check_and_cancel_remaining_orders(trade_info):
    """TP나 SL이 실행된 후 남은 주문을 취소합니다"""
    try:
        # 실제 포지션 확인
        positions = exchange.fetch_positions([symbol])
        has_position = False
        
        for position in positions:
            if position['symbol'] == 'BTC/USDT:USDT':
                amt = float(position['info']['positionAmt'])
                if abs(amt) > 0.0001:  # 매우 작은 수량은 무시
                    has_position = True
                    break
        
        # 포지션이 없는 경우에만 남은 주문 취소
        if not has_position:
            # 현재 미체결 주문 확인
            open_orders = exchange.fetch_open_orders(symbol)
            
            # SL 주문이 체결되었는지 확인
            if trade_info.get('sl_order_id'):  # get() 메서드를 사용하여 키가 없어도 안전하게 처리
                sl_order = next((order for order in open_orders if order['id'] == trade_info['sl_order_id']), None)
                if not sl_order:  # SL 주문이 체결되었으면 TP 주문 취소
                    if trade_info.get('tp_order_id'):  # TP 주문 ID가 있는 경우에만 취소 시도
                        try:
                            exchange.cancel_order(trade_info['tp_order_id'], symbol)
                            print("TP 주문이 취소되었습니다 (SL 체결)")
                        except Exception as e:
                            print(f"TP 주문 취소 중 오류 발생: {e}")
            
            # TP 주문이 체결되었는지 확인
            if trade_info.get('tp_order_id'):  # get() 메서드를 사용하여 키가 없어도 안전하게 처리
                tp_order = next((order for order in open_orders if order['id'] == trade_info['tp_order_id']), None)
                if not tp_order:  # TP 주문이 체결되었으면 SL 주문 취소
                    if trade_info.get('sl_order_id'):  # SL 주문 ID가 있는 경우에만 취소 시도
                        try:
                            exchange.cancel_order(trade_info['sl_order_id'], symbol)
                            print("SL 주문이 취소되었습니다 (TP 체결)")
                        except Exception as e:
                            print(f"SL 주문 취소 중 오류 발생: {e}")
    except Exception as e:
        print(f"남은 주문 취소 중 오류 발생: {str(e)}")

def check_manual_closures():
    """수동 청산된 거래 확인 및 업데이트"""
    try:
        # 바이낸스 API 연결 (exchange 객체는 전역 또는 매개변수로 받는 것이 효율적입니다.)
        # 여기서는 함수 내에서 다시 초기화하지만, 실제 운영 시에는 메인 루프의 exchange 객체를 재활용하는 것이 좋습니다.
        exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
        SELECT id, entry_price, amount, action, leverage, sl_order_id, tp_order_id
        FROM trades 
        WHERE status = 'OPEN'
        ''')
        open_trades = cursor.fetchall()
        
        for trade in open_trades:
            trade_id, entry_price, amount, action, leverage, sl_order_id, tp_order_id = trade
            
            try:
                # 바이낸스에서 주문 상태 확인 (SL 주문)
                if sl_order_id:
                    sl_order = exchange.fetch_order(sl_order_id, symbol) # symbol은 'BTC/USDT' 등 전역 변수 사용
                    if sl_order['status'] == 'closed':
                        logger.info(f"거래 ID {trade_id}: SL 주문({sl_order_id})이 체결됨. 실제 체결 내역 확인 중.")
                        # 실제 체결 내역 (My Trades)에서 정확한 청산 가격 및 수수료 가져오기
                        # SL 주문은 포지션을 청산하는 '매도' 또는 '매수' 체결 내역이 됩니다.
                        # 청산 주문 ID와 일치하는 체결 내역을 찾아야 합니다.
                        # `since` 매개변수로 주문 생성 시간부터 현재까지를 조회하면 효율적입니다.
                        
                        # Note: 'closed_at'이 아닌 'timestamp' (체결 시간)을 주로 사용합니다.
                        # ccxt는 timestamp를 밀리초 단위로 반환합니다.
                        trades_for_order = exchange.fetch_my_trades(symbol, since=sl_order['timestamp'] - 60000, limit=100)
                        
                        exit_price = None
                        fee_cost = 0
                        
                        # 해당 주문에 대한 체결 내역 필터링
                        relevant_trades = [
                            t for t in trades_for_order 
                            if t.get('order') == sl_order_id # 주문 ID가 일치하고
                            and t.get('side') != ('buy' if action == 'long' else 'sell') # 포지션 진입 방향과 반대 (청산)
                        ]
                        
                        # 체결 내역이 여러 개일 경우, 평균 가격 및 총 수수료 계산
                        if relevant_trades:
                            total_executed_amount = sum(float(t['amount']) for t in relevant_trades)
                            total_cost_or_revenue = sum(float(t['price']) * float(t['amount']) for t in relevant_trades)
                            fee_cost = sum(float(t['fee']['cost']) for t in relevant_trades)
                            if total_executed_amount > 0:
                                exit_price = total_cost_or_revenue / total_executed_amount
                            exit_timestamp = datetime.fromtimestamp(max(t['timestamp'] for t in relevant_trades) / 1000).isoformat()
                        else:
                            logger.warning(f"거래 ID {trade_id}: SL 주문({sl_order_id})은 closed 상태이나, 관련된 체결 내역을 찾을 수 없습니다. 현재 가격으로 손실 계산.")
                            exit_price = current_price # Fallback to current price if no trade record
                            exit_timestamp = datetime.now().isoformat()
                            
                        if exit_price:
                            profit_loss = calculate_profit_loss(entry_price, exit_price, amount, action, leverage) - fee_cost
                            # 손익률 계산 시, `entry_price * amount`는 '계약 금액'입니다. 실제 투자 원금은 레버리지를 고려해야 합니다.
                            # 레버리지 포지션의 손익률은 (최종자산-초기자산)/초기자산 * 100% 로 계산해야 정확합니다.
                            # 또는 (수익금 / 진입시 투자 원금) * 100
                            # 투자 원금 = (진입가격 * 수량) / 레버리지
                            
                            # 기존 코드의 `profit_loss_percentage = (profit_loss / (entry_price * amount)) * 100`는
                            # 레버리지를 고려하지 않은 총 계약 금액 대비 손익률이므로,
                            # 투자 원금 대비 손익률을 보려면 `(entry_price * amount / leverage)`를 사용해야 합니다.
                            initial_capital_at_risk = (entry_price * amount) / leverage
                            if initial_capital_at_risk > 0:
                                profit_loss_percentage = (profit_loss / initial_capital_at_risk) * 100
                            else:
                                profit_loss_percentage = 0
                            
                            update_trade_status(
                                trade_id, 'CLOSED_BY_SL', exit_price, exit_timestamp,
                                profit_loss, profit_loss_percentage
                            )
                            # SL이 체결되었으니, 혹시 남아있을 TP 주문은 취소
                            # ccxt.cancel_order는 orderId, symbol 필요
                            if tp_order_id:
                                try:
                                    exchange.cancel_order(tp_order_id, symbol)
                                    logger.info(f"거래 ID {trade_id}: TP 주문({tp_order_id}) 취소 완료.")
                                except Exception as cancel_e:
                                    logger.warning(f"거래 ID {trade_id}: TP 주문({tp_order_id}) 취소 실패: {cancel_e}")
                            logger.info(f"거래 ID {trade_id} (SL 체결) DB 업데이트 완료.")
                        continue # 다음 거래로 이동
                
                # TP 주문 확인 로직 (SL 주문 확인 로직과 거의 동일)
                if tp_order_id:
                    tp_order = exchange.fetch_order(tp_order_id, symbol)
                    if tp_order['status'] == 'closed':
                        logger.info(f"거래 ID {trade_id}: TP 주문({tp_order_id})이 체결됨. 실제 체결 내역 확인 중.")
                        trades_for_order = exchange.fetch_my_trades(symbol, since=tp_order['timestamp'] - 60000, limit=100)
                        
                        exit_price = None
                        fee_cost = 0
                        relevant_trades = [
                            t for t in trades_for_order 
                            if t.get('order') == tp_order_id
                            and t.get('side') != ('buy' if action == 'long' else 'sell')
                        ]
                        
                        if relevant_trades:
                            total_executed_amount = sum(float(t['amount']) for t in relevant_trades)
                            total_cost_or_revenue = sum(float(t['price']) * float(t['amount']) for t in relevant_trades)
                            fee_cost = sum(float(t['fee']['cost']) for t in relevant_trades)
                            if total_executed_amount > 0:
                                exit_price = total_cost_or_revenue / total_executed_amount
                            exit_timestamp = datetime.fromtimestamp(max(t['timestamp'] for t in relevant_trades) / 1000).isoformat()
                        else:
                            logger.warning(f"거래 ID {trade_id}: TP 주문({tp_order_id})은 closed 상태이나, 관련된 체결 내역을 찾을 수 없습니다. 현재 가격으로 이익 계산.")
                            exit_price = current_price
                            exit_timestamp = datetime.now().isoformat()

                        if exit_price:
                            profit_loss = calculate_profit_loss(entry_price, exit_price, amount, action, leverage) - fee_cost
                            initial_capital_at_risk = (entry_price * amount) / leverage
                            if initial_capital_at_risk > 0:
                                profit_loss_percentage = (profit_loss / initial_capital_at_risk) * 100
                            else:
                                profit_loss_percentage = 0

                            update_trade_status(
                                trade_id, 'CLOSED_BY_TP', exit_price, exit_timestamp,
                                profit_loss, profit_loss_percentage
                            )
                            if sl_order_id:
                                try:
                                    exchange.cancel_order(sl_order_id, symbol)
                                    logger.info(f"거래 ID {trade_id}: SL 주문({sl_order_id}) 취소 완료.")
                                except Exception as cancel_e:
                                    logger.warning(f"거래 ID {trade_id}: SL 주문({sl_order_id}) 취소 실패: {cancel_e}")
                            logger.info(f"거래 ID {trade_id} (TP 체결) DB 업데이트 완료.")
                        continue # 다음 거래로 이동
                
                # SL/TP가 아닌 수동 청산 또는 기타 이유로 포지션이 사라진 경우
                positions = exchange.fetch_positions([symbol])
                current_position_amt = 0 # 실제 포지션량
                for p in positions:
                    if p['symbol'] == symbol: # BTC/USDT:USDT 대신 전역 변수 symbol 사용
                        current_position_amt = float(p['info']['positionAmt'])
                        break
                
                # DB에는 OPEN인데 거래소에 포지션이 없는 경우
                if abs(current_position_amt) < 0.0001: # 포지션이 거의 0인 경우 (ccxt는 약간의 오차를 가질 수 있음)
                    logger.info(f"거래 ID {trade_id}: DB는 OPEN인데 거래소 포지션이 없습니다. 수동 청산 확인.")
                    # 가장 최근의 체결 내역을 검색하여 청산 가격 및 시간 추정
                    # 이때, 'amount'가 해당 거래의 'amount'와 유사한지 확인하는 것이 좋습니다.
                    # since는 최근 24시간 등 적절한 범위로 설정
                    recent_my_trades = exchange.fetch_my_trades(symbol, since=exchange.parse8601((datetime.now() - timedelta(hours=24)).isoformat())) # 최근 24시간 내역
                    
                    exit_trade_found = False
                    for my_trade in recent_my_trades:
                        # 포지션 방향과 반대되는 거래(청산 거래)인지, 그리고 수량도 유사한지 확인
                        # (예: 롱 포지션이었다면 'sell' 거래, 숏 포지션이었다면 'buy' 거래)
                        if (action == 'long' and my_trade['side'] == 'sell' and abs(float(my_trade['amount']) - amount) < amount * 0.1) or \
                           (action == 'short' and my_trade['side'] == 'buy' and abs(float(my_trade['amount']) - amount) < amount * 0.1):
                            
                            exit_price = float(my_trade['price'])
                            exit_timestamp = my_trade['datetime'] # ISO 8601 형식
                            fee_cost = float(my_trade['fee']['cost'])
                            
                            profit_loss = calculate_profit_loss(entry_price, exit_price, amount, action, leverage) - fee_cost
                            initial_capital_at_risk = (entry_price * amount) / leverage
                            if initial_capital_at_risk > 0:
                                profit_loss_percentage = (profit_loss / initial_capital_at_risk) * 100
                            else:
                                profit_loss_percentage = 0

                            update_trade_status(
                                trade_id, 'CLOSED_MANUALLY', exit_price, exit_timestamp,
                                profit_loss, profit_loss_percentage
                            )
                            logger.info(f"거래 ID {trade_id}: 수동 청산 감지 및 DB 업데이트 완료. 청산가: {exit_price}")
                            # 수동 청산 시 SL/TP 주문이 남아있을 수 있으므로 모두 취소
                            if sl_order_id:
                                try:
                                    exchange.cancel_order(sl_order_id, symbol)
                                except Exception: pass
                            if tp_order_id:
                                try:
                                    exchange.cancel_order(tp_order_id, symbol)
                                except Exception: pass
                            exit_trade_found = True
                            break # 관련된 청산 거래를 찾았으므로 루프 종료
                    
                    if not exit_trade_found:
                        logger.warning(f"거래 ID {trade_id}: DB에 OPEN이나 거래소 포지션 없음. 그러나 최근 체결 내역에서 청산 거래를 찾을 수 없습니다. (데이터 불일치 가능성)")
                        # 이 경우 수동으로 DB를 CLOSED로 변경하거나, 경고 로그를 남기고 다음 번에 다시 확인하도록 할 수 있습니다.
                        # 극단적인 경우 강제로 CLOSED 처리할 수도 있지만, 신중해야 합니다.
                        # 예: update_trade_status(trade_id, 'CLOSED_UNKNOWN', current_price, datetime.now().isoformat(), 0, 0)

            except ccxt.NetworkError as e:
                logger.error(f"거래소 네트워크 오류 (거래 ID {trade_id}): {e}. 재시도 필요.")
                # 네트워크 오류는 보통 일시적이므로, 다음 루프에서 다시 시도하도록 continue
                continue
            except ccxt.ExchangeError as e:
                logger.error(f"거래소 오류 (거래 ID {trade_id}): {e}. API 키/권한 확인 또는 주문 ID 오류 가능성.")
                # 주문 ID가 유효하지 않거나, 이미 취소/체결된 주문일 수 있음.
                # 필요시 해당 주문을 DB에서 제거하거나 상태를 변경하는 로직 추가.
                continue
            except Exception as e:
                logger.error(f"거래 {trade_id} 확인 중 일반 오류 발생: {e}", exc_info=True) # exc_info=True로 전체 스택 트레이스 출력

        conn.close()
        
    except Exception as e:
        logger.error(f"수동 청산 확인 메인 루프 중 오류 발생: {e}", exc_info=True)

def calculate_profit_loss(entry_price, exit_price, amount, action): # leverage 매개변수 제거
    """
    거래의 순수 가격 변동에 따른 손익을 계산합니다.
    수수료는 이 함수 밖에서 별도로 차감되어야 합니다.
    """
    if action == 'long':
        return (exit_price - entry_price) * amount
    else:  # short
        return (entry_price - exit_price) * amount

# profit_loss_percentage 계산 시 leverage 사용
# profit_loss_percentage = (profit_loss / ((entry_price * amount) / leverage)) * 100

def get_current_trade():
    """현재 오픈 포지션 조회"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, action, entry_price, amount, leverage, sl_price, tp_price, entry_timestamp
        FROM trades
        WHERE status = 'OPEN'
        ORDER BY timestamp DESC
        LIMIT 1
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'action': result[1],
                'entry_price': result[2],
                'amount': result[3],
                'leverage': result[4],
                'sl_price': result[5],
                'tp_price': result[6],
                'entry_timestamp': result[7]
            }
        return None
    except Exception as e:
        logger.error(f"현재 거래 조회 중 오류 발생: {e}")
        return None

def close_position_by_bot(exchange, trade_id, current_side, amount, current_price, reason="BOT_FORCED_CLOSE"):
    """
    봇이 명시적으로 포지션을 청산하는 함수.
    시장가 주문으로 포지션을 닫고, DB를 업데이트합니다.
    """
    global symbol # 전역 변수 symbol 사용
    try:
        order_side = 'sell' if current_side == 'long' else 'buy'
        logger.info(f"봇이 포지션을 강제 청산합니다: {reason}. 사이드: {order_side}, 수량: {amount}")

        # 시장가 청산 주문 실행
        close_order = exchange.create_market_order(symbol, order_side, amount)
        logger.info(f"청산 주문 ID: {close_order['id']}, 상태: {close_order['status']}")

        # 주문 체결 확인 및 최종 청산 가격 / 수수료 확인 (중요!)
        # Binance는 create_market_order 후 order 객체에 filled, cost 등을 바로 주지 않을 수 있으므로,
        # fetch_order 또는 fetch_my_trades를 사용하여 확인이 필요합니다.
        
        executed_price = current_price # 임시로 현재가 사용
        executed_amount = amount
        fee_cost = 0.05

        # 실제 체결 내역 조회
        time.sleep(2) # 주문 체결 대기
        # 특정 orderId에 해당하는 체결 내역만 가져오기 위해 params를 사용 (거래소마다 다를 수 있음)
        # Binance의 경우 'orderId' 파라미터가 동작합니다.
        recent_trades = exchange.fetch_my_trades(symbol, limit=10, params={'orderId': close_order['id']}) 
        
        if recent_trades:
            total_executed_amount = sum(t['amount'] for t in recent_trades)
            total_cost_or_revenue = sum(t['price'] * t['amount'] for t in recent_trades)
            fee_cost = sum(t['fee']['cost'] for t in recent_trades)
            if total_executed_amount > 0:
                executed_price = total_cost_or_revenue / total_executed_amount
            executed_amount = total_executed_amount
            logger.info(f"청산 주문({close_order['id']}) 체결 내역 발견. 체결가: {executed_price}, 수수료: {fee_cost}")
        else:
            logger.warning(f"청산 주문({close_order['id']})에 대한 체결 내역을 찾을 수 없습니다. 현재 가격({current_price})으로 처리.")
            
        # DB 업데이트
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT entry_price, leverage FROM trades WHERE id = ?", (trade_id,))
        trade_info = cursor.fetchone()
        entry_price = trade_info[0]
        leverage = trade_info[1]

        profit_loss = calculate_profit_loss(entry_price, executed_price, executed_amount, current_side) - fee_cost
        initial_capital_at_risk = (entry_price * executed_amount) / leverage
        profit_loss_percentage = (profit_loss / initial_capital_at_risk) * 100 if initial_capital_at_risk > 0 else 0

        update_trade_status(trade_id, reason, executed_price, datetime.now().isoformat(), profit_loss, profit_loss_percentage)
        logger.info(f"거래 ID {trade_id} 봇에 의해 청산됨. 상태: {reason}, 종료가: {executed_price:.2f}, 손익: {profit_loss:.2f} ({profit_loss_percentage:.2f}%)")

        # 남아있는 SL/TP 주문 취소 (확실하게)
        cursor.execute("SELECT sl_order_id, tp_order_id FROM trades WHERE id = ?", (trade_id,))
        order_ids = cursor.fetchone()
        if order_ids:
            sl_id, tp_id = order_ids
            if sl_id:
                try:
                    exchange.cancel_order(sl_id, symbol)
                    logger.info(f"청산 후 SL 주문({sl_id}) 취소 완료.")
                except ccxt.OrderNotFound: pass # 이미 취소되었거나 없는 주문
                except Exception as e: logger.warning(f"청산 후 SL 취소 실패: {e}")
            if tp_id:
                try:
                    exchange.cancel_order(tp_id, symbol)
                    logger.info(f"청산 후 TP 주문({tp_id}) 취소 완료.")
                except ccxt.OrderNotFound: pass # 이미 취소되었거나 없는 주문
                except Exception as e: logger.warning(f"청산 후 TP 취소 실패: {e}")
        conn.close()

    except Exception as e:
        logger.error(f"봇에 의한 포지션 청산 중 오류 발생: {e}", exc_info=True)


# ===== 메인 프로그램 시작 =====
print("\n=== Bitcoin Trading Bot Started ===")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Trading Pair:", symbol)
print("Dynamic Leverage: AI Optimized")
print("Dynamic SL/TP: AI Optimized")
print("Multi Timeframe Analysis: 5m, 15m, 1h, 4h")
print("News Sentiment Analysis: Enabled")
print("Historical Performance Learning: Enabled")
print("Database Logging: Enabled")
print("===================================\n")

# 데이터베이스 설정
setup_database()

# ===== 메인 트레이딩 루프 =====
while True:
    try:
        global last_ai_re_evaluation_time
        # 현재 시간 및 가격 조회
        current_time = datetime.now().strftime('%H:%M:%S')
        current_price = exchange.fetch_ticker(symbol)['last']
        print(f"\n[{current_time}] Current BTC Price: ${current_price:,.2f}")

        # ===== 1. 현재 포지션 확인 =====
        current_side = None  # 현재 포지션 방향 (long/short/None)
        amount = 0  # 포지션 수량

        # 바이낸스에서 현재 포지션 조회 (최대 3번 시도)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                positions = exchange.fetch_positions([symbol])
                for position in positions:
                    if position['symbol'] == 'BTC/USDT:USDT':
                        amt = float(position['info']['positionAmt'])
                        if abs(amt) > 0.0001:  # 매우 작은 수량은 무시
                            if amt > 0:
                                current_side = 'long'
                                amount = amt
                            elif amt < 0:
                                current_side = 'short'
                                amount = abs(amt)
                            break
                if current_side is not None:  # 유효한 포지션을 찾았으면 루프 종료
                    break
                time.sleep(1)  # 다음 시도 전에 잠시 대기
            except Exception as e:
                print(f"포지션 조회 시도 {attempt + 1} 실패: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    print("포지션 조회 실패. 프로그램을 종료합니다.")
                    exit(1)

        # 데이터베이스에서 현재 거래 정보 조회
        current_trade = get_current_trade()
        current_trade_id = current_trade['id'] if current_trade else None
        current_trade_in_db = current_trade  # Add this line to initialize current_trade_in_db

        # 포지션과 DB 정보가 일치하는지 확인
        if current_side and current_trade:
            if current_side != current_trade['action'].lower():
                print(f"경고: 바이낸스 포지션({current_side})과 DB 기록({current_trade['action']})이 일치하지 않습니다.")
                print("DB 정보를 업데이트합니다...")
                update_trade_status(current_trade_id, 'CLOSED', current_price, None, None, None)
                current_trade = None
                current_trade_id = None

        # ===== 2. 포지션이 있는 경우 TP/SL 주문 확인 및 취소 =====
        if current_side and current_trade:
            check_and_cancel_remaining_orders(current_trade)
            
        # ... (메인 루프 시작) ...

        # ===== 2-1. 포지션이 존재하는 경우 처리 =====
        if current_side and current_trade_in_db:
            
            # 바이낸스 포지션과 DB 기록이 일치하는지 확인 (방향)
            if current_side != current_trade_in_db['action'].lower():
                logger.warning(f"경고: 바이낸스 포지션({current_side})과 DB 기록({current_trade_in_db['action']}) 불일치. DB 기록을 닫힘으로 처리.")
                update_trade_status(current_trade_in_db['id'], 'CLOSED_MISMATCH', current_price, datetime.now().isoformat(), 0, 0)
                current_trade_in_db = None # DB 상태를 업데이트했으니 다시 None으로 설정

            else: # 포지션이 있고 DB 기록도 일치하는 경우
                print(f"현재 {current_side.upper()} 포지션 유지 중. 수량: {amount} BTC, 진입가: ${current_trade_in_db['entry_price']:,.2f}")
                print(f"SL: ${current_trade_in_db['sl_price']:,.2f}, TP: ${current_trade_in_db['tp_price']:,.2f}")
        
                # 현재 손익 계산 (AI에게 전달할 정보)
                current_profit_loss = calculate_profit_loss(
                    current_trade_in_db['entry_price'], current_price, amount, current_trade_in_db['action'].lower()
                )
                initial_capital_at_risk = (current_trade_in_db['entry_price'] * amount) / current_trade_in_db['leverage']
                current_profit_loss_percentage = (current_profit_loss / initial_capital_at_risk) * 100 if initial_capital_at_risk > 0 else 0

                # 포지션 진입 시간 (ISO 8601 문자열을 datetime 객체로 변환)
                entry_timestamp_dt = datetime.fromisoformat(current_trade_in_db['entry_timestamp'])
                time_since_entry = datetime.now() - entry_timestamp_dt
                time_since_entry_hours = time_since_entry.total_seconds() / 3600

                # 재평가 조건:
                # 1. 마지막 재평가로부터 1시간이 지났거나
                # 2. 포지션 진입 후 1시간이 지났거나
                # 3. 손익률이 ±5% 이상 변동했을 때
                should_re_evaluate = False
                
                # 마지막 재평가로부터 1시간 경과
                time_since_last_eval = (datetime.now() - trading_state.last_ai_re_evaluation_time).total_seconds() / 3600
                if time_since_last_eval >= 1:
                    should_re_evaluate = True
                    logger.info(f"마지막 재평가로부터 {time_since_last_eval:.1f}시간 경과, 재평가를 요청합니다.")
                
                # 포지션 진입 후 1시간 경과 (최초 1회만)
                elif time_since_entry_hours >= 1 and time_since_entry_hours < 1.04:  # 1시간~1시간 4분 사이에만
                    should_re_evaluate = True
                    logger.info(f"포지션 진입 후 {time_since_entry_hours:.1f}시간 경과, 최초 재평가를 요청합니다.")
                
                # 손익률 변동 확인
                else:
                    # 이전 손익률과 현재 손익률 비교
                    previous_pnl = current_trade_in_db.get('last_pnl_percentage', 0)
                    current_pnl = current_profit_loss_percentage
                    pnl_change = abs(current_pnl - previous_pnl)
                    
                    if pnl_change >= 5.0:  # 5% 이상 변동
                        should_re_evaluate = True
                        logger.info(f"손익률 {pnl_change:.1f}% 변동 감지 (이전: {previous_pnl:.1f}% → 현재: {current_pnl:.1f}%), 재평가를 요청합니다.")
                
                if not should_re_evaluate:
                    wait_time = 300  # 5분 대기
                    
                    logger.info(f"재평가 조건 미충족 - 다음 재평가까지 {wait_time/60:.0f}분 대기")
                    time.sleep(wait_time)  # 계산된 시간만큼 대기
                    continue

                logger.info(f"포지션 재평가 시작 (진입 후 {time_since_entry_hours:.1f}시간 경과)")
                
                # 시장 데이터 수집 (AI 재평가를 위해 다시 수집)
                multi_tf_data = fetch_multi_timeframe_data()
                order_book = fetch_order_book()
                large_trades = fetch_large_trades()
                recent_news = fetch_bitcoin_news()
                historical_trading_data = get_historical_trading_data(limit=10)
                performance_metrics = get_performance_metrics()
                
                # AI 분석을 위한 데이터 준비
                market_analysis = {
                    "timestamp": datetime.now().isoformat(),
                    "current_price": current_price,
                    "timeframes": {},
                    "order_book": order_book,
                    "large_trades": large_trades,
                    "recent_news": recent_news,
                    "historical_trading_data": historical_trading_data,
                    "performance_metrics": performance_metrics
                }
                for tf_name, df in multi_tf_data.items():
                    market_analysis["timeframes"][tf_name] = df.to_dict(orient="records")
                
                # AI 포지션 재평가 시스템 프롬프트
                re_evaluation_system_prompt = """
You are a highly analytical crypto trading expert with an OPEN BTC/USDT futures position.
Your primary task is to critically re-evaluate the market and advise whether to **MAINTAIN** this position or **CLOSE** it immediately.
Prioritize capital preservation and securing profits.
주어진 데이터를 천천히, 철저히 분석하여 신중한 판단을 내리세요. 

**Current Position Details:**
- Direction: {current_side_upper}
- Amount: {amount} BTC
- Entry Price: ${entry_price:,.2f}
- Current Price: ${current_price:,.2f}
- Stop Loss: ${sl_price:,.2f}
- Take Profit: ${tp_price:,.2f}
- Leverage: {leverage}x
- Current Profit/Loss: {current_profit_loss_percentage:.2f}% ({current_profit_loss:.2f} USDT)
- Time since entry: {time_since_entry_hours:.1f} hours

**Market Data:**
{market_analysis_json_string}

**Your Decision Process:**
1.  **Assess Position Performance:** Evaluate current P/L against SL/TP and time in trade.
2.  **Re-analyze Market:** Review provided market data (trends, volume, news) for any significant shifts or new risks.
3.  **Risk/Reward Reassessment:** Does the position still align with optimal risk/reward given current conditions?
4.  **Action Determination:**
    * **MAINTAIN:** If market outlook remains favorable and risk is controlled.
    * **CLOSE:** If conditions have deteriorated, risk has increased unacceptably, or a clear exit signal is present.

**IMPORTANT: Your response must be a valid JSON object with the following structure:**
{{
    "action": "MAINTAIN" or "CLOSE",
    "reasoning": "Your concise explanation for maintaining or closing the position"
}}

DO NOT include any markdown formatting, code blocks, or additional text. Return ONLY the JSON object.
"""
            
                # AI 재평가 프롬프트에 동적 정보 주입
                formatted_re_evaluation_prompt = re_evaluation_system_prompt.format(
                    current_side_upper=current_side.upper(),
                    amount=amount,
                    entry_price=current_trade_in_db['entry_price'],
                    current_price=current_price,
                    sl_price=current_trade_in_db['sl_price'],
                    tp_price=current_trade_in_db['tp_price'],
                    leverage=current_trade_in_db['leverage'],
                    current_profit_loss=current_profit_loss,
                    current_profit_loss_percentage=current_profit_loss_percentage,
                    time_since_entry_hours=time_since_entry_hours,
                    market_analysis_json_string=json.dumps(market_analysis, indent=2) # AI에게 전달할 시장 데이터 JSON
                )

                # AI에게 포지션 재평가 요청
                try:
                    re_evaluation_response = client.chat.completions.create(
                        model="gemini-2.5-flash-preview-05-20", # 또는 "gemini-1.5-pro-latest"
                        messages=[
                        {"role": "user", "content": formatted_re_evaluation_prompt}
                        ]
                    )
                
                    re_evaluation_content = re_evaluation_response.choices[0].message.content.strip()
                    logger.info(f"AI 재평가 Raw 응답: {re_evaluation_content}")

                    # JSON 형식 정리 (코드 블록 제거)
                    if re_evaluation_content.startswith("```"):
                        content_parts = re_evaluation_content.split("\n", 1)
                        if len(content_parts) > 1:
                            re_evaluation_content = content_parts[1]
                        if "```" in re_evaluation_content:
                            re_evaluation_content = re_evaluation_content.rsplit("```", 1)[0]
                        re_evaluation_content = re_evaluation_content.strip()

                    # JSON 파싱 시도
                    try:
                        re_evaluation_decision = json.loads(re_evaluation_content)
                    except json.JSONDecodeError as e:
                        # JSON 파싱 실패 시 응답 내용을 정리하고 다시 시도
                        logger.warning(f"Initial JSON parsing failed: {e}")
                        # 불필요한 공백과 줄바꿈 제거
                        re_evaluation_content = ' '.join(re_evaluation_content.split())
                        # JSON 파싱 재시도
                        re_evaluation_decision = json.loads(re_evaluation_content)

                    # 필수 키 확인
                    required_keys = ['action', 'reasoning']
                    missing_keys = [key for key in required_keys if key not in re_evaluation_decision]
                    
                    if missing_keys:
                        raise ValueError(f"Missing required keys in re-evaluation decision: {missing_keys}")

                    # action 값 검증
                    if re_evaluation_decision['action'] not in ['MAINTAIN', 'CLOSE']:
                        raise ValueError(f"Invalid action value: {re_evaluation_decision['action']}. Must be 'MAINTAIN' or 'CLOSE'")

                    logger.info(f"AI 재평가 결정: {re_evaluation_decision['action']}, 근거: {re_evaluation_decision['reasoning']}")

                    if re_evaluation_decision['action'] == "CLOSE":
                        logger.warning(f"AI가 포지션 청산을 권고합니다. 이유: {re_evaluation_decision['reasoning']}")
                        # 봇이 능동적으로 포지션 청산 함수 호출
                        close_position_by_bot(
                            exchange, 
                            current_trade_in_db['id'], 
                            current_trade_in_db['action'].lower(), 
                            amount, # 현재 바이낸스에서 조회된 포지션 양
                            current_price,
                            reason="CLOSED_BY_AI_RE_EVALUATION"
                        )
                        # 포지션이 닫혔으므로 다음 루프에서 새로운 포지션 진입을 고려하도록 함
                        current_trade_in_db = None 
                        current_side = None # 바이낸스 포지션도 닫혔다고 가정
                        time.sleep(10) # 청산 후 잠시 대기
                        continue # 즉시 다음 루프 시작 (새 포지션 탐색)
                    elif re_evaluation_decision['action'] == "MAINTAIN":
                        logger.info(f"AI가 포지션 유지를 권고합니다. 이유: {re_evaluation_decision['reasoning']}")
                        # 포지션 유지
                
                    trading_state.last_ai_re_evaluation_time = datetime.now() # 재평가 시간 업데이트

                except json.JSONDecodeError as e:
                    logger.error(f"AI 재평가 응답 JSON 파싱 오류: {e}")
                    logger.error(f"AI 재평가 Raw 응답: {re_evaluation_content}")
                    # 오류 발생 시 재평가 시간 업데이트하지 않음 (다음 루프에서 다시 시도)
                except ValueError as e:
                    logger.error(f"AI 재평가 데이터 검증 오류: {e}")
                    logger.error(f"AI 재평가 Raw 응답: {re_evaluation_content}")
                    # 오류 발생 시 재평가 시간 업데이트하지 않음
                except Exception as e:
                    logger.error(f"AI 재평가 요청 또는 처리 중 기타 오류: {e}", exc_info=True)
                    # 오류 발생 시 재평가 시간 업데이트하지 않음
            
            # 포지션이 있는 동안에는 AI 분석 및 신규 진입을 하지 않고 대기 (AI 재평가 시 제외)
            time.sleep(300) # 포지션 있을 때 5분 대기 (AI 재평가가 발생하지 않는 경우)
            continue # 다음 루프 시작
    
# ... (이하 기존 코드) ...
        
        # ===== 3. 포지션이 없는 경우 처리 =====
        else:
            # 이전에 포지션이 있었고 DB에 열린 거래가 있는 경우 (포지션 종료됨)
            if current_trade:
                handle_position_closure(current_price, current_trade['action'], current_trade['amount'], current_trade_id)
            
            # 포지션이 없을 경우, 남아있는 미체결 주문 취소
            try:
                open_orders = exchange.fetch_open_orders(symbol)
                if open_orders:
                    for order in open_orders:
                        exchange.cancel_order(order['id'], symbol)
                    print("Cancelled remaining open orders for", symbol)
                else:
                    print("No remaining open orders to cancel.")
            except Exception as e:
                print("Error cancelling orders:", e)
                
            # 잠시 대기 후 시장 분석 시작
            time.sleep(5)
            print("No position. Analyzing market...")

            # ===== 4. 시장 데이터 수집 =====
            # 멀티 타임프레임 차트 데이터 수집
            multi_tf_data = fetch_multi_timeframe_data()
            
            # 호가창 및 대량 거래 데이터 수집
            order_book = fetch_order_book()
            large_trades = fetch_large_trades()
            
            # 최신 비트코인 뉴스 수집
            recent_news = fetch_bitcoin_news()
            
            # 과거 거래 내역 및 AI 분석 결과 가져오기
            historical_trading_data = get_historical_trading_data(limit=10)  # 최근 10개 거래
            
            # 전체 거래 성과 메트릭스 계산
            performance_metrics = get_performance_metrics()
            
            # ===== 5. AI 분석을 위한 데이터 준비 =====
            market_analysis = {
                "timestamp": datetime.now().isoformat(),
                "current_price": current_price,
                "timeframes": {},
                "order_book": order_book,
                "large_trades": large_trades,
                "recent_news": recent_news,
                "historical_trading_data": historical_trading_data,
                "performance_metrics": performance_metrics
            }
            
            # 각 타임프레임 데이터를 dict로 변환하여 저장
            for tf_name, df in multi_tf_data.items():
                market_analysis["timeframes"][tf_name] = df.to_dict(orient="records")
            
            # ===== 6. AI 트레이딩 결정 요청 =====
            system_prompt = """
You are an elite crypto trading expert specializing in data-driven decisions and risk management for BTC/USDT futures.
Your core mission is to achieve a minimum 1% daily return while strictly preserving capital.
Every decision must prioritize minimizing losses and maximizing the win rate (target 2.5:1 Reward/Risk).
주어진 가정과 데이터를 천천히 그리고 철저히 분석하고 판단하여 거래 결정을 내리세요. 

**Decision Process:**

1.  **Analyze Historical Performance:**
    * Review past trades (profit/loss, win/loss ratio, average P/L %, SL/TP effectiveness).
    * Identify patterns, learn from successes/failures, and adapt strategy.
    * Compare LONG vs SHORT performance and leverage outcomes.

2.  **Evaluate Current Market Conditions:**
    * **Trend Analysis (Multi-timeframe: 4h, 1h, 15m, 5m):** Identify overall bias (4h, 1h, 15m for direction; 5m for entry timing).
    * **Volume & Liquidity:** Observe volume trends, large trades (>50 BTC), order book imbalances, and liquidity gaps.
    * **Volatility:** Assess volatility across timeframes.
    * **Key Levels:** Pinpoint crucial support/resistance levels with volume confirmation.
    * **News Sentiment:** Analyze recent news headlines for bullish/bearish sentiment.

3.  **Determine Trading Stance:**
    * **Direction:** LONG, SHORT, or NO_POSITION.
    * **Conviction:** Probability of success (range 70-99.9%). If conviction is below 80%, recommend NO_POSITION.

4.  **Calculate Position Sizing (Kelly Criterion & Risk Management):**
    * **Kelly Formula (Adjusted):** f* = (p - q/b) * 0.5 (Half-Kelly for reduced volatility).
        * `p` = Probability of success (your conviction).
        * `q` = Probability of failure (1 - p).
        * `b` = Reward/Risk ratio (e.g., if TP is 1% and SL is 0.4%, then b = 1/0.4 = 2.5).
    * **Capital Allocation:** Risk `f*` fraction of your available capital.
    * **Adjustment:** Be more conservative if recent trades resulted in losses or overall win rate is below 50%. Reduce size during abnormally low volume.

5.  **Set Leverage (1x-20x):**
    * **Dynamic Adjustment:** Based on market volatility, volume, and your conviction.
    * **High Conviction/Low Volatility/Trending:** Up to 20x.
    * **High Volatility/Uncertainty/Erratic Volume:** 1-3x.
    * **Learning:** Incorporate past leverage performance.

6.  **Define Stop Loss (SL) & Take Profit (TP) Levels:**
    * **Technical Placement:** Based on recent price action, support/resistance, and volume profiles.
    * **Volatility Adaptation:** Adjust to avoid premature stop-outs.
    * **Expression:** As percentages from the entry price (e.g., 0.005 for 0.5%).
    * **Target Ratio:** Aim for a 2.5:1 Reward/Risk ratio where possible.
    * **Learning:** Adapt based on historical SL/TP performance.

**IMPORTANT: Your response must be a valid JSON object with the following structure:**
{{
    "direction": "LONG" or "SHORT" or "NO_POSITION",
    "recommended_position_size": decimal between 0.1-1.0,
    "recommended_leverage": integer between 1-20,
    "stop_loss_percentage": decimal (e.g., 0.005 for 0.5%),
    "take_profit_percentage": decimal (e.g., 0.0125 for 1.25%),
    "reasoning": "Your concise explanation"
}}

DO NOT include any markdown formatting, code blocks, or additional text. Return ONLY the JSON object.
"""
            # AI 분석을 위한 시스템 프롬프트 설정
                 
             # Gemini API 호출하여 트레이딩 결정 요청
            response = client.chat.completions.create(
                model="gemini-2.5-flash-preview-05-20",  # 현재 사용 가능한 최신 모델
                messages=[
                    # 시스템 프롬프트를 사용자 메시지에 통합.
                    {"role": "user", "content": f"{system_prompt}\n\n{market_analysis}"}
                ]
            )
            
        
            # ===== 7. AI 응답 처리 및 거래 실행 =====
            try:
                # API 응답에서 내용 추출
                response_content = response.choices[0].message.content.strip()
                print(f"Raw AI response: {response_content}")  # 디버깅용 출력
                
                # JSON 형식 정리 (코드 블록 제거)
                if response_content.startswith("```"):
                    # 첫 번째 줄바꿈 이후부터 마지막 ``` 이전까지의 내용만 추출
                    content_parts = response_content.split("\n", 1)
                    if len(content_parts) > 1:
                        response_content = content_parts[1]
                    # 마지막 ``` 제거
                    if "```" in response_content:
                        response_content = response_content.rsplit("```", 1)[0]
                    response_content = response_content.strip()
                
                # JSON 파싱 시도
                try:
                    trading_decision = json.loads(response_content)
                except json.JSONDecodeError as e:
                    # JSON 파싱 실패 시 응답 내용을 정리하고 다시 시도
                    print(f"Initial JSON parsing failed: {e}")
                    # 불필요한 공백과 줄바꿈 제거
                    response_content = ' '.join(response_content.split())
                    # JSON 파싱 재시도
                    trading_decision = json.loads(response_content)
                
                # 필수 키 확인
                required_keys = ['direction', 'recommended_position_size', 'recommended_leverage', 
                               'stop_loss_percentage', 'take_profit_percentage', 'reasoning']
                missing_keys = [key for key in required_keys if key not in trading_decision]
                
                if missing_keys:
                    raise ValueError(f"Missing required keys in trading decision: {missing_keys}")
                
                # 결정 내용 출력
                print(f"AI 거래 결정:")
                print(f"방향: {trading_decision['direction']}")
                print(f"추천 포지션 크기: {trading_decision['recommended_position_size']*100:.1f}%")
                print(f"추천 레버리지: {trading_decision['recommended_leverage']}x")
                print(f"스탑로스 레버리지: {trading_decision['stop_loss_percentage']*100:.2f}%")
                print(f"테이크프로핏 레버리지: {trading_decision['take_profit_percentage']*100:.2f}%")
                print(f"근거: {trading_decision['reasoning']}")
                
                # AI 분석 결과를 데이터베이스에 저장
                analysis_data = {
                    'current_price': current_price,
                    'direction': trading_decision['direction'],
                    'recommended_position_size': trading_decision['recommended_position_size'],
                    'recommended_leverage': trading_decision['recommended_leverage'],
                    'stop_loss_percentage': trading_decision['stop_loss_percentage'],
                    'take_profit_percentage': trading_decision['take_profit_percentage'],
                    'reasoning': trading_decision['reasoning']
                }
                analysis_id = save_ai_analysis(analysis_data)
                
                # AI 추천 방향 가져오기
                action = trading_decision['direction'].lower()
                
                # ===== 8. 트레이딩 결정에 따른 액션 실행 =====
                # 포지션을 열지 말아야 하는 경우
                if action == "no_position":
                    print("현재 시장 상황에서는 포지션을 열지 않는 것이 좋습니다.")
                    print(f"이유: {trading_decision['reasoning']}")
                    time.sleep(900)  # 포지션 없을 때 15분 대기
                    continue
                    
                # ===== 9. 투자 금액 계산 =====
                # 현재 잔액 확인
                balance = exchange.fetch_balance()
                available_capital = balance['USDT']['free']  # 가용 USDT 잔액
                
                # AI 추천 포지션 크기 비율 적용
                position_size_percentage = trading_decision['recommended_position_size']
                investment_amount = available_capital * position_size_percentage
                
                # 바이낸스에서 최소 주문 금액 확인
                try:
                    market_info = exchange.fetch_market(symbol)
                    min_order_amount = float(market_info['limits']['amount']['min']) * current_price
                    if investment_amount < min_order_amount:
                        investment_amount = min_order_amount
                        print(f"최소 주문 금액({min_order_amount:.2f} USDT)으로 조정됨")
                except Exception as e:
                    print(f"최소 주문 금액 확인 중 오류 발생: {e}")
                    # 오류 발생 시 기본값 100 USDT 사용
                    if investment_amount < 100:
                        investment_amount = 100
                        print(f"최소 주문 금액(100 USDT)으로 조정됨")
                
                print(f"투자 금액: {investment_amount:.2f} USDT")
                
                # ===== 10. 주문 수량 계산 =====
                # BTC 수량 = 투자금액 / 현재가격, 소수점 3자리까지 반올림
                amount = math.ceil((investment_amount / current_price) * 1000) / 1000
                print(f"주문 수량: {amount} BTC")

                # ===== 11. 레버리지 설정 =====
                # AI 추천 레버리지 설정
                recommended_leverage = trading_decision['recommended_leverage']
                exchange.set_leverage(recommended_leverage, symbol)
                print(f"레버리지 설정: {recommended_leverage}x")

                # ===== 12. 스탑로스/테이크프로핏 설정 =====
                # AI 추천 SL/TP 비율 가져오기
                sl_percentage = trading_decision['stop_loss_percentage']
                tp_percentage = trading_decision['take_profit_percentage']

                # ===== 13. 포지션 진입 및 SL/TP 주문 실행 =====
                if action == "long":  # 롱 포지션
                    # 시장가 매수 주문
                    order = exchange.create_market_buy_order(symbol, amount)
                    
                    # 실제 체결 내역 조회
                    time.sleep(2)  # 주문 체결 대기
                    recent_trades = exchange.fetch_my_trades(symbol, limit=10, params={'orderId': order['id']})
                    
                    if recent_trades:
                        # 체결 내역에서 평균 체결가격 계산
                        total_executed_amount = sum(t['amount'] for t in recent_trades)
                        total_cost_or_revenue = sum(t['price'] * t['amount'] for t in recent_trades)
                        entry_price = total_cost_or_revenue / total_executed_amount if total_executed_amount > 0 else current_price
                        logger.info(f"실제 체결가격: {entry_price:,.2f} (주문 ID: {order['id']})")
                    else:
                        entry_price = current_price
                        logger.warning(f"체결 내역을 찾을 수 없어 현재가격({current_price:,.2f})을 진입가격으로 사용합니다.")
                    
                    # 스탑로스/테이크프로핏 가격 계산
                    sl_price = round(entry_price * (1 - sl_percentage), 2)   # AI 추천 비율만큼 하락
                    tp_price = round(entry_price * (1 + tp_percentage), 2)   # AI 추천 비율만큼 상승
                    
                    # SL/TP 주문 생성
                    exchange.create_order(symbol, 'STOP_MARKET', 'sell', amount, None, {'stopPrice': sl_price})
                    exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell', amount, None, {'stopPrice': tp_price})
                    
                    # 거래 데이터 저장
                    trade_data = {
                        'action': 'long',
                        'entry_price': entry_price,
                        'amount': amount,
                        'leverage': recommended_leverage,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'sl_percentage': sl_percentage,
                        'tp_percentage': tp_percentage,
                        'position_size_percentage': position_size_percentage,
                        'investment_amount': investment_amount
                    }
                    trade_id = save_trade(trade_data)
                    
                    # AI 분석 결과와 거래 연결
                    update_analysis_sql = "UPDATE ai_analysis SET trade_id = ? WHERE id = ?"
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute(update_analysis_sql, (trade_id, analysis_id))
                    conn.commit()
                    conn.close()
                    
                    # 포지션 진입 시점에 재평가 시간 초기화
                    trading_state.last_ai_re_evaluation_time = datetime.now()
                    
                    print(f"\n=== LONG Position Opened ===")
                    print(f"Entry: ${entry_price:,.2f}")
                    print(f"Stop Loss: ${sl_price:,.2f} (-{sl_percentage*100:.2f}%)")
                    print(f"Take Profit: ${tp_price:,.2f} (+{tp_percentage*100:.2f}%)")
                    print(f"Leverage: {recommended_leverage}x")
                    print(f"분석 근거: {trading_decision['reasoning']}")
                    print("===========================")

                elif action == "short":  # 숏 포지션
                    # 시장가 매도 주문
                    order = exchange.create_market_sell_order(symbol, amount)
                    
                    # 실제 체결 내역 조회
                    time.sleep(2)  # 주문 체결 대기
                    recent_trades = exchange.fetch_my_trades(symbol, limit=10, params={'orderId': order['id']})
                    
                    if recent_trades:
                        # 체결 내역에서 평균 체결가격 계산
                        total_executed_amount = sum(t['amount'] for t in recent_trades)
                        total_cost_or_revenue = sum(t['price'] * t['amount'] for t in recent_trades)
                        entry_price = total_cost_or_revenue / total_executed_amount if total_executed_amount > 0 else current_price
                        logger.info(f"실제 체결가격: {entry_price:,.2f} (주문 ID: {order['id']})")
                    else:
                        entry_price = current_price
                        logger.warning(f"체결 내역을 찾을 수 없어 현재가격({current_price:,.2f})을 진입가격으로 사용합니다.")
                    
                    # 스탑로스/테이크프로핏 가격 계산
                    sl_price = round(entry_price * (1 + sl_percentage), 2)   # AI 추천 비율만큼 상승
                    tp_price = round(entry_price * (1 - tp_percentage), 2)   # AI 추천 비율만큼 하락
                    
                    # SL/TP 주문 생성
                    exchange.create_order(symbol, 'STOP_MARKET', 'buy', amount, None, {'stopPrice': sl_price})
                    exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'buy', amount, None, {'stopPrice': tp_price})
                    
                    # 거래 데이터 저장
                    trade_data = {
                        'action': 'short',
                        'entry_price': entry_price,
                        'amount': amount,
                        'leverage': recommended_leverage,
                        'sl_price': sl_price,
                        'tp_price': tp_price,
                        'sl_percentage': sl_percentage,
                        'tp_percentage': tp_percentage,
                        'position_size_percentage': position_size_percentage,
                        'investment_amount': investment_amount
                    }
                    trade_id = save_trade(trade_data)
                    
                    # AI 분석 결과와 거래 연결
                    update_analysis_sql = "UPDATE ai_analysis SET trade_id = ? WHERE id = ?"
                    conn = sqlite3.connect(DB_FILE)
                    cursor = conn.cursor()
                    cursor.execute(update_analysis_sql, (trade_id, analysis_id))
                    conn.commit()
                    conn.close()
                    
                    # 포지션 진입 시점에 재평가 시간 초기화
                    trading_state.last_ai_re_evaluation_time = datetime.now()
                    
                    print(f"\n=== SHORT Position Opened ===")
                    print(f"Entry: ${entry_price:,.2f}")
                    print(f"Stop Loss: ${sl_price:,.2f} (+{sl_percentage*100:.2f}%)")
                    print(f"Take Profit: ${tp_price:,.2f} (-{tp_percentage*100:.2f}%)")
                    print(f"Leverage: {recommended_leverage}x")
                    print(f"분석 근거: {trading_decision['reasoning']}")
                    print("============================")
                else:
                    print("Action이 'long' 또는 'short'가 아니므로 주문을 실행하지 않습니다.")
                    
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류: {e}")
                print(f"AI 응답: {response.choices[0].message.content}")
                time.sleep(30)  # 대기 후 다시 시도
                continue
            except Exception as e:
                print(f"기타 오류: {e}")
                time.sleep(10)
                continue

        # ===== 14. 일정 시간 대기 후 다음 루프 실행 =====
        time.sleep(300)  # 메인 루프는 5분마다 실행

    except Exception as e:
        print(f"\n Error: {e}")
        time.sleep(5)

