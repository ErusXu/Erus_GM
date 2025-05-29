import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ccxt  # 암호화폐 거래소 API 라이브러리
import numpy as np
import os

# 페이지 설정
st.set_page_config(
    page_title="Bitcoin Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# 데이터베이스 파일 경로
DB_FILE = "bitcoin_trading.db"

# 기본 스타일
st.markdown("""
<style>
    .header {
        font-size: 2.5rem;
        color: #FF9900; /* Binance orange-ish */
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .subheader {
        font-size: 1.8rem;
        color: #FFFFFF;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .metric-card {
        background-color: #262730; /* Darker background */
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease-in-out;
        margin-bottom: 15px; /* Added margin for spacing */
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-title {
        font-size: 1rem;
        color: #AAAAAA;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #FFFFFF;
    }
    .positive {
        color: #28a745; /* Green */
    }
    .negative {
        color: #dc3545; /* Red */
    }
    .neutral {
        color: #ffc107; /* Yellow/Orange */
    }
</style>
""", unsafe_allow_html=True)

# ===== 데이터 로드 함수 =====
@st.cache_data(ttl=60) # 60초 캐싱
def load_data():
    conn = sqlite3.connect(DB_FILE)
    
    trades_df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC", conn)
    ai_analysis_df = pd.read_sql_query("SELECT * FROM ai_analysis ORDER BY timestamp DESC", conn)
    
    conn.close()
    
    # 데이터 전처리
    if not trades_df.empty:
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        # profit_loss가 없는 경우 0으로 채우기 (새로운 OPEN 상태 등)
        trades_df['profit_loss'] = trades_df['profit_loss'].fillna(0) 
        trades_df['profit_loss_percentage'] = trades_df['profit_loss_percentage'].fillna(0)
        
    if not ai_analysis_df.empty:
        ai_analysis_df['timestamp'] = pd.to_datetime(ai_analysis_df['timestamp'])
        
    return trades_df, ai_analysis_df

# ===== 성능 지표 계산 함수 =====
def calculate_performance_metrics(trades_df):
    metrics = {
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0.0,
        'total_profit_loss': 0.0,
        'avg_profit_per_trade': 0.0,
        'avg_loss_per_trade': 0.0,
        'max_drawdown': 0.0,
        'max_profit': 0.0,
        'max_loss': 0.0,
        'avg_profit_percentage': 0.0,
        'avg_loss_percentage': 0.0,
        'reward_risk_ratio': 0.0
    }

    if trades_df.empty:
        return metrics

    closed_trades = trades_df[trades_df['status'].str.contains('CLOSED')] # CLOSED 상태만 고려
    
    if closed_trades.empty:
        return metrics

    metrics['total_trades'] = len(closed_trades)
    
    winning_trades_df = closed_trades[closed_trades['profit_loss'] > 0]
    losing_trades_df = closed_trades[closed_trades['profit_loss'] < 0]

    metrics['winning_trades'] = len(winning_trades_df)
    metrics['losing_trades'] = len(losing_trades_df)

    if metrics['total_trades'] > 0:
        metrics['win_rate'] = (metrics['winning_trades'] / metrics['total_trades']) * 100

    metrics['total_profit_loss'] = closed_trades['profit_loss'].sum()

    if metrics['winning_trades'] > 0:
        metrics['avg_profit_per_trade'] = winning_trades_df['profit_loss'].mean()
        metrics['avg_profit_percentage'] = winning_trades_df['profit_loss_percentage'].mean()
        metrics['max_profit'] = winning_trades_df['profit_loss'].max()
    
    if metrics['losing_trades'] > 0:
        metrics['avg_loss_per_trade'] = losing_trades_df['profit_loss'].mean()
        metrics['avg_loss_percentage'] = losing_trades_df['profit_loss_percentage'].mean()
        metrics['max_loss'] = losing_trades_df['profit_loss'].min()

    # Reward/Risk Ratio
    if metrics['avg_loss_per_trade'] < 0: # Ensure there are losing trades to calculate ratio
        metrics['reward_risk_ratio'] = abs(metrics['avg_profit_per_trade'] / metrics['avg_loss_per_trade'])

    # Max Drawdown (Cumulative sum approach)
    cumulative_returns = closed_trades['profit_loss'].cumsum()
    if not cumulative_returns.empty:
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak.where(peak > 0, 1) # Avoid division by zero
        metrics['max_drawdown'] = abs(drawdown.min() * 100) if not drawdown.empty else 0.0
        
    return metrics

# ===== Streamlit UI =====
st.markdown("<h1 class='header'>📈 Bitcoin Futures AI Trading Dashboard</h1>", unsafe_allow_html=True)

trades_df, ai_analysis_df = load_data()

# ===== 1. 현재 포지션 정보 =====
st.markdown("<h2 class='subheader'>Current Position</h2>", unsafe_allow_html=True)

current_open_trade = trades_df[trades_df['status'] == 'OPEN'].head(1)

if not current_open_trade.empty:
    trade = current_open_trade.iloc[0]
    
    # 바이낸스에서 최신 가격 정보 가져오기 (API 키 필요)
    try:
        exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET_KEY'),
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        ticker = exchange.fetch_ticker('BTC/USDT')
        current_price = ticker['last']
    except Exception as e:
        st.warning(f"Failed to fetch current price from Binance: {e}. Displaying static price.")
        current_price = trade['entry_price'] # Fallback to entry price

    profit_loss = (current_price - trade['entry_price']) * trade['amount'] * (1 if trade['action'] == 'long' else -1)
    # profit_loss_percentage = (profit_loss / (trade['entry_price'] * trade['amount'] / trade['leverage'])) * 100
    # Profit/Loss Percentage calculation from erus_GM.py's calculate_profit_loss logic, assuming gross for display
    initial_capital_at_risk = (trade['entry_price'] * trade['amount']) / trade['leverage']
    profit_loss_percentage = (profit_loss / initial_capital_at_risk) * 100 if initial_capital_at_risk > 0 else 0


    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-title'>Direction</div>
            <div class='metric-value {'positive' if trade['action'] == 'long' else 'negative'}'>{trade['action'].upper()}</div>
        </div>
        <div class='metric-card'>
            <div class='metric-title'>Entry Price</div>
            <div class='metric-value'>${trade['entry_price']:,.2f}</div>
        </div>
        <div class='metric-card'>
            <div class='metric-title'>Current Price</div>
            <div class='metric-value'>${current_price:,.2f}</div>
        </div>
        <div class='metric-card'>
            <div class='metric-title'>Amount</div>
            <div class='metric-value'>{trade['amount']:.4f} BTC</div>
        </div>
        <div class='metric-card'>
            <div class='metric-title'>Leverage</div>
            <div class='metric-value'>{trade['leverage']}x</div>
        </div>
        <div class='metric-card'>
            <div class='metric-title'>Stop Loss</div>
            <div class='metric-value'>${trade['sl_price']:,.2f}</div>
        </div>
        <div class='metric-card'>
            <div class='metric-title'>Take Profit</div>
            <div class='metric-value'>${trade['tp_price']:,.2f}</div>
        </div>
        <div class='metric-card'>
            <div class='metric-title'>Current P/L</div>
            <div class='metric-value {'positive' if profit_loss > 0 else 'negative' if profit_loss < 0 else 'neutral'}>
                ${profit_loss:,.2f} ({profit_loss_percentage:,.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown("<hr>", unsafe_allow_html=True)

else:
    st.info("No open positions currently.")

# ===== 2. 전체 거래 성과 요약 =====
st.markdown("<h2 class='subheader'>Overall Trading Performance Summary</h2>", unsafe_allow_html=True)
overall_metrics = calculate_performance_metrics(trades_df)

metrics_cols = st.columns(4)
metrics_data = [
    ("Total P/L", f"${overall_metrics['total_profit_loss']:,.2f}", 'positive' if overall_metrics['total_profit_loss'] >= 0 else 'negative'),
    ("Win Rate", f"{overall_metrics['win_rate']:.2f}%", 'positive' if overall_metrics['win_rate'] >= 50 else 'negative'),
    ("Total Trades", f"{overall_metrics['total_trades']}", 'neutral'),
    ("Avg. Win %", f"{overall_metrics['avg_profit_percentage']:.2f}%", 'positive'),
    ("Avg. Loss %", f"{overall_metrics['avg_loss_percentage']:.2f}%", 'negative'),
    ("Max Drawdown", f"{overall_metrics['max_drawdown']:.2f}%", 'negative'),
    ("Avg. Reward/Risk", f"{overall_metrics['reward_risk_ratio']:.2f}", 'positive' if overall_metrics['reward_risk_ratio'] >= 1 else 'negative'),
    ("Max Profit", f"${overall_metrics['max_profit']:,.2f}", 'positive'),
    ("Max Loss", f"${overall_metrics['max_loss']:,.2f}", 'negative'),
]

col_idx = 0
for title, value, color_class in metrics_data:
    with metrics_cols[col_idx % 4]:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-title'>{title}</div>
                <div class='metric-value {color_class}'>{value}</div>
            </div>
        """, unsafe_allow_html=True)
    col_idx += 1


# ===== 3. 거래 성과 차트 =====
st.markdown("<h2 class='subheader'>Trading Performance Charts</h2>", unsafe_allow_html=True)
chart_cols = st.columns(2)

closed_trades = trades_df[trades_df['status'].str.contains('CLOSED')]

with chart_cols[0]:
    if not closed_trades.empty:
        # 누적 수익 차트
        trades_sorted = closed_trades.sort_values('timestamp')
        trades_sorted['cumulative_pl'] = trades_sorted['profit_loss'].cumsum()
        
        fig = px.line(
            trades_sorted, 
            x='timestamp', 
            y='cumulative_pl',
            title='Cumulative Profit/Loss (USDT)',
            labels={'timestamp': 'Date', 'cumulative_pl': 'P/L (USDT)'}
        )
        fig.update_traces(mode='lines+markers', marker_size=5)
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No closed trades to display cumulative P/L.")

with chart_cols[1]:
    if not closed_trades.empty:
        # 승률 파이 차트
        winning_trades = len(closed_trades[closed_trades['profit_loss'] > 0])
        losing_trades = len(closed_trades[closed_trades['profit_loss'] < 0])
        
        if winning_trades == 0 and losing_trades == 0:
            st.info("No winning or losing trades to display Win/Loss Ratio.")
        else:
            fig = px.pie(
                values=[winning_trades, losing_trades],
                names=['Winning', 'Losing'],
                title='Win/Loss Ratio',
                color_discrete_sequence=['#28a745', '#dc3545'] # Green for winning, Red for losing
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No closed trades to display Win/Loss Ratio.")

# ===== 4. 최근 거래 내역 =====
st.markdown("<h2 class='subheader'>Recent Trades</h2>", unsafe_allow_html=True)
if not trades_df.empty:
    # trade_id와 analysis_id 연결
    merged_df = pd.merge(trades_df, ai_analysis_df, left_on='analysis_id', right_on='id', suffixes=('_trade', '_ai'), how='left')
    
    # 필요한 컬럼만 선택하고 순서 조정
    display_cols = merged_df[[
        'timestamp_trade', 'action', 'amount', 'entry_price', 'exit_price', 'profit_loss', 'profit_loss_percentage', 
        'leverage', 'status', 'recommended_position_size', 'recommended_leverage', 
        'stop_loss_percentage', 'take_profit_percentage', 'reasoning'
    ]].copy()
    
    # 컬럼 이름 변경 (가독성 향상)
    display_cols.rename(columns={
        'timestamp_trade': 'Trade Time',
        'action': 'Direction',
        'amount': 'Amount (BTC)',
        'entry_price': 'Entry Price ($)',
        'exit_price': 'Exit Price ($)',
        'profit_loss': 'P/L (USDT)',
        'profit_loss_percentage': 'P/L (%)',
        'leverage': 'Actual Leverage',
        'status': 'Status',
        'recommended_position_size': 'AI Position Size %',
        'recommended_leverage': 'AI Leverage',
        'stop_loss_percentage': 'AI SL %',
        'take_profit_percentage': 'AI TP %',
        'reasoning': 'AI Reasoning'
    }, inplace=True)

    # 포맷팅 적용 (수익률, 금액)
    display_cols['Entry Price ($)'] = display_cols['Entry Price ($)'].apply(lambda x: f"${x:,.2f}")
    display_cols['Exit Price ($)'] = display_cols['Exit Price ($)'].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else 'N/A')
    display_cols['P/L (USDT)'] = display_cols['P/L (USDT)'].apply(lambda x: f"${x:,.2f}")
    display_cols['P/L (%)'] = display_cols['P/L (%)'].apply(lambda x: f"{x:,.2f}%")
    display_cols['AI Position Size %'] = display_cols['AI Position Size %'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else 'N/A')
    display_cols['AI SL %'] = display_cols['AI SL %'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else 'N/A')
    display_cols['AI TP %'] = display_cols['AI TP %'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else 'N/A')

    # 스타일링을 위한 HTML 출력
    def highlight_status(row):
        color = ''
        if 'OPEN' in row['Status']:
            color = 'background-color: #3333FF;' # Blue
        elif 'CLOSED_BY_TP' in row['Status']:
            color = 'background-color: #28a745;' # Green
        elif 'CLOSED_BY_SL' in row['Status']:
            color = 'background-color: #dc3545;' # Red
        elif 'CLOSED_MANUALLY' in row['Status']:
            color = 'background-color: #ffc107;' # Orange
        elif 'CLOSED_BY_AI_RE_EVALUATION' in row['Status']:
            color = 'background-color: #6f42c1;' # Purple
        elif 'CLOSED_MISMATCH' in row['Status']:
            color = 'background-color: #6c757d;' # Gray
        
        pl_color = ''
        if pd.notna(row['P/L (USDT)']):
            pl_value = float(row['P/L (USDT)'].replace('$', '').replace(',', ''))
            if pl_value > 0:
                pl_color = 'color: #28a745;'
            elif pl_value < 0:
                pl_color = 'color: #dc3545;'

        return [
            f'{color}', # for Status column
            f'{pl_color}', # for P/L (USDT) column
            f'{pl_color}', # for P/L (%) column
            '', '', '', '', '', '', '', '', '', '' # Default for other columns
        ]

    # DataFrame을 HTML로 변환
    st.dataframe(
        display_cols.style
        .apply(lambda row: highlight_status(row), 
               axis=1, 
               subset=['Status', 'P/L (USDT)', 'P/L (%)'] # Apply to specific columns
        ), 
        use_container_width=True
    )
else:
    st.info("No trades recorded yet.")

# ===== 5. AI 분석 기록 (AI의 의사결정 추적) =====
st.markdown("<h2 class='subheader'>AI Analysis History</h2>", unsafe_allow_html=True)

if not ai_analysis_df.empty:
    ai_display_cols = ai_analysis_df[[
        'timestamp', 'direction', 'conviction', 'recommended_position_size', 
        'recommended_leverage', 'stop_loss_percentage', 'take_profit_percentage', 'reasoning'
    ]].copy()

    ai_display_cols.rename(columns={
        'timestamp': 'Analysis Time',
        'direction': 'AI Direction',
        'conviction': 'AI Conviction',
        'recommended_position_size': 'AI Pos Size %',
        'recommended_leverage': 'AI Leverage',
        'stop_loss_percentage': 'AI SL %',
        'take_profit_percentage': 'AI TP %',
        'reasoning': 'AI Reasoning'
    }, inplace=True)
    
    ai_display_cols['AI Pos Size %'] = ai_display_cols['AI Pos Size %'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else 'N/A')
    ai_display_cols['AI Conviction'] = ai_display_cols['AI Conviction'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else 'N/A')
    ai_display_cols['AI SL %'] = ai_display_cols['AI SL %'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else 'N/A')
    ai_display_cols['AI TP %'] = ai_display_cols['AI TP %'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else 'N/A')

    st.dataframe(ai_display_cols, use_container_width=True)
else:
    st.info("No AI analysis records yet.")

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Data is refreshed every 60 seconds.")