#!/usr/bin/env python3
"""
럭셜마린더블비 KTR 크로스후 첫원비 전략 백테스팅 엔진
23~25년 완다차트 데이터 기반
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# 1. 데이터 로드
# =====================================================
def load_chart_data():
    """23~25년 차트 데이터 로드 및 통합"""
    c23 = pd.read_csv('23년 10분차트.csv')
    c24 = pd.read_csv('24년 10분차트.csv')
    c25 = pd.read_excel('25년 10분차트.xlsx')
    
    all_data = pd.concat([c23, c24, c25], ignore_index=True)
    all_data['time'] = pd.to_datetime(all_data['time'])
    all_data = all_data.sort_values('time').reset_index(drop=True)
    
    # 컬럼 이름 정리 (숫자형으로 변환)
    numeric_cols = ['open', 'high', 'low', 'close', '원비 매도', '원비 매수', 
                    '골든 크로스', '데드 크로스', '아시아 박스 상단값', '아시아 박스 하단값',
                    '20 SMA', '120 SMA']
    for col in numeric_cols:
        if col in all_data.columns:
            all_data[col] = pd.to_numeric(all_data[col], errors='coerce')
    
    return all_data

def load_trade_history():
    """실제 거래 내역 로드"""
    trades = pd.read_csv('23~25년 전략 거래목록 10분.csv')
    trades['날짜 및 시간'] = pd.to_datetime(trades['날짜 및 시간'])
    for col in ['가격 USD', '순손익 USD', '순손익 %', '누적 손익 USD', '누적 손익 %',
                '포지션 크기 (수량)', '포지션 크기 (가치)', '유리한 변동 USD', '불리한 변동 USD']:
        trades[col] = pd.to_numeric(trades[col], errors='coerce')
    return trades

# =====================================================
# 2. KTR 계산 (세션별 레인지)
# =====================================================
def get_session_info(dt, us_dst=None, euro_dst=None):
    """시간대별 세션 구분 (GMT+9 기준)"""
    h = dt.hour
    m = dt.minute
    cur_min = h * 60 + m
    
    # DST 처리 (간략화: 3월 둘째 일요일~11월 첫째 일요일)
    if us_dst is None:
        month = dt.month
        us_dst = 3 <= month <= 10  # 대략적 DST
    if euro_dst is None:
        month = dt.month
        euro_dst = 3 <= month <= 10
    
    asia_h = 7 if us_dst else 8
    euro_h = 16 if euro_dst else 17
    us_h = 22 if us_dst else 23
    us_m = 30
    
    asia_start = asia_h * 60
    euro_start = euro_h * 60
    us_start = us_h * 60 + us_m
    
    if cur_min >= asia_start and cur_min < euro_start:
        return 'asia', asia_h, us_dst, euro_dst
    elif cur_min >= euro_start and cur_min < us_start:
        return 'euro', asia_h, us_dst, euro_dst
    else:
        return 'us', asia_h, us_dst, euro_dst

def is_us_dst(dt):
    """미국 DST 여부 (3월 둘째 일요일 ~ 11월 첫째 일요일)"""
    year = dt.year
    # 3월 둘째 일요일
    d = datetime(year, 3, 1)
    sundays = 0
    while sundays < 2:
        if d.weekday() == 6:
            sundays += 1
            if sundays == 2:
                break
        d += timedelta(days=1)
    dst_start = d
    
    # 11월 첫째 일요일
    d = datetime(year, 11, 1)
    while d.weekday() != 6:
        d += timedelta(days=1)
    dst_end = d
    
    # tz-naive로 비교
    dt_naive = dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    return dst_start <= dt_naive < dst_end

def is_euro_dst(dt):
    """유럽 DST 여부 (3월 마지막 일요일 ~ 10월 마지막 일요일)"""
    year = dt.year
    # 3월 마지막 일요일
    d = datetime(year, 3, 31)
    while d.weekday() != 6:
        d -= timedelta(days=1)
    dst_start = d
    
    # 10월 마지막 일요일
    d = datetime(year, 10, 31)
    while d.weekday() != 6:
        d -= timedelta(days=1)
    dst_end = d
    
    dt_naive = dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    return dst_start <= dt_naive < dst_end

# =====================================================
# 3. 백테스팅 엔진
# =====================================================
class BacktestEngine:
    def __init__(self, data):
        self.data = data.copy()
        self.initial_capital = 10000.0
        self.risk_pct = 10.0
        self.counter_risk_pct = 5.0
        self.point_value = 1.0
        self.max_add_step = 5
        self.selected_stop_ktr = 5.5  # maxAddStep + 0.5
        self.max_regular_onebi = 3
        self.max_reverse_onebi = 2
        self.tp_ktr_mult = 3
        
        # 상태 변수 초기화
        self.reset_state()
        
        # 결과 저장
        self.trades = []
        self.equity_curve = []
        self.cumulative_pnl = 0.0
        
    def reset_state(self):
        self.cycle_state = 0
        self.cross_bar = None
        self.just_closed_bar = None
        self.filled_steps = 0
        
        self.engine_mode = 0
        self.engine_dir = 0
        self.last_cross_dir = 0
        self.setup_count = 0
        self.active_trade_mode = 0
        self.active_trade_dir = 0
        self.active_trade_session = 0
        
        self.long_onebi_count = 0
        self.short_onebi_count = 0
        self.last_long_onebi_bar = None
        self.last_short_onebi_bar = None
        
        self.setup_ktr = None
        self.base_price = None
        self.stop_price = None
        self.tp_price = None
        self.entry_prices = [None] * 6
        self.qtys = [None] * 6
        self.entry_risk_usd = None
        
        self.active_side = ""
        self.defense_mode = False
        self.force_close_issued = False
        self.pending_close_reason = ""
        
        self.reverse_arm = 0
        self.reverse_cross_bar = None
        self.reverse_next_dir = 0
        self.reverse_seed_bar = None
        
        self.b_used_today_long = False
        self.b_used_today_short = False
        self.b_break_armed = False
        self.b_break_dir = 0
        self.b_break_bar = None
        
        # 박스 상태
        self.today_box_top = None
        self.today_box_bottom = None
        self.box_ready = False
        self.active_box_session_key = None
        self.prev_session_key = None
        
        self.box_building = False
        self.build_high = None
        self.build_low = None
        self.build_session_key = None
        
        # 포지션
        self.position_size = 0.0
        self.position_avg_price = 0.0
        self.open_trades = 0
        
        # KTR
        self.ktr_values = {}
        self.active_ktr = None

    def get_stepped_capital(self, equity):
        if equity >= 10000:
            return max(int(equity / 5000) * 5000.0, 1000.0)
        else:
            return max(int(equity / 1000) * 1000.0, 1000.0)

    def calc_risk_split_qtys(self, risk_usd, ktr):
        if risk_usd is None or risk_usd <= 0 or ktr is None or ktr <= 0:
            return [None] * 6
        
        total_steps = self.max_add_step + 1.0
        per_step_risk = risk_usd / total_steps
        
        qtys = []
        for i in range(6):
            if i == 0:
                q = per_step_risk / (self.selected_stop_ktr * ktr * self.point_value)
            elif i <= self.max_add_step:
                divisor = (self.selected_stop_ktr - i) * ktr * self.point_value
                q = per_step_risk / divisor if divisor > 0 else None
            else:
                q = None
            qtys.append(q)
        return qtys

    def get_session_key(self, dt, asia_h):
        """세션 날짜 키 (아시아 시작 기준)"""
        h = dt.hour
        m = dt.minute
        cur_min = h * 60 + m
        if cur_min >= asia_h * 60:
            return dt.year * 10000 + dt.month * 100 + dt.day
        else:
            prev = dt - timedelta(days=1)
            return prev.year * 10000 + prev.month * 100 + prev.day

    def compute_ktr_from_data(self, i, session_type):
        """간략 KTR 계산: 세션 첫 2봉 합산 레인지 (10분 기준)"""
        # 실제로는 request.security를 통해 세션 시작시 계산
        # 여기서는 이전 봉의 레인지를 사용하는 간략화된 버전
        if i < 2:
            return None
        row = self.data.iloc[i]
        prev = self.data.iloc[i-1]
        combined_high = max(row['high'], prev['high'])
        combined_low = min(row['low'], prev['low'])
        ktr = combined_high - combined_low
        return ktr if ktr > 0 else None

    def is_above_today_box(self, close_price):
        return (self.box_ready and 
                self.today_box_top is not None and 
                close_price > self.today_box_top)

    def is_below_today_box(self, close_price):
        return (self.box_ready and 
                self.today_box_bottom is not None and 
                close_price < self.today_box_bottom)

    def close_position(self, exit_price, bar_idx, exit_time, reason=""):
        """포지션 청산"""
        if self.position_size == 0:
            return
        
        if self.active_side == "LONG":
            pnl = (exit_price - self.position_avg_price) * abs(self.position_size) * self.point_value
        else:
            pnl = (self.position_avg_price - exit_price) * abs(self.position_size) * self.point_value
        
        self.cumulative_pnl += pnl
        
        trade = {
            'entry_time': self.entry_time,
            'exit_time': exit_time,
            'side': self.active_side,
            'entry_price': self.position_avg_price,
            'exit_price': exit_price,
            'qty': abs(self.position_size),
            'pnl': pnl,
            'cumulative_pnl': self.cumulative_pnl,
            'mode': self.active_trade_mode,
            'reason': reason,
            'onebi_count': self.long_onebi_count if self.active_side == "LONG" else self.short_onebi_count,
            'filled_steps': self.filled_steps
        }
        self.trades.append(trade)
        
        # 상태 초기화
        self.position_size = 0.0
        self.position_avg_price = 0.0
        self.open_trades = 0
        self.filled_steps = 0
        
    def enter_position(self, side, price, qty, bar_idx, entry_time):
        """포지션 진입"""
        self.active_side = side
        self.position_avg_price = price
        self.position_size = qty if side == "LONG" else -qty
        self.open_trades = 1
        self.filled_steps = 1
        self.base_price = price
        self.entry_time = entry_time
        
        # 진입가/손절/익절 계산
        self.entry_prices[0] = price
        for i in range(1, 6):
            if i <= self.max_add_step:
                if side == "LONG":
                    self.entry_prices[i] = price - self.setup_ktr * i
                else:
                    self.entry_prices[i] = price + self.setup_ktr * i
            else:
                self.entry_prices[i] = None
        
        if side == "LONG":
            self.stop_price = price - self.setup_ktr * self.selected_stop_ktr
            self.tp_price = price + self.setup_ktr * self.tp_ktr_mult
        else:
            self.stop_price = price + self.setup_ktr * self.selected_stop_ktr
            self.tp_price = price - self.setup_ktr * self.tp_ktr_mult
    
    def check_fills_and_exits(self, i, row):
        """추가 체결 및 익절/손절 확인"""
        if self.position_size == 0:
            return
            
        high = row['high']
        low = row['low']
        close = row['close']
        
        # 손절/익절 확인
        if self.active_side == "LONG":
            if self.stop_price is not None and low <= self.stop_price:
                self.close_position(self.stop_price, i, row['time'], "손절")
                self.just_closed_bar = i
                return
            if self.tp_price is not None and high >= self.tp_price:
                self.close_position(self.tp_price, i, row['time'], "익절")
                self.just_closed_bar = i
                return
                
            # 추가 체결 확인
            for step in range(1, 6):
                if step <= self.max_add_step and self.filled_steps == step:
                    ep = self.entry_prices[step]
                    if ep is not None and low <= ep and self.qtys[step] is not None:
                        self.filled_steps += 1
                        self.open_trades += 1
                        # 평균가 재계산
                        total_qty = sum(self.qtys[j] for j in range(self.filled_steps) if self.qtys[j] is not None)
                        total_cost = sum(self.qtys[j] * self.entry_prices[j] for j in range(self.filled_steps) 
                                       if self.qtys[j] is not None and self.entry_prices[j] is not None)
                        if total_qty > 0:
                            self.position_avg_price = total_cost / total_qty
                            self.position_size = total_qty
                        
                        # 멀티구간이면 익절가를 진입가로 변경
                        if self.filled_steps >= 2:
                            self.tp_price = self.base_price
                        break
        else:  # SHORT
            if self.stop_price is not None and high >= self.stop_price:
                self.close_position(self.stop_price, i, row['time'], "손절")
                self.just_closed_bar = i
                return
            if self.tp_price is not None and low <= self.tp_price:
                self.close_position(self.tp_price, i, row['time'], "익절")
                self.just_closed_bar = i
                return
                
            # 추가 체결 확인
            for step in range(1, 6):
                if step <= self.max_add_step and self.filled_steps == step:
                    ep = self.entry_prices[step]
                    if ep is not None and high >= ep and self.qtys[step] is not None:
                        self.filled_steps += 1
                        self.open_trades += 1
                        total_qty = sum(self.qtys[j] for j in range(self.filled_steps) if self.qtys[j] is not None)
                        total_cost = sum(self.qtys[j] * self.entry_prices[j] for j in range(self.filled_steps)
                                       if self.qtys[j] is not None and self.entry_prices[j] is not None)
                        if total_qty > 0:
                            self.position_avg_price = total_cost / total_qty
                            self.position_size = -total_qty
                        
                        if self.filled_steps >= 2:
                            self.tp_price = self.base_price
                        break

    def run(self):
        """메인 백테스팅 루프"""
        data = self.data
        n = len(data)
        
        # 사전 계산: SMA 20, 120 (차트에서 이미 제공)
        # 골든크로스, 데드크로스도 이미 제공
        
        prev_close = None
        
        for i in range(n):
            row = data.iloc[i]
            dt = row['time']
            if pd.isna(dt):
                continue
                
            # dt가 Timestamp인지 확인
            if hasattr(dt, 'to_pydatetime'):
                dt_py = dt.to_pydatetime()
            else:
                dt_py = dt
            
            open_p = row['open']
            high_p = row['high']
            low_p = row['low']
            close_p = row['close']
            
            if pd.isna(close_p):
                continue
            
            # 신호 읽기
            golden_cross = row.get('골든 크로스', 0) == 1
            dead_cross = row.get('데드 크로스', 0) == 1
            buy_condition1 = row.get('원비 매수', 0) == 1
            sell_condition1 = row.get('원비 매도', 0) == 1
            
            # 박스 값
            box_top = row.get('아시아 박스 상단값', None)
            box_bottom = row.get('아시아 박스 하단값', None)
            
            if pd.notna(box_top) and pd.notna(box_bottom) and box_top != '' and box_bottom != '':
                try:
                    self.today_box_top = float(box_top)
                    self.today_box_bottom = float(box_bottom)
                    self.box_ready = True
                except (ValueError, TypeError):
                    pass
            
            # SMA 값
            sma20 = row.get('20 SMA', None)
            sma120 = row.get('120 SMA', None)
            
            # DST
            us_dst = is_us_dst(dt_py) if hasattr(dt_py, 'year') else False
            euro_dst = is_euro_dst(dt_py) if hasattr(dt_py, 'year') else False
            
            # 세션
            asia_h = 7 if us_dst else 8
            session_key = self.get_session_key(dt_py, asia_h)
            
            # 새 세션 감지
            if self.prev_session_key is None or session_key != self.prev_session_key:
                self.prev_session_key = session_key
                self.box_ready = False
                self.today_box_top = None
                self.today_box_bottom = None
                self.b_used_today_long = False
                self.b_used_today_short = False
                self.b_break_armed = False
                self.b_break_dir = 0
                self.b_break_bar = None
                
                # 박스 값이 있으면 설정
                if pd.notna(box_top) and pd.notna(box_bottom):
                    try:
                        self.today_box_top = float(box_top)
                        self.today_box_bottom = float(box_bottom)
                        self.box_ready = True
                    except (ValueError, TypeError):
                        pass
            
            # 무포지션 진입금지 시간
            us_sub_h = 22 if us_dst else 23
            block_start = (us_sub_h - 1) * 60
            block_end = block_start + 120
            cur_min = dt_py.hour * 60 + dt_py.minute
            in_flat_entry_block = block_start <= cur_min < block_end
            is_flat = self.position_size == 0 and self.open_trades == 0
            can_enter = not (is_flat and in_flat_entry_block)
            
            # 세션 타입
            euro_h = 16 if euro_dst else 17
            us_start_min = us_sub_h * 60 + 30
            if cur_min >= asia_h * 60 and cur_min < euro_h * 60:
                session_type = 1  # asia
            elif cur_min >= euro_h * 60 and cur_min < us_start_min:
                session_type = 2  # euro
            else:
                session_type = 3  # us
            
            # KTR (간략 계산: 차트에 없으므로 2봉 합산 레인지 사용)
            self.active_ktr = self.compute_ktr_from_data(i, session_type)
            ktr_ready = self.active_ktr is not None and self.active_ktr > 0
            
            # 1시간 역추세 필터 (간략화)
            htf_trend_dir = 0
            if pd.notna(sma20) and pd.notna(sma120):
                if sma20 > sma120:
                    htf_trend_dir = 1
                elif sma20 < sma120:
                    htf_trend_dir = -1
            
            # 시드 계산
            equity = self.initial_capital + self.cumulative_pnl
            stepped_capital = self.get_stepped_capital(equity)
            
            # ── 거래중 확인 ──
            if self.position_size != 0:
                # 반대 크로스 감시
                if self.cycle_state == 3 and dead_cross and not self.defense_mode:
                    self.defense_mode = True
                    self.reverse_arm = -1
                    self.reverse_cross_bar = i
                    # 방어모드: 익절가 조정
                    if self.filled_steps >= 2:
                        defense_step = max(1, self.filled_steps - 2)
                        self.tp_price = self.entry_prices[defense_step - 1] if self.entry_prices[defense_step - 1] else self.tp_price
                    
                if self.cycle_state == -3 and golden_cross and not self.defense_mode:
                    self.defense_mode = True
                    self.reverse_arm = 1
                    self.reverse_cross_bar = i
                    if self.filled_steps >= 2:
                        defense_step = max(1, self.filled_steps - 2)
                        self.tp_price = self.entry_prices[defense_step - 1] if self.entry_prices[defense_step - 1] else self.tp_price
                
                # 반대신호정리
                if (self.defense_mode and self.reverse_arm == -1 and 
                    i > (self.reverse_cross_bar or i) and sell_condition1 and not self.force_close_issued):
                    self.pending_close_reason = "반대신호정리"
                    self.force_close_issued = True
                    self.reverse_next_dir = -1
                    self.reverse_seed_bar = i
                    self.close_position(close_p, i, row['time'], "반대신호정리")
                    self.just_closed_bar = i
                    
                elif (self.defense_mode and self.reverse_arm == 1 and 
                      i > (self.reverse_cross_bar or i) and buy_condition1 and not self.force_close_issued):
                    self.pending_close_reason = "반대신호정리"
                    self.force_close_issued = True
                    self.reverse_next_dir = 1
                    self.reverse_seed_bar = i
                    self.close_position(close_p, i, row['time'], "반대신호정리")
                    self.just_closed_bar = i
                else:
                    # 일반 체결/익절/손절 확인
                    self.check_fills_and_exits(i, row)
                
                # 거래 종료 후 상태 전이
                if self.position_size == 0 and self.just_closed_bar == i:
                    self.cycle_state = 0
                    self.defense_mode = False
                    self.force_close_issued = False
                    
                    if self.pending_close_reason == "반대신호정리" and self.reverse_next_dir != 0:
                        self.engine_mode = 2
                        self.engine_dir = self.reverse_next_dir
                        self.setup_count = 0
                        self.cross_bar = self.reverse_cross_bar
                        self.cycle_state = 1 if self.reverse_next_dir == 1 else -1
                        
                        if self.reverse_next_dir == 1:
                            self.long_onebi_count = 0
                            self.last_long_onebi_bar = self.reverse_seed_bar
                            self.short_onebi_count = 0
                        else:
                            self.short_onebi_count = 0
                            self.last_short_onebi_bar = self.reverse_seed_bar
                            self.long_onebi_count = 0
                        
                        self.reverse_next_dir = 0
                        self.reverse_seed_bar = None
                    else:
                        if self.active_trade_mode == 1:
                            if self.setup_count >= self.max_regular_onebi:
                                self.engine_mode = 3
                            else:
                                self.engine_mode = 1
                            self.cycle_state = 1 if self.engine_dir == 1 else -1
                        elif self.active_trade_mode == 2:
                            if self.setup_count >= self.max_reverse_onebi:
                                self.engine_mode = 3
                            else:
                                self.engine_mode = 2
                            self.cycle_state = 1 if self.engine_dir == 1 else -1
                        elif self.active_trade_mode == 3:
                            self.engine_mode = 3
                            self.cycle_state = 1 if self.engine_dir == 1 else -1
                    
                    self.active_trade_mode = 0
                    self.active_trade_dir = 0
                    self.pending_close_reason = ""
                
                # 에쿼티 트래킹
                self.equity_curve.append({
                    'time': row['time'],
                    'equity': equity,
                    'position': self.active_side
                })
                prev_close = close_p
                continue
            
            # ── 무포지션 상태 ──
            can_rearm = (self.position_size == 0 and self.open_trades == 0 and 
                        self.cycle_state != 2 and self.cycle_state != -2 and 
                        (self.just_closed_bar is None or i > self.just_closed_bar))
            
            if can_rearm:
                # 새 크로스
                if golden_cross and self.last_cross_dir != 1:
                    self.last_cross_dir = 1
                    self.engine_mode = 1
                    self.engine_dir = 1
                    self.setup_count = 0
                    self.cycle_state = 1
                    self.cross_bar = i
                    self.long_onebi_count = 0
                    self.short_onebi_count = 0
                    self.last_long_onebi_bar = None
                    self.last_short_onebi_bar = None
                    self.b_break_armed = False
                    self.b_break_dir = 0
                    self.b_break_bar = None
                    
                elif dead_cross and self.last_cross_dir != -1:
                    self.last_cross_dir = -1
                    self.engine_mode = 1
                    self.engine_dir = -1
                    self.setup_count = 0
                    self.cycle_state = -1
                    self.cross_bar = i
                    self.long_onebi_count = 0
                    self.short_onebi_count = 0
                    self.last_long_onebi_bar = None
                    self.last_short_onebi_bar = None
                    self.b_break_armed = False
                    self.b_break_dir = 0
                    self.b_break_bar = None
                
                # A모드 / 예외모드
                if can_rearm and (self.engine_mode == 1 or self.engine_mode == 2):
                    allowed_count = self.max_regular_onebi if self.engine_mode == 1 else self.max_reverse_onebi
                    
                    # 롱
                    if self.engine_dir == 1 and self.cycle_state == 1:
                        long_onebi_raw = (buy_condition1 and 
                                         (self.last_long_onebi_bar is None or i > self.last_long_onebi_bar))
                        
                        if long_onebi_raw:
                            self.last_long_onebi_bar = i
                            self.setup_count += 1
                            self.long_onebi_count = self.setup_count
                            
                            # 카운트 범위 내이고 + 박스 밖이면 → 진입
                            if (self.setup_count <= allowed_count and 
                                self.is_above_today_box(close_p) and 
                                ktr_ready and can_enter):
                                
                                self.setup_ktr = self.active_ktr
                                
                                # 리스크 계산
                                is_counter = htf_trend_dir != 0 and 1 != htf_trend_dir
                                risk_pct = self.counter_risk_pct if is_counter else self.risk_pct
                                risk_usd = stepped_capital * risk_pct * 0.01
                                self.entry_risk_usd = risk_usd
                                
                                self.qtys = self.calc_risk_split_qtys(risk_usd, self.setup_ktr)
                                
                                if self.qtys[0] is not None and self.qtys[0] > 0:
                                    self.active_trade_mode = self.engine_mode
                                    self.active_trade_dir = self.engine_dir
                                    self.active_trade_session = session_type
                                    self.enter_position("LONG", close_p, self.qtys[0], i, row['time'])
                                    self.cycle_state = 3  # 직접 3으로 (체결 즉시)
                            
                            # 카운트 소진 → B모드
                            if self.setup_count >= allowed_count and self.cycle_state != 3 and self.cycle_state != 2:
                                self.engine_mode = 3
                                self.cycle_state = 1
                                self.b_break_armed = False
                                self.b_break_dir = 0
                                self.b_break_bar = None
                    
                    # 숏
                    if self.engine_dir == -1 and self.cycle_state == -1:
                        short_onebi_raw = (sell_condition1 and 
                                          (self.last_short_onebi_bar is None or i > self.last_short_onebi_bar))
                        
                        if short_onebi_raw:
                            self.last_short_onebi_bar = i
                            self.setup_count += 1
                            self.short_onebi_count = self.setup_count
                            
                            if (self.setup_count <= allowed_count and 
                                self.is_below_today_box(close_p) and 
                                ktr_ready and can_enter):
                                
                                self.setup_ktr = self.active_ktr
                                
                                is_counter = htf_trend_dir != 0 and -1 != htf_trend_dir
                                risk_pct = self.counter_risk_pct if is_counter else self.risk_pct
                                risk_usd = stepped_capital * risk_pct * 0.01
                                self.entry_risk_usd = risk_usd
                                
                                self.qtys = self.calc_risk_split_qtys(risk_usd, self.setup_ktr)
                                
                                if self.qtys[0] is not None and self.qtys[0] > 0:
                                    self.active_trade_mode = self.engine_mode
                                    self.active_trade_dir = self.engine_dir
                                    self.active_trade_session = session_type
                                    self.enter_position("SHORT", close_p, self.qtys[0], i, row['time'])
                                    self.cycle_state = -3
                            
                            if self.setup_count >= allowed_count and self.cycle_state != -3 and self.cycle_state != -2:
                                self.engine_mode = 3
                                self.cycle_state = -1
                                self.b_break_armed = False
                                self.b_break_dir = 0
                                self.b_break_bar = None
                
                # B모드
                if can_rearm and self.engine_mode == 3:
                    b_trend_dir = 0
                    if pd.notna(sma20) and pd.notna(sma120):
                        if sma20 > sma120:
                            b_trend_dir = 1
                        elif sma20 < sma120:
                            b_trend_dir = -1
                    
                    if b_trend_dir == 0:
                        self.b_break_armed = False
                        self.b_break_dir = 0
                        self.b_break_bar = None
                    
                    if b_trend_dir != 0 and (self.engine_dir != b_trend_dir or 
                        self.cycle_state != (1 if b_trend_dir == 1 else -1)):
                        self.engine_dir = b_trend_dir
                        self.cycle_state = 1 if b_trend_dir == 1 else -1
                        self.b_break_armed = False
                        self.b_break_dir = 0
                        self.b_break_bar = None
                    
                    # 박스 돌파 감지
                    box_up_break = (self.box_ready and self.today_box_top is not None and 
                                   close_p > self.today_box_top and 
                                   prev_close is not None and prev_close <= self.today_box_top)
                    box_dn_break = (self.box_ready and self.today_box_bottom is not None and 
                                   close_p < self.today_box_bottom and 
                                   prev_close is not None and prev_close >= self.today_box_bottom)
                    
                    # B모드 롱
                    if b_trend_dir == 1 and self.cycle_state == 1:
                        if box_dn_break and self.b_break_armed and self.b_break_dir == 1:
                            self.b_break_armed = False
                            self.b_break_dir = 0
                            self.b_break_bar = None
                        
                        if box_up_break and not self.b_used_today_long:
                            self.b_break_armed = True
                            self.b_break_dir = 1
                            self.b_break_bar = i
                        
                        if (self.b_break_armed and self.b_break_dir == 1 and 
                            i > (self.b_break_bar or i) and
                            self.today_box_top is not None and close_p > self.today_box_top and
                            buy_condition1 and
                            (self.last_long_onebi_bar is None or i > self.last_long_onebi_bar)):
                            
                            self.last_long_onebi_bar = i
                            self.long_onebi_count = 1
                            self.short_onebi_count = 0
                            self.b_used_today_long = True
                            self.b_break_armed = False
                            self.b_break_dir = 0
                            self.b_break_bar = None
                            
                            if ktr_ready and can_enter:
                                self.setup_ktr = self.active_ktr
                                is_counter = htf_trend_dir != 0 and 1 != htf_trend_dir
                                risk_pct = self.counter_risk_pct if is_counter else self.risk_pct
                                risk_usd = stepped_capital * risk_pct * 0.01
                                self.entry_risk_usd = risk_usd
                                self.qtys = self.calc_risk_split_qtys(risk_usd, self.setup_ktr)
                                
                                if self.qtys[0] is not None and self.qtys[0] > 0:
                                    self.active_trade_mode = 3
                                    self.active_trade_dir = 1
                                    self.active_trade_session = session_type
                                    self.engine_dir = 1
                                    self.enter_position("LONG", close_p, self.qtys[0], i, row['time'])
                                    self.cycle_state = 3
                    
                    # B모드 숏
                    if b_trend_dir == -1 and self.cycle_state == -1:
                        if box_up_break and self.b_break_armed and self.b_break_dir == -1:
                            self.b_break_armed = False
                            self.b_break_dir = 0
                            self.b_break_bar = None
                        
                        if box_dn_break and not self.b_used_today_short:
                            self.b_break_armed = True
                            self.b_break_dir = -1
                            self.b_break_bar = i
                        
                        if (self.b_break_armed and self.b_break_dir == -1 and
                            i > (self.b_break_bar or i) and
                            self.today_box_bottom is not None and close_p < self.today_box_bottom and
                            sell_condition1 and
                            (self.last_short_onebi_bar is None or i > self.last_short_onebi_bar)):
                            
                            self.last_short_onebi_bar = i
                            self.short_onebi_count = 1
                            self.long_onebi_count = 0
                            self.b_used_today_short = True
                            self.b_break_armed = False
                            self.b_break_dir = 0
                            self.b_break_bar = None
                            
                            if ktr_ready and can_enter:
                                self.setup_ktr = self.active_ktr
                                is_counter = htf_trend_dir != 0 and -1 != htf_trend_dir
                                risk_pct = self.counter_risk_pct if is_counter else self.risk_pct
                                risk_usd = stepped_capital * risk_pct * 0.01
                                self.entry_risk_usd = risk_usd
                                self.qtys = self.calc_risk_split_qtys(risk_usd, self.setup_ktr)
                                
                                if self.qtys[0] is not None and self.qtys[0] > 0:
                                    self.active_trade_mode = 3
                                    self.active_trade_dir = -1
                                    self.active_trade_session = session_type
                                    self.engine_dir = -1
                                    self.enter_position("SHORT", close_p, self.qtys[0], i, row['time'])
                                    self.cycle_state = -3
            
            self.equity_curve.append({
                'time': row['time'],
                'equity': equity
            })
            prev_close = close_p
        
        # 마지막에 열린 포지션 강제 정리
        if self.position_size != 0:
            last_row = data.iloc[-1]
            self.close_position(last_row['close'], len(data)-1, last_row['time'], "기간종료정리")
        
        return self.trades

# =====================================================
# 4. 비교 분석
# =====================================================
def analyze_actual_trades(trades_df):
    """실제 거래 내역 분석"""
    # 거래 번호별 그룹핑 (진입/청산 쌍)
    grouped = trades_df.groupby('거래 #')
    
    results = []
    for trade_num, group in grouped:
        entries = group[group['타입'].str.contains('진입')]
        exits = group[group['타입'].str.contains('청산')]
        
        if len(entries) == 0 or len(exits) == 0:
            continue
        
        first_entry = entries.iloc[0]
        first_exit = exits.iloc[0]
        
        side = "LONG" if "매수" in first_entry['타입'] else "SHORT"
        signal = first_entry['신호']
        
        # 모드 판별
        if signal in ['L1', 'S1']:
            mode_name = "A/예외"
        elif signal.startswith('L') or signal.startswith('S'):
            mode_name = "추가구간"
        else:
            mode_name = "기타"
        
        results.append({
            'trade_num': trade_num,
            'entry_time': first_entry['날짜 및 시간'],
            'exit_time': first_exit['날짜 및 시간'],
            'side': side,
            'signal': signal,
            'entry_price': first_entry['가격 USD'],
            'exit_price': first_exit['가격 USD'],
            'pnl': first_exit['순손익 USD'],
            'cumulative_pnl': first_exit['누적 손익 USD'],
            'qty': first_entry['포지션 크기 (수량)'],
            'favorable': first_exit['유리한 변동 USD'],
            'adverse': first_exit['불리한 변동 USD'],
            'exit_signal': first_exit['신호']
        })
    
    return pd.DataFrame(results)

def generate_report(backtest_trades, actual_df, actual_analysis):
    """비교 리포트 생성"""
    bt_df = pd.DataFrame(backtest_trades)
    
    report = []
    report.append("# 백테스팅 결과 비교 리포트")
    report.append(f"**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    
    # ── 실제 거래 요약 ──
    report.append("## 1. 실제 거래 내역 요약 (TradingView 전략)")
    report.append("")
    
    # L1/S1 진입만 필터 (첫진입)
    first_entries = actual_analysis[actual_analysis['signal'].isin(['L1', 'S1'])]
    
    total_actual = len(first_entries)
    actual_long = len(first_entries[first_entries['side'] == 'LONG'])
    actual_short = len(first_entries[first_entries['side'] == 'SHORT'])
    
    actual_wins = len(first_entries[first_entries['pnl'] > 0])
    actual_losses = len(first_entries[first_entries['pnl'] < 0])
    actual_be = len(first_entries[first_entries['pnl'] == 0])
    
    actual_total_pnl = actual_df['누적 손익 USD'].iloc[-1] if len(actual_df) > 0 else 0
    actual_total_pnl_pct = actual_df['누적 손익 %'].iloc[-1] if len(actual_df) > 0 else 0
    
    report.append(f"| 항목 | 값 |")
    report.append(f"|------|------|")
    report.append(f"| 총 거래 수 (전체 행) | {len(actual_df)} |")
    report.append(f"| 총 거래 번호 수 | {actual_df['거래 #'].max()} |")
    report.append(f"| 첫진입(L1/S1) 횟수 | {total_actual} |")
    report.append(f"| 롱 진입 | {actual_long} |")
    report.append(f"| 숏 진입 | {actual_short} |")
    report.append(f"| 승리 | {actual_wins} |")
    report.append(f"| 패배 | {actual_losses} |")
    report.append(f"| 보합 | {actual_be} |")
    report.append(f"| 승률 | {actual_wins/total_actual*100:.1f}% |" if total_actual > 0 else "| 승률 | N/A |")
    report.append(f"| 최종 누적 손익 (USD) | ${actual_total_pnl:,.2f} |")
    report.append(f"| 최종 누적 손익 (%) | {actual_total_pnl_pct:.2f}% |")
    report.append("")
    
    # 연도별 실제 거래
    report.append("### 연도별 실제 거래 결과")
    report.append("")
    
    first_entries_copy = first_entries.copy()
    first_entries_copy['year'] = first_entries_copy['entry_time'].dt.year
    
    report.append("| 연도 | 거래수 | 승리 | 패배 | 승률 | 총 손익 (USD) |")
    report.append("|------|--------|------|------|------|---------------|")
    
    for year in sorted(first_entries_copy['year'].unique()):
        year_data = first_entries_copy[first_entries_copy['year'] == year]
        yr_wins = len(year_data[year_data['pnl'] > 0])
        yr_losses = len(year_data[year_data['pnl'] < 0])
        yr_total = len(year_data)
        yr_pnl = year_data['pnl'].sum()
        yr_wr = yr_wins / yr_total * 100 if yr_total > 0 else 0
        report.append(f"| {year} | {yr_total} | {yr_wins} | {yr_losses} | {yr_wr:.1f}% | ${yr_pnl:,.2f} |")
    report.append("")
    
    # 청산 사유별
    report.append("### 청산 사유별 분포 (실제)")
    report.append("")
    exit_dist = first_entries['exit_signal'].value_counts()
    report.append("| 청산 신호 | 횟수 | 비율 |")
    report.append("|-----------|------|------|")
    for sig, cnt in exit_dist.items():
        report.append(f"| {sig} | {cnt} | {cnt/total_actual*100:.1f}% |")
    report.append("")
    
    # ── 백테스팅 요약 ──
    report.append("## 2. Python 백테스팅 결과")
    report.append("")
    
    if len(bt_df) > 0:
        bt_total = len(bt_df)
        bt_long = len(bt_df[bt_df['side'] == 'LONG'])
        bt_short = len(bt_df[bt_df['side'] == 'SHORT'])
        bt_wins = len(bt_df[bt_df['pnl'] > 0])
        bt_losses = len(bt_df[bt_df['pnl'] < 0])
        bt_be = len(bt_df[bt_df['pnl'] == 0])
        bt_total_pnl = bt_df['cumulative_pnl'].iloc[-1] if len(bt_df) > 0 else 0
        bt_wr = bt_wins / bt_total * 100 if bt_total > 0 else 0
        
        report.append(f"| 항목 | 값 |")
        report.append(f"|------|------|")
        report.append(f"| 총 거래 수 | {bt_total} |")
        report.append(f"| 롱 | {bt_long} |")
        report.append(f"| 숏 | {bt_short} |")
        report.append(f"| 승리 | {bt_wins} |")
        report.append(f"| 패배 | {bt_losses} |")
        report.append(f"| 보합 | {bt_be} |")
        report.append(f"| 승률 | {bt_wr:.1f}% |")
        report.append(f"| 최종 누적 손익 (USD) | ${bt_total_pnl:,.2f} |")
        report.append("")
        
        # 모드별
        report.append("### 모드별 백테스팅 결과")
        report.append("")
        report.append("| 모드 | 거래수 | 승리 | 패배 | 승률 | 총 손익 |")
        report.append("|------|--------|------|------|------|---------|")
        for mode in sorted(bt_df['mode'].unique()):
            mode_data = bt_df[bt_df['mode'] == mode]
            mode_name = {1: "A모드(정규)", 2: "A모드(예외)", 3: "B모드"}.get(mode, f"모드{mode}")
            m_wins = len(mode_data[mode_data['pnl'] > 0])
            m_total = len(mode_data)
            m_pnl = mode_data['pnl'].sum()
            m_wr = m_wins / m_total * 100 if m_total > 0 else 0
            report.append(f"| {mode_name} | {m_total} | {m_wins} | {len(mode_data[mode_data['pnl'] < 0])} | {m_wr:.1f}% | ${m_pnl:,.2f} |")
        report.append("")
        
        # 연도별
        bt_df_copy = bt_df.copy()
        bt_df_copy['entry_time'] = pd.to_datetime(bt_df_copy['entry_time'])
        bt_df_copy['year'] = bt_df_copy['entry_time'].dt.year
        
        report.append("### 연도별 백테스팅 결과")
        report.append("")
        report.append("| 연도 | 거래수 | 승리 | 패배 | 승률 | 총 손익 (USD) |")
        report.append("|------|--------|------|------|------|---------------|")
        for year in sorted(bt_df_copy['year'].unique()):
            year_data = bt_df_copy[bt_df_copy['year'] == year]
            yr_wins = len(year_data[year_data['pnl'] > 0])
            yr_total = len(year_data)
            yr_pnl = year_data['pnl'].sum()
            yr_wr = yr_wins / yr_total * 100 if yr_total > 0 else 0
            report.append(f"| {year} | {yr_total} | {yr_wins} | {len(year_data[year_data['pnl'] < 0])} | {yr_wr:.1f}% | ${yr_pnl:,.2f} |")
        report.append("")
        
        # 청산 사유별
        report.append("### 청산 사유별 분포 (백테스팅)")
        report.append("")
        reason_dist = bt_df['reason'].value_counts()
        report.append("| 청산 사유 | 횟수 | 비율 |")
        report.append("|-----------|------|------|")
        for reason, cnt in reason_dist.items():
            report.append(f"| {reason} | {cnt} | {cnt/bt_total*100:.1f}% |")
        report.append("")
    else:
        report.append("백테스팅 결과 거래가 없습니다.")
        report.append("")
    
    # ── 비교 ──
    report.append("## 3. 실제 vs 백테스팅 비교")
    report.append("")
    
    report.append("| 항목 | 실제 (TradingView) | 백테스팅 (Python) | 차이 |")
    report.append("|------|-------------------|-------------------|------|")
    
    if len(bt_df) > 0:
        report.append(f"| 총 거래 수 | {total_actual} | {bt_total} | {bt_total - total_actual} |")
        report.append(f"| 롱 | {actual_long} | {bt_long} | {bt_long - actual_long} |")
        report.append(f"| 숏 | {actual_short} | {bt_short} | {bt_short - actual_short} |")
        actual_wr = actual_wins / total_actual * 100 if total_actual > 0 else 0
        report.append(f"| 승률 | {actual_wr:.1f}% | {bt_wr:.1f}% | {bt_wr - actual_wr:+.1f}% |")
        report.append(f"| 최종 누적 손익 | ${actual_total_pnl:,.2f} | ${bt_total_pnl:,.2f} | ${bt_total_pnl - actual_total_pnl:+,.2f} |")
    report.append("")
    
    # ── 차이 설명 ──
    report.append("## 4. 백테스팅 차이 원인 분석")
    report.append("")
    report.append("### Python 백테스팅과 TradingView 실제 결과가 차이나는 주요 원인:")
    report.append("")
    report.append("1. **KTR 계산 차이**: TradingView는 `request.security()`로 다중 타임프레임 KTR을 계산하지만, Python에서는 간략화된 2봉 합산 레인지를 사용. 이로 인해 포지션 사이즈, 손절/익절 레벨이 달라짐.")
    report.append("2. **주문 체결 모델**: TradingView는 `calc_on_every_tick=true`로 틱마다 계산하지만, Python은 봉 종가 기준으로 체결. 실제 체결 타이밍에 차이 발생.")
    report.append("3. **박스 계산**: 아시아 시가 박스를 차트에서 직접 읽지만, 세션 시작/종료 경계 처리에 미세한 차이가 있을 수 있음.")
    report.append("4. **추가 구간 체결**: 지정가 주문(limit order)의 체결 시뮬레이션이 간략화되어 있어, 실제 멀티구간 체결 패턴과 다를 수 있음.")
    report.append("5. **1시간 역추세 필터**: 10분 차트의 SMA로 근사했지만, 실제는 1시간 봉 기준 SMA20/120을 사용.")
    report.append("")
    
    report.append("## 5. 전략 신호 통계 (차트 데이터)")
    report.append("")
    
    return "\n".join(report)

# =====================================================
# 5. 실행
# =====================================================
if __name__ == "__main__":
    print("=" * 60)
    print("럭셜마린더블비 KTR 크로스후 첫원비 전략 백테스팅")
    print("=" * 60)
    
    print("\n[1] 차트 데이터 로딩...")
    chart_data = load_chart_data()
    print(f"    총 {len(chart_data)} 봉 로드 완료")
    print(f"    기간: {chart_data['time'].min()} ~ {chart_data['time'].max()}")
    
    print("\n[2] 실제 거래 내역 로딩...")
    actual_trades = load_trade_history()
    print(f"    총 {len(actual_trades)} 행 로드 완료")
    
    print("\n[3] 실제 거래 분석...")
    actual_analysis = analyze_actual_trades(actual_trades)
    print(f"    총 {len(actual_analysis)} 개 거래 분석 완료")
    
    print("\n[4] 백테스팅 실행중...")
    engine = BacktestEngine(chart_data)
    bt_trades = engine.run()
    print(f"    백테스팅 완료: {len(bt_trades)} 개 거래 생성")
    
    print("\n[5] 비교 리포트 생성...")
    report = generate_report(bt_trades, actual_trades, actual_analysis)
    
    # 신호 통계 추가
    signal_stats = []
    signal_stats.append("")
    
    for year_label, year_range in [("23년 (6~12월)", (2023, 6, 2023, 12)), 
                                     ("24년", (2024, 1, 2024, 12)),
                                     ("25년", (2025, 1, 2025, 12))]:
        mask = (chart_data['time'].dt.year >= year_range[0]) & (chart_data['time'].dt.year <= year_range[2])
        if year_range[0] == year_range[2]:
            mask = mask & (chart_data['time'].dt.month >= year_range[1]) & (chart_data['time'].dt.month <= year_range[3])
        subset = chart_data[mask]
        
        gc = (subset['골든 크로스'] == 1).sum()
        dc = (subset['데드 크로스'] == 1).sum()
        ob_buy = (subset['원비 매수'] == 1).sum()
        ob_sell = (subset['원비 매도'] == 1).sum()
        
        signal_stats.append(f"### {year_label}")
        signal_stats.append(f"| 신호 | 횟수 |")
        signal_stats.append(f"|------|------|")
        signal_stats.append(f"| 골든 크로스 | {gc} |")
        signal_stats.append(f"| 데드 크로스 | {dc} |")
        signal_stats.append(f"| 원비 매수 | {ob_buy} |")
        signal_stats.append(f"| 원비 매도 | {ob_sell} |")
        signal_stats.append("")
    
    report += "\n".join(signal_stats)
    
    # 실제 거래 상세 (처음 30건)
    report += "\n\n## 6. 실제 거래 내역 샘플 (L1/S1 첫진입, 처음 30건)\n\n"
    first_entries = actual_analysis[actual_analysis['signal'].isin(['L1', 'S1'])].head(30)
    report += "| # | 진입시간 | 방향 | 신호 | 진입가 | 청산가 | 청산신호 | 손익(USD) | 누적(USD) |\n"
    report += "|---|----------|------|------|--------|--------|----------|-----------|----------|\n"
    for _, r in first_entries.iterrows():
        report += f"| {r['trade_num']} | {r['entry_time'].strftime('%Y-%m-%d %H:%M')} | {r['side']} | {r['signal']} | {r['entry_price']:.3f} | {r['exit_price']:.3f} | {r['exit_signal']} | {r['pnl']:+.2f} | {r['cumulative_pnl']:,.2f} |\n"
    
    # 백테스팅 거래 상세 (처음 30건)
    if len(bt_trades) > 0:
        report += "\n\n## 7. 백테스팅 거래 내역 샘플 (처음 30건)\n\n"
        report += "| # | 진입시간 | 방향 | 모드 | 진입가 | 청산가 | 사유 | 손익(USD) | 누적(USD) |\n"
        report += "|---|----------|------|------|--------|--------|------|-----------|----------|\n"
        for idx, t in enumerate(bt_trades[:30]):
            mode_name = {1: "A정규", 2: "A예외", 3: "B모드"}.get(t['mode'], str(t['mode']))
            entry_t = t['entry_time'].strftime('%Y-%m-%d %H:%M') if hasattr(t['entry_time'], 'strftime') else str(t['entry_time'])
            report += f"| {idx+1} | {entry_t} | {t['side']} | {mode_name} | {t['entry_price']:.3f} | {t['exit_price']:.3f} | {t['reason']} | {t['pnl']:+.2f} | {t['cumulative_pnl']:,.2f} |\n"
    
    # 최종 요약
    report += "\n\n## 8. 최종 요약\n\n"
    report += "이 백테스팅은 완다차트에서 export한 23~25년 10분 데이터의 **원비 매수/매도**, **골든 크로스/데드 크로스**, **아시아 박스 상단/하단** 신호를 직접 사용하여\n"
    report += "전략의 A모드(정규/예외) → B모드 전이 로직을 Python으로 재현한 것입니다.\n\n"
    report += "**핵심 전략 규칙 (재확인)**:\n"
    report += "- 원비 카운트는 박스 안팎 무관하게 첫원비부터 순서대로 증가\n"
    report += "- 진입은 박스 밖(돌파 후)일 때만 실행\n"
    report += "- A모드: 정규 3번, 예외 2번 카운트 소진 시 B모드 전환\n"
    report += "- B모드: 박스 돌파 후 다음 봉부터 원비 감지 → 당일 1회 진입\n"
    
    with open('backtest_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ 리포트 저장 완료: backtest_report.md")
    print(f"\n{'='*60}")
    
    # 콘솔 요약 출력
    if len(bt_trades) > 0:
        bt_df = pd.DataFrame(bt_trades)
        print(f"\n📊 백테스팅 요약:")
        print(f"   총 거래: {len(bt_df)}")
        print(f"   승/패: {len(bt_df[bt_df['pnl']>0])}/{len(bt_df[bt_df['pnl']<0])}")
        print(f"   승률: {len(bt_df[bt_df['pnl']>0])/len(bt_df)*100:.1f}%")
        print(f"   최종 손익: ${bt_df['cumulative_pnl'].iloc[-1]:,.2f}")
    
    first_entries = actual_analysis[actual_analysis['signal'].isin(['L1', 'S1'])]
    print(f"\n📊 실제 거래 요약:")
    print(f"   첫진입 거래: {len(first_entries)}")
    print(f"   승/패: {len(first_entries[first_entries['pnl']>0])}/{len(first_entries[first_entries['pnl']<0])}")
    if len(first_entries) > 0:
        print(f"   승률: {len(first_entries[first_entries['pnl']>0])/len(first_entries)*100:.1f}%")
    print(f"   최종 누적 손익: ${actual_trades['누적 손익 USD'].iloc[-1]:,.2f}")
