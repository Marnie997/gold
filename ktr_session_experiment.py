#!/usr/bin/env python3
"""
세션별 KTR 실험 + 미장 전 포지션 관리 비교
- 아시아장 진입: MAX(전일 미장 KTR, 당일 아시아 KTR)
- 유로장 진입: MAX(전일 미장 KTR, 당일 아시아 KTR)
- 미장: 그대로
- 25년은 변경 없음, 23/24년만 적용
- 미장 전 포지션 정리 vs 유지 vs 수익시만 정리 비교
"""

import sys
sys.path.insert(0, '.')

from backtest import BacktestEngine, load_chart_data, is_us_dst, is_euro_dst
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class SessionKTREngine(BacktestEngine):
    """세션별 KTR 선택 + 미장 전 관리 엔진"""

    def __init__(self, data, apply_session_ktr_years=None,
                 pre_us_mode='keep'):
        """
        apply_session_ktr_years: set of years to apply session KTR logic (e.g. {2023, 2024})
        pre_us_mode: 'keep' (기존), 'close_all' (미장 전 전체 정리),
                     'close_profit' (미장 전 수익중이면 정리)
        """
        super().__init__(data)
        self.apply_session_ktr_years = apply_session_ktr_years or set()
        self.pre_us_mode = pre_us_mode

        # 세션 KTR 저장소
        self.session_ktrs = {}  # {session_date_key: {'asia': ktr, 'euro': ktr, 'us': ktr}}
        self.prev_day_us_ktr = None
        self.current_session_date = None

        # 세션 KTR 사전 계산
        self._precompute_session_ktrs()

    def _precompute_session_ktrs(self):
        """모든 세션의 KTR을 사전 계산"""
        data = self.data
        n = len(data)

        current_session_key = None
        current_session_type = None
        session_bars = []

        for i in range(n):
            row = data.iloc[i]
            dt = row['time']
            if pd.isna(dt):
                continue

            if hasattr(dt, 'to_pydatetime'):
                dt_py = dt.to_pydatetime()
            else:
                dt_py = dt

            us_dst = is_us_dst(dt_py)
            euro_dst = is_euro_dst(dt_py)
            asia_h = 7 if us_dst else 8
            euro_h = 16 if euro_dst else 17
            us_sub_h = 22 if us_dst else 23
            us_start_min = us_sub_h * 60 + 30
            cur_min = dt_py.hour * 60 + dt_py.minute

            # 세션 타입 결정
            if cur_min >= asia_h * 60 and cur_min < euro_h * 60:
                sess_type = 'asia'
            elif cur_min >= euro_h * 60 and cur_min < us_start_min:
                sess_type = 'euro'
            else:
                sess_type = 'us'

            # 세션 날짜 키
            if cur_min >= asia_h * 60:
                sess_date = dt_py.year * 10000 + dt_py.month * 100 + dt_py.day
            else:
                prev = dt_py - timedelta(days=1)
                sess_date = prev.year * 10000 + prev.month * 100 + prev.day

            compound_key = (sess_date, sess_type)

            if compound_key != (current_session_key, current_session_type):
                # 이전 세션 KTR 저장
                if current_session_key is not None and current_session_type is not None and len(session_bars) >= 2:
                    combined_high = max(b['high'] for b in session_bars[:2])
                    combined_low = min(b['low'] for b in session_bars[:2])
                    ktr = combined_high - combined_low
                    if ktr > 0:
                        if current_session_key not in self.session_ktrs:
                            self.session_ktrs[current_session_key] = {}
                        self.session_ktrs[current_session_key][current_session_type] = ktr

                current_session_key = sess_date
                current_session_type = sess_type
                session_bars = []

            session_bars.append({'high': row['high'], 'low': row['low']})

        # 마지막 세션
        if current_session_key is not None and current_session_type is not None and len(session_bars) >= 2:
            combined_high = max(b['high'] for b in session_bars[:2])
            combined_low = min(b['low'] for b in session_bars[:2])
            ktr = combined_high - combined_low
            if ktr > 0:
                if current_session_key not in self.session_ktrs:
                    self.session_ktrs[current_session_key] = {}
                self.session_ktrs[current_session_key][current_session_type] = ktr

        # 전일 미장 KTR 매핑 만들기
        self.prev_us_ktr_map = {}
        sorted_keys = sorted(self.session_ktrs.keys())
        for idx, key in enumerate(sorted_keys):
            if 'us' in self.session_ktrs[key]:
                # 이 날짜의 US KTR이 다음 날의 "전일 미장 KTR"이 됨
                # 다음 세션 날짜 찾기
                for next_idx in range(idx + 1, len(sorted_keys)):
                    next_key = sorted_keys[next_idx]
                    if next_key != key:
                        self.prev_us_ktr_map[next_key] = self.session_ktrs[key]['us']
                        break

    def _get_session_ktr(self, session_date_key, session_type, year):
        """세션별 KTR 선택 로직"""
        # 25년이거나 적용 대상이 아니면 원래 KTR 사용
        if year not in self.apply_session_ktr_years:
            return None  # None이면 원래 compute_ktr_from_data 결과 사용

        base_ktr = self.session_ktrs.get(session_date_key, {}).get(session_type)
        prev_us_ktr = self.prev_us_ktr_map.get(session_date_key)
        today_asia_ktr = self.session_ktrs.get(session_date_key, {}).get('asia')

        if session_type == 'asia':
            # 아시아: MAX(전일 미장 KTR, 당일 아시아 KTR)
            candidates = []
            if prev_us_ktr is not None and prev_us_ktr > 0:
                candidates.append(prev_us_ktr)
            if today_asia_ktr is not None and today_asia_ktr > 0:
                candidates.append(today_asia_ktr)
            if candidates:
                return max(candidates)
            return base_ktr

        elif session_type == 'euro':
            # 유로: MAX(전일 미장 KTR, 당일 아시아 KTR)
            candidates = []
            if prev_us_ktr is not None and prev_us_ktr > 0:
                candidates.append(prev_us_ktr)
            if today_asia_ktr is not None and today_asia_ktr > 0:
                candidates.append(today_asia_ktr)
            if candidates:
                return max(candidates)
            return base_ktr

        else:  # us
            # 미장: 그대로
            return base_ktr

    def run(self):
        """메인 백테스팅 루프 (세션 KTR + 미장 전 관리 포함)"""
        data = self.data
        n = len(data)

        prev_close = None

        for i in range(n):
            row = data.iloc[i]
            dt = row['time']
            if pd.isna(dt):
                continue

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

            golden_cross = row.get('골든 크로스', 0) == 1
            dead_cross = row.get('데드 크로스', 0) == 1
            buy_condition1 = row.get('원비 매수', 0) == 1
            sell_condition1 = row.get('원비 매도', 0) == 1

            box_top = row.get('아시아 박스 상단값', None)
            box_bottom = row.get('아시아 박스 하단값', None)

            if pd.notna(box_top) and pd.notna(box_bottom) and box_top != '' and box_bottom != '':
                try:
                    self.today_box_top = float(box_top)
                    self.today_box_bottom = float(box_bottom)
                    self.box_ready = True
                except (ValueError, TypeError):
                    pass

            sma20 = row.get('20 SMA', None)
            sma120 = row.get('120 SMA', None)

            us_dst = is_us_dst(dt_py) if hasattr(dt_py, 'year') else False
            euro_dst = is_euro_dst(dt_py) if hasattr(dt_py, 'year') else False

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

                if pd.notna(box_top) and pd.notna(box_bottom):
                    try:
                        self.today_box_top = float(box_top)
                        self.today_box_bottom = float(box_bottom)
                        self.box_ready = True
                    except (ValueError, TypeError):
                        pass

            # 세션 타입
            us_sub_h = 22 if us_dst else 23
            euro_h = 16 if euro_dst else 17
            us_start_min = us_sub_h * 60 + 30
            cur_min = dt_py.hour * 60 + dt_py.minute

            if cur_min >= asia_h * 60 and cur_min < euro_h * 60:
                session_type = 1  # asia
                session_name = 'asia'
            elif cur_min >= euro_h * 60 and cur_min < us_start_min:
                session_type = 2  # euro
                session_name = 'euro'
            else:
                session_type = 3  # us
                session_name = 'us'

            # ★ 미장 전 포지션 관리 ★
            pre_us_window_start = us_start_min - 60  # 미장 1시간 전
            year = dt_py.year

            if (self.pre_us_mode != 'keep' and
                self.position_size != 0 and
                session_type != 3 and
                cur_min >= pre_us_window_start and cur_min < us_start_min):

                should_close = False
                if self.pre_us_mode == 'close_all':
                    should_close = True
                elif self.pre_us_mode == 'close_profit':
                    # 수익중이면 정리
                    if self.active_side == "LONG":
                        unrealized = (close_p - self.position_avg_price) * abs(self.position_size) * self.point_value
                    else:
                        unrealized = (self.position_avg_price - close_p) * abs(self.position_size) * self.point_value
                    should_close = unrealized > 0

                if should_close:
                    self.close_position(close_p, i, row['time'], "미장전정리")
                    self.just_closed_bar = i

                    # 상태 초기화
                    self.cycle_state = 0
                    self.defense_mode = False
                    self.force_close_issued = False

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

                    self.equity_curve.append({
                        'time': row['time'],
                        'equity': self.initial_capital + self.cumulative_pnl
                    })
                    prev_close = close_p
                    continue

            # 무포지션 진입금지 시간
            block_start = (us_sub_h - 1) * 60
            block_end = block_start + 120
            in_flat_entry_block = block_start <= cur_min < block_end
            is_flat = self.position_size == 0 and self.open_trades == 0
            can_enter = not (is_flat and in_flat_entry_block)

            # ★ 세션별 KTR 선택 ★
            base_ktr = self.compute_ktr_from_data(i, session_type)
            session_ktr = self._get_session_ktr(session_key, session_name, year)

            if session_ktr is not None and year in self.apply_session_ktr_years:
                self.active_ktr = session_ktr
            else:
                self.active_ktr = base_ktr

            ktr_ready = self.active_ktr is not None and self.active_ktr > 0

            # 1시간 역추세 필터
            htf_trend_dir = 0
            if pd.notna(sma20) and pd.notna(sma120):
                if sma20 > sma120:
                    htf_trend_dir = 1
                elif sma20 < sma120:
                    htf_trend_dir = -1

            equity = self.initial_capital + self.cumulative_pnl
            stepped_capital = self.get_stepped_capital(equity)

            # ── 거래중 확인 ──
            if self.position_size != 0:
                if self.cycle_state == 3 and dead_cross and not self.defense_mode:
                    self.defense_mode = True
                    self.reverse_arm = -1
                    self.reverse_cross_bar = i
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
                    self.check_fills_and_exits(i, row)

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

                            if (self.setup_count <= allowed_count and
                                self.is_above_today_box(close_p) and
                                ktr_ready and can_enter):

                                self.setup_ktr = self.active_ktr

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
                                    self.cycle_state = 3

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
# 실험 실행
# =====================================================
def yearly_stats(df, label, print_header=True):
    """연도별 통계 출력"""
    if len(df) == 0:
        print(f"  ── {label}: 거래 없음 ──")
        return

    df = df.copy()
    df['year'] = pd.to_datetime(df['entry_time']).dt.year

    if print_header:
        print(f"\n  ── {label} ──")
        header = f"  {'연도':>6} | {'거래수':>6} | {'승':>4} | {'패':>4} | {'승률':>7} | {'총손익':>14} | {'평균승':>10} | {'평균패':>10} | {'손익비':>6}"
        print(header)
        print("  " + "-" * 90)

    for year in sorted(df['year'].unique()):
        yr = df[df['year'] == year]
        wins = yr[yr['pnl'] > 0]
        losses = yr[yr['pnl'] < 0]
        wr = len(wins)/(len(wins)+len(losses))*100 if (len(wins)+len(losses)) > 0 else 0
        avg_w = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_l = losses['pnl'].mean() if len(losses) > 0 else 0
        rr = avg_w / abs(avg_l) if avg_l != 0 else 0
        total = yr['pnl'].sum()
        print(f"  {year:>6} | {len(yr):>6} | {len(wins):>4} | {len(losses):>4} | {wr:>6.1f}% | ${total:>12,.2f} | ${avg_w:>8,.2f} | ${avg_l:>8,.2f} | {rr:>5.2f}")

    total_all = df['pnl'].sum()
    wins_all = df[df['pnl'] > 0]
    losses_all = df[df['pnl'] < 0]
    wr_all = len(wins_all)/(len(wins_all)+len(losses_all))*100 if (len(wins_all)+len(losses_all)) > 0 else 0
    print(f"  {'합계':>6} | {len(df):>6} | {len(wins_all):>4} | {len(losses_all):>4} | {wr_all:>6.1f}% | ${total_all:>12,.2f}")


if __name__ == "__main__":
    print("=" * 70)
    print("🔬 세션별 KTR 실험 + 미장 전 포지션 관리 비교")
    print("=" * 70)

    print("\n[1] 데이터 로딩...")
    chart_data = load_chart_data()
    print(f"    총 {len(chart_data)} 봉")

    # ── 원본 (기준) ──
    print("\n[2] 원본 백테스팅 (기존 KTR, 미장 전 유지)...")
    engine_orig = BacktestEngine(chart_data)
    trades_orig = engine_orig.run()
    df_orig = pd.DataFrame(trades_orig)
    print(f"    거래 수: {len(df_orig)}")

    # ── 변형 A: 세션 KTR + 미장 전 유지 ──
    print("\n[3] 세션 KTR (23/24) + 미장 전 유지...")
    engine_a = SessionKTREngine(chart_data,
                                apply_session_ktr_years={2023, 2024},
                                pre_us_mode='keep')
    trades_a = engine_a.run()
    df_a = pd.DataFrame(trades_a)
    print(f"    거래 수: {len(df_a)}")

    # ── 변형 B: 세션 KTR + 미장 전 전체 정리 ──
    print("\n[4] 세션 KTR (23/24) + 미장 전 전체 정리...")
    engine_b = SessionKTREngine(chart_data,
                                apply_session_ktr_years={2023, 2024},
                                pre_us_mode='close_all')
    trades_b = engine_b.run()
    df_b = pd.DataFrame(trades_b)
    print(f"    거래 수: {len(df_b)}")

    # ── 변형 C: 세션 KTR + 미장 전 수익시만 정리 ──
    print("\n[5] 세션 KTR (23/24) + 미장 전 수익시만 정리...")
    engine_c = SessionKTREngine(chart_data,
                                apply_session_ktr_years={2023, 2024},
                                pre_us_mode='close_profit')
    trades_c = engine_c.run()
    df_c = pd.DataFrame(trades_c)
    print(f"    거래 수: {len(df_c)}")

    # ── 결과 비교 ──
    print("\n" + "=" * 90)
    print("📊 결과 비교: 4개 변형")
    print("=" * 90)

    variants = [
        ("① 원본 (기존)", df_orig),
        ("② 세션KTR + 유지", df_a),
        ("③ 세션KTR + 전체정리", df_b),
        ("④ 세션KTR + 수익정리", df_c),
    ]

    for label, df in variants:
        yearly_stats(df, label)

    # ── 연도별 변화 ──
    print("\n\n" + "=" * 90)
    print("📊 연도별 손익 변화 비교 (원본 대비)")
    print("=" * 90)

    for year in [2023, 2024, 2025]:
        print(f"\n  ── {year}년 ──")
        print(f"  {'변형':>25} | {'거래수':>6} | {'승률':>7} | {'총손익':>14} | {'vs 원본':>14}")
        print("  " + "-" * 80)

        for label, df in variants:
            df_yr = df.copy()
            df_yr['year'] = pd.to_datetime(df_yr['entry_time']).dt.year
            yr = df_yr[df_yr['year'] == year]

            orig_yr = df_orig.copy()
            orig_yr['year'] = pd.to_datetime(orig_yr['entry_time']).dt.year
            orig_yr = orig_yr[orig_yr['year'] == year]

            pnl = yr['pnl'].sum() if len(yr) > 0 else 0
            orig_pnl = orig_yr['pnl'].sum() if len(orig_yr) > 0 else 0
            diff = pnl - orig_pnl

            wins = len(yr[yr['pnl'] > 0]) if len(yr) > 0 else 0
            losses = len(yr[yr['pnl'] < 0]) if len(yr) > 0 else 0
            wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

            print(f"  {label:>25} | {len(yr):>6} | {wr:>6.1f}% | ${pnl:>12,.2f} | ${diff:>12,.2f}")

    # ── 세션별 KTR 통계 ──
    print("\n\n" + "=" * 90)
    print("📊 세션 KTR 통계 (사전 계산)")
    print("=" * 90)

    for year in [2023, 2024, 2025]:
        asia_ktrs = []
        euro_ktrs = []
        us_ktrs = []

        for key, sessions in engine_a.session_ktrs.items():
            key_year = key // 10000
            if key_year == year:
                if 'asia' in sessions:
                    asia_ktrs.append(sessions['asia'])
                if 'euro' in sessions:
                    euro_ktrs.append(sessions['euro'])
                if 'us' in sessions:
                    us_ktrs.append(sessions['us'])

        print(f"\n  {year}년:")
        if asia_ktrs:
            print(f"    아시아 KTR: 평균 {np.mean(asia_ktrs):.3f}, 중간값 {np.median(asia_ktrs):.3f}, 최대 {np.max(asia_ktrs):.3f}")
        if euro_ktrs:
            print(f"    유로   KTR: 평균 {np.mean(euro_ktrs):.3f}, 중간값 {np.median(euro_ktrs):.3f}, 최대 {np.max(euro_ktrs):.3f}")
        if us_ktrs:
            print(f"    미장   KTR: 평균 {np.mean(us_ktrs):.3f}, 중간값 {np.median(us_ktrs):.3f}, 최대 {np.max(us_ktrs):.3f}")

    # ── 청산 사유 비교 ──
    print("\n\n" + "=" * 90)
    print("📊 청산 사유별 비교 (23/24년)")
    print("=" * 90)

    for label, df in variants:
        print(f"\n  ── {label} ──")
        df_tmp = df.copy()
        df_tmp['year'] = pd.to_datetime(df_tmp['entry_time']).dt.year
        for year in [2023, 2024]:
            yr = df_tmp[df_tmp['year'] == year]
            print(f"    {year}년:")
            for reason in ['익절', '손절', '반대신호정리', '미장전정리', '기간종료정리']:
                rd = yr[yr['reason'] == reason]
                if len(rd) > 0:
                    print(f"      {reason}: {len(rd)}건 | 합산 ${rd['pnl'].sum():,.2f} | 평균 ${rd['pnl'].mean():,.2f}")

    # ── 모드별 비교 ──
    print("\n\n" + "=" * 90)
    print("📊 모드별 비교 (23/24년)")
    print("=" * 90)

    mode_names = {1: "A정규", 2: "A예외", 3: "B모드"}
    for label, df in variants:
        print(f"\n  ── {label} ──")
        df_tmp = df.copy()
        df_tmp['year'] = pd.to_datetime(df_tmp['entry_time']).dt.year
        for year in [2023, 2024]:
            yr = df_tmp[df_tmp['year'] == year]
            print(f"    {year}년:")
            for mode_id, mode_name in mode_names.items():
                md = yr[yr['mode'] == mode_id]
                if len(md) > 0:
                    wins = len(md[md['pnl'] > 0])
                    losses = len(md[md['pnl'] < 0])
                    wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
                    print(f"      {mode_name}: {len(md)}건 | 승률 {wr:.1f}% | 손익 ${md['pnl'].sum():,.2f}")

    print("\n\n✅ 실험 완료")
