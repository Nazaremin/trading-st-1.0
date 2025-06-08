"""
Криптовалютная торговая стратегия
Автор: Manus AI (Алгопрограммист)

Специализированная версия стратегии для криптовалютного рынка с учетом:
- Высокой волатильности
- 24/7 торговли
- Отсутствия сессионности
- Влияния новостей и социальных настроений
- Быстрых движений и гэпов
"""

from trading_system_base import *
from multi_timeframe_analysis import *
from risk_position_management import *
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import math

class CryptoVolatilityManager:
    """Менеджер волатильности для криптовалют"""
    
    def __init__(self):
        self.volatility_thresholds = {
            'low': 0.02,      # 2% за час
            'normal': 0.05,   # 5% за час
            'high': 0.10,     # 10% за час
            'extreme': 0.20   # 20% за час
        }
        
        self.volatility_adjustments = {
            'low': {'risk_multiplier': 1.2, 'position_multiplier': 1.1},
            'normal': {'risk_multiplier': 1.0, 'position_multiplier': 1.0},
            'high': {'risk_multiplier': 0.7, 'position_multiplier': 0.8},
            'extreme': {'risk_multiplier': 0.4, 'position_multiplier': 0.5}
        }
    
    def calculate_hourly_volatility(self, data: pd.DataFrame) -> float:
        """Расчет часовой волатильности"""
        if len(data) < 2:
            return 0.0
        
        # Расчет процентных изменений за последний час
        recent_data = data.tail(12)  # Последние 12 5-минутных свечей = 1 час
        if len(recent_data) < 2:
            return 0.0
        
        price_changes = recent_data['close'].pct_change().dropna()
        hourly_volatility = price_changes.std() * np.sqrt(12)  # Аннуализация до часа
        
        return hourly_volatility
    
    def classify_volatility(self, volatility: float) -> str:
        """Классификация уровня волатильности"""
        if volatility >= self.volatility_thresholds['extreme']:
            return 'extreme'
        elif volatility >= self.volatility_thresholds['high']:
            return 'high'
        elif volatility >= self.volatility_thresholds['normal']:
            return 'normal'
        else:
            return 'low'
    
    def get_volatility_adjustments(self, volatility_level: str) -> Dict:
        """Получение корректировок для уровня волатильности"""
        return self.volatility_adjustments.get(volatility_level, 
                                             self.volatility_adjustments['normal'])

class CryptoNewsAnalyzer:
    """Анализатор влияния новостей на криптовалюты"""
    
    def __init__(self):
        # Ключевые события, влияющие на криптовалюты
        self.high_impact_events = [
            'fed_meeting', 'inflation_data', 'regulatory_news',
            'major_adoption', 'security_breach', 'whale_movement'
        ]
        
        # Временные окна влияния событий
        self.event_impact_windows = {
            'fed_meeting': timedelta(hours=6),
            'inflation_data': timedelta(hours=4),
            'regulatory_news': timedelta(hours=12),
            'major_adoption': timedelta(hours=8),
            'security_breach': timedelta(hours=24),
            'whale_movement': timedelta(hours=2)
        }
    
    def is_high_impact_period(self, current_time: datetime = None) -> Tuple[bool, str]:
        """Проверка периода высокого влияния новостей"""
        if current_time is None:
            current_time = datetime.now()
        
        # В реальной системе здесь был бы API для получения новостей
        # Для демонстрации используем упрощенную логику
        
        # Проверка выходных (обычно меньше новостей)
        if current_time.weekday() >= 5:  # Суббота, воскресенье
            return False, "weekend"
        
        # Проверка времени основных новостей (UTC)
        hour = current_time.hour
        
        # Американские новости (13:30-15:30 UTC)
        if 13 <= hour <= 15:
            return True, "us_news_time"
        
        # Азиатские новости (00:00-02:00 UTC)
        elif 0 <= hour <= 2:
            return True, "asia_news_time"
        
        # Европейские новости (08:00-10:00 UTC)
        elif 8 <= hour <= 10:
            return True, "europe_news_time"
        
        return False, "normal"
    
    def calculate_news_risk_factor(self, current_time: datetime = None) -> float:
        """Расчет фактора риска новостей (0-1)"""
        is_high_impact, reason = self.is_high_impact_period(current_time)
        
        if is_high_impact:
            if reason == "us_news_time":
                return 0.8  # Высокий риск
            elif reason in ["asia_news_time", "europe_news_time"]:
                return 0.6  # Средний риск
        
        return 0.2  # Низкий риск

class CryptoMomentumDetector:
    """Детектор моментума для криптовалют"""
    
    def __init__(self):
        self.momentum_periods = [5, 15, 30, 60]  # Периоды в минутах
        
    def detect_momentum_breakout(self, data: pd.DataFrame, min_strength: float = 0.7) -> Optional[Dict]:
        """Детекция прорыва моментума"""
        if len(data) < 60:  # Минимум 5 часов данных
            return None
        
        recent_data = data.tail(60)  # Последние 5 часов
        
        # Расчет моментума на разных периодах
        momentum_scores = {}
        
        for period in self.momentum_periods:
            if len(recent_data) >= period:
                period_data = recent_data.tail(period)
                
                # Расчет направленного движения
                price_change = (period_data['close'].iloc[-1] - period_data['close'].iloc[0]) / period_data['close'].iloc[0]
                
                # Расчет консистентности движения
                price_changes = period_data['close'].pct_change().dropna()
                positive_moves = sum(1 for change in price_changes if change > 0)
                consistency = positive_moves / len(price_changes) if len(price_changes) > 0 else 0
                
                # Расчет силы движения (объем)
                if 'volume' in period_data.columns:
                    avg_volume = period_data['volume'].mean()
                    recent_volume = period_data['volume'].tail(5).mean()
                    volume_factor = recent_volume / avg_volume if avg_volume > 0 else 1
                else:
                    volume_factor = 1
                
                # Общий скор моментума
                momentum_score = abs(price_change) * consistency * min(volume_factor, 2.0)
                momentum_scores[period] = {
                    'score': momentum_score,
                    'direction': 'bullish' if price_change > 0 else 'bearish',
                    'strength': momentum_score
                }
        
        # Поиск согласованного моментума
        if len(momentum_scores) >= 3:
            # Проверка согласованности направления
            directions = [score['direction'] for score in momentum_scores.values()]
            if len(set(directions)) == 1:  # Все в одном направлении
                avg_strength = np.mean([score['strength'] for score in momentum_scores.values()])
                
                if avg_strength >= min_strength:
                    return {
                        'type': 'momentum_breakout',
                        'direction': directions[0],
                        'strength': avg_strength,
                        'timeframes': list(momentum_scores.keys()),
                        'confidence': min(avg_strength, 1.0)
                    }
        
        return None
    
    def detect_momentum_exhaustion(self, data: pd.DataFrame) -> Optional[Dict]:
        """Детекция истощения моментума"""
        if len(data) < 30:
            return None
        
        recent_data = data.tail(30)  # Последние 2.5 часа
        
        # Анализ дивергенции цены и объема
        price_trend = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
        
        if 'volume' in recent_data.columns:
            # Сравнение объема в первой и второй половине периода
            first_half_volume = recent_data.head(15)['volume'].mean()
            second_half_volume = recent_data.tail(15)['volume'].mean()
            volume_trend = (second_half_volume - first_half_volume) / first_half_volume if first_half_volume > 0 else 0
            
            # Дивергенция: цена растет, но объем падает (или наоборот)
            if (price_trend > 0.02 and volume_trend < -0.2) or (price_trend < -0.02 and volume_trend < -0.2):
                return {
                    'type': 'momentum_exhaustion',
                    'direction': 'bearish' if price_trend > 0 else 'bullish',
                    'price_trend': price_trend,
                    'volume_trend': volume_trend,
                    'confidence': min(abs(price_trend) + abs(volume_trend), 1.0)
                }
        
        return None

class CryptoStrategy(BaseStrategy):
    """Криптовалютная торговая стратегия"""
    
    def __init__(self):
        super().__init__(MarketConfig(MarketType.CRYPTO))
        self.volatility_manager = CryptoVolatilityManager()
        self.news_analyzer = CryptoNewsAnalyzer()
        self.momentum_detector = CryptoMomentumDetector()
        self.mtf_analyzer = MultiTimeframeAnalyzer(self.config)
        self.pattern_detector = PatternDetector(self.config)
        self.synchronizer = TimeframeSynchronizer(self.config)
        self.position_manager = PositionManager(self.config)
        
        # Крипто-специфичные параметры
        self.max_daily_trades = 8  # Максимум сделок в день
        self.gap_threshold = 0.03  # 3% гэп для особого внимания
        self.whale_movement_threshold = 0.05  # 5% движение за 15 минут
        self.daily_trade_count = 0
        self.last_trade_date = datetime.now().date()
        
    def analyze_market(self, data: Dict[TimeFrame, pd.DataFrame], symbol: str) -> Optional[Signal]:
        """Анализ рынка для криптовалют"""
        
        # Проверка дневного лимита сделок
        if not self._check_daily_trade_limit():
            return None
        
        # Анализ волатильности
        execution_data = data.get(self.config.timeframes['execution'])
        if execution_data is None or len(execution_data) < 10:
            return None
        
        current_volatility = self.volatility_manager.calculate_hourly_volatility(execution_data)
        volatility_level = self.volatility_manager.classify_volatility(current_volatility)
        
        # Избегаем торговли при экстремальной волатильности
        if volatility_level == 'extreme':
            self.logger.warning(f"Экстремальная волатильность: {current_volatility:.4f}")
            return None
        
        # Анализ новостного фона
        news_risk = self.news_analyzer.calculate_news_risk_factor()
        if news_risk > 0.7:
            self.logger.info("Высокий новостной риск, снижение активности")
        
        # Детекция гэпов и резких движений
        gap_analysis = self._analyze_gaps_and_spikes(execution_data)
        if gap_analysis['avoid_trading']:
            return None
        
        # Многотаймфреймный анализ
        timeframe_analyses = {}
        for tf_name, tf_enum in self.config.timeframes.items():
            if tf_enum in data:
                timeframe_analyses[tf_name] = self.mtf_analyzer.analyze_timeframe(data[tf_enum], tf_enum)
        
        if len(timeframe_analyses) < 3:
            return None
        
        # Синхронизация анализа
        sync_result = self.synchronizer.synchronize_analysis(timeframe_analyses)
        
        # Детекция моментума (приоритет для криптовалют)
        momentum_breakout = self.momentum_detector.detect_momentum_breakout(execution_data)
        momentum_exhaustion = self.momentum_detector.detect_momentum_exhaustion(execution_data)
        
        # Приоритет моментум-сигналам
        if momentum_breakout and momentum_breakout['confidence'] > 0.7:
            if sync_result.get('synchronized', False):
                return self._create_momentum_signal(
                    momentum_breakout, timeframe_analyses, volatility_level, news_risk, symbol
                )
        
        # Сигналы истощения моментума (разворотные)
        if momentum_exhaustion and momentum_exhaustion['confidence'] > 0.6:
            return self._create_reversal_signal(
                momentum_exhaustion, timeframe_analyses, volatility_level, news_risk, symbol
            )
        
        # Стандартный анализ паттернов AMD
        if sync_result.get('synchronized', False):
            signal_tf_data = data[self.config.timeframes['signal']]
            signal_analysis = timeframe_analyses['signal']
            
            amd_pattern = self.pattern_detector.detect_amd_pattern(signal_tf_data, signal_analysis)
            
            if amd_pattern and amd_pattern['confidence'] > 0.6:  # Ниже порог для крипто
                return self._create_crypto_amd_signal(
                    amd_pattern, timeframe_analyses, volatility_level, news_risk, symbol
                )
        
        return None
    
    def _check_daily_trade_limit(self) -> bool:
        """Проверка дневного лимита сделок"""
        current_date = datetime.now().date()
        
        if current_date != self.last_trade_date:
            self.daily_trade_count = 0
            self.last_trade_date = current_date
        
        return self.daily_trade_count < self.max_daily_trades
    
    def _analyze_gaps_and_spikes(self, data: pd.DataFrame) -> Dict:
        """Анализ гэпов и резких движений"""
        if len(data) < 5:
            return {'avoid_trading': False, 'gap_detected': False}
        
        recent_data = data.tail(5)
        
        # Проверка резких движений за последние 15 минут (3 свечи по 5 минут)
        price_changes = recent_data['close'].pct_change().abs()
        max_change = price_changes.max()
        
        # Проверка гэпов между свечами
        gaps = []
        for i in range(1, len(recent_data)):
            prev_close = recent_data['close'].iloc[i-1]
            current_open = recent_data['open'].iloc[i]
            gap = abs(current_open - prev_close) / prev_close
            gaps.append(gap)
        
        max_gap = max(gaps) if gaps else 0
        
        # Определение необходимости избегать торговли
        avoid_trading = (max_change > self.whale_movement_threshold or 
                        max_gap > self.gap_threshold)
        
        return {
            'avoid_trading': avoid_trading,
            'gap_detected': max_gap > self.gap_threshold,
            'spike_detected': max_change > self.whale_movement_threshold,
            'max_gap': max_gap,
            'max_change': max_change
        }
    
    def _create_momentum_signal(self, momentum: Dict, timeframe_analyses: Dict,
                              volatility_level: str, news_risk: float, symbol: str) -> Signal:
        """Создание сигнала на основе моментума"""
        
        direction = momentum['direction']
        signal_analysis = timeframe_analyses['signal']
        
        # Получение текущей цены (упрощенно)
        current_price = 50000.0  # Заглушка
        
        # Расчет входа с учетом волатильности
        volatility_adjustments = self.volatility_manager.get_volatility_adjustments(volatility_level)
        
        # Более агрессивные стопы для криптовалют
        atr_value = current_price * 0.02  # 2% от цены как примерное ATR
        stop_multiplier = self.config.stop_loss_atr_multiplier * volatility_adjustments['risk_multiplier']
        
        if direction == 'bullish':
            entry_price = current_price * 1.001  # Небольшая премия для входа
            stop_loss = entry_price - (atr_value * stop_multiplier)
            take_profits = [
                entry_price + (atr_value * multiplier * volatility_adjustments['position_multiplier'])
                for multiplier in self.config.take_profit_levels
            ]
        else:
            entry_price = current_price * 0.999  # Небольшая скидка для входа
            stop_loss = entry_price + (atr_value * stop_multiplier)
            take_profits = [
                entry_price - (atr_value * multiplier * volatility_adjustments['position_multiplier'])
                for multiplier in self.config.take_profit_levels
            ]
        
        # Расчет соотношения риск/прибыль
        risk_distance = abs(entry_price - stop_loss)
        reward_distance = abs(take_profits[0] - entry_price) if take_profits else 0
        rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
        
        # Корректировка уверенности с учетом крипто-факторов
        base_confidence = momentum['confidence']
        volatility_penalty = 0.1 if volatility_level == 'high' else 0
        news_penalty = news_risk * 0.2
        
        total_confidence = max(base_confidence - volatility_penalty - news_penalty, 0.3)
        
        return Signal(
            timestamp=datetime.now(),
            signal_type=SignalType.BUY if direction == 'bullish' else SignalType.SELL,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profits,
            confidence=total_confidence,
            timeframe_analysis={
                self.config.timeframes['context']: f"Momentum: {momentum['strength']:.2f}",
                self.config.timeframes['signal']: f"Direction: {direction}",
                self.config.timeframes['execution']: f"Volatility: {volatility_level}"
            },
            risk_reward_ratio=rr_ratio
        )
    
    def _create_reversal_signal(self, exhaustion: Dict, timeframe_analyses: Dict,
                              volatility_level: str, news_risk: float, symbol: str) -> Signal:
        """Создание разворотного сигнала"""
        
        direction = exhaustion['direction']  # Направление ожидаемого разворота
        
        # Более консервативные параметры для разворотных сигналов
        current_price = 50000.0  # Заглушка
        atr_value = current_price * 0.015  # 1.5% от цены
        
        volatility_adjustments = self.volatility_manager.get_volatility_adjustments(volatility_level)
        stop_multiplier = self.config.stop_loss_atr_multiplier * volatility_adjustments['risk_multiplier'] * 1.5
        
        if direction == 'bullish':
            entry_price = current_price * 0.998  # Более консервативный вход
            stop_loss = entry_price - (atr_value * stop_multiplier)
            take_profits = [
                entry_price + (atr_value * multiplier * 0.8)  # Более скромные цели
                for multiplier in self.config.take_profit_levels[:2]  # Только первые 2 уровня
            ]
        else:
            entry_price = current_price * 1.002
            stop_loss = entry_price + (atr_value * stop_multiplier)
            take_profits = [
                entry_price - (atr_value * multiplier * 0.8)
                for multiplier in self.config.take_profit_levels[:2]
            ]
        
        # Расчет соотношения риск/прибыль
        risk_distance = abs(entry_price - stop_loss)
        reward_distance = abs(take_profits[0] - entry_price) if take_profits else 0
        rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
        
        # Более низкая уверенность для разворотных сигналов
        base_confidence = exhaustion['confidence'] * 0.8
        volatility_penalty = 0.15 if volatility_level == 'high' else 0.05
        news_penalty = news_risk * 0.3
        
        total_confidence = max(base_confidence - volatility_penalty - news_penalty, 0.2)
        
        return Signal(
            timestamp=datetime.now(),
            signal_type=SignalType.BUY if direction == 'bullish' else SignalType.SELL,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profits,
            confidence=total_confidence,
            timeframe_analysis={
                self.config.timeframes['context']: f"Exhaustion: {exhaustion['type']}",
                self.config.timeframes['signal']: f"Reversal: {direction}",
                self.config.timeframes['execution']: f"Confidence: {exhaustion['confidence']:.2f}"
            },
            risk_reward_ratio=rr_ratio
        )
    
    def _create_crypto_amd_signal(self, amd_pattern: Dict, timeframe_analyses: Dict,
                                volatility_level: str, news_risk: float, symbol: str) -> Signal:
        """Создание AMD сигнала для криптовалют"""
        
        direction = amd_pattern['distribution_direction']
        entry_info = amd_pattern['entry_zone']
        
        # Корректировка с учетом крипто-специфики
        volatility_adjustments = self.volatility_manager.get_volatility_adjustments(volatility_level)
        
        # Более широкие стопы для криптовалют
        risk_distance = abs(entry_info['entry_price'] - entry_info['stop_loss'])
        adjusted_stop_distance = risk_distance * volatility_adjustments['risk_multiplier']
        
        if direction == 'bullish':
            stop_loss = entry_info['entry_price'] - adjusted_stop_distance
            take_profits = [
                entry_info['entry_price'] + (adjusted_stop_distance * multiplier)
                for multiplier in self.config.take_profit_levels
            ]
        else:
            stop_loss = entry_info['entry_price'] + adjusted_stop_distance
            take_profits = [
                entry_info['entry_price'] - (adjusted_stop_distance * multiplier)
                for multiplier in self.config.take_profit_levels
            ]
        
        # Расчет соотношения риск/прибыль
        new_risk_distance = abs(entry_info['entry_price'] - stop_loss)
        reward_distance = abs(take_profits[0] - entry_info['entry_price']) if take_profits else 0
        rr_ratio = reward_distance / new_risk_distance if new_risk_distance > 0 else 0
        
        # Корректировка уверенности
        base_confidence = amd_pattern['confidence']
        crypto_bonus = 0.1 if volatility_level == 'normal' else 0
        volatility_penalty = 0.2 if volatility_level == 'high' else 0
        news_penalty = news_risk * 0.15
        
        total_confidence = max(base_confidence + crypto_bonus - volatility_penalty - news_penalty, 0.3)
        
        return Signal(
            timestamp=datetime.now(),
            signal_type=SignalType.BUY if direction == 'bullish' else SignalType.SELL,
            entry_price=entry_info['entry_price'],
            stop_loss=stop_loss,
            take_profit=take_profits,
            confidence=total_confidence,
            timeframe_analysis={
                self.config.timeframes['context']: f"AMD: {amd_pattern['pattern']}",
                self.config.timeframes['signal']: f"Direction: {direction}",
                self.config.timeframes['execution']: f"Vol: {volatility_level}"
            },
            risk_reward_ratio=rr_ratio
        )
    
    def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """Крипто-специфичный расчет размера позиции"""
        
        # Анализ текущей волатильности
        current_volatility = 0.05  # Заглушка - в реальности получаем из данных
        volatility_level = self.volatility_manager.classify_volatility(current_volatility)
        volatility_adjustments = self.volatility_manager.get_volatility_adjustments(volatility_level)
        
        # Базовый расчет с учетом новостного риска
        news_risk = self.news_analyzer.calculate_news_risk_factor()
        adjusted_risk_per_trade = self.config.risk_per_trade * (1 - news_risk * 0.5)
        
        # Расчет размера позиции
        risk_amount = account_balance * adjusted_risk_per_trade
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        
        if stop_distance == 0:
            return 0
        
        base_position_size = risk_amount / stop_distance
        
        # Корректировка с учетом волатильности
        adjusted_position_size = base_position_size * volatility_adjustments['position_multiplier']
        
        # Дополнительные ограничения для криптовалют
        max_position_value = account_balance * 0.15  # Максимум 15% баланса в одной позиции
        max_position_size = max_position_value / signal.entry_price
        
        return min(adjusted_position_size, max_position_size)
    
    def should_close_position_early(self, position: Position, current_data: pd.DataFrame) -> Tuple[bool, str]:
        """Проверка досрочного закрытия позиции для криптовалют"""
        
        # Анализ резких изменений волатильности
        current_volatility = self.volatility_manager.calculate_hourly_volatility(current_data)
        volatility_level = self.volatility_manager.classify_volatility(current_volatility)
        
        if volatility_level == 'extreme':
            return True, "extreme_volatility"
        
        # Анализ новостного фона
        news_risk = self.news_analyzer.calculate_news_risk_factor()
        if news_risk > 0.8:
            return True, "high_news_risk"
        
        # Анализ истощения моментума
        momentum_exhaustion = self.momentum_detector.detect_momentum_exhaustion(current_data)
        if momentum_exhaustion and momentum_exhaustion['confidence'] > 0.7:
            # Если истощение в противоположном направлении нашей позиции
            if ((position.side == 'buy' and momentum_exhaustion['direction'] == 'bearish') or
                (position.side == 'sell' and momentum_exhaustion['direction'] == 'bullish')):
                return True, "momentum_exhaustion"
        
        return False, ""
    
    def get_crypto_market_state(self) -> Dict:
        """Получение состояния криптовалютного рынка"""
        
        current_time = datetime.now()
        news_risk = self.news_analyzer.calculate_news_risk_factor(current_time)
        is_high_impact, impact_reason = self.news_analyzer.is_high_impact_period(current_time)
        
        return {
            'timestamp': current_time,
            'news_risk_factor': news_risk,
            'high_impact_period': is_high_impact,
            'impact_reason': impact_reason,
            'daily_trades_used': self.daily_trade_count,
            'daily_trades_remaining': self.max_daily_trades - self.daily_trade_count,
            'market_hours': '24/7',
            'recommended_action': self._get_market_recommendation(news_risk, is_high_impact)
        }
    
    def _get_market_recommendation(self, news_risk: float, is_high_impact: bool) -> str:
        """Получение рекомендации по рынку"""
        
        if news_risk > 0.8:
            return "avoid_trading"
        elif news_risk > 0.6:
            return "reduce_position_sizes"
        elif is_high_impact:
            return "monitor_closely"
        else:
            return "normal_trading"
    
    def execute_trade(self, signal: Signal, account_balance: float, symbol: str) -> Optional[Position]:
        """Исполнение сделки с учетом крипто-специфики"""
        
        # Проверка дневного лимита
        if not self._check_daily_trade_limit():
            self.logger.warning("Достигнут дневной лимит сделок")
            return None
        
        # Расчет размера позиции
        position_size = self.calculate_position_size(signal, account_balance)
        
        if position_size <= 0:
            return None
        
        # Создание позиции
        position = self.position_manager.open_position(
            signal, account_balance, signal.entry_price, 
            signal.entry_price * 0.02, symbol  # Примерное ATR
        )
        
        if position:
            self.daily_trade_count += 1
            self.logger.info(f"Открыта крипто-позиция: {position.side} {position.size} {symbol}")
        
        return position

