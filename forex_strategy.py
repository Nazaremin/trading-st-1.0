"""
Forex-оптимизированная торговая стратегия

Специализированная версия стратегии для валютного рынка с учетом:
- Низкой волатильности
- Сессионности торгов
- Корреляций валютных пар
- Особенностей ликвидности
"""

from trading_system_base import *
from multi_timeframe_analysis import *
from risk_position_management import *
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, time
import pytz

class ForexSessionManager:
    """Менеджер торговых сессий для Forex"""
    
    def __init__(self):
        self.sessions = {
            'asian': {
                'start': time(0, 0),   # 00:00 UTC
                'end': time(9, 0),     # 09:00 UTC
                'major_pairs': ['USD/JPY', 'AUD/USD', 'NZD/USD'],
                'volatility_factor': 0.7
            },
            'european': {
                'start': time(7, 0),   # 07:00 UTC
                'end': time(16, 0),    # 16:00 UTC
                'major_pairs': ['EUR/USD', 'GBP/USD', 'EUR/GBP', 'EUR/JPY'],
                'volatility_factor': 1.0
            },
            'american': {
                'start': time(13, 0),  # 13:00 UTC
                'end': time(22, 0),    # 22:00 UTC
                'major_pairs': ['USD/CAD', 'EUR/USD', 'GBP/USD'],
                'volatility_factor': 1.2
            }
        }
        
        # Периоды пересечения сессий (высокая ликвидность)
        self.overlap_periods = {
            'european_american': {
                'start': time(13, 0),  # 13:00 UTC
                'end': time(16, 0),    # 16:00 UTC
                'volatility_factor': 1.5
            },
            'asian_european': {
                'start': time(7, 0),   # 07:00 UTC
                'end': time(9, 0),     # 09:00 UTC
                'volatility_factor': 1.1
            }
        }
    
    def get_current_session(self, current_time: datetime = None) -> Dict:
        """Определение текущей торговой сессии"""
        if current_time is None:
            current_time = datetime.now(pytz.UTC)
        
        current_time_only = current_time.time()
        
        # Проверка периодов пересечения
        for overlap_name, overlap_info in self.overlap_periods.items():
            if overlap_info['start'] <= current_time_only <= overlap_info['end']:
                return {
                    'type': 'overlap',
                    'name': overlap_name,
                    'volatility_factor': overlap_info['volatility_factor'],
                    'high_liquidity': True
                }
        
        # Проверка основных сессий
        for session_name, session_info in self.sessions.items():
            if session_info['start'] <= current_time_only <= session_info['end']:
                return {
                    'type': 'main',
                    'name': session_name,
                    'major_pairs': session_info['major_pairs'],
                    'volatility_factor': session_info['volatility_factor'],
                    'high_liquidity': session_name in ['european', 'american']
                }
        
        # Период низкой ликвидности
        return {
            'type': 'low_liquidity',
            'name': 'off_session',
            'volatility_factor': 0.5,
            'high_liquidity': False
        }
    
    def is_news_time(self, current_time: datetime = None) -> bool:
        """Проверка времени выхода важных новостей"""
        if current_time is None:
            current_time = datetime.now(pytz.UTC)
        
        # Основные времена выхода новостей (UTC)
        news_times = [
            time(8, 30),   # EUR новости
            time(12, 30),  # GBP новости
            time(13, 30),  # USD новости
            time(15, 30),  # USD новости
        ]
        
        current_time_only = current_time.time()
        
        # Проверка ±30 минут от времени новостей
        for news_time in news_times:
            news_datetime = datetime.combine(current_time.date(), news_time)
            time_diff = abs((current_time - news_datetime).total_seconds())
            if time_diff <= 1800:  # 30 минут
                return True
        
        return False

class ForexCorrelationAnalyzer:
    """Анализатор корреляций валютных пар"""
    
    def __init__(self):
        # Матрица корреляций основных валютных пар (упрощенная)
        self.correlation_matrix = {
            'EUR/USD': {
                'GBP/USD': 0.7,
                'AUD/USD': 0.6,
                'NZD/USD': 0.5,
                'USD/CHF': -0.8,
                'USD/JPY': -0.3,
                'EUR/GBP': 0.4,
                'EUR/JPY': 0.6
            },
            'GBP/USD': {
                'EUR/USD': 0.7,
                'AUD/USD': 0.5,
                'NZD/USD': 0.4,
                'USD/CHF': -0.6,
                'USD/JPY': -0.2,
                'EUR/GBP': -0.3,
                'GBP/JPY': 0.5
            },
            'USD/JPY': {
                'EUR/USD': -0.3,
                'GBP/USD': -0.2,
                'AUD/USD': -0.1,
                'USD/CHF': 0.4,
                'EUR/JPY': 0.8,
                'GBP/JPY': 0.7
            },
            'AUD/USD': {
                'EUR/USD': 0.6,
                'GBP/USD': 0.5,
                'NZD/USD': 0.8,
                'USD/CHF': -0.5,
                'USD/JPY': -0.1
            },
            'USD/CHF': {
                'EUR/USD': -0.8,
                'GBP/USD': -0.6,
                'AUD/USD': -0.5,
                'USD/JPY': 0.4
            }
        }
    
    def get_correlation(self, pair1: str, pair2: str) -> float:
        """Получение коэффициента корреляции между парами"""
        if pair1 in self.correlation_matrix and pair2 in self.correlation_matrix[pair1]:
            return self.correlation_matrix[pair1][pair2]
        elif pair2 in self.correlation_matrix and pair1 in self.correlation_matrix[pair2]:
            return self.correlation_matrix[pair2][pair1]
        else:
            return 0.0
    
    def calculate_portfolio_correlation_risk(self, positions: List[Position]) -> float:
        """Расчет корреляционного риска портфеля"""
        if len(positions) <= 1:
            return 0.0
        
        total_correlation_risk = 0.0
        pair_count = 0
        
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions[i+1:], i+1):
                correlation = self.get_correlation(pos1.symbol, pos2.symbol)
                
                # Риск увеличивается при одинаковом направлении и положительной корреляции
                # или при противоположном направлении и отрицательной корреляции
                if (pos1.side == pos2.side and correlation > 0) or \
                   (pos1.side != pos2.side and correlation < 0):
                    total_correlation_risk += abs(correlation)
                
                pair_count += 1
        
        return total_correlation_risk / pair_count if pair_count > 0 else 0.0
    
    def suggest_hedge_pairs(self, current_pair: str, position_side: str) -> List[Tuple[str, str]]:
        """Предложение пар для хеджирования"""
        hedge_suggestions = []
        
        if current_pair in self.correlation_matrix:
            for pair, correlation in self.correlation_matrix[current_pair].items():
                # Для хеджирования ищем сильно коррелирующие пары
                if abs(correlation) > 0.6:
                    if correlation > 0:
                        # Положительная корреляция - противоположная позиция
                        hedge_side = 'sell' if position_side == 'buy' else 'buy'
                    else:
                        # Отрицательная корреляция - такая же позиция
                        hedge_side = position_side
                    
                    hedge_suggestions.append((pair, hedge_side))
        
        return hedge_suggestions

class ForexStrategy(BaseStrategy):
    """Forex-оптимизированная торговая стратегия"""
    
    def __init__(self):
        super().__init__(MarketConfig(MarketType.FOREX))
        self.session_manager = ForexSessionManager()
        self.correlation_analyzer = ForexCorrelationAnalyzer()
        self.mtf_analyzer = MultiTimeframeAnalyzer(self.config)
        self.pattern_detector = PatternDetector(self.config)
        self.synchronizer = TimeframeSynchronizer(self.config)
        self.position_manager = PositionManager(self.config)
        
        # Forex-специфичные параметры
        self.min_pip_movement = 10  # Минимальное движение для входа (в пипсах)
        self.spread_filter = 3      # Максимальный спред для торговли (в пипсах)
        self.news_avoidance_minutes = 30  # Избегание торговли за 30 мин до/после новостей
        
    def analyze_market(self, data: Dict[TimeFrame, pd.DataFrame], symbol: str) -> Optional[Signal]:
        """Анализ рынка для Forex"""
        
        # Проверка торговой сессии
        current_session = self.session_manager.get_current_session()
        if not self._is_suitable_session(current_session, symbol):
            return None
        
        # Проверка времени новостей
        if self.session_manager.is_news_time():
            self.logger.info("Избегание торговли во время новостей")
            return None
        
        # Многотаймфреймный анализ
        timeframe_analyses = {}
        for tf_name, tf_enum in self.config.timeframes.items():
            if tf_enum in data:
                timeframe_analyses[tf_name] = self.mtf_analyzer.analyze_timeframe(data[tf_enum], tf_enum)
        
        if len(timeframe_analyses) < 3:
            self.logger.warning("Недостаточно данных для многотаймфреймного анализа")
            return None
        
        # Синхронизация анализа
        sync_result = self.synchronizer.synchronize_analysis(timeframe_analyses)
        if not sync_result.get('synchronized', False):
            return None
        
        # Детекция паттернов на сигнальном таймфрейме
        signal_tf_data = data[self.config.timeframes['signal']]
        signal_analysis = timeframe_analyses['signal']
        
        # Поиск паттерна AMD
        amd_pattern = self.pattern_detector.detect_amd_pattern(signal_tf_data, signal_analysis)
        
        if amd_pattern and amd_pattern['confidence'] > 0.7:
            # Проверка здорового Orderflow на исполнительном таймфрейме
            execution_tf_data = data[self.config.timeframes['execution']]
            orderflow_check = self.pattern_detector.detect_healthy_orderflow(
                execution_tf_data, amd_pattern['distribution_direction']
            )
            
            if orderflow_check['healthy']:
                return self._create_forex_signal(
                    amd_pattern, signal_analysis, sync_result, current_session, symbol
                )
        
        # Альтернативный поиск сигналов на основе структуры и имбалансов
        structure_signal = self._analyze_structure_signals(
            timeframe_analyses, sync_result, current_session, symbol
        )
        
        return structure_signal
    
    def _is_suitable_session(self, session_info: Dict, symbol: str) -> bool:
        """Проверка подходящей сессии для торговли"""
        
        # Избегаем торговли в периоды низкой ликвидности
        if session_info['type'] == 'low_liquidity':
            return False
        
        # Проверяем, подходит ли пара для текущей сессии
        if session_info['type'] == 'main':
            major_pairs = session_info.get('major_pairs', [])
            if symbol not in major_pairs and session_info['name'] == 'asian':
                return False
        
        return True
    
    def _create_forex_signal(self, amd_pattern: Dict, signal_analysis: TimeframeAnalysis,
                           sync_result: Dict, session_info: Dict, symbol: str) -> Signal:
        """Создание Forex-сигнала"""
        
        direction = amd_pattern['distribution_direction']
        entry_info = amd_pattern['entry_zone']
        
        # Корректировка входа с учетом сессии
        session_volatility_factor = session_info.get('volatility_factor', 1.0)
        
        # Расчет стоп-лосса с учетом ATR и ключевых уровней
        atr_value = signal_analysis.key_levels[-1] if signal_analysis.key_levels else 0.001  # Примерное значение
        stop_loss = self.position_manager.risk_manager.calculate_stop_loss(
            entry_info['entry_price'], direction, atr_value, signal_analysis.key_levels
        )
        
        # Расчет тейк-профитов с учетом имбалансов
        take_profits = self.position_manager.risk_manager.calculate_take_profits(
            entry_info['entry_price'], stop_loss, direction, signal_analysis.imbalances
        )
        
        # Корректировка целей с учетом волатильности сессии
        adjusted_take_profits = [
            tp * session_volatility_factor for tp in take_profits
        ]
        
        # Расчет соотношения риск/прибыль
        risk_distance = abs(entry_info['entry_price'] - stop_loss)
        reward_distance = abs(adjusted_take_profits[0] - entry_info['entry_price']) if adjusted_take_profits else 0
        rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
        
        # Общая уверенность с учетом Forex-факторов
        base_confidence = amd_pattern['confidence']
        session_bonus = 0.1 if session_info.get('high_liquidity', False) else 0
        sync_bonus = 0.1 if sync_result.get('type') == 'full_sync' else 0.05
        
        total_confidence = min(base_confidence + session_bonus + sync_bonus, 1.0)
        
        return Signal(
            timestamp=datetime.now(),
            signal_type=SignalType.BUY if direction == 'bullish' else SignalType.SELL,
            entry_price=entry_info['entry_price'],
            stop_loss=stop_loss,
            take_profit=adjusted_take_profits,
            confidence=total_confidence,
            timeframe_analysis={
                self.config.timeframes['context']: f"Trend: {sync_result.get('direction', 'unknown')}",
                self.config.timeframes['signal']: f"AMD Pattern: {amd_pattern['pattern']}",
                self.config.timeframes['execution']: f"Healthy Orderflow: {direction}"
            },
            risk_reward_ratio=rr_ratio
        )
    
    def _analyze_structure_signals(self, timeframe_analyses: Dict, sync_result: Dict,
                                 session_info: Dict, symbol: str) -> Optional[Signal]:
        """Анализ сигналов на основе структуры рынка"""
        
        if not sync_result.get('synchronized', False):
            return None
        
        signal_analysis = timeframe_analyses['signal']
        execution_analysis = timeframe_analyses['execution']
        
        # Поиск торговых возможностей в зонах премиума/дисконта
        current_price = 1.0  # Заглушка - в реальной системе получаем из данных
        
        premium_zone = signal_analysis.premium_discount_zones.get('premium')
        discount_zone = signal_analysis.premium_discount_zones.get('discount')
        
        direction = sync_result.get('direction')
        
        # Покупка в зоне дисконта при бычьем тренде
        if (direction == 'bullish' and discount_zone and 
            discount_zone[0] <= current_price <= discount_zone[1]):
            
            # Проверка качества Orderflow
            orderflow_check = self.pattern_detector.detect_healthy_orderflow(
                pd.DataFrame(), 'bullish'  # Заглушка для данных
            )
            
            if orderflow_check['score'] > 0.6:
                return self._create_structure_signal(
                    'bullish', current_price, signal_analysis, session_info
                )
        
        # Продажа в зоне премиума при медвежьем тренде
        elif (direction == 'bearish' and premium_zone and 
              premium_zone[0] <= current_price <= premium_zone[1]):
            
            orderflow_check = self.pattern_detector.detect_healthy_orderflow(
                pd.DataFrame(), 'bearish'  # Заглушка для данных
            )
            
            if orderflow_check['score'] > 0.6:
                return self._create_structure_signal(
                    'bearish', current_price, signal_analysis, session_info
                )
        
        return None
    
    def _create_structure_signal(self, direction: str, entry_price: float,
                               signal_analysis: TimeframeAnalysis, session_info: Dict) -> Signal:
        """Создание сигнала на основе структуры"""
        
        # Упрощенный расчет для демонстрации
        atr_value = 0.001  # Примерное значение ATR
        
        if direction == 'bullish':
            stop_loss = entry_price - (atr_value * self.config.stop_loss_atr_multiplier)
            take_profits = [
                entry_price + (atr_value * multiplier) 
                for multiplier in self.config.take_profit_levels
            ]
        else:
            stop_loss = entry_price + (atr_value * self.config.stop_loss_atr_multiplier)
            take_profits = [
                entry_price - (atr_value * multiplier) 
                for multiplier in self.config.take_profit_levels
            ]
        
        # Расчет соотношения риск/прибыль
        risk_distance = abs(entry_price - stop_loss)
        reward_distance = abs(take_profits[0] - entry_price) if take_profits else 0
        rr_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
        
        # Уверенность на основе качества структуры и сессии
        base_confidence = signal_analysis.structure_quality
        session_bonus = 0.1 if session_info.get('high_liquidity', False) else 0
        total_confidence = min(base_confidence + session_bonus, 1.0)
        
        return Signal(
            timestamp=datetime.now(),
            signal_type=SignalType.BUY if direction == 'bullish' else SignalType.SELL,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profits,
            confidence=total_confidence,
            timeframe_analysis={
                self.config.timeframes['context']: f"Structure: {direction}",
                self.config.timeframes['signal']: f"Zone: {'discount' if direction == 'bullish' else 'premium'}",
                self.config.timeframes['execution']: f"Quality: {signal_analysis.structure_quality:.2f}"
            },
            risk_reward_ratio=rr_ratio
        )
    
    def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """Forex-специфичный расчет размера позиции"""
        
        # Получение текущей сессии для корректировки
        current_session = self.session_manager.get_current_session()
        session_factor = current_session.get('volatility_factor', 1.0)
        
        # Базовый расчет
        base_size = self.position_manager.risk_manager.calculate_position_size(
            signal, account_balance, signal.entry_price, 0.001  # Примерное ATR
        )
        
        # Корректировка с учетом сессии
        # В периоды низкой волатильности можем увеличить размер
        if session_factor < 0.8:
            adjusted_size = base_size.position_size * 1.2
        elif session_factor > 1.3:
            adjusted_size = base_size.position_size * 0.8
        else:
            adjusted_size = base_size.position_size
        
        return adjusted_size
    
    def should_avoid_trading(self, symbol: str) -> Tuple[bool, str]:
        """Проверка условий для избегания торговли"""
        
        # Проверка времени новостей
        if self.session_manager.is_news_time():
            return True, "Время выхода новостей"
        
        # Проверка сессии
        current_session = self.session_manager.get_current_session()
        if not self._is_suitable_session(current_session, symbol):
            return True, f"Неподходящая сессия: {current_session.get('name', 'unknown')}"
        
        # Проверка корреляционного риска
        correlation_risk = self.correlation_analyzer.calculate_portfolio_correlation_risk(
            list(self.position_manager.positions.values())
        )
        if correlation_risk > 0.7:
            return True, "Высокий корреляционный риск портфеля"
        
        return False, ""
    
    def get_trading_recommendations(self) -> Dict:
        """Получение торговых рекомендаций для Forex"""
        
        current_session = self.session_manager.get_current_session()
        portfolio_summary = self.position_manager.get_portfolio_summary()
        
        recommendations = {
            'current_session': current_session,
            'recommended_pairs': [],
            'avoid_trading': False,
            'risk_level': 'normal',
            'portfolio_health': 'good'
        }
        
        # Рекомендации по парам в зависимости от сессии
        if current_session['type'] == 'main':
            recommendations['recommended_pairs'] = current_session.get('major_pairs', [])
        elif current_session['type'] == 'overlap':
            recommendations['recommended_pairs'] = ['EUR/USD', 'GBP/USD', 'USD/JPY']
        
        # Оценка уровня риска
        risk_metrics = portfolio_summary['risk_metrics']
        if risk_metrics.total_risk_exposure > 0.05:  # 5% от баланса
            recommendations['risk_level'] = 'high'
        elif risk_metrics.position_correlation > 0.7:
            recommendations['risk_level'] = 'elevated'
        
        # Оценка здоровья портфеля
        if risk_metrics.max_drawdown > 0.1:  # 10%
            recommendations['portfolio_health'] = 'poor'
        elif risk_metrics.win_rate < 0.4:
            recommendations['portfolio_health'] = 'fair'
        
        # Рекомендации по избеганию торговли
        avoid_trading, reason = self.should_avoid_trading('EUR/USD')  # Пример
        recommendations['avoid_trading'] = avoid_trading
        recommendations['avoid_reason'] = reason
        
        return recommendations

