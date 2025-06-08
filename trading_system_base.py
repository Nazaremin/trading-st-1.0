"""
Алгоритмическая торговая система - Базовая архитектура
Автор: Manus AI (Алгопрограммист)
Версия: 1.0

Система реализует оптимизированную многотаймфреймную торговую стратегию
с адаптацией под Forex (низкая волатильность) и криптовалюты (высокая волатильность)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
from abc import ABC, abstractmethod

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class MarketType(Enum):
    """Типы рынков для оптимизации стратегии"""
    FOREX = "forex"
    CRYPTO = "crypto"

class TimeFrame(Enum):
    """Временные рамки для анализа"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

class SignalType(Enum):
    """Типы торговых сигналов"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

class OrderType(Enum):
    """Типы ордеров"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

@dataclass
class MarketData:
    """Структура рыночных данных"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: TimeFrame

@dataclass
class Signal:
    """Структура торгового сигнала"""
    timestamp: datetime
    signal_type: SignalType
    entry_price: float
    stop_loss: float
    take_profit: List[float]
    confidence: float
    timeframe_analysis: Dict[TimeFrame, str]
    risk_reward_ratio: float

@dataclass
class Position:
    """Структура позиции"""
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: List[float]
    unrealized_pnl: float
    timestamp: datetime

class MarketConfig:
    """Конфигурация для различных типов рынков"""
    
    def __init__(self, market_type: MarketType):
        self.market_type = market_type
        self._setup_config()
    
    def _setup_config(self):
        """Настройка параметров в зависимости от типа рынка"""
        if self.market_type == MarketType.FOREX:
            # Конфигурация для Forex (низкая волатильность)
            self.risk_per_trade = 0.01  # 1% риска на сделку
            self.max_positions = 3      # Максимум 3 позиции одновременно
            self.min_rr_ratio = 2.0     # Минимальное соотношение риск/прибыль
            self.volatility_multiplier = 1.0
            self.stop_loss_atr_multiplier = 1.5
            self.take_profit_levels = [2.0, 3.0, 4.0]  # Множители для тейк-профитов
            self.timeframes = {
                'context': TimeFrame.D1,    # Контекстный таймфрейм
                'signal': TimeFrame.H4,     # Сигнальный таймфрейм  
                'execution': TimeFrame.H1   # Исполнительный таймфрейм
            }
            self.session_times = {
                'asian': (0, 9),
                'european': (7, 16),
                'american': (13, 22)
            }
            
        elif self.market_type == MarketType.CRYPTO:
            # Конфигурация для криптовалют (высокая волатильность)
            self.risk_per_trade = 0.005  # 0.5% риска на сделку (меньше из-за высокой волатильности)
            self.max_positions = 2       # Максимум 2 позиции одновременно
            self.min_rr_ratio = 3.0      # Более высокое соотношение риск/прибыль
            self.volatility_multiplier = 2.0
            self.stop_loss_atr_multiplier = 2.5  # Более широкие стопы
            self.take_profit_levels = [3.0, 5.0, 8.0]  # Более агрессивные цели
            self.timeframes = {
                'context': TimeFrame.H4,    # Более короткие таймфреймы
                'signal': TimeFrame.H1,
                'execution': TimeFrame.M15
            }
            self.session_times = None  # Криптовалюты торгуются 24/7

class TechnicalIndicators:
    """Класс для расчета технических индикаторов"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Простая скользящая средняя"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Экспоненциальная скользящая средняя"""
        return data.ewm(span=period).mean()
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range - индикатор волатильности"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Полосы Боллинджера"""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }

class FractalDetector:
    """Детектор фракталов"""
    
    @staticmethod
    def detect_fractals(high: pd.Series, low: pd.Series, period: int = 5) -> Dict[str, pd.Series]:
        """
        Обнаружение фракталов (паттерн из 5 свечей)
        Возвращает серии с булевыми значениями для фракталов вверх и вниз
        """
        fractals_up = pd.Series(False, index=high.index)
        fractals_down = pd.Series(False, index=low.index)
        
        half_period = period // 2
        
        for i in range(half_period, len(high) - half_period):
            # Фрактал вверх - максимум выше соседних свечей
            if all(high.iloc[i] > high.iloc[i-j] for j in range(1, half_period + 1)) and \
               all(high.iloc[i] > high.iloc[i+j] for j in range(1, half_period + 1)):
                fractals_up.iloc[i] = True
            
            # Фрактал вниз - минимум ниже соседних свечей
            if all(low.iloc[i] < low.iloc[i-j] for j in range(1, half_period + 1)) and \
               all(low.iloc[i] < low.iloc[i+j] for j in range(1, half_period + 1)):
                fractals_down.iloc[i] = True
        
        return {'up': fractals_up, 'down': fractals_down}

class ImbalanceDetector:
    """Детектор имбалансов (Fair Value Gaps)"""
    
    @staticmethod
    def detect_imbalances(high: pd.Series, low: pd.Series, close: pd.Series) -> List[Dict]:
        """
        Обнаружение имбалансов (FVG) - зон, где цена двигалась быстро
        Возвращает список словарей с информацией об имбалансах
        """
        imbalances = []
        
        for i in range(2, len(high)):
            # Бычий имбаланс: gap между high[i-2] и low[i]
            if high.iloc[i-2] < low.iloc[i]:
                imbalances.append({
                    'type': 'bullish',
                    'start_index': i-2,
                    'end_index': i,
                    'upper_level': low.iloc[i],
                    'lower_level': high.iloc[i-2],
                    'strength': abs(low.iloc[i] - high.iloc[i-2]) / close.iloc[i-1]
                })
            
            # Медвежий имбаланс: gap между low[i-2] и high[i]
            elif low.iloc[i-2] > high.iloc[i]:
                imbalances.append({
                    'type': 'bearish',
                    'start_index': i-2,
                    'end_index': i,
                    'upper_level': low.iloc[i-2],
                    'lower_level': high.iloc[i],
                    'strength': abs(low.iloc[i-2] - high.iloc[i]) / close.iloc[i-1]
                })
        
        return imbalances

class OrderflowAnalyzer:
    """Анализатор качества движения цены (Orderflow)"""
    
    @staticmethod
    def analyze_orderflow(open_prices: pd.Series, high: pd.Series, low: pd.Series, 
                         close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Анализ качества движения цены
        Возвращает оценку здоровья Orderflow (-1 до 1)
        """
        # Расчет размера тела свечи
        body_size = abs(close - open_prices)
        candle_range = high - low
        
        # Расчет теней
        upper_shadow = high - np.maximum(open_prices, close)
        lower_shadow = np.minimum(open_prices, close) - low
        
        # Оценка качества движения
        orderflow_score = pd.Series(0.0, index=close.index)
        
        for i in range(1, len(close)):
            score = 0.0
            
            # Размер тела относительно диапазона
            if candle_range.iloc[i] > 0:
                body_ratio = body_size.iloc[i] / candle_range.iloc[i]
                score += body_ratio * 0.3
            
            # Направление движения
            if close.iloc[i] > close.iloc[i-1]:
                score += 0.2
            elif close.iloc[i] < close.iloc[i-1]:
                score -= 0.2
            
            # Объем (если доступен)
            if not pd.isna(volume.iloc[i]) and i > 0:
                if volume.iloc[i] > volume.iloc[i-1]:
                    score += 0.1
                else:
                    score -= 0.1
            
            # Размер теней
            if candle_range.iloc[i] > 0:
                shadow_ratio = (upper_shadow.iloc[i] + lower_shadow.iloc[i]) / candle_range.iloc[i]
                score -= shadow_ratio * 0.2
            
            orderflow_score.iloc[i] = np.clip(score, -1.0, 1.0)
        
        return orderflow_score

class BaseStrategy(ABC):
    """Базовый класс для торговых стратегий"""
    
    def __init__(self, market_config: MarketConfig):
        self.config = market_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.positions: List[Position] = []
        self.signals_history: List[Signal] = []
        
    @abstractmethod
    def analyze_market(self, data: Dict[TimeFrame, pd.DataFrame]) -> Optional[Signal]:
        """Анализ рынка и генерация сигналов"""
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Signal, account_balance: float) -> float:
        """Расчет размера позиции"""
        pass
    
    def validate_signal(self, signal: Signal) -> bool:
        """Валидация торгового сигнала"""
        # Проверка соотношения риск/прибыль
        if signal.risk_reward_ratio < self.config.min_rr_ratio:
            self.logger.warning(f"Сигнал отклонен: RR {signal.risk_reward_ratio} < {self.config.min_rr_ratio}")
            return False
        
        # Проверка уровня уверенности
        if signal.confidence < 0.6:
            self.logger.warning(f"Сигнал отклонен: низкая уверенность {signal.confidence}")
            return False
        
        # Проверка максимального количества позиций
        if len(self.positions) >= self.config.max_positions:
            self.logger.warning("Сигнал отклонен: достигнуто максимальное количество позиций")
            return False
        
        return True

