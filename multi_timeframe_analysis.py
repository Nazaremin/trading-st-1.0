"""
Многотаймфреймный анализ и детекция паттернов

Реализует анализ по трем таймфреймам с детекцией паттернов AMD,
здорового Orderflow и синхронизацией сигналов
"""

from trading_system_base import *
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TimeframeAnalysis:
    """Результат анализа одного таймфрейма"""
    timeframe: TimeFrame
    trend_direction: str  # 'bullish', 'bearish', 'sideways'
    structure_quality: float  # 0-1, качество структуры
    key_levels: List[float]  # Ключевые уровни
    fractals: Dict[str, List[float]]  # Фракталы up/down
    imbalances: List[Dict]  # Имбалансы
    orderflow_score: float  # -1 до 1, качество движения
    premium_discount_zones: Dict[str, Tuple[float, float]]  # Зоны премиума/дисконта

class MultiTimeframeAnalyzer:
    """Анализатор множественных таймфреймов"""
    
    def __init__(self, market_config: MarketConfig):
        self.config = market_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.fractal_detector = FractalDetector()
        self.imbalance_detector = ImbalanceDetector()
        self.orderflow_analyzer = OrderflowAnalyzer()
        
    def analyze_timeframe(self, data: pd.DataFrame, timeframe: TimeFrame) -> TimeframeAnalysis:
        """Анализ одного таймфрейма"""
        
        # Расчет технических индикаторов
        data['sma_20'] = TechnicalIndicators.sma(data['close'], 20)
        data['sma_50'] = TechnicalIndicators.sma(data['close'], 50)
        data['ema_12'] = TechnicalIndicators.ema(data['close'], 12)
        data['ema_26'] = TechnicalIndicators.ema(data['close'], 26)
        data['atr'] = TechnicalIndicators.atr(data['high'], data['low'], data['close'])
        data['rsi'] = TechnicalIndicators.rsi(data['close'])
        
        # Определение тренда
        trend_direction = self._determine_trend(data)
        
        # Оценка качества структуры
        structure_quality = self._assess_structure_quality(data)
        
        # Поиск ключевых уровней
        key_levels = self._find_key_levels(data)
        
        # Детекция фракталов
        fractals = self.fractal_detector.detect_fractals(data['high'], data['low'])
        fractal_levels = {
            'up': data.loc[fractals['up'], 'high'].tolist(),
            'down': data.loc[fractals['down'], 'low'].tolist()
        }
        
        # Детекция имбалансов
        imbalances = self.imbalance_detector.detect_imbalances(
            data['high'], data['low'], data['close']
        )
        
        # Анализ Orderflow
        orderflow_score = self.orderflow_analyzer.analyze_orderflow(
            data['open'], data['high'], data['low'], data['close'], 
            data.get('volume', pd.Series([np.nan] * len(data)))
        ).iloc[-1] if len(data) > 0 else 0.0
        
        # Определение зон премиума/дисконта
        premium_discount_zones = self._calculate_premium_discount_zones(data)
        
        return TimeframeAnalysis(
            timeframe=timeframe,
            trend_direction=trend_direction,
            structure_quality=structure_quality,
            key_levels=key_levels,
            fractals=fractal_levels,
            imbalances=imbalances,
            orderflow_score=orderflow_score,
            premium_discount_zones=premium_discount_zones
        )
    
    def _determine_trend(self, data: pd.DataFrame) -> str:
        """Определение направления тренда"""
        if len(data) < 50:
            return 'sideways'
        
        # Анализ скользящих средних
        sma_20 = data['sma_20'].iloc[-1]
        sma_50 = data['sma_50'].iloc[-1]
        current_price = data['close'].iloc[-1]
        
        # Анализ структуры (Higher Highs, Higher Lows)
        recent_highs = data['high'].rolling(10).max().tail(5)
        recent_lows = data['low'].rolling(10).min().tail(5)
        
        hh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs.iloc[i] > recent_highs.iloc[i-1])
        hl_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows.iloc[i] > recent_lows.iloc[i-1])
        
        lh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs.iloc[i] < recent_highs.iloc[i-1])
        ll_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows.iloc[i] < recent_lows.iloc[i-1])
        
        # Определение тренда
        if (current_price > sma_20 > sma_50) and (hh_count >= 2 and hl_count >= 2):
            return 'bullish'
        elif (current_price < sma_20 < sma_50) and (lh_count >= 2 and ll_count >= 2):
            return 'bearish'
        else:
            return 'sideways'
    
    def _assess_structure_quality(self, data: pd.DataFrame) -> float:
        """Оценка качества рыночной структуры (0-1)"""
        if len(data) < 20:
            return 0.5
        
        score = 0.0
        
        # Качество тренда (согласованность скользящих средних)
        if 'sma_20' in data.columns and 'sma_50' in data.columns:
            sma_alignment = abs(data['sma_20'].iloc[-1] - data['sma_50'].iloc[-1]) / data['close'].iloc[-1]
            score += (1 - min(sma_alignment * 10, 1)) * 0.3
        
        # Волатильность (стабильность движения)
        if 'atr' in data.columns:
            atr_stability = 1 - (data['atr'].tail(10).std() / data['atr'].tail(10).mean())
            score += max(0, atr_stability) * 0.3
        
        # RSI (не перекупленность/перепроданность)
        if 'rsi' in data.columns:
            rsi_value = data['rsi'].iloc[-1]
            if 30 <= rsi_value <= 70:
                score += 0.4
            elif 20 <= rsi_value <= 80:
                score += 0.2
        
        return min(score, 1.0)
    
    def _find_key_levels(self, data: pd.DataFrame) -> List[float]:
        """Поиск ключевых уровней поддержки/сопротивления"""
        levels = []
        
        # Уровни на основе фракталов
        fractals = self.fractal_detector.detect_fractals(data['high'], data['low'])
        
        # Добавляем уровни фракталов
        fractal_highs = data.loc[fractals['up'], 'high'].tolist()
        fractal_lows = data.loc[fractals['down'], 'low'].tolist()
        
        levels.extend(fractal_highs)
        levels.extend(fractal_lows)
        
        # Психологические уровни (круглые числа)
        current_price = data['close'].iloc[-1]
        price_range = data['high'].max() - data['low'].min()
        
        # Определяем шаг для психологических уровней
        if price_range > 1000:
            step = 100
        elif price_range > 100:
            step = 10
        elif price_range > 10:
            step = 1
        else:
            step = 0.1
        
        # Добавляем ближайшие психологические уровни
        base_level = round(current_price / step) * step
        for i in range(-3, 4):
            levels.append(base_level + i * step)
        
        # Удаляем дубликаты и сортируем
        levels = sorted(list(set(levels)))
        
        # Фильтруем уровни в разумном диапазоне от текущей цены
        price_filter = price_range * 0.1
        levels = [level for level in levels 
                 if abs(level - current_price) <= price_filter]
        
        return levels
    
    def _calculate_premium_discount_zones(self, data: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Расчет зон премиума и дисконта"""
        if len(data) < 20:
            return {'premium': (0, 0), 'discount': (0, 0)}
        
        # Определяем диапазон для анализа (последние 50 свечей или все данные)
        analysis_period = min(50, len(data))
        recent_data = data.tail(analysis_period)
        
        high_level = recent_data['high'].max()
        low_level = recent_data['low'].min()
        range_size = high_level - low_level
        
        # Зоны премиума (верхние 30% диапазона)
        premium_start = high_level - (range_size * 0.3)
        premium_zone = (premium_start, high_level)
        
        # Зоны дисконта (нижние 30% диапазона)
        discount_end = low_level + (range_size * 0.3)
        discount_zone = (low_level, discount_end)
        
        return {
            'premium': premium_zone,
            'discount': discount_zone
        }

class PatternDetector:
    """Детектор торговых паттернов"""
    
    def __init__(self, market_config: MarketConfig):
        self.config = market_config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect_amd_pattern(self, data: pd.DataFrame, timeframe_analysis: TimeframeAnalysis) -> Optional[Dict]:
        """
        Детекция паттерна AMD (Accumulation-Manipulation-Distribution)
        """
        if len(data) < 50:
            return None
        
        # Ищем консолидацию (аккумуляция)
        consolidation = self._find_consolidation(data)
        if not consolidation:
            return None
        
        # Проверяем манипуляцию (ложный пробой)
        manipulation = self._detect_manipulation(data, consolidation)
        if not manipulation:
            return None
        
        # Определяем направление дистрибуции
        distribution_direction = self._predict_distribution(data, consolidation, manipulation)
        
        if distribution_direction:
            return {
                'pattern': 'AMD',
                'accumulation_zone': consolidation,
                'manipulation': manipulation,
                'distribution_direction': distribution_direction,
                'confidence': self._calculate_amd_confidence(data, consolidation, manipulation),
                'entry_zone': self._calculate_amd_entry(consolidation, manipulation, distribution_direction)
            }
        
        return None
    
    def _find_consolidation(self, data: pd.DataFrame, min_periods: int = 20) -> Optional[Dict]:
        """Поиск зоны консолидации"""
        for i in range(len(data) - min_periods, min_periods, -1):
            window_data = data.iloc[i-min_periods:i]
            
            # Проверяем, что цена находится в узком диапазоне
            high_level = window_data['high'].max()
            low_level = window_data['low'].min()
            range_size = high_level - low_level
            avg_price = window_data['close'].mean()
            
            # Консолидация: диапазон меньше 3% от средней цены
            if range_size / avg_price < 0.03:
                return {
                    'start_index': i - min_periods,
                    'end_index': i,
                    'high': high_level,
                    'low': low_level,
                    'range_size': range_size
                }
        
        return None
    
    def _detect_manipulation(self, data: pd.DataFrame, consolidation: Dict) -> Optional[Dict]:
        """Детекция манипуляции (ложного пробоя)"""
        cons_end = consolidation['end_index']
        
        # Ищем пробой в следующих 10 свечах после консолидации
        for i in range(cons_end, min(cons_end + 10, len(data))):
            current_candle = data.iloc[i]
            
            # Пробой вверх
            if current_candle['high'] > consolidation['high']:
                # Проверяем возврат в диапазон в следующих свечах
                for j in range(i + 1, min(i + 5, len(data))):
                    if data.iloc[j]['close'] < consolidation['high']:
                        return {
                            'type': 'false_breakout_up',
                            'breakout_index': i,
                            'return_index': j,
                            'max_penetration': current_candle['high'] - consolidation['high']
                        }
            
            # Пробой вниз
            elif current_candle['low'] < consolidation['low']:
                # Проверяем возврат в диапазон
                for j in range(i + 1, min(i + 5, len(data))):
                    if data.iloc[j]['close'] > consolidation['low']:
                        return {
                            'type': 'false_breakout_down',
                            'breakout_index': i,
                            'return_index': j,
                            'max_penetration': consolidation['low'] - current_candle['low']
                        }
        
        return None
    
    def _predict_distribution(self, data: pd.DataFrame, consolidation: Dict, manipulation: Dict) -> Optional[str]:
        """Предсказание направления дистрибуции"""
        if manipulation['type'] == 'false_breakout_up':
            # Ложный пробой вверх -> ожидаем движение вниз
            return 'bearish'
        elif manipulation['type'] == 'false_breakout_down':
            # Ложный пробой вниз -> ожидаем движение вверх
            return 'bullish'
        
        return None
    
    def _calculate_amd_confidence(self, data: pd.DataFrame, consolidation: Dict, manipulation: Dict) -> float:
        """Расчет уверенности в паттерне AMD"""
        confidence = 0.5
        
        # Качество консолидации (чем дольше, тем лучше)
        consolidation_length = consolidation['end_index'] - consolidation['start_index']
        confidence += min(consolidation_length / 50, 0.2)
        
        # Размер манипуляции (не слишком большой)
        range_size = consolidation['high'] - consolidation['low']
        if range_size > 0:
            manipulation_ratio = manipulation['max_penetration'] / range_size
            if 0.1 <= manipulation_ratio <= 0.5:
                confidence += 0.2
        
        # Скорость возврата после манипуляции
        return_speed = manipulation['return_index'] - manipulation['breakout_index']
        if return_speed <= 3:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_amd_entry(self, consolidation: Dict, manipulation: Dict, direction: str) -> Dict:
        """Расчет зоны входа для паттерна AMD"""
        if direction == 'bullish':
            entry_price = consolidation['low'] + (consolidation['high'] - consolidation['low']) * 0.3
            stop_loss = consolidation['low'] - manipulation['max_penetration']
            take_profit = consolidation['high'] + (consolidation['high'] - consolidation['low'])
        else:  # bearish
            entry_price = consolidation['high'] - (consolidation['high'] - consolidation['low']) * 0.3
            stop_loss = consolidation['high'] + manipulation['max_penetration']
            take_profit = consolidation['low'] - (consolidation['high'] - consolidation['low'])
        
        return {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def detect_healthy_orderflow(self, data: pd.DataFrame, direction: str, periods: int = 10) -> Dict:
        """
        Детекция здорового Orderflow
        """
        if len(data) < periods:
            return {'healthy': False, 'score': 0.0}
        
        recent_data = data.tail(periods)
        score = 0.0
        
        if direction == 'bullish':
            # Для бычьего движения
            # 1. Последовательные более высокие максимумы и минимумы
            hh_count = 0
            hl_count = 0
            
            for i in range(1, len(recent_data)):
                if recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1]:
                    hh_count += 1
                if recent_data['low'].iloc[i] > recent_data['low'].iloc[i-1]:
                    hl_count += 1
            
            structure_score = (hh_count + hl_count) / (2 * (periods - 1))
            score += structure_score * 0.4
            
            # 2. Преобладание бычьих свечей
            bullish_candles = sum(1 for i in range(len(recent_data)) 
                                if recent_data['close'].iloc[i] > recent_data['open'].iloc[i])
            bullish_ratio = bullish_candles / len(recent_data)
            score += bullish_ratio * 0.3
            
            # 3. Сильные тела свечей (малые тени)
            body_strength = 0
            for i in range(len(recent_data)):
                candle = recent_data.iloc[i]
                body_size = abs(candle['close'] - candle['open'])
                total_range = candle['high'] - candle['low']
                if total_range > 0:
                    body_strength += body_size / total_range
            
            avg_body_strength = body_strength / len(recent_data)
            score += avg_body_strength * 0.3
            
        else:  # bearish
            # Аналогично для медвежьего движения
            lh_count = 0
            ll_count = 0
            
            for i in range(1, len(recent_data)):
                if recent_data['high'].iloc[i] < recent_data['high'].iloc[i-1]:
                    lh_count += 1
                if recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1]:
                    ll_count += 1
            
            structure_score = (lh_count + ll_count) / (2 * (periods - 1))
            score += structure_score * 0.4
            
            bearish_candles = sum(1 for i in range(len(recent_data)) 
                                if recent_data['close'].iloc[i] < recent_data['open'].iloc[i])
            bearish_ratio = bearish_candles / len(recent_data)
            score += bearish_ratio * 0.3
            
            body_strength = 0
            for i in range(len(recent_data)):
                candle = recent_data.iloc[i]
                body_size = abs(candle['close'] - candle['open'])
                total_range = candle['high'] - candle['low']
                if total_range > 0:
                    body_strength += body_size / total_range
            
            avg_body_strength = body_strength / len(recent_data)
            score += avg_body_strength * 0.3
        
        return {
            'healthy': score > 0.7,
            'score': score,
            'direction': direction
        }

class TimeframeSynchronizer:
    """Синхронизатор сигналов между таймфреймами"""
    
    def __init__(self, market_config: MarketConfig):
        self.config = market_config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def synchronize_analysis(self, analyses: Dict[str, TimeframeAnalysis]) -> Dict:
        """
        Синхронизация анализа между таймфреймами
        Возвращает общую оценку и рекомендации
        """
        context_tf = analyses.get('context')
        signal_tf = analyses.get('signal') 
        execution_tf = analyses.get('execution')
        
        if not all([context_tf, signal_tf, execution_tf]):
            return {'synchronized': False, 'reason': 'Недостаточно данных для анализа'}
        
        # Проверка согласованности трендов
        trends = [context_tf.trend_direction, signal_tf.trend_direction, execution_tf.trend_direction]
        
        # Полная синхронизация (все таймфреймы в одном направлении)
        if len(set(trends)) == 1 and trends[0] != 'sideways':
            return {
                'synchronized': True,
                'type': 'full_sync',
                'direction': trends[0],
                'confidence': self._calculate_sync_confidence(analyses),
                'quality_score': self._calculate_quality_score(analyses)
            }
        
        # Частичная синхронизация (контекст + сигнал согласованы)
        elif (context_tf.trend_direction == signal_tf.trend_direction and 
              context_tf.trend_direction != 'sideways'):
            return {
                'synchronized': True,
                'type': 'partial_sync',
                'direction': context_tf.trend_direction,
                'confidence': self._calculate_sync_confidence(analyses) * 0.8,
                'quality_score': self._calculate_quality_score(analyses)
            }
        
        # Противоречивые сигналы
        else:
            return {
                'synchronized': False,
                'type': 'conflicting',
                'reason': f'Противоречивые тренды: {trends}',
                'recommendation': 'Ожидать прояснения ситуации'
            }
    
    def _calculate_sync_confidence(self, analyses: Dict[str, TimeframeAnalysis]) -> float:
        """Расчет уверенности в синхронизации"""
        confidence = 0.0
        
        # Качество структуры на всех таймфреймах
        avg_structure_quality = np.mean([
            analysis.structure_quality for analysis in analyses.values()
        ])
        confidence += avg_structure_quality * 0.4
        
        # Качество Orderflow
        avg_orderflow = np.mean([
            abs(analysis.orderflow_score) for analysis in analyses.values()
        ])
        confidence += avg_orderflow * 0.3
        
        # Количество подтверждающих факторов
        confirmation_factors = 0
        for analysis in analyses.values():
            if len(analysis.imbalances) > 0:
                confirmation_factors += 1
            if len(analysis.fractals['up']) > 0 or len(analysis.fractals['down']) > 0:
                confirmation_factors += 1
        
        confidence += min(confirmation_factors / 6, 0.3)
        
        return min(confidence, 1.0)
    
    def _calculate_quality_score(self, analyses: Dict[str, TimeframeAnalysis]) -> float:
        """Расчет общего качества анализа"""
        scores = []
        
        for analysis in analyses.values():
            score = 0.0
            
            # Качество структуры
            score += analysis.structure_quality * 0.4
            
            # Сила Orderflow
            score += abs(analysis.orderflow_score) * 0.3
            
            # Наличие ключевых уровней
            if len(analysis.key_levels) > 0:
                score += 0.15
            
            # Наличие имбалансов
            if len(analysis.imbalances) > 0:
                score += 0.15
            
            scores.append(score)
        
        return np.mean(scores)

