"""
Система управления рисками и позициями

Реализует продвинутые алгоритмы управления рисками, расчета размера позиций
и динамического управления стоп-лоссами и тейк-профитами
"""

from trading_system_base import *
from multi_timeframe_analysis import *
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import math

@dataclass
class RiskMetrics:
    """Метрики риска для портфеля"""
    total_risk_exposure: float
    position_correlation: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    avg_risk_reward: float
    var_95: float  # Value at Risk 95%

@dataclass
class PositionSizing:
    """Результат расчета размера позиции"""
    position_size: float
    risk_amount: float
    max_loss_pips: float
    position_value: float
    leverage_used: float
    margin_required: float

class RiskManager:
    """Менеджер рисков"""
    
    def __init__(self, market_config: MarketConfig):
        self.config = market_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.trade_history: List[Dict] = []
        self.current_positions: List[Position] = []
        
    def calculate_position_size(self, signal: Signal, account_balance: float, 
                              current_price: float, atr_value: float) -> PositionSizing:
        """
        Расчет размера позиции с учетом волатильности и типа рынка
        """
        # Базовый риск на сделку
        base_risk = account_balance * self.config.risk_per_trade
        
        # Корректировка риска в зависимости от волатильности
        volatility_adjusted_risk = self._adjust_risk_for_volatility(
            base_risk, atr_value, current_price
        )
        
        # Корректировка риска в зависимости от уверенности в сигнале
        confidence_adjusted_risk = volatility_adjusted_risk * signal.confidence
        
        # Корректировка риска в зависимости от корреляции с существующими позициями
        correlation_adjusted_risk = self._adjust_risk_for_correlation(
            confidence_adjusted_risk, signal
        )
        
        # Расчет размера позиции
        stop_distance = abs(current_price - signal.stop_loss)
        
        if stop_distance == 0:
            self.logger.error("Расстояние до стоп-лосса равно нулю")
            return PositionSizing(0, 0, 0, 0, 0, 0)
        
        # Для Forex: размер позиции в лотах
        # Для Crypto: размер позиции в базовой валюте
        if self.config.market_type == MarketType.FOREX:
            # 1 лот = 100,000 единиц базовой валюты
            # Размер позиции = Риск / (Расстояние до стопа в пипсах * Стоимость пипса)
            pip_value = self._calculate_pip_value(current_price)
            stop_distance_pips = stop_distance * 10000  # Конвертация в пипсы
            position_size = correlation_adjusted_risk / (stop_distance_pips * pip_value)
            
        else:  # CRYPTO
            # Размер позиции = Риск / Расстояние до стопа
            position_size = correlation_adjusted_risk / stop_distance
        
        # Применение максимальных лимитов
        max_position_size = self._calculate_max_position_size(account_balance)
        position_size = min(position_size, max_position_size)
        
        # Расчет дополнительных метрик
        position_value = position_size * current_price
        leverage_used = position_value / account_balance if account_balance > 0 else 0
        margin_required = position_value / 10 if self.config.market_type == MarketType.FOREX else position_value
        
        return PositionSizing(
            position_size=position_size,
            risk_amount=correlation_adjusted_risk,
            max_loss_pips=stop_distance_pips if self.config.market_type == MarketType.FOREX else stop_distance,
            position_value=position_value,
            leverage_used=leverage_used,
            margin_required=margin_required
        )
    
    def _adjust_risk_for_volatility(self, base_risk: float, atr_value: float, current_price: float) -> float:
        """Корректировка риска в зависимости от волатильности"""
        if atr_value == 0 or current_price == 0:
            return base_risk
        
        # Нормализованная волатильность (ATR / цена)
        normalized_volatility = atr_value / current_price
        
        # Для Forex: нормальная волатильность ~0.01 (1%)
        # Для Crypto: нормальная волатильность ~0.05 (5%)
        normal_volatility = 0.01 if self.config.market_type == MarketType.FOREX else 0.05
        
        # Коэффициент корректировки волатильности
        volatility_ratio = normalized_volatility / normal_volatility
        
        # Снижаем риск при высокой волатильности
        if volatility_ratio > 1.5:
            adjustment_factor = 0.7
        elif volatility_ratio > 1.2:
            adjustment_factor = 0.85
        elif volatility_ratio < 0.5:
            adjustment_factor = 1.2
        elif volatility_ratio < 0.8:
            adjustment_factor = 1.1
        else:
            adjustment_factor = 1.0
        
        return base_risk * adjustment_factor
    
    def _adjust_risk_for_correlation(self, base_risk: float, signal: Signal) -> float:
        """Корректировка риска в зависимости от корреляции с существующими позициями"""
        if not self.current_positions:
            return base_risk
        
        # Простая проверка корреляции (можно расширить)
        same_direction_positions = sum(1 for pos in self.current_positions 
                                     if pos.side == signal.signal_type.value)
        
        # Снижаем риск при наличии коррелирующих позиций
        if same_direction_positions >= 2:
            return base_risk * 0.5
        elif same_direction_positions == 1:
            return base_risk * 0.75
        
        return base_risk
    
    def _calculate_pip_value(self, current_price: float) -> float:
        """Расчет стоимости пипса для Forex"""
        # Упрощенный расчет для основных валютных пар
        # В реальной системе нужно учитывать конкретную валютную пару
        return 10.0  # $10 за пипс для стандартного лота
    
    def _calculate_max_position_size(self, account_balance: float) -> float:
        """Расчет максимального размера позиции"""
        # Максимум 10% от баланса в одной позиции
        max_risk_per_position = account_balance * 0.1
        
        if self.config.market_type == MarketType.FOREX:
            # Максимум 10 лотов
            return min(10.0, max_risk_per_position / 100000)
        else:  # CRYPTO
            # Максимум 10% от баланса
            return max_risk_per_position
    
    def calculate_stop_loss(self, entry_price: float, signal_direction: str, 
                          atr_value: float, key_levels: List[float]) -> float:
        """
        Расчет уровня стоп-лосса с учетом ATR и ключевых уровней
        """
        atr_multiplier = self.config.stop_loss_atr_multiplier
        
        if signal_direction == 'buy':
            # Для покупки: стоп ниже входа
            atr_stop = entry_price - (atr_value * atr_multiplier)
            
            # Поиск ближайшего уровня поддержки
            support_levels = [level for level in key_levels if level < entry_price]
            if support_levels:
                nearest_support = max(support_levels)
                # Стоп чуть ниже уровня поддержки
                level_stop = nearest_support - (atr_value * 0.5)
                # Выбираем более консервативный стоп
                return max(atr_stop, level_stop)
            
            return atr_stop
            
        else:  # sell
            # Для продажи: стоп выше входа
            atr_stop = entry_price + (atr_value * atr_multiplier)
            
            # Поиск ближайшего уровня сопротивления
            resistance_levels = [level for level in key_levels if level > entry_price]
            if resistance_levels:
                nearest_resistance = min(resistance_levels)
                # Стоп чуть выше уровня сопротивления
                level_stop = nearest_resistance + (atr_value * 0.5)
                # Выбираем более консервативный стоп
                return min(atr_stop, level_stop)
            
            return atr_stop
    
    def calculate_take_profits(self, entry_price: float, stop_loss: float, 
                             signal_direction: str, imbalances: List[Dict]) -> List[float]:
        """
        Расчет уровней тейк-профита с учетом имбалансов и соотношения риск/прибыль
        """
        risk_distance = abs(entry_price - stop_loss)
        take_profits = []
        
        # Базовые уровни тейк-профита на основе соотношения риск/прибыль
        for multiplier in self.config.take_profit_levels:
            if signal_direction == 'buy':
                tp_level = entry_price + (risk_distance * multiplier)
            else:  # sell
                tp_level = entry_price - (risk_distance * multiplier)
            
            take_profits.append(tp_level)
        
        # Корректировка с учетом имбалансов
        relevant_imbalances = self._find_relevant_imbalances(
            entry_price, signal_direction, imbalances
        )
        
        if relevant_imbalances:
            # Добавляем уровни имбалансов как дополнительные цели
            for imbalance in relevant_imbalances[:2]:  # Максимум 2 дополнительных уровня
                if signal_direction == 'buy':
                    imbalance_target = imbalance['upper_level']
                    if imbalance_target > entry_price:
                        take_profits.append(imbalance_target)
                else:  # sell
                    imbalance_target = imbalance['lower_level']
                    if imbalance_target < entry_price:
                        take_profits.append(imbalance_target)
        
        # Сортировка и ограничение количества уровней
        if signal_direction == 'buy':
            take_profits = sorted([tp for tp in take_profits if tp > entry_price])
        else:
            take_profits = sorted([tp for tp in take_profits if tp < entry_price], reverse=True)
        
        return take_profits[:5]  # Максимум 5 уровней
    
    def _find_relevant_imbalances(self, entry_price: float, direction: str, 
                                imbalances: List[Dict]) -> List[Dict]:
        """Поиск релевантных имбалансов для установки целей"""
        relevant = []
        
        for imbalance in imbalances:
            if direction == 'buy' and imbalance['type'] == 'bullish':
                if imbalance['lower_level'] > entry_price:
                    relevant.append(imbalance)
            elif direction == 'sell' and imbalance['type'] == 'bearish':
                if imbalance['upper_level'] < entry_price:
                    relevant.append(imbalance)
        
        # Сортировка по силе имбаланса
        return sorted(relevant, key=lambda x: x['strength'], reverse=True)
    
    def update_trailing_stop(self, position: Position, current_price: float, 
                           atr_value: float) -> float:
        """
        Обновление трейлинг стопа
        """
        if position.side == 'buy':
            # Для длинной позиции: стоп движется вверх
            new_stop = current_price - (atr_value * self.config.stop_loss_atr_multiplier)
            return max(position.stop_loss, new_stop)
        else:
            # Для короткой позиции: стоп движется вниз
            new_stop = current_price + (atr_value * self.config.stop_loss_atr_multiplier)
            return min(position.stop_loss, new_stop)
    
    def calculate_portfolio_risk(self) -> RiskMetrics:
        """Расчет метрик риска портфеля"""
        if not self.trade_history:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Общий риск экспозиции
        total_risk = sum(abs(pos.unrealized_pnl) for pos in self.current_positions)
        
        # Корреляция позиций (упрощенная)
        same_direction = sum(1 for pos in self.current_positions if pos.side == 'buy')
        total_positions = len(self.current_positions)
        correlation = abs(same_direction - total_positions/2) / max(total_positions/2, 1) if total_positions > 0 else 0
        
        # Анализ исторических сделок
        if len(self.trade_history) >= 10:
            returns = [trade.get('return', 0) for trade in self.trade_history]
            
            # Максимальная просадка
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
            
            # Коэффициент Шарпа (упрощенный)
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            
            # Винрейт
            winning_trades = sum(1 for ret in returns if ret > 0)
            win_rate = winning_trades / len(returns)
            
            # Среднее соотношение риск/прибыль
            profits = [ret for ret in returns if ret > 0]
            losses = [abs(ret) for ret in returns if ret < 0]
            avg_rr = (np.mean(profits) / np.mean(losses)) if losses and profits else 0
            
            # VaR 95%
            var_95 = np.percentile(returns, 5) if len(returns) >= 20 else 0
            
        else:
            max_drawdown = sharpe_ratio = win_rate = avg_rr = var_95 = 0
        
        return RiskMetrics(
            total_risk_exposure=total_risk,
            position_correlation=correlation,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            avg_risk_reward=avg_rr,
            var_95=var_95
        )
    
    def should_close_position(self, position: Position, current_price: float, 
                            market_analysis: Dict) -> Tuple[bool, str]:
        """
        Определение необходимости закрытия позиции
        """
        # Проверка стоп-лосса
        if position.side == 'buy' and current_price <= position.stop_loss:
            return True, "stop_loss"
        elif position.side == 'sell' and current_price >= position.stop_loss:
            return True, "stop_loss"
        
        # Проверка тейк-профита
        for tp_level in position.take_profit:
            if position.side == 'buy' and current_price >= tp_level:
                return True, "take_profit"
            elif position.side == 'sell' and current_price <= tp_level:
                return True, "take_profit"
        
        # Проверка изменения рыночной структуры
        if market_analysis.get('synchronized', False):
            sync_direction = market_analysis.get('direction', '')
            if ((position.side == 'buy' and sync_direction == 'bearish') or
                (position.side == 'sell' and sync_direction == 'bullish')):
                return True, "structure_change"
        
        # Проверка времени удержания позиции
        time_held = datetime.now() - position.timestamp
        max_hold_time = timedelta(days=7) if self.config.market_type == MarketType.FOREX else timedelta(days=3)
        
        if time_held > max_hold_time:
            return True, "time_limit"
        
        return False, ""

class PositionManager:
    """Менеджер позиций"""
    
    def __init__(self, market_config: MarketConfig):
        self.config = market_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.risk_manager = RiskManager(market_config)
        self.positions: Dict[str, Position] = {}
        
    def open_position(self, signal: Signal, account_balance: float, 
                     current_price: float, atr_value: float, symbol: str) -> Optional[Position]:
        """Открытие новой позиции"""
        
        # Проверка лимитов
        if len(self.positions) >= self.config.max_positions:
            self.logger.warning("Достигнуто максимальное количество позиций")
            return None
        
        # Расчет размера позиции
        position_sizing = self.risk_manager.calculate_position_size(
            signal, account_balance, current_price, atr_value
        )
        
        if position_sizing.position_size <= 0:
            self.logger.warning("Размер позиции равен нулю")
            return None
        
        # Создание позиции
        position = Position(
            symbol=symbol,
            side=signal.signal_type.value,
            size=position_sizing.position_size,
            entry_price=signal.entry_price,
            current_price=current_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            unrealized_pnl=0.0,
            timestamp=datetime.now()
        )
        
        # Добавление в портфель
        position_id = f"{symbol}_{signal.signal_type.value}_{datetime.now().timestamp()}"
        self.positions[position_id] = position
        self.risk_manager.current_positions.append(position)
        
        self.logger.info(f"Открыта позиция {position_id}: {position.side} {position.size} {symbol} @ {position.entry_price}")
        
        return position
    
    def close_position(self, position_id: str, current_price: float, reason: str = "") -> Optional[Dict]:
        """Закрытие позиции"""
        if position_id not in self.positions:
            self.logger.error(f"Позиция {position_id} не найдена")
            return None
        
        position = self.positions[position_id]
        
        # Расчет P&L
        if position.side == 'buy':
            pnl = (current_price - position.entry_price) * position.size
        else:  # sell
            pnl = (position.entry_price - current_price) * position.size
        
        # Создание записи о сделке
        trade_record = {
            'symbol': position.symbol,
            'side': position.side,
            'size': position.size,
            'entry_price': position.entry_price,
            'exit_price': current_price,
            'pnl': pnl,
            'return': pnl / (position.entry_price * position.size) if position.entry_price * position.size > 0 else 0,
            'duration': datetime.now() - position.timestamp,
            'reason': reason,
            'timestamp': datetime.now()
        }
        
        # Удаление позиции
        del self.positions[position_id]
        self.risk_manager.current_positions = [p for p in self.risk_manager.current_positions if p != position]
        self.risk_manager.trade_history.append(trade_record)
        
        self.logger.info(f"Закрыта позиция {position_id}: P&L = {pnl:.2f}, причина = {reason}")
        
        return trade_record
    
    def update_positions(self, current_prices: Dict[str, float], market_analyses: Dict[str, Dict]) -> List[Dict]:
        """Обновление всех позиций"""
        closed_trades = []
        positions_to_close = []
        
        for position_id, position in self.positions.items():
            symbol = position.symbol
            current_price = current_prices.get(symbol, position.current_price)
            
            # Обновление текущей цены и P&L
            position.current_price = current_price
            if position.side == 'buy':
                position.unrealized_pnl = (current_price - position.entry_price) * position.size
            else:
                position.unrealized_pnl = (position.entry_price - current_price) * position.size
            
            # Проверка необходимости закрытия
            market_analysis = market_analyses.get(symbol, {})
            should_close, reason = self.risk_manager.should_close_position(
                position, current_price, market_analysis
            )
            
            if should_close:
                positions_to_close.append((position_id, current_price, reason))
        
        # Закрытие позиций
        for position_id, price, reason in positions_to_close:
            trade_record = self.close_position(position_id, price, reason)
            if trade_record:
                closed_trades.append(trade_record)
        
        return closed_trades
    
    def get_portfolio_summary(self) -> Dict:
        """Получение сводки по портфелю"""
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_position_value = sum(pos.size * pos.current_price for pos in self.positions.values())
        
        risk_metrics = self.risk_manager.calculate_portfolio_risk()
        
        return {
            'total_positions': len(self.positions),
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_position_value': total_position_value,
            'risk_metrics': risk_metrics,
            'positions': list(self.positions.values())
        }

