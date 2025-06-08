"""
Главный файл алгоритмической торговой системы
Автор: Manus AI (Алгопрограммист)

Демонстрация работы Forex и Crypto стратегий
"""

from trading_system_base import *
from multi_timeframe_analysis import *
from risk_position_management import *
from forex_strategy import ForexStrategy
from crypto_strategy import CryptoStrategy
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def generate_sample_data(timeframe: TimeFrame, periods: int = 1000) -> pd.DataFrame:
    """Генерация примерных данных для тестирования"""
    
    # Базовые параметры в зависимости от таймфрейма
    if timeframe == TimeFrame.M5:
        base_price = 50000.0  # Для криптовалют
        volatility = 0.02
    elif timeframe == TimeFrame.H1:
        base_price = 1.1000   # Для Forex
        volatility = 0.005
    else:
        base_price = 1.1000
        volatility = 0.001
    
    # Генерация случайных данных с трендом
    np.random.seed(42)
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=periods//24), 
                         periods=periods, freq='5min')
    
    # Генерация цен с трендом и волатильностью
    returns = np.random.normal(0, volatility, periods)
    trend = np.linspace(0, 0.1, periods)  # Восходящий тренд
    
    prices = [base_price]
    for i in range(1, periods):
        new_price = prices[-1] * (1 + returns[i] + trend[i]/periods)
        prices.append(new_price)
    
    # Создание OHLCV данных
    data = []
    for i, price in enumerate(prices):
        high = price * (1 + abs(np.random.normal(0, volatility/2)))
        low = price * (1 - abs(np.random.normal(0, volatility/2)))
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'timestamp': dates[i],
            'open': open_price,
            'high': max(open_price, high, close_price),
            'low': min(open_price, low, close_price),
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df

def test_forex_strategy():
    """Тестирование Forex стратегии"""
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ FOREX СТРАТЕГИИ")
    print("="*60)
    
    # Создание стратегии
    forex_strategy = ForexStrategy()
    
    # Генерация тестовых данных для разных таймфреймов
    test_data = {
        TimeFrame.H4: generate_sample_data(TimeFrame.H4, 500),
        TimeFrame.H1: generate_sample_data(TimeFrame.H1, 1000),
        TimeFrame.M15: generate_sample_data(TimeFrame.M15, 2000)
    }
    
    print(f"Сгенерированы данные для {len(test_data)} таймфреймов")
    
    # Анализ рынка
    symbol = "EUR/USD"
    signal = forex_strategy.analyze_market(test_data, symbol)
    
    if signal:
        print(f"\n✅ СИГНАЛ НАЙДЕН для {symbol}:")
        print(f"   Тип: {signal.signal_type.value}")
        print(f"   Вход: {signal.entry_price:.5f}")
        print(f"   Стоп: {signal.stop_loss:.5f}")
        print(f"   Цели: {[f'{tp:.5f}' for tp in signal.take_profit]}")
        print(f"   Уверенность: {signal.confidence:.2%}")
        print(f"   R/R: {signal.risk_reward_ratio:.2f}")
        
        # Тестирование расчета размера позиции
        account_balance = 10000.0
        position_size = forex_strategy.calculate_position_size(signal, account_balance)
        print(f"   Размер позиции: {position_size:.2f} лотов")
        
    else:
        print(f"\n❌ Сигнал не найден для {symbol}")
    
    # Получение торговых рекомендаций
    recommendations = forex_strategy.get_trading_recommendations()
    print(f"\n📊 ТОРГОВЫЕ РЕКОМЕНДАЦИИ:")
    print(f"   Текущая сессия: {recommendations['current_session']['name']}")
    print(f"   Рекомендуемые пары: {recommendations['recommended_pairs']}")
    print(f"   Уровень риска: {recommendations['risk_level']}")
    print(f"   Здоровье портфеля: {recommendations['portfolio_health']}")
    
    if recommendations['avoid_trading']:
        print(f"   ⚠️  Избегать торговли: {recommendations['avoid_reason']}")

def test_crypto_strategy():
    """Тестирование Crypto стратегии"""
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ CRYPTO СТРАТЕГИИ")
    print("="*60)
    
    # Создание стратегии
    crypto_strategy = CryptoStrategy()
    
    # Генерация тестовых данных с высокой волатильностью
    test_data = {
        TimeFrame.H1: generate_sample_data(TimeFrame.H1, 500),
        TimeFrame.M15: generate_sample_data(TimeFrame.M15, 1000),
        TimeFrame.M5: generate_sample_data(TimeFrame.M5, 2000)
    }
    
    print(f"Сгенерированы данные для {len(test_data)} таймфреймов")
    
    # Анализ рынка
    symbol = "BTC/USDT"
    signal = crypto_strategy.analyze_market(test_data, symbol)
    
    if signal:
        print(f"\n✅ СИГНАЛ НАЙДЕН для {symbol}:")
        print(f"   Тип: {signal.signal_type.value}")
        print(f"   Вход: ${signal.entry_price:,.2f}")
        print(f"   Стоп: ${signal.stop_loss:,.2f}")
        print(f"   Цели: {[f'${tp:,.2f}' for tp in signal.take_profit]}")
        print(f"   Уверенность: {signal.confidence:.2%}")
        print(f"   R/R: {signal.risk_reward_ratio:.2f}")
        
        # Тестирование расчета размера позиции
        account_balance = 50000.0
        position_size = crypto_strategy.calculate_position_size(signal, account_balance)
        print(f"   Размер позиции: {position_size:.6f} BTC")
        
    else:
        print(f"\n❌ Сигнал не найден для {symbol}")
    
    # Получение состояния рынка
    market_state = crypto_strategy.get_crypto_market_state()
    print(f"\n📊 СОСТОЯНИЕ КРИПТО РЫНКА:")
    print(f"   Фактор новостного риска: {market_state['news_risk_factor']:.2%}")
    print(f"   Период высокого влияния: {market_state['high_impact_period']}")
    print(f"   Причина: {market_state['impact_reason']}")
    print(f"   Использовано сделок: {market_state['daily_trades_used']}/{crypto_strategy.max_daily_trades}")
    print(f"   Рекомендация: {market_state['recommended_action']}")

def demonstrate_risk_management():
    """Демонстрация системы управления рисками"""
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ УПРАВЛЕНИЯ РИСКАМИ")
    print("="*60)
    
    # Создание конфигурации для Forex
    forex_config = MarketConfig(MarketType.FOREX)
    risk_manager = RiskManager(forex_config)
    
    # Создание примерного сигнала
    signal = Signal(
        timestamp=datetime.now(),
        signal_type=SignalType.BUY,
        entry_price=1.1000,
        stop_loss=1.0950,
        take_profit=[1.1050, 1.1100, 1.1150],
        confidence=0.75,
        timeframe_analysis={},
        risk_reward_ratio=2.0
    )
    
    # Расчет размера позиции
    account_balance = 10000.0
    current_price = 1.1000
    atr_value = 0.0020
    
    position_sizing = risk_manager.calculate_position_size(
        signal, account_balance, current_price, atr_value
    )
    
    print(f"💰 РАСЧЕТ РАЗМЕРА ПОЗИЦИИ:")
    print(f"   Баланс счета: ${account_balance:,.2f}")
    print(f"   Риск на сделку: {forex_config.risk_per_trade:.1%}")
    print(f"   Размер позиции: {position_sizing.position_size:.2f} лотов")
    print(f"   Сумма риска: ${position_sizing.risk_amount:.2f}")
    print(f"   Максимальный убыток: {position_sizing.max_loss_pips:.1f} пипсов")
    print(f"   Стоимость позиции: ${position_sizing.position_value:,.2f}")
    print(f"   Используемое плечо: {position_sizing.leverage_used:.1f}x")
    print(f"   Требуемая маржа: ${position_sizing.margin_required:,.2f}")

def demonstrate_pattern_detection():
    """Демонстрация детекции паттернов"""
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ ДЕТЕКЦИИ ПАТТЕРНОВ")
    print("="*60)
    
    # Создание детектора паттернов
    config = MarketConfig(MarketType.FOREX)
    pattern_detector = PatternDetector(config)
    
    # Генерация данных с паттерном
    data = generate_sample_data(TimeFrame.M15, 200)
    
    # Создание примерного анализа таймфрейма
    analysis = TimeframeAnalysis(
        timeframe=TimeFrame.M15,
        trend_direction='bullish',
        structure_quality=0.8,
        orderflow_score=0.75,
        key_levels=[1.0950, 1.1000, 1.1050, 1.1100],
        premium_discount_zones={'premium': (1.1080, 1.1120), 'discount': (1.0980, 1.1020)},
        imbalances=[
            {'type': 'bullish', 'upper_level': 1.1030, 'lower_level': 1.1020, 'strength': 0.7},
            {'type': 'bullish', 'upper_level': 1.1070, 'lower_level': 1.1060, 'strength': 0.6}
        ],
        fractals={'resistance': [1.1100], 'support': [1.0950]}
    )
    
    # Детекция паттерна AMD
    amd_pattern = pattern_detector.detect_amd_pattern(data, analysis)
    
    if amd_pattern:
        print(f"🎯 ПАТТЕРН AMD ОБНАРУЖЕН:")
        print(f"   Тип паттерна: {amd_pattern['pattern']}")
        print(f"   Направление: {amd_pattern['distribution_direction']}")
        print(f"   Уверенность: {amd_pattern['confidence']:.2%}")
        print(f"   Зона входа: {amd_pattern['entry_zone']['entry_price']:.5f}")
        print(f"   Стоп-лосс: {amd_pattern['entry_zone']['stop_loss']:.5f}")
    else:
        print("❌ Паттерн AMD не обнаружен")
    
    # Детекция здорового Orderflow
    orderflow = pattern_detector.detect_healthy_orderflow(data, 'bullish')
    
    print(f"\n📈 АНАЛИЗ ORDERFLOW:")
    print(f"   Здоровый поток: {'Да' if orderflow['healthy'] else 'Нет'}")
    print(f"   Оценка: {orderflow['score']:.2%}")
    print(f"   Детали: {orderflow.get('details', 'Нет дополнительных деталей')}")

def main():
    """Главная функция демонстрации"""
    print("🚀 АЛГОРИТМИЧЕСКАЯ ТОРГОВАЯ СИСТЕМА")
    print("Автор: Manus AI (Алгопрограммист)")
    print("Версии: Forex (низкая волатильность) + Crypto (высокая волатильность)")
    
    try:
        # Тестирование компонентов
        demonstrate_pattern_detection()
        demonstrate_risk_management()
        
        # Тестирование стратегий
        test_forex_strategy()
        test_crypto_strategy()
        
        print("\n" + "="*60)
        print("✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ УСПЕШНО")
        print("="*60)
        
        print("\n📋 КРАТКАЯ СВОДКА СИСТЕМЫ:")
        print("   • Многотаймфреймный анализ (3 таймфрейма)")
        print("   • Детекция паттернов AMD и здорового Orderflow")
        print("   • Адаптивное управление рисками")
        print("   • Forex: учет сессионности и корреляций")
        print("   • Crypto: адаптация под высокую волатильность")
        print("   • Автоматическое управление позициями")
        print("   • Система мониторинга новостей")
        
    except Exception as e:
        print(f"\n❌ ОШИБКА ПРИ ТЕСТИРОВАНИИ: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

