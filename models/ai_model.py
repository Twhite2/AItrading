import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

class TradingAIModel:
    def __init__(self, model_path: str, confidence_threshold: float = 0.75):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.confidence_threshold = confidence_threshold

    def prepare_input(self, candles: pd.DataFrame) -> str:
        """Convert candle data to model input format."""
        # Format last 5 candles for input
        candle_desc = []
        for _, candle in candles.tail(5).iterrows():
            candle_desc.append(
                f"O:{candle['open']:.5f} H:{candle['high']:.5f} "
                f"L:{candle['low']:.5f} C:{candle['close']:.5f} "
                f"V:{candle['volume']:.2f}"
            )
        return " | ".join(candle_desc)

    def predict(self, candles: pd.DataFrame) -> Tuple[str, float]:
        """Generate trading prediction from candle data."""
        input_text = self.prepare_input(candles)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=50,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract action and confidence
        # Format expected: "ACTION:BUY|CONFIDENCE:0.85"
        try:
            parts = prediction.split("|")
            action = parts[0].split(":")[1].strip()
            confidence = float(parts[1].split(":")[1].strip())
            
            if confidence >= self.confidence_threshold:
                return action, confidence
            return "HOLD", confidence
        except:
            return "HOLD", 0.0

    def analyze_market_condition(self, candles: pd.DataFrame) -> Dict:
        """Analyze current market conditions for trade documentation."""
        return {
            "trend": self._calculate_trend(candles),
            "volatility": self._calculate_volatility(candles),
            "volume_profile": self._analyze_volume(candles),
            "key_levels": self._identify_key_levels(candles)
        }

    def _calculate_trend(self, candles: pd.DataFrame) -> str:
        sma20 = candles['close'].rolling(20).mean().iloc[-1]
        sma50 = candles['close'].rolling(50).mean().iloc[-1]
        current_price = candles['close'].iloc[-1]
        
        if current_price > sma20 > sma50:
            return "STRONG_UPTREND"
        elif current_price > sma20:
            return "UPTREND"
        elif current_price < sma20 < sma50:
            return "STRONG_DOWNTREND"
        elif current_price < sma20:
            return "DOWNTREND"
        return "SIDEWAYS"