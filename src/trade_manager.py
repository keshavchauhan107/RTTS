# src/trade_manager.py
import json
import math
import os
import numpy as np

from typing import Dict, Any, Tuple
from src.logger import logger
from src.env_reader import get_env_value

STATE_FILE = get_env_value("TRADE_STATE_FILE")
MAX_POSITION_PERCENT = float(get_env_value("MAX_POSITION_PERCENT")) 
AGGRESSIVENESS = float(get_env_value("TRADE_AGGRESSIVENESS"))  
MIN_TRADE_VALUE = float(get_env_value("MIN_TRADE_VALUE"))    
TRANSACTION_COST_PCT = float(get_env_value("TRANSACTION_COST_PCT"))
SELL_ALL_THRESHOLD = float(get_env_value("SELL_ALL_THRESHOLD"))
HOLD_THRESHOLD = float(get_env_value("HOLD_THRESHOLD")) 

def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
        except Exception as e:
            logger.error("Failed to load state file, initializing new state: %s", e, exc_info=True)
            state = {}
    else:
        state = {}

    # .setdefault(key, default) only sets the key if it’s missing.
    state.setdefault("Money", float(get_env_value("INITIAL_MONEY")))
    state.setdefault("AvailStocks", int(get_env_value("INITIAL_STOCKS")))
    return state

def save_state(state: Dict[str, Any]) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
        logger.info("State saved: %s", state)
    except Exception as e:
        logger.error("Failed to save state: %s", e, exc_info=True)

# Calculates your total account equity (net worth)
def get_equity(state: Dict[str, Any], price: float) -> float:
    try:
        return float(state.get("Money")) + float(state.get("AvailStocks")) * float(price)
    except Exception as e:
        logger.error("Failed to get Equity: %s", e, exc_info=True)

# Cost amount for a trade of Notional Value
def apply_transaction_cost(value: float) -> float:
    return abs(value) * TRANSACTION_COST_PCT


# -------------------------
# Decision & execution
# -------------------------
def decide_and_execute(probs: np.ndarray, price: float, state: Dict[str, Any], params: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Decide trade action from model probabilities and apply to state (simulation).
    probs: numpy array like [[p_hold, p_buy, p_sell]] or shape (3,)
    price: current price per share
    state: dict with 'Money' and 'AvailStocks'
    params: optional overrides for trading params

    Returns (new_state, info) where info contains what was done for logging/inspection.
    """
    # allow params override
    max_pos_pct = params.get("MAX_POSITION_PERCENT", MAX_POSITION_PERCENT) if params else MAX_POSITION_PERCENT
    aggressiveness = params.get("AGGRESSIVENESS", AGGRESSIVENESS) if params else AGGRESSIVENESS
    min_trade_value = params.get("MIN_TRADE_VALUE", MIN_TRADE_VALUE) if params else MIN_TRADE_VALUE
    sell_all_threshold = params.get("SELL_ALL_THRESHOLD", SELL_ALL_THRESHOLD) if params else SELL_ALL_THRESHOLD
    hold_threshold = params.get("HOLD_THRESHOLD", HOLD_THRESHOLD) if params else HOLD_THRESHOLD

    # Normalize probs shape
    p = np.asarray(probs).reshape(-1)
    if p.size == 3:
        p_hold, p_buy, p_sell = float(p[0]), float(p[1]), float(p[2])
    else:
        raise ValueError("probs must contain 3 values [p_hold, p_buy, p_sell]")

    # compute simple directional signal
    signal = p_buy - p_sell  # range [-1,1]
    abs_signal = abs(signal)

    equity = get_equity(state, price)

    info = {
        "p_hold": p_hold,
        "p_buy": p_buy,
        "p_sell": p_sell,
        "signal": signal,
        "equity": equity,
        "price": price
    }

    # Decide to hold if weak signal
    if abs_signal < hold_threshold:
        logger.info("Decision=HOLD (weak signal %.4f)", signal)
        info.update({"action": "HOLD", "shares_traded": 0, "trade_notional": 0.0})
        return state, info

    # Desired position fraction of equity in stock (target)
    # scale by |signal| * aggressiveness, cap at max_pos_pct
    target_frac = min(max_pos_pct, abs_signal * aggressiveness * max_pos_pct)
    desired_position_value = equity * target_frac
    desired_shares_total = math.floor(desired_position_value / price) if price > 0 else 0

    current_shares = int(state.get("AvailStocks", 0))
    current_position_value = current_shares * price

    # BUY logic
    if signal > 0:
        shares_to_buy = max(0, desired_shares_total - current_shares)
        trade_notional = shares_to_buy * price
        if trade_notional < min_trade_value or shares_to_buy <= 0:
            logger.info("Decision=HOLD (buy signal but trade too small or already at/above desired size). signal=%.4f, desired_shares=%d, current=%d",
                        signal, desired_shares_total, current_shares)
            info.update({"action": "HOLD", "shares_traded": 0, "trade_notional": 0.0})
            return state, info

        # check if enough cash
        cost = trade_notional + apply_transaction_cost(trade_notional)
        if cost > state["Money"]:
            # scale down to affordable shares
            affordable_shares = math.floor((state["Money"] / (price * (1+TRANSACTION_COST_PCT))))
            shares_to_buy = max(0, min(shares_to_buy, affordable_shares))
            trade_notional = shares_to_buy * price

        if shares_to_buy <= 0:
            logger.info("Decision=HOLD (not enough cash). needed=%d but affordable=0", shares_to_buy)
            info.update({"action": "HOLD", "shares_traded": 0, "trade_notional": 0.0})
            return state, info

        # execute (simulation)
        transaction_cost = apply_transaction_cost(trade_notional)
        total_cost = trade_notional + transaction_cost
        state["Money"] = round(state["Money"] - total_cost, 8)
        state["AvailStocks"] = current_shares + int(shares_to_buy)

        logger.info("Decision=BUY shares=%d price=%.4f notional=%.2f tx_cost=%.4f new_money=%.2f new_shares=%d",
                    int(shares_to_buy), price, trade_notional, transaction_cost, state["Money"], state["AvailStocks"])

        info.update({
            "action": "BUY",
            "shares_traded": int(shares_to_buy),
            "trade_notional": trade_notional,
            "transaction_cost": transaction_cost,
            "new_money": state["Money"],
            "new_shares": state["AvailStocks"]
        })
        save_state(state)
        return state, info

    # SELL logic (signal < 0)
    else:
        # if very strong negative, sell all
        if signal <= -sell_all_threshold:
            shares_to_sell = current_shares
        else:
            # sell fraction of current holdings proportional to strength
            shares_to_sell = math.floor(current_shares * (abs_signal * aggressiveness))
        trade_notional = shares_to_sell * price
        if shares_to_sell <= 0 or trade_notional < min_trade_value:
            logger.info("Decision=HOLD (sell signal but nothing to sell or too small). signal=%.4f, shares_to_sell=%d", signal, shares_to_sell)
            info.update({"action": "HOLD", "shares_traded": 0, "trade_notional": 0.0})
            return state, info

        # execute sell
        transaction_cost = apply_transaction_cost(trade_notional)
        total_proceed = trade_notional - transaction_cost
        state["Money"] = round(state["Money"] + total_proceed, 8)
        state["AvailStocks"] = current_shares - int(shares_to_sell)

        logger.info("Decision=SELL shares=%d price=%.4f notional=%.2f tx_cost=%.4f new_money=%.2f new_shares=%d",
                    int(shares_to_sell), price, trade_notional, transaction_cost, state["Money"], state["AvailStocks"])

        info.update({
            "action": "SELL",
            "shares_traded": int(shares_to_sell),
            "trade_notional": trade_notional,
            "transaction_cost": transaction_cost,
            "new_money": state["Money"],
            "new_shares": state["AvailStocks"]
        })
        save_state(state)
        return state, info

# convenience wrapper for your earlier execute_order
def execute_order_from_probs(probs, price):
    """
    probs can be numpy array or list-like. price = current market price (float).
    """
    state = load_state()
    new_state, info = decide_and_execute(probs, price, state)
    return new_state, info
