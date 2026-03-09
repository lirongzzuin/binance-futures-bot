"""
Correlation Guard — 상관 자산 과다 노출 방지.

BTC와 ETH는 ~0.85-0.95 상관관계를 가짐.
동일 방향 동시 포지션은 사실상 2배 노출 → 리스크 배가.

규칙:
  1. 상관 자산에 동일 방향 포지션 존재 시 → risk_scale 50% (사이즈 절반)
  2. 반대 방향(헤지) 포지션 → risk_scale 100% (허용)
  3. 총 방향성 노출이 equity의 MAX_TOTAL_EXPOSURE_PCT 초과 시 → 진입 차단
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.logger import get_logger

log = get_logger(__name__)

# 상관 자산 쌍 정의
_CORRELATED_PAIRS: Dict[str, List[str]] = {
    "BTCUSDT": ["ETHUSDT"],
    "ETHUSDT": ["BTCUSDT"],
}


class CorrelationGuard:
    """상관 자산 포트폴리오 리스크 관리."""

    def __init__(self, max_total_exposure_pct: float = 0.75):
        self._max_exposure = max_total_exposure_pct

    def check_entry(
        self,
        symbol: str,
        direction: int,         # +1 long, -1 short
        notional: float,        # 진입 예정 명목가
        positions: list,        # List[PositionInfo] — 현재 열린 포지션들
        equity: float,
    ) -> Tuple[bool, float]:
        """
        상관 자산 가드 체크.

        Returns:
            (allowed: bool, risk_scale: float 0.0–1.0)
            - allowed=False: 진입 차단
            - risk_scale<1.0: 포지션 사이즈 축소
        """
        correlated_symbols = _CORRELATED_PAIRS.get(symbol, [])
        risk_scale = 1.0

        for pos in positions:
            if hasattr(pos, 'is_flat') and pos.is_flat:
                continue
            if pos.symbol not in correlated_symbols:
                continue

            pos_direction = 1 if pos.is_long else -1

            if pos_direction == direction:
                # 동일 방향 상관 자산 → 사이즈 절반
                risk_scale = 0.50
                log.info(
                    "CorrelationGuard: %s %s — correlated %s same direction → scale=0.50",
                    symbol, "LONG" if direction == 1 else "SHORT", pos.symbol,
                )
            # 반대 방향 (헤지): risk_scale 유지 (1.0)

        # 총 방향성 노출 체크
        net_exposure = 0.0
        for pos in positions:
            if hasattr(pos, 'is_flat') and pos.is_flat:
                continue
            pos_dir = 1 if pos.is_long else -1
            pos_notional = pos.qty * pos.entry_price if pos.entry_price > 0 else 0
            net_exposure += pos_dir * pos_notional

        proposed_total = abs(net_exposure + direction * notional * risk_scale)

        if equity > 0 and proposed_total > equity * self._max_exposure:
            log.info(
                "CorrelationGuard: %s BLOCKED — total exposure %.2f > %.2f (%.0f%% equity)",
                symbol, proposed_total, equity * self._max_exposure,
                self._max_exposure * 100,
            )
            return False, 0.0

        return True, risk_scale
