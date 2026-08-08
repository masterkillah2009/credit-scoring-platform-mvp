"""Exact decimal money arithmetic.

IPSRS CST-06 is explicit: "All monetary computation in exact decimal
arithmetic; no binary floating point for money." Every amount that could reach
a customer's loan agreement passes through this module.

Two rounding policies are used deliberately:

  * money  - 2 decimal places, ROUND_HALF_UP (the convention a borrower and an
             auditor both expect)
  * ratio  - 6 decimal places, ROUND_HALF_UP, for DSR and similar quotients
             that are reported but never paid

Instalments are rounded UP to the minor unit. Rounding a repayment down would
leave a residual balance at maturity that the schedule never collects; rounding
up leaves a trivial overpayment on the final instalment, which is the standard
and conservative treatment.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP, getcontext

# 28 significant digits is ample for retail amounts and keeps annuity
# discounting exact well beyond the precision anyone will read.
getcontext().prec = 28

MONEY = Decimal("0.01")
RATIO = Decimal("0.000001")
ZERO = Decimal("0")


def money(value) -> Decimal:
    """Coerce to an exact 2dp monetary Decimal.

    Floats are accepted at the boundary (JSON payloads carry them) but are
    converted via ``str`` so that 0.1 becomes exactly 0.10 rather than
    0.1000000000000000055511151231257827.
    """
    if isinstance(value, Decimal):
        decimal_value = value
    elif isinstance(value, float):
        decimal_value = Decimal(str(value))
    elif value is None:
        raise ValueError("money() received None; missing amounts must be "
                         "handled explicitly, never coerced to zero")
    else:
        decimal_value = Decimal(str(value))
    return decimal_value.quantize(MONEY, rounding=ROUND_HALF_UP)


def ratio(value) -> Decimal:
    if isinstance(value, Decimal):
        decimal_value = value
    elif isinstance(value, float):
        decimal_value = Decimal(str(value))
    else:
        decimal_value = Decimal(str(value))
    return decimal_value.quantize(RATIO, rounding=ROUND_HALF_UP)


def instalment_for(principal: Decimal, annual_rate: Decimal,
                   tenor_months: int) -> Decimal:
    """Level instalment on an amortising loan, rounded up to the minor unit.

        I = P * r / (1 - (1 + r)^-n)      r = annual_rate / 12

    A zero rate degenerates to straight-line repayment.
    """
    if tenor_months <= 0:
        raise ValueError("tenor_months must be positive")
    principal = money(principal)
    if principal <= ZERO:
        return ZERO
    monthly = Decimal(annual_rate) / Decimal(12)
    if monthly == ZERO:
        return (principal / Decimal(tenor_months)).quantize(
            MONEY, rounding=ROUND_CEILING)
    discount = Decimal(1) - (Decimal(1) + monthly) ** (-tenor_months)
    return (principal * monthly / discount).quantize(
        MONEY, rounding=ROUND_CEILING)


def principal_for(instalment: Decimal, annual_rate: Decimal,
                  tenor_months: int) -> Decimal:
    """Largest principal serviceable by a given instalment - the inverse annuity.

        P = I * (1 - (1 + r)^-n) / r

    Rounded DOWN to the minor unit: offering a principal whose instalment
    exceeds assessed affordability, even by one ngwee, is precisely the error
    the affordability assessment exists to prevent.
    """
    if tenor_months <= 0:
        raise ValueError("tenor_months must be positive")
    instalment = money(instalment)
    if instalment <= ZERO:
        return ZERO
    monthly = Decimal(annual_rate) / Decimal(12)
    if monthly == ZERO:
        principal = instalment * Decimal(tenor_months)
    else:
        discount = Decimal(1) - (Decimal(1) + monthly) ** (-tenor_months)
        principal = instalment * discount / monthly
    # quantize toward zero (floor for positive amounts)
    return principal.quantize(MONEY, rounding="ROUND_DOWN")


def total_repayable(instalment: Decimal, tenor_months: int) -> Decimal:
    return money(money(instalment) * Decimal(tenor_months))


def total_cost_of_credit(principal: Decimal, instalment: Decimal,
                         tenor_months: int, fee: Decimal = ZERO) -> Decimal:
    """Interest plus fees over the life of the facility."""
    return money(total_repayable(instalment, tenor_months)
                 - money(principal) + money(fee))
