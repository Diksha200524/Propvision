import pandas as pd
import numpy as np


# ---------------------------------------------------------
# EMI Calculator
# ---------------------------------------------------------

def calculate_emi(
    loan_amount: float,
    annual_interest_rate: float,
    tenure_years: int
):
    """
    Calculates EMI details.

    Returns:
        monthly_emi
        total_interest
        total_payment
    """

    monthly_rate = annual_interest_rate / (12 * 100)

    months = tenure_years * 12

    if monthly_rate == 0:
        emi = loan_amount / months
    else:
        emi = (
            loan_amount
            * monthly_rate
            * (1 + monthly_rate) ** months
        ) / (
            (1 + monthly_rate) ** months - 1
        )

    total_payment = emi * months

    total_interest = total_payment - loan_amount

    return {
        "loan_amount": round(loan_amount, 2),
        "monthly_emi": round(emi, 2),
        "total_interest": round(total_interest, 2),
        "total_payment": round(total_payment, 2),
        "months": months
    }


# ---------------------------------------------------------
# Amortization Schedule
# ---------------------------------------------------------

def amortization_schedule(
    loan_amount: float,
    annual_interest_rate: float,
    tenure_years: int
):

    monthly_rate = annual_interest_rate / (12 * 100)

    months = tenure_years * 12

    emi = calculate_emi(
        loan_amount,
        annual_interest_rate,
        tenure_years
    )["monthly_emi"]

    balance = loan_amount

    schedule = []

    for month in range(1, months + 1):

        interest = balance * monthly_rate

        principal = emi - interest

        balance -= principal

        if balance < 0:
            balance = 0

        schedule.append({
            "Month": month,
            "EMI": round(emi, 2),
            "Principal": round(principal, 2),
            "Interest": round(interest, 2),
            "Remaining Balance": round(balance, 2)
        })

    return pd.DataFrame(schedule)


# ---------------------------------------------------------
# Loan Summary
# ---------------------------------------------------------

def loan_summary(
    loan_amount,
    annual_interest_rate,
    tenure_years
):

    emi = calculate_emi(
        loan_amount,
        annual_interest_rate,
        tenure_years
    )

    return {
        "Loan Amount": emi["loan_amount"],
        "Interest Rate": annual_interest_rate,
        "Tenure (Years)": tenure_years,
        "Monthly EMI": emi["monthly_emi"],
        "Total Interest": emi["total_interest"],
        "Total Payment": emi["total_payment"]
    }


# ---------------------------------------------------------
# Affordability Calculator
# ---------------------------------------------------------

def affordability(
    monthly_income,
    emi_percentage=40
):
    """
    Maximum EMI user should pay.
    Default = 40% of monthly income.
    """

    return round(
        monthly_income * emi_percentage / 100,
        2
    )


# ---------------------------------------------------------
# Loan Eligibility
# ---------------------------------------------------------

def eligible_loan(
    monthly_income,
    annual_interest_rate,
    tenure_years,
    emi_percentage=40
):

    max_emi = affordability(
        monthly_income,
        emi_percentage
    )

    r = annual_interest_rate / (12 * 100)

    n = tenure_years * 12

    if r == 0:
        return max_emi * n

    loan = (
        max_emi
        * ((1 + r) ** n - 1)
    ) / (
        r * (1 + r) ** n
    )

    return round(loan, 2)


# ---------------------------------------------------------
# Prepayment Calculator
# ---------------------------------------------------------

def prepayment_savings(
    loan_amount,
    annual_interest_rate,
    tenure_years,
    prepayment_amount
):

    original = calculate_emi(
        loan_amount,
        annual_interest_rate,
        tenure_years
    )

    new_loan = max(
        loan_amount - prepayment_amount,
        0
    )

    updated = calculate_emi(
        new_loan,
        annual_interest_rate,
        tenure_years
    )

    return {
        "Old EMI": original["monthly_emi"],
        "New EMI": updated["monthly_emi"],
        "Interest Saved":
            round(
                original["total_interest"] -
                updated["total_interest"],
                2
            )
    }