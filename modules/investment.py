import math


# ---------------------------------------------------------
# Future Property Value
# ---------------------------------------------------------

def future_value(
    current_price: float,
    annual_growth_rate: float,
    years: int
):
    """
    Calculates future property value using compound growth.
    """

    future_price = current_price * \
        ((1 + annual_growth_rate / 100) ** years)

    return round(future_price, 2)


# ---------------------------------------------------------
# Profit
# ---------------------------------------------------------

def profit(
    purchase_price: float,
    selling_price: float
):

    return round(
        selling_price - purchase_price,
        2
    )


# ---------------------------------------------------------
# Return On Investment
# ---------------------------------------------------------

def roi(
    purchase_price: float,
    selling_price: float
):

    return round(
        ((selling_price - purchase_price)
         / purchase_price) * 100,
        2
    )


# ---------------------------------------------------------
# CAGR
# ---------------------------------------------------------

def cagr(
    purchase_price: float,
    selling_price: float,
    years: int
):
    """
    Compound Annual Growth Rate
    """

    value = (
        (selling_price / purchase_price)
        ** (1 / years)
    ) - 1

    return round(value * 100, 2)


# ---------------------------------------------------------
# Rental Yield
# ---------------------------------------------------------

def rental_yield(
    property_price: float,
    monthly_rent: float
):

    annual_rent = monthly_rent * 12

    yield_percent = (
        annual_rent / property_price
    ) * 100

    return round(yield_percent, 2)


# ---------------------------------------------------------
# Break Even Time
# ---------------------------------------------------------

def break_even_years(
    property_price: float,
    monthly_rent: float
):

    annual_income = monthly_rent * 12

    if annual_income == 0:
        return None

    years = property_price / annual_income

    return round(years, 2)


# ---------------------------------------------------------
# Stamp Duty
# ---------------------------------------------------------

def stamp_duty(
    property_price: float,
    stamp_percent: float
):

    duty = (
        property_price
        * stamp_percent
        / 100
    )

    return round(duty, 2)


# ---------------------------------------------------------
# Registration Charge
# ---------------------------------------------------------

def registration_charge(
    property_price: float,
    registration_percent: float
):

    registration = (
        property_price
        * registration_percent
        / 100
    )

    return round(registration, 2)


# ---------------------------------------------------------
# Total Purchase Cost
# ---------------------------------------------------------

def total_purchase_cost(
    property_price: float,
    stamp_percent: float,
    registration_percent: float
):

    stamp = stamp_duty(
        property_price,
        stamp_percent
    )

    registration = registration_charge(
        property_price,
        registration_percent
    )

    return round(
        property_price +
        stamp +
        registration,
        2
    )


# ---------------------------------------------------------
# Investment Summary
# ---------------------------------------------------------

def investment_summary(
    purchase_price: float,
    annual_growth_rate: float,
    years: int,
    monthly_rent: float,
    stamp_percent: float = 7,
    registration_percent: float = 1
):

    future = future_value(
        purchase_price,
        annual_growth_rate,
        years
    )

    return {
        "Purchase Price": round(purchase_price, 2),

        "Future Value":
            future,

        "Profit":
            profit(
                purchase_price,
                future
            ),

        "ROI (%)":
            roi(
                purchase_price,
                future
            ),

        "CAGR (%)":
            cagr(
                purchase_price,
                future,
                years
            ),

        "Rental Yield (%)":
            rental_yield(
                purchase_price,
                monthly_rent
            ),

        "Break Even (Years)":
            break_even_years(
                purchase_price,
                monthly_rent
            ),

        "Stamp Duty":
            stamp_duty(
                purchase_price,
                stamp_percent
            ),

        "Registration":
            registration_charge(
                purchase_price,
                registration_percent
            ),

        "Total Purchase Cost":
            total_purchase_cost(
                purchase_price,
                stamp_percent,
                registration_percent
            )
    }