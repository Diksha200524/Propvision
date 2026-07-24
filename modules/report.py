from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch


# -------------------------------------------------------
# Create PDF Report
# -------------------------------------------------------

def generate_report(
        filename,
        property_details,
        predicted_price,
        emi_summary,
        investment_summary,
        similar_properties=None
):
    """
    Generates a PDF report.

    Parameters
    ----------
    filename : str
    property_details : dict
    predicted_price : float
    emi_summary : dict
    investment_summary : dict
    similar_properties : DataFrame (optional)
    """

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    elements = []

    # ---------------------------------------------------
    # Title
    # ---------------------------------------------------

    elements.append(
        Paragraph(
            "<b>Real Estate Analysis Report</b>",
            styles["Title"]
        )
    )

    elements.append(Spacer(1, 0.3 * inch))

    # ---------------------------------------------------
    # Property Details
    # ---------------------------------------------------

    elements.append(
        Paragraph(
            "<b>Property Details</b>",
            styles["Heading2"]
        )
    )

    property_table = [["Feature", "Value"]]

    for key, value in property_details.items():
        property_table.append([str(key), str(value)])

    table = Table(property_table)

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

            ("GRID", (0, 0), (-1, -1), 1, colors.grey),

            ("BACKGROUND", (0, 1), (-1, -1), colors.beige),

            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),

            ("ALIGN", (0, 0), (-1, -1), "CENTER")
        ])
    )

    elements.append(table)

    elements.append(Spacer(1, 0.3 * inch))

    # ---------------------------------------------------
    # Predicted Price
    # ---------------------------------------------------

    elements.append(
        Paragraph(
            "<b>Predicted Price</b>",
            styles["Heading2"]
        )
    )

    elements.append(
        Paragraph(
            f"₹ {predicted_price:,.2f}",
            styles["BodyText"]
        )
    )

    elements.append(Spacer(1, 0.3 * inch))

    # ---------------------------------------------------
    # EMI Summary
    # ---------------------------------------------------

    elements.append(
        Paragraph(
            "<b>EMI Summary</b>",
            styles["Heading2"]
        )
    )

    emi_table = [["Parameter", "Value"]]

    for key, value in emi_summary.items():
        emi_table.append([key, str(value)])

    table = Table(emi_table)

    table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 1, colors.black),

            ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),

            ("BOTTOMPADDING", (0, 0), (-1, 0), 10)
        ])
    )

    elements.append(table)

    elements.append(Spacer(1, 0.3 * inch))

    # ---------------------------------------------------
    # Investment Summary
    # ---------------------------------------------------

    elements.append(
        Paragraph(
            "<b>Investment Analysis</b>",
            styles["Heading2"]
        )
    )

    investment_table = [["Metric", "Value"]]

    for key, value in investment_summary.items():
        investment_table.append([key, str(value)])

    table = Table(investment_table)

    table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 1, colors.black),

            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgreen),

            ("BOTTOMPADDING", (0, 0), (-1, 0), 10)
        ])
    )

    elements.append(table)

    elements.append(Spacer(1, 0.3 * inch))

    # ---------------------------------------------------
    # Similar Properties
    # ---------------------------------------------------

    if similar_properties is not None and not similar_properties.empty:

        elements.append(
            Paragraph(
                "<b>Similar Properties</b>",
                styles["Heading2"]
            )
        )

        table_data = [list(similar_properties.columns)]

        for row in similar_properties.values.tolist():
            table_data.append(row)

        table = Table(table_data)

        table.setStyle(
            TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),

                ("BACKGROUND", (0, 0), (-1, 0), colors.orange),

                ("BOTTOMPADDING", (0, 0), (-1, 0), 8),

                ("FONTSIZE", (0, 0), (-1, -1), 8)
            ])
        )

        elements.append(table)

    # ---------------------------------------------------
    # Footer
    # ---------------------------------------------------

    elements.append(Spacer(1, 0.4 * inch))

    elements.append(
        Paragraph(
            "Generated by Real Estate Analytics Platform",
            styles["Italic"]
        )
    )

    doc.build(elements)

    return filename