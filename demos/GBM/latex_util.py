
import pandas as pd

def format_number(x):
    """Custom formatting function to avoid unnecessary trailing zeros"""
    if pd.isna(x) or x == '-':
        return '-'
    try:
        num = float(x)
        formatted = f"{num:.4f}"
        if '.' not in formatted and abs(num) < 1 and num != 0:
            formatted += '.0'
        return formatted
    except (ValueError, TypeError):
        return str(x)

def format_results_dataframe(df, numeric_columns):
    """Apply custom formatting to numeric columns in results dataframe"""
    results_formatted = df.copy()
    for col in numeric_columns:
        if col in results_formatted.columns:
            results_formatted[col] = results_formatted[col].apply(format_number)
    return results_formatted

def generate_latex_table(df, caption, label):
    """Generate LaTeX table with booktabs formatting"""
    latex_table = df.style.hide(axis='index').to_latex(
        caption=caption,
        label=label,
        position="btp",
        hrules=True,
        column_format="ll@{\hspace{0.4em}}r@{\hspace{0.4em}}r@{\hspace{0.4em}}r@{\hspace{0.4em}}r@{\hspace{0.4em}}r@{\hspace{0.4em}}r@{\hspace{0.4em}}r"
    )
    # Replace default LaTeX table environment with booktabs format
    latex_table = latex_table.replace("\\begin{table}[H]", "\\begin{table}[btp]")
    latex_table = latex_table.replace("\\hline", "\\toprule", 1)
    latex_table = latex_table.replace("\\hline", "\\midrule", 1)
    latex_table = latex_table.replace("\\hline", "\\bottomrule")
    return latex_table