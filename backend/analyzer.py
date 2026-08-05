import pandas as pd
import numpy as np
import math
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def clean_for_json(obj):
    """
    Recursively clean dictionary/list for JSON serialization.
    Handles:
    - NaN, Infinity, -Infinity -> None
    - Numpy types -> Native Python types
    """
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (float, np.float64, np.float32)):
        if pd.isna(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def generate_plots(df: pd.DataFrame):
    plots = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 1. Correlation Heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Correlation Heatmap")
        plots["heatmap"] = plot_to_base64(plt.gcf())
    
    # 2. Distributions (Top 3 numeric)
    for col in numeric_cols[:3]:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plots[f"dist_{col}"] = plot_to_base64(plt.gcf())
        
    return plots

def auto_eda(df: pd.DataFrame):
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Basic Info
    # Calculate true target prevalences and rate for EDA
    stats = get_failure_stats(df)
    failure_count = stats.get("total_failures", 0)
    failure_rate = stats.get("failure_rate", 0.0)

    summary = {
        "shape": df.shape,
        "failure_count": failure_count,
        "failure_rate": failure_rate,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols
    }

    # Descriptive Statistics
    desc = df.describe(include='all')
    summary["statistics"] = desc.to_dict()

    # Correlations (Numeric only)
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        summary["correlations"] = corr_matrix.to_dict()
    else:
        summary["correlations"] = {}

    # Sample Data (First 5 rows)
    summary["sample"] = df.head(5).to_dict(orient="records")

    # Simple Outlier Analysis (IQR Method) for numeric cols
    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if count > 0:
            outliers[col] = int(count)
            
    summary["outliers"] = outliers

    # Duplicate Rows Count
    summary["duplicate_rows"] = int(df.duplicated().sum())
    
    # Categorical Distributions (Top 10 counts)
    distributions = {}
    for col in categorical_cols:
        # Get top 10 values
        counts = df[col].value_counts().head(10).to_dict()
        distributions[col] = counts
    summary["distributions"] = distributions

    # Final Recursive Cleaning
    return clean_for_json(summary)

def find_target_column(df: pd.DataFrame, target_override: str = None):
    """
    Identifies target variable dynamically, prioritizing manual override.
    """
    if target_override and target_override in df.columns:
        return target_override

    # Check if target columns are present (priority ordered)
    possible_cols = [
        "Target", "target", "label", "Label", "y", "Machine failure", "Failure", "failure", 
        "Survived", "survived", "churn", "Churn", "default", "Default", "class", "Class",
        "output", "Output", "response", "Response", "clicked", "Clicked", "decision", "Decision"
    ]
    # Check exact match
    for col in possible_cols:
        if col in df.columns:
            return col
            
    # Check case-insensitive match
    cols_lower = {c.lower(): c for c in df.columns}
    for col in possible_cols:
        col_lower = col.lower()
        if col_lower in cols_lower:
            return cols_lower[col_lower]
            
    if len(df.columns) > 0:
        # Default to the last column
        return df.columns[-1]
    return None

class TargetAnalysisEngine:
    @staticmethod
    def get_target_stats(df: pd.DataFrame, target_override: str = None) -> dict:
        """
        Returns generic statistics on the target variable.
        (Previously get_failure_stats)
        """
        target_col = find_target_column(df, target_override=target_override)
        if not target_col:
            return {"error": "No target column found"}

        total_records = len(df)
        unique_vals = df[target_col].dropna().unique()
        
        # Determine type of target: classification (binary/categorical) or regression
        if pd.api.types.is_bool_dtype(df[target_col]) or (pd.api.types.is_numeric_dtype(df[target_col]) and len(unique_vals) <= 2):
            target_type = "classification"
        elif len(unique_vals) <= 10:
            target_type = "classification"
        else:
            target_type = "regression"

        target_instances = 0
        target_rate = 0.0
        modes = []

        if target_type == "classification" and len(unique_vals) > 0:
            # Identify positive class val
            positive_val = None
            for val in [1, True, 1.0, "1", "True", "true", "yes", "Yes", "Failure", "failure", "Survived", "survived"]:
                if val in unique_vals:
                    positive_val = val
                    break
            if positive_val is None:
                positive_val = unique_vals[1] if len(unique_vals) > 1 else unique_vals[0]

            target_instances = int((df[target_col] == positive_val).sum())
            target_rate = round((target_instances / total_records) * 100, 2) if total_records > 0 else 0.0

            # Sub-category/feature breakdown
            for col in df.columns:
                if col == target_col: continue
                if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
                    unique_vals_col = set(df[col].dropna().unique())
                    if unique_vals_col.issubset({0, 1, 0.0, 1.0, True, False}):
                        count = int(((df[col] == 1) & (df[target_col] == positive_val)).sum())
                        if count > 0:
                            pct = (count / target_instances * 100) if target_instances > 0 else 0
                            modes.append({"name": col, "count": count, "percent": pct})

        elif target_type == "regression":
            # Outlier counts (IQR method) as high-value target instances
            q1 = df[target_col].quantile(0.25)
            q3 = df[target_col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers_mask = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
            target_instances = int(outliers_mask.sum())
            
            if target_instances == 0:
                # Fallback to top 10% highest values
                threshold = df[target_col].quantile(0.90)
                outliers_mask = df[target_col] >= threshold
                target_instances = int(outliers_mask.sum())

            target_rate = round((target_instances / total_records) * 100, 2) if total_records > 0 else 0.0

            # Sub-category analysis by grouping categorical values for target outliers
            for col in df.columns:
                if col == target_col: continue
                if pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
                    top_cats = df[outliers_mask][col].value_counts().head(3)
                    for cat, count in top_cats.items():
                        pct = (count / target_instances * 100) if target_instances > 0 else 0
                        modes.append({"name": f"{col} = {cat}", "count": int(count), "percent": pct})

            if not modes:
                for col in df.columns:
                    if col == target_col: continue
                    if pd.api.types.is_numeric_dtype(df[col]):
                        unique_vals_col = set(df[col].dropna().unique())
                        if unique_vals_col.issubset({0, 1, 0.0, 1.0, True, False}):
                            count = int(((df[col] == 1) & outliers_mask).sum())
                            if count > 0:
                                pct = (count / target_instances * 100) if target_instances > 0 else 0
                                modes.append({"name": col, "count": count, "percent": pct})

        modes.sort(key=lambda x: x["count"], reverse=True)

        return {
            "target_column": target_col,
            "target_type": target_type,
            "total_records": total_records,
            "total_targets": target_instances,
            "target_rate": target_rate,
            "modes": modes,
            # Backward compatibility keys
            "total_failures": target_instances,
            "failure_rate": target_rate
        }

    @staticmethod
    def analyze_target_drivers(df: pd.DataFrame, target_override: str = None) -> str:
        """
        Generates analysis report of the target drivers.
        (Previously analyze_failure_modes)
        """
        stats = TargetAnalysisEngine.get_target_stats(df, target_override=target_override)
        if "error" in stats:
            return "No specific target variable column identified. Cannot profile data segments automatically."
            
        total_targets = stats["total_targets"]
        target_col = stats["target_column"]
        target_type = stats["target_type"]
        
        if total_targets == 0:
            return f"No occurrences of target events found for target variable '{target_col}'."

        report = [f"### Target Driver & Category Analysis"]
        report.append(f"**Target Variable**: `{target_col}` ({target_type} type)")
        report.append(f"**Total Highlighted Records**: {total_targets}")
        
        if stats["modes"]:
            report.append("\n**Breakdown of Target Contexts:**")
            for m in stats["modes"]:
                report.append(f"- **{m['name']}**: {m['count']} ({m['percent']:.1f}%)")
        else:
            report.append("\nNo specific subset indicators found for the target variable.")

        report.append("\n*This analysis was generated instantly based on dataset statistics.*")
        return "\n".join(report)

    @staticmethod
    def get_highlighted_records(df: pd.DataFrame, target_override: str = None) -> list:
        """
        Extracts highlighted rows based on target variable.
        (Previously get_failures)
        """
        target_col = find_target_column(df, target_override=target_override)
        if not target_col:
            return []

        unique_vals = df[target_col].dropna().unique()
        if pd.api.types.is_bool_dtype(df[target_col]) or (pd.api.types.is_numeric_dtype(df[target_col]) and len(unique_vals) <= 2):
            target_type = "classification"
        elif len(unique_vals) <= 10:
            target_type = "classification"
        else:
            target_type = "regression"

        if target_type == "classification" and len(unique_vals) > 0:
            positive_val = None
            for val in [1, True, 1.0, "1", "True", "true", "yes", "Yes", "Failure", "failure", "Survived", "survived"]:
                if val in unique_vals:
                    positive_val = val
                    break
            if positive_val is None:
                positive_val = unique_vals[1] if len(unique_vals) > 1 else unique_vals[0]
                
            highlighted_df = df[df[target_col] == positive_val]
        else:
            # Regression: Outliers or Top 10% highest values
            q1 = df[target_col].quantile(0.25)
            q3 = df[target_col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            mask = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
            if mask.sum() == 0:
                mask = df[target_col] >= df[target_col].quantile(0.90)
            highlighted_df = df[mask]

        if not highlighted_df.empty:
            # Limit to top 1000 to prevent huge payloads
            records = highlighted_df.head(1000).to_dict(orient="records")
            return clean_for_json(records)
        return []

def get_failure_stats(df: pd.DataFrame, target_override: str = None):
    return TargetAnalysisEngine.get_target_stats(df, target_override=target_override)

def analyze_failure_modes(df: pd.DataFrame, target_override: str = None):
    return TargetAnalysisEngine.analyze_target_drivers(df, target_override=target_override)

def get_correlation_stats(df: pd.DataFrame, target_override: str = None):
    """
    Returns correlation data.
    """
    target_col = find_target_column(df, target_override=target_override)
    if not target_col:
        return {"error": "No target column found"}
        
    unique_vals = df[target_col].dropna().unique()
    if pd.api.types.is_bool_dtype(df[target_col]) or (pd.api.types.is_numeric_dtype(df[target_col]) and len(unique_vals) <= 2):
        target_type = "classification"
    elif len(unique_vals) <= 10:
        target_type = "classification"
    else:
        target_type = "regression"

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = {"top_correlations": [], "shifts": []}
    
    try:
        # 1. Correlations
        target_series = df[target_col]
        if not pd.api.types.is_numeric_dtype(target_series):
            target_series = pd.Series(pd.factorize(target_series)[0], index=target_series.index)
            
        corrs = df[numeric_cols].corrwith(target_series).sort_values(ascending=False)
        top_corr = corrs[abs(corrs) > 0.05].drop(target_col, errors='ignore')
        
        for col, val in top_corr.head(5).items():
            if pd.isna(val): continue
            stats["top_correlations"].append({"feature": col, "value": val})
            
        # 2. Shifts
        if target_type == "classification" and len(unique_vals) > 0:
            positive_val = None
            for val in [1, True, 1.0, "1", "True", "true", "yes", "Yes", "Failure", "failure"]:
                if val in unique_vals:
                    positive_val = val
                    break
            if positive_val is None:
                positive_val = unique_vals[1] if len(unique_vals) > 1 else unique_vals[0]
                
            pos_mask = df[target_col] == positive_val
            neg_mask = df[target_col] != positive_val
        else:
            q1 = df[target_col].quantile(0.25)
            q3 = df[target_col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            pos_mask = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
            if pos_mask.sum() == 0:
                pos_mask = df[target_col] >= df[target_col].quantile(0.90)
            neg_mask = ~pos_mask
            
        pos_df = df[pos_mask]
        neg_df = df[neg_mask]
        
        if not pos_df.empty and not neg_df.empty:
            for col in numeric_cols:
                if col == target_col or "id" in col.lower(): continue
                pos_mean = pos_df[col].mean()
                neg_mean = neg_df[col].mean()
                if neg_mean != 0 and not pd.isna(pos_mean) and not pd.isna(neg_mean):
                    pct_diff = ((pos_mean - neg_mean) / neg_mean) * 100
                    if abs(pct_diff) > 5:
                        stats["shifts"].append({
                            "feature": col,
                            "pct_diff": pct_diff,
                            "fail_mean": pos_mean, # compatibility key
                            "norm_mean": neg_mean  # compatibility key
                        })
    except Exception as e:
        return {"error": str(e)}
        
    return stats

def analyze_correlations(df: pd.DataFrame, target_override: str = None):
    stats = get_correlation_stats(df, target_override=target_override)
    if "error" in stats:
        return f"Could not calculate correlations: {stats['error']}"
        
    summary = ["### Statistical Key Driver Analysis"]
    
    # Corrs
    if stats["top_correlations"]:
        summary.append("**Top Correlated Features with the Target Variable:**")
        for item in stats["top_correlations"]:
            summary.append(f"- {item['feature']}: {item['value']:.2f}")
    else:
        summary.append("No strong linear correlations found with the target variable.")
        
    # Shifts
    if stats["shifts"]:
        summary.append("\n**Feature Behavior During High/Target Value Events:**")
        for item in stats["shifts"]:
            direction = "HIGHER" if item['pct_diff'] > 0 else "LOWER"
            summary.append(f"- {item['feature']}: {abs(item['pct_diff']):.1f}% {direction} (Avg: {item['fail_mean']:.1f} vs {item['norm_mean']:.1f})")
            
    return "\n".join(summary)