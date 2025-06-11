import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor



def remove_duplicates_and_compare(df, target_col='quality'):
    """
    Removes duplicate rows from a dataframe and compares the distribution of a target column
    before and after duplicate removal.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The original dataframe to check for duplicates
    target_col : str, default='quality'
        The column to check distribution changes
        
    Returns:
    --------
    df_no_duplicates : pandas.DataFrame
        The dataframe with duplicates removed
    """
    
    df_no_duplicates = df.drop_duplicates(keep='first')
    
    print(f"Original dataset size: {df.shape[0]} rows")
    print(f"After removing duplicates: {df_no_duplicates.shape[0]} rows")
    print(f"Removed {df.shape[0] - df_no_duplicates.shape[0]} duplicate rows")
    
    
    print(f"\n{target_col.capitalize()} distribution in original dataset:")
    original_counts = df[target_col].value_counts(normalize=True).sort_index() * 100
    print(original_counts)
    
    print(f"\n{target_col.capitalize()} distribution after removing duplicates:")
    cleaned_counts = df_no_duplicates[target_col].value_counts(normalize=True).sort_index() * 100
    print(cleaned_counts)
    
   
    print("\nDifference in percentage points:")
    diff = cleaned_counts - original_counts
    print(diff)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.35
    x = original_counts.index
    
    ax.bar(x - width/2, original_counts, width, label='Original Dataset')
    ax.bar(x + width/2, cleaned_counts, width, label='After Removing Duplicates')
    
    ax.set_xlabel(f'{target_col.capitalize()} Rating')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Distribution of {target_col.capitalize()} Before and After Removing Duplicates')
    ax.set_xticks(x)
    ax.legend()
    
    plt.show()
    
    return df_no_duplicates



    

def check_duplicates(df):
    """
    Checks for duplicate rows in a dataframe and displays information about them.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to check for duplicates
        
    Returns:
    --------
    has_duplicates : bool
        True if duplicates were found, False otherwise
    """
    duplicate_rows = df.duplicated(keep='first')
    print(f"Number of duplicate rows: {duplicate_rows.sum()}")
    
    if duplicate_rows.sum() > 0:
        print("First few duplicate rows:")
        display(df[duplicate_rows].head())
        
        all_duplicates = df[df.duplicated(keep=False)]
        print(f"Total rows involved in duplication: {len(all_duplicates)}")
        
        if len(all_duplicates) > 0:
            print("Sample of duplicate sets (sorted to show identical rows together):")
            display(all_duplicates.sort_values(by=all_duplicates.columns.tolist()).head(10))
        
        return True
    else:
        print("No duplicates found in the dataset.")
        return False





def outlier_analysis(df):
    """
    Perform outlier analysis using Z-score and IQR methods, then compare results.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe to analyze
        
    Returns:
    --------
    results : dict
        Dictionary containing outliers detected by each method and comparison results
    """
    df_copy = df.copy()
    
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # ===========================================
    # Z-score method (threshold = 3)
    # ===========================================
    # Create z-scores for each column
    
    z_outliers_all = pd.DataFrame()
    for column in numeric_columns:
        series = df_copy[column]
        z_score = np.abs((series - series.mean()) / series.std())
        
        mask = z_score > 3
        if sum(mask) > 0:
    
            column_outliers = df_copy[mask].copy()
            column_outliers['outlier_feature'] = column
            column_outliers['z_score'] = z_score[mask]
            z_outliers_all = pd.concat([z_outliers_all, column_outliers])
    
    z_score_outliers = df_copy[df_copy.index.isin(z_outliers_all.index.unique())]
    
    print(f"Z-score method (threshold = 3):")
    print(f"- Found {len(z_score_outliers)} unique outliers ({len(z_score_outliers)/len(df)*100:.2f}% of data)")
    
    plt.figure(figsize=(12, 8))
    for i, column in enumerate(numeric_columns[:6]): 
        plt.subplot(2, 3, i+1)
        plt.hist(df_copy[column], bins=20, alpha=0.5, label='All data')
        if len(z_score_outliers) > 0:
            plt.hist(z_score_outliers[column], bins=20, alpha=0.7, color='red', label='Outliers')
        plt.title(f'{column}')
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.suptitle('Z-score Outliers by Column', y=1.02, fontsize=14)
    plt.show()
    
    # ===========================================
    # IQR method (multiplier = 1.5)
    # ===========================================
    iqr_outliers_all = pd.DataFrame()
    
    for column in numeric_columns:
        series = df_copy[column]
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (series < lower_bound) | (series > upper_bound)
        if sum(mask) > 0:
            column_outliers = df_copy[mask].copy()
            column_outliers['outlier_feature'] = column
            column_outliers['distance'] = np.where(
                series[mask] < lower_bound,
                lower_bound - series[mask],
                series[mask] - upper_bound
            )
            iqr_outliers_all = pd.concat([iqr_outliers_all, column_outliers])
    
    iqr_outliers = df_copy[df_copy.index.isin(iqr_outliers_all.index.unique())]
    
    print(f"\nIQR method (multiplier = 1.5):")
    print(f"- Found {len(iqr_outliers)} unique outliers ({len(iqr_outliers)/len(df)*100:.2f}% of data)")
    
    plt.figure(figsize=(14, 10))
    for i, column in enumerate(numeric_columns[:6]):  
        plt.subplot(2, 3, i+1)
        sns.boxplot(x=df_copy[column])
        plt.title(f'{column}')
    
    plt.tight_layout()
    plt.suptitle('Boxplots Showing IQR Outliers', y=1.02, fontsize=14)
    plt.show()
    
    # ===========================================
    # Compare methods
    # ===========================================
    z_indices = set(z_score_outliers.index)
    iqr_indices = set(iqr_outliers.index)
    
    common_indices = z_indices.intersection(iqr_indices)
    z_only_indices = z_indices - iqr_indices
    iqr_only_indices = iqr_indices - z_indices
    
    common_outliers = df_copy.loc[list(common_indices)] if common_indices else pd.DataFrame()
    z_only_outliers = df_copy.loc[list(z_only_indices)] if z_only_indices else pd.DataFrame()
    iqr_only_outliers = df_copy.loc[list(iqr_only_indices)] if iqr_only_indices else pd.DataFrame()
    
    print("\nDetailed Comparison of Methods:")
    print(f"- Z-score method identified {len(z_indices)} unique outliers")
    print(f"- IQR method identified {len(iqr_indices)} unique outliers")
    print(f"- Both methods agreed on {len(common_indices)} outliers")
    print(f"- {len(z_only_indices)} outliers were found ONLY by Z-score method")
    print(f"- {len(iqr_only_indices)} outliers were found ONLY by IQR method")
    
    plt.figure(figsize=(10, 6))
    plt.bar(['Z-score only', 'Both methods', 'IQR only'], 
            [len(z_only_indices), len(common_indices), len(iqr_only_indices)],
            color=['blue', 'purple', 'green'])
    plt.title('Comparison of Outlier Detection Methods')
    plt.ylabel('Number of Outliers')
    plt.show()
    
    return {
        'z_score_outliers': z_score_outliers,
        'iqr_outliers': iqr_outliers,
        'common_outliers': common_outliers,
        'z_only_outliers': z_only_outliers,
        'iqr_only_outliers': iqr_only_outliers,
        'all_outliers': pd.concat([z_score_outliers, iqr_outliers]).drop_duplicates()
    }




def get_zscore_outliers(df, threshold=3.0, visualize=True):
    """
    Identify outliers using the Z-score method with optional visualizations.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe to analyze
    threshold : float, default=3.0
        Z-score threshold for outlier detection
    visualize : bool, default=True
        Whether to create visualizations
        
    Returns:
    --------
    outliers : pandas DataFrame
        DataFrame containing only the outlier rows
    outlier_indices : list
        List of indices of outlier rows
    """
    from IPython.display import display, Markdown, HTML
    
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    all_z_scores = {}
    
    outlier_mask = pd.Series(False, index=df.index)
    outliers_by_column = {}
    column_outlier_counts = {}
    
    for column in numeric_columns:
        mean = df[column].mean()
        std = df[column].std()
        z_scores = np.abs((df[column] - mean) / std)
        all_z_scores[column] = z_scores
        
        column_mask = z_scores > threshold
        outlier_mask = outlier_mask | column_mask
        
        outliers_by_column[column] = df.index[column_mask].tolist()
        
        column_outlier_counts[column] = sum(column_mask)
    
    outliers = df[outlier_mask]
    outlier_indices = list(outliers.index)
    
    display(Markdown(f"## Z-score Outlier Analysis (threshold = {threshold})"))
    
    summary_df = pd.DataFrame({
        'Column': numeric_columns,
        'Outliers': [column_outlier_counts[col] for col in numeric_columns],
        'Percentage': [column_outlier_counts[col]/len(df)*100 for col in numeric_columns]
    })
    summary_df['Percentage'] = summary_df['Percentage'].round(2).astype(str) + '%'
    display(Markdown("### Outliers by Column"))
    display(summary_df.sort_values('Outliers', ascending=False))
    
    display(Markdown("### Overall Summary"))
    display(HTML(f"""
    <table style="width:50%">
        <tr>
            <td><b>Total rows analyzed:</b></td>
            <td>{len(df)}</td>
        </tr>
        <tr>
            <td><b>Unique rows with outliers:</b></td>
            <td>{len(outliers)}</td>
        </tr>
        <tr>
            <td><b>Percentage of data flagged as outliers:</b></td>
            <td>{len(outliers)/len(df)*100:.2f}%</td>
        </tr>
    </table>
    """))
    
    if visualize:
        n_cols = 3
        n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
        plt.figure(figsize=(16, n_rows * 4))
        
        for i, column in enumerate(numeric_columns):
            plt.subplot(n_rows, n_cols, i+1)
            
            sns.histplot(df[column], kde=True, color='skyblue', alpha=0.6, label='All data')
            
            if len(outliers_by_column[column]) > 0:
                sns.histplot(df.loc[outliers_by_column[column], column], 
                             color='red', alpha=0.7, label='Outliers')
            
            plt.title(f'Distribution of {column}')
            plt.axvline(df[column].mean(), color='darkblue', linestyle='--', 
                       label=f'Mean ({df[column].mean():.2f})')
            if i == 0:  # Only add legend to first plot to save space
                plt.legend()
        
        plt.tight_layout()
        plt.suptitle(f'Z-score Outlier Detection (threshold = {threshold})', 
                    y=1.02, fontsize=16)
        plt.show()
        
        plt.figure(figsize=(16, 10))
        for i, column in enumerate(numeric_columns[:12]): 
            plt.subplot(4, 3, i+1)
            sns.boxplot(x=df[column], color='skyblue')
            plt.title(f'Boxplot of {column}')
        
        plt.tight_layout()
        plt.suptitle('Boxplots Showing Distribution and Potential Outliers', 
                    y=1.02, fontsize=16)
        plt.show()
        
        if len(outliers) > 0:
            z_score_df = pd.DataFrame(index=outliers.index)
            for column in numeric_columns:
                z_score_df[column] = all_z_scores[column].loc[outliers.index]

            sample_size = min(20, len(outliers))
            top_outliers = z_score_df.max(axis=1).sort_values(ascending=False).index[:sample_size]
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(z_score_df.loc[top_outliers], cmap='YlOrRd', annot=True, fmt=".1f")
            plt.title(f'Z-scores of Top {sample_size} Outliers Across Features')
            plt.tight_layout()
            plt.show()
            
            key_vars = ['alcohol', 'volatile acidity', 'quality']
            if all(var in df.columns for var in key_vars):
                plt.figure(figsize=(16, 5))
                
                for i, pair in enumerate([(key_vars[0], key_vars[1]), 
                                         (key_vars[0], key_vars[2]), 
                                         (key_vars[1], key_vars[2])]):
                    plt.subplot(1, 3, i+1)
                
                    plt.scatter(df[pair[0]], df[pair[1]], 
                              alpha=0.5, color='blue', label='Normal')
                    # Outliers
                    plt.scatter(outliers[pair[0]], outliers[pair[1]], 
                              alpha=0.8, color='red', label='Outliers')
                    plt.xlabel(pair[0])
                    plt.ylabel(pair[1])
                    plt.title(f'{pair[1]} vs {pair[0]}')
                    if i == 0:
                        plt.legend()
                
                plt.tight_layout()
                plt.show()
    
    if len(outliers) > 0:
        display(Markdown("### Sample of Identified Outliers"))
        display(outliers.head(10))
    
    return outliers, outlier_indices







def compare_dataframes(df1, df2, name1='Original', name2='Modified', key_columns=None):
    """
    Compare two dataframes to see how distributions and relationships have changed.
    
    Parameters:
    -----------
    df1 : pandas DataFrame
        First dataframe (typically the original one)
    df2 : pandas DataFrame
        Second dataframe (typically after some transformation or filtering)
    name1 : str, default='Original'
        Name to use for the first dataframe in outputs
    name2 : str, default='Modified'
        Name to use for the second dataframe in outputs
    key_columns : list or None
        List of column names to focus on. If None, all numeric columns are used.
        
    Returns:
    --------
    comparison_stats : pandas DataFrame
        DataFrame with comparative statistics
    """
    from IPython.display import display, Markdown
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    display(Markdown(f"## Comparing {name1} ({len(df1)} rows) with {name2} ({len(df2)} rows)"))
    display(Markdown(f"* Difference: {len(df1) - len(df2)} rows ({(len(df1) - len(df2))/len(df1)*100:.2f}% reduction)"))
    
    if key_columns is None:
        key_columns = df1.select_dtypes(include=['number']).columns.tolist()
    
    key_columns = [col for col in key_columns if col in df1.columns and col in df2.columns]
    
    comparison_stats = pd.DataFrame(columns=[
        f'{name1} Mean', f'{name2} Mean', 'Mean % Change',
        f'{name1} Std', f'{name2} Std', 'Std % Change',
        f'{name1} Min', f'{name2} Min', 
        f'{name1} Max', f'{name2} Max'
    ])
    
    for col in key_columns:
        mean1, mean2 = df1[col].mean(), df2[col].mean()
        std1, std2 = df1[col].std(), df2[col].std()
        min1, min2 = df1[col].min(), df2[col].min()
        max1, max2 = df1[col].max(), df2[col].max()
        
        mean_pct = (mean2 - mean1) / mean1 * 100 if mean1 != 0 else np.nan
        std_pct = (std2 - std1) / std1 * 100 if std1 != 0 else np.nan
        
        comparison_stats.loc[col] = [
            mean1, mean2, mean_pct,
            std1, std2, std_pct,
            min1, min2,
            max1, max2
        ]
    
    for col in ['Mean % Change', 'Std % Change']:
        comparison_stats[col] = comparison_stats[col].round(2).astype(str) + '%'

    numeric_cols = [col for col in comparison_stats.columns if col not in ['Mean % Change', 'Std % Change']]
    comparison_stats[numeric_cols] = comparison_stats[numeric_cols].round(3)
    
    display(Markdown("### Summary Statistics Comparison"))
    display(comparison_stats)
    

    n_cols = min(3, len(key_columns))
    n_rows = (len(key_columns) + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, n_rows * 4))
    for i, col in enumerate(key_columns):
        plt.subplot(n_rows, n_cols, i+1)
        
        sns.kdeplot(df1[col], label=name1, color='blue', alpha=0.7)
        sns.kdeplot(df2[col], label=name2, color='red', alpha=0.7)
        
        plt.axvline(df1[col].mean(), color='blue', linestyle='--', 
                   label=f'{name1} Mean: {df1[col].mean():.2f}')
        plt.axvline(df2[col].mean(), color='red', linestyle='--', 
                   label=f'{name2} Mean: {df2[col].mean():.2f}')
        
        plt.title(f'Distribution of {col}')
        plt.legend()
    
    plt.tight_layout()
    plt.suptitle(f'Distribution Comparison: {name1} vs {name2}', y=1.02, fontsize=16)
    plt.show()
    
    if 'quality' in key_columns:
        target = 'quality'
        display(Markdown("### Correlation with Quality"))
        
        corr1 = df1.corr()[target].drop(target)
        corr2 = df2.corr()[target].drop(target)
        
        key_cols_no_target = [col for col in key_columns if col != target]
        corr1 = corr1[corr1.index.isin(key_cols_no_target)]
        corr2 = corr2[corr2.index.isin(key_cols_no_target)]
        
        corr_change = corr2 - corr1
        corr_pct_change = (corr2 - corr1) / corr1.abs() * 100
        
        corr_comparison = pd.DataFrame({
            f'{name1} Correlation': corr1,
            f'{name2} Correlation': corr2,
            'Absolute Change': corr_change,
            'Percentage Change': corr_pct_change
        })
        
        corr_comparison = corr_comparison.sort_values(f'{name2} Correlation', key=abs, ascending=False)
        
        corr_comparison = corr_comparison.round(3)
        corr_comparison['Percentage Change'] = corr_comparison['Percentage Change'].round(1).astype(str) + '%'
        
        display(corr_comparison)
        
        plt.figure(figsize=(12, 6))

        top_features = corr_comparison.index[:6]  # Top 6 features
        
        x = np.arange(len(top_features))
        width = 0.35
        
        plt.bar(x - width/2, corr_comparison.loc[top_features, f'{name1} Correlation'], 
               width, label=name1, color='blue', alpha=0.7)
        plt.bar(x + width/2, corr_comparison.loc[top_features, f'{name2} Correlation'], 
               width, label=name2, color='red', alpha=0.7)
        
        plt.xlabel('Feature')
        plt.ylabel(f'Correlation with {target}')
        plt.title(f'Feature Correlations with {target}: {name1} vs {name2}')
        plt.xticks(x, top_features, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # 4. Compare bivariate relationships for key pairs
    key_pairs = []
    if 'alcohol' in key_columns and 'volatile acidity' in key_columns:
        key_pairs.append(('alcohol', 'volatile acidity'))
    if 'alcohol' in key_columns and 'quality' in key_columns:
        key_pairs.append(('alcohol', 'quality'))
    if 'volatile acidity' in key_columns and 'quality' in key_columns:
        key_pairs.append(('volatile acidity', 'quality'))
    
    if key_pairs:
        plt.figure(figsize=(15, 5))
        for i, (x_col, y_col) in enumerate(key_pairs):
            plt.subplot(1, len(key_pairs), i+1)
            
            sns.regplot(x=x_col, y=y_col, data=df1, 
                      scatter_kws={'alpha':0.3, 's':10}, 
                      line_kws={'color':'blue'}, label=name1)
            sns.regplot(x=x_col, y=y_col, data=df2, 
                      scatter_kws={'alpha':0.3, 's':10}, 
                      line_kws={'color':'red'}, label=name2)
            
            plt.title(f'{y_col} vs {x_col}')
            plt.legend()
        
        plt.tight_layout()
        plt.suptitle('Key Relationship Comparisons', y=1.02)
        plt.show()
    
    return None






def plot_feature_distributions(df, figsize=(15, 12)):
    """
    Plots histograms and boxplots for all numeric features in the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the features
    figsize : tuple, default=(15, 12)
        Size of the figure
    """
    numeric_cols = [col for col in df.columns if col not in ['quality', 'good_wine']]
    n_cols = len(numeric_cols)
    
    n_rows = (n_cols + 1) // 2
    
    fig, axes = plt.subplots(n_rows, 4, figsize=figsize)
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        ax_hist = axes[i*2]
        sns.histplot(df[col], kde=True, ax=ax_hist, color='darkblue')
        ax_hist.set_title(f'Distribution of {col}', fontsize=12)
        ax_hist.set_xlabel(col, fontsize=10)
        ax_hist.set_ylabel('Frequency', fontsize=10)
        
        ax_box = axes[i*2+1]
        sns.boxplot(y=df[col], ax=ax_box, color='darkblue')
        ax_box.set_title(f'Boxplot of {col}', fontsize=12)
        ax_box.set_ylabel(col, fontsize=10)
    
    for j in range(i*2+2, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    stats_df = pd.DataFrame({
        'Mean': df[numeric_cols].mean(),
        'Median': df[numeric_cols].median(),
        'Std Dev': df[numeric_cols].std(),
        'Min': df[numeric_cols].min(),
        'Max': df[numeric_cols].max(),
        'Skewness': df[numeric_cols].skew(),
        'Kurtosis': df[numeric_cols].kurtosis()
    })
    
    print("Feature Statistics (including skewness and kurtosis):")
    display(stats_df)
    
    high_skew = stats_df[abs(stats_df['Skewness']) > 1].index.tolist()
    if high_skew:
        print(f"\nFeatures with high skewness (|skew| > 1) that might need transformation: {high_skew}")





def plot_correlation_heatmap(df, figsize=(12, 10)):
    """
    Plots a correlation heatmap for all numeric features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the features
    figsize : tuple, default=(12, 10)
        Size of the figure
        
    Returns:
    --------
    corr_matrix : pandas.DataFrame
        The correlation matrix
    """
    corr_matrix = df.corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    plt.figure(figsize=figsize)
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, linewidths=.5,
                cbar_kws={"shrink": .5})
    
    plt.title('Correlation Matrix of Wine Features', fontsize=14)
    plt.tight_layout()
    plt.show()
    

    return corr_matrix




def plot_quality_correlations(corr_matrix, figsize=(10, 6)):
    """
    Creates a bar plot of correlations with quality, sorted by absolute value.
    
    Parameters:
    -----------
    corr_matrix : pandas.DataFrame
        The correlation matrix
    figsize : tuple, default=(10, 6)
        Size of the figure
        
    Returns:
    --------
    quality_corrs : pandas.Series
        Series of correlations with quality
    """
    quality_corrs = corr_matrix['quality'].drop('quality')
    
    quality_corrs = quality_corrs.reindex(quality_corrs.abs().sort_values(ascending=False).index)
    
    plt.figure(figsize=figsize)
    bars = plt.bar(quality_corrs.index, quality_corrs.values, color=['darkblue' if x > 0 else 'darkred' for x in quality_corrs.values])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, 
                 height + 0.02 if height > 0 else height - 0.05,
                 f'{height:.2f}', 
                 ha='center', va='bottom' if height > 0 else 'top',
                 fontsize=9)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Correlation of Features with Wine Quality', fontsize=14)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Correlation Coefficient', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    print("Top positive correlations with quality:")
    print(quality_corrs[quality_corrs > 0].head(3))
    
    print("\nTop negative correlations with quality:")
    print(quality_corrs[quality_corrs < 0].head(3))
    
    return quality_corrs




    
def plot_feature_vs_quality(df, features, figsize=(15, 4)):
    """
    Creates scatter plots and box plots of selected features vs. quality.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the features
    features : list
        List of feature names to plot
    figsize : tuple, default=(15, 4)
        Size of the figure per feature
    """
    for feature in features:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        sns.regplot(x=feature, y='quality', data=df, ax=ax1, 
                   scatter_kws={'alpha':0.5, 'color':'darkblue'}, 
                   line_kws={'color':'red'})
        ax1.set_title(f'{feature} vs. Quality with Regression Line', fontsize=12)
        
        corr = df[[feature, 'quality']].corr().iloc[0, 1]
        ax1.annotate(f"Correlation: {corr:.2f}", 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.8))
        
        sns.boxplot(x='quality', y=feature, data=df, ax=ax2, palette='Blues')
        ax2.set_title(f'Distribution of {feature} by Quality', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        groups = [df[df['quality'] == q][feature].values for q in sorted(df['quality'].unique())]
        f_val, p_val = stats.f_oneway(*groups)
        print(f"ANOVA for {feature} across quality groups: F={f_val:.2f}, p={p_val:.4f}")
        if p_val < 0.05:
            print(f"The mean values of {feature} differ significantly across quality groups (p < 0.05).")
        else:
            print(f"No significant difference in {feature} means across quality groups (p >= 0.05).")
        print("-" * 50)





def check_multicollinearity(df, features):
    """
    Calculates Variance Inflation Factor (VIF) for each feature to check for multicollinearity.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the features
    features : list
        List of feature names to check
        
    Returns:
    --------
    vif_df : pandas.DataFrame
        DataFrame with VIF values for each feature
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = features
    
    X = df[features]
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    vif_data = vif_data.sort_values("VIF", ascending=False)
    
    print("Variance Inflation Factors (VIF):")
    print("VIF > 10 indicates high multicollinearity")
    print("VIF > 5 indicates moderate multicollinearity")
    display(vif_data)
    
    high_vif = vif_data[vif_data["VIF"] > 5]["Feature"].tolist()
    if high_vif:
        print(f"\nFeatures with concerning multicollinearity (VIF > 5): {high_vif}")
        
        if len(high_vif) > 1:
            print("\nCorrelation matrix of features with high VIF:")
            high_vif_corr = df[high_vif].corr()
            display(high_vif_corr)
    else:
        print("\nNo concerning multicollinearity detected among the features.")
    
    return vif_data




    
def explore_potential_transformations(df, features, figsize=(15, 4)):
    """
    Explores potential transformations for skewed features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the features
    features : list
        List of feature names to explore
    figsize : tuple, default=(15, 4)
        Size of the figure per feature
    """
    for feature in features:

        skew = df[feature].skew()
        
        if abs(skew) < 0.5:
            print(f"{feature} has low skewness ({skew:.2f}) and may not need transformation.")
            continue
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        sns.histplot(df[feature], kde=True, ax=axes[0], color='darkblue')
        axes[0].set_title(f'Original {feature}\nSkew: {skew:.2f}', fontsize=10)
        
        if df[feature].min() <= 0:
            log_data = np.log(df[feature] - df[feature].min() + 1)
        else:
            log_data = np.log(df[feature])
        
        log_skew = log_data.skew()
        sns.histplot(log_data, kde=True, ax=axes[1], color='darkgreen')
        axes[1].set_title(f'Log {feature}\nSkew: {log_skew:.2f}', fontsize=10)
        
        if df[feature].min() < 0:
            sqrt_data = np.sqrt(df[feature] - df[feature].min())
        else:
            sqrt_data = np.sqrt(df[feature])
        
        sqrt_skew = sqrt_data.skew()
        sns.histplot(sqrt_data, kde=True, ax=axes[2], color='darkred')
        axes[2].set_title(f'Sqrt {feature}\nSkew: {sqrt_skew:.2f}', fontsize=10)
        
        if df[feature].min() <= 0:
            boxcox_data = df[feature] - df[feature].min() + 1
            boxcox_data, _ = stats.boxcox(boxcox_data)
        else:
            boxcox_data, _ = stats.boxcox(df[feature])
        
        boxcox_skew = pd.Series(boxcox_data).skew()
        sns.histplot(boxcox_data, kde=True, ax=axes[3], color='darkorange')
        axes[3].set_title(f'Box-Cox {feature}\nSkew: {boxcox_skew:.2f}', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        transformations = {
            'Original': abs(skew),
            'Log': abs(log_skew),
            'Square Root': abs(sqrt_skew),
            'Box-Cox': abs(boxcox_skew)
        }
        
        best_transform = min(transformations.items(), key=lambda x: x[1])
        print(f"Recommended transformation for {feature}: {best_transform[0]} (skewness: {best_transform[1]:.2f})")
        print("-" * 50)





def display_summary_statistics(df):
    """
    Displays comprehensive summary statistics for a dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to summarize
    """
    print("Basic Summary Statistics:")
    summary = df.describe()
    display(summary)

    print("\nAdditional Statistics:")
    additional_stats = pd.DataFrame({
        'Median': df.median(),
        'Skewness': df.skew(),
        'Kurtosis': df.kurtosis(),
        'Missing Values': df.isnull().sum(),
        'Missing (%)': (df.isnull().sum() / len(df) * 100).round(2)
    })
    display(additional_stats)
    
    if 'quality' in df.columns:
        print("\nWine Quality Distribution:")
        quality_counts = df['quality'].value_counts().sort_index()
        quality_percentage = (df['quality'].value_counts(normalize=True).sort_index() * 100).round(2)
        
        quality_stats = pd.DataFrame({
            'Count': quality_counts,
            'Percentage (%)': quality_percentage
        })
        display(quality_stats)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(quality_counts.index, quality_counts.values, color='darkblue')
        
        max_count = quality_counts.max()
        
        plt.ylim(0, max_count * 1.20) 
        
        for bar, count, percentage in zip(bars, quality_counts.values, quality_percentage.values):
            height = bar.get_height()
            y_pos = height + (max_count * 0.05) 
            
            plt.text(
                bar.get_x() + bar.get_width()/2,
                y_pos,
                f'{count}\n({percentage}%)',
                ha='center',
                fontsize=10,
                fontweight='bold',
                color='darkblue'
            )
            
        plt.xlabel('Wine Quality Rating', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Distribution of Wine Quality Ratings', fontsize=14)
        plt.xticks(quality_counts.index, fontsize=10)
        plt.yticks(fontsize=10)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()





def analyze_quality_variable(df, figsize=(10, 6)):
    """
    Performs detailed analysis of the quality variable.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the quality variable
    figsize : tuple, default=(10, 6)
        Size of the figure
    """
    plt.figure(figsize=figsize)
    
    ax = sns.histplot(df['quality'], kde=True, discrete=True, color='darkblue')
    
    stats_text = f"Mean: {df['quality'].mean():.2f}\n"
    stats_text += f"Median: {df['quality'].median():.2f}\n"
    stats_text += f"Std Dev: {df['quality'].std():.2f}\n"
    stats_text += f"Range: {df['quality'].min()} - {df['quality'].max()}"
    
    plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.8))
    
    plt.title('Distribution of Wine Quality', fontsize=14)
    plt.xlabel('Quality Rating', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(sorted(df['quality'].unique()))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    print("\nPotential quality groupings:")
    
    df_temp = df.copy()
    

    df_temp.loc[:, 'quality_binary'] = np.where(df_temp['quality'] >= 7, 'Good', 'Average/Poor')
    binary_counts = df_temp['quality_binary'].value_counts()
    print("\nBinary grouping (threshold = 7):")
    print(binary_counts)
    print(f"Percentage good: {binary_counts['Good']/len(df_temp)*100:.2f}%")
    
    bins = [0, 4, 6, 10]
    labels = ['Low', 'Medium', 'High']
    df_temp.loc[:, 'quality_group'] = pd.cut(df_temp['quality'], bins=bins, labels=labels, include_lowest=True)
    tertiary_counts = df_temp['quality_group'].value_counts().sort_index()
    print("\nTertiary grouping:")
    print(tertiary_counts)
    for label in labels:
        print(f"Percentage {label}: {tertiary_counts[label]/len(df_temp)*100:.2f}%")
    
   
    plt.figure(figsize=figsize)
    
    plt.subplot(1, 2, 1)
    sns.countplot(x='quality_binary', hue='quality_binary', data=df_temp, 
                 palette={'Average/Poor': '#4682B4', 'Good': '#000080'}, 
                 legend=False)
    plt.title('Binary Quality Grouping', fontsize=12)
    plt.xlabel('Quality Group', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    
    plt.subplot(1, 2, 2)
    sns.countplot(x='quality_group', hue='quality_group', data=df_temp, 
                 palette={'Low': '#A9CCE3', 'Medium': '#4682B4', 'High': '#000080'}, 
                 legend=False)
    plt.title('Tertiary Quality Grouping', fontsize=12)
    plt.xlabel('Quality Group', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    
    plt.tight_layout()
    plt.show()





    

def plot_feature_interactions(df, features, quality_col='quality', figsize=(12, 10)):
    """
    Creates pairplots and interaction visualizations for selected features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the features
    features : list
        List of feature names to analyze
    quality_col : str, default='quality'
        The name of the quality column to use for coloring
    figsize : tuple, default=(12, 10)
        Size of the figure
    """
    if quality_col == 'quality':
        bins = [0, 4, 6, 10]
        labels = ['Low', 'Medium', 'High']
        df_temp = df.copy()
        df_temp['quality_group'] = pd.cut(df_temp['quality'], bins=bins, labels=labels, include_lowest=True)
        color_col = 'quality_group'
    else:
        color_col = quality_col
    
    

    g = sns.pairplot(df_temp, vars=features, hue=color_col, 
                    plot_kws={'alpha': 0.6}, diag_kind='kde',
                    palette='viridis')
    g.fig.suptitle('Pairwise Relationships Between Features', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    if len(features) >= 2:
        feat1, feat2 = features[0], features[1]
        
        plt.figure(figsize=figsize)
        
        df_heatmap = df.copy()
        df_heatmap[f'{feat1}_bin'] = pd.qcut(df_heatmap[feat1], q=5, duplicates='drop')
        df_heatmap[f'{feat2}_bin'] = pd.qcut(df_heatmap[feat2], q=5, duplicates='drop')
        
        pivot = df_heatmap.pivot_table(index=f'{feat1}_bin', 
                                     columns=f'{feat2}_bin', 
                                     values='quality', 
                                     aggfunc='mean')
        
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis')
        plt.title(f'Mean Quality by {feat1} and {feat2}', fontsize=14)
        plt.xlabel(feat2, fontsize=12)
        plt.ylabel(feat1, fontsize=12)
        plt.tight_layout()
        plt.show()
        
        
        df_interaction = df.copy()
        df_interaction[f'{feat1}_{feat2}_interaction'] = df_interaction[feat1] * df_interaction[feat2]

        interaction_corr = df_interaction[[f'{feat1}_{feat2}_interaction', 'quality']].corr().iloc[0, 1]
        print(f"Correlation of interaction term with quality: {interaction_corr:.4f}")
        
        corr1 = df_interaction[[feat1, 'quality']].corr().iloc[0, 1]
        corr2 = df_interaction[[feat2, 'quality']].corr().iloc[0, 1]
        print(f"Correlation of {feat1} with quality: {corr1:.4f}")
        print(f"Correlation of {feat2} with quality: {corr2:.4f}")
        
        if abs(interaction_corr) > (abs(corr1) + abs(corr2))/2:
            print(f"The interaction between {feat1} and {feat2} may be meaningful.")
        else:
            print(f"The interaction between {feat1} and {feat2} doesn't appear to add much value beyond the individual effects.")






            
def create_domain_knowledge_features(df):
    """
    Creates new features based on domain knowledge about wine chemistry.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the features
        
    Returns:
    --------
    df_new : pandas.DataFrame
        Dataframe with new features added
    """
    df_new = df.copy()
    
    # 1. Total acidity (fixed + volatile)
    df_new['total_acidity'] = df_new['fixed acidity'] + df_new['volatile acidity']
    
    # 2. Ratio of free to total sulfur dioxide (indicates preservation effectiveness)
    df_new['free_to_total_so2_ratio'] = df_new['free sulfur dioxide'] / df_new['total sulfur dioxide']
    
    # 3. Acidity to alcohol ratio
    df_new['acidity_to_alcohol_ratio'] = df_new['total_acidity'] / df_new['alcohol']
    
    # 4. Sugar to acid ratio (indicates sweetness balance)
    df_new['sugar_to_acid_ratio'] = df_new['residual sugar'] / df_new['total_acidity']
    
    # 5. Sulfur to alcohol ratio (preservation level relative to alcohol)
    df_new['sulfur_to_alcohol_ratio'] = df_new['total sulfur dioxide'] / df_new['alcohol']
    
   
    print("Created the following new features based on domain knowledge:")
    print("1. total_acidity: Sum of fixed and volatile acidity")
    print("2. free_to_total_so2_ratio: Ratio of free to total sulfur dioxide")
    print("3. acidity_to_alcohol_ratio: Ratio of total acidity to alcohol")
    print("4. sugar_to_acid_ratio: Ratio of residual sugar to total acidity")
    print("5. sulfur_to_alcohol_ratio: Ratio of total sulfur dioxide to alcohol")
    
    
    new_features = ['total_acidity', 'free_to_total_so2_ratio', 'acidity_to_alcohol_ratio', 
                   'sugar_to_acid_ratio', 'sulfur_to_alcohol_ratio']
    
    corrs = df_new[new_features + ['quality']].corr()['quality'].drop('quality').sort_values(ascending=False)
    
    print("\nCorrelations of new features with quality:")
    display(corrs)
    
   
    return df_new





def perform_model_diagnostics(model, X_train, y_train, figsize=(12, 6)):
    """
    Performs basic diagnostic plots for a linear regression model.
    
    Parameters:
    -----------
    model : statsmodels regression model
        The fitted model
    X_train : pandas DataFrame
        Training features used to fit the model
    y_train : pandas Series
        Training target values
    figsize : tuple, default=(12, 6)
        Size of the figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    

    y_train_pred = model.predict(X_train)
    residuals = y_train - y_train_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].scatter(y_train_pred, residuals, alpha=0.5, color='darkblue')
    axes[0].axhline(y=0, color='red', linestyle='--')
    axes[0].set_xlabel('Fitted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Fitted Values')
    axes[0].grid(True, alpha=0.3)
    
    sns.histplot(residuals, kde=True, ax=axes[1], color='darkblue')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Residuals')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Residuals summary statistics:")
    print(f"Mean of residuals: {np.mean(residuals):.4f}")
    print(f"Standard deviation of residuals: {np.std(residuals):.4f}")
    
    from scipy import stats
    shapiro_test = stats.shapiro(residuals)
    print(f"\nShapiro-Wilk test for normality of residuals:")
    print(f"Test statistic: {shapiro_test[0]:.4f}")
    print(f"p-value: {shapiro_test[1]:.4f}")
    if shapiro_test[1] < 0.05:
        print("The residuals do not appear to be normally distributed (p < 0.05)")
    else:
        print("The residuals appear to be normally distributed (p >= 0.05)")



        

def backward_selection(X_train, y_train, initial_features, significance_level=0.05):
    """
    Perform backward selection to remove insignificant variables
    
    Parameters:
    -----------
    X_train : pandas DataFrame
        Training features dataframe
    y_train : pandas Series
        Training target variable
    initial_features : list
        Initial set of features to consider
    significance_level : float, default=0.05
        Threshold p-value for feature removal
        
    Returns:
    --------
    selected_features : list
        Final selected features
    final_model : statsmodels regression model
        Final fitted model
    """
    import statsmodels.api as sm
    
    selected = list(initial_features)
    
    print("Starting backward selection with features:", selected)
    
    while True:
        X_selected = X_train[selected]
        X_selected = sm.add_constant(X_selected)
        
        model = sm.OLS(y_train, X_selected).fit()
        
        p_values = model.pvalues.drop('const')
        
        max_p_value = p_values.max()
        feature_with_max_p = p_values.idxmax()
        
        if max_p_value > significance_level:
            selected.remove(feature_with_max_p)
            print(f"Removed {feature_with_max_p} with p-value {max_p_value:.4f}")
        else:
            print("All remaining features are significant (p < 0.05)")
            break
    
    X_final = sm.add_constant(X_train[selected])
    final_model = sm.OLS(y_train, X_final).fit()
    
    return selected, final_model






    
def evaluate_model_on_test_data(model, X_test, y_test, features, model_name="Model", figsize=(10, 6)):
    """
    Evaluates a regression model on test data and creates performance visualizations.
    
    Parameters:
    -----------
    model : statsmodels regression model
        The fitted model
    X_test : pandas DataFrame
        Test features
    y_test : pandas Series
        Test target values
    features : list
        List of features used in the model
    model_name : str, default="Model"
        Name to identify the model in outputs
    figsize : tuple, default=(10, 6)
        Size of the figure
        
    Returns:
    --------
    metrics : dict
        Dictionary with performance metrics
    """
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error
    
   
    X_test_sm = sm.add_constant(X_test[features])
    

    y_test_pred = model.predict(X_test_sm)

    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"{model_name} - Test Set Performance:")
    print(f"R-squared: {test_r2:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    
    plt.figure(figsize=figsize)
    plt.scatter(y_test, y_test_pred, alpha=0.5, color='darkblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Quality')
    plt.ylabel('Predicted Quality')
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

    metrics = {
        'r2': test_r2,
        'rmse': test_rmse
    }
    
    return metrics






def compare_model_predictions(y_test, y_pred_model1, y_pred_model2, model1_name="Initial Model", model2_name="Expanded Model", figsize=(12, 5)):
    """
    Creates side-by-side scatter plots comparing actual vs. predicted values for two models.
    
    Parameters:
    -----------
    y_test : pandas Series or numpy array
        Actual values from test set
    y_pred_model1 : pandas Series or numpy array
        Predictions from first model
    y_pred_model2 : pandas Series or numpy array
        Predictions from second model
    model1_name : str, default="Initial Model"
        Name of the first model to display in the plot title
    model2_name : str, default="Expanded Model"
        Name of the second model to display in the plot title
    figsize : tuple, default=(12, 5)
        Size of the figure
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=figsize)
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred_model1, alpha=0.5, color='darkblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Quality')
    plt.ylabel('Predicted Quality')
    plt.title(f'{model1_name}: Actual vs Predicted')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_model2, alpha=0.5, color='darkblue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Quality')
    plt.ylabel('Predicted Quality')
    plt.title(f'{model2_name}: Actual vs Predicted')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np
    
    r2_model1 = r2_score(y_test, y_pred_model1)
    rmse_model1 = np.sqrt(mean_squared_error(y_test, y_pred_model1))
    r2_model2 = r2_score(y_test, y_pred_model2)
    rmse_model2 = np.sqrt(mean_squared_error(y_test, y_pred_model2))
    
    print("Model Evaluation on Test Data:")
    print(f"\n{model1_name}:")
    print(f"R-squared: {r2_model1:.4f}")
    print(f"RMSE: {rmse_model1:.4f}")
    
    print(f"\n{model2_name}:")
    print(f"R-squared: {r2_model2:.4f}")
    print(f"RMSE: {rmse_model2:.4f}")
    
    print(f"\nImprovement in R-squared: {r2_model2 - r2_model1:.4f}")
    print(f"Improvement in RMSE: {rmse_model1 - rmse_model2:.4f}")
    
    return {
        'model1_r2': r2_model1,
        'model1_rmse': rmse_model1,
        'model2_r2': r2_model2,
        'model2_rmse': rmse_model2
    }