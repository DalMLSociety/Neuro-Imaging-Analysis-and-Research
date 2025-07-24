import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def main():
    # 1. Load the data from the Excel file
    df = pd.read_excel('demoSCA7.xlsx')

    # 2. Create 'subject' (each 3 rows per person) and 'year' (1–3) columns
    df['subject'] = df.index // 3 + 1
    df['year']    = df.groupby('subject').cumcount() + 1

    # 3. Time‑series line plots for all six measures
    measures = ['yearOnset', 'cag', 'school', 'moca', 'mmse', 'sara']
    for measure in measures:
        fig, ax = plt.subplots(figsize=(8, 5))
        for subj in df['subject'].unique():
            sub_df = df[df['subject'] == subj]
            ax.plot(
                sub_df['year'],
                sub_df[measure],
                marker='o',
                label=f'P{subj}'
            )
        ax.set_title(f'{measure} over 3 Years')
        ax.set_xlabel('Year')
        ax.set_ylabel(measure)
        ax.legend(ncol=2, fontsize='small', loc='best')
        plt.tight_layout()
        fig.savefig(f'5_{measure}_over_3_years.png', dpi=150)
        plt.close(fig)

    # 4. Scatter matrix to examine pairwise relationships
    axes = scatter_matrix(
        df[measures],
        diagonal='hist',
        alpha=0.7,
        figsize=(12, 12)
    )
    plt.suptitle('Scatter Matrix of Onset, CAG, Education & Scores', y=1.02)
    plt.tight_layout()
    plt.savefig('5_extended_scatter_matrix.png', dpi=150)
    plt.close()

    # 5. Print the correlation matrix for all six variables
    corr = df[measures].corr()
    print("Correlation matrix (yearOnset, cag, school, moca, mmse, sara):")
    print(corr)

if __name__ == '__main__':
    main()
