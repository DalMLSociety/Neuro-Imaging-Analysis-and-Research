#!/usr/bin/env python3
"""
Simple SCA Pattern Analysis
===========================

A streamlined analysis to identify SCA biomarkers from brain imaging data.
This version is optimized for efficiency while maintaining scientific rigor.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn.input_data import NiftiMasker
import warnings
warnings.filterwarnings('ignore')

def load_brain_data(data_dir, max_files=10):
    """Load and process brain data efficiently"""
    print("Loading brain imaging data...")
    
    # Get file lists
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.nii.gz')]
    patient_files = sorted([f for f in all_files if f.startswith('Denoised_p')])[:max_files]
    control_files = sorted([f for f in all_files if f.startswith('Denoised_C')])[:max_files]
    
    print(f"Selected {len(patient_files)} patient files and {len(control_files)} control files")
    
    # Create simple masker
    sample_file = os.path.join(data_dir, patient_files[0])
    masker = NiftiMasker(
        mask_strategy='background',
        standardize=True,
        smoothing_fwhm=6
    )
    
    # Extract signals from all files
    all_signals = []
    all_labels = []
    all_metadata = []
    
    print("Processing files...")
    
    # Process patients
    for i, filename in enumerate(patient_files):
        try:
            filepath = os.path.join(data_dir, filename)
            img = nib.load(filepath)
            
            if i == 0:  # Fit masker on first image
                signals = masker.fit_transform(img)
            else:
                signals = masker.transform(img)
            
            # Take mean across time
            mean_signal = signals.mean(axis=0)
            all_signals.append(mean_signal)
            all_labels.append(1)  # Patient = 1
            all_metadata.append({'group': 'patient', 'file': filename})
            
            print(f"  Processed patient {i+1}/{len(patient_files)}")
            
        except Exception as e:
            print(f"  Error with {filename}: {e}")
            continue
    
    # Process controls
    for i, filename in enumerate(control_files):
        try:
            filepath = os.path.join(data_dir, filename)
            img = nib.load(filepath)
            signals = masker.transform(img)
            
            # Take mean across time
            mean_signal = signals.mean(axis=0)
            all_signals.append(mean_signal)
            all_labels.append(0)  # Control = 0
            all_metadata.append({'group': 'control', 'file': filename})
            
            print(f"  Processed control {i+1}/{len(control_files)}")
            
        except Exception as e:
            print(f"  Error with {filename}: {e}")
            continue
    
    brain_signals = np.array(all_signals)
    labels = np.array(all_labels)
    
    print(f"Data loading completed: {brain_signals.shape}")
    print(f"Patients: {np.sum(labels == 1)}, Controls: {np.sum(labels == 0)}")
    
    return brain_signals, labels, all_metadata, masker

def perform_component_analysis(brain_signals, n_components=15):
    """Perform PCA and ICA analysis"""
    print("Performing component analysis...")
    
    # Standardize data
    scaler = StandardScaler()
    signals_scaled = scaler.fit_transform(brain_signals)
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=30, random_state=42)
    pca_signals = pca.fit_transform(signals_scaled)
    
    # ICA for independent components
    ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
    ica_signals = ica.fit_transform(pca_signals)
    
    print(f"Component analysis completed")
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    print(f"ICA components shape: {ica_signals.shape}")
    
    return ica_signals, pca, ica, scaler

def identify_biomarkers(ica_signals, labels):
    """Identify biomarker components"""
    print("Identifying biomarker components...")
    
    patient_mask = labels == 1
    control_mask = labels == 0
    
    biomarker_results = []
    
    for comp_idx in range(ica_signals.shape[1]):
        patient_values = ica_signals[patient_mask, comp_idx]
        control_values = ica_signals[control_mask, comp_idx]
        
        # Statistical tests
        t_stat, p_value = ttest_ind(patient_values, control_values)
        u_stat, p_mw = mannwhitneyu(patient_values, control_values, alternative='two-sided')
        
        # Effect size
        pooled_std = np.sqrt(((len(patient_values) - 1) * np.var(patient_values, ddof=1) + 
                             (len(control_values) - 1) * np.var(control_values, ddof=1)) / 
                            (len(patient_values) + len(control_values) - 2))
        
        cohens_d = (np.mean(patient_values) - np.mean(control_values)) / pooled_std if pooled_std > 0 else 0
        
        result = {
            'component': comp_idx,
            'patient_mean': np.mean(patient_values),
            'patient_std': np.std(patient_values),
            'control_mean': np.mean(control_values),
            'control_std': np.std(control_values),
            'p_value': p_value,
            'p_value_mw': p_mw,
            'cohens_d': cohens_d,
            'abs_cohens_d': abs(cohens_d),
            'significant': (p_value < 0.05) and (abs(cohens_d) > 0.3)
        }
        
        biomarker_results.append(result)
    
    biomarker_df = pd.DataFrame(biomarker_results)
    significant_components = biomarker_df[biomarker_df['significant']]['component'].tolist()
    
    print(f"Found {len(significant_components)} significant biomarker components")
    
    return biomarker_df, significant_components

def train_classifier(ica_signals, labels, significant_components):
    """Train classifier on significant components"""
    print("Training classifier...")
    
    if len(significant_components) == 0:
        print("No significant components for classification")
        return None
    
    # Use significant components as features
    features = ica_signals[:, significant_components]
    
    # Train Random Forest
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Cross-validation
    cv_scores = cross_val_score(classifier, features, labels, cv=5)
    
    # Fit classifier
    classifier.fit(features, labels)
    
    results = {
        'cv_scores': cv_scores,
        'mean_accuracy': cv_scores.mean(),
        'std_accuracy': cv_scores.std(),
        'feature_importance': classifier.feature_importances_,
        'classifier': classifier
    }
    
    print(f"Classification accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return results

def create_visualizations(biomarker_df, ica_signals, labels, classification_results, 
                         significant_components, output_dir):
    """Create analysis visualizations"""
    print("Creating visualizations...")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set up plotting
    plt.style.use('default')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Significance volcano plot
    ax = axes[0, 0]
    colors = ['red' if sig else 'lightblue' for sig in biomarker_df['significant']]
    ax.scatter(biomarker_df['cohens_d'], -np.log10(biomarker_df['p_value']), 
              c=colors, alpha=0.7)
    ax.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.7)
    ax.axvline(0.3, color='gray', linestyle='--', alpha=0.7)
    ax.axvline(-0.3, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel("Cohen's d (Effect Size)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("Biomarker Significance")
    ax.grid(True, alpha=0.3)
    
    # 2. Top significant components
    ax = axes[0, 1]
    if len(significant_components) > 0:
        sig_data = biomarker_df[biomarker_df['significant']].sort_values('abs_cohens_d', ascending=False)
        top_5 = sig_data.head(5)
        ax.barh(range(len(top_5)), top_5['abs_cohens_d'], color='darkred', alpha=0.7)
        ax.set_yticks(range(len(top_5)))
        ax.set_yticklabels([f"Comp {c}" for c in top_5['component']])
        ax.set_xlabel("|Effect Size|")
        ax.set_title("Top Biomarker Components")
        ax.grid(True, alpha=0.3)
    
    # 3. Classification performance
    ax = axes[0, 2]
    if classification_results:
        cv_scores = classification_results['cv_scores']
        ax.bar(range(len(cv_scores)), cv_scores, color='darkblue', alpha=0.7)
        ax.axhline(cv_scores.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {cv_scores.mean():.3f}')
        ax.set_xlabel("CV Fold")
        ax.set_ylabel("Accuracy")
        ax.set_title("Classification Performance")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Sample distribution
    ax = axes[1, 0]
    groups = ['Patients', 'Controls']
    counts = [np.sum(labels == 1), np.sum(labels == 0)]
    ax.pie(counts, labels=groups, colors=['red', 'blue'], autopct='%1.1f%%', alpha=0.7)
    ax.set_title("Dataset Composition")
    
    # 5. Effect size distribution
    ax = axes[1, 1]
    ax.hist(biomarker_df['abs_cohens_d'], bins=10, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(0.3, color='red', linestyle='--', label='Threshold')
    ax.set_xlabel('|Effect Size|')
    ax.set_ylabel('Frequency')
    ax.set_title('Effect Size Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Top component distribution
    ax = axes[1, 2]
    if len(significant_components) > 0:
        top_comp = significant_components[0]
        patient_mask = labels == 1
        control_mask = labels == 0
        
        patient_values = ica_signals[patient_mask, top_comp]
        control_values = ica_signals[control_mask, top_comp]
        
        ax.boxplot([patient_values, control_values], labels=['Patients', 'Controls'],
                  patch_artist=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax.set_ylabel('Component Score')
        ax.set_title(f'Top Component {top_comp}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sca_biomarker_analysis.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def generate_report(biomarker_df, classification_results, significant_components, 
                   labels, output_dir):
    """Generate analysis report"""
    print("Generating report...")
    
    # Create summary statistics
    n_total = len(labels)
    n_patients = np.sum(labels == 1)
    n_controls = np.sum(labels == 0)
    n_significant = len(significant_components)
    
    report_content = f"""
# SCA Brain Pattern Analysis Report

## Executive Summary
This analysis identified **{n_significant}** significant biomarker components from brain imaging data using Independent Component Analysis (ICA).

## Dataset Overview
- **Total participants**: {n_total}
- **SCA patients**: {n_patients}
- **Healthy controls**: {n_controls}
- **Significant biomarkers found**: {n_significant}

## Key Findings

### Significant Biomarker Components
"""
    
    if n_significant > 0:
        sig_biomarkers = biomarker_df[biomarker_df['significant']].sort_values('abs_cohens_d', ascending=False)
        
        for i, (_, biomarker) in enumerate(sig_biomarkers.head(5).iterrows()):
            report_content += f"""
#### Component {biomarker['component']}
- **Effect Size (Cohen's d)**: {biomarker['cohens_d']:.3f}
- **P-value**: {biomarker['p_value']:.4f}
- **Patient Mean**: {biomarker['patient_mean']:.3f} ± {biomarker['patient_std']:.3f}
- **Control Mean**: {biomarker['control_mean']:.3f} ± {biomarker['control_std']:.3f}
"""
    
    if classification_results:
        report_content += f"""
### Classification Results
- **Cross-validation Accuracy**: {classification_results['mean_accuracy']:.3f} ± {classification_results['std_accuracy']:.3f}
- **Number of biomarker features**: {len(significant_components)}
"""
    
    report_content += """
### Methodology
1. **Data Processing**: Spatial smoothing and standardization
2. **Dimensionality Reduction**: PCA followed by ICA
3. **Statistical Testing**: Independent t-tests and Mann-Whitney U tests
4. **Effect Size**: Cohen's d calculation
5. **Classification**: Random Forest with 5-fold cross-validation

### Clinical Implications
- Network-level biomarkers could help monitor disease progression
- Components may reflect specific aspects of SCA pathology
- Potential for early detection and treatment monitoring

### Recommendations
1. Validate findings in larger cohorts
2. Correlate with clinical severity measures
3. Investigate longitudinal changes
4. Consider therapeutic trial applications

---
*Analysis completed successfully*
"""
    
    # Save report
    report_file = os.path.join(output_dir, 'sca_analysis_report.md')
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    # Save data
    biomarker_df.to_csv(os.path.join(output_dir, 'biomarker_components.csv'), index=False)
    
    print(f"Report saved to {report_file}")
    
    return report_content

def main():
    """Main analysis function"""
    print("="*60)
    print("SCA Brain Pattern Analysis")
    print("="*60)
    
    # Configuration
    data_directory = "extracted data/raw_data"
    output_directory = "sca_analysis_results"
    
    try:
        # Step 1: Load brain data
        brain_signals, labels, metadata, masker = load_brain_data(data_directory, max_files=12)
        
        # Step 2: Component analysis
        ica_signals, pca, ica, scaler = perform_component_analysis(brain_signals, n_components=15)
        
        # Step 3: Identify biomarkers
        biomarker_df, significant_components = identify_biomarkers(ica_signals, labels)
        
        # Step 4: Train classifier
        classification_results = train_classifier(ica_signals, labels, significant_components)
        
        # Step 5: Create visualizations
        create_visualizations(biomarker_df, ica_signals, labels, classification_results,
                            significant_components, output_directory)
        
        # Step 6: Generate report
        generate_report(biomarker_df, classification_results, significant_components,
                       labels, output_directory)
        
        print("="*60)
        print("✅ Analysis Completed Successfully!")
        print(f"📁 Results saved to: {output_directory}")
        print(f"🧬 Found {len(significant_components)} significant biomarkers")
        
        if classification_results:
            print(f"🎯 Classification accuracy: {classification_results['mean_accuracy']:.3f}")
        
        print("="*60)
        
        return output_directory
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results_dir = main()
    
    if results_dir:
        print(f"\n🎉 Analysis complete! Check '{results_dir}' for results.")
        print("📊 Key files:")
        print("   - sca_biomarker_analysis.png")
        print("   - sca_analysis_report.md") 
        print("   - biomarker_components.csv")
    else:
        print("\n❌ Analysis failed. Please check error messages above.") 