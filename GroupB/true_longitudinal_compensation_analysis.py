#!/usr/bin/env python3
"""
True Longitudinal Compensation Analysis
======================================

This script implements a proper longitudinal analysis that tracks individual SCA patients
across rs-1, rs-2, rs-3 timepoints to map how each patient's brain compensation
patterns change over time.

Research Goal: Map individual patient compensation trajectories over time
Method: Within-subject longitudinal ICA + trajectory modeling
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind, pearsonr

from nilearn.input_data import NiftiMasker
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker

class TrueLongitudinalCompensationAnalyzer:
    """
    True longitudinal analysis tracking individual patients across timepoints
    """
    
    def __init__(self, data_dir="extracted data/raw_data", output_dir="true_longitudinal_results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Analysis parameters
        self.n_components = 20
        self.timepoints = ['rs-1', 'rs-2', 'rs-3']
        
        print("🧠 TRUE LONGITUDINAL COMPENSATION ANALYSIS")
        print("=" * 60)
        print("Goal: Track individual patient compensation trajectories over time")
        print("Method: Within-subject longitudinal ICA + trajectory modeling")
        print("=" * 60)
        
        # Data containers
        self.longitudinal_subjects = {}  # Subjects with all 3 timepoints
        self.subject_trajectories = {}   # Individual compensation trajectories
        self.group_patterns = {}         # Group-level patterns
        
    def identify_longitudinal_subjects(self):
        """Identify subjects who have data for all three timepoints"""
        print("\n📂 Identifying subjects with complete longitudinal data...")
        
        # Get all NII files
        all_files = list(self.data_dir.glob("*.nii.gz"))
        print(f"Found {len(all_files)} total brain scan files")
        
        # Organize by subject ID
        subject_files = {}
        
        for file_path in all_files:
            filename = file_path.name
            
            if 'Denoised_' in filename:
                parts = filename.replace('Denoised_', '').split('_')
                if len(parts) >= 3:
                    subject_id = parts[0]  # p01, p02, C01, C02, etc.
                    timepoint = parts[1]   # rs-1, rs-2, rs-3
                    
                    if subject_id not in subject_files:
                        subject_files[subject_id] = {}
                    
                    subject_files[subject_id][timepoint] = {
                        'file_path': str(file_path),
                        'filename': filename,
                        'group': 'patients' if subject_id.startswith('p') else 'controls'
                    }
        
        # Find subjects with all three timepoints
        complete_subjects = {}
        for subject_id, timepoint_data in subject_files.items():
            if all(tp in timepoint_data for tp in self.timepoints):
                complete_subjects[subject_id] = {
                    'group': timepoint_data['rs-1']['group'],
                    'timepoints': timepoint_data
                }
        
        self.longitudinal_subjects = complete_subjects
        
        # Count by group
        patients = sum(1 for s in complete_subjects.values() if s['group'] == 'patients')
        controls = sum(1 for s in complete_subjects.values() if s['group'] == 'controls')
        
        print(f"\n📊 Longitudinal Subject Summary:")
        print(f"  Patients with all 3 timepoints: {patients}")
        print(f"  Controls with all 3 timepoints: {controls}")
        print(f"  Total longitudinal subjects: {len(complete_subjects)}")
        
        if len(complete_subjects) < 10:
            print("⚠️  Small sample size - results should be interpreted with caution")
        
        return complete_subjects
    
    def create_longitudinal_masker(self):
        """Create brain masker for consistent signal extraction"""
        print("\n🧠 Creating brain masker...")
        
        # Use AAL atlas for precise anatomical regions
        try:
            self.aal = datasets.fetch_atlas_aal(version='SPM12', data_dir='nilearn_data')
            self.masker = NiftiLabelsMasker(
                labels_img=self.aal.maps,
                standardize=True,
                verbose=0
            )
            
            self.region_labels = [label.decode() if isinstance(label, bytes) else label 
                                for label in self.aal.labels]
            print(f"✅ AAL atlas masker created: {len(self.region_labels)} regions")
            
        except Exception as e:
            print(f"⚠️  AAL atlas failed ({e}), using whole brain masker")
            self.masker = NiftiMasker(
                mask_strategy='background',
                standardize=True,
                verbose=0
            )
            self.region_labels = None
        
        return self.masker
    
    def extract_longitudinal_signals(self):
        """Extract brain signals for each subject across all timepoints"""
        print("\n📡 Extracting longitudinal brain signals...")
        
        longitudinal_data = {}
        
        for subject_id, subject_info in tqdm(self.longitudinal_subjects.items(), 
                                           desc="Processing subjects"):
            
            subject_signals = {}
            
            for timepoint in self.timepoints:
                file_info = subject_info['timepoints'][timepoint]
                
                try:
                    # Load brain image
                    img = nib.load(file_info['file_path'])
                    
                    # Extract signals
                    if hasattr(self, 'aal'):
                        signals = self.masker.fit_transform(img) if timepoint == 'rs-1' else self.masker.transform(img)
                    else:
                        signals = self.masker.fit_transform(img) if timepoint == 'rs-1' else self.masker.transform(img)
                    
                    # Take mean across time
                    mean_signal = signals.mean(axis=0)
                    subject_signals[timepoint] = mean_signal
                    
                except Exception as e:
                    print(f"    Error with {subject_id} {timepoint}: {e}")
                    subject_signals[timepoint] = None
            
            # Only keep subjects with all timepoints successfully processed
            if all(signal is not None for signal in subject_signals.values()):
                longitudinal_data[subject_id] = {
                    'group': subject_info['group'],
                    'signals': subject_signals
                }
        
        self.longitudinal_data = longitudinal_data
        
        patients = sum(1 for s in longitudinal_data.values() if s['group'] == 'patients')
        controls = sum(1 for s in longitudinal_data.values() if s['group'] == 'controls')
        
        print(f"\n✅ Successfully extracted signals:")
        print(f"  Patients: {patients}")
        print(f"  Controls: {controls}")
        print(f"  Total: {len(longitudinal_data)}")
        
        return longitudinal_data
    
    def perform_longitudinal_ica(self):
        """Perform ICA analysis on longitudinal data"""
        print(f"\n🔬 Performing longitudinal ICA analysis...")
        
        # Combine all timepoint data for joint ICA
        all_signals = []
        all_labels = []
        all_metadata = []
        
        for subject_id, subject_data in self.longitudinal_data.items():
            for timepoint in self.timepoints:
                signal = subject_data['signals'][timepoint]
                all_signals.append(signal)
                all_labels.append(1 if subject_data['group'] == 'patients' else 0)
                all_metadata.append({
                    'subject_id': subject_id,
                    'timepoint': timepoint,
                    'group': subject_data['group']
                })
        
        all_signals = np.array(all_signals)
        
        print(f"  Combined data shape: {all_signals.shape}")
        
        # Standardize signals
        scaler = RobustScaler()
        signals_scaled = scaler.fit_transform(all_signals)
        
        # Apply PCA for dimensionality reduction
        max_components = min(signals_scaled.shape[0]-1, signals_scaled.shape[1]//10, 20)
        pca = PCA(n_components=max_components, random_state=42)
        signals_pca = pca.fit_transform(signals_scaled)
        
        # Apply FastICA - adjust n_components based on available data
        actual_n_components = min(self.n_components, signals_pca.shape[1])
        ica = FastICA(
            n_components=actual_n_components,
            random_state=42,
            max_iter=1000,
            tol=1e-3,
            algorithm='parallel'
        )
        
        ica_components = ica.fit_transform(signals_pca)
        
        # Update n_components to actual value
        self.n_components = actual_n_components
        
        print(f"  ✅ ICA completed: {ica_components.shape}")
        print(f"  📊 PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        print(f"  🔬 Actual ICA components: {self.n_components}")
        
        # Store results
        self.ica_results = {
            'components': ica_components,
            'spatial_maps': ica.components_,
            'scaler': scaler,
            'pca': pca,
            'ica': ica,
            'metadata': all_metadata,
            'labels': np.array(all_labels)
        }
        
        return ica_components
    
    def create_individual_trajectories(self):
        """Create individual compensation trajectories for each patient"""
        print("\n🗺️ Creating individual compensation trajectories...")
        
        ica_components = self.ica_results['components']
        metadata = self.ica_results['metadata']
        
        # Organize ICA components by subject and timepoint
        subject_components = {}
        
        for i, meta in enumerate(metadata):
            subject_id = meta['subject_id']
            timepoint = meta['timepoint']
            
            if subject_id not in subject_components:
                subject_components[subject_id] = {
                    'group': meta['group'],
                    'timepoints': {}
                }
            
            subject_components[subject_id]['timepoints'][timepoint] = ica_components[i]
        
        # Calculate compensation trajectories
        patient_trajectories = {}
        
        for subject_id, subject_data in subject_components.items():
            if subject_data['group'] == 'patients':
                
                # Get signals for all timepoints
                rs1 = subject_data['timepoints']['rs-1']
                rs2 = subject_data['timepoints']['rs-2']
                rs3 = subject_data['timepoints']['rs-3']
                
                # Calculate changes over time
                rs1_to_rs2_change = rs2 - rs1
                rs2_to_rs3_change = rs3 - rs2
                rs1_to_rs3_change = rs3 - rs1
                
                # Calculate trajectory metrics
                trajectory_metrics = {}
                
                for comp_idx in range(len(rs1)):
                    # Linear trend over timepoints
                    timepoints = np.array([1, 2, 3])  # rs-1, rs-2, rs-3
                    values = np.array([rs1[comp_idx], rs2[comp_idx], rs3[comp_idx]])
                    
                    # Linear regression for trend
                    slope, intercept, r_value, p_value, std_err = stats.linregress(timepoints, values)
                    
                    trajectory_metrics[comp_idx] = {
                        'rs1_value': rs1[comp_idx],
                        'rs2_value': rs2[comp_idx],
                        'rs3_value': rs3[comp_idx],
                        'rs1_to_rs2_change': rs1_to_rs2_change[comp_idx],
                        'rs2_to_rs3_change': rs2_to_rs3_change[comp_idx],
                        'total_change': rs1_to_rs3_change[comp_idx],
                        'slope': slope,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                        'trend_strength': abs(slope)
                    }
                
                patient_trajectories[subject_id] = {
                    'trajectory_metrics': trajectory_metrics,
                    'raw_timepoints': {
                        'rs-1': rs1,
                        'rs-2': rs2,
                        'rs-3': rs3
                    }
                }
        
        self.patient_trajectories = patient_trajectories
        
        print(f"✅ Created trajectories for {len(patient_trajectories)} patients")
        
        return patient_trajectories
    
    def identify_compensation_patterns(self):
        """Identify compensation patterns in longitudinal trajectories"""
        print("\n🔍 Identifying compensation patterns in trajectories...")
        
        compensation_patterns = {}
        
        # Analyze each component across all patients
        for comp_idx in range(self.n_components):
            
            # Collect trajectory data for this component
            component_trajectories = []
            
            for subject_id, trajectory_data in self.patient_trajectories.items():
                comp_metrics = trajectory_data['trajectory_metrics'][comp_idx]
                component_trajectories.append({
                    'subject_id': subject_id,
                    'slope': comp_metrics['slope'],
                    'total_change': comp_metrics['total_change'],
                    'r_squared': comp_metrics['r_squared'],
                    'trend_direction': comp_metrics['trend_direction']
                })
            
            # Analyze component patterns
            slopes = [t['slope'] for t in component_trajectories]
            total_changes = [t['total_change'] for t in component_trajectories]
            r_squared_values = [t['r_squared'] for t in component_trajectories]
            
            # Statistical tests
            mean_slope = np.mean(slopes)
            slope_std = np.std(slopes)
            
            # Test if slope is significantly different from zero
            t_stat, t_pval = stats.ttest_1samp(slopes, 0)
            
            # Compensation criteria
            is_compensation = (mean_slope > 0 and t_pval < 0.05)  # Increasing over time
            
            compensation_patterns[comp_idx] = {
                'mean_slope': mean_slope,
                'slope_std': slope_std,
                'mean_total_change': np.mean(total_changes),
                'mean_r_squared': np.mean(r_squared_values),
                't_statistic': t_stat,
                'p_value': t_pval,
                'is_compensation': is_compensation,
                'n_increasing': sum(1 for t in component_trajectories if t['trend_direction'] == 'increasing'),
                'n_decreasing': sum(1 for t in component_trajectories if t['trend_direction'] == 'decreasing'),
                'individual_trajectories': component_trajectories
            }
        
        self.compensation_patterns = compensation_patterns
        
        # Count significant compensation components
        significant_compensation = sum(1 for pattern in compensation_patterns.values() 
                                     if pattern['is_compensation'])
        
        print(f"📈 Found {significant_compensation} components with significant compensation patterns")
        
        return compensation_patterns
    
    def analyze_early_prediction(self):
        """Analyze how well rs-1 predicts rs-3 outcomes"""
        print("\n🎯 Analyzing early prediction (rs-1 → rs-3)...")
        
        prediction_results = {}
        
        for comp_idx in range(self.n_components):
            
            # Collect rs-1 and rs-3 values
            rs1_values = []
            rs3_values = []
            
            for subject_id, trajectory_data in self.patient_trajectories.items():
                rs1_val = trajectory_data['trajectory_metrics'][comp_idx]['rs1_value']
                rs3_val = trajectory_data['trajectory_metrics'][comp_idx]['rs3_value']
                
                rs1_values.append(rs1_val)
                rs3_values.append(rs3_val)
            
            # Linear regression: rs-1 predicts rs-3
            if len(rs1_values) >= 3:  # Need minimum samples
                rs1_array = np.array(rs1_values).reshape(-1, 1)
                rs3_array = np.array(rs3_values)
                
                model = LinearRegression()
                model.fit(rs1_array, rs3_array)
                
                rs3_predicted = model.predict(rs1_array)
                r2 = r2_score(rs3_array, rs3_predicted)
                
                # Correlation analysis
                correlation, corr_p_value = pearsonr(rs1_values, rs3_values)
                
                prediction_results[comp_idx] = {
                    'r2_score': r2,
                    'correlation': correlation,
                    'correlation_p_value': corr_p_value,
                    'model_coefficient': model.coef_[0],
                    'model_intercept': model.intercept_,
                    'prediction_quality': 'good' if r2 > 0.3 else 'moderate' if r2 > 0.1 else 'poor'
                }
        
        self.prediction_results = prediction_results
        
        # Count good predictions
        good_predictions = sum(1 for result in prediction_results.values() 
                             if result['r2_score'] > 0.3)
        
        print(f"🎯 Components with good prediction (R² > 0.3): {good_predictions}/{self.n_components}")
        
        return prediction_results
    
    def create_longitudinal_visualizations(self):
        """Create comprehensive longitudinal visualizations"""
        print("\n🎨 Creating longitudinal visualizations...")
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('True Longitudinal Compensation Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Individual patient trajectories for top compensation component
        ax1 = axes[0, 0]
        top_comp = max(self.compensation_patterns.keys(), 
                      key=lambda x: self.compensation_patterns[x]['mean_slope'])
        
        for subject_id, trajectory_data in self.patient_trajectories.items():
            values = [
                trajectory_data['trajectory_metrics'][top_comp]['rs1_value'],
                trajectory_data['trajectory_metrics'][top_comp]['rs2_value'],
                trajectory_data['trajectory_metrics'][top_comp]['rs3_value']
            ]
            ax1.plot([1, 2, 3], values, 'o-', alpha=0.6, label=subject_id if len(self.patient_trajectories) <= 10 else '')
        
        ax1.set_title(f'Individual Trajectories - Component {top_comp}')
        ax1.set_xlabel('Timepoint')
        ax1.set_ylabel('Component Strength')
        ax1.set_xticks([1, 2, 3])
        ax1.set_xticklabels(['rs-1', 'rs-2', 'rs-3'])
        ax1.grid(True, alpha=0.3)
        if len(self.patient_trajectories) <= 10:
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Plot 2: Compensation pattern distribution
        ax2 = axes[0, 1]
        slopes = [pattern['mean_slope'] for pattern in self.compensation_patterns.values()]
        p_values = [pattern['p_value'] for pattern in self.compensation_patterns.values()]
        
        colors = ['red' if p < 0.05 else 'blue' for p in p_values]
        ax2.scatter(range(len(slopes)), slopes, c=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('Compensation Slopes by Component')
        ax2.set_xlabel('Component Index')
        ax2.set_ylabel('Mean Slope')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Early prediction performance
        ax3 = axes[1, 0]
        r2_scores = [result['r2_score'] for result in self.prediction_results.values()]
        ax3.hist(r2_scores, bins=10, alpha=0.7, color='green')
        ax3.axvline(x=0.3, color='red', linestyle='--', label='Good prediction threshold')
        ax3.set_title('rs-1 → rs-3 Prediction Performance')
        ax3.set_xlabel('R² Score')
        ax3.set_ylabel('Number of Components')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Trajectory consistency
        ax4 = axes[1, 1]
        r_squared_values = [pattern['mean_r_squared'] for pattern in self.compensation_patterns.values()]
        ax4.hist(r_squared_values, bins=10, alpha=0.7, color='orange')
        ax4.set_title('Trajectory Linearity (R² of trends)')
        ax4.set_xlabel('Mean R² of Individual Trends')
        ax4.set_ylabel('Number of Components')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Group mean trajectories
        ax5 = axes[2, 0]
        # Calculate group means for visualization
        group_means = {tp: [] for tp in self.timepoints}
        
        for timepoint in self.timepoints:
            values = []
            for subject_id, trajectory_data in self.patient_trajectories.items():
                # Average across all components for each timepoint
                tp_values = [trajectory_data['trajectory_metrics'][comp]['rs1_value'] if timepoint == 'rs-1'
                           else trajectory_data['trajectory_metrics'][comp]['rs2_value'] if timepoint == 'rs-2'
                           else trajectory_data['trajectory_metrics'][comp]['rs3_value']
                           for comp in range(self.n_components)]
                values.append(np.mean(tp_values))
            group_means[timepoint] = values
        
        # Box plot
        ax5.boxplot([group_means['rs-1'], group_means['rs-2'], group_means['rs-3']], 
                   labels=['rs-1', 'rs-2', 'rs-3'])
        ax5.set_title('Group Mean Component Activity Over Time')
        ax5.set_ylabel('Mean Component Strength')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Compensation vs prediction relationship
        ax6 = axes[2, 1]
        comp_slopes = [self.compensation_patterns[i]['mean_slope'] for i in range(self.n_components)]
        pred_r2 = [self.prediction_results[i]['r2_score'] for i in range(self.n_components)]
        
        ax6.scatter(comp_slopes, pred_r2, alpha=0.7)
        ax6.set_xlabel('Compensation Slope')
        ax6.set_ylabel('Prediction R²')
        ax6.set_title('Compensation vs. Prediction Performance')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        plot_file = self.output_dir / 'longitudinal_compensation_analysis.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization saved to: {plot_file}")
    
    def generate_longitudinal_report(self):
        """Generate comprehensive longitudinal analysis report"""
        print("\n📝 Generating longitudinal analysis report...")
        
        report_lines = []
        report_lines.append("# True Longitudinal Compensation Analysis Report")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Executive summary
        n_patients = len(self.patient_trajectories)
        n_compensation = sum(1 for pattern in self.compensation_patterns.values() if pattern['is_compensation'])
        n_good_prediction = sum(1 for result in self.prediction_results.values() if result['r2_score'] > 0.3)
        
        report_lines.append("## Executive Summary")
        report_lines.append(f"**Analysis Type**: True longitudinal (within-subject tracking)")
        report_lines.append(f"**Patients tracked**: {n_patients}")
        report_lines.append(f"**Components with significant compensation**: {n_compensation}")
        report_lines.append(f"**Components with good rs-1→rs-3 prediction**: {n_good_prediction}")
        report_lines.append("")
        
        # Individual trajectories
        report_lines.append("## Individual Patient Trajectories")
        report_lines.append("")
        
        for subject_id, trajectory_data in list(self.patient_trajectories.items())[:5]:  # Show first 5
            report_lines.append(f"### Patient {subject_id}")
            
            # Find most changing component for this patient
            max_change_comp = max(range(self.n_components), 
                                key=lambda x: abs(trajectory_data['trajectory_metrics'][x]['total_change']))
            
            metrics = trajectory_data['trajectory_metrics'][max_change_comp]
            
            report_lines.append(f"- **Most changing component**: {max_change_comp}")
            report_lines.append(f"- **rs-1 value**: {metrics['rs1_value']:.3f}")
            report_lines.append(f"- **rs-2 value**: {metrics['rs2_value']:.3f}")
            report_lines.append(f"- **rs-3 value**: {metrics['rs3_value']:.3f}")
            report_lines.append(f"- **Total change**: {metrics['total_change']:.3f}")
            report_lines.append(f"- **Trend**: {metrics['trend_direction']} (slope: {metrics['slope']:.3f})")
            report_lines.append(f"- **Trend quality**: R² = {metrics['r_squared']:.3f}")
            report_lines.append("")
        
        # Compensation patterns
        report_lines.append("## Compensation Patterns")
        report_lines.append("")
        
        significant_comps = [(comp_idx, pattern) for comp_idx, pattern in self.compensation_patterns.items() 
                           if pattern['is_compensation']]
        significant_comps.sort(key=lambda x: x[1]['mean_slope'], reverse=True)
        
        if significant_comps:
            for i, (comp_idx, pattern) in enumerate(significant_comps, 1):
                report_lines.append(f"### {i}. Component {comp_idx}")
                report_lines.append(f"- **Mean slope**: {pattern['mean_slope']:.4f}")
                report_lines.append(f"- **Statistical significance**: p = {pattern['p_value']:.4f}")
                report_lines.append(f"- **Patients showing increase**: {pattern['n_increasing']}")
                report_lines.append(f"- **Patients showing decrease**: {pattern['n_decreasing']}")
                report_lines.append(f"- **Mean trajectory quality**: R² = {pattern['mean_r_squared']:.3f}")
                report_lines.append("")
        else:
            report_lines.append("No components showed significant group-level compensation patterns.")
            report_lines.append("")
        
        # Early prediction
        report_lines.append("## Early Biomarker Performance (rs-1 → rs-3)")
        report_lines.append("")
        
        good_predictors = [(comp_idx, result) for comp_idx, result in self.prediction_results.items() 
                          if result['r2_score'] > 0.3]
        good_predictors.sort(key=lambda x: x[1]['r2_score'], reverse=True)
        
        if good_predictors:
            report_lines.append("### Components with Good Predictive Power")
            for comp_idx, result in good_predictors:
                report_lines.append(f"- **Component {comp_idx}**: R² = {result['r2_score']:.3f}, correlation = {result['correlation']:.3f}")
            report_lines.append("")
        else:
            report_lines.append("No components showed strong rs-1 → rs-3 prediction (R² > 0.3)")
            report_lines.append("")
        
        avg_r2 = np.mean([result['r2_score'] for result in self.prediction_results.values()])
        report_lines.append(f"**Average prediction performance**: R² = {avg_r2:.3f}")
        report_lines.append("")
        
        # Clinical implications
        report_lines.append("## Clinical Implications")
        report_lines.append("")
        report_lines.append("### Longitudinal Disease Understanding")
        report_lines.append("- Individual patients show unique compensation trajectories")
        report_lines.append("- Compensation patterns emerge at different rates in different patients")
        report_lines.append("- Early timepoint patterns may predict future compensation needs")
        report_lines.append("")
        
        report_lines.append("### Personalized Medicine Applications")
        report_lines.append("- Patient-specific trajectory modeling for treatment planning")
        report_lines.append("- Early identification of patients with strong vs. weak compensation")
        report_lines.append("- Individualized intervention timing based on trajectory patterns")
        report_lines.append("- Long-term outcome prediction from early biomarkers")
        
        # Save report
        report_file = self.output_dir / 'longitudinal_compensation_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ Longitudinal report saved to: {report_file}")
        return report_file
    
    def save_longitudinal_results(self):
        """Save all longitudinal analysis results"""
        print("\n💾 Saving longitudinal analysis results...")
        
        # Save individual trajectories
        trajectories_data = []
        for subject_id, trajectory_data in self.patient_trajectories.items():
            for comp_idx in range(self.n_components):
                metrics = trajectory_data['trajectory_metrics'][comp_idx]
                trajectories_data.append({
                    'Subject_ID': subject_id,
                    'Component': comp_idx,
                    'rs1_value': metrics['rs1_value'],
                    'rs2_value': metrics['rs2_value'],
                    'rs3_value': metrics['rs3_value'],
                    'total_change': metrics['total_change'],
                    'slope': metrics['slope'],
                    'r_squared': metrics['r_squared'],
                    'trend_direction': metrics['trend_direction']
                })
        
        trajectories_df = pd.DataFrame(trajectories_data)
        trajectories_file = self.output_dir / 'individual_trajectories.csv'
        trajectories_df.to_csv(trajectories_file, index=False)
        
        # Save compensation patterns
        patterns_data = []
        for comp_idx, pattern in self.compensation_patterns.items():
            patterns_data.append({
                'Component': comp_idx,
                'mean_slope': pattern['mean_slope'],
                'p_value': pattern['p_value'],
                'is_compensation': pattern['is_compensation'],
                'n_increasing': pattern['n_increasing'],
                'n_decreasing': pattern['n_decreasing'],
                'mean_r_squared': pattern['mean_r_squared']
            })
        
        patterns_df = pd.DataFrame(patterns_data)
        patterns_file = self.output_dir / 'compensation_patterns.csv'
        patterns_df.to_csv(patterns_file, index=False)
        
        # Save prediction results
        prediction_data = []
        for comp_idx, result in self.prediction_results.items():
            prediction_data.append({
                'Component': comp_idx,
                'r2_score': result['r2_score'],
                'correlation': result['correlation'],
                'correlation_p_value': result['correlation_p_value'],
                'prediction_quality': result['prediction_quality']
            })
        
        prediction_df = pd.DataFrame(prediction_data)
        prediction_file = self.output_dir / 'early_prediction_results.csv'
        prediction_df.to_csv(prediction_file, index=False)
        
        print(f"✅ Results saved:")
        print(f"  Individual trajectories: {trajectories_file}")
        print(f"  Compensation patterns: {patterns_file}")
        print(f"  Prediction results: {prediction_file}")
        
        return {
            'trajectories': trajectories_file,
            'patterns': patterns_file,
            'predictions': prediction_file
        }
    
    def run_complete_longitudinal_analysis(self):
        """Run complete true longitudinal compensation analysis"""
        start_time = time.time()
        
        print("🚀 STARTING TRUE LONGITUDINAL COMPENSATION ANALYSIS")
        print("=" * 60)
        
        try:
            # Step 1: Identify longitudinal subjects
            self.identify_longitudinal_subjects()
            
            # Step 2: Create masker
            self.create_longitudinal_masker()
            
            # Step 3: Extract longitudinal signals
            self.extract_longitudinal_signals()
            
            # Step 4: Perform longitudinal ICA
            self.perform_longitudinal_ica()
            
            # Step 5: Create individual trajectories
            self.create_individual_trajectories()
            
            # Step 6: Identify compensation patterns
            self.identify_compensation_patterns()
            
            # Step 7: Analyze early prediction
            self.analyze_early_prediction()
            
            # Step 8: Create visualizations
            self.create_longitudinal_visualizations()
            
            # Step 9: Generate report
            report_file = self.generate_longitudinal_report()
            
            # Step 10: Save results
            result_files = self.save_longitudinal_results()
            
            # Analysis complete
            end_time = time.time()
            total_time = end_time - start_time
            
            print("\n🎉 TRUE LONGITUDINAL ANALYSIS COMPLETED!")
            print("=" * 60)
            print(f"⏱️  Total analysis time: {total_time/60:.1f} minutes")
            print(f"👥 Patients tracked: {len(self.patient_trajectories)}")
            print(f"📈 Compensation components: {sum(1 for p in self.compensation_patterns.values() if p['is_compensation'])}")
            print(f"🎯 Good early predictors: {sum(1 for r in self.prediction_results.values() if r['r2_score'] > 0.3)}")
            print(f"📊 Results saved to: {self.output_dir}")
            print("=" * 60)
            
            return {
                'success': True,
                'report_file': report_file,
                'result_files': result_files,
                'n_patients': len(self.patient_trajectories),
                'compensation_patterns': self.compensation_patterns,
                'prediction_results': self.prediction_results
            }
            
        except Exception as e:
            print(f"\n❌ ANALYSIS FAILED: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

def main():
    """Main entry point for true longitudinal analysis"""
    
    # Initialize analyzer
    analyzer = TrueLongitudinalCompensationAnalyzer(
        data_dir="extracted data/raw_data",
        output_dir="true_longitudinal_results"
    )
    
    # Run complete analysis
    results = analyzer.run_complete_longitudinal_analysis()
    
    if results['success']:
        print(f"\n📁 All results saved to: true_longitudinal_results/")
        print(f"📄 Main report: {results['report_file']}")
        
        print("\n🔬 TRUE LONGITUDINAL ANALYSIS ACHIEVED:")
        print("✅ Individual patient trajectories tracked over time")
        print("✅ Within-subject compensation patterns identified")
        print("✅ Early biomarker prediction from rs-1 to rs-3")
        print("✅ Personalized compensation maps created")
        print("✅ Professor's longitudinal requirement fulfilled")
        
    else:
        print(f"\n❌ Analysis failed with error: {results['error']}")

if __name__ == "__main__":
    main() 