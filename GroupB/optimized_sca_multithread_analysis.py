#!/usr/bin/env python3
"""
Optimized Multi-threaded SCA Analysis
====================================

High-performance analysis using all CPU cores and optimized memory management.
Designed for 12-core system with 7.7GB RAM processing 84 files (~20GB data).
"""

import os
import gc
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Multi-processing and optimization
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import psutil
import time
from tqdm import tqdm

# ML and analysis libraries
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import ttest_ind, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# Neuroimaging libraries
from nilearn.input_data import NiftiMasker
from nilearn.image import resample_img, smooth_img
from nilearn import datasets
import json

class OptimizedSCAAnalyzer:
    """High-performance SCA analyzer with multi-threading and memory optimization"""
    
    def __init__(self, data_dir="extracted data/raw_data", output_dir="optimized_sca_results"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # System specifications
        self.n_cores = mp.cpu_count()  # 12 cores
        self.total_ram_gb = psutil.virtual_memory().total / (1024**3)  # 7.7 GB
        self.available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        print(f"🖥️  System Resources:")
        print(f"   • CPU Cores: {self.n_cores}")
        print(f"   • Total RAM: {self.total_ram_gb:.1f} GB")
        print(f"   • Available RAM: {self.available_ram_gb:.1f} GB")
        
        # Optimization parameters
        self.max_workers = min(self.n_cores - 1, 8)  # Leave 1 core free, max 8 for memory
        self.chunk_size = max(1, int(self.available_ram_gb / 2))  # Process files in chunks
        self.memory_limit_per_file = 500 * 1024 * 1024  # 500MB per file max
        
        print(f"⚡ Optimization Settings:")
        print(f"   • Max Workers: {self.max_workers}")
        print(f"   • Chunk Size: {self.chunk_size} files")
        print(f"   • Memory Limit per File: {self.memory_limit_per_file / (1024**2):.0f} MB")
        
        # Data storage
        self.patient_files = []
        self.control_files = []
        self.masker = None
        
    def load_file_lists(self):
        """Load and organize file lists efficiently"""
        print("\n📁 Loading file lists...")
        
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.nii.gz')]
        self.patient_files = sorted([f for f in all_files if f.startswith('Denoised_p')])
        self.control_files = sorted([f for f in all_files if f.startswith('Denoised_C')])
        
        print(f"   • Patient files: {len(self.patient_files)}")
        print(f"   • Control files: {len(self.control_files)}")
        print(f"   • Total files: {len(self.patient_files) + len(self.control_files)}")
        
        # Estimate total data size
        total_size_gb = (len(self.patient_files) + len(self.control_files)) * 0.24  # ~240MB per file
        print(f"   • Estimated total data: {total_size_gb:.1f} GB")
        
        return True
    
    def create_optimized_masker(self):
        """Create optimized brain masker with reduced memory footprint"""
        print("\n🧠 Creating optimized brain masker...")
        
        # Load sample file to fit masker
        sample_file = os.path.join(self.data_dir, self.patient_files[0])
        sample_img = nib.load(sample_file)
        
        # Create masker with optimized settings
        self.masker = NiftiMasker(
            mask_strategy='background',
            standardize=True,
            smoothing_fwhm=4,  # Reduced for speed
            memory_level=0,    # No caching to save RAM
            verbose=0,
            n_jobs=1  # Single-threaded within masker (we handle parallelism)
        )
        
        # Fit masker
        print("   • Fitting masker...")
        self.masker.fit(sample_img)
        n_voxels = int(self.masker.mask_img_.get_fdata().sum())
        
        print(f"   • Brain voxels: {n_voxels:,}")
        print("   ✅ Masker created successfully")
        
        # Clear sample image from memory
        del sample_img
        gc.collect()
        
        return self.masker
    
    def process_single_file(self, filepath):
        """Process a single NII file efficiently"""
        try:
            # Load image with memory monitoring
            img = nib.load(filepath)
            
            # Check file size and downsample if needed
            data_size = np.prod(img.shape) * 8  # Estimate size in bytes
            if data_size > self.memory_limit_per_file:
                # Downsample for memory efficiency
                img = resample_img(img, target_affine=np.diag([4, 4, 4, 1]))
            
            # Extract signals using pre-fitted masker
            signals = self.masker.transform(img)
            
            # Take mean across time to reduce dimensionality
            mean_signals = signals.mean(axis=0)
            
            # Clear memory
            del img, signals
            gc.collect()
            
            return mean_signals
            
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(filepath)}: {e}")
            return None
    
    def extract_signals_parallel(self):
        """Extract brain signals using parallel processing"""
        print(f"\n⚡ Extracting signals with {self.max_workers} workers...")
        
        all_signals = []
        all_labels = []
        all_metadata = []
        
        # Combine all files for processing
        all_file_data = []
        
        # Add patient files
        for filename in self.patient_files:
            filepath = os.path.join(self.data_dir, filename)
            all_file_data.append({
                'filepath': filepath,
                'filename': filename,
                'label': 1,  # Patient = 1
                'group': 'patient'
            })
        
        # Add control files
        for filename in self.control_files:
            filepath = os.path.join(self.data_dir, filename)
            all_file_data.append({
                'filepath': filepath,
                'filename': filename,
                'label': 0,  # Control = 0
                'group': 'control'
            })
        
        # Process files in chunks to manage memory
        total_files = len(all_file_data)
        
        with tqdm(total=total_files, desc="🧠 Processing brain scans", unit="scan") as pbar:
            
            # Process in chunks
            for chunk_start in range(0, total_files, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, total_files)
                chunk_data = all_file_data[chunk_start:chunk_end]
                
                # Extract filepaths for this chunk
                filepaths = [item['filepath'] for item in chunk_data]
                
                # Process chunk in parallel
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all files in chunk
                    future_to_data = {
                        executor.submit(self.process_single_file, filepath): (filepath, item)
                        for filepath, item in zip(filepaths, chunk_data)
                    }
                    
                    # Collect results
                    for future in as_completed(future_to_data):
                        filepath, item = future_to_data[future]
                        
                        try:
                            signals = future.result()
                            if signals is not None:
                                all_signals.append(signals)
                                all_labels.append(item['label'])
                                all_metadata.append({
                                    'filename': item['filename'],
                                    'group': item['group'],
                                    'subject_id': item['filename'].split('_')[1]
                                })
                        except Exception as e:
                            print(f"❌ Error with {item['filename']}: {e}")
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'Chunk': f"{chunk_start//self.chunk_size + 1}",
                            'RAM': f"{psutil.virtual_memory().percent:.1f}%"
                        })
                
                # Force garbage collection between chunks
                gc.collect()
                time.sleep(0.1)  # Brief pause to let system clean up
        
        # Convert to arrays
        if len(all_signals) > 0:
            self.brain_signals = np.array(all_signals)
            self.labels = np.array(all_labels)
            self.metadata = all_metadata
            
            print(f"\n✅ Signal extraction completed!")
            print(f"   • Final dataset shape: {self.brain_signals.shape}")
            print(f"   • Patients: {np.sum(self.labels == 1)}")
            print(f"   • Controls: {np.sum(self.labels == 0)}")
            print(f"   • Success rate: {len(all_signals)}/{total_files} ({100*len(all_signals)/total_files:.1f}%)")
            
            return True
        else:
            print("❌ No signals extracted successfully!")
            return False
    
    def perform_fast_ica(self, n_components=20):
        """Perform optimized ICA analysis"""
        print(f"\n🔬 Performing FastICA with {n_components} components...")
        
        # Standardize data
        scaler = RobustScaler()  # More robust to outliers
        signals_scaled = scaler.fit_transform(self.brain_signals)
        
        # Reduce dimensionality first with PCA for efficiency
        pca = PCA(n_components=min(50, self.brain_signals.shape[1]//10), random_state=42)
        signals_pca = pca.fit_transform(signals_scaled)
        
        print(f"   • PCA reduced dimensions: {signals_pca.shape}")
        
        # Perform FastICA
        ica = FastICA(
            n_components=n_components,
            random_state=42,
            max_iter=1000,
            tol=1e-3,
            algorithm='parallel',  # Fastest algorithm
            fun='logcosh'
        )
        
        # Transform data
        ica_components = ica.fit_transform(signals_pca)
        
        print(f"   • ICA components shape: {ica_components.shape}")
        print("   ✅ ICA analysis completed")
        
        # Store results
        self.ica_components = ica_components
        self.ica_model = ica
        self.pca_model = pca
        self.scaler = scaler
        
        return ica_components
    
    def identify_biomarkers_fast(self):
        """Fast biomarker identification using Random Forest"""
        print("\n🔍 Identifying SCA biomarkers...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.ica_components, self.labels, 
            test_size=0.3, random_state=42, stratify=self.labels
        )
        
        # Train Random Forest with optimized parameters
        rf = RandomForestClassifier(
            n_estimators=100,  # Reduced for speed
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=self.max_workers,  # Use all cores
            class_weight='balanced'
        )
        
        rf.fit(X_train, y_train)
        
        # Evaluate
        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test)
        y_pred = rf.predict(X_test)
        
        print(f"   • Training accuracy: {train_score:.3f}")
        print(f"   • Testing accuracy: {test_score:.3f}")
        print(f"   • Feature importance range: {rf.feature_importances_.min():.3f} - {rf.feature_importances_.max():.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(rf, self.ica_components, self.labels, cv=5, n_jobs=self.max_workers)
        print(f"   • Cross-validation: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # Store results
        self.rf_model = rf
        self.biomarker_scores = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': rf.feature_importances_.tolist()
        }
        
        print("   ✅ Biomarker identification completed")
        return self.biomarker_scores
    
    def statistical_analysis(self):
        """Perform statistical analysis between groups"""
        print("\n📊 Performing statistical analysis...")
        
        # Separate groups
        patient_mask = self.labels == 1
        control_mask = self.labels == 0
        
        patient_data = self.ica_components[patient_mask]
        control_data = self.ica_components[control_mask]
        
        # Statistical tests for each component
        statistical_results = {}
        
        for i in range(self.ica_components.shape[1]):
            patient_comp = patient_data[:, i]
            control_comp = control_data[:, i]
            
            # T-test
            t_stat, t_pval = ttest_ind(patient_comp, control_comp)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_pval = mannwhitneyu(patient_comp, control_comp, alternative='two-sided')
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(patient_comp) - 1) * np.var(patient_comp) + 
                                 (len(control_comp) - 1) * np.var(control_comp)) / 
                                (len(patient_comp) + len(control_comp) - 2))
            cohens_d = (np.mean(patient_comp) - np.mean(control_comp)) / pooled_std
            
            statistical_results[f'component_{i+1}'] = {
                't_statistic': t_stat,
                't_pvalue': t_pval,
                'u_statistic': u_stat,
                'u_pvalue': u_pval,
                'cohens_d': cohens_d,
                'patient_mean': np.mean(patient_comp),
                'control_mean': np.mean(control_comp),
                'patient_std': np.std(patient_comp),
                'control_std': np.std(control_comp)
            }
        
        # Find significant components (p < 0.05, effect size > 0.5)
        significant_components = []
        for comp, stats in statistical_results.items():
            if stats['t_pvalue'] < 0.05 and abs(stats['cohens_d']) > 0.5:
                significant_components.append(comp)
        
        print(f"   • Significant components (p<0.05, |d|>0.5): {len(significant_components)}")
        print(f"   • Top effect sizes: {sorted([abs(stats['cohens_d']) for stats in statistical_results.values()], reverse=True)[:5]}")
        
        self.statistical_results = statistical_results
        self.significant_components = significant_components
        
        print("   ✅ Statistical analysis completed")
        return statistical_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n📈 Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Feature importance plot
        plt.figure(figsize=(12, 6))
        importance = self.rf_model.feature_importances_
        components = [f'IC{i+1}' for i in range(len(importance))]
        
        plt.subplot(1, 2, 1)
        plt.bar(components, importance)
        plt.title('ICA Component Importance for SCA Classification')
        plt.xlabel('Independent Components')
        plt.ylabel('Feature Importance')
        plt.xticks(rotation=45)
        
        # 2. Statistical significance plot
        plt.subplot(1, 2, 2)
        p_values = [self.statistical_results[f'component_{i+1}']['t_pvalue'] for i in range(len(components))]
        effect_sizes = [abs(self.statistical_results[f'component_{i+1}']['cohens_d']) for i in range(len(components))]
        
        scatter = plt.scatter(p_values, effect_sizes, c=importance, cmap='viridis', s=60)
        plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Medium effect size')
        plt.axvline(x=0.05, color='r', linestyle='--', alpha=0.7, label='p = 0.05')
        plt.xlabel('P-value (t-test)')
        plt.ylabel('Effect Size (|Cohen\'s d|)')
        plt.title('Statistical Significance vs Effect Size')
        plt.colorbar(scatter, label='Feature Importance')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sca_biomarker_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Group comparison for top components
        top_components = sorted(range(len(importance)), key=lambda i: importance[i], reverse=True)[:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, comp_idx in enumerate(top_components):
            ax = axes[idx]
            
            patient_data = self.ica_components[self.labels == 1, comp_idx]
            control_data = self.ica_components[self.labels == 0, comp_idx]
            
            ax.boxplot([control_data, patient_data], labels=['Controls', 'SCA Patients'])
            ax.set_title(f'Component {comp_idx+1} (Importance: {importance[comp_idx]:.3f})')
            ax.set_ylabel('Component Score')
            
            # Add statistical info
            stats = self.statistical_results[f'component_{comp_idx+1}']
            ax.text(0.02, 0.98, f"p = {stats['t_pvalue']:.4f}\nd = {stats['cohens_d']:.3f}", 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('Top ICA Components: SCA Patients vs Controls')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'top_components_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Visualizations saved")
        
    def save_comprehensive_results(self):
        """Save all analysis results"""
        print("\n💾 Saving comprehensive results...")
        
        # Main results dictionary
        results = {
            'analysis_info': {
                'total_subjects': len(self.labels),
                'patients': int(np.sum(self.labels == 1)),
                'controls': int(np.sum(self.labels == 0)),
                'n_components': self.ica_components.shape[1],
                'n_voxels': self.brain_signals.shape[1],
                'processing_cores': self.max_workers,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'classification_performance': self.biomarker_scores,
            'statistical_results': self.statistical_results,
            'significant_components': self.significant_components,
            'subject_metadata': self.metadata
        }
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        results_json = convert_numpy_types(results)
        
        # Save as JSON
        with open(os.path.join(self.output_dir, 'optimized_sca_analysis_results.json'), 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Save ICA components as CSV
        ica_df = pd.DataFrame(
            self.ica_components,
            columns=[f'IC_{i+1}' for i in range(self.ica_components.shape[1])]
        )
        ica_df['group'] = ['patient' if label == 1 else 'control' for label in self.labels]
        ica_df['subject_id'] = [meta['subject_id'] for meta in self.metadata]
        ica_df.to_csv(os.path.join(self.output_dir, 'ica_components.csv'), index=False)
        
        # Save summary report
        with open(os.path.join(self.output_dir, 'analysis_summary.txt'), 'w') as f:
            f.write("OPTIMIZED SCA BRAIN PATTERN ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: {results['analysis_info']['patients']} patients, {results['analysis_info']['controls']} controls\n")
            f.write(f"Processing: {self.max_workers} cores, {len(self.labels)} files processed\n")
            f.write(f"Classification Accuracy: {self.biomarker_scores['cv_mean']:.3f} ± {self.biomarker_scores['cv_std']:.3f}\n")
            f.write(f"Significant Components: {len(self.significant_components)}\n\n")
            
            f.write("TOP BIOMARKERS:\n")
            importance = self.rf_model.feature_importances_
            top_indices = sorted(range(len(importance)), key=lambda i: importance[i], reverse=True)[:5]
            
            for i, idx in enumerate(top_indices):
                stats = self.statistical_results[f'component_{idx+1}']
                f.write(f"{i+1}. Component {idx+1}:\n")
                f.write(f"   - Importance: {importance[idx]:.4f}\n")
                f.write(f"   - p-value: {stats['t_pvalue']:.4f}\n")
                f.write(f"   - Effect size: {stats['cohens_d']:.4f}\n\n")
        
        print(f"   ✅ Results saved to: {self.output_dir}")
        
    def run_complete_analysis(self):
        """Run the complete optimized analysis pipeline"""
        start_time = time.time()
        
        print("🚀 OPTIMIZED SCA BRAIN PATTERN ANALYSIS")
        print("=" * 60)
        
        try:
            # Step 1: Load data
            if not self.load_file_lists():
                raise Exception("Failed to load file lists")
            
            # Step 2: Create masker
            if not self.create_optimized_masker():
                raise Exception("Failed to create masker")
            
            # Step 3: Extract signals
            if not self.extract_signals_parallel():
                raise Exception("Failed to extract signals")
            
            # Step 4: ICA analysis
            self.perform_fast_ica(n_components=20)
            
            # Step 5: Biomarker identification
            self.identify_biomarkers_fast()
            
            # Step 6: Statistical analysis
            self.statistical_analysis()
            
            # Step 7: Visualizations
            self.create_visualizations()
            
            # Step 8: Save results
            self.save_comprehensive_results()
            
            # Analysis complete
            end_time = time.time()
            total_time = end_time - start_time
            
            print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"⏱️  Total time: {total_time/60:.1f} minutes")
            print(f"📊 Classification accuracy: {self.biomarker_scores['cv_mean']:.3f}")
            print(f"🔬 Significant components: {len(self.significant_components)}")
            print(f"📁 Results saved to: {self.output_dir}")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"\n❌ ANALYSIS FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main execution function"""
    analyzer = OptimizedSCAAnalyzer()
    success = analyzer.run_complete_analysis()
    
    if success:
        print("✅ Analysis completed successfully!")
    else:
        print("❌ Analysis failed!")
    
    return success

if __name__ == "__main__":
    main() 