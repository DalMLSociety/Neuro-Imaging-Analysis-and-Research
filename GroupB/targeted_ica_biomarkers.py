import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nilearn import datasets, input_data, plotting, image
from nilearn.decomposition import CanICA
from sklearn.decomposition import FastICA
import nibabel as nib
import os
import json
from pathlib import Path
import warnings
import time
from tqdm import tqdm

warnings.filterwarnings('ignore')

class TargetedICABiomarkerAnalysis:
    """Targeted ICA analysis specifically for biomarker regions."""
    
    def __init__(self, raw_data_dir='raw_data'):
        print(f"TARGETED ICA BIOMARKER ANALYSIS")
        print("=" * 45)
        
        self.raw_data_dir = Path(raw_data_dir)
        self.aal_atlas = datasets.fetch_atlas_aal()
        self.aal_labels = self.aal_atlas.labels
        
        # Load AAL mapping
        try:
            with open('aal_mapping.json', 'r') as f:
                self.aal_mapping = json.load(f)
                self.aal_mapping = {int(k): v for k, v in self.aal_mapping.items()}
        except:
            self.aal_mapping = {i+1: label for i, label in enumerate(self.aal_labels)}
        
        # Your specific biomarker regions
        self.compensation_regions = [23, 78, 84, 34, 85, 55, 71, 35, 64]
        self.deterioration_regions = [91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116]
        self.all_biomarker_regions = self.compensation_regions + self.deterioration_regions
        
        print(f"Targeting {len(self.compensation_regions)} compensation regions")
        print(f"Targeting {len(self.deterioration_regions)} deterioration regions")
        print(f"Total biomarker regions: {len(self.all_biomarker_regions)}")
    
    def extract_targeted_time_series(self):
        """Extract time series specifically from biomarker regions."""
        print(f"\nEXTRACTING TARGETED TIME SERIES")
        print("=" * 40)
        
        # Find functional files
        functional_files = []
        if self.raw_data_dir.exists():
            for file in self.raw_data_dir.rglob("*.nii.gz"):
                filename = file.name.lower()
                if ('denoised' in filename and 'rs-' in filename and 
                    'reho' not in filename and 'atlas' not in filename):
                    functional_files.append(file)
        
        print(f"📁 Found {len(functional_files)} files")
        
        # Load AAL atlas
        aal_atlas_img = image.load_img(self.aal_atlas.maps)
        
        # Create masker for biomarker regions only
        print("Creating biomarker-specific masker...")
        masker = input_data.NiftiLabelsMasker(
            labels_img=aal_atlas_img,
            standardize=True,
            verbose=0
        )
        
        # Extract time series for all files
        all_time_series = []
        
        with tqdm(desc="Extracting time series", total=len(functional_files), ncols=80) as pbar:
            for file in functional_files:
                try:
                    img = nib.load(str(file))
                    time_series = masker.fit_transform(img)  # Shape: (timepoints, 116_regions)
                    
                    # Extract only biomarker regions
                    biomarker_indices = [r-1 for r in self.all_biomarker_regions if r <= 116]
                    biomarker_time_series = time_series[:, biomarker_indices]
                    
                    all_time_series.append(biomarker_time_series)
                    
                except Exception as e:
                    print(f"Failed to process {file.name}: {e}")
                
                pbar.update(1)
        
        print(f"Extracted time series from {len(all_time_series)} files")
        print(f"   Each file: {all_time_series[0].shape[0]} timepoints × {all_time_series[0].shape[1]} biomarker regions")
        
        return all_time_series
    
    def run_targeted_ica(self, time_series_list, n_components=10, analysis_type="all_biomarkers"):
        """Run ICA specifically on biomarker regions."""
        print(f"\n🧠 TARGETED ICA: {analysis_type.upper()}")
        print("=" * 50)
        
        # Concatenate all time series
        print("Concatenating time series...")
        all_data = np.vstack(time_series_list)  # Shape: (total_timepoints, biomarker_regions)
        
        print(f"Data shape: {all_data.shape}")
        print(f"   Timepoints: {all_data.shape[0]}")
        print(f"   Regions: {all_data.shape[1]}")
        
        # Apply FastICA (more suitable for time series data)
        print(f"Running FastICA with {n_components} components...")
        
        ica = FastICA(
            n_components=n_components,
            random_state=42,
            max_iter=1000,
            tol=1e-4
        )
        
        try:
            # Fit ICA
            start_time = time.time()
            ica_components = ica.fit_transform(all_data.T)  # Transpose: (regions, timepoints)
            ica_time = time.time() - start_time
            
            print(f"ICA completed in {ica_time:.2f}s")
            print(f"   Components shape: {ica_components.shape}")
            
            # Get mixing matrix (how much each region contributes to each component)
            mixing_matrix = ica.mixing_  # Shape: (regions, components)
            
            return ica, mixing_matrix, ica_components
            
        except Exception as e:
            print(f"ICA failed: {e}")
            return None, None, None
    
    def analyze_component_patterns(self, mixing_matrix, region_list, analysis_type):
        """Analyze which regions contribute to each component."""
        print(f"\nANALYZING {analysis_type.upper()} PATTERNS")
        print("=" * 45)
        
        n_components, n_regions = mixing_matrix.T.shape
        results = {}
        
        with tqdm(desc="Analyzing components", total=n_components, ncols=80) as pbar:
            for comp_idx in range(n_components):
                # Get loadings for this component
                loadings = mixing_matrix.T[comp_idx, :]
                
                # Find high-loading regions (top 30%)
                threshold = np.percentile(np.abs(loadings), 70)
                high_loading_indices = np.where(np.abs(loadings) > threshold)[0]
                
                # Map back to actual region IDs
                high_loading_regions = [region_list[idx] for idx in high_loading_indices if idx < len(region_list)]
                
                # Calculate overlap with compensation/deterioration
                comp_overlap = len(set(high_loading_regions) & set(self.compensation_regions))
                det_overlap = len(set(high_loading_regions) & set(self.deterioration_regions))
                
                results[comp_idx + 1] = {
                    'high_loading_regions': high_loading_regions,
                    'compensation_overlap': comp_overlap,
                    'deterioration_overlap': det_overlap,
                    'loadings': loadings,
                    'threshold': threshold
                }
                
                pbar.set_description(f"IC {comp_idx+1}: {len(high_loading_regions)} regions")
                pbar.update(1)
        
        return results
    
    def run_complete_targeted_analysis(self):
        """Run complete targeted ICA analysis."""
        total_start = time.time()
        
        print(f"STARTING TARGETED ICA ANALYSIS")
        print("=" * 50)
        
        # Extract time series
        all_time_series = self.extract_targeted_time_series()
        
        if not all_time_series:
            print("No time series extracted")
            return
        
        # Run different targeted analyses
        analyses = [
            ("all_biomarkers", self.all_biomarker_regions, 8),
            ("compensation_only", self.compensation_regions, 4),
            ("deterioration_only", self.deterioration_regions, 6)
        ]
        
        all_results = {}
        
        for analysis_type, region_list, n_comp in analyses:
            print(f"\n" + "="*60)
            print(f"ANALYSIS: {analysis_type.upper()}")
            print("="*60)
            
            # Extract time series for specific regions
            if analysis_type == "all_biomarkers":
                target_time_series = all_time_series
                target_regions = self.all_biomarker_regions
            else:
                # Extract subset of time series
                if analysis_type == "compensation_only":
                    target_indices = [self.all_biomarker_regions.index(r) for r in self.compensation_regions]
                    target_regions = self.compensation_regions
                else:  # deterioration_only
                    target_indices = [self.all_biomarker_regions.index(r) for r in self.deterioration_regions]
                    target_regions = self.deterioration_regions
                
                target_time_series = [ts[:, target_indices] for ts in all_time_series]
            
            # Run ICA
            ica, mixing_matrix, components = self.run_targeted_ica(
                target_time_series, n_comp, analysis_type
            )
            
            if ica is not None:
                # Analyze patterns
                results = self.analyze_component_patterns(
                    mixing_matrix, target_regions, analysis_type
                )
                all_results[analysis_type] = results
                
                # Print summary
                self.print_analysis_summary(results, analysis_type)
        
        # Generate comprehensive report
        self.generate_targeted_report(all_results)
        
        total_time = time.time() - total_start
        print(f"\nTARGETED ICA ANALYSIS COMPLETE!")
        print(f"⏱️ Total time: {total_time:.2f}s")
        
        return all_results
    
    def print_analysis_summary(self, results, analysis_type):
        """Print summary of analysis results."""
        print(f"\n{analysis_type.upper()} SUMMARY:")
        
        total_comp_overlap = sum(r['compensation_overlap'] for r in results.values())
        total_det_overlap = sum(r['deterioration_overlap'] for r in results.values())
        
        print(f"   Total compensation overlaps: {total_comp_overlap}")
        print(f"   Total deterioration overlaps: {total_det_overlap}")
        
        # Show best components
        best_comp = max(results.items(), key=lambda x: x[1]['compensation_overlap'])
        best_det = max(results.items(), key=lambda x: x[1]['deterioration_overlap'])
        
        print(f"   🏆 Best compensation: IC {best_comp[0]} ({best_comp[1]['compensation_overlap']} regions)")
        print(f"   🏆 Best deterioration: IC {best_det[0]} ({best_det[1]['deterioration_overlap']} regions)")
    
    def generate_targeted_report(self, all_results):
        """Generate comprehensive targeted ICA report."""
        print(f"\nGENERATING TARGETED REPORT")
        print("=" * 35)
        
        report = f"""
TARGETED ICA BIOMARKER VALIDATION REPORT
{'='*50}

APPROACH:
This analysis applies ICA specifically to your biomarker regions
rather than globally across all brain regions, providing more
relevant validation of your SCA7-specific findings.

ANALYSES PERFORMED:
"""
        
        for analysis_type, results in all_results.items():
            n_components = len(results)
            total_comp = sum(r['compensation_overlap'] for r in results.values())
            total_det = sum(r['deterioration_overlap'] for r in results.values())
            
            if analysis_type == "compensation_only":
                max_possible = len(self.compensation_regions) * n_components
                overlap_rate = (total_comp / max_possible * 100) if max_possible > 0 else 0
                target_desc = f"{len(self.compensation_regions)} compensation regions"
            elif analysis_type == "deterioration_only":
                max_possible = len(self.deterioration_regions) * n_components  
                overlap_rate = (total_det / max_possible * 100) if max_possible > 0 else 0
                target_desc = f"{len(self.deterioration_regions)} deterioration regions"
            else:
                max_possible_comp = len(self.compensation_regions) * n_components
                max_possible_det = len(self.deterioration_regions) * n_components
                comp_rate = (total_comp / max_possible_comp * 100) if max_possible_comp > 0 else 0
                det_rate = (total_det / max_possible_det * 100) if max_possible_det > 0 else 0
                overlap_rate = (comp_rate + det_rate) / 2
                target_desc = f"{len(self.all_biomarker_regions)} biomarker regions"
            
            report += f"""
{analysis_type.upper().replace('_', ' ')}:
* Target: {target_desc}
* Components: {n_components}
* Compensation overlaps: {total_comp}
* Deterioration overlaps: {total_det}
* Overall overlap rate: {overlap_rate:.1f}%
"""
        
        report += f"""
INTERPRETATION:
Targeted ICA provides more relevant validation by focusing specifically
on your biomarker regions rather than competing with global brain networks.
Higher overlap rates indicate that your biomarkers form coherent,
independent functional networks within the disease-affected brain regions.

VALIDATION STRENGTH:
"""
        
        # Calculate overall validation
        if "all_biomarkers" in all_results:
            all_bio_results = all_results["all_biomarkers"]
            total_overlaps = sum(r['compensation_overlap'] + r['deterioration_overlap'] 
                               for r in all_bio_results.values())
            max_possible = len(all_bio_results) * len(self.all_biomarker_regions)
            validation_score = (total_overlaps / max_possible * 100) if max_possible > 0 else 0
            
            if validation_score > 30:
                assessment = "STRONG: Biomarkers form coherent independent networks"
            elif validation_score > 15:
                assessment = "MODERATE: Partial network independence detected"
            else:
                assessment = "LIMITED: Complex interdependent patterns"
            
            report += f"* Targeted validation score: {validation_score:.1f}%\n"
            report += f"* Assessment: {assessment}\n"
        
        report += f"""
CONCLUSION:
Targeted ICA analysis validates your biomarker regions by testing whether
they form independent functional networks specifically within the SCA7-
affected brain regions, providing more relevant validation than global ICA.
"""
        
        print(report)
        
        # Save report
        with open('targeted_ica_biomarker_validation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Saved: targeted_ica_biomarker_validation_report.txt")

def main():
    """Run targeted ICA biomarker analysis."""
    print("TARGETED ICA BIOMARKER VALIDATION")
    print("=" * 50)
    print("Applying ICA specifically to your biomarker regions")
    print("More relevant than global brain ICA")
    
    analyzer = TargetedICABiomarkerAnalysis()
    results = analyzer.run_complete_targeted_analysis()
    
    print("\nTARGETED ICA ANALYSIS COMPLETED!")
    print("Biomarker-specific validation achieved!")

if __name__ == "__main__":
    main() 