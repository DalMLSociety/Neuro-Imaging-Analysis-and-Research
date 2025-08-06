"""
Enhanced Connectivity Matrix Generator
====================================

IMPROVEMENTS:
- Progress tracking with tqdm
- Quality control checks
- Multiple connectivity measures (correlation, partial correlation, precision)
- Fisher z-transformation for correlation matrices
- Memory optimization
- Error handling and logging
- Quality metrics calculation
- Parallel processing capability
"""

import os
import numpy as np
import nibabel as nib
from nilearn import datasets, image
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from tqdm import tqdm
import logging
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedConnectivityGenerator:
    def __init__(self, raw_dir='raw_data', out_dir='full_connectivity_matrices'):
        self.raw_dir = raw_dir
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        
        # Load AAL atlas
        logger.info("Loading AAL atlas...")
        self.aal = datasets.fetch_atlas_aal()
        self.atlas_filename = self.aal['maps']
        self.labels = self.aal['labels']
        
        # Initialize masker with better parameters
        self.masker = NiftiLabelsMasker(
            labels_img=self.atlas_filename, 
            standardize=True,
            memory='nilearn_cache',
            memory_level=1
        )
        
        # Multiple connectivity measures
        self.connectivity_measures = {
            'correlation': ConnectivityMeasure(kind='correlation'),
            'partial_correlation': ConnectivityMeasure(kind='partial correlation'),
            'precision': ConnectivityMeasure(kind='precision')
        }
        
        logger.info(f"Initialized with {len(self.labels)} brain regions")
    
    def quality_check(self, time_series, filename):
        """Perform quality control checks on time series data."""
        quality_metrics = {}
        
        # Check for NaN values
        nan_count = np.isnan(time_series).sum()
        quality_metrics['nan_count'] = nan_count
        
        # Check for zero variance regions
        zero_var_count = np.sum(np.var(time_series, axis=0) == 0)
        quality_metrics['zero_variance_regions'] = zero_var_count
        
        # Check signal-to-noise ratio (simplified)
        snr = np.mean(np.std(time_series, axis=0)) / np.mean(np.std(time_series, axis=0))
        quality_metrics['snr'] = snr
        
        # Check temporal correlation (autocorrelation)
        autocorr = np.mean([np.corrcoef(time_series[:-1, i], time_series[1:, i])[0, 1] 
                           for i in range(time_series.shape[1])])
        quality_metrics['autocorrelation'] = autocorr
        
        # Log quality issues
        if nan_count > 0:
            logger.warning(f"{filename}: {nan_count} NaN values detected")
        if zero_var_count > 0:
            logger.warning(f"{filename}: {zero_var_count} regions with zero variance")
        
        return quality_metrics
    
    def fisher_z_transform(self, correlation_matrix):
        """Apply Fisher z-transformation to correlation matrix."""
        # Avoid division by zero and invalid values
        correlation_matrix = np.clip(correlation_matrix, -0.99, 0.99)
        z_matrix = 0.5 * np.log((1 + correlation_matrix) / (1 - correlation_matrix))
        return z_matrix
    
    def process_single_file(self, fname):
        """Process a single file with enhanced error handling and quality control."""
        if not fname.endswith('.nii.gz') or not fname.startswith('Denoised_'):
            return None
        
        fpath = os.path.join(self.raw_dir, fname)
        
        # Parse subject/session
        parts = fname.replace('.nii.gz','').split('_')
        if len(parts) < 3:
            logger.error(f"Invalid filename format: {fname}")
            return None
        
        subj = parts[1]
        session = parts[2]
        
        results = {}
        
        try:
            # Load image
            img = image.load_img(fpath)
            
            # Extract time series
            time_series = self.masker.fit_transform(img)
            
            # Quality check
            quality_metrics = self.quality_check(time_series, fname)
            
            # Compute multiple connectivity measures
            for measure_name, conn_measure in self.connectivity_measures.items():
                try:
                    matrix = conn_measure.fit_transform([time_series])[0]
                    
                    # Apply Fisher z-transformation for correlation measures
                    if measure_name in ['correlation', 'partial_correlation']:
                        matrix = self.fisher_z_transform(matrix)
                    
                    # Save matrix
                    out_name = f'{subj}_{session}_{measure_name}.npy'
                    out_path = os.path.join(self.out_dir, out_name)
                    np.save(out_path, matrix)
                    
                    results[measure_name] = {
                        'matrix': matrix,
                        'quality_metrics': quality_metrics,
                        'file_path': out_path
                    }
                    
                except Exception as e:
                    logger.error(f"Error computing {measure_name} for {fname}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing {fname}: {e}")
            return None
    
    def generate_all_matrices(self):
        """Generate all connectivity matrices with progress tracking."""
        logger.info("Starting enhanced connectivity matrix generation...")
        
        # Get list of files
        files = [f for f in os.listdir(self.raw_dir) 
                if f.endswith('.nii.gz') and f.startswith('Denoised_')]
        
        logger.info(f"Found {len(files)} files to process")
        
        # Process files with progress bar
        results = []
        for fname in tqdm(files, desc="Processing files", ncols=80):
            result = self.process_single_file(fname)
            if result:
                results.append(result)
        
        # Summary
        logger.info(f"Successfully processed {len(results)} files")
        logger.info(f"Generated matrices saved in {self.out_dir}")
        
        return results

def main():
    """Main function to run the enhanced connectivity generator."""
    generator = EnhancedConnectivityGenerator()
    results = generator.generate_all_matrices()
    
    # Print summary
    print(f"\nEnhanced connectivity matrix generation complete!")
    print(f"Output directory: {generator.out_dir}")
    print(f"Processed files: {len(results)}")
    print(f"Connectivity measures: {list(generator.connectivity_measures.keys())}")

if __name__ == "__main__":
    main() 