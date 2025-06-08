OUTPUT_ROOT = 'output'
NIAR = 'raw_data'

seed_points = {
    "PCC": [-2, -52, 26],       # Posterior cingulate cortex – core node of the Default Mode Network (DMN), involved in self-referential thinking and memory retrieval
    "mPFC": [0, 52, -2],        # Medial prefrontal cortex – anterior DMN, associated with decision-making and internal mentation
    "L_IPL": [-45, -67, 36],    # Left inferior parietal lobule – part of DMN, involved in social cognition and perspective-taking
    "R_IPL": [49, -63, 33],     # Right inferior parietal lobule – DMN region, symmetrical to L_IPL

    "ACC": [0, 22, 35],         # Anterior cingulate cortex – part of the Salience Network, related to conflict monitoring and emotion regulation
    "L_Insula": [-38, 22, -10], # Left insula – detects salient stimuli and supports interoceptive awareness
    "R_Insula": [38, 22, -10],  # Right insula – same function as left, part of Salience Network

    "DLPFC_L": [-44, 36, 20],   # Left dorsolateral prefrontal cortex – Executive Control Network, involved in working memory and top-down attention
    "DLPFC_R": [44, 36, 20],    # Right DLPFC – right-side counterpart

    "L_IPS": [-28, -58, 46],    # Left intraparietal sulcus – Dorsal Attention Network, supports goal-directed attention
    "R_IPS": [28, -58, 46],     # Right IPS – same function on the right hemisphere

    "L_FEF": [-26, -4, 50],     # Left frontal eye field – involved in eye movements and spatial attention
    "R_FEF": [26, -4, 50],      # Right FEF – counterpart to left FEF

    "L_SMG": [-60, -22, 18],    # Left supramarginal gyrus – part of the Ventral Attention Network, supports stimulus-driven attention shifts
    "R_SMG": [60, -22, 18],     # Right SMG – same function on the right

    "L_M1": [-38, -26, 62],     # Left primary motor cortex – controls voluntary movements of the right side of the body
    "R_M1": [38, -26, 62],      # Right M1 – controls left-side movements

    "L_A1": [-54, -14, 8],      # Left primary auditory cortex – processes auditory information from the right ear
    "R_A1": [54, -14, 8],       # Right A1 – processes sound from the left ear

    "V1": [0, -94, 4],          # Primary visual cortex – responsible for processing basic visual features
}
