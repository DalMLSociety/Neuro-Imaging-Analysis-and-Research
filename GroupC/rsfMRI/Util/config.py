OUTPUT_ROOT = '../../../output'
NIAR = 'raw_data'

Z_MIN = -72
Z_MAX = 110

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

# MNI center coordinates for 33 DMN ROIs
dmn_coords_33 = [
    (-11,  55,  -5), ( 11,  53,  -6),
    (-10,  50,  20), ( 10,  50,  19),
    (-20,  31,  46), ( 23,  32,  46),
    ( -5, -50,  35), (  7, -51,  34),
    ( -6, -55,  12), (  6, -54,  13),
    (-46, -64,  33), ( 50, -59,  34),
    (-58, -21, -15), ( 59, -17, -18),
    (-38,  17, -34), ( 43,  15, -35),
    (-36,  23, -16), ( 37,  25, -16),
    (-24, -30, -16), ( 26, -26, -18),
    (-15,  -9, -18), ( 17,  -8, -16),
    (-11,  12,   7), ( 13,  11,   9),
    (-26, -82, -33), ( 29, -79, -34),
    ( -6, -57, -45), (  8, -53, -48),
    ( -7, -14,   8), (  7, -11,   8),
    ( -7,  12, -12), (  7,   9, -12),
    (  0, -22, -21)
]

# **Naming 33 ROIs in the order given in the paper's 'Functional space' figure**
dmn_names_33 = [
    # Right hemisphere
    "R VMPFC", "R AMPFC", "R DLPFC", "R PCC", "R Rsp", "R PH", "R Amy",
    "R VLPFC","R TP",   "R MTG",   "R PPC", "R T",   "R BF",  "R C",
    "R CbH",   "R CbT",  "MidB",
    # Left hemisphere
    "L VMPFC", "L AMPFC","L DLPFC", "L PCC", "L Rsp", "L PH", "L Amy",
    "L VLPFC","L TP",   "L MTG",   "L PPC", "L T",   "L BF",  "L C",
    "L CbH",   "L CbT"
]
