import warnings
warnings.simplefilter("ignore")

import os
import mudata as mu
import scanpy as sc
import pandas as pd
import numpy as np
import pertpy as pt
import patsy
import argparse

os.environ['R_HOME'] = '/rfs/project/rfs-iCNyzSAaucw/lk530/miniconda3/envs/pertpy_1.0/lib/R'

from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

# Parse experiment id
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_id', type=int)
args = parser.parse_args()

# Load experiment plan
exp_df = pd.read_csv("/rds/project/rds-C9woKbOCf2Y/lk530/T_cell_infection_atlas/Multi_milo/Manuscript_final/milo_experiment_plan_CD4_fig1.csv")

for col in ['group_1', 'group_2', 'subset_condition']:
    exp_df[col] = exp_df[col].str.replace('SARS-CoV-2', 'SARS_CoV_2')
    exp_df[col] = exp_df[col].str.replace('HSV-2', 'HSV_2')

exp = exp_df.loc[exp_df['id'] == args.experiment_id].squeeze()
print(exp)


def get_valid_formula(obs, confounders, comparison_variable, verbose=True):
    """
    Build a full-rank model formula of the form:
    ~ confounders + comparison_variable

    Drops uninformative or collinear confounders, prioritizing:
        study > isolation > modalities

    Returns:
        formula (str), kept_confounders (list), dropped_confounders (list)
    """
    confounder_priority = ["study", "isolation", "modalities"]
    confounders = [c for c in confounders if c in obs.columns]

    kept = []
    dropped = []

    for conf in confounders:
        n_unique = obs[conf].nunique(dropna=True)
        if n_unique < 2:
            dropped.append(conf)
            if verbose:
                print(f"Dropping confounder '{conf}' — only {n_unique} unique value(s)")
        else:
            kept.append(conf)

    kept = sorted(kept, key=lambda x: confounder_priority.index(x) if x in confounder_priority else len(confounder_priority))

    final_confounders = kept.copy()
    while final_confounders:
        formula = "~ " + " + ".join(final_confounders + [comparison_variable])
        try:
            design = patsy.dmatrix(formula, data=obs, return_type='dataframe')
            rank = np.linalg.matrix_rank(design)
            if rank == design.shape[1]:
                return formula, final_confounders, dropped
            else:
                to_drop = final_confounders.pop(-1)
                dropped.append(to_drop)
                if verbose:
                    print(f"Dropping confounder '{to_drop}' — collinearity detected")
        except Exception as e:
            if verbose:
                print(f"Error while testing formula '{formula}': {e}")
            break

    formula = "~ " + comparison_variable
    if verbose:
        print("All confounders dropped — using only comparison variable")
    return formula, [], dropped


# Load Data
obs = pd.read_csv(
    '/rfs/project/rfs-iCNyzSAaucw/lmd76/pan_infection/datasets/pan_infection_atlas/snakemake_toolbox/out/checkpoint_objects/checkpoint_6/obs.csv',
    index_col=0,
    low_memory=False,
)

# new BEAM annotations from Lorenz
annot = pd.read_csv('/rfs/project/rfs-iCNyzSAaucw/lmd76/pan_infection/datasets/pan_infection_atlas/annotations_LMD/BEAM_manual_annotation-2.csv')
annot.loc[annot.annotation_level_4 == 'B', 'annotation_level_1'] = 'B'
reverse_barcode_dict = (obs[obs.study == 'BEAM'].donor_id + '_' + obs[obs.study == 'BEAM'].barcode.str.split('_|None', expand=True)[1]).to_dict()
annot.index = annot.Donor_barcodes.map({v: k for k, v in reverse_barcode_dict.items()})
annot = annot[annot.index.isin(obs.index)]
cols = ['annotation_level_1', 'annotation_level_2', 'annotation_level_3', 'annotation_level_4']
obs.loc[annot.index, cols] = annot.loc[:, cols].values

# new VZV BEAM data annotations
annot = pd.read_csv('/rfs/project/rfs-iCNyzSAaucw/lmd76/pan_infection/datasets/pan_infection_atlas/annotations_LMD/GSE275633_BEAM_annotation.csv')
annot.index = 'GSE275633' + annot['index']
barcode_map = obs[obs.study == 'GSE275633'].reset_index().set_index('barcode')['index'].to_dict()
annot['new_barcode'] = annot.index.map(barcode_map)
obs.loc[annot.new_barcode, cols] = annot.loc[:, cols].values

# split BEAM EBV-CMV-flu cells
beam_labels = pd.read_csv('/rfs/project/rfs-iCNyzSAaucw/lmd76/pan_infection/datasets/pan_infection_atlas/miscellaneous/milo_proper/BEAM_pathogen_split_detail.csv', index_col=1)
beam_labels['pathogen'] = beam_labels.virus_detail.str.split('_', n=2, expand=True)[2].fillna('Influenza_CMV_EBV')
obs.loc[beam_labels.index, 'pathogen'] = beam_labels.pathogen.values

# split BEAM VZV-EBV cells
obs.loc[obs.study == 'GSE275633', 'pathogen'] = obs.loc[obs.study == 'GSE275633', 'antigen'].apply(
    lambda x: 'EBV_VZV' if isinstance(x, float) else 'EBV' if 'EBV' in x else 'VZV' if 'VZV' in x else np.nan
)

# Absorb CD8 Activated CTL
replace_dict = {'T CD8 Activated CTL': 'T CD8 Activated', 'T CD8 Activated CTL Cycling': 'T CD8 Activated Cycling'}
for col in cols:
    obs[col] = obs[col].replace(replace_dict)

# add more metadata
pathogen_dataset_dict = obs[['pathogen', 'dataset']].drop_duplicates().query('pathogen!="Healthy"').dropna().drop_duplicates('dataset').set_index('dataset').pathogen.to_dict()
obs['pathogen_wo_healthy_label'] = obs.dataset.map(pathogen_dataset_dict)
obs['isolation'] = obs['isolation_strategy']

for col in ['pathogen', 'pathogen_wo_healthy_label']:
    obs[col] = obs[col].str.replace('SARS-CoV-2', 'SARS_CoV_2')
    obs[col] = obs[col].str.replace('HSV-2', 'HSV_2')

# Construct the adata object
adata = sc.AnnData(obs=obs)
UMAP = np.load('/rfs/project/rfs-iCNyzSAaucw/lmd76/pan_infection/datasets/pan_infection_atlas/snakemake_toolbox/out/checkpoint_objects/checkpoint_6/scANVI_improved_val/X_umap.npy')
adata.obsm["X_umap"] = UMAP
SCANVI = np.load('/rfs/project/rfs-iCNyzSAaucw/lmd76/pan_infection/datasets/pan_infection_atlas/snakemake_toolbox/out/checkpoint_objects/checkpoint_6/scANVI_improved_val/X_scANVI.npy')
adata.obsm["X_scANVI"] = SCANVI

# Update pathogen column using detailed pathogen CSV
pathogen_detailed = pd.read_csv('/rds/project/rds-C9woKbOCf2Y/lk530/T_cell_infection_atlas/260121_T_cell_atlas_pathogen_detailed.csv')
mapping_detailed = (
    pathogen_detailed
    .groupby('barcode')['pathogen']
    .agg(lambda x: ';'.join(sorted(set(x))))
)
mask_wang = adata.obs['study'] == 'HRA000190'
adata.obs.loc[mask_wang, 'pathogen_detailed'] = adata.obs.loc[mask_wang, 'pathogen']
mask_other = ~mask_wang
adata.obs.loc[mask_other, 'pathogen_detailed'] = (
    adata.obs.loc[mask_other, 'barcode'].astype(str).map(mapping_detailed)
)
adata.obs['pathogen'] = adata.obs['pathogen_detailed']

for col in ['pathogen']:
    adata.obs[col] = adata.obs[col].str.replace('SARS-CoV-2', 'SARS_CoV_2')
    adata.obs[col] = adata.obs[col].str.replace('HSV-2', 'HSV_2')

# Subset to CD4 T cells
adata = adata[adata.obs.annotation_level_1 == 'T CD4']

# rename Yoshida_2021 sample_id to make them unique
adata.obs.loc[adata.obs.study == 'Yoshida_2021', 'sample_id'] = (
    adata.obs.loc[adata.obs.study == 'Yoshida_2021', 'sample_id'] + '_Yoshida_2021'
)

# Standardise severity labels for Dengue
adata.obs['severity'] = adata.obs['severity'].replace({
    'Dengue fever': 'Dengue_fever',
    'Dengue hemorrhagic fever': 'Dengue_hemorrhagic_fever'
})

# Tuberculosis: barcode-level mapping to CASE/CONTROL
mtb_meta = pd.read_csv('/rds/project/rds-C9woKbOCf2Y/lk530/T_cell_infection_atlas/Nathan_Mtb_disease_status.csv')
if 'barcode' in mtb_meta.columns:
    mtb_meta = mtb_meta.set_index('barcode')
mask_mtb_raw = adata.obs['pathogen'] == 'Tuberculosis'
adata.obs['pathogen'] = adata.obs['pathogen'].astype(str)
adata.obs.loc[mask_mtb_raw, 'pathogen'] = (
    adata.obs.loc[mask_mtb_raw, 'barcode']
    .map(mtb_meta['disease_status'])
    .map(lambda x: f'Tuberculosis_{x}' if pd.notna(x) else 'Tuberculosis')
    .values
)

# HIV: Wang disease_stage update
hiv_meta = pd.read_csv('/rds/project/rds-C9woKbOCf2Y/lk530/T_cell_infection_atlas/Wang_HIV_disease_status.csv')
hiv_meta['barcode'] = 'HRA000190_' + hiv_meta['barcode']
hiv_meta = hiv_meta.drop_duplicates(subset='barcode').set_index('barcode')
adata.obs['disease_stage'] = adata.obs['disease_stage'].astype(str)
mask_hiv_wang = (adata.obs['pathogen'] == 'HIV') & (adata.obs['study'] == 'HRA000190')
adata.obs.loc[mask_hiv_wang, 'disease_stage'] = (
    adata.obs.loc[mask_hiv_wang, 'barcode']
    .map(hiv_meta['Disease_status'])
    .values
)

# Influenza: relabel by study
mask_flu = adata.obs['pathogen'] == 'Influenza_virus'
adata.obs.loc[mask_flu & (adata.obs['study'] == 'COMBAT_2022'), 'pathogen'] = 'Influenza_Primary'
adata.obs.loc[mask_flu & (adata.obs['study'] == 'BEAM'), 'pathogen'] = 'Influenza_Memory'  # CD8 only

# Initialise from pathogen
adata.obs['severity_stage_final'] = adata.obs['pathogen'].copy()

# Healthy
adata.obs.loc[adata.obs['pathogen'] == 'Healthy', 'severity_stage_final'] = 'Healthy'

# ANALYSIS 1: compare infection vs. healthy

# HIV: HIV_acute / HIV_chronic (Chronic + Untreated merged)
mask_hiv = adata.obs['pathogen'] == 'HIV'
adata.obs.loc[mask_hiv, 'severity_stage_final'] = (
    adata.obs.loc[mask_hiv, 'disease_stage']
    .map({'Acute': 'HIV_acute', 'Chronic': 'HIV_chronic', 'Untreated': 'HIV_chronic'})
)

# Tuberculosis
mask_mtb = adata.obs['pathogen'].str.startswith('Tuberculosis', na=False)
adata.obs.loc[mask_mtb, 'severity_stage_final'] = 'Tuberculosis'

# Influenza: rename Primary/Memory
adata.obs['severity_stage_final'] = adata.obs['severity_stage_final'].replace({
    'Influenza_Primary': 'Influenza_acute',
    'Influenza_Memory':  'Influenza_memory',
})

# HBV: all powered chronic subtypes → HBV_chronic (broad label for Analysis 1)
#      acute / acute_resolved / chronic_resolved → Healthy (excluded by exp plan)
mask_hbv = adata.obs['pathogen'] == 'HBV'
hbv_final_map = {
    # Hatje_2024
    'VHB1':  'HBV_chronic', 'VHB2':  'HBV_chronic', 'VHB3':  'HBV_chronic',
    'VHB4':  'HBV_chronic', 'VHB6':  'HBV_chronic', 'VHB7':  'HBV_chronic',
    'VHB8':  'HBV_chronic', 'VHB9':  'HBV_chronic', 'VHB10': 'HBV_chronic',
    'VHB11': 'HBV_chronic', 'VHB12': 'HBV_chronic', 'VHB14': 'HBV_chronic',
    'VHB16': 'HBV_chronic', 'VHB21': 'HBV_chronic', 'VHB26': 'HBV_chronic',
    'VHB27': 'HBV_chronic', 'VHB28': 'HBV_chronic', 'VHB34': 'HBV_chronic',
    'VHB35': 'HBV_chronic', 'VHB36': 'HBV_chronic', 'VHB37': 'HBV_chronic',
    # GSE234241 (Genshaft)
    '10-1003': 'HBV_chronic', '10-1005': 'HBV_chronic', '10-1006': 'HBV_chronic',
    '10-1007': 'HBV_chronic', '20-1002': 'HBV_chronic', '20-1003': 'HBV_chronic',
    '20-3001': 'HBV_chronic', '20-3002': 'HBV_chronic', '30-1021': 'HBV_chronic',
    '30-1022': 'HBV_chronic', '30-1023': 'HBV_chronic', '30-1024': 'HBV_chronic',
    '30-1025': 'HBV_chronic',
    '22-0002': 'Healthy', '22-0004': 'Healthy', '22-0010': 'Healthy',
    # GSE182159 (Zhang) — acute / acute_resolved / chronic_resolved → None
    'P190326': 'HBV_chronic', 'P190402': 'HBV_chronic', 'P190604': 'HBV_chronic',
    'P190716': None,           'P190719': 'HBV_chronic', 'P190801': 'HBV_chronic',
    'P190808': 'HBV_chronic', 'P190902': 'HBV_chronic', 'P190910': 'HBV_chronic',
    'P190911': 'HBV_chronic', 'P191008': None,           'P191028': 'HBV_chronic',
    'P191112': 'HBV_chronic', 'P191126': None,           'P191127': None,
    'P191210': None,           'P191217': None,
    # GSE182159 (Zhang) healthy controls
    'D528848': 'Healthy', 'D529074': 'Healthy', 'D529351': 'Healthy',
    'D529354': 'Healthy', 'D529409': 'Healthy', 'Dhc570':  'Healthy',
    # GSE259231 (Heim) — acute / acute_resolved → None
    'HBV_1':  None,            'HBV_2':  None,            'HBV_3':  None,
    'HBV_4':  'HBV_chronic',   'HBV_5':  'HBV_chronic',   'HBV_6':  None,
    'HBV_7':  'HBV_chronic',   'HBV_8':  'HBV_chronic',   'HBV_9':  'HBV_chronic',
    'HBV_10': 'HBV_chronic',   'HBV_11': 'HBV_chronic',   'HBV_12': 'HBV_chronic',
    'HBV_13': 'HBV_chronic',   'HBV_14': 'HBV_chronic',   'HBV_15': 'HBV_chronic',
    'HBV_16': 'HBV_chronic',   'HBV_17': 'HBV_chronic',   'HBV_18': 'HBV_chronic',
    'HBV_19': 'HBV_chronic',   'HBV_20': 'HBV_chronic',   'HBV_21': 'HBV_chronic',
}
adata.obs.loc[mask_hbv, 'severity_stage_final'] = (
    adata.obs.loc[mask_hbv, 'donor_id'].map(hbv_final_map).values
)
# Unmapped HBV cells (excluded subtypes) → Healthy
adata.obs.loc[mask_hbv & adata.obs['severity_stage_final'].isna(), 'severity_stage_final'] = 'Healthy'

# ANALYSIS 2: stratified by disease severity
adata.obs['severity_stage_detail2'] = adata.obs['severity_stage_final'].copy()

# SARS-CoV-2: prefix + severity
mask_cov2 = adata.obs['pathogen'] == 'SARS_CoV_2'
adata.obs.loc[mask_cov2, 'severity_stage_detail2'] = (
    'SARS_CoV_2_' + adata.obs.loc[mask_cov2, 'severity'].astype(str)
)

# Dengue: explicit map from standardised severity labels
mask_dengue = adata.obs['pathogen'] == 'Dengue_virus'
adata.obs.loc[mask_dengue, 'severity_stage_detail2'] = (
    adata.obs.loc[mask_dengue, 'severity']
    .map({
        'Asymptomatic':             'Dengue_Asymptomatic',
        'Dengue_fever':             'Dengue_fever',
        'Dengue_hemorrhagic_fever': 'Dengue_hemorrhagic_fever',
    })
)

# HBV: detailed subtypes — treated / untreated / IT / IA
#      acute / acute_resolved / chronic_resolved → None → Healthy (excluded by exp plan)
hbv_detail2_map = {
    # Hatje_2024 — all treated (NUC therapy)
    'VHB1':  'HBV_chronic_treated',    'VHB2':  'HBV_chronic_treated',
    'VHB3':  'HBV_chronic_treated',    'VHB4':  'HBV_chronic_treated',
    'VHB6':  'HBV_chronic_treated',    'VHB7':  'HBV_chronic_treated',
    'VHB8':  'HBV_chronic_treated',    'VHB9':  'HBV_chronic_treated',
    'VHB10': 'HBV_chronic_treated',    'VHB11': 'HBV_chronic_treated',
    'VHB12': 'HBV_chronic_treated',    'VHB14': 'HBV_chronic_treated',
    'VHB16': 'HBV_chronic_treated',    'VHB21': 'HBV_chronic_treated',
    'VHB26': 'HBV_chronic_treated',    'VHB27': 'HBV_chronic_treated',
    'VHB28': 'HBV_chronic_treated',    'VHB34': 'HBV_chronic_treated',
    'VHB35': 'HBV_chronic_treated',    'VHB36': 'HBV_chronic_treated',
    'VHB37': 'HBV_chronic_treated',
    # GSE234241 (Genshaft) — untreated
    '10-1003': 'HBV_chronic_untreated', '10-1005': 'HBV_chronic_untreated',
    '10-1006': 'HBV_chronic_untreated', '10-1007': 'HBV_chronic_untreated',
    '20-1002': 'HBV_chronic_untreated', '20-1003': 'HBV_chronic_untreated',
    '20-3001': 'HBV_chronic_untreated', '20-3002': 'HBV_chronic_untreated',
    '30-1021': 'HBV_chronic_untreated', '30-1022': 'HBV_chronic_untreated',
    '30-1023': 'HBV_chronic_untreated', '30-1024': 'HBV_chronic_untreated',
    '30-1025': 'HBV_chronic_untreated',
    '22-0002': 'Healthy', '22-0004': 'Healthy', '22-0010': 'Healthy',
    # GSE182159 (Zhang) — IT / IA retained; excluded subtypes → None
    'P190326': 'HBV_chronic_IT',  'P190402': 'HBV_chronic_IT',
    'P190604': 'HBV_chronic_IT',  'P190716': None,   # acute_resolved → excluded
    'P190719': 'HBV_chronic_IA',  'P190801': 'HBV_chronic_IA',
    'P190808': 'HBV_chronic_IT',  'P190902': 'HBV_chronic_IT',
    'P190910': 'HBV_chronic_IT',  'P190911': 'HBV_chronic_IA',
    'P191008': None,               # acute_resolved → excluded
    'P191028': 'HBV_chronic_IA',  'P191112': 'HBV_chronic_IA',
    'P191126': None,               # chronic_resolved → excluded
    'P191127': None,               # chronic_resolved → excluded
    'P191210': None,               # chronic_resolved → excluded
    'P191217': None,               # acute_resolved → excluded
    # GSE182159 (Zhang) healthy controls
    'D528848': 'Healthy', 'D529074': 'Healthy', 'D529351': 'Healthy',
    'D529354': 'Healthy', 'D529409': 'Healthy', 'Dhc570':  'Healthy',
    # GSE259231 (Heim) — untreated chronic retained; acute / acute_resolved → None
    'HBV_1':  None,                    # acute → excluded
    'HBV_2':  None,                    # acute_resolved → excluded
    'HBV_3':  None,                    # acute_resolved → excluded
    'HBV_4':  'HBV_chronic_untreated', 'HBV_5':  'HBV_chronic_untreated',
    'HBV_6':  None,                    # acute → excluded
    'HBV_7':  'HBV_chronic_untreated', 'HBV_8':  'HBV_chronic_untreated',
    'HBV_9':  'HBV_chronic_untreated', 'HBV_10': 'HBV_chronic_untreated',
    'HBV_11': 'HBV_chronic_untreated', 'HBV_12': 'HBV_chronic_untreated',
    'HBV_13': 'HBV_chronic_untreated', 'HBV_14': 'HBV_chronic_untreated',
    'HBV_15': 'HBV_chronic_untreated', 'HBV_16': 'HBV_chronic_untreated',
    'HBV_17': 'HBV_chronic_untreated', 'HBV_18': 'HBV_chronic_untreated',
    'HBV_19': 'HBV_chronic_untreated', 'HBV_20': 'HBV_chronic_untreated',
    'HBV_21': 'HBV_chronic_untreated',
}
adata.obs.loc[mask_hbv, 'severity_stage_detail2'] = (
    adata.obs.loc[mask_hbv, 'donor_id'].map(hbv_detail2_map).values
)
# Unmapped HBV cells (excluded subtypes) → Healthy
adata.obs.loc[mask_hbv & adata.obs['severity_stage_detail2'].isna(), 'severity_stage_detail2'] = 'Healthy'

print("\n=== severity_stage_final ===")
print(adata.obs['severity_stage_final'].value_counts(dropna=False))

print("\n=== severity_stage_detail2: SARS-CoV-2 ===")
print(adata.obs.loc[mask_cov2, 'severity_stage_detail2'].value_counts(dropna=False))

print("\n=== severity_stage_detail2: Dengue ===")
print(adata.obs.loc[mask_dengue, 'severity_stage_detail2'].value_counts(dropna=False))

print("\n=== severity_stage_detail2: HBV ===")
print(adata.obs.loc[mask_hbv, 'severity_stage_detail2'].value_counts(dropna=False))

# Subset & run Milo

# no nan values for any relevant categories
adata.obs['isolation'] = adata.obs['isolation'].fillna('None')
adata.obs['severity'] = adata.obs['severity'].fillna('None')
adata.obs['disease_stage'] = adata.obs['disease_stage'].fillna('None')
adata.obs['severity_stage_final'] = adata.obs['severity_stage_final'].fillna('None')
adata.obs['severity_stage_detail2'] = adata.obs['severity_stage_detail2'].fillna('None')

print('Before subsetting:', adata.shape)

adata = adata[adata.obs.query(exp['subset_condition']).index]
print('After subsetting:', adata.shape)

print(adata.obs.groupby(exp['compare_variable'])['barcode'].nunique())

# Filter to only the two groups being compared
adata = adata[adata.obs[exp['compare_variable']].isin([exp['group_1'], exp['group_2']])].copy()

# Set categorical order with group_2 (reference, e.g. Healthy) first
from pandas.api.types import CategoricalDtype
milo_order = [exp['group_2'], exp['group_1']]
cat = CategoricalDtype(categories=milo_order, ordered=True)
adata.obs[exp['compare_variable']] = adata.obs[exp['compare_variable']].astype(str).astype(cat)

# Remove donors with ambiguous severity assignments
donors_with_multi_severity = (
    adata.obs[['donor_id', 'severity']]
    .drop_duplicates()
    .donor_id.value_counts(dropna=False)
    .to_frame()
    .query('count == 2')
    .index.tolist()
)
remove_samples = (
    adata[adata.obs.donor_id.isin(donors_with_multi_severity)]
    .obs[['sample_id', 'donor_id', 'severity', 'disease_stage']]
    .drop_duplicates()
    .set_index('sample_id')
    .duplicated('donor_id')
    .to_frame()
    .rename(columns={0: 'which'})
    .query('which == True')
    .index.tolist()
)
adata = adata[~adata.obs.sample_id.isin(remove_samples)].copy()

## Initialize object for Milo analysis
milo = pt.tl.Milo()
mdata = milo.load(adata)

# Build kNN graph
n_neighbors = min(500, max(mdata['rna'].obs.donor_id.nunique() * 2, 100))
print(f'Build k-NN graph with {n_neighbors} neighbors')
sc.pp.neighbors(mdata["rna"], use_rep="X_scANVI", n_neighbors=n_neighbors)

milo.make_nhoods(mdata["rna"], prop=0.1)
mdata = milo.count_nhoods(mdata, sample_col="sample_id")

formula, kept, dropped = get_valid_formula(
    mdata['rna'].obs, eval(exp['correct_for']), exp['compare_variable']
)
print('Using formula', formula)

formula = f"~ {exp['compare_variable']}"
print('ATTENTION: Overwriting formula to',formula)

model_contrasts = f"{exp['compare_variable']}{exp['group_1']}-{exp['compare_variable']}{exp['group_2']}"
print('Using model_contrasts', model_contrasts)

milo.da_nhoods(mdata, design=formula, model_contrasts=model_contrasts)
milo.build_nhood_graph(mdata)

for annotation in ['annotation_level_3']:
    milo.annotate_nhoods(mdata, anno_col=annotation)
    mdata["milo"].var.loc[mdata["milo"].var["nhood_annotation_frac"] < 0.5, "nhood_annotation"] = "Mixed"

    df = (
        mdata['milo'].var
        .drop(columns=['index_cell'])
        .select_dtypes(include=[np.number])
        .join(mdata['milo'].var['nhood_annotation'])
        .groupby('nhood_annotation')
        .mean()
        .sort_values('SpatialFDR')
    )
    df['formula_used'] = formula
    df['contrast_used'] = model_contrasts
    df['nhoods_total'] = mdata["milo"].var.shape[0]
    df['nhoods_mixed'] = (mdata["milo"].var["nhood_annotation_frac"] < 0.6).sum()
    df.to_csv(f'/rds/project/rds-C9woKbOCf2Y/lk530/T_cell_infection_atlas/Multi_milo/Manuscript_final/Fig_1/results_CD4_{exp["id"]}.csv')
    print('Saved', annotation)
