# Pan-Infection Atlas analysis code

This repository contains reviewer-facing analysis and figure-generation notebooks supporting the Pan-Infection Atlas manuscript and the associated Pan-Infection T cell atlas website: https://tcellatlas.org.

## Manuscript

**A pan-infection single-cell atlas of human T cells unlocks systematic antigen-specificity inference**

Lisa M. Dratva, Yizhou Yu, Elizaveta K. Vlasova, Min Gyu Im, Krzysztof Polanski, Maximilian Alexandrov, Lisa M. Milchsack, Rakeshlal Kapuge, Alexander V. Predeus, Mikhail Shugay, Lorenz Kretschmer, and Sarah A. Teichmann.

## Repository layout

- `analysis/pan_infection_atlas/01_reviewer_figures.ipynb`: reviewer-facing manuscript figure code.
- `analysis/pan_infection_atlas/02_website_and_cell2specificity_exports.ipynb`: website and Cell2Specificity export code.
- `analysis/structural_modeling/`: notes and future workflows for TCR-peptide-HLA structural modeling analyses.
- `analysis/cell_state_analysis/`: notes and future workflows for cell-state analyses.

## Usage

Open the notebooks in `analysis/pan_infection_atlas/`, review the setup or path cells, and adjust local input and output paths before running. Input data and generated outputs are expected to be provided or written locally and are not included in this repository.

## Adding analyses

Collaborators can add structural modeling, cell-state analysis, or other manuscript-supporting workflows under `analysis/`. Please keep each workflow self-contained, document required local inputs, and avoid committing large generated files.

## Data and outputs

Large input data and generated outputs are not tracked in git. Controlled-access, sensitive, or derived large files should remain outside the repository or in approved storage locations.

## Related resources

- Pan-Infection T cell atlas website: https://tcellatlas.org
- Cell2Specificity: https://github.com/lisadratva/cell2specificity
