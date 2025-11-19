# Causal DDI Agent - Drug-Drug Interaction Adverse Effect Discovery

A causal inference agent for discovering drug-drug interactions (DDI) that cause adverse effects using FDA's FAERS (FDA Adverse Event Reporting System) database.

## Overview

This agent performs three main tasks:
1. **Candidate Selection**: Identifies drug combinations with sufficient cases for analysis
2. **Causal Inference**: Estimates causal effects using Mantel-Haenszel stratified analysis
3. **Literature Verification**: Checks if findings are documented in literature (PubMed integration ready)

## Features

- Loads and processes FAERS data from multiple quarters
- Identifies drug combinations (2+ drugs taken together)
- Extracts clinical features (age, sex, weight)
- Compares drug combinations vs monotherapy (proper control group)
- Estimates causal effects with confounder adjustment (age/sex stratification)
- Computes statistical significance (chi-square test)
- Provides adjusted risk ratios (Mantel-Haenszel method)
- Ready for PubMed API integration

## Installation

### Option 1: Using Homebrew + Poetry (Easiest for macOS)

```bash
# Install Poetry via Homebrew (avoids SSL certificate issues)
brew install poetry

# Install project dependencies
cd /path/to/hackathon_anthropic
poetry install

# Run the agent
poetry run python causal_ddi_agent.py
```

### Option 2: Using the Installation Helper Script

If you encounter SSL certificate errors:

```bash
# Run the helper script (will try Homebrew first, then fallback methods)
./install_poetry.sh

# Add Poetry to PATH
export PATH="$HOME/.local/bin:$PATH"

# Install dependencies
poetry install
```

### Option 3: Fix SSL Certificates First

If you have Python from python.org:

```bash
# Run the certificate installer
open /Applications/Python\ 3.11/Install\ Certificates.command

# Then install Poetry normally
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

### Option 4: Using pip (No Poetry)

```bash
pip install pandas numpy scipy requests scikit-learn
python causal_ddi_agent.py
```

## Usage

### Basic Usage

```bash
poetry run python causal_ddi_agent.py
```

### Programmatic Usage

```python
from causal_ddi_agent import CausalDDIAgent

# Initialize agent
agent = CausalDDIAgent(data_dir="/path/to/faers_data")
agent.initialize()

# Discover DDI adverse effects
candidates, causal_results = agent.discover_ddi_adverse_effects(top_n=10)

# Access results
for result in causal_results:
    if result.get('significant'):  # Only show significant findings
        print(f"{result['drug_combination']} → {result['adverse_event']}")
        print(f"  Adjusted Risk Ratio: {result['adjusted_risk_ratio']:.2f}")
        print(f"  P-value: {result['p_value']:.4f}")
```

## Data Structure

The FAERS database contains:
- **DEMO.txt**: Patient demographics (age, sex, weight, country)
- **DRUG.txt**: Drug information (drug names, role codes, dosage)
- **REAC.txt**: Adverse reactions (preferred terms)
- **INDI.txt**: Indications (conditions being treated)
- **OUTC.txt**: Outcomes (hospitalization, death, etc.)

## Methodology

### 1. Candidate Selection
- Identifies drug combinations appearing together in cases
- Filters by minimum frequency threshold (≥10 cases)
- Ranks candidates by number of co-occurrences

### 2. Causal Inference (Stratified Analysis)
- **Treatment group**: Cases with both drugs (combination therapy)
- **Control group**: Cases with either drug alone (monotherapy)
- **Confounder adjustment**: Mantel-Haenszel stratification by age and sex
- **Outcomes**: 
  - Crude risk ratio (unadjusted)
  - Adjusted risk ratio (stratified)
  - P-value (chi-square test)
  - Number of strata analyzed

### 3. Literature Verification
- Checks if DDI-AE association is documented
- Provides PubMed search queries
- Can be extended with full PubMed API integration

## Interpreting Results

**Adjusted RR > 2.0**: Strong causal effect (combination doubles risk)  
**Adjusted RR 1.5-2.0**: Moderate causal effect  
**Adjusted RR 1.1-1.5**: Weak causal effect  
**P-value < 0.05**: Statistically significant  
**P-value ≥ 0.05**: Not significant (could be chance finding)

## Example Output

```
Top candidates by frequency:
METFORMIN+INSULIN → Hypoglycaemia: 245 cases

Causal Inference with Stratification:

METFORMIN + INSULIN → Hypoglycaemia
  Combination risk: 0.156 (n=1250)
  Monotherapy risk: 0.045 (n=3840)
  Crude RR: 3.47
  Adjusted RR: 3.12
  P-value: 0.0001 ***
  Strata analyzed: 6
```

## Limitations & Future Work

- Uses Mantel-Haenszel stratification (can be extended with propensity score matching)
- Only adjusts for age and sex (other confounders like comorbidities not included)
- No temporal analysis (can add time-to-event modeling)
- No multiple testing correction (Bonferroni/FDR should be applied for many tests)
- Literature verification is a placeholder (needs PubMed API integration)
- No interaction with external knowledge graphs (can integrate DrugBank, SIDER)

## Why This Approach?

**Removed correlation analysis**: The "lift" metric was redundant - we can directly perform causal inference on all candidates.

**Better control group**: Comparing combination vs monotherapy (not "either drug vs both"), which is more clinically relevant.

**Statistical rigor**: Added p-values and significance testing, which was missing before.

**Confounder adjustment**: Implemented Mantel-Haenszel stratification to control for age/sex confounding.

## References

- "Causal AI Scientist: Facilitating Causal Data Science with Large Language Models"
- FAERS Database: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html

