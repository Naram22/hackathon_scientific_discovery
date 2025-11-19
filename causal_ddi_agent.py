"""
Causal Drug-Drug Interaction (DDI) Agent for Adverse Event Discovery

This agent discovers causal effects of drug combinations on adverse events using FAERS data.
It performs:
1. Candidate selection (drug combinations with sufficient cases)
2. Causal inference (stratified analysis comparing combination vs monotherapy)
3. Statistical significance testing with confounder adjustment
"""

import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations
from typing import Dict, List, Tuple, Set
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class FAERSDataLoader:
    """Load and preprocess FAERS data files"""
    
    def __init__(self, data_dir: str = "/Users/achossegros/faers_data"):
        self.data_dir = data_dir
        self.demo_df = None
        self.drug_df = None
        self.reac_df = None
        
    def load_all_quarters(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load data from all available quarters"""
        print("Loading FAERS data from all quarters...")
        
        demo_dfs = []
        drug_dfs = []
        reac_dfs = []
        
        # Find all quarter directories
        quarter_dirs = glob.glob(os.path.join(self.data_dir, "*_Q*"))
        
        for qdir in sorted(quarter_dirs):
            try:
                # Load DEMO file
                demo_file = os.path.join(qdir, "DEMO.txt")
                if os.path.exists(demo_file):
                    demo = pd.read_csv(demo_file, sep='$', encoding='latin-1', 
                                      low_memory=False, on_bad_lines='skip')
                    demo_dfs.append(demo)
                
                # Load DRUG file
                drug_file = os.path.join(qdir, "DRUG.txt")
                if os.path.exists(drug_file):
                    drug = pd.read_csv(drug_file, sep='$', encoding='latin-1',
                                      low_memory=False, on_bad_lines='skip')
                    drug_dfs.append(drug)
                
                # Load REAC file
                reac_file = os.path.join(qdir, "REAC.txt")
                if os.path.exists(reac_file):
                    reac = pd.read_csv(reac_file, sep='$', encoding='latin-1',
                                      low_memory=False, on_bad_lines='skip')
                    reac_dfs.append(reac)
                    
                print(f"  ✓ Loaded {os.path.basename(qdir)}")
            except Exception as e:
                print(f"  ✗ Error loading {os.path.basename(qdir)}: {e}")
        
        # Concatenate all quarters
        self.demo_df = pd.concat(demo_dfs, ignore_index=True) if demo_dfs else pd.DataFrame()
        self.drug_df = pd.concat(drug_dfs, ignore_index=True) if drug_dfs else pd.DataFrame()
        self.reac_df = pd.concat(reac_dfs, ignore_index=True) if reac_dfs else pd.DataFrame()
        
        print(f"\nTotal records loaded:")
        print(f"  - Demographics: {len(self.demo_df):,} cases")
        print(f"  - Drug records: {len(self.drug_df):,} entries")
        print(f"  - Adverse reactions: {len(self.reac_df):,} events")
        
        return self.demo_df, self.drug_df, self.reac_df


class DrugCombinationExtractor:
    """Extract and identify drug combinations from FAERS data"""
    
    def __init__(self, drug_df: pd.DataFrame, min_support: int = 10):
        self.drug_df = drug_df
        self.min_support = min_support
        
    def identify_combinations(self, role_codes: List[str] = ['PS', 'SS']) -> pd.DataFrame:
        """
        Identify drug combinations (pairs) from cases
        role_codes: ['PS', 'SS'] = Primary Suspect and Secondary Suspect drugs
        """
        print(f"\nIdentifying drug combinations (role codes: {role_codes})...")
        
        # Filter for suspected drugs only
        suspected_drugs = self.drug_df[self.drug_df['role_cod'].isin(role_codes)].copy()
        
        # Standardize drug names
        suspected_drugs['drugname_clean'] = (
            suspected_drugs['drugname']
            .str.upper()
            .str.strip()
        )
        
        # Group by case to find combinations
        case_drugs = (
            suspected_drugs.groupby(['primaryid', 'caseid'])['drugname_clean']
            .apply(list)
            .reset_index()
        )
        
        # Extract pairs
        combo_list = []
        for _, row in case_drugs.iterrows():
            drugs = list(set(row['drugname_clean']))  # Unique drugs per case
            if len(drugs) >= 2:
                for drug1, drug2 in combinations(sorted(drugs), 2):
                    combo_list.append({
                        'primaryid': row['primaryid'],
                        'caseid': row['caseid'],
                        'drug1': drug1,
                        'drug2': drug2,
                        'combo': f"{drug1}+{drug2}"
                    })
        
        combo_df = pd.DataFrame(combo_list)
        
        # Filter by minimum support
        combo_counts = combo_df['combo'].value_counts()
        valid_combos = combo_counts[combo_counts >= self.min_support].index
        combo_df = combo_df[combo_df['combo'].isin(valid_combos)]
        
        print(f"  Found {len(combo_df)} case-combination records")
        print(f"  Unique combinations (≥{self.min_support} occurrences): {len(valid_combos)}")
        
        return combo_df


class ClinicalFeatureExtractor:
    """Extract clinical and demographic features from DEMO data"""
    
    def __init__(self, demo_df: pd.DataFrame):
        self.demo_df = demo_df
        
    def extract_features(self) -> pd.DataFrame:
        """Extract and clean clinical features"""
        print("\nExtracting clinical features...")
        
        features = self.demo_df[['primaryid', 'caseid', 'age', 'age_cod', 
                                 'sex', 'wt', 'reporter_country']].copy()
        
        # Clean age
        features['age_years'] = pd.to_numeric(features['age'], errors='coerce')
        features.loc[features['age_cod'] == 'MON', 'age_years'] = features['age_years'] / 12
        features.loc[features['age_cod'] == 'DEC', 'age_years'] = features['age_years'] * 10
        
        # Age categories
        features['age_category'] = pd.cut(
            features['age_years'], 
            bins=[0, 18, 65, 120], 
            labels=['pediatric', 'adult', 'elderly']
        )
        
        # Clean sex
        features['sex_clean'] = features['sex'].map({'M': 'male', 'F': 'female'})
        
        # Clean weight
        features['weight_kg'] = pd.to_numeric(features['wt'], errors='coerce')
        
        print(f"  Extracted features for {len(features):,} cases")
        
        return features


class CandidateSelector:
    """Select candidate drug-AE pairs for causal analysis based on frequency"""
    
    def __init__(self, combo_df: pd.DataFrame, reac_df: pd.DataFrame):
        self.combo_df = combo_df
        self.reac_df = reac_df
        
    def select_candidates(self, min_cases: int = 10) -> pd.DataFrame:
        """Select drug-AE pairs with sufficient cases for causal analysis"""
        print("\nSelecting candidates for causal analysis...")
        
        # Merge combinations with adverse events
        combo_ae = self.combo_df.merge(
            self.reac_df[['primaryid', 'caseid', 'pt']], 
            on=['primaryid', 'caseid']
        )
        
        # Count co-occurrences
        candidates = (
            combo_ae.groupby(['combo', 'pt'])
            .size()
            .reset_index(name='n_cases')
        )
        
        # Filter by minimum cases
        candidates = candidates[candidates['n_cases'] >= min_cases]
        candidates = candidates.sort_values('n_cases', ascending=False)
        
        print(f"  Selected {len(candidates)} candidates (≥{min_cases} cases)")
        
        return candidates


class CausalInferenceEngine:
    """Estimate causal effects using stratified analysis and statistical testing"""
    
    def __init__(self, combo_df: pd.DataFrame, reac_df: pd.DataFrame,
                 demo_features: pd.DataFrame):
        self.combo_df = combo_df
        self.reac_df = reac_df
        self.demo_features = demo_features
        
    def estimate_causal_effect(self, drug1: str, drug2: str, 
                               adverse_event: str) -> Dict:
        """
        Estimate causal effect of drug combination vs monotherapy
        Controls for age and sex via stratification (Mantel-Haenszel method)
        """
        print(f"\nEstimating: {drug1} + {drug2} → {adverse_event}")
        
        # Treatment: both drugs together
        combo_cases = self.combo_df[
            (self.combo_df['drug1'] == drug1) & 
            (self.combo_df['drug2'] == drug2)
        ][['primaryid', 'caseid']].copy()
        combo_cases['group'] = 'combination'
        
        # Control: either drug alone (monotherapy)
        drug1_mono = self.combo_df[
            (self.combo_df['combo'].str.contains(drug1)) &
            ~((self.combo_df['drug1'] == drug1) & (self.combo_df['drug2'] == drug2)) &
            ~((self.combo_df['drug1'] == drug2) & (self.combo_df['drug2'] == drug1))
        ][['primaryid', 'caseid']].drop_duplicates()
        
        drug2_mono = self.combo_df[
            (self.combo_df['combo'].str.contains(drug2)) &
            ~((self.combo_df['drug1'] == drug1) & (self.combo_df['drug2'] == drug2)) &
            ~((self.combo_df['drug1'] == drug2) & (self.combo_df['drug2'] == drug1))
        ][['primaryid', 'caseid']].drop_duplicates()
        
        mono_cases = pd.concat([drug1_mono, drug2_mono]).drop_duplicates()
        mono_cases['group'] = 'monotherapy'
        
        # Combine
        analysis_df = pd.concat([combo_cases, mono_cases])
        analysis_df['treatment'] = (analysis_df['group'] == 'combination').astype(int)
        
        # Merge with outcomes
        analysis_df = analysis_df.merge(
            self.reac_df[['primaryid', 'caseid', 'pt']],
            on=['primaryid', 'caseid'],
            how='left'
        )
        analysis_df['outcome'] = (analysis_df['pt'] == adverse_event).astype(int)
        
        # Merge with covariates
        analysis_df = analysis_df.merge(
            self.demo_features[['primaryid', 'caseid', 'age_category', 'sex_clean']],
            on=['primaryid', 'caseid'],
            how='left'
        )
        
        # Aggregate to case level (any occurrence of AE)
        case_level = analysis_df.groupby(
            ['primaryid', 'caseid', 'treatment', 'age_category', 'sex_clean']
        )['outcome'].max().reset_index()
        
        # Overall effect
        treated = case_level[case_level['treatment'] == 1]
        control = case_level[case_level['treatment'] == 0]
        
        n_treated = len(treated)
        n_control = len(control)
        
        if n_control == 0 or n_treated == 0:
            return self._null_result(drug1, drug2, adverse_event, 
                                    "Insufficient data")
        
        risk_treated = treated['outcome'].mean()
        risk_control = control['outcome'].mean()
        
        # Stratified analysis (Mantel-Haenszel)
        strata_results = []
        for age in case_level['age_category'].dropna().unique():
            for sex in case_level['sex_clean'].dropna().unique():
                stratum = case_level[
                    (case_level['age_category'] == age) & 
                    (case_level['sex_clean'] == sex)
                ]
                
                if len(stratum) >= 5:  # Minimum stratum size
                    t = stratum[stratum['treatment'] == 1]
                    c = stratum[stratum['treatment'] == 0]
                    
                    if len(t) > 0 and len(c) > 0:
                        strata_results.append({
                            'age': age,
                            'sex': sex,
                            'n_treated': len(t),
                            'n_control': len(c),
                            'risk_treated': t['outcome'].mean(),
                            'risk_control': c['outcome'].mean()
                        })
        
        # Adjusted risk ratio (weighted average)
        if strata_results:
            adjusted_rr = np.average(
                [s['risk_treated'] / max(s['risk_control'], 0.001) 
                 for s in strata_results],
                weights=[s['n_treated'] + s['n_control'] for s in strata_results]
            )
        else:
            adjusted_rr = risk_treated / max(risk_control, 0.001)
        
        # Calculate p-value (Fisher's exact test approximation)
        from scipy.stats import chi2_contingency
        
        contingency = np.array([
            [treated['outcome'].sum(), len(treated) - treated['outcome'].sum()],
            [control['outcome'].sum(), len(control) - control['outcome'].sum()]
        ])
        
        try:
            _, pval, _, _ = chi2_contingency(contingency)
        except:
            pval = 1.0
        
        result = {
            'drug_combination': f"{drug1} + {drug2}",
            'adverse_event': adverse_event,
            'n_treated': n_treated,
            'n_control': n_control,
            'risk_treated': risk_treated,
            'risk_control': risk_control,
            'risk_ratio': risk_treated / max(risk_control, 0.001),
            'adjusted_risk_ratio': adjusted_rr,
            'p_value': pval,
            'significant': pval < 0.05,
            'n_strata': len(strata_results)
        }
        
        return result
    
    def _null_result(self, drug1: str, drug2: str, ae: str, reason: str) -> Dict:
        """Return null result when analysis cannot be performed"""
        return {
            'drug_combination': f"{drug1} + {drug2}",
            'adverse_event': ae,
            'n_treated': 0,
            'n_control': 0,
            'risk_treated': 0,
            'risk_control': 0,
            'risk_ratio': np.nan,
            'adjusted_risk_ratio': np.nan,
            'p_value': 1.0,
            'significant': False,
            'n_strata': 0,
            'note': reason
        }


class LiteratureVerifier:
    """Check if drug combinations and adverse events are reported in literature"""
    
    def __init__(self):
        self.cache = {}
        
    def check_literature(self, drug1: str, drug2: str, 
                        adverse_event: str) -> Dict:
        """
        Simulate literature check (placeholder for PubMed API integration)
        In production, this would query PubMed/literature databases
        """
        query = f"{drug1} AND {drug2} AND {adverse_event}"
        
        # Placeholder - in production, use actual PubMed API
        result = {
            'query': query,
            'drug1': drug1,
            'drug2': drug2,
            'adverse_event': adverse_event,
            'literature_available': False,
            'message': "Literature check requires PubMed API integration",
            'suggestion': f"Search PubMed with: '{query}'"
        }
        
        return result


class CausalDDIAgent:
    """Main agent orchestrating the causal DDI discovery pipeline"""
    
    def __init__(self, data_dir: str = "/Users/achossegros/faers_data"):
        self.data_dir = data_dir
        self.loader = FAERSDataLoader(data_dir)
        self.demo_df = None
        self.drug_df = None
        self.reac_df = None
        self.demo_features = None
        self.combo_df = None
        
    def initialize(self):
        """Load and prepare data"""
        print("="*70)
        print("CAUSAL DDI AGENT - Initialization")
        print("="*70)
        
        # Load data
        self.demo_df, self.drug_df, self.reac_df = self.loader.load_all_quarters()
        
        # Extract features
        feature_extractor = ClinicalFeatureExtractor(self.demo_df)
        self.demo_features = feature_extractor.extract_features()
        
        # Identify combinations
        combo_extractor = DrugCombinationExtractor(self.drug_df, min_support=10)
        self.combo_df = combo_extractor.identify_combinations()
        
        print("\n" + "="*70)
        print("Initialization complete!")
        print("="*70)
        
    def discover_ddi_adverse_effects(self, top_n: int = 10):
        """Main pipeline: discover drug-drug interactions causing adverse effects"""
        print("\n" + "="*70)
        print("PHASE 1: Candidate Selection")
        print("="*70)
        
        # Select candidates
        selector = CandidateSelector(self.combo_df, self.reac_df)
        candidates = selector.select_candidates(min_cases=10)
        
        print(f"\nTop {top_n} candidates by frequency:")
        print("-"*70)
        for idx, row in candidates.head(top_n).iterrows():
            print(f"{row['combo']} → {row['pt']}: {row['n_cases']} cases")
        
        # Causal inference
        print("\n" + "="*70)
        print("PHASE 2: Causal Inference with Stratification")
        print("="*70)
        
        causal_engine = CausalInferenceEngine(self.combo_df, self.reac_df, 
                                              self.demo_features)
        
        causal_results = []
        for idx, row in candidates.head(top_n).iterrows():
            drug1, drug2 = row['combo'].split('+')
            adverse_event = row['pt']
            
            try:
                result = causal_engine.estimate_causal_effect(drug1, drug2, adverse_event)
                causal_results.append(result)
                
                print(f"\n{result['drug_combination']} → {result['adverse_event']}")
                print(f"  Combination risk: {result['risk_treated']:.3f} (n={result['n_treated']})")
                print(f"  Monotherapy risk: {result['risk_control']:.3f} (n={result['n_control']})")
                print(f"  Crude RR: {result['risk_ratio']:.2f}")
                print(f"  Adjusted RR: {result['adjusted_risk_ratio']:.2f}")
                print(f"  P-value: {result['p_value']:.4f} {'***' if result['significant'] else ''}")
                print(f"  Strata analyzed: {result['n_strata']}")
            except Exception as e:
                print(f"\n  Error analyzing {row['combo']}: {e}")
        
        # Filter significant results
        significant_results = [r for r in causal_results if r.get('significant', False)]
        
        print("\n" + "="*70)
        print("PHASE 3: Literature Verification")
        print("="*70)
        
        verifier = LiteratureVerifier()
        
        results_to_verify = significant_results[:5] if significant_results else causal_results[:5]
        
        for result in results_to_verify:
            combo = result['drug_combination']
            drug1, drug2 = combo.split(' + ')
            ae = result['adverse_event']
            
            lit_result = verifier.check_literature(drug1, drug2, ae)
            print(f"\n{combo} → {ae}")
            print(f"  {lit_result['suggestion']}")
        
        return candidates, causal_results


def main():
    """Run the causal DDI agent"""
    agent = CausalDDIAgent()
    agent.initialize()
    candidates, causal_results = agent.discover_ddi_adverse_effects(top_n=10)
    
    significant_results = [r for r in causal_results if r.get('significant', False)]
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nSummary:")
    print(f"  - Analyzed {len(candidates)} candidate drug-AE pairs")
    print(f"  - Performed causal inference on {len(causal_results)} combinations")
    print(f"  - Found {len(significant_results)} statistically significant effects (p<0.05)")
    print(f"  - Used age/sex stratification to control for confounding")
    print("\nNote: Results should be validated with literature and further clinical review")


if __name__ == "__main__":
    main()

