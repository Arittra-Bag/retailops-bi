import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json

# Statistical testing imports
try:
    #from scipy import stats
    #import pingouin as pg
    from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind, bootstrap
    from sklearn.utils import resample
    STATS_AVAILABLE = True
except ImportError as e:
    print(f"Statistical packages not available: {e}")
    STATS_AVAILABLE = False

logger = logging.getLogger(__name__)

class ABTestingFramework:
    """Comprehensive A/B testing and experimentation framework"""
    
    def __init__(self):
        """Initialize A/B testing framework"""
        self.project_root = Path(__file__).parent.parent.parent
        self.data_path = self.project_root / "data" / "processed"
        self.experiments_path = self.project_root / "data" / "experiments"
        self.experiments_path.mkdir(exist_ok=True)
        
        # Storage for experiments
        self.experiments = {}
        self.results = {}
        
        logger.info("✅ A/B Testing Framework initialized")
    '''
    def power_analysis(self, effect_size: float, alpha: float = 0.05, power: float = 0.8, 
                      test_type: str = "two_sample") -> Dict[str, Any]:
        """Calculate required sample size for experiment"""
        try:
            if not STATS_AVAILABLE:
                return {
                    "status": "error",
                    "error": "Statistical packages not available",
                    "required_sample_size_per_group": None
                }
            
            # Using Cohen's conventions for effect sizes
            if test_type == "two_sample":
                # For two-sample t-test
                from statsmodels.stats.power import ttest_power
                
                # Calculate required sample size per group
                sample_size = ttest_power(effect_size, power, alpha, alternative='two-sided')
                
                # Handle NaN results
                if np.isnan(sample_size) or np.isinf(sample_size):
                    return {
                        "status": "error",
                        "error": "Cannot calculate sample size with given parameters. Try adjusting effect size, power, or alpha.",
                        "effect_size": effect_size,
                        "alpha": alpha,
                        "power": power,
                        "required_sample_size_per_group": None,
                        "suggestion": "Try effect_size >= 0.1, alpha between 0.01-0.1, power between 0.7-0.95"
                    }
                
                # Ensure we have a reasonable sample size
                if sample_size < 2:
                    sample_size = 30  # Minimum practical sample size
                
                return {
                    "status": "success",
                    "test_type": test_type,
                    "effect_size": float(effect_size),
                    "alpha": float(alpha),
                    "power": float(power),
                    "required_sample_size_per_group": int(np.ceil(sample_size)),
                    "total_sample_size": int(np.ceil(sample_size * 2)),
                    "interpretation": {
                        "small_effect": 0.2,
                        "medium_effect": 0.5,
                        "large_effect": 0.8,
                        "your_effect": float(effect_size)
                    }
                }
            
        except Exception as e:
            logger.error(f"Error in power analysis: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "required_sample_size_per_group": None,
                "total_sample_size": None,  # <-- Add this line to prevent frontend crash
                "suggestion": "Check if effect size, alpha, and power values are valid"
            }
    '''
    def design_experiment(self, experiment_name: str, hypothesis: str, 
                         metric: str, segments: List[str] = None,
                         allocation_ratio: List[float] = None) -> Dict[str, Any]:
        """Design a new A/B experiment"""
        try:
            if segments is None:
                segments = ["control", "treatment"]
            
            if allocation_ratio is None:
                allocation_ratio = [0.5, 0.5]  # 50/50 split
            
            if len(segments) != len(allocation_ratio):
                return {"error": "Segments and allocation ratio must have same length"}
            
            if not np.isclose(sum(allocation_ratio), 1.0):
                return {"error": "Allocation ratios must sum to 1.0"}
            
            experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            experiment_design = {
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
                "hypothesis": hypothesis,
                "primary_metric": metric,
                "segments": segments,
                "allocation_ratio": allocation_ratio,
                "status": "designed",
                "created_at": datetime.now().isoformat(),
                "start_date": None,
                "end_date": None,
                "sample_size_per_segment": None,
                "statistical_power": 0.8,
                "significance_level": 0.05,
                "minimum_detectable_effect": 0.05  # 5% relative change
            }
            
            # Store experiment
            self.experiments[experiment_id] = experiment_design
            
            # Save to file
            experiment_file = self.experiments_path / f"{experiment_id}.json"
            with open(experiment_file, 'w') as f:
                json.dump(experiment_design, f, indent=2)
            
            logger.info(f"✅ Experiment designed: {experiment_id}")
            
            return {
                "status": "success",
                "experiment_id": experiment_id,
                "design": experiment_design,
                "next_steps": [
                    "1. Calculate sample size using power_analysis()",
                    "2. Prepare user assignment strategy",
                    "3. Start experiment with start_experiment()",
                    "4. Monitor metrics during experiment",
                    "5. Analyze results with analyze_experiment()"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error designing experiment: {str(e)}")
            return {"error": str(e)}
    
    def simulate_ab_test(self, experiment_name: str, control_mean: float, 
                        treatment_mean: float, std_dev: float, 
                        sample_size_per_group: int = 1000) -> Dict[str, Any]:
        """Simulate A/B test with synthetic data for demonstration"""
        try:
            logger.info(f"🧪 Simulating A/B test: {experiment_name}")
            
            # Generate synthetic data
            np.random.seed(42)  # Reproducible results
            
            # Control group
            control_data = np.random.normal(control_mean, std_dev, sample_size_per_group)
            
            # Treatment group
            treatment_data = np.random.normal(treatment_mean, std_dev, sample_size_per_group)
            
            # Calculate effect size
            pooled_std = np.sqrt(((sample_size_per_group - 1) * std_dev**2 + 
                                 (sample_size_per_group - 1) * std_dev**2) / 
                                (sample_size_per_group + sample_size_per_group - 2))
            effect_size = (treatment_mean - control_mean) / pooled_std
            
            # Perform statistical tests
            t_stat, p_value = ttest_ind(control_data, treatment_data)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p_value = mannwhitneyu(control_data, treatment_data, alternative='two-sided')
            
            # Bootstrap confidence interval for difference
            def mean_diff(x, y):
                return np.mean(y) - np.mean(x)
            
            bootstrap_diffs = []
            for _ in range(1000):
                boot_control = resample(control_data)
                boot_treatment = resample(treatment_data)
                bootstrap_diffs.append(mean_diff(boot_control, boot_treatment))
            
            ci_lower = np.percentile(bootstrap_diffs, 2.5)
            ci_upper = np.percentile(bootstrap_diffs, 97.5)
            
            # Calculate business metrics
            control_stats = {
                "mean": float(np.mean(control_data)),
                "std": float(np.std(control_data)),
                "median": float(np.median(control_data)),
                "sample_size": sample_size_per_group
            }
            
            treatment_stats = {
                "mean": float(np.mean(treatment_data)),
                "std": float(np.std(treatment_data)),
                "median": float(np.median(treatment_data)),
                "sample_size": sample_size_per_group
            }
            
            # Relative change
            relative_change = ((treatment_stats["mean"] - control_stats["mean"]) / 
                             control_stats["mean"]) * 100
            
            # Statistical significance (convert to Python bool)
            is_significant = bool(p_value < 0.05)
            
            # Store results
            experiment_results = {
                "experiment_name": experiment_name,
                "experiment_type": "simulated_ab_test",
                "control_group": control_stats,
                "treatment_group": treatment_stats,
                "statistical_tests": {
                    "t_test": {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": is_significant
                    },
                    "mann_whitney_u": {
                        "u_statistic": float(u_stat),
                        "p_value": float(u_p_value),
                        "significant": bool(u_p_value < 0.05)
                    }
                },
                "effect_analysis": {
                    "effect_size_cohens_d": float(effect_size),
                    "absolute_difference": float(treatment_stats["mean"] - control_stats["mean"]),
                    "relative_change_percent": float(relative_change),
                    "confidence_interval_95": [float(ci_lower), float(ci_upper)]
                },
                "business_impact": {
                    "significant": bool(is_significant),
                    "recommendation": "Launch" if is_significant and relative_change > 0 else "Don't Launch",
                    "confidence_level": "High" if p_value < 0.01 else "Medium" if p_value < 0.05 else "Low"
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Save results
            result_id = f"simulation_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.results[result_id] = experiment_results
            
            result_file = self.experiments_path / f"results_{result_id}.json"
            with open(result_file, 'w') as f:
                json.dump(experiment_results, f, indent=2)
            
            logger.info("✅ A/B test simulation complete")
            
            return {
                "status": "success",
                "result_id": result_id,
                "results": experiment_results,
                "interpretation": self._interpret_results(experiment_results)
            }
            
        except Exception as e:
            logger.error(f"Error in A/B test simulation: {str(e)}")
            return {"error": str(e)}
    
    def analyze_customer_segments_ab(self) -> Dict[str, Any]:
        """Analyze customer segments using A/B testing principles"""
        try:
            logger.info("📊 Analyzing customer segments with A/B testing framework...")
            
            # Load customer data
            customer_file = self.data_path / "customer_analysis.csv"
            if not customer_file.exists():
                return {"error": "Customer analysis data not found"}
            
            customers_df = pd.read_csv(customer_file)
            
            # Create customer segments based on spending
            customers_df['Spending_Quartile'] = pd.qcut(customers_df['Total_Spent'], 
                                                       q=4, labels=['Low', 'Medium', 'High', 'Premium'])
            
            # Analyze differences between segments
            segments = ['Low', 'Medium', 'High', 'Premium']
            comparisons = []
            
            for i in range(len(segments)):
                for j in range(i+1, len(segments)):
                    seg1, seg2 = segments[i], segments[j]
                    
                    data1 = customers_df[customers_df['Spending_Quartile'] == seg1]['Total_Spent']
                    data2 = customers_df[customers_df['Spending_Quartile'] == seg2]['Total_Spent']
                    
                    if len(data1) > 10 and len(data2) > 10:  # Minimum sample size
                        # Statistical test
                        t_stat, p_value = ttest_ind(data1, data2)
                        effect_size = (np.mean(data2) - np.mean(data1)) / np.sqrt(
                            (np.var(data1) + np.var(data2)) / 2
                        )
                        
                        comparisons.append({
                            "segment_1": seg1,
                            "segment_2": seg2,
                            "segment_1_mean": float(np.mean(data1)),
                            "segment_2_mean": float(np.mean(data2)),
                            "difference": float(np.mean(data2) - np.mean(data1)),
                            "t_statistic": float(t_stat),
                            "p_value": float(p_value),
                            "effect_size": float(effect_size),
                            "significant": p_value < 0.05,
                            "sample_size_1": len(data1),
                            "sample_size_2": len(data2)
                        })
            
            # Overall segment analysis
            segment_summary = customers_df.groupby('Spending_Quartile').agg({
                'Total_Spent': ['count', 'mean', 'std', 'median'],
                'Transaction_Count': 'mean',
                'Avg_Order_Value': 'mean'
            }).round(2)
            
            return {
                "status": "success",
                "analysis_type": "Customer Segment A/B Analysis",
                "segments_analyzed": len(segments),
                "total_comparisons": len(comparisons),
                "significant_differences": sum(1 for c in comparisons if c['significant']),
                "segment_comparisons": comparisons,
                "segment_summary": segment_summary.to_dict(),
                "methodology": "Independent t-tests with effect size calculation",
                "interpretation": "Statistical analysis of customer segment differences",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in customer segment analysis: {str(e)}")
            return {"error": str(e)}
    
    def bayesian_ab_analysis(self, control_conversions: int, control_total: int,
                           treatment_conversions: int, treatment_total: int) -> Dict[str, Any]:
        """Bayesian A/B test analysis for conversion rates"""
        try:
            logger.info("🎯 Performing Bayesian A/B analysis...")
            
            # Beta-Binomial model for conversion rates
            # Prior: Beta(1, 1) - uniform prior
            alpha_prior, beta_prior = 1, 1
            
            # Posterior distributions
            alpha_control = alpha_prior + control_conversions
            beta_control = beta_prior + (control_total - control_conversions)
            
            alpha_treatment = alpha_prior + treatment_conversions
            beta_treatment = beta_prior + (treatment_total - treatment_conversions)
            
            # Sample from posteriors
            n_samples = 10000
            control_posterior = np.random.beta(alpha_control, beta_control, n_samples)
            treatment_posterior = np.random.beta(alpha_treatment, beta_treatment, n_samples)
            
            # Calculate probability that treatment > control
            prob_treatment_better = np.mean(treatment_posterior > control_posterior)
            
            # Credible intervals
            control_ci = np.percentile(control_posterior, [2.5, 97.5])
            treatment_ci = np.percentile(treatment_posterior, [2.5, 97.5])
            
            # Expected values
            control_rate = control_conversions / control_total
            treatment_rate = treatment_conversions / treatment_total
            
            # Relative lift
            lift_posterior = (treatment_posterior - control_posterior) / control_posterior
            expected_lift = np.mean(lift_posterior) * 100
            lift_ci = np.percentile(lift_posterior, [2.5, 97.5]) * 100
            
            return {
                "status": "success",
                "analysis_type": "Bayesian A/B Test",
                "control_group": {
                    "conversions": control_conversions,
                    "total": control_total,
                    "rate": float(control_rate),
                    "credible_interval_95": [float(control_ci[0]), float(control_ci[1])]
                },
                "treatment_group": {
                    "conversions": treatment_conversions,
                    "total": treatment_total,
                    "rate": float(treatment_rate),
                    "credible_interval_95": [float(treatment_ci[0]), float(treatment_ci[1])]
                },
                "bayesian_results": {
                    "prob_treatment_better": float(prob_treatment_better),
                    "expected_lift_percent": float(expected_lift),
                    "lift_credible_interval": [float(lift_ci[0]), float(lift_ci[1])],
                    "decision": "Launch" if prob_treatment_better > 0.95 else "Don't Launch",
                    "confidence": "High" if prob_treatment_better > 0.95 else "Medium" if prob_treatment_better > 0.8 else "Low"
                },
                "interpretation": {
                    "strong_evidence": bool(prob_treatment_better > 0.95),
                    "moderate_evidence": bool(0.8 < prob_treatment_better <= 0.95),
                    "weak_evidence": bool(prob_treatment_better <= 0.8)
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in Bayesian analysis: {str(e)}")
            return {"error": str(e)}
    
    def sequential_testing(self, data: List[Dict], metric_name: str) -> Dict[str, Any]:
        """Sequential testing for early stopping"""
        try:
            logger.info("⏰ Performing sequential testing analysis...")
            
            if len(data) < 100:
                return {"error": "Insufficient data for sequential testing"}
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            if 'group' not in df.columns or metric_name not in df.columns:
                return {"error": f"Required columns 'group' and '{metric_name}' not found"}
            
            # Sequential p-values
            control_data = df[df['group'] == 'control'][metric_name].values
            treatment_data = df[df['group'] == 'treatment'][metric_name].values
            
            min_samples = 50  # Minimum samples before testing
            p_values = []
            sample_sizes = []
            
            max_samples = min(len(control_data), len(treatment_data))
            
            for n in range(min_samples, max_samples, 10):
                if n <= len(control_data) and n <= len(treatment_data):
                    _, p_val = ttest_ind(control_data[:n], treatment_data[:n])
                    p_values.append(p_val)
                    sample_sizes.append(n)
            
            # Alpha spending function (simple Bonferroni correction)
            alpha_spent = 0.05 / len(p_values)
            
            # Find early stopping point
            early_stop = None
            for i, p_val in enumerate(p_values):
                if p_val < alpha_spent:
                    early_stop = sample_sizes[i]
                    break
            
            return {
                "status": "success",
                "analysis_type": "Sequential Testing",
                "total_samples_available": max_samples,
                "tests_performed": len(p_values),
                "alpha_spent_per_test": alpha_spent,
                "early_stopping_point": early_stop,
                "final_p_value": float(p_values[-1]) if p_values else None,
                "sequential_p_values": [float(p) for p in p_values],
                "sample_sizes": sample_sizes,
                "recommendation": "Stop early" if early_stop else "Continue to planned sample size",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in sequential testing: {str(e)}")
            return {"error": str(e)}
    
    def _interpret_results(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Interpret experiment results for business stakeholders"""
        try:
            p_value = results['statistical_tests']['t_test']['p_value']
            relative_change = results['effect_analysis']['relative_change_percent']
            effect_size = results['effect_analysis']['effect_size_cohens_d']
            
            interpretation = {}
            
            # Statistical significance
            if p_value < 0.01:
                interpretation['significance'] = "Highly significant (p < 0.01)"
            elif p_value < 0.05:
                interpretation['significance'] = "Significant (p < 0.05)"
            else:
                interpretation['significance'] = "Not significant (p >= 0.05)"
            
            # Effect size interpretation
            if abs(effect_size) < 0.2:
                interpretation['effect_magnitude'] = "Small effect"
            elif abs(effect_size) < 0.5:
                interpretation['effect_magnitude'] = "Medium effect"
            else:
                interpretation['effect_magnitude'] = "Large effect"
            
            # Business recommendation
            if p_value < 0.05 and relative_change > 2:
                interpretation['recommendation'] = "Strong recommendation to launch"
            elif p_value < 0.05 and relative_change > 0:
                interpretation['recommendation'] = "Moderate recommendation to launch"
            elif p_value >= 0.05:
                interpretation['recommendation'] = "Insufficient evidence to launch"
            else:
                interpretation['recommendation'] = "Do not launch - negative impact"
            
            return interpretation
            
        except Exception as e:
            logger.error(f"Error interpreting results: {str(e)}")
            return {"error": str(e)}
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments"""
        try:
            return {
                "total_experiments": len(self.experiments),
                "total_results": len(self.results),
                "experiments": list(self.experiments.keys()),
                "results": list(self.results.keys()),
                "framework_capabilities": [
                    "Power Analysis & Sample Size Calculation",
                    "Experiment Design & Management",
                    "Classical Frequentist Testing",
                    "Bayesian A/B Analysis",
                    "Sequential Testing",
                    "Customer Segment Analysis",
                    "Effect Size Calculation",
                    "Business Impact Assessment"
                ],
                "statistical_methods": [
                    "Independent t-tests",
                    "Mann-Whitney U tests",
                    "Bootstrap confidence intervals",
                    "Beta-Binomial Bayesian models",
                    "Sequential hypothesis testing",
                    "Effect size (Cohen's d)"
                ],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting experiment summary: {str(e)}")
            return {"error": str(e)}


def main():
    """Main function to test A/B testing framework"""
    try:
        # Initialize A/B testing framework
        ab_framework = ABTestingFramework()
        
        print("🧪 Testing A/B Testing & Experimentation Framework")
        print("=" * 70)
        '''
        # Test 1: Power Analysis
        print("\n1️⃣ Testing Power Analysis...")
        power_result = ab_framework.power_analysis(effect_size=0.3, power=0.8)
        if power_result['status'] == 'success':
            print(f"✅ Power analysis complete:")
            print(f"   📊 Sample size needed: {power_result['required_sample_size_per_group']} per group")
            print(f"   🎯 Effect size: {power_result['effect_size']} (Medium)")
        else:
            print(f"❌ Power analysis failed: {power_result['error']}")
        '''
        # Test 2: Experiment Design
        print("\n2️⃣ Testing Experiment Design...")
        design_result = ab_framework.design_experiment(
            experiment_name="Customer_Retention_Campaign",
            hypothesis="Email marketing campaign will increase customer retention by 5%",
            metric="retention_rate"
        )
        if design_result.get('status') == 'success':
            print(f"✅ Experiment designed:")
            print(f"   🔬 ID: {design_result['experiment_id']}")
            print(f"   📈 Metric: {design_result['design']['primary_metric']}")
        
        # Test 3: A/B Test Simulation
        print("\n3️⃣ Testing A/B Test Simulation...")
        simulation_result = ab_framework.simulate_ab_test(
            experiment_name="Revenue_Optimization",
            control_mean=100.0,
            treatment_mean=105.0,  # 5% improvement
            std_dev=20.0,
            sample_size_per_group=1000
        )
        if simulation_result.get('status') == 'success':
            results = simulation_result['results']
            print(f"✅ A/B simulation complete:")
            print(f"   📊 Control mean: £{results['control_group']['mean']:.2f}")
            print(f"   📈 Treatment mean: £{results['treatment_group']['mean']:.2f}")
            print(f"   🎯 P-value: {results['statistical_tests']['t_test']['p_value']:.4f}")
            print(f"   💡 Recommendation: {results['business_impact']['recommendation']}")
        
        # Test 4: Customer Segment Analysis
        print("\n4️⃣ Testing Customer Segment Analysis...")
        segment_result = ab_framework.analyze_customer_segments_ab()
        if segment_result.get('status') == 'success':
            print(f"✅ Segment analysis complete:")
            print(f"   📊 Segments: {segment_result['segments_analyzed']}")
            print(f"   🔍 Comparisons: {segment_result['total_comparisons']}")
            print(f"   ✨ Significant: {segment_result['significant_differences']}")
        
        # Test 5: Bayesian Analysis
        print("\n5️⃣ Testing Bayesian Analysis...")
        bayesian_result = ab_framework.bayesian_ab_analysis(
            control_conversions=85, control_total=1000,
            treatment_conversions=95, treatment_total=1000
        )
        if bayesian_result.get('status') == 'success':
            bayesian = bayesian_result['bayesian_results']
            print(f"✅ Bayesian analysis complete:")
            print(f"   🎯 Prob treatment better: {bayesian['prob_treatment_better']:.3f}")
            print(f"   📈 Expected lift: {bayesian['expected_lift_percent']:.1f}%")
            print(f"   💡 Decision: {bayesian['decision']}")
        
        # Test 6: Framework Summary
        print("\n6️⃣ Framework Summary...")
        summary = ab_framework.get_experiment_summary()
        print(f"✅ Framework capabilities: {len(summary['framework_capabilities'])}")
        print(f"✅ Statistical methods: {len(summary['statistical_methods'])}")
        
        print("\n" + "=" * 70)
        print("🎉 A/B Testing Framework Test Complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()