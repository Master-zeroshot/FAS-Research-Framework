"""
Visualization and Analysis Tools for Face Anti-Spoofing

This module provides comprehensive visualization and analysis tools
for understanding and interpreting face anti-spoofing models.
"""

from .attention_visualization import (
    AttentionVisualizer, AttentionAnalyzer, AttentionHeatmapGenerator,
    visualize_attention, analyze_attention_patterns
)

from .feature_analysis import (
    FeatureAnalyzer, FeatureVisualizer, FeatureClustering,
    analyze_features, visualize_features
)

from .model_interpretability import (
    GradCAM, IntegratedGradients, LIME, ModelInterpretabilityAnalyzer,
    analyze_model_interpretability, create_interpretability_visualization
)

from .failure_case_analysis import (
    FailureCaseAnalyzer, FailureCaseReporter,
    analyze_failure_cases, create_failure_visualization
)

from .interactive_dashboards import (
    InteractiveDashboard, StreamlitDashboard,
    create_interactive_dashboard, create_streamlit_dashboard
)

__all__ = [
    # Attention Visualization
    'AttentionVisualizer', 'AttentionAnalyzer', 'AttentionHeatmapGenerator',
    'visualize_attention', 'analyze_attention_patterns',
    
    # Feature Analysis
    'FeatureAnalyzer', 'FeatureVisualizer', 'FeatureClustering',
    'analyze_features', 'visualize_features',
    
    # Model Interpretability
    'GradCAM', 'IntegratedGradients', 'LIME', 'ModelInterpretabilityAnalyzer',
    'analyze_model_interpretability', 'create_interpretability_visualization',
    
    # Failure Case Analysis
    'FailureCaseAnalyzer', 'FailureCaseReporter',
    'analyze_failure_cases', 'create_failure_visualization',
    
    # Interactive Dashboards
    'InteractiveDashboard', 'StreamlitDashboard',
    'create_interactive_dashboard', 'create_streamlit_dashboard'
]
