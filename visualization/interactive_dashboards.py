"""
Interactive Dashboards for Face Anti-Spoofing

This module provides tools for creating interactive dashboards
for visualizing and analyzing face anti-spoofing models.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd

from utils.logging_config import get_logger

logger = get_logger(__name__)


class InteractiveDashboard:
    """Interactive dashboard for face anti-spoofing analysis."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize interactive dashboard.
        
        Args:
            model: Model to analyze
            device: Device to run analysis on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        logger.info("Initialized interactive dashboard")
    
    def create_performance_dashboard(self, evaluation_results: Dict[str, Any]) -> go.Figure:
        """
        Create performance dashboard.
        
        Args:
            evaluation_results: Results from model evaluation
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy by Dataset', 'Confusion Matrix', 'ROC Curve', 'Performance Metrics'),
            specs=[[{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Accuracy by dataset
        if 'dataset_accuracy' in evaluation_results:
            datasets = list(evaluation_results['dataset_accuracy'].keys())
            accuracies = list(evaluation_results['dataset_accuracy'].values())
            
            fig.add_trace(
                go.Bar(x=datasets, y=accuracies, name='Accuracy'),
                row=1, col=1
            )
        
        # Confusion matrix
        if 'confusion_matrix' in evaluation_results:
            cm = evaluation_results['confusion_matrix']
            fig.add_trace(
                go.Heatmap(z=cm, x=['Predicted Real', 'Predicted Spoof'], 
                          y=['Actual Real', 'Actual Spoof'], name='Confusion Matrix'),
                row=1, col=2
            )
        
        # ROC curve
        if 'roc_curve' in evaluation_results:
            fpr, tpr = evaluation_results['roc_curve']
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'),
                row=2, col=1
            )
        
        # Performance metrics
        if 'metrics' in evaluation_results:
            metrics = evaluation_results['metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            fig.add_trace(
                go.Bar(x=metric_names, y=metric_values, name='Metrics'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, title_text="Model Performance Dashboard")
        return fig
    
    def create_feature_dashboard(self, feature_data: Dict[str, Any]) -> go.Figure:
        """
        Create feature analysis dashboard.
        
        Args:
            feature_data: Feature analysis data
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Distribution', 'Feature Correlation', 't-SNE Visualization', 'Feature Importance'),
            specs=[[{"type": "histogram"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Feature distribution
        if 'feature_distribution' in feature_data:
            features = feature_data['feature_distribution']
            fig.add_trace(
                go.Histogram(x=features, name='Feature Distribution'),
                row=1, col=1
            )
        
        # Feature correlation
        if 'correlation_matrix' in feature_data:
            corr_matrix = feature_data['correlation_matrix']
            fig.add_trace(
                go.Heatmap(z=corr_matrix, name='Correlation'),
                row=1, col=2
            )
        
        # t-SNE visualization
        if 'tsne_features' in feature_data and 'tsne_labels' in feature_data:
            tsne_features = feature_data['tsne_features']
            tsne_labels = feature_data['tsne_labels']
            
            # Create scatter plot
            for label in np.unique(tsne_labels):
                mask = tsne_labels == label
                fig.add_trace(
                    go.Scatter(x=tsne_features[mask, 0], y=tsne_features[mask, 1],
                              mode='markers', name=f'Class {label}'),
                    row=2, col=1
                )
        
        # Feature importance
        if 'feature_importance' in feature_data:
            importance = feature_data['feature_importance']
            feature_names = [f'Feature {i}' for i in range(len(importance))]
            
            fig.add_trace(
                go.Bar(x=feature_names, y=importance, name='Importance'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, title_text="Feature Analysis Dashboard")
        return fig
    
    def create_attention_dashboard(self, attention_data: Dict[str, Any]) -> go.Figure:
        """
        Create attention visualization dashboard.
        
        Args:
            attention_data: Attention analysis data
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Attention Heatmap', 'Attention Statistics', 'Attention Distribution', 'Attention Overlay'),
            specs=[[{"type": "heatmap"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "heatmap"}]]
        )
        
        # Attention heatmap
        if 'attention_map' in attention_data:
            attention_map = attention_data['attention_map']
            fig.add_trace(
                go.Heatmap(z=attention_map, name='Attention Map'),
                row=1, col=1
            )
        
        # Attention statistics
        if 'attention_stats' in attention_data:
            stats = attention_data['attention_stats']
            stat_names = list(stats.keys())
            stat_values = list(stats.values())
            
            fig.add_trace(
                go.Bar(x=stat_names, y=stat_values, name='Statistics'),
                row=1, col=2
            )
        
        # Attention distribution
        if 'attention_values' in attention_data:
            attention_values = attention_data['attention_values']
            fig.add_trace(
                go.Histogram(x=attention_values, name='Distribution'),
                row=2, col=1
            )
        
        # Attention overlay
        if 'attention_overlay' in attention_data:
            overlay = attention_data['attention_overlay']
            fig.add_trace(
                go.Heatmap(z=overlay, name='Overlay'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, title_text="Attention Analysis Dashboard")
        return fig
    
    def create_failure_analysis_dashboard(self, failure_data: Dict[str, Any]) -> go.Figure:
        """
        Create failure analysis dashboard.
        
        Args:
            failure_data: Failure analysis data
            
        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Failure Cases by Type', 'Confidence Distribution', 'Failure Patterns', 'Error Analysis'),
            specs=[[{"type": "pie"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Failure cases by type
        if 'failure_counts' in failure_data:
            failure_counts = failure_data['failure_counts']
            fig.add_trace(
                go.Pie(labels=list(failure_counts.keys()), values=list(failure_counts.values()),
                      name='Failure Types'),
                row=1, col=1
            )
        
        # Confidence distribution
        if 'confidence_distribution' in failure_data:
            confidences = failure_data['confidence_distribution']
            fig.add_trace(
                go.Histogram(x=confidences, name='Confidence'),
                row=1, col=2
            )
        
        # Failure patterns
        if 'failure_patterns' in failure_data:
            patterns = failure_data['failure_patterns']
            pattern_names = list(patterns.keys())
            pattern_counts = list(patterns.values())
            
            fig.add_trace(
                go.Bar(x=pattern_names, y=pattern_counts, name='Patterns'),
                row=2, col=1
            )
        
        # Error analysis
        if 'error_analysis' in failure_data:
            error_data = failure_data['error_analysis']
            if 'false_positives' in error_data and 'false_negatives' in error_data:
                fig.add_trace(
                    go.Scatter(x=error_data['false_positives'], y=error_data['false_negatives'],
                              mode='markers', name='Error Analysis'),
                    row=2, col=2
                )
        
        fig.update_layout(height=800, showlegend=True, title_text="Failure Analysis Dashboard")
        return fig


class StreamlitDashboard:
    """Streamlit-based interactive dashboard."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        """
        Initialize Streamlit dashboard.
        
        Args:
            model: Model to analyze
            device: Device to run analysis on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        logger.info("Initialized Streamlit dashboard")
    
    def create_main_dashboard(self):
        """Create main Streamlit dashboard."""
        st.set_page_config(page_title="Face Anti-Spoofing Dashboard", layout="wide")
        
        st.title("Face Anti-Spoofing Analysis Dashboard")
        
        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Select Page", [
            "Model Performance", "Feature Analysis", "Attention Visualization", 
            "Failure Analysis", "Real-time Prediction"
        ])
        
        if page == "Model Performance":
            self._create_performance_page()
        elif page == "Feature Analysis":
            self._create_feature_page()
        elif page == "Attention Visualization":
            self._create_attention_page()
        elif page == "Failure Analysis":
            self._create_failure_page()
        elif page == "Real-time Prediction":
            self._create_prediction_page()
    
    def _create_performance_page(self):
        """Create performance analysis page."""
        st.header("Model Performance Analysis")
        
        # Upload evaluation results
        uploaded_file = st.file_uploader("Upload evaluation results (JSON)", type=['json'])
        
        if uploaded_file is not None:
            try:
                results = json.load(uploaded_file)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{results.get('accuracy', 0):.4f}")
                
                with col2:
                    st.metric("Precision", f"{results.get('precision', 0):.4f}")
                
                with col3:
                    st.metric("Recall", f"{results.get('recall', 0):.4f}")
                
                with col4:
                    st.metric("F1 Score", f"{results.get('f1_score', 0):.4f}")
                
                # ROC Curve
                if 'roc_curve' in results:
                    st.subheader("ROC Curve")
                    fpr, tpr = results['roc_curve']
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
                    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Confusion Matrix
                if 'confusion_matrix' in results:
                    st.subheader("Confusion Matrix")
                    cm = results['confusion_matrix']
                    
                    fig = go.Figure(data=go.Heatmap(z=cm, x=['Predicted Real', 'Predicted Spoof'], 
                                                  y=['Actual Real', 'Actual Spoof']))
                    fig.update_layout(title="Confusion Matrix")
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading results: {e}")
    
    def _create_feature_page(self):
        """Create feature analysis page."""
        st.header("Feature Analysis")
        
        # Upload feature data
        uploaded_file = st.file_uploader("Upload feature data (JSON)", type=['json'])
        
        if uploaded_file is not None:
            try:
                feature_data = json.load(uploaded_file)
                
                # Feature distribution
                if 'feature_distribution' in feature_data:
                    st.subheader("Feature Distribution")
                    features = feature_data['feature_distribution']
                    
                    fig = go.Figure(data=go.Histogram(x=features))
                    fig.update_layout(title="Feature Distribution", xaxis_title="Feature Value", yaxis_title="Frequency")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature correlation
                if 'correlation_matrix' in feature_data:
                    st.subheader("Feature Correlation")
                    corr_matrix = feature_data['correlation_matrix']
                    
                    fig = go.Figure(data=go.Heatmap(z=corr_matrix))
                    fig.update_layout(title="Feature Correlation Matrix")
                    st.plotly_chart(fig, use_container_width=True)
                
                # t-SNE visualization
                if 'tsne_features' in feature_data and 'tsne_labels' in feature_data:
                    st.subheader("t-SNE Visualization")
                    tsne_features = np.array(feature_data['tsne_features'])
                    tsne_labels = np.array(feature_data['tsne_labels'])
                    
                    fig = go.Figure()
                    for label in np.unique(tsne_labels):
                        mask = tsne_labels == label
                        fig.add_trace(go.Scatter(x=tsne_features[mask, 0], y=tsne_features[mask, 1],
                                               mode='markers', name=f'Class {label}'))
                    fig.update_layout(title="t-SNE Visualization", xaxis_title="t-SNE 1", yaxis_title="t-SNE 2")
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error loading feature data: {e}")
    
    def _create_attention_page(self):
        """Create attention visualization page."""
        st.header("Attention Visualization")
        
        # Upload image
        uploaded_file = st.file_uploader("Upload image for attention analysis", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            try:
                # Load and preprocess image
                image = Image.open(uploaded_file)
                image_tensor = self._preprocess_image(image)
                
                # Generate attention maps
                attention_maps = self._generate_attention_maps(image_tensor)
                
                # Display original image
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
                
                # Display attention maps
                for layer_name, attention_map in attention_maps.items():
                    st.subheader(f"Attention Map - {layer_name}")
                    
                    # Create attention visualization
                    fig = go.Figure(data=go.Heatmap(z=attention_map))
                    fig.update_layout(title=f"Attention Map - {layer_name}")
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    def _create_failure_page(self):
        """Create failure analysis page."""
        st.header("Failure Case Analysis")
        
        # Upload failure data
        uploaded_file = st.file_uploader("Upload failure analysis data (JSON)", type=['json'])
        
        if uploaded_file is not None:
            try:
                failure_data = json.load(uploaded_file)
                
                # Failure statistics
                st.subheader("Failure Statistics")
                
                if 'failure_counts' in failure_data:
                    failure_counts = failure_data['failure_counts']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = go.Figure(data=go.Pie(labels=list(failure_counts.keys()), 
                                                   values=list(failure_counts.values())))
                        fig.update_layout(title="Failure Cases by Type")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = go.Figure(data=go.Bar(x=list(failure_counts.keys()), 
                                                   y=list(failure_counts.values())))
                        fig.update_layout(title="Failure Cases Count", xaxis_title="Failure Type", yaxis_title="Count")
                        st.plotly_chart(fig, use_container_width=True)
                
                # Confidence analysis
                if 'confidence_analysis' in failure_data:
                    st.subheader("Confidence Analysis")
                    conf_analysis = failure_data['confidence_analysis']
                    
                    # Display confidence statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Mean Confidence", f"{conf_analysis.get('mean', 0):.4f}")
                    
                    with col2:
                        st.metric("Std Confidence", f"{conf_analysis.get('std', 0):.4f}")
                    
                    with col3:
                        st.metric("Min Confidence", f"{conf_analysis.get('min', 0):.4f}")
                
            except Exception as e:
                st.error(f"Error loading failure data: {e}")
    
    def _create_prediction_page(self):
        """Create real-time prediction page."""
        st.header("Real-time Prediction")
        
        # Upload image
        uploaded_file = st.file_uploader("Upload image for prediction", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            try:
                # Load and preprocess image
                image = Image.open(uploaded_file)
                image_tensor = self._preprocess_image(image)
                
                # Make prediction
                with torch.no_grad():
                    output = self.model(image_tensor.to(self.device))
                    prediction = output.argmax(dim=1).item()
                    confidence = F.softmax(output, dim=1).max(dim=1)[0].item()
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Input Image")
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.subheader("Prediction Results")
                    
                    # Prediction
                    prediction_text = "Real" if prediction == 0 else "Spoof"
                    st.metric("Prediction", prediction_text)
                    
                    # Confidence
                    st.metric("Confidence", f"{confidence:.4f}")
                    
                    # Confidence bar
                    st.progress(confidence)
                    
                    # Interpretation
                    if confidence > 0.8:
                        st.success("High confidence prediction")
                    elif confidence > 0.5:
                        st.warning("Medium confidence prediction")
                    else:
                        st.error("Low confidence prediction")
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize image
        image = image.resize((224, 224))
        
        # Convert to tensor
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return image_tensor
    
    def _generate_attention_maps(self, image_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """Generate attention maps for image."""
        # This is a placeholder implementation
        # In practice, you would use the actual attention visualization methods
        attention_maps = {}
        
        # Simulate attention maps
        attention_map = np.random.rand(14, 14)  # 14x14 attention map
        attention_maps['layer_1'] = attention_map
        
        return attention_maps


def create_interactive_dashboard(model: nn.Module, device: str = "cuda") -> InteractiveDashboard:
    """
    Create interactive dashboard.
    
    Args:
        model: Model to analyze
        device: Device to run analysis on
        
    Returns:
        Interactive dashboard instance
    """
    return InteractiveDashboard(model, device)


def create_streamlit_dashboard(model: nn.Module, device: str = "cuda") -> StreamlitDashboard:
    """
    Create Streamlit dashboard.
    
    Args:
        model: Model to analyze
        device: Device to run analysis on
        
    Returns:
        Streamlit dashboard instance
    """
    return StreamlitDashboard(model, device)
