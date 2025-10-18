#!/usr/bin/env python3
"""
FAS-Research-Framework Test Runner

Test runner for the FAS-Research-Framework project.
Provides comprehensive testing with coverage reporting.
"""

import argparse
import sys
from pathlib import Path

import pytest


def main():
    """
    Run tests with various options.
    
    This function provides a comprehensive test runner for the face anti-spoofing project,
    supporting coverage reporting, verbose output, fast test filtering, and category-specific
    test execution.
    """
    parser = argparse.ArgumentParser(description="Run tests for face anti-spoofing project")
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Run tests in verbose mode"
    )
    parser.add_argument(
        "--fast", 
        action="store_true", 
        help="Run only fast tests (skip slow integration tests)"
    )
    parser.add_argument(
        "--models", 
        action="store_true", 
        help="Run only model tests"
    )
    parser.add_argument(
        "--utils", 
        action="store_true", 
        help="Run only utility tests"
    )
    parser.add_argument(
        "--config", 
        action="store_true", 
        help="Run only configuration tests"
    )
    parser.add_argument(
        "--file", 
        type=str, 
        help="Run tests from specific file"
    )
    
    args = parser.parse_args()
    
    # Build pytest arguments based on user input
    pytest_args = []
    
    if args.verbose:
        pytest_args.append("-v")
    
    if args.coverage:
        # Add coverage reporting with HTML and terminal output
        # Fail if coverage is below 80%
        pytest_args.extend([
            "--cov=.",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=80"
        ])
    
    if args.fast:
        # Skip slow integration tests for faster execution
        pytest_args.extend(["-m", "not slow"])
    
    # Select specific test categories based on user preference
    if args.models:
        pytest_args.append("tests/test_models.py")
    elif args.utils:
        pytest_args.append("tests/test_utils.py")
    elif args.config:
        pytest_args.append("tests/test_config_validation.py")
    elif args.file:
        pytest_args.append(args.file)
    else:
        pytest_args.append("tests/")
    
    # Add additional pytest options
    pytest_args.extend([
        "--tb=short",
        "--strict-markers",
        "--disable-warnings"
    ])
    
    print(f"Running tests with args: {' '.join(pytest_args)}")
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\nSUCCESS: All tests passed!")
    else:
        print(f"\nFAILED: Tests failed with exit code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
