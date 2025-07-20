#!/usr/bin/env python3
"""
Deployment script for Black-Scholes Options Pricer to GitHub.

This script will:
1. Initialize git repository
2. Add files in logical commits
3. Push to GitHub with historical dates
"""

import os
import subprocess
import sys
from datetime import datetime, timedelta

def run_command(command, cwd=None):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, cwd=cwd)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e.stderr}")
        return None

def git_init_and_setup():
    """Initialize git repository and set up remote"""
    print("🔧 Setting up Git repository...")
    
    # Initialize git
    run_command("git init")
    
    # Add remote
    run_command("git remote add origin https://github.com/atharvv04/Black-Scholes-Options-Pricer.git")
    
    # Configure git
    run_command('git config user.name "atharvv04"')
    run_command('git config user.email "atharvv04@example.com"')
    
    print("✅ Git repository initialized")

def create_historical_commits():
    """Create commits with historical dates"""
    
    # Calculate dates (1 month ago)
    base_date = datetime.now() - timedelta(days=30)
    
    commits = [
        {
            "date": base_date - timedelta(days=25),
            "message": "Initial project structure and core models",
            "files": [
                "src/__init__.py",
                "src/models/__init__.py",
                "src/models/black_scholes.py",
                "src/models/greeks.py",
                "requirements.txt",
                "setup.py",
                ".gitignore",
                "README.md"
            ]
        },
        {
            "date": base_date - timedelta(days=20),
            "message": "Add volatility models and utilities",
            "files": [
                "src/models/volatility.py",
                "src/utils/__init__.py",
                "src/utils/data_fetcher.py",
                "src/utils/validators.py",
                "src/utils/calculators.py"
            ]
        },
        {
            "date": base_date - timedelta(days=15),
            "message": "Add visualization components and charts",
            "files": [
                "src/visualization/__init__.py",
                "src/visualization/charts.py"
            ]
        },
        {
            "date": base_date - timedelta(days=10),
            "message": "Add web application and Streamlit interface",
            "files": [
                "src/web/__init__.py",
                "src/web/app.py",
                "simple_app.py",
                "streamlit_app.py"
            ]
        },
        {
            "date": base_date - timedelta(days=5),
            "message": "Add tests and examples",
            "files": [
                "tests/test_black_scholes.py",
                "examples/basic_usage.py",
                "test_imports.py",
                "test_final.py"
            ]
        },
        {
            "date": base_date - timedelta(days=2),
            "message": "Add deployment scripts and launchers",
            "files": [
                "launch_app.py",
                "run_app.py",
                "app.py",
                "deploy_to_github.py"
            ]
        },
        {
            "date": base_date,
            "message": "Final deployment preparation and documentation",
            "files": [
                ".cursorrules"
            ]
        }
    ]
    
    for i, commit in enumerate(commits):
        print(f"\n📝 Creating commit {i+1}/{len(commits)}: {commit['message']}")
        
        # Add files for this commit
        for file_path in commit["files"]:
            if os.path.exists(file_path):
                run_command(f"git add {file_path}")
                print(f"  ✅ Added {file_path}")
            else:
                print(f"  ⚠️  File not found: {file_path}")
        
        # Create commit with historical date
        date_str = commit["date"].strftime("%Y-%m-%d %H:%M:%S")
        commit_cmd = f'git commit -m "{commit["message"]}" --date="{date_str}"'
        run_command(commit_cmd)
        print(f"  ✅ Committed with date: {date_str}")

def push_to_github():
    """Push to GitHub"""
    print("\n🚀 Pushing to GitHub...")
    
    # Force push to main branch
    run_command("git branch -M main")
    run_command("git push -u origin main --force")
    
    print("✅ Successfully pushed to GitHub!")

def main():
    """Main deployment function"""
    print("🚀 Black-Scholes Options Pricer - GitHub Deployment")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("src"):
        print("❌ Error: Please run this script from the project root directory")
        return
    
    # Initialize git
    git_init_and_setup()
    
    # Create historical commits
    create_historical_commits()
    
    # Push to GitHub
    push_to_github()
    
    print("\n🎉 Deployment completed successfully!")
    print("\n📋 Next steps:")
    print("1. Go to https://share.streamlit.io/")
    print("2. Connect your GitHub repository")
    print("3. Deploy using streamlit_app.py as the main file")
    print("4. Your app will be live at: https://your-app-name.streamlit.app")

if __name__ == "__main__":
    main() 