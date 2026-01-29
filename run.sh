#!/bin/bash
# Easy run script for Fake News Detection System

echo "========================================================================"
echo "             FAKE NEWS DETECTION SYSTEM"
echo "========================================================================"
echo ""

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "⚠️  Warning: 'models' directory not found!"
    echo "Please ensure your models are in the following structure:"
    echo "  models/"
    echo "  ├── bert/final_model/"
    echo "  ├── roberta/final_model/"
    echo "  └── tf_idf/"
    echo ""
fi

# Function to show menu
show_menu() {
    echo "Choose an option:"
    echo ""
    echo "  1) Run Demo (sample texts)"
    echo "  2) Interactive Mode (enter text manually)"
    echo "  3) Quick Test (verify system works)"
    echo "  4) Run Examples"
    echo "  5) Analyze from text input"
    echo "  6) Exit"
    echo ""
}

# Main loop
while true; do
    show_menu
    read -p "Enter choice [1-6]: " choice
    echo ""
    
    case $choice in
        1)
            echo "Running demo..."
            python3 demo.py --demo
            ;;
        2)
            echo "Starting interactive mode..."
            echo "Type 'quit' to return to menu"
            python3 demo.py --interactive
            ;;
        3)
            echo "Running system test..."
            python3 test.py
            ;;
        4)
            echo "Available examples:"
            echo "  1. Basic Usage"
            echo "  2. Model Selection"
            echo "  3. Ensemble Methods"
            echo "  4. Batch Processing"
            echo "  5. Detailed Analysis"
            echo "  6. Component Usage"
            echo "  7. Risk Assessment"
            echo "  8. Custom Weights"
            echo ""
            read -p "Enter example number [1-8] or 'all': " ex_choice
            echo ""
            
            if [ "$ex_choice" = "all" ]; then
                python3 examples.py --all
            else
                python3 examples.py --example $ex_choice
            fi
            ;;
        5)
            echo "Enter text to analyze (press Enter twice when done):"
            text=""
            while IFS= read -r line; do
                [ -z "$line" ] && break
                text="${text}${line} "
            done
            
            if [ -n "$text" ]; then
                python3 main.py --text "$text"
            else
                echo "No text entered."
            fi
            ;;
        6)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please enter 1-6."
            ;;
    esac
    
    echo ""
    echo "------------------------------------------------------------------------"
    echo ""
done
