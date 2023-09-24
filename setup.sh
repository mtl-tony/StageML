# 1. First, navigate to the directory where you want to create your virtual environment
cd C:/Users/tony/Documents/

# 2. Use the python3 -m venv command to create a virtual environment.
# You can replace 'myenv' with the name you want to give to your virtual environment.
python -m venv staged_ml_testing

# 3. To start using the virtual environment, you need to activate it.
# If you're using bash or a bash-compatible shell (like zsh), use:
. ./staged_ml_testing/Scripts/activate

# Now, your command prompt changes to show the name of the activated virtual environment.
# At this point, you're inside your virtual environment and can start installing packages into it.
cd C:/Users/tony/Documents/git/StagedML/
pip install -r requirements.txt
