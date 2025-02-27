# DSAI-course
## Setting Up Your Virtual Environment  

A virtual environment helps isolate dependencies and manage Python packages efficiently. Follow the steps below based on your operating system:  

### 1. Install Python and pip  
Ensure you have Python installed on your system. If you haven't installed it yet:  

- **Windows & macOS:** Download and install Python from [python.org](https://www.python.org/downloads/).  
- **Linux (Debian-based):**  
```bash
sudo apt-get install python3-pip
```

After installation, verify python and pip versions
```bash
python --version
pip --version
```

### 2. Create a virtual environment

Once Python and pip are installed, install `virtualenv` by running:
```bash
pip install virtualenv
```

Confirm the installation 
```bash
virtualenv --version
```

Choose a name for your environment (e.g., `DSAI_env`) and run:
```bash
virtualenv DSAI_env
```

For a specific Python version:
```bash
virtualenv -p python3 DSAI_env
```

There are different ways to activate virtual environment;

- Windows (Command Prompt):
```bash
DSAI_env\Scripts\activate
```

- Windows (PowerShell):
```bash
DSAI_env\Scripts\Activate.ps1
```

- macOS & Linux:
```bash
source DSAI_env/bin/activate
```

Once activated, you will see the environment name in your terminal prompt.

To exit the virtual environment, run:
```bash
deactivate
```

Now you're ready to install packages and work within an isolated environment for your Data Science and AI projects! 🚀

## Module 1