## Prerequisites

Before running the application, ensure you have Python 3.10+ and a virtual environment (`.venv`) configured.

## Setup Instructions

### 1. Clone & Navigate to Project
```bash
cd "Summer internship thing/project"
```

### 2. Configure Environment Variables
The project requires an API key to communicate with the model services. 

1. Create a new file named `.env` in the root directory of the project.
2. Add your API key using the exact format below:

```env
API_KEY=your_actual_api_key_here
```

> ⚠️ **Security Warning:** Never commit your `.env` file to GitHub or version control.

### 3. Install Dependencies
Activate your virtual environment and install the required core packages:

```bash
# Activate your environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate your environment (Linux/macOS or Git Bash)
source .venv/bin/activate

# Install the reduced requirements list
pip install -r requirements.txt
```

### 4. Run the Application
To launch the Streamlit frontend user interface:

```bash
streamlit run app.py
```
