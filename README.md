# Clone the repository
git clone https://github.com/yourusername/skin-cancer-app.git
cd skin-cancer-app

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
