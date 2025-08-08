# About this repo


python3 -m venv env-ai-measurements-backend
source env-ai-measurements-backend/bin/activate

# Show dependencies in a file

## Uvicorn is the server that makes the endpoints available


pip freeze > requirements.txt
# A different way to obtain the same (but once the project is done, as it gets the project )
pip install pipreqs
pipreqs /path/to/your/project



# For development running the server and all the apps
uvicorn main:app --reload





# IT needs the ESOA backend.ai.signal_cleaner package
## You can install it as follows
pip install -e ../backend.ai.signal_cleaner