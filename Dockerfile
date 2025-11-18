# Use the official Python image as a base
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
# We assume the required packages (streamlit, pandas, numpy, scikit-learn, plotly)
# are listed in a requirements.txt file.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files (the main app and the pages directory)
COPY . /app

# The application requires the 'OnlineRetail.csv' file to be present at /app/OnlineRetail.csv.
# This file must be mounted as a volume when running the container.

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "app.py"]