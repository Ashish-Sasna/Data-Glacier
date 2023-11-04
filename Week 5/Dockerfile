# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /DataGlacier/Week5

# Copy the current directory contents into the container
COPY Scripts/ /DataGlacier/Week5/Scripts/
COPY model.pkl /DataGlacier/Week5/
COPY requirements.txt /DataGlacier/Week5/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port 5000
EXPOSE 5000

# Entrypoint for executing the app
ENTRYPOINT ["python", "Scripts/app.py"]
