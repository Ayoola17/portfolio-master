# Use an official Python runtime as a parent image
FROM python:3.6-slim-buster

# Set the working directory to /app
WORKDIR /Image

# Copy the current directory contents into the container at /app
COPY . /Image/

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable


# Run app.py when the container launches
CMD ["python", "manage.py", "runserver", "0.0.0.0:80"]
