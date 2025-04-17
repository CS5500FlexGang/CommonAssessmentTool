Team Flexgang

Project -- Feature Development Backend: Create CRUD API's for Client

User Story

As a user of the backend API's, I want to call API's that can retrieve, update, and delete information of clients who have already registered with the CaseManagment service so that I more efficiently help previous clients make better decisions on how to be gainfully employed.

Acceptance Criteria

- Provide REST API endpoints so that the Frontend can use them to get information on an existing client.
- Document how to use the REST API
- Choose and create a database to hold client information
- Add tests

This will contain the model used for the project that based on the input information will give the social workers the clients baseline level of success and what their success will be after certain interventions.

The model works off of dummy data of several combinations of clients alongside the interventions chosen for them as well as their success rate at finding a job afterward. The model will be updated by the case workers by inputing new data for clients with their updated outcome information, and it can be updated on a daily, weekly, or monthly basis.

This also has an API file to interact with the front end, and logic in order to process the interventions coming from the front end. This includes functions to clean data, create a matrix of all possible combinations in order to get the ones with the highest increase of success, and output the results in a way the front end can interact with.

-------------------------How to Use (Locally)-------------------------

1. In the virtual environment you've created for this project, install all dependencies in requirements.txt (`pip install -r requirements.txt`)

2. Run the app (`uvicorn app.main:app --reload`)

3. Load data into database (`python initialize_data.py`)

4. Go to SwaggerUI (`http://127.0.0.1:8000/docs`)

5. Log in as admin (username: `admin` password: `admin123`)

6. Click on each endpoint to use -Create User (Only users in admin role can create new users. The role field needs to be either "admin" or "case_worker")

-------------------------How to Use (With Docker)-------------------------
Option 1: Using Docker CLI

1. Build the Docker image (`docker build -t fastapi-app .`)

2. Run the Docker container (`docker run -d -p 8000:8000 --name fastapi-backend fastapi-app`)

3. Load data into database (`docker exec -it fastapi-backend python initialize_data.py`)

4. Go to SwaggerUI (`http://127.0.0.1:8000/docs`)

5. Log in as admin (username: `admin` password: `admin123`)

6. Click on each endpoint to use -Create User (Only users in admin role can create new users. The role field needs to be either "admin" or "case_worker")

Option 2: Using Docker Compose

1. Start the application(`docker-compose up -d`)

2. Load data into database (`docker exec -it fastapi-backend python initialize_data.py`)

3. Go to SwaggerUI (`http://127.0.0.1:8000/docs`)

4. Log in as admin (username: `admin` password: `admin123`)

5. Click on each endpoint to use -Create User (Only users in admin role can create new users. The role field needs to be either "admin" or "case_worker")

-Create User (Only users in admin role can create new users. The role field needs to be either "admin" or "case_worker")

-Get clients (Display all the clients that are in the database)

-Get client (Allow authorized users to search for a client by id. If the id is not in database, an error message will show.)

-Update client (Allow authorized users to update a client's basic info by inputting in client_id and providing updated values.)

-Delete client (Allow authorized users to delete a client by id. If an id is no longer in the database, an error message will show.)

-Get clients by criteria (Allow authorized users to get a list of clients who meet a certain combination of criteria.)

-Get Clients by services (Allow authorized users to get a list of clients who meet a certain combination of service statuses.)

-Get clients services (Allow authorized users to view a client's services' status.)

-Get clients by success rate (Allow authorized users to search for clients whose cases have a success rate beyond a certain number.)

-Get clients by case worker (Allow users to view which clients are assigned to a specific case worker.)

-Update client services (Allow users to update the service status of a case.)

-Create case assignment (Allow authorized users to create a new case assignment.)

-------------------------CI/CD Pipeline-------------------------

This project includes a continuous integration (CI) pipeline built with GitHub Actions that helps maintain code quality and ensure the application runs correctly.

### CI Pipeline Features

- **Automated Code Checks**: The pipeline runs linters (pylint, flake8) and formatters (black) to ensure code quality and consistency.
- **Automated Testing**: All tests are automatically run to verify that changes don't break existing functionality.
- **Docker Validation**: The pipeline builds and runs the Docker container to ensure it works as expected.

### CI Pipeline Triggers

The CI pipeline is automatically triggered on:
- Pushes to main, docker, elisaa, and ci-pipeline branches
- Pull requests to the main branch

### How to View Pipeline Results

1. Go to the "Actions" tab in the GitHub repository
2. Click on the most recent workflow run to view details
3. Expand job sections to see individual steps and their output

### Local Development with CI in Mind

Before pushing your changes, you can run the same checks locally:

```bash
# Run linters
pylint app/
flake8 app/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Run formatter
black app/

# Run tests
python -m pytest tests/

# Build and test Docker image
docker build -t fastapi-app .
docker run -d -p 8000:8000 --name fastapi-backend fastapi-app

------------------------- Backend Application - Common Assessment Tool-------------------------
## Public Access
The backend API is deployed and accessible at:
http://3.142.42.78:8000

Swagger API documentation is available at:
http://3.142.42.78:8000/docs

## Deployment Information
- Hosted on AWS EC2
- Region: us-east-2
```

------------------------- Setting Up the Deployment Pipeline -------------------------
## SSH Key Configuration
### Generate an SSH key pair
```bash
ssh-keygen -t rsa -b 4096 -C "github-actions"
```
### Add the public key to your VM's authorized_keys file
```bash
cat ~/.ssh/id_rsa.pub | ssh your_username@your-vm-ip "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

### Add the private key as a GitHub secret:
Go to your GitHub repository → Settings → Secrets and variables → Actions
Create a new secret named SSH_PRIVATE_KEY with the content of your private key

### Add two more required secrets:
VM_HOST: The hostname or IP address of your VM
VM_USER: The username for SSH login (e.g., ec2-user)

## Docker Setup on VM
### Ensure Docker is installed and running on your VM
### Note: This step is only required for initial setup. If you already have a working CD pipeline, Docker should already be installed on your VM.
For Ubuntu/Debian-based VMs:
```bash
# SSH into VM
ssh your_username@vm-sm
# Update packages
sudo apt update
# Install Docker if not already installed
sudo apt install -y docker.io
# Add your user to the docker group to avoid using sudo
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect
exit
ssh your_username@vm-sm
```

For Amazon Linux-based VMs:
```bash
# SSH into VM
ssh ec2-user@your-vm-ip
# Update packages and install Docker
sudo yum update -y
sudo yum install -y docker
# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker
# Add your user to the docker group
sudo usermod -a -G docker $USER
# Log out and back in for changes to take effect
exit
ssh ec2-user@your-vm-ip
```

## The CD Workflow File
The CD workflow is defined in .github/workflows/cd.yml. It triggers whenever a release is created from the master branch.

## Creating a New Release
To deploy a new version of the application:
1. Make your code changes and push them to the main branch
2. Go to your GitHub repository
3. Click on "Releases" in the right sidebar
4. Click "Create a new release"
5. Enter a tag version (e.g., v1.0.4)
6. Add a title and description
7. Click "Publish release"
The CD workflow will automatically trigger and deploy your changes to the VM.

## Monitoring Deployments
To monitor the deployment process:
1. Go to the "Actions" tab in your GitHub repository
2. Click on the running or most recent "Continuous Deployment" workflow
3. 1View the logs for each step of the deployment

## Verifying Deployment
To verify that your deployment was successful:
1. Access your API documentation at: http://your-vm-ip:8000/docs
2. Check that your changes are visible
3. Test the API endpoints to ensure functionality

## Troubleshooting
### Port Already in Use
If you see an error like:
```bash
Error: listen tcp4 0.0.0.0:8000: bind: address already in use
```
SSH into your VM and:
```bash
# SSH into your VM
ssh ec2-user@3.142.42.78
# Find the process using port 8000
sudo lsof -i :8000
# Kill the process
sudo kill -9 [PID]
```


