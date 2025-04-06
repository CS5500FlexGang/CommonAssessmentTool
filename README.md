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
